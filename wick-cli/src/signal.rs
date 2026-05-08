//! SIGINT handling for the line-REPL chat loop.
//!
//! The TUI handles Ctrl+C entirely through crossterm's raw-mode key events
//! (it sees `KeyCode::Char('c')` with `KeyModifiers::CONTROL` before the
//! kernel can deliver SIGINT to the process); this module is for the
//! line-mode `wick chat --no-tui` path where Ctrl+C would otherwise kill
//! the entire REPL and wipe the user's conversation.
//!
//! ## Semantics
//!
//! Ctrl+C during a turn (prefill or generate) → flip the session's cancel
//! atomic. The session checks it between tokens and unwinds with
//! `WickError::Cancelled` / `FinishReason::Cancelled`; the REPL pops the
//! user turn, prints "(cancelled)", and continues.
//!
//! Ctrl+C at the prompt (waiting on stdin) → exit the process with code
//! 130 (= 128 + SIGINT, the conventional exit code for "killed by SIGINT").
//! Same effective behavior as the default SIGINT handler, just explicit.
//!
//! The two modes are distinguished by an `intercepting: AtomicBool` shared
//! between the REPL loop and the handler closure: the REPL flips it true
//! around `session.append_*` and `session.generate(...)`, false otherwise.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

/// Conventional Unix exit code for "killed by SIGINT" (128 + signal number).
const SIGINT_EXIT_CODE: i32 = 130;

/// Handler closure body. NOT an async-signal-safe handler in the POSIX
/// sense — `ctrlc` runs the closure on a dedicated thread it spawns at
/// `set_handler` time (Unix `sigwait` loop / Windows console control
/// handler), and `process::exit` itself is not async-signal-safe — but
/// the closure runs in normal Rust thread context, so the only
/// constraint is the usual thread-safe atomic / sync-primitive usage.
/// Extracted into a free function so the `intercepting=true` branch can
/// be unit-tested without registering a real handler (which is
/// process-global and would conflict with other tests).
///
/// `sigint_fired` is set in addition to the session's `cancel` flag so the
/// REPL can distinguish "user pressed Ctrl+C" from "stdout went away and
/// `ChatSink` self-cancelled" — both flip `cancel` but only the former
/// should keep the REPL alive afterward.
pub(crate) fn handle_sigint(
    cancel: &AtomicBool,
    intercepting: &AtomicBool,
    sigint_fired: &AtomicBool,
) {
    if intercepting.load(Ordering::Relaxed) {
        cancel.store(true, Ordering::Relaxed);
        sigint_fired.store(true, Ordering::Relaxed);
    } else {
        // At the prompt — fall back to default SIGINT behavior so users
        // can exit with one keystroke. The line REPL has no scrollback
        // / readline state to clean up, so a hard exit is fine.
        std::process::exit(SIGINT_EXIT_CODE);
    }
}

/// Install a process-wide SIGINT handler wired to the supplied flags. Call
/// this once when entering the line REPL; the handler stays installed for
/// the rest of the process. `ctrlc` is cross-platform (Unix signals +
/// Windows console control handler) so behavior is consistent on
/// macOS / Linux / Windows.
pub(crate) fn install_line_repl_handler(
    cancel: Arc<AtomicBool>,
    intercepting: Arc<AtomicBool>,
    sigint_fired: Arc<AtomicBool>,
) -> Result<()> {
    ctrlc::set_handler(move || handle_sigint(&cancel, &intercepting, &sigint_fired))
        .context("install SIGINT handler for line REPL")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intercepting_true_sets_cancel_and_fired() {
        let cancel = AtomicBool::new(false);
        let intercepting = AtomicBool::new(true);
        let fired = AtomicBool::new(false);
        handle_sigint(&cancel, &intercepting, &fired);
        assert!(cancel.load(Ordering::Relaxed));
        assert!(fired.load(Ordering::Relaxed));
        // intercepting is unchanged; only the REPL loop toggles it.
        assert!(intercepting.load(Ordering::Relaxed));
    }

    #[test]
    fn intercepting_true_idempotent() {
        // Repeated SIGINTs while still intercepting must keep cancel set;
        // the handler must not toggle / clear it.
        let cancel = AtomicBool::new(false);
        let intercepting = AtomicBool::new(true);
        let fired = AtomicBool::new(false);
        handle_sigint(&cancel, &intercepting, &fired);
        handle_sigint(&cancel, &intercepting, &fired);
        handle_sigint(&cancel, &intercepting, &fired);
        assert!(cancel.load(Ordering::Relaxed));
        assert!(fired.load(Ordering::Relaxed));
    }

    // The intercepting=false branch calls `std::process::exit(130)` and
    // can't be unit-tested without bringing down the test runner.
    // Manual smoke covers it: spawning `wick chat --no-tui` and sending
    // SIGINT before any input verifies the process exits with 130.
}
