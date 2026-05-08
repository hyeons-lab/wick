//! Inline TUI for `wick chat` — Claude Code / Junie style.
//!
//! Instead of taking over the whole terminal with a full-screen
//! alternate buffer, we use ratatui's `Viewport::Inline` mode: a
//! small (2-line) widget area pinned at the bottom of the terminal,
//! with everything above flowing to the scrollback as normal terminal
//! output. User and assistant messages get committed to scrollback
//! via `Terminal::insert_before` once they're finalised; live
//! streaming + the input prompt live in the inline viewport.
//!
//! Layout (bottom of terminal):
//!
//!   ```
//!   …prior messages in scrollback above…
//!   user> What is 2+2?
//!   assistant> 4
//!
//!   > _                                    <- input line (cursor here)
//!   ↵ send · /help · Ctrl+C to exit         <- hint line
//!   ```
//!
//! While the worker is mid-generate, the input line shows
//! `assistant> <streaming tokens>` and the hint line says
//! "generating… Ctrl+C to cancel". When the turn finishes, the
//! assembled assistant text gets emitted to scrollback and the input
//! line returns to its empty prompt.
//!
//! Slash commands (`/help`, `/clear`, `/save <path>`,
//! `/system <text>`, `/exit`, `/quit`) match the line-based REPL
//! exactly — same dispatch logic, same `crate::write_transcript`
//! helper for `/save`. Status output (help text, "history
//! cleared", "saved N messages", error reports) is emitted to
//! scrollback via `insert_before` so it doesn't get lost in
//! viewport refreshes.

use std::io::{self, Stdout, Write};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::{Terminal, TerminalOptions, Viewport};
use unicode_width::UnicodeWidthChar;
use wick::tokenizer::{BpeTokenizer, ChatMessage};
use wick::{FinishReason, GenerateOpts, ModalitySink, Session};

/// Updates flowing from the inference worker thread to the UI. The
/// `ImageFetched` variant is also delivered from a `/image <url>`
/// fetch thread (not the inference worker) — the channel is shared
/// so the UI sees both kinds of async results in the same drain loop.
enum UiUpdate {
    /// Decoded text — append to the in-flight assistant reply.
    Chunk(String),
    /// Turn finished; carries the finish reason for the hint line.
    Done(FinishReason),
    /// `session.append_tokens` or `session.generate` errored. The
    /// in-flight user turn gets popped on the worker side; the UI
    /// surfaces the message and returns to the prompt.
    Error(String),
    /// A `/image <url>` fetch finished. `Ok(bytes)` pushes onto
    /// `pending_images`; `Err(message)` is surfaced as a status line.
    /// Always clears `state.fetching_url` so the UI unlocks.
    ImageFetched {
        url: String,
        result: std::result::Result<Vec<u8>, String>,
    },
}

/// User-driven turns dispatched from the UI to the worker. The full
/// history (including the new user turn at the end) ships per turn —
/// the worker is stateless w.r.t. history so `/clear` on the UI side
/// just stops including the dropped messages on the next turn,
/// without needing a "reset history" signal back to the worker.
///
/// `history_images` is parallel to `history` (one entry per
/// message; empty Vec for system / assistant / text-only user)
/// and carries the image attachments for every turn that had
/// any. Worker re-feeds every image through the encoder on
/// every turn so the model has actual pixel access for
/// fine-grained follow-ups, not just its own turn-1 description.
/// `Arc<Vec<u8>>` keeps the per-turn channel send cheap
/// (refcount bumps, not memcpy of up to 50 MB per attachment).
enum TurnRequest {
    Turn {
        history: Vec<ChatMessage>,
        history_images: Vec<Vec<std::sync::Arc<Vec<u8>>>>,
    },
    Shutdown,
}

struct ChatState {
    history: Vec<ChatMessage>,
    /// In-flight assistant text streamed from the worker. `None`
    /// between turns; `Some` while generating.
    pending: Option<String>,
    input: String,
    /// Cursor column inside `input` measured in bytes (always on a
    /// char boundary).
    cursor: usize,
    /// `true` while the worker is busy.
    generating: bool,
    /// Last finish reason — surfaced briefly in the hint line.
    last_finish: Option<FinishReason>,
    /// Image bytes attached via `/image <path-or-url>`; the next user
    /// turn copies these into `history_images[user_idx]` (cheap
    /// refcount bump per attachment) and drains
    /// `pending_images` on success. Preserved on
    /// `UiUpdate::Error` so the user can retry without
    /// re-attaching.
    pending_images: Vec<std::sync::Arc<Vec<u8>>>,
    /// Parallel to `history`: each entry holds the images
    /// attached to the corresponding turn (empty for
    /// system / assistant / text-only user). Used to re-render
    /// the conversation as multimodal on every turn so the
    /// model has actual pixel access for fine-grained
    /// follow-ups across many turns, not just its own turn-1
    /// description.
    history_images: Vec<Vec<std::sync::Arc<Vec<u8>>>>,
    /// `Some(url)` while a `/image <url>` fetch is running on a
    /// background thread. Locks input dispatch the same way
    /// `generating` does, but renders / Ctrl+C / exit stay
    /// responsive — the prior synchronous fetch could freeze
    /// the UI for up to 30s on a misbehaving server.
    fetching_url: Option<String>,
}

/// RAII guard that disables raw mode + emits a final newline when
/// dropped, even on panic. Ensures the user's terminal is left in a
/// usable state if anything in the TUI loop unwinds — without this,
/// a panic mid-render would leave the shell stuck in raw mode and
/// the user would have to `reset` or close the window.
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        // Restore cursor visibility — ratatui hides it during draws
        // and a panic mid-frame would otherwise leave the user's
        // shell with no visible cursor. Then emit a newline so the
        // shell prompt lands on a fresh row after the inline
        // viewport's bottom line.
        let _ = execute!(
            io::stdout(),
            crossterm::cursor::Show,
            crossterm::cursor::MoveToColumn(0)
        );
        let _ = writeln!(io::stdout());
    }
}

/// Run the inline TUI. Consumes the `Session` because the worker
/// thread takes ownership for the duration; returns the final
/// history when the user exits cleanly.
///
/// `tokenizer` is taken as `Arc<BpeTokenizer>` rather than a value
/// clone because `WickEngine` already stores the tokenizer behind an
/// `Arc` (see `engine.tokenizer_arc()`) and the vocab + merge tables
/// are large enough that a deep clone is wasteful. The worker thread
/// receives its own `Arc` by clone, sharing the underlying maps.
pub(crate) fn run(
    session: Session,
    tokenizer: std::sync::Arc<BpeTokenizer>,
    initial_history: Vec<ChatMessage>,
    opts: GenerateOpts,
) -> Result<Vec<ChatMessage>> {
    enable_raw_mode()?;
    // Install the panic guard FIRST so any later `?` early-return
    // (or panic from `Terminal::with_options`) restores the terminal
    // before unwinding past this frame.
    let _guard = RawModeGuard;

    let mut stdout = io::stdout();
    // No alternate-screen toggle — we want messages to remain in the
    // user's scrollback after exit.
    let backend = CrosstermBackend::new(stdout.by_ref());
    // Inline(3): up to 2 lines for input/streaming with wrap, plus
    // 1 hint line. Multi-line input fits within the same 2-row pane;
    // wrapped streaming output auto-scrolls to keep the latest tokens
    // visible.
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(3),
        },
    )?;

    let result = run_inner(&mut terminal, session, tokenizer, initial_history, opts);

    // On clean exit also clear the inline viewport region + restore
    // the cursor explicitly (the guard's Drop only handles raw mode
    // and the trailing newline).
    let _ = terminal.clear();
    let _ = terminal.show_cursor();
    result
}

fn run_inner(
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
    session: Session,
    tokenizer: std::sync::Arc<BpeTokenizer>,
    initial_history: Vec<ChatMessage>,
    opts: GenerateOpts,
) -> Result<Vec<ChatMessage>> {
    let cancel = session.cancel_handle();
    // No path today pre-loads images, so `history_images` is
    // initialized to N empty Vecs aligned with `initial_history`.
    let history_images: Vec<Vec<std::sync::Arc<Vec<u8>>>> = vec![Vec::new(); initial_history.len()];
    let mut state = ChatState {
        history: initial_history,
        pending: None,
        input: String::new(),
        cursor: 0,
        generating: false,
        last_finish: None,
        pending_images: Vec::new(),
        history_images,
        fetching_url: None,
    };

    // Print system prompt + any pre-loaded history into scrollback so
    // the user sees what's already in context.
    for msg in &state.history {
        emit_message_to_scrollback(terminal, &msg.role, &msg.content)?;
    }
    if !state.history.is_empty() {
        // Blank separator before the input area.
        emit_blank_line(terminal)?;
    }

    let (tx_turn, rx_turn) = mpsc::channel::<TurnRequest>();
    let (tx_update, rx_update) = mpsc::channel::<UiUpdate>();

    let worker_tokenizer = tokenizer;
    let worker_opts = opts.clone();
    // Clone for the worker so the UI keeps an owned `Sender` to pass to
    // `/image <url>` fetch threads (and to `handle_key`).
    let worker_tx_update = tx_update.clone();
    let worker_handle = thread::spawn(move || {
        worker_loop(
            session,
            worker_tokenizer,
            worker_opts,
            rx_turn,
            worker_tx_update,
        );
    });

    'outer: loop {
        terminal.draw(|frame| draw_inline(frame, &state))?;

        // Drain worker updates between draws.
        loop {
            match rx_update.try_recv() {
                Ok(UiUpdate::Chunk(s)) => {
                    // Strip leading whitespace ONLY while `pending`
                    // is still empty/whitespace. Chat templates often
                    // emit a leading "\n" or "\n " right after the
                    // assistant tag; we want that gone, but any
                    // whitespace AFTER the first non-whitespace
                    // character is real (code-block indent, list
                    // markers, etc.) and must be preserved.
                    let pending = state.pending.get_or_insert_with(String::new);
                    if pending.chars().all(char::is_whitespace) {
                        pending.clear();
                        pending.push_str(s.trim_start());
                    } else {
                        pending.push_str(&s);
                    }
                }
                Ok(UiUpdate::Done(reason)) => {
                    // Always commit the assistant turn, mirroring
                    // the line REPL's `history.push(...content:
                    // sink.into_text())` after every successful
                    // generate. `Session::generate` can legitimately
                    // produce zero text tokens (e.g. first sampled
                    // token is EOS) — `state.pending` is `Some("")`
                    // from the dispatch initializer in that case
                    // and `take().unwrap_or_default()` collapses
                    // both that and any future code path that left
                    // it `None` to an empty assistant message.
                    // Skipping the push would leave two consecutive
                    // user turns in `history` and break subsequent
                    // chat-template rendering.
                    let text = state.pending.take().unwrap_or_default();
                    emit_message_to_scrollback(terminal, "assistant", &text)?;
                    emit_blank_line(terminal)?;
                    state.history.push(ChatMessage {
                        role: "assistant".into(),
                        content: text,
                    });
                    state.history_images.push(Vec::new());
                    // Drain pending images only after a full-turn
                    // success (worker finished prefill AND
                    // generate, assistant text just committed). On
                    // error the `Error` arm leaves them intact so
                    // the user can retry without re-attaching —
                    // same discipline as the line REPL (PR #136).
                    // Note: the user-turn images already moved
                    // into `history_images` at send time (see
                    // `dispatch_user_input`), so they're permanent
                    // for the rest of the session (until
                    // `/clear`).
                    state.pending_images.clear();
                    state.generating = false;
                    state.last_finish = Some(reason);
                }
                Ok(UiUpdate::Error(e)) => {
                    if let Some(last) = state.history.last()
                        && last.role == "user"
                    {
                        state.history.pop();
                        state.history_images.pop();
                    }
                    state.pending = None;
                    state.generating = false;
                    emit_status_to_scrollback(terminal, &format!("error: {e}"))?;
                    emit_blank_line(terminal)?;
                }
                Ok(UiUpdate::ImageFetched { url, result }) => {
                    // Background `/image <url>` fetch returned. If the
                    // user already left the TUI or hit `/clear` we'd
                    // still get here (the channel doesn't know about
                    // state); apply the result regardless — `/clear`
                    // dropped the previous `pending_images`, but a
                    // late-arriving fetch is fine to attach.
                    state.fetching_url = None;
                    match result {
                        Ok(bytes) => {
                            emit_status_to_scrollback(
                                terminal,
                                &format!(
                                    "(image attached: {url}, {} bytes; sends with next message)",
                                    bytes.len()
                                ),
                            )?;
                            emit_blank_line(terminal)?;
                            state.pending_images.push(std::sync::Arc::new(bytes));
                        }
                        Err(e) => {
                            emit_status_to_scrollback(
                                terminal,
                                &format!("error: /image {url}: {e}"),
                            )?;
                            emit_blank_line(terminal)?;
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break 'outer,
            }
        }

        if !crossterm::event::poll(Duration::from_millis(50))? {
            continue;
        }
        match crossterm::event::read()? {
            Event::Key(k) if k.kind == KeyEventKind::Press => {
                match handle_key(k, &mut state, &tx_turn, &tx_update, &cancel, terminal)? {
                    KeyAction::Continue => {}
                    KeyAction::Exit => break 'outer,
                }
            }
            _ => {}
        }
    }

    let _ = tx_turn.send(TurnRequest::Shutdown);
    drop(tx_turn);
    let _ = worker_handle.join();
    Ok(state.history)
}

enum KeyAction {
    Continue,
    Exit,
}

fn handle_key(
    k: KeyEvent,
    state: &mut ChatState,
    tx_turn: &Sender<TurnRequest>,
    tx_update: &Sender<UiUpdate>,
    cancel: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
) -> Result<KeyAction> {
    // Ctrl+C / Ctrl+D semantics, matching the hint line:
    // - mid-generate → only cancel the in-flight turn, stay in the
    //   TUI. The worker observes `cancel` between tokens, returns
    //   `Cancelled`, and `state.generating` flips to false on the
    //   next `UiUpdate::Done` drain.
    // - mid-image-fetch → exit. `reqwest::blocking` doesn't expose a
    //   mid-read cancel, so the fetch thread keeps running until it
    //   finishes or hits its 30s timeout; the eventual `tx_update`
    //   send no-ops once the channel is dropped on exit. Treating
    //   fetch as "idle for exit purposes" gives the user a way out
    //   of a stuck download without waiting for the timeout.
    // - idle → exit the TUI cleanly.
    if k.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('d'))
    {
        if state.generating {
            cancel.store(true, Ordering::Relaxed);
            return Ok(KeyAction::Continue);
        }
        return Ok(KeyAction::Exit);
    }

    if state.generating || state.fetching_url.is_some() {
        // No editing while a turn is in flight or an image is
        // downloading; only Ctrl+C above.
        return Ok(KeyAction::Continue);
    }

    match k.code {
        KeyCode::Esc => {
            if state.input.is_empty() {
                return Ok(KeyAction::Exit);
            }
            state.input.clear();
            state.cursor = 0;
        }
        KeyCode::Enter => {
            // Shift+Enter inserts a newline (multi-line composition);
            // bare Enter submits. Some terminals don't distinguish
            // Shift+Enter from Enter at the keyboard layer; users on
            // those terminals can paste multi-line text or use Alt+Enter
            // (handled below) instead.
            if k.modifiers.contains(KeyModifiers::SHIFT) || k.modifiers.contains(KeyModifiers::ALT)
            {
                state.input.insert(state.cursor, '\n');
                state.cursor += 1;
                return Ok(KeyAction::Continue);
            }
            let line = std::mem::take(&mut state.input);
            state.cursor = 0;
            if line.trim().is_empty() {
                return Ok(KeyAction::Continue);
            }
            return dispatch_user_input(line, state, tx_turn, tx_update, terminal);
        }
        KeyCode::Backspace if state.cursor > 0 => {
            let prev = prev_char_boundary(&state.input, state.cursor);
            state.input.drain(prev..state.cursor);
            state.cursor = prev;
        }
        KeyCode::Delete if state.cursor < state.input.len() => {
            let next = next_char_boundary(&state.input, state.cursor);
            state.input.drain(state.cursor..next);
        }
        KeyCode::Left if state.cursor > 0 => {
            state.cursor = prev_char_boundary(&state.input, state.cursor);
        }
        KeyCode::Right if state.cursor < state.input.len() => {
            state.cursor = next_char_boundary(&state.input, state.cursor);
        }
        KeyCode::Home => state.cursor = 0,
        KeyCode::End => state.cursor = state.input.len(),
        KeyCode::Char(c) if !k.modifiers.contains(KeyModifiers::CONTROL) => {
            state.input.insert(state.cursor, c);
            state.cursor += c.len_utf8();
        }
        _ => {}
    }
    Ok(KeyAction::Continue)
}

fn dispatch_user_input(
    line: String,
    state: &mut ChatState,
    tx_turn: &Sender<TurnRequest>,
    tx_update: &Sender<UiUpdate>,
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
) -> Result<KeyAction> {
    if line.starts_with('/') {
        // Trim trailing whitespace before matching so commands like
        // `/help ` or `/exit\t` still dispatch — same fix as the
        // line-REPL slash dispatch (PR #125 review).
        //
        // Split on the first whitespace so commands with arguments
        // (`/save out.txt`, `/system You are ...`) dispatch on the
        // verb only.
        let trimmed = line.trim_end();
        let (cmd, rest) = match trimmed.split_once(char::is_whitespace) {
            Some((c, r)) => (c, r.trim()),
            None => (trimmed, ""),
        };
        // Trailing-arg policy mirrors the line REPL: no-arg
        // commands reject extras strictly so a typo like
        // `/clear save` doesn't silently wipe the conversation.
        if !rest.is_empty() && matches!(cmd, "/exit" | "/quit" | "/help" | "/clear") {
            emit_status_to_scrollback(terminal, &format!("{cmd} takes no arguments — type /help"))?;
            emit_blank_line(terminal)?;
            return Ok(KeyAction::Continue);
        }
        match cmd {
            "/exit" | "/quit" => return Ok(KeyAction::Exit),
            "/help" => {
                // One status line per command — keeps a wide list
                // readable on an 80-col terminal where a single
                // concatenated line wraps awkwardly.
                emit_status_to_scrollback(terminal, "Commands:")?;
                emit_status_to_scrollback(
                    terminal,
                    "  /clear           Clear conversation history (system prompt preserved)",
                )?;
                emit_status_to_scrollback(
                    terminal,
                    "  /system <text>   Replace (or set) the system prompt; empty arg removes it",
                )?;
                emit_status_to_scrollback(
                    terminal,
                    "  /save <path>     Save the conversation transcript to a file",
                )?;
                emit_status_to_scrollback(
                    terminal,
                    "  /image <path-or-url>    Attach an image (file path or http(s):// URL) to the next user turn (repeat for multi-image)",
                )?;
                emit_status_to_scrollback(terminal, "  /help            Show this help")?;
                emit_status_to_scrollback(terminal, "  /exit, /quit     Exit the REPL")?;
                emit_blank_line(terminal)?;
                return Ok(KeyAction::Continue);
            }
            "/clear" => {
                // Preserve the system prompt (if any) by truncating
                // to the first message rather than clear+clone+push.
                // The worker is stateless w.r.t. history and runs
                // `session.reset()` itself at the start of each
                // turn, so no explicit reset signal is needed.
                let had_system = state.history.first().is_some_and(|m| m.role == "system");
                if had_system {
                    state.history.truncate(1);
                    state.history_images.truncate(1);
                } else {
                    state.history.clear();
                    state.history_images.clear();
                }
                let pending_dropped = !state.pending_images.is_empty();
                state.pending_images.clear();
                emit_status_to_scrollback(
                    terminal,
                    match (had_system, pending_dropped) {
                        (true, true) => {
                            "history cleared (system prompt preserved); pending images dropped"
                        }
                        (true, false) => "history cleared (system prompt preserved)",
                        (false, true) => "history cleared; pending images dropped",
                        (false, false) => "history cleared",
                    },
                )?;
                emit_blank_line(terminal)?;
                return Ok(KeyAction::Continue);
            }
            "/save" => {
                if rest.is_empty() {
                    emit_status_to_scrollback(terminal, "usage: /save <path>")?;
                    emit_blank_line(terminal)?;
                    return Ok(KeyAction::Continue);
                }
                // "messages" is the honest unit (a "turn" is
                // colloquially user+assistant; `len()` counts the
                // system message as one entry).
                let msg = match crate::write_transcript(&state.history, Path::new(rest)) {
                    Ok(()) => format!(
                        "saved {} message{} to {rest}",
                        state.history.len(),
                        if state.history.len() == 1 { "" } else { "s" }
                    ),
                    Err(e) => format!("error: /save failed: {e}"),
                };
                emit_status_to_scrollback(terminal, &msg)?;
                emit_blank_line(terminal)?;
                return Ok(KeyAction::Continue);
            }
            "/image" => {
                // Arg is a filesystem path or `http(s)://` URL;
                // `image_source::load` resolves both with the same 50 MB
                // cap. Path reads are fast and stay synchronous; URL
                // fetches dispatch to a background thread so the UI
                // keeps rendering / responds to Ctrl+C while the fetch
                // is in flight (a misbehaving server can otherwise
                // freeze the TUI for up to the 30s timeout).
                if rest.is_empty() {
                    emit_status_to_scrollback(terminal, "usage: /image <path-or-url>")?;
                    emit_blank_line(terminal)?;
                    return Ok(KeyAction::Continue);
                }
                if !crate::image_source::looks_like_url(rest) {
                    match crate::image_source::load(rest, crate::image_source::MAX_IMAGE_BYTES) {
                        Ok(bytes) => {
                            emit_status_to_scrollback(
                                terminal,
                                &format!(
                                    "(image attached: {rest}, {} bytes; sends with next message)",
                                    bytes.len()
                                ),
                            )?;
                            emit_blank_line(terminal)?;
                            state.pending_images.push(std::sync::Arc::new(bytes));
                        }
                        Err(e) => {
                            emit_status_to_scrollback(
                                terminal,
                                &format!("error: /image {rest}: {e:#}"),
                            )?;
                            emit_blank_line(terminal)?;
                        }
                    }
                    return Ok(KeyAction::Continue);
                }

                // URL branch: serial — only one in-flight fetch at a
                // time. Concurrent fetches would race on
                // `pending_images` ordering and the
                // `state.fetching_url` lock; sequential is simpler and
                // matches typical user input cadence.
                if state.fetching_url.is_some() {
                    emit_status_to_scrollback(
                        terminal,
                        "(another image fetch is still in flight; please wait)",
                    )?;
                    emit_blank_line(terminal)?;
                    return Ok(KeyAction::Continue);
                }
                emit_status_to_scrollback(terminal, &format!("(downloading {rest}...)"))?;
                emit_blank_line(terminal)?;
                let url = rest.to_string();
                state.fetching_url = Some(url.clone());
                let tx = tx_update.clone();
                std::thread::spawn(move || {
                    let result =
                        crate::image_source::load(&url, crate::image_source::MAX_IMAGE_BYTES)
                            .map_err(|e| format!("{e:#}"));
                    // Errors here mean the UI dropped the receiver
                    // (clean exit) — drop the result silently.
                    let _ = tx.send(UiUpdate::ImageFetched { url, result });
                });
                return Ok(KeyAction::Continue);
            }
            "/system" => {
                // Empty arg removes the system message (the
                // user-facing off-switch). Non-empty arg replaces
                // or inserts at index 0. `state.pending` is
                // already `None` here (the worker can't be
                // generating — `handle_key` blocks all input
                // dispatch while `state.generating` is true), so
                // we don't touch it.
                if rest.is_empty() {
                    let removed = state.history.first().is_some_and(|m| m.role == "system");
                    if removed {
                        state.history.remove(0);
                        state.history_images.remove(0);
                        emit_status_to_scrollback(terminal, "system prompt removed")?;
                    } else {
                        emit_status_to_scrollback(terminal, "no system prompt was set")?;
                    }
                    emit_blank_line(terminal)?;
                    return Ok(KeyAction::Continue);
                }
                let new_msg = ChatMessage {
                    role: "system".into(),
                    content: rest.to_string(),
                };
                if state.history.first().is_some_and(|m| m.role == "system") {
                    state.history[0] = new_msg;
                    // history_images[0] stays as empty Vec — system
                    // messages never carry image attachments.
                } else {
                    state.history.insert(0, new_msg);
                    state.history_images.insert(0, Vec::new());
                }
                emit_status_to_scrollback(terminal, "system prompt updated")?;
                emit_blank_line(terminal)?;
                return Ok(KeyAction::Continue);
            }
            other => {
                emit_status_to_scrollback(
                    terminal,
                    &format!("unknown command: {other} — type /help"),
                )?;
                emit_blank_line(terminal)?;
                return Ok(KeyAction::Continue);
            }
        }
    }

    // Real user message: emit to scrollback, push to history, kick
    // the worker with the full history (worker is stateless).
    emit_message_to_scrollback(terminal, "user", &line)?;
    state.history.push(ChatMessage {
        role: "user".into(),
        content: line,
    });
    // Move the pending images onto the new user turn's
    // history_images entry (cloning Arcs is just refcount
    // bumps). pending_images itself drains on `UiUpdate::Done`.
    state.history_images.push(state.pending_images.clone());
    state.pending = Some(String::new());
    state.generating = true;
    state.last_finish = None;
    // Clone history + history_images for the channel send. With
    // `Arc<Vec<u8>>` storage, the per-turn channel send is a
    // refcount bump per attachment rather than a deep memcpy.
    let history_for_worker = state.history.clone();
    let history_images_for_worker = state.history_images.clone();
    if tx_turn
        .send(TurnRequest::Turn {
            history: history_for_worker,
            history_images: history_images_for_worker,
        })
        .is_err()
    {
        emit_status_to_scrollback(terminal, "worker thread exited unexpectedly")?;
        // Roll back the user turn — it never reached the worker, so
        // leaving it in `history` would mis-represent the
        // conversation in the next attempt and in the returned
        // history on exit. `pending_images` stays intact so the
        // user can retry without re-attaching.
        state.history.pop();
        state.history_images.pop();
        state.generating = false;
        state.pending = None;
    }
    Ok(KeyAction::Continue)
}

// ── Worker thread ──────────────────────────────────────────────────────────

fn worker_loop(
    mut session: Session,
    tokenizer: std::sync::Arc<BpeTokenizer>,
    opts: GenerateOpts,
    rx_turn: Receiver<TurnRequest>,
    tx_update: Sender<UiUpdate>,
) {
    while let Ok(req) = rx_turn.recv() {
        let TurnRequest::Turn {
            history,
            history_images,
        } = req
        else {
            break;
        };

        // Stateless: history is the full conversation including the
        // just-pushed user turn at the end. UI is the source of
        // truth. `/clear` on UI side just sends a smaller history
        // next turn — no separate reset needed here.
        session.reset();
        let any_images = history_images.iter().any(|v| !v.is_empty());
        let prefill_outcome: Result<(), String> = if any_images {
            // Multimodal path: synthesize multimodal messages by
            // zipping history with history_images. Each user turn
            // that had attachments rebuilds as
            // `[Image*N, Text(content)?]`; turns without
            // attachments rebuild as `[Text(content)]`. Image bytes
            // flatten across all turns in document order, matching
            // the chat template's `<image>` marker walk.
            let messages: Vec<wick::tokenizer::ChatMessageMultimodal> = history
                .iter()
                .zip(history_images.iter())
                .map(|(msg, imgs)| {
                    let mut content: Vec<wick::tokenizer::ContentItem> =
                        vec![wick::tokenizer::ContentItem::Image; imgs.len()];
                    if !msg.content.is_empty() {
                        content.push(wick::tokenizer::ContentItem::Text {
                            text: msg.content.clone(),
                        });
                    }
                    wick::tokenizer::ChatMessageMultimodal {
                        role: msg.role.clone(),
                        content,
                    }
                })
                .collect();
            let images_refs: Vec<&[u8]> = history_images
                .iter()
                .flat_map(|v| v.iter().map(|a| a.as_slice()))
                .collect();
            session
                .append_chat_with_images(&messages, &images_refs, true)
                .map_err(|e| format!("append_chat_with_images failed: {e}"))
        } else {
            match wick::tokenizer::apply_chat_template(&tokenizer, &history, true) {
                Ok(formatted) => {
                    let tokens = tokenizer.encode(&formatted);
                    session
                        .append_tokens(&tokens)
                        .map_err(|e| format!("append_tokens failed: {e}"))
                }
                Err(e) => Err(format!("chat-template render failed: {e}")),
            }
        };
        if let Err(msg) = prefill_outcome {
            let _ = tx_update.send(UiUpdate::Error(msg));
            continue;
        }

        let mut sink = TuiSink {
            tokenizer: &tokenizer,
            tx: tx_update.clone(),
        };
        let summary = match session.generate(&opts, &mut sink) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx_update.send(UiUpdate::Error(format!("generate failed: {e}")));
                continue;
            }
        };
        // Drop sink so its `tx` clone is released before the final
        // send below; chunks have already been streamed to the UI
        // token-by-token via `on_text_tokens`.
        drop(sink);
        let _ = tx_update.send(UiUpdate::Done(summary.finish_reason));
    }
}

struct TuiSink<'a> {
    tokenizer: &'a BpeTokenizer,
    tx: Sender<UiUpdate>,
}

impl ModalitySink for TuiSink<'_> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        let piece = self.tokenizer.decode(tokens);
        let _ = self.tx.send(UiUpdate::Chunk(piece));
    }
    fn on_done(&mut self, _reason: FinishReason) {
        // Final `Done` carries the FinishReason and is emitted by
        // `worker_loop` after `generate` returns.
    }
}

// ── Rendering: inline viewport ─────────────────────────────────────────────

fn draw_inline(frame: &mut ratatui::Frame, state: &ChatState) {
    let area = frame.area();
    // Layout: 2 lines for input/streaming (with wrap), 1 hint line.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(1)])
        .split(area);

    draw_input_or_stream(frame, chunks[0], state);
    draw_hint(frame, chunks[1], state);
}

fn draw_input_or_stream(frame: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    use ratatui::widgets::Wrap;
    let dim = Style::default().add_modifier(Modifier::DIM);
    let bold = Style::default().add_modifier(Modifier::BOLD);
    if state.generating {
        // Show the streaming assistant text in place of the input
        // line. Wrap so long replies are visible across viewport
        // rows; auto-scroll keeps the latest content in view when
        // the wrapped + multi-line text exceeds the available rows.
        //
        // Multi-line: the model can stream `\n` mid-response (code
        // blocks, lists, paragraph breaks). `Paragraph` doesn't
        // honor `\n` inside a `Span` — it's only a row-break across
        // separate `Line` objects — so we split `pending` on `\n`
        // and emit one `Line` per source-line, matching how the
        // finalized message is rendered to scrollback. First line
        // carries the "assistant> " prefix; continuation lines
        // are indented under the content for visual alignment.
        //
        // Compute total rows by feeding the COMBINED prefix + pending
        // string through `wrapped_row_count` at `area.width`. Only
        // the first row carries the "assistant> " prefix; subsequent
        // wrapped continuation rows have the full `area.width`
        // available (ratatui's Wrap doesn't preserve indent across
        // a soft wrap). Computing on the prefixed combined string
        // at the full width matches both hard breaks (`\n`) and
        // soft wraps exactly.
        let pending = state.pending.as_deref().unwrap_or("");
        // `wrapped_height_for_message` mirrors how the message is
        // laid out below: per-source-line, with role label on row 0
        // and continuation indent on subsequent source-lines, soft
        // wrapping at `area.width`. Using it here keeps the scroll
        // offset in sync with the actual rendered Lines.
        let total_rows = wrapped_height_for_message("assistant", pending, area.width);
        let scroll = total_rows.saturating_sub(area.height);
        let label = "assistant> ";
        let indent: String = " ".repeat(label.len());
        let mut lines: Vec<Line> = Vec::new();
        let mut first = true;
        for src in pending.split('\n') {
            let mut spans = Vec::new();
            if first {
                spans.push(Span::styled(label, bold));
                first = false;
            } else {
                spans.push(Span::raw(indent.clone()));
            }
            spans.push(Span::raw(src.to_string()));
            lines.push(Line::from(spans));
        }
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0));
        frame.render_widget(para, area);
    } else {
        // Multi-line input: render as one Line per source-line. Each
        // gets the "> " prompt; continuation lines get a 2-col
        // indent so wrapped + multi-line text reads naturally. Note
        // `"".split('\n')` already yields one empty element, so we
        // always have at least one Line — no fallback push needed.
        let mut lines: Vec<Line> = Vec::new();
        for (i, src_line) in state.input.split('\n').enumerate() {
            let lead: Span = if i == 0 {
                Span::styled("> ", dim)
            } else {
                Span::raw("  ")
            };
            lines.push(Line::from(vec![lead, Span::raw(src_line.to_string())]));
        }
        // Cursor position in the unscrolled paragraph coordinate
        // space. Walk the input up to byte offset `cursor`, tracking
        // display columns via `unicode-width` so wide CJK and emoji
        // chars take 2 cells.
        //
        // Rules (matching how the rendered Lines + Wrap behave):
        // - Row 0 starts at col 2 (after the "> " prompt span).
        // - A '\n' starts a new source-line; the next row begins at
        //   col 2 (the "  " continuation indent).
        // - A SOFT wrap (col + w > area.width within one source-
        //   line) starts a new row at col 0 — ratatui's Wrap does
        //   not preserve any leading indent across the wrap.
        let mut row: u16 = 0;
        let mut col: u16 = 2; // first row starts after "> "
        for ch in state.input[..state.cursor].chars() {
            if ch == '\n' {
                row = row.saturating_add(1);
                col = 2;
                continue;
            }
            let w = ch.width().unwrap_or(0) as u16;
            if col + w > area.width && area.width > 0 {
                row = row.saturating_add(1);
                col = 0;
            }
            col += w;
        }
        // Vertical scroll: if the cursor row sits below the viewport,
        // scroll the Paragraph so the cursor row is the LAST visible
        // row. Without this, multi-line input wider than 2 source
        // rows or anything that wraps past the inline pane would
        // hide the caret. The same offset is subtracted from the
        // rendered cursor row so it stays inside `area`.
        let scroll_y = row.saturating_sub(area.height.saturating_sub(1));
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll_y, 0));
        frame.render_widget(para, area);

        let mut cursor_row = row.saturating_sub(scroll_y);
        let mut cursor_col = col;
        if cursor_row >= area.height {
            cursor_row = area.height.saturating_sub(1);
            cursor_col = area.width.saturating_sub(1);
        } else if cursor_col >= area.width {
            cursor_col = area.width.saturating_sub(1);
        }
        frame.set_cursor_position((
            area.x.saturating_add(cursor_col),
            area.y.saturating_add(cursor_row),
        ));
    }
}

/// Compute how many rows a wrapped string takes inside `width`
/// columns, treating each '\n' as a hard break and each `\u{...}`
/// width as `unicode-width` reports. Used to size the
/// streaming-pane scroll offset.
fn wrapped_row_count(s: &str, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    let mut rows: u16 = 1;
    let mut col: u16 = 0;
    for ch in s.chars() {
        if ch == '\n' {
            rows = rows.saturating_add(1);
            col = 0;
            continue;
        }
        let w = ch.width().unwrap_or(0) as u16;
        if col + w > width {
            rows = rows.saturating_add(1);
            col = 0;
        }
        col += w;
    }
    rows
}

fn draw_hint(frame: &mut ratatui::Frame, area: Rect, state: &ChatState) {
    let dim = Style::default().add_modifier(Modifier::DIM);
    let text = if state.generating {
        String::from("generating… Ctrl+C to cancel")
    } else if let Some(url) = &state.fetching_url {
        format!("downloading {url}… Ctrl+C to exit")
    } else if let Some(reason) = &state.last_finish {
        format!("↵ send · /help · Ctrl+C to exit  ·  last turn: {reason:?}")
    } else {
        String::from("↵ send · /help · Ctrl+C to exit")
    };
    frame.render_widget(Paragraph::new(Line::from(Span::styled(text, dim))), area);
}

// ── Scrollback emission ────────────────────────────────────────────────────
//
// `Terminal::insert_before` writes lines into the scrollback above
// the inline viewport. Used to commit user/assistant turns and
// status messages so they persist outside the redraw cycle.

fn emit_message_to_scrollback(
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
    role: &str,
    content: &str,
) -> Result<()> {
    use ratatui::widgets::{Widget, Wrap};
    let term_width = terminal.size()?.width.max(1);
    let lines = render_message_lines(role, content);
    // Pre-compute the wrapped row count so `insert_before` reserves
    // enough rows for the Paragraph's wrap output. Without this we'd
    // either truncate (set_line into a 1-row buffer per source-line,
    // the previous bug) or over-/under-allocate.
    let height = wrapped_height_for_message(role, content, term_width).max(1);
    terminal.insert_before(height, |buf| {
        let para = Paragraph::new(lines).wrap(Wrap { trim: false });
        para.render(buf.area, buf);
    })?;
    Ok(())
}

/// Total wrapped row count for a `role + content` pair given a
/// terminal `width`. Mirrors how `Paragraph::wrap` lays the message
/// out: the role label sits at the start of source-line 0, every
/// other source-line gets a continuation indent of the label's
/// width (so wrapped/newline content aligns under the first line's
/// content), and each row breaks at `width` columns measured by
/// `unicode-width`.
fn wrapped_height_for_message(role: &str, content: &str, width: u16) -> u16 {
    let label = format!("{role}> ");
    let indent: String = " ".repeat(label.len());
    let mut total: u16 = 0;
    let mut first = true;
    for src in content.split('\n') {
        let prefix = if first {
            label.as_str()
        } else {
            indent.as_str()
        };
        first = false;
        let combined: String = format!("{prefix}{src}");
        total = total.saturating_add(wrapped_row_count(&combined, width));
    }
    total
}

fn emit_status_to_scrollback(
    terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>,
    text: &str,
) -> Result<()> {
    use ratatui::widgets::{Widget, Wrap};
    let term_width = terminal.size()?.width.max(1);
    let dim = Style::default().add_modifier(Modifier::DIM);
    let line = Line::from(vec![Span::raw("· "), Span::styled(text.to_string(), dim)]);
    // Compute wrapped height up-front so `insert_before` reserves
    // enough rows. Mirrors `emit_message_to_scrollback`'s pattern —
    // the previous `set_line` call truncated long status / help
    // text in the user's scrollback because it doesn't wrap.
    let combined = format!("· {text}");
    let height = wrapped_row_count(&combined, term_width).max(1);
    terminal.insert_before(height, |buf| {
        let para = Paragraph::new(line).wrap(Wrap { trim: false });
        para.render(buf.area, buf);
    })?;
    Ok(())
}

fn emit_blank_line(terminal: &mut Terminal<CrosstermBackend<&mut Stdout>>) -> Result<()> {
    terminal.insert_before(1, |_| {})?;
    Ok(())
}

fn render_message_lines(role: &str, content: &str) -> Vec<Line<'static>> {
    let label_style = match role {
        "user" => Style::default().add_modifier(Modifier::BOLD),
        "assistant" => Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC),
        "system" => Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC),
        _ => Style::default(),
    };
    let label = format!("{role}> ");
    // Indent continuation lines by the label's display width so
    // multi-line messages line up under the first line's content.
    // Role labels are ASCII so byte-len == column-width here; wider
    // role names would still be safe under this assumption since
    // ratatui will lay the indent out as that many cells regardless
    // of how `width()` would measure them.
    let indent: String = " ".repeat(label.len());
    let mut lines = Vec::new();
    let mut first = true;
    for source_line in content.split('\n') {
        let mut spans = Vec::new();
        if first {
            spans.push(Span::styled(label.clone(), label_style));
            first = false;
        } else {
            spans.push(Span::raw(indent.clone()));
        }
        spans.push(Span::raw(source_line.to_string()));
        lines.push(Line::from(spans));
    }
    // `"".split('\n')` always yields one element so `lines` is never
    // empty here; no fallback push required.
    lines
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn prev_char_boundary(s: &str, byte: usize) -> usize {
    let mut i = byte.saturating_sub(1);
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn next_char_boundary(s: &str, byte: usize) -> usize {
    let mut i = byte + 1;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}
