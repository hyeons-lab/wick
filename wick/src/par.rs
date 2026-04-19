//! Parallelism facade over `rayon`.
//!
//! When the `parallel` feature is enabled (the default), this module
//! re-exports `rayon`'s `prelude` + slice-iterator traits so call sites
//! can write `slice.par_chunks_mut(n).enumerate().for_each(...)` and
//! get parallel execution for free.
//!
//! When `parallel` is disabled — the target case is `wasm32-unknown-unknown`,
//! where rayon can't run single-threadedly in browsers — this module
//! provides drop-in sequential shim traits with the same method names.
//! The shims return `std::slice::Chunks[Mut]`, which already implements
//! `Iterator` with `enumerate()` / `zip()` / `for_each()` — so no call
//! site needs a `#[cfg]` branch.
//!
//! Float-reassociation note: rayon's `par_iter().reduce()` /
//! chunked-sum patterns are NOT identical to their sequential forms at
//! the ULP level (summation order differs). This matters for exact-float
//! tests. `wick` today has no reduction call sites via rayon — all
//! parallel usage is `par_chunks_mut` + in-place mutation — so
//! greedy-sampling output is bit-for-bit stable across `parallel` on/off.
//! If that changes, add explicit reduction wrappers here with documented
//! ordering guarantees.

/// Number of worker threads the runtime will use. Returns `1` when
/// `parallel` is disabled so callers who size thread-local buffers
/// (GEMM scratch, flash-attention chunks, …) degrade gracefully.
#[cfg(feature = "parallel")]
pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "parallel"))]
pub fn current_num_threads() -> usize {
    1
}

// ---------------------------------------------------------------------------
// `parallel` ON — thin re-exports of rayon's traits + prelude.
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub use rayon::slice::{ParallelSlice, ParallelSliceMut};

// ---------------------------------------------------------------------------
// `parallel` OFF — sequential shim traits with matching method names.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "parallel"))]
mod seq {
    /// Sequential drop-in for `rayon::slice::ParallelSlice`. Returns the
    /// std iterator so downstream `.enumerate()` / `.zip()` / `.for_each()`
    /// calls resolve to their `Iterator` counterparts.
    pub trait ParallelSlice<T: Sync> {
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T>;
    }

    impl<T: Sync> ParallelSlice<T> for [T] {
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T> {
            self.chunks(chunk_size)
        }
    }

    /// Sequential drop-in for `rayon::slice::ParallelSliceMut`.
    pub trait ParallelSliceMut<T: Send> {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;
    }

    impl<T: Send> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }
    }
}

#[cfg(not(feature = "parallel"))]
pub use seq::{ParallelSlice, ParallelSliceMut};

// Alias rayon's iterator traits to std's `Iterator` when `parallel` is
// off, so call-site `use crate::par::{IndexedParallelIterator, ParallelIterator}`
// keeps resolving. Method calls like `.enumerate()` / `.zip()` /
// `.for_each()` already exist on `Iterator`, so the sequential paths
// Just Work.
#[cfg(not(feature = "parallel"))]
pub use core::iter::Iterator as IndexedParallelIterator;
#[cfg(not(feature = "parallel"))]
pub use core::iter::Iterator as ParallelIterator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_num_threads_is_positive() {
        assert!(current_num_threads() >= 1);
    }

    #[test]
    fn par_chunks_mut_shape_matches_sequential() {
        // Sanity check that the facade's `par_chunks_mut` produces the
        // expected chunk layout regardless of which backend is active —
        // guards against accidentally wiring in a different chunk size
        // convention in the sequential shim.
        let mut v = vec![0i32; 8];
        v.par_chunks_mut(4)
            .enumerate()
            .for_each(|(i, chunk)| chunk.iter_mut().for_each(|x| *x = i as i32));
        assert_eq!(v, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }
}
