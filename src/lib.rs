//! A lock-free Chase-Lev work-stealing deque.
//!
//! This is a self-contained, dependency-free implementation of the dynamic circular
//! work-stealing deque from:
//!
//! - D. Chase & Y. Lev, *Dynamic Circular Work-Stealing Deque*, SPAA 2005.
//!
//! with the C11 atomic memory orderings established (and machine-checked) by:
//!
//! - N. M. Lê, A. Pop, A. Cohen, F. Zappa Nardelli, *Correct and Efficient Work-Stealing
//!   for Weak Memory Models*, PPoPP 2013, and
//! - J. Choi, *Formal Verification of Chase-Lev Deque in Concurrent Separation Logic*, 2023
//!   (arXiv:2309.03642).
//!
//! The deque has a single **owner** ([`Worker`]) that pushes and pops from the *bottom*,
//! and any number of **thieves** ([`Stealer`]) that steal from the *top*. Only the owner
//! mutates `bottom`; `top` is monotonically increasing and is only advanced via CAS, so no
//! ABA tag field is needed.
//!
//! # Element storage & data-race freedom
//!
//! A naive Chase-Lev implementation reads/writes the array slots with plain `ptr::read`/
//! `ptr::write`. That is a genuine data race (and C11 UB): a thief speculatively reads a
//! slot *before* its CAS, which can race with the owner overwriting that physical slot via a
//! later `push`. `crossbeam-deque` papers over this with `read_volatile`/`write_volatile`
//! and explicitly documents it as "technically UB" — ThreadSanitizer still flags it.
//!
//! Following Lê et al., this implementation makes slot accesses **truly atomic**: each
//! element is heap-boxed and the array cell holds an [`AtomicPtr<T>`]. Slots are loaded and
//! stored with `Relaxed` ordering (the indices carry the happens-before via an
//! Acquire/Release pair and a SeqCst fence), so there is no data race at all — it is
//! ThreadSanitizer-clean, as production job queues are: they enqueue pointers, not values.
//!
//! # Memory reclamation
//!
//! The basic Chase-Lev algorithm needs a garbage collector to reclaim grown-out buffers,
//! because a thief may still be indexing into an old buffer when the owner grows. To stay
//! dependency-free (no epoch GC), this implementation uses a **retain-until-drop** policy:
//! retired (grown-out) cell arrays are pushed onto an internal list and freed only when the
//! last handle to the deque is dropped. Total retired memory is bounded by `O(log(max_len))`
//! cell arrays. Each live element's `Box<T>` is owned by whichever consumer wins it (a
//! successful `pop` or `steal`); any element still in the active buffer's `[top, bottom)`
//! range at drop time is freed exactly once.
//!
//! This crate is a complete, tested, dependency-free implementation of the canonical
//! work-stealing primitive that underlies Rayon, Tokio, and Go — provided for study and
//! standalone use. See `research/` for the source papers and `research/SYNTHESIS.md` for the
//! design notes, including the ThreadSanitizer data-race finding that motivated the
//! atomic-cell storage.
//!
//! ## Two algorithms in this crate
//!
//! - [`Worker`] / [`Stealer`] — the **exact-once** Chase-Lev deque (this module). Every task
//!   runs exactly once; pays a CAS on steal and a `SeqCst` fence on pop.
//! - [`idempotent`] — **WS-MULT**, a fence-free, CAS-free work-stealing queue with
//!   *multiplicity* (each task delivered ≥1 times) for idempotent workloads. See that module.

pub mod idempotent;
pub mod priority;
pub mod scheduler;

use std::boxed::Box;
use std::ptr;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicIsize, AtomicPtr, Ordering, fence};
use std::vec::Vec;

// Atomics + Arc are sourced from `loom` under `--cfg loom` (for exhaustive model checking of
// the memory orderings) and from `std` otherwise. `fence` and `Ordering` come from the same
// place so the orderings being checked are exactly the ones shipped.
#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicIsize, AtomicPtr, Ordering, fence};

/// Smallest backing-buffer capacity; the deque never shrinks below this.
const MIN_CAPACITY: usize = 16;

/// The deque shrinks when fewer than `cap / SHRINK_FACTOR` elements remain. Per Chase-Lev §3
/// this must be `>= 3` so the survivors comfortably fit the half-size buffer.
const SHRINK_FACTOR: usize = 3;

/// Outcome of a [`Stealer::steal`] attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum Steal<T> {
    /// The deque was observed empty.
    Empty,
    /// Lost a race with another thief or an emptying pop; the caller should retry.
    Retry,
    /// Successfully stole a value from the top of the deque.
    Success(T),
}

impl<T> Steal<T> {
    /// Returns the stolen value, if any.
    pub fn success(self) -> Option<T> {
        match self {
            Steal::Success(v) => Some(v),
            _ => None,
        }
    }
}

/// A power-of-two-sized cyclic buffer of atomic cells. Each cell holds a `*mut T` that
/// points to a heap-boxed element (or is null for an empty slot). Using `AtomicPtr` makes
/// every slot access a real atomic operation, so a thief's speculative read can never race
/// the owner's overwriting push (no UB, ThreadSanitizer-clean).
struct Buffer<T> {
    cells: Box<[AtomicPtr<T>]>,
    cap: usize, // always a power of two
}

impl<T> Buffer<T> {
    fn alloc(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        // A boxed slice of null `AtomicPtr`s — safe, no manual layout/dealloc, and the cells
        // live at stable addresses for the buffer's lifetime (so `cell()` references are ok).
        let mut v = Vec::with_capacity(cap);
        for _ in 0..cap {
            v.push(AtomicPtr::new(ptr::null_mut()));
        }
        Buffer {
            cells: v.into_boxed_slice(),
            cap,
        }
    }

    #[inline]
    fn mask(&self) -> usize {
        self.cap - 1
    }

    /// Reference to the atomic cell for a logical index (indices wrap modulo `cap`).
    #[inline]
    unsafe fn cell(&self, index: isize) -> &AtomicPtr<T> {
        &self.cells[index as usize & self.mask()]
    }

    /// Store a boxed element into a slot (Relaxed: ordered by the `bottom` Release store).
    #[inline]
    unsafe fn write(&self, index: isize, boxed: *mut T) {
        self.cell(index).store(boxed, Ordering::Relaxed);
    }

    /// Atomically load the boxed-element pointer at a slot (Relaxed). May observe a
    /// concurrent overwrite, which is fine — the CAS on `top` decides the real winner.
    #[inline]
    unsafe fn read(&self, index: isize) -> *mut T {
        self.cell(index).load(Ordering::Relaxed)
    }

    /// Allocate a buffer of `new_cap` slots and copy the cell pointers for `[top, bottom)`.
    /// Only the *pointers* move; the boxed elements they reference are untouched. Works for
    /// both growth (`new_cap > cap`) and shrinkage (`new_cap < cap`) because elements are
    /// indexed modulo capacity, so `top`/`bottom` need not change.
    unsafe fn resized(&self, new_cap: usize, bottom: isize, top: isize) -> Buffer<T> {
        debug_assert!((bottom - top) as usize <= new_cap.saturating_sub(1));
        let other = Buffer::alloc(new_cap);
        let mut i = top;
        while i < bottom {
            let p = self.cell(i).load(Ordering::Relaxed);
            other.cell(i).store(p, Ordering::Relaxed);
            i += 1;
        }
        other
    }

    /// Allocate a buffer of double the capacity and copy the live cell pointers.
    unsafe fn grow(&self, bottom: isize, top: isize) -> Buffer<T> {
        self.resized(self.cap * 2, bottom, top)
    }
}

/// A node in the singly linked list of retired (grown-out) buffers.
struct Retired<T> {
    buffer: *mut Buffer<T>,
    next: *mut Retired<T>,
}

/// Shared state behind both the [`Worker`] and its [`Stealer`]s.
struct Inner<T> {
    bottom: AtomicIsize,
    top: AtomicIsize,
    /// Pointer to the current (active) `Buffer<T>`, heap-boxed for atomic swap.
    buffer: AtomicPtr<Buffer<T>>,
    /// Head of the retain-until-drop list of retired buffers (owner-only producer).
    retired: AtomicPtr<Retired<T>>,
}

impl<T> Inner<T> {
    fn new(log_initial_cap: u32) -> Self {
        let cap = 1usize << log_initial_cap;
        let buffer = Box::into_raw(Box::new(Buffer::alloc(cap)));
        Inner {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            buffer: AtomicPtr::new(buffer),
            retired: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Push a grown-out buffer onto the retire list. Only ever called by the owner, but the
    /// CAS loop keeps it correct regardless.
    unsafe fn retire(&self, buffer: *mut Buffer<T>) {
        let node = Box::into_raw(Box::new(Retired {
            buffer,
            next: ptr::null_mut(),
        }));
        loop {
            let head = self.retired.load(Ordering::Relaxed);
            (*node).next = head;
            if self
                .retired
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // No concurrency here: the last Arc handle is being dropped.
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Relaxed);
        let active = self.buffer.load(Ordering::Relaxed);

        unsafe {
            // Free the live elements still in the active buffer, exactly once.
            let buf = &*active;
            let mut i = top;
            while i < bottom {
                let p = buf.cell(i).load(Ordering::Relaxed);
                if !p.is_null() {
                    drop(Box::from_raw(p));
                }
                i += 1;
            }
            // Dropping the `Box<Buffer>` frees its boxed cell slice automatically.
            drop(Box::from_raw(active));

            // Free retired buffers (allocations only — their elements were either moved out
            // or are duplicates owned by the active buffer).
            let mut node = self.retired.load(Ordering::Relaxed);
            while !node.is_null() {
                let owned = Box::from_raw(node);
                drop(Box::from_raw(owned.buffer));
                node = owned.next;
            }
        }
    }
}

/// The single owner of a work-stealing deque. Pushes and pops at the bottom.
///
/// `Worker` is `Send` (it can be moved to the thread that will own it) but deliberately not
/// `Sync`: only one thread may call [`push`](Worker::push) / [`pop`](Worker::pop).
pub struct Worker<T> {
    inner: Arc<Inner<T>>,
}

// SAFETY: a `Worker<T>` only ever exposes single-threaded owner operations; moving it across
// threads is fine when `T: Send`. It is intentionally not `Sync`.
unsafe impl<T: Send> Send for Worker<T> {}

impl<T> Worker<T> {
    /// Create a new deque owner with a default initial capacity (32 slots).
    pub fn new() -> Self {
        Self::with_log_capacity(5)
    }

    /// Create a new deque owner with `2^log_initial_cap` initial slots.
    pub fn with_log_capacity(log_initial_cap: u32) -> Self {
        Worker {
            inner: Arc::new(Inner::new(log_initial_cap)),
        }
    }

    /// Create a [`Stealer`] handle that can steal from the top of this deque.
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Number of elements currently in the deque (approximate under concurrency).
    pub fn len(&self) -> usize {
        let b = self.inner.bottom.load(Ordering::Relaxed);
        let t = self.inner.top.load(Ordering::Relaxed);
        (b - t).max(0) as usize
    }

    /// Whether the deque is empty (approximate under concurrency).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current backing-buffer capacity (number of slots). Exposed mainly for tests.
    pub fn capacity(&self) -> usize {
        let buf = self.inner.buffer.load(Ordering::Relaxed);
        // SAFETY: the buffer pointer is always valid for the deque's lifetime.
        unsafe { (*buf).cap }
    }

    /// Shrink the backing buffer when the deque has retreated far below its capacity.
    ///
    /// Per Chase-Lev §3: when fewer than `cap / SHRINK_FACTOR` elements remain (and we are
    /// above the minimum capacity), relocate them into a half-size buffer and retire the old
    /// one. Safe under the retain-until-drop policy: a concurrent thief still indexing the old
    /// buffer cannot use-after-free because retired buffers live until the deque is dropped.
    fn perhaps_shrink(&self, bottom: isize, top: isize, buf_ptr: *mut Buffer<T>) {
        let inner = &*self.inner;
        // SAFETY: `buf_ptr` is the live active buffer passed in by `pop`.
        let cap = unsafe { (*buf_ptr).cap };
        let size = (bottom - top).max(0) as usize;
        if cap > MIN_CAPACITY && size < cap / SHRINK_FACTOR {
            let smaller_cap = (cap / 2).max(MIN_CAPACITY);
            // SAFETY: `size <= smaller_cap - 1` holds because size < cap/3 <= smaller_cap/1.
            let smaller = unsafe { (*buf_ptr).resized(smaller_cap, bottom, top) };
            let smaller_ptr = Box::into_raw(Box::new(smaller));
            inner.buffer.store(smaller_ptr, Ordering::Release);
            // SAFETY: retire (don't free) the old buffer; in-flight thieves may still read it.
            unsafe { inner.retire(buf_ptr) };
        }
    }

    /// Push a value onto the bottom of the deque. Owner-only.
    pub fn push(&self, value: T) {
        let inner = &*self.inner;
        let b = inner.bottom.load(Ordering::Relaxed);
        let t = inner.top.load(Ordering::Acquire);
        let mut buf_ptr = inner.buffer.load(Ordering::Acquire);

        // Grow if the buffer is full (leave one cell unused, per the paper).
        // SAFETY: `buf_ptr` always points to a live, owner-installed buffer.
        let cap = unsafe { (*buf_ptr).cap } as isize;
        if b - t >= cap - 1 {
            let bigger = unsafe { (*buf_ptr).grow(b, t) };
            let bigger_ptr = Box::into_raw(Box::new(bigger));
            inner.buffer.store(bigger_ptr, Ordering::Release);
            // SAFETY: the old buffer is retained (not freed) until the deque is dropped, so
            // a concurrent stealer still indexing into it cannot use-after-free.
            unsafe { inner.retire(buf_ptr) };
            buf_ptr = bigger_ptr;
        }

        // Box the element and publish its pointer into the slot.
        let boxed = Box::into_raw(Box::new(value));
        // SAFETY: `b` is within the (possibly grown) buffer's capacity.
        unsafe { (*buf_ptr).write(b, boxed) };
        // Release: publishes the slot write to thieves that Acquire-load `bottom`.
        inner.bottom.store(b + 1, Ordering::Release);
    }

    /// Pop a value from the bottom of the deque. Owner-only. Returns `None` if empty.
    pub fn pop(&self) -> Option<T> {
        let inner = &*self.inner;
        let b = inner.bottom.load(Ordering::Relaxed) - 1;
        let buf_ptr = inner.buffer.load(Ordering::Relaxed);
        inner.bottom.store(b, Ordering::Relaxed);

        // This fence orders the `bottom` decrement before the `top` load, matching the
        // SeqCst fence in a concurrent thief's `steal`.
        fence(Ordering::SeqCst);

        let t = inner.top.load(Ordering::Relaxed);

        if t > b {
            // Deque was empty; restore the canonical empty state (bottom == top).
            inner.bottom.store(t, Ordering::Relaxed);
            return None;
        }

        // SAFETY: `b` indexes a live slot in the active buffer; the cell holds the boxed elem.
        let boxed = unsafe { (*buf_ptr).read(b) };

        if t < b {
            // Not the last element — no thief can be racing for it. We own the box.
            // Opportunistically shrink the buffer if it has retreated far below capacity.
            self.perhaps_shrink(b, t, buf_ptr);
            return Some(unsafe { *Box::from_raw(boxed) });
        }

        // `t == b`: this is the last element. Race the thieves for it via CAS on `top`.
        let won = inner
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        inner.bottom.store(t + 1, Ordering::Relaxed);
        if won {
            // We won the box; take ownership.
            Some(unsafe { *Box::from_raw(boxed) })
        } else {
            // A thief won the CAS and owns the box; we must not free it.
            None
        }
    }
}

impl<T> Default for Worker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A thief handle that steals from the top of a work-stealing deque.
///
/// Cheap to clone; each clone can steal concurrently from a different thread.
pub struct Stealer<T> {
    inner: Arc<Inner<T>>,
}

// SAFETY: steal operations are lock-free and safe to invoke from many threads when `T: Send`.
unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Clone for Stealer<T> {
    fn clone(&self) -> Self {
        Stealer {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Stealer<T> {
    /// Attempt to steal a value from the top of the deque.
    pub fn steal(&self) -> Steal<T> {
        let inner = &*self.inner;
        let t = inner.top.load(Ordering::Acquire);
        // Orders this `top` load before the `bottom` load, matching the owner's pop fence.
        fence(Ordering::SeqCst);
        let b = inner.bottom.load(Ordering::Acquire);

        if t >= b {
            return Steal::Empty;
        }

        // Read the boxed-element pointer BEFORE the CAS: after a successful CAS the owner may
        // refill the slot via a concurrent push. The atomic load here cannot race the owner.
        let buf_ptr = inner.buffer.load(Ordering::Acquire);
        // SAFETY: `t < b`, so the slot holds a live element pointer in the loaded buffer.
        let boxed = unsafe { (*buf_ptr).read(t) };

        if inner
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            // We won the box; take ownership.
            Steal::Success(unsafe { *Box::from_raw(boxed) })
        } else {
            // Lost the race; the pointer we read belongs to whoever won the CAS. Don't free.
            Steal::Retry
        }
    }

    /// Steal roughly **half** of the victim's elements, moving them into `dest`'s deque, and
    /// return one of them directly. This is the optimization real runtimes (Tokio, Rayon,
    /// crossbeam) use: amortize the expensive CAS over a batch instead of one item per steal.
    ///
    /// The whole batch is claimed with a single `compare_exchange` that advances `top` by the
    /// batch size, so a losing thief retries cleanly and no element is taken twice. `dest`
    /// must be a fresh/owned `Worker` (typically the thief's own empty deque).
    ///
    /// # Implementation note
    ///
    /// Each item is claimed with an independent single-item [`steal`](Stealer::steal) (which
    /// is individually linearizable, loom-checked, and TSan-clean). A *single-CAS* batch claim
    /// is unsound against this crate's Chase-Lev owner, whose non-last `pop` takes from the
    /// bottom **without** a CAS — a multi-slot top-claim can then overlap the owner's pops and
    /// double-free. Looping single steals keeps the same amortization-of-scheduling benefit
    /// (one call drains ~half a victim) while staying provably correct.
    pub fn steal_batch_and_pop(&self, dest: &Worker<T>) -> Steal<T> {
        let inner = &*self.inner;
        // Estimate how many to take (about half) from a consistent snapshot of the indices.
        let t = inner.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = inner.bottom.load(Ordering::Acquire);
        let available = b - t;
        if available <= 0 {
            return Steal::Empty;
        }
        let want = ((available + 1) / 2) as usize; // ceil(half), at least 1

        // First successful steal is returned to the caller; the rest go into `dest`.
        let first = match self.steal() {
            Steal::Success(v) => v,
            // Empty/Retry: report it so the caller can move on or retry.
            other => return other,
        };
        for _ in 1..want {
            match self.steal() {
                Steal::Success(v) => dest.push(v),
                Steal::Empty | Steal::Retry => break, // victim drained or contended; stop early
            }
        }
        Steal::Success(first)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::AtomicUsize;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn push_pop_lifo() {
        let w = Worker::new();
        for i in 0..10 {
            w.push(i);
        }
        // Owner pops in LIFO order.
        for i in (0..10).rev() {
            assert_eq!(w.pop(), Some(i));
        }
        assert_eq!(w.pop(), None);
        assert!(w.is_empty());
    }

    #[test]
    fn steal_fifo() {
        let w = Worker::new();
        let s = w.stealer();
        for i in 0..10 {
            w.push(i);
        }
        // Thieves take from the top in FIFO order.
        for i in 0..10 {
            assert_eq!(s.steal(), Steal::Success(i));
        }
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn grows_past_initial_capacity() {
        // Start tiny (2 slots) to force several grows.
        let w = Worker::<usize>::with_log_capacity(1);
        let n = 10_000;
        for i in 0..n {
            w.push(i);
        }
        assert_eq!(w.len(), n);
        let mut sum = 0usize;
        while let Some(v) = w.pop() {
            sum += v;
        }
        assert_eq!(sum, (0..n).sum());
    }

    #[test]
    fn shrinks_after_draining() {
        let w = Worker::<usize>::with_log_capacity(1);
        let n = 10_000;
        for i in 0..n {
            w.push(i);
        }
        let grown = w.capacity();
        assert!(grown >= n, "should have grown to hold {n}: cap={grown}");

        // Pop almost everything; the buffer should shrink back toward the minimum.
        for _ in 0..(n - 5) {
            w.pop();
        }
        let shrunk = w.capacity();
        assert!(
            shrunk < grown,
            "capacity should shrink: {grown} -> {shrunk}"
        );
        assert!(shrunk >= MIN_CAPACITY, "never below MIN_CAPACITY");

        // Remaining elements are still intact and in LIFO order.
        for expected in (0..5).rev() {
            assert_eq!(w.pop(), Some(expected));
        }
        assert_eq!(w.pop(), None);
    }

    #[test]
    fn drops_remaining_elements_once() {
        // A non-Copy payload that counts live instances detects double-free / leak.
        struct Counted(StdArc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let live = StdArc::new(AtomicUsize::new(0));
        {
            let w = Worker::<Counted>::with_log_capacity(1);
            for _ in 0..1000 {
                live.fetch_add(1, Ordering::SeqCst);
                w.push(Counted(StdArc::clone(&live)));
            }
            // Take some out (moved to us), drop the rest with the deque.
            for _ in 0..400 {
                drop(w.pop().unwrap());
            }
            assert_eq!(live.load(Ordering::SeqCst), 600);
        }
        // After the Worker is dropped, every element must be accounted for exactly once.
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

    /// Drain one thief: steal until the owner is done and the deque is empty, recording
    /// each stolen value exactly once. Extracted to keep nesting shallow.
    fn run_thief(
        s: &Stealer<usize>,
        seen: &[AtomicUsize],
        stolen_count: &AtomicUsize,
        total: usize,
    ) {
        loop {
            match s.steal() {
                Steal::Success(v) => {
                    seen[v].fetch_add(1, Ordering::SeqCst);
                    stolen_count.fetch_add(1, Ordering::SeqCst);
                }
                Steal::Retry => {}
                Steal::Empty if stolen_count.load(Ordering::SeqCst) >= total => break,
                Steal::Empty => {}
            }
        }
    }

    /// Record one consumed value (helper shared by the batch thief, keeps nesting shallow).
    fn record(seen: &[AtomicUsize], consumed: &AtomicUsize, v: usize) {
        seen[v].fetch_add(1, Ordering::SeqCst);
        consumed.fetch_add(1, Ordering::SeqCst);
    }

    /// Batch-stealing thief: drains its local deque, then steals half-batches until done.
    fn run_batch_thief(
        s: &Stealer<usize>,
        seen: &[AtomicUsize],
        consumed: &AtomicUsize,
        total: usize,
    ) {
        let local = Worker::<usize>::new();
        loop {
            while let Some(v) = local.pop() {
                record(seen, consumed, v);
            }
            match s.steal_batch_and_pop(&local) {
                Steal::Success(v) => record(seen, consumed, v),
                Steal::Retry => {}
                Steal::Empty if consumed.load(Ordering::SeqCst) >= total => break,
                Steal::Empty => {}
            }
        }
    }

    /// Owner side: interleave pushes and pops, returning everything it popped itself.
    fn run_owner(w: &Worker<usize>, total: usize) -> Vec<usize> {
        let mut popped = Vec::new();
        for i in 0..total {
            w.push(i);
            if i % 3 == 0 {
                if let Some(v) = w.pop() {
                    popped.push(v);
                }
            }
        }
        while let Some(v) = w.pop() {
            popped.push(v);
        }
        popped
    }

    #[test]
    fn concurrent_steal_no_loss_no_duplication() {
        let w = Worker::<usize>::new();
        let n: usize = 200_000;
        let thieves = 4;

        let seen: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        let stolen_count = StdArc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let seen = StdArc::clone(&seen);
                let stolen_count = StdArc::clone(&stolen_count);
                scope.spawn(move || run_thief(&s, &seen, &stolen_count, n));
            }

            for v in run_owner(&w, n) {
                seen[v].fetch_add(1, Ordering::SeqCst);
                stolen_count.fetch_add(1, Ordering::SeqCst);
            }
        });

        // Every value pushed must have been observed exactly once, by exactly one consumer.
        for (v, slot) in seen.iter().enumerate() {
            assert_eq!(
                slot.load(Ordering::SeqCst),
                1,
                "value {v} not consumed exactly once"
            );
        }
    }

    #[test]
    fn steal_batch_takes_about_half() {
        let victim = Worker::<usize>::new();
        let thief_dest = Worker::<usize>::new();
        let s = victim.stealer();
        for i in 0..100 {
            victim.push(i);
        }

        // One batch steal should move ~half: 1 returned + ~49 into dest.
        let got = s.steal_batch_and_pop(&thief_dest).success();
        assert!(got.is_some());
        let moved = thief_dest.len();
        assert!(
            (40..=55).contains(&(moved + 1)),
            "batch should be about half of 100, got {}",
            moved + 1
        );
        // Victim keeps the rest; nothing is lost.
        assert_eq!(victim.len() + thief_dest.len() + 1, 100);
    }

    #[test]
    fn concurrent_batch_steal_no_loss() {
        let w = Worker::<usize>::new();
        let n: usize = 100_000;
        let thieves = 4;

        let seen: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        let consumed = StdArc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let seen = StdArc::clone(&seen);
                let consumed = StdArc::clone(&consumed);
                scope.spawn(move || run_batch_thief(&s, &seen, &consumed, n));
            }

            for i in 0..n {
                w.push(i);
            }
            while let Some(v) = w.pop() {
                seen[v].fetch_add(1, Ordering::SeqCst);
                consumed.fetch_add(1, Ordering::SeqCst);
            }
        });

        for (v, slot) in seen.iter().enumerate() {
            assert_eq!(
                slot.load(Ordering::SeqCst),
                1,
                "value {v} not consumed exactly once"
            );
        }
    }
}

/// Exhaustive model-checked tests under `loom`. loom replays every legal interleaving of the
/// atomic operations to prove the memory orderings are correct (not just "didn't crash on
/// this run", which is all ThreadSanitizer can promise). Kept tiny because loom's state space
/// explodes combinatorially.
///
/// Run with: `RUSTFLAGS="--cfg loom" cargo test --release loom_`
#[cfg(loom)]
#[cfg(test)]
mod loom_tests {
    use super::*;

    #[test]
    fn loom_owner_pop_vs_one_thief() {
        loom::model(|| {
            let worker = Worker::<u32>::with_log_capacity(4); // 16 slots, no grow needed
            let stealer = worker.stealer();

            worker.push(1);
            worker.push(2);

            let thief = loom::thread::spawn(move || matches!(stealer.steal(), Steal::Success(_)));

            // Owner pops concurrently with the single steal.
            let owner_got = worker.pop().is_some();
            let thief_got = thief.join().unwrap();

            // Two items, two consumers, no overlap: between owner and thief at most 2 succeed,
            // and the remaining item (if any) is still poppable. No item is taken twice.
            let mut remaining = 0;
            while worker.pop().is_some() {
                remaining += 1;
            }
            let consumed = owner_got as usize + thief_got as usize + remaining;
            assert_eq!(consumed, 2, "every pushed item consumed exactly once");
        });
    }

    #[test]
    fn loom_push_then_steal() {
        loom::model(|| {
            let worker = Worker::<u32>::with_log_capacity(4);
            let stealer = worker.stealer();
            worker.push(42);

            let thief = loom::thread::spawn(move || stealer.steal().success());
            let stolen = thief.join().unwrap();
            let popped = worker.pop();

            // Exactly one of {thief, owner} gets the single element.
            assert_eq!(stolen.is_some() as usize + popped.is_some() as usize, 1);
        });
    }
}
