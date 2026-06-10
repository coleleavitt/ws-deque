//! Fence-free, CAS-free work-stealing with **multiplicity** (WS-MULT).
//!
//! Implements the WS-MULT algorithm of
//!
//! - A. Castañeda & M. Piña, *Fully Read/Write Fence-Free Work-Stealing with Multiplicity*,
//!   arXiv:2008.04424 (Fig. 3).
//!
//! # The breakthrough
//!
//! Attiya et al.'s well-known impossibility result says *exact* work-stealing (every task
//! taken exactly once) must use either a read-modify-write instruction (CAS/swap) **or** a
//! memory fence on the hot path. Chase-Lev (this crate's [`crate::Worker`]) pays with a CAS
//! on `steal`/last-`pop` and a `SeqCst` fence on every `pop`.
//!
//! WS-MULT sidesteps the impossibility by *relaxing the contract*: every task is delivered
//! **at least once** (with multiplicity bounded by the number of concurrent stealers), rather
//! than exactly once. Under that relaxation:
//!
//! - **`put` is fully read/write** — a plain store, **no CAS, no fence**.
//! - **`take`/`steal` never retry** — they read a monotone `head` (a *MaxRegister*, realized
//!   here with `fetch_max`) and a slot, with **no CAS-abort loop** like Chase-Lev's.
//!
//! The monotone `head` is the whole trick: it can only move forward, so a slow consumer can
//! never "rewind" the queue. Two consumers only ever take the same task if they read the same
//! `head` concurrently — bounding the multiplicity by the thread count.
//!
//! # When this is the right tool
//!
//! Use WS-MULT for **idempotent** workloads where executing a task more than once is wasteful
//! but not incorrect: parallel SAT/SMT solving, graph traversal / reachability, fixpoint
//! iteration, branch-and-bound, speculative search. There the lower synchronization cost
//! wins, and the rare double-execution is absorbed by an idempotency check.
//!
//! Do **not** use it where each task must run exactly once (side effects, accounting) — use
//! the exactly-once Chase-Lev [`crate::Worker`] for that.
//!
//! Because a task may be handed out more than once, the element type is `T: Clone` and
//! consumers receive clones; the master copy is dropped when the queue is dropped.

use std::boxed::Box;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::vec::Vec;

#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

/// Initial number of task slots.
const INITIAL_CAPACITY: usize = 16;

/// Outcome of an [`IdempotentStealer::steal`] (or owner [`IdempotentWorker::take`]).
#[derive(Debug, PartialEq, Eq)]
pub enum Take<T> {
    /// No task currently available at the head.
    Empty,
    /// A task (delivered *at least once* across all consumers).
    Got(T),
}

impl<T> Take<T> {
    /// The task, if any.
    pub fn got(self) -> Option<T> {
        match self {
            Take::Got(v) => Some(v),
            Take::Empty => None,
        }
    }
}

/// A growable array of atomic cells. Each cell holds a `*mut T` (a boxed task) or null (`⊥`,
/// "no task"). The owner only ever appends; consumers only ever read. The owner grows by
/// publishing a larger array via `AtomicPtr` (old arrays are retained until the queue drops,
/// so a slow thief still reading an old array can't use-after-free).
struct Tasks<T> {
    cells: Box<[AtomicPtr<T>]>,
    /// Parallel "claimed by a thief" flags, used only by the bounded-multiplicity
    /// [`IdempotentStealer::steal_exclusive`]. A thief that wins the `false→true` CAS on
    /// `claimed[head]` is the unique thief allowed to take that slot, so no two *thieves* take
    /// the same task (the owner's `take` ignores these flags — a `take` and a `steal` may still
    /// both deliver one task, which is permitted by bounded multiplicity).
    claimed: Box<[AtomicBool]>,
    len: usize,
}

impl<T> Tasks<T> {
    fn with_len(len: usize) -> Self {
        let mut v = Vec::with_capacity(len);
        let mut c = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(AtomicPtr::new(core::ptr::null_mut()));
            c.push(AtomicBool::new(false));
        }
        Tasks {
            cells: v.into_boxed_slice(),
            claimed: c.into_boxed_slice(),
            len,
        }
    }
}

struct Inner<T> {
    /// MaxRegister `head`: the index of the next task to take. Only ever moves forward.
    /// `fetch_max` realizes `MaxWrite`; `load` realizes `MaxRead`.
    head: AtomicUsize,
    /// Pointer to the active `Tasks<T>` array (swapped on growth, unless `bounded`).
    tasks: AtomicPtr<Tasks<T>>,
    /// Retired (grown-out) arrays, freed when the queue drops.
    retired: AtomicPtr<Retired<T>>,
    /// When `true` the array never grows (fixed capacity). Required for the bounded-
    /// multiplicity [`IdempotentStealer::steal_exclusive`]: per-slot claim flags can only be
    /// race-free when there is exactly **one** array for the queue's lifetime — otherwise a
    /// thief on a retired array and a thief on the grown array could both claim the same slot.
    bounded: bool,
}

struct Retired<T> {
    tasks: *mut Tasks<T>,
    next: *mut Retired<T>,
}

impl<T> Inner<T> {
    unsafe fn retire(&self, old: *mut Tasks<T>) {
        let node = Box::into_raw(Box::new(Retired {
            tasks: old,
            next: core::ptr::null_mut(),
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
        unsafe {
            // Free every distinct boxed task exactly once. Tasks live in the *active* array
            // in `[0, tail_written)`; we walk the whole active array and free non-null cells.
            let active = self.tasks.load(Ordering::Relaxed);
            let arr = &*active;
            for cell in arr.cells.iter() {
                let p = cell.load(Ordering::Relaxed);
                if !p.is_null() {
                    drop(Box::from_raw(p));
                }
            }
            drop(Box::from_raw(active));

            // Retired arrays only ever held *clones of pointers* that were copied forward into
            // the active array on growth, so we must NOT double-free their tasks — free only
            // the array allocations.
            let mut node = self.retired.load(Ordering::Relaxed);
            while !node.is_null() {
                let owned = Box::from_raw(node);
                drop(Box::from_raw(owned.tasks));
                node = owned.next;
            }
        }
    }
}

/// The single owner of a WS-MULT queue. Puts at the tail, takes from the head. Not `Sync`.
pub struct IdempotentWorker<T> {
    inner: Arc<Inner<T>>,
    /// Persistent local tail (owner-only, never shared). Number of tasks ever put.
    tail: usize,
    /// Max tasks accepted (only meaningful when bounded; `usize::MAX` when growable).
    cap: usize,
}

// SAFETY: the owner handle moves between threads only as a whole; owner ops are single-thread.
unsafe impl<T: Send> Send for IdempotentWorker<T> {}

impl<T: Clone> IdempotentWorker<T> {
    /// Create an empty, growable WS-MULT queue. Use [`steal`](IdempotentStealer::steal) /
    /// [`take`](Self::take) (unbounded multiplicity).
    pub fn new() -> Self {
        Self::make(
            INITIAL_CAPACITY.next_power_of_two().max(INITIAL_CAPACITY),
            false,
        )
    }

    /// Create a **bounded** (fixed-capacity) WS-MULT queue holding up to `capacity` tasks. The
    /// array never grows, which is what makes [`steal_exclusive`](IdempotentStealer::steal_exclusive)
    /// race-free (single array for the queue's lifetime). [`put`](Self::put) returns `false`
    /// once `capacity` tasks have been inserted.
    pub fn bounded(capacity: usize) -> Self {
        // +2 for the two trailing ⊥ sentinels the algorithm keeps; capacity is enforced
        // explicitly in `put` via the `bounded`/`cap` check, not by the array length alone.
        let mut w = Self::make(capacity + 2, true);
        w.cap = capacity;
        w
    }

    fn make(len: usize, bounded: bool) -> Self {
        let tasks = Box::into_raw(Box::new(Tasks::with_len(len)));
        IdempotentWorker {
            inner: Arc::new(Inner {
                head: AtomicUsize::new(0),
                tasks: AtomicPtr::new(tasks),
                retired: AtomicPtr::new(core::ptr::null_mut()),
                bounded,
            }),
            tail: 0,
            cap: usize::MAX,
        }
    }

    /// A cheaply-clonable thief handle.
    pub fn stealer(&self) -> IdempotentStealer<T> {
        IdempotentStealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Number of tasks ever put (the tail). Tasks taken are not subtracted — this is the
    /// high-water mark, exposed mainly for tests.
    pub fn tail(&self) -> usize {
        self.tail
    }

    /// Current backing-array capacity.
    pub fn capacity(&self) -> usize {
        // SAFETY: the active array pointer is always valid.
        unsafe { (*self.inner.tasks.load(Ordering::Relaxed)).len }
    }

    /// **Put** a task at the tail. Fully read/write: **no CAS, no fence** on the hot path
    /// (Castañeda-Piña Fig. 3, lines 1-2). A growable queue enlarges the array as needed and
    /// always returns `true`; a [`bounded`](Self::bounded) queue returns `false` (and drops the
    /// task) when the fixed array is full.
    pub fn put(&mut self, task: T) -> bool {
        let inner = &*self.inner;
        if inner.bounded && self.tail >= self.cap {
            return false; // fixed-capacity queue is full; never grow (keeps a single array)
        }
        let mut arr = inner.tasks.load(Ordering::Acquire);
        // Need slots for index `tail` and the trailing `⊥` sentinel at `tail + 1`.
        // SAFETY: `arr` is a live owner-installed array.
        if self.tail + 1 >= unsafe { (*arr).len } {
            arr = self.grow(arr);
        }
        let boxed = Box::into_raw(Box::new(task));
        // SAFETY: indices `tail` and `tail+1` are within the (possibly grown) array.
        unsafe {
            // Line 2: store the task and re-mark the next slot as ⊥. Order is irrelevant
            // (the paper writes them as an unordered set), and both are plain Relaxed stores —
            // no Release needed because `head` (the synchronizer) is touched separately and a
            // consumer that reads a slot has already observed `head`'s monotone progress.
            (*arr).cells[self.tail].store(boxed, Ordering::Release);
            (*arr).cells[self.tail + 1].store(core::ptr::null_mut(), Ordering::Relaxed);
        }
        self.tail += 1;
        true
    }

    /// Grow the task array to double length, copying existing cell pointers forward.
    fn grow(&self, old: *mut Tasks<T>) -> *mut Tasks<T> {
        let inner = &*self.inner;
        // SAFETY: `old` is the live active array.
        let old_ref = unsafe { &*old };
        // Growth only ever happens on a non-bounded queue, where `steal_exclusive` (the sole
        // user of the `claimed` flags) is forbidden — so only the task pointers need copying.
        let bigger = Tasks::with_len(old_ref.len * 2);
        for i in 0..old_ref.len {
            let p = old_ref.cells[i].load(Ordering::Relaxed);
            bigger.cells[i].store(p, Ordering::Relaxed);
        }
        let bigger_ptr = Box::into_raw(Box::new(bigger));
        inner.tasks.store(bigger_ptr, Ordering::Release);
        // SAFETY: retire (don't free) the old array; a slow thief may still index it.
        unsafe { inner.retire(old) };
        bigger_ptr
    }

    /// **Take** from the head (owner). No retry, no CAS-abort loop: read `head`, and if a task
    /// is available, read it and advance `head` via `fetch_max` (Fig. 3, lines 4-9).
    pub fn take(&self) -> Take<T> {
        let inner = &*self.inner;
        let head = inner.head.load(Ordering::Acquire);
        if head >= self.tail {
            return Take::Empty;
        }
        let arr = inner.tasks.load(Ordering::Acquire);
        // SAFETY: `head < tail <= len`, so the cell is a live, owner-written slot.
        let p = unsafe { (*arr).cells[head].load(Ordering::Acquire) };
        if p.is_null() {
            return Take::Empty;
        }
        // MaxWrite(head + 1): advance the monotone head. `fetch_max` keeps it non-decreasing
        // even if a thief raced ahead.
        inner.head.fetch_max(head + 1, Ordering::AcqRel);
        // Multiplicity: clone out; the master box stays in the array (freed at queue drop).
        // SAFETY: `p` points to a live boxed task owned by the queue.
        Take::Got(unsafe { (*p).clone() })
    }
}

impl<T: Clone> Default for IdempotentWorker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A thief handle for a WS-MULT queue. `Clone` + `Send` + `Sync`.
pub struct IdempotentStealer<T> {
    inner: Arc<Inner<T>>,
}

// SAFETY: steal is read/write + a monotone fetch_max; safe from many threads when `T: Send`.
unsafe impl<T: Send> Send for IdempotentStealer<T> {}
unsafe impl<T: Send> Sync for IdempotentStealer<T> {}

impl<T> Clone for IdempotentStealer<T> {
    fn clone(&self) -> Self {
        IdempotentStealer {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Clone> IdempotentStealer<T> {
    /// **Steal** from the head. No CAS, no retry loop (Fig. 3, lines 10-16): read `head`, read
    /// that slot; if it holds a task, advance `head` with `fetch_max` and return a clone.
    pub fn steal(&self) -> Take<T> {
        let inner = &*self.inner;
        let head = inner.head.load(Ordering::Acquire);
        let arr = inner.tasks.load(Ordering::Acquire);
        // SAFETY: `arr` is a live array pointer; bind an explicit reference to its cells before
        // indexing (avoids an implicit autoref through the raw pointer).
        let cells: &[AtomicPtr<T>] = unsafe { &(*arr).cells };
        let p = match cells.get(head) {
            Some(c) => c.load(Ordering::Acquire),
            None => return Take::Empty, // head beyond a stale array view; nothing to steal
        };
        if p.is_null() {
            return Take::Empty;
        }
        inner.head.fetch_max(head + 1, Ordering::AcqRel);
        // SAFETY: `p` points to a live boxed task owned by the queue.
        Take::Got(unsafe { (*p).clone() })
    }

    /// **Bounded-multiplicity steal** (Castañeda-Piña B-WS-MULT). Like [`steal`](Self::steal),
    /// but a thief first claims the slot with a single `false→true` CAS on a per-slot flag, so
    /// **no two thieves ever take the same task**. (A concurrent owner [`take`] may still take
    /// it once — the paper's bounded variant excludes only thief/thief duplication, not
    /// take/steal.) Returns [`Take::Empty`] when the head slot is empty or already claimed.
    ///
    /// # Panics (debug)
    ///
    /// Requires a [`bounded`](IdempotentWorker::bounded) queue. The per-slot claim flags are
    /// only race-free when the array never grows: with growth, a thief on a retired array and a
    /// thief on the grown array could both win the claim for the same logical slot. A growable
    /// queue would silently break the no-double-take guarantee, so this asserts bounded mode.
    pub fn steal_exclusive(&self) -> Take<T> {
        let inner = &*self.inner;
        debug_assert!(
            inner.bounded,
            "steal_exclusive requires IdempotentWorker::bounded (growth breaks per-slot claims)"
        );
        let head = inner.head.load(Ordering::Acquire);
        let arr = inner.tasks.load(Ordering::Acquire);
        // SAFETY: `arr` is a live array pointer.
        let cells: &[AtomicPtr<T>] = unsafe { &(*arr).cells };
        let claims: &[AtomicBool] = unsafe { &(*arr).claimed };
        let p = match cells.get(head) {
            Some(c) => c.load(Ordering::Acquire),
            None => return Take::Empty,
        };
        if p.is_null() {
            return Take::Empty;
        }
        // Claim the slot: exactly one thief wins the false→true transition.
        if claims[head]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            // Another thief already claimed this slot. Nudge head forward so the next call
            // makes progress, then report empty for this attempt.
            inner.head.fetch_max(head + 1, Ordering::AcqRel);
            return Take::Empty;
        }
        inner.head.fetch_max(head + 1, Ordering::AcqRel);
        // SAFETY: `p` points to a live boxed task owned by the queue.
        Take::Got(unsafe { (*p).clone() })
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn put_take_fifo() {
        let mut w = IdempotentWorker::new();
        for i in 0..10 {
            w.put(i);
        }
        for i in 0..10 {
            assert_eq!(w.take(), Take::Got(i)); // FIFO: head advances 0,1,2,...
        }
        assert_eq!(w.take(), Take::Empty);
    }

    #[test]
    fn steal_fifo_from_head() {
        let mut w = IdempotentWorker::new();
        let s = w.stealer();
        for i in 0..5 {
            w.put(i);
        }
        for i in 0..5 {
            assert_eq!(s.steal(), Take::Got(i));
        }
        assert_eq!(s.steal(), Take::Empty);
    }

    #[test]
    fn grows_and_preserves_order() {
        let mut w = IdempotentWorker::<usize>::new();
        let n = 5_000;
        for i in 0..n {
            w.put(i);
        }
        assert!(w.capacity() > n);
        for i in 0..n {
            assert_eq!(w.take(), Take::Got(i));
        }
        assert_eq!(w.take(), Take::Empty);
    }

    #[test]
    fn concurrent_multiplicity_bounded() {
        let mut w = IdempotentWorker::<usize>::new();
        let n = 100_000;
        let thieves = 4;
        for i in 0..n {
            w.put(i);
        }

        // Count how many times each task is delivered. The WS-MULT contract: >= 1 each, and
        // total multiplicity bounded by (1 owner + thieves) — never unbounded like idempotent.
        let counts: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let counts = StdArc::clone(&counts);
                scope.spawn(move || {
                    while let Take::Got(v) = s.steal() {
                        counts[v].fetch_add(1, StdOrdering::SeqCst);
                    }
                });
            }
            while let Take::Got(v) = w.take() {
                counts[v].fetch_add(1, StdOrdering::SeqCst);
            }
        });

        let max_consumers = thieves + 1;
        for (v, c) in counts.iter().enumerate() {
            let got = c.load(StdOrdering::SeqCst);
            assert!(got >= 1, "task {v} never taken (violates 'at least once')");
            assert!(
                got <= max_consumers,
                "task {v} taken {got} times, exceeds multiplicity bound {max_consumers}"
            );
        }
    }

    #[test]
    fn bounded_put_rejects_when_full() {
        let mut w = IdempotentWorker::<usize>::bounded(4);
        for i in 0..4 {
            assert!(w.put(i), "slot {i} within capacity");
        }
        assert!(!w.put(99), "put past capacity must return false");
        // The four accepted tasks are still retrievable in order.
        for i in 0..4 {
            assert_eq!(w.take(), Take::Got(i));
        }
        assert_eq!(w.take(), Take::Empty);
    }

    #[test]
    fn bounded_steal_no_two_thieves_same_task() {
        // With steal_exclusive and NO owner take, every task is taken by at most one thief.
        let n = 100_000;
        let thieves = 6;
        let mut w = IdempotentWorker::<usize>::bounded(n);
        for i in 0..n {
            assert!(w.put(i), "bounded queue should accept up to capacity");
        }
        let counts: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        let done = StdArc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let counts = StdArc::clone(&counts);
                let done = StdArc::clone(&done);
                scope.spawn(move || {
                    while done.load(StdOrdering::SeqCst) < n {
                        if let Take::Got(v) = s.steal_exclusive() {
                            counts[v].fetch_add(1, StdOrdering::SeqCst);
                            done.fetch_add(1, StdOrdering::SeqCst);
                        }
                    }
                });
            }
        });

        // Bounded-multiplicity (thieves only): every task taken exactly once, never twice.
        for (v, c) in counts.iter().enumerate() {
            let got = c.load(StdOrdering::SeqCst);
            assert_eq!(
                got, 1,
                "task {v} taken {got} times by thieves (must be exactly 1)"
            );
        }
    }

    #[test]
    fn no_phantom_tasks() {
        // Every delivered value must be one that was actually put (no reading uninit slots).
        let mut w = IdempotentWorker::<usize>::new();
        let n = 1000;
        for i in 0..n {
            w.put(i * 7 + 1); // distinctive values
        }
        let valid: HashSet<usize> = (0..n).map(|i| i * 7 + 1).collect();
        let s = w.stealer();
        let mut seen = 0;
        while let Take::Got(v) = s.steal() {
            assert!(valid.contains(&v), "phantom task {v} never put");
            seen += 1;
        }
        assert_eq!(seen, n);
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;

    #[test]
    fn loom_put_take_steal() {
        loom::model(|| {
            let mut w = IdempotentWorker::<u32>::new();
            w.put(1);
            w.put(2);
            let s = w.stealer();

            let thief = loom::thread::spawn(move || {
                let mut got = Vec::new();
                while let Take::Got(v) = s.steal() {
                    got.push(v);
                }
                got
            });

            let mut owner_got = Vec::new();
            while let Take::Got(v) = w.take() {
                owner_got.push(v);
            }
            let thief_got = thief.join().unwrap();

            // Multiplicity contract: every task delivered at least once across both consumers.
            for task in [1u32, 2] {
                let n = owner_got.iter().filter(|&&v| v == task).count()
                    + thief_got.iter().filter(|&&v| v == task).count();
                assert!(n >= 1, "task {task} never delivered");
                assert!(
                    n <= 2,
                    "task {task} delivered {n} times, exceeds 2 consumers"
                );
            }
        });
    }
}
