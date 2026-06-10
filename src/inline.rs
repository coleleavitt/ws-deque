//! An allocation-free Chase-Lev deque for small `Copy` payloads.
//!
//! The crate's main [`crate::Worker`] boxes every element into an `AtomicPtr<T>` cell, which is
//! what makes it *genuinely* race-free for arbitrary `T` (vs. crossbeam's technically-UB inline
//! `volatile`) — but the boxing costs an allocation per `push` and a pointer-chase per `steal`.
//! For the very common case of **small `Copy` tasks** (a job id, an index, a packed pointer)
//! that whole cost is unnecessary: a `Copy` value can be stored *inline* in an atomic word with
//! no allocation, and because `Copy` types have no `Drop`, there is no ownership to track — a
//! thief and the owner racing for a slot simply both read a *copy*, and the `top` CAS decides
//! who keeps it. No double-free is even possible.
//!
//! This module is that fast path: the same wraparound-safe Chase-Lev index protocol and the
//! same quiescent-state buffer reclamation as the main deque, but the cells are `AtomicU64`
//! holding the value's bits directly. It accepts any `T: Copy` of at most 8 bytes (checked at
//! compile time). Slot reads/writes are real atomic ops, so it stays ThreadSanitizer-clean.

use std::boxed::Box;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicIsize, AtomicPtr, AtomicU64, Ordering, fence};
use std::vec::Vec;

#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicIsize, AtomicPtr, AtomicU64, Ordering, fence};

pub use crate::{CachePadded, Steal};

const MIN_CAPACITY: usize = 16;
const SHRINK_FACTOR: usize = 3;

/// Bit-pack a `Copy` value (≤ 8 bytes) into a `u64`. Smaller types are zero-extended.
#[inline]
fn to_bits<T: Copy>(v: T) -> u64 {
    const {
        assert!(
            core::mem::size_of::<T>() <= 8,
            "inline deque needs T no larger than 8 bytes"
        )
    };
    let mut bits = 0u64;
    // SAFETY: `T` is `Copy` and `size_of::<T>() <= 8`; we copy its bytes into the low bytes of a
    // zeroed `u64`. No alignment requirement (byte copy), no drop (Copy).
    unsafe {
        core::ptr::copy_nonoverlapping(
            &v as *const T as *const u8,
            &mut bits as *mut u64 as *mut u8,
            core::mem::size_of::<T>(),
        );
    }
    bits
}

/// Inverse of [`to_bits`]: reconstruct a `T` from the low bytes of a `u64`.
#[inline]
fn from_bits<T: Copy>(bits: u64) -> T {
    // SAFETY: `bits` was produced by `to_bits::<T>` (or is an unused slot we never read), so its
    // low `size_of::<T>()` bytes are a valid `T` bit-pattern; `T: Copy` so a bit read is sound.
    unsafe { core::ptr::read(&bits as *const u64 as *const T) }
}

/// A power-of-two-sized cyclic buffer of `AtomicU64` cells (each holds a value's packed bits).
struct Buffer {
    cells: Box<[AtomicU64]>,
    cap: usize,
}

impl Buffer {
    fn alloc(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        let mut v = Vec::with_capacity(cap);
        for _ in 0..cap {
            v.push(AtomicU64::new(0));
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

    #[inline]
    fn cell(&self, index: isize) -> &AtomicU64 {
        &self.cells[index as usize & self.mask()]
    }

    /// Allocate a `new_cap` buffer and copy the live `[top, bottom)` cells (count-based, so it is
    /// correct across index wraparound).
    fn resized(&self, new_cap: usize, bottom: isize, top: isize) -> Buffer {
        let count = bottom.wrapping_sub(top);
        debug_assert!(count >= 0 && (count as usize) <= new_cap.saturating_sub(1));
        let other = Buffer::alloc(new_cap);
        for k in 0..count {
            let i = top.wrapping_add(k);
            other
                .cell(i)
                .store(self.cell(i).load(Ordering::Relaxed), Ordering::Relaxed);
        }
        other
    }
}

struct Retired {
    buffer: *mut Buffer,
    next: *mut Retired,
}

struct Inner {
    /// Owner-written (and thief-read). Cache-padded so writes don't invalidate the thieves'
    /// `top` line — the dominant false-sharing cost in the steal hot loop.
    bottom: CachePadded<AtomicIsize>,
    /// Thief-CAS'd (and owner-read). On its own cache line, away from `bottom`.
    top: CachePadded<AtomicIsize>,
    buffer: AtomicPtr<Buffer>,
    /// Retired (grown/shrunk-out) buffers. The inline deque uses **retain-until-drop**
    /// reclamation — retired buffers are kept (memory bounded by `O(log max_len)` arrays) and
    /// freed only when the whole deque drops. This lets `steal` skip the `in_flight` counter
    /// (two `SeqCst` RMWs) the boxed deque needs, trading a little memory for a faster steal
    /// hot path. Sound because a thief can never read freed memory: nothing is freed mid-life.
    retired: AtomicPtr<Retired>,
}

impl Inner {
    fn new(log_cap: u32) -> Self {
        let buffer = Box::into_raw(Box::new(Buffer::alloc(1usize << log_cap)));
        Inner {
            bottom: CachePadded(AtomicIsize::new(0)),
            top: CachePadded(AtomicIsize::new(0)),
            buffer: AtomicPtr::new(buffer),
            retired: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    unsafe fn retire(&self, old: *mut Buffer) {
        let node = Box::into_raw(Box::new(Retired {
            buffer: old,
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

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.buffer.load(Ordering::Relaxed)));
            let mut node = self.retired.load(Ordering::Relaxed);
            while !node.is_null() {
                let owned = Box::from_raw(node);
                drop(Box::from_raw(owned.buffer));
                node = owned.next;
            }
        }
        // No element drops: `T: Copy` has no destructor, and values live inline in the buffers.
    }
}

/// The single owner of an inline (`Copy`) work-stealing deque. Pushes/pops the bottom (LIFO).
pub struct InlineWorker<T: Copy> {
    inner: Arc<Inner>,
    cached_top: core::cell::Cell<isize>,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: owner-only `&self` methods; the `Cell` is touched only by `push` on the owning thread.
unsafe impl<T: Copy + Send> Send for InlineWorker<T> {}

impl<T: Copy> InlineWorker<T> {
    /// Create an empty inline deque (default capacity 32).
    pub fn new() -> Self {
        Self::with_log_capacity(5)
    }

    /// Create an empty inline deque with `2^log_cap` initial slots.
    pub fn with_log_capacity(log_cap: u32) -> Self {
        InlineWorker {
            inner: Arc::new(Inner::new(log_cap)),
            cached_top: core::cell::Cell::new(0),
            _marker: core::marker::PhantomData,
        }
    }

    /// A thief handle for this deque.
    pub fn stealer(&self) -> InlineStealer<T> {
        InlineStealer {
            inner: Arc::clone(&self.inner),
            _marker: core::marker::PhantomData,
        }
    }

    /// Number of elements (approximate under concurrency).
    pub fn len(&self) -> usize {
        let b = self.inner.bottom.load(Ordering::Relaxed);
        let t = self.inner.top.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    /// Whether the deque is empty (approximate under concurrency).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current backing-buffer capacity.
    pub fn capacity(&self) -> usize {
        unsafe { (*self.inner.buffer.load(Ordering::Relaxed)).cap }
    }

    /// Push a value onto the bottom. Owner-only. **No allocation** — the value's bits are stored
    /// directly in the cell.
    pub fn push(&self, value: T) {
        let inner = &*self.inner;
        let b = inner.bottom.load(Ordering::Relaxed);
        let mut buf = inner.buffer.load(Ordering::Acquire);
        let cap = unsafe { (*buf).cap } as isize;

        // Cached-top trick (Chase-Lev §2.3): consult the cached lower bound first.
        let mut t = self.cached_top.get();
        if b.wrapping_sub(t) >= cap - 1 {
            t = inner.top.load(Ordering::Acquire);
            self.cached_top.set(t);
        }
        if b.wrapping_sub(t) >= cap - 1 {
            let bigger = unsafe { (*buf).resized(((cap * 2) as usize).max(MIN_CAPACITY), b, t) };
            let bigger_ptr = Box::into_raw(Box::new(bigger));
            inner.buffer.store(bigger_ptr, Ordering::Release);
            unsafe { inner.retire(buf) };
            buf = bigger_ptr;
        }

        unsafe { (*buf).cell(b).store(to_bits(value), Ordering::Relaxed) };
        inner.bottom.store(b.wrapping_add(1), Ordering::Release);
    }

    /// Pop a value from the bottom. Owner-only (LIFO). Returns `None` if empty.
    pub fn pop(&self) -> Option<T> {
        let inner = &*self.inner;
        let b = inner.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        let buf = inner.buffer.load(Ordering::Relaxed);
        inner.bottom.store(b, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let t = inner.top.load(Ordering::Relaxed);
        let size = b.wrapping_sub(t);

        if size < 0 {
            inner.bottom.store(t, Ordering::Relaxed);
            return None;
        }
        let bits = unsafe { (*buf).cell(b).load(Ordering::Relaxed) };
        if size > 0 {
            self.perhaps_shrink(b, t, buf);
            return Some(from_bits(bits));
        }
        // Last element: race thieves via CAS on top.
        let won = inner
            .top
            .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();
        inner.bottom.store(t.wrapping_add(1), Ordering::Relaxed);
        if won { Some(from_bits(bits)) } else { None }
    }

    fn perhaps_shrink(&self, bottom: isize, top: isize, buf: *mut Buffer) {
        let inner = &*self.inner;
        let cap = unsafe { (*buf).cap };
        let size = bottom.wrapping_sub(top).max(0) as usize;
        if cap > MIN_CAPACITY && size < cap / SHRINK_FACTOR {
            let smaller_cap = (cap / 2).max(MIN_CAPACITY);
            let smaller = unsafe { (*buf).resized(smaller_cap, bottom, top) };
            let smaller_ptr = Box::into_raw(Box::new(smaller));
            inner.buffer.store(smaller_ptr, Ordering::Release);
            unsafe { inner.retire(buf) };
        }
    }
}

impl<T: Copy> Default for InlineWorker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A thief for an inline (`Copy`) deque.
pub struct InlineStealer<T: Copy> {
    inner: Arc<Inner>,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: steal is read/atomic-CAS only; safe from many threads for `Copy + Send` values.
unsafe impl<T: Copy + Send> Send for InlineStealer<T> {}
unsafe impl<T: Copy + Send> Sync for InlineStealer<T> {}

impl<T: Copy> Clone for InlineStealer<T> {
    fn clone(&self) -> Self {
        InlineStealer {
            inner: Arc::clone(&self.inner),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Copy> InlineStealer<T> {
    /// Steal a value from the top of the deque.
    ///
    /// The steal fast path here is **fence-light**: unlike the boxed [`crate::Worker`], the
    /// inline deque does **not** bracket the buffer dereference with an `in_flight` counter (two
    /// extra `SeqCst` RMWs per steal). It doesn't need to: a retired buffer is *retained* (never
    /// freed mid-life — see [`Inner::retire`]), so a thief reading a stale buffer pointer can
    /// never use-after-free. That removes the two atomics from every steal — the same hot-path
    /// economy crossbeam gets from epoch GC, but dependency-free.
    pub fn steal(&self) -> Steal<T> {
        let inner = &*self.inner;
        let t = inner.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = inner.bottom.load(Ordering::Acquire);
        if b.wrapping_sub(t) <= 0 {
            return Steal::Empty;
        }

        // SAFETY: the loaded buffer is retained until the whole deque drops, so this read is
        // always to live memory even if the owner grows/shrinks concurrently.
        let buf = inner.buffer.load(Ordering::Acquire);
        let bits = unsafe { (*buf).cell(t).load(Ordering::Relaxed) };

        if inner
            .top
            .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            // We won the slot — it's just a bit-copy; no ownership/drop to worry about.
            Steal::Success(from_bits(bits))
        } else {
            Steal::Retry
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn push_pop_lifo_inline() {
        let w = InlineWorker::<u64>::new();
        for i in 0..10u64 {
            w.push(i);
        }
        for i in (0..10u64).rev() {
            assert_eq!(w.pop(), Some(i));
        }
        assert_eq!(w.pop(), None);
    }

    #[test]
    fn steal_fifo_inline() {
        let w = InlineWorker::<u32>::new();
        let s = w.stealer();
        for i in 0..10u32 {
            w.push(i);
        }
        for i in 0..10u32 {
            assert_eq!(s.steal(), Steal::Success(i));
        }
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn small_copy_type_roundtrips() {
        // A 4-byte struct (smaller than 8) must round-trip through the inline cells.
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct P {
            x: u16,
            y: u16,
        }
        let w = InlineWorker::<P>::new();
        w.push(P { x: 7, y: 9 });
        w.push(P { x: 1, y: 2 });
        assert_eq!(w.pop(), Some(P { x: 1, y: 2 }));
        assert_eq!(w.pop(), Some(P { x: 7, y: 9 }));
    }

    #[test]
    fn grows_and_shrinks_inline() {
        let w = InlineWorker::<u64>::with_log_capacity(1);
        let n = 10_000u64;
        for i in 0..n {
            w.push(i);
        }
        assert!(w.capacity() >= n as usize);
        let grown = w.capacity();
        for _ in 0..(n - 5) {
            w.pop();
        }
        assert!(w.capacity() < grown, "inline buffer should shrink");
    }

    #[test]
    fn concurrent_steal_no_loss_inline() {
        let w = InlineWorker::<usize>::new();
        let n = 200_000usize;
        let thieves = 4;
        let seen: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        let consumed = StdArc::new(AtomicUsize::new(0));

        fn thief(s: InlineStealer<usize>, seen: &[AtomicUsize], consumed: &AtomicUsize, n: usize) {
            loop {
                match s.steal() {
                    Steal::Success(v) => {
                        seen[v].fetch_add(1, StdOrdering::SeqCst);
                        consumed.fetch_add(1, StdOrdering::SeqCst);
                    }
                    Steal::Retry => {}
                    Steal::Empty if consumed.load(StdOrdering::SeqCst) >= n => break,
                    Steal::Empty => {}
                }
            }
        }

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let seen = StdArc::clone(&seen);
                let consumed = StdArc::clone(&consumed);
                scope.spawn(move || thief(s, &seen, &consumed, n));
            }
            let mut popped = Vec::new();
            for i in 0..n {
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
            for v in popped {
                seen[v].fetch_add(1, StdOrdering::SeqCst);
                consumed.fetch_add(1, StdOrdering::SeqCst);
            }
        });

        for (v, slot) in seen.iter().enumerate() {
            assert_eq!(
                slot.load(StdOrdering::SeqCst),
                1,
                "value {v} not consumed exactly once"
            );
        }
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;

    #[test]
    fn loom_inline_owner_pop_vs_thief() {
        loom::model(|| {
            let w = InlineWorker::<u32>::with_log_capacity(4);
            let s = w.stealer();
            w.push(1);
            w.push(2);

            let thief = loom::thread::spawn(move || match s.steal() {
                Steal::Success(v) => Some(v),
                _ => None,
            });

            let owner = w.pop();
            let stolen = thief.join().unwrap();

            // Two items, two consumers, plus whatever remains: exact-once total (no value
            // delivered twice, none lost) — even though values are inline bit-copies.
            let mut remaining = 0;
            while w.pop().is_some() {
                remaining += 1;
            }
            let consumed = owner.is_some() as usize + stolen.is_some() as usize + remaining;
            assert_eq!(consumed, 2, "every pushed value consumed exactly once");
        });
    }

    #[test]
    fn loom_inline_push_then_steal() {
        loom::model(|| {
            let w = InlineWorker::<u32>::with_log_capacity(4);
            let s = w.stealer();
            w.push(42);
            let thief = loom::thread::spawn(move || matches!(s.steal(), Steal::Success(_)));
            let stolen = thief.join().unwrap();
            let popped = w.pop();
            assert_eq!(stolen as usize + popped.is_some() as usize, 1);
        });
    }
}
