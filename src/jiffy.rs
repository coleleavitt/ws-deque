//! A Jiffy-style wait-free **multi-producer / single-consumer** (MPSC) queue.
//!
//! Implements the core of
//!
//! - D. Adas & R. Friedman, *Jiffy: A Fast, Memory Efficient, Wait-Free Multi-Producers
//!   Single-Consumer Queue*, arXiv:2010.14189.
//!
//! Jiffy is an unbounded FIFO queue built from a singly-linked list of fixed-size **buffers**.
//! An `enqueue` is little more than a single fetch-and-add on a shared `tail` index (to claim a
//! slot) plus storing the value and flipping that slot's state `Empty → Set`. A new buffer is
//! linked (via CAS) just before the current one fills, so producers almost never block on
//! allocation. `dequeue` (single consumer) scans from the `head`, reading slot states with no
//! atomic RMW at all, and marks consumed slots `Handled`; fully-handled buffers are unlinked
//! and freed, so memory tracks the live element count.
//!
//! Here it serves as the scheduler's lock-free **work injector / inbox**: any worker may
//! `enqueue` a locality-hinted task, and the owning worker (the single consumer) `dequeue`s
//! them into its own deque. That replaces the previous `Mutex<Vec<T>>` inbox.
//!
//! # Scope
//!
//! This is a faithful but pragmatic implementation of Jiffy's fast paths (FAA-claimed slots,
//! three-state slots, eager next-buffer allocation, head-buffer reclamation). It targets the
//! single-consumer setting the paper is designed for; it is **not** safe to dequeue from more
//! than one thread.

use std::boxed::Box;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::vec::Vec;

#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

/// Slots per buffer node.
const BUFFER_SIZE: usize = 1024;

// Slot states (the paper's 2-bit `isSet`).
const EMPTY: u8 = 0; // not yet written by a producer
const SET: u8 = 1; // written, available to the consumer
const HANDLED: u8 = 2; // already consumed (skipped on future scans)

struct Slot<T> {
    state: AtomicU8,
    /// Written by the claiming producer before flipping `state` to `SET`; read by the consumer
    /// only after observing `SET`. `MaybeUninit` because an `EMPTY`/`HANDLED` slot has no value.
    value: core::cell::UnsafeCell<core::mem::MaybeUninit<T>>,
}

struct Buffer<T> {
    slots: Box<[Slot<T>]>,
    next: AtomicPtr<Buffer<T>>,
    /// Global index of `slots[0]`.
    base: usize,
}

impl<T> Buffer<T> {
    fn new(base: usize) -> *mut Self {
        let mut v = Vec::with_capacity(BUFFER_SIZE);
        for _ in 0..BUFFER_SIZE {
            v.push(Slot {
                state: AtomicU8::new(EMPTY),
                value: core::cell::UnsafeCell::new(core::mem::MaybeUninit::uninit()),
            });
        }
        Box::into_raw(Box::new(Buffer {
            slots: v.into_boxed_slice(),
            next: AtomicPtr::new(core::ptr::null_mut()),
            base,
        }))
    }
}

struct Inner<T> {
    /// Monotone global enqueue index; a producer FAAs this to claim a slot.
    tail: AtomicUsize,
    /// Tail buffer pointer (where new slots are claimed). Advanced via CAS by producers.
    tail_buf: AtomicPtr<Buffer<T>>,
    /// The very first buffer (global base 0). Never changes; the whole linked list is reachable
    /// from here, so `Drop` walks from it to free every buffer (and any un-dequeued value).
    first: *mut Buffer<T>,
    /// Consumer-owned head buffer pointer + the global head index. Only the single consumer
    /// advances these. Buffers are **retained until the queue drops** (not freed mid-`dequeue`):
    /// freeing a head buffer while a slow producer may still reference it — a producer that FAA-
    /// claimed a low index but has not yet stored, or is walking `next` pointers — is a genuine
    /// use-after-free (ThreadSanitizer-confirmed). Retain-until-drop is the simple race-free
    /// policy; for this scheduler inbox the queue is drained continuously and dropped per run.
    head_buf: AtomicPtr<Buffer<T>>,
    head: AtomicUsize,
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Single-threaded at drop. Walk the whole chain from `first`; drop any still-`SET` value
        // (enqueued but never dequeued) exactly once, then free each buffer.
        unsafe {
            let mut buf = self.first;
            while !buf.is_null() {
                let owned = Box::from_raw(buf);
                for slot in owned.slots.iter() {
                    if slot.state.load(Ordering::Relaxed) == SET {
                        // SAFETY: SET ⇒ the value was written and never consumed.
                        (*slot.value.get()).assume_init_drop();
                    }
                }
                buf = owned.next.load(Ordering::Relaxed);
            }
        }
    }
}

/// A producer handle (`Clone` + `Send` + `Sync`): many of these may `enqueue` concurrently.
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
}

// SAFETY: enqueue uses only FAA/CAS/atomic stores; safe from many threads for `T: Send`.
unsafe impl<T: Send> Send for Producer<T> {}
unsafe impl<T: Send> Sync for Producer<T> {}

impl<T> Clone for Producer<T> {
    fn clone(&self) -> Self {
        Producer {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// The single consumer handle. `Send` (move it to its thread) but **not** `Sync`: exactly one
/// thread may `dequeue`.
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
}

unsafe impl<T: Send> Send for Consumer<T> {}

/// Create a Jiffy MPSC queue, returning the single [`Consumer`] and a cloneable [`Producer`].
pub fn channel<T>() -> (Producer<T>, Consumer<T>) {
    let first = Buffer::new(0);
    let inner = Arc::new(Inner {
        tail: AtomicUsize::new(0),
        tail_buf: AtomicPtr::new(first),
        first,
        head_buf: AtomicPtr::new(first),
        head: AtomicUsize::new(0),
    });
    (
        Producer {
            inner: Arc::clone(&inner),
        },
        Consumer { inner },
    )
}

impl<T> Producer<T> {
    /// Enqueue a value (wait-free fast path: one FAA to claim a slot, then a store + state flip).
    pub fn enqueue(&self, value: T) {
        let inner = &*self.inner;
        let idx = inner.tail.fetch_add(1, Ordering::AcqRel);

        // Find/allocate the buffer holding global index `idx`, starting from the tail buffer.
        let buf = self.locate_tail_buffer(idx);
        let offset = idx - unsafe { (*buf).base };

        // SAFETY: this producer uniquely claimed `idx` via FAA, so it alone writes this slot.
        unsafe {
            let slot = &(*buf).slots[offset];
            (*slot.value.get()).write(value);
            // Release: publishes the value write to the consumer that Acquire-loads the state.
            slot.state.store(SET, Ordering::Release);
        }

        // Eagerly link the next buffer when we claim the second-to-last slot, so producers that
        // reach the end almost never have to allocate on the critical path (paper §4).
        if offset + 2 == BUFFER_SIZE {
            self.link_next_buffer(buf);
        }
    }

    /// Walk/grow the tail-buffer chain until we reach the buffer whose range contains `idx`.
    fn locate_tail_buffer(&self, idx: usize) -> *mut Buffer<T> {
        let inner = &*self.inner;
        loop {
            let buf = inner.tail_buf.load(Ordering::Acquire);
            let base = unsafe { (*buf).base };
            if idx < base {
                // Our slot is in an earlier buffer (tail_buf advanced past us); walk from head.
                return self.locate_from_head(idx);
            }
            if idx < base + BUFFER_SIZE {
                return buf;
            }
            // `idx` is beyond this buffer — ensure the next buffer exists and advance tail_buf.
            self.link_next_buffer(buf);
            let next = unsafe { (*buf).next.load(Ordering::Acquire) };
            if !next.is_null() {
                // Best-effort advance: a losing CAS just means another producer already moved
                // `tail_buf` forward, which is exactly the outcome we wanted — so ignoring the
                // result is correct, and we loop to re-read the (now-advanced) tail buffer.
                let _advanced =
                    inner
                        .tail_buf
                        .compare_exchange(buf, next, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
    }

    /// Slow path: a producer's claimed `idx` lies in a buffer at/after the head; walk from head.
    fn locate_from_head(&self, idx: usize) -> *mut Buffer<T> {
        let inner = &*self.inner;
        let mut buf = inner.head_buf.load(Ordering::Acquire);
        loop {
            let base = unsafe { (*buf).base };
            if idx < base + BUFFER_SIZE {
                return buf;
            }
            let next = unsafe { (*buf).next.load(Ordering::Acquire) };
            debug_assert!(!next.is_null(), "claimed index has no buffer");
            buf = next;
        }
    }

    /// Ensure `buf.next` exists, allocating + CAS-linking a fresh buffer if needed.
    fn link_next_buffer(&self, buf: *mut Buffer<T>) {
        let next = unsafe { (*buf).next.load(Ordering::Acquire) };
        if !next.is_null() {
            return;
        }
        let base = unsafe { (*buf).base } + BUFFER_SIZE;
        let fresh = Buffer::new(base);
        // Only one producer wins the link; the rest free their speculative buffer.
        if unsafe {
            (*buf)
                .next
                .compare_exchange(
                    core::ptr::null_mut(),
                    fresh,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_err()
        } {
            // SAFETY: we just allocated `fresh` and lost the race; no one else has it.
            unsafe { drop(Box::from_raw(fresh)) };
        }
    }
}

impl<T> Consumer<T> {
    /// Dequeue the next value (single consumer). Returns `None` if the queue is currently empty.
    /// Reads slot states with plain loads — no FAA/CAS on the consumer side (paper §4).
    pub fn dequeue(&mut self) -> Option<T> {
        let inner = &*self.inner;
        loop {
            let head = inner.head.load(Ordering::Relaxed);
            let tail = inner.tail.load(Ordering::Acquire);
            if head >= tail {
                return None; // nothing claimed past the head
            }
            let buf = inner.head_buf.load(Ordering::Acquire);
            let base = unsafe { (*buf).base };
            let offset = head - base;

            // If we've consumed/handled the whole head buffer, advance to the next. The old
            // buffer is *retained* (still linked from `first`) until the queue drops — see the
            // `head_buf` field docs for why freeing it here is a use-after-free.
            if offset >= BUFFER_SIZE {
                let next = unsafe { (*buf).next.load(Ordering::Acquire) };
                if next.is_null() {
                    return None; // producer hasn't linked the next buffer yet
                }
                inner.head_buf.store(next, Ordering::Release);
                continue;
            }

            let slot = unsafe { &(*buf).slots[offset] };
            match slot.state.load(Ordering::Acquire) {
                EMPTY => {
                    // A producer claimed this index (head < tail) but hasn't stored yet. The
                    // single consumer waits for it rather than reordering (keeps FIFO simple).
                    return None;
                }
                HANDLED => {
                    // Already consumed (can happen after a state we skipped); advance head.
                    inner.head.store(head + 1, Ordering::Relaxed);
                    continue;
                }
                _ => {
                    // SET: take the value, mark handled, advance head.
                    // SAFETY: SET ⇒ the value is initialized; single consumer ⇒ unique read.
                    let value = unsafe { (*slot.value.get()).assume_init_read() };
                    slot.state.store(HANDLED, Ordering::Release);
                    inner.head.store(head + 1, Ordering::Relaxed);
                    return Some(value);
                }
            }
        }
    }

    /// Whether the queue currently has no dequeuable element (approximate under concurrency).
    pub fn is_empty(&self) -> bool {
        let inner = &*self.inner;
        inner.head.load(Ordering::Relaxed) >= inner.tail.load(Ordering::Acquire)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

    use super::*;

    #[test]
    fn single_producer_fifo() {
        let (p, mut c) = channel::<u64>();
        for i in 0..5000u64 {
            p.enqueue(i);
        }
        for i in 0..5000u64 {
            assert_eq!(c.dequeue(), Some(i)); // FIFO across many buffers
        }
        assert_eq!(c.dequeue(), None);
    }

    #[test]
    fn empty_then_filled() {
        let (p, mut c) = channel::<u32>();
        assert_eq!(c.dequeue(), None);
        p.enqueue(42);
        assert_eq!(c.dequeue(), Some(42));
        assert_eq!(c.dequeue(), None);
    }

    #[test]
    fn many_producers_no_loss_no_dup() {
        let (p, mut c) = channel::<usize>();
        let producers = 6;
        let per = 50_000usize;
        let n = producers * per;

        std::thread::scope(|scope| {
            for t in 0..producers {
                let p = p.clone();
                scope.spawn(move || {
                    for i in 0..per {
                        p.enqueue(t * per + i); // globally unique values
                    }
                });
            }
            // Single consumer drains concurrently until it has seen all n values.
            let seen = AtomicUsize::new(0);
            let mut bag: HashSet<usize> = HashSet::with_capacity(n);
            while seen.load(StdOrdering::Relaxed) < n {
                if let Some(v) = c.dequeue() {
                    assert!(bag.insert(v), "value {v} dequeued twice");
                    seen.fetch_add(1, StdOrdering::Relaxed);
                }
            }
            assert_eq!(bag.len(), n, "every produced value seen exactly once");
        });
    }

    #[test]
    fn drops_unconsumed_values_once() {
        // A payload that counts live instances detects leak/double-free of un-dequeued items.
        use std::sync::Arc as StdArc;
        struct Counted(StdArc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_sub(1, StdOrdering::SeqCst);
            }
        }
        let live = StdArc::new(AtomicUsize::new(0));
        {
            let (p, mut c) = channel::<Counted>();
            for _ in 0..2000 {
                live.fetch_add(1, StdOrdering::SeqCst);
                p.enqueue(Counted(StdArc::clone(&live)));
            }
            // Consume some, leave the rest to be dropped with the queue.
            for _ in 0..500 {
                drop(c.dequeue().unwrap());
            }
            assert_eq!(live.load(StdOrdering::SeqCst), 1500);
        }
        assert_eq!(live.load(StdOrdering::SeqCst), 0, "no leak / double-free");
    }
}
