//! Linked-list-backed work-stealing with multiplicity (Castañeda-Piña *approach 2*).
//!
//! Implements the second array-backing strategy from §6 of
//!
//! - A. Castañeda & M. Piña, *Fully Read/Write Fence-Free Work-Stealing with Multiplicity*,
//!   arXiv:2008.04424.
//!
//! Instead of a growable contiguous array (the [`crate::idempotent`] module, *approach 1*,
//! which doubles + copies on overflow and must reclaim retired arrays), the task store is a
//! **singly-linked list of fixed-size nodes**. This buys two properties the contiguous version
//! cannot have at once:
//!
//! 1. **`put` is true constant-time.** When the current node fills, the owner links a fresh
//!    node and continues — there is never an O(n) copy of existing tasks.
//! 2. **No memory reclamation problem at all.** Old nodes are never abandoned or moved, so a
//!    slow thief can always follow `next` pointers safely. The whole list lives until the
//!    structure is dropped — no epoch GC, no quiescent-state counter, no retired list.
//!
//! Semantics are the same **multiplicity** relaxation as [`crate::idempotent`]: a task is
//! delivered *at least once* across all consumers (bounded by the number of concurrent
//! consumers), so `T: Clone` and consumers receive clones. Use it for idempotent workloads.
//!
//! The trade-off versus the contiguous deque is locality: slot access follows node pointers
//! rather than a single `& mask`, and each consumer walks the list forward from a cached node.
//! Because `head` only advances and nodes are append-only, that walk is amortized O(1).

use std::boxed::Box;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::vec::Vec;

#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub use crate::idempotent::Take;

/// Tasks per linked-list node (power of two for cheap division/modulo).
const NODE_SIZE: usize = 64;

/// A fixed-size node in the task list. `base` is the global index of `cells[0]`.
struct Node<T> {
    cells: Box<[AtomicPtr<T>]>,
    next: AtomicPtr<Node<T>>,
    base: usize,
}

impl<T> Node<T> {
    fn new(base: usize) -> *mut Self {
        let mut v = Vec::with_capacity(NODE_SIZE);
        for _ in 0..NODE_SIZE {
            v.push(AtomicPtr::new(core::ptr::null_mut()));
        }
        Box::into_raw(Box::new(Node {
            cells: v.into_boxed_slice(),
            next: AtomicPtr::new(core::ptr::null_mut()),
            base,
        }))
    }
}

struct Inner<T> {
    /// MaxRegister `head`: global index of the next task to take. Only ever advances.
    head: AtomicUsize,
    /// First node (global base 0). Never changes; owns the list for `Drop`.
    first: *mut Node<T>,
    /// Highest global index the owner has written + 1 (published for consumers' bounds check).
    tail: AtomicUsize,
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Single-threaded at drop. Walk every node; free each non-null master box exactly once
        // (tasks are cloned out, so the master stays in its cell), then free the node itself.
        unsafe {
            let mut node = self.first;
            while !node.is_null() {
                let owned = Box::from_raw(node);
                for cell in owned.cells.iter() {
                    let p = cell.load(Ordering::Relaxed);
                    if !p.is_null() {
                        drop(Box::from_raw(p));
                    }
                }
                node = owned.next.load(Ordering::Relaxed);
                // `owned` (the Box<Node>) is dropped here, freeing the node allocation.
            }
        }
    }
}

/// The single owner of a linked WS-MULT queue. Puts at the tail, takes from the head.
pub struct LinkedWorker<T> {
    inner: Arc<Inner<T>>,
    /// Owner-private: the node currently being appended to, and the global tail index.
    tail_node: *mut Node<T>,
    tail: usize,
}

// SAFETY: owner operations are single-threaded; the handle is `Send` (move it to its thread).
unsafe impl<T: Send> Send for LinkedWorker<T> {}

impl<T: Clone> LinkedWorker<T> {
    /// Create an empty linked WS-MULT queue.
    pub fn new() -> Self {
        let first = Node::new(0);
        LinkedWorker {
            inner: Arc::new(Inner {
                head: AtomicUsize::new(0),
                first,
                tail: AtomicUsize::new(0),
            }),
            tail_node: first,
            tail: 0,
        }
    }

    /// A cheaply-clonable thief handle.
    pub fn stealer(&self) -> LinkedStealer<T> {
        LinkedStealer {
            inner: Arc::clone(&self.inner),
            cur_node: self.inner.first,
            cur_base: 0,
        }
    }

    /// Number of nodes currently allocated (test/diagnostic).
    pub fn node_count(&self) -> usize {
        let mut n = 0;
        let mut node = self.inner.first;
        while !node.is_null() {
            n += 1;
            // SAFETY: node list is append-only and never freed before drop.
            node = unsafe { (*node).next.load(Ordering::Acquire) };
        }
        n
    }

    /// **Put** a task at the tail in true constant time — append to the current node, linking a
    /// fresh node when it fills (never copying existing tasks). No CAS on the common path.
    pub fn put(&mut self, task: T) {
        let offset = self.tail - unsafe { (*self.tail_node).base };
        // SAFETY: `tail_node` is a live owner-held node.
        let node = unsafe { &*self.tail_node };
        let boxed = Box::into_raw(Box::new(task));
        node.cells[offset].store(boxed, Ordering::Release);

        // Advance tail; if the node is now full, link a new one for the next put.
        self.tail += 1;
        self.inner.tail.store(self.tail, Ordering::Release);
        if offset + 1 == NODE_SIZE {
            let fresh = Node::new(self.tail);
            // Publish the new node so consumers can follow `next` to it.
            node.next.store(fresh, Ordering::Release);
            self.tail_node = fresh;
        }
    }

    /// **Take** the head task (owner). Advances `head` via `fetch_max` (monotone MaxRegister).
    pub fn take(&self) -> Take<T> {
        let head = self.inner.head.load(Ordering::Acquire);
        if head >= self.inner.tail.load(Ordering::Acquire) {
            return Take::Empty;
        }
        // SAFETY: head < tail, so the slot has been written; walk from `first` to its node.
        let p = unsafe { load_slot(self.inner.first, head) };
        if p.is_null() {
            return Take::Empty;
        }
        self.inner.head.fetch_max(head + 1, Ordering::AcqRel);
        // SAFETY: `p` is a live boxed task owned by the queue.
        Take::Got(unsafe { (*p).clone() })
    }
}

impl<T: Clone> Default for LinkedWorker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A thief for a linked WS-MULT queue. Carries a cached node pointer so its forward walk to the
/// head node is amortized O(1); therefore `steal` takes `&mut self`.
pub struct LinkedStealer<T> {
    inner: Arc<Inner<T>>,
    cur_node: *mut Node<T>,
    cur_base: usize,
}

// SAFETY: single-consumer per handle (`&mut self`); `Send` so it can move to its thread.
unsafe impl<T: Send> Send for LinkedStealer<T> {}

impl<T> Clone for LinkedStealer<T> {
    fn clone(&self) -> Self {
        // A fresh clone restarts its walk from the first node (always valid).
        LinkedStealer {
            inner: Arc::clone(&self.inner),
            cur_node: self.inner.first,
            cur_base: 0,
        }
    }
}

impl<T: Clone> LinkedStealer<T> {
    /// Steal the head task (multiplicity: delivered at least once across consumers). Reads the
    /// shared head, walks forward (from this thief's cached node) to the node holding it, reads
    /// the slot, and advances head via `fetch_max`.
    pub fn steal(&mut self) -> Take<T> {
        let head = self.inner.head.load(Ordering::Acquire);
        if head >= self.inner.tail.load(Ordering::Acquire) {
            return Take::Empty;
        }
        // Walk forward from the cached node to the node containing global index `head`.
        // `head` only advances and nodes are append-only, so this never moves backward and is
        // amortized O(1) over the thief's lifetime.
        while head >= self.cur_base + NODE_SIZE {
            // SAFETY: cur_node is live; its `next` is published before `tail`/`head` advance
            // past this node, so a readable `head` guarantees `next` is non-null here.
            let next = unsafe { (*self.cur_node).next.load(Ordering::Acquire) };
            if next.is_null() {
                return Take::Empty; // node not yet linked from our view; try again later
            }
            self.cur_node = next;
            self.cur_base += NODE_SIZE;
        }
        let offset = head - self.cur_base;
        // SAFETY: offset < NODE_SIZE and the slot is within the written range.
        let p = unsafe { (*self.cur_node).cells[offset].load(Ordering::Acquire) };
        if p.is_null() {
            return Take::Empty;
        }
        self.inner.head.fetch_max(head + 1, Ordering::AcqRel);
        // SAFETY: `p` is a live boxed task owned by the queue.
        Take::Got(unsafe { (*p).clone() })
    }
}

/// Walk from `first` to the node holding global index `idx` and load its slot pointer.
/// SAFETY: caller guarantees `idx < tail` (the slot has been written) and the node list is live.
unsafe fn load_slot<T>(first: *mut Node<T>, idx: usize) -> *mut T {
    let mut node = first;
    while idx >= (*node).base + NODE_SIZE {
        node = (*node).next.load(Ordering::Acquire);
        debug_assert!(!node.is_null());
    }
    let offset = idx - (*node).base;
    (*node).cells[offset].load(Ordering::Acquire)
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn put_take_fifo_across_many_nodes() {
        let mut w = LinkedWorker::<usize>::new();
        let n = NODE_SIZE * 10 + 7; // force many node links
        for i in 0..n {
            w.put(i);
        }
        assert!(w.node_count() >= 10, "should have linked many nodes");
        for i in 0..n {
            assert_eq!(w.take(), Take::Got(i)); // FIFO
        }
        assert_eq!(w.take(), Take::Empty);
    }

    #[test]
    fn steal_fifo_single_thief() {
        let mut w = LinkedWorker::<usize>::new();
        let mut s = w.stealer();
        let n = NODE_SIZE * 3 + 5;
        for i in 0..n {
            w.put(i);
        }
        for i in 0..n {
            assert_eq!(s.steal(), Take::Got(i));
        }
        assert_eq!(s.steal(), Take::Empty);
    }

    #[test]
    fn put_does_not_copy_on_grow() {
        // A purely diagnostic check that growth only *adds* nodes (constant-time put), never a
        // re-allocation of the whole store: node_count grows by exactly 1 per NODE_SIZE puts.
        let mut w = LinkedWorker::<usize>::new();
        assert_eq!(w.node_count(), 1);
        for i in 0..NODE_SIZE {
            w.put(i);
        }
        assert_eq!(
            w.node_count(),
            2,
            "one fresh node linked after filling the first"
        );
    }

    fn drain_thief(mut s: LinkedStealer<usize>, counts: &[AtomicUsize]) {
        let mut empties = 0;
        while empties < 1000 {
            match s.steal() {
                Take::Got(v) => {
                    counts[v].fetch_add(1, StdOrdering::SeqCst);
                    empties = 0;
                }
                Take::Empty => empties += 1,
            }
        }
    }

    #[test]
    fn concurrent_steal_at_least_once() {
        let mut w = LinkedWorker::<usize>::new();
        let n = 200_000;
        let thieves = 4;
        for i in 0..n {
            w.put(i);
        }
        let counts: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let counts = StdArc::clone(&counts);
                scope.spawn(move || drain_thief(s, &counts));
            }
            while let Take::Got(v) = w.take() {
                counts[v].fetch_add(1, StdOrdering::SeqCst);
            }
        });

        let max_consumers = thieves + 1;
        for (v, c) in counts.iter().enumerate() {
            let got = c.load(StdOrdering::SeqCst);
            assert!(got >= 1, "task {v} never delivered");
            assert!(
                got <= max_consumers,
                "task {v} delivered {got}× (> {max_consumers})"
            );
        }
    }
}
