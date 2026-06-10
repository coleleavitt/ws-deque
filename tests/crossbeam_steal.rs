//! Steal / batch-steal tests **ported and adapted from `crossbeam-deque`**
//! (`crossbeam-deque/tests/steal.rs`).
//!
//! Note on batch semantics: crossbeam's `steal_batch` moves *exactly* a deterministic prefix and
//! its tests assert specific element identities. `ws-deque` deliberately implements `steal_batch`
//! /`steal_batch_and_pop` as a loop of individually-linearizable single steals (a single-CAS
//! multi-slot claim is unsound against this crate's CAS-free owner `pop` — see the `Stealer`
//! docs). So the exact count/identity per batch is not contractually fixed; what *is* guaranteed
//! is: a batch moves ≥ 1 element when work exists, takes from the top (oldest) end, and never
//! duplicates or loses an element. These tests assert that contract.

use ws_deque::Steal::{self, Success};
use ws_deque::Worker;

fn busy_retry<T>(mut f: impl FnMut() -> Steal<T>) -> Steal<T> {
    loop {
        let s = f();
        if !s.is_retry() {
            return s;
        }
    }
}

#[test]
fn steal_fifo() {
    let w = Worker::new_fifo();
    for i in 1..=3 {
        w.push(i);
    }

    let s = w.stealer();
    assert_eq!(s.steal(), Success(1));
    assert_eq!(s.steal(), Success(2));
    assert_eq!(s.steal(), Success(3));
}

#[test]
fn steal_lifo() {
    // Stealing always takes from the *top* (oldest), even on a LIFO worker — only the owner's
    // `pop` is LIFO. So a thief sees 1, 2, 3 in order, matching crossbeam.
    let w = Worker::new();
    for i in 1..=3 {
        w.push(i);
    }

    let s = w.stealer();
    assert_eq!(s.steal(), Success(1));
    assert_eq!(s.steal(), Success(2));
    assert_eq!(s.steal(), Success(3));
}

#[test]
fn steal_batch_moves_from_top_no_loss() {
    // A batch steal moves ≥ 1 element from the top into `dest`, in top (oldest-first) order,
    // and the union of what's left + what moved is exactly the original, once each.
    let w = Worker::new();
    for i in 1..=8 {
        w.push(i);
    }
    let s = w.stealer();
    let w2 = Worker::new_fifo();

    let res = busy_retry(|| s.steal_batch(&w2));
    assert_eq!(res, Success(()), "batch must succeed when work exists");

    let mut moved = Vec::new();
    while let Some(v) = w2.pop() {
        moved.push(v);
    }
    assert!(!moved.is_empty(), "batch moved at least one element");
    // Moved elements come from the top (oldest) end, so they are a prefix 1, 2, ... in order.
    for (k, v) in moved.iter().enumerate() {
        assert_eq!(*v, k + 1, "batch takes the oldest elements in order");
    }

    // Drain the rest and confirm the whole set 1..=8 is accounted for exactly once.
    let mut rest = moved.clone();
    while let Steal::Success(v) = s.steal() {
        rest.push(v);
    }
    while let Some(v) = w.pop() {
        rest.push(v);
    }
    rest.sort_unstable();
    assert_eq!(rest, (1..=8).collect::<Vec<_>>(), "no loss, no duplication");
}

#[test]
fn steal_batch_and_pop_returns_one_and_moves_rest() {
    let w = Worker::new();
    for i in 1..=6 {
        w.push(i);
    }
    let s = w.stealer();
    let w2 = Worker::new_fifo();

    // Returns one stolen element directly (the oldest), and may move more into w2.
    let popped = match busy_retry(|| s.steal_batch_and_pop(&w2)) {
        Success(v) => v,
        other => panic!("expected Success, got {other:?}"),
    };
    assert_eq!(popped, 1, "the directly-returned element is the oldest");

    // Collect everything and verify the full set is intact, once each.
    let mut all = vec![popped];
    while let Some(v) = w2.pop() {
        all.push(v);
    }
    while let Steal::Success(v) = s.steal() {
        all.push(v);
    }
    while let Some(v) = w.pop() {
        all.push(v);
    }
    all.sort_unstable();
    assert_eq!(all, (1..=6).collect::<Vec<_>>(), "no loss, no duplication");
}

#[test]
fn steal_batch_empty_is_empty() {
    let w: Worker<i32> = Worker::new();
    let s = w.stealer();
    let w2 = Worker::new();
    assert_eq!(s.steal_batch(&w2), Steal::Empty);
    assert_eq!(s.steal_batch_and_pop(&w2), Steal::Empty);
}
