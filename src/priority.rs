//! Priority work-stealing: per-task priority over a small set of levels.
//!
//! Implements the *steal-order / execution-order* idea of
//!
//! - M. Wimmer, D. Cederman, J. L. Träff, P. Tsigas, *Configurable Strategies for
//!   Work-Stealing*, arXiv:1305.6474.
//!
//! Standard work-stealing is oblivious to task importance — execution order is whatever the
//! deque's LIFO/FIFO discipline happens to give. Wimmer et al. show that letting a task carry
//! a **priority** (a steal/execution-order hint) can reduce the *total work* for search-style
//! algorithms: branch-and-bound and best-first / shortest-path explore the most promising
//! branches first, pruning the rest before they are ever expanded.
//!
//! # Design
//!
//! A [`PriorityWorker`] is simply `K` independent [`crate::Worker`] deques, one per priority
//! level (`0` = highest). `push(task, level)` routes to a level; `pop` and `steal` scan from
//! the highest priority downward and take the first available task. Because each level is an
//! ordinary verified Chase-Lev deque, this composition inherits the deque's exact-once
//! semantics, race-freedom (ThreadSanitizer-clean), and loom-checked orderings — the only new
//! logic is the highest-first scan, which is plain control flow.
//!
//! `K` is a const generic so the level array is stack-allocated and the scan is a fixed,
//! branch-predictable loop.

use crate::{Steal, Stealer, Worker};

/// Owner handle for a `K`-level priority work-stealing pool. Level `0` is the highest priority.
pub struct PriorityWorker<T, const K: usize> {
    levels: [Worker<T>; K],
}

/// A thief handle for a [`PriorityWorker`]. `Clone` + `Send` + `Sync`.
pub struct PriorityStealer<T, const K: usize> {
    levels: [Stealer<T>; K],
}

impl<T, const K: usize> PriorityWorker<T, K> {
    /// Create a priority pool with `K` levels. `K` must be at least 1.
    pub fn new() -> Self {
        assert!(K >= 1, "priority pool needs at least one level");
        PriorityWorker {
            levels: core::array::from_fn(|_| Worker::new()),
        }
    }

    /// A thief handle that steals highest-priority-first across all levels.
    pub fn stealer(&self) -> PriorityStealer<T, K> {
        PriorityStealer {
            levels: core::array::from_fn(|i| self.levels[i].stealer()),
        }
    }

    /// Number of priority levels (`K`).
    pub fn levels(&self) -> usize {
        K
    }

    /// Total queued tasks across all levels (approximate under concurrency).
    pub fn len(&self) -> usize {
        self.levels.iter().map(|w| w.len()).sum()
    }

    /// Whether every level is empty (approximate under concurrency).
    pub fn is_empty(&self) -> bool {
        self.levels.iter().all(|w| w.is_empty())
    }

    /// Push `task` at priority `level` (clamped to `[0, K)`; `0` = highest). Owner-only.
    pub fn push(&self, task: T, level: usize) {
        let lvl = level.min(K - 1);
        self.levels[lvl].push(task);
    }

    /// Pop the highest-priority available task (owner-only). Scans level `0` upward.
    pub fn pop(&self) -> Option<T> {
        for w in &self.levels {
            if let Some(task) = w.pop() {
                return Some(task);
            }
        }
        None
    }
}

impl<T, const K: usize> Default for PriorityWorker<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const K: usize> Clone for PriorityStealer<T, K> {
    fn clone(&self) -> Self {
        PriorityStealer {
            levels: core::array::from_fn(|i| self.levels[i].clone()),
        }
    }
}

impl<T, const K: usize> PriorityStealer<T, K> {
    /// Steal the highest-priority available task. Scans level `0` upward; a `Retry` at one
    /// level does not block lower levels — we report `Retry` only if *some* level was contended
    /// and none yielded a task, so the caller knows to try again.
    pub fn steal(&self) -> Steal<T> {
        let mut saw_retry = false;
        for s in &self.levels {
            match s.steal() {
                Steal::Success(task) => return Steal::Success(task),
                Steal::Retry => saw_retry = true,
                Steal::Empty => {}
            }
        }
        if saw_retry {
            Steal::Retry
        } else {
            Steal::Empty
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn pops_highest_priority_first() {
        let w = PriorityWorker::<&str, 3>::new();
        w.push("low", 2);
        w.push("high", 0);
        w.push("mid", 1);
        // Even though "low" was pushed first, priority 0 wins.
        assert_eq!(w.pop(), Some("high"));
        assert_eq!(w.pop(), Some("mid"));
        assert_eq!(w.pop(), Some("low"));
        assert_eq!(w.pop(), None);
    }

    #[test]
    fn steal_prefers_higher_priority() {
        let w = PriorityWorker::<u32, 2>::new();
        let s = w.stealer();
        w.push(100, 1); // low priority
        w.push(1, 0); // high priority
        // Thief takes the high-priority task first.
        assert_eq!(s.steal(), Steal::Success(1));
        assert_eq!(s.steal(), Steal::Success(100));
        assert_eq!(s.steal(), Steal::Empty);
    }

    #[test]
    fn level_clamps_to_range() {
        let w = PriorityWorker::<u32, 2>::new();
        w.push(7, 99); // clamps to level 1 (lowest)
        assert_eq!(w.len(), 1);
        assert_eq!(w.pop(), Some(7));
    }

    fn priority_thief(
        s: &PriorityStealer<usize, 4>,
        seen: &[AtomicUsize],
        consumed: &AtomicUsize,
        total: usize,
    ) {
        loop {
            match s.steal() {
                Steal::Success(v) => {
                    seen[v].fetch_add(1, Ordering::SeqCst);
                    consumed.fetch_add(1, Ordering::SeqCst);
                }
                Steal::Retry => {}
                Steal::Empty if consumed.load(Ordering::SeqCst) >= total => break,
                Steal::Empty => {}
            }
        }
    }

    #[test]
    fn concurrent_priority_steal_no_loss() {
        let w = PriorityWorker::<usize, 4>::new();
        let n = if cfg!(miri) { 400 } else { 100_000 };
        let thieves = 4;
        let seen: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        let consumed = StdArc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let seen = StdArc::clone(&seen);
                let consumed = StdArc::clone(&consumed);
                scope.spawn(move || priority_thief(&s, &seen, &consumed, n));
            }
            for i in 0..n {
                w.push(i, i % 4); // spread across all priority levels
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
