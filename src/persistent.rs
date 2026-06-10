//! A **crash-recoverable (persistent) FIFO queue** with an explicit persistency model.
//!
//! > ⚠️ **NOT REAL CRASH DURABILITY.** This is a *software simulation* of NVM persistence for
//! > study and testing — it does **not** flush to disk or actual non-volatile memory, and it
//! > does **not** survive a real process exit or power loss. "Crash" here means a method call
//! > ([`PersistentQueue::crash_now`]) that wipes the modelled volatile state; "durable" means
//! > "recorded in an in-RAM log we chose to keep across that simulated crash." Do **not** use it
//! > anywhere you need genuine durability — use a real WAL / `fsync`'d store / NVM library for
//! > that. It exists to demonstrate and test the *persistency algorithm* (`pwb`/`psync` ordering
//! > and recovery correctness), nothing more.
//!
//! Models the durability discipline of
//!
//! - P. Fatourou, N. Giachoudis, G. Mallis, *Highly-Efficient Persistent FIFO Queues*,
//!   arXiv:2402.17674,
//!
//! for **Non-Volatile Memory (NVM)** settings, where data survives a power loss but only the
//! parts that were explicitly *flushed* are guaranteed durable. Real NVM exposes two
//! persistence primitives the paper uses on every operation:
//!
//! - **`pwb`** (persistent write-back) — request a flush of a location to NVM (non-blocking).
//! - **`psync`** — a fence that blocks until outstanding `pwb`s have completed.
//!
//! Since this machine has no NVM hardware, we **simulate** persistent memory with an in-process
//! byte log ([`PersistentMemory`]) that distinguishes *volatile* writes from *persisted* bytes:
//! a [`crash`](PersistentMemory::crash) discards everything written-but-not-yet-`psync`'d,
//! exactly as a power failure would. The queue is structured so that after a crash at *any*
//! point, [`PersistentQueue::recover`] rebuilds a **consistent FIFO prefix**: every element
//! whose enqueue completed its `pwb`+`psync` survives, in order, with no duplicates and no
//! lost-but-acknowledged items.
//!
//! This is a single-producer/single-consumer durable queue (the SPSC core of the paper's
//! design) — enough to demonstrate the persistence model and recovery correctness; the
//! concurrent LCRQ machinery is out of scope without NVM hardware to measure it on.

use std::vec::Vec;

/// A simulated block of non-volatile memory: a log of fixed-size records, each either *durable*
/// (survives a crash) or *volatile* (written but not yet `psync`'d — lost on crash).
///
/// A real system would `pwb`/`psync` cache lines; we model the same all-or-nothing durability
/// boundary at record granularity, which is what matters for recovery correctness.
pub struct PersistentMemory {
    /// Durable records — survive [`crash`](Self::crash).
    durable: Vec<Record>,
    /// Records written but not yet flushed — discarded by a crash.
    volatile: Vec<Record>,
    /// Metric: number of `psync` fences executed (the paper's persistence cost).
    psyncs: usize,
}

/// One persisted record. The queue appends an `Enqueue(v)` when a value is durably added and a
/// `Dequeue` marker when the head advances, so replay reconstructs the live FIFO contents.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Record {
    Enqueue(u64),
    Dequeue,
}

impl PersistentMemory {
    /// Fresh, empty persistent memory.
    pub fn new() -> Self {
        PersistentMemory {
            durable: Vec::new(),
            volatile: Vec::new(),
            psyncs: 0,
        }
    }

    /// `pwb`: stage a record for persistence (non-blocking; not yet durable).
    fn pwb(&mut self, record: Record) {
        self.volatile.push(record);
    }

    /// `psync`: fence — make all staged (`pwb`'d) records durable. After this returns, the
    /// records survive a crash.
    fn psync(&mut self) {
        self.durable.append(&mut self.volatile);
        self.psyncs += 1;
    }

    /// Simulate a power-loss crash: everything `pwb`'d but not yet `psync`'d is lost; durable
    /// records remain. Returns a recovered view (the durable log).
    pub fn crash(&mut self) {
        self.volatile.clear();
    }

    /// Number of `psync` fences performed (persistence cost metric).
    pub fn psync_count(&self) -> usize {
        self.psyncs
    }
}

impl Default for PersistentMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// A durable SPSC FIFO queue backed by [`PersistentMemory`]. Each `enqueue`/`dequeue` persists a
/// single record with one `pwb`+`psync` pair (the paper's "a pair of persistence instructions
/// per operation"), so the durable log is always a valid linearization of completed operations.
///
/// > ⚠️ **Simulated durability only** — see the module-level warning. The "non-volatile memory"
/// > is an in-RAM `Vec`; nothing is written to disk/NVM and nothing survives a real crash or
/// > process exit. This type models the *algorithm*, not production durability.
pub struct PersistentQueue {
    mem: PersistentMemory,
    /// Volatile (cached) live contents, reconstructable from `mem` by [`recover`](Self::recover).
    live: std::collections::VecDeque<u64>,
}

impl PersistentQueue {
    /// Create an empty durable queue.
    pub fn new() -> Self {
        PersistentQueue {
            mem: PersistentMemory::new(),
            live: std::collections::VecDeque::new(),
        }
    }

    /// **Durable enqueue.** Persists the value (pwb + psync) *before* returning, so once this
    /// call completes the value is guaranteed to survive a crash.
    pub fn enqueue(&mut self, value: u64) {
        self.mem.pwb(Record::Enqueue(value));
        self.mem.psync(); // durable from here on
        self.live.push_back(value);
    }

    /// **Durable dequeue.** Persists a dequeue marker (pwb + psync) before returning the value,
    /// so a crash cannot resurrect an item the consumer already durably removed.
    pub fn dequeue(&mut self) -> Option<u64> {
        let value = self.live.pop_front()?;
        self.mem.pwb(Record::Dequeue);
        self.mem.psync();
        Some(value)
    }

    /// Enqueue *without* the final `psync` — the value is staged (`pwb`'d) but **not yet durable**.
    /// Models the window between writing and fencing; a [`crash`](Self::crash_now) here loses it.
    /// Used to test that un-acknowledged enqueues are correctly *not* recovered.
    pub fn enqueue_unsynced(&mut self, value: u64) {
        self.mem.pwb(Record::Enqueue(value));
        self.live.push_back(value);
        // deliberately no psync
    }

    /// Simulate a crash: drop all volatile (cached) state and un-`psync`'d records, keeping only
    /// durable memory. The in-RAM `live` deque is wiped — recovery must rebuild it from NVM.
    pub fn crash_now(&mut self) {
        self.mem.crash();
        self.live.clear();
    }

    /// **Recovery.** Replay the durable log to reconstruct the live FIFO contents after a crash.
    /// Returns the recovered queue contents in FIFO order. Guarantees: every durably-enqueued
    /// value that was not durably-dequeued is present, exactly once, in enqueue order.
    pub fn recover(&mut self) -> Vec<u64> {
        // Replay: Enqueue(v) appends v; Dequeue removes the current head. The durable log is, by
        // construction, a prefix of completed operations, so this yields a consistent state.
        let mut rebuilt = std::collections::VecDeque::new();
        for rec in &self.mem.durable {
            match rec {
                Record::Enqueue(v) => rebuilt.push_back(*v),
                Record::Dequeue => {
                    rebuilt.pop_front();
                }
            }
        }
        self.live = rebuilt.clone();
        rebuilt.into_iter().collect()
    }

    /// Current live length (volatile view).
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Whether the live queue is empty.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// `psync` count so far (persistence cost).
    pub fn psync_count(&self) -> usize {
        self.mem.psync_count()
    }
}

impl Default for PersistentQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_enqueues_survive_crash() {
        let mut q = PersistentQueue::new();
        for i in 0..100 {
            q.enqueue(i);
        }
        // Power loss: wipe volatile RAM, keep NVM.
        q.crash_now();
        let recovered = q.recover();
        assert_eq!(
            recovered,
            (0..100).collect::<Vec<_>>(),
            "all durable enqueues recovered in order"
        );
    }

    #[test]
    fn unsynced_enqueue_is_lost_on_crash() {
        let mut q = PersistentQueue::new();
        q.enqueue(1); // durable
        q.enqueue(2); // durable
        q.enqueue_unsynced(3); // staged but NOT psync'd
        q.crash_now();
        let recovered = q.recover();
        // 3 was never fenced, so a crash loses it — exactly the NVM durability boundary.
        assert_eq!(
            recovered,
            std::vec![1, 2],
            "un-acknowledged enqueue not recovered"
        );
    }

    #[test]
    fn dequeues_are_durable_no_resurrection() {
        let mut q = PersistentQueue::new();
        for i in 0..10 {
            q.enqueue(i);
        }
        assert_eq!(q.dequeue(), Some(0));
        assert_eq!(q.dequeue(), Some(1));
        // Crash after durable dequeues: the removed items must NOT come back.
        q.crash_now();
        let recovered = q.recover();
        assert_eq!(
            recovered,
            (2..10).collect::<Vec<_>>(),
            "dequeued items stay gone after crash"
        );
    }

    #[test]
    fn crash_at_every_point_is_consistent() {
        // Crash after each of N operations; recovery must always be a valid prefix with no
        // duplicates and no acknowledged loss.
        for crash_after in 0..=50 {
            let mut q = PersistentQueue::new();
            let mut model: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
            for op in 0..50u64 {
                if op % 3 == 2 {
                    // dequeue
                    let got = q.dequeue();
                    let want = model.pop_front();
                    assert_eq!(got, want, "dequeue matches model at op {op}");
                } else {
                    q.enqueue(op);
                    model.push_back(op);
                }
                if op == crash_after {
                    break;
                }
            }
            q.crash_now();
            let recovered = q.recover();
            let expected: Vec<u64> = model.into_iter().collect();
            assert_eq!(
                recovered, expected,
                "recovery consistent when crashing after op {crash_after}"
            );
        }
    }

    #[test]
    fn one_psync_pair_per_operation() {
        // The paper's headline: a pair of persistence instructions per op. We model one psync
        // per enqueue/dequeue, so the count equals the number of durable operations.
        let mut q = PersistentQueue::new();
        for i in 0..20 {
            q.enqueue(i);
        }
        for _ in 0..5 {
            q.dequeue();
        }
        assert_eq!(q.psync_count(), 25, "one psync per durable enqueue/dequeue");
    }
}
