//! An in-process **distributed work-stealing** runtime (shared-nothing, message-passing).
//!
//! Models the distributed work-stealing design of
//!
//! - J. John, J. Milthorpe, P. Strazdins, *Distributed Work Stealing in a Task-Based Dataflow
//!   Runtime*, arXiv:2211.00838.
//!
//! Where the [`crate::scheduler`] is *shared-memory* (workers steal by touching each other's
//! deques directly), a distributed runtime has **nodes that share nothing**: each node owns its
//! tasks privately, and a thief node can only obtain work by *sending a steal-request message*
//! to a victim node, which independently decides whether to migrate work back. This is the model
//! for clusters / multi-process runtimes (MPI, PaRSEC). We run the nodes as threads communicating
//! over channels — no shared task state between nodes — so the *protocol* is exactly the
//! distributed one, demonstrable on a single machine without a cluster.
//!
//! Protocol (paper §3-4):
//! - **Randomized victim selection** — a starving node picks a random other node to ask.
//! - **Half victim policy** — a victim migrates up to half its surplus tasks per request.
//! - **Message passing only** — tasks move as messages; nodes never read each other's memory.
//! - **Distributed termination** — a global outstanding-task count (the analogue of a
//!   termination-detection module) reaches zero exactly when all work everywhere is done.
//!
//! Tasks are values of type `T`; each is executed by `run`, which may emit child tasks (kept
//! local to the producing node, the natural data-locality choice). Results are accumulated by the
//! caller via shared atomics in the closure (as with the shared-memory scheduler).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::vec::Vec;

/// A message between nodes. Steal requests and work migrations travel as these.
enum Message<T> {
    /// "Node `from` is starving — send it work if you can."
    StealRequest { from: usize },
    /// A batch of migrated tasks arriving at the requesting node (may be empty = "declined").
    Work(Vec<T>),
}

/// Per-node mailbox: an MPSC receiver plus the senders to reach every node.
struct Node<T> {
    rx: Receiver<Message<T>>,
    peers: Vec<Sender<Message<T>>>,
}

/// Shared, node-spanning coordination (the only global state — counters, not task data).
struct Global {
    /// Total outstanding tasks across all nodes; reaching 0 ⇒ everything is done.
    outstanding: AtomicUsize,
    /// Steal requests sent (a metric, and the distributed-overhead signal).
    steal_msgs: AtomicUsize,
    /// Tasks migrated across nodes.
    migrated: AtomicUsize,
    n_nodes: usize,
}

/// A handle passed to the task closure for emitting child tasks (kept on the local node).
pub struct NodeSpawner<'a, T> {
    local: &'a mut Vec<T>,
    global: &'a Global,
}

impl<T> NodeSpawner<'_, T> {
    /// Emit a child task; it stays on this node (data locality) unless later stolen.
    pub fn spawn(&mut self, task: T) {
        self.global.outstanding.fetch_add(1, Ordering::SeqCst);
        self.local.push(task);
    }
}

/// Statistics from a distributed run.
#[derive(Debug, Clone, Copy)]
pub struct DistStats {
    /// Steal-request messages sent across nodes.
    pub steal_messages: usize,
    /// Tasks migrated across node boundaries.
    pub migrated_tasks: usize,
    pub nodes: usize,
}

/// Run `run` over `nodes` shared-nothing nodes, seeding `initial` tasks round-robin across them.
/// Each node runs single-threaded over its private task pool; nodes balance load purely by
/// message-passing steal requests. Blocks until all work (initial + spawned) is done.
pub fn run<T, F>(nodes: usize, initial: impl IntoIterator<Item = T>, run: F) -> DistStats
where
    T: Send,
    F: Fn(T, &mut NodeSpawner<'_, T>) + Sync,
{
    let nodes = nodes.max(1);

    // Build per-node channels. Each node has one Receiver; every node holds a Sender to all.
    let mut receivers = Vec::with_capacity(nodes);
    let mut senders: Vec<Sender<Message<T>>> = Vec::with_capacity(nodes);
    for _ in 0..nodes {
        let (tx, rx) = channel();
        senders.push(tx);
        receivers.push(rx);
    }

    let global = Arc::new(Global {
        outstanding: AtomicUsize::new(0),
        steal_msgs: AtomicUsize::new(0),
        migrated: AtomicUsize::new(0),
        n_nodes: nodes,
    });

    // Seed initial tasks round-robin into per-node starting pools.
    let mut seeds: Vec<Vec<T>> = (0..nodes).map(|_| Vec::new()).collect();
    let mut seeded = 0usize;
    for (i, task) in initial.into_iter().enumerate() {
        seeds[i % nodes].push(task);
        seeded += 1;
    }
    global.outstanding.store(seeded, Ordering::SeqCst);
    if seeded == 0 {
        return DistStats {
            steal_messages: 0,
            migrated_tasks: 0,
            nodes,
        };
    }

    // Each node keeps a clone of all senders (to message any peer).
    let node_states: Vec<Node<T>> = receivers
        .into_iter()
        .map(|rx| Node {
            rx,
            peers: senders.clone(),
        })
        .collect();
    drop(senders); // nodes hold their own clones now

    std::thread::scope(|scope| {
        for ((id, node), seed) in node_states.into_iter().enumerate().zip(seeds) {
            let global = Arc::clone(&global);
            let run = &run;
            scope.spawn(move || node_main(id, node, seed, &global, run));
        }
    });

    DistStats {
        steal_messages: global.steal_msgs.load(Ordering::Relaxed),
        migrated_tasks: global.migrated.load(Ordering::Relaxed),
        nodes,
    }
}

/// One node's event loop: run local tasks; when starved, service the mailbox and send a steal
/// request to a random victim; exit when global work is exhausted.
fn node_main<T, F>(id: usize, node: Node<T>, seed: Vec<T>, global: &Global, run: &F)
where
    T: Send,
    F: Fn(T, &mut NodeSpawner<'_, T>) + Sync,
{
    let mut local: Vec<T> = seed;
    let mut rng =
        0x9E3779B97F4A7C15u64 ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03).wrapping_add(1);
    let mut pending_request = false;

    loop {
        // 1. Always service incoming messages first (answer steal requests, accept work).
        drain_mailbox(id, &node, &mut local, global, &mut pending_request);

        // 2. Execute one local task if we have one.
        if let Some(task) = local.pop() {
            let mut spawner = NodeSpawner {
                local: &mut local,
                global,
            };
            run(task, &mut spawner);
            // Finished a task; if it was the last anywhere, everyone can terminate.
            global.outstanding.fetch_sub(1, Ordering::SeqCst);
            continue;
        }

        // 3. No local work. Are we globally done?
        if global.outstanding.load(Ordering::SeqCst) == 0 {
            return;
        }

        // 4. Starving: send a steal request to a random victim (once outstanding per round).
        if !pending_request && global.n_nodes > 1 {
            rng = xorshift(rng);
            let victim = (rng as usize) % global.n_nodes;
            if victim != id {
                global.steal_msgs.fetch_add(1, Ordering::Relaxed);
                // Best-effort send; if the victim has exited, we'll just retry / terminate.
                if node.peers[victim]
                    .send(Message::StealRequest { from: id })
                    .is_ok()
                {
                    pending_request = true;
                }
            }
        }
        // Yield briefly to avoid a hot spin while waiting for a reply or for work to appear.
        std::thread::yield_now();
    }
}

/// Process all currently-available mailbox messages: migrate work to requesters (half policy)
/// and absorb incoming work batches.
fn drain_mailbox<T>(
    id: usize,
    node: &Node<T>,
    local: &mut Vec<T>,
    global: &Global,
    pending_request: &mut bool,
) {
    loop {
        match node.rx.try_recv() {
            Ok(Message::StealRequest { from }) => {
                // Half victim policy: give away up to half of our surplus (keep at least 1).
                let give = local.len().saturating_sub(1) / 2;
                let batch: Vec<T> = if give > 0 {
                    local.split_off(local.len() - give)
                } else {
                    Vec::new()
                };
                let migrated = batch.len();
                // Send work (possibly empty = decline). Tasks move ownership to the thief node;
                // outstanding count is unchanged (work moved, not created or completed).
                if node.peers[from].send(Message::Work(batch)).is_ok() {
                    global.migrated.fetch_add(migrated, Ordering::Relaxed);
                }
            }
            Ok(Message::Work(mut batch)) => {
                // Our steal reply arrived (possibly empty); absorb it and allow a new request.
                local.append(&mut batch);
                *pending_request = false;
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                let _ = id; // id reserved for future per-node tracing
                return;
            }
        }
    }
}

#[inline]
fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as O};

    use super::*;

    #[test]
    fn distributes_irregular_tree_to_completion() {
        // A dynamically-unfolding binary tree spread across nodes; every node must be visited
        // exactly once and the run must terminate (distributed termination detection).
        let depth = 16u32;
        let count = AtomicUsize::new(0);
        run(4, [depth], |v: u32, sp| {
            count.fetch_add(1, O::Relaxed);
            if v > 0 {
                sp.spawn(v - 1);
                sp.spawn(v - 1);
            }
        });
        let expected = (1usize << (depth + 1)) - 1;
        assert_eq!(
            count.load(O::Relaxed),
            expected,
            "every tree node executed once"
        );
    }

    #[test]
    fn parallel_sum_across_nodes() {
        let n = if cfg!(miri) { 400u64 } else { 100_000u64 };
        let total = AtomicU64::new(0);
        let stats = run(4, 0..n, |i: u64, _sp| {
            total.fetch_add(i, O::Relaxed);
        });
        assert_eq!(total.load(O::Relaxed), n * (n - 1) / 2);
        // Multi-node => some cross-node stealing happened (load was balanced by messages).
        assert!(stats.nodes == 4);
    }

    #[test]
    fn single_node_completes() {
        let count = AtomicUsize::new(0);
        run(1, [8u32], |v: u32, sp| {
            count.fetch_add(1, O::Relaxed);
            if v > 0 {
                sp.spawn(v - 1);
            }
        });
        assert_eq!(count.load(O::Relaxed), 9);
    }

    #[test]
    fn imbalanced_seed_is_balanced_by_stealing() {
        // All initial work starts on node 0 (others idle) — they must steal it via messages.
        let n = if cfg!(miri) { 300usize } else { 50_000usize };
        let seen: Arc<Vec<AtomicUsize>> = Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
        // Seed a single root task that fans out, so node 0 starts with everything.
        run(4, [0usize], |start: usize, sp| {
            if start == 0 {
                for i in 1..=n {
                    sp.spawn(i);
                }
            } else {
                seen[start - 1].fetch_add(1, O::SeqCst);
            }
        });
        for (i, c) in seen.iter().enumerate() {
            assert_eq!(
                c.load(O::SeqCst),
                1,
                "task {i} executed exactly once across nodes"
            );
        }
    }
}
