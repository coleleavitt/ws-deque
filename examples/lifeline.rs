//! The lifeline-graph scheduler driving an irregular, dynamically-unfolding workload.
//!
//! Unlike `examples/fib.rs` (which hand-rolls a busy-wait stealing loop directly on the
//! deque), this uses the high-level [`ws_deque::scheduler::run`] API: idle workers park
//! spin-free and are woken via the hypercube lifeline graph when work appears, with clean
//! distributed termination.
//!
//! Workload: an *Unbalanced Tree Search* — the canonical irregular-parallelism benchmark.
//! Each node pseudo-randomly decides how many children to spawn, so the tree shape is wildly
//! uneven and static partitioning would fail badly; only dynamic load balancing keeps all
//! cores busy.
//!
//! ```sh
//! cargo run --example lifeline --release -- 8 30 3   # workers, max_depth, branch
//! ```
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

use ws_deque::scheduler::{Spawner, run};

/// One UTS node: a 64-bit hash seed. Children derive their seeds from the parent's.
#[derive(Clone, Copy)]
struct Node {
    seed: u64,
    depth: u32,
}

/// splitmix64 — a fast, well-distributed hash for deriving child seeds deterministically.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Expand one node: decide its child count from its hash, spawn them, and count this node.
fn visit(node: Node, max_depth: u32, branch: u64, nodes: &AtomicU64, sp: &Spawner<'_, Node>) {
    nodes.fetch_add(1, Ordering::Relaxed);
    if node.depth >= max_depth {
        return;
    }
    let h = splitmix64(node.seed);
    // Branching factor 0..=branch, biased so the tree is irregular but finite.
    let children = h % (branch + 1);
    for k in 0..children {
        sp.spawn(Node {
            seed: splitmix64(node.seed ^ (k.wrapping_mul(0x2545F4914F6CDD1D))),
            depth: node.depth + 1,
        });
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let workers: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    });
    // `max_depth` caps recursion; `branch` is the max children per node. Larger = bigger tree.
    let max_depth: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(30);
    let branch: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);

    let nodes = AtomicU64::new(0);
    let root = Node {
        seed: 0xC0FFEE,
        depth: 0,
    };

    let start = std::time::Instant::now();
    run(workers, [root], |node, sp| {
        visit(node, max_depth, branch, &nodes, sp);
    });
    let elapsed = start.elapsed();

    let total = nodes.load(Ordering::Relaxed);
    println!(
        "Unbalanced Tree Search: {total} nodes across {workers} workers \
         (max_depth={max_depth}, branch=0..={branch}) in {elapsed:?}"
    );
    println!(
        "throughput: {:.1} M nodes/s",
        total as f64 / elapsed.as_secs_f64() / 1e6
    );
}
