//! Acar-style **work-inflation decomposition** of parallel cost.
//!
//! From U. Acar, A. Charguéraud, M. Rainey, *Parallel Work Inflation, Memory Effects, and
//! their Empirical Analysis* (arXiv:1709.03767): a parallel program's slowdown vs. its
//! sequential baseline decomposes into four non-overlapping factors —
//!
//! 1. **Algorithmic overhead** — the parallel *algorithm* doing more total work than the
//!    sequential baseline (here ~0: same arithmetic).
//! 2. **Scheduling overhead** — creating/stealing tasks, load balancing. Isolated by running
//!    the parallel code on **one** worker (no stealing) and comparing to the sequential baseline.
//! 3. **Lack of parallelism** — idle workers; not the focus here.
//! 4. **Work inflation** — the *same* operations costing more under parallel execution
//!    (cache coherence, false sharing, atomics, memory bus contention). Isolated as the extra
//!    total CPU-work when going from 1 worker to N workers.
//!
//! We also use it to attribute the **boxed-vs-inline deque** gap: the boxed deque's extra cost
//! is work inflation (an allocation + atomic-pointer chase per element), not algorithm.
//!
//! ```sh
//! cargo run --example work_inflation --release
//! ```
use std::time::Instant;

use ws_deque::Worker;
use ws_deque::inline::InlineWorker;
use ws_deque::scheduler::run_with_config;

const N: u64 = 4_000_000;

/// The kernel: a compute-heavy hash mix per element, so the work is CPU-bound (parallelism
/// actually pays off) rather than memory-bound (where it wouldn't).
fn kernel(lo: u64, hi: u64) -> u64 {
    let mut acc = 0u64;
    for i in lo..hi {
        let mut x = i.wrapping_mul(0x9E3779B97F4A7C15);
        // A few rounds of splitmix-style mixing to give each element real ALU work.
        for _ in 0..8 {
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58476D1CE4E5B9);
            x ^= x >> 27;
        }
        acc = acc.wrapping_add(x);
    }
    acc
}

fn time<R>(label: &str, f: impl FnOnce() -> R) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  {label:<34} {ms:>9.2} ms");
    (r, ms)
}

fn main() {
    println!("Work-inflation decomposition (sum kernel, N = {N})\n");

    // 1. Sequential baseline.
    let (seq_result, seq_ms) = time("sequential baseline", || kernel(0, N));

    // 2. Parallel on ONE worker — isolates scheduling overhead (task creation, no stealing).
    // `div_ceil` so 64 chunks fully cover [0, N) (the last chunk is capped at N).
    let chunk = N.div_ceil(64).max(1);
    let par1 = std::sync::atomic::AtomicU64::new(0);
    let (_stats1, p1_ms) = time("parallel, 1 worker", || {
        run_with_config(1, 1, 1, 0..64u64, |c, _sp| {
            let lo = c * chunk;
            let hi = (lo + chunk).min(N);
            par1.fetch_add(kernel(lo, hi), std::sync::atomic::Ordering::Relaxed);
        })
    });
    assert_eq!(par1.load(std::sync::atomic::Ordering::Relaxed), seq_result);

    // 3. Parallel on N workers — total wall-clock; with the per-core work we can infer inflation.
    let workers = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let parn = std::sync::atomic::AtomicU64::new(0);
    let (statsn, pn_ms) = time(&format!("parallel, {workers} workers"), || {
        run_with_config(workers, 1, 1, 0..64u64, |c, _sp| {
            let lo = c * chunk;
            let hi = (lo + chunk).min(N);
            parn.fetch_add(kernel(lo, hi), std::sync::atomic::Ordering::Relaxed);
        })
    });
    assert_eq!(parn.load(std::sync::atomic::Ordering::Relaxed), seq_result);

    println!("\n  decomposition:");
    println!(
        "    scheduling overhead   = {:>6.1}%   (1-worker parallel vs sequential)",
        (p1_ms / seq_ms - 1.0) * 100.0
    );
    println!(
        "    parallel speedup      = {:>6.2}x   ({workers} workers vs sequential)",
        seq_ms / pn_ms
    );
    println!("    ideal speedup         = {:>6.2}x", workers as f64);
    let efficiency = (seq_ms / pn_ms) / workers as f64;
    println!(
        "    efficiency            = {:>6.1}%   (lost to inflation + imbalance)",
        efficiency * 100.0
    );
    println!(
        "    steals = {}  attempts = {}  hit-rate = {:.1}%",
        statsn.steals,
        statsn.steal_attempts,
        if statsn.steal_attempts > 0 {
            100.0 * statsn.steals as f64 / statsn.steal_attempts as f64
        } else {
            0.0
        }
    );

    // --- Deque boxing as work inflation: same op count, different per-op cost. ---
    println!("\n  deque per-op cost (boxed AtomicPtr vs inline AtomicU64, push+pop {N} items):");
    let boxed = Worker::<u64>::new();
    let (_b, b_ms) = time("boxed deque", || {
        for i in 0..N {
            boxed.push(i);
        }
        let mut s = 0u64;
        while let Some(v) = boxed.pop() {
            s = s.wrapping_add(v);
        }
        s
    });
    let inline = InlineWorker::<u64>::new();
    let (_i, i_ms) = time("inline deque", || {
        for i in 0..N {
            inline.push(i);
        }
        let mut s = 0u64;
        while let Some(v) = inline.pop() {
            s = s.wrapping_add(v);
        }
        s
    });
    println!(
        "    boxed/inline ratio    = {:>6.2}x   (this gap is work inflation: alloc + ptr-chase)",
        b_ms / i_ms
    );
}
