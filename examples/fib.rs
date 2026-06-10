//! A tiny multi-threaded work-stealing scheduler built on `ws-deque`.
//!
//! Each worker owns a `Worker<Task>` deque and holds `Stealer`s for every other worker.
//! It runs its own tasks LIFO (good locality) and, when its deque drains, steals FIFO from
//! a random victim — the classic Chase-Lev / Cilk scheduling loop. The demo workload is a
//! deliberately naive parallel `fib(n)` that spawns a task per recursive call, which stress
//! tests push/pop/steal far harder than a real fib ever would.
//!
//! ```sh
//! cargo run --example fib --release -- 34 8
//! ```
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use ws_deque::{Steal, Stealer, Worker};

/// A unit of work: compute `fib(n)` and add it into the shared total.
enum Task {
    Fib(u64),
}

struct Shared {
    stealers: Vec<Stealer<Task>>,
    total: AtomicU64,
    /// Number of tasks not yet completed; the run ends when this hits zero.
    outstanding: AtomicUsize,
}

fn worker_loop(me: usize, local: &Worker<Task>, shared: &Shared) {
    let n_workers = shared.stealers.len();
    let mut rng = 0x9E3779B97F4A7C15u64 ^ (me as u64).wrapping_mul(0xD1B54A32D192ED03);

    loop {
        // 1. Drain our own deque LIFO.
        while let Some(task) = local.pop() {
            run_task(task, local, shared);
        }

        // 2. Nothing left to do anywhere? We're done.
        if shared.outstanding.load(Ordering::Acquire) == 0 {
            return;
        }

        // 3. Try to steal from a random victim (xorshift RNG, no deps).
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let victim = (rng as usize) % n_workers;
        if victim != me {
            if let Steal::Success(task) = shared.stealers[victim].steal() {
                run_task(task, local, shared);
            }
        }
    }
}

fn run_task(task: Task, local: &Worker<Task>, shared: &Shared) {
    match task {
        Task::Fib(n) => {
            if n < 2 {
                shared.total.fetch_add(n, Ordering::Relaxed);
            } else {
                // Spawn two child tasks; account for the extra outstanding work first.
                shared.outstanding.fetch_add(2, Ordering::Release);
                local.push(Task::Fib(n - 1));
                local.push(Task::Fib(n - 2));
            }
        }
    }
    // This task itself is now complete.
    shared.outstanding.fetch_sub(1, Ordering::Release);
}

fn fib_reference(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

fn main() {
    let mut args = env::args().skip(1);
    let n: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(34);
    let workers: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    });

    // Build one Worker per thread and share every Stealer with everyone.
    let owners: Vec<Worker<Task>> = (0..workers).map(|_| Worker::new()).collect();
    let stealers: Vec<Stealer<Task>> = owners.iter().map(|w| w.stealer()).collect();

    let shared = Arc::new(Shared {
        stealers,
        total: AtomicU64::new(0),
        outstanding: AtomicUsize::new(1),
    });

    // Seed the root task onto worker 0's deque.
    owners[0].push(Task::Fib(n));

    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        // Move each owned `Worker` into its own thread (`Worker` is `Send` but not `Sync`).
        for (me, local) in owners.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            scope.spawn(move || worker_loop(me, &local, &shared));
        }
    });
    let elapsed = start.elapsed();

    let got = shared.total.load(Ordering::Relaxed);
    let want = fib_reference(n);
    println!("fib({n}) = {got} over {workers} workers in {elapsed:?}");
    assert_eq!(got, want, "work-stealing result {got} != reference {want}");
    println!("OK (matches reference {want})");
}
