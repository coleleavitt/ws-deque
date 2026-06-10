//! Head-to-head throughput: `ws-deque` vs `crossbeam-deque`.
//!
//! Two regimes that matter for a work-stealing deque:
//! 1. **owner push/pop** — the uncontended common path (no stealing).
//! 2. **owner vs N thieves** — the contended path that actually exercises the CAS protocol.
//!
//! `crossbeam-deque` stores elements inline via volatile (faster, but documented as
//! technically-UB and flagged by ThreadSanitizer); `ws-deque` boxes elements into `AtomicPtr`
//! cells (race-free, one allocation per push). This bench quantifies that trade-off.
//!
//! ```sh
//! cargo bench
//! ```
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use crossbeam_deque::{Steal as CbSteal, Worker as CbWorker};
use ws_deque::bwos::BwosWorker;
use ws_deque::idempotent::{IdempotentWorker, Take};
use ws_deque::inline::InlineWorker;
use ws_deque::{Steal, Worker};

const N: usize = 4096;

/// BWoS (block-based) vs crossbeam on owner push/pop. BWoS's in-block fast path is a plain slot
/// write + `Release` bump (no fence, no CAS) until a block boundary, so it should be competitive
/// with — or beat — crossbeam's inline volatile, while staying race-free (no UB).
fn bench_bwos_push_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_pop_bwos");

    group.bench_function("ws_bwos", |b| {
        // 64 blocks * 256 = 16384 capacity, comfortably above N so no boundary churn dominates.
        let w = BwosWorker::<u64>::with_blocks(64, 256);
        b.iter(|| {
            for i in 0..N as u64 {
                w.put(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.bench_function("crossbeam", |b| {
        let w = CbWorker::<u64>::new_lifo();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Boxed deque vs the inline (`Copy`) fast path vs crossbeam, on owner push/pop. Quantifies how
/// much of the boxed deque's overhead is the per-element allocation.
fn bench_inline_vs_boxed(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_pop_copy");

    group.bench_function("ws_inline", |b| {
        let w = InlineWorker::<u64>::new();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.bench_function("ws_boxed", |b| {
        let w = Worker::<u64>::new();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.bench_function("crossbeam", |b| {
        let w = CbWorker::<u64>::new_lifo();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Owner-only put/take throughput: the fence-free, CAS-free WS-MULT queue vs the exact-once
/// Chase-Lev deque. WS-MULT's `put` is a plain store (no fence), so this isolates the cost of
/// Chase-Lev's `SeqCst` fence on every `pop`.
fn bench_fencefree_put_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("put_take_owner");

    group.bench_function("ws_mult_fencefree", |b| {
        let mut w = IdempotentWorker::<u64>::new();
        b.iter(|| {
            for i in 0..N as u64 {
                w.put(black_box(i));
            }
            let mut sum = 0u64;
            while let Take::Got(v) = w.take() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.bench_function("chase_lev_exactonce", |b| {
        let w = Worker::<u64>::new();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.finish();
}

fn bench_push_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_pop");

    group.bench_function("ws_deque", |b| {
        let w = Worker::<u64>::new();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.bench_function("crossbeam", |b| {
        let w = CbWorker::<u64>::new_lifo();
        b.iter(|| {
            for i in 0..N as u64 {
                w.push(black_box(i));
            }
            let mut sum = 0u64;
            while let Some(v) = w.pop() {
                sum += v;
            }
            black_box(sum);
        });
    });

    group.finish();
}

const THIEVES: usize = 3;

fn ws_thief_loop(s: &ws_deque::Stealer<u64>, stop: &AtomicBool, stolen: &AtomicU64) {
    while !stop.load(Ordering::Relaxed) {
        if let Steal::Success(v) = s.steal() {
            stolen.fetch_add(v, Ordering::Relaxed);
        }
    }
}

fn cb_thief_loop(s: &crossbeam_deque::Stealer<u64>, stop: &AtomicBool, stolen: &AtomicU64) {
    while !stop.load(Ordering::Relaxed) {
        if let CbSteal::Success(v) = s.steal() {
            stolen.fetch_add(v, Ordering::Relaxed);
        }
    }
}

fn run_ws_contended() {
    let w = Worker::<u64>::new();
    let stop = Arc::new(AtomicBool::new(false));
    let stolen = Arc::new(AtomicU64::new(0));
    std::thread::scope(|scope| {
        for _ in 0..THIEVES {
            let s = w.stealer();
            let (stop, stolen) = (Arc::clone(&stop), Arc::clone(&stolen));
            scope.spawn(move || ws_thief_loop(&s, &stop, &stolen));
        }
        for i in 0..N as u64 {
            w.push(i);
        }
        while w.pop().is_some() {}
        stop.store(true, Ordering::Relaxed);
    });
    black_box(stolen.load(Ordering::Relaxed));
}

fn run_cb_contended() {
    let w = CbWorker::<u64>::new_lifo();
    let stop = Arc::new(AtomicBool::new(false));
    let stolen = Arc::new(AtomicU64::new(0));
    std::thread::scope(|scope| {
        for _ in 0..THIEVES {
            let s = w.stealer();
            let (stop, stolen) = (Arc::clone(&stop), Arc::clone(&stolen));
            scope.spawn(move || cb_thief_loop(&s, &stop, &stolen));
        }
        for i in 0..N as u64 {
            w.push(i);
        }
        while w.pop().is_some() {}
        stop.store(true, Ordering::Relaxed);
    });
    black_box(stolen.load(Ordering::Relaxed));
}

fn bench_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("owner_vs_thieves");
    group.bench_function("ws_deque", |b| b.iter(run_ws_contended));
    group.bench_function("crossbeam", |b| b.iter(run_cb_contended));
    group.finish();
}

/// The *realistic* job-queue case: the payload is an already-heap-allocated task
/// (`Box<dyn FnOnce()>`), as in any real executor (Rayon/Tokio enqueue boxed closures). Here
/// **both** deques pay one allocation per task regardless of internal storage, so ws-deque's
/// extra `AtomicPtr` box is amortized against work that was going to allocate anyway — the two
/// implementations should converge. This is the workload that matters for an executor, vs. the
/// `u64`-payload microbench above which maximizes the relative storage overhead.
fn bench_task_queue(c: &mut Criterion) {
    type Task = Box<dyn FnOnce() -> u64 + Send>;
    let mut group = c.benchmark_group("task_queue_boxed");
    let n = 2048usize;

    group.bench_function("ws_deque", |b| {
        let w = Worker::<Task>::new();
        b.iter(|| {
            for i in 0..n as u64 {
                w.push(Box::new(move || i.wrapping_mul(2654435761)));
            }
            let mut sum = 0u64;
            while let Some(task) = w.pop() {
                sum = sum.wrapping_add(task());
            }
            black_box(sum);
        });
    });

    group.bench_function("crossbeam", |b| {
        let w = CbWorker::<Task>::new_lifo();
        b.iter(|| {
            for i in 0..n as u64 {
                w.push(Box::new(move || i.wrapping_mul(2654435761)));
            }
            let mut sum = 0u64;
            while let Some(task) = w.pop() {
                sum = sum.wrapping_add(task());
            }
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_push_pop,
    bench_contended,
    bench_fencefree_put_take,
    bench_inline_vs_boxed,
    bench_task_queue,
    bench_bwos_push_pop
);
criterion_main!(benches);
