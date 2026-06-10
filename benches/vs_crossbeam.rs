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
use ws_deque::idempotent::{IdempotentWorker, Take};
use ws_deque::{Steal, Worker};

const N: usize = 4096;

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

criterion_group!(
    benches,
    bench_push_pop,
    bench_contended,
    bench_fencefree_put_take
);
criterion_main!(benches);
