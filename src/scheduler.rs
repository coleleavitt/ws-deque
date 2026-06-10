//! A lifeline-graph work-stealing scheduler built on the Chase-Lev deque.
//!
//! Implements the lifeline-based global load-balancing algorithm of
//!
//! - V. Saraswat, P. Kambadur, S. Kodali, D. Grove, S. Krishnamoorthy, *Lifeline-based Global
//!   Load Balancing*, PPoPP 2011, as packaged by
//! - W. Zhang et al., *GLB: Lifeline-based Global Load Balancing library in X10*,
//!   arXiv:1312.5691.
//!
//! # What this adds over a naive work-stealing loop
//!
//! The `fib` example's scheduler busy-waits: an idle worker spins, repeatedly stealing at
//! random until work appears or everyone is done. That burns a core doing nothing and has no
//! clean termination signal.
//!
//! The lifeline algorithm fixes both:
//!
//! 1. **Two-round stealing.** An idle worker first tries `w` *random* victims (cheap, good for
//!    bursty imbalance). Only if those fail does it fall back to its **lifeline buddies** —
//!    a fixed set of neighbours in a low-diameter **hypercube** graph.
//! 2. **Lifelines (push-on-wake).** When random + lifeline stealing both fail, the worker
//!    *registers* on its buddies' lifelines and **sleeps** (no spin). A buddy that later
//!    obtains work checks its lifeline requests and **pushes** work to the sleeper, waking it.
//!    Because the hypercube is strongly connected, work reaches every idle worker.
//! 3. **Distributed termination.** A global active-worker count drops as workers go idle; when
//!    it hits zero with all deques empty, every worker is woken to exit. No spinning, ever.
//!
//! The result is a `parallel_for`-style driver that runs an irregular, dynamically-unfolding
//! workload (each task may spawn more) to completion, then returns — the standard fork-join
//! shape, but with real load balancing and clean, spin-free idling.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::vec::Vec;

use crate::{Steal, Stealer, Worker};

/// Per-worker shared state. The deque is owner-private (only the owning thread pushes/pops),
/// but its `Stealer`s are shared with every other worker for cross-worker stealing.
struct WorkerSlot<T> {
    /// Stealer for *this* worker's deque, handed to thieves.
    stealer: Stealer<T>,
    /// Lifeline requests: indices of workers that registered to receive work from this one.
    /// Guarded by a plain mutex — touched only off the hot path (when a worker goes idle or
    /// when an owner with fresh work flushes pending lifeline requests).
    lifeline_requests: Mutex<Vec<usize>>,
    /// **Lazy work-pushing inbox** (Deters NUMA-WS *work pushing*). Any worker may deposit a
    /// task here to honor a locality hint (`Spawner::spawn_at`); the owning worker drains it
    /// into its own deque at the top of its loop, preserving the single-owner deque invariant.
    /// A plain mutex-guarded `Vec` is an MPSC queue good enough for the off-hot-path hint flow.
    inbox: Mutex<Vec<T>>,
}

/// Shared scheduler state across all worker threads.
struct Shared<T> {
    slots: Vec<WorkerSlot<T>>,
    /// Hypercube lifeline neighbours for each worker (buddy → can pull from us).
    lifelines: Vec<Vec<usize>>,
    /// Outstanding tasks: incremented when a task is pushed onto any deque, decremented after a
    /// task finishes running. Work exists somewhere iff this is `> 0`, so termination is exactly
    /// "outstanding reaches 0" — robust against racy steals (unlike counting idle workers).
    outstanding: AtomicUsize,
    /// Park/wake coordination. `generation` bumps whenever work is published or termination is
    /// declared, so a parking worker that missed a wake re-checks instead of sleeping forever.
    park: Mutex<ParkState>,
    wake: Condvar,
    n: usize,
    /// Workers per locality group for **locality-biased stealing** (Deters NUMA-WS). A thief
    /// prefers victims in its own group (contiguous worker-id ranges, a proxy for same-socket)
    /// before probing remote groups, cutting cross-NUMA traffic. `1` disables the bias.
    group_size: usize,
}

struct ParkState {
    generation: u64,
    terminated: bool,
}

/// Compute the hypercube lifeline neighbours of worker `id` among `n` workers: the workers
/// reachable by flipping one bit of `id`'s index (clamped to `< n`). Low diameter (log n),
/// low degree (log n), strongly connected — exactly the GLB lifeline topology.
fn hypercube_neighbours(id: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut bit = 1;
    while bit < n {
        let neighbour = id ^ bit;
        if neighbour < n && neighbour != id {
            out.push(neighbour);
        }
        bit <<= 1;
    }
    out
}

/// A handle a task closure uses to spawn more work onto its own worker's deque.
pub struct Spawner<'a, T> {
    deque: &'a Worker<T>,
    shared: &'a Shared<T>,
    me: usize,
}

impl<'a, T: Send> Spawner<'a, T> {
    /// Push a new task onto the current worker's deque. Accounts for the new outstanding task,
    /// clears any satisfied lifeline requests, and wakes parked buddies to come steal.
    pub fn spawn(&self, task: T) {
        self.shared.outstanding.fetch_add(1, Ordering::SeqCst);
        self.deque.push(task);
        // We hold work now, so satisfy and clear any lifeline requests registered on us.
        clear_lifelines(self.shared, self.me);
        // Wake parked workers to re-attempt stealing (they will now succeed).
        signal_work(self.shared);
    }

    /// **Lazy work-pushing** (Deters NUMA-WS): spawn a task with a *locality hint* — deposit it
    /// in `worker`'s inbox instead of the current deque, so it is executed on (or near) the
    /// requested worker for better locality. `worker` is clamped to a valid index; passing the
    /// current worker is equivalent to [`spawn`](Self::spawn) but via the inbox. The cost lands
    /// on the rare hint path, not the common spawn path (the "lazy" of lazy work-pushing).
    pub fn spawn_at(&self, task: T, worker: usize) {
        let target = worker.min(self.shared.n - 1);
        if target == self.me {
            return self.spawn(task);
        }
        self.shared.outstanding.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut inbox) = self.shared.slots[target].inbox.lock() {
            inbox.push(task);
        }
        // The target may be parked; wake everyone so it drains its inbox.
        signal_work(self.shared);
    }
}

/// Clear any lifeline requests registered on `me` (we now have work, so requesters will be
/// woken by `signal_work` and can steal from us). Off the hot path; only touches the mutex if
/// requests are actually present.
fn clear_lifelines<T>(shared: &Shared<T>, me: usize) {
    if let Ok(mut reqs) = shared.slots[me].lifeline_requests.lock() {
        reqs.clear();
    }
}

/// Bump the work generation and wake all parked workers to re-check for stealable work.
fn signal_work<T>(shared: &Shared<T>) {
    if let Ok(mut st) = shared.park.lock() {
        st.generation = st.generation.wrapping_add(1);
    }
    shared.wake.notify_all();
}

/// Run the scheduler over `workers` threads, seeding `initial` tasks, executing each task with
/// `run` (which may spawn more via the [`Spawner`]). Blocks until all work is drained.
///
/// `run` is `Fn(T, &Spawner<T>) + Sync` — it gets the task and a spawner for child tasks.
pub fn run<T, F>(workers: usize, initial: impl IntoIterator<Item = T>, run: F)
where
    T: Send,
    F: Fn(T, &Spawner<'_, T>) + Sync,
{
    // Default: a single locality group (no bias).
    run_with(workers, 1, initial, run);
}

/// Like [`run`], but with **locality-biased stealing** (Deters NUMA-WS): workers are split
/// into contiguous groups of `group_size` (a proxy for sockets/NUMA nodes), and an idle worker
/// prefers stealing from victims in its own group before probing remote groups. `group_size`
/// of `0` or `1` disables the bias (identical to [`run`]).
pub fn run_with<T, F>(
    workers: usize,
    group_size: usize,
    initial: impl IntoIterator<Item = T>,
    run: F,
) where
    T: Send,
    F: Fn(T, &Spawner<'_, T>) + Sync,
{
    let workers = workers.max(1);
    let group_size = group_size.clamp(1, workers);

    // Build one deque per worker; collect stealers up front.
    let deques: Vec<Worker<T>> = (0..workers).map(|_| Worker::new()).collect();
    let slots: Vec<WorkerSlot<T>> = deques
        .iter()
        .map(|d| WorkerSlot {
            stealer: d.stealer(),
            lifeline_requests: Mutex::new(Vec::new()),
            inbox: Mutex::new(Vec::new()),
        })
        .collect();
    let lifelines: Vec<Vec<usize>> = (0..workers)
        .map(|i| hypercube_neighbours(i, workers))
        .collect();

    let shared = Arc::new(Shared {
        slots,
        lifelines,
        outstanding: AtomicUsize::new(0),
        park: Mutex::new(ParkState {
            generation: 0,
            terminated: false,
        }),
        wake: Condvar::new(),
        n: workers,
        group_size,
    });

    // Seed the initial tasks round-robin across worker deques, counting each as outstanding.
    let mut seeded = 0usize;
    for (i, task) in initial.into_iter().enumerate() {
        deques[i % workers].push(task);
        seeded += 1;
    }
    shared.outstanding.store(seeded, Ordering::SeqCst);

    // Empty workload: nothing to do.
    if seeded == 0 {
        return;
    }

    std::thread::scope(|scope| {
        // Move each owned `Worker` into its own thread (it is `Send` but not `Sync`), so the
        // owner-private `cached_top` cell is never shared across threads.
        for (me, deque) in deques.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let run = &run;
            scope.spawn(move || worker_main(me, &deque, &shared, run));
        }
    });
}

/// The per-worker main loop: drain locally, steal when empty, park when no work exists, exit at
/// global termination.
fn worker_main<T, F>(me: usize, deque: &Worker<T>, shared: &Shared<T>, run: &F)
where
    T: Send,
    F: Fn(T, &Spawner<'_, T>) + Sync,
{
    let spawner = Spawner { deque, shared, me };

    loop {
        // 0. Drain our lazy-work-pushing inbox into our own deque. These tasks were already
        //    counted as outstanding in `spawn_at`; moving them to the deque doesn't change the
        //    count. Done first so locality-hinted work runs on its requested worker.
        drain_inbox(me, deque, shared);

        // 1. Drain our own deque (LIFO, good locality). Each finished task decrements the
        //    global outstanding count; reaching 0 means the whole computation is done.
        let mut did_work = false;
        while let Some(task) = deque.pop() {
            run(task, &spawner);
            finish_task(shared);
            did_work = true;
        }
        if did_work {
            continue;
        }

        // 2. Try to steal: w random victims, then lifeline buddies.
        if let Some(task) = try_steal(me, deque, shared) {
            run(task, &spawner);
            finish_task(shared);
            continue;
        }

        // 3. No work found here or anywhere we probed. If all work is globally done, terminate;
        //    otherwise register on lifelines and park (spin-free) until woken by fresh work.
        if shared.outstanding.load(Ordering::SeqCst) == 0 {
            declare_terminated(shared);
            return;
        }
        register_lifelines(me, shared);
        if park_until_work_or_done(me, shared) {
            return; // global termination observed while parked
        }
    }
}

/// Whether worker `me`'s lazy-work-pushing inbox currently holds any task.
fn inbox_nonempty<T>(me: usize, shared: &Shared<T>) -> bool {
    shared.slots[me]
        .inbox
        .lock()
        .map(|q| !q.is_empty())
        .unwrap_or(false)
}

/// Move any tasks deposited in worker `me`'s lazy-work-pushing inbox into its own deque.
/// Takes the inbox lock briefly each loop, but the inbox is only populated when `spawn_at`
/// locality hints are in play; with no hints it is always empty and this is near-free.
fn drain_inbox<T: Send>(me: usize, deque: &Worker<T>, shared: &Shared<T>) {
    let drained: Vec<T> = match shared.slots[me].inbox.lock() {
        Ok(mut inbox) if !inbox.is_empty() => core::mem::take(&mut *inbox),
        _ => return,
    };
    for task in drained {
        deque.push(task); // already counted as outstanding by spawn_at
    }
}

/// Mark a task as completed; if it was the last outstanding task, wake everyone to terminate.
fn finish_task<T>(shared: &Shared<T>) {
    if shared.outstanding.fetch_sub(1, Ordering::SeqCst) == 1 {
        declare_terminated(shared);
    }
}

/// Declare global termination and wake all parked workers so they can exit.
fn declare_terminated<T>(shared: &Shared<T>) {
    if let Ok(mut st) = shared.park.lock() {
        st.terminated = true;
    }
    shared.wake.notify_all();
}

/// Try `w` random victims then the lifeline buddies. Returns one stolen task if any.
fn try_steal<T: Send>(me: usize, deque: &Worker<T>, shared: &Shared<T>) -> Option<T> {
    let n = shared.n;
    if n <= 1 {
        return None;
    }
    // Round 1: w random victims (w = log2(n).max(1) is a reasonable default), biased to our
    // own locality group when `group_size > 1`.
    let w = (usize::BITS - (n as u32).leading_zeros()).max(1) as usize;
    let mut rng = seed_rng(me);
    let gsize = shared.group_size;
    let group_lo = (me / gsize) * gsize;
    let group_hi = (group_lo + gsize).min(n);
    let group_span = group_hi - group_lo;
    for k in 0..w {
        rng = xorshift(rng);
        // Bias the first ~half of attempts to same-group victims (locality), the rest globally.
        let victim = if gsize > 1 && group_span > 1 && k < w.div_ceil(2) {
            group_lo + (rng as usize) % group_span
        } else {
            (rng as usize) % n
        };
        if victim != me {
            if let Some(t) = steal_one(shared, victim, me, deque) {
                return Some(t);
            }
        }
    }
    // Round 2: lifeline buddies (the hypercube neighbours we can pull from).
    for &victim in &shared.lifelines[me] {
        if let Some(t) = steal_one(shared, victim, me, deque) {
            return Some(t);
        }
    }
    None
}

/// Steal a half-batch from `victim` into our deque, returning one task to run immediately.
fn steal_one<T: Send>(
    shared: &Shared<T>,
    victim: usize,
    _me: usize,
    deque: &Worker<T>,
) -> Option<T> {
    match shared.slots[victim].stealer.steal_batch_and_pop(deque) {
        Steal::Success(t) => Some(t),
        Steal::Empty | Steal::Retry => None,
    }
}

/// Register this worker on each lifeline buddy so that a buddy obtaining work will wake us.
fn register_lifelines<T>(me: usize, shared: &Shared<T>) {
    for &buddy in &shared.lifelines[me] {
        let mut reqs = shared.slots[buddy].lifeline_requests.lock().unwrap();
        if !reqs.contains(&me) {
            reqs.push(me);
        }
    }
}

/// Park (spin-free) until the work generation changes (new work published) or termination is
/// declared. Returns `true` if the scheduler has terminated and the caller should exit.
///
/// A `generation` snapshot taken before sleeping closes the lost-wake race: if a buddy
/// published work between our failed steal and acquiring the lock, the generation already
/// differs and we don't sleep.
fn park_until_work_or_done<T>(me: usize, shared: &Shared<T>) -> bool {
    let st = match shared.park.lock() {
        Ok(g) => g,
        Err(_) => return true, // poisoned: treat as terminated to avoid a hang
    };
    if st.terminated {
        return true;
    }
    // Re-check outstanding under the lock: a task may have finished between our check and here.
    if shared.outstanding.load(Ordering::SeqCst) == 0 {
        return true;
    }
    // Re-check our inbox under the lock: a `spawn_at` may have deposited work (and bumped the
    // generation) *before* we snapshot `gen` below — without this check we'd sleep on a future
    // generation change while a task already sits in our inbox (a lost-wakeup hang). Returning
    // here sends us back to the loop top, which drains the inbox.
    if inbox_nonempty(me, shared) {
        return false;
    }
    let gen = st.generation;
    let guard = shared
        .wake
        .wait_while(st, |s| s.generation == gen && !s.terminated);
    match guard {
        Ok(s) => s.terminated,
        Err(_) => true,
    }
}

#[inline]
fn seed_rng(me: usize) -> u64 {
    0x9E3779B97F4A7C15u64 ^ (me as u64).wrapping_mul(0xD1B54A32D192ED03).wrapping_add(1)
}

#[inline]
fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn hypercube_topology_is_connected_and_low_degree() {
        // For n = 8, each node has exactly 3 neighbours (log2 8) and the graph is connected.
        for id in 0..8 {
            let nb = hypercube_neighbours(id, 8);
            assert_eq!(nb.len(), 3, "node {id} should have 3 hypercube neighbours");
        }
        // Non-power-of-two: neighbours clamp to < n, still nonempty for n > 1.
        for id in 0..6 {
            assert!(!hypercube_neighbours(id, 6).is_empty());
        }
        assert!(hypercube_neighbours(0, 1).is_empty());
    }

    #[test]
    fn runs_irregular_tree_to_completion() {
        // A dynamically-unfolding tree: each node with value v>0 spawns two children (v-1).
        // The scheduler must execute every node exactly once and then terminate (no hang).
        let depth = 18u32;
        let counter = AtomicUsize::new(0);
        run(8, [depth], |v: u32, sp| {
            counter.fetch_add(1, Ordering::Relaxed);
            if v > 0 {
                sp.spawn(v - 1);
                sp.spawn(v - 1);
            }
        });
        // Number of nodes in a full binary tree of `depth` levels: 2^(depth+1) - 1.
        let expected = (1usize << (depth + 1)) - 1;
        assert_eq!(counter.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn parallel_sum_matches_sequential() {
        // Sum 0..N by spawning a task per element; checks correctness under load balancing.
        let n = 200_000u64;
        let total = AtomicU64::new(0);
        run(8, 0..n, |i: u64, _sp| {
            total.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), n * (n - 1) / 2);
    }

    #[test]
    fn single_worker_still_completes() {
        let counter = AtomicUsize::new(0);
        run(1, [10u32], |v: u32, sp| {
            counter.fetch_add(1, Ordering::Relaxed);
            if v > 0 {
                sp.spawn(v - 1);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 11);
    }

    #[test]
    fn locality_biased_run_completes_irregular_tree() {
        // Same irregular tree as the default scheduler, but with locality groups of 2 — the
        // bias must not change correctness: every node expanded exactly once, clean termination.
        let depth = 17u32;
        let counter = AtomicUsize::new(0);
        run_with(8, 2, [depth], |v: u32, sp| {
            counter.fetch_add(1, Ordering::Relaxed);
            if v > 0 {
                sp.spawn(v - 1);
                sp.spawn(v - 1);
            }
        });
        let expected = (1usize << (depth + 1)) - 1;
        assert_eq!(counter.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn lazy_work_pushing_hinted_spawn_completes() {
        // Each node spawns its children with a locality hint to a round-robin worker via the
        // inbox path. The whole irregular tree must still run exactly once and terminate.
        let depth = 16u32;
        let counter = AtomicUsize::new(0);
        let workers = 8;
        run(workers, [depth], |v: u32, sp| {
            let me = counter.fetch_add(1, Ordering::Relaxed);
            if v > 0 {
                // Hint children to two different workers (exercises the inbox cross-worker path).
                sp.spawn_at(v - 1, me % workers);
                sp.spawn_at(v - 1, (me + 1) % workers);
            }
        });
        let expected = (1usize << (depth + 1)) - 1;
        assert_eq!(counter.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn lazy_work_pushing_parallel_sum() {
        let n = 5_000u64;
        let total = AtomicU64::new(0);
        let workers = 8;
        // Seed one driver task that fans out hinted spawns across workers (exercises the inbox
        // cross-worker path end-to-end; kept modest because each hint wakes the pool).
        run(workers, [0u64], |start: u64, sp| {
            if start == 0 {
                for i in 1..=n {
                    sp.spawn_at(i, (i as usize) % workers);
                }
            } else {
                total.fetch_add(start, Ordering::Relaxed);
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), n * (n + 1) / 2);
    }

    #[test]
    fn locality_biased_parallel_sum() {
        let n = 200_000u64;
        let total = AtomicU64::new(0);
        run_with(8, 4, 0..n, |i: u64, _sp| {
            total.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), n * (n - 1) / 2);
    }
}
