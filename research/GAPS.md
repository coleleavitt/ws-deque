# Gap analysis: papers → `ws-deque`

What the source papers (see `SYNTHESIS.md`) contain, and what this crate does vs defers.
Status: ✅ implemented · 🟡 partial · ⬜ deferred (with rationale).

## Chase & Lev, *Dynamic Circular Work-Stealing Deque* (SPAA'05) — the core paper

| § | Idea | Status | Notes |
| --- | --- | --- | --- |
| §2 | Cyclic array, monotone `top`, owner push/pop bottom + thief CAS steal top | ✅ | `Worker` / `Stealer`. |
| §2 | Growable buffer (double on full) | ✅ | `Buffer::grow`. |
| §2.3 | Cache a local upper bound on `top` to avoid reading contended `top` every push | ⬜ | Pure micro-opt; would help the push hot path. Deferred. |
| §3 | **Array shrinking** when `b−t < cap/K`, K≥3 | ✅ | `Worker::perhaps_shrink` on the non-racing pop branch + `shrinks_after_draining` test. |
| §3.1 | Shrink-without-copying (retain chain of smaller arrays, low-water-mark) | ⬜ | Optimization on top of §3; we always relocate. |
| §3.2 | Combine multiple shrinks (a5→a1 directly) | ⬜ | Same. |
| §4 | **Shared buffer pool / GC-free reclamation** by bumping `top`+`bottom` by array size to abort in-flight thieves | 🟡 | We instead use **retain-until-drop**: retired buffers live until the deque drops (bounded by `O(log max_len)` arrays). Simpler, dependency-free, race-free; trades a bounded amount of memory for not needing the abort trick or epoch GC. |
| §2 | Correct C11 memory orderings | ✅ | Per Lê et al.; **loom**-model-checked + **ThreadSanitizer**-clean. |

## Lê, Pop, Cohen, Zappa Nardelli, *Correct and Efficient Work-Stealing for Weak Memory Models* (PPoPP'13)

| Idea | Status | Notes |
| --- | --- | --- |
| Relaxed/Acquire/Release + SeqCst-fence ordering scheme | ✅ | Exactly the orderings used; see module docs. |
| **Race-free slot access** (vs crossbeam's technically-UB `volatile`) | ✅ | Slots are `AtomicPtr<T>` cells. This is the key correctness win — see `benches/` for the cost. |

## Choi, *Formal Verification of Chase-Lev Deque in Concurrent Separation Logic* (arXiv 2309.03642)

| Idea | Status | Notes |
| --- | --- | --- |
| Machine-checked linearizability of push/pop/steal | 🟡 | We don't reproduce the Iris proof, but we approximate it operationally with **loom** (exhaustive interleaving model-checking of the same orderings). |

## Deters, Wu, Xu, Lee, *NUMA-Aware Work-First Platform / NUMA-WS* (arXiv 1806.11128)

| Mechanism | Status | Notes |
| --- | --- | --- |
| **Locality-biased steals** (bias victim choice to same-socket) | ⬜ | Needs a multi-deque scheduler + topology; out of scope for a single-deque primitive. A scheduler built on `ws-deque` (see `examples/fib.rs`) is where this belongs. |
| **Lazy work pushing** (push task to honor a locality hint, charge span not work) | ⬜ | Scheduler-level, same as above. |
| Locality-hint API | ⬜ | Scheduler-level. |
| Work-first principle (cost on the steal, not the work, path) | ✅ | Honored: owner push/pop take no CAS on the common path; only steals and last-element pops CAS. |

## Production-deque features (industry practice, beyond the papers)

| Feature | Status | Notes |
| --- | --- | --- |
| **`steal_batch` / steal-half** (Tokio, crossbeam, Rayon) | ✅ | `Stealer::steal_batch_and_pop` moves ~half the victim's work in one CAS. Concurrent + TSan-tested. |
| Index wraparound on overflow | ⬜ | We use monotone `isize`; Chase-Lev assumes no overflow (a 64-bit index at 10⁹ ops/s lasts ~500 years). crossbeam handles wrapping. |
| **loom model checking** | ✅ | `--cfg loom` suite. |
| **ThreadSanitizer** | ✅ | All concurrent tests + the `fib` example run clean under `-Zsanitizer=thread`. |
| FIFO (`new_fifo`) variant | ⬜ | Only the LIFO owner deque is implemented. |
| Bench vs `crossbeam-deque` | ✅ | `benches/vs_crossbeam.rs`. |

### Benchmark reality (this machine, `cargo bench`, N=4096)

| Workload | ws-deque | crossbeam | Ratio |
| --- | ---: | ---: | ---: |
| push/pop (uncontended) | ~119 µs | ~34 µs | ~3.5× slower |
| owner vs 3 thieves | ~1.97 ms | ~277 µs | ~7× slower |

**Why, and the trade-off:** `ws-deque` boxes every element into an `AtomicPtr<T>` cell, so
each `push` allocates and the steal path chases a pointer. That is the price of being
*genuinely* race-free (TSan- and loom-clean). `crossbeam-deque` stores elements inline using
`read_volatile`/`write_volatile`, which it documents as "technically a data race and therefore
UB" — faster, but TSan flags it. For a job queue that enqueues `Box`/`Arc` tasks anyway (the
common case), the extra allocation is largely amortized. Closing the gap for `T: Copy` /
small-value payloads (an inline atomic-cell fast path) is the obvious next optimization.

## Additional literature pulled this round

- John, Milthorpe, Strazdins, *Distributed Work Stealing in a Task-Based Dataflow Runtime*
  (arXiv 2211.00838) — extends work stealing across nodes; relevant only to a distributed
  scheduler, not a shared-memory deque.
- Khatiri, Trystram, Wagner, *Work Stealing Simulator* (arXiv 1910.02803) — models steal
  latency; useful for evaluating victim-selection policies if we add NUMA bias.
- Suksompong, Leiserson, Schardl, *On the Efficiency of Localized Work Stealing* (arXiv
  1804.04773) — bounds the overhead of locality-biased stealing; the theory backing a future
  NUMA scheduler.

> Searched arXiv (semantic + keyword) and DOI/Unpaywall for newer non-blocking / weak-memory
> deque work (idempotent/relaxed deques, BWoS-style bounded queues). The canonical newer
> results are paywalled or conference-only; the verification (Choi) and weak-memory (Lê)
> papers already captured here are the load-bearing ones for a correct implementation.
