# Gap analysis: papers → `ws-deque`

What the source papers (see `SYNTHESIS.md`) contain, and what this crate does vs defers.
Status: ✅ implemented · 🟡 partial · ⬜ deferred (with rationale).

## ⭐ Breakthrough: priority work-stealing (`src/priority.rs`)

Wimmer, Cederman, Träff & Tsigas, *Configurable Strategies for Work-Stealing*
(arXiv:1305.6474). Implemented as `priority::PriorityWorker<T, K>` / `PriorityStealer<T, K>`.

**Why it matters.** Standard work-stealing is *oblivious* to task importance — execution order
is whatever LIFO/FIFO falls out of the deque. Wimmer et al. show that letting a task carry a
**priority** (a steal/execution-order hint) reduces the *total work* for search algorithms:
branch-and-bound and best-first/shortest-path expand promising nodes first, pruning the rest
before they are ever touched.

**Design:** `K` const-generic priority levels, each an independent **verified Chase-Lev
deque**. `push(task, level)`, and `pop`/`steal` scan highest-priority-first. Because each level
is the already-proven deque, the composition inherits exact-once semantics, race-freedom, and
the loom-checked orderings; the only new logic is the highest-first scan (plain control flow).

**Status:** ✅ `PriorityWorker`/`PriorityStealer`, level clamping, highest-first pop/steal.
4 tests (priority order, steal preference, clamp, concurrent no-loss), **ThreadSanitizer-clean**.
`examples/branch_and_bound.rs` runs A\* vs Dijkstra on a 120×120 obstacle grid: A\*
(heuristic-prioritized) expands **~5× fewer nodes** for the identical optimal cost — the
Wimmer "priority reduces total work" result, made measurable.

⬜ Deferred: a true unbounded priority (we bucket into `K` fixed levels — fine for A\*/B&B,
coarser than a comparison heap); per-task *strategies* beyond priority (call-conversion,
granularity hints) from the same paper.

## ⭐ Breakthrough: lifeline-graph scheduler (`src/scheduler.rs`)

Saraswat et al., *Lifeline-based Global Load Balancing* (PPoPP'11), packaged by Zhang et al.,
*GLB* (arXiv:1312.5691). Implemented as `scheduler::run` — a `parallel_for`/fork-join driver
over N worker threads, each owning a Chase-Lev deque.

**Why it matters.** A naive work-stealing loop (like `examples/fib.rs`) *busy-waits*: idle
workers spin-steal at random, burning cores, with no clean way to detect "everyone is done."
The lifeline algorithm fixes both problems:

- **Two-round stealing:** `w = ⌈log₂ N⌉` random victims first (cheap, handles bursty
  imbalance), then fall back to **lifeline buddies** — neighbours in a **hypercube** graph
  (degree & diameter both `log N`, strongly connected).
- **Spin-free idling:** when stealing fails, a worker *registers* on its buddies' lifelines and
  **parks** on a condvar (no spin). A worker that obtains work bumps a generation counter and
  wakes parked buddies, who then steal successfully.
- **Robust distributed termination:** a global `outstanding` task counter (incremented on
  spawn, decremented on completion) hits 0 *exactly* when the computation is done — wake-all
  and exit. This is sound against racy steals, unlike counting idle workers.

**Status:** ✅ `scheduler::run(workers, initial, |task, spawner| …)`; hypercube lifelines,
two-round stealing via `steal_batch_and_pop`, condvar parking, outstanding-based termination.
Tested (hypercube topology, irregular tree of 2¹⁹−1 dynamically-spawned nodes, 200k parallel
sum, single-worker), **25× stress-looped** for termination races, **ThreadSanitizer-clean**
(lib tests + a live 6-worker example run). `examples/lifeline.rs` runs an Unbalanced Tree
Search (~830k irregular nodes, 8 workers).

> **Bug found & fixed via TSan during this work:** the scheduler's near-empty deques exposed a
> latent double-free in `steal_batch_and_pop` — a single top-CAS claiming a multi-slot range
> could overlap the owner's CAS-free bottom `pop`. Reworked batch stealing to loop the
> individually-correct single `steal`, preserving the amortization while being provably sound.
> (Both the original single-CAS batch *and* its test passed before; only the scheduler's tight
> deques surfaced it — exactly why a real consumer + a sanitizer matter.)

## ⭐ Breakthrough: WS-MULT — fence-free, CAS-free work-stealing (`src/idempotent.rs`)

Castañeda & Piña, *Fully Read/Write Fence-Free Work-Stealing with Multiplicity*
(arXiv:2008.04424), implemented as `idempotent::IdempotentWorker` / `IdempotentStealer`.

**Why it matters.** Attiya et al.'s impossibility result proves that *exact-once*
work-stealing **must** use a read-modify-write (CAS) **or** a memory fence on the hot path —
Chase-Lev pays both (CAS on steal, `SeqCst` fence on every pop). WS-MULT escapes the
impossibility by relaxing to **multiplicity**: each task is delivered **≥1 times**, with the
count bounded by the number of concurrent consumers. Under that relaxation:

| Operation | Chase-Lev (exact-once) | WS-MULT (multiplicity) |
| --- | --- | --- |
| `put` / `push` | store + `Release` | **plain store, no CAS, no fence** |
| `take` / `pop` | `SeqCst` fence + maybe CAS | monotone `fetch_max`, **no retry** |
| `steal` | CAS-abort loop (`Retry`) | `fetch_max`, **no retry** |

**Mechanism:** the `head` is a monotone *MaxRegister* (`fetch_max`); it can only move forward,
so a slow consumer can never rewind the queue. Two consumers take the same task only if they
read the same `head` concurrently → multiplicity ≤ thread count (strictly better than the
*unbounded* re-extraction of Michael et al.'s idempotent work-stealing, which WS-MULT improves
on — see the paper's "Idempotent ≠ Multiplicity" section).

**Status:** ✅ FIFO WS-MULT with growable array, owner `put`/`take`, thief `steal`, `T: Clone`.
Tested (FIFO order, growth, no-phantom-tasks, concurrent multiplicity-bounded), **loom**
model-checked, **ThreadSanitizer-clean** (genuinely race-free *without* a fence — the whole
point). Measured ~1.4× faster owner put/take than Chase-Lev (137 µs vs 192 µs, N=4096),
isolating the eliminated fence+CAS. Use for idempotent workloads (parallel SAT, graph search,
fixpoint, branch-and-bound); use `Worker` when exactly-once is required.

✅ **WS-WMULT weak multiplicity** is now implemented (`idempotent::WeakStealer`,
`IdempotentStealer::into_weak`). Castañeda-Piña §4: the shared monotone `head` plus each
consumer's private cached max `r` form a *RangeMaxRegister* (Fig. 6) whose `RMaxRead`/`RMaxWrite`
are plain reads/writes — so the **entire consumer path is fence-free, with no `fetch_max` / no
RMW at all** (one notch cheaper than WS-MULT's `fetch_max`). The trade-off is weaker FIFO
ordering (a thief may briefly observe a stale head). Per-consumer state means `steal_weak` takes
`&mut self`. Tested (single-thief FIFO + concurrent at-least-once/bounded), TSan-clean.

✅ **Linked-list array backing** (Castañeda-Piña *approach 2*) is now implemented as the
`linked` module (`LinkedWorker`/`LinkedStealer`). The task store is a singly-linked list of
fixed-size nodes indexed by `(node, offset)`, giving two properties the contiguous deque can't
have together: **`put` is true constant-time** (link a fresh node instead of doubling+copying)
and **no reclamation problem at all** (nodes are append-only and never abandoned — no epoch GC,
no quiescent counter, no retired list). Same multiplicity semantics. Tested (FIFO across many
nodes, constant-time growth, concurrent at-least-once), TSan-clean. Trade-off: pointer-chasing
slot access vs. the contiguous deque's `& mask`.

✅ **B-WS-MULT bounded-multiplicity steal** is now implemented:
`IdempotentWorker::bounded(capacity)` + `IdempotentStealer::steal_exclusive()`. A thief claims
the head slot with a single `false→true` CAS on a per-slot flag, so **no two thieves ever take
the same task** (a `take`/`steal` pair still may — that's the paper's bounded guarantee). This
requires a *fixed-capacity* queue: per-slot claim flags are only race-free when there is exactly
one array for the queue's lifetime, so `steal_exclusive` debug-asserts bounded mode and `put`
returns `false` when full rather than growing. (This soundness boundary was caught by the
advisor — a growable claim scheme lets a thief on a retired array and one on the grown array
both claim the same logical slot.) Tested with a 6-thief no-double-take stress, TSan-clean.

All variants from this paper are now implemented (WS-MULT, B-WS-MULT, WS-WMULT, and both array
backings — contiguous in `idempotent`, linked-list in `linked`).

## Chase & Lev, *Dynamic Circular Work-Stealing Deque* (SPAA'05) — the core paper

| § | Idea | Status | Notes |
| --- | --- | --- | --- |
| §2 | Cyclic array, monotone `top`, owner push/pop bottom + thief CAS steal top | ✅ | `Worker` / `Stealer`. |
| §2 | Growable buffer (double on full) | ✅ | `Buffer::grow`. |
| §2.3 | Cache a local upper bound on `top` to avoid reading contended `top` every push | ✅ | `Worker.cached_top` (owner-only `Cell`): `push` consults the cached lower bound and only Acquire-loads the shared `top` when the buffer might be full. ~3% faster push/pop; TSan-clean. |
| §3 | **Array shrinking** when `b−t < cap/K`, K≥3 | ✅ | `Worker::perhaps_shrink` on the non-racing pop branch + `shrinks_after_draining` test. |
| §3.1 | Shrink-without-copying (retain chain of smaller arrays, low-water-mark) | ⬜ | Optimization on top of §3; we always relocate. |
| §3.2 | Combine multiple shrinks (a5→a1 directly) | ⬜ | Same. |
| §4 | **Buffer reclamation** (free grown/shrunk-out buffers mid-life) | ✅ | **Quiescent-state reclamation**: an `in_flight` steal counter brackets each thief's buffer dereference with symmetric `SeqCst` fences (the same protocol shape as top/bottom). The owner's `try_reclaim` frees the retired list only when it observes `in_flight == 0` — a point at which no thief holds a retired-buffer pointer and a new thief can only load the live buffer. Replaces retain-until-drop, so the backlog stays bounded under grow/shrink cycling. **loom-verified** (exhaustive grow-during-steal model), **TSan-clean** under a 200k grow-during-steal stress, 20× stress-looped. |
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
| **Locality-biased steals** (bias victim choice to same-socket) | ✅ | `scheduler::run_with(workers, group_size, …)`: workers split into contiguous locality groups; an idle thief biases ~half its random-victim probes to its own group before going global. Correctness-preserving (irregular tree + parallel sum complete under bias), TSan-clean + 15× stress-looped. (Behavioural locality *win* is workload/hardware-dependent and not asserted.) |
| **Lazy work pushing** (push task to honor a locality hint, charge span not work) | ✅ | `Spawner::spawn_at(task, worker)` deposits into the target worker's **lock-free Jiffy MPSC inbox** (`jiffy` module); the target (single consumer) drains it into its own deque at the loop top (preserving the single-owner invariant). Cost lands on the rare hint path. A **lost-wakeup hang** (a worker parking while its inbox held work) was caught by the parallel-sum test and fixed by re-checking the inbox before sleeping. TSan-clean + 15× stress. |
| Locality-hint API | ✅ | `Spawner::spawn_at` is the per-task hint; group-level bias (`run_with`) covers coarse NUMA placement. |
| Work-first principle (cost on the steal, not the work, path) | ✅ | Honored: owner push/pop take no CAS on the common path; only steals and last-element pops CAS. |

## Granularity control — heartbeat scheduling (Acar/Rainey; Wimmer spawn-to-call)

The single most impactful *performance* idea common to every work-stealing system: **too many
tiny tasks drown in scheduling overhead.** `scheduler::Spawner::spawn_or_call(task, serial)` and
`should_promote()` + `run_with_config(workers, group_size, heartbeat_interval, …)` implement
**amortized granularity control**: most spawn decisions run the work *inline* (a direct call, no
deque push / no atomic / no wake), and only one in every `heartbeat_interval` is *promoted* to a
real stealable task. This bounds scheduling overhead to a `1 / heartbeat_interval` fraction *no
matter how finely the program spawns* — the principled form of Wimmer's "spawn-to-call
conversion." Tested: a depth-16 binary tree (≈131k spawn decisions) completes correctly while
*promoting* < 25% of them; a divide-and-conquer sum via `spawn_or_call` returns the exact total.
TSan-clean.

## Empirical analysis — work-inflation decomposition (Acar, arXiv:1709.03767)

The crate now *attributes* its costs instead of reporting one number. `examples/work_inflation.rs`
decomposes parallel slowdown into Acar's four factors (algorithmic overhead / scheduling overhead
/ lack of parallelism / **work inflation** — the same ops costing more in parallel due to cache
coherence, false sharing, atomics). The scheduler exposes `Stats { tasks, steals, steal_attempts }`
via `run_with_config`. Measured on this machine: scheduling overhead ≈ **0.8%** (1-worker
parallel vs sequential), and crucially the **boxed-vs-inline deque gap is 3.66× of pure work
inflation** (allocation + atomic-pointer chase per element) — not algorithm. That number is the
honest "why" behind the `inline` fast path.

## Verification coverage matrix

Two tools, with honest limits. **loom** exhaustively model-checks the atomic interleavings of
*small hand-written scenarios* under a C11-style model (it cannot run the full test suite — the
state space explodes). **ThreadSanitizer** runs the *full* concurrent tests but only observes
interleavings that actually executed (it cannot *prove* race-freedom). Together they are strong;
neither alone is a proof.

| Module | loom | TSan | Notes |
| --- | :-: | :-: | --- |
| `lib` (Chase-Lev deque) | ✅ ×3 | ✅ | incl. grow-during-steal reclamation model |
| `inline` (Copy deque) | ✅ ×2 | ✅ | owner-pop-vs-thief + push-then-steal |
| `idempotent` (WS-MULT family) | ✅ | ✅ | multiplicity contract under interleaving |
| `linked` (linked backing) | ✅ | ✅ | take-vs-steal at-least-once |
| `jiffy` (MPSC injector) | ✅ | ✅ | 2-producer/1-consumer no-loss (tiny `BUFFER_SIZE` under loom) |
| `priority` | — | ✅ | composes verified deques; no new orderings |
| `scheduler` | — | ✅ | 10 tests incl. heartbeat + lazy-push, TSan-clean |

**Beyond loom — `RustMC` / GenMC (deferred tooling, arXiv:2502.06293).** loom only checks code
rewritten against `loom::sync`. **RustMC** extends the **GenMC** stateless model checker to real
Rust via LLVM IR, exhaustively exploring *all* executions of the actual compiled program —
including `unsafe`, raw-pointer atomics, and C/C++ FFI — under RC11. That is the principled way
to machine-check the hand-justified `AtomicPtr`/CAS orderings these modules rely on (loom
*approximates* this; GenMC would be a real proof). Documented as the next verification step; not
yet wired into CI. **Miri** is also currently broken on this toolchain (`cargo miri` panics in
its phase wrapper), so UB-detection (provenance/alignment/uninit) is presently via TSan only.

## Production-deque features (industry practice, beyond the papers)

| Feature | Status | Notes |
| --- | --- | --- |
| **`steal_batch` / steal-half** (Tokio, crossbeam, Rayon) | ✅ | `Stealer::steal_batch_and_pop` moves ~half the victim's work in one CAS. Concurrent + TSan-tested. |
| Index wraparound on overflow | ✅ | All `top`/`bottom` arithmetic uses `wrapping_add`/`wrapping_sub` and count-based loops, so the deque is correct even past `isize::MAX`. Tested by seeding indices at `isize::MAX - k` (`correct_across_index_wraparound`, `wraparound_concurrent_no_loss`). |
| **loom model checking** | ✅ | `--cfg loom` suite. |
| **ThreadSanitizer** | ✅ | All concurrent tests + the `fib` example run clean under `-Zsanitizer=thread`. |
| FIFO (`new_fifo`) variant | ✅ | `Worker::new_fifo()`: the owner pops the *oldest* task from the top via the shared steal core (a retry loop), matching crossbeam/Tokio `new_fifo`. FIFO-order + concurrent-no-loss tests, TSan-clean. |
| Bench vs `crossbeam-deque` | ✅ | `benches/vs_crossbeam.rs`. |

### Benchmark reality (this machine, `cargo bench`, N=4096)

| Workload (`cargo bench`) | ws-deque (boxed) | ws-deque (inline `Copy`) | crossbeam |
| --- | ---: | ---: | ---: |
| `push_pop` of `u64` | ~169 µs | **~47 µs** | ~32 µs |
| `owner_vs_thieves` (u64) | ~1.97 ms | — | ~277 µs |
| **`task_queue_boxed`** (`Box<dyn FnOnce>` jobs) | **~77 µs** | n/a | ~83 µs |

**Two honest stories, by payload.** (1) On a `u64` microbench the general `Worker`'s per-element
`AtomicPtr` box (an allocation crossbeam avoids via inline-but-technically-UB `volatile`) makes it
~5× slower — and the **`inline::InlineWorker<T: Copy>`** fast path closes that to ~1.5× by storing
small `Copy` values directly in `AtomicU64` cells (no alloc, still race-free since `Copy` has no
`Drop`). (2) But the `u64` case *maximizes* relative storage overhead and is **not** how executors
work. The realistic `task_queue_boxed` row uses `Box<dyn FnOnce>` jobs (what Rayon/Tokio actually
enqueue): there both deques pay the same per-task allocation, so ws-deque's box amortizes and the
two **converge to parity** (~77 µs vs ~83 µs). So: `inline` for small `Copy` payloads, default
`Worker` for task queues where it's already competitive *and* genuinely race-free (no UB).

## Additional literature pulled

**Implemented:**
- ⭐ Wimmer, Cederman, Träff & Tsigas, *Configurable Strategies for Work-Stealing*
  (arXiv 1305.6474) — **the priority work-stealing breakthrough above** (`src/priority.rs`).
- ⭐ Castañeda & Piña, *Fully Read/Write Fence-Free Work-Stealing with Multiplicity*
  (arXiv 2008.04424) — **the WS-MULT breakthrough** (`src/idempotent.rs`).
- ⭐ Saraswat et al. / Zhang et al., *Lifeline-based Global Load Balancing* / *GLB*
  (arXiv 1312.5691) — **the lifeline scheduler** (`src/scheduler.rs`).

**Safe memory reclamation — IMPLEMENTED (quiescent-state), with SMR papers for context:**
The deque now reclaims retired buffers mid-life via a **quiescent-state** scheme (an `in_flight`
steal counter + symmetric `SeqCst` fences; the owner frees the retired list only at `in_flight
== 0`). This is the dependency-free analogue of epoch-based reclamation — sufficient because the
deque has a *single* retiring thread (the owner). It is loom-verified and TSan-clean. The
following heavier SMR schemes were read as the alternative design points (deferred — our
single-retirer setting does not need their generality):
- Nikolaev & Ravindran, *Hyaline: Snapshot-Free, Transparent, Robust Memory Reclamation*
  (arXiv 1905.07903) — reference-counting-on-retire SMR with balanced reclamation work.
- Nikolaev & Ravindran, *Crystalline: Fast and Memory-Efficient Wait-Free Reclamation*
  (arXiv 2108.02763) and *WFE: Universal Wait-Free Memory Reclamation* (arXiv 2001.01999) —
  wait-free reclamation; needed only for multi-retirer lock-free structures.

**Read, deferred (queue building blocks for a scheduler's global injector):**
- Nikolaev & Ravindran, *wCQ: A Fast Wait-Free Queue with Bounded Memory Usage*
  (arXiv 2201.02179) — wait-free MPMC FIFO with bounded memory; the strongest candidate if
  the scheduler grows a *shared global* (multi-consumer) injector. Deferred: the per-worker
  Jiffy MPSC inboxes + lifelines suffice for the current pull-based design.
- ✅ Adas & Friedman, *Jiffy: Wait-Free Multi-Producer Single-Consumer Queue* (arXiv 2010.14189)
  — **implemented** as the `jiffy` module (FAA-claimed slots, 3-state slots, eager next-buffer
  linking, single-consumer dequeue) and wired in as the scheduler's lock-free work-pushing
  inbox, replacing the old `Mutex<Vec>`. A TSan-caught buffer use-after-free (freeing a head
  buffer mid-`dequeue` while a slow producer still referenced it) was fixed with retain-until-
  drop reclamation. Many-producer/one-consumer no-loss/no-dup + drop tests, TSan-clean.
- von Geijer & Tsigas, *How to Relax Instantly: Elastic Relaxation of Concurrent Data
  Structures* (arXiv 2403.13644) and Rukundo, Atalar & Tsigas, *Relaxing Concurrent
  Data-structure Semantics … 2D Framework* (arXiv 1906.07105) — tunable semantic relaxation
  (trade strict order for scalability); conceptual cousins of WS-MULT's multiplicity, a path
  to a *relaxed* injector if contention ever dominates.

**Implemented this round (verification, granularity, analysis):**
- ✅ Acar, Charguéraud, Rainey, *Parallel Work Inflation, Memory Effects, and their Empirical
  Analysis* (arXiv 1709.03767) — the **work-inflation decomposition** (`examples/work_inflation.rs`
  + scheduler `Stats`). Attributes the boxed-vs-inline gap to work inflation (3.66×).
- ✅ (granularity) heartbeat scheduling — `spawn_or_call` / `run_with_config`, the Acar/Rainey
  amortized granularity-control idea; bounds scheduling overhead independent of spawn fineness.

**Implemented (final round — CI, race detection, distributed, persistence):**
- ✅ Pearce, Lange, O'Keeffe, *RustMC: Extending GenMC to Rust* (arXiv 2502.06293) — the
  verification *path* is now realized operationally: `.github/workflows/ci.yml` runs the full
  loom suite **and** ThreadSanitizer (`-Zsanitizer=thread -Zbuild-std`) on every push, plus
  fmt/clippy/MSRV. (A literal GenMC run still needs that external tool; CI gives continuous
  loom+TSan coverage of the matrix above.)
- ✅ Westrick, Wang, Acar, *DePa: Order Maintenance for Task Parallelism* (arXiv 2204.14168) —
  the `race` module: SP-order (English/Hebrew rank) determinacy-race detection. Flags parallel
  write-conflicts (shared accumulator) and clears race-free fork-join reductions, *independent
  of schedule* — stronger than TSan's sampling. 8 tests.
- ✅ John, Milthorpe, Strazdins, *Distributed Work Stealing* (arXiv 2211.00838) — the
  `distributed` module: shared-nothing nodes, message-passing steal requests, randomized victim
  selection, half victim policy, distributed termination. Balances an all-on-node-0 seed purely
  by messages. 4 tests, TSan-clean.
- ✅ Fatourou, Giachoudis, Mallis, *Highly-Efficient Persistent FIFO Queues* (arXiv 2402.17674) —
  the `persistent` module: an explicit NVM persistency model (`pwb`/`psync`, simulated since no
  NVRAM hardware) with crash + recovery. Durably-enqueued items survive a crash at *any* point;
  un-`psync`'d ones are correctly lost; no duplicates/resurrection. 5 tests incl. crash-at-every-
  point consistency. **WARNING: durability is simulated in RAM, not real** — no NVRAM hardware
  here, so "NVM" is a `Vec`, "crash" is a method call, and nothing survives a real process exit.
  Models/tests the *algorithm*, not production durability (use a WAL/`fsync`/NVM library).

**Read, deferred (genuinely different scope):**
- Motiwala, *No Cords Attached: Coordination-Free Concurrent Lock-Free Queues*
  (arXiv 2511.09410, 2025) — coordination-free MPMC FIFO; a multi-producer/multi-consumer shape,
  only relevant as an alternate *multi-consumer* injector (our Jiffy inbox is MPSC, which the
  scheduler needs).
- Khatiri, Trystram, Wagner, *Work Stealing Simulator* (arXiv 1910.02803) — a modelling/eval
  tool, not a data structure.
- Suksompong, Leiserson, Schardl, *On the Efficiency of Localized Work Stealing* (arXiv
  1804.04773) — analysis backing the locality bias already implemented in `run_with`.
- A literal **GenMC/RustMC** run and a **real-NVM** persistent-queue benchmark both need external
  tooling/hardware this machine lacks; the CI loom+TSan jobs and the software persistency model
  are the faithful in-reach substitutes.

> Method: arXiv (semantic + keyword), DOI/Unpaywall, and OpenAlex. The fence-free WS-MULT
> result (Castañeda-Piña) was the highest-value implementable find — it changes the
> *asymptotics of synchronization* (removes the mandatory fence/CAS), not just constants.
