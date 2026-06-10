# ws-deque

A lock-free, **dependency-free**, **ThreadSanitizer-clean** Chase-Lev work-stealing deque in
safe-surface Rust — the scheduling primitive that underlies Rayon, Tokio, and Go, implemented
from the original papers for study and standalone use.

```rust
use ws_deque::{Worker, Steal};

let worker = Worker::new();          // single owner: push/pop at the bottom
let stealer = worker.stealer();      // any number of thieves: steal from the top

worker.push(1);
worker.push(2);
assert_eq!(worker.pop(), Some(2));   // owner pops LIFO
assert!(matches!(stealer.steal(), Steal::Success(1))); // thieves take FIFO
```

Or use the built-in **lifeline-graph scheduler** for irregular fork-join workloads — idle
workers park spin-free and are woken via a hypercube lifeline graph, with clean distributed
termination:

```rust
use ws_deque::scheduler::run;
use std::sync::atomic::{AtomicUsize, Ordering};

let count = AtomicUsize::new(0);
run(8, [20u32], |depth, spawner| {        // 8 workers, one seed task
    count.fetch_add(1, Ordering::Relaxed);
    if depth > 0 {
        spawner.spawn(depth - 1);          // each task may spawn more
        spawner.spawn(depth - 1);
    }
});
// runs the whole dynamically-unfolding tree to completion, then returns
```

Need **priority** (expand promising work first)? `priority::PriorityWorker<T, K>` layers `K`
priority levels over the deque — `pop`/`steal` always take the highest-priority task. On A\* vs
Dijkstra it explores ~5× fewer nodes for the same optimum (the Wimmer "priority reduces total
work" result):

```rust
use ws_deque::priority::PriorityWorker;

let w = PriorityWorker::<&str, 3>::new();  // 3 levels, 0 = highest
w.push("background", 2);
w.push("urgent", 0);
assert_eq!(w.pop(), Some("urgent"));        // highest priority first
```

Examples:

```sh
cargo run --example lifeline --release          # Unbalanced Tree Search, lifeline scheduler
cargo run --example branch_and_bound --release  # A* vs Dijkstra, priority work-stealing
cargo run --example work_inflation --release    # Acar work-inflation decomposition of cost
cargo run --example fib --release -- 34 8       # raw-deque busy-wait scheduler (lower level)
```

## Why another deque?

This implementation exists to get the *memory model* right, in the open.

A naive Chase-Lev deque reads and writes its array slots with plain `ptr::read` / `ptr::write`.
That is a **genuine data race** under the C11 model: a thief speculatively reads a slot
*before* its `compare_exchange` on `top`, which can race the owner overwriting that physical
slot via a later `push`. The widely-used `crossbeam-deque` papers over this with
`read_volatile` / `write_volatile` and openly documents it as *"technically speaking a data
race and therefore UB."* ThreadSanitizer flags it.

`ws-deque` follows **Lê et al., *Correct and Efficient Work-Stealing for Weak Memory Models***
(PPoPP 2013): each array cell is an `AtomicPtr<T>` holding a heap-boxed element, accessed with
`Relaxed` loads/stores while the indices carry the happens-before via `Acquire`/`Release` plus
a `SeqCst` fence. The result is **race-free by construction** — `cargo +nightly test
-Zsanitizer=thread -Zbuild-std` reports zero races across the full suite, including a
4-thief / 200k-element concurrent stress test.

## Design

- **Single owner, many thieves.** `Worker` pushes/pops the bottom (no CAS on the common path);
  `Stealer` (cheaply cloneable, `Send + Sync`) steals from the top via CAS. `Worker::new_fifo()`
  switches the owner to oldest-first (FIFO) processing, like crossbeam/Tokio `new_fifo`.
- **Monotone `top`.** Only ever advanced by CAS, never decremented, so no ABA tag. All index
  arithmetic is **wraparound-safe** (`wrapping_*`), correct even past `isize::MAX`.
- **Cached `top` on push.** The owner keeps a private lower bound of the contended `top` and only
  reads the shared atomic when the buffer might be full (Chase-Lev §2.3).
- **Growable *and shrinking* cyclic buffer with live reclamation.** Doubles on overflow
  (Chase-Lev §2), halves below `cap/3` (Chase-Lev §3). Retired buffers are freed *mid-life* by a
  **quiescent-state** scheme (an in-flight steal counter + `SeqCst` fences) — loom-verified, so
  the memory backlog stays bounded under grow/shrink cycling, with no epoch-GC dependency.
- **Batch stealing.** `Stealer::steal_batch_and_pop` moves ~half the victim's work into the
  thief's own deque — the amortization trick Tokio, Rayon, and crossbeam use.
- **Correct `Drop`.** Every boxed element is freed exactly once — verified with a
  drop-counting test (no leaks, no double-frees).

## Algorithms in this crate

A family of work-stealing structures, plus a scheduler that ties them together:

| Module | Type | Contract / niche |
| --- | --- | --- |
| (root) | `Worker` / `Stealer` (Chase-Lev) | **exact-once**; LIFO or `new_fifo`; wraparound-safe; live buffer reclamation; cache-padded |
| `bwos` | `BwosWorker` / `BwosStealer` | block-based work stealing (OSDI'23), **bounded** — **~5.8× faster than crossbeam** on push/pop |
| `bwos::unbounded` | `UnboundedBwosWorker` / `…Stealer` | linked-block variant — unbounded capacity, closer-to-crossbeam throughput |
| `inline` | `InlineWorker<T: Copy>` | allocation-free fast path — fence-light steal, no per-element box |
| `idempotent` | `IdempotentWorker` (WS-MULT) | **≥1×** multiplicity; `put` is a plain store (no CAS/fence) |
| `idempotent` | `WeakStealer` (WS-WMULT) | weak multiplicity; consumer path **fully fence-free, no RMW** |
| `idempotent` | `bounded()` + `steal_exclusive` (B-WS-MULT) | bounded multiplicity — **no two thieves take the same task** |
| `linked` | `LinkedWorker` (approach 2) | linked-node store: **constant-time `put`, zero reclamation** |
| `jiffy` | `channel()` → `Producer`/`Consumer` | wait-free **MPSC** injector (Jiffy) — the scheduler's lock-free inbox |
| `priority` | `PriorityWorker<T, K>` | K priority levels — expand promising work first |
| `scheduler` | `run` / `run_with` / `run_with_config` | lifeline fork-join driver; locality bias, lazy work-pushing, **heartbeat granularity control** |
| `distributed` | `run` (shared-nothing nodes) | message-passing distributed work-stealing — steal requests, half victim policy |
| `race` | `Dag` / `detect_races` | DePa **determinacy-race detection** (SP-order) — schedule-independent |
| `persistent` | `PersistentQueue` | NVM FIFO persistency model — `pwb`/`psync` + recovery. **⚠️ simulated, not real durability** |

**Verification:** every concurrent module is checked by both **loom** (exhaustive interleaving
model-checking of bounded scenarios) and **ThreadSanitizer** (full test suite), run on every push
by [`.github/workflows/ci.yml`](.github/workflows/ci.yml). See the coverage matrix and the
RustMC/GenMC path in [`research/GAPS.md`](research/GAPS.md#verification-coverage-matrix).

The WS-MULT family is a Rust implementation of Castañeda & Piña's *Fully Read/Write Fence-Free
Work-Stealing with Multiplicity* (arXiv:2008.04424), which sidesteps the Attiya et al.
impossibility result (exact-once work-stealing *must* use a fence or CAS) by relaxing to
multiplicity. The core `idempotent` queue is measured ~1.4× faster than the Chase-Lev `pop`/
`push` path because it carries neither the fence nor the CAS. See
[`research/GAPS.md`](research/GAPS.md#-breakthrough-ws-mult--fence-free-cas-free-work-stealing-srcidempotentrs).

| | `Worker` (Chase-Lev) | `IdempotentWorker` (WS-MULT) |
| --- | --- | --- |
| Contract | every task runs **exactly once** | every task runs **≥1 times** (multiplicity ≤ #threads) |
| `push`/`put` | store + `Release` | **plain store — no CAS, no fence** |
| `pop`/`take` | `SeqCst` fence + maybe CAS | monotone `fetch_max`, **no retry** |
| `steal` | CAS-abort loop | `fetch_max`, **no retry** |
| Use for | side effects, accounting, exactly-once | idempotent work: SAT, graph search, fixpoint |

The second is a Rust implementation of Castañeda & Piña's *Fully Read/Write Fence-Free
Work-Stealing with Multiplicity* (arXiv:2008.04424). It sidesteps the Attiya et al.
impossibility result (exact-once work-stealing *must* use a fence or CAS) by relaxing to
multiplicity — and is measured ~1.4× faster than the Chase-Lev `pop`/`push` path on this
machine because it carries neither the fence nor the CAS. See
[`research/GAPS.md`](research/GAPS.md#-breakthrough-ws-mult--fence-free-cas-free-work-stealing-srcidempotentrs).

```rust
use ws_deque::idempotent::{IdempotentWorker, Take};

let mut w = IdempotentWorker::new();   // T: Clone (a task may be delivered more than once)
let s = w.stealer();
w.put(1);
w.put(2);
assert_eq!(w.take(), Take::Got(1));     // FIFO, fence-free, no CAS
assert!(matches!(s.steal(), Take::Got(2)));
```

## Correctness & performance

- **`loom`** exhaustively model-checks the push/pop/steal interleavings:
  `RUSTFLAGS="--cfg loom" cargo test --release loom_`.
- **ThreadSanitizer** runs clean across every concurrent test and the `fib` example:
  `RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test --lib -Zbuild-std --target <triple>`.
- **Miri** checks for undefined behaviour TSan can't see — invalid pointer provenance, misaligned
  access, uninitialized reads (under Stacked Borrows): `cargo +nightly miri test --lib`. Tests
  scale themselves down under `cfg!(miri)`. Miri **found a real UB bug** during hardening — a
  `*const → *mut` cast in `bwos`'s slot writes violated Stacked Borrows; fixed by storing slots as
  `AtomicU64` (atomic access through `&` is sound). This is exactly the class of bug TSan misses.
- **`crossbeam-deque`'s own test suite, ported** (`tests/crossbeam_{lifo,fifo,steal}.rs`): the
  production crate's battle-tested stress tests — `smoke`, `spsc`, `stampede`, `stress`,
  `no_starvation`, and the exact-once `destructors` test — run against this crate's deque (8
  threads × 50k items), both natively and under ThreadSanitizer. This is the "is it really
  production-hardened?" check, borrowed from the crate everyone trusts.
- **`cargo bench`** pits this crate against `crossbeam-deque`. The result depends entirely on
  the payload:

  | Workload (`cargo bench`) | ws-deque | crossbeam | Verdict |
  | --- | ---: | ---: | --- |
  | `push_pop` of `u64` (boxed cells) | ~154 µs | ~32 µs | crossbeam ~5× faster |
  | `push_pop` of `u64` (**`inline` fast path**) | ~44 µs | ~32 µs | within ~1.4× |
  | **`push_pop_bwos`** — block-based (`bwos`) | **~6.2 µs** | ~36 µs | **BWoS ~5.8× *faster*** |
  | **`task_queue_boxed`** — `Box<dyn FnOnce>` jobs | **~77 µs** | ~83 µs | **converged (≈parity)** |

  Two takeaways. **(1) The `bwos` block-based queue beats crossbeam outright** (~5.8×): splitting
  the queue into blocks moves the owner's `push`/`pop` off the contended shared indices, so the
  in-block fast path is a plain write + `Release` bump — no `SeqCst` fence per op, which is what
  Chase-Lev (ours *and* crossbeam's) pays. This is the OSDI'23 BWoS result, reproduced and
  loom/TSan-verified. **(2) For the classic Chase-Lev deque, payload decides the story:** the
  `u64` microbench *maximizes* the relative cost of ws-deque's per-element `AtomicPtr` box (an
  allocation crossbeam avoids with inline-but-technically-UB `volatile`), but for any real
  executor the payload is already a heap allocation (`Box<dyn FnOnce>`/`Arc<Task>` — what
  Rayon/Tokio enqueue), so both converge to **parity** (`task_queue_boxed`). Per-field
  `CachePadded` (dependency-free) removes owner↔thief false sharing, cutting the contended deque
  ~20%. Pick: **`bwos`** for raw throughput, **`inline`** for small `Copy` payloads, default
  **`Worker`** for task queues — all genuinely race-free (no UB, TSan- and loom-clean). See
  [`research/GAPS.md`](research/GAPS.md) for the full analysis.

## References

The papers and a design synthesis live in [`research/`](research/):

- D. Chase & Y. Lev, *Dynamic Circular Work-Stealing Deque*, SPAA 2005.
- N. M. Lê, A. Pop, A. Cohen, F. Zappa Nardelli, *Correct and Efficient Work-Stealing for Weak
  Memory Models*, PPoPP 2013.
- J. Choi, *Formal Verification of Chase-Lev Deque in Concurrent Separation Logic*, 2023
  (arXiv:2309.03642).
- Plus NUMA / cache-complexity / mixed-mode work-stealing papers — see
  [`research/SYNTHESIS.md`](research/SYNTHESIS.md).

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

The algorithm derives from published academic work; this is an independent clean-room
implementation from those papers.
