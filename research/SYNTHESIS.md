# Parallelism Research Synthesis

> **Note on the papers.** The raw PDFs and `pdftotext` extractions are *not* committed to this
> public repo (they are third-party copyrighted academic works). The table below lists every
> source with its arXiv / DOI link — fetch them yourself. This synthesis and the `ws-deque`
> implementation are original.

Each section maps a paper to a concrete, implementable idea. This document was originally
written while prototyping inside the `ForkUnion` fork; references to that project's fork-join
+ NUMA APIs (`for_n`, `fold_with_scratch`, `RoundRobinVec`, `CacheAligned`) are retained as
the motivating context for the deque in this crate (`src/lib.rs`).

## Papers downloaded (research/)

| File | Paper | arXiv / source | Relevance |
| --- | --- | --- | --- |
| `chase_lev_deque.pdf` | Chase & Lev, *Dynamic Circular Work-Stealing Deque* (SPAA'05) | dre.vanderbilt mirror | The deque every runtime (Rayon/Tokio/Go) implements |
| `choi_chaselev_verification.pdf` | Choi, *Formal Verification of Chase-Lev Deque in Concurrent Separation Logic* (2023) | arXiv 2309.03642 | Correct C11 atomic orderings (relaxed/acquire/release/seq_cst) |
| `suksompong_localized_workstealing.pdf` | Suksompong, Leiserson, Schardl, *On the Efficiency of Localized Work Stealing* (2018) | arXiv 1804.04773 | Locality-biased victim selection — NUMA argument |
| `deters_numa_workfirst.pdf` | Deters, Wu, Xu, Lee, *A NUMA-Aware Provably-Efficient Task-Parallel Platform Based on the Work-First Principle* (2018) | arXiv 1806.11128 | NUMA-WS: mitigating work inflation, socket-local deques |
| `gu_workstealing_cache.pdf` | Gu, Napier, Sun, *Analysis of Work-Stealing and Parallel Cache Complexity* (2021) | arXiv 2111.04994 | Modern cache-miss bounds for randomized work stealing |
| `cole_cache_miss.pdf` | Cole & Ramachandran, *Bounding Cache Miss Costs of Multithreaded Computations Under General Schedulers* (2017) | arXiv 1705.08350 | False-sharing / cache-line cost analysis |
| `wimmer_mixedmode.pdf` | Wimmer & Träff, *Work-stealing for mixed-mode parallelism by deterministic team-building* (2010) | arXiv 1012.5030 | Tasks needing r≥1 threads → Fork Union `for_slices` |

> Note: the original Blumofe–Leiserson JACM'99 paper and Neill–Wierman (CMU'09) are
> closed-access (verified via Unpaywall + multiple mirror attempts). The Choi 2023 paper
> reproduces and formally verifies the Lê et al. C11 implementation, which supersedes the
> 1999 pseudocode for *implementation* purposes.

---

## 1. Chase-Lev deque → `src/deque.rs` (IMPLEMENTED)

**Paper idea (chase_lev_deque.pdf §2):** A single-owner / multi-stealer deque on a cyclic
array with two monotone indices `top` and `bottom`.
- Owner: `push_bottom` (write slot, `bottom += 1`); `pop_bottom` (`bottom -= 1`, read).
- Thieves: `steal` reads `top` then `bottom`, reads slot, then `CAS(top, t→t+1)`.
- `top` is **never decremented** ⇒ no ABA tag field needed.
- Last-element race: an emptying `pop_bottom` must also `CAS` `top` so a concurrent
  `steal` cannot also take the final element.

**Correct memory orderings (choi/Lê, choi_chaselev_verification.txt §"Relaxed Memory"):**
- `push_bottom`: load `bottom` Relaxed, load `top` Acquire, store slot Relaxed, store
  `bottom` **Release**.
- `steal`: load `top` Acquire, **SeqCst fence**, load `bottom` Acquire, load slot Relaxed,
  `CAS(top)` SeqCst (success) / Relaxed (failure).
- `pop_bottom`: store `bottom` Relaxed, **SeqCst fence**, load `top` Relaxed, CAS SeqCst.

This is exactly what `crossbeam-deque` does; we implement it standalone with no deps so it
fits Fork Union's "no external crates" philosophy.

**The slot data race — and the truly-race-free fix (validated with ThreadSanitizer).**
A first cut stored elements inline and accessed slots with `ptr::read`/`ptr::write`.
ThreadSanitizer (`-Zsanitizer=thread -Zbuild-std`) immediately flagged a *real* data race in
`core::ptr::write::<usize>`: a thief speculatively reads slot `t` **before** its CAS, which
races the owner overwriting that physical slot via a later `push`. `crossbeam-deque` knows
about this and papers over it with `read_volatile`/`write_volatile`, documenting it as
"technically speaking a data race and therefore UB" (crossbeam-deque-0.8.6/src/deque.rs:70).
Per **Lê et al.**, the genuinely race-free fix is to make slot accesses *atomic*: each cell
is an `AtomicPtr<T>` holding a heap-boxed element, loaded/stored `Relaxed`. After this change
ThreadSanitizer reports **zero** races across the whole deque suite (single-thread + the
4-thief / 200k-element concurrent stress test), and 15 release stress runs pass. This is how
production job queues actually behave — they enqueue pointers, not inline values.

## 2. Localized / NUMA-biased stealing → victim selection policy

**Paper idea (suksompong_localized_workstealing.pdf):** Pure random victim choice ignores
locality. The *localized* variant biases a thief toward victims whose work it "owns"
(was assigned to it), bounding the extra steals. **Deters NUMA-WS (deters_numa_workfirst.pdf
§III):** keep a deque per socket; steal socket-locally first, cross-socket only when the
local socket is drained — this mitigates *work inflation* (the slowdown from touching
remote DRAM). Fork Union already has the colocation/NUMA primitives (`count_colocations`,
`RoundRobinVec`), so the implementable policy is: **two-level steal — first try victims in
the same colocation index, fall back to global random.** Documented as the design rationale
for `RoundRobinVec`'s per-colocation layout.

## 3. Work-first principle → static scheduling default (already in Fork Union)

**Paper idea (deters §II "work-first"):** Pay scheduling cost on the *steal* (rare) path,
never on the *work* (common) path. Fork Union's `for_n` / static scheduler embodies this:
no per-task heap allocation, no CAS on the common path; the dynamic stealing counter is only
touched under `for_n_dynamic`. This validates the README's static-default choice and the
N-body numbers (static beats dynamic on uniform workloads).

## 4. Cache complexity & false sharing → `CacheAligned` (already in Fork Union)

**Paper idea (cole_cache_miss.pdf, gu_workstealing_cache.pdf):** Total cache-miss cost of a
work-stealing schedule is bounded by sequential misses plus O(steals × cache footprint of a
steal). Minimizing steals (static) and avoiding false sharing on the per-thread counters is
what keeps the constant small. Fork Union's `CacheAligned<T>` (128-byte align) and
`fold_with_scratch` (per-thread accumulator) are the direct application — one scratch slot
per thread, no shared atomic in the reduction hot path.

## 5. Mixed-mode parallelism → `for_slices` (already in Fork Union)

**Paper idea (wimmer_mixedmode.pdf):** Some tasks need a *team* of r≥1 threads, not a single
worker. Deterministic team-building assigns contiguous worker teams to such tasks. Fork
Union's `for_slices` (nested `schedule(static)`) is the lightweight analog: a slice is a
contiguous team-sized chunk handled cooperatively. No code change needed; this is the
theoretical justification for the third scheduling primitive.

---

## What this session implements in the repo

1. `src/deque.rs` — a standalone, `no_std`-friendly Chase-Lev work-stealing deque with the
   verified C11 orderings, `push`/`pop`/`steal`, growable cyclic buffer, and concurrent tests.
   (An additive, self-contained Rust module implementing the algorithm from the papers,
   feature-gated on `std` so it stays independent of the precompiled-C core.)
2. `scripts/reduce_bench.rs` — a runnable parallel-reduction (sum) benchmark replicating the
   blog methodology: Fork Union `fold_with_scratch` vs Rayon `broadcast` vs Tokio, printing
   throughput so users measure *their* hardware instead of the README's numbers.
