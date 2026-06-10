# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — work-stealing data structures
- **Chase-Lev deque** (`Worker` / `Stealer`): exact-once, LIFO or `new_fifo`, wraparound-safe
  indices, a cached-`top` push fast path, growable **and** shrinking buffer with quiescent-state
  mid-life reclamation, and `steal_batch_and_pop`. Race-free via `AtomicPtr` cells (no UB).
- **`inline::InlineWorker<T: Copy>`**: an allocation-free fast path storing small `Copy` values
  directly in `AtomicU64` cells — ~3.6× faster push/pop than the boxed deque.
- **`idempotent`** (WS-MULT family, Castañeda–Piña): `IdempotentWorker`/`IdempotentStealer`
  (fence-free, CAS-free `put`), `WeakStealer` (WS-WMULT, fully fence-free consumer), and
  `bounded()` + `steal_exclusive` (B-WS-MULT, no two thieves take the same task).
- **`linked::LinkedWorker`**: linked-list-backed WS-MULT (constant-time `put`, zero reclamation).
- **`jiffy`**: a wait-free MPSC injector queue (Adas–Friedman), used as the scheduler's inbox.
- **`priority::PriorityWorker<T, K>`**: K-level priority work-stealing (Wimmer).

### Added — schedulers & runtimes
- **`scheduler`**: a lifeline-graph fork-join scheduler (`run` / `run_with` / `run_with_config`)
  with hypercube lifelines, spin-free parking, distributed termination, locality-biased stealing
  (NUMA-WS), lazy work-pushing (Jiffy inbox), and **heartbeat granularity control**
  (`spawn_or_call`). Returns work-inflation `Stats`.
- **`distributed`**: a shared-nothing, message-passing distributed work-stealing runtime
  (John et al.) — randomized victim selection, half victim policy, distributed termination.

### Added — analysis, verification & tooling
- **`race`**: DePa SP-order determinacy-race detection (schedule-independent).
- **`persistent`**: an NVM `pwb`/`psync` persistency model with crash + recovery.
  ⚠️ Durability is **simulated in RAM**, not real — for studying the algorithm, not production use.
- **loom** model-checked tests across every concurrent module; **ThreadSanitizer**-clean test
  suite; **criterion** benchmarks vs `crossbeam-deque` (incl. a realistic boxed-task workload
  where the two converge); a GitHub Actions CI running fmt/clippy/test/loom/TSan/MSRV.

## [0.1.0]

Initial release.
