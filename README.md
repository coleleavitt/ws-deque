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

See `examples/fib.rs` for a complete multi-threaded work-stealing scheduler:

```sh
cargo run --example fib --release -- 34 8     # parallel fib(34) across 8 workers
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
  `Stealer` (cheaply cloneable, `Send + Sync`) steals from the top via CAS.
- **Monotone `top`.** `top` is only ever advanced by CAS, never decremented, so no ABA tag.
- **Growable cyclic buffer.** Doubles on overflow. Grown-out buffers are retained until the
  deque is dropped (a dependency-free alternative to epoch GC); retired memory is bounded by
  `O(log max_len)` arrays.
- **Correct `Drop`.** Every boxed element is freed exactly once — verified with a
  drop-counting test (no leaks, no double-frees).

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
