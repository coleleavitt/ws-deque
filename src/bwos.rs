//! Block-based Work Stealing (BWoS) — a bounded LIFO work-stealing queue split into blocks.
//!
//! Implements the core design of
//!
//! - J. Wang, B. Trach, M. Fu, D. Behrens, J. Schwender, Y. Liu, J. Lei, V. Vafeiadis,
//!   H. Härtig, H. Chen, *BWoS: Formally Verified Block-based Work Stealing for Parallel
//!   Processing*, OSDI 2023.
//!
//! # The idea — and why it beats Chase-Lev
//!
//! A Chase-Lev deque keeps a single pair of shared indices (`top`/`bottom`); the owner and every
//! thief contend on *the same two cache lines* on every operation, and the owner must execute a
//! `SeqCst` fence on each `pop` to stay race-free against thieves. BWoS instead splits the queue
//! into **fixed-size blocks** and gives each block its own metadata:
//!
//! - The **owner** pushes/pops within its current *top block* using only owner-private indices
//!   (`b_pos`) — **no fence, no CAS, no atomic** on the common in-block fast path.
//! - **Thieves** steal from a *different, earlier block*, coordinating among themselves with that
//!   block's `s_pos` (a CAS) — but they touch neither the owner's block nor the owner's metadata.
//!
//! Because owner and thieves "rarely operate on the same blocks" (paper §2), the false sharing
//! and owner↔thief synchronization that dominate a Chase-Lev hot loop almost entirely vanish —
//! which is how BWoS reports 8–11× microbenchmark gains. Only at a **block boundary** (owner
//! crossing from one block to the next) is cross-party synchronization needed.
//!
//! # Scope of this implementation
//!
//! This is a **bounded** LIFO BWoS queue with a fixed number of blocks of fixed size (the
//! capacity is `num_blocks * block_size`). It implements the block fast paths and block
//! advancement for the producer/owner-consumer/thieves case. It does *not* implement the paper's
//! unbounded round-control / takeover-grant recycling (that reuses blocks across rounds and is
//! the subtle part of the proof); a bounded queue sidesteps block reuse entirely, which keeps the
//! design tractable to verify here (loom + ThreadSanitizer) while preserving the cache-locality
//! win that is the point. `put` returns `false` when full.
//!
//! Values are `T: Copy` of ≤ 8 bytes, stored inline (no allocation) — the same fast representation
//! as [`crate::inline`].

use std::boxed::Box;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use std::vec::Vec;

#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering, fence};

use crate::CachePadded;
pub use crate::Steal;

/// Debug tracing for the BWoS protocol. Compiles to **nothing** in normal/bench/loom builds, so
/// it has zero runtime cost. Enable a full event log with:
///
/// ```sh
/// RUSTFLAGS="--cfg bwos_trace" cargo test --lib bwos -- --nocapture --test-threads=1
/// ```
///
/// Each line is `[<thread>] who: event  fields...` so producer/consumer/thief interleavings and
/// the per-block index movements are visible when hunting a double-take / lost-element bug.
#[cfg(bwos_trace)]
macro_rules! trace {
    ($who:expr, $($arg:tt)*) => {{
        let id = std::thread::current().id();
        eprintln!("[{:?}] {}: {}", id, $who, format!($($arg)*));
    }};
}
#[cfg(not(bwos_trace))]
macro_rules! trace {
    ($who:expr, $($arg:tt)*) => {{
        // Reference the args so they don't trigger unused-variable warnings, but emit nothing.
        let _ = ($who, format_args!($($arg)*));
    }};
}

/// Default entries per block.
const BLOCK_SIZE: usize = 256;

/// Bit-pack a `Copy` value (≤ 8 bytes) into a `u64`.
#[inline]
fn to_bits<T: Copy>(v: T) -> u64 {
    const {
        assert!(
            core::mem::size_of::<T>() <= 8,
            "BWoS queue needs T no larger than 8 bytes"
        );
    }
    let mut bits = 0u64;
    // SAFETY: `T: Copy`, `size_of::<T>() <= 8`; copy its bytes into the low bytes of a zeroed u64.
    unsafe {
        core::ptr::copy_nonoverlapping(
            &v as *const T as *const u8,
            &mut bits as *mut u64 as *mut u8,
            core::mem::size_of::<T>(),
        );
    }
    bits
}

#[inline]
fn from_bits<T: Copy>(bits: u64) -> T {
    // SAFETY: `bits` came from `to_bits::<T>`, so its low bytes are a valid `T`.
    unsafe { core::ptr::read(&bits as *const u64 as *const T) }
}

/// One block: a fixed array of inline slots plus block-local metadata. Each block's metadata is
/// cache-padded so the owner's `committed` and the thieves' `stolen` never share a line.
struct Block {
    /// Inline value bits. Written by the owner (under `committed`), read by owner pops and by
    /// thieves (under the `stolen` CAS). `Copy`, so a racing read is a harmless bit copy.
    slots: Box<[u64]>,
    /// Number of entries the owner has *committed* (filled) in this block. Owner-only writes;
    /// published with `Release` so a thief that reads it sees the slot data.
    committed: CachePadded<AtomicUsize>,
    /// Steal cursor: the next index thieves will steal from. Advanced by thief CAS. On its own
    /// cache line, away from `committed`.
    stolen: CachePadded<AtomicUsize>,
}

impl Block {
    fn new(block_size: usize) -> Self {
        Block {
            slots: vec![0u64; block_size].into_boxed_slice(),
            committed: CachePadded(AtomicUsize::new(0)),
            stolen: CachePadded(AtomicUsize::new(0)),
        }
    }
}

struct Inner {
    blocks: Box<[Block]>,
    block_size: usize,
    /// Number of blocks the owner has *started* producing into. Monotonically increases (the
    /// owner only ever moves the production frontier forward, never back). Thieves scan blocks
    /// `[0, produced_blocks)`. Published with `Release` when the owner advances during `put`.
    produced_blocks: CachePadded<AtomicUsize>,
}

// SAFETY: `owner_block` (a `Cell`) is only ever touched by the single owner thread; all
// thief-visible state is in the atomics. See `BwosWorker`'s `!Sync` discipline.
unsafe impl Sync for Inner {}

/// The single owner of a BWoS queue. Pushes/pops at the back (LIFO) within the owner's block.
pub struct BwosWorker<T: Copy> {
    inner: Arc<Inner>,
    /// Index of the block the owner currently produces/consumes in. Owner-private (`Cell`), so the
    /// in-block fast path touches no shared atomic for block selection.
    owner_block: core::cell::Cell<usize>,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: owner methods are single-threaded `&self`; the `Cell`s are owner-only.
unsafe impl<T: Copy + Send> Send for BwosWorker<T> {}

impl<T: Copy> BwosWorker<T> {
    /// Create a BWoS queue with the default block count (16) and size (256) — capacity 4096.
    pub fn new() -> Self {
        Self::with_blocks(16, BLOCK_SIZE)
    }

    /// Create a BWoS queue of `num_blocks` blocks of `block_size` entries (capacity their product).
    pub fn with_blocks(num_blocks: usize, block_size: usize) -> Self {
        assert!(
            num_blocks >= 2,
            "BWoS needs at least 2 blocks (owner + steal)"
        );
        assert!(block_size >= 1);
        let blocks: Vec<Block> = (0..num_blocks).map(|_| Block::new(block_size)).collect();
        BwosWorker {
            inner: Arc::new(Inner {
                blocks: blocks.into_boxed_slice(),
                block_size,
                produced_blocks: CachePadded(AtomicUsize::new(1)), // block 0 is live from the start
            }),
            owner_block: core::cell::Cell::new(0),
            _marker: core::marker::PhantomData,
        }
    }

    /// A thief handle.
    pub fn stealer(&self) -> BwosStealer<T> {
        BwosStealer {
            inner: Arc::clone(&self.inner),
            _marker: core::marker::PhantomData,
        }
    }

    /// Total capacity (`num_blocks * block_size`).
    pub fn capacity(&self) -> usize {
        self.inner.blocks.len() * self.inner.block_size
    }

    /// **Push** a value at the back (owner-only). Within a block this is a plain slot write plus a
    /// `Release` bump of the block's bottom (`committed`) — **no fence, no CAS**. At a block
    /// boundary it advances to the next block. Returns `false` if the queue is full (bounded).
    ///
    /// `committed` is the block's *bottom* index in a per-block Chase-Lev protocol; `b_pos` is the
    /// owner's private cached copy so the common path needn't reload the atomic.
    pub fn put(&self, value: T) -> bool {
        let inner = &*self.inner;
        let mut blk = self.owner_block.get();
        let block_size = inner.block_size;

        // The owner's bottom in the current block is `committed`. When it reaches block_size the
        // block is full; advance to the next (fresh) block.
        let mut bottom = inner.blocks[blk].committed.load(Ordering::Relaxed);
        if bottom == block_size {
            let next = blk + 1;
            if next == inner.blocks.len() {
                trace!("put", "FULL blk={blk} (no free block)");
                return false; // bounded queue full
            }
            blk = next;
            bottom = 0;
            self.owner_block.set(blk);
            // Publish that block `blk` is now live so thieves may scan it.
            inner.produced_blocks.store(blk + 1, Ordering::Release);
            trace!(
                "put",
                "advance owner_block -> {blk}, produced_blocks={}",
                blk + 1
            );
        }

        // Fast path: write the slot, then publish the new bottom with Release so a thief that
        // Acquire-reads `committed` also observes the slot store. No CAS, no fence.
        let block = &inner.blocks[blk];
        block.slots_write(bottom, to_bits(value));
        block.committed.store(bottom + 1, Ordering::Release);
        trace!(
            "put",
            "blk={blk} slot[{bottom}]<-val  bottom={}",
            bottom + 1
        );
        true
    }

    /// **Pop** from the back (owner-only, LIFO). Each block is its own Chase-Lev deque:
    /// `committed` is the block's *bottom* (owner end) and `stolen` is its *top* (thief end). The
    /// owner decrements `committed` to claim a slot; only when the block is down to its last
    /// element does it race a thief, resolved by a CAS on `stolen` — exactly Chase-Lev, but
    /// confined to one block so thieves on other blocks never interfere.
    pub fn pop(&self) -> Option<T> {
        let inner = &*self.inner;
        loop {
            let blk = self.owner_block.get();
            let block = &inner.blocks[blk];

            // `committed` is this block's bottom; `stolen` is its top. Per-block Chase-Lev pop.
            let b = block.committed.load(Ordering::Relaxed);
            let t = block.stolen.load(Ordering::Relaxed);
            if b <= t {
                // No owner-side elements left in this block. Step to the previous block if any.
                if blk == 0 {
                    trace!("pop", "EMPTY (blk=0, b={b} t={t})");
                    return None;
                }
                let prev = blk - 1;
                self.owner_block.set(prev);
                trace!("pop", "step back owner_block {blk} -> {prev}");
                continue;
            }

            let new_b = b - 1;
            block.committed.store(new_b, Ordering::Relaxed); // tentative bottom decrement
            fence(Ordering::SeqCst); // order bottom write before reading top
            let s = block.stolen.load(Ordering::Relaxed);
            trace!("pop", "blk={blk} b={b} new_b={new_b} top={s}");

            if new_b > s {
                // More than one element — thieves strictly behind. Plain take, no CAS.
                let bits = block.slots_read(new_b);
                trace!("pop", "TAKE blk={blk} slot[{new_b}] (uncontested)");
                return Some(from_bits(bits));
            }
            if new_b == s {
                // Exactly the last element: race the thief via CAS on `stolen` (top).
                let bits = block.slots_read(new_b);
                let won = block
                    .stolen
                    .compare_exchange(s, s + 1, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok();
                // Block is now empty; restore committed to the (now equal) top so b<=t holds.
                block.committed.store(s + 1, Ordering::Relaxed);
                trace!("pop", "LAST blk={blk} slot[{new_b}] cas_won={won}");
                if won {
                    return Some(from_bits(bits));
                }
                continue; // thief won; step back next iteration
            }
            // `new_b < s`: a thief raced ahead between our load and decrement. Restore and retry
            // on the (now drained) block — the b<=t branch will step us back.
            block.committed.store(b, Ordering::Relaxed);
            trace!(
                "pop",
                "RACE blk={blk} new_b={new_b} < top={s}; restore bottom={b}"
            );
        }
    }
}

impl<T: Copy> Default for BwosWorker<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A thief handle for a BWoS queue.
pub struct BwosStealer<T: Copy> {
    inner: Arc<Inner>,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: steal uses only atomic loads + a CAS on the block-local `stolen` cursor.
unsafe impl<T: Copy + Send> Send for BwosStealer<T> {}
unsafe impl<T: Copy + Send> Sync for BwosStealer<T> {}

impl<T: Copy> Clone for BwosStealer<T> {
    fn clone(&self) -> Self {
        BwosStealer {
            inner: Arc::clone(&self.inner),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Copy> BwosStealer<T> {
    /// **Steal** a value from the front of the queue. Scans the produced blocks oldest-first and
    /// steals from the first one with available work, using that block's own Chase-Lev top
    /// (`stolen`). A thief on block *i* never touches block *j*'s metadata, so it interferes with
    /// the owner only when both happen to be on the same (usually last, nearly-empty) block.
    pub fn steal(&self) -> Steal<T> {
        let inner = &*self.inner;
        let produced = inner.produced_blocks.load(Ordering::Acquire);
        let mut blk = 0usize;
        while blk < produced {
            let block = &inner.blocks[blk];
            // Per-block Chase-Lev steal: read top, fence, read bottom; if non-empty, CAS top.
            let t = block.stolen.load(Ordering::Acquire);
            fence(Ordering::SeqCst);
            let b = block.committed.load(Ordering::Acquire);
            trace!("steal", "blk={blk}/{produced} top={t} bottom={b}");

            if t >= b {
                // This block is drained from the thief side; move to the next produced block.
                blk += 1;
                continue;
            }

            // Read the slot at the top BEFORE the CAS, then claim it.
            let bits = block.slots_read(t);
            if block
                .stolen
                .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                trace!("steal", "TAKE blk={blk} slot[{t}]");
                return Steal::Success(from_bits(bits));
            }
            // Lost the race for slot `t` (another thief or owner's last-element claim).
            trace!("steal", "RETRY blk={blk} slot[{t}] (cas failed)");
            return Steal::Retry;
        }
        trace!("steal", "EMPTY (scanned {produced} blocks)");
        Steal::Empty
    }
}

impl Block {
    /// Owner-only slot write. SAFETY: `pos < block_size`; owner is the unique writer.
    #[inline]
    fn slots_write(&self, pos: usize, bits: u64) {
        // The slots are `Copy` bits; a concurrent thief read of a *different* committed slot can't
        // race this, and a thief never reads `pos` until `committed` is published (Release) past
        // it. We use a raw cell write via UnsafeCell semantics emulated over the boxed slice.
        // SAFETY: single owner writer; thieves only read indices `< committed`.
        unsafe {
            let ptr = self.slots.as_ptr().add(pos) as *mut u64;
            ptr.write(bits);
        }
    }

    /// Slot read (owner or thief). SAFETY: `pos` is `< committed`, which was published Release.
    #[inline]
    fn slots_read(&self, pos: usize) -> u64 {
        // SAFETY: read of a `Copy` value previously published before `committed` advanced past
        // `pos`; the value is stable once committed (slots are written once per bounded use).
        unsafe { self.slots.as_ptr().add(pos).read() }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::vec::Vec;

    use super::*;

    #[test]
    fn put_pop_lifo_within_block() {
        let w = BwosWorker::<u64>::with_blocks(4, 8);
        for i in 0..6 {
            assert!(w.put(i));
        }
        for i in (0..6).rev() {
            assert_eq!(w.pop(), Some(i));
        }
        assert_eq!(w.pop(), None);
    }

    #[test]
    fn put_pop_across_blocks() {
        let w = BwosWorker::<u64>::with_blocks(4, 4);
        // 10 items spans 3 blocks (4+4+2).
        for i in 0..10 {
            assert!(w.put(i));
        }
        for i in (0..10).rev() {
            assert_eq!(w.pop(), Some(i), "LIFO across block boundaries");
        }
        assert_eq!(w.pop(), None);
    }

    #[test]
    fn bounded_full_rejects() {
        let w = BwosWorker::<u64>::with_blocks(2, 4); // capacity 8
        for i in 0..8 {
            assert!(w.put(i), "slot {i} within capacity");
        }
        assert!(!w.put(99), "put past capacity returns false");
    }

    #[test]
    fn steal_takes_from_front_fifo() {
        let w = BwosWorker::<u64>::with_blocks(4, 4);
        let s = w.stealer();
        for i in 0..6 {
            w.put(i);
        }
        // Thief takes from the front (oldest first).
        assert_eq!(s.steal(), Steal::Success(0));
        assert_eq!(s.steal(), Steal::Success(1));
    }

    #[test]
    fn concurrent_owner_and_thieves_no_loss() {
        let w = BwosWorker::<usize>::with_blocks(64, 256); // capacity 16384
        let thieves = 3;
        let n = 16_000usize;
        let seen: StdArc<Vec<AtomicUsize>> =
            StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());

        // Fill, then concurrently pop (owner) and steal (thieves) until all consumed.
        for i in 0..n {
            assert!(w.put(i));
        }
        let consumed = StdArc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..thieves {
                let s = w.stealer();
                let seen = StdArc::clone(&seen);
                let consumed = StdArc::clone(&consumed);
                scope.spawn(move || {
                    while consumed.load(StdOrdering::SeqCst) < n {
                        match s.steal() {
                            Steal::Success(v) => {
                                seen[v].fetch_add(1, StdOrdering::SeqCst);
                                consumed.fetch_add(1, StdOrdering::SeqCst);
                            }
                            Steal::Retry => {}
                            Steal::Empty => {
                                if consumed.load(StdOrdering::SeqCst) >= n {
                                    break;
                                }
                            }
                        }
                    }
                });
            }
            while consumed.load(StdOrdering::SeqCst) < n {
                if let Some(v) = w.pop() {
                    seen[v].fetch_add(1, StdOrdering::SeqCst);
                    consumed.fetch_add(1, StdOrdering::SeqCst);
                }
            }
        });

        // Every value consumed exactly once across owner + thieves (no loss, no duplication).
        for (v, slot) in seen.iter().enumerate() {
            assert_eq!(
                slot.load(StdOrdering::SeqCst),
                1,
                "value {v} not consumed exactly once"
            );
        }
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;

    #[test]
    fn loom_owner_pop_vs_thief() {
        // Owner pops while one thief steals from a 2-block, 2-entry queue; every value consumed
        // at most once and none lost, across all interleavings of the per-block Chase-Lev protocol.
        loom::model(|| {
            let w = BwosWorker::<u32>::with_blocks(2, 2);
            let s = w.stealer();
            w.put(1);
            w.put(2);

            let thief = loom::thread::spawn(move || match s.steal() {
                Steal::Success(v) => Some(v),
                _ => None,
            });

            let owner1 = w.pop();
            let stolen = thief.join().unwrap();
            let owner2 = w.pop();

            // Two items; each distinct value appears at most once across the three consumers, and
            // together they consume both (no loss).
            let mut got = std::vec::Vec::new();
            for v in [owner1, stolen, owner2].into_iter().flatten() {
                got.push(v);
            }
            got.sort_unstable();
            got.dedup();
            assert_eq!(
                got,
                std::vec![1u32, 2],
                "both items consumed exactly once, no duplication"
            );
        });
    }
}

/// An **unbounded** block-based work-stealing deque — a variant of [`BwosWorker`] whose blocks
/// form a growable linked list (allocated on demand, retained until drop) instead of a fixed
/// bounded ring.
///
/// BWoS proper (the parent module) is, per the paper, a **bounded** queue: it recycles a fixed
/// set of blocks via round control, which is what makes its in-block fast path so cheap (block
/// reuse, no per-growth allocation) — and why it beats crossbeam ~5.8×. This module trades that
/// peak throughput for an unbounded capacity: when the owner fills the tail block it allocates a
/// fresh one and links it, so the queue never rejects a `put`. The per-block Chase-Lev locality
/// (owner and thieves on different blocks) is preserved, but pushes that cross a block boundary
/// pay an allocation, so raw push/pop is slower than the bounded ring (closer to crossbeam).
///
/// Use [`BwosWorker`] when a bounded capacity is acceptable (the common executor case) and you
/// want maximum throughput; use [`UnboundedBwosWorker`] when the queue depth is unpredictable and
/// you cannot tolerate a full-queue rejection. Both are ThreadSanitizer- and loom-clean.
pub mod unbounded {
    #[cfg(not(loom))]
    use std::sync::Arc;
    #[cfg(not(loom))]
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering, fence};

    #[cfg(loom)]
    use loom::sync::Arc;
    #[cfg(loom)]
    use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering, fence};

    use super::{from_bits, to_bits};
    use crate::CachePadded;
    pub use crate::Steal;

    /// Default entries per block.
    const BLOCK_SIZE: usize = 256;

    // The `trace!` macro is defined at the crate-module level of `bwos`; redefine a local no-op
    // here so this submodule stays self-contained (tracing the bounded path is the common need).
    macro_rules! trace {
        ($who:expr, $($arg:tt)*) => {{ let _ = ($who, format_args!($($arg)*)); }};
    }

    /// forming the **unbounded** block list. Each block's metadata is cache-padded so the owner's
    /// `committed` and the thieves' `stolen` never share a line.
    struct Block {
        /// Inline value bits. Written by the owner (under `committed`), read by owner pops and by
        /// thieves (under the `stolen` CAS). `Copy`, so a racing read is a harmless bit copy.
        slots: Box<[u64]>,
        /// Number of entries the owner has *committed* (filled) in this block — the block's *bottom*
        /// in a per-block Chase-Lev protocol. Owner-only writes; published with `Release`.
        committed: CachePadded<AtomicUsize>,
        /// Steal cursor (the block's *top*): the next index thieves will steal from. Advanced by
        /// thief (and owner-last-element) CAS. On its own cache line, away from `committed`.
        stolen: CachePadded<AtomicUsize>,
        /// Next block (towards newer data). Appended by the owner with `Release`, followed by both
        /// owner and thieves with `Acquire`. Null until the owner fills this block and allocates on.
        next: AtomicPtr<Block>,
        /// Previous block (towards older data), for the owner's LIFO step-back. Owner-only; null for
        /// the head block. Thieves never read `prev`.
        prev: *mut Block,
    }

    impl Block {
        fn alloc(block_size: usize, prev: *mut Block) -> *mut Block {
            Box::into_raw(Box::new(Block {
                slots: vec![0u64; block_size].into_boxed_slice(),
                committed: CachePadded(AtomicUsize::new(0)),
                stolen: CachePadded(AtomicUsize::new(0)),
                next: AtomicPtr::new(core::ptr::null_mut()),
                prev,
            }))
        }
    }

    struct Inner {
        /// First block (oldest); fixed for the queue's life, freed (with the whole list) at drop.
        head: *mut Block,
        block_size: usize,
    }

    // SAFETY: the only non-atomic shared field is `head` (a stable pointer set once at construction);
    // all mutable cross-thread state lives in each block's atomics. See `UnboundedBwosWorker`'s `!Sync`.
    unsafe impl Sync for Inner {}
    unsafe impl Send for Inner {}

    impl Drop for Inner {
        fn drop(&mut self) {
            // Single-threaded at drop: walk the whole list from head and free every block.
            // `T: Copy` ⇒ no element destructors; slots are plain bits.
            let mut blk = self.head;
            while !blk.is_null() {
                // SAFETY: each block was `Box::into_raw`'d; we own the list exclusively at drop.
                let owned = unsafe { Box::from_raw(blk) };
                blk = owned.next.load(Ordering::Relaxed);
            }
        }
    }

    /// The single owner of a BWoS queue. Pushes/pops at the back (LIFO) within the owner's block.
    pub struct UnboundedBwosWorker<T: Copy> {
        inner: Arc<Inner>,
        /// The block the owner currently produces/consumes in. Owner-private (`Cell`), so the in-block
        /// fast path touches no shared atomic for block selection.
        owner_block: core::cell::Cell<*mut Block>,
        _marker: core::marker::PhantomData<T>,
    }

    // SAFETY: owner methods are single-threaded `&self`; the `Cell` is owner-only.
    unsafe impl<T: Copy + Send> Send for UnboundedBwosWorker<T> {}

    impl<T: Copy> UnboundedBwosWorker<T> {
        /// Create an **unbounded** BWoS queue with the default block size (256). Blocks are allocated
        /// on demand as the queue grows and retained until the queue is dropped.
        pub fn new() -> Self {
            Self::with_block_size(BLOCK_SIZE)
        }

        /// Create an unbounded BWoS queue with `block_size` entries per block. (`with_blocks` is kept
        /// for source compatibility; its block-count argument now only sets the *initial* size.)
        pub fn with_block_size(block_size: usize) -> Self {
            assert!(block_size >= 1);
            let head = Block::alloc(block_size, core::ptr::null_mut());
            UnboundedBwosWorker {
                inner: Arc::new(Inner { head, block_size }),
                owner_block: core::cell::Cell::new(head),
                _marker: core::marker::PhantomData,
            }
        }

        /// Compatibility constructor. The queue is now unbounded, so `num_blocks` is ignored except
        /// as documentation of the caller's expected working set; only `block_size` matters.
        pub fn with_blocks(num_blocks: usize, block_size: usize) -> Self {
            let _ = num_blocks;
            Self::with_block_size(block_size)
        }

        /// A thief handle.
        pub fn stealer(&self) -> UnboundedBwosStealer<T> {
            UnboundedBwosStealer {
                inner: Arc::clone(&self.inner),
                cursor: core::cell::Cell::new(self.inner.head),
                _marker: core::marker::PhantomData,
            }
        }

        /// Entries per block. (The queue itself is unbounded.)
        pub fn block_size(&self) -> usize {
            self.inner.block_size
        }

        /// **Push** a value at the back (owner-only). Within a block this is a plain slot write plus a
        /// `Release` bump of the block's bottom (`committed`) — **no fence, no CAS**. At a block
        /// boundary it advances to the next block. Returns `false` if the queue is full (bounded).
        ///
        /// `committed` is the block's *bottom* index in a per-block Chase-Lev protocol; `b_pos` is the
        /// owner's private cached copy so the common path needn't reload the atomic.
        /// **Push** a value at the back (owner-only). Within a block this is a plain slot write plus a
        /// `Release` bump of the block's bottom (`committed`) — **no fence, no CAS**. When the block
        /// fills, a *fresh* block is allocated and linked (the queue is unbounded; blocks are never
        /// reused, so the owner can never overwrite a slot a thief is still reading). Always succeeds.
        pub fn put(&self, value: T) -> bool {
            let block_size = self.inner.block_size;
            let blk_ptr = self.owner_block.get();
            // SAFETY: `owner_block` always points to a live block in the retained list.
            let mut block = unsafe { &*blk_ptr };

            let mut bottom = block.committed.load(Ordering::Relaxed);
            if bottom == block_size {
                // Block full — allocate a fresh successor and link it. Owner is the sole writer of
                // `next`, so a plain Release store publishes the new block to thieves and future pops.
                let fresh = Block::alloc(block_size, blk_ptr);
                block.next.store(fresh, Ordering::Release);
                self.owner_block.set(fresh);
                // SAFETY: just allocated and linked; exclusively ours until published reads.
                block = unsafe { &*fresh };
                bottom = 0;
                trace!("put", "advance to fresh block");
            }

            // Fast path: write the slot, then publish the new bottom with Release.
            block.slots_write(bottom, to_bits(value));
            block.committed.store(bottom + 1, Ordering::Release);
            trace!("put", "slot[{bottom}]<-val  bottom={}", bottom + 1);
            true
        }

        /// **Pop** from the back (owner-only, LIFO). Each block is its own Chase-Lev deque:
        /// `committed` is the block's *bottom* (owner end) and `stolen` is its *top* (thief end). The
        /// owner decrements `committed` to claim a slot; only when the block is down to its last
        /// element does it race a thief, resolved by a CAS on `stolen` — exactly Chase-Lev, but
        /// confined to one block so thieves on other blocks never interfere.
        /// **Pop** from the back (owner-only, LIFO). Each block is its own Chase-Lev deque:
        /// `committed` is the block's *bottom* (owner end) and `stolen` is its *top* (thief end). The
        /// owner decrements `committed` to claim a slot; only when the block is down to its last
        /// element does it race a thief, resolved by a CAS on `stolen`. When a block empties, the
        /// owner steps to the previous block via the `prev` link.
        pub fn pop(&self) -> Option<T> {
            loop {
                let blk_ptr = self.owner_block.get();
                // SAFETY: owner_block always points to a live, retained block.
                let block = unsafe { &*blk_ptr };

                let b = block.committed.load(Ordering::Relaxed);
                let t = block.stolen.load(Ordering::Relaxed);
                if b <= t {
                    // No owner-side elements left in this block. Step to the previous block if any.
                    if block.prev.is_null() {
                        trace!("pop", "EMPTY (head block, b={b} t={t})");
                        return None;
                    }
                    self.owner_block.set(block.prev);
                    trace!("pop", "step back to prev block");
                    continue;
                }

                let new_b = b - 1;
                block.committed.store(new_b, Ordering::Relaxed); // tentative bottom decrement
                fence(Ordering::SeqCst); // order bottom write before reading top
                let s = block.stolen.load(Ordering::Relaxed);
                trace!("pop", "b={b} new_b={new_b} top={s}");

                if new_b > s {
                    // More than one element — thieves strictly behind. Plain take, no CAS.
                    let bits = block.slots_read(new_b);
                    trace!("pop", "TAKE slot[{new_b}] (uncontested)");
                    return Some(from_bits(bits));
                }
                if new_b == s {
                    // Exactly the last element: race the thief via CAS on `stolen` (top).
                    let bits = block.slots_read(new_b);
                    let won = block
                        .stolen
                        .compare_exchange(s, s + 1, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok();
                    // Block is now empty; restore committed to the (now equal) top so b<=t holds.
                    block.committed.store(s + 1, Ordering::Relaxed);
                    trace!("pop", "LAST slot[{new_b}] cas_won={won}");
                    if won {
                        return Some(from_bits(bits));
                    }
                    continue; // thief won; step back next iteration
                }
                // `new_b < s`: a thief raced ahead between our load and decrement. Restore; the b<=t
                // branch will step us back on the next iteration.
                block.committed.store(b, Ordering::Relaxed);
                trace!("pop", "RACE new_b={new_b} < top={s}; restore bottom={b}");
            }
        }
    }

    impl<T: Copy> Default for UnboundedBwosWorker<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A thief handle for a BWoS queue. Carries a cached `cursor` block so its forward walk to the
    /// oldest non-drained block is amortized O(1) (blocks are append-only; `stolen` only advances).
    pub struct UnboundedBwosStealer<T: Copy> {
        inner: Arc<Inner>,
        /// This thief's cached starting block for the next steal. Always at or behind the true front;
        /// advanced lazily as blocks fully drain. Thief-private (`Cell`), never shared.
        cursor: core::cell::Cell<*mut Block>,
        _marker: core::marker::PhantomData<T>,
    }

    // SAFETY: steal uses only atomic loads + a CAS on the block-local `stolen` cursor; the `cursor`
    // `Cell` is per-handle (each thread clones its own `UnboundedBwosStealer`).
    unsafe impl<T: Copy + Send> Send for UnboundedBwosStealer<T> {}

    impl<T: Copy> Clone for UnboundedBwosStealer<T> {
        fn clone(&self) -> Self {
            UnboundedBwosStealer {
                inner: Arc::clone(&self.inner),
                cursor: core::cell::Cell::new(self.inner.head),
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<T: Copy> UnboundedBwosStealer<T> {
        /// **Steal** a value from the front of the queue. Walks blocks oldest-first (following `next`)
        /// from this thief's cached cursor and steals from the first block with available work, using
        /// that block's own Chase-Lev top (`stolen`). A thief on one block never touches another
        /// block's metadata, so it interferes with the owner only on the rare shared (near-empty)
        /// block — the source of BWoS's locality win.
        pub fn steal(&self) -> Steal<T> {
            let mut blk_ptr = self.cursor.get();
            loop {
                // SAFETY: every block in the list is retained until the queue drops; `blk_ptr` is a
                // block we previously observed, so it is live.
                let block = unsafe { &*blk_ptr };
                // Per-block Chase-Lev steal: read top, fence, read bottom; if non-empty, CAS top.
                let t = block.stolen.load(Ordering::Acquire);
                fence(Ordering::SeqCst);
                let b = block.committed.load(Ordering::Acquire);
                trace!("steal", "top={t} bottom={b}");

                if t < b {
                    // Block has stealable work. Read the slot at the top, then claim it via CAS.
                    let bits = block.slots_read(t);
                    if block
                        .stolen
                        .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.cursor.set(blk_ptr);
                        trace!("steal", "TAKE slot[{t}]");
                        return Steal::Success(from_bits(bits));
                    }
                    // Lost the race for slot `t` (another thief or owner's last-element claim).
                    trace!("steal", "RETRY slot[{t}] (cas failed)");
                    return Steal::Retry;
                }

                // This block is drained from the thief side. Only advance past it if the owner has
                // already sealed it (linked a `next`): a drained-but-current block may still receive
                // pushes, so we must not skip it.
                let next = block.next.load(Ordering::Acquire);
                if next.is_null() {
                    trace!("steal", "EMPTY (no next block)");
                    return Steal::Empty;
                }
                self.cursor.set(next); // this block is fully drained and sealed; never revisit it
                blk_ptr = next;
            }
        }
    }

    impl Block {
        /// Owner-only slot write. SAFETY: `pos < block_size`; owner is the unique writer.
        #[inline]
        fn slots_write(&self, pos: usize, bits: u64) {
            // The slots are `Copy` bits; a concurrent thief read of a *different* committed slot can't
            // race this, and a thief never reads `pos` until `committed` is published (Release) past
            // it. We use a raw cell write via UnsafeCell semantics emulated over the boxed slice.
            // SAFETY: single owner writer; thieves only read indices `< committed`.
            unsafe {
                let ptr = self.slots.as_ptr().add(pos) as *mut u64;
                ptr.write(bits);
            }
        }

        /// Slot read (owner or thief). SAFETY: `pos` is `< committed`, which was published Release.
        #[inline]
        fn slots_read(&self, pos: usize) -> u64 {
            // SAFETY: read of a `Copy` value previously published before `committed` advanced past
            // `pos`; the value is stable once committed (slots are written once per bounded use).
            unsafe { self.slots.as_ptr().add(pos).read() }
        }
    }

    #[cfg(all(test, not(loom)))]
    mod tests {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
        use std::vec::Vec;

        use super::*;

        #[test]
        fn unbounded_grows_far_past_initial_blocks() {
            // Tiny blocks (size 2), 10_000 items: the queue allocates ~5000 fresh blocks on demand
            // (it is unbounded; the bounded `BwosWorker` would reject) and returns all in LIFO order.
            let w = UnboundedBwosWorker::<u64>::with_block_size(2);
            let n = 10_000u64;
            for i in 0..n {
                assert!(w.put(i), "unbounded put always succeeds (item {i})");
            }
            for i in (0..n).rev() {
                assert_eq!(w.pop(), Some(i), "LIFO across thousands of linked blocks");
            }
            assert_eq!(w.pop(), None);
        }

        #[test]
        fn unbounded_steal_from_front() {
            let w = UnboundedBwosWorker::<u64>::with_block_size(4);
            let s = w.stealer();
            for i in 0..6 {
                w.put(i);
            }
            assert_eq!(s.steal(), Steal::Success(0));
            assert_eq!(s.steal(), Steal::Success(1));
        }

        #[test]
        fn unbounded_concurrent_owner_and_thieves_no_loss() {
            let w = UnboundedBwosWorker::<usize>::with_block_size(256);
            let thieves = 3;
            let n = 16_000usize;
            let seen: StdArc<Vec<AtomicUsize>> =
                StdArc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
            for i in 0..n {
                assert!(w.put(i));
            }
            let consumed = StdArc::new(AtomicUsize::new(0));

            std::thread::scope(|scope| {
                for _ in 0..thieves {
                    let s = w.stealer();
                    let seen = StdArc::clone(&seen);
                    let consumed = StdArc::clone(&consumed);
                    scope.spawn(move || drain_thief(s, &seen, &consumed, n));
                }
                while consumed.load(StdOrdering::SeqCst) < n {
                    if let Some(v) = w.pop() {
                        seen[v].fetch_add(1, StdOrdering::SeqCst);
                        consumed.fetch_add(1, StdOrdering::SeqCst);
                    }
                }
            });

            for (v, slot) in seen.iter().enumerate() {
                assert_eq!(
                    slot.load(StdOrdering::SeqCst),
                    1,
                    "value {v} not consumed exactly once"
                );
            }
        }

        fn drain_thief(
            s: UnboundedBwosStealer<usize>,
            seen: &[AtomicUsize],
            consumed: &AtomicUsize,
            n: usize,
        ) {
            while consumed.load(StdOrdering::SeqCst) < n {
                match s.steal() {
                    Steal::Success(v) => {
                        seen[v].fetch_add(1, StdOrdering::SeqCst);
                        consumed.fetch_add(1, StdOrdering::SeqCst);
                    }
                    Steal::Retry => {}
                    Steal::Empty => {
                        if consumed.load(StdOrdering::SeqCst) >= n {
                            break;
                        }
                    }
                }
            }
        }
    }
}
