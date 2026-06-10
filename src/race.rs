//! On-the-fly **determinacy-race detection** for structured fork-join parallelism.
//!
//! Implements the order-maintenance core of
//!
//! - S. Westrick, L. Wang, U. Acar, *DePa: Simple, Provably Efficient, and Practical Order
//!   Maintenance for Task Parallelism*, arXiv:2204.14168,
//!
//! using the classic **SP-order** characterization (Feng & Leiserson): a fork-join program's
//! computation is a *series-parallel* tree whose leaves are sequential **strands**; two strands
//! run *logically in parallel* iff their lowest common ancestor is a **parallel** (spawn) node,
//! and run *in series* (one happens-before the other) iff it is a **series** (sequence) node.
//!
//! A **determinacy race** exists iff two strands that run logically in parallel access the same
//! memory location and at least one access is a write. Crucially this is a property of the
//! *program structure*, not of any particular schedule — so a single (even sequential) traversal
//! detects races that could occur under *any* parallel execution. That is what makes it stronger
//! than a sampling tool like ThreadSanitizer, which only sees the interleavings that ran.
//!
//! # How "in parallel?" is answered in O(1)
//!
//! We assign every strand two ranks by depth-first numbering of the SP-tree:
//! - **English** order: at every node, visit children left-to-right.
//! - **Hebrew** order: at *parallel* nodes, visit children right-to-left; at series nodes,
//!   left-to-right.
//!
//! Then for strands `u`, `v`: `u` happens-before `v` iff `u` precedes `v` in **both** orders.
//! They are **in parallel** iff the two orders *disagree* (`u` before `v` in one, after in the
//! other). This `(english, hebrew)` pair is exactly an SP-order labelling; the comparison is a
//! pair of integer compares.
//!
//! # Scope
//!
//! This is an *analysis* structure: you describe the computation's spawn/sync structure and the
//! memory accesses each strand makes (e.g. by instrumenting a program), then call
//! [`Dag::detect_races`]. It does not execute user closures in parallel itself — determinacy-race
//! detection deliberately reasons about the structure, independent of schedule.

use std::vec::Vec;

/// Identifier for a shared memory location being tracked (any stable `usize` key, e.g. an array
/// index or an address cast to `usize`).
pub type Location = usize;

/// A single memory access by a strand.
#[derive(Clone, Copy, Debug)]
struct Access {
    location: Location,
    is_write: bool,
}

/// A node in the series-parallel computation tree.
enum Node {
    /// A sequential strand: a leaf that performs a list of memory accesses in order.
    Strand(Vec<Access>),
    /// Series composition: children execute one-after-another (left happens-before right).
    Series(Vec<Node>),
    /// Parallel composition (a spawn): children execute logically in parallel.
    Parallel(Vec<Node>),
}

/// A reported determinacy race: two parallel strands touched `location`, at least one writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Race {
    /// The contended memory location.
    pub location: Location,
    /// Strand ids (English-order leaf indices) of the two racing accesses.
    pub strand_a: usize,
    pub strand_b: usize,
    /// Whether each access was a write (at least one is always `true`).
    pub write_a: bool,
    pub write_b: bool,
}

/// Builder for one strand's accesses, handed to the closure passed to [`Dag::strand`].
pub struct StrandBuilder {
    accesses: Vec<Access>,
}

impl StrandBuilder {
    /// Record a read of `location` by this strand.
    pub fn read(&mut self, location: Location) {
        self.accesses.push(Access {
            location,
            is_write: false,
        });
    }

    /// Record a write of `location` by this strand.
    pub fn write(&mut self, location: Location) {
        self.accesses.push(Access {
            location,
            is_write: true,
        });
    }
}

/// A builder for a series-parallel computation. Use [`series`](Dag::series) / [`parallel`](Dag::parallel)
/// to nest composition, and [`strand`](Dag::strand) for leaves that touch memory; then call
/// [`detect_races`](Dag::detect_races).
///
/// ```
/// use ws_deque::race::Dag;
/// // Two parallel strands both writing location 0 → a determinacy race.
/// let mut dag = Dag::new();
/// dag.parallel(|d| {
///     d.strand(|s| s.write(0));
///     d.strand(|s| s.write(0));
/// });
/// assert_eq!(dag.detect_races().len(), 1);
/// ```
pub struct Dag {
    children: Vec<Node>,
}

impl Dag {
    /// Start an empty computation (an implicit series root).
    pub fn new() -> Self {
        Dag {
            children: Vec::new(),
        }
    }

    /// Add a sequential strand leaf, recording its memory accesses via the builder closure.
    pub fn strand(&mut self, build: impl FnOnce(&mut StrandBuilder)) {
        let mut sb = StrandBuilder {
            accesses: Vec::new(),
        };
        build(&mut sb);
        self.children.push(Node::Strand(sb.accesses));
    }

    /// Add a **series** composition: everything added inside `build` runs in sequence.
    pub fn series(&mut self, build: impl FnOnce(&mut Dag)) {
        let mut inner = Dag::new();
        build(&mut inner);
        self.children.push(Node::Series(inner.children));
    }

    /// Add a **parallel** composition (a spawn): everything added inside `build` runs logically
    /// in parallel with the others.
    pub fn parallel(&mut self, build: impl FnOnce(&mut Dag)) {
        let mut inner = Dag::new();
        build(&mut inner);
        self.children.push(Node::Parallel(inner.children));
    }

    /// Analyze the computation and return every determinacy race (parallel strands touching the
    /// same location with at least one write). Empty result ⇒ the program is determinacy-race-free
    /// and therefore produces the same result under *any* parallel schedule.
    pub fn detect_races(&self) -> Vec<Race> {
        // 1. Assign every strand an (english, hebrew) SP-order rank. The root composes its
        //    children in series, so we walk the children list directly with a series discipline.
        let mut strands: Vec<StrandInfo> = Vec::new();
        let mut english = 0usize;
        for c in &self.children {
            assign_english(c, &mut english, &mut strands);
        }
        // Hebrew pass: the root is series, so children are visited left-to-right; each child's
        // english base is the strand count of the children before it.
        let mut hebrew = 0usize;
        let mut base = 0usize;
        for c in &self.children {
            let mut cursor = base;
            hebrew_walk(c, &mut hebrew, &mut strands, &mut cursor);
            base += count_strands(c);
        }

        // 2. Group accesses by location, then check every pair of accessing strands for a
        //    parallel write-conflict.
        find_races(&strands)
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-strand analysis record: its two SP-order ranks and the accesses it made.
struct StrandInfo {
    english: usize,
    hebrew: usize,
    accesses: Vec<Access>,
}

/// English DFS: children left-to-right at every node. Assigns each strand its english rank and
/// captures its accesses (first visit creates the record).
fn assign_english(node: &Node, counter: &mut usize, out: &mut Vec<StrandInfo>) {
    match node {
        Node::Strand(accesses) => {
            out.push(StrandInfo {
                english: *counter,
                hebrew: usize::MAX, // filled by the Hebrew pass
                accesses: accesses.clone(),
            });
            *counter += 1;
        }
        Node::Series(children) | Node::Parallel(children) => {
            for c in children {
                assign_english(c, counter, out);
            }
        }
    }
}

/// Returns the number of strands in `node` (so callers can advance the english cursor over
/// subtrees visited out of english order).
fn count_strands(node: &Node) -> usize {
    match node {
        Node::Strand(_) => 1,
        Node::Series(children) | Node::Parallel(children) => {
            children.iter().map(count_strands).sum()
        }
    }
}

/// Walk in Hebrew order while tracking each strand's english index, so we can write the hebrew
/// rank into the right `StrandInfo`. `english_cursor` is the english index of the *first* strand
/// in the subtree currently being entered.
fn hebrew_walk(
    node: &Node,
    hebrew: &mut usize,
    out: &mut [StrandInfo],
    english_cursor: &mut usize,
) {
    match node {
        Node::Strand(_) => {
            let idx = *english_cursor;
            out[idx].hebrew = *hebrew;
            *hebrew += 1;
            *english_cursor += 1;
        }
        Node::Series(children) => {
            // Series: same order in both English and Hebrew (left-to-right).
            for c in children {
                hebrew_walk(c, hebrew, out, english_cursor);
            }
        }
        Node::Parallel(children) => {
            // Parallel: Hebrew visits children right-to-left. The english cursor must still map
            // each strand to its english index, so we precompute each child's english base.
            let mut base = *english_cursor;
            let mut bases = Vec::with_capacity(children.len());
            for c in children {
                bases.push(base);
                base += count_strands(c);
            }
            // Visit right-to-left for Hebrew ordering, using each child's own english base.
            for (c, &child_base) in children.iter().zip(bases.iter()).rev() {
                let mut cursor = child_base;
                hebrew_walk(c, hebrew, out, &mut cursor);
            }
            // Advance the outer english cursor past all children.
            *english_cursor = base;
        }
    }
}

/// Two strands run logically in parallel iff the English and Hebrew orders disagree on them.
fn in_parallel(a: &StrandInfo, b: &StrandInfo) -> bool {
    (a.english < b.english) != (a.hebrew < b.hebrew)
}

/// Find all parallel write-conflicts. For each location, examine every pair of strands that
/// accessed it; report a race if they are in parallel and at least one wrote.
fn find_races(strands: &[StrandInfo]) -> Vec<Race> {
    use std::collections::HashMap;
    // location -> list of (strand index, is_write)
    let mut by_loc: HashMap<Location, Vec<(usize, bool)>> = HashMap::new();
    for (si, s) in strands.iter().enumerate() {
        for a in &s.accesses {
            by_loc.entry(a.location).or_default().push((si, a.is_write));
        }
    }

    let mut races = Vec::new();
    for (&location, accessors) in &by_loc {
        races_at_location(location, accessors, strands, &mut races);
    }
    races
}

/// Check every pair of accesses to one location and append any parallel write-conflicts.
/// Extracted from `find_races` to keep nesting shallow.
fn races_at_location(
    location: Location,
    accessors: &[(usize, bool)],
    strands: &[StrandInfo],
    races: &mut Vec<Race>,
) {
    for i in 0..accessors.len() {
        for j in (i + 1)..accessors.len() {
            let (sa, wa) = accessors[i];
            let (sb, wb) = accessors[j];
            // same strand = sequential; read/read never races; otherwise need parallelism.
            if sa == sb || !(wa || wb) || !in_parallel(&strands[sa], &strands[sb]) {
                continue;
            }
            races.push(Race {
                location,
                strand_a: strands[sa].english,
                strand_b: strands[sb].english,
                write_a: wa,
                write_b: wb,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_write_write_is_a_race() {
        let mut dag = Dag::new();
        dag.parallel(|d| {
            d.strand(|s| s.write(0));
            d.strand(|s| s.write(0));
        });
        let races = dag.detect_races();
        assert_eq!(races.len(), 1, "two parallel writers race");
        assert_eq!(races[0].location, 0);
    }

    #[test]
    fn parallel_read_write_is_a_race() {
        let mut dag = Dag::new();
        dag.parallel(|d| {
            d.strand(|s| s.read(7));
            d.strand(|s| s.write(7));
        });
        assert_eq!(dag.detect_races().len(), 1, "parallel read+write race");
    }

    #[test]
    fn parallel_read_read_is_not_a_race() {
        let mut dag = Dag::new();
        dag.parallel(|d| {
            d.strand(|s| s.read(0));
            d.strand(|s| s.read(0));
        });
        assert!(dag.detect_races().is_empty(), "two readers never race");
    }

    #[test]
    fn series_write_write_is_not_a_race() {
        // Sequenced writes happen-before each other: no race regardless of value.
        let mut dag = Dag::new();
        dag.series(|d| {
            d.strand(|s| s.write(0));
            d.strand(|s| s.write(0));
        });
        assert!(
            dag.detect_races().is_empty(),
            "sequential writes don't race"
        );
    }

    #[test]
    fn parallel_but_disjoint_locations_no_race() {
        let mut dag = Dag::new();
        dag.parallel(|d| {
            d.strand(|s| s.write(1));
            d.strand(|s| s.write(2));
        });
        assert!(
            dag.detect_races().is_empty(),
            "disjoint locations don't race"
        );
    }

    #[test]
    fn nested_reduction_is_race_free() {
        // The canonical safe pattern: each parallel strand writes its OWN slot, then a serial
        // combine reads them all. No two parallel strands touch the same location.
        let mut dag = Dag::new();
        dag.series(|d| {
            d.parallel(|p| {
                p.strand(|s| s.write(0));
                p.strand(|s| s.write(1));
                p.strand(|s| s.write(2));
                p.strand(|s| s.write(3));
            });
            d.strand(|s| {
                // Serial combine, after the join: reads are sequenced after all writes.
                s.read(0);
                s.read(1);
                s.read(2);
                s.read(3);
            });
        });
        assert!(
            dag.detect_races().is_empty(),
            "fork-then-join reduction is race-free"
        );
    }

    #[test]
    fn accumulator_into_shared_cell_is_a_race() {
        // The canonical BUG: parallel strands all += into one shared accumulator.
        let mut dag = Dag::new();
        dag.parallel(|d| {
            for _ in 0..4 {
                d.strand(|s| {
                    s.read(99); // read acc
                    s.write(99); // write acc  → parallel write-conflict
                });
            }
        });
        let races = dag.detect_races();
        assert!(!races.is_empty(), "shared accumulator must be flagged");
        assert!(races.iter().all(|r| r.location == 99));
    }

    #[test]
    fn series_of_parallels_no_cross_race() {
        // Two separate parallel phases that each write disjoint cells, sequenced: no race even
        // though cells repeat across phases (the phases are in series).
        let mut dag = Dag::new();
        dag.series(|d| {
            d.parallel(|p| {
                p.strand(|s| s.write(0));
                p.strand(|s| s.write(1));
            });
            d.parallel(|p| {
                p.strand(|s| s.write(0)); // same cell as phase 1, but sequenced after it
                p.strand(|s| s.write(1));
            });
        });
        assert!(
            dag.detect_races().is_empty(),
            "sequenced phases don't cross-race"
        );
    }
}
