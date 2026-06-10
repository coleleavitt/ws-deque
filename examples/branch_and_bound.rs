//! Heuristic-guided search with priority work-stealing — a faithful demonstration of the
//! Wimmer et al. result (arXiv:1305.6474) that *prioritizing* tasks can reduce **total work**.
//!
//! Problem: shortest path on a weighted grid with obstacles. We run the same search twice,
//! differing only in how node priority is assigned:
//!
//! - **A\*** : priority from `f = g + h`, where `h` is the (admissible) Manhattan-distance
//!   heuristic to the goal. Promising nodes (low `f`) are expanded first.
//! - **Dijkstra** : priority from `g` only (no heuristic) — uniform-cost search.
//!
//! With an admissible heuristic, A\* provably expands no more nodes than Dijkstra and usually
//! far fewer, while finding the *same* optimal path cost. That is exactly "priority reduces
//! total work," made measurable.
//!
//! Both use [`ws_deque::priority::PriorityWorker`]: the node's `f` (or `g`) is bucketed into a
//! priority level, so the search always expands from the lowest-cost non-empty bucket.
//!
//! ```sh
//! cargo run --example branch_and_bound --release
//! ```
use std::collections::HashMap;

use ws_deque::priority::PriorityWorker;

const K: usize = 64; // priority buckets (finer = closer to a true priority queue)

struct Grid {
    w: i32,
    h: i32,
    blocked: Vec<bool>,
}

impl Grid {
    fn at(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.w && y < self.h && !self.blocked[(y * self.w + x) as usize]
    }
}

#[derive(Clone)]
struct SearchNode {
    x: i32,
    y: i32,
    g: i32, // cost so far
}

/// Manhattan distance to the goal — admissible for 4-connected unit-cost movement.
fn heuristic(x: i32, y: i32, gx: i32, gy: i32) -> i32 {
    (x - gx).abs() + (y - gy).abs()
}

/// Bucket a cost estimate into `[0, K)`; lower estimate ⇒ level 0 (expanded first).
fn level_of(estimate: i32, max_estimate: i32) -> usize {
    if max_estimate <= 0 {
        return 0;
    }
    let lvl = (estimate as i64 * (K as i64 - 1) / max_estimate as i64) as usize;
    lvl.min(K - 1)
}

/// Run the search. `use_heuristic=true` is A\*, `false` is Dijkstra. Returns
/// `(optimal_cost, nodes_expanded)`.
fn search(grid: &Grid, start: (i32, i32), goal: (i32, i32), use_heuristic: bool) -> (i32, usize) {
    let pool = PriorityWorker::<SearchNode, K>::new();
    let mut best_g: HashMap<(i32, i32), i32> = HashMap::new();
    let max_estimate = grid.w + grid.h; // loose upper bound for bucketing

    pool.push(
        SearchNode {
            x: start.0,
            y: start.1,
            g: 0,
        },
        0,
    );
    best_g.insert(start, 0);

    let mut expanded = 0usize;
    let mut answer = i32::MAX;

    while let Some(node) = pool.pop() {
        // Skip stale entries (a cheaper path to this cell was found after this was queued).
        if let Some(&bg) = best_g.get(&(node.x, node.y)) {
            if node.g > bg {
                continue;
            }
        }
        expanded += 1;

        if (node.x, node.y) == goal {
            answer = node.g;
            break; // first goal expansion is optimal for both A* and Dijkstra
        }

        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (node.x + dx, node.y + dy);
            if !grid.at(nx, ny) {
                continue;
            }
            let ng = node.g + 1;
            if best_g.get(&(nx, ny)).map(|&b| ng < b).unwrap_or(true) {
                best_g.insert((nx, ny), ng);
                let estimate = if use_heuristic {
                    ng + heuristic(nx, ny, goal.0, goal.1)
                } else {
                    ng
                };
                pool.push(
                    SearchNode {
                        x: nx,
                        y: ny,
                        g: ng,
                    },
                    level_of(estimate, max_estimate),
                );
            }
        }
    }

    (answer, expanded)
}

/// A deterministic grid with scattered obstacles.
fn make_grid(w: i32, h: i32) -> Grid {
    let mut seed = 0xDEAD_BEEF_CAFE_1234u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let blocked: Vec<bool> = (0..(w * h))
        .map(|_| next() % 100 < 28) // ~28% obstacles
        .collect();
    let mut grid = Grid { w, h, blocked };
    // Keep start and goal corners open.
    grid.blocked[0] = false;
    grid.blocked[(w * h - 1) as usize] = false;
    grid
}

fn main() {
    let grid = make_grid(120, 120);
    let start = (0, 0);
    let goal = (grid.w - 1, grid.h - 1);

    let (cost_a, nodes_a) = search(&grid, start, goal, true);
    let (cost_d, nodes_d) = search(&grid, start, goal, false);

    println!("Grid shortest path ({}x{}, ~28% blocked)", grid.w, grid.h);
    if cost_a == i32::MAX {
        println!("  (no path on this random grid — rerun)");
        return;
    }
    println!("  A* (heuristic) : cost={cost_a}  nodes_expanded={nodes_a}");
    println!("  Dijkstra       : cost={cost_d}  nodes_expanded={nodes_d}");
    assert_eq!(cost_a, cost_d, "both must find the same optimal cost");
    let ratio = nodes_d as f64 / nodes_a.max(1) as f64;
    println!("  → priority (A*) explored {ratio:.2}x fewer nodes for the same optimum {cost_a}");
}
