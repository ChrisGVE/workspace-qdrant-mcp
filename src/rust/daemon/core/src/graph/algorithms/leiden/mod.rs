//! In-house deterministic Leiden community detection over [`AdjacencyExport`].
//!
//! Implements the Leiden algorithm from Traag, Waltman & van Eck (2019),
//! "From Louvain to Leiden: guaranteeing well-connected communities",
//! Scientific Reports 9:5233, <https://doi.org/10.1038/s41598-019-41695-z>.
//!
//! The objective is the **Constant Potts Model (CPM)**:
//! ```text
//!   H = Σ_c [ e_c − γ · (n_c choose 2) ]
//! ```
//! where `e_c` = total internal edge weight of community `c`,
//! `n_c` = number of nodes in `c`, and `γ` is the resolution parameter.
//!
//! **Three-phase loop** (Leiden, not Louvain):
//! 1. Local moving — visit nodes in index order; move each to the neighbouring
//!    community that maximises ΔH; repeat until stable.
//! 2. Refinement — within each phase-1 community, start every node in its own
//!    singleton sub-community and merge only to γ-well-connected partners.
//!    This step is what distinguishes Leiden from Louvain and guarantees the
//!    well-connectedness property (Thm 3, Traag 2019).
//! 3. Aggregation — build an aggregate graph from the refined sub-communities
//!    and recurse until the partition stops changing.
//!
//! On top of the flat result a **recursive size-gated split** (DOM-06) re-runs
//! Leiden at an increased resolution `γ' = γ × resolution_step` on any
//! community whose size ≥ `max_community_members`, recursing until the
//! sub-community size falls below `min_community_members`.
//!
//! **Determinism guarantees (DOM-01)**:
//! - All adjacency, community, and assignment structures use `BTreeMap` /
//!   `BTreeSet` — iteration order is stable across daemon restarts.
//! - Single-threaded. No `rayon`, no parallel iterators.
//! - The PRNG (used only for initial community labels) is seeded once from
//!   `LeidenConfig::seed`. Node-index order is preferred over RNG wherever
//!   possible; ties in ΔH are broken by lowest community index.
//! - Two runs on identical input yield byte-identical community membership.

mod aggregate;
mod refine;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use crate::graph::AdjacencyExport;

// Re-export sub-module internals needed by tests.
pub(crate) use aggregate::aggregate_graph;
pub(crate) use refine::refine_partition;

// ─── Public API types ─────────────────────────────────────────────────────────

/// Configuration for the Leiden community detection algorithm.
#[derive(Debug, Clone)]
pub struct LeidenConfig {
    /// CPM resolution parameter γ at the base level. Higher values produce
    /// smaller, more internally dense communities.
    pub resolution: f64,
    /// Multiplier applied to `resolution` at each recursive split level
    /// (DOM-06). Must be > 1.0 to guarantee termination.
    pub resolution_step: f64,
    /// Communities with ≥ this many members are recursively split at the next
    /// resolution level.
    pub max_community_members: usize,
    /// Recursion stops when a (sub)community has fewer than this many members.
    pub min_community_members: usize,
    /// Seed for the internal PRNG (used for one-time initialisation only).
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            resolution_step: 1.5,
            max_community_members: 200,
            min_community_members: 4,
            seed: 42,
        }
    }
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Detect communities in `export` using the in-house Leiden algorithm.
///
/// Returns a `Vec` of communities, where each community is a sorted `Vec` of
/// node indices (indices into `export.node_ids`).  An empty graph returns an
/// empty `Vec`; a single-node graph returns `vec![vec![0]]`.
///
/// The graph is treated as **undirected**: edge `(i, j, w)` contributes weight
/// `w` to both `i↔j`; if both `(i, j)` and `(j, i)` appear in `export.edges`,
/// their weights are summed.
pub fn detect_communities_leiden(
    export: &AdjacencyExport,
    config: &LeidenConfig,
) -> Vec<Vec<usize>> {
    let n = export.node_ids.len();
    if n == 0 {
        return Vec::new();
    }

    // Build the symmetrised, weighted undirected adjacency from the export.
    let adj = build_undirected_adj(export);

    // Run flat Leiden at the base resolution.
    let flat = flat_leiden(&adj, n, config.resolution, config.seed);

    // Apply recursive size-gated splitting (DOM-06).
    recursive_split(flat, &adj, config, config.resolution)
}

// ─── Undirected adjacency ──────────────────────────────────────────────────────

/// Symmetrised weighted adjacency list: `adj[i]` maps neighbour index → weight.
/// Uses `BTreeMap` throughout for deterministic iteration (DOM-01).
pub(crate) type UndirAdj = Vec<BTreeMap<usize, f64>>;

/// Build an undirected, weight-summed adjacency from `AdjacencyExport`.
fn build_undirected_adj(export: &AdjacencyExport) -> UndirAdj {
    let n = export.node_ids.len();
    let mut adj: UndirAdj = vec![BTreeMap::new(); n];
    for &(src, tgt, w) in &export.edges {
        if src == tgt {
            continue; // skip self-loops
        }
        *adj[src].entry(tgt).or_insert(0.0) += w;
        *adj[tgt].entry(src).or_insert(0.0) += w;
    }
    adj
}

// ─── Flat Leiden (phases 1-3) ──────────────────────────────────────────────

/// Run the full Leiden loop (local move → refine → aggregate, repeat) and
/// return the flat partition as a map `node_index → community_id`.
///
/// Returns a `BTreeMap<usize, usize>` for deterministic downstream access.
pub(crate) fn flat_leiden(
    adj: &UndirAdj,
    n: usize,
    resolution: f64,
    seed: u64,
) -> BTreeMap<usize, usize> {
    if n == 0 {
        return BTreeMap::new();
    }
    if n == 1 {
        return std::iter::once((0, 0)).collect();
    }

    // Initial assignment: each node in its own singleton community.
    let mut partition: BTreeMap<usize, usize> = (0..n).map(|i| (i, i)).collect();

    loop {
        // Phase 1 — local moving on the current graph (starts from partition).
        let moved = local_move_phase(adj, n, &mut partition, resolution);

        // Phase 2 — refinement within each phase-1 community.
        let refined = refine_partition(adj, n, &partition, resolution, seed);

        // Phase 3 — aggregate the refined partition and build the aggregate graph.
        let (agg_adj, agg_n, node_to_agg) = aggregate_graph(adj, n, &refined);

        // Map the current partition to aggregate nodes and check for change.
        let prev_agg_partition = map_to_agg(&partition, &node_to_agg, agg_n);
        let mut agg_partition = prev_agg_partition.clone();

        // Run local moving on the aggregate graph.
        let agg_moved = local_move_phase(&agg_adj, agg_n, &mut agg_partition, resolution);

        // Lift the aggregate partition back to the original nodes.
        let new_partition = lift_partition(n, &node_to_agg, &agg_partition);

        // Converge when neither phase produced a move.
        if !moved && !agg_moved {
            partition = new_partition;
            break;
        }

        partition = new_partition;

        // Convergence check (NOT vacuous): `prev_agg_partition` is the aggregate
        // assignment captured BEFORE the aggregate local-move (line above), while
        // `partition` now holds the result lifted from the POST-move aggregate
        // assignment.  Lifting both back to original nodes and comparing detects
        // whether the aggregate local-move produced a net structural change.  We
        // stop only when it did not — i.e. the partition is stable — so the loop
        // runs to true convergence rather than exiting one iteration early.
        if partition == lift_partition(n, &node_to_agg, &prev_agg_partition) {
            break;
        }
    }

    // Re-label community ids to be contiguous starting from 0 (for stability).
    relabel_partition(partition)
}

/// Map original-node partition to aggregate-node partition.
/// Each aggregate node represents a refined sub-community; its initial
/// assignment is the phase-1 community of the constituent original nodes.
fn map_to_agg(
    partition: &BTreeMap<usize, usize>,
    node_to_agg: &BTreeMap<usize, usize>,
    agg_n: usize,
) -> BTreeMap<usize, usize> {
    // agg_node → phase-1 community (from any constituent original node).
    let mut agg_part: BTreeMap<usize, usize> = BTreeMap::new();
    for (&orig, &agg_node) in node_to_agg {
        let comm = partition[&orig];
        // All original nodes in the same refined sub-community belong to the
        // same phase-1 community, so the first write is correct.
        agg_part.entry(agg_node).or_insert(comm);
    }
    // Ensure every aggregate node has an entry (should always hold).
    for a in 0..agg_n {
        agg_part.entry(a).or_insert(a);
    }
    agg_part
}

/// Lift the aggregate partition back to original nodes.
fn lift_partition(
    n: usize,
    node_to_agg: &BTreeMap<usize, usize>,
    agg_partition: &BTreeMap<usize, usize>,
) -> BTreeMap<usize, usize> {
    (0..n)
        .map(|orig| {
            let agg_node = node_to_agg[&orig];
            let comm = agg_partition[&agg_node];
            (orig, comm)
        })
        .collect()
}

/// Relabel community ids to 0..k (contiguous), lowest original id first.
fn relabel_partition(partition: BTreeMap<usize, usize>) -> BTreeMap<usize, usize> {
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut counter = 0usize;
    // Iterate in node-index order so relabelling is deterministic.
    let mut result: BTreeMap<usize, usize> = BTreeMap::new();
    for (&node, &comm) in &partition {
        let new_comm = *old_to_new.entry(comm).or_insert_with(|| {
            let c = counter;
            counter += 1;
            c
        });
        result.insert(node, new_comm);
    }
    result
}

// ─── Phase 1: local moving ─────────────────────────────────────────────────

/// Visit all nodes in index order and move each to the neighbouring community
/// that maximises the CPM gain ΔH.  Repeat until no node moves.
/// Returns `true` if at least one node moved during the entire pass.
pub(crate) fn local_move_phase(
    adj: &UndirAdj,
    n: usize,
    partition: &mut BTreeMap<usize, usize>,
    resolution: f64,
) -> bool {
    // Precompute community sizes and internal weights.
    let mut comm_size: BTreeMap<usize, usize> = BTreeMap::new();
    let mut comm_internal: BTreeMap<usize, f64> = BTreeMap::new();

    for node in 0..n {
        let c = partition[&node];
        *comm_size.entry(c).or_insert(0) += 1;
        // Internal weight: sum of edges from node to neighbours in same community.
        let w_in: f64 = adj[node]
            .iter()
            .filter(|(&nb, _)| partition[&nb] == c)
            .map(|(_, &w)| w)
            .sum::<f64>()
            / 2.0; // each undirected edge counted once from both sides
        *comm_internal.entry(c).or_insert(0.0) += w_in;
    }

    let mut any_moved = false;
    let mut changed = true;
    while changed {
        changed = false;
        for node in 0..n {
            let c_cur = partition[&node];
            let size_cur = comm_size[&c_cur];

            // Compute weight of edges from `node` to each neighbouring community.
            let mut w_to_comm: BTreeMap<usize, f64> = BTreeMap::new();
            for (&nb, &w) in &adj[node] {
                let c_nb = partition[&nb];
                *w_to_comm.entry(c_nb).or_insert(0.0) += w;
            }

            let w_to_cur = w_to_comm.get(&c_cur).copied().unwrap_or(0.0);

            // ΔH for removing node from its current community:
            // removing node from c_cur reduces internal weight by w_to_cur
            // and changes the (n choose 2) term.
            // ΔH_remove = -(w_to_cur) + γ * (size_cur - 1)
            let delta_remove = -w_to_cur + resolution * (size_cur as f64 - 1.0);

            // Find the best target community (try all neighbouring communities).
            let mut best_comm = c_cur;
            let mut best_gain = 0.0; // net gain must exceed 0 to move

            for (&c_cand, &w_cand) in &w_to_comm {
                if c_cand == c_cur {
                    continue;
                }
                let size_cand = comm_size.get(&c_cand).copied().unwrap_or(0);
                // ΔH for adding node to c_cand:
                // ΔH_add = w_cand - γ * size_cand
                let delta_add = w_cand - resolution * size_cand as f64;
                let gain = delta_remove + delta_add;
                if gain > best_gain || (gain == best_gain && c_cand < best_comm) {
                    best_gain = gain;
                    best_comm = c_cand;
                }
            }

            if best_comm != c_cur {
                // Apply the move.
                *comm_size.entry(c_cur).or_insert(0) -= 1;
                *comm_internal.entry(c_cur).or_insert(0.0) -= w_to_cur;
                let w_to_best = w_to_comm.get(&best_comm).copied().unwrap_or(0.0);
                *comm_size.entry(best_comm).or_insert(0) += 1;
                *comm_internal.entry(best_comm).or_insert(0.0) += w_to_best;
                partition.insert(node, best_comm);
                changed = true;
                any_moved = true;
            }
        }
    }
    any_moved
}

// ─── Recursive size-gated split (DOM-06) ──────────────────────────────────

/// Convert a `BTreeMap<node → community>` partition to the canonical output
/// format `Vec<Vec<usize>>` (sorted communities of sorted node indices).
fn partition_to_output(partition: &BTreeMap<usize, usize>) -> Vec<Vec<usize>> {
    let mut by_comm: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&node, &comm) in partition {
        by_comm.entry(comm).or_default().push(node);
    }
    let mut result: Vec<Vec<usize>> = by_comm
        .into_values()
        .map(|mut members| {
            members.sort_unstable();
            members
        })
        .collect();
    // Sort communities by smallest member index for stable output ordering.
    result.sort_by_key(|members| members[0]);
    result
}

/// Recursively split large communities at higher resolution (DOM-06).
///
/// Applies [`split_community`] to every community of the flat partition.
fn recursive_split(
    partition: BTreeMap<usize, usize>,
    adj: &UndirAdj,
    config: &LeidenConfig,
    current_resolution: f64,
) -> Vec<Vec<usize>> {
    let mut final_communities: Vec<Vec<usize>> = Vec::new();
    for members in partition_to_output(&partition) {
        final_communities.extend(split_community(members, adj, config, current_resolution));
    }
    // Sort by smallest member index for stability.
    final_communities.sort_by_key(|m| m[0]);
    final_communities
}

/// Split a single community recursively at increasing resolution (DOM-06).
///
/// A community is split only if, at the higher resolution `γ' = current × step`,
/// flat Leiden partitions it into ≥2 sub-communities that ALL satisfy
/// `min_community_members`. If the higher resolution instead shatters it into
/// sub-`min` fragments — as happens for a clique, which CPM cannot subdivide
/// into mid-size well-connected parts (any partition at γ > 1 collapses to
/// singletons) — the community is *irreducible* and is kept whole even though it
/// exceeds `max_community_members`. This honours DOM-06's "recurse only while
/// members ≥ `min_community_members`" and prevents fragmenting a maximally
/// cohesive concept into meaningless pieces.
fn split_community(
    members: Vec<usize>,
    adj: &UndirAdj,
    config: &LeidenConfig,
    current_resolution: f64,
) -> Vec<Vec<usize>> {
    // Leaf: not over the size cap, or already too small to split further.
    if members.len() < config.max_community_members || members.len() < config.min_community_members
    {
        return vec![members];
    }

    // Build the induced subgraph for these members (original index → subgraph index).
    let sub_n = members.len();
    let orig_to_sub: BTreeMap<usize, usize> = members
        .iter()
        .enumerate()
        .map(|(i, &orig)| (orig, i))
        .collect();
    let mut sub_adj: UndirAdj = vec![BTreeMap::new(); sub_n];
    for (sub_i, &orig_i) in members.iter().enumerate() {
        for (&orig_j, &w) in &adj[orig_i] {
            if let Some(&sub_j) = orig_to_sub.get(&orig_j) {
                *sub_adj[sub_i].entry(sub_j).or_insert(0.0) += w;
            }
        }
    }

    // Re-cluster at the increased resolution and map parts back to original indices.
    let next_resolution = current_resolution * config.resolution_step;
    let sub_partition = flat_leiden(&sub_adj, sub_n, next_resolution, config.seed);
    let sub_parts: Vec<Vec<usize>> = partition_to_output(&sub_partition)
        .into_iter()
        .map(|part| part.into_iter().map(|sub_i| members[sub_i]).collect())
        .collect();

    // Accept the split only if it is a genuine multi-way partition whose every
    // part respects `min_community_members`; otherwise the community is
    // irreducible at this resolution and is kept whole.
    let clean = sub_parts.len() >= 2
        && sub_parts
            .iter()
            .all(|p| p.len() >= config.min_community_members);
    if !clean {
        return vec![members];
    }

    // Recurse on each accepted part (a part may itself still exceed the cap).
    let mut out: Vec<Vec<usize>> = Vec::new();
    for part in sub_parts {
        out.extend(split_community(part, adj, config, next_resolution));
    }
    out
}
