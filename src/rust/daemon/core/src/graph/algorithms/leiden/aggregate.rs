//! Community aggregation phase for the Leiden algorithm.
//!
//! After the refinement phase produces a set of sub-communities, this module
//! builds the **aggregate graph** (also called the "contracted" or "reduced"
//! graph) that becomes the input to the next Leiden iteration:
//!
//! - Each refined sub-community becomes a single aggregate node.
//! - The weight of an aggregate edge `(A, B)` is the sum of all original
//!   edge weights between original nodes in sub-community `A` and original
//!   nodes in sub-community `B`.
//! - Self-loops in the aggregate graph (internal edges of a sub-community) are
//!   **retained** because local-moving on the aggregate graph uses them to
//!   compute the CPM objective correctly.
//!
//! **Returns** a triple:
//! - `UndirAdj` — the aggregate adjacency (symmetrised, with self-loops).
//! - `usize` — number of aggregate nodes.
//! - `BTreeMap<usize, usize>` — mapping from original node index to aggregate
//!   node index.
//!
//! **Determinism**: aggregate node ids are assigned in sub-community-id order
//! (the minimum original node index within each sub-community, since we use
//! original node indices as sub-community ids in the refinement phase).  All
//! structures use `BTreeMap` for stable iteration.

use std::collections::BTreeMap;

use crate::graph::algorithms::leiden::UndirAdj;

/// Build the aggregate graph from a refined partition.
///
/// `refined_partition`: `original_node_index → refined_sub_community_id`
pub(crate) fn aggregate_graph(
    adj: &UndirAdj,
    n: usize,
    refined_partition: &BTreeMap<usize, usize>,
) -> (UndirAdj, usize, BTreeMap<usize, usize>) {
    // Collect the distinct sub-community ids, sorted for determinism.
    let mut sub_ids: Vec<usize> = refined_partition
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    sub_ids.sort_unstable();

    // Assign contiguous aggregate-node indices 0..k in sub-id order.
    let sub_to_agg: BTreeMap<usize, usize> = sub_ids
        .iter()
        .enumerate()
        .map(|(agg_idx, &sub_id)| (sub_id, agg_idx))
        .collect();

    let agg_n = sub_ids.len();

    // Map original nodes to aggregate nodes.
    let node_to_agg: BTreeMap<usize, usize> = (0..n)
        .map(|orig| {
            let sub = refined_partition[&orig];
            let agg = sub_to_agg[&sub];
            (orig, agg)
        })
        .collect();

    // Build the aggregate adjacency.
    // Self-loops (intra-sub-community edges) are included as agg[a][a].
    let mut agg_adj: UndirAdj = vec![BTreeMap::new(); agg_n];

    for orig_i in 0..n {
        let agg_i = node_to_agg[&orig_i];
        for (&orig_j, &w) in &adj[orig_i] {
            let agg_j = node_to_agg[&orig_j];
            if agg_i == agg_j {
                // Intra-sub-community edge — add as self-loop (halved, since the
                // undirected edge is seen once from each endpoint).
                *agg_adj[agg_i].entry(agg_i).or_insert(0.0) += w / 2.0;
            } else {
                // Inter-sub-community edge.
                *agg_adj[agg_i].entry(agg_j).or_insert(0.0) += w / 2.0;
                *agg_adj[agg_j].entry(agg_i).or_insert(0.0) += w / 2.0;
            }
        }
    }

    (agg_adj, agg_n, node_to_agg)
}
