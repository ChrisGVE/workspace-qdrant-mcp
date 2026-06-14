//! Leiden refinement phase.
//!
//! Within each phase-1 community, every node starts in its own singleton
//! sub-community.  Nodes are then merged into neighbouring sub-communities
//! **only when** two conditions hold (Traag 2019, §2 / Algorithm 2):
//!
//! 1. The candidate sub-community `T` is γ-well-connected to the rest of its
//!    phase-1 community — i.e., the edge weight from `T` to the remainder of
//!    the phase-1 community exceeds `γ · |T| · (|C| − |T|)` where `|C|` is
//!    the phase-1 community size.
//! 2. Merging increases the CPM objective: ΔH > 0.
//!
//! The well-connectedness check prevents the creation of poorly-connected
//! sub-communities and is the key property that distinguishes Leiden from
//! Louvain.
//!
//! **Determinism**: all data structures use `BTreeMap`/`BTreeSet`.  Nodes are
//! visited in index order.  Tie-breaking on ΔH uses lowest sub-community id.

use std::collections::{BTreeMap, BTreeSet};

use crate::graph::algorithms::leiden::UndirAdj;

/// Run the Leiden refinement phase.
///
/// Returns a mapping `original_node_index → refined_sub_community_id` where
/// each sub-community is a subset of one phase-1 community.  Sub-community ids
/// are arbitrary non-negative integers (not necessarily contiguous).
pub(crate) fn refine_partition(
    adj: &UndirAdj,
    n: usize,
    phase1_partition: &BTreeMap<usize, usize>,
    resolution: f64,
    _seed: u64, // reserved for future randomised visit order; not used (deterministic)
) -> BTreeMap<usize, usize> {
    if n == 0 {
        return BTreeMap::new();
    }

    // Group nodes by phase-1 community.
    let mut communities: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&node, &comm) in phase1_partition {
        communities.entry(comm).or_default().push(node);
    }

    // Start every node in its own singleton sub-community.
    // Use the original node index as the initial sub-community id.
    let mut refined: BTreeMap<usize, usize> = (0..n).map(|i| (i, i)).collect();

    // For each phase-1 community, attempt to merge nodes/sub-communities.
    for (_comm_id, members) in &communities {
        let comm_size = members.len();
        if comm_size <= 1 {
            continue; // singleton or empty — nothing to merge
        }

        // Build a set of member nodes for fast membership tests.
        let member_set: BTreeSet<usize> = members.iter().copied().collect();

        // Iteratively try to merge sub-communities within this phase-1 community.
        // We visit each node and attempt to move it to a neighbouring
        // sub-community inside the same phase-1 community.
        let mut changed = true;
        while changed {
            changed = false;
            for &node in members {
                let cur_sub = refined[&node];

                // Collect members of `node`'s current sub-community.
                let cur_sub_members: Vec<usize> = members
                    .iter()
                    .copied()
                    .filter(|&m| refined[&m] == cur_sub)
                    .collect();
                let cur_sub_size = cur_sub_members.len();

                // Compute edges from `node` to each neighbouring sub-community
                // within the same phase-1 community.
                let mut w_to_sub: BTreeMap<usize, f64> = BTreeMap::new();
                for (&nb, &w) in &adj[node] {
                    if member_set.contains(&nb) {
                        let nb_sub = refined[&nb];
                        *w_to_sub.entry(nb_sub).or_insert(0.0) += w;
                    }
                }

                let w_to_cur_sub = w_to_sub.get(&cur_sub).copied().unwrap_or(0.0);

                // ΔH_remove: cost of removing node from its current sub-community.
                let delta_remove = -w_to_cur_sub + resolution * (cur_sub_size as f64 - 1.0);

                let mut best_sub = cur_sub;
                let mut best_gain = 0.0;

                for (&cand_sub, &w_cand) in &w_to_sub {
                    if cand_sub == cur_sub {
                        continue;
                    }

                    let cand_members: Vec<usize> = members
                        .iter()
                        .copied()
                        .filter(|&m| refined[&m] == cand_sub)
                        .collect();
                    let cand_size = cand_members.len();

                    // Well-connectedness check (Traag 2019 Theorem 3 / Algorithm 2):
                    // The candidate sub-community T must be γ-well-connected to the
                    // rest of the phase-1 community C.
                    // w(T, C \ T) > γ · |T| · (|C| − |T|)
                    let w_cand_to_rest: f64 = cand_members
                        .iter()
                        .flat_map(|&m| adj[m].iter())
                        .filter(|(&nb, _)| member_set.contains(&nb) && !cand_members.contains(&nb))
                        .map(|(_, &w)| w)
                        .sum::<f64>()
                        / 2.0; // undirected: each edge counted once from each end

                    let well_connected_threshold =
                        resolution * cand_size as f64 * (comm_size - cand_size) as f64;

                    if w_cand_to_rest <= well_connected_threshold {
                        continue; // not γ-well-connected — skip
                    }

                    // ΔH_add: gain from adding node to the candidate sub-community.
                    let delta_add = w_cand - resolution * cand_size as f64;
                    let gain = delta_remove + delta_add;

                    if gain > best_gain || (gain == best_gain && cand_sub < best_sub) {
                        best_gain = gain;
                        best_sub = cand_sub;
                    }
                }

                if best_sub != cur_sub {
                    refined.insert(node, best_sub);
                    changed = true;
                }
            }
        }
    }

    refined
}
