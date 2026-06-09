//! SQLite-free scope resolution: group/all tenant filtering and relevance
//! decay (WI-d4, #82).
//!
//! Mirrors `resolveScopeFilter` + `applyRelevanceDecay` (`search.ts`). GitHub #81.
//!
//! Project isolation (`scope=project`) is the tenant-id filter alone. The former
//! per-file `base_point` "worktree isolation" path was removed (#115): the
//! daemon populates `tracked_files.base_point` with a content-addressed dedup
//! hash, not a worktree root, so the TS-ported isolation read a hash as a path,
//! matched nothing, and degraded recall on any project above the 500-file cap.

use std::collections::HashMap;

use crate::qdrant::fusion::TaggedResult;
use crate::workspace_daemon::ResolveSearchScopeResponse;

/// Default decay multiplier for tenants absent from the decay map (TS `?? 0.4`).
const DEFAULT_DECAY_MULTIPLIER: f64 = 0.4;

/// Refusal message for empty group membership (TS `search.ts`).
pub const GROUP_EMPTY_REFUSAL: &str =
    "Group scope requires a resolved project context. Could not determine project group membership.";

/// Resolved scope context threaded into the search pipeline and post-processing.
#[derive(Debug, Default, Clone)]
pub struct ScopeContext {
    /// Tenant IDs to restrict the search to (scope=group with `filter_by_tenant`).
    pub group_tenant_ids: Option<Vec<String>>,
    /// Per-tenant relevance-decay multipliers (scope=group/all).
    pub decay_map: Option<HashMap<String, f64>>,
}

/// Build `(group_tenant_ids, decay_map)` from a daemon `resolveSearchScope`
/// response. Mirrors the body of TS `resolveScopeFilter`.
pub fn scope_filter_from_response(
    resp: &ResolveSearchScopeResponse,
) -> (Option<Vec<String>>, Option<HashMap<String, f64>>) {
    let mut decay_map: HashMap<String, f64> = HashMap::new();
    for entry in &resp.decay_map {
        decay_map.insert(entry.tenant_id.clone(), entry.multiplier as f64);
    }
    let group_tenant_ids = if resp.filter_by_tenant {
        Some(resp.tenant_ids.clone())
    } else {
        None
    };
    let decay = if decay_map.is_empty() {
        None
    } else {
        Some(decay_map)
    };
    (group_tenant_ids, decay)
}

/// Apply per-tenant relevance decay to fused results, then re-sort by score.
///
/// Mirrors TS `applyRelevanceDecay`: multiply each result's score by its
/// tenant's multiplier (default 0.4 for tenants absent from the map); results
/// without a `tenant_id` are left unchanged. Applied to the combined results
/// BEFORE RRF fusion so the induced ordering feeds the rank-based fusion.
pub fn apply_relevance_decay(results: &mut [TaggedResult], decay_map: &HashMap<String, f64>) {
    for r in results.iter_mut() {
        let tenant = r.payload.get("tenant_id").and_then(|v| v.as_str());
        if let Some(t) = tenant {
            let multiplier = decay_map
                .get(t)
                .copied()
                .unwrap_or(DEFAULT_DECAY_MULTIPLIER);
            r.score *= multiplier;
        }
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
