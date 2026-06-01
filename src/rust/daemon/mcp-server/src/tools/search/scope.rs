//! Scope resolution: group/all tenant filtering, relevance decay, base-points.
//!
//! Mirrors `resolveScopeFilter` + `applyRelevanceDecay` (`search.ts`) and the
//! base-point resolution in `resolveProjectContext` (`search-helpers.ts`).
//! GitHub #81.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::Connection;

use crate::proto::ResolveSearchScopeResponse;
use crate::qdrant::fusion::TaggedResult;
use crate::sqlite::project_queries::{get_active_base_points, get_watch_folder_id_by_tenant};

use super::types::SearchScope;

/// F-012 base-point filter cap (TS `BASE_POINTS_FILTER_CAP`).
pub const BASE_POINTS_FILTER_CAP: usize = 500;

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
    /// Per-instance base points for worktree isolation (scope=project).
    pub base_points: Option<Vec<String>>,
    /// Per-tenant relevance-decay multipliers (scope=group/all).
    pub decay_map: Option<HashMap<String, f64>>,
    /// True when the base-point set exceeded the cap and no primary point
    /// matched cwd (F-014 instance-isolation degraded).
    pub base_points_degraded: bool,
    /// Active base-point count when degraded (for the status reason).
    pub base_points_active_count: Option<usize>,
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

/// Resolve base points for worktree/instance isolation.
///
/// Mirrors the base-point block of TS `resolveProjectContext`: only for
/// `scope=project` with a known project. Returns
/// `(base_points, degraded, active_count)`.
///
/// - 0 active points → `(None, false, None)`.
/// - `1..=CAP` → `(Some(points), false, None)`.
/// - `> CAP` → narrow to the single base point that is a prefix of `cwd`
///   (`(Some([primary]), false, None)`); if none matches, degrade
///   (`(None, true, Some(count))`).
pub fn resolve_base_points(
    conn: Option<&Connection>,
    project_id: Option<&str>,
    scope: SearchScope,
    cwd: &Path,
) -> (Option<Vec<String>>, bool, Option<usize>) {
    let project_id = match (project_id, scope) {
        (Some(p), SearchScope::Project) => p,
        _ => return (None, false, None),
    };
    let Some(watch_id) = get_watch_folder_id_by_tenant(conn, project_id) else {
        return (None, false, None);
    };
    let points = get_active_base_points(conn, &watch_id, false);
    if points.is_empty() {
        return (None, false, None);
    }
    if points.len() <= BASE_POINTS_FILTER_CAP {
        return (Some(points), false, None);
    }
    // > cap: narrow to the primary base point that contains cwd (F-012).
    // Path-segment-aware prefix: `cwd == bp` or `cwd` starts with `bp` + a path
    // separator. This hardens beyond TS's raw `cwd.startsWith(bp)` (search-helpers.ts:90),
    // which false-positives on sibling roots sharing a prefix (e.g. `/repo` vs `/repo-a`).
    let cwd_str = cwd.to_string_lossy();
    match points.iter().find(|bp| cwd_under_base_point(&cwd_str, bp)) {
        Some(primary) => (Some(vec![primary.clone()]), false, None),
        None => (None, true, Some(points.len())),
    }
}

/// True when `cwd` is the base point itself or a descendant of it, using
/// path-segment boundaries (not a raw string prefix). Trailing separators on the
/// base point are tolerated.
fn cwd_under_base_point(cwd: &str, base_point: &str) -> bool {
    let sep = std::path::MAIN_SEPARATOR;
    let bp = base_point.trim_end_matches(sep);
    if cwd == bp {
        return true;
    }
    cwd.strip_prefix(bp)
        .is_some_and(|rest| rest.starts_with(sep))
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

/// Format the F-014 base-point-degraded status reason (TS
/// `formatBasePointsDegradedReason`).
pub fn format_base_points_degraded_reason(active_count: Option<usize>) -> String {
    let count = active_count
        .map(|c| c.to_string())
        .unwrap_or_else(|| "too many".to_string());
    format!(
        "Worktree/instance isolation degraded: project has {count} active base points, \
         exceeding the 500-filter cap; tenant filter still applies but base-point \
         narrowing was bypassed. Narrow further with pathGlob, branch, or component to \
         restore worktree-level isolation."
    )
}

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
