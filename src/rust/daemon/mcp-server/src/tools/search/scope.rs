//! SQLite-bound scope adapter: base-point resolution.
//!
//! The SQLite-free scope helpers (decay-map building, relevance decay,
//! path-segment matching, degraded-reason formatting, `ScopeContext`) now live
//! in the shared `wqm-client` crate (`wqm_client::search::scope`, WI-d4 #82).
//! This module re-exports them so existing `crate::tools::search::scope::…`
//! paths keep resolving, and keeps the one piece that reads the state DB:
//! [`resolve_base_points`], which reads `watch_folders` + `tracked_files`.

use std::path::Path;

use rusqlite::Connection;

use crate::sqlite::project_queries::{get_active_base_points, get_watch_folder_id_by_tenant};

use super::types::SearchScope;

// Re-export the SQLite-free scope surface from the shared client so consumers
// (search_tool, the pipeline, tests) reach a single definition.
pub use wqm_client::search::scope::{
    apply_relevance_decay, cwd_under_base_point, format_base_points_degraded_reason,
    scope_filter_from_response, ScopeContext, BASE_POINTS_FILTER_CAP, GROUP_EMPTY_REFUSAL,
};

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

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
