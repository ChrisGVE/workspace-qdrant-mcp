//! Fallback search (daemon-down path) and F-001 refusal.
//!
//! Extracted from `flow.rs` for size compliance.
//!
//! Mirrors `fallbackSearch` in `search-qdrant.ts:367-416`.

use std::collections::HashMap;

use serde_json::Value;

use crate::qdrant::client::QdrantRetrievedPoint;
use crate::qdrant::filters::{build_filter, FilterParams};
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::flow::SearchQdrant;
use super::options::SearchOptions;
use crate::models::{SearchResponse, SearchResult, SearchScope};

// ---------------------------------------------------------------------------
// F-001 refusal constants
// ---------------------------------------------------------------------------

/// F-001 refusal reason — must match `search-qdrant.ts:404` byte-for-byte.
pub fn f001_refusal_reason(refused: &[String]) -> String {
    format!(
        "Daemon unavailable and project scope unresolved - cannot run cross-tenant fallback. \
         Refused collections: {}",
        refused.join(", ")
    )
}

/// Fallback status reason for degraded (non-refused) path — matches `search-qdrant.ts:405`.
pub const FALLBACK_STATUS_REASON: &str = "Daemon unavailable - using fallback text search";

// ---------------------------------------------------------------------------
// Fallback search
// ---------------------------------------------------------------------------

/// Fallback search when daemon is unavailable.
///
/// Mirrors `fallbackSearch` in `search-qdrant.ts:367-416`.
pub async fn fallback_search<Q>(
    qdrant: &Q,
    opts: &SearchOptions,
    collections: &[String],
    project_id: Option<&str>,
) -> SearchResponse
where
    Q: SearchQdrant,
{
    let query_lower = opts.query.to_lowercase();
    let scope = opts.scope;
    let mut results: Vec<SearchResult> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    let mut attempted = 0usize;
    let fetch_limit = (opts.limit * 3) as u32;

    for coll in collections {
        // M1 (SECURITY F-001): refuse the collection when scope=Project AND project_id is
        // unresolved, for ALL collections — not just "projects"/"scratchpad". This matches
        // TS `buildFallbackFilter` which returns null for every collection whenever
        // `scope === 'project' && !context.currentProjectId` (search-qdrant.ts:333-337).
        // M2 (SECURITY F-001): treat project_id that is None OR empty/whitespace-only as
        // unresolved. Mirrors TS `!currentProjectId` (empty string is falsy → refuse).
        if scope == SearchScope::Project && project_id_is_unresolved(project_id) {
            refused.push(coll.clone());
            continue;
        }
        // SECURITY: group scope fails closed in the daemon-down fallback. TS
        // `buildFallbackFilter` sets `groupTenantIds: undefined`, so `buildProjectCondition`
        // THROWS ("Group scope requires non-empty tenant ID set", search-filters.ts:67-69) —
        // i.e. it refuses rather than scrolling unfiltered. Rust `build_project_condition`
        // returns None (no tenant filter) in that state, which would scroll cross-tenant and
        // leak. Refuse instead to preserve the fail-closed contract.
        if scope == SearchScope::Group {
            refused.push(coll.clone());
            continue;
        }
        let filter_params = fallback_filter_params(coll, opts, project_id);
        let filter = build_filter(&filter_params);
        attempted += 1;
        if let Ok(points) = qdrant.scroll_page(coll, filter, fetch_limit).await {
            let matched: Vec<SearchResult> = points
                .into_iter()
                .filter_map(|p| scroll_point_to_result(p, coll, &query_lower))
                .collect();
            results.extend(matched);
        }
    }

    let limited: Vec<SearchResult> = results.into_iter().take(opts.limit).collect();
    let total = limited.len();
    let is_degraded = attempted == 0 && !refused.is_empty();
    let status_reason = if is_degraded {
        f001_refusal_reason(&refused)
    } else {
        FALLBACK_STATUS_REASON.to_string()
    };
    SearchResponse {
        results: limited,
        total,
        query: opts.query.clone(),
        mode: opts.mode,
        scope,
        collections_searched: collections.to_vec(),
        status: Some("uncertain".to_string()),
        status_reason: Some(status_reason),
        branch: None,
        diversity_score: None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true when the project_id should be treated as unresolved.
///
/// Mirrors TS `!currentProjectId` (empty string is falsy — M2 fix).
/// Both `None` and `Some("")` / `Some("   ")` are considered unresolved.
pub fn project_id_is_unresolved(project_id: Option<&str>) -> bool {
    match project_id {
        None => true,
        Some(s) => s.trim().is_empty(),
    }
}

pub(super) fn fallback_filter_params<'a>(
    collection: &'a str,
    opts: &'a SearchOptions,
    project_id: Option<&'a str>,
) -> FilterParams {
    FilterParams {
        collection: collection.to_string(),
        scope: opts.scope.as_str().to_string(),
        // M2: treat empty/whitespace project_id as unresolved (mirrors TS !currentProjectId).
        project_id: project_id
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string),
        // TS `buildFallbackFilter` sets `groupTenantIds: undefined` — the
        // daemon-down fallback is a project-tenant-scoped substring scan and does
        // not apply group/all tenant filtering (search-qdrant.ts:342).
        group_tenant_ids: None,
        branch: opts.branch.clone(),
        file_type: opts.file_type.clone(),
        library_name: if collection == COLLECTION_LIBRARIES {
            opts.library_name.clone()
        } else {
            None
        },
        library_path: if collection == COLLECTION_LIBRARIES {
            opts.library_path.clone()
        } else {
            None
        },
        tag: opts.tag.clone(),
        tags: opts.tags.clone(),
        path_glob: opts.path_glob.clone(),
        component: opts.component.clone(),
    }
}

pub(super) fn scroll_point_to_result(
    p: QdrantRetrievedPoint,
    collection: &str,
    query_lower: &str,
) -> Option<SearchResult> {
    let content = p
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let title_str = p
        .payload
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !content.to_lowercase().contains(query_lower)
        && !title_str.to_lowercase().contains(query_lower)
    {
        return None;
    }
    let mut metadata: HashMap<String, Value> = p.payload.clone();
    metadata.insert(
        "_search_type".to_string(),
        Value::String("fallback".to_string()),
    );
    Some(SearchResult {
        id: p.id,
        score: 0.5,
        collection: collection.to_string(),
        content,
        title: if title_str.is_empty() {
            None
        } else {
            Some(title_str)
        },
        metadata,
        provenance: None,
        parent_context: None,
        graph_context: None,
    })
}
