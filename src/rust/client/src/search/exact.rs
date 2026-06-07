//! FTS5 exact/substring search via daemon's TextSearchService.
//!
//! Mirrors `searchExact` in `src/typescript/mcp-server/src/tools/search-exact.ts`.
//!
//! ## Behaviour
//! - Resolves tenant scope (project vs all).
//! - Refuses to broaden to all tenants when scope=project and no tenant found (F-004).
//! - Maps `TextSearchMatch` to `SearchResult` with score `1.0 - idx * 0.001`.
//! - Collections_searched: `["projects"]` on success, `[]` on unresolved.

use std::collections::HashMap;

use serde_json::Value;

use super::options::{SearchOptions, DEFAULT_EXACT_LIMIT};
use crate::models::{SearchMode, SearchResponse, SearchResult};
use crate::workspace_daemon::TextSearchRequest;
use wqm_common::constants::COLLECTION_PROJECTS;

// ---------------------------------------------------------------------------
// Dependency traits (injectable for tests)
// ---------------------------------------------------------------------------

/// Trait for daemon text-search — injectable in tests.
pub trait ExactSearchDaemon: Send + Sync {
    fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> impl std::future::Future<
        Output = Result<crate::workspace_daemon::TextSearchResponse, tonic::Status>,
    > + Send;
}

// ---------------------------------------------------------------------------
// Tenant resolution
// ---------------------------------------------------------------------------

enum TenantResolution {
    Tenant(String),
    Unscoped,
    Unresolved,
}

fn resolve_tenant(opts: &SearchOptions) -> TenantResolution {
    if opts.scope == crate::models::SearchScope::All {
        return TenantResolution::Unscoped;
    }
    if let Some(ref pid) = opts.project_id {
        return TenantResolution::Tenant(pid.clone());
    }
    TenantResolution::Unresolved
}

// ---------------------------------------------------------------------------
// Result mapping
// ---------------------------------------------------------------------------

/// Map `TextSearchMatch` proto rows to `SearchResult`.
///
/// Mirrors `mapExactResults` in `search-exact.ts:39-65`.
/// Score: `1.0 - idx * 0.001` (search-exact.ts:52).
fn map_exact_results(matches: Vec<crate::workspace_daemon::TextSearchMatch>) -> Vec<SearchResult> {
    matches
        .into_iter()
        .enumerate()
        .map(|(idx, m)| {
            let mut metadata: HashMap<String, Value> = HashMap::new();
            metadata.insert("file_path".to_string(), Value::String(m.file_path.clone()));
            metadata.insert(
                "line_number".to_string(),
                Value::Number(m.line_number.into()),
            );
            metadata.insert("tenant_id".to_string(), Value::String(m.tenant_id));
            if let Some(ref b) = m.branch {
                metadata.insert("branch".to_string(), Value::String(b.clone()));
            }
            // M4: always emit context_before/context_after (even empty []).
            // TS `mapExactResults` always includes both keys (search-exact.ts:55-63);
            // JSON.stringify keeps `[]`. Omitting them when empty causes byte mismatch
            // for contextLines=0.
            metadata.insert(
                "context_before".to_string(),
                Value::Array(m.context_before.into_iter().map(Value::String).collect()),
            );
            metadata.insert(
                "context_after".to_string(),
                Value::Array(m.context_after.into_iter().map(Value::String).collect()),
            );
            metadata.insert(
                "_search_type".to_string(),
                Value::String("exact".to_string()),
            );
            SearchResult {
                id: format!("{}:{}", m.file_path, m.line_number),
                score: 1.0 - idx as f64 * 0.001,
                collection: COLLECTION_PROJECTS.to_string(),
                content: m.content,
                title: None,
                metadata,
                provenance: None,
                parent_context: None,
                graph_context: None,
            }
        })
        .collect()
}

/// Build the response when tenant scope is unresolved (F-004).
///
/// Mirrors `unresolvedTenantResponse` in `search-exact.ts:105-119`.
fn unresolved_tenant_response(opts: &SearchOptions) -> SearchResponse {
    SearchResponse {
        results: vec![],
        total: 0,
        query: opts.query.clone(),
        mode: SearchMode::Keyword,
        scope: opts.scope,
        collections_searched: vec![],
        status: Some("uncertain".to_string()),
        status_reason: Some(
            "Project scope requested but no project could be resolved. \
             Pass `projectId` explicitly, run from a registered project directory, \
             or set `scope: \"all\"` to search across every indexed tenant."
                .to_string(),
        ),
        branch: None,
        diversity_score: None,
    }
}

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/// Execute FTS5 exact/substring search via daemon TextSearchService.
///
/// Mirrors `searchExact` in `search-exact.ts:125-165`.
pub async fn search_exact<D>(daemon: &mut D, opts: &SearchOptions) -> SearchResponse
where
    D: ExactSearchDaemon,
{
    let resolution = resolve_tenant(opts);

    let tenant_id = match resolution {
        TenantResolution::Unresolved => return unresolved_tenant_response(opts),
        TenantResolution::Tenant(ref id) => Some(id.clone()),
        TenantResolution::Unscoped => None,
    };

    let request = build_text_search_request(opts, tenant_id);

    match daemon.text_search(request).await {
        Ok(response) => {
            let results = map_exact_results(response.matches);
            SearchResponse {
                total: response.total_matches as usize,
                query: opts.query.clone(),
                mode: SearchMode::Keyword,
                scope: opts.scope,
                collections_searched: vec![COLLECTION_PROJECTS.to_string()],
                status: None,
                status_reason: None,
                branch: None,
                diversity_score: None,
                results,
            }
        }
        Err(e) => SearchResponse {
            results: vec![],
            total: 0,
            query: opts.query.clone(),
            mode: SearchMode::Keyword,
            scope: opts.scope,
            collections_searched: vec![],
            status: Some("uncertain".to_string()),
            status_reason: Some(format!(
                "Exact search failed: {}",
                crate::grpc::status_user_message(&e)
            )),
            branch: None,
            diversity_score: None,
        },
    }
}

/// Build the TextSearchRequest from SearchOptions.
///
/// Mirrors `buildExactSearchRequest` in `search-exact.ts:68-101`.
/// M5: when caller did not specify limit, exact mode uses `DEFAULT_EXACT_LIMIT` (100).
/// TS: `max_results: options.limit ?? 100` (search-exact.ts:95).
fn build_text_search_request(opts: &SearchOptions, tenant_id: Option<String>) -> TextSearchRequest {
    let max_results = if opts.limit_explicit {
        opts.limit
    } else {
        DEFAULT_EXACT_LIMIT
    };
    TextSearchRequest {
        pattern: opts.query.clone(),
        regex: false,
        case_sensitive: true,
        context_lines: opts.context_lines as i32,
        max_results: max_results as i32,
        tenant_id,
        branch: opts.branch.clone(),
        path_glob: opts.path_glob.clone(),
        path_prefix: None,
    }
}
