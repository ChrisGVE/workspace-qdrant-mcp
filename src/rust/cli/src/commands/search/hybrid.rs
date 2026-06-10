//! Hybrid search execution for the CLI (#125).
//!
//! Located at: `src/rust/cli/src/commands/search/hybrid.rs`
//!
//! Reuses the shared SQLite-free pipeline in `wqm_client::search` — the same
//! code path the MCP server runs: embeddings via the daemon's
//! EmbeddingService, dense + sparse Qdrant queries, RRF fusion. The CLI
//! pre-resolves the SQLite-bound input (project id from cwd, longest-prefix
//! `watch_folders` match) exactly like the MCP server, and passes no
//! tag-expansion keywords (that adapter is MCP-server-specific).
//!
//! Neighbors: `mod.rs` (subcommand dispatch), `render.rs` (terminal output),
//! `tui/views/search_semantic.rs` (TUI Semantic mode, same entry point).

use anyhow::{Context, Result};
use secrecy::SecretString;

use wqm_client::models::{SearchResponse, SearchScope};
use wqm_client::search::options::SearchOptions;
use wqm_client::search::scope::{scope_filter_from_response, ScopeContext};
use wqm_client::workspace_daemon::ResolveSearchScopeRequest;
use wqm_client::{DaemonClient, QdrantReadClient};

/// Run the shared hybrid/semantic/keyword search pipeline.
///
/// Connects to the daemon (embeddings + scope resolution) and Qdrant
/// (dense/sparse queries), then executes `run_search_pipeline` with the
/// CLI-resolved project id.
pub async fn run_hybrid_search(
    opts: &SearchOptions,
    project_id: Option<&str>,
) -> Result<SearchResponse> {
    let mut daemon = crate::grpc::connect_default()
        .await
        .context("Daemon not running. Start with: wqm service start")?;

    let qdrant = QdrantReadClient::new(
        crate::config::resolve_qdrant_url(),
        crate::config::resolve_qdrant_api_key().map(SecretString::from),
    );

    let scope_ctx = resolve_scope_ctx(&mut daemon, project_id, opts.scope).await;

    Ok(wqm_client::search::run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(), // tag-expansion keywords are an MCP-server-side feature
        opts,
        project_id,
        false, // no tag expansion without the keyword adapter
        &scope_ctx,
        &(), // no fallback-metrics backend in the CLI
    )
    .await)
}

/// Resolve the group/all scope tenant filter + relevance decay via the daemon.
///
/// Mirrors the MCP server's `resolve_scope_filter`: no daemon call for
/// `scope=project` or when the project is unresolved; daemon errors degrade
/// to an empty context (best-effort).
async fn resolve_scope_ctx(
    daemon: &mut DaemonClient,
    project_id: Option<&str>,
    scope: SearchScope,
) -> ScopeContext {
    let project_id = match (scope, project_id) {
        (SearchScope::Project, _) | (_, None) => return ScopeContext::default(),
        (_, Some(p)) => p,
    };
    let req = ResolveSearchScopeRequest {
        tenant_id: project_id.to_string(),
        scope: scope.as_str().to_string(),
    };
    match daemon.resolve_search_scope(req).await {
        Ok(resp) => {
            let (group_tenant_ids, decay_map) = scope_filter_from_response(&resp);
            ScopeContext {
                group_tenant_ids,
                decay_map,
            }
        }
        Err(_) => ScopeContext::default(),
    }
}

/// Resolve the current project's tenant id from the working directory via
/// longest-prefix `watch_folders` match (read-only state DB access).
pub fn resolve_project_id_from_cwd() -> Option<String> {
    let db_path = crate::config::get_database_path().ok()?;
    let cwd = std::env::current_dir().ok()?;
    wqm_common::project_id::resolve_path_to_project(&db_path, &cwd).map(|(tenant, _path)| tenant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_project_id_does_not_panic_without_db() {
        // Returns None (or a real id on a dev machine) — must never panic.
        let _ = resolve_project_id_from_cwd();
    }
}
