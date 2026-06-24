//! `ReadStoreFacade` — the read-side storage entrypoint (arch §6.2, AC-F10).
//!
//! File: `wqm-storage/src/facade/read/mod.rs`
//! Location: `src/rust/storage/src/facade/read/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch §6.2). This struct
//!   is the single entrypoint for all read operations: hybrid search, FTS5
//!   search, file listing, and project resolution. It holds a `ProjectRegistry`
//!   for CWD->tenant resolution (AC-F10.10) and opens per-project `store.db`
//!   pools on demand.
//!
//!   Hard boundaries (arch §9):
//!     - No DDL, no INSERT/UPDATE/DELETE, no git2.
//!     - No `wqm-storage-write` dependency (Guard 1).
//!     - FTS5 exclusively via `crate::fts::search::fts_search` (AC-F10.5);
//!       `facade/read/fts.rs` must NOT exist.
//!
//!   F17 adds `search_scoped` — the fan-out entry point for scope=group|all.
//!   The method delegates enumeration to `ProjectRegistry::enumerate_by_scope`,
//!   checks the cliff for scope=all, fans out via bounded concurrency, caps per-
//!   project results, merges via RRF (per-project normalized), and re-assembles
//!   an ordered `Vec<EnrichedHit>`.
//!
//!   Live wiring deferral: `search_scoped` is complete and tested offline.
//!   Wiring to MCP server / CLI rides the read-facade cutover (same posture as
//!   F8/F20; tracked separately; see docs/architecture/branch-storage-model.md
//!   §8 Path note).
//!
//! Neighbors: `search.rs` (Qdrant + SQLite enrich), `list.rs` (file listing),
//!   `fanout.rs` (fan-out primitives, AC-F17), `crate::fts::search` (FTS5),
//!   `crate::project::ProjectRegistry`.

pub mod fanout;
pub mod list;
pub mod search;

// NOTE: fts.rs is intentionally absent from this directory. All FTS5 logic
// lives in crate::fts::search (AC-F10.5 / arch §9 FP-2). Any attempt to
// create facade/read/fts.rs is caught by the structural test in fts::search.

use std::path::Path;

use sqlx::SqlitePool;
use wqm_common::{error::StorageError, search::types::SearchResult};

use crate::connection::open_store_readonly;
use crate::fts::fts_search;
use crate::project::{ProjectRegistry, SearchScope};
use crate::qdrant::QdrantReadClient;
use crate::types::binding::ProjectBinding;
use crate::types::results::{FileEntry, FtsResult};

use fanout::{build_project_collection, merge_project_results, run_bounded, FanoutConfig};
use list::list_branch as list_branch_inner;
use search::{branch_search, EnrichedHit};

// ---------------------------------------------------------------------------
// ReadStoreFacade
// ---------------------------------------------------------------------------

/// The read-side storage facade (arch §6.2).
///
/// All read operations — hybrid search, FTS5, file listing — go through this
/// struct. It holds shared resources (registry, Qdrant client) and opens
/// per-project `store.db` pools on demand via `open_store_readonly`.
pub struct ReadStoreFacade {
    /// CWD->tenant resolver (state.db). Minted by F10, extended by F16.
    registry: ProjectRegistry,
    /// Read-only Qdrant client (Guard 2 — no mutation).
    qdrant: QdrantReadClient,
}

impl ReadStoreFacade {
    /// Construct a facade from an already-opened registry and Qdrant client.
    pub fn new(registry: ProjectRegistry, qdrant: QdrantReadClient) -> Self {
        Self { registry, qdrant }
    }

    // -----------------------------------------------------------------------
    // Project resolution (AC-F10.4, AC-F10.10)
    // -----------------------------------------------------------------------

    /// Resolve `cwd` to the owning `ProjectBinding`.
    ///
    /// Canonicalizes the path before any query (arch §6.5). Returns `None`
    /// when no registered root matches. Callers that receive `None` MUST
    /// return an error or empty result — never fall through to an all-tenant
    /// query (SEC-3, AC-F10.2).
    pub async fn resolve_project(
        &self,
        cwd: impl AsRef<Path>,
    ) -> Result<Option<ProjectBinding>, StorageError> {
        self.registry.resolve_project(cwd).await
    }

    // -----------------------------------------------------------------------
    // Hybrid search (AC-F10.1, AC-F10.2)
    // -----------------------------------------------------------------------

    /// Hybrid branch-scoped search: dense + sparse Qdrant, RRF fusion,
    /// SQLite enrichment via `idx_blob_refs_covering` JOIN (arch §4.4).
    ///
    /// `binding` must come from `resolve_project`; `None` binding is rejected
    /// before any Qdrant call (SEC-3 / AC-F10.2).
    pub async fn search(
        &self,
        binding: &ProjectBinding,
        dense_vec: Vec<f32>,
        sparse_indices: Vec<u32>,
        sparse_values: Vec<f32>,
        top_k: u64,
    ) -> Result<Vec<EnrichedHit>, StorageError> {
        let pool = self.open_store(binding).await?;
        branch_search(
            &self.qdrant,
            &pool,
            binding.tenant_id.as_str(),
            binding.branch_id.as_str(),
            dense_vec,
            sparse_indices,
            sparse_values,
            top_k,
        )
        .await
    }

    // -----------------------------------------------------------------------
    // Scoped fan-out search (AC-F17)
    // -----------------------------------------------------------------------

    /// Multi-project hybrid search for `scope=group|all` (AC-F17, arch R5).
    ///
    /// For `scope=project` this delegates to `search` (the existing F10 path).
    /// For `scope=group|all` it fans out across all projects in the scope with
    /// bounded concurrency, merges by RRF normalized per project, and returns
    /// a ranked `Vec<EnrichedHit>` ordered by fused score.
    ///
    /// `scope=all` above `config.cliff` projects returns `ScopeTooBroad`
    /// immediately — never a silent slow path (AC-F17.2).
    ///
    /// `config` is `FanoutConfig::default()` unless the caller needs custom
    /// cliff/concurrency values.
    pub async fn search_scoped(
        &self,
        anchor: &ProjectBinding,
        scope: SearchScope,
        dense_vec: Vec<f32>,
        sparse_indices: Vec<u32>,
        sparse_values: Vec<f32>,
        top_k: u64,
        config: FanoutConfig,
    ) -> Result<Vec<EnrichedHit>, StorageError> {
        // scope=project is the existing single-project path — no fan-out.
        if scope == SearchScope::Project {
            return self
                .search(anchor, dense_vec, sparse_indices, sparse_values, top_k)
                .await;
        }

        // Enumerate bindings for the requested scope.
        let bindings = self
            .registry
            .enumerate_by_scope(scope, anchor.tenant_id.as_str())
            .await?;

        // scope=all cliff guard (AC-F17.2).
        if scope == SearchScope::All {
            config.check_cliff(bindings.len(), "all")?;
        }

        if bindings.is_empty() {
            return Ok(vec![]);
        }

        // Fan out: one task per binding, bounded concurrency.
        let per_project = self
            .run_per_project_searches(
                bindings,
                dense_vec,
                sparse_indices,
                sparse_values,
                top_k,
                config.concurrency,
            )
            .await?;

        // Cross-project RRF merge (AC-F17.1).
        Ok(assemble_merged_hits(per_project, top_k))
    }

    /// Spawn one search task per binding under the concurrency semaphore.
    ///
    /// Returns per-project `(tenant_id, Vec<EnrichedHit>)` in submission order.
    async fn run_per_project_searches(
        &self,
        bindings: Vec<ProjectBinding>,
        dense_vec: Vec<f32>,
        sparse_indices: Vec<u32>,
        sparse_values: Vec<f32>,
        top_k: u64,
        concurrency: usize,
    ) -> Result<Vec<(String, Vec<EnrichedHit>)>, StorageError> {
        let qdrant = self.qdrant.clone();

        let tasks: Vec<_> = bindings
            .into_iter()
            .map(|binding| {
                let qdrant = qdrant.clone();
                let dv = dense_vec.clone();
                let si = sparse_indices.clone();
                let sv = sparse_values.clone();
                move || async move {
                    let pool = open_store_readonly(&binding.db_path).await?;
                    let hits = branch_search(
                        &qdrant,
                        &pool,
                        binding.tenant_id.as_str(),
                        binding.branch_id.as_str(),
                        dv,
                        si,
                        sv,
                        top_k,
                    )
                    .await?;
                    Ok((binding.tenant_id.as_str().to_string(), hits))
                }
            })
            .collect();

        run_bounded(tasks, concurrency).await
    }

    // -----------------------------------------------------------------------
    // FTS5 search (AC-F10.3, AC-F10.5)
    // -----------------------------------------------------------------------

    /// Branch-scoped FTS5 full-text search (arch §5.2, AC-F10.3).
    ///
    /// Delegates exclusively to `crate::fts::fts_search` (the sole FTS5
    /// module, AC-F10.5). The query is sanitized before binding (arch §6.5 A5).
    pub async fn fts_search(
        &self,
        binding: &ProjectBinding,
        query: &str,
        limit: u32,
    ) -> Result<Vec<FtsResult>, StorageError> {
        let pool = self.open_store(binding).await?;
        fts_search(&pool, query, binding.branch_id.as_str(), limit).await
    }

    // -----------------------------------------------------------------------
    // File listing (AC-F10.1 — list_branch)
    // -----------------------------------------------------------------------

    /// List all files known to the branch in `binding`.
    pub async fn list_branch(
        &self,
        binding: &ProjectBinding,
    ) -> Result<Vec<FileEntry>, StorageError> {
        let pool = self.open_store(binding).await?;
        list_branch_inner(&pool, binding.branch_id.as_str()).await
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Open (or re-open) the per-project `store.db` pool for `binding`.
    ///
    /// Uses `open_store_readonly` which enforces the two-layer read-only
    /// guarantee (SQLITE_OPEN_READONLY + query_only = ON, AC-F14.2).
    async fn open_store(&self, binding: &ProjectBinding) -> Result<SqlitePool, StorageError> {
        open_store_readonly(&binding.db_path).await
    }
}

// ---------------------------------------------------------------------------
// Fan-out assembly helpers
// ---------------------------------------------------------------------------

/// Merge per-project `EnrichedHit` lists via cross-project RRF, then return
/// top-`top_k` hits ordered by fused score.
///
/// Steps (AC-F17.1, AC-F17.3):
///   1. Truncate each project's hit list to `top_k` (bounds candidate set to
///      P*K — AC-F17.3). `EnrichedHit` is already sorted best-first by
///      `branch_search`.
///   2. Convert each capped hit to `SearchResult` (id = point_id) for the
///      RRF primitives.
///   3. Merge all project collections via `rrf_merge` keyed by tenant_id —
///      one ranked list per project, normalized by the RRF formula (AC-F17.1,
///      DR GP-9: wqm_common::search::rrf, no fork).
///   4. Re-assemble `EnrichedHit`s from the fused order via a point_id map.
///   5. Truncate final output to `top_k`.
fn assemble_merged_hits(
    per_project: Vec<(String, Vec<EnrichedHit>)>,
    top_k: u64,
) -> Vec<EnrichedHit> {
    let k = top_k as usize;

    let mut hit_map: std::collections::HashMap<String, EnrichedHit> =
        std::collections::HashMap::new();
    let mut collections: Vec<(String, Vec<SearchResult>)> = Vec::with_capacity(per_project.len());

    for (tenant_id, hits) in per_project {
        // Inline top-K cap on EnrichedHit (AC-F17.3).
        let capped: Vec<EnrichedHit> = hits.into_iter().take(k).collect();
        let search_results = enriched_to_search_results(&capped);
        let (key, sr) = build_project_collection(&tenant_id, search_results);
        for hit in capped {
            hit_map.insert(hit.point_id.clone(), hit);
        }
        collections.push((key, sr));
    }

    // Cross-project RRF merge (wqm_common — no fork, AC-F17.1 DR GP-9).
    let merged = merge_project_results(collections, 60.0);

    // Reassemble EnrichedHits in fused rank order, truncated to top_k.
    merged
        .into_iter()
        .take(k)
        .filter_map(|ccr| hit_map.remove(&ccr.result.id))
        .collect()
}

/// Convert `EnrichedHit`s to `SearchResult`s for the cross-project RRF input.
///
/// Only `id` (= `point_id`) and `score` are used by `rrf_merge`; payload and
/// vector fields are left empty to avoid cloning large blobs unnecessarily.
fn enriched_to_search_results(hits: &[EnrichedHit]) -> Vec<SearchResult> {
    hits.iter()
        .map(|h| SearchResult {
            id: h.point_id.clone(),
            score: h.score,
            payload: std::collections::HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        })
        .collect()
}
