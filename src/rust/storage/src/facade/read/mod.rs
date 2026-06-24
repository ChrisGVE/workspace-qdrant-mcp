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
//! Neighbors: `search.rs` (Qdrant + SQLite enrich), `list.rs` (file listing),
//!   `crate::fts::search` (FTS5 — sole module), `crate::project::ProjectRegistry`.

pub mod list;
pub mod search;

// NOTE: fts.rs is intentionally absent from this directory. All FTS5 logic
// lives in crate::fts::search (AC-F10.5 / arch §9 FP-2). Any attempt to
// create facade/read/fts.rs is caught by the structural test in fts::search.

use std::path::Path;

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::connection::open_store_readonly;
use crate::fts::fts_search;
use crate::project::ProjectRegistry;
use crate::qdrant::QdrantReadClient;
use crate::types::binding::ProjectBinding;
use crate::types::results::{FileEntry, FtsResult};

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
