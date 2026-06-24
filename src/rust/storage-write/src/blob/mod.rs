//! The blob layer — content-addressed dedup ingest (arch §4.1, §6.3, §8 nexus).
//!
//! File: `wqm-storage-write/src/blob/mod.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: This module realizes the F6 dedup ladder and its concurrency control —
//!   the §8 nexus invariant that NO blob or Qdrant blob point is written outside a
//!   per-`content_key` lock. It is the chunk-grain greenfield replacement of the
//!   retired 989-line `daemon/core/src/branch_index/tagger.rs` BranchTagger (the
//!   daemon cutover that wires this in is a follow-up — see ARCHITECTURE.md). Split
//!   across four files along responsibility boundaries (coding.md §X, AC-F6.6):
//!     - [`lock`]   — `ContentKeyLockManager` (one async lock per content_key,
//!                    eviction-bounded, sorted multi-lock = deadlock-free).
//!     - [`embed`]  — the `Embedder` trait (the lazy embed seam; embeds ONLY on a miss).
//!     - [`ladder`] — one chunk's write cycle (the two cases: hit / miss).
//!     - [`dedup`]  — file-level orchestration (`files`/`concrete` upsert + chunk loop).
//!
//! Neighbors: [`crate::schema`] (the tables written here), [`crate::qdrant`] (the
//!   eventual flush target of the enqueued ops), [`crate::connection`] (the store.db
//!   pool factory the ladder writes through).

pub mod dedup;
pub mod embed;
pub mod file_delete;
pub mod gc;
pub mod ladder;
pub mod lock;
pub mod membership;
pub mod vector_codec;

pub use dedup::{ingest_file, IngestParams};
pub use embed::{EmbeddedChunk, Embedder};
pub use file_delete::delete_file_from_branch;
pub use gc::{blob_refcount, delete_orphan_blob_row};
pub use ladder::{BlobPayload, QdrantOp, QdrantSink};
pub use lock::{ContentKeyLock, ContentKeyLockManager, LockManagerConfig};
pub use vector_codec::{decode_dense, decode_sparse, encode_dense, encode_sparse};

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared fixtures for the blob-layer ladder/dedup tests.
    //!
    //! Builds a temp `store.db` with the full schema applied, the `store_meta` tenant
    //! row populated (so the cross-tenant guard trigger passes), and a `branches` row
    //! for the test branch so the FK chain (`files`/`blob_refs`) is satisfiable.

    use sqlx::SqlitePool;
    use tempfile::TempDir;

    use crate::connection::open_store_write;
    use crate::schema::ddl_statements;

    /// The tenant every fixture uses; must match `store_meta` and every blob insert.
    pub const TENANT: &str = "tenant-test";

    /// A built fixture store: the pool plus the `TempDir` that owns the on-disk file
    /// (kept alive so the DB is not deleted while the pool is open).
    pub struct Fixture {
        pub pool: SqlitePool,
        _dir: TempDir,
    }

    /// Build a fresh store.db with schema + `store_meta` + one `branches` row.
    pub async fn fixture(branch_id: &str) -> Fixture {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("store.db");
        let pool = open_store_write(&path).await.expect("open_store_write");

        for stmt in ddl_statements() {
            sqlx::query(stmt).execute(&pool).await.expect("ddl");
        }
        sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
            .bind(TENANT)
            .execute(&pool)
            .await
            .expect("store_meta");
        add_branch(&pool, branch_id).await;

        Fixture { pool, _dir: dir }
    }

    /// Add another `branches` row (a second branch for membership-set tests).
    pub async fn add_branch(pool: &SqlitePool, branch_id: &str) {
        sqlx::query(
            "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
             VALUES (?, ?, '/repo', '2024-01-01', '2024-01-01')",
        )
        .bind(branch_id)
        .bind(branch_id)
        .execute(pool)
        .await
        .expect("branch insert");
    }
}
