//! `file_identity_id` allocation — mint-vs-inherit.
//!
//! File: src/rust/daemon/core/src/tracked_files_schema/identity.rs
//! Location: tracked_files_schema/ (the v48 state.db inventory layer).
//! Context: the branch-lineage indexing subsystem
//! (docs/architecture/branch-lineage-indexing.md). This module owns the rule
//! that decides a file's `file_identity_id` — the content-INDEPENDENT random
//! UUID that, together with `(tenant_id, file_hash)`, forms the `content_key`
//! dedup key (arch §4.3). Its sibling `operations.rs` holds the v40-era tracked
//! file CRUD (superseded by the v48 model); this module is new, self-contained,
//! and reads only the v48 `tracked_files` + `branch_lineage` tables.
//!
//! The allocation result feeds the BranchTagger (F6), which writes it into the
//! `tracked_files.file_identity_id` column and into `content_key`.

use sqlx::{Row, SqlitePool};
use uuid::Uuid;

/// The outcome of allocating a `file_identity_id` for a view row.
///
/// `Minted` means a genuinely new file-lineage was born (a parent-less first
/// ingest); `Inherited` means the id was copied from a lineage ancestor's view
/// row for the same logical file. The variant is purely informational for the
/// caller's telemetry — both variants carry the resolved `file_identity_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileIdentity {
    /// A fresh UUID was minted because the path has no lineage parent.
    Minted(Uuid),
    /// The lineage parent's id was inherited (lazy/materialized/rename).
    Inherited(Uuid),
}

impl FileIdentity {
    /// The resolved identity, regardless of how it was obtained.
    pub fn id(&self) -> Uuid {
        match self {
            FileIdentity::Minted(id) | FileIdentity::Inherited(id) => *id,
        }
    }
}

/// Allocate the `file_identity_id` for a view row at `(tenant_id, branch,
/// relative_path)`, per arch §4.4.1 (D2 = no-share, Q5 resolved).
///
/// The rule has no gap:
///
/// - **MINT** a fresh `Uuid::new_v4()` ONLY when the path has no lineage parent
///   to inherit from — a genuinely new file on this branch, or the first-ever
///   ingest of this path. These cases always carry a real ingest event, so
///   there is always something to mint at.
/// - **INHERIT** the lineage parent's `file_identity_id` on everything else:
///   lazy inheritance down the lineage chain, materialization of a previously
///   lazy row, and rename/move. Inheritance follows the **view row** (the
///   parent branch's row for the same logical path), not the bytes — so a move
///   keeps the same `file_identity_id` and P4 holds.
///
/// Parent resolution walks `branch_lineage` from `branch` toward its root
/// (depth-capped and cycle-guarded by [`MAX_LINEAGE_DEPTH`]); at each ancestor
/// branch it looks for that branch's `tracked_files` row at the SAME
/// `relative_path` and returns that row's `file_identity_id`. The first ancestor
/// that owns the path wins (nearest-ancestor-wins, matching the read-path
/// nearest-branch-wins resolution of arch §5.2). If no ancestor owns the path,
/// the row is genuinely new on this lineage and a fresh UUID is minted.
///
/// This is a state.db-only resolution: two indexed probes per ancestor
/// (`branch_lineage` PK and `idx_tracked_files_branch`), never a Qdrant read.
pub async fn allocate_file_identity(
    pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
    relative_path: &str,
) -> Result<FileIdentity, sqlx::Error> {
    match inherit_from_lineage(pool, tenant_id, branch, relative_path).await? {
        Some(parent_id) => Ok(FileIdentity::Inherited(parent_id)),
        None => Ok(FileIdentity::Minted(Uuid::new_v4())),
    }
}

/// Maximum number of lineage hops to walk before treating the chain as a root.
///
/// A safety cap mirroring the read-path depth guard (arch §5.2): a corrupt or
/// transiently-relinked `branch_lineage` row could otherwise loop or run
/// unboundedly. 64 is far beyond any realistic branch-off-branch depth.
const MAX_LINEAGE_DEPTH: usize = 64;

/// Walk the lineage chain from `branch` toward its root, returning the
/// `file_identity_id` of the nearest ancestor branch that owns a
/// `tracked_files` row at `relative_path`.
///
/// Returns `Ok(None)` when no ancestor owns the path (mint case). Cycle-guarded
/// by a visited set and depth-capped by [`MAX_LINEAGE_DEPTH`].
async fn inherit_from_lineage(
    pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
    relative_path: &str,
) -> Result<Option<Uuid>, sqlx::Error> {
    let mut visited = std::collections::HashSet::new();
    let mut current = branch.to_string();

    for _ in 0..MAX_LINEAGE_DEPTH {
        if !visited.insert(current.clone()) {
            // A cycle in branch_lineage — stop and treat as a root.
            return Ok(None);
        }

        let parent = parent_branch(pool, tenant_id, &current).await?;
        let Some(parent) = parent else {
            // Reached a lineage root with no further ancestor.
            return Ok(None);
        };

        if let Some(id) = file_identity_at(pool, tenant_id, &parent, relative_path).await? {
            return Ok(Some(id));
        }

        current = parent;
    }

    Ok(None)
}

/// The parent branch of `(tenant_id, branch)` from `branch_lineage`, or `None`
/// for a lineage root (NULL `parent_branch`) or an unknown branch.
async fn parent_branch(
    pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
) -> Result<Option<String>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT parent_branch FROM branch_lineage \
         WHERE tenant_id = ?1 AND branch = ?2",
    )
    .bind(tenant_id)
    .bind(branch)
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|r| r.get::<Option<String>, _>("parent_branch")))
}

/// The `file_identity_id` of `branch`'s view row at `relative_path`, if that
/// branch owns one. Prefers a live (non-deleted) row; a tombstoned ancestor
/// row still carries the identity to inherit (the identity outlives the
/// presence). Returns `None` when the branch has no row at this path.
async fn file_identity_at(
    pool: &SqlitePool,
    tenant_id: &str,
    branch: &str,
    relative_path: &str,
) -> Result<Option<Uuid>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT file_identity_id FROM tracked_files \
         WHERE tenant_id = ?1 AND branch = ?2 AND relative_path = ?3 \
         ORDER BY (state = 'present') DESC, created_at ASC \
         LIMIT 1",
    )
    .bind(tenant_id)
    .bind(branch)
    .bind(relative_path)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => {
            let raw: String = r.get("file_identity_id");
            // A malformed UUID in the DB is a corrupted-row condition; surface
            // it as a decode error rather than silently minting a new identity
            // (which would defeat the inheritance invariant).
            Uuid::parse_str(&raw)
                .map(Some)
                .map_err(|e| sqlx::Error::Decode(Box::new(e)))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;
    use wqm_common::hashing::content_key_v3;

    const TENANT: &str = "tenant1";
    const NOW: &str = "2025-01-01T00:00:00.000Z";

    /// An in-memory state.db migrated to the current schema (includes the v48
    /// `tracked_files` rebuild + `branch_lineage` table).
    async fn v48_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        insert_watch_folder(&pool, "wf1").await;
        pool
    }

    /// Satisfy the `tracked_files.watch_folder_id` FK.
    async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str) {
        sqlx::query(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES (?1, '/tmp/test', 'projects', ?2, ?3, ?3)",
        )
        .bind(watch_id)
        .bind(TENANT)
        .bind(NOW)
        .execute(pool)
        .await
        .unwrap();
    }

    /// Record a `branch_lineage` edge (parent = NULL marks a root).
    async fn add_lineage(pool: &SqlitePool, branch: &str, parent: Option<&str>) {
        let origin = if parent.is_some() { "event" } else { "root" };
        sqlx::query(
            "INSERT INTO branch_lineage \
             (tenant_id, branch, parent_branch, origin, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        )
        .bind(TENANT)
        .bind(branch)
        .bind(parent)
        .bind(origin)
        .bind(NOW)
        .execute(pool)
        .await
        .unwrap();
    }

    /// Insert a `tracked_files` view row carrying a known `file_identity_id`.
    /// `content_key` is derived from the same `(tenant, identity, file_hash)`
    /// the production tagger uses, so the test exercises the real key contract.
    #[allow(clippy::too_many_arguments)]
    async fn insert_view_row(
        pool: &SqlitePool,
        branch: &str,
        relative_path: &str,
        file_identity_id: &Uuid,
        file_hash: &str,
    ) {
        let id = file_identity_id.to_string();
        let ck = content_key_v3(TENANT, &id, file_hash);
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              state, file_mtime, file_hash, relative_path, created_at, updated_at) \
             VALUES ('wf1', ?1, ?2, ?3, ?4, 'present', ?5, ?6, ?7, ?5, ?5)",
        )
        .bind(TENANT)
        .bind(branch)
        .bind(&id)
        .bind(&ck)
        .bind(NOW)
        .bind(file_hash)
        .bind(relative_path)
        .execute(pool)
        .await
        .unwrap();
    }

    /// T-F4-axis-A-one-identity: the same file down a 3-branch lineage chain
    /// (main → dev → feat) resolves to ONE `file_identity_id` and ONE
    /// `content_key` — inheritance flows down the chain, no re-mint.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_axis_a_one_identity() {
        let pool = v48_pool().await;
        add_lineage(&pool, "main", None).await;
        add_lineage(&pool, "dev", Some("main")).await;
        add_lineage(&pool, "feat", Some("dev")).await;

        let path = "src/lib.rs";
        let hash = "deadbeef";

        // main is parent-less for this path → MINT.
        let on_main = allocate_file_identity(&pool, TENANT, "main", path)
            .await
            .unwrap();
        assert!(
            matches!(on_main, FileIdentity::Minted(_)),
            "first ingest on a root branch must mint, got {on_main:?}"
        );
        insert_view_row(&pool, "main", path, &on_main.id(), hash).await;

        // dev inherits main's identity for the same path.
        let on_dev = allocate_file_identity(&pool, TENANT, "dev", path)
            .await
            .unwrap();
        assert_eq!(
            on_dev,
            FileIdentity::Inherited(on_main.id()),
            "dev must inherit main's identity"
        );
        insert_view_row(&pool, "dev", path, &on_dev.id(), hash).await;

        // feat inherits transitively through dev.
        let on_feat = allocate_file_identity(&pool, TENANT, "feat", path)
            .await
            .unwrap();
        assert_eq!(
            on_feat,
            FileIdentity::Inherited(on_main.id()),
            "feat must inherit the same identity through the chain"
        );

        // ONE identity across all three branches.
        assert_eq!(on_main.id(), on_dev.id());
        assert_eq!(on_dev.id(), on_feat.id());

        // ONE content_key (same tenant, same identity, same bytes).
        let id = on_main.id().to_string();
        let ck_main = content_key_v3(TENANT, &id, hash);
        let ck_feat = content_key_v3(TENANT, &on_feat.id().to_string(), hash);
        assert_eq!(ck_main, ck_feat, "axis A must yield one content_key");
    }

    /// T-F4-D2-distinct-bytes-distinct-id: two genuinely-distinct files that
    /// happen to share identical bytes (two parent-less first ingests) get
    /// DISTINCT `file_identity_id`s and DISTINCT `content_key`s — mint is
    /// per-ingest, never per-content (the no-share invariant, D2).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_d2_distinct_bytes_distinct_id() {
        let pool = v48_pool().await;
        add_lineage(&pool, "main", None).await;

        let same_bytes = "identicalhash";

        // Two different paths, both parent-less first ingests on main.
        let file_a = allocate_file_identity(&pool, TENANT, "main", "src/a.rs")
            .await
            .unwrap();
        assert!(matches!(file_a, FileIdentity::Minted(_)));
        insert_view_row(&pool, "main", "src/a.rs", &file_a.id(), same_bytes).await;

        let file_b = allocate_file_identity(&pool, TENANT, "main", "src/b.rs")
            .await
            .unwrap();
        assert!(matches!(file_b, FileIdentity::Minted(_)));

        // Distinct identities despite identical bytes.
        assert_ne!(
            file_a.id(),
            file_b.id(),
            "two distinct files must mint distinct identities (D2)"
        );

        // Distinct content_keys → distinct real points.
        let ck_a = content_key_v3(TENANT, &file_a.id().to_string(), same_bytes);
        let ck_b = content_key_v3(TENANT, &file_b.id().to_string(), same_bytes);
        assert_ne!(
            ck_a, ck_b,
            "distinct identities over identical bytes must yield distinct content_keys"
        );
    }

    /// T-F4-rename-preserves-identity: a rename (path change, same view row /
    /// same content) preserves `file_identity_id`. The new path inherits from
    /// the SAME branch's existing row at the old path along the chain — and
    /// because identity is content-independent, the renamed file keeps its id.
    ///
    /// Modeled as: dev was branched off main; main owns `old.rs`. dev renames it
    /// to `new.rs`. Allocation for the new path on dev still inherits main's
    /// identity for the lineage (the in-place rename UPDATE of §4.5.1 keeps the
    /// row, so its identity is unchanged), and the original identity is the one
    /// that flows forward.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f4_rename_preserves_identity() {
        let pool = v48_pool().await;
        add_lineage(&pool, "main", None).await;
        add_lineage(&pool, "dev", Some("main")).await;

        let old_path = "src/old.rs";
        let hash = "renamehash";

        // main ingests the file at the old path.
        let original = allocate_file_identity(&pool, TENANT, "main", old_path)
            .await
            .unwrap();
        insert_view_row(&pool, "main", old_path, &original.id(), hash).await;

        // dev inherits the identity at the old path (its own view row).
        let on_dev = allocate_file_identity(&pool, TENANT, "dev", old_path)
            .await
            .unwrap();
        assert_eq!(on_dev, FileIdentity::Inherited(original.id()));
        insert_view_row(&pool, "dev", old_path, &on_dev.id(), hash).await;

        // The rename on dev is an in-place UPDATE of the existing row's path
        // (§4.5.1) — the identity must NOT change.
        sqlx::query(
            "UPDATE tracked_files SET relative_path = 'src/new.rs', updated_at = ?3 \
             WHERE tenant_id = ?1 AND branch = 'dev' AND relative_path = ?2",
        )
        .bind(TENANT)
        .bind(old_path)
        .bind(NOW)
        .execute(&pool)
        .await
        .unwrap();

        let after_rename: String = sqlx::query_scalar(
            "SELECT file_identity_id FROM tracked_files \
             WHERE tenant_id = ?1 AND branch = 'dev' AND relative_path = 'src/new.rs'",
        )
        .bind(TENANT)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(
            after_rename,
            original.id().to_string(),
            "a rename must preserve file_identity_id (P4)"
        );
    }
}
