//! Pool-based database operations for tracked files and Qdrant chunks

use sqlx::{sqlite::SqliteRow, Row, Sqlite, SqlitePool};
use std::path::Path;
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::paths::RelativePath;
use wqm_common::timestamps;

use super::types::{ChunkType, ProcessingStatus, TrackedFile};

// Re-export hashing functions from wqm-common
pub use wqm_common::hashing::{compute_content_hash, compute_file_hash};

/// The full v48 `tracked_files` column list read by [`tracked_file_from_row`].
///
/// Kept as a single source of truth so every `SELECT *`-style read (the two
/// [`lookup_tracked_file`] arms, [`super::reconcile::get_files_needing_reconcile`])
/// stays in lock-step with the row decoder and cannot drift column-by-column.
pub(crate) const TRACKED_FILE_COLUMNS: &str =
    "file_id, watch_folder_id, relative_path, tenant_id, \
     branch, file_identity_id, content_key, is_virtual, state, \
     file_type, language, file_mtime, file_hash, chunk_count, chunking_method, \
     lsp_status, treesitter_status, last_error, needs_reconcile, reconcile_reason, \
     extension, is_test, collection, base_point, incremental, \
     component, routing_reason, created_at, updated_at";

/// Build a TrackedFile from a SQLite row.
///
/// Post-v37: the row carries a single relative path column (`relative_path`).
/// Post-v48 (branch-lineage): the v40 `primary_branch`/`branches` JSON columns
/// were dropped; the row carries the scalar `branch`, `tenant_id`, `content_key`,
/// `file_identity_id`, `is_virtual`, and `state` instead (`schema_version/v48.rs`).
/// The caller's SELECT must list [`TRACKED_FILE_COLUMNS`].
pub(crate) fn tracked_file_from_row(r: &SqliteRow) -> TrackedFile {
    let raw_relative: String = r.get("relative_path");
    let relative_path = RelativePath::from_validated(raw_relative.clone()).unwrap_or_else(|e| {
        // The DB invariant established by v37 says every row has a
        // non-null, normalized relative_path. A failure here is a
        // corrupted-row condition; fail closed by treating the path as
        // an opaque single-segment relative — but still record the
        // diagnostic so it surfaces in tests/logs.
        tracing::error!(
            "tracked_files row decode: invalid relative_path {:?}: {} — substituting normalized form",
            raw_relative,
            e
        );
        RelativePath::from_user_input(raw_relative.trim_start_matches('/'))
            .unwrap_or_else(|_| {
                RelativePath::from_user_input("invalid_path").expect("literal is valid")
            })
    });

    TrackedFile {
        file_id: r.get("file_id"),
        watch_folder_id: r.get("watch_folder_id"),
        relative_path,
        tenant_id: r.get("tenant_id"),
        branch: r.get("branch"),
        file_identity_id: r.get("file_identity_id"),
        content_key: r.get("content_key"),
        is_virtual: r.get::<i64, _>("is_virtual") != 0,
        state: r.get("state"),
        file_type: r.get("file_type"),
        language: r.get("language"),
        file_mtime: r.get("file_mtime"),
        file_hash: r.get("file_hash"),
        chunk_count: r.get("chunk_count"),
        chunking_method: r.get("chunking_method"),
        lsp_status: ProcessingStatus::from_str(
            r.get::<Option<String>, _>("lsp_status")
                .as_deref()
                .unwrap_or("none"),
        )
        .unwrap_or(ProcessingStatus::None),
        treesitter_status: ProcessingStatus::from_str(
            r.get::<Option<String>, _>("treesitter_status")
                .as_deref()
                .unwrap_or("none"),
        )
        .unwrap_or(ProcessingStatus::None),
        last_error: r.get("last_error"),
        needs_reconcile: r.get::<i32, _>("needs_reconcile") != 0,
        reconcile_reason: r.get("reconcile_reason"),
        extension: r.get("extension"),
        is_test: r.get::<Option<i32>, _>("is_test").unwrap_or(0) != 0,
        collection: r
            .get::<Option<String>, _>("collection")
            .unwrap_or_else(|| COLLECTION_PROJECTS.to_string()),
        base_point: r.get("base_point"),
        incremental: r.get::<Option<i32>, _>("incremental").unwrap_or(0) != 0,
        component: r.get("component"),
        routing_reason: r.get("routing_reason"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    }
}

/// Get file modification time as ISO 8601 string
pub fn get_file_mtime(path: &Path) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path)?;
    let mtime = metadata.modified()?;
    let datetime: chrono::DateTime<chrono::Utc> = mtime.into();
    Ok(timestamps::format_utc(&datetime))
}

/// Look up a watch_folder by tenant_id and collection, return (watch_id, path)
pub async fn lookup_watch_folder(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
) -> Result<Option<(String, String)>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT watch_id, path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 AND enabled = 1 LIMIT 1"
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| (r.get("watch_id"), r.get("path"))))
}

/// Look up a live tracked file by (watch_folder_id, relative_path[, branch]).
///
/// v48 (branch-lineage) keeps one live row per `(branch, path)`, so the lookup
/// is a direct equality match — no `branches` JSON membership fallback:
///
/// - `Some(branch)` → the row for exactly that branch and path.
/// - `None` → any live row for the path (first match) — used by callers that do
///   not care which branch owns the row.
///
/// Both arms filter on `state = 'present'`, so logically-deleted tombstones are
/// never returned. Returns `Ok(None)` when no live row matches.
pub async fn lookup_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    branch: Option<&str>,
) -> Result<Option<TrackedFile>, sqlx::Error> {
    let row = match branch {
        Some(b) => {
            let sql = format!(
                "SELECT {TRACKED_FILE_COLUMNS} FROM tracked_files \
                 WHERE watch_folder_id = ?1 AND relative_path = ?2 \
                   AND branch = ?3 AND state = 'present'"
            );
            sqlx::query(&sql)
                .bind(watch_folder_id)
                .bind(relative_path)
                .bind(b)
                .fetch_optional(pool)
                .await?
        }
        None => {
            let sql = format!(
                "SELECT {TRACKED_FILE_COLUMNS} FROM tracked_files \
                 WHERE watch_folder_id = ?1 AND relative_path = ?2 \
                   AND state = 'present' LIMIT 1"
            );
            sqlx::query(&sql)
                .bind(watch_folder_id)
                .bind(relative_path)
                .fetch_optional(pool)
                .await?
        }
    };

    Ok(row.map(|r| tracked_file_from_row(&r)))
}

/// Insert a v48 `tracked_files` row, returning the `file_id` (AUTOINCREMENT PK).
///
/// The v48 branch-lineage model (one row per `(branch, path)`, virtual shadows,
/// lifecycle `state`) replaced the v40 columns (`primary_branch`, `branches`
/// JSON array) — see `schema_version/v48.rs`. This is the sole tracked-file
/// insert path; it is called by the branch tagger (`branch_index::tagger`, F6)
/// for all three dedup cases and by the zero-byte recorder.
///
/// `created_at`/`updated_at` are generated internally via
/// [`wqm_common::timestamps::now_utc`] (the established idiom). `lsp_status`/
/// `treesitter_status` should be [`ProcessingStatus::None`] for virtual/initial
/// rows. `is_virtual`/`is_test` are stored as `0`/`1`; `needs_reconcile` is `1`
/// for the crash-safe real-point cases (cleared on success) and `0` for virtual
/// writes (arch §4.2/§5.1). `last_error`, `incremental`, and `routing_reason`
/// take their DDL defaults (NULL / 0).
#[allow(clippy::too_many_arguments)]
pub async fn insert_tracked_file_v48(
    pool: &SqlitePool,
    watch_folder_id: &str,
    tenant_id: &str,
    branch: &str,
    file_identity_id: &str,
    content_key: &str,
    is_virtual: bool,
    state: &str,
    file_type: Option<&str>,
    language: Option<&str>,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    collection: &str,
    extension: Option<&str>,
    is_test: bool,
    base_point: Option<&str>,
    component: Option<&str>,
    relative_path: &str,
    needs_reconcile: bool,
    reconcile_reason: Option<&str>,
) -> Result<i64, sqlx::Error> {
    let now = timestamps::now_utc();
    let result = sqlx::query(
        "INSERT INTO tracked_files (
            watch_folder_id, tenant_id, branch, file_identity_id, content_key,
            is_virtual, state, file_type, language, file_mtime, file_hash,
            chunk_count, chunking_method, lsp_status, treesitter_status,
            collection, extension, is_test, base_point, component, relative_path,
            needs_reconcile, reconcile_reason, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                 ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25)",
    )
    .bind(watch_folder_id)
    .bind(tenant_id)
    .bind(branch)
    .bind(file_identity_id)
    .bind(content_key)
    .bind(is_virtual as i32)
    .bind(state)
    .bind(file_type)
    .bind(language)
    .bind(file_mtime)
    .bind(file_hash)
    .bind(chunk_count)
    .bind(chunking_method)
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(collection)
    .bind(extension)
    .bind(is_test as i32)
    .bind(base_point)
    .bind(component)
    .bind(relative_path)
    .bind(needs_reconcile as i32)
    .bind(reconcile_reason)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok(result.last_insert_rowid())
}

/// Update an existing tracked file record
pub async fn update_tracked_file(
    pool: &SqlitePool,
    file_id: i64,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    base_point: Option<&str>,
    component: Option<&str>,
) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET file_mtime = ?1, file_hash = ?2, chunk_count = ?3,
         chunking_method = ?4, lsp_status = ?5, treesitter_status = ?6,
         base_point = ?7, component = ?8, last_error = NULL, updated_at = ?9
         WHERE file_id = ?10",
    )
    .bind(file_mtime)
    .bind(file_hash)
    .bind(chunk_count)
    .bind(chunking_method)
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(base_point)
    .bind(component)
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;

    Ok(())
}

/// Delete a tracked file by file_id (CASCADE deletes qdrant_chunks)
pub async fn delete_tracked_file(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Maximum chunks per batch insert. With 9 params per chunk, 100 * 9 = 900,
/// safely under SQLite's default SQLITE_MAX_VARIABLE_NUMBER (999).
pub(crate) const CHUNK_INSERT_BATCH_SIZE: usize = 100;

/// Insert qdrant_chunks for a file using batched multi-row INSERT.
///
/// Reduces round-trips from O(n) to O(n/BATCH_SIZE). A file with 1000 chunks
/// executes 10 batched queries instead of 1000 individual inserts.
pub async fn insert_qdrant_chunks(
    pool: &SqlitePool,
    file_id: i64,
    chunks: &[(
        String,
        i32,
        String,
        Option<ChunkType>,
        Option<String>,
        Option<i32>,
        Option<i32>,
    )],
    // Each tuple: (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line)
) -> Result<(), sqlx::Error> {
    if chunks.is_empty() {
        return Ok(());
    }
    let now = timestamps::now_utc();
    let mut tx = pool.begin().await?;
    for batch in chunks.chunks(CHUNK_INSERT_BATCH_SIZE) {
        execute_chunk_batch_insert(&mut tx, file_id, batch, &now).await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Execute a single batched INSERT for a slice of chunks within an existing transaction.
///
/// Builds a dynamic multi-row VALUES clause to insert all chunks in one query.
pub(crate) async fn execute_chunk_batch_insert(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    file_id: i64,
    batch: &[(
        String,
        i32,
        String,
        Option<ChunkType>,
        Option<String>,
        Option<i32>,
        Option<i32>,
    )],
    now: &str,
) -> Result<(), sqlx::Error> {
    // Build "VALUES (?,?,?,?,?,?,?,?,?), (?,?,?,?,?,?,?,?,?), ..." with 9 params per row
    let placeholders: Vec<String> = (0..batch.len())
        .map(|i| {
            let base = i * 9 + 1;
            format!(
                "(?{}, ?{}, ?{}, ?{}, ?{}, ?{}, ?{}, ?{}, ?{})",
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                base + 8
            )
        })
        .collect();

    let sql = format!(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, \
         chunk_type, symbol_name, start_line, end_line, created_at) VALUES {}",
        placeholders.join(", ")
    );

    let mut query = sqlx::query(&sql);
    for (point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line) in
        batch
    {
        query = query
            .bind(file_id)
            .bind(point_id)
            .bind(chunk_index)
            .bind(content_hash)
            .bind(chunk_type.as_ref().map(|ct| ct.to_string()))
            .bind(symbol_name.as_deref())
            .bind(start_line)
            .bind(end_line)
            .bind(now);
    }
    query.execute(&mut **tx).await?;
    Ok(())
}

/// Delete all qdrant_chunks for a file_id
pub async fn delete_qdrant_chunks(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM qdrant_chunks WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Get all point_ids for a tracked file's chunks
pub async fn get_chunk_point_ids(
    pool: &SqlitePool,
    file_id: i64,
) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query("SELECT point_id FROM qdrant_chunks WHERE file_id = ?1")
        .bind(file_id)
        .fetch_all(pool)
        .await?;

    Ok(rows.iter().map(|r| r.get("point_id")).collect())
}

/// Check if a file has the incremental (do-not-delete) flag set.
///
/// Lookup keyed by absolute filesystem path: the row is matched by joining
/// `watch_folders.path || '/' || tracked_files.relative_path` against the
/// supplied absolute path. Returns true if at least one match exists with
/// `incremental = 1`.
pub async fn is_incremental(pool: &SqlitePool, abs_file_path: &str) -> Result<bool, sqlx::Error> {
    let result: Option<i32> = sqlx::query_scalar(
        "SELECT tf.incremental \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.path || '/' || tf.relative_path = ?1 \
         LIMIT 1",
    )
    .bind(abs_file_path)
    .fetch_optional(pool)
    .await?;

    Ok(result.unwrap_or(0) != 0)
}

/// Set the incremental (do-not-delete) flag on a tracked file by absolute path.
pub async fn set_incremental(
    pool: &SqlitePool,
    abs_file_path: &str,
    incremental: bool,
) -> Result<u64, sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();
    let result = sqlx::query(
        "UPDATE tracked_files \
         SET incremental = ?1, updated_at = ?2 \
         WHERE file_id IN ( \
             SELECT tf.file_id FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.path || '/' || tf.relative_path = ?3 \
         )",
    )
    .bind(if incremental { 1i32 } else { 0i32 })
    .bind(&now)
    .bind(abs_file_path)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Get all tracked file paths for a watch_folder (for cleanup/recovery).
///
/// Returns `(file_id, relative_path, branch)`. v48: `branch` is the scalar
/// NOT-NULL branch column (was the v40 `primary_branch`).
pub async fn get_tracked_file_paths(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<Vec<(i64, String, String)>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT file_id, relative_path, branch FROM tracked_files WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| (r.get("file_id"), r.get("relative_path"), r.get("branch")))
        .collect())
}

/// Get `(relative_path, file_hash, file_mtime)` for every tracked file in a
/// watch_folder.
///
/// Used by startup recovery to detect content changes that happened while the
/// daemon was not running (the live watcher only sees events while up). The
/// stored `file_mtime` lets recovery gate the expensive content re-hash on a
/// cheap mtime comparison, so unchanged files are not read from disk on every
/// boot.
pub async fn get_tracked_files_with_hashes(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<Vec<(String, String, String)>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT relative_path, file_hash, file_mtime FROM tracked_files WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| {
            (
                r.get("relative_path"),
                r.get("file_hash"),
                r.get("file_mtime"),
            )
        })
        .collect())
}

/// Get tracked files under a folder path prefix for a watch_folder.
///
/// Used for folder-level delete/move operations. Matches files whose
/// relative path starts with `folder_prefix/` (ensuring proper directory
/// boundary). Returns `(file_id, relative_path, branch)` (v48 scalar `branch`).
pub async fn get_tracked_files_by_prefix(
    pool: &SqlitePool,
    watch_folder_id: &str,
    folder_prefix: &str,
) -> Result<Vec<(i64, String, String)>, sqlx::Error> {
    // Ensure prefix ends with '/' for proper directory boundary matching
    let prefix = if folder_prefix.ends_with('/') {
        folder_prefix.to_string()
    } else {
        format!("{}/", folder_prefix)
    };

    let rows = sqlx::query(
        "SELECT file_id, relative_path, branch FROM tracked_files
         WHERE watch_folder_id = ?1 AND (relative_path LIKE ?2 OR relative_path = ?3)",
    )
    .bind(watch_folder_id)
    .bind(format!("{}%", prefix))
    .bind(folder_prefix) // Also match the exact path (in case it's a file, not a dir)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| (r.get("file_id"), r.get("relative_path"), r.get("branch")))
        .collect())
}

/// Compute relative path from absolute file path and watch folder base path
pub fn compute_relative_path(abs_path: &str, base_path: &str) -> Option<String> {
    let abs = Path::new(abs_path);
    let base = Path::new(base_path);
    abs.strip_prefix(base)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Branch-lineage F5: tenant-wide byte-identical locator (v48).
//
// New, self-contained, v48-correct. This locator is the Case-2 "copy-vector"
// probe of arch §5.1: a tenant-wide hash lookup spanning projects+libraries,
// used by the F6 tagger to dedup compute across distinct file-identities /
// collections by copying the vector of an existing byte-identical real point
// instead of re-embedding.
// ---------------------------------------------------------------------------

/// A byte-identical existing view row located tenant-wide by `file_hash`.
///
/// Purpose-built for the Case-2 copy-vector path (arch §5.1): the caller (the F6
/// tagger) needs to find the REAL Qdrant point holding the byte-identical
/// content's vector and copy it. The point to copy from is derivable from
/// `content_key` alone via [`real_point_id_for`] (see its doc), so no Qdrant
/// read is needed in the LOCATE step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteIdenticalHit {
    /// The located row's content identity. `point_id(content_key, chunk_index)`
    /// addresses the real point's chunks regardless of whether the row is real
    /// or virtual (a virtual row shares its real point's `content_key`).
    pub content_key: String,
    /// The Qdrant collection the real point lives in (the row's `collection`).
    pub collection: String,
    /// Whether the located row is itself a virtual/view row.
    pub is_virtual: bool,
    /// The located row's `file_id` (for diagnostics / follow-up tracker reads).
    pub file_id: i64,
}

/// Locate a tenant-wide byte-identical existing row by `file_hash`, spanning
/// projects and libraries (arch §5.1 Case-2 locator, F5).
///
/// Runs `WHERE tenant_id = ?1 AND file_hash = ?2 ORDER BY created_at ASC LIMIT 1`
/// over the `idx_tracked_files_file_hash` index — a state.db indexed probe, NOT
/// a Qdrant scan. The index is NOT unique (multiple identities/collections can
/// hold identical bytes under one tenant), so `ORDER BY created_at ASC LIMIT 1`
/// (oldest wins) makes the selection deterministic; correctness does not depend
/// on which row is chosen, since all share the bytes.
///
/// Returns `Ok(None)` when no row under the tenant has those bytes.
///
/// To get the real point id to copy the vector from, the caller derives it from
/// the located `content_key` with [`real_point_id_for`] — this holds whether the
/// located row is real or virtual, because a virtual row shares the real point's
/// `content_key` and `point_id` is a pure function of `(content_key, chunk_index)`.
pub async fn locate_byte_identical(
    pool: &SqlitePool,
    tenant_id: &str,
    file_hash: &str,
) -> Result<Option<ByteIdenticalHit>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT file_id, content_key, collection, is_virtual \
         FROM tracked_files \
         WHERE tenant_id = ?1 AND file_hash = ?2 \
         ORDER BY created_at ASC \
         LIMIT 1",
    )
    .bind(tenant_id)
    .bind(file_hash)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| ByteIdenticalHit {
        content_key: r.get("content_key"),
        collection: r.get("collection"),
        is_virtual: r.get::<i64, _>("is_virtual") != 0,
        file_id: r.get("file_id"),
    }))
}

/// The real point id for chunk `chunk_index` of `content_key`.
///
/// Settled per arch §5.1 / §4.3: `point_id = point_id(content_key, chunk_index)`
/// is a pure function of `content_key` alone, and a virtual point shares the
/// `content_key` of the real point it shadows. So the real point id is
/// derivable from a located row's `content_key` whether that row is real or
/// virtual — the LOCATE step stays state.db-only (no Qdrant read to "follow"
/// a `real_point_id` link). The arch's Qdrant-payload `real_point_id` link is an
/// equivalent alternative the tagger MAY use, but is not required here.
pub fn real_point_id_for(content_key: &str, chunk_index: u32) -> uuid::Uuid {
    wqm_common::hashing::point_id(content_key, chunk_index)
}

#[cfg(test)]
mod f5_locator_tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    const TENANT: &str = "tenant1";

    /// In-memory state.db migrated to v48 (SchemaManager runs all migrations).
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
        // Two watch folders, in two collections, to prove the locator spans both.
        insert_watch_folder(&pool, "wf_proj", "projects").await;
        insert_watch_folder(&pool, "wf_lib", "libraries").await;
        pool
    }

    async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, collection: &str) {
        sqlx::query(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES (?1, '/tmp/' || ?1, ?2, ?3, '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')",
        )
        .bind(watch_id)
        .bind(collection)
        .bind(TENANT)
        .execute(pool)
        .await
        .unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_row(
        pool: &SqlitePool,
        watch_id: &str,
        collection: &str,
        content_key: &str,
        file_hash: &str,
        is_virtual: bool,
        relative_path: &str,
        created_at: &str,
    ) {
        sqlx::query(
            "INSERT INTO tracked_files \
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key, \
              is_virtual, state, file_mtime, file_hash, relative_path, collection, \
              created_at, updated_at) \
             VALUES (?1, ?2, 'main', ?3, ?4, ?5, 'present', ?6, ?7, ?8, ?9, ?6, ?6)",
        )
        .bind(watch_id)
        .bind(TENANT)
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(content_key)
        .bind(if is_virtual { 1i64 } else { 0i64 })
        .bind(created_at)
        .bind(file_hash)
        .bind(relative_path)
        .bind(collection)
        .execute(pool)
        .await
        .unwrap();
    }

    /// T-F5-locator-oldest-deterministic: several rows share one
    /// `(tenant_id, file_hash)` at differing `created_at` (and across the
    /// projects + libraries collections); the locator returns the OLDEST.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f5_locator_oldest_deterministic() {
        let pool = v48_pool().await;
        let hash = "samebytes";

        // Oldest: a real point in projects.
        insert_row(
            &pool,
            "wf_proj",
            "projects",
            "ck_oldest",
            hash,
            false,
            "src/a.rs",
            "2025-01-01T00:00:00.000Z",
        )
        .await;
        // Newer: a real point in libraries (different identity / collection).
        insert_row(
            &pool,
            "wf_lib",
            "libraries",
            "ck_newer",
            hash,
            false,
            "doc/b.md",
            "2025-02-01T00:00:00.000Z",
        )
        .await;
        // Newest: a virtual view row.
        insert_row(
            &pool,
            "wf_proj",
            "projects",
            "ck_newest",
            hash,
            true,
            "src/c.rs",
            "2025-03-01T00:00:00.000Z",
        )
        .await;

        let hit = locate_byte_identical(&pool, TENANT, hash)
            .await
            .unwrap()
            .expect("a byte-identical row must be located");

        assert_eq!(
            hit.content_key, "ck_oldest",
            "the oldest row (by created_at ASC) must win"
        );
        assert_eq!(hit.collection, "projects");
        assert!(!hit.is_virtual);
    }

    /// The locator spans collections: a libraries-only byte-identical row is
    /// found even when no projects row matches (arch §5.1 axis C).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f5_locator_spans_collections() {
        let pool = v48_pool().await;
        let hash = "libonly";
        insert_row(
            &pool,
            "wf_lib",
            "libraries",
            "ck_lib",
            hash,
            false,
            "doc/x.md",
            "2025-01-01T00:00:00.000Z",
        )
        .await;

        let hit = locate_byte_identical(&pool, TENANT, hash)
            .await
            .unwrap()
            .expect("a libraries row must be located tenant-wide");
        assert_eq!(hit.collection, "libraries");
        assert_eq!(hit.content_key, "ck_lib");
    }

    /// No row under the tenant has those bytes → `Ok(None)`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f5_locator_none_on_miss() {
        let pool = v48_pool().await;
        let hit = locate_byte_identical(&pool, TENANT, "absent")
            .await
            .unwrap();
        assert!(hit.is_none(), "an absent hash must locate nothing");
    }

    /// `real_point_id_for` derives the same id `point_id(content_key, n)` would,
    /// for both real and virtual located rows (they share `content_key`).
    #[test]
    fn t_f5_real_point_id_derivation() {
        let ck = "ck_oldest";
        assert_eq!(
            real_point_id_for(ck, 0),
            wqm_common::hashing::point_id(ck, 0),
            "derivation must equal the canonical point_id helper"
        );
        assert_ne!(
            real_point_id_for(ck, 0),
            real_point_id_for(ck, 1),
            "different chunk indices must derive different point ids"
        );
    }

    /// T-F6b: `insert_tracked_file_v48` writes all branch-lineage v48 columns
    /// and returns the AUTOINCREMENT file_id; the inserted row is then locatable
    /// by the F5 byte-identical probe (the tagger's Case-2 path).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6b_insert_tracked_file_v48_real() {
        let pool = v48_pool().await;
        let fid_uuid = uuid::Uuid::new_v4().to_string();

        let file_id = insert_tracked_file_v48(
            &pool,
            "wf_proj",
            TENANT,
            "main",
            &fid_uuid,
            "ck_real",
            false,        // is_virtual
            "present",    // state
            Some("code"), // file_type
            Some("rust"), // language
            "2025-01-01T00:00:00.000Z",
            "hash_real",
            3, // chunk_count
            Some("tree_sitter"),
            ProcessingStatus::None,
            ProcessingStatus::None,
            "projects",    // collection
            Some("rs"),    // extension
            false,         // is_test
            Some("bp1"),   // base_point
            Some("core"),  // component
            "src/main.rs", // relative_path
            true,          // needs_reconcile (Case 2/3 crash-safety)
            Some("additive_crash"),
        )
        .await
        .expect("insert_tracked_file_v48 must succeed against v48");
        assert!(file_id > 0, "AUTOINCREMENT file_id must be positive");

        let row = sqlx::query(
            "SELECT branch, file_identity_id, content_key, is_virtual, state, \
             chunk_count, collection, component, relative_path, needs_reconcile, \
             reconcile_reason FROM tracked_files WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.get::<String, _>("branch"), "main");
        assert_eq!(row.get::<String, _>("file_identity_id"), fid_uuid);
        assert_eq!(row.get::<String, _>("content_key"), "ck_real");
        assert_eq!(row.get::<i64, _>("is_virtual"), 0);
        assert_eq!(row.get::<String, _>("state"), "present");
        assert_eq!(row.get::<i64, _>("chunk_count"), 3);
        assert_eq!(row.get::<String, _>("collection"), "projects");
        assert_eq!(row.get::<String, _>("component"), "core");
        assert_eq!(row.get::<String, _>("relative_path"), "src/main.rs");
        assert_eq!(row.get::<i64, _>("needs_reconcile"), 1);
        assert_eq!(row.get::<String, _>("reconcile_reason"), "additive_crash");

        // Integration with the F5 locator (Case-2 copy-vector probe).
        let hit = locate_byte_identical(&pool, TENANT, "hash_real")
            .await
            .unwrap()
            .expect("the inserted row must be byte-locatable");
        assert_eq!(hit.content_key, "ck_real");
        assert!(!hit.is_virtual);
    }

    /// T-F6b-virtual: a virtual write stores `is_virtual=1`, `needs_reconcile=0`,
    /// NULL reconcile_reason (arch §5.1 Case 1 — no crash-safety flag needed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn t_f6b_insert_tracked_file_v48_virtual() {
        let pool = v48_pool().await;
        let file_id = insert_tracked_file_v48(
            &pool,
            "wf_proj",
            TENANT,
            "feature/x",
            &uuid::Uuid::new_v4().to_string(),
            "ck_virt",
            true, // is_virtual
            "present",
            None,
            None,
            "2025-01-01T00:00:00.000Z",
            "hash_virt",
            2,
            None,
            ProcessingStatus::None,
            ProcessingStatus::None,
            "projects",
            None,
            false,
            None,
            None,
            "src/v.rs",
            false, // needs_reconcile
            None,
        )
        .await
        .expect("virtual insert must succeed");

        let row = sqlx::query(
            "SELECT is_virtual, needs_reconcile, reconcile_reason \
             FROM tracked_files WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.get::<i64, _>("is_virtual"), 1);
        assert_eq!(row.get::<i64, _>("needs_reconcile"), 0);
        assert!(row.get::<Option<String>, _>("reconcile_reason").is_none());
    }
}
