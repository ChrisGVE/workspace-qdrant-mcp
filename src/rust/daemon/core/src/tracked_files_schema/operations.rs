//! Pool-based database operations for tracked files and Qdrant chunks

use sqlx::{sqlite::SqliteRow, Row, Sqlite, SqlitePool};
use std::path::Path;
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::paths::RelativePath;
use wqm_common::timestamps;

use super::types::{ChunkType, ProcessingStatus, TrackedFile};

// Re-export hashing functions from wqm-common
pub use wqm_common::hashing::{compute_content_hash, compute_file_hash};

/// Build a TrackedFile from a SQLite row.
///
/// Post-v37: the row carries a single relative path column (`relative_path`).
/// The legacy absolute `file_path` column was dropped by migration v37 and is
/// no longer hydrated here.
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
        primary_branch: r.get("primary_branch"),
        branches: r
            .get::<Option<String>, _>("branches")
            .unwrap_or_else(|| "[]".to_string()),
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

/// Look up a tracked file by (watch_folder_id, relative_path, branch).
///
/// First checks `primary_branch` for an exact match. If not found, checks
/// `branches` JSON array membership — this handles content-hash deduped
/// files that share a single row across multiple branches.
pub async fn lookup_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    branch: Option<&str>,
) -> Result<Option<TrackedFile>, sqlx::Error> {
    let row = match branch {
        Some(b) => {
            let r = sqlx::query(
                "SELECT file_id, watch_folder_id, relative_path, primary_branch, branches,
                        file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error,
                        needs_reconcile, reconcile_reason, extension, is_test,
                        collection, base_point, incremental,
                        component, routing_reason, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND relative_path = ?2 AND primary_branch = ?3",
            )
            .bind(watch_folder_id)
            .bind(relative_path)
            .bind(b)
            .fetch_optional(pool)
            .await?;
            if r.is_some() {
                r
            } else {
                sqlx::query(
                    "SELECT file_id, watch_folder_id, relative_path, primary_branch, branches,
                            file_type, language,
                            file_mtime, file_hash, chunk_count, chunking_method,
                            lsp_status, treesitter_status, last_error,
                            needs_reconcile, reconcile_reason, extension, is_test,
                            collection, base_point, incremental,
                            component, routing_reason, created_at, updated_at
                     FROM tracked_files
                     WHERE watch_folder_id = ?1 AND relative_path = ?2
                       AND EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = ?3)",
                )
                .bind(watch_folder_id)
                .bind(relative_path)
                .bind(b)
                .fetch_optional(pool)
                .await?
            }
        }
        None => {
            sqlx::query(
                "SELECT file_id, watch_folder_id, relative_path, primary_branch, branches,
                        file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error,
                        needs_reconcile, reconcile_reason, extension, is_test,
                        collection, base_point, incremental,
                        component, routing_reason, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND relative_path = ?2 AND primary_branch IS NULL",
            )
            .bind(watch_folder_id)
            .bind(relative_path)
            .fetch_optional(pool)
            .await?
        }
    };

    Ok(row.map(|r| tracked_file_from_row(&r)))
}

/// Look up a tracked file by content hash for cross-branch deduplication.
///
/// Finds an existing `tracked_files` row for the same `watch_folder_id` and
/// `relative_path` that has the same `file_hash` but a **different**
/// `primary_branch`. This is the key query for content-hash dedup: if a
/// matching row exists, the Qdrant points already contain the embeddings and
/// only the `branches` array needs updating.
///
/// Returns `None` when no cross-branch duplicate exists.
pub async fn lookup_tracked_file_by_hash(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    file_hash: &str,
    exclude_branch: &str,
) -> Result<Option<TrackedFile>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT file_id, watch_folder_id, relative_path, primary_branch, branches,
                file_type, language,
                file_mtime, file_hash, chunk_count, chunking_method,
                lsp_status, treesitter_status, last_error,
                needs_reconcile, reconcile_reason, extension, is_test,
                collection, base_point, incremental,
                component, routing_reason, created_at, updated_at
         FROM tracked_files
         WHERE watch_folder_id = ?1
           AND relative_path = ?2
           AND file_hash = ?3
           AND primary_branch IS NOT NULL
           AND primary_branch != ?4
         LIMIT 1",
    )
    .bind(watch_folder_id)
    .bind(relative_path)
    .bind(file_hash)
    .bind(exclude_branch)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| tracked_file_from_row(&r)))
}

/// Add a branch name to the `branches` JSON array of a tracked file.
///
/// Uses SQLite's `json_insert` to append the branch at the end of the array
/// (`'$[#]'`). The caller must hold the per-tenant branch lock to prevent
/// concurrent read-modify-write races.
///
/// Returns the updated `branches` JSON string after the mutation.
pub async fn add_branch_to_tracked_file(
    pool: &SqlitePool,
    file_id: i64,
    branch: &str,
) -> Result<String, sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files
         SET branches = json_insert(branches, '$[#]', ?1),
             updated_at = ?2
         WHERE file_id = ?3",
    )
    .bind(branch)
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;

    // Fetch the updated branches value
    let row: (String,) =
        sqlx::query_as("SELECT COALESCE(branches, '[]') FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .fetch_one(pool)
            .await?;

    Ok(row.0)
}

/// Insert a new tracked file record, returning the file_id.
///
/// `relative_path` is the post-v37 anchored relative path stored alongside
/// the row. Callers must have validated/normalized the string upstream.
#[allow(clippy::too_many_arguments)]
pub async fn insert_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    branch: Option<&str>,
    file_type: Option<&str>,
    language: Option<&str>,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: i32,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    collection: Option<&str>,
    extension: Option<&str>,
    is_test: bool,
    base_point: Option<&str>,
    component: Option<&str>,
) -> Result<i64, sqlx::Error> {
    let now = timestamps::now_utc();
    let collection = collection.unwrap_or(COLLECTION_PROJECTS);
    // Build the branches JSON array from the branch parameter.
    let branches_json = match branch {
        Some(b) => format!(r#"["{}"]"#, b),
        None => "[]".to_string(),
    };
    let result = sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, relative_path, primary_branch, branches,
         file_type, language,
         file_mtime, file_hash, chunk_count, chunking_method, lsp_status, treesitter_status,
         extension, is_test, collection, base_point, component, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
    )
    .bind(watch_folder_id)
    .bind(relative_path)
    .bind(branch)
    .bind(&branches_json)
    .bind(file_type)
    .bind(language)
    .bind(file_mtime)
    .bind(file_hash)
    .bind(chunk_count)
    .bind(chunking_method)
    .bind(lsp_status.to_string())
    .bind(treesitter_status.to_string())
    .bind(extension)
    .bind(is_test as i32)
    .bind(collection)
    .bind(base_point)
    .bind(component)
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
/// Returns `(file_id, relative_path, primary_branch)`.
pub async fn get_tracked_file_paths(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<Vec<(i64, String, Option<String>)>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT file_id, relative_path, primary_branch FROM tracked_files WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| {
            (
                r.get("file_id"),
                r.get("relative_path"),
                r.get("primary_branch"),
            )
        })
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
/// boundary).
pub async fn get_tracked_files_by_prefix(
    pool: &SqlitePool,
    watch_folder_id: &str,
    folder_prefix: &str,
) -> Result<Vec<(i64, String, Option<String>)>, sqlx::Error> {
    // Ensure prefix ends with '/' for proper directory boundary matching
    let prefix = if folder_prefix.ends_with('/') {
        folder_prefix.to_string()
    } else {
        format!("{}/", folder_prefix)
    };

    let rows = sqlx::query(
        "SELECT file_id, relative_path, primary_branch FROM tracked_files
         WHERE watch_folder_id = ?1 AND (relative_path LIKE ?2 OR relative_path = ?3)",
    )
    .bind(watch_folder_id)
    .bind(format!("{}%", prefix))
    .bind(folder_prefix) // Also match the exact path (in case it's a file, not a dir)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| {
            (
                r.get("file_id"),
                r.get("relative_path"),
                r.get("primary_branch"),
            )
        })
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
