//! Pool-based database operations for tracked files and Qdrant chunks

use sqlx::{sqlite::SqliteRow, Row, Sqlite, SqlitePool};
use std::path::Path;
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::timestamps;

use super::types::{ChunkType, ProcessingStatus, TrackedFile};

// Re-export hashing functions from wqm-common
pub use wqm_common::hashing::{compute_content_hash, compute_file_hash};

/// Build a TrackedFile from a SQLite row
pub(crate) fn tracked_file_from_row(r: &SqliteRow) -> TrackedFile {
    TrackedFile {
        file_id: r.get("file_id"),
        watch_folder_id: r.get("watch_folder_id"),
        file_path: r.get("file_path"),
        branch: r.get("branch"),
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
        relative_path: r.get("relative_path"),
        incremental: r.get::<Option<i32>, _>("incremental").unwrap_or(0) != 0,
        component: r.get("component"),
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

/// Look up a tracked file by (watch_folder_id, relative_path, branch)
pub async fn lookup_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    file_path: &str,
    branch: Option<&str>,
) -> Result<Option<TrackedFile>, sqlx::Error> {
    let row = match branch {
        Some(b) => {
            sqlx::query(
                "SELECT file_id, watch_folder_id, file_path, branch, file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error,
                        needs_reconcile, reconcile_reason, extension, is_test,
                        collection, base_point, relative_path, incremental,
                        component, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND file_path = ?2 AND branch = ?3",
            )
            .bind(watch_folder_id)
            .bind(file_path)
            .bind(b)
            .fetch_optional(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT file_id, watch_folder_id, file_path, branch, file_type, language,
                        file_mtime, file_hash, chunk_count, chunking_method,
                        lsp_status, treesitter_status, last_error,
                        needs_reconcile, reconcile_reason, extension, is_test,
                        collection, base_point, relative_path, incremental,
                        component, created_at, updated_at
                 FROM tracked_files
                 WHERE watch_folder_id = ?1 AND file_path = ?2 AND branch IS NULL",
            )
            .bind(watch_folder_id)
            .bind(file_path)
            .fetch_optional(pool)
            .await?
        }
    };

    Ok(row.map(|r| tracked_file_from_row(&r)))
}

/// Insert a new tracked file record, returning the file_id
pub async fn insert_tracked_file(
    pool: &SqlitePool,
    watch_folder_id: &str,
    file_path: &str,
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
    relative_path: Option<&str>,
    component: Option<&str>,
) -> Result<i64, sqlx::Error> {
    let now = timestamps::now_utc();
    let collection = collection.unwrap_or(COLLECTION_PROJECTS);
    let result = sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_type, language,
         file_mtime, file_hash, chunk_count, chunking_method, lsp_status, treesitter_status,
         extension, is_test, collection, base_point, relative_path, component, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)"
    )
    .bind(watch_folder_id)
    .bind(file_path)
    .bind(branch)
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
    .bind(relative_path)
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
/// Looks up by absolute file_path across all watch_folders. Returns true if the
/// file exists in tracked_files with `incremental = 1`.
pub async fn is_incremental(pool: &SqlitePool, file_path: &str) -> Result<bool, sqlx::Error> {
    let result: Option<i32> =
        sqlx::query_scalar("SELECT incremental FROM tracked_files WHERE file_path = ?1 LIMIT 1")
            .bind(file_path)
            .fetch_optional(pool)
            .await?;

    Ok(result.unwrap_or(0) != 0)
}

/// Set the incremental (do-not-delete) flag on a tracked file.
pub async fn set_incremental(
    pool: &SqlitePool,
    file_path: &str,
    incremental: bool,
) -> Result<u64, sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();
    let result = sqlx::query(
        "UPDATE tracked_files SET incremental = ?1, updated_at = ?2 WHERE file_path = ?3",
    )
    .bind(if incremental { 1i32 } else { 0i32 })
    .bind(&now)
    .bind(file_path)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Get all tracked file paths for a watch_folder (for cleanup/recovery)
pub async fn get_tracked_file_paths(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<Vec<(i64, String, Option<String>)>, sqlx::Error> {
    // Returns (file_id, file_path, branch)
    let rows = sqlx::query(
        "SELECT file_id, file_path, branch FROM tracked_files WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| (r.get("file_id"), r.get("file_path"), r.get("branch")))
        .collect())
}

/// Get tracked files under a folder path prefix for a watch_folder
///
/// Used for folder-level delete/move operations. Matches files whose relative
/// path starts with `folder_prefix/` (ensuring proper directory boundary).
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
        "SELECT file_id, file_path, branch FROM tracked_files
         WHERE watch_folder_id = ?1 AND (file_path LIKE ?2 OR file_path = ?3)",
    )
    .bind(watch_folder_id)
    .bind(format!("{}%", prefix))
    .bind(folder_prefix) // Also match the exact path (in case it's a file, not a dir)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| (r.get("file_id"), r.get("file_path"), r.get("branch")))
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
