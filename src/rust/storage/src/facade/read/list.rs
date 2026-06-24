//! Branch file listing — `ReadStoreFacade::list_branch` (AC-F10.1, §6.2).
//!
//! File: `wqm-storage/src/facade/read/list.rs`
//! Location: `src/rust/storage/src/facade/read/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch §6.2, §5.2).
//!   Queries the per-project `store.db` to list every file visible on a branch,
//!   with its content hash and chunk count. All parameters are bound (GP-8).
//!
//! Neighbors: `super::mod` (ReadStoreFacade), `crate::types::results::FileEntry`
//!   (output type), `crate::schema::columns` (column name constants).

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::types::results::FileEntry;

/// List every file known to `branch_id` in the per-project `store.db`.
///
/// Returns one `FileEntry` per `(branch_id, relative_path)` pair, with the
/// file's content hash (from `concrete`) and chunk count (from `blob_refs`).
///
/// All SQL values come from bound parameters (arch §6.5 GP-8).
pub async fn list_branch(
    pool: &SqlitePool,
    branch_id: &str,
) -> Result<Vec<FileEntry>, StorageError> {
    let rows = sqlx::query_as::<_, FileRow>(
        r#"
        SELECT
            f.file_id,
            f.relative_path                           AS path,
            COALESCE(c.file_hash, '')                 AS content_hash,
            CAST(COUNT(br.ref_id) AS INTEGER)         AS chunk_count
        FROM files f
        LEFT JOIN concrete c  ON c.file_id  = f.file_id AND c.branch_id = f.branch_id
        LEFT JOIN blob_refs br ON br.file_id = f.file_id AND br.branch_id = f.branch_id
        WHERE f.branch_id = ?1
        GROUP BY f.file_id
        ORDER BY f.relative_path ASC
        "#,
    )
    .bind(branch_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("list_branch failed: {e}")))?;

    Ok(rows.into_iter().map(FileRow::into_entry).collect())
}

// ---------------------------------------------------------------------------
// Internal row type
// ---------------------------------------------------------------------------

#[derive(sqlx::FromRow)]
struct FileRow {
    file_id: i64,
    path: String,
    content_hash: String,
    chunk_count: i64,
}

impl FileRow {
    fn into_entry(self) -> FileEntry {
        FileEntry {
            file_id: self.file_id,
            path: self.path,
            content_hash: self.content_hash,
            chunk_count: self.chunk_count as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "list_tests.rs"]
mod tests;
