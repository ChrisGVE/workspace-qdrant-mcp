//! Transaction-aware write operations for tracked files and Qdrant chunks

use sqlx::Sqlite;
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::timestamps;

use super::operations::{execute_chunk_batch_insert, CHUNK_INSERT_BATCH_SIZE};
use super::types::{ChunkType, ProcessingStatus};

/// Insert a new tracked file record within a transaction, returning the file_id
pub async fn insert_tracked_file_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
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
    .execute(&mut **tx)
    .await?;

    Ok(result.last_insert_rowid())
}

/// Update an existing tracked file record within a transaction
pub async fn update_tracked_file_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
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
         base_point = ?7, component = ?8, last_error = NULL, needs_reconcile = 0, reconcile_reason = NULL, updated_at = ?9
         WHERE file_id = ?10"
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
    .execute(&mut **tx)
    .await?;

    Ok(())
}

/// Delete a tracked file by file_id within a transaction (CASCADE deletes qdrant_chunks)
pub async fn delete_tracked_file_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    file_id: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

/// Insert qdrant_chunks for a file within a transaction using batched multi-row INSERT.
pub async fn insert_qdrant_chunks_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
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
) -> Result<(), sqlx::Error> {
    if chunks.is_empty() {
        return Ok(());
    }
    let now = timestamps::now_utc();
    for batch in chunks.chunks(CHUNK_INSERT_BATCH_SIZE) {
        execute_chunk_batch_insert(tx, file_id, batch, &now).await?;
    }
    Ok(())
}

/// Delete all qdrant_chunks for a file_id within a transaction
pub async fn delete_qdrant_chunks_tx(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    file_id: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM qdrant_chunks WHERE file_id = ?1")
        .bind(file_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
