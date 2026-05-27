//! Database operations for branch discovery.

use std::collections::HashMap;

use sqlx::SqlitePool;
use tracing::warn;
use wqm_common::timestamps;

/// Known file entry from tracked_files: (relative_path, file_hash) → (file_id, branches, base_point).
pub struct KnownFile {
    pub file_id: i64,
    pub branches: Vec<String>,
    pub base_point: Option<String>,
}

/// Load all known files for a project, keyed by (relative_path, file_hash).
pub async fn load_known_files(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<HashMap<(String, String), KnownFile>, String> {
    let rows: Vec<(i64, String, String, String, Option<String>)> = sqlx::query_as(
        "SELECT file_id, relative_path, COALESCE(file_hash, ''), COALESCE(branches, '[]'), base_point
         FROM tracked_files
         WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to load known files: {}", e))?;

    let mut map = HashMap::with_capacity(rows.len());
    for (file_id, relative_path, file_hash, branches_json, base_point) in rows {
        let branches: Vec<String> = serde_json::from_str(&branches_json).unwrap_or_default();
        map.insert(
            (relative_path, file_hash),
            KnownFile {
                file_id,
                branches,
                base_point,
            },
        );
    }
    Ok(map)
}

/// Batch add a branch to multiple tracked_files rows.
pub async fn batch_add_branch(
    pool: &SqlitePool,
    file_ids: &[i64],
    branch: &str,
) -> Result<u64, String> {
    if file_ids.is_empty() {
        return Ok(0);
    }

    let now = timestamps::now_utc();
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("Failed to begin transaction: {}", e))?;

    let mut updated = 0u64;
    for &file_id in file_ids {
        let result = sqlx::query(
            "UPDATE tracked_files
             SET branches = json_insert(branches, '$[#]', ?1),
                 updated_at = ?2
             WHERE file_id = ?3",
        )
        .bind(branch)
        .bind(&now)
        .bind(file_id)
        .execute(&mut *tx)
        .await;

        match result {
            Ok(r) => updated += r.rows_affected(),
            Err(e) => {
                warn!("Failed to add branch to file_id={}: {}", file_id, e);
            }
        }
    }

    tx.commit()
        .await
        .map_err(|e| format!("Failed to commit batch branch add: {}", e))?;

    Ok(updated)
}

/// Insert file_metadata rows for discovered shared files.
pub async fn batch_insert_file_metadata(
    search_pool: &SqlitePool,
    entries: &[(i64, &str, Option<&str>, Option<&str>, Option<&str>)],
    tenant_id: &str,
    branch: &str,
    watch_root: &str,
) {
    for &(file_id, relative_path, base_point, _rel_path_alias, file_hash) in entries {
        let abs_path = format!("{}/{}", watch_root.trim_end_matches('/'), relative_path);
        let result = sqlx::query(
            "INSERT INTO file_metadata (file_id, tenant_id, branch, file_path, base_point, relative_path, file_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(file_id, branch) DO NOTHING",
        )
        .bind(file_id)
        .bind(tenant_id)
        .bind(branch)
        .bind(&abs_path)
        .bind(base_point)
        .bind(relative_path)
        .bind(file_hash)
        .execute(search_pool)
        .await;

        if let Err(e) = result {
            warn!(
                "Failed to insert file_metadata for file_id={}, branch='{}': {}",
                file_id, branch, e
            );
        }
    }
}
