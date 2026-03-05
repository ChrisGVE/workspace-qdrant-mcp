//! Qdrant-to-SQLite reconstruction logic for state recovery.
//!
//! Converts scrolled Qdrant point payloads into SQLite rows for
//! watch_folders, tracked_files, qdrant_chunks, and rules_mirror.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rusqlite::params;
use uuid::Uuid;

/// Statistics returned from a single collection reconstruction pass.
pub(super) struct ReconstructStats {
    pub watch_folders: u64,
    pub tracked_files: u64,
    pub chunks: u64,
}

/// Reconstruct watch_folders, tracked_files, and qdrant_chunks from
/// projects collection points.
pub(super) fn reconstruct_project_state(
    conn: &rusqlite::Connection,
    points: &[serde_json::Value],
) -> Result<ReconstructStats> {
    let now = wqm_common::timestamps::now_utc();

    let mut tenant_files: BTreeMap<String, Vec<&serde_json::Value>> = BTreeMap::new();
    for point in points {
        let tenant_id = point["payload"]["tenant_id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        tenant_files.entry(tenant_id).or_default().push(point);
    }

    let mut stats = ReconstructStats {
        watch_folders: 0,
        tracked_files: 0,
        chunks: 0,
    };

    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    for (tenant_id, points) in &tenant_files {
        let project_root = infer_project_root(points);
        let watch_id = Uuid::new_v4().to_string();
        tx.execute(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, is_active, enabled, created_at, updated_at) \
             VALUES (?1, ?2, 'projects', ?3, 0, 1, ?4, ?5)",
            params![watch_id, project_root, tenant_id, now, now],
        )
        .context("Failed to insert watch_folder")?;
        stats.watch_folders += 1;

        let file_groups = group_points_by_file(points);
        for ((file_path, branch), file_points) in &file_groups {
            let file_id =
                insert_project_tracked_file(&tx, &watch_id, file_path, branch, file_points, &now)?;
            if let Some(fid) = file_id {
                stats.tracked_files += 1;
                stats.chunks += insert_qdrant_chunks_with_metadata(&tx, fid, file_points, &now)?;
            }
        }
    }

    tx.commit()
        .context("Failed to commit project reconstruction")?;
    Ok(stats)
}

/// Reconstruct watch_folders, tracked_files, and qdrant_chunks from
/// libraries collection points.
pub(super) fn reconstruct_library_state(
    conn: &rusqlite::Connection,
    points: &[serde_json::Value],
) -> Result<ReconstructStats> {
    let now = wqm_common::timestamps::now_utc();

    // Group by library_name (tenant field for libraries)
    let mut library_groups: BTreeMap<String, Vec<&serde_json::Value>> = BTreeMap::new();
    for point in points {
        let library_name = point["payload"]["library_name"]
            .as_str()
            .or_else(|| point["payload"]["tenant_id"].as_str())
            .unwrap_or("unknown")
            .to_string();
        library_groups.entry(library_name).or_default().push(point);
    }

    let mut watch_folders_created = 0u64;
    let mut tracked_files_created = 0u64;
    let mut chunks_created = 0u64;

    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    for (library_name, points) in &library_groups {
        let watch_id = Uuid::new_v4().to_string();

        // Libraries don't have file paths — use a placeholder directory
        let lib_path = format!("/recovered-libraries/{}", library_name);

        tx.execute(
            "INSERT OR IGNORE INTO watch_folders \
             (watch_id, path, collection, tenant_id, library_mode, enabled, created_at, updated_at) \
             VALUES (?1, ?2, 'libraries', ?3, 'sync', 1, ?4, ?5)",
            params![watch_id, lib_path, library_name, now, now],
        )
        .context("Failed to insert library watch_folder")?;
        watch_folders_created += 1;

        // Group by document_id for tracked files
        let mut doc_groups: BTreeMap<String, Vec<&serde_json::Value>> = BTreeMap::new();
        for point in points {
            let doc_id = point["payload"]["document_id"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            doc_groups.entry(doc_id).or_default().push(point);
        }

        for (doc_id, doc_points) in &doc_groups {
            let first = doc_points[0];
            let file_path = first["payload"]["file_path"]
                .as_str()
                .or_else(|| first["payload"]["source_url"].as_str())
                .unwrap_or(doc_id)
                .to_string();
            let file_hash = first["payload"]["file_hash"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let branch = first["payload"]["branch"]
                .as_str()
                .unwrap_or("main")
                .to_string();

            let result = tx
                .execute(
                    "INSERT OR IGNORE INTO tracked_files \
                 (watch_folder_id, file_path, branch, file_mtime, file_hash, chunk_count, \
                  collection, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'libraries', ?7, ?8)",
                    params![
                        watch_id,
                        file_path,
                        branch,
                        now,
                        file_hash,
                        doc_points.len() as i64,
                        now,
                        now
                    ],
                )
                .context("Failed to insert library tracked_file")?;

            if result == 0 {
                continue;
            }

            let file_id = tx.last_insert_rowid();
            tracked_files_created += 1;

            for point in doc_points {
                let Some(point_id_str) = extract_point_id(point) else {
                    continue;
                };
                let chunk_index = point["payload"]["chunk_index"].as_u64().unwrap_or(0) as i64;
                let content = point["payload"]["content"].as_str().unwrap_or("");
                let content_hash = wqm_common::hashing::compute_content_hash(content);

                tx.execute(
                    "INSERT OR IGNORE INTO qdrant_chunks \
                     (file_id, point_id, chunk_index, content_hash, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![file_id, point_id_str, chunk_index, &content_hash[..32], now],
                )
                .context("Failed to insert library qdrant_chunk")?;
                chunks_created += 1;
            }
        }
    }

    tx.commit()
        .context("Failed to commit library reconstruction")?;

    Ok(ReconstructStats {
        watch_folders: watch_folders_created,
        tracked_files: tracked_files_created,
        chunks: chunks_created,
    })
}

/// Reconstruct rules_mirror from rules collection points. Returns the number of rows inserted.
pub(super) fn reconstruct_rules_state(
    conn: &rusqlite::Connection,
    points: &[serde_json::Value],
) -> Result<u64> {
    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    let mut count = 0u64;

    for point in points {
        let Some(point_id) = extract_point_id(point) else {
            continue;
        };

        let content = point["payload"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let scope = point["payload"]["scope"].as_str().map(|s| s.to_string());
        let tenant_id = point["payload"]["tenant_id"]
            .as_str()
            .or_else(|| point["payload"]["project_id"].as_str())
            .map(|s| s.to_string());
        let label = point["payload"]["label"].as_str().map(|s| s.to_string());
        let created_at = point["payload"]["created_at"]
            .as_str()
            .unwrap_or("2025-01-01T00:00:00Z")
            .to_string();
        let updated_at = point["payload"]["updated_at"]
            .as_str()
            .unwrap_or(&created_at)
            .to_string();

        // Prefer label as rule_id for stable identity; fall back to point_id
        let rule_id = label.as_deref().unwrap_or(&point_id);

        tx.execute(
            "INSERT OR IGNORE INTO rules_mirror \
             (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![rule_id, content, scope, tenant_id, created_at, updated_at],
        )
        .context("Failed to insert rules_mirror")?;
        count += 1;
    }

    tx.commit()
        .context("Failed to commit rules reconstruction")?;

    Ok(count)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Group points by (file_path, branch) for tracked file reconstruction.
fn group_points_by_file<'a>(
    points: &[&'a serde_json::Value],
) -> BTreeMap<(String, String), Vec<&'a serde_json::Value>> {
    let mut groups: BTreeMap<(String, String), Vec<&serde_json::Value>> = BTreeMap::new();
    for point in points {
        let file_path = point["payload"]["file_path"]
            .as_str()
            .or_else(|| point["payload"]["absolute_path"].as_str())
            .unwrap_or("")
            .to_string();
        let branch = point["payload"]["branch"]
            .as_str()
            .unwrap_or("main")
            .to_string();
        if !file_path.is_empty() {
            groups.entry((file_path, branch)).or_default().push(point);
        }
    }
    groups
}

/// Insert a tracked_file row for a project file.
/// Returns `Some(file_id)` when the row was inserted, `None` for duplicates.
fn insert_project_tracked_file(
    tx: &rusqlite::Transaction<'_>,
    watch_id: &str,
    file_path: &str,
    branch: &str,
    file_points: &[&serde_json::Value],
    now: &str,
) -> Result<Option<i64>> {
    let first = file_points[0];
    let file_hash = first["payload"]["file_hash"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let language = first["payload"]["language"].as_str().map(|s| s.to_string());
    let file_type = first["payload"]["file_type"]
        .as_str()
        .map(|s| s.to_string());
    let base_point = first["payload"]["base_point"]
        .as_str()
        .map(|s| s.to_string());
    let relative_path = first["payload"]["relative_path"]
        .as_str()
        .map(|s| s.to_string());
    let extension = first["payload"]["file_extension"]
        .as_str()
        .map(|s| s.to_string());

    let result = tx
        .execute(
            "INSERT OR IGNORE INTO tracked_files \
             (watch_folder_id, file_path, branch, file_type, language, file_mtime, file_hash, \
              chunk_count, collection, base_point, relative_path, extension, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'projects', ?9, ?10, ?11, ?12, ?13)",
            params![
                watch_id,
                file_path,
                branch,
                file_type,
                language,
                now,
                file_hash,
                file_points.len() as i64,
                base_point,
                relative_path,
                extension,
                now,
                now
            ],
        )
        .context("Failed to insert tracked_file")?;

    Ok(if result > 0 {
        Some(tx.last_insert_rowid())
    } else {
        None
    })
}

/// Insert qdrant_chunks with full metadata (chunk_type, symbol, line range).
/// Returns the number of chunks inserted.
fn insert_qdrant_chunks_with_metadata(
    tx: &rusqlite::Transaction<'_>,
    file_id: i64,
    file_points: &[&serde_json::Value],
    now: &str,
) -> Result<u64> {
    let mut count = 0u64;
    for point in file_points {
        let Some(point_id_str) = extract_point_id(point) else {
            continue;
        };
        let chunk_index = point["payload"]["chunk_index"].as_u64().unwrap_or(0) as i64;
        let content = point["payload"]["content"].as_str().unwrap_or("");
        let content_hash = wqm_common::hashing::compute_content_hash(content);
        let chunk_type = point["payload"]["chunk_type"]
            .as_str()
            .map(|s| s.to_string());
        let symbol_name = point["payload"]["chunk_symbol_name"]
            .as_str()
            .map(|s| s.to_string());

        tx.execute(
            "INSERT OR IGNORE INTO qdrant_chunks \
             (file_id, point_id, chunk_index, content_hash, chunk_type, symbol_name, \
              start_line, end_line, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                file_id,
                point_id_str,
                chunk_index,
                &content_hash[..32],
                chunk_type,
                symbol_name,
                point["payload"]["chunk_start_line"].as_i64(),
                point["payload"]["chunk_end_line"].as_i64(),
                now
            ],
        )
        .context("Failed to insert qdrant_chunk")?;
        count += 1;
    }
    Ok(count)
}

/// Extract the Qdrant point ID as a string from a point JSON value.
fn extract_point_id(point: &serde_json::Value) -> Option<String> {
    point["id"]
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| point["id"].as_u64().map(|n| n.to_string()))
}

/// Infer the project root directory from absolute paths in Qdrant points.
///
/// Collects all `absolute_path` (or `file_path`) values and returns their
/// longest common ancestor directory.
pub(super) fn infer_project_root(points: &[&serde_json::Value]) -> String {
    let paths: Vec<&str> = points
        .iter()
        .filter_map(|p| {
            p["payload"]["absolute_path"]
                .as_str()
                .or_else(|| p["payload"]["file_path"].as_str())
        })
        .collect();

    if paths.is_empty() {
        return "/unknown-project".to_string();
    }

    if paths.len() == 1 {
        return PathBuf::from(paths[0])
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/unknown-project".to_string());
    }

    let first = PathBuf::from(paths[0]);
    let components: Vec<_> = first.components().collect();
    let mut common_len = components.len();

    for path in &paths[1..] {
        let p = PathBuf::from(path);
        let p_components: Vec<_> = p.components().collect();
        let mut match_len = 0;
        for (a, b) in components.iter().zip(p_components.iter()) {
            if a == b {
                match_len += 1;
            } else {
                break;
            }
        }
        common_len = common_len.min(match_len);
    }

    if common_len == 0 {
        return "/unknown-project".to_string();
    }

    let common_path: PathBuf = components[..common_len].iter().collect();
    common_path.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_project_root_single_file() {
        let point = serde_json::json!({
            "payload": {
                "absolute_path": "/home/user/project/src/main.rs"
            }
        });
        let result = infer_project_root(&[&point]);
        assert_eq!(result, "/home/user/project/src");
    }

    #[test]
    fn test_infer_project_root_multiple_files() {
        let p1 = serde_json::json!({
            "payload": { "absolute_path": "/home/user/project/src/main.rs" }
        });
        let p2 = serde_json::json!({
            "payload": { "absolute_path": "/home/user/project/src/lib.rs" }
        });
        let p3 = serde_json::json!({
            "payload": { "absolute_path": "/home/user/project/tests/test.rs" }
        });
        let result = infer_project_root(&[&p1, &p2, &p3]);
        assert_eq!(result, "/home/user/project");
    }

    #[test]
    fn test_infer_project_root_empty() {
        let result = infer_project_root(&[]);
        assert_eq!(result, "/unknown-project");
    }

    #[test]
    fn test_infer_project_root_no_common() {
        let p1 = serde_json::json!({
            "payload": { "absolute_path": "/home/user/a/file.rs" }
        });
        let p2 = serde_json::json!({
            "payload": { "absolute_path": "/opt/other/b/file.rs" }
        });
        let result = infer_project_root(&[&p1, &p2]);
        assert_eq!(result, "/");
    }
}
