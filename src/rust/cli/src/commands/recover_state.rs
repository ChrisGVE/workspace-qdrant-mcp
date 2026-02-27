//! Recover state.db from Qdrant collections
//!
//! Scrolls all 4 canonical Qdrant collections and reconstructs:
//! - watch_folders: inferred from unique tenant_id + absolute_path prefixes
//! - tracked_files: one row per unique (tenant_id, file_path, branch)
//! - qdrant_chunks: one row per Qdrant point (for file-type points)
//! - rules_mirror: reconstructed from rules collection points

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::params;
use uuid::Uuid;

use crate::output;
use super::qdrant_helpers;
use wqm_common::constants::{
    COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

/// All 4 canonical collections
const ALL_COLLECTIONS: &[&str] = &[
    COLLECTION_PROJECTS,
    COLLECTION_LIBRARIES,
    COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
];

/// Execute recover-state command
pub async fn execute(confirm: bool) -> Result<()> {
    output::section("State Recovery from Qdrant");

    if !confirm {
        output::info("This will rebuild state.db from Qdrant point payloads.");
        output::info("Existing state.db will be backed up to state.db.bak.");
        output::warning("Sparse vocabulary and keyword/tag data cannot be recovered.");
        output::info("They will be rebuilt by the daemon on restart.");
        println!();
        output::info("Run with --confirm to proceed.");
        return Ok(());
    }

    let db_path = crate::config::get_database_path()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Step 1: Backup existing database
    let bak_path = db_path.with_extension("db.bak");
    if db_path.exists() {
        std::fs::copy(&db_path, &bak_path)
            .context("Failed to backup state.db")?;
        output::success(format!("Backed up to {}", bak_path.display()));
        std::fs::remove_file(&db_path)
            .context("Failed to remove old state.db")?;
    }

    // Step 2: Create fresh database with full schema
    let conn = create_fresh_database(&db_path)?;
    output::success("Created fresh state.db with full schema");

    // Step 3: Connect to Qdrant
    let http_client = qdrant_helpers::build_qdrant_http_client()?;
    let base_url = qdrant_helpers::qdrant_base_url();
    output::kv("Qdrant URL", &base_url);
    output::separator();

    // Step 4: Scroll each collection and reconstruct
    let mut total_points = 0u64;
    let mut total_watch_folders = 0u64;
    let mut total_tracked_files = 0u64;
    let mut total_chunks = 0u64;
    let mut total_rules = 0u64;

    for collection in ALL_COLLECTIONS {
        output::info(format!("Scrolling {}...", collection));
        let points = qdrant_helpers::scroll_all_points(
            &http_client, &base_url, collection,
        ).await?;

        let count = points.len();
        total_points += count as u64;
        output::kv(&format!("  {} points", collection), &count.to_string());

        if points.is_empty() {
            continue;
        }

        match *collection {
            c if c == COLLECTION_PROJECTS => {
                let stats = reconstruct_project_state(&conn, &points)?;
                total_watch_folders += stats.watch_folders;
                total_tracked_files += stats.tracked_files;
                total_chunks += stats.chunks;
            }
            c if c == COLLECTION_LIBRARIES => {
                let stats = reconstruct_library_state(&conn, &points)?;
                total_watch_folders += stats.watch_folders;
                total_tracked_files += stats.tracked_files;
                total_chunks += stats.chunks;
            }
            c if c == COLLECTION_RULES => {
                total_rules += reconstruct_rules_state(&conn, &points)?;
            }
            _ => {
                // Scratchpad: no SQLite state needed, points exist only in Qdrant
            }
        }
    }

    // Step 5: Summary
    output::separator();
    output::section("Recovery Summary");
    output::kv("Total Qdrant points", &total_points.to_string());
    output::kv("Watch folders created", &total_watch_folders.to_string());
    output::kv("Tracked files created", &total_tracked_files.to_string());
    output::kv("Qdrant chunks mapped", &total_chunks.to_string());
    output::kv("Rules mirrored", &total_rules.to_string());
    output::separator();
    output::success("Recovery complete. Restart daemon to rebuild vocabulary and tags.");
    output::info("Verify with: wqm admin health");

    Ok(())
}

/// Statistics returned from reconstruction
struct ReconstructStats {
    watch_folders: u64,
    tracked_files: u64,
    chunks: u64,
}

/// Create core tables: watch_folders, tracked_files, qdrant_chunks.
fn create_core_tables(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            submodule_path TEXT,
            git_remote_url TEXT,
            remote_hash TEXT,
            disambiguation_path TEXT,
            is_active INTEGER DEFAULT 0,
            last_activity_at TEXT,
            is_paused INTEGER DEFAULT 0,
            pause_start_time TEXT,
            is_archived INTEGER DEFAULT 0,
            last_commit_hash TEXT,
            is_git_tracked INTEGER DEFAULT 0,
            library_mode TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            cleanup_on_disable INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_scan TEXT,
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )"
    ).context("Failed to create watch_folders")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            branch TEXT,
            file_type TEXT,
            language TEXT,
            file_mtime TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            chunking_method TEXT,
            lsp_status TEXT DEFAULT 'none',
            treesitter_status TEXT DEFAULT 'none',
            last_error TEXT,
            needs_reconcile INTEGER DEFAULT 0,
            reconcile_reason TEXT,
            extension TEXT,
            is_test INTEGER DEFAULT 0,
            collection TEXT NOT NULL DEFAULT 'projects',
            base_point TEXT,
            relative_path TEXT,
            incremental INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
            UNIQUE(watch_folder_id, file_path, branch)
        )"
    ).context("Failed to create tracked_files")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS qdrant_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            point_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_type TEXT,
            symbol_name TEXT,
            start_line INTEGER,
            end_line INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE,
            UNIQUE(file_id, chunk_index)
        )"
    ).context("Failed to create qdrant_chunks")
}

/// Create auxiliary tables: rules_mirror, unified_queue, submodules, operational_state.
fn create_auxiliary_tables(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS rules_mirror (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            scope TEXT,
            tenant_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"
    ).context("Failed to create rules_mirror")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS unified_queue (
            queue_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            status TEXT DEFAULT 'pending',
            branch TEXT,
            payload_json TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0,
            last_error TEXT,
            leased_by TEXT,
            lease_expires_at TEXT
        )"
    ).context("Failed to create unified_queue")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS watch_folder_submodules (
            parent_watch_id TEXT NOT NULL,
            child_watch_id TEXT NOT NULL,
            submodule_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (parent_watch_id, child_watch_id),
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE,
            FOREIGN KEY (child_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )"
    ).context("Failed to create watch_folder_submodules")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS operational_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"
    ).context("Failed to create operational_state")
}

/// Create indexes for recovery tables.
fn create_recovery_indexes(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_watch_collection_tenant ON watch_folders(collection, tenant_id);
         CREATE INDEX IF NOT EXISTS idx_watch_path ON watch_folders(path);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_watch ON tracked_files(watch_folder_id);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_path ON tracked_files(file_path);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_base_point ON tracked_files(base_point);
         CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_point ON qdrant_chunks(point_id);
         CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_file ON qdrant_chunks(file_id);"
    ).context("Failed to create indexes")
}

/// Create a fresh SQLite database with all required tables.
///
/// We create only the tables needed for recovery. The daemon will
/// handle any remaining migrations on its next startup.
fn create_fresh_database(db_path: &Path) -> Result<rusqlite::Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create database directory")?;
    }

    let conn = rusqlite::Connection::open(db_path)
        .context("Failed to create state database")?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=ON;"
    ).context("Failed to set SQLite pragmas")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        )"
    ).context("Failed to create schema_version")?;

    let now = wqm_common::timestamps::now_utc();
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?1, ?2)",
        params![21, now],
    ).context("Failed to insert schema version")?;

    create_core_tables(&conn)?;
    create_auxiliary_tables(&conn)?;

    conn.execute(
        "INSERT INTO operational_state (key, value, updated_at) VALUES ('recovery_done', 'true', ?1)",
        params![now],
    ).context("Failed to set recovery flag")?;

    create_recovery_indexes(&conn)?;

    Ok(conn)
}

/// Reconstruct watch_folders + tracked_files + qdrant_chunks from projects collection points.
fn reconstruct_project_state(
    conn: &rusqlite::Connection,
    points: &[serde_json::Value],
) -> Result<ReconstructStats> {
    let now = wqm_common::timestamps::now_utc();

    // Group points by tenant_id to infer watch folders
    let mut tenant_files: BTreeMap<String, Vec<&serde_json::Value>> = BTreeMap::new();
    for point in points {
        let tenant_id = point["payload"]["tenant_id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        tenant_files.entry(tenant_id).or_default().push(point);
    }

    let mut watch_folders_created = 0u64;
    let mut tracked_files_created = 0u64;
    let mut chunks_created = 0u64;

    let tx = conn.unchecked_transaction()
        .context("Failed to begin transaction")?;

    for (tenant_id, points) in &tenant_files {
        // Infer project root from absolute paths
        let project_root = infer_project_root(points);

        let watch_id = Uuid::new_v4().to_string();
        tx.execute(
            "INSERT OR IGNORE INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, created_at, updated_at)
             VALUES (?1, ?2, 'projects', ?3, 0, 1, ?4, ?5)",
            params![watch_id, project_root, tenant_id, now, now],
        ).context("Failed to insert watch_folder")?;
        watch_folders_created += 1;

        // Group points by (file_path, branch) for tracked_files
        let mut file_groups: BTreeMap<(String, String), Vec<&serde_json::Value>> = BTreeMap::new();
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

            if file_path.is_empty() {
                continue;
            }
            file_groups.entry((file_path, branch)).or_default().push(point);
        }

        for ((file_path, branch), file_points) in &file_groups {
            // Take metadata from the first chunk
            let first = file_points[0];
            let file_hash = first["payload"]["file_hash"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let language = first["payload"]["language"]
                .as_str()
                .map(|s| s.to_string());
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

            let chunk_count = file_points.len() as i64;

            let result = tx.execute(
                "INSERT OR IGNORE INTO tracked_files
                 (watch_folder_id, file_path, branch, file_type, language, file_mtime, file_hash,
                  chunk_count, collection, base_point, relative_path, extension, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'projects', ?9, ?10, ?11, ?12, ?13)",
                params![
                    watch_id, file_path, branch, file_type, language,
                    now, file_hash, chunk_count, base_point, relative_path,
                    extension, now, now
                ],
            ).context("Failed to insert tracked_file")?;

            if result == 0 {
                continue; // Duplicate
            }

            let file_id = tx.last_insert_rowid();
            tracked_files_created += 1;

            // Insert qdrant_chunks
            for point in file_points {
                let point_id_str = if let Some(s) = point["id"].as_str() {
                    s.to_string()
                } else if let Some(n) = point["id"].as_u64() {
                    n.to_string()
                } else {
                    continue;
                };
                let chunk_index = point["payload"]["chunk_index"]
                    .as_u64()
                    .unwrap_or(0) as i64;
                let content = point["payload"]["content"]
                    .as_str()
                    .unwrap_or("");
                let content_hash = wqm_common::hashing::compute_content_hash(content);
                let chunk_type = point["payload"]["chunk_type"]
                    .as_str()
                    .map(|s| s.to_string());
                let symbol_name = point["payload"]["chunk_symbol_name"]
                    .as_str()
                    .map(|s| s.to_string());
                let start_line = point["payload"]["chunk_start_line"]
                    .as_i64();
                let end_line = point["payload"]["chunk_end_line"]
                    .as_i64();

                tx.execute(
                    "INSERT OR IGNORE INTO qdrant_chunks
                     (file_id, point_id, chunk_index, content_hash, chunk_type, symbol_name, start_line, end_line, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        file_id, point_id_str, chunk_index,
                        &content_hash[..32], chunk_type, symbol_name,
                        start_line, end_line, now
                    ],
                ).context("Failed to insert qdrant_chunk")?;
                chunks_created += 1;
            }
        }
    }

    tx.commit().context("Failed to commit project reconstruction")?;

    Ok(ReconstructStats {
        watch_folders: watch_folders_created,
        tracked_files: tracked_files_created,
        chunks: chunks_created,
    })
}

/// Reconstruct watch_folders + tracked_files from libraries collection points.
fn reconstruct_library_state(
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

    let tx = conn.unchecked_transaction()
        .context("Failed to begin transaction")?;

    for (library_name, points) in &library_groups {
        let watch_id = Uuid::new_v4().to_string();

        // Libraries don't have file paths -- use a placeholder directory
        let lib_path = format!("/recovered-libraries/{}", library_name);

        tx.execute(
            "INSERT OR IGNORE INTO watch_folders
             (watch_id, path, collection, tenant_id, library_mode, enabled, created_at, updated_at)
             VALUES (?1, ?2, 'libraries', ?3, 'sync', 1, ?4, ?5)",
            params![watch_id, lib_path, library_name, now, now],
        ).context("Failed to insert library watch_folder")?;
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
                .unwrap_or(&doc_id)
                .to_string();
            let file_hash = first["payload"]["file_hash"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let branch = first["payload"]["branch"]
                .as_str()
                .unwrap_or("main")
                .to_string();

            let chunk_count = doc_points.len() as i64;

            let result = tx.execute(
                "INSERT OR IGNORE INTO tracked_files
                 (watch_folder_id, file_path, branch, file_mtime, file_hash, chunk_count, collection, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'libraries', ?7, ?8)",
                params![watch_id, file_path, branch, now, file_hash, chunk_count, now, now],
            ).context("Failed to insert library tracked_file")?;

            if result == 0 {
                continue;
            }

            let file_id = tx.last_insert_rowid();
            tracked_files_created += 1;

            for point in doc_points {
                let point_id_str = if let Some(s) = point["id"].as_str() {
                    s.to_string()
                } else if let Some(n) = point["id"].as_u64() {
                    n.to_string()
                } else {
                    continue;
                };
                let chunk_index = point["payload"]["chunk_index"]
                    .as_u64()
                    .unwrap_or(0) as i64;
                let content = point["payload"]["content"]
                    .as_str()
                    .unwrap_or("");
                let content_hash = wqm_common::hashing::compute_content_hash(content);

                tx.execute(
                    "INSERT OR IGNORE INTO qdrant_chunks
                     (file_id, point_id, chunk_index, content_hash, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![file_id, point_id_str, chunk_index, &content_hash[..32], now],
                ).context("Failed to insert library qdrant_chunk")?;
                chunks_created += 1;
            }
        }
    }

    tx.commit().context("Failed to commit library reconstruction")?;

    Ok(ReconstructStats {
        watch_folders: watch_folders_created,
        tracked_files: tracked_files_created,
        chunks: chunks_created,
    })
}

/// Reconstruct rules_mirror from rules collection points.
fn reconstruct_rules_state(
    conn: &rusqlite::Connection,
    points: &[serde_json::Value],
) -> Result<u64> {
    let tx = conn.unchecked_transaction()
        .context("Failed to begin transaction")?;

    let mut count = 0u64;

    for point in points {
        let point_id = if let Some(s) = point["id"].as_str() {
            s.to_string()
        } else if let Some(n) = point["id"].as_u64() {
            n.to_string()
        } else {
            continue;
        };

        let content = point["payload"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let scope = point["payload"]["scope"]
            .as_str()
            .map(|s| s.to_string());
        let tenant_id = point["payload"]["tenant_id"]
            .as_str()
            .or_else(|| point["payload"]["project_id"].as_str())
            .map(|s| s.to_string());
        let label = point["payload"]["label"]
            .as_str()
            .map(|s| s.to_string());
        let created_at = point["payload"]["created_at"]
            .as_str()
            .unwrap_or_else(|| {
                // Use a fixed fallback since we can't borrow `now` across the closure
                "2025-01-01T00:00:00Z"
            })
            .to_string();
        let updated_at = point["payload"]["updated_at"]
            .as_str()
            .unwrap_or(&created_at)
            .to_string();

        // Use label as rule_id if available, otherwise point_id
        let rule_id = label.as_deref().unwrap_or(&point_id);

        tx.execute(
            "INSERT OR IGNORE INTO rules_mirror
             (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![rule_id, content, scope, tenant_id, created_at, updated_at],
        ).context("Failed to insert rules_mirror")?;
        count += 1;
    }

    tx.commit().context("Failed to commit rules reconstruction")?;

    Ok(count)
}

/// Infer the project root directory from absolute paths in Qdrant points.
///
/// Collects all absolute_path values and finds the longest common path prefix.
fn infer_project_root(points: &[&serde_json::Value]) -> String {
    let paths: Vec<&str> = points
        .iter()
        .filter_map(|p| {
            p["payload"]["absolute_path"].as_str()
                .or_else(|| p["payload"]["file_path"].as_str())
        })
        .collect();

    if paths.is_empty() {
        return "/unknown-project".to_string();
    }

    if paths.len() == 1 {
        // Single file: use its parent directory
        return PathBuf::from(paths[0])
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/unknown-project".to_string());
    }

    // Find longest common path prefix
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
