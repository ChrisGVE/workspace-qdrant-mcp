//! Ignore-file reconciliation engine.
//!
//! Given a project root, walks the tree applying current .gitignore +
//! .wqmignore rules, diffs against the set of already-indexed files in
//! the DB, and enqueues file/delete for stale entries and file/add for
//! missing ones. This keeps the index consistent when ignore rules change
//! while the daemon was offline (startup) or at runtime (watcher trigger).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use ignore::WalkBuilder;
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use sqlx::Row;

use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{ItemType, QueueOperation};

/// Outcome of a single reconciliation run.
#[derive(Debug, Default)]
pub struct ReconcileStats {
    pub stale_deleted: u64,
    pub missing_added: u64,
}

/// Reconcile ignore rules for a single project.
///
/// 1. Walk the project tree with WalkBuilder (respects .gitignore + .wqmignore)
/// 2. Query tracked_files for all indexed file paths in that project
/// 3. Diff: stale = indexed but now excluded, missing = on disk but not indexed
/// 4. Enqueue file/delete for stale, file/add for missing
pub async fn reconcile_ignore_rules(
    project_root: &Path,
    tenant_id: &str,
    collection: &str,
    pool: &SqlitePool,
    queue_manager: &Arc<QueueManager>,
) -> Result<ReconcileStats, String> {
    let watch_id = fetch_watch_id(pool, tenant_id, collection).await?;

    let eligible_files = walk_eligible_files(project_root)?;
    let indexed_files = get_indexed_file_paths(pool, &watch_id).await?;

    let stale: Vec<&String> = indexed_files
        .iter()
        .filter(|p| !eligible_files.contains(p.as_str()))
        .collect();
    let missing: Vec<&String> = eligible_files
        .iter()
        .filter(|p| !indexed_files.contains(p.as_str()))
        .collect();

    if stale.is_empty() && missing.is_empty() {
        debug!(
            "[ignore_sync] {} — no changes (indexed={}, eligible={})",
            tenant_id,
            indexed_files.len(),
            eligible_files.len()
        );
        return Ok(ReconcileStats::default());
    }

    info!(
        "[ignore_sync] {} — {} stale, {} missing (indexed={}, eligible={})",
        tenant_id,
        stale.len(),
        missing.len(),
        indexed_files.len(),
        eligible_files.len()
    );

    enqueue_reconcile_ops(
        queue_manager,
        tenant_id,
        collection,
        project_root,
        &stale,
        &missing,
    )
    .await
}

/// Look up the watch_id for a tenant+collection combination.
async fn fetch_watch_id(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
) -> Result<String, String> {
    let row = sqlx::query(
        "SELECT watch_id FROM watch_folders \
         WHERE tenant_id = ?1 AND collection = ?2 AND enabled = 1 LIMIT 1",
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("lookup_watch_folder failed: {e}"))?
    .ok_or_else(|| {
        format!(
            "No watch folder for tenant={} collection={}",
            tenant_id, collection
        )
    })?;
    Ok(row.get("watch_id"))
}

/// Enqueue delete + add operations for stale and missing files.
///
/// `stale` and `missing` carry relative paths (forward-slash, normalized) —
/// the JSON payload's `file_path` field is reconstructed as an absolute path
/// by joining `project_root` so downstream file processors receive paths in
/// the same shape they did before the v37 `tracked_files` rename.
async fn enqueue_reconcile_ops(
    queue_manager: &Arc<QueueManager>,
    tenant_id: &str,
    collection: &str,
    project_root: &Path,
    stale: &[&String],
    missing: &[&String],
) -> Result<ReconcileStats, String> {
    let mut stats = ReconcileStats::default();

    stats.stale_deleted = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
        project_root,
        QueueOperation::Delete,
        stale,
        "ignore_rule_change",
    )
    .await;

    stats.missing_added = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
        project_root,
        QueueOperation::Add,
        missing,
        "ignore_reconciliation",
    )
    .await;

    info!(
        "[ignore_sync] {} — enqueued {} deletes, {} adds",
        tenant_id, stats.stale_deleted, stats.missing_added
    );

    Ok(stats)
}

/// Default batch size for ignore-sync enqueues.
///
/// Each batch is committed as a single SQLite transaction so we amortise
/// lock contention across hundreds of inserts. 500 is a balance between
/// commit latency and transaction size that matches the `#59` acceptance
/// criteria.
pub const IGNORE_SYNC_BATCH_SIZE: usize = 500;

/// Enqueue `file_paths` as `(File, op)` items in batches using a single
/// SQLite transaction per batch. Progress is logged every batch so large
/// backfills are observable in the daemon log.
async fn enqueue_ignore_ops(
    queue_manager: &Arc<QueueManager>,
    tenant_id: &str,
    collection: &str,
    project_root: &Path,
    op: QueueOperation,
    file_paths: &[&String],
    reason: &str,
) -> u64 {
    let total = file_paths.len();
    if total == 0 {
        return 0;
    }

    let mut enqueued: u64 = 0;
    let op_label = match op {
        QueueOperation::Delete => "delete",
        QueueOperation::Add => "add",
        _ => "op",
    };

    for chunk in file_paths.chunks(IGNORE_SYNC_BATCH_SIZE) {
        let payloads: Vec<String> = chunk
            .iter()
            .map(|rel_path| {
                let abs = project_root
                    .join(rel_path.as_str())
                    .to_string_lossy()
                    .to_string();
                match op {
                    QueueOperation::Delete => serde_json::json!({
                        "file_path": abs,
                        "reason": reason,
                    }),
                    _ => serde_json::json!({
                        "file_path": abs,
                        "source": reason,
                    }),
                }
                .to_string()
            })
            .collect();

        match queue_manager
            .enqueue_unified_batch(ItemType::File, op, tenant_id, collection, &payloads, None)
            .await
        {
            Ok(n) => {
                enqueued += n;
                debug!(
                    "[ignore_sync] {} — {}: batch committed {}/{} items ({} total enqueued)",
                    tenant_id,
                    op_label,
                    n,
                    chunk.len(),
                    enqueued
                );
            }
            Err(e) => warn!(
                "[ignore_sync] {} — {} batch failed (size={}): {}",
                tenant_id,
                op_label,
                chunk.len(),
                e
            ),
        }
    }

    enqueued
}

/// Walk project tree and collect all eligible file paths (not excluded
/// by .gitignore or .wqmignore). Returns paths relative to `project_root`,
/// normalized to forward-slash separators so comparison against the
/// `tracked_files.relative_path` column works identically on Windows.
fn walk_eligible_files(project_root: &Path) -> Result<HashSet<String>, String> {
    let mut builder = WalkBuilder::new(project_root);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .add_custom_ignore_filename(".gitignore")
        .add_custom_ignore_filename(".wqmignore");

    let mut files = HashSet::new();
    for entry in builder.build().flatten() {
        if entry.file_type().map_or(false, |ft| ft.is_file()) {
            if let Some(rel) = entry
                .path()
                .strip_prefix(project_root)
                .ok()
                .map(normalize_relative)
            {
                files.insert(rel);
            }
        }
    }

    Ok(files)
}

/// Normalize a relative path to the storage format used by
/// `tracked_files.relative_path` — forward-slash separators, lossy UTF-8.
fn normalize_relative(rel: &Path) -> String {
    let s = rel.to_string_lossy().to_string();
    if std::path::MAIN_SEPARATOR == '/' {
        s
    } else {
        s.replace(std::path::MAIN_SEPARATOR, "/")
    }
}

/// Get all tracked relative paths for a watch folder from the DB.
///
/// Reads the canonical `relative_path` column. Pre-v37 the schema had a
/// denormalized absolute `file_path`; that column was dropped by the v37
/// `tracked_files` rebuild (see `tracked_files_schema::schema`).
async fn get_indexed_file_paths(
    pool: &SqlitePool,
    watch_folder_id: &str,
) -> Result<HashSet<String>, String> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT relative_path FROM tracked_files WHERE watch_folder_id = ?1")
            .bind(watch_folder_id)
            .fetch_all(pool)
            .await
            .map_err(|e| format!("query tracked_files failed: {e}"))?;

    Ok(rows.into_iter().map(|(p,)| p).collect())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// Unit test the diff logic without DB (just the set operations)
    #[test]
    fn diff_stale_and_missing() {
        let indexed: HashSet<String> = ["/a/foo.rs", "/a/bar.rs", "/a/old.rs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let eligible: HashSet<String> = ["/a/foo.rs", "/a/bar.rs", "/a/new.rs"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let stale: Vec<&String> = indexed
            .iter()
            .filter(|p| !eligible.contains(p.as_str()))
            .collect();
        let missing: Vec<&String> = eligible
            .iter()
            .filter(|p| !indexed.contains(p.as_str()))
            .collect();

        assert_eq!(stale.len(), 1);
        assert!(stale.iter().any(|p| p.as_str() == "/a/old.rs"));
        assert_eq!(missing.len(), 1);
        assert!(missing.iter().any(|p| p.as_str() == "/a/new.rs"));
    }

    #[test]
    fn diff_no_changes() {
        let files: HashSet<String> = ["/a/foo.rs"].iter().map(|s| s.to_string()).collect();
        let stale: Vec<&String> = files
            .iter()
            .filter(|p| !files.contains(p.as_str()))
            .collect();
        let missing: Vec<&String> = files
            .iter()
            .filter(|p| !files.contains(p.as_str()))
            .collect();
        assert!(stale.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn walk_eligible_files_respects_gitignore() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".gitignore"), "dist/\n").unwrap();
        let dist = root.path().join("dist");
        fs::create_dir(&dist).unwrap();
        fs::write(dist.join("bundle.js"), "//").unwrap();
        let src = root.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path()).unwrap();
        // src/main.rs should be eligible
        assert!(files.iter().any(|f| f.ends_with("main.rs")));
        // dist/bundle.js should NOT be eligible
        assert!(!files.iter().any(|f| f.ends_with("bundle.js")));
    }

    #[test]
    fn walk_eligible_files_respects_wqmignore_exclusion() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join(".wqmignore"), "data/\n").unwrap();
        let data = root.path().join("data");
        fs::create_dir(&data).unwrap();
        fs::write(data.join("big.csv"), "a,b,c").unwrap();
        fs::write(root.path().join("readme.md"), "# hi").unwrap();

        let files = walk_eligible_files(root.path()).unwrap();
        assert!(files.iter().any(|f| f.ends_with("readme.md")));
        assert!(!files.iter().any(|f| f.ends_with("big.csv")));
    }

    #[test]
    fn walk_eligible_files_emits_relative_paths_with_forward_slashes() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("src").join("api");
        std::fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("server.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path()).unwrap();

        // Output must be a relative path joined by '/', matching the format
        // used in tracked_files.relative_path. On Windows this validates the
        // separator normalization. The absolute path must NOT leak through.
        assert!(
            files.contains("src/api/server.rs"),
            "expected 'src/api/server.rs', got {:?}",
            files
        );
        assert!(
            !files.iter().any(|f| f.contains(':') || f.starts_with('/')),
            "no entry should look absolute, got {:?}",
            files
        );
    }

    #[tokio::test]
    async fn get_indexed_file_paths_reads_relative_path_column() {
        // Bootstrap the full SchemaManager pipeline so tracked_files has the
        // post-v37 shape (no `file_path` column, `relative_path` canonical).
        let pool = super::super::tests::create_test_pool().await;
        super::super::tests::setup_schema(&pool).await;

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at) \
             VALUES ('wf1', '/some/root', 'projects', 'tenant1', 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        for rel in ["src/main.rs", "docs/readme.md"] {
            sqlx::query(
                "INSERT INTO tracked_files \
                 (watch_folder_id, relative_path, branch, file_mtime, file_hash, collection, base_point, created_at, updated_at) \
                 VALUES ('wf1', ?1, 'main', '2025-01-01T00:00:00Z', 'h', 'projects', 'bp', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
            )
            .bind(rel)
            .execute(&pool)
            .await
            .unwrap();
        }

        let got = get_indexed_file_paths(&pool, "wf1").await.unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.contains("src/main.rs"));
        assert!(got.contains("docs/readme.md"));
    }
}
