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

use crate::patterns::exclusion::should_exclude_file;
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

    enqueue_reconcile_ops(queue_manager, tenant_id, collection, &stale, &missing).await
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
async fn enqueue_reconcile_ops(
    queue_manager: &Arc<QueueManager>,
    tenant_id: &str,
    collection: &str,
    stale: &[&String],
    missing: &[&String],
) -> Result<ReconcileStats, String> {
    let mut stats = ReconcileStats::default();

    stats.stale_deleted = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
        QueueOperation::Delete,
        stale,
        "ignore_rule_change",
    )
    .await;

    stats.missing_added = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
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
            .map(|file_path| {
                match op {
                    QueueOperation::Delete => serde_json::json!({
                        "file_path": file_path,
                        "reason": reason,
                    }),
                    _ => serde_json::json!({
                        "file_path": file_path,
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
/// by .gitignore or .wqmignore). Returns relative path strings (relative
/// to `project_root`), matching the post-v37 `tracked_files.relative_path`
/// column format.
///
/// `hidden(false)` lets the walker descend into dotted paths so the
/// `.github/` whitelist still works, but every candidate is then run
/// through `should_exclude_file` — the same exclusion engine the file
/// watcher and folder scanner use. Without this, `.git/` internals
/// (objects, packs, submodule hook samples) are walked, flagged as
/// "missing", and enqueued as file/add, flooding the queue.
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
            let rel = entry
                .path()
                .strip_prefix(project_root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();
            // Mirror the watcher/scanner exclusion policy so the eligible
            // set matches what the rest of the pipeline will actually index.
            if should_exclude_file(&rel) {
                continue;
            }
            files.insert(rel);
        }
    }

    Ok(files)
}

/// Get all tracked relative file paths for a watch folder from the DB.
///
/// Post-v37: the absolute `file_path` column no longer exists; relative paths
/// are stored in `relative_path`.
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
    fn walk_eligible_files_excludes_git_internals() {
        let root = tempfile::tempdir().unwrap();
        // Simulate a real repo's .git internals (objects, packs, submodule hooks).
        let objects = root.path().join(".git/objects/3c");
        fs::create_dir_all(&objects).unwrap();
        fs::write(objects.join("9b3dda0a7b0788f289fd1e2edc07306fea374b"), "x").unwrap();
        let hooks = root.path().join(".git/modules/sub/hooks");
        fs::create_dir_all(&hooks).unwrap();
        fs::write(hooks.join("prepare-commit-msg.sample"), "#!/bin/sh").unwrap();
        // A legitimate source file that MUST remain eligible.
        let src = root.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path()).unwrap();

        assert!(
            files.iter().any(|f| f.ends_with("main.rs")),
            "source files stay eligible"
        );
        assert!(
            !files.iter().any(|f| f.contains(".git/")),
            ".git internals must never be eligible, got: {files:?}"
        );
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
}
