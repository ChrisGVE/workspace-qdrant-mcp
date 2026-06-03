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

use crate::patterns::global_ignore;
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
///
/// `global_ignore_path` — optional path to `global.wqmignore`, applied on top
/// of per-project ignore files. When `None` or when the file does not exist,
/// only the per-project `.gitignore` / `.wqmignore` files are used.
pub async fn reconcile_ignore_rules(
    project_root: &Path,
    tenant_id: &str,
    collection: &str,
    pool: &SqlitePool,
    queue_manager: &Arc<QueueManager>,
    global_ignore_path: Option<&Path>,
) -> Result<ReconcileStats, String> {
    let watch_id = fetch_watch_id(pool, tenant_id, collection).await?;

    let eligible_files = walk_eligible_files(project_root, global_ignore_path)?;
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
///
/// `stale` and `missing` carry relative paths (forward-slash, normalized).
/// The JSON payload's `file_path` field is the relative form — `FilePayload`
/// types it as `RelativePath` and the downstream strategy reanchors via
/// `RelativePath::to_absolute(watch_folder_root)`. Sending an absolute path
/// here makes the strategy double-join (root + absolute), producing a path
/// like `<root>//<root>/<rel>` that does not exist on disk, which then
/// triggers `handle_missing_file` silently — items drain without indexing.
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
            .map(|rel_path| {
                // FilePayload.file_path is RelativePath. Sending an absolute
                // path here is a silent footgun: serde derives are transparent
                // and the strategy re-anchors via to_absolute(root), producing
                // <root>//<absolute> which does not exist on disk.
                let rel = rel_path.as_str();
                match op {
                    QueueOperation::Delete => serde_json::json!({
                        "file_path": rel,
                        "reason": reason,
                    }),
                    _ => serde_json::json!({
                        "file_path": rel,
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
///
/// `global_ignore_path` — if `Some` and the file exists on disk, its patterns
/// are applied as a base-level ignore layer across the entire walk (equivalent
/// to a project-root `.wqmignore` but sourced from outside the project tree).
fn walk_eligible_files(
    project_root: &Path,
    global_ignore_path: Option<&Path>,
) -> Result<HashSet<String>, String> {
    let mut builder = WalkBuilder::new(project_root);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .add_custom_ignore_filename(".gitignore")
        .add_custom_ignore_filename(".wqmignore");

    // Explicitly apply the project-root `.wqmignore` as a base layer. The
    // `add_custom_ignore_filename(".wqmignore")` above is meant to pick it up
    // during the walk, but in a git repo (`git_ignore(true)`) it did NOT
    // reliably exclude root-anchored deep paths (e.g.
    // `src/typescript/mcp-server/reports/`), so reconciliation eligibility
    // diverged from the scan path's `ProjectIgnoreMatcher` (which DOES honor
    // the root `.wqmignore`). `add_ignore` anchors the file's patterns to its
    // parent dir (= `project_root`), matching the scan path exactly — without
    // this, reconciliation would keep re-adding files the scan path excludes
    // (add/delete reconcile loop). The bind-mounted `global.wqmignore` is
    // applied separately below.
    let project_wqmignore = project_root.join(".wqmignore");
    if project_wqmignore.is_file() {
        builder.add_ignore(&project_wqmignore);
    }

    // Apply global ignore rules (daemon-wide, outside the project tree).
    // `add_ignore` applies the file's patterns as a base layer that every
    // project walk inherits; `add_custom_ignore_filename` only finds files
    // inside the walked tree, so it cannot reference the global file here.
    if let Some(global_path) = global_ignore_path {
        if global_path.is_file() {
            builder.add_ignore(global_path);
            debug!(
                "[ignore_sync] applying global ignore rules from {}",
                global_path.display()
            );
        }
    }

    // Belt-and-suspenders global filter. `WalkBuilder::add_ignore(global)` above
    // anchors the global patterns to the file's parent dir (`/var/lib/memexd/`),
    // which reliably matches only depth-1 project paths — nested matches leak
    // (e.g. `state/qdrant/...`, `<proj>/generated/...` survive, so reconciliation
    // never marks them stale and the residuals persist). Re-check every candidate
    // against the SAME root-anchored matcher the watcher and folder-scan use
    // (`patterns::global_ignore`), built once from the same file, so all three
    // paths agree on eligibility. Only DROPS files (never adds), so it cannot
    // resurrect a walk-pruned path.
    let global_matcher = global_ignore_path.and_then(global_ignore::matcher_from);

    let mut files = HashSet::new();
    for entry in builder.build().flatten() {
        if entry.file_type().map_or(false, |ft| ft.is_file()) {
            if let Some(ref m) = global_matcher {
                if m.matched_path_or_any_parents(entry.path(), false).is_ignore() {
                    continue;
                }
            }
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

        let files = walk_eligible_files(root.path(), None).unwrap();
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

        let files = walk_eligible_files(root.path(), None).unwrap();
        assert!(files.iter().any(|f| f.ends_with("readme.md")));
        assert!(!files.iter().any(|f| f.ends_with("big.csv")));
    }

    #[test]
    fn walk_eligible_files_respects_global_ignore() {
        let global_dir = tempfile::tempdir().unwrap();
        let global_ignore = global_dir.path().join("global.wqmignore");
        fs::write(&global_ignore, "vendors/\n*.zip\n").unwrap();

        let root = tempfile::tempdir().unwrap();
        let vendors = root.path().join("vendors");
        fs::create_dir(&vendors).unwrap();
        fs::write(vendors.join("library.js"), "// lib").unwrap();
        fs::write(root.path().join("archive.zip"), "PK..").unwrap();
        fs::write(root.path().join("main.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path(), Some(&global_ignore)).unwrap();
        // main.rs is eligible
        assert!(files.contains("main.rs"), "expected main.rs, got {files:?}");
        // vendors/ and *.zip are globally excluded
        assert!(
            !files.iter().any(|f| f.contains("library.js")),
            "vendors/ should be excluded"
        );
        assert!(!files.contains("archive.zip"), "*.zip should be excluded");
    }

    #[test]
    fn walk_eligible_files_excludes_deep_global_match() {
        // Regression: `WalkBuilder::add_ignore` anchors global patterns to the
        // ignore file's parent dir, so a `**/`-pattern leaks for DEEP (depth-2+)
        // project paths — `state/qdrant/...` survived reconciliation and was
        // never marked stale. The post-filter via `global_ignore::matcher_from`
        // must drop it regardless of depth.
        let global_dir = tempfile::tempdir().unwrap();
        let global_ignore = global_dir.path().join("global.wqmignore");
        fs::write(&global_ignore, "**/state/qdrant/\n**/generated/\n").unwrap();

        let root = tempfile::tempdir().unwrap();
        let deep = root.path().join("sub").join("state").join("qdrant");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("segment.json"), "{}").unwrap();
        let gen = root.path().join("pkg").join("generated");
        fs::create_dir_all(&gen).unwrap();
        fs::write(gen.join("api.pb.dart"), "// gen").unwrap();
        fs::write(root.path().join("keep.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path(), Some(&global_ignore)).unwrap();
        assert!(files.contains("keep.rs"), "hand-authored file kept, got {files:?}");
        assert!(
            !files.iter().any(|f| f.contains("state/qdrant")),
            "deep state/qdrant must be excluded, got {files:?}"
        );
        assert!(
            !files.iter().any(|f| f.contains("generated")),
            "deep generated/ must be excluded, got {files:?}"
        );
    }

    #[test]
    fn walk_eligible_files_emits_relative_paths_with_forward_slashes() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("src").join("api");
        std::fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("server.rs"), "fn main() {}").unwrap();

        let files = walk_eligible_files(root.path(), None).unwrap();

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
