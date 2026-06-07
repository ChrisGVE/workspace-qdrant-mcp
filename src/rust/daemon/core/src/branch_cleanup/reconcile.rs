//! Stale-branch reconciliation sweep (#102).
//!
//! Branch deletions are normally handled live by the branch lifecycle event
//! consumer, but events are missed when the daemon is down (or were never
//! emitted before the lifecycle code existed). The stale rows then linger in
//! `tracked_files.branches[]`, Qdrant payloads, and search.db `file_metadata`
//! — where each stale `file_metadata` row duplicates every unfiltered grep
//! match for that file.
//!
//! This sweep compares the branches recorded in both databases against the
//! repository's actual local branches and runs the regular
//! [`cleanup_deleted_branch`] path (which re-checks local AND remote
//! existence before touching anything) for each branch that is gone.

use std::collections::BTreeSet;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::branch_switch::BranchUpdateContext;

use super::cleanup_deleted_branch;

/// Outcome of one reconciliation sweep.
#[derive(Debug, Default)]
pub struct ReconcileStats {
    /// Git-tracked project watch folders examined.
    pub folders_checked: u64,
    /// Branches confirmed gone and cleaned up.
    pub branches_pruned: u64,
    /// Branches skipped (still exist remotely, or existence check failed).
    pub branches_skipped: u64,
    /// Orphaned code_lines rows pruned (files with no file_metadata AND no
    /// tracked_files row).
    pub orphaned_lines_pruned: u64,
    /// Errors encountered (non-fatal; sweep continues).
    pub errors: u64,
}

/// A project watch folder eligible for reconciliation.
#[derive(sqlx::FromRow)]
struct WatchFolderRow {
    watch_id: String,
    tenant_id: String,
    path: String,
}

/// Sweep all active git-tracked project watch folders for branches that are
/// recorded in `tracked_files` / `file_metadata` but no longer exist in the
/// repository, and clean them up via [`cleanup_deleted_branch`].
pub async fn reconcile_stale_branches(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
) -> ReconcileStats {
    let mut stats = ReconcileStats::default();

    let folders: Vec<WatchFolderRow> = match sqlx::query_as(
        "SELECT watch_id, tenant_id, path FROM watch_folders \
         WHERE collection = 'projects' AND is_git_tracked = 1 AND is_archived = 0",
    )
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!("Branch reconcile: failed to list watch folders: {}", e);
            stats.errors += 1;
            return stats;
        }
    };

    for folder in folders {
        let root = Path::new(&folder.path);
        // Skip folders whose checkout is missing or not a git repo — branch
        // existence cannot be determined, so pruning would be unsafe.
        if !root.join(".git").exists() {
            debug!(
                "Branch reconcile: skipping '{}' (no .git at path)",
                folder.path
            );
            continue;
        }
        stats.folders_checked += 1;

        let stored = stored_branches(pool, branch_ctx, &folder).await;
        if stored.is_empty() {
            continue;
        }
        let local = match local_branches(root) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    "Branch reconcile: branch listing failed for '{}': {}",
                    folder.path, e
                );
                stats.errors += 1;
                continue;
            }
        };

        for branch in stored.difference(&local) {
            // cleanup_deleted_branch re-checks local AND remote existence and
            // skips (no data loss) when the branch still exists anywhere or
            // the check fails.
            let result = cleanup_deleted_branch(
                pool,
                branch_ctx,
                &folder.watch_id,
                &folder.tenant_id,
                root,
                branch,
            )
            .await;
            if result.skipped {
                stats.branches_skipped += 1;
            } else {
                info!(
                    "Branch reconcile: pruned stale branch '{}' from '{}' \
                     ({} updated, {} deleted)",
                    branch, folder.path, result.updated, result.deleted
                );
                stats.branches_pruned += 1;
            }
            stats.errors += result.errors;
        }
    }

    // Final pass: code_lines whose file has NO file_metadata rows left are
    // unreachable by every scoped search but still match unscoped FTS queries
    // with stale content. Prune them — but only when the file is also gone
    // from tracked_files (a live tracked file missing its metadata is a
    // different inconsistency; deleting its lines would lose search data).
    if let Some(ref sdb) = branch_ctx.search_db {
        match prune_orphaned_code_lines(pool, sdb.pool()).await {
            Ok(pruned) => stats.orphaned_lines_pruned = pruned,
            Err(e) => {
                warn!("Branch reconcile: orphaned code_lines prune failed: {}", e);
                stats.errors += 1;
            }
        }
    }

    stats
}

/// Delete code_lines (and their FTS5 index entries) for files that have no
/// file_metadata row in search.db AND no tracked_files row in state.db.
async fn prune_orphaned_code_lines(
    state_pool: &SqlitePool,
    search_pool: &SqlitePool,
) -> Result<u64, String> {
    let candidates: Vec<i64> = sqlx::query_scalar(
        "SELECT DISTINCT file_id FROM code_lines \
         WHERE file_id NOT IN (SELECT file_id FROM file_metadata)",
    )
    .fetch_all(search_pool)
    .await
    .map_err(|e| format!("orphan candidate query: {}", e))?;

    let mut pruned = 0u64;
    let mut still_tracked = 0u64;
    for file_id in candidates {
        // Cross-database existence check (state.db and search.db are separate
        // pools — no SQL-level join available).
        let tracked: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE file_id = ?1")
                .bind(file_id)
                .fetch_one(state_pool)
                .await
                .map_err(|e| format!("tracked_files check for file_id={}: {}", file_id, e))?;
        if tracked > 0 {
            still_tracked += 1;
            continue;
        }

        let mut tx = search_pool
            .begin()
            .await
            .map_err(|e| format!("begin orphan prune: {}", e))?;
        // External-content FTS5: remove index entries incrementally before
        // the content rows (mirrors FtsBatchProcessor::delete_file).
        let old_rows: Vec<(i64, String)> =
            sqlx::query_as("SELECT line_id, content FROM code_lines WHERE file_id = ?1")
                .bind(file_id)
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| format!("orphan line fetch for file_id={}: {}", file_id, e))?;
        for (line_id, old_content) in &old_rows {
            if let Err(e) = sqlx::query(crate::code_lines_schema::FTS5_DELETE_ROW_SQL)
                .bind(*line_id)
                .bind(old_content.as_str())
                .execute(&mut *tx)
                .await
            {
                warn!("FTS5 delete failed for line_id={}: {}", line_id, e);
            }
        }
        let result = sqlx::query("DELETE FROM code_lines WHERE file_id = ?1")
            .bind(file_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("orphan delete for file_id={}: {}", file_id, e))?;
        tx.commit()
            .await
            .map_err(|e| format!("commit orphan prune: {}", e))?;
        pruned += result.rows_affected();
    }

    if still_tracked > 0 {
        warn!(
            "Branch reconcile: {} file(s) have code_lines but no file_metadata while still \
             tracked — scoped search misses them until re-indexed",
            still_tracked
        );
    }
    if pruned > 0 {
        info!("Branch reconcile: pruned {} orphaned code_lines", pruned);
    }
    Ok(pruned)
}

/// Union of branches recorded for this watch folder in state.db
/// (`tracked_files.branches[]`) and search.db (`file_metadata.branch`).
async fn stored_branches(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    folder: &WatchFolderRow,
) -> BTreeSet<String> {
    let mut stored: BTreeSet<String> = BTreeSet::new();

    let tracked: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT j.value FROM tracked_files tf, json_each(tf.branches) j \
         WHERE tf.watch_folder_id = ?1",
    )
    .bind(&folder.watch_id)
    .fetch_all(pool)
    .await
    .unwrap_or_else(|e| {
        warn!(
            "Branch reconcile: tracked_files branch listing failed for '{}': {}",
            folder.watch_id, e
        );
        Vec::new()
    });
    stored.extend(tracked);

    if let Some(ref sdb) = branch_ctx.search_db {
        let metadata: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT branch FROM file_metadata \
             WHERE tenant_id = ?1 AND branch IS NOT NULL",
        )
        .bind(&folder.tenant_id)
        .fetch_all(sdb.pool())
        .await
        .unwrap_or_else(|e| {
            warn!(
                "Branch reconcile: file_metadata branch listing failed for '{}': {}",
                folder.tenant_id, e
            );
            Vec::new()
        });
        stored.extend(metadata);
    }

    stored
}

/// List the repository's current local branches.
fn local_branches(root: &Path) -> Result<BTreeSet<String>, String> {
    let out = std::process::Command::new("git")
        .args([
            "-C",
            &root.to_string_lossy(),
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        ])
        .output()
        .map_err(|e| format!("git for-each-ref: {}", e))?;
    if !out.status.success() {
        return Err(format!("git for-each-ref exit={}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn mem_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    async fn setup_state(pool: &SqlitePool) {
        // Minimal tracked_files shape for the existence check.
        sqlx::query("CREATE TABLE tracked_files (file_id INTEGER PRIMARY KEY)")
            .execute(pool)
            .await
            .unwrap();
    }

    async fn setup_search(pool: &SqlitePool) {
        sqlx::query(crate::code_lines_schema::CREATE_FILE_METADATA_V7_SQL)
            .execute(pool)
            .await
            .unwrap();
        sqlx::query(crate::code_lines_schema::CREATE_CODE_LINES_SQL)
            .execute(pool)
            .await
            .unwrap();
        sqlx::query(crate::code_lines_schema::CREATE_CODE_LINES_FTS_SQL)
            .execute(pool)
            .await
            .unwrap();
    }

    async fn insert_line(pool: &SqlitePool, file_id: i64, content: &str) {
        let line_id: i64 = sqlx::query_scalar(
            "INSERT INTO code_lines (file_id, seq, content, line_number)
             VALUES (?1, 1000.0, ?2, 1) RETURNING line_id",
        )
        .bind(file_id)
        .bind(content)
        .fetch_one(pool)
        .await
        .unwrap();
        sqlx::query(crate::code_lines_schema::FTS5_INSERT_ROW_SQL)
            .bind(line_id)
            .bind(content)
            .execute(pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn prune_deletes_fully_dead_orphans_only() {
        let state = mem_pool().await;
        let search = mem_pool().await;
        setup_state(&state).await;
        setup_search(&search).await;

        // file 1: orphaned in search.db AND gone from tracked_files → prune
        insert_line(&search, 1, "dead orphan").await;
        // file 2: orphaned in search.db but STILL tracked → keep
        insert_line(&search, 2, "live but unindexed").await;
        sqlx::query("INSERT INTO tracked_files (file_id) VALUES (2)")
            .execute(&state)
            .await
            .unwrap();
        // file 3: has file_metadata → not an orphan candidate at all
        insert_line(&search, 3, "healthy").await;
        sqlx::query(
            "INSERT INTO file_metadata (file_id, tenant_id, branch, file_path)
             VALUES (3, 't1', 'main', '/p/f.rs')",
        )
        .execute(&search)
        .await
        .unwrap();

        let pruned = prune_orphaned_code_lines(&state, &search).await.unwrap();
        assert_eq!(pruned, 1);

        let remaining: Vec<i64> =
            sqlx::query_scalar("SELECT DISTINCT file_id FROM code_lines ORDER BY file_id")
                .fetch_all(&search)
                .await
                .unwrap();
        assert_eq!(remaining, vec![2, 3]);

        // FTS index consistent: dead content unmatchable, live content intact.
        let dead_hits: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH '\"dead orphan\"'",
        )
        .fetch_one(&search)
        .await
        .unwrap();
        assert_eq!(dead_hits, 0);
        let live_hits: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH '\"healthy\"'",
        )
        .fetch_one(&search)
        .await
        .unwrap();
        assert_eq!(live_hits, 1);
    }

    #[test]
    fn local_branches_lists_refs() {
        // Real temp repo with one branch ref.
        let dir = tempfile::TempDir::new().unwrap();
        let run = |args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(args)
                .output()
                .unwrap()
        };
        run(&["init", "-b", "main"]);
        run(&[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "x",
        ]);
        run(&["branch", "feature-a"]);

        let branches = local_branches(dir.path()).unwrap();
        assert!(branches.contains("main"), "branches: {branches:?}");
        assert!(branches.contains("feature-a"), "branches: {branches:?}");
    }
}
