//! Startup pruning of orphaned search.db FTS entries (#130).
//!
//! `tracked_files` (state.db) is the authoritative registry of every ingested
//! file. The FTS index lives in the separate **search.db** (`code_lines`,
//! `code_lines_fts`, `file_metadata`) and mirrors a subset of those files for
//! fast literal/regex line search (the `grep` surface). When a file's
//! `tracked_files` row is removed through a path that does not also run the FTS
//! delete — historical bulk removals, pre-FTS-era ingests, an interrupted
//! delete — its search.db rows are left behind. Those orphans surface as
//! `grep`/`search` hits for files that no longer exist on disk (e.g. the
//! long-removed `src/typescript/**` tree that triggered #130).
//!
//! This mirrors [`super::remove_orphan_chunks`], which prunes the state.db
//! `qdrant_chunks` mirror, but for the **cross-database** FTS index: it deletes
//! every search.db file whose `file_id` is absent from `tracked_files`,
//! reusing the transactional [`FtsBatchProcessor::delete_file`] so the
//! external-content FTS5 index stays consistent.
//!
//! It runs in the background reconciliation task (not the fast readiness path):
//! a backlog of thousands of orphan deletes must not gate gRPC readiness
//! (issue #59).

use std::collections::HashSet;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use crate::search_db::SearchDbManager;

/// Prune search.db FTS entries whose `file_id` is not present in the state.db
/// `tracked_files` table (#130).
///
/// Returns the number of orphaned files removed from the FTS index.
///
/// # Safety
///
/// If `tracked_files` is empty the prune is skipped entirely. An empty registry
/// against a populated search.db almost always means state.db failed to open or
/// the two databases were mismatched — deleting every FTS row in that situation
/// would be catastrophic and is never the intended outcome.
pub async fn prune_orphan_fts_entries(
    state_pool: &SqlitePool,
    search_db: &SearchDbManager,
) -> Result<u64, String> {
    info!("Pruning orphaned search.db FTS entries (#130)...");

    // Authoritative set of live file_ids from state.db.
    let live_ids: HashSet<i64> = sqlx::query_scalar::<_, i64>("SELECT file_id FROM tracked_files")
        .fetch_all(state_pool)
        .await
        .map_err(|e| format!("Failed to load tracked_files ids: {e}"))?
        .into_iter()
        .collect();

    if live_ids.is_empty() {
        warn!(
            "tracked_files is empty — skipping FTS orphan prune to avoid wiping a \
             populated search.db (possible state.db load failure or DB mismatch)"
        );
        return Ok(0);
    }

    // Candidate file_ids present in the FTS index. Union both tables: a partial
    // historical delete can leave a file with `code_lines` but no
    // `file_metadata` (or the reverse), and either form is an orphan worth
    // clearing.
    let candidate_ids: Vec<i64> = sqlx::query_scalar::<_, i64>(
        "SELECT file_id FROM file_metadata \
         UNION \
         SELECT file_id FROM code_lines",
    )
    .fetch_all(search_db.pool())
    .await
    .map_err(|e| format!("Failed to load search.db file ids: {e}"))?;

    let orphan_ids: Vec<i64> = candidate_ids
        .into_iter()
        .filter(|id| !live_ids.contains(id))
        .collect();

    if orphan_ids.is_empty() {
        debug!("No orphaned FTS entries found in search.db");
        return Ok(0);
    }

    info!(
        "Found {} orphaned FTS file(s) in search.db — removing",
        orphan_ids.len()
    );

    // `delete_file` deletes `code_lines`, the external-content FTS5 rows, and
    // all `file_metadata` rows for a `file_id` (every branch) in one
    // transaction, so a single call per orphan fully clears it.
    let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
    let mut removed: u64 = 0;
    for file_id in orphan_ids {
        match processor.delete_file(file_id).await {
            Ok(_) => removed += 1,
            Err(e) => warn!(
                "Failed to delete orphaned FTS entries for file_id={file_id}: {e} (non-fatal)"
            ),
        }
    }

    if removed > 0 {
        info!("Removed {removed} orphaned FTS file(s) from search.db");
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fts_batch_processor::FileChange;
    use sqlx::sqlite::SqlitePoolOptions;
    use tempfile::TempDir;

    /// Minimal state pool exposing just the `tracked_files.file_id` column the
    /// prune reads. The real schema carries far more, but the prune only needs
    /// the live id set, so a one-column table keeps the test focused.
    async fn state_pool_with_ids(ids: &[i64]) -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query("CREATE TABLE tracked_files (file_id INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .unwrap();
        for id in ids {
            sqlx::query("INSERT INTO tracked_files (file_id) VALUES (?1)")
                .bind(id)
                .execute(&pool)
                .await
                .unwrap();
        }
        pool
    }

    /// Ingest one file into search.db (code_lines + FTS + file_metadata) the
    /// same way the real pipeline does, via `FtsBatchProcessor::flush`.
    async fn write_fts_file(db: &SearchDbManager, file_id: i64, path: &str) {
        let mut processor = FtsBatchProcessor::new(db, FtsBatchConfig::default());
        processor.add_change(FileChange {
            file_id,
            old_content: String::new(),
            new_content: "fn main() {}\nlet x = 1;\n".to_string(),
            tenant_id: "t".to_string(),
            branch: Some("main".to_string()),
            file_path: path.to_string(),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
        processor.flush(0).await.unwrap();
    }

    async fn count(db: &SearchDbManager, sql: &str) -> i64 {
        sqlx::query_scalar(sql).fetch_one(db.pool()).await.unwrap()
    }

    #[tokio::test]
    async fn prunes_only_orphans() {
        let tmp = TempDir::new().unwrap();
        let db = SearchDbManager::new(tmp.path().join("search.db"))
            .await
            .unwrap();
        // 1, 2 are live; 3, 4 are ghosts of files deleted from disk (#130).
        write_fts_file(&db, 1, "/live/a.rs").await;
        write_fts_file(&db, 2, "/live/b.rs").await;
        write_fts_file(&db, 3, "/ghost/c.ts").await;
        write_fts_file(&db, 4, "/ghost/d.ts").await;

        // Sanity: ghosts are present before the prune.
        assert!(
            count(
                &db,
                "SELECT COUNT(*) FROM code_lines WHERE file_id IN (3,4)"
            )
            .await
                > 0
        );

        let state = state_pool_with_ids(&[1, 2]).await;
        let removed = prune_orphan_fts_entries(&state, &db).await.unwrap();
        assert_eq!(removed, 2);

        // Live files untouched.
        assert!(
            count(
                &db,
                "SELECT COUNT(*) FROM code_lines WHERE file_id IN (1,2)"
            )
            .await
                > 0
        );
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM file_metadata WHERE file_id IN (1,2)"
            )
            .await,
            2
        );
        // Ghosts fully gone from both tables.
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM code_lines WHERE file_id IN (3,4)"
            )
            .await,
            0
        );
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM file_metadata WHERE file_id IN (3,4)"
            )
            .await,
            0
        );
    }

    #[tokio::test]
    async fn empty_tracked_files_skips_prune() {
        // Guard against catastrophic wipe: an empty registry against a populated
        // search.db means state.db failed to load or the DBs are mismatched —
        // the prune must do nothing, not delete everything.
        let tmp = TempDir::new().unwrap();
        let db = SearchDbManager::new(tmp.path().join("search.db"))
            .await
            .unwrap();
        write_fts_file(&db, 1, "/x.rs").await;

        let state = state_pool_with_ids(&[]).await;
        let removed = prune_orphan_fts_entries(&state, &db).await.unwrap();
        assert_eq!(removed, 0);
        assert!(
            count(&db, "SELECT COUNT(*) FROM code_lines").await > 0,
            "must NOT wipe FTS when tracked_files is empty"
        );
    }
}
