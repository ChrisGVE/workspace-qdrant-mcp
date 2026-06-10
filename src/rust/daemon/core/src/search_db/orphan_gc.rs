//! Orphaned search-content GC.
//!
//! `code_lines.file_id` references `tracked_files.file_id` in state.db, but
//! SQLite cannot enforce a cross-database foreign key — the link is
//! "enforced at the application level" (see `code_lines_schema`). The
//! per-file Delete path upholds it only for files that are still tracked
//! when the Delete processes: content indexed for a file whose tracked row
//! vanished by another route (an ignore reconciliation that enqueued the
//! delete before the FTS5 content landed, a reembed reset, a crash between
//! the two databases) is invisible to every cleanup mechanism and lingers
//! in `code_lines`/`code_lines_fts`/`file_metadata` forever — `grep` keeps
//! returning ghost matches for files excluded long ago (observed: 1,717
//! matches under an ignored `proto/src/generated/` tree).
//!
//! This module is the missing FK enforcement: diff the `file_id`s present
//! in search.db against `tracked_files`, then remove the orphans by one of
//! two paths sized to the backlog — surgical per-file deletes through
//! [`FtsBatchProcessor::delete_file`] (incremental FTS5, no rebuild) for a
//! handful of orphans, or chunked bulk `DELETE`s plus a full FTS5 rebuild
//! for an accumulated backlog (the production case above carried 45k
//! orphaned files / 19.6M lines — per-line FTS5 deletes would mean ~40M
//! statements at startup).
//!
//! Run at startup BEFORE the queue processor starts (no FTS5 work in
//! flight, so the diff cannot race the batch writer) — the same window as
//! `reset_orphaned_destinations_at_startup`.

use std::collections::HashSet;

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};

use super::types::SearchDbResult;
use super::SearchDbManager;

/// Outcome of one GC pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct OrphanGcStats {
    /// Orphaned files whose search content was removed.
    pub files_deleted: u64,
    /// `code_lines` rows removed across those files.
    pub lines_deleted: u64,
}

/// Orphan-count threshold above which the GC switches from surgical
/// per-file deletes (incremental FTS5 maintenance) to bulk `DELETE`s plus a
/// full FTS5 rebuild.
///
/// The surgical path issues one FTS5 'delete' command per line — fine for a
/// handful of files, catastrophic for an accumulated backlog (observed in
/// production: 45k orphaned files carrying 19.6M code_lines ≈ 40M
/// statements). Past this threshold, chunked `DELETE ... WHERE file_id IN`
/// plus one `INSERT INTO code_lines_fts(code_lines_fts) VALUES('rebuild')`
/// is orders of magnitude cheaper.
const BULK_PATH_FILE_THRESHOLD: usize = 64;

/// Max ids per `IN`-list chunk (keeps bind-variable count well under
/// SQLite's limit).
const ID_CHUNK: usize = 500;

/// Remove search.db content whose `tracked_files` row no longer exists.
///
/// `state_pool` is the state.db pool (`tracked_files` lives there). An
/// empty `tracked_files` is treated as authoritative: with no tracked files
/// there is no live content, so everything in search.db is orphaned — the
/// rescan/reembed path rebuilds it from disk.
///
/// `tracked_files.file_id` is `AUTOINCREMENT`, so ids are never reused and
/// the orphan snapshot can never come to refer to a different, newer file.
pub async fn gc_orphaned_files(
    search_db: &SearchDbManager,
    state_pool: &SqlitePool,
) -> SearchDbResult<OrphanGcStats> {
    gc_orphaned_files_with_threshold(search_db, state_pool, BULK_PATH_FILE_THRESHOLD).await
}

/// [`gc_orphaned_files`] with an explicit bulk-path threshold (tests force
/// either path with it).
pub(crate) async fn gc_orphaned_files_with_threshold(
    search_db: &SearchDbManager,
    state_pool: &SqlitePool,
    bulk_threshold: usize,
) -> SearchDbResult<OrphanGcStats> {
    let search_pool = search_db.pool();

    // Union both tables: a `file_metadata` row without `code_lines` (a
    // size-capped fts5_skipped file) and `code_lines` without metadata (a
    // half-applied write) are both reachable orphan shapes.
    let search_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT file_id FROM file_metadata UNION SELECT file_id FROM code_lines",
    )
    .fetch_all(search_pool)
    .await?;

    if search_ids.is_empty() {
        return Ok(OrphanGcStats::default());
    }

    let tracked: HashSet<i64> = sqlx::query_scalar("SELECT file_id FROM tracked_files")
        .fetch_all(state_pool)
        .await?
        .into_iter()
        .collect();

    let orphans: Vec<i64> = search_ids
        .into_iter()
        .filter(|id| !tracked.contains(id))
        .collect();

    if orphans.is_empty() {
        return Ok(OrphanGcStats::default());
    }

    let stats = if orphans.len() <= bulk_threshold {
        info!(
            "[search_gc] {} orphaned search-db file(s) — surgical per-file delete",
            orphans.len()
        );
        surgical_delete(search_db, &orphans).await
    } else {
        info!(
            "[search_gc] {} orphaned search-db file(s) — bulk delete + FTS5 rebuild",
            orphans.len()
        );
        bulk_delete(search_db, &orphans).await?
    };

    info!(
        "[search_gc] done: {} file(s), {} line(s) removed",
        stats.files_deleted, stats.lines_deleted
    );
    Ok(stats)
}

/// Per-file delete through [`FtsBatchProcessor::delete_file`] — keeps the
/// FTS5 index incrementally consistent, no rebuild needed. Per-file failures
/// are logged and skipped so one bad row cannot strand the rest.
async fn surgical_delete(search_db: &SearchDbManager, orphans: &[i64]) -> OrphanGcStats {
    let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
    let mut stats = OrphanGcStats::default();
    for file_id in orphans {
        match processor.delete_file(*file_id).await {
            Ok(lines) => {
                stats.files_deleted += 1;
                stats.lines_deleted += lines as u64;
            }
            Err(e) => {
                warn!("[search_gc] failed to delete file_id={}: {}", file_id, e);
            }
        }
    }
    stats
}

/// Chunked bulk `DELETE` of `code_lines` + `file_metadata`, then a full FTS5
/// rebuild.
///
/// The bulk `DELETE` bypasses the per-row FTS5 'delete' command, leaving the
/// external-content index referencing dead rowids; the trailing 'rebuild'
/// re-derives it from the surviving `code_lines`. Until the rebuild commits,
/// FTS queries may transiently return phantom rowids — acceptable in the
/// pre-start window this GC runs in (and self-healing once rebuilt).
async fn bulk_delete(
    search_db: &SearchDbManager,
    orphans: &[i64],
) -> SearchDbResult<OrphanGcStats> {
    let pool = search_db.pool();
    let mut stats = OrphanGcStats::default();

    let mut tx = pool.begin().await?;
    for chunk in orphans.chunks(ID_CHUNK) {
        let placeholders = vec!["?"; chunk.len()].join(",");

        let sql = format!("DELETE FROM code_lines WHERE file_id IN ({placeholders})");
        let mut q = sqlx::query(&sql);
        for id in chunk {
            q = q.bind(id);
        }
        stats.lines_deleted += q.execute(&mut *tx).await?.rows_affected();

        let sql = format!("DELETE FROM file_metadata WHERE file_id IN ({placeholders})");
        let mut q = sqlx::query(&sql);
        for id in chunk {
            q = q.bind(id);
        }
        q.execute(&mut *tx).await?;

        stats.files_deleted += chunk.len() as u64;
    }
    tx.commit().await?;

    info!(
        "[search_gc] bulk delete committed ({} lines) — rebuilding FTS5 index",
        stats.lines_deleted
    );
    search_db
        .rebuild_and_maybe_optimize_fts(stats.lines_deleted as usize)
        .await?;

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory state.db stand-in: only the `tracked_files.file_id` column
    /// is read by the GC, so a minimal table keeps the test self-contained.
    async fn state_pool_with_tracked(file_ids: &[i64]) -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query("CREATE TABLE tracked_files (file_id INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .unwrap();
        for id in file_ids {
            sqlx::query("INSERT INTO tracked_files (file_id) VALUES (?1)")
                .bind(id)
                .execute(&pool)
                .await
                .unwrap();
        }
        pool
    }

    async fn search_db_in_temp() -> (tempfile::TempDir, SearchDbManager) {
        let dir = tempfile::tempdir().unwrap();
        let db = SearchDbManager::new(dir.path().join("search.db"))
            .await
            .unwrap();
        (dir, db)
    }

    async fn seed_file(db: &SearchDbManager, file_id: i64, content: &str) {
        let processor = FtsBatchProcessor::new(db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                file_id,
                content,
                "tenant-a",
                Some("main"),
                &format!("/repo/file{file_id}.java"),
                Some(&format!("bp{file_id}")),
                Some(&format!("file{file_id}.java")),
                Some("hash"),
            )
            .await
            .unwrap();
    }

    async fn count(pool: &SqlitePool, sql: &str) -> i64 {
        sqlx::query_scalar(sql).fetch_one(pool).await.unwrap()
    }

    #[tokio::test]
    async fn removes_only_untracked_files() {
        let (_d, db) = search_db_in_temp().await;
        seed_file(&db, 1, "tracked line one\ntracked line two").await;
        seed_file(&db, 2, "orphan generated stub alpha\norphan generated stub beta").await;
        seed_file(&db, 3, "another orphan body").await;
        let state = state_pool_with_tracked(&[1]).await;

        let stats = gc_orphaned_files(&db, &state).await.unwrap();

        assert_eq!(stats.files_deleted, 2);
        assert!(stats.lines_deleted >= 3);
        let pool = db.pool();
        assert_eq!(
            count(pool, "SELECT COUNT(*) FROM file_metadata").await,
            1,
            "only the tracked file's metadata survives"
        );
        assert_eq!(
            count(pool, "SELECT COUNT(*) FROM code_lines WHERE file_id != 1").await,
            0
        );
        // The FTS index must not match the deleted content anymore (trigram
        // tokenizer — query must be >= 3 chars), while tracked content does.
        assert_eq!(
            count(
                pool,
                "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH 'generated stub'"
            )
            .await,
            0,
            "ghost FTS matches must be gone"
        );
        assert!(
            count(
                pool,
                "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH 'tracked line'"
            )
            .await
                >= 2
        );
    }

    #[tokio::test]
    async fn bulk_path_removes_orphans_and_rebuilds_fts() {
        // Force the bulk path with a threshold below the orphan count: the
        // chunked DELETEs bypass per-row FTS5 maintenance, so this also
        // proves the trailing rebuild leaves the index consistent.
        let (_d, db) = search_db_in_temp().await;
        seed_file(&db, 1, "tracked line one\ntracked line two").await;
        seed_file(&db, 2, "orphan generated stub alpha").await;
        seed_file(&db, 3, "orphan generated stub beta").await;
        seed_file(&db, 4, "orphan generated stub gamma").await;
        let state = state_pool_with_tracked(&[1]).await;

        let stats = gc_orphaned_files_with_threshold(&db, &state, 1)
            .await
            .unwrap();

        assert_eq!(stats.files_deleted, 3);
        assert_eq!(stats.lines_deleted, 3);
        let pool = db.pool();
        assert_eq!(count(pool, "SELECT COUNT(*) FROM file_metadata").await, 1);
        assert_eq!(
            count(pool, "SELECT COUNT(*) FROM code_lines WHERE file_id != 1").await,
            0
        );
        assert_eq!(
            count(
                pool,
                "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH 'generated stub'"
            )
            .await,
            0,
            "rebuilt FTS index must not ghost-match deleted content"
        );
        assert!(
            count(
                pool,
                "SELECT COUNT(*) FROM code_lines_fts WHERE code_lines_fts MATCH 'tracked line'"
            )
            .await
                >= 2,
            "rebuilt FTS index must still match surviving content"
        );
    }

    #[tokio::test]
    async fn noop_when_everything_is_tracked() {
        let (_d, db) = search_db_in_temp().await;
        seed_file(&db, 7, "alpha\nbeta").await;
        let state = state_pool_with_tracked(&[7]).await;

        let stats = gc_orphaned_files(&db, &state).await.unwrap();

        assert_eq!(stats.files_deleted, 0);
        assert_eq!(count(db.pool(), "SELECT COUNT(*) FROM code_lines").await, 2);
    }

    #[tokio::test]
    async fn noop_on_empty_search_db() {
        let (_d, db) = search_db_in_temp().await;
        let state = state_pool_with_tracked(&[1, 2]).await;
        let stats = gc_orphaned_files(&db, &state).await.unwrap();
        assert_eq!(stats.files_deleted, 0);
    }

    #[tokio::test]
    async fn empty_tracked_files_is_authoritative_and_clears_all() {
        let (_d, db) = search_db_in_temp().await;
        seed_file(&db, 1, "stale content").await;
        let state = state_pool_with_tracked(&[]).await;

        let stats = gc_orphaned_files(&db, &state).await.unwrap();

        assert_eq!(stats.files_deleted, 1);
        assert_eq!(count(db.pool(), "SELECT COUNT(*) FROM code_lines").await, 0);
        assert_eq!(
            count(db.pool(), "SELECT COUNT(*) FROM file_metadata").await,
            0
        );
    }

    #[tokio::test]
    async fn metadata_only_orphan_is_removed() {
        // fts5_skipped files have a file_metadata row but no code_lines —
        // the UNION must still surface them as orphans.
        let (_d, db) = search_db_in_temp().await;
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(42_i64)
            .bind("tenant-a")
            .bind("main")
            .bind("/repo/huge.csv")
            .bind("bp42")
            .bind("huge.csv")
            .bind("hash")
            .bind(123_i64)
            .bind(1_i64) // fts5_skipped
            .execute(db.pool())
            .await
            .unwrap();
        let state = state_pool_with_tracked(&[]).await;

        let stats = gc_orphaned_files(&db, &state).await.unwrap();

        assert_eq!(stats.files_deleted, 1);
        assert_eq!(
            count(db.pool(), "SELECT COUNT(*) FROM file_metadata").await,
            0
        );
    }
}
