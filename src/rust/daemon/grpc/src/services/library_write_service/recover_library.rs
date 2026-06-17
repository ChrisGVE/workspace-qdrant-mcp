//! Library recover / re-point handler (issue #140).
//!
//! Located at
//! `src/rust/daemon/grpc/src/services/library_write_service/recover_library.rs`,
//! a submodule of `library_write_service.rs`. It implements `RecoverLibrary`:
//! re-pointing a library to a new source path.
//!
//! A library's `tenant_id` is its tag (see `exec_add_library`), so re-pointing
//! a library never changes tenancy — only paths move. The cascade is therefore
//! the path-rewrite half of the project recover: swap the stored `path` in
//! `watch_folders` (collection = `libraries`), prefix-rewrite the absolute path
//! columns in state.db and search.db, and prefix-rewrite the absolute path
//! payload of the library's Qdrant points. Relative paths are watch-root
//! relative and untouched.
//!
//! Unlike the other LibraryWriteService RPCs (which delegate to the WriteActor),
//! recover needs the state.db pool + Qdrant client directly; they are wired via
//! `LibraryWriteServiceImpl::with_recover_deps`. When they are absent the RPC
//! returns an internal error rather than silently doing nothing.
//!
//! # Atomicity
//!
//! state.db is the single transactional unit reconciled here: the watch-root
//! swap and the queue-path splice both live in `recover::repoint_path_state_db`,
//! which runs them in one transaction. search.db is a *separate* database file
//! on its own pool, so it cannot share that transaction; like the project
//! recover, the SQLite↔search.db↔Qdrant boundary is non-atomic and a failure
//! past the state.db commit is logged and safe to re-run (the boundary-anchored
//! prefix match makes a re-run a no-op).

use std::path::Path;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::SqlitePool;
use tonic::Status;
use tracing::{info, warn};

use workspace_qdrant_core::recover;
use workspace_qdrant_core::search_db::search_db_path_from_state;
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::paths::CanonicalPath;

use crate::proto::{RecoverLibraryRequest, RecoverLibraryResponse};

use super::LibraryWriteServiceImpl;

impl LibraryWriteServiceImpl {
    /// Handle the RecoverLibrary RPC (#140).
    pub(super) async fn handle_recover_library(
        &self,
        req: RecoverLibraryRequest,
    ) -> Result<RecoverLibraryResponse, Status> {
        if req.tag.is_empty() {
            return Err(Status::invalid_argument("tag cannot be empty"));
        }

        let db_pool = self.db_pool.as_ref().ok_or_else(|| {
            Status::internal("RecoverLibrary: service not configured with a database pool")
        })?;

        // Look up the library's stored source path (tenant_id == tag).
        let old_path: Option<String> = sqlx::query_scalar(
            "SELECT path FROM watch_folders \
             WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(&req.tag)
        .bind(COLLECTION_LIBRARIES)
        .fetch_optional(db_pool)
        .await
        .map_err(|e| Status::internal(format!("Database error: {e}")))?;

        let old_path =
            old_path.ok_or_else(|| Status::not_found(format!("Library not found: {}", req.tag)))?;

        // Resolve the new path: validate as canonical when provided, else keep.
        let new_path = match req.new_path.as_deref().filter(|p| !p.is_empty()) {
            Some(p) => CanonicalPath::from_user_input(p)
                .map_err(|e| Status::invalid_argument(format!("invalid new_path: {e}")))?
                .into_string(),
            None => old_path.clone(),
        };

        if new_path == old_path {
            return Ok(RecoverLibraryResponse {
                success: true,
                dry_run: req.dry_run,
                changed: false,
                old_path: old_path.clone(),
                new_path,
                sqlite_rows_updated: 0,
                qdrant_points_updated: 0,
                message: format!(
                    "Library '{}' already at {old_path} — nothing to recover",
                    req.tag
                ),
            });
        }

        if req.dry_run {
            self.recover_library_dry_run(&req.tag, &old_path, &new_path)
                .await
        } else {
            self.recover_library_apply(&req.tag, &old_path, &new_path)
                .await
        }
    }

    async fn recover_library_dry_run(
        &self,
        tag: &str,
        old_path: &str,
        new_path: &str,
    ) -> Result<RecoverLibraryResponse, Status> {
        let db_pool = self.db_pool.as_ref().expect("checked by caller");

        let mut sqlite_rows = recover::count_repoint_rows(db_pool, tag, old_path)
            .await
            .map_err(map_schema_err)?;

        // search.db absolute path rows that the rewrite would touch.
        if let Some(state_path) = state_db_path(db_pool).await {
            let search_path = search_db_path_from_state(Path::new(&state_path));
            if search_path.exists() {
                if let Ok(pool) = open_rw_pool(&search_path).await {
                    sqlite_rows +=
                        count_search_db_paths(&pool, tag, old_path)
                            .await
                            .map_err(|e| {
                                Status::internal(format!("search.db dry-run count failed: {e}"))
                            })?;
                    pool.close().await;
                }
            }
        }

        let mut qdrant_points = 0i64;
        if let Some(storage) = self.storage.as_ref() {
            let n = storage
                .rewrite_path_payload_prefix_by_tenant(
                    COLLECTION_LIBRARIES,
                    tag,
                    old_path,
                    new_path,
                    true,
                )
                .await
                .map_err(|e| Status::internal(format!("Qdrant dry-run failed: {e}")))?;
            qdrant_points += n as i64;
        }

        Ok(RecoverLibraryResponse {
            success: true,
            dry_run: true,
            changed: sqlite_rows > 0 || qdrant_points > 0,
            message: format!(
                "DRY RUN: would update {sqlite_rows} SQLite row(s) and {qdrant_points} \
                 Qdrant point(s); path {old_path} -> {new_path}"
            ),
            old_path: old_path.to_string(),
            new_path: new_path.to_string(),
            sqlite_rows_updated: clamp_i32(sqlite_rows),
            qdrant_points_updated: clamp_i32(qdrant_points),
        })
    }

    async fn recover_library_apply(
        &self,
        tag: &str,
        old_path: &str,
        new_path: &str,
    ) -> Result<RecoverLibraryResponse, Status> {
        let db_pool = self.db_pool.as_ref().expect("checked by caller");

        // state.db: swap watch root + splice queue paths (one transaction).
        let mut sqlite_rows = recover::repoint_path_state_db(db_pool, tag, old_path, new_path)
            .await
            .map_err(map_schema_err)?;

        // search.db absolute file paths.
        if let Some(state_path) = state_db_path(db_pool).await {
            let search_path = search_db_path_from_state(Path::new(&state_path));
            if search_path.exists() {
                match open_rw_pool(&search_path).await {
                    Ok(pool) => {
                        sqlite_rows +=
                            recover::rewrite_paths_search_db(&pool, tag, old_path, new_path)
                                .await
                                .map_err(map_schema_err)?;
                        pool.close().await;
                    }
                    Err(e) => warn!("RecoverLibrary: search.db open failed (skipped): {e}"),
                }
            }
        }

        // Qdrant: rewrite absolute path payload on the library's points.
        let mut qdrant_points = 0i64;
        if let Some(storage) = self.storage.as_ref() {
            let n = storage
                .rewrite_path_payload_prefix_by_tenant(
                    COLLECTION_LIBRARIES,
                    tag,
                    old_path,
                    new_path,
                    false,
                )
                .await
                .map_err(|e| Status::internal(format!("Qdrant repath failed: {e}")))?;
            qdrant_points += n as i64;
        }

        info!(
            tag = %tag,
            old_path = %old_path,
            new_path = %new_path,
            sqlite_rows,
            qdrant_points,
            "RecoverLibrary applied"
        );

        Ok(RecoverLibraryResponse {
            success: true,
            dry_run: false,
            changed: true,
            message: format!(
                "Recovered library '{tag}': path {old_path} -> {new_path}; \
                 {sqlite_rows} SQLite row(s) and {qdrant_points} Qdrant point(s) updated"
            ),
            old_path: old_path.to_string(),
            new_path: new_path.to_string(),
            sqlite_rows_updated: clamp_i32(sqlite_rows),
            qdrant_points_updated: clamp_i32(qdrant_points),
        })
    }
}

/// Count `file_metadata` rows whose absolute path begins with `old_prefix`.
/// Uses the same escaped, boundary-anchored `LIKE 'old_prefix/%'` as the
/// rewrite (`recover::rewrite_paths_search_db`), so dry-run and apply agree, and
/// propagates the DB error instead of swallowing it.
async fn count_search_db_paths(
    pool: &SqlitePool,
    tenant_id: &str,
    old_prefix: &str,
) -> Result<i64, sqlx::Error> {
    let pattern = format!("{}/%", recover::escape_like(old_prefix));
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM file_metadata \
         WHERE tenant_id = ?1 AND file_path LIKE ?2 ESCAPE '\\'",
    )
    .bind(tenant_id)
    .bind(pattern)
    .fetch_one(pool)
    .await
}

/// The state.db file path, via `pragma_database_list`.
async fn state_db_path(pool: &SqlitePool) -> Option<String> {
    sqlx::query_scalar("SELECT file FROM pragma_database_list WHERE name = 'main'")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .filter(|p: &String| !p.is_empty())
}

/// Open a read-write SQLite pool on an existing satellite database file.
///
/// WAL + a 5s `busy_timeout` so a concurrent writer yields a retried
/// `SQLITE_BUSY` instead of failing immediately. Built from
/// [`SqliteConnectOptions`] rather than a `sqlite://` URL string so a path with
/// special characters never needs URL-escaping.
async fn open_rw_pool(path: &Path) -> Result<SqlitePool, sqlx::Error> {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(Duration::from_secs(5));
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
}

/// Clamp an `i64` count into the proto `int32` field, saturating rather than
/// silently wrapping on the (practically unreachable) overflow.
fn clamp_i32(n: i64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn map_schema_err(e: workspace_qdrant_core::SchemaError) -> Status {
    Status::internal(format!("Recover cascade failed: {e}"))
}
