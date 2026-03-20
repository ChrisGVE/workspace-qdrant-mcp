//! SQL execution for WatchWriteService commands.

use sqlx::SqlitePool;
use wqm_common::timestamps;

use super::actor::WriteActor;
use super::commands::*;

impl WriteActor {
    pub(super) async fn exec_pause_watchers(&self) -> WriteResult<u32> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 1, \
             pause_start_time = ?1, updated_at = ?1 \
             WHERE enabled = 1 AND is_paused = 0",
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }

    pub(super) async fn exec_resume_watchers(&self) -> WriteResult<u32> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 0, \
             pause_start_time = NULL, updated_at = ?1 \
             WHERE enabled = 1 AND is_paused = 1",
        )
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }

    pub(super) async fn exec_enable_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        let watch_id = resolve_watch_id(&self.pool, &data.watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET enabled = 1, updated_at = ?1 WHERE watch_id = ?2",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }

    pub(super) async fn exec_disable_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        let watch_id = resolve_watch_id(&self.pool, &data.watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET enabled = 0, updated_at = ?1 WHERE watch_id = ?2",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }

    pub(super) async fn exec_archive_watch(
        &self,
        data: ArchiveWatchData,
    ) -> WriteResult<ArchiveWatchResult> {
        let watch_id = resolve_watch_id(&self.pool, &data.watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
             WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        let affected = result.rows_affected() as u32;
        if affected == 0 {
            return Ok(ArchiveWatchResult {
                affected_count: 0,
                submodules_archived: 0,
                submodules_skipped: 0,
            });
        }

        let (archived, skipped) = if data.cascade_submodules {
            archive_submodules_safely(&self.pool, &watch_id, &now).await?
        } else {
            (0, 0)
        };

        Ok(ArchiveWatchResult {
            affected_count: affected,
            submodules_archived: archived,
            submodules_skipped: skipped,
        })
    }

    pub(super) async fn exec_unarchive_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        let watch_id = resolve_watch_id(&self.pool, &data.watch_id).await?;
        let now = timestamps::now_utc();

        let result = sqlx::query(
            "UPDATE watch_folders SET is_archived = 0, updated_at = ?1 \
             WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 1",
        )
        .bind(&now)
        .bind(&watch_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(result.rows_affected() as u32)
    }
}

/// Resolve a watch_id that may be an ID or a filesystem path.
async fn resolve_watch_id(pool: &SqlitePool, input: &str) -> WriteResult<String> {
    let exists =
        sqlx::query_scalar::<_, String>("SELECT watch_id FROM watch_folders WHERE watch_id = ?1")
            .bind(input)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

    if let Some(id) = exists {
        return Ok(id);
    }

    let prefix = format!("{}%", input);
    let matches = sqlx::query_scalar::<_, String>(
        "SELECT watch_id FROM watch_folders WHERE watch_id LIKE ?1",
    )
    .bind(&prefix)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("database error: {}", e))?;

    if matches.len() == 1 {
        return Ok(matches.into_iter().next().unwrap());
    }

    let canonical = std::path::Path::new(input)
        .canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| input.to_string());

    let by_path =
        sqlx::query_scalar::<_, String>("SELECT watch_id FROM watch_folders WHERE path = ?1")
            .bind(&canonical)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

    by_path.ok_or_else(|| format!("watch folder not found: {}", input))
}

/// Archive submodules with cross-reference safety checks.
async fn archive_submodules_safely(
    pool: &SqlitePool,
    parent_watch_id: &str,
    now: &str,
) -> WriteResult<(u32, u32)> {
    let submodules = sqlx::query_as::<_, (String, Option<String>, Option<String>)>(
        "SELECT wf.watch_id, wf.remote_hash, wf.git_remote_url \
         FROM watch_folders wf \
         INNER JOIN watch_folder_submodules j ON wf.watch_id = j.child_watch_id \
         WHERE j.parent_watch_id = ?1",
    )
    .bind(parent_watch_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("database error: {}", e))?;

    let mut archived = 0u32;
    let mut skipped = 0u32;

    for (sub_id, remote_hash, git_remote_url) in &submodules {
        let rh = remote_hash.as_deref().unwrap_or("");
        let url = git_remote_url.as_deref().unwrap_or("");

        if rh.is_empty() && url.is_empty() {
            let result = sqlx::query(
                "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
                 WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
            )
            .bind(now)
            .bind(sub_id)
            .execute(pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;
            if result.rows_affected() > 0 {
                archived += 1;
            }
            continue;
        }

        let other_refs = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM watch_folders sub \
             WHERE sub.remote_hash = ?1 AND sub.git_remote_url = ?2 \
             AND sub.parent_watch_id != ?3 \
             AND COALESCE(sub.is_archived, 0) = 0 \
             AND EXISTS ( \
               SELECT 1 FROM watch_folders parent \
               WHERE parent.watch_id = sub.parent_watch_id \
               AND COALESCE(parent.is_archived, 0) = 0 \
             )",
        )
        .bind(rh)
        .bind(url)
        .bind(parent_watch_id)
        .fetch_one(pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        if other_refs > 0 {
            skipped += 1;
        } else {
            let result = sqlx::query(
                "UPDATE watch_folders SET is_archived = 1, updated_at = ?1 \
                 WHERE watch_id = ?2 AND COALESCE(is_archived, 0) = 0",
            )
            .bind(now)
            .bind(sub_id)
            .execute(pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;
            if result.rows_affected() > 0 {
                archived += 1;
            }
        }
    }

    Ok((archived, skipped))
}
