//! SQL execution for LibraryWriteService commands.

use wqm_common::timestamps;

use super::actor::WriteActor;
use super::commands::*;

impl WriteActor {
    pub(super) async fn exec_add_library(
        &self,
        data: AddLibraryData,
    ) -> WriteResult<AddLibraryResult> {
        let watch_id = format!("lib-{}", data.tag);
        let now = timestamps::now_utc();

        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| format!("database error: {}", e))?;

        if exists > 0 {
            return Ok(AddLibraryResult {
                success: false,
                watch_id,
                message: format!("Library '{}' already exists", data.tag),
            });
        }

        let path_exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE path = ?1")
                .bind(&data.path)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| format!("database error: {}", e))?;

        if path_exists > 0 {
            return Ok(AddLibraryResult {
                success: false,
                watch_id,
                message: format!("Path '{}' is already registered", data.path),
            });
        }

        sqlx::query(
            "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
              follow_symlinks, cleanup_on_disable, created_at, updated_at) \
             VALUES (?1, ?2, 'libraries', ?3, ?4, 0, 0, 0, 0, ?5, ?5)",
        )
        .bind(&watch_id)
        .bind(&data.path)
        .bind(&data.tag)
        .bind(&data.mode)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("database error: {}", e))?;

        Ok(AddLibraryResult {
            success: true,
            watch_id,
            message: format!("Library '{}' added (not watching yet)", data.tag),
        })
    }

    pub(super) async fn exec_remove_library(
        &self,
        data: RemoveLibraryData,
    ) -> WriteResult<RemoveLibraryResult> {
        let watch_id = format!("lib-{}", data.tag);

        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| format!("database error: {}", e))?;

        if exists == 0 {
            return Err(format!("library '{}' not found", data.tag));
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("transaction error: {}", e))?;

        // Delete children first (dependency order) to respect FK constraints
        let cancelled = sqlx::query(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND collection = 'libraries'",
        )
        .bind(&data.tag)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("database error: {}", e))?
        .rows_affected() as u32;

        let components = sqlx::query("DELETE FROM project_components WHERE watch_folder_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("database error: {}", e))?
            .rows_affected() as u32;

        let tracked = sqlx::query("DELETE FROM tracked_files WHERE watch_folder_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("database error: {}", e))?
            .rows_affected() as u32;

        // Parent deleted last — all children already removed
        sqlx::query("DELETE FROM watch_folders WHERE watch_id = ?1")
            .bind(&watch_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("database error: {}", e))?;

        tx.commit()
            .await
            .map_err(|e| format!("commit error: {}", e))?;

        Ok(RemoveLibraryResult {
            success: true,
            queue_items_cancelled: cancelled,
            tracked_files_deleted: tracked,
            components_deleted: components,
            message: format!("Library '{}' removed", data.tag),
        })
    }

    pub(super) async fn exec_watch_library(
        &self,
        data: WatchLibraryData,
    ) -> WriteResult<WatchLibraryResult> {
        let watch_id = format!("lib-{}", data.tag);
        let now = timestamps::now_utc();

        let exists =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE watch_id = ?1")
                .bind(&watch_id)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| format!("database error: {}", e))?;

        if exists > 0 {
            sqlx::query(
                "UPDATE watch_folders SET enabled = 1, library_mode = ?1, path = ?2, \
                 updated_at = ?3, last_activity_at = ?3 WHERE watch_id = ?4",
            )
            .bind(&data.mode)
            .bind(&data.path)
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

            Ok(WatchLibraryResult {
                success: true,
                is_new: false,
                watch_id,
                message: format!("Library '{}' watching enabled", data.tag),
            })
        } else {
            sqlx::query(
                "INSERT INTO watch_folders \
                 (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
                  follow_symlinks, cleanup_on_disable, created_at, updated_at, last_activity_at) \
                 VALUES (?1, ?2, 'libraries', ?3, ?4, 1, 0, 0, 0, ?5, ?5, ?5)",
            )
            .bind(&watch_id)
            .bind(&data.path)
            .bind(&data.tag)
            .bind(&data.mode)
            .bind(&now)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

            Ok(WatchLibraryResult {
                success: true,
                is_new: true,
                watch_id,
                message: format!("Library '{}' added and watching enabled", data.tag),
            })
        }
    }

    pub(super) async fn exec_unwatch_library(&self, data: UnwatchLibraryData) -> WriteResult<u32> {
        let watch_id = format!("lib-{}", data.tag);
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

    pub(super) async fn exec_configure_library(
        &self,
        data: ConfigureLibraryData,
    ) -> WriteResult<u32> {
        if data.enable == Some(true) && data.disable == Some(true) {
            return Err("cannot enable and disable a library simultaneously".into());
        }

        let watch_id = format!("lib-{}", data.tag);
        let now = timestamps::now_utc();
        let mut affected = 0u32;

        if let Some(ref mode) = data.mode {
            let result = sqlx::query(
                "UPDATE watch_folders SET library_mode = ?1, updated_at = ?2 WHERE watch_id = ?3",
            )
            .bind(mode)
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;
            affected += result.rows_affected() as u32;
        }

        if data.enable == Some(true) {
            let result = sqlx::query(
                "UPDATE watch_folders SET enabled = 1, updated_at = ?1 WHERE watch_id = ?2",
            )
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;
            affected += result.rows_affected() as u32;
        }

        if data.disable == Some(true) {
            let result = sqlx::query(
                "UPDATE watch_folders SET enabled = 0, updated_at = ?1 WHERE watch_id = ?2",
            )
            .bind(&now)
            .bind(&watch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;
            affected += result.rows_affected() as u32;
        }

        Ok(affected)
    }

    pub(super) async fn exec_set_incremental(
        &self,
        data: SetIncrementalData,
    ) -> WriteResult<SetIncrementalResult> {
        let value: i32 = if data.clear { 0 } else { 1 };
        let now = timestamps::now_utc();
        let mut updated = 0u32;
        let mut not_found = 0u32;

        for path in &data.file_paths {
            // `path` is a relative content path validated by the gRPC layer
            // (extract_relative_paths! macro). Match directly against
            // tracked_files.relative_path (the column is relative post-v37).
            let result = sqlx::query(
                "UPDATE tracked_files \
                 SET incremental = ?1, updated_at = ?2 \
                 WHERE relative_path = ?3",
            )
            .bind(value)
            .bind(&now)
            .bind(path)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("database error: {}", e))?;

            if result.rows_affected() > 0 {
                updated += 1;
            } else {
                not_found += 1;
            }
        }

        Ok(SetIncrementalResult { updated, not_found })
    }
}
