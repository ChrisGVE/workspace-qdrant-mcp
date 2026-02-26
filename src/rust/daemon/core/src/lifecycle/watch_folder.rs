//! Watch Folder Lifecycle State Machine
//!
//! Centralizes ALL `watch_folders.is_active` mutations into a single module.
//! Every activation or deactivation of a watch folder **must** go through
//! [`WatchFolderLifecycle`] — direct SQL against `is_active` is forbidden
//! elsewhere in the codebase.
//!
//! ## Mutation Surface
//!
//! | Caller                  | Method                          |
//! |-------------------------|---------------------------------|
//! | DaemonStateManager      | `activate_project_group`        |
//! | DaemonStateManager      | `deactivate_project_group`      |
//! | PriorityManager         | `register_session`              |
//! | PriorityManager         | `unregister_session`            |
//! | PriorityManager         | `set_priority`                  |
//! | PriorityManager         | `cleanup_orphaned_sessions`     |
//! | SystemService (gRPC)    | `set_server_state`              |
//! | startup::reconciliation  | `validate_watch_folders`        |

use sqlx::SqlitePool;
use tracing::info;
use wqm_common::timestamps;

/// Errors originating from watch folder lifecycle transitions.
#[derive(Debug, thiserror::Error)]
pub enum WatchFolderLifecycleError {
    /// An underlying SQLite operation failed.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// The targeted watch folder does not exist.
    #[error("watch folder not found: {0}")]
    NotFound(String),
}

/// Centralised state machine for `watch_folders.is_active` mutations.
///
/// All callers that previously wrote `is_active` directly now delegate to one
/// of the methods on this struct. This ensures every transition is logged and
/// occurs through a single code path.
#[derive(Clone, Debug)]
pub struct WatchFolderLifecycle {
    pool: SqlitePool,
}

impl WatchFolderLifecycle {
    /// Create a new lifecycle manager backed by the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    // ── project-group operations (recursive CTE) ─────────────────────

    /// Activate a project **and** all descendant submodules.
    ///
    /// Uses a recursive CTE over `watch_folder_submodules` so that every
    /// child inherits the activation.
    pub async fn activate_project_group(
        &self,
        watch_id: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE descendants AS (
                SELECT ?1 AS watch_id
                UNION
                SELECT j.child_watch_id FROM watch_folder_submodules j
                JOIN descendants d ON j.parent_watch_id = d.watch_id
            )
            UPDATE watch_folders
            SET is_active = 1,
                last_activity_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM descendants)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: activate_project_group watch_id={} \
             -> is_active=1 ({} rows)",
            watch_id, rows
        );
        Ok(rows)
    }

    /// Deactivate a project **and** all descendant submodules.
    pub async fn deactivate_project_group(
        &self,
        watch_id: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let result = sqlx::query(
            r#"
            WITH RECURSIVE descendants AS (
                SELECT ?1 AS watch_id
                UNION
                SELECT j.child_watch_id FROM watch_folder_submodules j
                JOIN descendants d ON j.parent_watch_id = d.watch_id
            )
            UPDATE watch_folders
            SET is_active = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE watch_id IN (SELECT watch_id FROM descendants)
            "#,
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: deactivate_project_group watch_id={} \
             -> is_active=0 ({} rows)",
            watch_id, rows
        );
        Ok(rows)
    }

    // ── tenant-level operations ──────────────────────────────────────

    /// Activate all watch folders for a `(tenant_id, collection)` pair.
    ///
    /// Used by `PriorityManager::register_session` and `set_priority("high")`.
    pub async fn activate_by_tenant(
        &self,
        tenant_id: &str,
        collection: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let now = timestamps::format_utc(&chrono::Utc::now());
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_active = 1,
                last_activity_at = ?1,
                updated_at = ?1
            WHERE tenant_id = ?2
              AND collection = ?3
            "#,
        )
        .bind(&now)
        .bind(tenant_id)
        .bind(collection)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: tenant_id={} collection={} \
             -> is_active=1 ({} rows)",
            tenant_id, collection, rows
        );
        Ok(rows)
    }

    /// Deactivate all watch folders for a `(tenant_id, collection)` pair.
    ///
    /// Used by `PriorityManager::unregister_session` and `set_priority("normal")`.
    pub async fn deactivate_by_tenant(
        &self,
        tenant_id: &str,
        collection: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_active = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE tenant_id = ?1
              AND collection = ?2
            "#,
        )
        .bind(tenant_id)
        .bind(collection)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: tenant_id={} collection={} \
             -> is_active=0 ({} rows)",
            tenant_id, collection, rows
        );
        Ok(rows)
    }

    /// Set `is_active` to a specific value for a `(tenant_id, collection)` pair.
    ///
    /// Used by `PriorityManager::set_priority` which maps `"high"` -> `true`,
    /// `"normal"` -> `false`.
    pub async fn set_active_by_tenant(
        &self,
        tenant_id: &str,
        collection: &str,
        active: bool,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let now = timestamps::format_utc(&chrono::Utc::now());
        let result = sqlx::query(
            r#"
            UPDATE watch_folders
            SET is_active = ?1,
                last_activity_at = ?2,
                updated_at = ?2
            WHERE tenant_id = ?3
              AND collection = ?4
            "#,
        )
        .bind(active as i32)
        .bind(&now)
        .bind(tenant_id)
        .bind(collection)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: tenant_id={} collection={} \
             -> is_active={} ({} rows)",
            tenant_id,
            collection,
            i32::from(active),
            rows
        );
        Ok(rows)
    }

    // ── path-level operations ────────────────────────────────────────

    /// Set `is_active` for a watch folder identified by its filesystem path.
    ///
    /// Used by the gRPC `SystemService::set_server_state` endpoint.
    pub async fn set_active_by_path(
        &self,
        path: &str,
        active: bool,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let now = timestamps::now_utc();
        let result = sqlx::query(
            "UPDATE watch_folders \
             SET is_active = ?1, last_activity_at = ?2, updated_at = ?3 \
             WHERE path = ?4",
        )
        .bind(active as i32)
        .bind(&now)
        .bind(&now)
        .bind(path)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: path={} -> is_active={} ({} rows)",
            path,
            i32::from(active),
            rows
        );
        Ok(rows)
    }

    // ── watch-id-level operations ────────────────────────────────────

    /// Deactivate a single watch folder by `watch_id`.
    ///
    /// Used by `startup::reconciliation::validate_watch_folders` when a
    /// path no longer exists on disk.
    pub async fn deactivate_by_watch_id(
        &self,
        watch_id: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        let result = sqlx::query(
            "UPDATE watch_folders SET is_active = 0, \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE watch_id = ?1",
        )
        .bind(watch_id)
        .execute(&self.pool)
        .await?;

        let rows = result.rows_affected();
        info!(
            "watch_folder lifecycle: deactivate watch_id={} \
             -> is_active=0 ({} rows)",
            watch_id, rows
        );
        Ok(rows)
    }

    // ── bulk operations ──────────────────────────────────────────────

    /// Deactivate multiple tenants inside a single transaction.
    ///
    /// Used by `PriorityManager::cleanup_orphaned_sessions` to atomically
    /// demote all stale projects.
    pub async fn deactivate_orphaned_tenants(
        &self,
        tenant_ids: &[String],
        collection: &str,
    ) -> Result<u64, WatchFolderLifecycleError> {
        if tenant_ids.is_empty() {
            return Ok(0);
        }

        let mut tx = self.pool.begin().await?;
        let mut total: u64 = 0;

        for tenant_id in tenant_ids {
            let result = sqlx::query(
                r#"
                UPDATE watch_folders
                SET is_active = 0,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE tenant_id = ?1
                  AND collection = ?2
                "#,
            )
            .bind(tenant_id)
            .bind(collection)
            .execute(&mut *tx)
            .await?;

            total += result.rows_affected();
        }

        tx.commit().await?;

        info!(
            "watch_folder lifecycle: deactivated {} orphaned tenants \
             ({} rows) in collection={}",
            tenant_ids.len(),
            total,
            collection
        );
        Ok(total)
    }

    /// Find tenant IDs whose heartbeat has gone stale.
    ///
    /// Returns the tenant IDs that are currently active but whose
    /// `last_activity_at` is older than `cutoff_iso`. This is a read-only
    /// helper -- the caller is responsible for deciding what to do with the
    /// result (typically calling [`Self::deactivate_orphaned_tenants`]).
    pub async fn find_stale_active_tenants(
        &self,
        collection: &str,
        cutoff_iso: &str,
    ) -> Result<Vec<String>, WatchFolderLifecycleError> {
        let rows = sqlx::query_scalar::<_, String>(
            r#"
            SELECT tenant_id
            FROM watch_folders
            WHERE is_active = 1
              AND collection = ?1
              AND last_activity_at IS NOT NULL
              AND last_activity_at < ?2
            "#,
        )
        .bind(collection)
        .bind(cutoff_iso)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }
}

#[cfg(test)]
#[path = "watch_folder_tests.rs"]
mod tests;
