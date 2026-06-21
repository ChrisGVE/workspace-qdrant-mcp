//! Migration v38: add `routing_reason` column to `tracked_files`.
//!
//! Format-based routing (Feature 3) stores why a file was routed to a
//! particular collection (e.g. `"format_based"` for PDF/EPUB/DOCX files
//! redirected from projects to libraries). The column is nullable because
//! the vast majority of files are not format-routed.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V38Migration;

#[async_trait]
impl Migration for V38Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v38: add routing_reason column to tracked_files");

        let mut conn = pool.acquire().await?;

        // Check if column already exists (idempotent for crash recovery)
        let table_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
        )
        .fetch_one(&mut *conn)
        .await?;

        if table_sql.contains("routing_reason") {
            debug!("Migration v38: routing_reason column already exists, skipping");
            return Ok(());
        }

        conn.execute("ALTER TABLE tracked_files ADD COLUMN routing_reason TEXT")
            .await?;

        debug!("Migration v38: added routing_reason column to tracked_files");
        info!("Migration v38 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        38
    }

    fn description(&self) -> &'static str {
        "Add routing_reason column to tracked_files for format-based routing traceability"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Clear v35 fail-point in case a concurrent test left it set
        crate::schema_version::v35::INJECT_FAILURE_AFTER_WATCH_FOLDERS
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // Migrate only THROUGH v38 — this test asserts the pre-v48 tracked_files
        // shape (its INSERTs omit the v48 NOT-NULL columns). The full chain
        // would land at v48 and reject those INSERTs.
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations_through(38).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn v38_adds_routing_reason_column() {
        let pool = setup_pool().await;

        // Verify tracked_files exists and has routing_reason
        let sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert!(
            sql.contains("routing_reason"),
            "tracked_files should have routing_reason column after v38"
        );
    }

    #[tokio::test]
    async fn v38_is_idempotent() {
        let pool = setup_pool().await;

        // Run v38 again — should not error
        let migration = V38Migration;
        let result = migration.up(&pool).await;
        assert!(result.is_ok(), "v38 should be idempotent");
    }

    #[tokio::test]
    async fn v38_routing_reason_is_nullable() {
        let pool = setup_pool().await;

        // Insert a tracked_file without routing_reason — should succeed.
        // Post-v40: the `branch` column no longer exists; use the v40
        // schema which has `primary_branch` and `branches` instead.
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, tenant_id, collection, path, created_at, updated_at)
             VALUES ('wf1', 'tenant1', 'projects', '/tmp/test', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let result = sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, relative_path, file_mtime, file_hash, created_at, updated_at)
             VALUES ('wf1', 'src/main.rs', '2024-01-01T00:00:00Z', 'abc123', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await;
        assert!(result.is_ok(), "NULL routing_reason should be accepted");

        // Insert with routing_reason set — should also succeed
        let result = sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, relative_path, file_mtime, file_hash, routing_reason, created_at, updated_at)
             VALUES ('wf1', 'docs/manual.pdf', '2024-01-01T00:00:00Z', 'def456', 'format_based', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await;
        assert!(result.is_ok(), "explicit routing_reason should be accepted");

        // Verify the value roundtrips
        let reason: Option<String> = sqlx::query_scalar(
            "SELECT routing_reason FROM tracked_files WHERE relative_path = 'docs/manual.pdf'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(reason.as_deref(), Some("format_based"));
    }
}
