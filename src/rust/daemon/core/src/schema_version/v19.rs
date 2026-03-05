//! Migration v19: Add base_point, relative_path, incremental columns to tracked_files.
//!
//! These columns support content-addressed identity for state integrity.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V19Migration;

#[async_trait]
impl Migration for V19Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v19: Adding base_point, relative_path, incremental to tracked_files");

        use crate::tracked_files_schema::{
            CREATE_BASE_POINT_INDEX_SQL, CREATE_REFCOUNT_INDEX_SQL, MIGRATE_V19_ADD_COLUMNS_SQL,
        };

        let has_base_point: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'base_point'",
        )
        .fetch_one(pool)
        .await?;

        if !has_base_point {
            for alter_sql in MIGRATE_V19_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("base_point column already exists, skipping ALTER TABLE");
        }

        // Backfill relative_path from file_path
        let backfilled: u64 = sqlx::query(
            "UPDATE tracked_files SET relative_path = file_path WHERE relative_path IS NULL",
        )
        .execute(pool)
        .await?
        .rows_affected();

        if backfilled > 0 {
            info!("Backfilled relative_path for {} rows", backfilled);
        }

        // Backfill base_point using Rust hash computation
        let rows: Vec<(i64, String, String, String, String)> = sqlx::query_as(
            "SELECT tf.file_id, wf.tenant_id, COALESCE(tf.branch, 'default'), tf.file_path, tf.file_hash
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
             WHERE tf.base_point IS NULL"
        )
        .fetch_all(pool).await?;

        if !rows.is_empty() {
            info!("Backfilling base_point for {} existing rows", rows.len());
            for (file_id, tenant_id, branch, file_path, file_hash) in &rows {
                let base_point = wqm_common::hashing::compute_base_point(
                    tenant_id, branch, file_path, file_hash,
                );
                sqlx::query("UPDATE tracked_files SET base_point = ?1 WHERE file_id = ?2")
                    .bind(&base_point)
                    .bind(file_id)
                    .execute(pool)
                    .await?;
            }
            info!("base_point backfill complete");
        }

        sqlx::query(CREATE_BASE_POINT_INDEX_SQL)
            .execute(pool)
            .await?;
        sqlx::query(CREATE_REFCOUNT_INDEX_SQL).execute(pool).await?;

        info!("Migration v19 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        19
    }
    fn description(&self) -> &'static str {
        "Add base_point and relative_path to tracked_files"
    }
}
