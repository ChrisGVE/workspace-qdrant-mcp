//! Migration v8: Add extension and is_test columns to tracked_files with backfill.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V08Migration;

#[async_trait]
impl Migration for V08Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v8: Adding extension and is_test columns to tracked_files");

        use crate::file_classification::{get_extension_for_storage, is_test_file};
        use crate::tracked_files_schema::MIGRATE_V8_ADD_COLUMNS_SQL;
        use std::path::Path;

        let has_extension: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'extension'",
        )
        .fetch_one(pool)
        .await?;

        if !has_extension {
            for alter_sql in MIGRATE_V8_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("extension column already exists, skipping ALTER TABLE");
        }

        // Backfill extension and is_test for existing rows
        let rows: Vec<(i64, String)> =
            sqlx::query_as("SELECT rowid, file_path FROM tracked_files WHERE extension IS NULL")
                .fetch_all(pool)
                .await?;

        if !rows.is_empty() {
            info!(
                "Backfilling extension and is_test for {} existing rows",
                rows.len()
            );
            for (rowid, file_path) in &rows {
                let path = Path::new(file_path);
                let extension = get_extension_for_storage(path);
                let is_test = is_test_file(path);
                sqlx::query(
                    "UPDATE tracked_files SET extension = ?1, is_test = ?2 WHERE rowid = ?3",
                )
                .bind(extension.as_deref())
                .bind(is_test as i32)
                .bind(rowid)
                .execute(pool)
                .await?;
            }
            info!("Backfill complete");
        }

        info!("Migration v8 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        8
    }
    fn description(&self) -> &'static str {
        "Add extension and is_test columns to tracked_files"
    }
}
