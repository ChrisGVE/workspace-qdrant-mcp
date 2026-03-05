//! Migration v10: Fix stale file_type='test' rows in tracked_files.
//!
//! Before v8, some code paths wrote file_type='test' instead of using
//! classify_file_type() + is_test. Reclassify using canonical logic.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V10Migration;

#[async_trait]
impl Migration for V10Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v10: Fixing stale file_type='test' rows in tracked_files");

        use crate::file_classification::{classify_file_type, is_test_file};
        use std::path::Path;

        let rows: Vec<(i64, String)> =
            sqlx::query_as("SELECT rowid, file_path FROM tracked_files WHERE file_type = 'test'")
                .fetch_all(pool)
                .await?;

        if rows.is_empty() {
            info!("No stale file_type='test' rows found");
        } else {
            info!("Reclassifying {} rows with file_type='test'", rows.len());
            for (rowid, file_path) in &rows {
                let path = Path::new(file_path);
                let file_type = classify_file_type(path);
                let is_test = is_test_file(path);
                sqlx::query(
                    "UPDATE tracked_files SET file_type = ?1, is_test = ?2 WHERE rowid = ?3",
                )
                .bind(file_type.as_str())
                .bind(is_test as i32)
                .bind(rowid)
                .execute(pool)
                .await?;
            }
            info!("Reclassification complete");
        }

        info!("Migration v10 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        10
    }
    fn description(&self) -> &'static str {
        "Fix stale file_type=test rows"
    }
}
