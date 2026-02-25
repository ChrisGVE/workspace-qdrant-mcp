//! Migration v3: Add needs_reconcile columns to tracked_files.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::SchemaError;
use super::migration::Migration;

pub struct V03Migration;

#[async_trait]
impl Migration for V03Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v3: Adding needs_reconcile columns to tracked_files");

        use crate::tracked_files_schema::{MIGRATE_V3_SQL, CREATE_RECONCILE_INDEX_SQL};

        let has_reconcile: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
        )
        .fetch_one(pool).await?;

        if !has_reconcile {
            for alter_sql in MIGRATE_V3_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("needs_reconcile columns already exist, skipping ALTER TABLE");
        }

        debug!("Creating reconcile index");
        sqlx::query(CREATE_RECONCILE_INDEX_SQL)
            .execute(pool).await?;

        info!("Migration v3 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 3 }
    fn description(&self) -> &'static str { "Add needs_reconcile columns to tracked_files" }
}
