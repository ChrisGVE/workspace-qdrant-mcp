//! Migration v5: Create metrics_history table for time-series storage.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::SchemaError;
use super::migration::Migration;

pub struct V05Migration;

#[async_trait]
impl Migration for V05Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v5: Creating metrics_history table");

        use crate::metrics_history_schema::{
            CREATE_METRICS_HISTORY_SQL, CREATE_METRICS_HISTORY_INDEXES_SQL,
        };

        debug!("Creating metrics_history table");
        sqlx::query(CREATE_METRICS_HISTORY_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_METRICS_HISTORY_INDEXES_SQL {
            debug!("Creating metrics_history index");
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v5 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 5 }
    fn description(&self) -> &'static str { "Create metrics_history table" }
}
