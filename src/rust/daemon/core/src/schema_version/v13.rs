//! Migration v13: Create resolution_events table.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V13Migration;

#[async_trait]
impl Migration for V13Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v13: Creating resolution_events table");

        use crate::resolution_events_schema::{
            CREATE_RESOLUTION_EVENTS_INDEXES_SQL, CREATE_RESOLUTION_EVENTS_SQL,
        };

        sqlx::query(CREATE_RESOLUTION_EVENTS_SQL)
            .execute(pool)
            .await?;

        for index_sql in CREATE_RESOLUTION_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v13 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        13
    }
    fn description(&self) -> &'static str {
        "Create resolution_events table"
    }
}
