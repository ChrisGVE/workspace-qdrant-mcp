//! Migration v12: Create search_events table for pipeline instrumentation.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V12Migration;

#[async_trait]
impl Migration for V12Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v12: Creating search_events table");

        use crate::search_events_schema::{
            CREATE_SEARCH_EVENTS_SQL, CREATE_SEARCH_EVENTS_INDEXES_SQL,
        };

        sqlx::query(CREATE_SEARCH_EVENTS_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v12 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 12 }
    fn description(&self) -> &'static str { "Create search_events table" }
}
