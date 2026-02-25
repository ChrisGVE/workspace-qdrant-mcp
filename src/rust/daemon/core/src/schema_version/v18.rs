//! Migration v18: Create indexed_content cache table for diff-based skip detection.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V18Migration;

#[async_trait]
impl Migration for V18Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v18: Creating indexed_content cache table");

        use crate::indexed_content_schema::{
            CREATE_INDEXED_CONTENT_SQL, CREATE_INDEXED_CONTENT_INDEXES_SQL,
        };

        sqlx::query(CREATE_INDEXED_CONTENT_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_INDEXED_CONTENT_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v18 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 18 }
    fn description(&self) -> &'static str { "Create indexed_content cache table" }
}
