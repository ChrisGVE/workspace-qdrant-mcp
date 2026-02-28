//! Migration v23: Create symbol co-occurrence table for concept graph.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V23Migration;

#[async_trait]
impl Migration for V23Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v23: Creating symbol_cooccurrence table");

        use crate::cooccurrence_schema::{
            CREATE_SYMBOL_COOCCURRENCE_SQL, CREATE_COOCCURRENCE_INDEXES_SQL,
        };

        sqlx::query(CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_COOCCURRENCE_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v23 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 23 }
    fn description(&self) -> &'static str { "Create symbol_cooccurrence table" }
}
