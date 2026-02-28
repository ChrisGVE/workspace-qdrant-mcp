//! Migration v25: Create project_embeddings and affinity_labels tables.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V25Migration;

#[async_trait]
impl Migration for V25Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v25: Creating project_embeddings and affinity_labels tables");

        use crate::affinity_grouper::{
            CREATE_PROJECT_EMBEDDINGS_SQL, CREATE_AFFINITY_LABELS_SQL,
        };

        sqlx::query(CREATE_PROJECT_EMBEDDINGS_SQL)
            .execute(pool).await?;

        sqlx::query(CREATE_AFFINITY_LABELS_SQL)
            .execute(pool).await?;

        info!("Migration v25 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 25 }
    fn description(&self) -> &'static str { "Create project_embeddings and affinity_labels tables" }
}
