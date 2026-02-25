//! Migration v2: Create tracked_files and qdrant_chunks tables.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::SchemaError;
use super::migration::Migration;

pub struct V02Migration;

#[async_trait]
impl Migration for V02Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v2: Creating tracked_files and qdrant_chunks tables");

        use crate::tracked_files_schema::{
            CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL,
            CREATE_QDRANT_CHUNKS_SQL, CREATE_QDRANT_CHUNKS_INDEXES_SQL,
        };

        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool).await?;

        debug!("Creating tracked_files table");
        sqlx::query(CREATE_TRACKED_FILES_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_TRACKED_FILES_INDEXES_SQL {
            debug!("Creating tracked_files index");
            sqlx::query(index_sql).execute(pool).await?;
        }

        debug!("Creating qdrant_chunks table");
        sqlx::query(CREATE_QDRANT_CHUNKS_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
            debug!("Creating qdrant_chunks index");
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v2 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 2 }
    fn description(&self) -> &'static str { "Create tracked_files and qdrant_chunks tables" }
}
