//! Migration v1: Create initial spec-compliant tables (watch_folders, unified_queue).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V01Migration;

#[async_trait]
impl Migration for V01Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v1: Creating spec-compliant tables");

        use crate::unified_queue_schema::{
            CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
        };
        use crate::watch_folders_schema::{
            CREATE_WATCH_FOLDERS_INDEXES_SQL, CREATE_WATCH_FOLDERS_SQL,
        };

        debug!("Creating watch_folders table");
        sqlx::query(CREATE_WATCH_FOLDERS_SQL).execute(pool).await?;

        for index_sql in CREATE_WATCH_FOLDERS_INDEXES_SQL {
            debug!("Creating watch_folders index");
            sqlx::query(index_sql).execute(pool).await?;
        }

        debug!("Creating unified_queue table");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await?;

        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            debug!("Creating unified_queue index");
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v1 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        1
    }
    fn description(&self) -> &'static str {
        "Create initial spec-compliant tables"
    }
}
