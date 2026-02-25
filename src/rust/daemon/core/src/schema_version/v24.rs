//! Migration v24: Create project_groups and project_dependencies tables.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V24Migration;

#[async_trait]
impl Migration for V24Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v24: Creating project_groups and project_dependencies tables");

        use crate::project_groups_schema::{
            CREATE_PROJECT_GROUPS_SQL, CREATE_PROJECT_GROUPS_INDEXES_SQL,
        };
        use crate::dependency_grouper::{
            CREATE_PROJECT_DEPENDENCIES_SQL, CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL,
        };

        sqlx::query(CREATE_PROJECT_GROUPS_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        sqlx::query(CREATE_PROJECT_DEPENDENCIES_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v24 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 24 }
    fn description(&self) -> &'static str { "Create project_groups and project_dependencies tables" }
}
