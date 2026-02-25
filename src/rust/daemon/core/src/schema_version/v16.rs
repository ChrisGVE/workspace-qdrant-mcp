//! Migration v16: Create keyword/tag extraction tables.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V16Migration;

#[async_trait]
impl Migration for V16Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v16: Creating keyword/tag extraction tables");

        use crate::keywords_schema::{
            CREATE_KEYWORDS_SQL, CREATE_KEYWORDS_INDEXES_SQL,
            CREATE_TAGS_SQL, CREATE_TAGS_INDEXES_SQL,
            CREATE_KEYWORD_BASKETS_SQL, CREATE_KEYWORD_BASKETS_INDEXES_SQL,
            CREATE_CANONICAL_TAGS_SQL, CREATE_CANONICAL_TAGS_INDEXES_SQL,
            CREATE_TAG_HIERARCHY_EDGES_SQL, CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL,
        };

        // Create tables in dependency order
        sqlx::query(CREATE_KEYWORDS_SQL).execute(pool).await?;
        sqlx::query(CREATE_TAGS_SQL).execute(pool).await?;
        sqlx::query(CREATE_KEYWORD_BASKETS_SQL).execute(pool).await?;
        sqlx::query(CREATE_CANONICAL_TAGS_SQL).execute(pool).await?;
        sqlx::query(CREATE_TAG_HIERARCHY_EDGES_SQL).execute(pool).await?;

        // Create all indexes
        for index_sql in CREATE_KEYWORDS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }
        for index_sql in CREATE_TAGS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }
        for index_sql in CREATE_KEYWORD_BASKETS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }
        for index_sql in CREATE_CANONICAL_TAGS_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }
        for index_sql in CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v16 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 16 }
    fn description(&self) -> &'static str { "Create keyword/tag extraction tables" }
}
