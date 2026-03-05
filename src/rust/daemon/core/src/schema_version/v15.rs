//! Migration v15: Create sparse_vocabulary and corpus_statistics tables for BM25 IDF.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V15Migration;

#[async_trait]
impl Migration for V15Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v15: Creating BM25 IDF tables (sparse_vocabulary, corpus_statistics)");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sparse_vocabulary (
                term_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                collection TEXT NOT NULL,
                document_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (term_id, collection),
                UNIQUE (term, collection)
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sparse_vocabulary_collection ON sparse_vocabulary (collection)",
        )
        .execute(pool).await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS corpus_statistics (
                collection TEXT PRIMARY KEY NOT NULL,
                total_documents INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            "#,
        )
        .execute(pool)
        .await?;

        info!("Migration v15 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        15
    }
    fn description(&self) -> &'static str {
        "Create BM25 IDF tables"
    }
}
