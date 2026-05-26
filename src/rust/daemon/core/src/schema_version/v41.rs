//! Migration v41: add `model_name` column to `taxonomy_cache` table.
//!
//! When the embedding model changes, cached taxonomy embeddings are invalid.
//! Adding `model_name` to the primary key ensures cache misses on model switch
//! without requiring a full table purge. Requires table reconstruction since
//! SQLite cannot ALTER a PRIMARY KEY.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V41Migration;

const RECREATE_SQL: &str = r#"
CREATE TABLE taxonomy_cache_new (
    taxonomy_hash TEXT NOT NULL,
    term_label    TEXT NOT NULL,
    category      TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    model_name    TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (taxonomy_hash, model_name, term_label)
)
"#;

#[async_trait]
impl Migration for V41Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v41: add model_name to taxonomy_cache (table reconstruction)");

        let mut conn = pool.acquire().await?;

        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy_cache')",
        )
        .fetch_one(&mut *conn)
        .await?;

        if !table_exists {
            debug!("Migration v41: taxonomy_cache does not exist, creating with model_name");
            conn.execute(
                RECREATE_SQL
                    .replace("taxonomy_cache_new", "taxonomy_cache")
                    .as_str(),
            )
            .await?;
            info!("Migration v41 complete (fresh table)");
            return Ok(());
        }

        conn.execute("BEGIN IMMEDIATE").await?;

        conn.execute(RECREATE_SQL).await?;

        conn.execute(
            "INSERT INTO taxonomy_cache_new (taxonomy_hash, term_label, category, embedding, created_at) \
             SELECT taxonomy_hash, term_label, category, embedding, created_at FROM taxonomy_cache",
        )
        .await?;

        conn.execute("DROP TABLE taxonomy_cache").await?;

        conn.execute("ALTER TABLE taxonomy_cache_new RENAME TO taxonomy_cache")
            .await?;

        conn.execute("COMMIT").await?;

        debug!("Migration v41: taxonomy_cache reconstructed with model_name in PK");
        info!("Migration v41 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        41
    }

    fn description(&self) -> &'static str {
        "Add model_name column to taxonomy_cache for model-aware cache invalidation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    #[tokio::test]
    async fn v41_adds_model_name_column() {
        let pool = setup_pool().await;

        let result: Result<Option<(String,)>, _> =
            sqlx::query_as("SELECT model_name FROM taxonomy_cache LIMIT 0")
                .fetch_optional(&pool)
                .await;
        assert!(result.is_ok(), "model_name column should exist");
    }

    #[tokio::test]
    async fn v41_default_model_name_applied() {
        let pool = setup_pool().await;

        let emb_bytes: Vec<u8> = vec![0u8; 384 * 4];
        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('hash1', 'term1', 'cat1', ?1)",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await
        .unwrap();

        let model: String = sqlx::query_scalar(
            "SELECT model_name FROM taxonomy_cache WHERE taxonomy_hash = 'hash1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(model, "all-MiniLM-L6-v2");
    }

    #[tokio::test]
    async fn v41_new_pk_allows_different_models() {
        let pool = setup_pool().await;

        let emb_bytes: Vec<u8> = vec![0u8; 384 * 4];

        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding, model_name) \
             VALUES ('hash1', 'term1', 'cat1', ?1, 'model-a')",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await
        .unwrap();

        let result = sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding, model_name) \
             VALUES ('hash1', 'term1', 'cat1', ?1, 'model-b')",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await;
        assert!(
            result.is_ok(),
            "Different model_name should allow same term"
        );
    }

    #[tokio::test]
    async fn v41_pk_rejects_full_duplicate() {
        let pool = setup_pool().await;

        let emb_bytes: Vec<u8> = vec![0u8; 384 * 4];

        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding, model_name) \
             VALUES ('hash1', 'term1', 'cat1', ?1, 'model-a')",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await
        .unwrap();

        let dup = sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding, model_name) \
             VALUES ('hash1', 'term1', 'cat1', ?1, 'model-a')",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await;
        assert!(
            dup.is_err(),
            "Duplicate (hash, model, term) should be rejected"
        );
    }

    #[tokio::test]
    async fn v41_preserves_existing_data() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Run migrations up to v40 only
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.ok();

        // Manually revert version to 40 to test data preservation
        sqlx::query("DELETE FROM schema_version WHERE version = 41")
            .execute(&pool)
            .await
            .ok();

        let emb_bytes: Vec<u8> = vec![1u8; 384 * 4];
        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('old_hash', 'old_term', 'old_cat', ?1)",
        )
        .bind(&emb_bytes)
        .execute(&pool)
        .await
        .unwrap();

        // Run v41 migration
        let migration = V41Migration;
        migration.up(&pool).await.unwrap();

        let (term, model): (String, String) = sqlx::query_as(
            "SELECT term_label, model_name FROM taxonomy_cache WHERE taxonomy_hash = 'old_hash'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(term, "old_term");
        assert_eq!(model, "all-MiniLM-L6-v2");
    }
}
