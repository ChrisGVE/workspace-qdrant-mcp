//! Migration v39: create `taxonomy_cache` table for Tier 2 embedding-based tagging.
//!
//! Tier 2 tagging embeds ~180 taxonomy concept terms at startup and computes
//! cosine similarity against document aggregate embeddings. Embedding all
//! terms at each startup is expensive (~5s on CPU), so this migration creates
//! a persistent cache keyed by a SHA-256 hash of the taxonomy YAML content.
//! When the taxonomy YAML changes, the hash changes and the cache is
//! invalidated (stale rows deleted).

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V39Migration;

/// DDL for the taxonomy embedding cache table.
///
/// - `taxonomy_hash`: SHA-256 of the full taxonomy YAML content (cache key).
/// - `term_label`: the taxonomy term text (e.g. "rust programming").
/// - `category`: the category this term belongs to (e.g. "programming-languages").
/// - `embedding`: the 384-dim `f32` vector serialised as little-endian bytes (BLOB).
/// - `created_at`: ISO 8601 timestamp with `Z` suffix.
pub const CREATE_TAXONOMY_CACHE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS taxonomy_cache (
    taxonomy_hash TEXT NOT NULL,
    term_label    TEXT NOT NULL,
    category      TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (taxonomy_hash, term_label)
)
"#;

#[async_trait]
impl Migration for V39Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v39: create taxonomy_cache table for Tier 2 tagging");

        let mut conn = pool.acquire().await?;

        // Idempotent: CREATE TABLE IF NOT EXISTS
        conn.execute(CREATE_TAXONOMY_CACHE_SQL).await?;

        debug!("Migration v39: taxonomy_cache table created");
        info!("Migration v39 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        39
    }

    fn description(&self) -> &'static str {
        "Create taxonomy_cache table for persistent Tier 2 taxonomy embedding cache"
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

        // Run all prior migrations so the schema is up to date
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    #[tokio::test]
    async fn v39_creates_taxonomy_cache_table() {
        let pool = setup_pool().await;

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy_cache')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert!(exists, "taxonomy_cache table should exist after v39");
    }

    #[tokio::test]
    async fn v39_is_idempotent() {
        let pool = setup_pool().await;

        // Run v39 again -- should not error
        let migration = V39Migration;
        let result = migration.up(&pool).await;
        assert!(result.is_ok(), "v39 should be idempotent");
    }

    #[tokio::test]
    async fn v39_taxonomy_cache_accepts_rows() {
        let pool = setup_pool().await;

        // Insert a test row
        let embedding_bytes: Vec<u8> = vec![0u8; 384 * 4]; // 384 f32s as bytes
        let result = sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES (?1, ?2, ?3, ?4)",
        )
        .bind("abc123hash")
        .bind("rust programming")
        .bind("programming-languages")
        .bind(&embedding_bytes)
        .execute(&pool)
        .await;

        assert!(result.is_ok(), "Should accept valid taxonomy_cache row");

        // Verify roundtrip
        let (label, category): (String, String) = sqlx::query_as(
            "SELECT term_label, category FROM taxonomy_cache WHERE taxonomy_hash = 'abc123hash'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(label, "rust programming");
        assert_eq!(category, "programming-languages");
    }

    #[tokio::test]
    async fn v39_primary_key_enforces_uniqueness() {
        let pool = setup_pool().await;

        let embedding_bytes: Vec<u8> = vec![0u8; 384 * 4];

        // Insert first row
        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('hash1', 'term1', 'cat1', ?1)",
        )
        .bind(&embedding_bytes)
        .execute(&pool)
        .await
        .unwrap();

        // Insert duplicate primary key -- should fail
        let dup = sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('hash1', 'term1', 'cat1', ?1)",
        )
        .bind(&embedding_bytes)
        .execute(&pool)
        .await;

        assert!(
            dup.is_err(),
            "Duplicate (taxonomy_hash, term_label) should be rejected"
        );
    }

    #[tokio::test]
    async fn v39_different_hash_same_term_allowed() {
        let pool = setup_pool().await;

        let embedding_bytes: Vec<u8> = vec![0u8; 384 * 4];

        // Same term, different taxonomy_hash -- should both succeed
        sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('hashA', 'term1', 'cat1', ?1)",
        )
        .bind(&embedding_bytes)
        .execute(&pool)
        .await
        .unwrap();

        let result = sqlx::query(
            "INSERT INTO taxonomy_cache (taxonomy_hash, term_label, category, embedding) \
             VALUES ('hashB', 'term1', 'cat1', ?1)",
        )
        .bind(&embedding_bytes)
        .execute(&pool)
        .await;

        assert!(
            result.is_ok(),
            "Different taxonomy_hash should allow same term_label"
        );
    }
}
