//! Dynamic Lexicon Manager (Task 17).
//!
//! Manages per-collection BM25 vocabulary with SQLite persistence.
//! Provides document frequency lookup for the keyword extraction pipeline
//! and corpus statistics for IDF weighting.
//!
//! On startup, loads persisted vocabulary from `sparse_vocabulary` and
//! `corpus_statistics` tables. During processing, accumulates new terms
//! in memory and periodically flushes to SQLite.

use std::collections::HashMap;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::embedding::BM25;

/// Manages per-collection BM25 vocabulary with SQLite persistence.
pub struct LexiconManager {
    pool: SqlitePool,
    /// Per-collection BM25 instances (lazy loaded from SQLite).
    instances: Arc<RwLock<HashMap<String, BM25>>>,
    /// Tracks how many documents have been added since last persist.
    dirty_counts: Arc<RwLock<HashMap<String, u32>>>,
    /// BM25 k1 parameter for new instances.
    k1: f32,
}

/// Threshold for auto-persisting (number of documents processed since last persist).
const AUTO_PERSIST_THRESHOLD: u32 = 50;

impl LexiconManager {
    pub fn new(pool: SqlitePool, k1: f32) -> Self {
        Self {
            pool,
            instances: Arc::new(RwLock::new(HashMap::new())),
            dirty_counts: Arc::new(RwLock::new(HashMap::new())),
            k1,
        }
    }

    /// Load BM25 state from SQLite for a specific collection.
    ///
    /// Reads `sparse_vocabulary` and `corpus_statistics` tables.
    /// If no data exists, creates an empty BM25 instance.
    pub async fn load_collection(&self, collection: &str) -> Result<(), sqlx::Error> {
        let rows = sqlx::query_as::<_, (i64, String, i64)>(
            "SELECT term_id, term, document_count FROM sparse_vocabulary WHERE collection = ?1",
        )
        .bind(collection)
        .fetch_all(&self.pool)
        .await?;

        let mut vocab = HashMap::new();
        let mut doc_freq = HashMap::new();

        for (term_id, term, document_count) in rows {
            let tid = term_id as u32;
            vocab.insert(term, tid);
            doc_freq.insert(tid, document_count as u32);
        }

        let total_docs: Option<i64> = sqlx::query_scalar(
            "SELECT total_documents FROM corpus_statistics WHERE collection = ?1",
        )
        .bind(collection)
        .fetch_optional(&self.pool)
        .await?;

        let total = total_docs.unwrap_or(0) as u32;

        let bm25 = BM25::from_persisted(self.k1, vocab, doc_freq, total);
        info!(
            "Loaded lexicon for collection '{}': {} terms, {} documents",
            collection,
            bm25.vocab_size(),
            bm25.total_docs(),
        );

        let mut instances = self.instances.write().await;
        instances.insert(collection.to_string(), bm25);

        let mut dirty = self.dirty_counts.write().await;
        dirty.insert(collection.to_string(), 0);

        Ok(())
    }

    /// Ensure a collection's BM25 instance is loaded (lazy load from SQLite if needed).
    async fn ensure_loaded(&self, collection: &str) -> Result<(), sqlx::Error> {
        {
            let instances = self.instances.read().await;
            if instances.contains_key(collection) {
                return Ok(());
            }
        }
        self.load_collection(collection).await
    }

    /// Get document frequency for a term in a collection.
    ///
    /// Returns 0 if the term is not in the vocabulary or the collection isn't loaded.
    pub async fn document_frequency(&self, collection: &str, term: &str) -> u32 {
        if let Err(e) = self.ensure_loaded(collection).await {
            warn!("Failed to load lexicon for '{}': {}", collection, e);
            return 0;
        }
        let instances = self.instances.read().await;
        if let Some(bm25) = instances.get(collection) {
            if let Some(&term_id) = bm25.vocab().get(term) {
                bm25.doc_freq().get(&term_id).copied().unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Get corpus size (total documents) for a collection.
    pub async fn corpus_size(&self, collection: &str) -> u64 {
        if let Err(e) = self.ensure_loaded(collection).await {
            warn!("Failed to load lexicon for '{}': {}", collection, e);
            return 0;
        }
        let instances = self.instances.read().await;
        instances
            .get(collection)
            .map(|bm25| bm25.total_docs() as u64)
            .unwrap_or(0)
    }

    /// Add a document's tokens to the collection's vocabulary.
    ///
    /// Updates in-memory BM25 state. Call `persist()` or rely on auto-persist
    /// to flush changes to SQLite.
    pub async fn add_document(
        &self,
        collection: &str,
        tokens: &[String],
    ) -> Result<(), sqlx::Error> {
        self.ensure_loaded(collection).await?;

        {
            let mut instances = self.instances.write().await;
            if let Some(bm25) = instances.get_mut(collection) {
                bm25.add_document(tokens);
            }
        }

        // Track dirty count for auto-persist
        let should_persist = {
            let mut dirty = self.dirty_counts.write().await;
            let count = dirty.entry(collection.to_string()).or_insert(0);
            *count += 1;
            *count >= AUTO_PERSIST_THRESHOLD
        };

        if should_persist {
            debug!(
                "Auto-persisting lexicon for '{}' (threshold {} reached)",
                collection, AUTO_PERSIST_THRESHOLD
            );
            self.persist(collection).await?;
        }

        Ok(())
    }

    /// Persist current BM25 state for a collection to SQLite.
    ///
    /// Uses INSERT OR REPLACE to upsert vocabulary entries and corpus statistics.
    pub async fn persist(&self, collection: &str) -> Result<(), sqlx::Error> {
        let instances = self.instances.read().await;
        let bm25 = match instances.get(collection) {
            Some(b) => b,
            None => return Ok(()),
        };

        let vocab = bm25.vocab().clone();
        let doc_freq = bm25.doc_freq().clone();
        let total_docs = bm25.total_docs();
        drop(instances);

        let now = timestamps::now_utc();

        // Batch upsert vocabulary in a transaction
        let mut tx = self.pool.begin().await?;

        for (term, term_id) in &vocab {
            let df = doc_freq.get(term_id).copied().unwrap_or(0);
            sqlx::query(
                r#"INSERT INTO sparse_vocabulary (term_id, term, collection, document_count, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5)
                   ON CONFLICT (term, collection)
                   DO UPDATE SET document_count = ?4"#,
            )
            .bind(*term_id as i64)
            .bind(term)
            .bind(collection)
            .bind(df as i64)
            .bind(&now)
            .execute(&mut *tx)
            .await?;
        }

        // Upsert corpus statistics
        sqlx::query(
            r#"INSERT INTO corpus_statistics (collection, total_documents, updated_at)
               VALUES (?1, ?2, ?3)
               ON CONFLICT (collection)
               DO UPDATE SET total_documents = ?2, updated_at = ?3"#,
        )
        .bind(collection)
        .bind(total_docs as i64)
        .bind(&now)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        // Reset dirty counter
        let mut dirty = self.dirty_counts.write().await;
        dirty.insert(collection.to_string(), 0);

        debug!(
            "Persisted lexicon for '{}': {} terms, {} total docs",
            collection,
            vocab.len(),
            total_docs,
        );

        Ok(())
    }

    /// Persist all loaded collections (call on shutdown or periodically).
    pub async fn persist_all(&self) -> Result<(), sqlx::Error> {
        let collections: Vec<String> = {
            let instances = self.instances.read().await;
            instances.keys().cloned().collect()
        };
        for collection in collections {
            if let Err(e) = self.persist(&collection).await {
                warn!("Failed to persist lexicon for '{}': {}", collection, e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;

    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    async fn setup_tables(pool: &SqlitePool) {
        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS sparse_vocabulary (
                term_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                collection TEXT NOT NULL,
                document_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (term_id, collection),
                UNIQUE (term, collection)
            )"#,
        )
        .execute(pool)
        .await
        .unwrap();

        sqlx::query(
            r#"CREATE TABLE IF NOT EXISTS corpus_statistics (
                collection TEXT PRIMARY KEY NOT NULL,
                total_documents INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )"#,
        )
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_new_collection_starts_empty() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool, 1.2);
        assert_eq!(mgr.corpus_size("projects").await, 0);
        assert_eq!(mgr.document_frequency("projects", "test").await, 0);
    }

    #[tokio::test]
    async fn test_add_document_tracks_terms() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool, 1.2);
        mgr.add_document("projects", &["hello".into(), "world".into()])
            .await
            .unwrap();
        mgr.add_document("projects", &["hello".into(), "rust".into()])
            .await
            .unwrap();

        assert_eq!(mgr.corpus_size("projects").await, 2);
        assert_eq!(mgr.document_frequency("projects", "hello").await, 2);
        assert_eq!(mgr.document_frequency("projects", "world").await, 1);
        assert_eq!(mgr.document_frequency("projects", "rust").await, 1);
        assert_eq!(mgr.document_frequency("projects", "missing").await, 0);
    }

    #[tokio::test]
    async fn test_persist_and_reload() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.add_document("projects", &["fn".into(), "main".into()])
            .await
            .unwrap();
        mgr.add_document("projects", &["fn".into(), "test".into()])
            .await
            .unwrap();
        mgr.persist("projects").await.unwrap();

        // Create a new manager to test loading from SQLite
        let mgr2 = LexiconManager::new(pool, 1.2);
        mgr2.load_collection("projects").await.unwrap();

        assert_eq!(mgr2.corpus_size("projects").await, 2);
        assert_eq!(mgr2.document_frequency("projects", "fn").await, 2);
        assert_eq!(mgr2.document_frequency("projects", "main").await, 1);
        assert_eq!(mgr2.document_frequency("projects", "test").await, 1);
    }

    #[tokio::test]
    async fn test_collections_are_isolated() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool, 1.2);
        mgr.add_document("projects", &["hello".into()])
            .await
            .unwrap();
        mgr.add_document("libraries", &["world".into()])
            .await
            .unwrap();

        assert_eq!(mgr.corpus_size("projects").await, 1);
        assert_eq!(mgr.corpus_size("libraries").await, 1);
        assert_eq!(mgr.document_frequency("projects", "hello").await, 1);
        assert_eq!(mgr.document_frequency("projects", "world").await, 0);
        assert_eq!(mgr.document_frequency("libraries", "world").await, 1);
        assert_eq!(mgr.document_frequency("libraries", "hello").await, 0);
    }

    #[tokio::test]
    async fn test_persist_all() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.add_document("projects", &["a".into()])
            .await
            .unwrap();
        mgr.add_document("libraries", &["b".into()])
            .await
            .unwrap();
        mgr.persist_all().await.unwrap();

        // Verify both persisted
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'libraries'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_persist_is_idempotent() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.add_document("projects", &["hello".into()])
            .await
            .unwrap();
        mgr.persist("projects").await.unwrap();
        mgr.persist("projects").await.unwrap(); // Second persist should not error

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_incremental_persist_updates_df() {
        let pool = create_test_pool().await;
        setup_tables(&pool).await;

        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.add_document("projects", &["hello".into()])
            .await
            .unwrap();
        mgr.persist("projects").await.unwrap();

        // Add more documents
        mgr.add_document("projects", &["hello".into(), "world".into()])
            .await
            .unwrap();
        mgr.persist("projects").await.unwrap();

        // Verify updated DF
        let df: i64 = sqlx::query_scalar(
            "SELECT document_count FROM sparse_vocabulary WHERE term = 'hello' AND collection = 'projects'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(df, 2);

        let total: i64 = sqlx::query_scalar(
            "SELECT total_documents FROM corpus_statistics WHERE collection = 'projects'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(total, 2);
    }
}
