//! Document and query operations on the LexiconManager.

use tracing::{debug, warn};
use wqm_common::timestamps;

use super::manager::LexiconManager;

/// Threshold for auto-persisting (number of documents processed since last persist).
pub(super) const AUTO_PERSIST_THRESHOLD: u32 = 50;

impl LexiconManager {
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

    /// Generate a sparse vector using the collection's persisted BM25 vocabulary.
    ///
    /// Uses the per-collection IDF statistics for true BM25 scoring,
    /// unlike the EmbeddingGenerator's ephemeral BM25 which starts empty each session.
    pub async fn generate_sparse_vector(
        &self,
        collection: &str,
        tokens: &[String],
    ) -> crate::embedding::SparseEmbedding {
        if let Err(e) = self.ensure_loaded(collection).await {
            warn!("Failed to load lexicon for '{}': {}", collection, e);
            return crate::embedding::SparseEmbedding {
                indices: Vec::new(),
                values: Vec::new(),
                vocab_size: 0,
            };
        }
        let instances = self.instances.read().await;
        if let Some(bm25) = instances.get(collection) {
            bm25.generate_sparse_vector(tokens)
        } else {
            crate::embedding::SparseEmbedding {
                indices: Vec::new(),
                values: Vec::new(),
                vocab_size: 0,
            }
        }
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

    /// Remove junk terms from sparse_vocabulary that were created before the
    /// BM25 tokenizer was upgraded. Matches patterns: pure digits, version strings,
    /// hex hashes (8+ chars), hex literals, terms containing path separators.
    ///
    /// Call once on startup before loading vocabulary into BM25 instances.
    pub async fn cleanup_junk_terms(&self) -> Result<u64, sqlx::Error> {
        let result = sqlx::query(
            r#"DELETE FROM sparse_vocabulary WHERE
               -- Pure digits
               (term GLOB '[0-9]*' AND term NOT GLOB '*[^0-9]*')
               -- Version strings (v1.2, 2.0.0, etc.)
               OR term GLOB '[0-9]*.[0-9]*'
               OR term GLOB 'v[0-9]*.[0-9]*'
               -- Hex hashes (8+ hex chars)
               OR (length(term) >= 8 AND term NOT GLOB '*[^a-f0-9]*')
               -- Hex literals (0x...)
               OR term GLOB '0x*'
               -- Contains path separators
               OR term LIKE '%/%' OR term LIKE '%\%'
               -- Single character
               OR length(term) <= 1"#,
        )
        .execute(&self.pool)
        .await?;

        let removed = result.rows_affected();
        if removed > 0 {
            tracing::info!("Cleaned {} junk terms from sparse_vocabulary", removed);
        }
        Ok(removed)
    }
}
