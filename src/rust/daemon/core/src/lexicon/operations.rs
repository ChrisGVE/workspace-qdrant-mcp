//! Document and query operations on the LexiconManager.

use tracing::{debug, warn};
use wqm_common::timestamps;

use super::background_persist::PersistRequest;

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

    /// Batch document-frequency lookup for a set of terms.
    ///
    /// Acquires the read lock once for all terms instead of once per term,
    /// eliminating the N async-lock overhead of calling `document_frequency` in a loop.
    /// Returns a map from term → document_count (absent = 0).
    pub async fn document_frequencies_batch(
        &self,
        collection: &str,
        terms: &std::collections::HashSet<String>,
    ) -> std::collections::HashMap<String, u64> {
        if terms.is_empty() {
            return std::collections::HashMap::new();
        }
        if let Err(e) = self.ensure_loaded(collection).await {
            warn!("Failed to load lexicon for '{}': {}", collection, e);
            return std::collections::HashMap::new();
        }
        let instances = self.instances.read().await;
        let bm25 = match instances.get(collection) {
            Some(b) => b,
            None => return std::collections::HashMap::new(),
        };
        let mut result = std::collections::HashMap::with_capacity(terms.len());
        for term in terms {
            if let Some(&term_id) = bm25.vocab().get(term.as_str()) {
                let df = bm25.doc_freq().get(&term_id).copied().unwrap_or(0);
                if df > 0 {
                    result.insert(term.clone(), df as u64);
                }
            }
        }
        result
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
            // Try to dispatch to background task (non-blocking).
            // Fall back to inline persist if the background task isn't running.
            let tx = {
                let guard = self.persist_tx.read().await;
                guard.clone()
            };
            match tx {
                Some(sender) => {
                    let request = PersistRequest::Persist {
                        collection: collection.to_string(),
                    };
                    if let Err(e) = sender.try_send(request) {
                        debug!(
                            "Background persist channel full for '{}', falling back to inline: {}",
                            collection, e
                        );
                        self.persist(collection).await?;
                    } else {
                        debug!(
                            "Queued background persist for '{}' (threshold {} reached)",
                            collection, AUTO_PERSIST_THRESHOLD
                        );
                    }
                }
                None => {
                    // Background task not started — persist inline
                    debug!(
                        "Auto-persisting lexicon for '{}' inline (threshold {} reached)",
                        collection, AUTO_PERSIST_THRESHOLD
                    );
                    self.persist(collection).await?;
                }
            }
        }

        Ok(())
    }

    /// Persist modified BM25 terms for a collection to SQLite.
    ///
    /// Only writes the terms that changed since the last persist (dirty tracking),
    /// rather than the entire vocabulary. This reduces the write volume from
    /// O(vocab_size) to O(terms_in_last_N_docs), eliminating multi-second stalls.
    pub async fn persist(&self, collection: &str) -> Result<(), sqlx::Error> {
        let (dirty_entries, total_docs) = {
            let mut instances = self.instances.write().await;
            let bm25 = match instances.get_mut(collection) {
                Some(b) => b,
                None => return Ok(()),
            };
            (bm25.take_dirty_entries(), bm25.total_docs())
        };

        if dirty_entries.is_empty() && total_docs == 0 {
            return Ok(());
        }

        let now = timestamps::now_utc();
        let mut tx = self.pool.begin().await?;

        if !dirty_entries.is_empty() {
            sqlx::query(
                "CREATE TEMP TABLE IF NOT EXISTS _vocab_batch(\
                 term_id INTEGER, term TEXT, doc_count INTEGER)",
            )
            .execute(&mut *tx)
            .await?;

            sqlx::query("DELETE FROM _vocab_batch")
                .execute(&mut *tx)
                .await?;

            // Batch-insert dirty terms into temp table (chunks of 100 → 300 params)
            const CHUNK_SIZE: usize = 100;
            for chunk in dirty_entries.chunks(CHUNK_SIZE) {
                let placeholders: Vec<String> = (0..chunk.len())
                    .map(|i| format!("(?{}, ?{}, ?{})", i * 3 + 1, i * 3 + 2, i * 3 + 3))
                    .collect();
                let sql = format!(
                    "INSERT INTO _vocab_batch(term_id, term, doc_count) VALUES {}",
                    placeholders.join(", ")
                );
                let mut q = sqlx::query(&sql);
                for (term, term_id, df) in chunk {
                    q = q.bind(*term_id as i64).bind(term).bind(*df as i64);
                }
                q.execute(&mut *tx).await?;
            }

            // Update existing terms (uses UNIQUE index on (term, collection))
            sqlx::query(
                "UPDATE sparse_vocabulary SET document_count = b.doc_count \
                 FROM _vocab_batch b \
                 WHERE sparse_vocabulary.term = b.term AND sparse_vocabulary.collection = ?1",
            )
            .bind(collection)
            .execute(&mut *tx)
            .await?;

            // Insert new terms (those not already present)
            sqlx::query(
                "INSERT OR IGNORE INTO sparse_vocabulary \
                 (term_id, term, collection, document_count, created_at) \
                 SELECT b.term_id, b.term, ?1, b.doc_count, ?2 FROM _vocab_batch b",
            )
            .bind(collection)
            .bind(&now)
            .execute(&mut *tx)
            .await?;

            sqlx::query("DROP TABLE IF EXISTS _vocab_batch")
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
        drop(dirty);

        // Evict hapax legomena from in-memory vocab and SQLite.
        // Done after commit so the DELETE is independent of the upsert tx.
        let evicted_count = {
            let mut instances = self.instances.write().await;
            if let Some(bm25) = instances.get_mut(collection) {
                bm25.evict_hapax()
            } else {
                0
            }
        };
        if evicted_count > 0 {
            sqlx::query(
                "DELETE FROM sparse_vocabulary WHERE collection = ?1 AND document_count = 1",
            )
            .bind(collection)
            .execute(&self.pool)
            .await?;
            debug!(
                "Evicted {} hapax legomena from '{}' vocabulary",
                evicted_count, collection
            );
        }

        debug!(
            "Persisted lexicon for '{}': {} dirty terms, {} total docs",
            collection,
            dirty_entries.len(),
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
