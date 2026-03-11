//! LexiconManager struct definition and lifecycle methods.

use std::collections::HashMap;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::{oneshot, mpsc, RwLock};
use tracing::info;

use crate::embedding::BM25;

use super::background_persist::{spawn_background_persister, PersistRequest};

/// Manages per-collection BM25 vocabulary with SQLite persistence.
pub struct LexiconManager {
    pub(super) pool: SqlitePool,
    /// Per-collection BM25 instances (lazy loaded from SQLite).
    pub(super) instances: Arc<RwLock<HashMap<String, BM25>>>,
    /// Tracks how many documents have been added since last persist.
    pub(super) dirty_counts: Arc<RwLock<HashMap<String, u32>>>,
    /// BM25 k1 parameter for new instances.
    pub(super) k1: f32,
    /// Channel to the background persistence task.
    /// `None` until `start_background_persister()` is called.
    pub(super) persist_tx: Arc<RwLock<Option<mpsc::Sender<PersistRequest>>>>,
}

impl LexiconManager {
    pub fn new(pool: SqlitePool, k1: f32) -> Self {
        Self {
            pool,
            instances: Arc::new(RwLock::new(HashMap::new())),
            dirty_counts: Arc::new(RwLock::new(HashMap::new())),
            k1,
            persist_tx: Arc::new(RwLock::new(None)),
        }
    }

    /// Spawn the background persistence task.
    ///
    /// Must be called once after wrapping `LexiconManager` in `Arc`.
    /// After this call, `add_document()` dispatches persist requests to the
    /// background task instead of blocking on SQLite writes inline.
    pub async fn start_background_persister(self: &Arc<Self>) {
        let tx = spawn_background_persister(Arc::clone(self));
        *self.persist_tx.write().await = Some(tx);
        info!("LexiconManager background persistence task started");
    }

    /// Flush all pending persist requests and wait for completion.
    ///
    /// Call before daemon shutdown to ensure all in-flight dirty terms
    /// are written to SQLite before the process exits.
    pub async fn flush_all_background(&self) {
        let tx = {
            let guard = self.persist_tx.read().await;
            guard.clone()
        };
        if let Some(tx) = tx {
            let (reply_tx, reply_rx) = oneshot::channel();
            if tx.send(PersistRequest::Flush { reply: reply_tx }).await.is_ok() {
                let _ = reply_rx.await;
                info!("LexiconManager background flush complete");
            }
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
    pub(super) async fn ensure_loaded(&self, collection: &str) -> Result<(), sqlx::Error> {
        {
            let instances = self.instances.read().await;
            if instances.contains_key(collection) {
                return Ok(());
            }
        }
        self.load_collection(collection).await
    }

    /// Clear all in-memory BM25 instances and dirty counts.
    ///
    /// Call after deleting vocabulary data from SQLite to ensure
    /// the in-memory state is consistent. New instances will be lazy-loaded
    /// from SQLite on next `load_collection()` call.
    pub async fn clear_all(&self) {
        let mut instances = self.instances.write().await;
        instances.clear();
        let mut dirty = self.dirty_counts.write().await;
        dirty.clear();
    }

    /// Persist all loaded collections (call on shutdown or periodically).
    pub async fn persist_all(&self) -> Result<(), sqlx::Error> {
        let collections: Vec<String> = {
            let instances = self.instances.read().await;
            instances.keys().cloned().collect()
        };
        for collection in collections {
            if let Err(e) = self.persist(&collection).await {
                tracing::warn!("Failed to persist lexicon for '{}': {}", collection, e);
            }
        }
        Ok(())
    }
}
