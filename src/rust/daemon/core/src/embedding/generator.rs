//! Embedding generator and text preprocessor.
//!
//! Dense embedding generation is delegated to an injected
//! `Arc<dyn DenseProvider>`. Sparse generation (BM25 / SPLADE++) and the
//! cross-document phrase cache stay in this module unchanged.

use fastembed::{SparseInitOptions, SparseModel, SparseTextEmbedding};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;

use super::bm25::{tokenize_for_bm25, BM25};
use super::phrase_cache::PhraseCache;
use super::provider::DenseProvider;
use super::types::{
    DenseEmbedding, EmbeddingConfig, EmbeddingError, EmbeddingResult, PreprocessedText,
    SparseEmbedding,
};

/// Embedding generator that delegates dense work to a `DenseProvider`.
pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
    dense_provider: Arc<dyn DenseProvider>,
    bm25: Arc<Mutex<BM25>>,
    /// Optional directory used by SPLADE++ initialisation.
    model_cache_dir: Option<PathBuf>,
    /// SPLADE++ sparse embedding model (lazy-initialised).
    splade_model: Arc<Mutex<Option<SparseTextEmbedding>>>,
    /// LRU cache for short-phrase dense embeddings (cross-document reuse).
    phrase_cache: PhraseCache,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("config", &self.config)
            .field("dense_provider", &self.dense_provider.provider_label())
            .field("sparse_mode", &self.config.sparse_vector_mode)
            .finish()
    }
}

impl EmbeddingGenerator {
    /// Create a new generator backed by the provided dense provider.
    pub fn new(
        config: EmbeddingConfig,
        dense_provider: Arc<dyn DenseProvider>,
    ) -> Result<Self, EmbeddingError> {
        let model_cache_dir = config.model_cache_dir.clone();
        let phrase_cache = PhraseCache::new(config.max_cache_size);
        Ok(Self {
            config: config.clone(),
            dense_provider,
            bm25: Arc::new(Mutex::new(BM25::new(config.bm25_k1))),
            model_cache_dir,
            splade_model: Arc::new(Mutex::new(None)),
            phrase_cache,
        })
    }

    /// Output dimensionality of the underlying dense provider.
    pub fn dense_dim(&self) -> usize {
        self.dense_provider.output_dim()
    }

    /// Maximum UTF-8 bytes allowed in a single embedding input for the active
    /// provider. `usize::MAX` means no caller-side limit. The ingestion layer
    /// splits chunks above this budget before embedding.
    pub fn max_input_bytes(&self) -> usize {
        self.dense_provider.max_input_bytes()
    }

    /// Label identifying the dense embedding model (e.g. "fastembed/all-MiniLM-L6-v2").
    pub fn provider_label(&self) -> &str {
        self.dense_provider.provider_label()
    }

    /// Bounded-cardinality engine label for the active dense provider, one of
    /// `fastembed|openai|azure_openai|lmstudio|llama_cpp|openai_compatible_other`.
    /// Used as the `embedding_engine` metric label (A2).
    pub fn metrics_label(&self) -> &'static str {
        self.dense_provider.metrics_label()
    }

    /// Issue a single probe call against the dense provider.
    pub async fn probe_provider(&self) -> Result<(), EmbeddingError> {
        self.dense_provider.probe().await
    }

    pub async fn initialize_model(&self, _model_name: &str) -> Result<(), EmbeddingError> {
        self.probe_provider().await
    }

    pub async fn generate_embedding(
        &self,
        text: &str,
        _model_name: &str,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        // Check phrase cache before delegating to the provider.
        let cached_dense = self.phrase_cache.get(text).await;

        let dense = if let Some(v) = cached_dense {
            DenseEmbedding {
                vector: v,
                model_name: self.dense_provider.provider_label().to_string(),
                sequence_length: text.len(),
            }
        } else {
            let embed_start = Instant::now();
            // Track embedder saturation for the duration of the provider call
            // (decremented on drop, including the early-return `?` paths).
            let _inflight = crate::monitoring::metrics_core::METRICS.embedding_inflight_guard();
            let mut embeddings = self.dense_provider.embed(&[text]).await?;
            drop(_inflight);
            let embed_ms = embed_start.elapsed().as_millis();

            let dense = embeddings
                .pop()
                .ok_or_else(|| EmbeddingError::GenerationError {
                    message: "Provider returned no embedding".to_string(),
                })?;

            info!(
                text_len = text.len(),
                dim = dense.vector.len(),
                embed_ms = embed_ms,
                provider = self.dense_provider.metrics_label(),
                "dense embedding generated"
            );

            // Populate cache for eligible phrases.
            self.phrase_cache.put(text, dense.vector.clone()).await;
            dense
        };

        // Generate sparse embedding using BM25.
        let bm25_start = Instant::now();
        let tokens = tokenize_for_bm25(text);

        let sparse = {
            let mut bm25 = self.bm25.lock().await;
            bm25.add_document(&tokens);
            bm25.generate_sparse_vector(&tokens)
        };
        let bm25_ms = bm25_start.elapsed().as_millis();

        info!(
            token_count = tokens.len(),
            sparse_nnz = sparse.indices.len(),
            bm25_ms = bm25_ms,
            "BM25 sparse vector generated"
        );

        let text_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };

        Ok(EmbeddingResult {
            text_hash,
            dense,
            sparse,
            generated_at: chrono::Utc::now(),
        })
    }

    #[tracing::instrument(
        name = "embedding.generate_batch",
        skip_all,
        fields(model = %model_name, batch_size = texts.len())
    )]
    pub async fn generate_embeddings_batch(
        &self,
        texts: &[String],
        model_name: &str,
    ) -> Result<Vec<EmbeddingResult>, EmbeddingError> {
        let batch_start = Instant::now();
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Partition into cache hits and misses.
        let mut dense_results: Vec<Option<DenseEmbedding>> = vec![None; batch_size];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<&str> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(cached) = self.phrase_cache.get(text).await {
                dense_results[i] = Some(DenseEmbedding {
                    vector: cached,
                    model_name: self.dense_provider.provider_label().to_string(),
                    sequence_length: text.len(),
                });
            } else {
                miss_indices.push(i);
                miss_texts.push(text.as_str());
            }
        }

        // Batch-embed all cache misses in one provider call (provider handles
        // sub-chunking by remote_batch_size internally).
        if !miss_texts.is_empty() {
            let embed_start = Instant::now();
            let embeddings = self.dense_provider.embed(&miss_texts).await?;
            let embed_ms = embed_start.elapsed().as_millis();

            if embeddings.len() != miss_texts.len() {
                return Err(EmbeddingError::GenerationError {
                    message: format!(
                        "Provider returned {} embeddings for {} inputs",
                        embeddings.len(),
                        miss_texts.len()
                    ),
                });
            }

            info!(
                miss_count = miss_texts.len(),
                cache_hits = batch_size - miss_texts.len(),
                dim = embeddings.first().map(|e| e.vector.len()).unwrap_or(0),
                embed_ms = embed_ms,
                provider = self.dense_provider.metrics_label(),
                "dense batch embedded"
            );

            for (j, dense) in embeddings.into_iter().enumerate() {
                let idx = miss_indices[j];
                self.phrase_cache
                    .put(&texts[idx], dense.vector.clone())
                    .await;
                dense_results[idx] = Some(dense);
            }
        }

        // Generate sparse vectors and assemble results.
        let mut results = Vec::with_capacity(batch_size);
        for (i, text) in texts.iter().enumerate() {
            let dense = dense_results[i].take().expect("all dense slots populated");

            let tokens = tokenize_for_bm25(text);
            let sparse = {
                let mut bm25 = self.bm25.lock().await;
                bm25.add_document(&tokens);
                bm25.generate_sparse_vector(&tokens)
            };

            let text_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut hasher);
                hasher.finish()
            };

            results.push(EmbeddingResult {
                text_hash,
                dense,
                sparse,
                generated_at: chrono::Utc::now(),
            });
        }

        let batch_ms = batch_start.elapsed().as_millis();
        let per_item_ms = if batch_size > 0 {
            batch_ms / batch_size as u128
        } else {
            0
        };
        info!(
            batch_size = batch_size,
            batch_ms = batch_ms,
            per_item_ms = per_item_ms,
            "embedding batch completed"
        );
        // Route this batch's telemetry through the metrics switchboard (the hub
        // owns ALL telemetry — no emitter writes DaemonMetrics directly). The
        // drain task reproduces the exact `record_embedding(model, batch_size,
        // elapsed)` observation, so the Prometheus series are byte-identical;
        // the only change is the path. The model label is interned to `&'static`
        // (the sample is `Copy`) so it is preserved exactly across the buffer.
        // Emitted per batch (coarse), not the per-chunk hot loop. If the
        // switchboard is not yet sealed (very early init, before any processing),
        // the emit is skipped — embedding never runs in that window.
        if let Some(sw) = crate::switchboard::switchboard() {
            let handle = sw.handle(
                crate::switchboard::MetricId::EmbedderBatch,
                crate::switchboard::intern_model_label(model_name),
            );
            sw.emit_embedder_batch(
                handle,
                crate::switchboard::EmbedderBatchRec {
                    batch_size,
                    elapsed: batch_start.elapsed(),
                },
            );
        }
        Ok(results)
    }

    pub async fn add_document_to_corpus(&self, text: &str) {
        let tokens = tokenize_for_bm25(text);
        let mut bm25 = self.bm25.lock().await;
        bm25.add_document(&tokens);
    }

    /// Returns `(hits, misses)` for the phrase embedding cache.
    pub async fn cache_stats(&self) -> (usize, usize) {
        let (hits, misses, _) = self.phrase_cache.stats().await;
        (hits as usize, misses as usize)
    }

    /// Clear the phrase embedding cache.
    pub async fn clear_cache(&self) {
        self.phrase_cache.clear().await;
    }

    pub fn available_models(&self) -> Vec<String> {
        vec![self.dense_provider.provider_label().to_string()]
    }

    pub async fn is_model_ready(&self, _model_name: &str) -> bool {
        self.probe_provider().await.is_ok()
    }

    /// Get the configured sparse vector mode ("bm25" or "splade").
    pub fn sparse_vector_mode(&self) -> &str {
        &self.config.sparse_vector_mode
    }

    /// Generate a sparse vector using the SPLADE++ model.
    ///
    /// Lazy-initialises the SPLADE++ model on first call (~150MB download).
    pub async fn generate_splade_sparse_vector(
        &self,
        text: &str,
    ) -> Result<SparseEmbedding, EmbeddingError> {
        let mut guard = self.splade_model.lock().await;
        if guard.is_none() {
            info!("Initializing SPLADE++ model (first call, ~150MB download)...");
            let mut init_opts =
                SparseInitOptions::new(SparseModel::SPLADEPPV1).with_show_download_progress(true);
            if let Some(ref cache_dir) = self.model_cache_dir {
                init_opts = init_opts.with_cache_dir(cache_dir.clone());
            }
            let model = SparseTextEmbedding::try_new(init_opts).map_err(|e| {
                EmbeddingError::InitializationError {
                    message: format!("SPLADE++ init failed: {}", e),
                }
            })?;
            *guard = Some(model);
            info!("SPLADE++ model initialized");
        }

        let model = guard.as_mut().unwrap();
        let splade_start = Instant::now();
        let results = model.embed(vec![text.to_string()], None).map_err(|e| {
            EmbeddingError::GenerationError {
                message: format!("SPLADE++ embedding failed: {}", e),
            }
        })?;
        let splade_ms = splade_start.elapsed().as_millis();

        let fe_sparse =
            results
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::GenerationError {
                    message: "SPLADE++ returned no embeddings".to_string(),
                })?;

        info!(
            text_len = text.len(),
            nnz = fe_sparse.indices.len(),
            splade_ms = splade_ms,
            "SPLADE++ sparse vector generated"
        );

        // Convert fastembed usize indices to our u32 indices
        Ok(SparseEmbedding {
            indices: fe_sparse.indices.into_iter().map(|i| i as u32).collect(),
            values: fe_sparse.values,
            vocab_size: 30522, // BERT vocab size for SPLADE++
        })
    }
}

/// Text preprocessor
#[derive(Debug)]
pub struct TextPreprocessor {
    enable_preprocessing: bool,
}

impl TextPreprocessor {
    pub fn new(enable_preprocessing: bool) -> Self {
        Self {
            enable_preprocessing,
        }
    }

    pub fn preprocess(&self, text: &str) -> PreprocessedText {
        let cleaned = if self.enable_preprocessing {
            text.split_whitespace().collect::<Vec<_>>().join(" ")
        } else {
            text.to_string()
        };

        let tokens = tokenize_for_bm25(&cleaned);

        PreprocessedText {
            original: text.to_string(),
            cleaned,
            tokens,
            token_ids: vec![],
        }
    }
}
