//! `FastEmbedProvider` — local ONNX-backed dense embedding provider.
//!
//! Wraps the existing FastEmbed `TextEmbedding` model loaded with the
//! `AllMiniLML6V2` checkpoint. Initialisation is lazy and rate-limited by
//! the same exponential backoff schedule used by `EmbeddingGenerator`,
//! ensuring repeated failures do not hammer the local disk or download
//! endpoints.
//!
//! The provider returns L2-normalised 384-dim vectors per the
//! `DenseProvider` invariant. The MiniLM checkpoint emits normalised
//! vectors natively; defensive normalisation only kicks in on the
//! pathological zero-norm case (rejected with `GenerationError`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::DenseProvider;
use crate::embedding::types::{DenseEmbedding, EmbeddingError};

/// Output dimensionality of the FastEmbed `AllMiniLML6V2` checkpoint.
pub const FASTEMBED_OUTPUT_DIM: usize = 384;

/// Backoff schedule (seconds) for failed init attempts. Matches the schedule
/// previously used inside `EmbeddingGenerator` so behaviour is preserved
/// during the trait-extraction migration.
pub(crate) const INIT_BACKOFF_SECS: &[u64] = &[30, 60, 120, 300, 600];

/// Local FastEmbed-backed implementation of `DenseProvider`.
pub struct FastEmbedProvider {
    inner: Arc<Mutex<Option<TextEmbedding>>>,
    model_cache_dir: Option<PathBuf>,
    num_threads: Option<usize>,
    batch_size: usize,
    initialized: Arc<AtomicBool>,
    last_failed_init: Arc<Mutex<Option<Instant>>>,
    init_failure_count: Arc<AtomicU32>,
}

impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider")
            .field("model_cache_dir", &self.model_cache_dir)
            .field("num_threads", &self.num_threads)
            .field("batch_size", &self.batch_size)
            .field("initialized", &self.initialized.load(Ordering::Relaxed))
            .field(
                "init_failure_count",
                &self.init_failure_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl FastEmbedProvider {
    /// Construct a new provider. Synchronous: no model load happens here.
    pub fn new(
        batch_size: usize,
        model_cache_dir: Option<PathBuf>,
        num_threads: Option<usize>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            model_cache_dir,
            num_threads,
            batch_size: batch_size.max(1),
            initialized: Arc::new(AtomicBool::new(false)),
            last_failed_init: Arc::new(Mutex::new(None)),
            init_failure_count: Arc::new(AtomicU32::new(0)),
        }
    }

    async fn ensure_initialized(&self) -> Result<(), EmbeddingError> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        let failure_count = self.init_failure_count.load(Ordering::SeqCst);
        if failure_count > 0 {
            let guard = self.last_failed_init.lock().await;
            if let Some(last_attempt) = *guard {
                let backoff_idx = (failure_count as usize - 1).min(INIT_BACKOFF_SECS.len() - 1);
                let backoff = INIT_BACKOFF_SECS[backoff_idx];
                let elapsed = last_attempt.elapsed().as_secs();
                if elapsed < backoff {
                    return Err(EmbeddingError::TemporarilyUnavailable {
                        retry_after_secs: backoff - elapsed,
                    });
                }
            }
        }

        let mut model_guard = self.inner.lock().await;
        if model_guard.is_some() {
            return Ok(());
        }

        let mut init_options =
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true);
        if let Some(threads) = self.num_threads {
            init_options = init_options.with_num_threads(threads);
        }
        if let Some(ref cache_dir) = self.model_cache_dir {
            init_options = init_options.with_cache_dir(cache_dir.clone());
        }

        match TextEmbedding::try_new(init_options) {
            Ok(model) => {
                *model_guard = Some(model);
                self.initialized.store(true, Ordering::SeqCst);
                self.init_failure_count.store(0, Ordering::SeqCst);
                *self.last_failed_init.lock().await = None;
                info!("FastEmbedProvider initialised (all-MiniLM-L6-v2)");
                Ok(())
            }
            Err(e) => {
                let new_count = self.init_failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                *self.last_failed_init.lock().await = Some(Instant::now());
                let backoff_idx = (new_count as usize - 1).min(INIT_BACKOFF_SECS.len() - 1);
                warn!(
                    failure_count = new_count,
                    next_retry_secs = INIT_BACKOFF_SECS[backoff_idx],
                    "FastEmbedProvider initialisation failed: {e}"
                );
                Err(EmbeddingError::InitializationError {
                    message: format!("Failed to initialize FastEmbed: {e}"),
                })
            }
        }
    }

    /// Test-only hook: prime the failure-backoff state without actually
    /// attempting initialisation. Used by `init_backoff_respected` to
    /// avoid touching the network or disk.
    #[cfg(test)]
    pub(crate) async fn force_init_failed_for_test(&self) {
        self.init_failure_count.store(1, Ordering::SeqCst);
        *self.last_failed_init.lock().await = Some(Instant::now());
    }
}

#[async_trait]
impl DenseProvider for FastEmbedProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_initialized().await?;

        let mut model_guard = self.inner.lock().await;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| EmbeddingError::InitializationError {
                message: "Model not initialized".to_string(),
            })?;

        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let docs: Vec<&str> = chunk.iter().copied().collect();
            let embed_start = Instant::now();
            let vectors = model
                .embed(docs, None)
                .map_err(|e| EmbeddingError::GenerationError {
                    message: format!("Embedding generation failed: {e}"),
                })?;
            let embed_ms = embed_start.elapsed().as_millis();
            for (idx, mut v) in vectors.into_iter().enumerate() {
                normalize_in_place(&mut v)?;
                let original = chunk[idx];
                out.push(DenseEmbedding {
                    vector: v,
                    model_name: "all-MiniLM-L6-v2".to_string(),
                    sequence_length: original.len(),
                });
            }
            info!(
                chunk = chunk.len(),
                embed_ms = embed_ms,
                "FastEmbedProvider chunk embedded"
            );
        }
        Ok(out)
    }

    fn output_dim(&self) -> usize {
        FASTEMBED_OUTPUT_DIM
    }

    fn provider_label(&self) -> &str {
        "fastembed/all-MiniLM-L6-v2"
    }

    fn metrics_label(&self) -> &'static str {
        "fastembed"
    }

    async fn probe(&self) -> Result<(), EmbeddingError> {
        self.ensure_initialized().await?;
        let _ = self.embed(&["probe"]).await?;
        Ok(())
    }
}

/// L2-normalise `v` in place. Returns `GenerationError` for vectors with
/// effectively-zero norm so a NaN never propagates downstream.
fn normalize_in_place(v: &mut [f32]) -> Result<(), EmbeddingError> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return Err(EmbeddingError::GenerationError {
            message: "Embedding returned zero-norm vector".to_string(),
        });
    }
    if (norm - 1.0).abs() < 1e-4 {
        // Already normalised within FP tolerance — skip the rescale to keep
        // the bitwise output identical to the upstream model.
        return Ok(());
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> FastEmbedProvider {
        FastEmbedProvider::new(32, None, Some(2))
    }

    #[test]
    fn output_dim_is_384() {
        assert_eq!(provider().output_dim(), FASTEMBED_OUTPUT_DIM);
    }

    #[test]
    fn provider_label() {
        assert_eq!(provider().provider_label(), "fastembed/all-MiniLM-L6-v2");
    }

    #[test]
    fn metrics_label_is_fastembed() {
        assert_eq!(provider().metrics_label(), "fastembed");
    }

    #[tokio::test]
    async fn init_backoff_respected() {
        let p = provider();
        p.force_init_failed_for_test().await;
        let err = p
            .ensure_initialized()
            .await
            .expect_err("backoff window must short-circuit init");
        match err {
            EmbeddingError::TemporarilyUnavailable { retry_after_secs } => {
                assert!(
                    retry_after_secs > 0 && retry_after_secs <= INIT_BACKOFF_SECS[0],
                    "retry_after_secs out of bounds: {retry_after_secs}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn normalize_in_place_rejects_zero_norm() {
        let mut v = vec![0.0_f32; 8];
        let err = normalize_in_place(&mut v).expect_err("zero-norm must error");
        assert!(matches!(err, EmbeddingError::GenerationError { .. }));
    }

    #[test]
    fn normalize_in_place_scales_to_unit() {
        let mut v = vec![3.0_f32, 4.0]; // length 5
        normalize_in_place(&mut v).unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm should be 1, got {norm}");
    }
}
