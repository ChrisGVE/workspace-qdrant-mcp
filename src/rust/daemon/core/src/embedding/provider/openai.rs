//! `OpenAiCompatibleProvider` — HTTP-based dense embedding provider speaking
//! the OpenAI `/v1/embeddings` protocol.
//!
//! Construction is **non-blocking**: `OpenAiCompatibleProvider::new` performs
//! no network I/O. The first probe is performed asynchronously by the
//! background `ProviderHealthMonitor` (see `health_monitor.rs`).
//!
//! Output dimensionality is initialised from `expected_output_dim` (config)
//! and stored as an `AtomicUsize`. The probe owns drift detection: if the
//! upstream returns a vector whose length differs from the cached value, the
//! probe writes the actual length back via `AtomicUsize::store` and emits a
//! WARN log. There is no separate trait method for dim updates.
//!
//! API keys are stored as `secrecy::SecretString`; their `Debug` impl is
//! `[REDACTED]` so the value never leaks through `tracing` spans, panic
//! payloads, or metric labels.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use secrecy::SecretString;
use tokio::sync::Mutex;
use tracing::warn;

use super::rate_limit::RateLimitAdapter;
use super::DenseProvider;
use crate::embedding::types::{DenseEmbedding, EmbeddingError};

mod types;

mod utils;
pub(crate) use utils::classify_provider;
use utils::normalize_in_place;

mod http;

/// Default HTTP request timeout (seconds).
const HTTP_TIMEOUT_SECS: u64 = 60;

/// HTTP client for any OpenAI-compatible `/v1/embeddings` endpoint.
///
/// Construction is non-blocking — no network call in `new`.
pub struct OpenAiCompatibleProvider {
    pub(super) base_url: String,
    pub(super) model: String,
    pub(super) batch_size: usize,
    pub(super) api_key: SecretString,
    pub(super) output_dim: AtomicUsize,
    pub(super) http: Client,
    pub(super) rate_limiter: Arc<RateLimitAdapter>,
    pub(super) metrics_tag: &'static str,
    pub(super) provider_label_value: String,
    pub(super) probe_cache: Mutex<Option<(Instant, Result<(), Arc<EmbeddingError>>)>>,
    pub(super) probe_cache_ttl: Duration,
    /// Char budget for a single input, derived from the model's token cap.
    pub(super) max_input_chars: usize,
}

/// Conservative lower bound on characters-per-token for cl100k/o200k BPE on
/// the code + prose this index holds. English averages ~4 and code ~3.3, so
/// assuming 3 keeps the derived char budget safely under the token cap while
/// avoiding needless over-splitting of normal-sized chunks.
const MIN_CHARS_PER_TOKEN: usize = 3;

impl std::fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("batch_size", &self.batch_size)
            .field("output_dim", &self.output_dim.load(Ordering::Relaxed))
            .field("metrics_tag", &self.metrics_tag)
            .field("probe_cache_ttl", &self.probe_cache_ttl)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatibleProvider {
    /// Construct the provider. Non-blocking — no network call.
    ///
    /// `api_key_env_var` names the environment variable that holds the API
    /// key. Returns `EmbeddingError::InitializationError` if the variable is
    /// absent or empty.
    pub fn new(
        base_url: String,
        model: String,
        batch_size: usize,
        expected_output_dim: usize,
        api_key_env_var: &str,
        probe_cache_ttl: Duration,
        max_input_tokens: usize,
    ) -> Result<Self, EmbeddingError> {
        let raw_key =
            std::env::var(api_key_env_var).map_err(|_| EmbeddingError::InitializationError {
                message: format!("API key environment variable {api_key_env_var} is not set"),
            })?;
        if raw_key.trim().is_empty() {
            return Err(EmbeddingError::InitializationError {
                message: format!("API key environment variable {api_key_env_var} is empty"),
            });
        }

        let http = Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .map_err(|e| EmbeddingError::InitializationError {
                message: format!("Failed to construct HTTP client: {e}"),
            })?;

        let metrics_tag = classify_provider(&base_url);
        let provider_label_value = format!("{}/{}", base_url, model);

        Ok(Self {
            base_url,
            model,
            batch_size: batch_size.max(1),
            api_key: SecretString::from(raw_key),
            output_dim: AtomicUsize::new(expected_output_dim),
            http,
            rate_limiter: Arc::new(RateLimitAdapter::new(batch_size.max(1))),
            metrics_tag,
            provider_label_value,
            probe_cache: Mutex::new(None),
            probe_cache_ttl,
            max_input_chars: max_input_tokens.saturating_mul(MIN_CHARS_PER_TOKEN),
        })
    }

    /// Endpoint URL: `{base_url}/v1/embeddings`.
    pub(super) fn endpoint_url(&self) -> String {
        format!("{}/v1/embeddings", self.base_url)
    }

    /// Issue a single batch request to the upstream and parse the response.
    /// Returns embeddings in the input order.
    async fn embed_chunk(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        http::embed_chunk(self, texts).await
    }
}

#[async_trait]
impl DenseProvider for OpenAiCompatibleProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<DenseEmbedding> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let raw_vectors = self.embed_chunk(chunk).await?;
            if raw_vectors.len() != chunk.len() {
                return Err(EmbeddingError::GenerationError {
                    message: format!(
                        "Provider returned {} embeddings for {} inputs",
                        raw_vectors.len(),
                        chunk.len()
                    ),
                });
            }
            for (idx, mut vector) in raw_vectors.into_iter().enumerate() {
                normalize_in_place(&mut vector)?;
                let original = chunk[idx];
                out.push(DenseEmbedding {
                    vector,
                    model_name: self.model.clone(),
                    sequence_length: original.len(),
                });
            }
        }
        Ok(out)
    }

    fn output_dim(&self) -> usize {
        self.output_dim.load(Ordering::Relaxed)
    }

    fn max_input_chars(&self) -> usize {
        self.max_input_chars
    }

    fn provider_label(&self) -> &str {
        &self.provider_label_value
    }

    fn metrics_label(&self) -> &'static str {
        self.metrics_tag
    }

    async fn probe(&self) -> Result<(), EmbeddingError> {
        // Cache hit returns clone of cached result.
        {
            let guard = self.probe_cache.lock().await;
            if let Some((when, result)) = guard.as_ref() {
                if when.elapsed() < self.probe_cache_ttl {
                    return match result {
                        Ok(()) => Ok(()),
                        Err(arc) => Err((**arc).clone()),
                    };
                }
            }
        }

        let probe_result = self.embed_chunk(&["probe"]).await;
        let cache_value: Result<(), Arc<EmbeddingError>> = match &probe_result {
            Ok(vectors) => {
                if let Some(first) = vectors.first() {
                    let actual = first.len();
                    let cached = self.output_dim.load(Ordering::Relaxed);
                    if actual != cached {
                        warn!(
                            cached_dim = cached,
                            actual_dim = actual,
                            provider = %self.provider_label_value,
                            "Embedding provider output dim drift detected; updating runtime atomic"
                        );
                        self.output_dim.store(actual, Ordering::Relaxed);
                    }
                }
                Ok(())
            }
            Err(e) => Err(Arc::new(e.clone())),
        };

        let mut guard = self.probe_cache.lock().await;
        *guard = Some((Instant::now(), cache_value.clone()));
        drop(guard);

        match cache_value {
            Ok(()) => Ok(()),
            Err(arc) => Err((*arc).clone()),
        }
    }
}

#[cfg(test)]
mod tests;
