//! Pluggable dense-embedding provider abstraction.
//!
//! `DenseProvider` is the runtime trait every dense-embedding backend
//! implements. The factory `build_dense_provider` selects the active
//! implementation from configuration. Only the trait + factory live here in
//! the round-1 commit; concrete implementations land in subsequent commits
//! (see PRD §9 commit plan: FastEmbed in commit 4, OpenAI-compatible in
//! commit 5).
//!
//! Invariant: every implementation MUST return L2-normalized vectors.
//! For any returned vector with norm ≤ `f32::EPSILON` the implementation
//! MUST return `EmbeddingError::GenerationError` rather than emit a NaN.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::EmbeddingSettings;

use super::types::{DenseEmbedding, EmbeddingError};

/// Runtime contract for a dense-embedding backend.
///
/// Implementors are stored behind `Arc<dyn DenseProvider>` and must therefore
/// be `Send + Sync` and trivially `Debug`-able. `embed` and `probe` are async
/// because remote providers perform network I/O; the local FastEmbed
/// implementation simply hands work to a synchronous CPU pool.
#[async_trait]
pub trait DenseProvider: Send + Sync + std::fmt::Debug {
    /// Generate one dense embedding per input text.
    ///
    /// Returns `Vec<DenseEmbedding>` in the same order as `texts`. On any
    /// error returns `EmbeddingError`; never silently falls back to a
    /// different provider.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError>;

    /// Output dimensionality of vectors produced by this provider+model.
    fn output_dim(&self) -> usize;

    /// Free-form provider identifier used for logging and health messages.
    /// May include the full base URL.
    fn provider_label(&self) -> &str;

    /// Fixed-cardinality label used for Prometheus metrics. One of:
    /// `"openai"`, `"azure_openai"`, `"lmstudio"`, `"llama_cpp"`,
    /// `"fastembed"`, `"openai_compatible_other"`.
    fn metrics_label(&self) -> &'static str;

    /// Single probe call (short test string) verifying the endpoint is
    /// reachable and any credentials are valid. Used by the health check;
    /// the result may be cached for `health_probe_cache_secs`.
    async fn probe(&self) -> Result<(), EmbeddingError>;
}

/// Construct the active dense provider from configuration.
///
/// Synchronous: no network I/O. Concrete provider implementations land in
/// later commits; this factory currently rejects every value with
/// `EmbeddingError::InitializationError` so callers integrate against the
/// final signature now.
pub fn build_dense_provider(
    settings: &EmbeddingSettings,
    _num_threads: Option<usize>,
) -> Result<Arc<dyn DenseProvider>, EmbeddingError> {
    Err(EmbeddingError::InitializationError {
        message: format!(
            "dense provider not yet wired (cache_max_entries={}, model_cache_dir={:?})",
            settings.cache_max_entries, settings.model_cache_dir
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_returns_initialization_error_until_impls_land() {
        let settings = EmbeddingSettings::default();
        let err = build_dense_provider(&settings, None)
            .expect_err("factory must reject construction until impls land");
        match err {
            EmbeddingError::InitializationError { message } => {
                assert!(
                    message.contains("not yet wired"),
                    "unexpected msg: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn factory_signature_accepts_optional_num_threads() {
        let settings = EmbeddingSettings::default();
        // Both arms must compile and yield the same error variant.
        let _ = build_dense_provider(&settings, Some(4));
        let _ = build_dense_provider(&settings, None);
    }
}
