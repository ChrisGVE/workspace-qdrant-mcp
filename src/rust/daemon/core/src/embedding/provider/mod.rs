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
use std::time::Duration;

use async_trait::async_trait;

use crate::config::EmbeddingSettings;

pub mod fastembed;
pub mod health_monitor;
pub mod openai;
pub mod rate_limit;
pub use fastembed::FastEmbedProvider;
pub use health_monitor::ProviderHealthMonitor;
pub use openai::OpenAiCompatibleProvider;
pub use rate_limit::RateLimitAdapter;

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
/// Synchronous: no network I/O. Dispatch is config-driven via
/// `settings.provider` (`"fastembed"` or `"openai_compatible"`). Any other
/// value yields `EmbeddingError::InitializationError`.
pub fn build_dense_provider(
    settings: &EmbeddingSettings,
    num_threads: Option<usize>,
) -> Result<Arc<dyn DenseProvider>, EmbeddingError> {
    match settings.provider.as_str() {
        "fastembed" => {
            let provider = FastEmbedProvider::new(
                DEFAULT_FASTEMBED_BATCH_SIZE,
                settings.model_cache_dir.clone(),
                num_threads,
            );
            Ok(Arc::new(provider))
        }
        "openai_compatible" => {
            let provider = OpenAiCompatibleProvider::new(
                settings.base_url.clone(),
                settings.model.clone(),
                settings.remote_batch_size,
                settings.max_input_chars,
                settings.output_dim,
                &settings.api_key_env_var,
                Duration::from_secs(settings.health_probe_cache_secs),
            )?;
            Ok(Arc::new(provider))
        }
        other => Err(EmbeddingError::InitializationError {
            message: format!(
                "Unknown embedding provider '{}': expected 'fastembed' or 'openai_compatible'",
                other
            ),
        }),
    }
}

const DEFAULT_FASTEMBED_BATCH_SIZE: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn fastembed_settings() -> EmbeddingSettings {
        let mut settings = EmbeddingSettings::default();
        settings.provider = "fastembed".to_string();
        settings
    }

    #[test]
    fn factory_constructs_fastembed_provider() {
        let settings = fastembed_settings();
        let provider = build_dense_provider(&settings, None).expect("factory must succeed");
        assert_eq!(provider.output_dim(), 384);
        assert_eq!(provider.metrics_label(), "fastembed");
    }

    #[test]
    fn factory_signature_accepts_optional_num_threads() {
        let settings = fastembed_settings();
        let _ = build_dense_provider(&settings, Some(4)).unwrap();
        let _ = build_dense_provider(&settings, None).unwrap();
    }

    #[test]
    fn test_factory_fastembed() {
        let settings = fastembed_settings();
        let provider = build_dense_provider(&settings, None).expect("factory must succeed");
        assert_eq!(provider.output_dim(), 384);
    }

    #[test]
    #[serial]
    fn test_factory_openai_compatible_no_network() {
        let env_var = "WQM_TEST_FACTORY_OPENAI_KEY";
        std::env::set_var(env_var, "sk-test-no-network");

        let mut settings = EmbeddingSettings::default();
        settings.provider = "openai_compatible".to_string();
        settings.api_key_env_var = env_var.to_string();
        settings.output_dim = 1536;

        let provider = build_dense_provider(&settings, None).expect("factory must succeed");
        assert_eq!(provider.output_dim(), 1536);

        std::env::remove_var(env_var);
    }

    #[test]
    fn test_factory_unknown_provider() {
        let mut settings = EmbeddingSettings::default();
        settings.provider = "unknown".to_string();
        let err = build_dense_provider(&settings, None).expect_err("must reject unknown provider");
        assert!(matches!(err, EmbeddingError::InitializationError { .. }));
    }
}
