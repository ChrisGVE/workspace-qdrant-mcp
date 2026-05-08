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
}

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
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const TEST_KEY_VAR: &str = "WQM_OPENAI_TEST_KEY";

    /// Per-test env-var guard. Sets `key=value` on construction; restores the
    /// previous value on drop. Each call uses a unique env-var name so tests
    /// can safely run in parallel without contaminating one another.
    struct EnvGuard {
        key: String,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self {
                key: key.to_string(),
                previous,
            }
        }

        fn unset(key: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            Self {
                key: key.to_string(),
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }

    fn unique_var(suffix: &str) -> String {
        format!("{TEST_KEY_VAR}_{suffix}")
    }

    fn build_provider(server: &MockServer, env_var: &str) -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new(
            server.uri(),
            "text-embedding-3-small".to_string(),
            128,
            1536,
            env_var,
            Duration::from_secs(60),
        )
        .expect("provider construction must succeed when env var is set")
    }

    fn unit_vector(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        // Use a non-zero, easy-to-validate pattern.
        for (i, x) in v.iter_mut().enumerate() {
            *x = if i == idx { 1.0 } else { 0.0 };
        }
        v
    }

    #[tokio::test]
    async fn test_openai_new_is_nonblocking() {
        let var = unique_var("nonblock");
        let _guard = EnvGuard::set(&var, "sk-test-fake");
        // No mock server running — construction must still succeed.
        let provider = OpenAiCompatibleProvider::new(
            "http://127.0.0.1:1".to_string(),
            "text-embedding-3-small".to_string(),
            64,
            1536,
            &var,
            Duration::from_secs(60),
        )
        .expect("new must not perform any network I/O");
        assert_eq!(provider.output_dim(), 1536);
    }

    #[tokio::test]
    async fn test_openai_embed_single_text() {
        let var = unique_var("embed_single");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![3.0_f32, 4.0],"index":0}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let result = provider.embed(&["hello"]).await.unwrap();
        assert_eq!(result.len(), 1);
        let norm: f32 = result[0].vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "vector must be L2-normalized");
    }

    #[tokio::test]
    async fn test_openai_embed_batch_ordering() {
        let var = unique_var("batch_order");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"object":"embedding","embedding": unit_vector(4, 2),"index":2},
                    {"object":"embedding","embedding": unit_vector(4, 0),"index":0},
                    {"object":"embedding","embedding": unit_vector(4, 1),"index":1}
                ]
            })))
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let result = provider.embed(&["a", "b", "c"]).await.unwrap();
        assert_eq!(result.len(), 3);
        // Vector at position i should be unit vector with 1.0 at index i.
        for (i, emb) in result.iter().enumerate() {
            assert!((emb.vector[i] - 1.0).abs() < 1e-5, "position {i} mismatch");
        }
    }

    #[tokio::test]
    async fn test_openai_embed_401_returns_remote_error() {
        let var = unique_var("err_401");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let err = provider.embed(&["hello"]).await.unwrap_err();
        match err {
            EmbeddingError::RemoteError { status_code, .. } => assert_eq!(status_code, 401),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_openai_embed_500_returns_remote_error() {
        let var = unique_var("err_500");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let err = provider.embed(&["hello"]).await.unwrap_err();
        match err {
            EmbeddingError::RemoteError { status_code, .. } => assert_eq!(status_code, 500),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_openai_embed_503_with_retry_after_rate_limits() {
        let var = unique_var("err_503");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(503)
                    .insert_header("Retry-After", "1")
                    .set_body_string("retry"),
            )
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let err = provider.embed(&["hello"]).await.unwrap_err();
        match err {
            EmbeddingError::RemoteError { status_code, .. } => assert_eq!(status_code, 503),
            other => panic!("unexpected: {other:?}"),
        }
        assert_eq!(provider.rate_limiter.consecutive_429s(), 1);
    }

    /// Wiremock responder that mirrors the input array length back as a
    /// batch of unit vectors. Lets the chunking test serve variable-size
    /// requests from a single mounted mock.
    struct EchoLenResponder;
    impl Respond for EchoLenResponder {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value =
                serde_json::from_slice(&request.body).unwrap_or_else(|_| json!({"input": []}));
            let n = body["input"].as_array().map(|a| a.len()).unwrap_or(0);
            let data: Vec<serde_json::Value> = (0..n)
                .map(|i| {
                    json!({
                        "object": "embedding",
                        "embedding": [1.0_f32],
                        "index": i,
                    })
                })
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": data,
            }))
        }
    }

    #[tokio::test]
    async fn test_openai_embed_texts_chunked_by_batch_size() {
        let var = unique_var("chunking");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(EchoLenResponder)
            .expect(3) // 5 texts at batch_size=2 → 2+2+1 = 3 requests
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            server.uri(),
            "m".to_string(),
            2,
            1,
            &var,
            Duration::from_secs(60),
        )
        .unwrap();
        let result = provider
            .embed(&["a", "b", "c", "d", "e"])
            .await
            .expect("embed must succeed");
        assert_eq!(result.len(), 5);
        // Mock's expect(3) is asserted at server drop.
    }

    #[tokio::test]
    async fn test_openai_output_dim_from_config() {
        let var = unique_var("dim_cfg");
        let _guard = EnvGuard::set(&var, "sk-test");
        let provider = OpenAiCompatibleProvider::new(
            "http://127.0.0.1:1".to_string(),
            "m".to_string(),
            32,
            1536,
            &var,
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(provider.output_dim(), 1536);
    }

    #[tokio::test]
    async fn test_openai_probe_updates_output_dim_on_mismatch() {
        let var = unique_var("dim_drift");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        let real_dim = 768;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![0.1_f32; real_dim],"index":0}]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(
            server.uri(),
            "m".to_string(),
            32,
            1536,
            &var,
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(provider.output_dim(), 1536);
        provider.probe().await.expect("probe must succeed");
        assert_eq!(
            provider.output_dim(),
            real_dim,
            "probe must update atomic on dim drift"
        );
    }

    #[tokio::test]
    async fn test_openai_probe_returns_ok_on_200() {
        let var = unique_var("probe_ok");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![1.0_f32; 1536],"index":0}]
            })))
            .mount(&server)
            .await;
        let provider = build_provider(&server, &var);
        provider.probe().await.expect("probe must succeed");
    }

    #[tokio::test]
    async fn test_openai_probe_returns_err_on_401() {
        let var = unique_var("probe_401");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("auth"))
            .mount(&server)
            .await;
        let provider = build_provider(&server, &var);
        let err = provider.probe().await.expect_err("probe must fail on 401");
        match err {
            EmbeddingError::RemoteError { status_code, .. } => assert_eq!(status_code, 401),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_openai_probe_cached_within_ttl() {
        let var = unique_var("probe_cache");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![1.0_f32; 1536],"index":0}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let provider = build_provider(&server, &var);
        provider.probe().await.unwrap();
        provider.probe().await.unwrap();
        // Mock's expect(1) is asserted on server drop.
    }

    #[tokio::test]
    async fn test_api_key_absent_returns_init_error() {
        let var = unique_var("absent");
        let _guard = EnvGuard::unset(&var);
        let result = OpenAiCompatibleProvider::new(
            "http://127.0.0.1:1".to_string(),
            "m".to_string(),
            32,
            1536,
            &var,
            Duration::from_secs(60),
        );
        match result {
            Err(EmbeddingError::InitializationError { .. }) => {}
            Ok(_) => panic!("missing env var must error"),
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_api_key_not_in_tracing_or_debug_output() {
        let var = unique_var("debug_redact");
        let secret = "sk-do-not-leak-12345";
        let _guard = EnvGuard::set(&var, secret);
        let provider = OpenAiCompatibleProvider::new(
            "http://127.0.0.1:1".to_string(),
            "m".to_string(),
            32,
            1536,
            &var,
            Duration::from_secs(60),
        )
        .unwrap();
        let dbg = format!("{provider:?}");
        assert!(!dbg.contains(secret), "Debug must not leak the API key");
        assert!(
            !dbg.contains("Bearer "),
            "Debug must not include auth header"
        );
    }

    #[tokio::test]
    async fn test_metrics_label_cardinality() {
        let var = unique_var("metrics");
        let _guard = EnvGuard::set(&var, "sk-test");
        let cases: &[(&str, &str)] = &[
            ("https://api.openai.com", "openai"),
            ("https://my-resource.openai.azure.com", "azure_openai"),
            ("http://localhost:1234", "lmstudio"),
            ("http://localhost:8080", "llama_cpp"),
            ("http://example.org", "openai_compatible_other"),
        ];
        for (url, expected) in cases {
            let provider = OpenAiCompatibleProvider::new(
                url.to_string(),
                "m".to_string(),
                32,
                1536,
                &var,
                Duration::from_secs(60),
            )
            .unwrap();
            assert_eq!(provider.metrics_label(), *expected, "url={url}");
        }
    }

    #[tokio::test]
    async fn test_openai_embed_zero_norm_vector_returns_error() {
        let var = unique_var("zero_norm");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![0.0_f32; 4],"index":0}]
            })))
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        let err = provider.embed(&["x"]).await.unwrap_err();
        match err {
            EmbeddingError::GenerationError { message } => {
                assert!(message.contains("zero-norm"), "msg={message}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_authorization_header_sent() {
        let var = unique_var("auth_header");
        let _guard = EnvGuard::set(&var, "sk-test");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object":"embedding","embedding": vec![1.0_f32],"index":0}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = build_provider(&server, &var);
        provider.embed(&["hello"]).await.unwrap();
    }
}
