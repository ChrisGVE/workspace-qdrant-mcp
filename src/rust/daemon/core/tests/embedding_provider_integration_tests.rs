//! End-to-end integration tests for the OpenAI-compatible embedding
//! provider against a wiremock HTTP server.
//!
//! Verifies the full provider chain — request shape, batching, response
//! reordering, L2 normalization, dim drift detection, and rate-limit
//! handling — without touching any real network endpoint.

use std::sync::Arc;

use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use workspace_qdrant_core::config::EmbeddingSettings;
use workspace_qdrant_core::embedding::build_dense_provider;
use workspace_qdrant_core::EmbeddingError;

/// Per-test env-var guard. Sets a unique key=value on construction; restores
/// the previous value on drop. Ensures parallel tests do not contaminate
/// each other's environment.
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
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => std::env::set_var(&self.key, v),
            None => std::env::remove_var(&self.key),
        }
    }
}

/// A 4-D unit vector pointing along axis `idx`. Trivially normalized to
/// length 1, so the provider's `normalize_in_place` should leave it intact.
fn unit_vector(dim: usize, idx: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if idx < dim {
        v[idx] = 1.0;
    }
    v
}

/// Build an `EmbeddingSettings` configured to talk to the supplied mock
/// server with the given env var holding a fake API key. Output dim is
/// kept small so unit-vector responses stay readable.
fn build_settings(server_uri: &str, env_var: &str, output_dim: usize) -> EmbeddingSettings {
    let mut settings = EmbeddingSettings::default();
    settings.provider = "openai_compatible".to_string();
    settings.base_url = server_uri.to_string();
    settings.model = "text-embedding-test".to_string();
    settings.api_key_env_var = env_var.to_string();
    settings.output_dim = output_dim;
    settings.remote_batch_size = 2;
    settings.health_probe_cache_secs = 60;
    settings
}

#[tokio::test]
async fn embed_batch_returns_in_input_order_and_normalized() {
    let var = "WQM_INTEG_EMBED_ORDER";
    let _guard = EnvGuard::set(var, "sk-test-integ");

    let server = MockServer::start().await;
    // Respond with the two unit vectors but in REVERSED `index` order to
    // exercise reorder_by_index. The provider must hand the caller the
    // vectors in the same order as the input texts.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("Authorization", "Bearer sk-test-integ"))
        .and(header("Content-Type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object":"embedding","embedding": unit_vector(4, 1), "index": 1},
                {"object":"embedding","embedding": unit_vector(4, 0), "index": 0},
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let settings = build_settings(&server.uri(), var, 4);
    let provider = build_dense_provider(&settings, None).expect("factory must succeed");

    let texts = ["alpha", "beta"];
    let results = provider
        .embed(&texts.to_vec())
        .await
        .expect("embed must succeed");

    assert_eq!(results.len(), 2);

    // Order is preserved by index reorder: results[0] corresponds to the
    // index=0 vector (axis 0), results[1] to the index=1 vector (axis 1).
    assert!((results[0].vector[0] - 1.0).abs() < 1e-5);
    assert!(results[0].vector[1].abs() < 1e-5);
    assert!(results[1].vector[0].abs() < 1e-5);
    assert!((results[1].vector[1] - 1.0).abs() < 1e-5);

    for emb in &results {
        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "all returned vectors must be L2-normalized; got {norm}"
        );
        assert_eq!(emb.model_name, "text-embedding-test");
    }
}

#[tokio::test]
async fn embed_chunks_batches_above_remote_batch_size() {
    let var = "WQM_INTEG_EMBED_CHUNKS";
    let _guard = EnvGuard::set(var, "sk-test-integ");

    // remote_batch_size = 2 but we send 5 texts, so the provider must issue
    // ceil(5/2) = 3 HTTP requests.
    let server = MockServer::start().await;

    /// Responder that mirrors the input array length, returning a unit
    /// vector along axis `i` for the `i`-th input.
    #[derive(Clone)]
    struct Mirror;
    impl Respond for Mirror {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let inputs = body["input"].as_array().unwrap();
            let data: Vec<Value> = inputs
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0_f32; 4];
                    if i < 4 {
                        v[i] = 1.0;
                    }
                    json!({
                        "object": "embedding",
                        "embedding": v,
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

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(Mirror)
        .expect(3)
        .mount(&server)
        .await;

    let settings = build_settings(&server.uri(), var, 4);
    let provider = build_dense_provider(&settings, None).unwrap();

    let texts = ["a", "b", "c", "d", "e"];
    let results = provider
        .embed(&texts.to_vec())
        .await
        .expect("embed must succeed across multiple batches");
    assert_eq!(results.len(), 5);
    for emb in &results {
        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}

#[tokio::test]
async fn probe_records_dim_drift_against_settings() {
    let var = "WQM_INTEG_PROBE_DRIFT";
    let _guard = EnvGuard::set(var, "sk-test-integ");

    // Settings declare output_dim=4 but the upstream returns a 3-D vector.
    // The probe must succeed (the runtime atomic adapts) but the cached
    // dim should now be 3 — verifiable via `provider.output_dim()`.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object":"embedding","embedding": [1.0_f32, 0.0, 0.0], "index": 0}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let settings = build_settings(&server.uri(), var, 4);
    let provider = build_dense_provider(&settings, None).unwrap();
    assert_eq!(
        provider.output_dim(),
        4,
        "before the first probe, output_dim mirrors the configured dim"
    );

    provider.probe().await.expect("probe must succeed");
    assert_eq!(
        provider.output_dim(),
        3,
        "probe must record the actual upstream dim and update the runtime atomic"
    );
}

#[tokio::test]
async fn rate_limit_streak_surfaces_rate_limit_exhausted() {
    let var = "WQM_INTEG_RATE_LIMIT";
    let _guard = EnvGuard::set(var, "sk-test-integ");

    let server = MockServer::start().await;
    // Respond with 429 every time. The provider's RateLimitAdapter has a
    // streak budget of 5 — the 6th consecutive call should surface a
    // `RateLimitExhausted` error.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "1")
                .set_body_string("{\"error\":\"rate limited\"}"),
        )
        .mount(&server)
        .await;

    let settings = build_settings(&server.uri(), var, 4);
    let provider = build_dense_provider(&settings, None).unwrap();

    // 1..6: each call returns RemoteError(429); 7th call must escalate to
    // RateLimitExhausted because the streak has now exceeded the budget.
    let mut last_err: Option<EmbeddingError> = None;
    for _ in 0..7 {
        let res = provider.embed(&["x"]).await;
        match res {
            Err(e) => last_err = Some(e),
            Ok(_) => panic!("call must error against an all-429 mock server"),
        }
    }
    let err = last_err.expect("at least one error captured");
    assert!(
        matches!(err, EmbeddingError::RateLimitExhausted { .. }),
        "expected RateLimitExhausted after streak exhaustion, got: {err:?}"
    );
}

#[tokio::test]
async fn auth_failure_surfaces_remote_error() {
    let var = "WQM_INTEG_AUTH_FAIL";
    let _guard = EnvGuard::set(var, "sk-bad");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_string("{\"error\":\"unauthorized\"}"))
        .mount(&server)
        .await;

    let settings = build_settings(&server.uri(), var, 4);
    let provider = build_dense_provider(&settings, None).unwrap();

    let err = provider
        .embed(&["x"])
        .await
        .expect_err("401 must surface as RemoteError");
    match err {
        EmbeddingError::RemoteError {
            status_code,
            message,
        } => {
            assert_eq!(status_code, 401);
            assert!(
                message.contains("unauthorized"),
                "body must be propagated; got {message}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn factory_constructs_arc_dyn_dense_provider() {
    // Smoke test: build_dense_provider returns an Arc<dyn DenseProvider>
    // suitable for storage and cloning across handlers.
    let var = "WQM_INTEG_FACTORY";
    let _guard = EnvGuard::set(var, "sk-test-integ");
    let settings = build_settings("http://127.0.0.1:1", var, 4);
    let provider = build_dense_provider(&settings, None).expect("factory must succeed");
    let _clone: Arc<dyn workspace_qdrant_core::embedding::provider::DenseProvider> =
        Arc::clone(&provider);
    assert_eq!(provider.output_dim(), 4);
    assert_eq!(provider.metrics_label(), "openai_compatible_other");
}
