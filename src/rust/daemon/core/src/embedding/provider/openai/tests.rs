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
        120_000,
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
        120_000,
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
        120_000,
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
        120_000,
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
        120_000,
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
        120_000,
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
        120_000,
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
            120_000,
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

/// Wiremock responder that mirrors the live remote endpoint's behavior of
/// rejecting any input string longer than a character cap with HTTP 422
/// `string_too_long` — the failure that permanently failed oversized
/// plain-text chunks in the unified queue. Compliant requests get unit
/// vectors back.
struct StringTooLongResponder {
    max_chars: usize,
}

impl Respond for StringTooLongResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).unwrap_or_else(|_| json!({"input": []}));
        let inputs = body["input"].as_array().cloned().unwrap_or_default();
        for input in &inputs {
            let chars = input.as_str().map(|s| s.chars().count()).unwrap_or(0);
            if chars > self.max_chars {
                return ResponseTemplate::new(422).set_body_json(json!({
                    "detail": [{
                        "type": "string_too_long",
                        "loc": ["body", "text", "input", "list[constrained-str]", 0],
                        "msg": format!(
                            "String should have at most {} characters",
                            self.max_chars
                        ),
                    }]
                }));
            }
        }
        let data: Vec<serde_json::Value> = (0..inputs.len())
            .map(|i| json!({"object": "embedding", "embedding": [1.0_f32], "index": i}))
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({"object": "list", "data": data}))
    }
}

#[tokio::test]
async fn test_openai_embed_truncates_oversized_input_to_char_cap() {
    let var = unique_var("trunc_cap");
    let _guard = EnvGuard::set(&var, "sk-test");
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(StringTooLongResponder { max_chars: 64 })
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(
        server.uri(),
        "m".to_string(),
        8,
        64, // provider-side max_input_chars matches the endpoint's cap
        1,
        &var,
        Duration::from_secs(60),
    )
    .unwrap();

    // Mirrors the dense text built for a plain-text file chunk: document
    // prefix + relative path header + body, far above the endpoint cap.
    let oversized = format!("passage: project/notes.txt\n\n{}", "x".repeat(200_000));
    let result = provider
        .embed(&[oversized.as_str()])
        .await
        .expect("oversized input must be truncated locally, not rejected with HTTP 422");
    assert_eq!(result.len(), 1);
    assert!(
        result[0].sequence_length <= 64,
        "embedded text must respect the cap, got {}",
        result[0].sequence_length
    );
}

#[tokio::test]
async fn test_openai_char_cap_counts_chars_not_bytes() {
    let var = unique_var("trunc_chars");
    let _guard = EnvGuard::set(&var, "sk-test");
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(StringTooLongResponder { max_chars: 50 })
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new(
        server.uri(),
        "m".to_string(),
        8,
        50,
        1,
        &var,
        Duration::from_secs(60),
    )
    .unwrap();

    // 40 box-drawing chars = 120 bytes. The cap counts CHARACTERS (what the
    // remote constrained-str validates), so this must pass through intact.
    let text = "\u{2500}".repeat(40);
    provider
        .embed(&[text.as_str()])
        .await
        .expect("40-char input must not be truncated by a 50-char cap");

    let requests = server
        .received_requests()
        .await
        .expect("request recording enabled");
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let sent = body["input"][0].as_str().unwrap();
    assert_eq!(
        sent.chars().count(),
        40,
        "multi-byte input under the char cap must be sent verbatim"
    );
}

#[test]
fn test_truncate_chars_cuts_on_char_boundary() {
    use super::utils::truncate_chars;

    assert_eq!(truncate_chars("hello", 10), "hello");
    assert_eq!(truncate_chars("hello", 5), "hello");
    assert_eq!(truncate_chars("hello", 3), "hel");
    // Multi-byte: each box-drawing char is 3 bytes, 1 char.
    let s = "\u{2500}\u{2500}\u{2500}";
    assert_eq!(truncate_chars(s, 2), "\u{2500}\u{2500}");
    assert_eq!(truncate_chars(s, 0), "");
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
