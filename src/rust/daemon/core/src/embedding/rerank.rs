//! Remote cross-encoder reranker speaking the Infinity/Cohere-style
//! `/v1/rerank` protocol.
//!
//! Activated by `WQM_RERANK_BASE_URL` (e.g. `http://wqm-embeddings-gpu:7997`,
//! the same Infinity sidecar that serves dense embeddings); when set, the gRPC
//! EmbeddingService POSTs rerank requests there instead of lazy-loading the
//! in-process fastembed cross-encoder (jina-turbo, English-only — measured
//! strictly worse on code queries). `WQM_RERANK_MODEL` selects the served
//! model (default: BAAI/bge-reranker-v2-m3, multilingual).
//!
//! Mirrors `OpenAiCompatibleProvider`'s transport conventions: non-blocking
//! construction, `{base_url}/v1/<route>` endpoint derivation, JSON body,
//! bounded timeout. No Authorization header — this targets the unauthenticated
//! in-stack sidecars, not a public rerank SaaS. There is deliberately no
//! fallback to the local fastembed reranker on remote failure: silently
//! swapping to a different (worse) model would change scoring semantics
//! mid-flight; the search layer already fails open to the pre-rerank order.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::embedding::types::EmbeddingError;

/// Env var that activates the remote reranker (endpoint base URL, no path).
pub const RERANK_BASE_URL_ENV: &str = "WQM_RERANK_BASE_URL";

/// Env var that selects the served rerank model.
pub const RERANK_MODEL_ENV: &str = "WQM_RERANK_MODEL";

/// Default multilingual cross-encoder (568M params, 8k context, PT/EN/100+
/// languages). Served by the Infinity GPU sidecar alongside the e5 embedder.
pub const DEFAULT_RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";

/// HTTP request timeout. Reranking a ~30-doc pool on the GPU sidecar takes
/// tens of milliseconds; the bound only guards against a wedged endpoint.
const HTTP_TIMEOUT_SECS: u64 = 30;

#[derive(Serialize)]
struct RerankHttpRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    /// Scores only — the caller already holds the documents by index.
    return_documents: bool,
}

#[derive(Deserialize)]
struct RerankHttpResponse {
    results: Vec<RerankHttpResult>,
}

#[derive(Deserialize)]
struct RerankHttpResult {
    index: usize,
    relevance_score: f32,
}

/// Remote `/v1/rerank` client. Cheap to clone (`reqwest::Client` is an `Arc`
/// internally).
#[derive(Debug, Clone)]
pub struct RemoteReranker {
    endpoint: String,
    model: String,
    http: reqwest::Client,
}

impl RemoteReranker {
    /// Build from `WQM_RERANK_BASE_URL` / `WQM_RERANK_MODEL`. Returns `None`
    /// when the base URL is unset or blank (remote reranking disabled —
    /// callers fall back to the in-process fastembed cross-encoder).
    pub fn from_env() -> Option<Self> {
        let base = std::env::var(RERANK_BASE_URL_ENV).ok()?;
        let base = base.trim();
        if base.is_empty() {
            return None;
        }
        let model = std::env::var(RERANK_MODEL_ENV)
            .ok()
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| DEFAULT_RERANK_MODEL.to_string());
        match Self::new(base, model) {
            Ok(reranker) => Some(reranker),
            Err(e) => {
                tracing::warn!("Remote reranker configured but unusable: {e}");
                None
            }
        }
    }

    /// Construct against `base_url` (`scheme://host[:port]`, no trailing
    /// path). Non-blocking — no network call.
    pub fn new(base_url: &str, model: String) -> Result<Self, EmbeddingError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
            .build()
            .map_err(|e| EmbeddingError::InitializationError {
                message: format!("Failed to construct rerank HTTP client: {e}"),
            })?;
        Ok(Self {
            endpoint: format!("{}/v1/rerank", base_url.trim_end_matches('/')),
            model,
            http,
        })
    }

    /// Served model id (for logs/labels).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Full endpoint URL (for logs/labels).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Score `documents` against `query`, returning `(index, score)` pairs
    /// sorted by score descending. `index` refers to the input order; entries
    /// with an out-of-range index (malformed upstream) are dropped.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<(usize, f32)>, EmbeddingError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let request = RerankHttpRequest {
            model: &self.model,
            query,
            documents,
            return_documents: false,
        };
        let response = self
            .http
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::GenerationError {
                message: format!("Rerank HTTP request failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::RemoteError {
                status_code: status.as_u16(),
                // char-based truncation: byte slicing could split a UTF-8
                // sequence in a non-ASCII error body and panic.
                message: body.chars().take(256).collect(),
            });
        }

        let parsed: RerankHttpResponse =
            response
                .json()
                .await
                .map_err(|e| EmbeddingError::GenerationError {
                    message: format!("Failed to decode rerank response: {e}"),
                })?;

        let mut ranked: Vec<(usize, f32)> = parsed
            .results
            .into_iter()
            .filter(|r| r.index < documents.len())
            .map(|r| (r.index, r.relevance_score))
            .collect();
        // Upstream returns descending order, but don't depend on it.
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn docs(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("document {i}")).collect()
    }

    #[test]
    fn endpoint_derivation_strips_trailing_slash() {
        let r = RemoteReranker::new("http://host:7997/", "m".to_string()).unwrap();
        assert_eq!(r.endpoint(), "http://host:7997/v1/rerank");
        let r = RemoteReranker::new("http://host:7997", "m".to_string()).unwrap();
        assert_eq!(r.endpoint(), "http://host:7997/v1/rerank");
    }

    #[tokio::test]
    async fn rerank_parses_and_sorts_descending() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(body_partial_json(json!({
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "q",
                "return_documents": false,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [
                    { "index": 0, "relevance_score": 0.1 },
                    { "index": 2, "relevance_score": 0.9 },
                    { "index": 1, "relevance_score": 0.5 },
                ]
            })))
            .mount(&server)
            .await;

        let reranker =
            RemoteReranker::new(&server.uri(), DEFAULT_RERANK_MODEL.to_string()).unwrap();
        let ranked = reranker.rerank("q", &docs(3)).await.unwrap();
        assert_eq!(ranked, vec![(2, 0.9), (1, 0.5), (0, 0.1)]);
    }

    #[tokio::test]
    async fn rerank_drops_out_of_range_indices() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [
                    { "index": 7, "relevance_score": 0.9 },
                    { "index": 1, "relevance_score": 0.5 },
                ]
            })))
            .mount(&server)
            .await;

        let reranker = RemoteReranker::new(&server.uri(), "m".to_string()).unwrap();
        let ranked = reranker.rerank("q", &docs(2)).await.unwrap();
        assert_eq!(ranked, vec![(1, 0.5)]);
    }

    #[tokio::test]
    async fn rerank_surfaces_remote_error_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(503).set_body_string("model loading"))
            .mount(&server)
            .await;

        let reranker = RemoteReranker::new(&server.uri(), "m".to_string()).unwrap();
        let err = reranker.rerank("q", &docs(1)).await.unwrap_err();
        match err {
            EmbeddingError::RemoteError {
                status_code,
                message,
            } => {
                assert_eq!(status_code, 503);
                assert_eq!(message, "model loading");
            }
            other => panic!("expected RemoteError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rerank_empty_documents_short_circuits() {
        // No mock server needed — must not issue a request at all.
        let reranker = RemoteReranker::new("http://127.0.0.1:1", "m".to_string()).unwrap();
        let ranked = reranker.rerank("q", &[]).await.unwrap();
        assert!(ranked.is_empty());
    }
}
