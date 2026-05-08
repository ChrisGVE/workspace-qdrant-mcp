//! HTTP layer for the OpenAI-compatible embedding provider.
//!
//! `embed_chunk` issues a single batch request and parses the response.

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use secrecy::ExposeSecret;

use crate::embedding::types::EmbeddingError;

use super::types::{OpenAiEmbeddingRequest, OpenAiEmbeddingResponse};
use super::utils::{parse_retry_after_secs, reorder_by_index};
use super::OpenAiCompatibleProvider;

/// Maximum streak of 429/503 responses before `embed()` surfaces
/// `RateLimitExhausted` to the caller.
pub(super) const RATE_LIMIT_STREAK_BUDGET: u32 = 5;

/// Issue a single batch request to the upstream and parse the response.
/// Returns embeddings in the input order.
pub(super) async fn embed_chunk(
    provider: &OpenAiCompatibleProvider,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let waits_before = provider.rate_limiter.waits_observed();
    provider.rate_limiter.pre_request().await;
    if provider.rate_limiter.waits_observed() > waits_before {
        crate::monitoring::embedding_metrics::record_rate_limit_wait(provider.metrics_tag);
    }
    let started = std::time::Instant::now();

    let request = OpenAiEmbeddingRequest {
        input: texts.to_vec(),
        model: &provider.model,
    };

    let response = provider
        .http
        .post(provider.endpoint_url())
        .header(
            AUTHORIZATION,
            format!("Bearer {}", provider.api_key.expose_secret()),
        )
        .header(CONTENT_TYPE, "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| {
            crate::monitoring::embedding_metrics::record_request(
                provider.metrics_tag,
                &provider.model,
                None,
                started.elapsed().as_secs_f64(),
            );
            EmbeddingError::GenerationError {
                message: format!("Embedding HTTP request failed: {e}"),
            }
        })?;

    let status = response.status();
    let headers = response.headers().clone();
    provider
        .rate_limiter
        .observe_response(&headers, status.as_u16());
    crate::monitoring::embedding_metrics::record_request(
        provider.metrics_tag,
        &provider.model,
        Some(status.as_u16()),
        started.elapsed().as_secs_f64(),
    );

    if status.is_success() {
        let parsed: OpenAiEmbeddingResponse =
            response
                .json()
                .await
                .map_err(|e| EmbeddingError::GenerationError {
                    message: format!("Failed to decode embedding response: {e}"),
                })?;
        return Ok(reorder_by_index(parsed.data));
    }

    if matches!(status.as_u16(), 429 | 503) {
        let streak = provider.rate_limiter.consecutive_429s();
        if streak > RATE_LIMIT_STREAK_BUDGET {
            return Err(EmbeddingError::RateLimitExhausted {
                consecutive_429s: streak,
                retry_after_secs: parse_retry_after_secs(&headers).unwrap_or(0),
            });
        }
    }

    let body = response.text().await.unwrap_or_default();
    let truncated = if body.len() > 256 {
        &body[..256]
    } else {
        &body
    };
    Err(EmbeddingError::RemoteError {
        status_code: status.as_u16(),
        message: truncated.to_string(),
    })
}
