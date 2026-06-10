//! Pure helper functions for the OpenAI-compatible embedding provider.

use reqwest::header::HeaderMap;

use crate::embedding::types::EmbeddingError;

use super::types::OpenAiEmbedding;

/// Map a base URL onto the fixed metrics-label enum. Cardinality is bounded
/// to: `openai`, `azure_openai`, `lmstudio`, `llama_cpp`,
/// `openai_compatible_other`. `fastembed` is owned by `FastEmbedProvider`.
pub(crate) fn classify_provider(base_url: &str) -> &'static str {
    let lower = base_url.to_ascii_lowercase();
    if lower.contains("openai.azure.com") {
        "azure_openai"
    } else if lower.contains("api.openai.com") {
        "openai"
    } else if lower.contains("localhost:1234") || lower.contains("127.0.0.1:1234") {
        "lmstudio"
    } else if lower.contains("localhost:8080") || lower.contains("127.0.0.1:8080") {
        "llama_cpp"
    } else {
        "openai_compatible_other"
    }
}

/// Truncate to at most `max_chars` characters (Unicode scalar values, the
/// unit OpenAI-compatible endpoints count for `string_too_long` limits),
/// always cutting on a UTF-8 char boundary.
pub(super) fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

/// L2-normalize a vector in place; reject zero-norm with `GenerationError`.
pub(super) fn normalize_in_place(v: &mut [f32]) -> Result<(), EmbeddingError> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return Err(EmbeddingError::GenerationError {
            message: "zero-norm vector returned by provider; cannot normalize".to_string(),
        });
    }
    if (norm - 1.0).abs() < 1e-4 {
        return Ok(());
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(())
}

/// Reorder the response data array by `index` so the output preserves the
/// input order even when the upstream returns out-of-order entries.
pub(super) fn reorder_by_index(mut data: Vec<OpenAiEmbedding>) -> Vec<Vec<f32>> {
    data.sort_by_key(|e| e.index);
    data.into_iter().map(|e| e.embedding).collect()
}

pub(super) fn parse_retry_after_secs(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}
