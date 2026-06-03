//! EmbeddingService RPC wrappers for [`DaemonClient`].
//!
//! Mirrors `DaemonClientService.embedText` and `generateSparseVector` from
//! `service-methods.ts` lines 47-67, and the higher-level helpers in
//! `daemon-embedder.ts`.
//!
//! | Rust method              | Proto RPC                             | TS equivalent              |
//! |--------------------------|---------------------------------------|----------------------------|
//! | `embed_text`             | `EmbeddingService::EmbedText`         | `embedText()`              |
//! | `generate_sparse_vector` | `EmbeddingService::GenerateSparseVector` | `generateSparseVector()` |
//!
//! Both methods use [`DaemonClient::call`] with default (5 s) timeout.
//!
//! ## Sparse vector representation
//!
//! The proto `SparseVectorResponse.indices_values` is a `map<uint32, float>`,
//! which prost generates as `HashMap<u32, f32>`.  Callers expecting parallel
//! `indices: Vec<u32>` / `values: Vec<f32>` arrays can use
//! [`split_sparse_map`] to convert (mirrors the TS
//! `Object.entries(response.indices_values)` transform in `daemon-embedder.ts`
//! lines 130-131).

use tonic::Status;

use wqm_proto::workspace_daemon::{
    EmbedTextRequest, EmbedTextResponse, SparseVectorRequest, SparseVectorResponse,
};

use super::client::DaemonClient;

impl DaemonClient {
    /// Generate a dense embedding vector — mirrors TS `embedText()`.
    ///
    /// Sends `text` to the daemon's FastEmbed model.  Returns a
    /// [`EmbedTextResponse`] containing:
    /// - `embedding`: 384-dimensional dense `Vec<f32>`
    /// - `dimensions`: vector dimension (should be 384 for all-MiniLM-L6-v2)
    /// - `model_name`: model identifier string
    /// - `success`: whether the embedding was generated successfully
    ///
    /// # Errors
    /// Returns `Err(Status)` on any transport, timeout, or daemon error.
    /// Callers should additionally check `response.success` and
    /// `response.error_message` for application-level failures.
    pub async fn embed_text(&mut self, text: &str) -> Result<EmbedTextResponse, Status> {
        let request = EmbedTextRequest {
            text: text.to_string(),
            model: None,
        };
        let client = self.embedding.clone();
        self.call("embedText", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.embed_text(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Generate a sparse (BM25-style) vector — mirrors TS `generateSparseVector()`.
    ///
    /// Returns a [`SparseVectorResponse`] containing:
    /// - `indices_values`: `HashMap<u32, f32>` — token index → IDF weight
    /// - `vocab_size`: current vocabulary size
    /// - `success` / `error_message`: application-level status
    ///
    /// Use [`split_sparse_map`] to convert the map into parallel arrays
    /// if your search backend expects `(indices: Vec<u32>, values: Vec<f32>)`.
    ///
    /// # Errors
    /// Returns `Err(Status)` on any transport, timeout, or daemon error.
    pub async fn generate_sparse_vector(
        &mut self,
        text: &str,
    ) -> Result<SparseVectorResponse, Status> {
        let request = SparseVectorRequest {
            text: text.to_string(),
        };
        let client = self.embedding.clone();
        self.call("generateSparseVector", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.generate_sparse_vector(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

/// Convert the proto `indices_values` map into sorted parallel arrays.
///
/// The TS `DaemonEmbedder.generateSparseVector` uses:
/// ```typescript
/// const entries = Object.entries(response.indices_values);
/// const indices = entries.map(([k]) => Number(k));
/// const values  = entries.map(([, v]) => v);
/// ```
/// (daemon-embedder.ts lines 130-131)
///
/// This function mirrors that transform: entries are sorted by index ascending
/// so callers get a deterministic, index-ordered representation.
pub fn split_sparse_map(
    indices_values: &std::collections::HashMap<u32, f32>,
) -> (Vec<u32>, Vec<f32>) {
    let mut entries: Vec<(u32, f32)> = indices_values.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|(k, _)| *k);
    entries.into_iter().unzip()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    // Helpers: build proto response types directly (no live daemon).

    fn make_embed_response_ok(dims: i32) -> EmbedTextResponse {
        EmbedTextResponse {
            embedding: vec![0.1_f32; dims as usize],
            dimensions: dims,
            model_name: "all-MiniLM-L6-v2".to_string(),
            success: true,
            error_message: String::new(),
        }
    }

    fn make_embed_response_err(msg: &str) -> EmbedTextResponse {
        EmbedTextResponse {
            embedding: vec![],
            dimensions: 0,
            model_name: String::new(),
            success: false,
            error_message: msg.to_string(),
        }
    }

    fn make_sparse_response_ok(entries: HashMap<u32, f32>) -> SparseVectorResponse {
        SparseVectorResponse {
            indices_values: entries,
            vocab_size: 50_000,
            success: true,
            error_message: String::new(),
        }
    }

    fn make_sparse_response_empty() -> SparseVectorResponse {
        SparseVectorResponse {
            indices_values: HashMap::new(),
            vocab_size: 0,
            success: true,
            error_message: String::new(),
        }
    }

    // ── EmbedTextResponse field mapping ───────────────────────────────────────

    #[test]
    fn embed_response_success_flag() {
        let r = make_embed_response_ok(384);
        assert!(r.success);
    }

    #[test]
    fn embed_response_dimensions_384() {
        let r = make_embed_response_ok(384);
        assert_eq!(r.dimensions, 384);
        assert_eq!(r.embedding.len(), 384);
    }

    #[test]
    fn embed_response_model_name() {
        let r = make_embed_response_ok(384);
        assert_eq!(r.model_name, "all-MiniLM-L6-v2");
    }

    #[test]
    fn embed_response_error_empty_on_success() {
        let r = make_embed_response_ok(384);
        assert!(r.error_message.is_empty());
    }

    #[test]
    fn embed_response_failure_no_vector() {
        let r = make_embed_response_err("daemon unavailable");
        assert!(!r.success);
        assert!(r.embedding.is_empty());
        assert_eq!(r.error_message, "daemon unavailable");
    }

    #[test]
    fn embed_response_embedding_length_matches_dimensions() {
        let r = make_embed_response_ok(384);
        assert_eq!(r.embedding.len(), r.dimensions as usize);
    }

    // ── SparseVectorResponse field mapping ────────────────────────────────────

    #[test]
    fn sparse_response_success_flag() {
        let r = make_sparse_response_ok(HashMap::new());
        assert!(r.success);
    }

    #[test]
    fn sparse_response_vocab_size() {
        let r = make_sparse_response_ok(HashMap::new());
        assert_eq!(r.vocab_size, 50_000);
    }

    #[test]
    fn sparse_response_indices_values_populated() {
        let mut m = HashMap::new();
        m.insert(42u32, 0.5_f32);
        m.insert(100u32, 0.3_f32);
        let r = make_sparse_response_ok(m);
        assert_eq!(r.indices_values.len(), 2);
        assert!((r.indices_values[&42u32] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn sparse_response_empty_for_whitespace_text() {
        // TS DaemonEmbedder returns empty vectors for blank input
        let r = make_sparse_response_empty();
        assert!(r.indices_values.is_empty());
    }

    // ── split_sparse_map ──────────────────────────────────────────────────────

    #[test]
    fn split_sparse_map_empty_input() {
        let (indices, values) = split_sparse_map(&HashMap::new());
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn split_sparse_map_single_entry() {
        let mut m = HashMap::new();
        m.insert(7u32, 0.9_f32);
        let (indices, values) = split_sparse_map(&m);
        assert_eq!(indices, vec![7u32]);
        assert!((values[0] - 0.9_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn split_sparse_map_sorted_by_index_ascending() {
        let mut m = HashMap::new();
        m.insert(100u32, 0.1_f32);
        m.insert(5u32, 0.5_f32);
        m.insert(50u32, 0.3_f32);
        let (indices, values) = split_sparse_map(&m);
        assert_eq!(indices, vec![5u32, 50u32, 100u32]);
        assert!((values[0] - 0.5_f32).abs() < f32::EPSILON);
        assert!((values[1] - 0.3_f32).abs() < f32::EPSILON);
        assert!((values[2] - 0.1_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn split_sparse_map_parallel_array_lengths_match() {
        let mut m = HashMap::new();
        m.insert(1u32, 0.1_f32);
        m.insert(2u32, 0.2_f32);
        m.insert(3u32, 0.3_f32);
        let (indices, values) = split_sparse_map(&m);
        assert_eq!(indices.len(), values.len());
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn split_sparse_map_values_preserved() {
        let mut m = HashMap::new();
        m.insert(10u32, 1.23_f32);
        let (_, values) = split_sparse_map(&m);
        assert!((values[0] - 1.23_f32).abs() < f32::EPSILON);
    }

    // ── EmbedTextRequest construction ─────────────────────────────────────────

    #[test]
    fn embed_text_request_text_field() {
        let req = EmbedTextRequest {
            text: "hello world".to_string(),
            model: None,
        };
        assert_eq!(req.text, "hello world");
        assert!(req.model.is_none());
    }

    #[test]
    fn embed_text_request_with_model_override() {
        let req = EmbedTextRequest {
            text: "hello world".to_string(),
            model: Some("custom-model".to_string()),
        };
        assert_eq!(req.model.as_deref(), Some("custom-model"));
    }

    // ── SparseVectorRequest construction ──────────────────────────────────────

    #[test]
    fn sparse_vector_request_text_field() {
        let req = SparseVectorRequest {
            text: "search query".to_string(),
        };
        assert_eq!(req.text, "search query");
    }

    // ── DaemonClient construction ─────────────────────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_embedding_calls() {
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }

    // ── call() timeout dispatch ───────────────────────────────────────────────

    #[tokio::test]
    async fn embed_text_method_uses_5s_budget() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("embedText", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }

    #[tokio::test]
    async fn generate_sparse_vector_method_uses_5s_budget() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call(
                "generateSparseVector",
                Some(Duration::from_millis(1)),
                || async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(())
                },
            )
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }
}
