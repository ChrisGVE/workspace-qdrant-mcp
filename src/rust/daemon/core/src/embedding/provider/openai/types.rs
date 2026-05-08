//! Wire types for the OpenAI `/v1/embeddings` HTTP protocol.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub(super) struct OpenAiEmbeddingRequest<'a> {
    pub(super) input: Vec<&'a str>,
    pub(super) model: &'a str,
}

#[derive(Deserialize)]
pub(super) struct OpenAiEmbeddingResponse {
    pub(super) data: Vec<OpenAiEmbedding>,
}

#[derive(Deserialize)]
pub(super) struct OpenAiEmbedding {
    pub(super) embedding: Vec<f32>,
    pub(super) index: usize,
}
