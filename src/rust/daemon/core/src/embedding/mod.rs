//! Embedding generation using FastEmbed
//!
//! This module provides embedding generation capabilities using the fastembed crate.
//! It generates both dense (semantic) and sparse (BM25) vectors for hybrid search.
//!
//! # Submodules
//! - `types`: Error types, configuration, and result structs
//! - `bm25`: BM25 tokenizer and sparse vector scorer
//! - `generator`: FastEmbed-based dense + sparse embedding generator

mod bm25;
mod generator;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items so external callers use the same paths as before
pub use bm25::{tokenize_for_bm25, BM25};
pub use generator::{EmbeddingGenerator, TextPreprocessor};
pub use types::{
    DenseEmbedding, EmbeddingConfig, EmbeddingError, EmbeddingResult,
    PreprocessedText, SparseEmbedding,
};
