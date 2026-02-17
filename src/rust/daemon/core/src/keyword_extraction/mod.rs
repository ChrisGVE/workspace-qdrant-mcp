//! Keyword and tag extraction pipeline.
//!
//! Two-stage extraction: lexical candidate generation (TF-IDF) followed by
//! semantic reranking (cosine similarity to parent summary vector).
//!
//! Modules:
//! - `lexical_candidates`: TF-IDF n-gram extraction with filtering
//! - (future) `semantic_rerank`: FastEmbed-based reranking
//! - (future) `keyword_selector`: top-K selection with DF penalty
//! - (future) `tag_selector`: MMR diversity-based tag selection

pub mod lexical_candidates;
