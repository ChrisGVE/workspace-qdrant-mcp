//! Keyword and tag extraction pipeline.
//!
//! Two-stage extraction: lexical candidate generation (TF-IDF) followed by
//! semantic reranking (cosine similarity to parent summary vector).
//!
//! Modules:
//! - `lexical_candidates`: TF-IDF n-gram extraction with filtering
//! - `semantic_rerank`: FastEmbed-based reranking
//! - `keyword_selector`: top-K selection with DF penalty
//! - `tag_selector`: MMR diversity-based tag selection

pub mod basket_assignment;
pub mod keyword_selector;
pub mod lexical_candidates;
pub mod semantic_rerank;
pub mod tag_selector;
