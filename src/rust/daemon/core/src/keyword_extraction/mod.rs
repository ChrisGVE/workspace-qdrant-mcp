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
pub mod canonical_tags;
pub mod hierarchy_builder;
pub mod keyword_selector;
pub mod lexical_candidates;
pub mod lsp_candidates;
pub mod pipeline;
pub mod quasi_summary;
pub mod semantic_rerank;
pub mod structural_tags;
pub mod tag_selector;
