//! Shared helpers extracted from scattered inline patterns.
//!
//! These modules consolidate duplicated logic that was previously
//! copy-pasted across multiple processing functions.

pub mod collection_ensure;
pub mod embedding_pipeline;
pub mod payload_builder;
pub mod point_builder;

pub use collection_ensure::ensure_collection;
pub use embedding_pipeline::{embed_with_sparse, EmbedResult};
pub use payload_builder::PayloadBuilder;
pub use point_builder::PointBuilder;
