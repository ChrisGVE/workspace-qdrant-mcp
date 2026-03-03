//! Symbol co-occurrence schema and operations.
//!
//! Stores pairwise symbol co-occurrence counts per tenant/collection.
//! Used by the keyword extraction pipeline to compute degree centrality
//! for concept tag boosting.
//!
//! Edges are stored with canonical ordering (`symbol_a < symbol_b`)
//! to halve storage and simplify lookups.

mod operations;
mod schema;
mod tests;
mod types;

pub use operations::{
    find_clusters, get_betweenness_centrality, get_degree_centrality, get_neighbors,
    upsert_cooccurrences,
};
pub use schema::{CREATE_COOCCURRENCE_INDEXES_SQL, CREATE_SYMBOL_COOCCURRENCE_SQL};
pub use types::CooccurrenceCluster;
