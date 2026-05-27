//! GraphService gRPC implementation
//!
//! Provides code relationship graph queries: traversal, impact analysis,
//! statistics, PageRank, community detection, betweenness centrality,
//! backend migration, and narrative-code traversal. All queries use a
//! shared read lock on the graph store.

mod analytics_handlers;
mod handlers;
mod helpers;
mod narrative_query;
mod service_impl;
mod validation;

// Re-export primary types
pub use service_impl::GraphServiceImpl;

#[cfg(test)]
mod tests;
