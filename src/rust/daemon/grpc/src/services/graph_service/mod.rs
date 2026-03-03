//! GraphService gRPC implementation
//!
//! Provides code relationship graph queries: traversal, impact analysis,
//! statistics, PageRank, community detection, betweenness centrality,
//! and backend migration. All queries use a shared read lock on the graph store.

mod helpers;
mod service_impl;
mod handlers;

// Re-export primary types
pub use service_impl::GraphServiceImpl;

#[cfg(test)]
mod tests;
