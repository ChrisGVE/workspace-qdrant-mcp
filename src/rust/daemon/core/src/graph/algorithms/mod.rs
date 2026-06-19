/// Graph algorithms: PageRank, community detection, betweenness centrality,
/// and in-house deterministic Leiden community detection.
///
/// All algorithms operate as pure functions over [`crate::graph::AdjacencyExport`]
/// — an owned, index-based adjacency snapshot produced by
/// [`crate::graph::GraphStore::export_adjacency`].  No database I/O occurs
/// inside any algorithm; the caller acquires the export, releases any read lock,
/// and then invokes the algorithm (LOCK-SCOPE contract).
mod betweenness;
mod community;
pub mod leiden;
mod pagerank;

pub use betweenness::{compute_betweenness_centrality, BetweennessEntry};
pub use community::{detect_communities, Community, CommunityConfig, CommunityMember};
pub use leiden::{detect_communities_leiden, LeidenConfig};
pub use pagerank::{compute_pagerank, PageRankConfig, PageRankEntry};

#[cfg(test)]
mod tests;
