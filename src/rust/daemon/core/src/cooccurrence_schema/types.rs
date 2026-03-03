//! Types for symbol co-occurrence data structures.

/// A co-occurrence cluster: a group of symbols that appear together frequently.
#[derive(Debug, Clone)]
pub struct CooccurrenceCluster {
    /// Symbols in this cluster.
    pub symbols: Vec<String>,
    /// Minimum edge weight within the cluster.
    pub min_weight: i64,
}
