/// Configuration for affinity grouping.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Minimum cosine similarity between two project aggregate embeddings
    /// to group them (default: 0.7).
    pub similarity_threshold: f64,
    /// Maximum number of projects in a single affinity group (0 = unlimited).
    pub max_group_size: usize,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_group_size: 0,
        }
    }
}
