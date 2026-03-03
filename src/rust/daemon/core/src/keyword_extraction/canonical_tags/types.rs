//! Data types for canonical tag deduplication and hierarchical clustering.

/// A tag with its embedding, collected from across documents.
#[derive(Debug, Clone)]
pub struct TagWithVector {
    /// The tag phrase
    pub phrase: String,
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Number of documents containing this tag
    pub doc_count: u32,
}

/// A canonical (deduplicated) tag.
#[derive(Debug, Clone)]
pub struct CanonicalTag {
    /// Canonical label (chosen from cluster members)
    pub label: String,
    /// Alternative labels (other members merged into this)
    pub aliases: Vec<String>,
    /// Centroid vector (mean of member vectors)
    pub centroid: Vec<f32>,
    /// Total document count across merged tags
    pub doc_count: u32,
    /// Hierarchy level (1=broad, 2=mid, 3=fine)
    pub level: u8,
    /// Index of parent cluster (None for top-level)
    pub parent_index: Option<usize>,
    /// Cosine similarity to parent cluster centroid (None for top-level)
    pub parent_similarity: Option<f64>,
}

/// Configuration for canonical tag building.
#[derive(Debug, Clone)]
pub struct CanonicalConfig {
    /// Similarity threshold for merging near-duplicates
    pub merge_threshold: f64,
    /// Similarity thresholds for hierarchy levels (broad, mid)
    /// Level 3 = all canonical tags (no merging)
    pub level_thresholds: [f64; 2],
}

impl Default for CanonicalConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.85,
            level_thresholds: [0.50, 0.70],
        }
    }
}

/// Result of canonical tag building.
#[derive(Debug, Clone)]
pub struct CanonicalHierarchy {
    /// Level 3: fine-grained canonical tags (after dedup)
    pub level3: Vec<CanonicalTag>,
    /// Level 2: mid-level clusters
    pub level2: Vec<CanonicalTag>,
    /// Level 1: broad topic clusters
    pub level1: Vec<CanonicalTag>,
}
