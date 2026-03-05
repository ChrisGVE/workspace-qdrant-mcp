//! Canonical tag deduplication and hierarchical clustering.
//!
//! Merges near-duplicate tags across documents within a tenant, then
//! builds a 3-level hierarchy via agglomerative clustering.

mod clustering;
mod types;

pub use types::{CanonicalConfig, CanonicalHierarchy, CanonicalTag, TagWithVector};

use clustering::{cluster_tags, merge_duplicates};

/// Build canonical tag hierarchy from collected tags.
///
/// 1. Merge near-duplicates (similarity > merge_threshold)
/// 2. Cluster at level 2 (mid threshold)
/// 3. Cluster at level 1 (broad threshold)
pub fn build_hierarchy(tags: &[TagWithVector], config: &CanonicalConfig) -> CanonicalHierarchy {
    if tags.is_empty() {
        return CanonicalHierarchy {
            level3: Vec::new(),
            level2: Vec::new(),
            level1: Vec::new(),
        };
    }

    // Step 1: Merge near-duplicates into canonical tags (level 3)
    let mut level3 = merge_duplicates(tags, config.merge_threshold);

    // Step 2: Cluster level 3 into level 2 (sets parent_index on level3 tags)
    let mut level2 = cluster_tags(&mut level3, config.level_thresholds[1], 2);

    // Step 3: Cluster level 2 into level 1 (sets parent_index on level2 tags)
    let level1 = cluster_tags(&mut level2, config.level_thresholds[0], 1);

    CanonicalHierarchy {
        level3,
        level2,
        level1,
    }
}

#[cfg(test)]
mod tests;
