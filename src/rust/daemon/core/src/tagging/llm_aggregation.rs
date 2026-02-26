//! LLM tag aggregation for Tier 3 tagging.
//!
//! Aggregates raw string tags from multiple document chunks into deduplicated,
//! frequency-ranked `SelectedTag` values with `"llm:"` prefix.

use std::collections::HashMap;

use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

/// Aggregate tags from multiple chunks by frequency.
///
/// Tags are ranked by how many chunks mention them. The top `max_tags`
/// are returned as `SelectedTag` with `"llm:"` prefix.
pub(super) fn aggregate_llm_tags(
    chunk_tags: &[Vec<String>],
    max_tags: usize,
) -> Vec<SelectedTag> {
    if chunk_tags.is_empty() {
        return Vec::new();
    }

    let total_chunks = chunk_tags.len() as f64;
    let mut freq: HashMap<String, usize> = HashMap::new();

    for tags in chunk_tags {
        // Deduplicate within a single chunk before counting
        let mut seen = std::collections::HashSet::new();
        for tag in tags {
            if seen.insert(tag.clone()) {
                *freq.entry(tag.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(max_tags);

    ranked
        .into_iter()
        .map(|(tag, count)| SelectedTag {
            phrase: format!("llm:{}", tag),
            tag_type: TagType::Concept,
            score: count as f64 / total_chunks,
            diversity_score: 1.0,
            semantic_score: 0.0,
            ngram_size: 1,
        })
        .collect()
}
