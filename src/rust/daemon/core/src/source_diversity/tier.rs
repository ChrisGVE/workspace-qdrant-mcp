/// Tier-based diversity re-ranking and diversity metrics.
///
/// Groups results into score tiers and interleaves sources within
/// each tier to prevent clustering. Also provides diversity score
/// calculations for monitoring.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{extract_file, extract_source};
use crate::storage::SearchResult;

/// Diversity re-ranking configuration (tier-based interleaving).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Whether diversity re-ranking is enabled.
    pub enabled: bool,
    /// Maximum results from a single source in the final output.
    pub max_per_source: usize,
    /// Score difference threshold for grouping into tiers.
    /// Results within this delta of each other are in the same tier.
    pub score_tier_threshold: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_per_source: 3,
            score_tier_threshold: 0.05,
        }
    }
}

/// Apply source diversity re-ranking to search results.
///
/// Algorithm:
/// 1. Group results into score tiers (within `config.score_tier_threshold`)
/// 2. Within each tier, interleave results from different sources
/// 3. Enforce `config.max_per_source` across the full result set
/// 4. Preserve overall score ordering across tiers
///
/// Returns the re-ranked results (may be shorter if sources are capped).
pub fn diversify_results(
    results: Vec<SearchResult>,
    config: &DiversityConfig,
) -> Vec<SearchResult> {
    if !config.enabled || results.is_empty() {
        return results;
    }

    let tiers = build_score_tiers(&results, config.score_tier_threshold);

    let mut output = Vec::with_capacity(results.len());
    let mut source_counts: HashMap<String, usize> = HashMap::new();

    for tier in tiers {
        let diversified_tier = interleave_tier(tier);

        for result in diversified_tier {
            let source = extract_source(&result);
            let count = source_counts.entry(source).or_insert(0);

            if *count < config.max_per_source {
                *count += 1;
                output.push(result);
            }
        }
    }

    output
}

/// Build score tiers from sorted results.
///
/// A tier is a group of results where each result's score is within
/// `threshold` of the first result in the tier.
pub fn build_score_tiers(results: &[SearchResult], threshold: f32) -> Vec<Vec<SearchResult>> {
    if results.is_empty() {
        return Vec::new();
    }

    let mut tiers: Vec<Vec<SearchResult>> = Vec::new();
    let mut current_tier: Vec<SearchResult> = vec![results[0].clone()];
    let mut tier_top_score = results[0].score;

    for result in results.iter().skip(1) {
        if (tier_top_score - result.score).abs() <= threshold {
            current_tier.push(result.clone());
        } else {
            tiers.push(current_tier);
            current_tier = vec![result.clone()];
            tier_top_score = result.score;
        }
    }

    if !current_tier.is_empty() {
        tiers.push(current_tier);
    }

    tiers
}

/// Interleave results within a tier by source.
///
/// Takes results from the same score tier and reorders them so that
/// results from different sources alternate, maintaining relative
/// ordering within each source.
fn interleave_tier(mut tier: Vec<SearchResult>) -> Vec<SearchResult> {
    if tier.len() <= 1 {
        return tier;
    }

    let mut source_groups: HashMap<String, Vec<SearchResult>> = HashMap::new();
    let mut source_order: Vec<String> = Vec::new();

    for result in tier.drain(..) {
        let source = extract_source(&result);
        if !source_groups.contains_key(&source) {
            source_order.push(source.clone());
        }
        source_groups.entry(source).or_default().push(result);
    }

    let mut output = Vec::new();
    let mut exhausted = 0;
    let total_sources = source_order.len();
    let mut indices: Vec<usize> = vec![0; total_sources];

    while exhausted < total_sources {
        for (i, source) in source_order.iter().enumerate() {
            if let Some(group) = source_groups.get(source) {
                if indices[i] < group.len() {
                    output.push(group[indices[i]].clone());
                    indices[i] += 1;
                    if indices[i] == group.len() {
                        exhausted += 1;
                    }
                }
            }
        }
    }

    output
}

/// Calculate source diversity score for a set of results.
///
/// Returns a value between 0.0 and 1.0 representing how diverse the
/// result sources are. 1.0 = every result from a unique source.
/// 0.0 = all from one source (or empty).
pub fn diversity_score(results: &[SearchResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }

    let unique_sources: std::collections::HashSet<String> =
        results.iter().map(|r| extract_source(r)).collect();

    unique_sources.len() as f32 / results.len() as f32
}

/// Calculate file-level diversity score for a set of results.
///
/// Returns a value between 0.0 and 1.0. 1.0 = every result from a
/// unique file. 0.0 = all from one file (or empty).
pub fn file_diversity_score(results: &[SearchResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }

    let unique_files: std::collections::HashSet<String> =
        results.iter().map(|r| extract_file(r)).collect();

    unique_files.len() as f32 / results.len() as f32
}
