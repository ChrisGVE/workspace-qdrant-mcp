/// Source diversity re-ranking for search results.
///
/// After retrieving candidates from Qdrant, groups results by source
/// (tenant_id or library_name) and re-ranks to ensure diverse representation
/// across sources within score tiers.
use serde::{Deserialize, Serialize};

use crate::storage::SearchResult;

// ─── Configuration ─────────────────────────────────────────────────────

/// Diversity re-ranking configuration.
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

// ─── Source extraction ─────────────────────────────────────────────────

/// Extract the source identifier from a search result's payload.
///
/// Checks `library_name` first (for library results), then `tenant_id`
/// (for project results). Falls back to "unknown" if neither is present.
fn extract_source(result: &SearchResult) -> String {
    result
        .payload
        .get("library_name")
        .and_then(|v| v.as_str())
        .or_else(|| result.payload.get("tenant_id").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string()
}

// ─── Diversity re-ranking ──────────────────────────────────────────────

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

    // Split into score tiers
    let tiers = build_score_tiers(&results, config.score_tier_threshold);

    // Re-rank within each tier for diversity, then flatten
    let mut output = Vec::with_capacity(results.len());
    let mut source_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

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
fn build_score_tiers(results: &[SearchResult], threshold: f32) -> Vec<Vec<SearchResult>> {
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

    // Group by source, preserving order within each source
    let mut source_groups: std::collections::HashMap<String, Vec<SearchResult>> =
        std::collections::HashMap::new();
    let mut source_order: Vec<String> = Vec::new();

    for result in tier.drain(..) {
        let source = extract_source(&result);
        if !source_groups.contains_key(&source) {
            source_order.push(source.clone());
        }
        source_groups.entry(source).or_default().push(result);
    }

    // Round-robin interleave
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_result(id: &str, score: f32, source_field: &str, source_val: &str) -> SearchResult {
        let mut payload = HashMap::new();
        payload.insert(source_field.to_string(), serde_json::json!(source_val));

        SearchResult {
            id: id.to_string(),
            score,
            payload,
            dense_vector: None,
            sparse_vector: None,
        }
    }

    fn make_project_result(id: &str, score: f32, tenant: &str) -> SearchResult {
        make_result(id, score, "tenant_id", tenant)
    }

    fn make_library_result(id: &str, score: f32, library: &str) -> SearchResult {
        make_result(id, score, "library_name", library)
    }

    // ─── Extract source tests ───────────────────────────────────────────

    #[test]
    fn test_extract_source_library() {
        let r = make_library_result("r1", 0.9, "my-lib");
        assert_eq!(extract_source(&r), "my-lib");
    }

    #[test]
    fn test_extract_source_project() {
        let r = make_project_result("r1", 0.9, "proj-a");
        assert_eq!(extract_source(&r), "proj-a");
    }

    #[test]
    fn test_extract_source_unknown() {
        let r = SearchResult {
            id: "r1".to_string(),
            score: 0.9,
            payload: HashMap::new(),
            dense_vector: None,
            sparse_vector: None,
        };
        assert_eq!(extract_source(&r), "unknown");
    }

    // ─── Score tier tests ───────────────────────────────────────────────

    #[test]
    fn test_build_tiers_single_tier() {
        let results = vec![
            make_project_result("r1", 0.90, "a"),
            make_project_result("r2", 0.88, "b"),
            make_project_result("r3", 0.87, "c"),
        ];
        let tiers = build_score_tiers(&results, 0.05);
        assert_eq!(tiers.len(), 1, "All within 0.05 of 0.90");
    }

    #[test]
    fn test_build_tiers_multiple() {
        let results = vec![
            make_project_result("r1", 0.95, "a"),
            make_project_result("r2", 0.93, "b"),
            // Gap > 0.05
            make_project_result("r3", 0.80, "c"),
            make_project_result("r4", 0.78, "d"),
        ];
        let tiers = build_score_tiers(&results, 0.05);
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].len(), 2);
        assert_eq!(tiers[1].len(), 2);
    }

    #[test]
    fn test_build_tiers_empty() {
        let tiers = build_score_tiers(&[], 0.05);
        assert!(tiers.is_empty());
    }

    // ─── Interleave tests ───────────────────────────────────────────────

    #[test]
    fn test_interleave_single_source() {
        let tier = vec![
            make_project_result("r1", 0.9, "a"),
            make_project_result("r2", 0.89, "a"),
        ];
        let result = interleave_tier(tier);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "r1");
        assert_eq!(result[1].id, "r2");
    }

    #[test]
    fn test_interleave_multiple_sources() {
        let tier = vec![
            make_project_result("a1", 0.9, "a"),
            make_project_result("a2", 0.9, "a"),
            make_project_result("b1", 0.9, "b"),
            make_project_result("b2", 0.9, "b"),
        ];
        let result = interleave_tier(tier);
        // Should alternate: a1, b1, a2, b2
        assert_eq!(result[0].id, "a1");
        assert_eq!(result[1].id, "b1");
        assert_eq!(result[2].id, "a2");
        assert_eq!(result[3].id, "b2");
    }

    // ─── Diversity re-ranking tests ─────────────────────────────────────

    #[test]
    fn test_diversify_disabled() {
        let config = DiversityConfig {
            enabled: false,
            ..Default::default()
        };
        let results = vec![
            make_project_result("r1", 0.9, "a"),
            make_project_result("r2", 0.8, "a"),
        ];
        let diversified = diversify_results(results.clone(), &config);
        assert_eq!(diversified.len(), results.len());
    }

    #[test]
    fn test_diversify_caps_per_source() {
        let config = DiversityConfig {
            enabled: true,
            max_per_source: 2,
            score_tier_threshold: 0.05,
        };
        let results = vec![
            make_project_result("a1", 0.95, "a"),
            make_project_result("a2", 0.93, "a"),
            make_project_result("a3", 0.91, "a"),
            make_project_result("a4", 0.89, "a"),
            make_project_result("b1", 0.87, "b"),
        ];
        let diversified = diversify_results(results, &config);

        // Source "a" capped at 2, "b" has 1 → total 3
        let a_count = diversified
            .iter()
            .filter(|r| extract_source(r) == "a")
            .count();
        assert_eq!(a_count, 2, "Source 'a' should be capped at 2");
        assert!(diversified.iter().any(|r| r.id == "b1"));
    }

    #[test]
    fn test_diversify_preserves_cross_tier_order() {
        let config = DiversityConfig {
            enabled: true,
            max_per_source: 5,
            score_tier_threshold: 0.05,
        };
        let results = vec![
            make_project_result("high1", 0.95, "a"),
            make_project_result("high2", 0.93, "b"),
            make_project_result("low1", 0.70, "a"),
            make_project_result("low2", 0.68, "b"),
        ];
        let diversified = diversify_results(results, &config);

        // High-tier results should come before low-tier
        let high1_pos = diversified.iter().position(|r| r.id == "high1").unwrap();
        let low1_pos = diversified.iter().position(|r| r.id == "low1").unwrap();
        assert!(high1_pos < low1_pos, "High-tier before low-tier");
    }

    #[test]
    fn test_diversify_same_tier_interleaves() {
        let config = DiversityConfig {
            enabled: true,
            max_per_source: 5,
            score_tier_threshold: 0.10,
        };
        let results = vec![
            make_project_result("a1", 0.95, "a"),
            make_project_result("a2", 0.94, "a"),
            make_project_result("b1", 0.93, "b"),
            make_project_result("b2", 0.92, "b"),
        ];
        let diversified = diversify_results(results, &config);

        // Within the same tier, sources should interleave
        assert_eq!(diversified[0].id, "a1");
        assert_eq!(diversified[1].id, "b1");
    }

    #[test]
    fn test_diversify_empty() {
        let config = DiversityConfig::default();
        let diversified = diversify_results(Vec::new(), &config);
        assert!(diversified.is_empty());
    }

    #[test]
    fn test_diversify_single_result() {
        let config = DiversityConfig::default();
        let results = vec![make_project_result("r1", 0.9, "a")];
        let diversified = diversify_results(results, &config);
        assert_eq!(diversified.len(), 1);
    }

    // ─── Diversity score tests ──────────────────────────────────────────

    #[test]
    fn test_diversity_score_all_unique() {
        let results = vec![
            make_project_result("r1", 0.9, "a"),
            make_project_result("r2", 0.8, "b"),
            make_project_result("r3", 0.7, "c"),
        ];
        let score = diversity_score(&results);
        assert!((score - 1.0).abs() < 1e-6, "All unique → 1.0");
    }

    #[test]
    fn test_diversity_score_all_same() {
        let results = vec![
            make_project_result("r1", 0.9, "a"),
            make_project_result("r2", 0.8, "a"),
            make_project_result("r3", 0.7, "a"),
        ];
        let score = diversity_score(&results);
        assert!(
            (score - 1.0 / 3.0).abs() < 1e-6,
            "1 unique / 3 total = 0.333"
        );
    }

    #[test]
    fn test_diversity_score_empty() {
        assert_eq!(diversity_score(&[]), 0.0);
    }

    #[test]
    fn test_diversity_score_mixed() {
        let results = vec![
            make_project_result("r1", 0.9, "a"),
            make_project_result("r2", 0.8, "b"),
            make_library_result("r3", 0.7, "lib-1"),
            make_project_result("r4", 0.6, "a"),
        ];
        let score = diversity_score(&results);
        // 3 unique sources / 4 results = 0.75
        assert!((score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_diversity_library_name_takes_precedence() {
        // A result with both library_name and tenant_id → library_name wins
        let mut payload = HashMap::new();
        payload.insert("library_name".to_string(), serde_json::json!("my-lib"));
        payload.insert("tenant_id".to_string(), serde_json::json!("proj-a"));

        let r = SearchResult {
            id: "r1".to_string(),
            score: 0.9,
            payload,
            dense_vector: None,
            sparse_vector: None,
        };
        assert_eq!(extract_source(&r), "my-lib");
    }
}
