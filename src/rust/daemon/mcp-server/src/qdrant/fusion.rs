//! RRF fusion, score threshold, and source-diversity re-ranking.
//!
//! ## RRF formula
//!
//! Mirrors `applyRRFFusion` in `search-qdrant.ts` lines 164–194:
//!
//! ```text
//! // TypeScript (search-qdrant.ts:176)
//! semanticResults.forEach((result, rank) => {
//!     const rrfScore = 1 / (RRF_K + rank + 1);
//!     ...
//! });
//! ```
//!
//! Ranks are **0-based** (TS `forEach` index starts at 0).  The formula is:
//!
//! ```text
//! score_i = Σ  1 / (k + rank + 1)   for each result list that contains the item
//!               where rank is the 0-based position in that list, k = RRF_K = 60
//! ```
//!
//! So the first result (rank 0) contributes `1/(60+0+1) = 1/61`.
//!
//! ## Diversity re-ranking
//!
//! Mirrors `diversifyResults` in `search-diversity.ts`:
//!
//! 1. Walk results sorted by score descending (expected from caller).
//! 2. Group consecutive results within `score_tier_threshold` of each other
//!    into tiers (threshold measured from the **top** of the tier).
//! 3. Within each tier, round-robin across sources to interleave them.
//! 4. Enforce `max_per_source` globally — skip any result that would push a
//!    source beyond the cap.
//! 5. Backfill from overflow to preserve total count.
//! 6. Return re-ranked list and diversity score in [0, 1].
//!
//! Source key = `"collection:library_name"` for library results, or
//! `"collection:tenant_id"` for project results (falling back to
//! `"collection:unknown"`).

use std::collections::HashMap;

use serde_json::Value;

use super::client::QdrantPoint;

// ── Public constants ──────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion k constant — `60` is the standard value.
///
/// Matches `RRF_K = 60` in `search-types.ts`.
pub const RRF_K: usize = 60;

/// Default score threshold applied after fusion.
///
/// Matches `DEFAULT_SCORE_THRESHOLD = 0.3` in `search-types.ts`.
pub const DEFAULT_SCORE_THRESHOLD: f64 = 0.3;

/// Vector name for dense embeddings — matches `DENSE_VECTOR_NAME = 'dense'`.
pub const DENSE_VECTOR_NAME: &str = "dense";

/// Vector name for sparse embeddings — matches `SPARSE_VECTOR_NAME = 'sparse'`.
pub const SPARSE_VECTOR_NAME: &str = "sparse";

// ── Search-type tag ───────────────────────────────────────────────────────────

/// Discriminator carried in the result's metadata to identify which search
/// leg produced it (mirrors `_search_type` metadata field in TS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchType {
    Semantic,
    Keyword,
    Hybrid,
    Fallback,
}

impl SearchType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Keyword => "keyword",
            Self::Hybrid => "hybrid",
            Self::Fallback => "fallback",
        }
    }
}

/// A search result tagged with which leg produced it.
///
/// This is the internal type used by the fusion and diversity layers.
/// The `payload` map is the same as `QdrantPoint::payload`.
#[derive(Debug, Clone)]
pub struct TaggedResult {
    /// Point ID (UUID or numeric string).
    pub id: String,
    /// Score — raw from Qdrant for single-leg results; RRF sum after fusion.
    pub score: f64,
    /// Collection the result came from.
    pub collection: String,
    /// Decoded payload.
    pub payload: HashMap<String, Value>,
    /// Which search leg produced this result.
    pub search_type: SearchType,
}

impl TaggedResult {
    /// Derive the stable source key used by diversity re-ranking.
    ///
    /// Mirrors `extractSource` in `search-diversity.ts`:
    /// - Libraries: `collection:library_name`
    /// - Other:     `collection:tenant_id` (falls back to `collection:unknown`)
    pub fn source_key(&self) -> String {
        let collection = &self.collection;
        // TS `libraryName ? ... : tenantId ?? 'unknown'` — an empty-string (or
        // non-string) library_name is FALSY in JS and falls through to
        // tenant_id, so treat empty/non-string library_name as absent.
        if let Some(lib) = self
            .payload
            .get("library_name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            return format!("{collection}:{lib}");
        }
        let tenant = self
            .payload
            .get("tenant_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        format!("{collection}:{tenant}")
    }
}

// ── RRF fusion ────────────────────────────────────────────────────────────────

/// Apply Reciprocal Rank Fusion to a mixed list of semantic + keyword results.
///
/// **Input:** a flat slice of [`TaggedResult`] where each item has
/// `search_type` set to either `Semantic` or `Keyword`.  The slice may
/// contain items from multiple collections; the dedup key is
/// `"collection:id"` (mirrors TS `${result.collection}:${result.id}`).
///
/// **Preconditions (callers must ensure):**
/// - Input is already in descending relevance order within each leg (as
///   returned by Qdrant).
/// - Only called when both legs have at least one result; pass-through
///   happens when `mode != "hybrid"` or a leg is empty (see `applyRRFFusion`
///   in TS lines 164–168).
///
/// **Returns:** results with scores replaced by their RRF sums, tagged as
/// `SearchType::Hybrid`.  NOT sorted — caller sorts after filtering by
/// threshold.
///
/// Formula: `score = Σ 1 / (RRF_K + rank + 1)` where rank is 0-based.
pub fn apply_rrf_fusion(results: &[TaggedResult]) -> Vec<TaggedResult> {
    let semantic: Vec<&TaggedResult> = results
        .iter()
        .filter(|r| r.search_type == SearchType::Semantic)
        .collect();
    let keyword: Vec<&TaggedResult> = results
        .iter()
        .filter(|r| r.search_type == SearchType::Keyword)
        .collect();

    // Mirror TS line 170: if either leg is empty, return original slice as-is.
    if semantic.is_empty() || keyword.is_empty() {
        return results.to_vec();
    }

    // key → (rrf_score_accumulator, first-seen TaggedResult)
    let mut scores: HashMap<String, (f64, TaggedResult)> = HashMap::new();

    // Semantic leg — ranks are 0-based (forEach index)
    for (rank, result) in semantic.iter().enumerate() {
        let key = format!("{}:{}", result.collection, result.id);
        let rrf = 1.0 / (RRF_K as f64 + rank as f64 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| {
            let mut r = (*result).clone();
            r.score = 0.0;
            (0.0, r)
        });
        entry.0 += rrf;
    }

    // Keyword leg — ranks are 0-based
    for (rank, result) in keyword.iter().enumerate() {
        let key = format!("{}:{}", result.collection, result.id);
        let rrf = 1.0 / (RRF_K as f64 + rank as f64 + 1.0);
        let entry = scores.entry(key).or_insert_with(|| {
            let mut r = (*result).clone();
            r.score = 0.0;
            (0.0, r)
        });
        entry.0 += rrf;
    }

    scores
        .into_values()
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result.search_type = SearchType::Hybrid;
            result
        })
        .collect()
}

/// Keep only results whose score is >= `threshold`.
///
/// Mirrors TS default threshold usage (`DEFAULT_SCORE_THRESHOLD = 0.3`).
/// Call after sorting by score descending.
pub fn apply_score_threshold(results: Vec<TaggedResult>, threshold: f64) -> Vec<TaggedResult> {
    results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect()
}

// ── Diversity re-ranking ──────────────────────────────────────────────────────

/// Configuration for source-diversity re-ranking.
///
/// Mirrors `DiversityConfig` in `search-diversity.ts`.
#[derive(Debug, Clone)]
pub struct DiversityConfig {
    /// When `false`, diversity re-ranking is skipped and results are returned
    /// unchanged.
    pub enabled: bool,
    /// Maximum results from one source in the final output.
    pub max_per_source: usize,
    /// Score delta within which results are grouped into the same tier.
    pub score_tier_threshold: f64,
}

/// Default diversity configuration.
///
/// Matches `DEFAULT_DIVERSITY_CONFIG` in `search-diversity.ts`:
/// `enabled: true, maxPerSource: 3, scoreTierThreshold: 0.05`.
pub const DEFAULT_DIVERSITY_CONFIG: DiversityConfig = DiversityConfig {
    enabled: true,
    max_per_source: 3,
    score_tier_threshold: 0.05,
};

/// Compute the diversity score for a result list.
///
/// Returns `unique_sources / total_results` in [0, 1].  An empty list
/// returns 1.0 (no diversity concern).
///
/// Mirrors `computeDiversityScore` in `search-diversity.ts`.
pub fn compute_diversity_score(results: &[TaggedResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let unique: std::collections::HashSet<String> =
        results.iter().map(|r| r.source_key()).collect();
    unique.len() as f64 / results.len() as f64
}

/// Apply source diversity re-ranking.
///
/// **Input must already be sorted by score descending** (as produced by
/// `apply_rrf_fusion` + sort, or a single-leg result list).
///
/// Returns the re-ranked list and the computed diversity score in [0, 1].
///
/// Mirrors `diversifyResults` in `search-diversity.ts`.
pub fn diversify_results(
    results: Vec<TaggedResult>,
    config: &DiversityConfig,
) -> (Vec<TaggedResult>, f64) {
    if !config.enabled || results.is_empty() {
        let score = compute_diversity_score(&results);
        return (results, score);
    }

    let tiers = build_score_tiers(&results, config.score_tier_threshold);
    let mut source_counts: HashMap<String, usize> = HashMap::new();
    let mut output: Vec<TaggedResult> = Vec::with_capacity(results.len());
    let mut spillover: Vec<TaggedResult> = Vec::new();
    let target_count = results.len();

    for tier in tiers {
        let interleaved = interleave_tier(tier);
        for r in interleaved {
            let source = r.source_key();
            let count = *source_counts.get(&source).unwrap_or(&0);
            if count < config.max_per_source {
                *source_counts.entry(source).or_insert(0) += 1;
                output.push(r);
            } else {
                spillover.push(r);
            }
        }
    }

    // Backfill from spillover to preserve the requested result count.
    for r in spillover {
        if output.len() >= target_count {
            break;
        }
        output.push(r);
    }

    let diversity_score = compute_diversity_score(&output);
    (output, diversity_score)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Group sorted results into tiers by score proximity.
///
/// Threshold is measured from the **top** of the current tier (matching TS
/// `buildScoreTiers` where `tierTopScore` is set once per new tier).
fn build_score_tiers(results: &[TaggedResult], threshold: f64) -> Vec<Vec<TaggedResult>> {
    if results.is_empty() {
        return vec![];
    }

    let mut tiers: Vec<Vec<TaggedResult>> = Vec::new();
    let mut current_tier: Vec<TaggedResult> = vec![results[0].clone()];
    let mut tier_top_score = results[0].score;

    for r in &results[1..] {
        if (tier_top_score - r.score).abs() <= threshold {
            current_tier.push(r.clone());
        } else {
            tiers.push(current_tier);
            current_tier = vec![r.clone()];
            tier_top_score = r.score;
        }
    }
    if !current_tier.is_empty() {
        tiers.push(current_tier);
    }
    tiers
}

/// Round-robin interleave a single tier by source.
///
/// Mirrors `interleaveTier` in `search-diversity.ts`.
fn interleave_tier(tier: Vec<TaggedResult>) -> Vec<TaggedResult> {
    if tier.len() <= 1 {
        return tier;
    }

    let mut source_order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<TaggedResult>> = HashMap::new();

    for r in tier {
        let source = r.source_key();
        if !groups.contains_key(&source) {
            source_order.push(source.clone());
            groups.insert(source.clone(), Vec::new());
        }
        groups.get_mut(&source).unwrap().push(r);
    }

    let mut indices = vec![0usize; source_order.len()];
    let mut exhausted = 0usize;
    let mut output: Vec<TaggedResult> = Vec::new();

    while exhausted < source_order.len() {
        for i in 0..source_order.len() {
            let source = &source_order[i];
            let group = groups.get(source).unwrap();
            let idx = indices[i];
            if idx < group.len() {
                output.push(group[idx].clone());
                indices[i] += 1;
                if indices[i] == group.len() {
                    exhausted += 1;
                }
            }
        }
    }

    output
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a [`QdrantPoint`] to a [`TaggedResult`] with the given collection
/// and search type.
pub fn point_to_tagged(
    point: QdrantPoint,
    collection: String,
    search_type: SearchType,
) -> TaggedResult {
    TaggedResult {
        id: point.id,
        score: point.score,
        collection,
        payload: point.payload,
        search_type,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "fusion_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "fusion_diversity_tests.rs"]
mod diversity_tests;
