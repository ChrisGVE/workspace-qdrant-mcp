/// Source diversity re-ranking for search results.
///
/// Two complementary strategies:
///
/// 1. **Tier-based interleaving** (`diversify_results`): groups results into
///    score tiers and interleaves sources with a per-source cap.
///
/// 2. **Penalty-based re-ranking** (`apply_diversity_penalty`): penalizes
///    consecutive results from the same file or project, then re-sorts.
///    This prevents result clustering from a single source without hard caps.
mod penalty;
mod tier;

#[cfg(test)]
mod tests;

// Re-export all public API types and functions.
pub use penalty::{apply_diversity_penalty, DiversityPenaltyConfig};
pub use tier::{
    build_score_tiers, diversify_results, diversity_score, file_diversity_score, DiversityConfig,
};

use crate::storage::SearchResult;

// ─── Shared extraction helpers ────────────────────────────────────────

/// Extract the source identifier from a search result's payload.
///
/// Checks `library_name` first (for library results), then `tenant_id`
/// (for project results). Falls back to "unknown" if neither is present.
pub(crate) fn extract_source(result: &SearchResult) -> String {
    result
        .payload
        .get("library_name")
        .and_then(|v| v.as_str())
        .or_else(|| result.payload.get("tenant_id").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string()
}

/// Extract the project identifier from a search result's payload.
///
/// Checks `tenant_id` first, then `library_name`. Falls back to
/// "unknown" when neither is present.
pub(crate) fn extract_project(result: &SearchResult) -> String {
    result
        .payload
        .get("tenant_id")
        .and_then(|v| v.as_str())
        .or_else(|| result.payload.get("library_name").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string()
}

/// Extract the file identifier from a search result's payload.
///
/// Checks `file_path` first, then `relative_path`. Falls back to
/// the result's `id` so that each result is at least self-distinguishable.
pub(crate) fn extract_file(result: &SearchResult) -> String {
    result
        .payload
        .get("file_path")
        .and_then(|v| v.as_str())
        .or_else(|| result.payload.get("relative_path").and_then(|v| v.as_str()))
        .unwrap_or(&result.id)
        .to_string()
}
