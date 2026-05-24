/// Penalty-based diversity re-ranking.
///
/// Penalizes consecutive results from the same file or project by
/// compounding multipliers, then re-sorts by adjusted score.
use serde::{Deserialize, Serialize};

use super::{extract_file, extract_project};
use crate::storage::SearchResult;

/// Penalty-based diversity configuration.
///
/// When consecutive results share the same file or project, their scores
/// are multiplied by the corresponding penalty factor. Penalties compound
/// across consecutive runs: the Nth consecutive same-file result receives
/// `same_file_penalty.powi(N - 1)` as its multiplier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityPenaltyConfig {
    /// Whether penalty-based re-ranking is enabled.
    pub enabled: bool,
    /// Multiplier applied to a result when the previous result shares the
    /// same `file_path` or `relative_path`. Default: 0.85.
    pub same_file_penalty: f32,
    /// Multiplier applied to a result when the previous result shares the
    /// same project (`tenant_id`) but a different file. Default: 0.92.
    pub same_project_penalty: f32,
}

impl Default for DiversityPenaltyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            same_file_penalty: 0.85,
            same_project_penalty: 0.92,
        }
    }
}

/// Apply diversity penalties to consecutive same-source results, then re-sort.
///
/// Algorithm:
/// 1. Walk the score-sorted result list.
/// 2. For each result, compare its file and project with the previous result.
/// 3. If the file matches, apply `config.same_file_penalty` (compounding).
/// 4. Else if only the project matches, apply `config.same_project_penalty`
///    (compounding).
/// 5. Re-sort by adjusted scores descending.
///
/// The compounding ensures that long runs from the same file are
/// increasingly penalized: the 3rd consecutive same-file result
/// receives `penalty^2`, the 4th `penalty^3`, etc.
pub fn apply_diversity_penalty(
    mut results: Vec<SearchResult>,
    config: &DiversityPenaltyConfig,
) -> Vec<SearchResult> {
    if !config.enabled || results.len() <= 1 {
        return results;
    }

    // First pass: compute penalty-adjusted scores.
    // We track consecutive run lengths for file and project separately.
    let mut adjusted_scores: Vec<f32> = Vec::with_capacity(results.len());
    let mut prev_file = String::new();
    let mut prev_project = String::new();
    let mut file_run: u32 = 0;
    let mut project_run: u32 = 0;

    for result in &results {
        let file = extract_file(result);
        let project = extract_project(result);

        if file == prev_file {
            file_run += 1;
            // Same file implies same project; use the stronger file penalty.
            let multiplier = config.same_file_penalty.powi(file_run as i32);
            adjusted_scores.push(result.score * multiplier);
        } else if project == prev_project {
            // Different file, same project.
            file_run = 0;
            project_run += 1;
            let multiplier = config.same_project_penalty.powi(project_run as i32);
            adjusted_scores.push(result.score * multiplier);
        } else {
            // Completely different source — no penalty.
            file_run = 0;
            project_run = 0;
            adjusted_scores.push(result.score);
        }

        prev_file = file;
        prev_project = project;
    }

    // Apply adjusted scores to results.
    for (result, &adj_score) in results.iter_mut().zip(adjusted_scores.iter()) {
        result.score = adj_score;
    }

    // Re-sort by adjusted score descending.
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}
