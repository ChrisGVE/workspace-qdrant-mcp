//! Search-quality metric math for the known-item eval (#135).
//!
//! Located at: `src/rust/cli/src/commands/benchmark/quality/metrics.rs`
//!
//! Pure functions over a list of result paths and a query's expected files.
//! No daemon, no Qdrant, no SQLite — so the metric math is unit-tested directly
//! on synthetic path lists. The live runner (`mod.rs`) extracts the ranked path
//! list from a real `SearchResponse` and feeds it here.
//!
//! ## Definitions (parity with the deleted TS harness `semantic-search.ts`)
//!
//! For one query, given the ranked result paths (top-k, duplicates kept) and the
//! deduplicated expected files:
//! - **topN hit** — at least one expected file appears in the first N *raw*
//!   (duplicate-bearing) result paths. Hits are measured on raw ranks because a
//!   user reads results in returned order, duplicates included.
//! - **firstRelevantRank** — 1-based rank of the first raw path that matches any
//!   expectation; `None` if none match.
//! - **recall@10** — fraction of *distinct* expected files matched anywhere in
//!   the deduplicated top-k.
//! - **precision@10** — fraction of the *deduplicated* top-k paths that match an
//!   expectation.
//! - **duplicateRate** — `1 - deduped_len / raw_len`: how much of the ranked list
//!   was repeat paths.
//! - **MRR** — `1 / firstRelevantRank` (0 when no hit).
//!
//! A mode summary averages each per-query metric across queries. The verdict
//! gates on two independent signals (top-3 useful rate and recall@10); see
//! [`classify`].
//!
//! Neighbors: `path_match.rs` (expectation matching), `report.rs` (shapes these
//! into the printed/JSON output), `dataset.rs` (supplies expected files).

use super::path_match::{normalize_path, ExpectedMatcher};

/// The verdict gates, with the rationale baked into the names.
///
/// Both gates are *independent* known-item signals — recall@10 asks "did we
/// surface the relevant file at all", top-3 useful rate asks "was it ranked high
/// enough to be seen". precision@10 is intentionally NOT a gate: with only 1–2
/// relevant files per query it is ≈ recall@10 × (meanExpected/10), i.e. a
/// rescaled copy of recall, so gating on both would double-count one signal. It
/// stays in the reported metrics for visibility. Values mirror the TS harness'
/// `DEFAULT_SEMANTIC_QUALITY_THRESHOLDS`.
#[derive(Debug, Clone, Copy)]
pub struct QualityThresholds {
    /// Minimum acceptable top-3 hit rate (a query is "useful" if a relevant file
    /// is in the first three results).
    pub top3_useful_rate: f64,
    /// Minimum acceptable recall@10.
    pub recall_at10: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            top3_useful_rate: 0.8,
            recall_at10: 0.7,
        }
    }
}

/// Per-query evaluation against the expected files.
#[derive(Debug, Clone)]
pub struct QueryEvaluation {
    /// Expected files that were matched somewhere in the deduplicated top-k.
    pub matched_expected: Vec<String>,
    pub top1_hit: bool,
    pub top3_hit: bool,
    pub top10_hit: bool,
    /// 1-based rank of the first relevant raw path, `None` if no hit.
    pub first_relevant_rank: Option<usize>,
    pub precision_at10: f64,
    pub recall_at10: f64,
    pub duplicate_rate: f64,
    pub mrr: f64,
}

/// Aggregate metrics for one ranked mode over all queries.
#[derive(Debug, Clone, Default)]
pub struct ModeSummary {
    pub runs: usize,
    pub top1_hit_rate: f64,
    pub top3_hit_rate: f64,
    pub top10_hit_rate: f64,
    pub precision_at10: f64,
    pub recall_at10: f64,
    pub mrr: f64,
    pub duplicate_rate: f64,
    pub avg_latency_ms: f64,
}

/// Quality grade for a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    Good,
    Mixed,
    Poor,
}

impl Grade {
    pub fn as_str(self) -> &'static str {
        match self {
            Grade::Good => "good",
            Grade::Mixed => "mixed",
            Grade::Poor => "poor",
        }
    }
}

/// A graded verdict with human-readable reasons for any failed gate.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub grade: Grade,
    pub reasons: Vec<String>,
    pub thresholds: QualityThresholds,
}

/// Evaluate one query's ranked result paths against its expected files.
///
/// `raw_ranked_paths` is the live search's top-k result paths in rank order,
/// duplicates kept; `expected` is the gold list (literal paths or globs).
/// `top_k` bounds how many ranked paths are considered. `workspace_root` is the
/// prefix stripped from result paths (empty when paths are already repo-relative).
pub fn evaluate_query(
    raw_ranked_paths: &[String],
    expected: &[String],
    workspace_root: &str,
    top_k: usize,
) -> QueryEvaluation {
    let expected_files = normalize_and_dedupe(expected, workspace_root);
    let matchers: Vec<ExpectedMatcher> = expected_files
        .iter()
        .map(|e| ExpectedMatcher::new(e))
        .collect();

    // Raw paths: normalized, top-k, duplicates kept, empties dropped.
    let raw_paths: Vec<String> = raw_ranked_paths
        .iter()
        .take(top_k)
        .map(|p| normalize_path(p, workspace_root))
        .filter(|p| !p.is_empty())
        .collect();

    // Deduped paths: first occurrence of each path, rank order preserved.
    let deduped_paths = dedupe_preserving_order(&raw_paths);

    let is_relevant = |path: &str| matchers.iter().any(|m| m.matches(path));

    let first_relevant_rank = raw_paths
        .iter()
        .position(|p| is_relevant(p))
        .map(|idx| idx + 1);

    let top1_hit = raw_paths.first().map(|p| is_relevant(p)).unwrap_or(false);
    let top3_hit = raw_paths.iter().take(3).any(|p| is_relevant(p));
    let top10_hit = raw_paths.iter().any(|p| is_relevant(p));

    // matched_expected / recall: over distinct expectations against deduped paths.
    let matched_expected: Vec<String> = expected_files
        .iter()
        .zip(matchers.iter())
        .filter(|(_, matcher)| deduped_paths.iter().any(|p| matcher.matches(p)))
        .map(|(expected, _)| expected.clone())
        .collect();

    let relevant_unique = deduped_paths.iter().filter(|p| is_relevant(p)).count();
    let precision_at10 = ratio(relevant_unique, deduped_paths.len());
    let recall_at10 = ratio(matched_expected.len(), expected_files.len());
    let duplicate_rate = if raw_paths.is_empty() {
        0.0
    } else {
        1.0 - (deduped_paths.len() as f64 / raw_paths.len() as f64)
    };
    let mrr = first_relevant_rank.map(|r| 1.0 / r as f64).unwrap_or(0.0);

    QueryEvaluation {
        matched_expected,
        top1_hit,
        top3_hit,
        top10_hit,
        first_relevant_rank,
        precision_at10,
        recall_at10,
        duplicate_rate,
        mrr,
    }
}

/// Summarize a mode by averaging each per-query metric across queries.
///
/// `latencies_ms` is paired per query (one latency per evaluation). An empty set
/// yields an all-zero summary rather than panicking — a degraded run still
/// reports.
pub fn summarize_mode(evaluations: &[QueryEvaluation], latencies_ms: &[f64]) -> ModeSummary {
    let n = evaluations.len();
    if n == 0 {
        return ModeSummary::default();
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let bool_rate = |f: fn(&QueryEvaluation) -> bool| {
        evaluations.iter().filter(|e| f(e)).count() as f64 / n as f64
    };

    ModeSummary {
        runs: n,
        top1_hit_rate: bool_rate(|e| e.top1_hit),
        top3_hit_rate: bool_rate(|e| e.top3_hit),
        top10_hit_rate: bool_rate(|e| e.top10_hit),
        precision_at10: mean(&collect(evaluations, |e| e.precision_at10)),
        recall_at10: mean(&collect(evaluations, |e| e.recall_at10)),
        mrr: mean(&collect(evaluations, |e| e.mrr)),
        duplicate_rate: mean(&collect(evaluations, |e| e.duplicate_rate)),
        avg_latency_ms: if latencies_ms.is_empty() {
            0.0
        } else {
            mean(latencies_ms)
        },
    }
}

/// Grade a mode summary against the quality gates.
///
/// Zero failed gates → `good`, one → `mixed`, both → `poor`. Each failed gate
/// contributes a reason naming the metric, its value, and the bar it missed.
pub fn classify(summary: &ModeSummary, thresholds: QualityThresholds) -> Verdict {
    let mut reasons = Vec::new();
    if summary.top3_hit_rate < thresholds.top3_useful_rate {
        reasons.push(format!(
            "top-3 useful rate {} is below {}",
            pct(summary.top3_hit_rate),
            pct(thresholds.top3_useful_rate)
        ));
    }
    if summary.recall_at10 < thresholds.recall_at10 {
        reasons.push(format!(
            "recall@10 {} is below {}",
            pct(summary.recall_at10),
            pct(thresholds.recall_at10)
        ));
    }
    let grade = match reasons.len() {
        0 => Grade::Good,
        1 => Grade::Mixed,
        _ => Grade::Poor,
    };
    Verdict {
        grade,
        reasons,
        thresholds,
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Normalize each expected file and drop duplicates, preserving order.
fn normalize_and_dedupe(expected: &[String], workspace_root: &str) -> Vec<String> {
    let normalized: Vec<String> = expected
        .iter()
        .map(|e| normalize_path(e, workspace_root))
        .filter(|e| !e.is_empty())
        .collect();
    dedupe_preserving_order(&normalized)
}

/// Keep the first occurrence of each value, preserving order.
fn dedupe_preserving_order(values: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values
        .iter()
        .filter(|v| seen.insert((*v).clone()))
        .cloned()
        .collect()
}

/// Safe ratio: 0 when the denominator is 0.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Collect one f64 field across evaluations.
fn collect(evaluations: &[QueryEvaluation], f: fn(&QueryEvaluation) -> f64) -> Vec<f64> {
    evaluations.iter().map(f).collect()
}

/// Format a 0–1 rate as a one-decimal percentage string.
pub fn pct(rate: f64) -> String {
    format!("{:.1}%", rate * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn top1_hit_when_expected_is_first() {
        let ev = evaluate_query(&paths(&["a.rs", "b.rs"]), &paths(&["a.rs"]), "", 10);
        assert!(ev.top1_hit);
        assert!(ev.top3_hit);
        assert!(ev.top10_hit);
        assert_eq!(ev.first_relevant_rank, Some(1));
        assert_eq!(ev.mrr, 1.0);
    }

    #[test]
    fn top3_but_not_top1_when_expected_at_rank_3() {
        let ev = evaluate_query(
            &paths(&["x.rs", "y.rs", "a.rs", "z.rs"]),
            &paths(&["a.rs"]),
            "",
            10,
        );
        assert!(!ev.top1_hit);
        assert!(ev.top3_hit);
        assert_eq!(ev.first_relevant_rank, Some(3));
        assert!((ev.mrr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn no_hit_yields_zero_mrr_and_none_rank() {
        let ev = evaluate_query(&paths(&["x.rs", "y.rs"]), &paths(&["a.rs"]), "", 10);
        assert!(!ev.top10_hit);
        assert_eq!(ev.first_relevant_rank, None);
        assert_eq!(ev.mrr, 0.0);
        assert_eq!(ev.recall_at10, 0.0);
    }

    #[test]
    fn recall_counts_distinct_expected_matched() {
        // Two expected; only one present → recall 0.5.
        let ev = evaluate_query(&paths(&["a.rs", "z.rs"]), &paths(&["a.rs", "b.rs"]), "", 10);
        assert_eq!(ev.matched_expected, vec!["a.rs".to_string()]);
        assert!((ev.recall_at10 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn precision_is_over_deduped_paths() {
        // 4 deduped paths, 1 relevant → precision 0.25.
        let ev = evaluate_query(
            &paths(&["a.rs", "x.rs", "y.rs", "z.rs"]),
            &paths(&["a.rs"]),
            "",
            10,
        );
        assert!((ev.precision_at10 - 0.25).abs() < 1e-9);
    }

    #[test]
    fn duplicate_rate_reflects_repeated_paths() {
        // 4 raw, 2 distinct → duplicate rate 0.5.
        let ev = evaluate_query(
            &paths(&["a.rs", "a.rs", "b.rs", "b.rs"]),
            &paths(&["a.rs"]),
            "",
            10,
        );
        // 4 raw paths, 2 distinct → half the list was duplicates.
        assert!((ev.duplicate_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hits_use_raw_ranks_not_deduped() {
        // Duplicate of an irrelevant path pushes the expected file to raw rank 3,
        // but only raw rank 2 after the dedupe — top1/top3 must use raw ranks.
        let ev = evaluate_query(&paths(&["x.rs", "x.rs", "a.rs"]), &paths(&["a.rs"]), "", 10);
        assert_eq!(ev.first_relevant_rank, Some(3));
        assert!(!ev.top1_hit);
        assert!(ev.top3_hit);
    }

    #[test]
    fn glob_expectation_matches() {
        let ev = evaluate_query(
            &paths(&["src/rust/daemon/proto/workspace_daemon.proto"]),
            &paths(&["**/proto/*.proto"]),
            "",
            10,
        );
        assert!(ev.top1_hit);
        assert!((ev.recall_at10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn top_k_bounds_considered_paths() {
        // Expected at rank 11; top_k=10 must not see it.
        let mut ranked: Vec<String> = (0..10).map(|i| format!("f{i}.rs")).collect();
        ranked.push("a.rs".to_string());
        let ev = evaluate_query(&ranked, &paths(&["a.rs"]), "", 10);
        assert!(!ev.top10_hit);
        assert_eq!(ev.first_relevant_rank, None);
    }

    #[test]
    fn workspace_root_prefix_is_stripped_before_matching() {
        let ev = evaluate_query(
            &paths(&["/repo/src/a.rs"]),
            &paths(&["src/a.rs"]),
            "/repo",
            10,
        );
        assert!(ev.top1_hit);
    }

    fn make_eval(top1: bool, top3: bool, top10: bool, recall: f64) -> QueryEvaluation {
        QueryEvaluation {
            matched_expected: if top10 {
                vec!["a.rs".to_string()]
            } else {
                vec![]
            },
            top1_hit: top1,
            top3_hit: top3,
            top10_hit: top10,
            first_relevant_rank: if top1 { Some(1) } else { None },
            precision_at10: 0.0,
            recall_at10: recall,
            duplicate_rate: 0.0,
            mrr: if top1 { 1.0 } else { 0.0 },
        }
    }

    #[test]
    fn summarize_averages_rates() {
        let evals = vec![
            make_eval(true, true, true, 1.0),
            make_eval(false, true, true, 1.0),
            make_eval(false, false, false, 0.0),
        ];
        let s = summarize_mode(&evals, &[10.0, 20.0, 30.0]);
        assert_eq!(s.runs, 3);
        assert!((s.top1_hit_rate - 1.0 / 3.0).abs() < 1e-9);
        assert!((s.top3_hit_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.top10_hit_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.recall_at10 - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.avg_latency_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_empty_is_all_zero() {
        let s = summarize_mode(&[], &[]);
        assert_eq!(s.runs, 0);
        assert_eq!(s.top3_hit_rate, 0.0);
    }

    #[test]
    fn verdict_good_when_both_gates_pass() {
        let s = ModeSummary {
            top3_hit_rate: 0.9,
            recall_at10: 0.8,
            ..Default::default()
        };
        let v = classify(&s, QualityThresholds::default());
        assert_eq!(v.grade, Grade::Good);
        assert!(v.reasons.is_empty());
    }

    #[test]
    fn verdict_mixed_when_one_gate_fails() {
        let s = ModeSummary {
            top3_hit_rate: 0.5,
            recall_at10: 0.8,
            ..Default::default()
        };
        let v = classify(&s, QualityThresholds::default());
        assert_eq!(v.grade, Grade::Mixed);
        assert_eq!(v.reasons.len(), 1);
        assert!(v.reasons[0].contains("top-3"));
    }

    #[test]
    fn verdict_poor_when_both_gates_fail() {
        let s = ModeSummary {
            top3_hit_rate: 0.5,
            recall_at10: 0.4,
            ..Default::default()
        };
        let v = classify(&s, QualityThresholds::default());
        assert_eq!(v.grade, Grade::Poor);
        assert_eq!(v.reasons.len(), 2);
    }

    #[test]
    fn pct_formats_one_decimal() {
        assert_eq!(pct(0.834), "83.4%");
        assert_eq!(pct(1.0), "100.0%");
    }
}
