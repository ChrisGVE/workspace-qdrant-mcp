//! Report shaping and rendering for the quality eval (#135).
//!
//! Located at: `src/rust/cli/src/commands/benchmark/quality/report.rs`
//!
//! Turns the per-query evaluations and mode summaries into (a) a human-readable
//! terminal report and (b) a serializable JSON report. The shapes mirror the
//! deleted TS `search_eval` tool output (modes / byCategory / perQuery / verdict)
//! so any tooling built against that format still reads. Percentages are rounded
//! to one decimal for display; raw rates stay in the metrics structs.
//!
//! Neighbors: `metrics.rs` (the numbers shaped here), `dataset.rs` (category
//! keys), `mod.rs` (drives the run and calls into this module).

use std::collections::BTreeMap;

use serde::Serialize;

use super::dataset::category_of;
use super::metrics::{ModeSummary, QueryEvaluation, Verdict};

/// The three ranked modes the eval reports, in display order.
pub const REPORT_MODES: [&str; 3] = ["semantic", "hybrid", "exact"];

/// One ranked mode's metrics, rounded for output.
#[derive(Debug, Clone, Serialize)]
pub struct ModeReport {
    /// Number of queries averaged into this mode's metrics.
    pub runs: usize,
    pub top1: f64,
    pub top3: f64,
    pub top10: f64,
    #[serde(rename = "recallAt10")]
    pub recall_at10: f64,
    #[serde(rename = "precisionAt10")]
    pub precision_at10: f64,
    pub mrr: f64,
    #[serde(rename = "duplicateRate")]
    pub duplicate_rate: f64,
    #[serde(rename = "avgLatencyMs")]
    pub avg_latency_ms: f64,
}

impl ModeReport {
    pub fn from_summary(s: &ModeSummary) -> Self {
        Self {
            runs: s.runs,
            top1: pct1(s.top1_hit_rate),
            top3: pct1(s.top3_hit_rate),
            top10: pct1(s.top10_hit_rate),
            recall_at10: pct1(s.recall_at10),
            precision_at10: pct1(s.precision_at10),
            mrr: round2(s.mrr),
            duplicate_rate: pct1(s.duplicate_rate),
            avg_latency_ms: round1(s.avg_latency_ms),
        }
    }
}

/// Per-category hit counts for one ranked mode.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CategoryHits {
    pub n: usize,
    pub top1: f64,
    pub top3: f64,
    pub top10: f64,
}

/// Per-category breakdown for the two ranked modes that matter most.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CategoryReport {
    pub semantic: CategoryHits,
    pub hybrid: CategoryHits,
}

/// One query's per-mode hit detail.
#[derive(Debug, Clone, Serialize)]
pub struct PerModeDetail {
    pub top1: bool,
    pub top3: bool,
    pub top10: bool,
    #[serde(rename = "firstRelevantRank")]
    pub first_relevant_rank: Option<usize>,
    /// Which expected files this mode actually surfaced in the top-k — lets a
    /// reader see *what* matched, not just that something did.
    pub matched: Vec<String>,
}

impl PerModeDetail {
    pub fn from_eval(e: &QueryEvaluation) -> Self {
        Self {
            top1: e.top1_hit,
            top3: e.top3_hit,
            top10: e.top10_hit,
            first_relevant_rank: e.first_relevant_rank,
            matched: e.matched_expected.clone(),
        }
    }
}

/// Per-query report row.
#[derive(Debug, Clone, Serialize)]
pub struct PerQueryReport {
    pub id: String,
    pub query: String,
    pub expected: Vec<String>,
    pub semantic: PerModeDetail,
    pub hybrid: PerModeDetail,
    pub exact: PerModeDetail,
}

/// The two gate thresholds that produced a verdict, as percentages.
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdReport {
    #[serde(rename = "top3UsefulRate")]
    pub top3_useful_rate: f64,
    #[serde(rename = "recallAt10")]
    pub recall_at10: f64,
}

/// The graded verdict, shaped for output.
#[derive(Debug, Clone, Serialize)]
pub struct VerdictReport {
    pub grade: String,
    pub reasons: Vec<String>,
    /// The gate thresholds the grade was measured against — so a reader can see
    /// the bar, not just whether it was cleared.
    pub thresholds: ThresholdReport,
}

impl VerdictReport {
    pub fn from_verdict(v: &Verdict) -> Self {
        Self {
            grade: v.grade.as_str().to_string(),
            reasons: v.reasons.clone(),
            thresholds: ThresholdReport {
                top3_useful_rate: pct1(v.thresholds.top3_useful_rate),
                recall_at10: pct1(v.thresholds.recall_at10),
            },
        }
    }
}

/// The full serializable eval report.
#[derive(Debug, Clone, Serialize)]
pub struct QualityReport {
    #[serde(rename = "datasetSource")]
    pub dataset_source: String,
    #[serde(rename = "queryCount")]
    pub query_count: usize,
    #[serde(rename = "projectId", skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    pub verdict: VerdictReport,
    pub modes: BTreeMap<String, ModeReport>,
    #[serde(rename = "byCategory")]
    pub by_category: BTreeMap<String, CategoryReport>,
    #[serde(rename = "perQuery")]
    pub per_query: Vec<PerQueryReport>,
}

/// Build the per-category breakdown (semantic + hybrid) from per-query evals.
///
/// `query_ids` is parallel to each evaluation slice. The category of each query
/// id buckets its semantic and hybrid hits.
pub fn build_by_category(
    query_ids: &[String],
    semantic: &[QueryEvaluation],
    hybrid: &[QueryEvaluation],
) -> BTreeMap<String, CategoryReport> {
    // Accumulate raw counts first, then convert to one-decimal percentages.
    #[derive(Default)]
    struct Counts {
        n: usize,
        top1: usize,
        top3: usize,
        top10: usize,
    }
    let mut sem: BTreeMap<String, Counts> = BTreeMap::new();
    let mut hyb: BTreeMap<String, Counts> = BTreeMap::new();

    let tally = |acc: &mut BTreeMap<String, Counts>, cat: &str, e: &QueryEvaluation| {
        let c = acc.entry(cat.to_string()).or_default();
        c.n += 1;
        c.top1 += e.top1_hit as usize;
        c.top3 += e.top3_hit as usize;
        c.top10 += e.top10_hit as usize;
    };

    for (id, e) in query_ids.iter().zip(semantic) {
        tally(&mut sem, &category_of(id), e);
    }
    for (id, e) in query_ids.iter().zip(hybrid) {
        tally(&mut hyb, &category_of(id), e);
    }

    let shape = |c: &Counts| CategoryHits {
        n: c.n,
        top1: rate_pct(c.top1, c.n),
        top3: rate_pct(c.top3, c.n),
        top10: rate_pct(c.top10, c.n),
    };

    let mut out: BTreeMap<String, CategoryReport> = BTreeMap::new();
    for (cat, c) in &sem {
        out.entry(cat.clone()).or_default().semantic = shape(c);
    }
    for (cat, c) in &hyb {
        out.entry(cat.clone()).or_default().hybrid = shape(c);
    }
    out
}

/// Render the report to stdout in a compact, readable layout.
pub fn print_report(report: &QualityReport) {
    println!("Search Quality Eval");
    println!("===================");
    println!("Dataset:  {}", report.dataset_source);
    if let Some(pid) = &report.project_id {
        println!("Project:  {pid}");
    }
    println!("Queries:  {}", report.query_count);
    println!();

    println!(
        "Verdict (semantic):  {}",
        report.verdict.grade.to_uppercase()
    );
    println!(
        "  gates: top-3 useful >= {}%, recall@10 >= {}%",
        report.verdict.thresholds.top3_useful_rate, report.verdict.thresholds.recall_at10
    );
    for reason in &report.verdict.reasons {
        println!("  - {reason}");
    }
    println!();

    println!("Per-mode metrics (%, except n / MRR / latency):");
    println!(
        "  {:<9} {:>4} {:>6} {:>6} {:>6} {:>9} {:>10} {:>6} {:>10} {:>9}",
        "mode", "n", "top1", "top3", "top10", "recall@10", "prec@10", "mrr", "dup-rate", "lat-ms"
    );
    for mode in REPORT_MODES {
        if let Some(m) = report.modes.get(mode) {
            println!(
                "  {:<9} {:>4} {:>6} {:>6} {:>6} {:>9} {:>10} {:>6} {:>10} {:>9}",
                mode,
                m.runs,
                m.top1,
                m.top3,
                m.top10,
                m.recall_at10,
                m.precision_at10,
                m.mrr,
                m.duplicate_rate,
                m.avg_latency_ms,
            );
        }
    }
    println!();

    println!("By category (semantic / hybrid top-3 hit %):");
    println!(
        "  {:<8} {:>4} {:>10} {:>10}",
        "category", "n", "sem-top3", "hyb-top3"
    );
    for (cat, c) in &report.by_category {
        println!(
            "  {:<8} {:>4} {:>10} {:>10}",
            cat, c.semantic.n, c.semantic.top3, c.hybrid.top3
        );
    }
}

// ---------------------------------------------------------------------------
// Rounding helpers
// ---------------------------------------------------------------------------

/// A 0–1 rate as a one-decimal percentage value (e.g. 0.834 → 83.4).
fn pct1(rate: f64) -> f64 {
    round1(rate * 100.0)
}

/// Hit count over n as a one-decimal percentage (0 when n == 0).
fn rate_pct(hits: usize, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        pct1(hits as f64 / n as f64)
    }
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::super::metrics::evaluate_query;
    use super::*;

    fn eval(raw: &[&str], expected: &[&str]) -> QueryEvaluation {
        let raw: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
        let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        evaluate_query(&raw, &expected, "", 10)
    }

    #[test]
    fn mode_report_rounds_to_one_decimal() {
        let s = ModeSummary {
            top3_hit_rate: 0.8334,
            recall_at10: 0.6667,
            mrr: 0.4567,
            avg_latency_ms: 12.34,
            ..Default::default()
        };
        let r = ModeReport::from_summary(&s);
        assert_eq!(r.top3, 83.3);
        assert_eq!(r.recall_at10, 66.7);
        assert_eq!(r.mrr, 0.46);
        assert_eq!(r.avg_latency_ms, 12.3);
    }

    #[test]
    fn by_category_buckets_by_prefix() {
        let ids = vec![
            "sym-a".to_string(),
            "sym-b".to_string(),
            "doc-c".to_string(),
        ];
        // sym-a hits top1, sym-b misses, doc-c hits top1.
        let sem = vec![
            eval(&["a.rs"], &["a.rs"]),
            eval(&["x.rs"], &["b.rs"]),
            eval(&["c.rs"], &["c.rs"]),
        ];
        let hyb = sem.clone();
        let by_cat = build_by_category(&ids, &sem, &hyb);

        let sym = &by_cat["sym"];
        assert_eq!(sym.semantic.n, 2);
        assert_eq!(sym.semantic.top1, 50.0, "1 of 2 sym queries hit top1");
        let doc = &by_cat["doc"];
        assert_eq!(doc.semantic.n, 1);
        assert_eq!(doc.semantic.top1, 100.0);
    }

    #[test]
    fn by_category_handles_empty() {
        let by_cat = build_by_category(&[], &[], &[]);
        assert!(by_cat.is_empty());
    }

    #[test]
    fn per_mode_detail_carries_rank() {
        let e = eval(&["x.rs", "y.rs", "a.rs"], &["a.rs"]);
        let d = PerModeDetail::from_eval(&e);
        assert!(!d.top1);
        assert!(d.top3);
        assert_eq!(d.first_relevant_rank, Some(3));
    }

    #[test]
    fn report_serializes_to_json_with_expected_keys() {
        let s = ModeSummary {
            top3_hit_rate: 0.9,
            recall_at10: 0.8,
            ..Default::default()
        };
        let mut modes = BTreeMap::new();
        modes.insert("semantic".to_string(), ModeReport::from_summary(&s));
        let report = QualityReport {
            dataset_source: "bundled".to_string(),
            query_count: 1,
            project_id: Some("tenant-1".to_string()),
            verdict: good_verdict(),
            modes,
            by_category: BTreeMap::new(),
            per_query: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"datasetSource\":\"bundled\""));
        assert!(json.contains("\"recallAt10\""));
        assert!(json.contains("\"byCategory\""));
        assert!(json.contains("\"projectId\":\"tenant-1\""));
    }

    #[test]
    fn report_omits_absent_project_id() {
        let report = QualityReport {
            dataset_source: "bundled".to_string(),
            query_count: 0,
            project_id: None,
            verdict: good_verdict(),
            modes: BTreeMap::new(),
            by_category: BTreeMap::new(),
            per_query: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("projectId"));
    }

    /// A "good" verdict report built through the real conversion path.
    fn good_verdict() -> VerdictReport {
        use super::super::metrics::{classify, ModeSummary, QualityThresholds};
        let summary = ModeSummary {
            top3_hit_rate: 0.95,
            recall_at10: 0.9,
            ..Default::default()
        };
        VerdictReport::from_verdict(&classify(&summary, QualityThresholds::default()))
    }
}
