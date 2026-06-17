//! Search-quality eval — hit-rate of the live pipeline against a gold set (#135).
//!
//! Located at: `src/rust/cli/src/commands/benchmark/quality/mod.rs`
//!
//! `wqm benchmark search-quality` runs each known-item query from the gold
//! dataset through the LIVE search pipeline in three ranked modes (semantic,
//! hybrid, exact), then reports hit@k / recall@10 / precision@10 / MRR /
//! duplicate-rate / latency per mode, a per-category breakdown, a per-query
//! detail, and a quality verdict. It complements the latency benchmark
//! (`wqm benchmark search`) by measuring *quality* rather than speed.
//!
//! The pipeline is the real one: semantic/hybrid go through
//! `wqm_client::search::run_search_pipeline` (the same code path the MCP server
//! and `wqm search` run); exact goes through `wqm_client::search::search_exact`.
//! No searcher is reimplemented here. Each issued search is logged to
//! `search_events` with `actor = 'benchmark'` so organic-query mining can
//! exclude the eval's own traffic (migration v47; recipe in the module docs and
//! `docs/testing/semantic-search-benchmarking.md`).
//!
//! Module layout:
//! - `dataset`    — gold YAML loading + query categorization
//! - `path_match` — repo-relative path normalization + glob matching
//! - `metrics`    — pure metric math + verdict (unit-tested on synthetic data)
//! - `report`     — output shaping (terminal + JSON)
//!
//! Neighbors: `../mod.rs` (benchmark subcommand dispatch), `../search/` (the
//! latency benchmark sibling), `../../search/hybrid.rs` (the live-pipeline glue
//! this runner mirrors).

mod dataset;
mod metrics;
mod path_match;
mod report;

use anyhow::{Context, Result};
use secrecy::SecretString;

use wqm_client::models::{SearchMode, SearchResponse, SearchScope};
use wqm_client::search::options::{SearchInput, SearchOptions};
use wqm_client::search::scope::ScopeContext;
use wqm_client::{DaemonClient, QdrantReadClient};

use crate::commands::search::hybrid::resolve_project_id_from_cwd;
use crate::output;

use dataset::{GoldDataset, GoldQuery};
use metrics::{classify, evaluate_query, summarize_mode, QualityThresholds, QueryEvaluation};
use report::{
    build_by_category, print_report, ModeReport, PerModeDetail, PerQueryReport, QualityReport,
    VerdictReport,
};

/// Result paths are already repo-relative in the payload, so no workspace-root
/// prefix needs stripping — keep it explicit and empty.
const WORKSPACE_ROOT: &str = "";

/// `actor` tag for every search the eval issues, so mining can exclude it.
const BENCHMARK_ACTOR: &str = "benchmark";

/// The three ranked modes the eval runs, paired with how each is issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalMode {
    Semantic,
    Hybrid,
    Exact,
}

impl EvalMode {
    fn name(self) -> &'static str {
        match self {
            EvalMode::Semantic => "semantic",
            EvalMode::Hybrid => "hybrid",
            EvalMode::Exact => "exact",
        }
    }
}

/// Execute the search-quality eval.
///
/// `dataset_path` overrides the bundled gold set; `top_k` bounds the ranked
/// paths scored; `output_file` optionally writes the JSON report.
pub async fn execute(
    dataset_path: Option<String>,
    top_k: usize,
    output_file: Option<String>,
) -> Result<()> {
    let (gold, source) = load_dataset(dataset_path)?;
    let project_id = resolve_project_id_from_cwd();

    output::section("Search Quality Eval");
    output::kv("Dataset", &gold.name);
    output::kv("Source", &source);
    if let Some(description) = &gold.description {
        output::kv("About", description);
    }
    output::kv("Queries", &gold.queries.len().to_string());
    print_category_counts(&gold);

    let Some(project_id) = project_id else {
        output::error("Current directory is not inside a registered project.");
        output::info("Register it first:  wqm project register");
        output::info("Or run from a registered project directory.");
        return Ok(());
    };
    output::kv("Project", &project_id);

    // The dataset's `defaults.limit` sets the result count when present;
    // otherwise the `--top-k` argument applies. The verdict and hit@k are scored
    // over `top_k` ranked results either way.
    let limit = gold.defaults.limit.unwrap_or(top_k);
    output::kv("Result limit", &limit.to_string());
    output::separator();

    let mut runner = LiveRunner::connect(&project_id).await?;
    let outcome = runner.run_all(&gold.queries, limit, top_k).await;

    let report = build_report(&gold, &source, Some(project_id), &outcome, top_k);
    print_report(&report);

    if let Some(path) = output_file {
        let json = serde_json::to_string_pretty(&report).context("serializing JSON report")?;
        std::fs::write(&path, json).with_context(|| format!("writing report to {path}"))?;
        println!("\nReport written to: {path}");
    }
    Ok(())
}

/// Load the gold dataset from an override path or the bundled set.
fn load_dataset(dataset_path: Option<String>) -> Result<(GoldDataset, String)> {
    match dataset_path {
        Some(path) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading dataset {path}"))?;
            let gold = GoldDataset::from_yaml(&text)?;
            Ok((gold, path))
        }
        None => {
            let gold = GoldDataset::bundled()?;
            Ok((gold, dataset::BUNDLED_DATASET_SOURCE.to_string()))
        }
    }
}

/// Print the per-category query counts so the user sees dataset composition.
fn print_category_counts(gold: &GoldDataset) {
    let counts = gold.category_counts();
    let rendered: Vec<String> = counts.iter().map(|(cat, n)| format!("{cat}={n}")).collect();
    output::kv("Categories", &rendered.join("  "));
}

// ---------------------------------------------------------------------------
// Per-query, per-mode outcome accumulation
// ---------------------------------------------------------------------------

/// All evaluations and latencies for one ranked mode, parallel to the query list.
#[derive(Default)]
struct ModeOutcome {
    evaluations: Vec<QueryEvaluation>,
    latencies_ms: Vec<f64>,
}

/// The full eval outcome: query ids plus one [`ModeOutcome`] per ranked mode.
#[derive(Default)]
struct EvalOutcome {
    query_ids: Vec<String>,
    semantic: ModeOutcome,
    hybrid: ModeOutcome,
    exact: ModeOutcome,
}

impl EvalOutcome {
    fn mode_mut(&mut self, mode: EvalMode) -> &mut ModeOutcome {
        match mode {
            EvalMode::Semantic => &mut self.semantic,
            EvalMode::Hybrid => &mut self.hybrid,
            EvalMode::Exact => &mut self.exact,
        }
    }
}

// ---------------------------------------------------------------------------
// Live runner — connects once, runs every query through the real pipeline
// ---------------------------------------------------------------------------

/// Holds the live connections and project scope used for every query.
struct LiveRunner {
    daemon: DaemonClient,
    qdrant: QdrantReadClient,
    project_id: String,
    scope_ctx: ScopeContext,
}

impl LiveRunner {
    /// Connect to the daemon + Qdrant and pre-resolve the project scope context.
    async fn connect(project_id: &str) -> Result<Self> {
        let mut daemon = crate::grpc::connect_default()
            .await
            .context("Daemon not running. Start with: wqm service start")?;
        let qdrant = QdrantReadClient::new(
            crate::config::resolve_qdrant_url(),
            crate::config::resolve_qdrant_api_key().map(SecretString::from),
        );
        // Project scope needs no daemon scope-resolution call (no group/all
        // tenant filter), mirroring `resolve_scope_ctx` for scope=project.
        let scope_ctx = resolve_project_scope_ctx(&mut daemon, project_id).await;
        Ok(Self {
            daemon,
            qdrant,
            project_id: project_id.to_string(),
            scope_ctx,
        })
    }

    /// Run every gold query through all three ranked modes.
    ///
    /// `limit` is the result count requested from each search; `top_k` bounds
    /// how deep the ranked list is scored (hit@k / recall). They differ when the
    /// dataset's `defaults.limit` asks for more results than the scoring depth.
    async fn run_all(&mut self, queries: &[GoldQuery], limit: usize, top_k: usize) -> EvalOutcome {
        let mut outcome = EvalOutcome::default();
        for query in queries {
            outcome.query_ids.push(query.id.clone());
            for mode in [EvalMode::Semantic, EvalMode::Hybrid, EvalMode::Exact] {
                let (response, latency_ms) = self.run_one(query, mode, limit).await;
                let evaluation = evaluate_response(&response, &query.expected_files, top_k);
                let acc = outcome.mode_mut(mode);
                acc.evaluations.push(evaluation);
                acc.latencies_ms.push(latency_ms);
            }
        }
        outcome
    }

    /// Run one query in one mode, returning the response and the wall latency.
    ///
    /// Logs the search to `search_events` with `actor='benchmark'`
    /// (fire-and-forget — instrumentation never breaks the eval).
    async fn run_one(
        &mut self,
        query: &GoldQuery,
        mode: EvalMode,
        limit: usize,
    ) -> (SearchResponse, f64) {
        let opts = build_options(query, mode, limit, &self.project_id);
        let start = std::time::Instant::now();
        let response = self.dispatch(mode, &opts).await;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.log_benchmark_event(query, mode, &response, latency_ms)
            .await;
        (response, latency_ms)
    }

    /// Dispatch to the real pipeline for the given mode.
    async fn dispatch(&mut self, mode: EvalMode, opts: &SearchOptions) -> SearchResponse {
        match mode {
            EvalMode::Exact => wqm_client::search::search_exact(&mut self.daemon, opts).await,
            EvalMode::Semantic | EvalMode::Hybrid => {
                wqm_client::search::run_search_pipeline(
                    &mut self.daemon,
                    &self.qdrant,
                    Vec::new(), // tag-expansion keywords are an MCP-server feature
                    opts,
                    Some(&self.project_id),
                    false, // no tag expansion without the keyword adapter
                    &self.scope_ctx,
                    &(), // no fallback-metrics backend in the CLI
                )
                .await
            }
        }
    }

    /// Fire-and-forget `search_events` log tagged `actor='benchmark'`.
    async fn log_benchmark_event(
        &mut self,
        query: &GoldQuery,
        mode: EvalMode,
        response: &SearchResponse,
        latency_ms: f64,
    ) {
        let event_id = uuid::Uuid::new_v4().to_string();
        // `tool` must be one of the search_events CHECK values; 'mcp_qdrant'
        // is the indexed-search tool. `filters` records the eval mode so mined
        // history can tell the modes apart.
        let _ = self
            .daemon
            .log_search_event(
                event_id,
                BENCHMARK_ACTOR.to_string(),
                "mcp_qdrant".to_string(),
                "search".to_string(),
                None,
                Some(self.project_id.clone()),
                Some(query.query.clone()),
                Some(format!("{{\"mode\":\"{}\"}}", mode.name())),
                None,
                Some(response.results.len() as i32),
                Some(latency_ms as i64),
                None,
                None,
                None,
            )
            .await;
    }
}

/// Resolve the scope context for a project-scoped run.
///
/// scope=project applies no group/all tenant filter, so (mirroring
/// `hybrid::resolve_scope_ctx`) no daemon scope-resolution call is needed; the
/// default empty context is correct. The signature keeps `daemon` for symmetry
/// and future group/all support.
async fn resolve_project_scope_ctx(_daemon: &mut DaemonClient, _project_id: &str) -> ScopeContext {
    ScopeContext::default()
}

// ---------------------------------------------------------------------------
// Option building + response evaluation
// ---------------------------------------------------------------------------

/// Build per-mode `SearchOptions` for one query.
///
/// All modes are project-scoped and cross-branch (`branch: None`) so the eval
/// measures index quality, not branch filtering. Semantic and hybrid differ only
/// in `mode`; exact sets the `exact` flag, which routes [`LiveRunner::dispatch`]
/// to `search_exact`.
fn build_options(
    query: &GoldQuery,
    mode: EvalMode,
    limit: usize,
    project_id: &str,
) -> SearchOptions {
    let input = SearchInput {
        query: query.query.clone(),
        limit: Some(limit),
        scope: Some(SearchScope::Project),
        branch: None, // cross-branch: don't let branch filtering hide hits
        project_id: Some(project_id.to_string()),
        mode: Some(match mode {
            EvalMode::Semantic => SearchMode::Semantic,
            // Exact's mode field is unused (the `exact` flag routes it), but
            // Hybrid is the honest label for the fused pipeline.
            EvalMode::Hybrid | EvalMode::Exact => SearchMode::Hybrid,
        }),
        exact: Some(mode == EvalMode::Exact),
        ..Default::default()
    };
    SearchOptions::from_input(input, None)
}

/// Extract the ranked result paths from a response and evaluate them.
fn evaluate_response(
    response: &SearchResponse,
    expected: &[String],
    top_k: usize,
) -> QueryEvaluation {
    let raw_paths: Vec<String> = response
        .results
        .iter()
        .filter_map(extract_result_path)
        .collect();
    evaluate_query(&raw_paths, expected, WORKSPACE_ROOT, top_k)
}

/// Extract the repo-relative path from a result's metadata.
///
/// Preference order mirrors the TS harness: `relative_path` (the repo-relative
/// payload field) → `file_path` → `path`. Returns `None` when none is a
/// non-empty string.
fn extract_result_path(result: &wqm_client::models::SearchResult) -> Option<String> {
    for key in ["relative_path", "file_path", "path"] {
        if let Some(value) = result.metadata.get(key).and_then(|v| v.as_str()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Report assembly
// ---------------------------------------------------------------------------

/// Assemble the full report from the run outcome.
fn build_report(
    gold: &GoldDataset,
    source: &str,
    project_id: Option<String>,
    outcome: &EvalOutcome,
    _top_k: usize,
) -> QualityReport {
    let thresholds = QualityThresholds::default();
    let semantic_summary = summarize_mode(
        &outcome.semantic.evaluations,
        &outcome.semantic.latencies_ms,
    );
    let hybrid_summary = summarize_mode(&outcome.hybrid.evaluations, &outcome.hybrid.latencies_ms);
    let exact_summary = summarize_mode(&outcome.exact.evaluations, &outcome.exact.latencies_ms);

    let verdict = classify(&semantic_summary, thresholds);

    let mut modes = std::collections::BTreeMap::new();
    modes.insert(
        "semantic".to_string(),
        ModeReport::from_summary(&semantic_summary),
    );
    modes.insert(
        "hybrid".to_string(),
        ModeReport::from_summary(&hybrid_summary),
    );
    modes.insert(
        "exact".to_string(),
        ModeReport::from_summary(&exact_summary),
    );

    let by_category = build_by_category(
        &outcome.query_ids,
        &outcome.semantic.evaluations,
        &outcome.hybrid.evaluations,
    );

    let per_query = build_per_query(gold, outcome);

    QualityReport {
        dataset_source: source.to_string(),
        query_count: gold.queries.len(),
        project_id,
        verdict: VerdictReport::from_verdict(&verdict),
        modes,
        by_category,
        per_query,
    }
}

/// Build the per-query detail rows.
fn build_per_query(gold: &GoldDataset, outcome: &EvalOutcome) -> Vec<PerQueryReport> {
    gold.queries
        .iter()
        .enumerate()
        .map(|(i, q)| PerQueryReport {
            id: q.id.clone(),
            query: q.query.clone(),
            expected: q.expected_files.clone(),
            semantic: PerModeDetail::from_eval(&outcome.semantic.evaluations[i]),
            hybrid: PerModeDetail::from_eval(&outcome.hybrid.evaluations[i]),
            exact: PerModeDetail::from_eval(&outcome.exact.evaluations[i]),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use wqm_client::models::SearchResult;

    fn result_with_path(key: &str, value: &str) -> SearchResult {
        let mut metadata = HashMap::new();
        metadata.insert(key.to_string(), json!(value));
        SearchResult {
            id: "x".to_string(),
            score: 1.0,
            collection: "projects".to_string(),
            content: String::new(),
            title: None,
            metadata,
            provenance: None,
            parent_context: None,
            graph_context: None,
        }
    }

    #[test]
    fn extract_path_prefers_relative_path() {
        let mut r = result_with_path("relative_path", "src/a.rs");
        r.metadata
            .insert("file_path".to_string(), json!("/abs/src/a.rs"));
        assert_eq!(extract_result_path(&r).as_deref(), Some("src/a.rs"));
    }

    #[test]
    fn extract_path_falls_back_to_file_path_then_path() {
        let r = result_with_path("file_path", "src/b.rs");
        assert_eq!(extract_result_path(&r).as_deref(), Some("src/b.rs"));
        let r = result_with_path("path", "src/c.rs");
        assert_eq!(extract_result_path(&r).as_deref(), Some("src/c.rs"));
    }

    #[test]
    fn extract_path_none_when_blank_or_absent() {
        let r = result_with_path("relative_path", "   ");
        assert_eq!(extract_result_path(&r), None);
        let r = result_with_path("unrelated", "v");
        assert_eq!(extract_result_path(&r), None);
    }

    #[test]
    fn build_options_sets_exact_flag_only_for_exact_mode() {
        let q = GoldQuery {
            id: "x".to_string(),
            query: "find it".to_string(),
            expected_files: vec!["a.rs".to_string()],
        };
        assert!(!build_options(&q, EvalMode::Semantic, 10, "t").exact);
        assert!(!build_options(&q, EvalMode::Hybrid, 10, "t").exact);
        assert!(build_options(&q, EvalMode::Exact, 10, "t").exact);
    }

    #[test]
    fn build_options_is_project_scoped_cross_branch() {
        let q = GoldQuery {
            id: "x".to_string(),
            query: "find it".to_string(),
            expected_files: vec!["a.rs".to_string()],
        };
        let opts = build_options(&q, EvalMode::Semantic, 7, "tenant-9");
        assert_eq!(opts.scope, SearchScope::Project);
        assert_eq!(opts.limit, 7);
        assert!(opts.branch.is_none(), "eval is cross-branch");
        assert_eq!(opts.project_id.as_deref(), Some("tenant-9"));
        assert_eq!(opts.mode, SearchMode::Semantic);
    }

    fn empty_response() -> SearchResponse {
        SearchResponse {
            results: vec![],
            total: 0,
            query: "q".to_string(),
            mode: SearchMode::Hybrid,
            scope: SearchScope::Project,
            collections_searched: vec![],
            status: None,
            status_reason: None,
            branch: None,
            diversity_score: None,
        }
    }

    #[test]
    fn evaluate_response_extracts_and_scores_paths() {
        let mut response = empty_response();
        response.results = vec![
            result_with_path("relative_path", "src/x.rs"),
            result_with_path("relative_path", "src/a.rs"),
        ];
        let ev = evaluate_response(&response, &["src/a.rs".to_string()], 10);
        assert!(ev.top3_hit);
        assert_eq!(ev.first_relevant_rank, Some(2));
    }

    #[test]
    fn build_report_grades_from_semantic_and_keeps_all_modes() {
        // Two queries; semantic hits both top1, so the verdict is good.
        let gold = GoldDataset::from_yaml(
            "name: t\nqueries:\n  - id: a\n    query: q\n    expectedFiles: [src/a.rs]\n  - id: doc-b\n    query: q\n    expectedFiles: [src/b.rs]\n",
        )
        .unwrap();

        let mut outcome = EvalOutcome::default();
        for q in &gold.queries {
            outcome.query_ids.push(q.id.clone());
            let hit = evaluate_query(&[q.expected_files[0].clone()], &q.expected_files, "", 10);
            outcome.semantic.evaluations.push(hit.clone());
            outcome.semantic.latencies_ms.push(5.0);
            outcome.hybrid.evaluations.push(hit.clone());
            outcome.hybrid.latencies_ms.push(6.0);
            outcome.exact.evaluations.push(hit);
            outcome.exact.latencies_ms.push(1.0);
        }

        let report = build_report(&gold, "bundled", Some("t".to_string()), &outcome, 10);
        assert_eq!(report.query_count, 2);
        assert_eq!(report.verdict.grade, "good");
        assert_eq!(report.modes.len(), 3);
        assert_eq!(report.per_query.len(), 2);
        // Categories: 'a' has no recognized prefix → orig; 'doc-b' → doc.
        assert!(report.by_category.contains_key("orig"));
        assert!(report.by_category.contains_key("doc"));
    }
}
