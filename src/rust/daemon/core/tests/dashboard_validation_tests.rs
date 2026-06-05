//! Grafana dashboard validation (Task 84, D6).
//!
//! Loads every `docker/grafana/dashboards/*.json`, and asserts:
//!   1. valid JSON (malformed dashboards fail the suite),
//!   2. `schemaVersion` present,
//!   3. every `datasource.uid` is templated (`${datasource}`) or the
//!      conventional fixed `prometheus` — never an org-specific hash UID,
//!   4. every `wqm_memexd_*` metric a panel references is actually defined by
//!      the daemon (catches typos, dropped metrics, and spurious suffixes such
//!      as a `_total` glued onto a gauge).
//!
//! ## Canonical metric-name source (IMPL-N1)
//! Prometheus `*Vec` metrics with no live samples do NOT appear in
//! `Registry::gather()` / `encode()`, so a freshly-booted registry snapshot is
//! an *incomplete* list of names. The authoritative, complete set is the
//! `wqm_memexd_*` string literals the daemon defines in its metric modules
//! (`src/monitoring/**` + `src/graph/metrics.rs`) — metric names must be static
//! literals, so scanning the definition sources is exact. A separate test still
//! boots the registry and renders `encode()` to guard the *runtime* invariant
//! that names carry a single `wqm_memexd_` prefix (no `memexd_memexd_` regression).
//!
//! Non-`wqm_memexd_` references (Qdrant's `qdrant_*`, the MCP server's
//! `wqm_mcp_*`, Prometheus's `up`, Grafana template vars) are out of the
//! daemon's scope and intentionally not checked here.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

/// Metric names a dashboard may reference that the daemon does not (yet) emit —
/// documented forward-compatible panels. Keep this list tiny and justified.
const KNOWN_PENDING: &[&str] = &[
    // codegraph.json has a forward-compatible query-latency panel; the daemon
    // exposes no graph query-duration histogram yet (handover / PRD §D3).
    "wqm_memexd_graph_query_duration_seconds",
];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/src/rust/daemon/core
    manifest_dir()
        .join("../../../..")
        // CATEGORY-B: test-local repo-root resolution; never persisted or sent.
        .canonicalize()
        .expect("resolve repo root from CARGO_MANIFEST_DIR")
}

fn dashboards_dir() -> PathBuf {
    repo_root().join("docker/grafana/dashboards")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().map_or(false, |x| x == "rs") {
            let name = p.file_name().unwrap().to_string_lossy();
            // Skip dedicated test files (they may contain illustrative,
            // non-real metric names).
            if !name.ends_with("_tests.rs") && name != "tests.rs" {
                out.push(p);
            }
        }
    }
}

/// The complete set of `wqm_memexd_*` metric names the daemon defines, scanned
/// from the metric-definition source modules.
fn defined_metric_names() -> BTreeSet<String> {
    let src = manifest_dir().join("src");
    let mut files = vec![src.join("graph/metrics.rs")];
    collect_rs(&src.join("monitoring"), &mut files);

    let re = Regex::new(r"wqm_memexd_[a-z0-9_]+").unwrap();
    let mut names = BTreeSet::new();
    for f in &files {
        if let Ok(text) = fs::read_to_string(f) {
            for m in re.find_iter(&text) {
                names.insert(m.as_str().trim_end_matches('_').to_string());
            }
        }
    }
    names
}

/// Strip a Prometheus histogram-derived suffix so a `..._bucket` / `..._sum` /
/// `..._count` reference resolves to its base histogram name. Only used as a
/// fallback after a full-name match fails, so genuine gauges whose name ends in
/// `_count` (e.g. `wqm_memexd_search_result_count`) are matched directly first.
fn strip_histogram_suffix(name: &str) -> &str {
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

/// Recursively assert every `datasource.uid` is templated or conventional.
fn check_datasource_uids(v: &Value, path: &Path) {
    match v {
        Value::Object(map) => {
            if let Some(ds) = map.get("datasource") {
                let uid = match ds {
                    Value::Object(o) => o.get("uid").and_then(Value::as_str),
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                };
                if let Some(uid) = uid {
                    assert!(
                        uid == "prometheus" || uid == "${datasource}",
                        "{:?}: non-templated datasource uid {:?} (use \"prometheus\" or \"${{datasource}}\")",
                        path,
                        uid
                    );
                }
            }
            for child in map.values() {
                check_datasource_uids(child, path);
            }
        }
        Value::Array(arr) => {
            for child in arr {
                check_datasource_uids(child, path);
            }
        }
        _ => {}
    }
}

fn dashboard_files() -> Vec<PathBuf> {
    let dir = dashboards_dir();
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read dashboards dir {:?}: {e}", dir))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "json"))
        .collect();
    files.sort();
    files
}

#[test]
fn dashboards_valid_templated_and_reference_real_metrics() {
    let defined = defined_metric_names();
    assert!(
        defined.contains("wqm_memexd_uptime_seconds"),
        "sanity: defined-metric scan should find a known core metric"
    );
    let pending: BTreeSet<&str> = KNOWN_PENDING.iter().copied().collect();
    let metric_re = Regex::new(r"wqm_memexd_[a-z0-9_]+").unwrap();

    let files = dashboard_files();
    assert!(
        files.len() >= 5,
        "expected the dashboard set (found {})",
        files.len()
    );

    for path in &files {
        let text = fs::read_to_string(path).unwrap();
        let json: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("malformed JSON in {:?}: {e}", path));

        assert!(
            json.get("schemaVersion").is_some(),
            "{:?}: missing schemaVersion",
            path
        );

        check_datasource_uids(&json, path);

        for m in metric_re.find_iter(&text) {
            let name = m.as_str();
            if defined.contains(name) || pending.contains(name) {
                continue;
            }
            let base = strip_histogram_suffix(name);
            assert!(
                defined.contains(base) || pending.contains(base),
                "{:?}: panel references unknown metric {:?} (not defined by the daemon)",
                path,
                name
            );
        }
    }
}

#[test]
fn daemon_metrics_render_with_single_prefix() {
    use workspace_qdrant_core::monitoring::metrics_core::METRICS;

    // Exercise a metric so encode() yields output, then guard the runtime
    // single-prefix invariant (A4: no `memexd_memexd_` double prefix).
    METRICS.set_uptime(1.0);
    let out = METRICS.encode().expect("encode /metrics snapshot");
    assert!(!out.is_empty(), "encoded metrics must not be empty");
    assert!(
        !out.contains("memexd_memexd_"),
        "double-prefixed metric name in /metrics output"
    );
    for line in out.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            if let Some(name) = rest.split_whitespace().next() {
                if name.starts_with("wqm_") {
                    assert!(
                        name.starts_with("wqm_memexd_"),
                        "unexpected wqm metric prefix: {}",
                        name
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_dashboard_json_is_rejected() {
    assert!(serde_json::from_str::<Value>("{ not: valid json").is_err());
}

#[test]
fn unknown_metric_is_not_in_defined_set() {
    let defined = defined_metric_names();
    assert!(!defined.is_empty());
    assert!(!defined.contains("wqm_memexd_bogus_does_not_exist"));
}
