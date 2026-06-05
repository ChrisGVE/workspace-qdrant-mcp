//! Metric-inventory doc completeness (Task 85, A3).
//!
//! Fails CI if a `wqm_memexd_*` metric is defined in the daemon's metric
//! modules but not documented in `docker/docs/telemetry.md`. Uses the same
//! authoritative source as the dashboard validation test: static metric-name
//! literals scanned from the definition modules (complete where a registry
//! snapshot is not — empty *Vec metrics never appear in encode()).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    manifest_dir()
        .join("../../../..")
        // CATEGORY-B: test-local repo-root resolution; never persisted or sent.
        .canonicalize()
        .expect("resolve repo root from CARGO_MANIFEST_DIR")
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
            if !name.ends_with("_tests.rs") && name != "tests.rs" {
                out.push(p);
            }
        }
    }
}

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

#[test]
fn every_daemon_metric_is_documented() {
    let doc_path = repo_root().join("docker/docs/telemetry.md");
    let doc = fs::read_to_string(&doc_path).unwrap_or_else(|e| panic!("read {:?}: {e}", doc_path));

    let defined = defined_metric_names();
    assert!(
        defined.contains("wqm_memexd_uptime_seconds"),
        "sanity: scan should find a known core metric"
    );

    let missing: Vec<&String> = defined
        .iter()
        .filter(|n| !doc.contains(n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "metrics defined in code but absent from docker/docs/telemetry.md: {:?}",
        missing
    );
}
