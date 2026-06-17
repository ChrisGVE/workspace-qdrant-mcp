//! Gold dataset loading + query categorization for the quality eval (#135).
//!
//! Located at: `src/rust/cli/src/commands/benchmark/quality/dataset.rs`
//!
//! The dataset is a YAML file of known-item queries, each with `expectedFiles`
//! (literal repo-relative paths or globs) that a good search should surface. It
//! is compiled into the binary via `include_str!` so the eval runs without a
//! bind-mounted repo (the deleted TS harness needed `WQM_REPO_DIR`); an explicit
//! `--dataset <path>` still overrides it for ad-hoc sets.
//!
//! Queries are grouped into categories by an id prefix (`sym-`, `impl-`, `doc-`,
//! `real-`, `pt-`, else `orig`), so a weak category (e.g. Portuguese on an
//! English-only embedding model) is visible per-category instead of silently
//! dragging the aggregate verdict. See [`category_of`].
//!
//! Neighbors: `metrics.rs` (scores each query's results), `path_match.rs`
//! (matches `expected_files` globs), the bundled
//! `benchmark-data/semantic-search-quality.yaml`.

use anyhow::{Context, Result};
use serde::Deserialize;

/// The gold dataset compiled into the binary. Verified against the live Rust
/// tree at authoring time; see the file's own header for the verification date.
pub const BUNDLED_DATASET_YAML: &str = include_str!("benchmark-data/semantic-search-quality.yaml");

/// The bundled dataset's logical source label (shown in the report).
pub const BUNDLED_DATASET_SOURCE: &str = "bundled:semantic-search-quality.yaml";

/// Category id prefixes. A query id whose segment before the first `-` is one of
/// these names that category; anything else is `orig`.
pub const CATEGORY_PREFIXES: &[&str] = &["pt", "sym", "impl", "doc", "real"];

/// The fallback category for ids without a recognized prefix.
pub const ORIG_CATEGORY: &str = "orig";

/// One known-item query and the files a good search should surface for it.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldQuery {
    pub id: String,
    pub query: String,
    /// Repo-relative paths or globs; a hit is any that appears in the top-k.
    #[serde(rename = "expectedFiles")]
    pub expected_files: Vec<String>,
}

/// Dataset-wide defaults.
///
/// Only `limit` (the result count) is honored by the eval; the known-item eval
/// always runs project-scoped over the `projects` collection, so any `scope` /
/// `collection` / `includeLibraries` keys in the YAML are accepted (serde
/// ignores unknown fields) and serve as human-readable documentation of intent,
/// but do not change behavior. They are intentionally NOT struct fields so the
/// only default with an effect is the one the code actually applies.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GoldDefaults {
    /// Result count requested per search (falls back to the `--top-k` argument).
    pub limit: Option<usize>,
}

/// The parsed gold dataset.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldDataset {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub defaults: GoldDefaults,
    pub queries: Vec<GoldQuery>,
}

impl GoldDataset {
    /// Parse the bundled (compiled-in) dataset.
    pub fn bundled() -> Result<Self> {
        Self::from_yaml(BUNDLED_DATASET_YAML).context("parsing the bundled gold dataset")
    }

    /// Parse a dataset from YAML text, then validate it.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let dataset: GoldDataset =
            serde_yaml_ng::from_str(text).context("dataset YAML is not valid")?;
        dataset.validate()?;
        Ok(dataset)
    }

    /// Reject empty/duplicate ids and empty expectation lists — a malformed gold
    /// set silently scores everything as a miss, so fail loudly instead.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.queries.is_empty(), "dataset has no queries");
        let mut seen = std::collections::HashSet::new();
        for q in &self.queries {
            anyhow::ensure!(!q.id.trim().is_empty(), "a query has an empty id");
            anyhow::ensure!(seen.insert(q.id.clone()), "duplicate query id: {}", q.id);
            anyhow::ensure!(
                !q.expected_files.is_empty(),
                "query {} has no expectedFiles",
                q.id
            );
            for f in &q.expected_files {
                anyhow::ensure!(
                    !f.trim().is_empty(),
                    "query {} has an empty expectedFile entry",
                    q.id
                );
            }
        }
        Ok(())
    }

    /// Count of queries per category (sorted by category name).
    pub fn category_counts(&self) -> Vec<(String, usize)> {
        let mut counts = std::collections::BTreeMap::new();
        for q in &self.queries {
            *counts.entry(category_of(&q.id)).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }
}

/// The category of a query id: its prefix before the first `-` when that prefix
/// is a recognized category name, else `orig`. Mirrors TS `categoryOf`.
pub fn category_of(id: &str) -> String {
    match id.split_once('-') {
        Some((prefix, _)) if CATEGORY_PREFIXES.contains(&prefix) => prefix.to_string(),
        _ => ORIG_CATEGORY.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_recognizes_known_prefixes() {
        assert_eq!(category_of("sym-rrf-fusion"), "sym");
        assert_eq!(category_of("impl-bm25-tokenizer"), "impl");
        assert_eq!(category_of("doc-canonical-collections"), "doc");
        assert_eq!(category_of("real-rrf-keywords"), "real");
        assert_eq!(category_of("pt-fila-retry"), "pt");
    }

    #[test]
    fn category_falls_back_to_orig() {
        // No dash → orig.
        assert_eq!(category_of("hybridrrf"), "orig");
        // Unknown prefix → orig (the whole id is treated as a known-item label).
        assert_eq!(category_of("project-id-resolution"), "orig");
        assert_eq!(category_of("score-threshold"), "orig");
    }

    #[test]
    fn category_empty_prefix_is_orig() {
        // Leading dash → empty prefix, not a known category.
        assert_eq!(category_of("-weird"), "orig");
    }

    #[test]
    fn from_yaml_parses_minimal_dataset() {
        let yaml = r#"
name: test
description: a small set
queries:
  - id: sym-foo
    query: foo lookup
    expectedFiles:
      - src/a.rs
  - id: known-item
    query: where is bar
    expectedFiles:
      - src/b.rs
      - "**/c.rs"
"#;
        let ds = GoldDataset::from_yaml(yaml).unwrap();
        assert_eq!(ds.name, "test");
        assert_eq!(ds.queries.len(), 2);
        assert_eq!(ds.queries[0].expected_files, vec!["src/a.rs"]);
        assert_eq!(ds.queries[1].expected_files.len(), 2);
    }

    #[test]
    fn from_yaml_rejects_duplicate_ids() {
        let yaml = r#"
name: dup
queries:
  - id: x
    query: q1
    expectedFiles: [a.rs]
  - id: x
    query: q2
    expectedFiles: [b.rs]
"#;
        let err = GoldDataset::from_yaml(yaml).unwrap_err().to_string();
        assert!(err.contains("duplicate query id"), "got: {err}");
    }

    #[test]
    fn from_yaml_rejects_empty_expected_files() {
        let yaml = r#"
name: empty
queries:
  - id: x
    query: q
    expectedFiles: []
"#;
        let err = GoldDataset::from_yaml(yaml).unwrap_err().to_string();
        assert!(err.contains("no expectedFiles"), "got: {err}");
    }

    #[test]
    fn category_counts_groups_and_sorts() {
        let yaml = r#"
name: counts
queries:
  - id: sym-a
    query: q
    expectedFiles: [a.rs]
  - id: sym-b
    query: q
    expectedFiles: [b.rs]
  - id: doc-c
    query: q
    expectedFiles: [c.rs]
  - id: plain-item
    query: q
    expectedFiles: [d.rs]
"#;
        let ds = GoldDataset::from_yaml(yaml).unwrap();
        let counts = ds.category_counts();
        assert_eq!(
            counts,
            vec![
                ("doc".to_string(), 1),
                ("orig".to_string(), 1),
                ("sym".to_string(), 2),
            ]
        );
    }

    #[test]
    fn bundled_dataset_parses_and_validates() {
        // The compiled-in gold set must always be well-formed.
        let ds = GoldDataset::bundled().expect("bundled dataset must parse");
        assert!(
            ds.queries.len() >= 30,
            "expected ~30-50 gold queries, got {}",
            ds.queries.len()
        );
    }
}
