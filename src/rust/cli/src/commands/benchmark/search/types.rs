//! Types and static dataset for the search benchmark.

use std::collections::HashSet;

use super::super::stats::LatencyStats;

/// A single benchmark query with its search parameters.
pub(super) struct BenchQuery {
    pub label: &'static str,
    pub pattern: &'static str,
    pub regex: bool,
}

/// Results from a single query run on one engine.
pub(super) struct EngineResult {
    pub latency_ms: f64,
    pub match_count: usize,
    pub file_paths: HashSet<String>,
}

/// Aggregated comparison for a single query.
pub(super) struct QueryComparison {
    pub label: String,
    pub pattern: String,
    pub fts5_stats: LatencyStats,
    pub rg_stats: LatencyStats,
    pub fts5_match_count: usize,
    pub rg_match_count: usize,
    pub fts5_files: HashSet<String>,
    pub rg_files: HashSet<String>,
    pub shared_files: usize,
    pub fts5_only_files: usize,
    pub rg_only_files: usize,
}

/// Default query set covering typical code search patterns.
pub(super) static DEFAULT_QUERIES: &[BenchQuery] = &[
    BenchQuery {
        label: "exact: common keyword",
        pattern: "async fn",
        regex: false,
    },
    BenchQuery {
        label: "exact: struct name",
        pattern: "ProcessingContext",
        regex: false,
    },
    BenchQuery {
        label: "exact: import path",
        pattern: "use std::collections::HashMap",
        regex: false,
    },
    BenchQuery {
        label: "exact: error handling",
        pattern: "anyhow::Result",
        regex: false,
    },
    BenchQuery {
        label: "exact: rare symbol",
        pattern: "CentralityCache",
        regex: false,
    },
    BenchQuery {
        label: "exact: trait impl",
        pattern: "impl Default for",
        regex: false,
    },
    BenchQuery {
        label: "exact: test annotation",
        pattern: "#[cfg(test)]",
        regex: false,
    },
    BenchQuery {
        label: "exact: multi-word",
        pattern: "queue processor",
        regex: false,
    },
    BenchQuery {
        label: "regex: fn signature",
        pattern: r"pub async fn \w+\(",
        regex: true,
    },
    BenchQuery {
        label: "regex: struct definition",
        pattern: r"pub struct \w+ \{",
        regex: true,
    },
    BenchQuery {
        label: "regex: fn definition",
        pattern: r"fn \w+\(",
        regex: true,
    },
    BenchQuery {
        label: "regex: mutable binding",
        pattern: r"let mut \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: trait impl",
        pattern: r"impl \w+ for \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: std imports",
        pattern: r"use (std|tokio|serde)::\w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: derive macros",
        pattern: r"#\[derive\(\w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: public decls",
        pattern: r"pub (fn|struct|enum|trait|type) \w+",
        regex: true,
    },
    BenchQuery {
        label: "regex: async Result",
        pattern: r"async fn \w+.*-> Result",
        regex: true,
    },
    BenchQuery {
        label: "regex: method chains",
        pattern: r"\.(await|unwrap|expect)\b",
        regex: true,
    },
];
