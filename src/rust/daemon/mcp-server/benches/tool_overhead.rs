//! Criterion micro-benchmarks for MCP tool-call overhead.
//!
//! # What is measured
//!
//! These benchmarks isolate **pure-CPU per-call costs** — envelope
//! construction, input parsing, and payload canonicalisation — that are
//! incurred on every tool invocation regardless of backend I/O.  They do NOT
//! hit the daemon, Qdrant, or SQLite.
//!
//! Three benchmark groups:
//!
//! 1. **envelope** — `unknown_tool` / `error_text` / `ok_text` +
//!    `serde_json::to_string_pretty` of a representative `SearchResponse`.
//! 2. **input_parsing** — `SearchInput`, `ListInput`, `RetrieveInput`, and
//!    `GrepInput` `from_args` from representative argument maps.
//! 3. **stable_stringify** — canonical payload serialisation of a
//!    representative store/rule payload map.
//!
//! # Warmup discipline (PERF-01)
//!
//! Criterion runs its own warm-up phase before collecting samples.  The
//! default warm-up time is 3 s (Criterion default) and sample count is 100.
//! For the full comparison run (see `benches/README.md`) the settings below
//! are used:
//!
//! - `warm_up_time`: 3 s (equivalent to discarding ≥20 calls at realistic
//!   throughput, satisfying the N=20 warmup-discard requirement of PERF-01)
//! - `measurement_time`: 5 s
//! - `sample_size`: 100
//!
//! Criterion reports p50 and p95 from its steady-state measurement window
//! in the HTML report (`target/criterion/`) and on stdout as `mean ± std`.
//! The p95 corresponds approximately to the 95th-percentile line of the
//! "PDF" plot in the HTML output.
//!
//! # TS-vs-Rust comparison
//!
//! The TypeScript side is measured separately (see `benches/README.md`).
//! This file benchmarks only the Rust server.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::{json, Map, Value};
use std::time::Duration;

use mcp_server::canonicalize::stable_stringify::stable_stringify;
use mcp_server::tools::envelope::{error_text, ok_text, unknown_tool};
use mcp_server::tools::grep::GrepInput;
use mcp_server::tools::list::ListInput;
use mcp_server::tools::retrieve::RetrieveInput;
use mcp_server::tools::search::{SearchMode, SearchOptions, SearchResponse, SearchScope};

// ---------------------------------------------------------------------------
// Representative fixtures
// ---------------------------------------------------------------------------

/// Build a representative `SearchResponse` with 3 results — mirrors the
/// size seen in golden tests and typical queries in production.
fn representative_search_response() -> SearchResponse {
    use mcp_server::tools::search::types::SearchResult;
    use std::collections::HashMap;

    let make_result = |i: usize| -> SearchResult {
        let mut metadata = HashMap::new();
        metadata.insert("file_path".to_string(), json!("src/tools/search/mod.rs"));
        metadata.insert("language".to_string(), json!("rust"));
        metadata.insert("chunk_index".to_string(), json!(i));
        metadata.insert("_search_type".to_string(), json!("hybrid"));
        SearchResult {
            id: format!("aaaabbbb-cccc-dddd-eeee-{i:012}"),
            score: 0.85 - (i as f64 * 0.05),
            collection: "workspace-qdrant-projects".to_string(),
            content: "pub fn search_tool(args: &Map<String, Value>) -> CallToolResult {"
                .to_string(),
            title: Some("search_tool".to_string()),
            metadata,
            provenance: None,
            parent_context: None,
            graph_context: None,
        }
    };

    SearchResponse {
        results: (0..3).map(make_result).collect(),
        total: 3,
        query: "search tool implementation".to_string(),
        mode: SearchMode::Hybrid,
        scope: SearchScope::Project,
        collections_searched: vec!["workspace-qdrant-projects".to_string()],
        status: None,
        status_reason: None,
        branch: Some("main".to_string()),
        diversity_score: Some(0.72),
    }
}

/// Build a representative `SearchInput` args map.
fn search_args() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("query".to_string(), json!("search tool implementation"));
    m.insert("mode".to_string(), json!("hybrid"));
    m.insert("limit".to_string(), json!(10));
    m.insert("scope".to_string(), json!("project"));
    m.insert("branch".to_string(), json!("main"));
    m.insert("fileType".to_string(), json!("rust"));
    m
}

/// Build a representative `ListInput` args map.
fn list_args() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("path".to_string(), json!("src/tools"));
    m.insert("depth".to_string(), json!(3u64));
    m.insert("format".to_string(), json!("tree"));
    m.insert("language".to_string(), json!("rust"));
    m.insert("limit".to_string(), json!(200u64));
    m
}

/// Build a representative `RetrieveInput` args map.
fn retrieve_args() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("collection".to_string(), json!("projects"));
    m.insert("limit".to_string(), json!(10u64));
    m.insert("offset".to_string(), json!(0u64));
    m.insert(
        "documentId".to_string(),
        json!("aaaabbbb-cccc-dddd-eeee-000000000001"),
    );
    m
}

/// Build a representative `GrepInput` args map.
fn grep_args() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("pattern".to_string(), json!("dispatch_tool"));
    m.insert("regex".to_string(), json!(false));
    m.insert("caseSensitive".to_string(), json!(true));
    m.insert("scope".to_string(), json!("project"));
    m.insert("contextLines".to_string(), json!(2u64));
    m.insert("maxResults".to_string(), json!(50u64));
    m
}

/// Build a representative store/rule payload map for `stable_stringify`.
fn rule_payload() -> Value {
    json!({
        "label": "no-direct-qdrant-writes",
        "action": "add",
        "sourceType": "rule",
        "content": "Daemon owns all Qdrant writes. MCP server is read-only.",
        "scope": "global",
        "title": "No direct Qdrant writes from MCP",
        "tags": ["architecture", "guardrail"],
        "priority": 1
    })
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

/// Group 1: envelope construction + JSON serialisation.
///
/// Measures the three envelope helpers and the `serde_json::to_string_pretty`
/// call on a realistic `SearchResponse`.  Together these constitute the output
/// path cost on every successful or failed tool call.
fn bench_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope");

    group.bench_function("unknown_tool", |b| {
        b.iter(|| {
            let r = unknown_tool("nonexistent_tool_name");
            // Prevent the compiler from optimising the call away.
            std::hint::black_box(r);
        });
    });

    group.bench_function("error_text", |b| {
        b.iter(|| {
            let r = error_text("Daemon unavailable: connection refused after 3 retries");
            std::hint::black_box(r);
        });
    });

    group.bench_function("ok_text_search_response", |b| {
        let resp = representative_search_response();
        b.iter(|| {
            let r = ok_text(std::hint::black_box(&resp));
            std::hint::black_box(r);
        });
    });

    group.bench_function("to_string_pretty_search_response", |b| {
        let resp = representative_search_response();
        b.iter(|| {
            let s = serde_json::to_string_pretty(std::hint::black_box(&resp)).unwrap();
            std::hint::black_box(s);
        });
    });

    group.finish();
}

/// Group 2: input argument parsing.
///
/// Measures `from_args` for each of the four tools with inputs.  This is the
/// deserialization cost paid before the backend call.
fn bench_input_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("input_parsing");

    let s_args = search_args();
    group.bench_with_input(
        BenchmarkId::new("SearchInput", "typical"),
        &s_args,
        |b, a| {
            b.iter(|| {
                // Use parse_args (permissive manual extraction) rather than serde
                // deserialization — SearchInput no longer derives Deserialize.
                let input = SearchOptions::parse_args(a).unwrap_or_default();
                std::hint::black_box(input);
            });
        },
    );

    let l_args = list_args();
    group.bench_with_input(BenchmarkId::new("ListInput", "typical"), &l_args, |b, a| {
        b.iter(|| {
            let input = ListInput::from_args(std::hint::black_box(a));
            std::hint::black_box(input);
        });
    });

    let r_args = retrieve_args();
    group.bench_with_input(
        BenchmarkId::new("RetrieveInput", "typical"),
        &r_args,
        |b, a| {
            b.iter(|| {
                let input = RetrieveInput::from_args(std::hint::black_box(a));
                std::hint::black_box(input);
            });
        },
    );

    let g_args = grep_args();
    group.bench_with_input(BenchmarkId::new("GrepInput", "typical"), &g_args, |b, a| {
        b.iter(|| {
            let input = GrepInput::from_args(std::hint::black_box(a));
            std::hint::black_box(input);
        });
    });

    group.finish();
}

/// Group 3: `stable_stringify` canonical payload serialisation.
///
/// Measures the per-call CPU cost of `stable_stringify` on a representative
/// rule payload with string, array, and integer fields.  This is called on
/// every write-path tool invocation (store, rules mutations) to derive the
/// idempotency key.
fn bench_stable_stringify(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_stringify");

    let payload = rule_payload();
    group.bench_function("rule_payload", |b| {
        b.iter(|| {
            let s = stable_stringify(std::hint::black_box(&payload));
            std::hint::black_box(s);
        });
    });

    // Also benchmark a larger nested payload to characterise scaling.
    let nested = json!({
        "collection": "workspace-qdrant-projects",
        "tenant_id": "proj_abc123def456",
        "content": "pub fn dispatch_tool(name: &str, args: &Map<String, Value>, ctx: &mut DispatchContext) -> CallToolResult",
        "metadata": {
            "file_path": "src/rust/daemon/mcp-server/src/tools/dispatch.rs",
            "language": "rust",
            "chunk_index": 0,
            "source_type": "project"
        },
        "priority": 0
    });
    group.bench_function("nested_store_payload", |b| {
        b.iter(|| {
            let s = stable_stringify(std::hint::black_box(&nested));
            std::hint::black_box(s);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion configuration
// ---------------------------------------------------------------------------

/// Configure warm-up and measurement times per PERF-01 requirements.
///
/// - `warm_up_time(3s)` ensures ≥20 calls are discarded before measurement
///   at any realistic throughput (3 s / 5 ms-per-call ≥ 600 calls >> 20).
/// - `measurement_time(5s)` collects a statistically stable sample window.
/// - `sample_size(100)` provides sufficient confidence for p50/p95 estimates.
fn configured_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = configured_criterion();
    targets = bench_envelope, bench_input_parsing, bench_stable_stringify
}
criterion_main!(benches);
