//! Fusion-pipeline baseline benchmark (WI / #82 task 1, TEMPORARY).
//!
//! Captures the pre-move median for `fuse_and_sort` + `diversify_slice_convert`
//! over a fixed synthetic corpus (2 collections × 200 `TaggedResult` each) so
//! PR3 (the SQLite-free search-pipeline split, tasks 24-28) can gate against a
//! +5% regression reference. Delete once that comparison is recorded.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::Value;

use mcp_server::qdrant::fusion::{SearchType, TaggedResult};
use mcp_server::tools::search::flow_collect::{diversify_slice_convert, fuse_and_sort};
use mcp_server::tools::search::options::{SearchInput, SearchOptions};
use mcp_server::tools::search::scope::ScopeContext;
use mcp_server::tools::search::types::SearchMode;

/// Build a fixed synthetic corpus: 2 collections × 200 results each, alternating
/// Semantic/Keyword legs so the RRF fusion path is exercised.
fn make_corpus() -> Vec<TaggedResult> {
    let mut out = Vec::with_capacity(400);
    for (ci, collection) in ["projects", "libraries"].iter().enumerate() {
        for i in 0..200 {
            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert(
                "tenant_id".to_string(),
                Value::String(format!("tenant-{}", ci)),
            );
            if *collection == "libraries" {
                payload.insert(
                    "library_name".to_string(),
                    Value::String(format!("lib-{}", i % 5)),
                );
            }
            payload.insert(
                "content".to_string(),
                Value::String(format!("synthetic content row {i} in {collection}")),
            );
            out.push(TaggedResult {
                id: format!("{collection}-{i}"),
                // Deterministic spread of scores; descending within each leg.
                score: 1.0 - (i as f64) / 250.0,
                collection: (*collection).to_string(),
                payload,
                search_type: if i % 2 == 0 {
                    SearchType::Semantic
                } else {
                    SearchType::Keyword
                },
            });
        }
    }
    out
}

fn bench_fusion_pipeline(c: &mut Criterion) {
    let corpus = make_corpus();
    let collections = vec!["projects".to_string(), "libraries".to_string()];
    let opts = SearchOptions::from_input(
        SearchInput {
            query: "bench".to_string(),
            limit: Some(50),
            diverse: Some(true),
            ..Default::default()
        },
        None,
    );

    c.bench_function("fusion_pipeline_baseline", |b| {
        b.iter(|| {
            let fused = fuse_and_sort(
                black_box(corpus.clone()),
                SearchMode::Hybrid,
                &ScopeContext::default(),
            );
            let (results, _) = diversify_slice_convert(fused, &opts, &collections);
            black_box(results)
        })
    });
}

criterion_group!(benches, bench_fusion_pipeline);
criterion_main!(benches);
