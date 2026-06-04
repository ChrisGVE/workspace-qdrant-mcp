//! Fusion-pipeline benchmark for the shared search pipeline (WI-d4 #82 task 36).
//!
//! Drives the ACTUAL fusion path now living in `wqm-client` over a fixed
//! synthetic corpus, so CI can gate against a +5% regression vs the pre-move
//! baseline captured in Phase 0 (median ≈ 1.32 ms, +5% ceiling ≈ 1.39 ms; temp
//! `mcp-server/benches/fusion_baseline.rs`).
//!
//! Three groups:
//! - `fusion_pipeline` — `fuse_and_sort` + `diversify_slice_convert` (the gated
//!   metric; mirrors the baseline bench exactly so the comparison is valid).
//! - `search_collection_multi` — the per-collection dense+sparse leg over 2
//!   collections via a mock `SearchQdrant`.
//! - `embedding_leg` — the sparse-vector merge that runs on the embedding leg.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::Value;

use wqm_client::models::SearchMode;
use wqm_client::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use wqm_client::qdrant::fusion::{SearchType, TaggedResult};
use wqm_client::search::expansion::merge_sparse_vectors;
use wqm_client::search::flow::SearchQdrant;
use wqm_client::search::flow_collect::{diversify_slice_convert, fuse_and_sort, search_collection};
use wqm_client::search::options::{SearchInput, SearchOptions};
use wqm_client::search::scope::ScopeContext;

/// Build a fixed synthetic corpus: 2 collections × 200 results each, alternating
/// Semantic/Keyword legs so the RRF fusion path is exercised. Identical shape to
/// the Phase-0 baseline bench for a valid apples-to-apples comparison.
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

fn bench_opts() -> SearchOptions {
    SearchOptions::from_input(
        SearchInput {
            query: "bench".to_string(),
            limit: Some(50),
            diverse: Some(true),
            ..Default::default()
        },
        None,
    )
}

// ── Group 1: fusion (gated metric, mirrors baseline) ─────────────────────────

fn bench_fusion_pipeline(c: &mut Criterion) {
    let corpus = make_corpus();
    let collections = vec!["projects".to_string(), "libraries".to_string()];
    let opts = bench_opts();

    c.bench_function("fusion_pipeline", |b| {
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

// ── Group 2: per-collection search_collection over 2 collections ─────────────

/// Mock Qdrant returning ~200 dense + ~200 sparse points per leg.
struct CorpusQdrant {
    dense: Vec<QdrantPoint>,
    sparse: Vec<QdrantPoint>,
}

impl SearchQdrant for CorpusQdrant {
    async fn search_dense(
        &self,
        _c: &str,
        _v: Vec<f32>,
        _l: u64,
        _t: Option<f32>,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        Ok(self.dense.clone())
    }
    async fn search_sparse(
        &self,
        _c: &str,
        _i: Vec<u32>,
        _v: Vec<f32>,
        _l: u64,
        _t: Option<f32>,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        Ok(self.sparse.clone())
    }
    async fn scroll_page(
        &self,
        _c: &str,
        _f: Option<qdrant_client::qdrant::Filter>,
        _l: u32,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }
    async fn retrieve_by_ids(
        &self,
        _c: &str,
        _ids: Vec<String>,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }
}

fn make_points(collection: &str, n: usize) -> Vec<QdrantPoint> {
    (0..n)
        .map(|i| {
            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert(
                "content".to_string(),
                Value::String(format!("row {i} in {collection}")),
            );
            QdrantPoint {
                id: format!("{collection}-{i}"),
                score: 1.0 - (i as f64) / 250.0,
                payload,
            }
        })
        .collect()
}

fn bench_search_collection_multi(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    let collections = ["projects", "libraries"];
    let qdrants: Vec<CorpusQdrant> = collections
        .iter()
        .map(|coll| CorpusQdrant {
            dense: make_points(coll, 200),
            sparse: make_points(coll, 200),
        })
        .collect();
    let dense = vec![0.1f32; 384];
    let sparse: HashMap<u32, f32> = (0..32u32).map(|k| (k, 0.5)).collect();

    c.bench_function("search_collection_multi", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut all: Vec<TaggedResult> = Vec::new();
                for (coll, q) in collections.iter().zip(qdrants.iter()) {
                    let leg = search_collection(
                        q,
                        coll,
                        SearchMode::Hybrid,
                        Some(dense.as_slice()),
                        Some(&sparse),
                        None,
                        100,
                        0.3,
                    )
                    .await;
                    all.extend(leg);
                }
                black_box(all)
            })
        })
    });
}

// ── Group 3: embedding leg (sparse-vector merge) ─────────────────────────────

fn bench_embedding_leg(c: &mut Criterion) {
    let original: HashMap<u32, f32> = (0..256u32).map(|k| (k, 0.7)).collect();
    // Half overlapping, half new indices — exercises the no-overwrite merge.
    let expansion: HashMap<u32, f32> = (128..384u32).map(|k| (k, 0.4)).collect();

    c.bench_function("embedding_leg_merge_sparse", |b| {
        b.iter(|| {
            black_box(merge_sparse_vectors(
                black_box(&original),
                black_box(&expansion),
                0.5,
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_fusion_pipeline,
    bench_search_collection_multi,
    bench_embedding_leg
);
criterion_main!(benches);
