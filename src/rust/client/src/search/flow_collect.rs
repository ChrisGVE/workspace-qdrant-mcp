//! Per-collection search + result construction helpers.
//!
//! Extracted from `flow.rs` for size compliance.
//!
//! - `search_collection`: dense + sparse Qdrant legs in parallel.
//! - `build_provenance`, `tagged_to_search_result`: TaggedResult → SearchResult.
//! - `expand_parent_context`: optional parent-unit context fetch.

use std::collections::HashMap;

use serde_json::Value;

use crate::qdrant::client::QdrantRetrievedPoint;
use crate::qdrant::fusion::{
    apply_rrf_fusion, diversify_results, point_to_tagged, SearchType, TaggedResult,
    DEFAULT_DIVERSITY_CONFIG,
};
use wqm_common::constants::{COLLECTION_LIBRARIES, RANKING_AID_KEYS};

use super::flow::SearchQdrant;
use super::options::SearchOptions;
use crate::models::{ParentContext, Provenance, SearchMode, SearchResult};

// ---------------------------------------------------------------------------
// Fusion + finalize phases
// ---------------------------------------------------------------------------

/// Phase 3: relevance decay → RRF fusion → sort by score desc.
///
/// Split from the diversity/slice phases (`diversify_slice_convert`) so the
/// graph-expansion fusion pass (GitHub #80) can run between them — TS interposes
/// `expandAndFuseWithGraph` after the fusion sort and before diversity.
pub fn fuse_and_sort(
    all_tagged: Vec<TaggedResult>,
    mode: SearchMode,
    scope_ctx: &super::scope::ScopeContext,
) -> Vec<TaggedResult> {
    // Phase 2b: relevance decay (scope=group/all). Applied to the combined
    // results BEFORE fusion so the decay-induced ordering feeds the rank-based
    // RRF (mirrors TS `applyRelevanceDecay` before `finalizeResults`).
    let mut all_tagged = all_tagged;
    if let Some(decay_map) = &scope_ctx.decay_map {
        super::scope::apply_relevance_decay(&mut all_tagged, decay_map);
    }

    // Phase 3: RRF fusion (hybrid only) → sort by score desc.
    let mut sorted = if mode == SearchMode::Hybrid {
        apply_rrf_fusion(&all_tagged)
    } else {
        all_tagged
    };
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

/// Phases 4–6: source diversity re-ranking → slice to limit → convert.
pub fn diversify_slice_convert(
    fused: Vec<TaggedResult>,
    opts: &SearchOptions,
    collections: &[String],
) -> (Vec<SearchResult>, Option<f64>) {
    // Phase 4: Source diversity re-ranking (when >1 collection).
    let (diverse_results, diversity_score) = if opts.diverse && collections.len() > 1 {
        let (dr, ds) = diversify_results(fused, &DEFAULT_DIVERSITY_CONFIG);
        (dr, Some(ds))
    } else {
        (fused, None)
    };

    // Phase 5-6: Slice to limit and convert.
    let results: Vec<SearchResult> = diverse_results
        .into_iter()
        .take(opts.limit)
        .map(tagged_to_search_result)
        .collect();

    (results, diversity_score)
}

// ---------------------------------------------------------------------------
// Per-collection search
// ---------------------------------------------------------------------------

/// Search one collection: dense + sparse legs concurrently.
///
/// Mirrors `searchCollection` in `search-qdrant.ts:149-158` and
/// `searchAllCollections` in `search-helpers.ts:242-252`.
///
/// Each leg (`searchDense` / `searchSparse` in TS) has its own `try/catch`
/// that returns `[]` on failure — `searchCollection` itself NEVER throws.
/// `searchAllCollections` sets `status='uncertain'` only when `searchCollection`
/// *itself* throws (a collection-level error, not a leg-level error).
///
/// This function mirrors that: leg failures are silently swallowed (empty
/// results for that leg), and `Ok(combined)` is returned in all cases.
/// Only a collection-level panic (which cannot happen here) would be `Err`.
///
/// Score threshold applied at query level: dense = threshold, sparse = threshold * 0.5.
/// (Matches TS `search-qdrant.ts:105` and `:135`.)
pub async fn search_collection<Q>(
    qdrant: &Q,
    collection: &str,
    mode: SearchMode,
    dense: Option<&[f32]>,
    sparse: Option<&HashMap<u32, f32>>,
    filter: Option<qdrant_client::qdrant::Filter>,
    limit: u64,
    score_threshold: f64,
) -> Vec<TaggedResult>
where
    Q: SearchQdrant,
{
    let dense_threshold = score_threshold as f32;
    let sparse_threshold = (score_threshold * 0.5) as f32;

    // Dense and sparse legs run concurrently (mirrors TS `Promise.all` in
    // `search-qdrant.ts:153`). Issuing them sequentially doubled search latency
    // (two serial Qdrant gRPC round-trips) — see GitHub #83. `tokio::join!`
    // polls both on the same task; the shared `&Q` / `&filter` borrows are
    // immutable so concurrent polling is sound.
    let dense_fut = async {
        if (mode == SearchMode::Hybrid || mode == SearchMode::Semantic) && dense.is_some() {
            qdrant
                .search_dense(
                    collection,
                    dense.unwrap().to_vec(),
                    limit,
                    Some(dense_threshold),
                    filter.clone(),
                )
                .await
                .map_err(|e| e.to_string())
        } else {
            Ok(vec![])
        }
    };

    let sparse_fut = async {
        if (mode == SearchMode::Hybrid || mode == SearchMode::Keyword) && sparse.is_some() {
            let sv = sparse.unwrap();
            let mut entries: Vec<(u32, f32)> = sv.iter().map(|(&k, &v)| (k, v)).collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            let (indices, values): (Vec<u32>, Vec<f32>) = entries.into_iter().unzip();
            if indices.is_empty() {
                Ok(vec![])
            } else {
                qdrant
                    .search_sparse(
                        collection,
                        indices,
                        values,
                        limit,
                        Some(sparse_threshold),
                        filter.clone(),
                    )
                    .await
                    .map_err(|e| e.to_string())
            }
        } else {
            Ok(vec![])
        }
    };

    let (dense_result, sparse_result) = tokio::join!(dense_fut, sparse_fut);

    // Leg failures are silently swallowed: TS `searchDense`/`searchSparse` each
    // have their own try/catch returning [] on failure.  `searchCollection` in TS
    // never throws — it just combines the two (possibly empty) arrays.
    // Mirror that: use the results from whichever legs succeeded.
    combine_legs(
        collection,
        dense_result.unwrap_or_default(),
        sparse_result.unwrap_or_default(),
    )
}

/// Tag and concatenate the dense (`Semantic`) and sparse (`Keyword`) leg points.
///
/// Mirrors the array concatenation in TS `searchCollection`.
fn combine_legs(
    collection: &str,
    dense_pts: Vec<crate::qdrant::client::QdrantPoint>,
    sparse_pts: Vec<crate::qdrant::client::QdrantPoint>,
) -> Vec<TaggedResult> {
    let mut combined: Vec<TaggedResult> = Vec::new();
    combined.extend(
        dense_pts
            .into_iter()
            .map(|p| point_to_tagged(p, collection.to_string(), SearchType::Semantic)),
    );
    combined.extend(
        sparse_pts
            .into_iter()
            .map(|p| point_to_tagged(p, collection.to_string(), SearchType::Keyword)),
    );
    combined
}

// ---------------------------------------------------------------------------
// Result construction
// ---------------------------------------------------------------------------

/// Build provenance from payload and collection name.
///
/// Mirrors `buildProvenance` in `search-qdrant.ts:33-67`.
pub fn build_provenance(payload: &HashMap<String, Value>, collection: &str) -> Provenance {
    let source = match collection {
        c if c == COLLECTION_LIBRARIES => "libraries",
        "scratchpad" => "scratchpad",
        "rules" => "rules",
        _ => "projects",
    };
    let library_name = payload
        .get("library_name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let library_path = payload
        .get("library_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let doc_title = payload
        .get("document_name")
        .or_else(|| payload.get("title"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let source_project_id = payload
        .get("tenant_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Provenance {
        source: source.to_string(),
        library_name,
        library_path,
        doc_title,
        source_project_id,
    }
}

pub fn tagged_to_search_result(tagged: TaggedResult) -> SearchResult {
    let collection = tagged.collection.clone();
    let payload = tagged.payload.clone();
    let content = payload
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let title = payload
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // Graph-expansion nodes mirror TS `nodeToSearchResult` (search-graph-expansion.ts:84):
    // a NARROW metadata object (no `content`/`title`/`_search_type`) and no
    // provenance. All other results carry the full payload as metadata, a
    // `_search_type` tag, and computed provenance.
    let (metadata, provenance) = if tagged.search_type == SearchType::Graph {
        let mut meta = payload.clone();
        meta.remove("content");
        meta.remove("title");
        (meta, None)
    } else {
        let mut meta = payload.clone();
        // Strip the daemon's internal ranking-aid keys (~1.5–2k tokens/hit) that
        // a reading agent never consumes — salvaged from alkmimm PR #134.
        // The graph branch already returns a narrow metadata object.
        for key in RANKING_AID_KEYS {
            meta.remove(*key);
        }
        meta.insert(
            "_search_type".to_string(),
            Value::String(tagged.search_type.as_str().to_string()),
        );
        (meta, Some(build_provenance(&payload, &collection)))
    };

    SearchResult {
        id: tagged.id,
        score: tagged.score,
        collection,
        content,
        title,
        metadata,
        provenance,
        parent_context: None,
        graph_context: None,
    }
}

// ---------------------------------------------------------------------------
// Parent context expansion
// ---------------------------------------------------------------------------

/// Expand parent_context for results that carry a `parent_unit_id` payload key.
///
/// Mirrors `expandParentContext` in `search-qdrant.ts:246-254`.
pub async fn expand_parent_context<Q>(qdrant: &Q, results: &mut Vec<SearchResult>)
where
    Q: SearchQdrant,
{
    let mut by_coll: HashMap<String, HashMap<String, Vec<usize>>> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let parent_id = r
            .metadata
            .get("parent_unit_id")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if let Some(pid) = parent_id {
            by_coll
                .entry(r.collection.clone())
                .or_default()
                .entry(pid)
                .or_default()
                .push(i);
        }
    }
    for (collection, parent_map) in &by_coll {
        let parent_ids: Vec<String> = parent_map.keys().cloned().collect();
        if parent_ids.is_empty() {
            continue;
        }
        if let Ok(points) = qdrant.retrieve_by_ids(collection, parent_ids).await {
            assign_parent_contexts(results, parent_map, points);
        }
    }
}

fn assign_parent_contexts(
    results: &mut Vec<SearchResult>,
    parent_map: &HashMap<String, Vec<usize>>,
    points: Vec<QdrantRetrievedPoint>,
) {
    for point in points {
        if let Some(indices) = parent_map.get(&point.id) {
            let ctx = ParentContext {
                parent_unit_id: point.id.clone(),
                unit_type: point
                    .payload
                    .get("unit_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                unit_text: point
                    .payload
                    .get("unit_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                locator: point.payload.get("locator").and_then(|v| {
                    v.as_object()
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                }),
            };
            for &idx in indices {
                results[idx].parent_context = Some(ctx.clone());
            }
        }
    }
}

#[cfg(test)]
mod ranking_aid_strip_tests {
    use super::*;

    fn payload_with_ranking_aids() -> HashMap<String, Value> {
        let mut p = HashMap::new();
        p.insert("content".to_string(), Value::String("body".to_string()));
        p.insert(
            "source_file".to_string(),
            Value::String("src/main.rs".to_string()),
        );
        for key in RANKING_AID_KEYS {
            p.insert((*key).to_string(), Value::String("noise".to_string()));
        }
        p
    }

    #[test]
    fn default_search_strips_ranking_aid_keys() {
        // Default (non-graph) results must drop the daemon's ranking-aid keys
        // while keeping ordinary payload fields — alkmimm #134 salvage.
        let tagged = TaggedResult {
            id: "doc-1".to_string(),
            score: 1.0,
            collection: "projects".to_string(),
            payload: payload_with_ranking_aids(),
            search_type: SearchType::Hybrid,
        };
        let result = tagged_to_search_result(tagged);
        let meta = &result.metadata;
        for key in RANKING_AID_KEYS {
            assert!(
                !meta.contains_key(*key),
                "ranking-aid key {key:?} must be stripped from default search metadata"
            );
        }
        assert!(meta.contains_key("source_file"), "keeps ordinary fields");
        assert!(
            meta.contains_key("_search_type"),
            "keeps the search-type tag"
        );
    }
}
