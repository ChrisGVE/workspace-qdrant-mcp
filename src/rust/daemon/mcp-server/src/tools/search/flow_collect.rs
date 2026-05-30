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
use crate::qdrant::fusion::{point_to_tagged, SearchType, TaggedResult};
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::flow::SearchQdrant;
use super::types::{ParentContext, Provenance, SearchMode, SearchResult};

// ---------------------------------------------------------------------------
// Per-collection search
// ---------------------------------------------------------------------------

/// Search one collection: dense + sparse legs concurrently.
///
/// Mirrors `searchCollection` in `search-qdrant.ts:149-158`.
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
                .unwrap_or_default()
        } else {
            vec![]
        }
    };

    let sparse_fut = async {
        if (mode == SearchMode::Hybrid || mode == SearchMode::Keyword) && sparse.is_some() {
            let sv = sparse.unwrap();
            let mut entries: Vec<(u32, f32)> = sv.iter().map(|(&k, &v)| (k, v)).collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            let (indices, values): (Vec<u32>, Vec<f32>) = entries.into_iter().unzip();
            if indices.is_empty() {
                return vec![];
            }
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
                .unwrap_or_default()
        } else {
            vec![]
        }
    };

    let (dense_pts, sparse_pts) = tokio::join!(dense_fut, sparse_fut);
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
    let mut metadata: HashMap<String, Value> = payload.clone();
    metadata.insert(
        "_search_type".to_string(),
        Value::String(tagged.search_type.as_str().to_string()),
    );
    let provenance = build_provenance(&payload, &collection);
    SearchResult {
        id: tagged.id,
        score: tagged.score,
        collection,
        content,
        title,
        metadata,
        provenance: Some(provenance),
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
