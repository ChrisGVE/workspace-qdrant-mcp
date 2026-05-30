//! Hybrid / semantic / keyword search pipeline.
//!
//! Mirrors the orchestration in `search-helpers.ts` and `search-qdrant.ts`.
//!
//! ## Pipeline phases (search-helpers.ts finalizeResults:309-342)
//! 1. Fan-out per-collection search (dense + sparse legs in parallel per collection)
//! 2. RRF fusion (`apply_rrf_fusion`)
//! 3. Sort by score desc
//! 4. Optional source diversity re-ranking (when >1 collection)
//! 5. Slice to limit
//! 6. Optional parent context expansion
//! 7. Optional per-result graph context enrichment
//!
//! ## Score threshold
//! Applied ONLY at the Qdrant query level (per-leg), NOT after RRF fusion.
//! See scratchpad note "MCP-Rust task 30 search wiring: do NOT apply
//! post-fusion score threshold".

use std::collections::HashMap;

// rusqlite::Connection is no longer imported here — expansion keywords are
// pre-computed by the caller (search_tool) synchronously before any await,
// then passed as an owned Vec<String> so the future remains Send.

use crate::observability::metrics::record_daemon_fallback;
use crate::qdrant::client::{QdrantPoint, QdrantReadClient, QdrantRetrievedPoint};
use crate::qdrant::filters::{build_filter, determine_collections, FilterParams};
use crate::qdrant::fusion::{
    apply_rrf_fusion, diversify_results, TaggedResult, DEFAULT_DIVERSITY_CONFIG,
};
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::graph_context::{expand_graph_context, GraphQueryDaemon};
use super::options::{SearchOptions, DEFAULT_EXPANSION_WEIGHT};
use super::types::{SearchMode, SearchResponse, SearchResult, SearchScope};

pub use super::flow_collect::{
    build_provenance, expand_parent_context, search_collection, tagged_to_search_result,
};
pub use super::flow_fallback::{f001_refusal_reason, fallback_search, FALLBACK_STATUS_REASON};

// ---------------------------------------------------------------------------
// Dependency traits
// ---------------------------------------------------------------------------

/// Embedding generation and sparse vector — injectable for tests.
pub trait EmbedDaemon: Send + Sync {
    fn embed_text(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, tonic::Status>> + Send;

    fn generate_sparse_vector(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<HashMap<u32, f32>, tonic::Status>> + Send;
}

/// Qdrant read access — injectable for tests.
pub trait SearchQdrant: Send + Sync {
    fn search_dense(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantPoint>>> + Send;

    fn search_sparse(
        &self,
        collection: &str,
        indices: Vec<u32>,
        values: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantPoint>>> + Send;

    fn scroll_page(
        &self,
        collection: &str,
        filter: Option<qdrant_client::qdrant::Filter>,
        limit: u32,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantRetrievedPoint>>> + Send;

    fn retrieve_by_ids(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantRetrievedPoint>>> + Send;
}

// ---------------------------------------------------------------------------
// `QdrantReadClient` adapter
// ---------------------------------------------------------------------------

impl SearchQdrant for QdrantReadClient {
    fn search_dense(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantPoint>>> + Send {
        let collection = collection.to_string();
        let client = self.clone();
        async move {
            client
                .search(
                    &collection,
                    crate::qdrant::fusion::DENSE_VECTOR_NAME,
                    vector,
                    limit,
                    score_threshold,
                    filter,
                )
                .await
        }
    }

    fn search_sparse(
        &self,
        collection: &str,
        indices: Vec<u32>,
        values: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantPoint>>> + Send {
        let collection = collection.to_string();
        let client = self.clone();
        async move {
            client
                .search_sparse(
                    &collection,
                    crate::qdrant::fusion::SPARSE_VECTOR_NAME,
                    indices,
                    values,
                    limit,
                    score_threshold,
                    filter,
                )
                .await
        }
    }

    fn scroll_page(
        &self,
        collection: &str,
        filter: Option<qdrant_client::qdrant::Filter>,
        limit: u32,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantRetrievedPoint>>> + Send {
        let collection = collection.to_string();
        let client = self.clone();
        async move {
            let (points, _) = client.scroll(&collection, filter, limit, None).await?;
            Ok(points)
        }
    }

    fn retrieve_by_ids(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> impl std::future::Future<Output = anyhow::Result<Vec<QdrantRetrievedPoint>>> + Send {
        let collection = collection.to_string();
        let client = self.clone();
        async move { client.retrieve(&collection, ids).await }
    }
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

/// Run the full hybrid/semantic/keyword search pipeline.
///
/// Mirrors `searchAllCollections` + `finalizeResults` in `search-helpers.ts`.
/// `D` must implement both `EmbedDaemon` and `GraphQueryDaemon` to avoid
/// double-borrow when the same `DaemonClient` is used for both phases.
/// Run the hybrid / semantic / keyword search pipeline.
///
/// `expansion_keywords` replaces the former `conn: Option<&Connection>` to keep
/// the future `Send`.  Callers must pre-compute expansion keywords synchronously
/// (via [`super::expansion::collect_expansion_keywords`] while holding the
/// SQLite lock) and pass the result here.  An empty `Vec` is equivalent to
/// `None` — no tag-basket expansion is performed.
pub async fn run_search_pipeline<D, Q>(
    daemon: &mut D,
    qdrant: &Q,
    expansion_keywords: Vec<String>,
    opts: &SearchOptions,
    project_id: Option<&str>,
    enable_tag_expansion: bool,
) -> SearchResponse
where
    D: EmbedDaemon + GraphQueryDaemon,
    Q: SearchQdrant,
{
    let collections = determine_collections(
        opts.collection.as_deref(),
        opts.scope.as_str(),
        opts.include_libraries,
    );
    let scope = opts.scope;
    let mode = opts.mode;

    // Phase 1: Generate embeddings + optional tag expansion.
    let (dense_embedding, sparse_vector) = match embed_and_expand(
        daemon,
        opts,
        expansion_keywords,
        enable_tag_expansion,
        mode,
    )
    .await
    {
        Ok(pair) => pair,
        Err(_) => {
            record_daemon_fallback("search", "embed_failed");
            return fallback_search(qdrant, opts, &collections, project_id).await;
        }
    };

    // Phase 2: Fan-out per-collection search.
    let all_tagged = search_all_collections(
        qdrant,
        &collections,
        opts,
        mode,
        &dense_embedding,
        &sparse_vector,
        project_id,
    )
    .await;

    // Phases 3-6: Fuse, rank, diversify, slice, convert.
    let (mut results, diversity_score) = finalize_results(all_tagged, opts, mode, &collections);

    // Phases 7-8: Context enrichment.
    enrich_results(daemon, qdrant, opts, &mut results).await;

    build_response(results, opts, scope, mode, collections, diversity_score)
}

/// Phase 1: Generate dense + sparse embeddings, then optionally expand sparse
/// with tag-basket keywords (Phase 1b, search-expansion.ts:50-77).
async fn embed_and_expand<D>(
    daemon: &mut D,
    opts: &SearchOptions,
    expansion_keywords: Vec<String>,
    enable_tag_expansion: bool,
    mode: SearchMode,
) -> Result<(Option<Vec<f32>>, Option<HashMap<u32, f32>>), ()>
where
    D: EmbedDaemon,
{
    let (dense, mut sparse) = generate_embeddings(daemon, opts).await?;

    // Phase 1b: tag-basket sparse expansion.
    if enable_tag_expansion
        && sparse.is_some()
        && (mode == SearchMode::Hybrid || mode == SearchMode::Keyword)
        && !expansion_keywords.is_empty()
    {
        let sv = sparse.take().unwrap();
        if let Ok(exp_sv) = daemon
            .generate_sparse_vector(&expansion_keywords.join(" "))
            .await
        {
            sparse = Some(super::expansion::merge_sparse_vectors(
                &sv,
                &exp_sv,
                DEFAULT_EXPANSION_WEIGHT,
            ));
        } else {
            sparse = Some(sv);
        }
    }
    Ok((dense, sparse))
}

/// Phase 2: Fan-out search across all target collections.
///
/// Each call to `search_collection` swallows per-leg errors (TS parity).
async fn search_all_collections<Q>(
    qdrant: &Q,
    collections: &[String],
    opts: &SearchOptions,
    mode: SearchMode,
    dense: &Option<Vec<f32>>,
    sparse: &Option<HashMap<u32, f32>>,
    project_id: Option<&str>,
) -> Vec<TaggedResult>
where
    Q: SearchQdrant,
{
    let mut all_tagged: Vec<TaggedResult> = Vec::new();
    let search_limit = (opts.limit * 2) as u64;

    for coll in collections {
        let filter_params = search_filter_params(coll, opts, project_id);
        let filter = build_filter(&filter_params);
        let leg = search_collection(
            qdrant,
            coll,
            mode,
            dense.as_deref(),
            sparse.as_ref(),
            filter,
            search_limit,
            opts.score_threshold,
        )
        .await;
        all_tagged.extend(leg);
    }
    all_tagged
}

/// Phases 3–6: RRF fusion → sort → diversity → slice → convert to SearchResult.
fn finalize_results(
    all_tagged: Vec<TaggedResult>,
    opts: &SearchOptions,
    mode: SearchMode,
    collections: &[String],
) -> (Vec<SearchResult>, Option<f64>) {
    // Phase 3: RRF fusion (hybrid only) → sort by score desc.
    let fused = if mode == SearchMode::Hybrid {
        apply_rrf_fusion(&all_tagged)
    } else {
        all_tagged
    };
    let mut sorted = fused;
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase 4: Source diversity re-ranking (when >1 collection).
    let (diverse_results, diversity_score) = if opts.diverse && collections.len() > 1 {
        let (dr, ds) = diversify_results(sorted, &DEFAULT_DIVERSITY_CONFIG);
        (dr, Some(ds))
    } else {
        (sorted, None)
    };

    // Phase 5-6: Slice to limit and convert.
    let results: Vec<SearchResult> = diverse_results
        .into_iter()
        .take(opts.limit)
        .map(tagged_to_search_result)
        .collect();

    (results, diversity_score)
}

/// Phases 7–8: Parent context and graph context enrichment.
async fn enrich_results<D, Q>(
    daemon: &mut D,
    qdrant: &Q,
    opts: &SearchOptions,
    results: &mut Vec<SearchResult>,
) where
    D: EmbedDaemon + GraphQueryDaemon,
    Q: SearchQdrant,
{
    if opts.expand_context {
        expand_parent_context(qdrant, results).await;
    }
    if opts.include_graph_context {
        // DEFERRED (task 30 follow-up, GitHub #80): expandAndFuseWithGraph
        // (graph-expansion fusion pass before diversity re-ranking) is not yet
        // implemented.  Only expandGraphContext (post-slice per-result enrichment)
        // runs here.  TS executes `expandAndFuseWithGraph` BEFORE diversity +
        // slice (`finalizeResults` in search-helpers.ts:313-315), allowing
        // graph-expanded results to participate in diversity scoring.
        expand_graph_context(daemon, results).await;
    }
}

/// Assemble the final SearchResponse from pipeline outputs.
fn build_response(
    results: Vec<SearchResult>,
    opts: &SearchOptions,
    scope: SearchScope,
    mode: SearchMode,
    collections: Vec<String>,
    diversity_score: Option<f64>,
) -> SearchResponse {
    let total = results.len();
    let mut resp = SearchResponse {
        results,
        total,
        query: opts.query.clone(),
        mode,
        scope,
        collections_searched: collections,
        status: None,
        status_reason: None,
        branch: None,
        diversity_score,
    };
    if let Some(ref b) = opts.branch {
        resp.branch = Some(b.clone());
    }
    resp
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate dense and sparse embeddings from daemon.
///
/// Returns `Err(())` on any transport error → caller triggers fallback search.
async fn generate_embeddings<D>(
    daemon: &mut D,
    opts: &SearchOptions,
) -> Result<(Option<Vec<f32>>, Option<HashMap<u32, f32>>), ()>
where
    D: EmbedDaemon,
{
    let mut dense: Option<Vec<f32>> = None;
    let mut sparse: Option<HashMap<u32, f32>> = None;

    if opts.mode == SearchMode::Hybrid || opts.mode == SearchMode::Semantic {
        let v = daemon.embed_text(&opts.query).await.map_err(|_| ())?;
        if !v.is_empty() {
            dense = Some(v);
        }
    }
    if opts.mode == SearchMode::Hybrid || opts.mode == SearchMode::Keyword {
        let sv = daemon
            .generate_sparse_vector(&opts.query)
            .await
            .map_err(|_| ())?;
        if !sv.is_empty() {
            sparse = Some(sv);
        }
    }
    Ok((dense, sparse))
}

fn search_filter_params<'a>(
    collection: &'a str,
    opts: &'a SearchOptions,
    project_id: Option<&'a str>,
) -> FilterParams {
    FilterParams {
        collection: collection.to_string(),
        scope: opts.scope.as_str().to_string(),
        project_id: project_id.map(str::to_string),
        group_tenant_ids: None,
        branch: opts.branch.clone(),
        file_type: opts.file_type.clone(),
        library_name: if collection == COLLECTION_LIBRARIES {
            opts.library_name.clone()
        } else {
            None
        },
        library_path: if collection == COLLECTION_LIBRARIES {
            opts.library_path.clone()
        } else {
            None
        },
        tag: opts.tag.clone(),
        tags: opts.tags.clone(),
        path_glob: opts.path_glob.clone(),
        component: opts.component.clone(),
        base_points: None,
    }
}
