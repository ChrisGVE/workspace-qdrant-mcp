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
use crate::qdrant::fusion::TaggedResult;
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};

use super::graph_context::{expand_graph_context, GraphQueryDaemon};
use super::options::{SearchOptions, DEFAULT_EXPANSION_WEIGHT};
use super::types::{SearchMode, SearchResponse, SearchResult, SearchScope};

pub use super::flow_collect::{
    build_provenance, diversify_slice_convert, expand_parent_context, fuse_and_sort,
    search_collection, tagged_to_search_result,
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
    scope_ctx: &super::scope::ScopeContext,
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
            return fallback_search(qdrant, opts, &collections, project_id, scope_ctx).await;
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
        scope_ctx,
    )
    .await;

    // Phase 3: relevance decay → RRF fusion → sort by score.
    let mut fused = fuse_and_sort(all_tagged, mode, scope_ctx);

    // Phase 3b: graph-expansion fusion (GitHub #80). Mirrors TS
    // `expandAndFuseWithGraph` at `finalizeResults` (search-helpers.ts:313-316) —
    // runs BEFORE diversity so graph-expanded nodes participate in diversity
    // scoring and the slice. Primary collection = first searched (TS `[0] ?? 'projects'`).
    if opts.include_graph_context {
        let primary = collections
            .first()
            .map(String::as_str)
            .unwrap_or(COLLECTION_PROJECTS);
        super::graph_fusion::expand_and_fuse_with_graph(daemon, &mut fused, primary).await;
    }

    // Phases 4-6: diversify, slice, convert.
    let (mut results, diversity_score) = diversify_slice_convert(fused, opts, &collections);

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
    scope_ctx: &super::scope::ScopeContext,
) -> Vec<TaggedResult>
where
    Q: SearchQdrant,
{
    let mut all_tagged: Vec<TaggedResult> = Vec::new();
    let search_limit = (opts.limit * 2) as u64;

    for coll in collections {
        let filter_params = search_filter_params(coll, opts, project_id, scope_ctx);
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
        // Post-slice per-result caller/callee enrichment (TS `expandGraphContext`,
        // search-helpers.ts:333). The pre-diversity graph-expansion fusion pass
        // (`expandAndFuseWithGraph`, GitHub #80) runs earlier in `run_search_pipeline`.
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
    scope_ctx: &'a super::scope::ScopeContext,
) -> FilterParams {
    FilterParams {
        collection: collection.to_string(),
        scope: opts.scope.as_str().to_string(),
        project_id: project_id.map(str::to_string),
        group_tenant_ids: scope_ctx.group_tenant_ids.clone(),
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
        // Base points only constrain the projects collection (TS
        // search-helpers.ts:216 `coll === PROJECTS_COLLECTION ? basePoints : undefined`);
        // threading them into libraries/scratchpad/rules would wrongly suppress hits.
        base_points: if collection == COLLECTION_PROJECTS {
            scope_ctx.base_points.clone()
        } else {
            None
        },
    }
}
