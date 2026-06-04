//! Shared SQLite-free search pipeline (WI-d4, #82).
//!
//! This module owns the hybrid / semantic / keyword / exact search pipeline that
//! both wqm clients run against the daemon + Qdrant. It is deliberately free of
//! SQLite, session, and Prometheus dependencies: the SQLite-bound parts
//! (base-point resolution, tag-basket keyword collection) and the concrete
//! metric backend stay in the consuming crate (the MCP server), which
//! pre-resolves owned values and threads them in.
//!
//! ## Module layout
//! - `options`       — `SearchInput`, `SearchOptions`, defaults
//! - `flow`          — hybrid/semantic/keyword pipeline (`run_search_pipeline`)
//! - `flow_collect`  — per-collection legs + fusion/finalize phases
//! - `flow_fallback` — daemon-down scroll fallback + F-001 refusal
//! - `exact`         — FTS5 exact search via daemon TextSearchService
//! - `graph_context` — 1-hop graph context enrichment via daemon GraphService
//! - `graph_fusion`  — graph-augmented RAG expansion + score fusion (#80)
//! - `scope`         — SQLite-free scope resolution (decay, base-point filter)
//! - `expansion`     — SQLite-free sparse-vector merge
//! - `metrics`       — `FallbackMetrics` hook trait

pub mod exact;
pub mod expansion;
pub mod flow;
pub mod flow_collect;
pub mod flow_fallback;
pub mod graph_context;
pub mod graph_fusion;
pub mod metrics;
pub mod options;
pub mod scope;

pub use exact::{search_exact, ExactSearchDaemon};
pub use flow::{run_search_pipeline, EmbedDaemon, SearchQdrant};
pub use flow_fallback::{f001_refusal_reason, fallback_search, FALLBACK_STATUS_REASON};
pub use graph_context::{expand_graph_context, GraphQueryDaemon};
pub use metrics::FallbackMetrics;
pub use options::{SearchInput, SearchOptions, DEFAULT_SCORE_THRESHOLD};
pub use scope::ScopeContext;

use std::collections::HashMap;

use crate::grpc::client::DaemonClient;
use crate::workspace_daemon::{
    QueryRelatedRequest, QueryRelatedResponse, TextSearchRequest, TextSearchResponse,
};

// ---------------------------------------------------------------------------
// DaemonClient adapter impls
// ---------------------------------------------------------------------------
//
// The pipeline trait bounds (`EmbedDaemon`, `ExactSearchDaemon`,
// `GraphQueryDaemon`) are satisfied by the shared `DaemonClient`. Both the
// traits and `DaemonClient` live in this crate, so these blanket adapter impls
// must live here too (orphan rule). They are thin wrappers over the typed RPC
// methods — no MCP-private types involved.

impl EmbedDaemon for DaemonClient {
    fn embed_text(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, tonic::Status>> + Send {
        let text = text.to_string();
        async move {
            let resp = DaemonClient::embed_text(self, &text).await?;
            Ok(resp.embedding)
        }
    }

    fn generate_sparse_vector(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<HashMap<u32, f32>, tonic::Status>> + Send {
        let text = text.to_string();
        async move {
            let resp = DaemonClient::generate_sparse_vector(self, &text).await?;
            Ok(resp.indices_values)
        }
    }
}

impl ExactSearchDaemon for DaemonClient {
    fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> impl std::future::Future<Output = Result<TextSearchResponse, tonic::Status>> + Send {
        DaemonClient::text_search(self, request)
    }
}

impl GraphQueryDaemon for DaemonClient {
    fn query_related(
        &mut self,
        request: QueryRelatedRequest,
    ) -> impl std::future::Future<Output = Result<QueryRelatedResponse, tonic::Status>> + Send {
        DaemonClient::query_related(self, request)
    }
}
