//! Search tool response types.
//!
//! These neutral result models now live in the shared `wqm-client` crate
//! (`wqm_client::models`, WI-d6 #82) so both clients share one shape. This
//! module re-exports them so existing `crate::tools::search::types::…` paths
//! keep resolving; the renderers (CLI output, MCP `envelope.rs`) stay
//! per-component. The serialisation contract (TS field order, omit-if-none) is
//! preserved by the moved definitions.

pub use wqm_client::models::{
    GraphContext, GraphContextNode, ParentContext, Provenance, SearchMode, SearchResponse,
    SearchResult, SearchScope,
};
