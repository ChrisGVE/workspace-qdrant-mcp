//! Scope adapter: re-exports the SQLite-free scope surface.
//!
//! The scope helpers (group-tenant filtering, decay-map building, relevance
//! decay, `ScopeContext`) live in the shared `wqm-client` crate
//! (`wqm_client::search::scope`, WI-d4 #82). This module re-exports them so
//! existing `crate::tools::search::scope::…` paths keep resolving.
//!
//! Project isolation for `scope=project` is the tenant-id filter alone
//! (`build_project_condition` in `wqm_client::qdrant::filters`). The former
//! per-file `base_point` "worktree isolation" path was removed (#115): the
//! daemon populates `tracked_files.base_point` with a content-addressed hash
//! for cross-branch dedup, not a worktree root, so the TS-ported isolation read
//! a hash as a path — it matched nothing and degraded recall on any project
//! with more than the 500-file cap. Multi-clone repos are already isolated by
//! distinct tenant ids, and worktrees by the branch filter.

// Re-export the SQLite-free scope surface from the shared client so consumers
// (search_tool, the pipeline, tests) reach a single definition.
pub use wqm_client::search::scope::{
    apply_relevance_decay, scope_filter_from_response, ScopeContext, GROUP_EMPTY_REFUSAL,
};
