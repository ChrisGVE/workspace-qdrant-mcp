//! Branch-level write operations — delete + truth-table probe (arch §4.3, F9).
//!
//! File: `wqm-storage-write/src/branch/mod.rs`
//! Location: `src/rust/storage-write/src/branch/` (write-crate branch layer)
//! Context: This module houses the whole-branch delete path (arch §4.3, AC-F9.3)
//!   and the git2-backed deletion truth-table probe (AC-F9.1 / DR GP-3). The
//!   single-file delete variant lives next door in [`crate::blob::file_delete`]
//!   (AC-F9.9 line-budget split along responsibility).
//!
//!   Three-file split (AC-F9.9 line-budget compliance, arch §9):
//!     - [`probe`]  — GP-4 truth-table types + git2 probe (`DeleteAction`, `GitBranchProbe`,
//!                    `delete_decision`, `probe_branch`). Pure + unit-testable sans git repo.
//!     - [`steps`]  — SQL step helpers `step1`–`step8` + `BlobCandidate`. All SQL mutations.
//!     - [`delete`] — Thin orchestrator: `branch_delete` public async fn + test module.
//!
//!   The `branch_cleanup/` module in `daemon/core` is NOT retired here: the daemon
//!   cutover that wires `wqm-storage-write` into `memexd` is a separate deferred
//!   task (#175). Until then, `branch_cleanup/` continues to serve the daemon; this
//!   module is the replacement that takes over once the cutover lands.
//!
//! Neighbors: [`crate::blob::gc`] (orphan GC by refcount), [`crate::blob::file_delete`]
//!   (single-file delete variant), [`crate::qdrant::membership_batch`] (batched
//!   survivor PUT flush outside the lock, F19 strategy).

pub mod delete;
pub mod onboard;
pub(crate) mod probe;
pub(crate) mod steps;

pub use delete::{branch_delete, delete_decision, DeleteAction, GitBranchProbe};
pub use onboard::{
    apply_git_diff, branch_onboard, resume_pending_onboards, FileContentProvider, OnboardConfig,
    PendingBranch, PendingDiffProvider,
};
pub use probe::probe_branch;
