//! Git-derived value types shared across the storage crates and daemon-core.
//!
//! Location: `wqm-common/src/git/` — leaf crate, no git2 dependency. Context:
//! F0 canonical home for the git diff-tree value types (`FileChange` /
//! `FileChangeStatus`) that arch §4.6's `apply_git_diff` and F8 consume,
//! relocated from `daemon-core/git/diff_tree.rs` so the write crate
//! (`wqm-storage-write`) shares ONE definition (FP-2 / DR GP-9). The git2-backed
//! `diff_tree()` producer STAYS in daemon-core — git access is write/daemon-side
//! only (no-shell-git is enforced there). Neighbors: `file_change`.

pub mod file_change;
