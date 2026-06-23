//! Git diff-tree file-change value types.
//!
//! Location: `wqm-common/src/git/file_change.rs`. Context: canonical home (F0)
//! of `FileChange` and `FileChangeStatus`, the per-entry result of a git
//! diff-tree — relocated verbatim from `daemon-core/git/diff_tree.rs` so the
//! write crate's incremental ingest (F8) and daemon-core's `apply_git_diff`
//! (arch §4.6) share ONE definition (FP-2). Daemon-core re-exports both from
//! `crate::git` so existing call sites are unchanged.
//!
//! NOTE: `fts_batch_processor::FileChange` is a DIFFERENT, unrelated type that
//! merely shares this name (an FTS batch-apply payload — `file_id`,
//! `old_content`, `new_content`, …, with zero field overlap). It is NOT this
//! nexus and is intentionally left in place; see the F0 commit rationale.

/// File change status from git diff-tree output
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileChangeStatus {
    /// Modified file (M)
    Modified,
    /// Added file (A)
    Added,
    /// Deleted file (D)
    Deleted,
    /// Renamed file (R) with similarity percentage
    Renamed { old_path: String, similarity: u8 },
    /// Copied file (C) with similarity percentage
    Copied { src_path: String, similarity: u8 },
    /// Type changed (T) -- e.g., file became symlink
    TypeChanged,
}

/// A single file change from git diff-tree
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Change status (modified, added, deleted, etc.)
    pub status: FileChangeStatus,
    /// Path relative to repo root
    pub path: String,
}
