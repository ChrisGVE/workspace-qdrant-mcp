mod branch_detector;
mod branch_lifecycle;
mod diff_tree;
mod reflog;
mod types;
mod watcher;
mod watcher_types;

// Re-export from types
pub use types::{detect_git_status, CacheStats, GitError, GitResult, GitStatus};

// Re-export from branch_detector
pub use branch_detector::GitBranchDetector;

// Re-export from branch_lifecycle
pub use branch_lifecycle::{
    branch_schema, BranchEvent, BranchEventHandler, BranchLifecycleConfig, BranchLifecycleDetector,
    BranchLifecycleStats,
};

// Re-export from watcher_types
pub use watcher_types::{GitEvent, GitEventType, GitWatcherError, GitWatcherResult};

// Re-export from watcher
pub use watcher::GitWatcher;

// Re-export from reflog
pub use reflog::{
    parse_reflog_last_entry, parse_reflog_line, read_current_branch, resolve_common_dir,
    resolve_git_dir,
};

// Re-export from diff_tree
pub use diff_tree::{
    diff_tree, get_blob_hash, ls_tree_submodules, parse_diff_tree_output, FileChange,
    FileChangeStatus,
};
