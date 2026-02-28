mod types;
mod branch_detector;
mod branch_lifecycle;
mod watcher_types;
mod watcher;
mod reflog;
mod diff_tree;

// Re-export from types
pub use types::{GitError, GitResult, GitStatus, CacheStats, detect_git_status};

// Re-export from branch_detector
pub use branch_detector::GitBranchDetector;

// Re-export from branch_lifecycle
pub use branch_lifecycle::{
    BranchEvent, BranchLifecycleConfig, BranchLifecycleDetector, BranchLifecycleStats,
    BranchEventHandler, branch_schema,
};

// Re-export from watcher_types
pub use watcher_types::{GitWatcherError, GitWatcherResult, GitEventType, GitEvent};

// Re-export from watcher
pub use watcher::GitWatcher;

// Re-export from reflog
pub use reflog::{
    resolve_git_dir, resolve_common_dir, read_current_branch,
    parse_reflog_last_entry, parse_reflog_line,
};

// Re-export from diff_tree
pub use diff_tree::{
    FileChangeStatus, FileChange, diff_tree, parse_diff_tree_output,
    get_blob_hash, ls_tree_submodules,
};
