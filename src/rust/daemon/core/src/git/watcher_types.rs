use thiserror::Error;

/// Errors from git watcher operations
#[derive(Error, Debug)]
pub enum GitWatcherError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Not a git repository: {0}")]
    NotGitRepo(String),

    #[error("Git directory not found: {0}")]
    GitDirNotFound(String),
}

pub type GitWatcherResult<T> = Result<T, GitWatcherError>;

/// Types of git operations detected from reflog parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitEventType {
    /// Branch switch: `checkout: moving from X to Y`
    BranchSwitch,
    /// New commit: `commit:` or `commit (amend):`
    Commit,
    /// Merge commit: `merge`
    Merge,
    /// Pull (fetch+merge): `pull:`
    Pull,
    /// Rebase: `rebase`
    Rebase,
    /// Reset: `reset:`
    Reset,
    /// Stash operation
    Stash,
    /// Unknown operation
    Unknown,
}

impl GitEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GitEventType::BranchSwitch => "branch_switch",
            GitEventType::Commit => "commit",
            GitEventType::Merge => "merge",
            GitEventType::Pull => "pull",
            GitEventType::Rebase => "rebase",
            GitEventType::Reset => "reset",
            GitEventType::Stash => "stash",
            GitEventType::Unknown => "unknown",
        }
    }
}

/// A git event detected by the watcher
#[derive(Debug, Clone)]
pub struct GitEvent {
    /// Watch folder ID this event belongs to
    pub watch_folder_id: String,
    /// Type of git operation
    pub event_type: GitEventType,
    /// Old commit SHA (before the operation)
    pub old_sha: String,
    /// New commit SHA (after the operation)
    pub new_sha: String,
    /// Current branch name (read from HEAD)
    pub branch: Option<String>,
    /// Previous branch name (for branch switches)
    pub old_branch: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_event_type_as_str() {
        assert_eq!(GitEventType::BranchSwitch.as_str(), "branch_switch");
        assert_eq!(GitEventType::Commit.as_str(), "commit");
        assert_eq!(GitEventType::Merge.as_str(), "merge");
        assert_eq!(GitEventType::Pull.as_str(), "pull");
        assert_eq!(GitEventType::Rebase.as_str(), "rebase");
        assert_eq!(GitEventType::Reset.as_str(), "reset");
        assert_eq!(GitEventType::Stash.as_str(), "stash");
        assert_eq!(GitEventType::Unknown.as_str(), "unknown");
    }
}
