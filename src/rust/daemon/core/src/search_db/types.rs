//! Types, constants, and helper functions for the search database.

use std::path::{Path, PathBuf};
use thiserror::Error;

/// Current schema version for search.db
pub const SEARCH_SCHEMA_VERSION: i32 = 6;

/// Default search database filename
pub const SEARCH_DB_FILENAME: &str = "search.db";

/// Errors from search database operations
#[derive(Error, Debug)]
pub enum SearchDbError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Schema migration error: {0}")]
    Migration(String),

    #[error("Downgrade not supported: database version {db_version} > code version {code_version}")]
    DowngradeNotSupported { db_version: i32, code_version: i32 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid search pattern: {0}")]
    InvalidPattern(String),
}

/// Result type for search database operations
pub type SearchDbResult<T> = Result<T, SearchDbError>;

/// Result of a code line insertion.
#[derive(Debug, Clone)]
pub struct InsertedLine {
    pub line_id: i64,
    pub seq: f64,
}

/// Result of a rebalance operation.
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub lines_rebalanced: usize,
    pub new_gap: f64,
}

/// Derive the search.db path from the state.db path.
///
/// Given `~/.workspace-qdrant/state.db`, returns `~/.workspace-qdrant/search.db`.
pub fn search_db_path_from_state(state_db_path: &Path) -> PathBuf {
    let parent = state_db_path.parent().unwrap_or(Path::new("."));
    if parent.as_os_str().is_empty() {
        PathBuf::from(SEARCH_DB_FILENAME)
    } else {
        parent.join(SEARCH_DB_FILENAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_db_path_from_state() {
        let state_path = PathBuf::from("/home/user/.workspace-qdrant/state.db");
        let search_path = search_db_path_from_state(&state_path);
        assert_eq!(
            search_path,
            PathBuf::from("/home/user/.workspace-qdrant/search.db")
        );
    }

    #[test]
    fn test_search_db_path_from_state_relative() {
        let state_path = PathBuf::from("state.db");
        let search_path = search_db_path_from_state(&state_path);
        assert_eq!(search_path, PathBuf::from("search.db"));
    }
}
