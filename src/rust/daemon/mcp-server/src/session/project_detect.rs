//! Project detection: git root, branch, remote URL, and project ID.
//!
//! The pure filesystem detection + the [`ProjectInfo`] type + the
//! [`ProjectLookup`] trait now live in the shared `wqm-client` crate
//! (`wqm_client::project`, WI-d5 #82). This module re-exports them so existing
//! `crate::session::project_detect::…` paths keep resolving, and supplies the
//! MCP-side glue: the SQLite registered-project lookup ([`ProjectLookup`] impl
//! for [`StateManager`]) and the `&StateManager`-taking wrappers callers use.
//!
//! Keeping the SQLite read here preserves the single-writer/SQLite-ownership
//! boundary — wqm-client stays SQLite-free.

use std::path::Path;

use crate::sqlite::manager::StateManager;
use wqm_client::project::ProjectLookup;

pub use wqm_client::project::{
    detect_branch, find_git_root, find_project_root, get_git_remote_url, ProjectInfo,
};

/// SQLite-backed registered-project lookup (the read the daemon owns).
///
/// Uses the `watch_folders` table (collection = 'projects'), longest-prefix
/// match on `path`. Returns `None` when the database is degraded or no match
/// is found.
impl ProjectLookup for StateManager {
    fn lookup_project_id(&self, path: &Path) -> Option<String> {
        let conn = self.connection()?;
        let path_str = path.to_str()?;

        let sql = "SELECT tenant_id FROM watch_folders \
                   WHERE collection = 'projects' \
                     AND (? = path OR ? LIKE path || '/' || '%') \
                   ORDER BY length(path) DESC \
                   LIMIT 1";

        conn.query_row(sql, rusqlite::params![path_str, path_str], |row| {
            row.get::<_, String>(0)
        })
        .ok()
    }
}

/// Detect the project for `cwd`, resolving the registered project id via the
/// daemon's SQLite. Thin wrapper over [`wqm_client::project::detect_project`].
///
/// Mirrors `detectProjectForSession` + `ProjectDetector.getProjectInfo`.
pub fn detect_project(cwd: &Path, state_manager: &StateManager) -> Option<ProjectInfo> {
    wqm_client::project::detect_project(cwd, state_manager)
}

/// Look up the registered project ID for a path from the daemon's SQLite.
///
/// Exposed so the search tool can resolve a tenant from cwd without the
/// git-remote / branch work that [`detect_project`] also performs (GitHub #83).
pub fn lookup_project_id(state_manager: &StateManager, project_root: &Path) -> Option<String> {
    state_manager.lookup_project_id(project_root)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "project_detect_tests.rs"]
mod tests;
