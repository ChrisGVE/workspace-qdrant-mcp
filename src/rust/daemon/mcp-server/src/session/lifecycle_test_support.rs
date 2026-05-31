//! Shared test doubles + helpers for the `session::lifecycle` tests.
//!
//! Extracted into its own sibling module so `lifecycle_tests.rs` stays under
//! the 500-line limit. Both `lifecycle_tests.rs` and `lifecycle_tests_part2.rs`
//! reach these via `use super::lifecycle_test_support::*`.

use std::path::PathBuf;

use crate::session::project_detect::ProjectInfo;

use super::{DaemonOps, RegisterResponse};

// ─────────────────────────────────────────────────────────────────────────────
// MockDaemonOps — test double
// ─────────────────────────────────────────────────────────────────────────────

/// Test double for [`DaemonOps`].
///
/// Configure `health_ok`, `register_response`, and `heartbeat_ok` before
/// the test.  Call counts are recorded in `health_calls`, `register_calls`,
/// `heartbeat_calls`, and `deprioritize_calls`.
#[derive(Debug)]
pub(crate) struct MockDaemonOps {
    pub health_ok: bool,
    pub register_response: Option<RegisterResponse>,
    pub heartbeat_ok: bool,
    pub deprioritize_ok: bool,
    pub health_calls: u32,
    pub register_calls: u32,
    pub heartbeat_calls: u32,
    pub deprioritize_calls: u32,
}

impl MockDaemonOps {
    pub fn new() -> Self {
        Self {
            health_ok: true,
            register_response: Some(RegisterResponse {
                project_id: "mock-proj-id".to_string(),
                is_worktree: false,
                watch_path: None,
                is_active: true,
                created: false,
            }),
            heartbeat_ok: true,
            deprioritize_ok: true,
            health_calls: 0,
            register_calls: 0,
            heartbeat_calls: 0,
            deprioritize_calls: 0,
        }
    }
}

impl DaemonOps for MockDaemonOps {
    async fn health(&mut self) -> Result<(), String> {
        self.health_calls += 1;
        if self.health_ok {
            Ok(())
        } else {
            Err("mock: health failed".to_string())
        }
    }

    async fn register_project(
        &mut self,
        _path: &str,
        _project_id: &str,
        _name: &str,
        _git_remote: Option<&str>,
    ) -> Result<RegisterResponse, String> {
        self.register_calls += 1;
        self.register_response
            .clone()
            .ok_or_else(|| "mock: register failed".to_string())
    }

    async fn heartbeat(&mut self, _project_id: &str) -> Result<bool, String> {
        self.heartbeat_calls += 1;
        if self.heartbeat_ok {
            Ok(true)
        } else {
            Err("mock: heartbeat failed".to_string())
        }
    }

    async fn deprioritize_project(
        &mut self,
        _project_id: &str,
        _watch_path: Option<&str>,
    ) -> Result<(), String> {
        self.deprioritize_calls += 1;
        if self.deprioritize_ok {
            Ok(())
        } else {
            Err("mock: deprioritize failed".to_string())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-detected "no project" result — simulates "no project detected".
pub(crate) fn no_project_detect() -> Option<ProjectInfo> {
    None
}

/// Pre-detected `ProjectInfo` (branch fixed to "main").
pub(crate) fn fixed_detect(
    project_path: PathBuf,
    project_id: Option<String>,
) -> Option<ProjectInfo> {
    Some(ProjectInfo {
        project_path,
        project_id,
        git_remote: None,
        branch: "main".to_string(),
    })
}
