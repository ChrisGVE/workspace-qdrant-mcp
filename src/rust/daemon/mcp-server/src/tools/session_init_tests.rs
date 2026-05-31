//! Hermetic test for `run_session_initialize` (the rmcp `initialize` glue).

use std::sync::Arc;

use tokio::sync::Mutex;

use super::run_session_initialize;
use crate::server_types::SessionState;
use crate::session::{DaemonOps, RegisterResponse};
use crate::sqlite::{SharedStateManager, StateManager};

/// Minimal `DaemonOps` mock: health/register/heartbeat all succeed.
struct OkDaemon;

impl DaemonOps for OkDaemon {
    async fn health(&mut self) -> Result<(), String> {
        Ok(())
    }
    async fn register_project(
        &mut self,
        _path: &str,
        _project_id: &str,
        _name: &str,
        _git_remote: Option<&str>,
    ) -> Result<RegisterResponse, String> {
        Ok(RegisterResponse {
            project_id: "T_CWD".to_string(),
            is_worktree: false,
            watch_path: None,
            is_active: true,
            created: false,
        })
    }
    async fn heartbeat(&mut self, _project_id: &str) -> Result<bool, String> {
        Ok(true)
    }
    async fn deprioritize_project(
        &mut self,
        _project_id: &str,
        _watch_path: Option<&str>,
    ) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn run_session_initialize_detects_registers_and_starts_heartbeat() {
    // Temp project root registered in a temp state.db; cwd is the root.
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("Cargo.toml"), b"[package]\n").unwrap();

    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (
             tenant_id TEXT NOT NULL,
             path TEXT NOT NULL,
             collection TEXT NOT NULL DEFAULT 'projects'
         )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_CWD', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let state = Arc::new(SharedStateManager::new(StateManager::open_at(&db_path)));
    let session = Arc::new(Mutex::new(SessionState::new()));
    let daemon = Arc::new(Mutex::new(OkDaemon));
    let hb_slot = Arc::new(std::sync::Mutex::new(None));

    run_session_initialize(&state, &session, &daemon, &hb_slot, &root).await;

    {
        let s = session.lock().await;
        assert!(s.initialized, "session must be marked initialized");
        assert!(
            s.daemon_connected,
            "daemon_connected must be true (health ok)"
        );
        assert_eq!(
            s.project_id.as_deref(),
            Some("T_CWD"),
            "project_id must be the cwd-detected tenant"
        );
        assert_eq!(
            s.project_path.as_deref(),
            Some(root.as_path()),
            "project_path must be the detected project root"
        );
    }

    // A heartbeat task must have been started and its handle stored.
    let handle = hb_slot.lock().unwrap().take();
    assert!(handle.is_some(), "heartbeat AbortHandle must be stored");
    handle.unwrap().abort(); // stop the spawned task

    // Idempotency: a second call must be a no-op (no re-detect / re-register).
    run_session_initialize(&state, &session, &daemon, &hb_slot, &root).await;
    assert!(
        hb_slot.lock().unwrap().is_none(),
        "second initialize must not start another heartbeat"
    );
}
