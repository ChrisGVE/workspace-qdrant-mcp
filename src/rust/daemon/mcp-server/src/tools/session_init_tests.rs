//! Hermetic test for `run_session_initialize` (the rmcp `initialize` glue).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;

use super::run_session_initialize;
use crate::server_types::SessionState;
use crate::session::{DaemonOps, RegisterResponse};
use crate::sqlite::{SharedStateManager, StateManager};

/// `DaemonOps` mock with call counters. `register_project` returns a tenant id
/// DISTINCT from the registry-detected one so the test can tell which source
/// `project_id` came from.
struct CountingDaemon {
    register_calls: Arc<AtomicU32>,
    heartbeat_calls: Arc<AtomicU32>,
}

impl DaemonOps for CountingDaemon {
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
        self.register_calls.fetch_add(1, Ordering::SeqCst);
        Ok(RegisterResponse {
            project_id: "T_REGISTERED".to_string(), // distinct from detected
            is_worktree: false,
            watch_path: None,
            is_active: true,
            created: false,
        })
    }
    async fn heartbeat(&mut self, _project_id: &str) -> Result<bool, String> {
        self.heartbeat_calls.fetch_add(1, Ordering::SeqCst);
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
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_DETECTED', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let register_calls = Arc::new(AtomicU32::new(0));
    let heartbeat_calls = Arc::new(AtomicU32::new(0));

    let state = Arc::new(SharedStateManager::new(StateManager::open_at(&db_path)));
    let session = Arc::new(Mutex::new(SessionState::new()));
    let daemon = Arc::new(Mutex::new(CountingDaemon {
        register_calls: Arc::clone(&register_calls),
        heartbeat_calls: Arc::clone(&heartbeat_calls),
    }));
    let hb_slot = Arc::new(std::sync::Mutex::new(None));

    run_session_initialize(&state, &session, &daemon, &hb_slot, &root).await;

    // Registration must have been invoked exactly once.
    assert_eq!(
        register_calls.load(Ordering::SeqCst),
        1,
        "register_project must be invoked once"
    );

    {
        let s = session.lock().await;
        assert!(s.initialized, "session must be marked initialized");
        assert!(
            s.daemon_connected,
            "daemon_connected must be true (health ok)"
        );
        // project_id comes from REGISTRY DETECTION ("T_DETECTED"), not the
        // daemon register response ("T_REGISTERED") — apply_registration_response
        // only fills project_id when detection left it empty. This proves the
        // cwd registry lookup actually ran.
        assert_eq!(
            s.project_id.as_deref(),
            Some("T_DETECTED"),
            "project_id must be the cwd-detected registry tenant"
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

    // Idempotency: a second call must be a complete no-op (no re-register, no
    // new heartbeat).
    run_session_initialize(&state, &session, &daemon, &hb_slot, &root).await;
    assert_eq!(
        register_calls.load(Ordering::SeqCst),
        1,
        "second initialize must not re-register"
    );
    assert!(
        hb_slot.lock().unwrap().is_none(),
        "second initialize must not start another heartbeat"
    );
}
