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

// ---------------------------------------------------------------------------
// file_uri_to_path (#97)
// ---------------------------------------------------------------------------

#[test]
fn file_uri_plain_absolute_path() {
    assert_eq!(
        super::file_uri_to_path("file:///Users/x/proj"),
        Some(std::path::PathBuf::from("/Users/x/proj"))
    );
}

#[test]
fn file_uri_percent_encoded_space() {
    assert_eq!(
        super::file_uri_to_path("file:///Users/x/My%20Project"),
        Some(std::path::PathBuf::from("/Users/x/My Project"))
    );
}

#[test]
fn file_uri_with_authority_host() {
    assert_eq!(
        super::file_uri_to_path("file://localhost/var/data"),
        Some(std::path::PathBuf::from("/var/data"))
    );
}

#[test]
fn file_uri_rejects_non_file_schemes() {
    assert_eq!(super::file_uri_to_path("https://example.com/x"), None);
    assert_eq!(super::file_uri_to_path("vscode-remote://wsl/home"), None);
}

#[test]
fn file_uri_invalid_escape_passes_through() {
    assert_eq!(
        super::file_uri_to_path("file:///a/%zz"),
        Some(std::path::PathBuf::from("/a/%zz"))
    );
}

// ---------------------------------------------------------------------------
// rebind_session_project (#97)
// ---------------------------------------------------------------------------

/// Launch-cwd detection found nothing; the client root resolves to a
/// registered project → session must be re-bound and registered.
#[tokio::test]
async fn rebind_binds_project_when_initial_detection_missed() {
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().join("conv-root");
    std::fs::create_dir_all(&root).unwrap();

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
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_ROOT', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let register_calls = Arc::new(AtomicU32::new(0));
    let state = Arc::new(SharedStateManager::new(StateManager::open_at(&db_path)));
    let session = Arc::new(Mutex::new(SessionState::new()));
    {
        // Simulate a completed initialize that found NO project but did
        // connect the daemon.
        let mut s = session.lock().await;
        s.initialized = true;
        s.daemon_connected = true;
    }
    let daemon = Arc::new(Mutex::new(CountingDaemon {
        register_calls: Arc::clone(&register_calls),
        heartbeat_calls: Arc::new(AtomicU32::new(0)),
    }));

    super::rebind_session_project(&state, &session, &daemon, root.clone()).await;

    let s = session.lock().await;
    assert_eq!(s.client_cwd.as_deref(), Some(root.as_path()));
    assert_eq!(
        s.project_id.as_deref(),
        Some("T_ROOT"),
        "rebind must adopt the root-detected project"
    );
    assert_eq!(s.project_path.as_deref(), Some(root.as_path()));
    assert_eq!(
        register_calls.load(Ordering::SeqCst),
        1,
        "rebind must register the newly detected project"
    );
}

/// The client root resolves to NO registered project → existing binding kept,
/// client_cwd still recorded.
#[tokio::test]
async fn rebind_keeps_existing_binding_when_root_unknown() {
    let dir = tempfile::TempDir::new().unwrap();
    let unknown_root = dir.path().join("unknown");
    std::fs::create_dir_all(&unknown_root).unwrap();

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
    drop(conn);

    let register_calls = Arc::new(AtomicU32::new(0));
    let state = Arc::new(SharedStateManager::new(StateManager::open_at(&db_path)));
    let session = Arc::new(Mutex::new(SessionState::new()));
    {
        let mut s = session.lock().await;
        s.initialized = true;
        s.daemon_connected = true;
        s.project_id = Some("T_EXISTING".to_string());
        s.project_path = Some(std::path::PathBuf::from("/elsewhere"));
    }
    let daemon = Arc::new(Mutex::new(CountingDaemon {
        register_calls: Arc::clone(&register_calls),
        heartbeat_calls: Arc::new(AtomicU32::new(0)),
    }));

    super::rebind_session_project(&state, &session, &daemon, unknown_root.clone()).await;

    let s = session.lock().await;
    assert_eq!(
        s.project_id.as_deref(),
        Some("T_EXISTING"),
        "unknown root must not clear an existing binding"
    );
    assert_eq!(s.client_cwd.as_deref(), Some(unknown_root.as_path()));
    assert_eq!(register_calls.load(Ordering::SeqCst), 0);
}

/// Root resolves to the SAME project the session already has → no re-register.
#[tokio::test]
async fn rebind_same_project_is_noop() {
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().join("same");
    std::fs::create_dir_all(&root).unwrap();

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
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_SAME', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let register_calls = Arc::new(AtomicU32::new(0));
    let state = Arc::new(SharedStateManager::new(StateManager::open_at(&db_path)));
    let session = Arc::new(Mutex::new(SessionState::new()));
    {
        let mut s = session.lock().await;
        s.initialized = true;
        s.daemon_connected = true;
        s.project_id = Some("T_SAME".to_string());
        s.project_path = Some(root.clone());
    }
    let daemon = Arc::new(Mutex::new(CountingDaemon {
        register_calls: Arc::clone(&register_calls),
        heartbeat_calls: Arc::new(AtomicU32::new(0)),
    }));

    super::rebind_session_project(&state, &session, &daemon, root.clone()).await;

    let s = session.lock().await;
    assert_eq!(s.project_id.as_deref(), Some("T_SAME"));
    assert_eq!(
        register_calls.load(Ordering::SeqCst),
        0,
        "identical project must not re-register"
    );
}
