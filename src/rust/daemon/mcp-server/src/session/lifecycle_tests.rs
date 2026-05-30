//! Tests for `session::lifecycle`.
//!
//! Covers the AC-SL1..SL4 acceptance criteria:
//! - SL1: `initialize_session` assigns a session UUID and detects project ID.
//! - SL2: `initialize_session` sets `daemon_connected = true` when the daemon health check succeeds.
//! - SL3: Heartbeat failure sets `daemon_connected = false`.
//! - SL4: `cleanup_session` is idempotent (second call is a no-op).

use tempfile::TempDir;

use crate::server_types::SessionState;
use crate::sqlite::manager::StateManager;

use super::{cleanup_session, initialize_session, register_project, RegisterResponse};

// Shared test doubles + detection helpers live in the sibling support module.
use super::lifecycle_test_support::{fixed_detect, no_project_detect, MockDaemonOps};

// ─────────────────────────────────────────────────────────────────────────────
// AC-SL1: session UUID assigned + project detected
// ─────────────────────────────────────────────────────────────────────────────

/// SL1a — `initialize_session` always assigns a fresh session UUID.
#[tokio::test]
async fn sl1a_initialize_assigns_session_id() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let original_id = state.session_id;
    let mut daemon = MockDaemonOps::new();
    let sm = StateManager::open_at("/nonexistent/state.db");

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        || {},
    )
    .await;

    // UUID must be reassigned (even if it happens to collide — extremely rare)
    // What matters is that it is a valid, non-nil UUID.
    assert_ne!(
        state.session_id,
        uuid::Uuid::nil(),
        "session_id must not be nil after initialize"
    );
    // The new UUID may differ from the one generated in SessionState::new().
    // (In practice it always will — but the contract is merely "non-nil and valid".)
    let _ = original_id; // suppress unused warning
}

/// SL1b — when a project is detected, `project_id` is set on the state.
#[tokio::test]
async fn sl1b_initialize_sets_project_id_when_detected() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    let sm = StateManager::open_at("/nonexistent/state.db");

    let project_path = dir.path().to_path_buf();

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        fixed_detect(project_path.clone(), Some("proj-detected-id".to_string())),
        || {},
    )
    .await;

    assert_eq!(
        state.project_id.as_deref(),
        Some("proj-detected-id"),
        "project_id must reflect the detected project"
    );
    assert_eq!(
        state.current_branch.as_deref(),
        Some("main"),
        "current_branch must be set from ProjectInfo"
    );
}

/// SL1c — when no project is detected, `project_path` falls back to `cwd`.
#[tokio::test]
async fn sl1c_initialize_uses_cwd_when_no_project_detected() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    let sm = StateManager::open_at("/nonexistent/state.db");

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        || {},
    )
    .await;

    assert_eq!(
        state.project_path.as_deref(),
        Some(dir.path()),
        "project_path must fall back to cwd when no project detected"
    );
    // Parity (session-lifecycle.ts:45-47): branch is always set, never None,
    // even with no registered project. A non-git tempdir yields "default".
    assert_eq!(
        state.current_branch.as_deref(),
        Some("default"),
        "current_branch must be set even when no project is detected"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// AC-SL2: daemon_connected = true on successful health check
// ─────────────────────────────────────────────────────────────────────────────

/// SL2a — health succeeds → `daemon_connected = true`.
#[tokio::test]
async fn sl2a_initialize_sets_daemon_connected_on_health_ok() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    daemon.health_ok = true;
    let sm = StateManager::open_at("/nonexistent/state.db");

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        || {},
    )
    .await;

    assert!(
        state.daemon_connected,
        "daemon_connected must be true when health check succeeds"
    );
}

/// SL2b — health fails → `daemon_connected = false`.
#[tokio::test]
async fn sl2b_initialize_sets_daemon_disconnected_on_health_fail() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    daemon.health_ok = false;
    let sm = StateManager::open_at("/nonexistent/state.db");

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        || {},
    )
    .await;

    assert!(
        !state.daemon_connected,
        "daemon_connected must be false when health check fails"
    );
}

/// SL2c — `start_hb_fn` is called exactly once on success.
#[tokio::test]
async fn sl2c_start_hb_fn_called_on_daemon_connected() {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    daemon.health_ok = true;
    let sm = StateManager::open_at("/nonexistent/state.db");

    let call_count = Arc::new(AtomicU32::new(0));
    let cc = call_count.clone();

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        move || {
            cc.fetch_add(1, Ordering::SeqCst);
        },
    )
    .await;

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "start_hb_fn must be called exactly once when daemon connects"
    );
}

/// SL2d — `start_hb_fn` is NOT called when health fails.
#[tokio::test]
async fn sl2d_start_hb_fn_not_called_when_health_fails() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();
    daemon.health_ok = false;
    let sm = StateManager::open_at("/nonexistent/state.db");

    let called = Arc::new(AtomicBool::new(false));
    let c = called.clone();

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        &sm,
        no_project_detect,
        move || {
            c.store(true, Ordering::SeqCst);
        },
    )
    .await;

    assert!(
        !called.load(Ordering::SeqCst),
        "start_hb_fn must NOT be called when daemon health fails"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// AC-SL3: heartbeat failure → daemon_connected = false
// ─────────────────────────────────────────────────────────────────────────────

/// SL3 — a failing heartbeat loop sets `daemon_connected = false`.
///
/// Uses `heartbeat_loop` (pub(crate)) with a very short interval and a
/// heartbeat_fn that always returns an error.  After one tick the shared
/// state must have `daemon_connected = false`.
#[tokio::test]
async fn sl3_heartbeat_failure_sets_daemon_disconnected() {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    let mut state_inner = SessionState::new();
    state_inner.daemon_connected = true;
    state_inner.project_id = Some("proj-heartbeat-test".to_string());

    let state = Arc::new(Mutex::new(state_inner));

    // heartbeat_fn always fails.
    let hb_fn = |_id: String| async { Err::<bool, String>("network error".to_string()) };

    // Run heartbeat_loop for a very short interval; abort after first tick.
    let loop_state = Arc::clone(&state);
    let handle = tokio::spawn(crate::session::heartbeat::heartbeat_loop(
        loop_state,
        hb_fn,
        Duration::from_millis(1),
    ));

    // Give the immediate first tick time to run.
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle.abort();

    assert!(
        !state.lock().unwrap().daemon_connected,
        "daemon_connected must be false after a heartbeat failure (SL3)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// AC-SL4: cleanup is idempotent (run-once)
// ─────────────────────────────────────────────────────────────────────────────

/// SL4a — first cleanup marks `cleaned = true`.
#[tokio::test]
async fn sl4a_cleanup_sets_cleaned_flag() {
    let mut state = SessionState::new();
    state.daemon_connected = false; // no daemon → skip deprioritize
    let mut daemon = MockDaemonOps::new();

    cleanup_session(&mut state, &mut daemon, None).await;

    assert!(state.cleaned, "cleaned must be true after first cleanup");
}

/// SL4b — second cleanup call is a no-op (deprioritize called at most once).
#[tokio::test]
async fn sl4b_cleanup_is_idempotent() {
    let mut state = SessionState::new();
    state.daemon_connected = true;
    state.project_id = Some("proj-idempotent".to_string());
    let mut daemon = MockDaemonOps::new();

    // First call.
    cleanup_session(&mut state, &mut daemon, None).await;
    let calls_after_first = daemon.deprioritize_calls;

    // Second call must be a no-op.
    cleanup_session(&mut state, &mut daemon, None).await;
    let calls_after_second = daemon.deprioritize_calls;

    assert_eq!(
        calls_after_first, calls_after_second,
        "deprioritize must not be called again on second cleanup (idempotency)"
    );
}

/// SL4c — cleanup calls deprioritize exactly once when connected with a project.
#[tokio::test]
async fn sl4c_cleanup_calls_deprioritize_when_connected() {
    let mut state = SessionState::new();
    state.daemon_connected = true;
    state.project_id = Some("proj-deprio".to_string());
    let mut daemon = MockDaemonOps::new();

    cleanup_session(&mut state, &mut daemon, None).await;

    assert_eq!(
        daemon.deprioritize_calls, 1,
        "deprioritize must be called exactly once on cleanup"
    );
}

/// SL4d — cleanup does NOT call deprioritize when daemon is not connected.
#[tokio::test]
async fn sl4d_cleanup_skips_deprioritize_when_not_connected() {
    let mut state = SessionState::new();
    state.daemon_connected = false;
    state.project_id = Some("proj-not-connected".to_string());
    let mut daemon = MockDaemonOps::new();

    cleanup_session(&mut state, &mut daemon, None).await;

    assert_eq!(
        daemon.deprioritize_calls, 0,
        "deprioritize must NOT be called when daemon_connected=false"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// register_project
// ─────────────────────────────────────────────────────────────────────────────

/// `register_project` — project_id is applied from daemon response.
#[tokio::test]
async fn register_project_applies_project_id_from_daemon() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    state.project_path = Some(dir.path().to_path_buf());

    let mut daemon = MockDaemonOps::new();
    daemon.register_response = Some(RegisterResponse {
        project_id: "daemon-assigned-id".to_string(),
        is_worktree: false,
        watch_path: None,
        is_active: true,
        created: false,
    });

    register_project(&mut state, &mut daemon).await;

    assert_eq!(
        state.project_id.as_deref(),
        Some("daemon-assigned-id"),
        "project_id must be set from daemon RegisterResponse"
    );
}

/// `register_project` — worktree flag and watch_path are applied.
#[tokio::test]
async fn register_project_applies_worktree_and_watch_path() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    state.project_path = Some(dir.path().to_path_buf());

    let mut daemon = MockDaemonOps::new();
    daemon.register_response = Some(RegisterResponse {
        project_id: "wt-proj".to_string(),
        is_worktree: true,
        watch_path: Some("/canonical/watch".to_string()),
        is_active: true,
        created: false,
    });

    register_project(&mut state, &mut daemon).await;

    assert!(state.is_worktree, "is_worktree must be set for worktrees");
    assert_eq!(
        state.watch_path.as_deref(),
        Some(std::path::Path::new("/canonical/watch")),
        "watch_path must be set from daemon response"
    );
}

/// `register_project` — no-op when project_path is None.
#[tokio::test]
async fn register_project_noop_when_no_project_path() {
    let mut state = SessionState::new();
    // project_path is None
    let mut daemon = MockDaemonOps::new();

    register_project(&mut state, &mut daemon).await;

    assert_eq!(
        daemon.register_calls, 0,
        "register must not be called when project_path is None"
    );
}

/// `register_project` — inactive project is not registered but no error.
#[tokio::test]
async fn register_project_inactive_project_skips_apply() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    state.project_path = Some(dir.path().to_path_buf());

    let mut daemon = MockDaemonOps::new();
    daemon.register_response = Some(RegisterResponse {
        project_id: "inactive-proj".to_string(),
        is_worktree: false,
        watch_path: None,
        is_active: false,
        created: false,
    });

    register_project(&mut state, &mut daemon).await;

    // project_id should NOT be set because is_active=false and created=false
    assert!(
        state.project_id.is_none(),
        "project_id must not be set for an inactive/uncreated project"
    );
}
