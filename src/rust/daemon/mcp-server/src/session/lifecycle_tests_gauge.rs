//! Session gauge metric tests for `session::lifecycle`.
//!
//! Separated from `lifecycle_tests.rs` to keep that file under 500 lines.
//!
//! All three tests must run **serially** (`#[serial]`) because they read the
//! shared `SESSION_COUNT` gauge before and after a call.  Concurrent test
//! execution would interleave increments/decrements from other tests and
//! break the `before + delta == after` assertions.

use serial_test::serial;
use tempfile::TempDir;

use crate::observability::metrics::SESSION_COUNT;
use crate::server_types::SessionState;

use super::lifecycle_test_support::{no_project_detect, MockDaemonOps};
use super::{cleanup_session, initialize_session};

// ─────────────────────────────────────────────────────────────────────────────
// Session gauge tests
// ─────────────────────────────────────────────────────────────────────────────

/// Gauge increments by 1 after `initialize_session`.
#[serial]
#[tokio::test]
async fn session_gauge_incremented_on_initialize() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();

    let before = SESSION_COUNT.get();

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        no_project_detect(),
        || {},
    )
    .await;

    let after = SESSION_COUNT.get();
    assert_eq!(
        after,
        before + 1.0,
        "SESSION_COUNT must increment by 1 after initialize_session"
    );
}

/// Gauge decrements by 1 after `cleanup_session`.
#[serial]
#[tokio::test]
async fn session_gauge_decremented_on_cleanup() {
    let mut state = SessionState::new();
    state.daemon_connected = false;
    let mut daemon = MockDaemonOps::new();

    let before = SESSION_COUNT.get();

    cleanup_session(&mut state, &mut daemon, None).await;

    let after = SESSION_COUNT.get();
    assert_eq!(
        after,
        before - 1.0,
        "SESSION_COUNT must decrement by 1 after cleanup_session"
    );
}

/// End-to-end: initialize then cleanup leaves gauge net-zero.
#[serial]
#[tokio::test]
async fn session_gauge_net_zero_after_init_and_cleanup() {
    let dir = TempDir::new().unwrap();
    let mut state = SessionState::new();
    let mut daemon = MockDaemonOps::new();

    let before = SESSION_COUNT.get();

    initialize_session(
        &mut state,
        &mut daemon,
        dir.path(),
        no_project_detect(),
        || {},
    )
    .await;

    state.daemon_connected = false; // skip deprioritize
    cleanup_session(&mut state, &mut daemon, None).await;

    let after = SESSION_COUNT.get();
    assert_eq!(
        after, before,
        "SESSION_COUNT must be net-zero after init+cleanup"
    );
}
