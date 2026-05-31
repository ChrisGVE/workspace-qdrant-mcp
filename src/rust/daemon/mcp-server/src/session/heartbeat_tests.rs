//! Tests for `session::heartbeat`.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::Duration;

use tokio::sync::Mutex;

use crate::server_types::SessionState;

use super::{heartbeat_loop, tick_heartbeat};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn connected_state_with_project(project_id: &str) -> Arc<Mutex<SessionState>> {
    let mut s = SessionState::new();
    s.daemon_connected = true;
    s.project_id = Some(project_id.to_string());
    Arc::new(Mutex::new(s))
}

fn disconnected_state() -> Arc<Mutex<SessionState>> {
    Arc::new(Mutex::new(SessionState::new()))
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_heartbeat
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tick_heartbeat_calls_fn_when_connected() {
    let state = connected_state_with_project("proj-abc");
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    let mut hb_fn = |_id: String| {
        called_clone.store(true, Ordering::SeqCst);
        async { Ok::<bool, String>(true) }
    };

    tick_heartbeat(&state, &mut hb_fn).await;
    assert!(called.load(Ordering::SeqCst), "heartbeat_fn must be called");
}

#[tokio::test]
async fn tick_heartbeat_skips_when_no_project() {
    let state = disconnected_state(); // no project_id, not connected
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    let mut hb_fn = |_id: String| {
        called_clone.store(true, Ordering::SeqCst);
        async { Ok::<bool, String>(true) }
    };

    tick_heartbeat(&state, &mut hb_fn).await;
    assert!(
        !called.load(Ordering::SeqCst),
        "heartbeat_fn must NOT be called when disconnected"
    );
}

#[tokio::test]
async fn tick_heartbeat_skips_when_daemon_disconnected() {
    // Has project_id but daemon_connected = false
    let state = Arc::new(Mutex::new({
        let mut s = SessionState::new();
        s.daemon_connected = false;
        s.project_id = Some("proj-xyz".to_string());
        s
    }));
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    let mut hb_fn = |_id: String| {
        called_clone.store(true, Ordering::SeqCst);
        async { Ok::<bool, String>(true) }
    };

    tick_heartbeat(&state, &mut hb_fn).await;
    assert!(
        !called.load(Ordering::SeqCst),
        "heartbeat_fn must NOT be called when daemon_connected=false"
    );
}

#[tokio::test]
async fn tick_heartbeat_failure_sets_daemon_disconnected() {
    let state = connected_state_with_project("proj-abc");

    let mut hb_fn = |_id: String| async { Err::<bool, String>("rpc failed".to_string()) };

    tick_heartbeat(&state, &mut hb_fn).await;

    assert!(
        !state.lock().await.daemon_connected,
        "daemon_connected must be false after heartbeat failure"
    );
}

#[tokio::test]
async fn tick_heartbeat_success_keeps_daemon_connected() {
    let state = connected_state_with_project("proj-abc");

    let mut hb_fn = |_id: String| async { Ok::<bool, String>(true) };

    tick_heartbeat(&state, &mut hb_fn).await;

    assert!(
        state.lock().await.daemon_connected,
        "daemon_connected must remain true after successful heartbeat"
    );
}

#[tokio::test]
async fn tick_heartbeat_passes_correct_project_id() {
    let state = connected_state_with_project("my-project-id");
    let captured_id = Arc::new(StdMutex::new(String::new()));
    let cap_clone = captured_id.clone();

    let mut hb_fn = move |id: String| {
        *cap_clone.lock().unwrap() = id;
        async { Ok::<bool, String>(false) }
    };

    tick_heartbeat(&state, &mut hb_fn).await;
    assert_eq!(&*captured_id.lock().unwrap(), "my-project-id");
}

// ─────────────────────────────────────────────────────────────────────────────
// heartbeat_loop (short interval for fast test)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn heartbeat_loop_fires_immediately_then_interval() {
    let state = connected_state_with_project("proj-abc");
    let count = Arc::new(AtomicU32::new(0));
    let count_clone = count.clone();

    let hb_fn = move |_: String| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        async { Ok::<bool, String>(true) }
    };

    // Very short interval (5 ms) so we can observe 2+ ticks.
    let loop_state = Arc::clone(&state);
    let handle = tokio::spawn(heartbeat_loop(loop_state, hb_fn, Duration::from_millis(5)));

    // Wait long enough for immediate + ≥1 interval ticks.
    tokio::time::sleep(Duration::from_millis(30)).await;
    handle.abort();

    let n = count.load(Ordering::SeqCst);
    assert!(n >= 2, "expected at least 2 heartbeats, got {n}");
}

#[tokio::test]
async fn heartbeat_loop_abort_handle_stops_loop() {
    let state = connected_state_with_project("proj-abc");
    let count = Arc::new(AtomicU32::new(0));
    let count_clone = count.clone();

    let hb_fn = move |_: String| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        async { Ok::<bool, String>(true) }
    };

    let loop_state = Arc::clone(&state);
    let handle = tokio::spawn(heartbeat_loop(loop_state, hb_fn, Duration::from_millis(5)));
    let abort = handle.abort_handle();

    // Let it run briefly, then abort.
    tokio::time::sleep(Duration::from_millis(10)).await;
    abort.abort();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let count_after_abort = count.load(Ordering::SeqCst);

    // Wait more — count must not grow after abort.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let count_final = count.load(Ordering::SeqCst);
    assert_eq!(
        count_after_abort, count_final,
        "heartbeat must stop after abort"
    );
}
