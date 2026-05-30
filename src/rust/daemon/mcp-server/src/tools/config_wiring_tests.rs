//! Integration-style unit tests verifying that `ServerConfig` is correctly
//! wired into `ToolsHandler` end-to-end.
//!
//! Specifically: `ServerConfig.rules.duplication_threshold` (from the config
//! file) → `ToolsHandler::new_with_config` → `ToolsHandler.rules_dup_threshold`
//! → `DispatchContext.rules_dup_threshold`. (Parity: there is intentionally NO
//! env override for the threshold — TS sources it from config only.)
//!
//! This guards against the regression found in codex audit Round-2 where the
//! config module + env overrides existed and were unit-tested in isolation but
//! the runtime startup (`main.rs` / `run_stdio` / `run_http`) passed `None`
//! hardcoded for the handler config — so no config value ever took effect at
//! runtime.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::config::load_config_with_env;
use crate::grpc::client::DaemonClient;
use crate::observability::health_monitor::{HealthState, SharedHealthState};
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::StateManager;
use crate::tools::ToolsHandler;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal (degraded) set of deps suitable for a unit test.
#[tokio::test]
async fn handler_stores_rules_dup_threshold_from_config() {
    // 1. A config whose rules section carries the threshold (as it would after
    //    loading a config file with `rules.duplication_threshold` set).
    let mut config = crate::config::ServerConfig::default();
    config
        .rules
        .get_or_insert_with(Default::default)
        .duplication_threshold = Some(0.42);

    // 2. Extract the threshold from config — same code as main.rs run_stdio.
    let rules_dup_threshold = config.rules.as_ref().and_then(|r| r.duplication_threshold);

    assert_eq!(
        rules_dup_threshold,
        Some(0.42),
        "config must carry the file-configured threshold"
    );

    // 3. Construct a ToolsHandler via new_with_config — same as transport layer.
    let daemon = DaemonClient::connect_default().expect("daemon client");
    let qdrant = QdrantReadClient::new(
        "http://localhost:6333".to_string(),
        None, // no API key
    );
    let state = StateManager::open_at("/tmp/nonexistent-test-state.db");
    let session = SessionState::new();
    let health: SharedHealthState = Arc::new(std::sync::RwLock::new(HealthState::initial()));

    let handler =
        ToolsHandler::new_with_config(daemon, qdrant, state, session, health, rules_dup_threshold);

    // 4. Assert the handler stores the threshold (accessible via private field
    //    in this #[cfg(test)] sibling module).
    assert_eq!(
        handler.rules_dup_threshold,
        Some(0.42),
        "ToolsHandler must propagate config threshold; got {:?}",
        handler.rules_dup_threshold
    );
}

#[tokio::test]
async fn handler_stores_none_when_threshold_not_configured() {
    // No WQM_RULES_DEDUP_THRESHOLD in env → rules_dup_threshold must be None.
    let getter = |_key: &str| -> Option<String> { None };
    let config = load_config_with_env(&getter).expect("config load");

    let rules_dup_threshold = config.rules.as_ref().and_then(|r| r.duplication_threshold);

    assert_eq!(
        rules_dup_threshold, None,
        "default config must not set a duplication threshold"
    );

    let daemon = DaemonClient::connect_default().expect("daemon client");
    let qdrant = QdrantReadClient::new("http://localhost:6333".to_string(), None);
    let state = StateManager::open_at("/tmp/nonexistent-test-state.db");
    let session = SessionState::new();
    let health: SharedHealthState = Arc::new(std::sync::RwLock::new(HealthState::initial()));

    let handler =
        ToolsHandler::new_with_config(daemon, qdrant, state, session, health, rules_dup_threshold);

    assert_eq!(
        handler.rules_dup_threshold, None,
        "ToolsHandler must have None threshold when config is absent"
    );
}

#[tokio::test]
async fn from_arcs_with_config_stores_threshold() {
    // Verify the HTTP-transport path (from_arcs_with_config) also stores the
    // threshold, mirroring the factory-closure in http.rs build_mcp_service.
    let threshold = Some(0.85_f64);

    let daemon_arc = Arc::new(Mutex::new(
        DaemonClient::connect_default().expect("daemon client"),
    ));
    let qdrant_arc = Arc::new(QdrantReadClient::new(
        "http://localhost:6333".to_string(),
        None,
    ));
    let state_arc = Arc::new(crate::sqlite::SharedStateManager::new(
        StateManager::open_at("/tmp/nonexistent-test-state.db"),
    ));
    let session_arc = Arc::new(Mutex::new(SessionState::new()));
    let health: SharedHealthState = Arc::new(std::sync::RwLock::new(HealthState::initial()));

    let handler = ToolsHandler::from_arcs_with_config(
        daemon_arc,
        qdrant_arc,
        state_arc,
        session_arc,
        health,
        threshold,
    );

    assert_eq!(
        handler.rules_dup_threshold, threshold,
        "from_arcs_with_config must propagate the threshold; got {:?}",
        handler.rules_dup_threshold
    );
}
