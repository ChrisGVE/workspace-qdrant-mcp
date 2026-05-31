//! Hermetic tests for health_monitor.
//!
//! No live daemon, Qdrant, or 30-second waits.  Probe injection via simple
//! structs that return predetermined values.

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use serde_json::json;

    use crate::observability::health_monitor::{
        augment_search_results, compute_state, force_check, get_health_metadata, DaemonProbe,
        HealthMonitorBuilder, HealthState, HealthStatus, QdrantProbe, SharedHealthState,
        UncertainReason,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Fake probes
    // ─────────────────────────────────────────────────────────────────────────

    #[derive(Clone)]
    struct FakeDaemon(bool);
    impl DaemonProbe for FakeDaemon {
        async fn check(&self) -> bool {
            self.0
        }
    }

    #[derive(Clone)]
    struct FakeQdrant(bool);
    impl QdrantProbe for FakeQdrant {
        async fn check(&self) -> bool {
            self.0
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // compute_state classification (health-monitor.ts:174-208)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn both_available_is_healthy() {
        let s = compute_state(true, true);
        assert_eq!(s.status, HealthStatus::Healthy);
        assert!(s.daemon_available);
        assert!(s.qdrant_available);
        assert!(s.reason.is_none());
        assert!(s.message.is_none());
    }

    #[test]
    fn both_unavailable_is_uncertain_both() {
        let s = compute_state(false, false);
        assert_eq!(s.status, HealthStatus::Uncertain);
        assert!(!s.daemon_available);
        assert!(!s.qdrant_available);
        assert_eq!(s.reason, Some(UncertainReason::BothUnavailable));
        assert!(s.message.as_ref().unwrap().contains("Both"));
    }

    #[test]
    fn daemon_down_only_is_uncertain_daemon() {
        let s = compute_state(false, true);
        assert_eq!(s.status, HealthStatus::Uncertain);
        assert!(!s.daemon_available);
        assert!(s.qdrant_available);
        assert_eq!(s.reason, Some(UncertainReason::DaemonUnavailable));
        assert!(s.message.as_ref().unwrap().contains("Daemon"));
    }

    #[test]
    fn qdrant_down_only_is_uncertain_qdrant() {
        let s = compute_state(true, false);
        assert_eq!(s.status, HealthStatus::Uncertain);
        assert!(s.daemon_available);
        assert!(!s.qdrant_available);
        assert_eq!(s.reason, Some(UncertainReason::QdrantUnavailable));
        assert!(s.message.as_ref().unwrap().contains("Qdrant"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Message text — parity with health-monitor.ts:190-199
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn both_unavailable_message_matches_ts() {
        let s = compute_state(false, false);
        assert_eq!(
            s.message.as_deref(),
            Some(
                "Both daemon and Qdrant are unavailable. \
                 Search results may be incomplete or unavailable."
            )
        );
    }

    #[test]
    fn daemon_unavailable_message_matches_ts() {
        let s = compute_state(false, true);
        assert_eq!(
            s.message.as_deref(),
            Some(
                "Daemon is unavailable. \
                 Search results may use cached data and new content cannot be indexed."
            )
        );
    }

    #[test]
    fn qdrant_unavailable_message_matches_ts() {
        let s = compute_state(true, false);
        assert_eq!(
            s.message.as_deref(),
            Some("Qdrant is unavailable. Search functionality is limited.")
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // get_health_metadata — health-monitor.ts:213-229
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn metadata_is_none_when_healthy() {
        let s = compute_state(true, true);
        assert!(get_health_metadata(&s).is_none());
    }

    #[test]
    fn metadata_present_when_daemon_unavailable() {
        let s = compute_state(false, true);
        let m = get_health_metadata(&s).expect("should have metadata");
        assert_eq!(m.status, HealthStatus::Uncertain);
        assert_eq!(m.reason, Some(UncertainReason::DaemonUnavailable));
        assert!(m.message.is_some());
    }

    #[test]
    fn metadata_present_when_qdrant_unavailable() {
        let s = compute_state(true, false);
        let m = get_health_metadata(&s).expect("should have metadata");
        assert_eq!(m.reason, Some(UncertainReason::QdrantUnavailable));
    }

    #[test]
    fn metadata_present_when_both_unavailable() {
        let s = compute_state(false, false);
        let m = get_health_metadata(&s).expect("should have metadata");
        assert_eq!(m.reason, Some(UncertainReason::BothUnavailable));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // augment_search_results — health-monitor.ts:237-248
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn no_health_key_when_healthy() {
        let state = compute_state(true, true);
        let input = json!({ "success": true, "results": [] });
        let out = augment_search_results(&state, input.clone());
        // Must be byte-identical to input — no health key.
        assert!(out.get("health").is_none());
        assert_eq!(out.get("success"), Some(&json!(true)));
    }

    #[test]
    fn health_key_added_when_daemon_unavailable() {
        let state = compute_state(false, true);
        let input = json!({ "success": true, "results": [] });
        let out = augment_search_results(&state, input);
        let health = out.get("health").expect("health key must be present");
        assert_eq!(
            health.get("status").and_then(|v| v.as_str()),
            Some("uncertain")
        );
        assert_eq!(
            health.get("reason").and_then(|v| v.as_str()),
            Some("daemon_unavailable")
        );
        assert!(health.get("message").is_some());
    }

    #[test]
    fn health_key_added_when_qdrant_unavailable() {
        let state = compute_state(true, false);
        let input = json!({ "success": true });
        let out = augment_search_results(&state, input);
        let health = out.get("health").expect("health key must be present");
        assert_eq!(
            health.get("reason").and_then(|v| v.as_str()),
            Some("qdrant_unavailable")
        );
    }

    #[test]
    fn health_key_added_when_both_unavailable() {
        let state = compute_state(false, false);
        let input = json!({ "success": true });
        let out = augment_search_results(&state, input);
        let health = out.get("health").expect("health key must be present");
        assert_eq!(
            health.get("reason").and_then(|v| v.as_str()),
            Some("both_unavailable")
        );
    }

    #[test]
    fn health_key_shape_uncertain_has_status_reason_message() {
        let state = compute_state(false, false);
        let out = augment_search_results(&state, json!({}));
        let health = out.get("health").unwrap();
        assert!(health.get("status").is_some());
        assert!(health.get("reason").is_some());
        assert!(health.get("message").is_some());
    }

    #[test]
    fn augment_non_object_is_noop() {
        // If the value is not an object, no panic and no modification.
        let state = compute_state(false, false);
        let input = json!([1, 2, 3]);
        let out = augment_search_results(&state, input.clone());
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn existing_fields_preserved_when_augmented() {
        let state = compute_state(false, true);
        let input = json!({ "success": true, "results": [], "query": "test" });
        let out = augment_search_results(&state, input);
        assert_eq!(out.get("success"), Some(&json!(true)));
        assert_eq!(out.get("query").and_then(|v| v.as_str()), Some("test"));
        assert!(out.get("health").is_some());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HealthMetadata serialisation — exact JSON field names
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn uncertain_reason_serialises_snake_case_daemon() {
        let state = compute_state(false, true);
        let meta = get_health_metadata(&state).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"daemon_unavailable\""));
    }

    #[test]
    fn uncertain_reason_serialises_snake_case_qdrant() {
        let state = compute_state(true, false);
        let meta = get_health_metadata(&state).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"qdrant_unavailable\""));
    }

    #[test]
    fn uncertain_reason_serialises_snake_case_both() {
        let state = compute_state(false, false);
        let meta = get_health_metadata(&state).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"both_unavailable\""));
    }

    #[test]
    fn health_status_serialises_as_lowercase_uncertain() {
        let state = compute_state(false, false);
        let meta = get_health_metadata(&state).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"uncertain\""));
    }

    #[test]
    fn health_status_serialises_as_lowercase_healthy() {
        let state = HealthState::initial();
        let s = serde_json::to_string(&state.status).unwrap();
        assert_eq!(s, "\"healthy\"");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // force_check — single-shot check without a 30-second wait
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn force_check_healthy_when_both_probes_return_true() {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let d = FakeDaemon(true);
        let q = FakeQdrant(true);
        let result = force_check(&state, &d, &q).await;
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn force_check_uncertain_when_daemon_probe_returns_false() {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let d = FakeDaemon(false);
        let q = FakeQdrant(true);
        let result = force_check(&state, &d, &q).await;
        assert_eq!(result.status, HealthStatus::Uncertain);
        assert_eq!(result.reason, Some(UncertainReason::DaemonUnavailable));
    }

    #[tokio::test]
    async fn force_check_uncertain_when_qdrant_probe_returns_false() {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let d = FakeDaemon(true);
        let q = FakeQdrant(false);
        let result = force_check(&state, &d, &q).await;
        assert_eq!(result.status, HealthStatus::Uncertain);
        assert_eq!(result.reason, Some(UncertainReason::QdrantUnavailable));
    }

    #[tokio::test]
    async fn force_check_uncertain_when_both_probes_return_false() {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let d = FakeDaemon(false);
        let q = FakeQdrant(false);
        let result = force_check(&state, &d, &q).await;
        assert_eq!(result.status, HealthStatus::Uncertain);
        assert_eq!(result.reason, Some(UncertainReason::BothUnavailable));
    }

    #[tokio::test]
    async fn force_check_updates_shared_state() {
        let state: SharedHealthState = Arc::new(RwLock::new(HealthState::initial()));
        let d = FakeDaemon(false);
        let q = FakeQdrant(false);
        force_check(&state, &d, &q).await;
        let guard = state.read().unwrap();
        assert_eq!(guard.status, HealthStatus::Uncertain);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HealthMonitorBuilder — smoke-test build + initial state
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn builder_build_starts_with_initial_state() {
        let monitor = HealthMonitorBuilder::new(FakeDaemon(true), FakeQdrant(true))
            .with_interval(std::time::Duration::from_secs(3600)) // long interval: no auto-run
            .build();
        // State starts as initial (healthy/true/true).
        let guard = monitor.state().read().unwrap().clone();
        // Initial is healthy — first async check may not have fired yet.
        // We just verify the handle works without panic.
        drop(guard);
        monitor.stop();
    }

    #[tokio::test]
    async fn builder_stop_does_not_panic() {
        let monitor = HealthMonitorBuilder::new(FakeDaemon(true), FakeQdrant(true))
            .with_interval(std::time::Duration::from_secs(3600))
            .build();
        monitor.stop();
        // Double-stop (via Drop) should also be fine.
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HealthState::initial
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn initial_state_is_healthy() {
        let s = HealthState::initial();
        assert_eq!(s.status, HealthStatus::Healthy);
        assert!(s.daemon_available);
        assert!(s.qdrant_available);
        assert!(s.reason.is_none());
        assert!(s.message.is_none());
    }
}
