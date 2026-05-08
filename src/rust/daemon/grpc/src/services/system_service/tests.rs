//! Tests for SystemService gRPC implementation

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::SystemTime;

use tonic::Request;
use workspace_qdrant_core::QueueProcessorHealth;

use crate::proto::{
    system_service_server::SystemService, QueueType, RefreshSignalRequest, ServerState,
    ServerStatusNotification, ServiceStatus,
};

use super::service_impl::SystemServiceImpl;

#[tokio::test]
async fn test_service_creation() {
    let service = SystemServiceImpl::new();
    assert!(service.start_time <= SystemTime::now());
}

#[tokio::test]
async fn test_default_trait() {
    let _service = SystemServiceImpl::default();
    // Should not panic
}

#[tokio::test]
async fn test_service_with_queue_health() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    health.set_queue_depth(42);

    let service = SystemServiceImpl::new().with_queue_health(health.clone());
    assert!(service.queue_health.is_some());

    // Test health check includes queue processor
    let response = service.health(Request::new(())).await.unwrap();
    let health_response = response.into_inner();
    assert!(health_response.components.len() >= 2);
    assert!(health_response
        .components
        .iter()
        .any(|c| c.component_name == "queue_processor"));
}

#[tokio::test]
async fn test_queue_processor_health_metrics() {
    let health = QueueProcessorHealth::new();

    // Test initial state
    assert!(!health.is_running.load(Ordering::SeqCst));
    assert_eq!(health.error_count.load(Ordering::SeqCst), 0);

    // Test running state
    health.set_running(true);
    assert!(health.is_running.load(Ordering::SeqCst));

    // Test error recording
    health.record_error();
    health.record_error();
    assert_eq!(health.error_count.load(Ordering::SeqCst), 2);

    // Test success recording
    health.record_success(100);
    health.record_success(200);
    assert_eq!(health.items_processed.load(Ordering::SeqCst), 2);
    // Average should be approximately 150
    let avg = health.avg_processing_time_ms.load(Ordering::SeqCst);
    assert!(avg > 0);

    // Test failure recording
    health.record_failure();
    assert_eq!(health.items_failed.load(Ordering::SeqCst), 1);

    // Test queue depth
    health.set_queue_depth(100);
    assert_eq!(health.queue_depth.load(Ordering::SeqCst), 100);
}

#[tokio::test]
async fn test_queue_processor_health_poll_time() {
    let health = QueueProcessorHealth::new();

    // Before any poll, should return MAX
    assert_eq!(health.seconds_since_last_poll(), u64::MAX);

    // After poll, should return small number
    health.record_poll();
    let secs = health.seconds_since_last_poll();
    assert!(secs < 2); // Should be nearly instant
}

#[tokio::test]
async fn test_heartbeat_initial_state_is_max() {
    let health = QueueProcessorHealth::new();
    assert_eq!(health.seconds_since_last_heartbeat(), u64::MAX);
}

#[tokio::test]
async fn test_heartbeat_after_record_is_recent() {
    let health = QueueProcessorHealth::new();
    health.record_heartbeat();
    let secs = health.seconds_since_last_heartbeat();
    assert!(secs < 2);
}

#[tokio::test]
async fn test_healthy_when_only_heartbeat_is_recent() {
    // Simulates: poll is old (long batch), but heartbeat is fresh (item just processed)
    // Stalled = min(poll, heartbeat) > 60 — should stay Healthy when heartbeat is fresh
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    // poll stays at 0 (MAX seconds ago)
    health.record_heartbeat(); // heartbeat is fresh

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.health(Request::new(())).await.unwrap();
    let health_response = response.into_inner();
    let queue_comp = health_response
        .components
        .iter()
        .find(|c| c.component_name == "queue_processor")
        .unwrap();
    // min(MAX, ~0) = ~0 which is not > 60, so Healthy
    assert_eq!(queue_comp.status, ServiceStatus::Healthy as i32);
}

#[tokio::test]
async fn test_healthy_when_only_poll_is_recent() {
    // Simulates: poll just fired, but no items have been processed yet
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    health.record_poll(); // poll is fresh
                          // heartbeat stays at 0 (MAX seconds ago)

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.health(Request::new(())).await.unwrap();
    let health_response = response.into_inner();
    let queue_comp = health_response
        .components
        .iter()
        .find(|c| c.component_name == "queue_processor")
        .unwrap();
    // min(~0, MAX) = ~0 which is not > 60, so Healthy
    assert_eq!(queue_comp.status, ServiceStatus::Healthy as i32);
}

#[tokio::test]
async fn test_metrics_include_queue_metrics() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    health.set_queue_depth(10);
    health.record_success(50);

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.get_metrics(Request::new(())).await.unwrap();
    let metrics = response.into_inner().metrics;

    // Check that queue metrics are present
    let metric_names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
    assert!(metric_names.contains(&"queue_pending"));
    assert!(metric_names.contains(&"queue_processed"));
    assert!(metric_names.contains(&"queue_failed"));
    assert!(metric_names.contains(&"queue_processor_running"));

    // Verify values
    let pending = metrics.iter().find(|m| m.name == "queue_pending").unwrap();
    assert_eq!(pending.value, 10.0);

    let running = metrics
        .iter()
        .find(|m| m.name == "queue_processor_running")
        .unwrap();
    assert_eq!(running.value, 1.0);
}

#[tokio::test]
async fn test_status_includes_queue_depth() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_queue_depth(25);

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.get_status(Request::new(())).await.unwrap();
    let status = response.into_inner();

    assert_eq!(status.metrics.unwrap().pending_operations, 25);
}

#[tokio::test]
async fn test_health_status_degraded_on_high_errors() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    health.record_poll();

    // Record many errors
    for _ in 0..101 {
        health.record_error();
    }

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.health(Request::new(())).await.unwrap();
    let health_response = response.into_inner();

    let queue_comp = health_response
        .components
        .iter()
        .find(|c| c.component_name == "queue_processor")
        .unwrap();
    assert_eq!(queue_comp.status, ServiceStatus::Degraded as i32);
}

#[tokio::test]
async fn test_health_status_unhealthy_when_not_running() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(false);

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.health(Request::new(())).await.unwrap();
    let health_response = response.into_inner();

    let queue_comp = health_response
        .components
        .iter()
        .find(|c| c.component_name == "queue_processor")
        .unwrap();
    assert_eq!(queue_comp.status, ServiceStatus::Unhealthy as i32);
}

#[tokio::test]
async fn test_get_queue_stats() {
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_queue_depth(15);
    health.record_success(100);
    health.record_success(200);
    health.record_failure();

    let service = SystemServiceImpl::new().with_queue_health(health);
    let response = service.get_queue_stats(Request::new(())).await.unwrap();
    let stats = response.into_inner();

    assert_eq!(stats.pending_count, 15);
    assert_eq!(stats.completed_count, 2);
    assert_eq!(stats.failed_count, 1);
}

#[tokio::test]
async fn test_shutdown() {
    let service = SystemServiceImpl::new();
    let response = service.shutdown(Request::new(())).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_status_store_initialized_empty() {
    let service = SystemServiceImpl::new();
    let store = service.status_store.read().await;
    assert!(store.is_empty());
}

#[tokio::test]
async fn test_notify_server_status_stores_entry() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("test-project".to_string()),
        project_root: Some("/tmp/test-project".to_string()),
    };

    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    // Verify the entry was stored
    let store = service.status_store.read().await;
    assert_eq!(store.len(), 1);
    let entry = store.get("test-project").unwrap();
    assert_eq!(entry.state, ServerState::Up);
    assert_eq!(entry.project_name.as_deref(), Some("test-project"));
    assert_eq!(entry.project_root.as_deref(), Some("/tmp/test-project"));
}

#[tokio::test]
async fn test_notify_server_status_transitions() {
    let service = SystemServiceImpl::new();

    // First: UP
    let up_notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("my-app".to_string()),
        project_root: Some("/home/user/my-app".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(up_notification))
        .await;
    assert!(response.is_ok());

    // Then: DOWN
    let down_notification = ServerStatusNotification {
        state: ServerState::Down as i32,
        project_name: Some("my-app".to_string()),
        project_root: Some("/home/user/my-app".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(down_notification))
        .await;
    assert!(response.is_ok());

    // Verify the final state is DOWN
    let store = service.status_store.read().await;
    let entry = store.get("my-app").unwrap();
    assert_eq!(entry.state, ServerState::Down);
}

#[tokio::test]
async fn test_notify_server_status_uses_project_root_as_fallback_key() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: None,
        project_root: Some("/tmp/fallback".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    let store = service.status_store.read().await;
    assert!(store.contains_key("/tmp/fallback"));
}

#[tokio::test]
async fn test_notify_server_status_unknown_fallback() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: None,
        project_root: None,
    };
    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    let store = service.status_store.read().await;
    assert!(store.contains_key("unknown"));
}

#[tokio::test]
async fn test_send_refresh_signal_without_db_pool() {
    // Without a database pool, refresh signal should return Ok but do nothing
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_tools_available() {
    // ToolsAvailable is informational and should always succeed
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::ToolsAvailable as i32,
        lsp_languages: vec!["rust".to_string(), "python".to_string()],
        grammar_languages: vec!["javascript".to_string()],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_unspecified() {
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::Unspecified as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_multiple_components_tracked_independently() {
    let service = SystemServiceImpl::new();

    // Register two different components
    let notification1 = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("project-a".to_string()),
        project_root: Some("/tmp/a".to_string()),
    };
    let notification2 = ServerStatusNotification {
        state: ServerState::Down as i32,
        project_name: Some("project-b".to_string()),
        project_root: Some("/tmp/b".to_string()),
    };

    service
        .notify_server_status(Request::new(notification1))
        .await
        .unwrap();
    service
        .notify_server_status(Request::new(notification2))
        .await
        .unwrap();

    let store = service.status_store.read().await;
    assert_eq!(store.len(), 2);
    assert_eq!(store.get("project-a").unwrap().state, ServerState::Up);
    assert_eq!(store.get("project-b").unwrap().state, ServerState::Down);
}

#[tokio::test]
async fn test_pause_all_watchers_without_db_pool() {
    let service = SystemServiceImpl::new();
    let response = service.pause_all_watchers(Request::new(())).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_resume_all_watchers_without_db_pool() {
    let service = SystemServiceImpl::new();
    let response = service.resume_all_watchers(Request::new(())).await;
    assert!(response.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────
// §7.9 — embedding_provider Health component + GetEmbeddingProviderStatus
// ─────────────────────────────────────────────────────────────────────────

mod embedding_provider_health {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use workspace_qdrant_core::config::EmbeddingSettings;
    use workspace_qdrant_core::embedding::provider::DenseProvider;
    use workspace_qdrant_core::embedding::{DenseEmbedding, EmbeddingError};

    #[allow(unused_imports)]
    use crate::proto::system_service_server::SystemService as _;

    /// Stub provider whose `probe()` returns whatever the supplied closure
    /// produces. Counts probe invocations so tests can assert on caching.
    struct StubProvider {
        result_factory: StdMutex<Box<dyn FnMut() -> Result<(), EmbeddingError> + Send>>,
        probe_calls: std::sync::atomic::AtomicUsize,
    }

    impl std::fmt::Debug for StubProvider {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("StubProvider")
                .field("probe_calls", &self.probe_calls)
                .finish()
        }
    }

    impl StubProvider {
        fn new<F>(f: F) -> Arc<Self>
        where
            F: FnMut() -> Result<(), EmbeddingError> + Send + 'static,
        {
            Arc::new(Self {
                result_factory: StdMutex::new(Box::new(f)),
                probe_calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.probe_calls.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl DenseProvider for StubProvider {
        async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(Vec::new())
        }
        fn output_dim(&self) -> usize {
            1536
        }
        fn provider_label(&self) -> &str {
            "stub-openai"
        }
        fn metrics_label(&self) -> &'static str {
            "openai"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            self.probe_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (self.result_factory.lock().unwrap())()
        }
    }

    fn settings_with(ttl_secs: u64) -> Arc<EmbeddingSettings> {
        let mut s = EmbeddingSettings::default();
        s.health_probe_cache_secs = ttl_secs;
        s.output_dim = 1536;
        Arc::new(s)
    }

    fn find_component<'a>(
        components: &'a [crate::proto::ComponentHealth],
        name: &str,
    ) -> Option<&'a crate::proto::ComponentHealth> {
        components.iter().find(|c| c.component_name == name)
    }

    #[tokio::test]
    async fn test_health_includes_embedding_provider_component() {
        let provider = StubProvider::new(|| Ok(()));
        let svc = SystemServiceImpl::new()
            .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
            .with_embedding_settings(settings_with(60));
        // Seed the cache so Health does not report `probe pending`.
        svc.seed_embedding_probe_cache(Ok(())).await;

        let resp = svc.health(Request::new(())).await.unwrap().into_inner();
        let comp = find_component(&resp.components, "embedding_provider")
            .expect("Health response must include embedding_provider component");
        assert_eq!(comp.status, ServiceStatus::Healthy as i32);
        assert!(comp.message.contains("Running"));
    }

    #[tokio::test]
    async fn test_health_embedding_provider_unhealthy_propagates() {
        let provider = StubProvider::new(|| {
            Err(EmbeddingError::RemoteError {
                status_code: 401,
                message: "Unauthorized".to_string(),
            })
        });
        let svc = SystemServiceImpl::new()
            .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
            .with_embedding_settings(settings_with(60));
        svc.seed_embedding_probe_cache(Err(EmbeddingError::RemoteError {
            status_code: 401,
            message: "Unauthorized".to_string(),
        }))
        .await;

        let resp = svc.health(Request::new(())).await.unwrap().into_inner();
        let comp = find_component(&resp.components, "embedding_provider").unwrap();
        assert_eq!(comp.status, ServiceStatus::Unhealthy as i32);
        assert!(
            comp.message.contains("auth failure"),
            "expected auth-failure message, got: {}",
            comp.message
        );

        // Overall health must reflect the unhealthy component.
        assert_eq!(resp.status, ServiceStatus::Unhealthy as i32);
    }

    #[tokio::test]
    async fn test_health_probe_cached_within_ttl() {
        let provider = StubProvider::new(|| Ok(()));
        let svc = SystemServiceImpl::new()
            .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
            .with_embedding_settings(settings_with(60));

        // Two on-demand status RPCs in quick succession should produce only
        // one underlying probe call thanks to the TTL cache.
        let _ = svc
            .get_embedding_provider_status(Request::new(()))
            .await
            .unwrap();
        let _ = svc
            .get_embedding_provider_status(Request::new(()))
            .await
            .unwrap();
        assert_eq!(
            provider.calls(),
            1,
            "second status call within TTL must reuse cached probe; got {} probes",
            provider.calls()
        );
    }

    #[tokio::test]
    async fn test_health_embedding_provider_probe_pending() {
        // Provider wired but no probe has run yet — Health must report
        // Degraded with a `probe pending` message, not block on a probe.
        let provider = StubProvider::new(|| Ok(()));
        let svc = SystemServiceImpl::new()
            .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
            .with_embedding_settings(settings_with(60));

        let resp = svc.health(Request::new(())).await.unwrap().into_inner();
        let comp = find_component(&resp.components, "embedding_provider").unwrap();
        assert_eq!(comp.status, ServiceStatus::Degraded as i32);
        assert!(
            comp.message.contains("probe pending"),
            "got: {}",
            comp.message
        );
        assert_eq!(
            provider.calls(),
            0,
            "Health must not issue a probe when the cache is empty"
        );
    }
}
