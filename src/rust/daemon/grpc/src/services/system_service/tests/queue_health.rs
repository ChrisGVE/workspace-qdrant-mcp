//! Tests for queue processor health metrics, probes, heartbeats, and health
//! status derivation.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tonic::Request;
use workspace_qdrant_core::QueueProcessorHealth;

use crate::proto::{system_service_server::SystemService, ServiceStatus};

use super::super::service_impl::SystemServiceImpl;

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
async fn test_service_with_ewma_state_shares_arc_with_processor() {
    use workspace_qdrant_core::config::QueueHealthConfig;
    use workspace_qdrant_core::EwmaState;

    // The processor and the gRPC service are wired with the SAME `Arc<EwmaState>`
    // (see queue_init::attach_adaptive_and_health). Modeling that here: a sample
    // fed through one handle must be visible through the service's handle.
    let ewma = Arc::new(EwmaState::new(&QueueHealthConfig::default()));
    let service = SystemServiceImpl::new().with_ewma_state(Arc::clone(&ewma));
    assert!(service.ewma_state.is_some());

    // Feed a ms/KB sample through the "processor-side" Arc.
    ewma.update_ms_per_kb(12.5);

    // The service, holding the same Arc, observes the seeded lane.
    let snapshot = service.ewma_state.as_ref().unwrap().ms_per_kb_snapshot();
    assert!(
        snapshot.seeded,
        "service should see the processor-fed sample"
    );
    assert_eq!(snapshot.fast, 12.5);
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
async fn test_high_error_count_alone_stays_healthy() {
    // #131: `error_count` is a lifetime-cumulative counter and must NOT drive
    // the health verdict. A running processor with a recent poll stays Healthy
    // no matter how many lifetime errors have accumulated; the false-positive
    // "High error count detected" degradation has been removed. (A real
    // functional-degradation model is tracked in #133.)
    let health = Arc::new(QueueProcessorHealth::new());
    health.set_running(true);
    health.record_poll();
    health.record_heartbeat();

    // Record many lifetime errors — previously this forced Degraded.
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
    assert_eq!(queue_comp.status, ServiceStatus::Healthy as i32);
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
