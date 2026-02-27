//! Tests for WatchQueueCoordinator capacity management (Task 461.9).

use super::super::*;

#[test]
fn test_coordinator_config_default() {
    let config = CoordinatorConfig::default();
    assert_eq!(config.total_capacity, 10000);
    assert_eq!(config.min_per_watch, 100);
    assert_eq!(config.max_per_watch, 2000);
}

#[test]
fn test_coordinator_new() {
    let coordinator = WatchQueueCoordinator::new();
    assert_eq!(coordinator.get_total_capacity(), 10000);
    assert_eq!(coordinator.get_available_capacity(), 10000);
    assert_eq!(coordinator.get_allocated_capacity(), 0);
}

#[tokio::test]
async fn test_coordinator_request_and_release_capacity() {
    let coordinator = WatchQueueCoordinator::new();

    // Request capacity
    let granted = coordinator.request_capacity("watch-1", 100).await;
    assert!(granted);
    assert_eq!(coordinator.get_allocated_capacity(), 100);
    assert_eq!(coordinator.get_available_capacity(), 9900);

    // Verify watch allocation
    let alloc = coordinator.get_watch_allocation("watch-1").await;
    assert_eq!(alloc, Some(100));

    // Release capacity
    coordinator.release_capacity("watch-1", 50).await;
    assert_eq!(coordinator.get_allocated_capacity(), 50);
    assert_eq!(coordinator.get_available_capacity(), 9950);

    // Verify watch allocation after release
    let alloc = coordinator.get_watch_allocation("watch-1").await;
    assert_eq!(alloc, Some(50));
}

#[tokio::test]
async fn test_coordinator_capacity_limit() {
    let config = CoordinatorConfig {
        total_capacity: 1000,
        min_per_watch: 10,
        max_per_watch: 200,
    };
    let coordinator = WatchQueueCoordinator::with_config(config);

    // Request up to max_per_watch
    let granted = coordinator.request_capacity("watch-1", 200).await;
    assert!(granted);

    // Request more than max_per_watch should fail
    let granted = coordinator.request_capacity("watch-1", 100).await;
    assert!(!granted);
    assert_eq!(coordinator.get_allocated_capacity(), 200);
}

#[tokio::test]
async fn test_coordinator_total_capacity_limit() {
    let config = CoordinatorConfig {
        total_capacity: 500,
        min_per_watch: 10,
        max_per_watch: 1000,
    };
    let coordinator = WatchQueueCoordinator::with_config(config);

    // Request up to total_capacity
    let granted = coordinator.request_capacity("watch-1", 300).await;
    assert!(granted);
    let granted = coordinator.request_capacity("watch-2", 200).await;
    assert!(granted);

    // Request more should fail (total would exceed)
    let granted = coordinator.request_capacity("watch-3", 100).await;
    assert!(!granted);
    assert_eq!(coordinator.get_allocated_capacity(), 500);
}

#[tokio::test]
async fn test_coordinator_multiple_watches() {
    let coordinator = WatchQueueCoordinator::new();

    // Multiple watches can request capacity
    assert!(coordinator.request_capacity("watch-1", 100).await);
    assert!(coordinator.request_capacity("watch-2", 200).await);
    assert!(coordinator.request_capacity("watch-3", 300).await);

    assert_eq!(coordinator.get_allocated_capacity(), 600);

    // Check summary
    let summary = coordinator.get_allocation_summary().await;
    assert_eq!(summary.num_watches, 3);
    assert_eq!(summary.allocated_capacity, 600);
    assert_eq!(summary.per_watch_allocation.len(), 3);
    assert_eq!(summary.per_watch_allocation.get("watch-1"), Some(&100));
    assert_eq!(summary.per_watch_allocation.get("watch-2"), Some(&200));
    assert_eq!(summary.per_watch_allocation.get("watch-3"), Some(&300));
}

#[tokio::test]
async fn test_coordinator_reset_watch() {
    let coordinator = WatchQueueCoordinator::new();

    coordinator.request_capacity("watch-1", 500).await;
    assert_eq!(coordinator.get_allocated_capacity(), 500);

    coordinator.reset_watch("watch-1").await;
    assert_eq!(coordinator.get_allocated_capacity(), 0);
    assert_eq!(coordinator.get_watch_allocation("watch-1").await, None);
}

#[tokio::test]
async fn test_coordinator_release_more_than_held() {
    let coordinator = WatchQueueCoordinator::new();

    coordinator.request_capacity("watch-1", 100).await;

    // Releasing more than held should only release what's held
    coordinator.release_capacity("watch-1", 500).await;
    assert_eq!(coordinator.get_allocated_capacity(), 0);
    assert_eq!(coordinator.get_watch_allocation("watch-1").await, Some(0));
}

#[tokio::test]
async fn test_coordinator_release_unknown_watch() {
    let coordinator = WatchQueueCoordinator::new();

    // Should not panic, just log a warning
    coordinator.release_capacity("unknown-watch", 100).await;
    assert_eq!(coordinator.get_allocated_capacity(), 0);
}
