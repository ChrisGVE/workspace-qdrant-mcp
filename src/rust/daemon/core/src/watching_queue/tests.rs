//! Tests for the watching_queue module.

use super::*;
use tempfile::tempdir;

#[test]
fn test_get_current_branch_non_git() {
    let temp_dir = tempdir().unwrap();
    let branch = get_current_branch(temp_dir.path());
    assert_eq!(branch, "main");
}

// Multi-tenant routing tests
#[test]
fn test_watch_type_default() {
    assert_eq!(WatchType::default(), WatchType::Project);
}

#[test]
fn test_watch_type_from_str() {
    assert_eq!(WatchType::from_str("project"), Some(WatchType::Project));
    assert_eq!(WatchType::from_str("library"), Some(WatchType::Library));
    assert_eq!(WatchType::from_str("PROJECT"), Some(WatchType::Project));
    assert_eq!(WatchType::from_str("LIBRARY"), Some(WatchType::Library));
    assert_eq!(WatchType::from_str("invalid"), None);
}

#[test]
fn test_watch_type_as_str() {
    assert_eq!(WatchType::Project.as_str(), "project");
    assert_eq!(WatchType::Library.as_str(), "library");
}

#[test]
fn test_unified_collection_constants() {
    use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES};
    // Canonical collection names (without underscore prefix)
    assert_eq!(COLLECTION_PROJECTS, "projects");
    assert_eq!(COLLECTION_LIBRARIES, "libraries");
}

#[test]
fn test_determine_collection_and_tenant_project() {
    let temp_dir = tempdir().unwrap();
    let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Project,
        temp_dir.path(),
        None,
        "_old_collection",
    );

    use wqm_common::constants::COLLECTION_PROJECTS;
    assert_eq!(collection, COLLECTION_PROJECTS);
    // Tenant ID should be local_ prefixed since temp_dir is not a git repo
    assert!(tenant_id.starts_with("local_"));
}

#[test]
fn test_determine_collection_and_tenant_library_with_name() {
    let temp_dir = tempdir().unwrap();
    let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Library,
        temp_dir.path(),
        Some("my_library"),
        "_old_collection",
    );

    use wqm_common::constants::COLLECTION_LIBRARIES;
    assert_eq!(collection, COLLECTION_LIBRARIES);
    assert_eq!(tenant_id, "my_library");
}

#[test]
fn test_determine_collection_and_tenant_library_fallback_from_collection() {
    let temp_dir = tempdir().unwrap();
    let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Library,
        temp_dir.path(),
        None, // No library_name provided
        "_langchain", // Legacy collection
    );

    use wqm_common::constants::COLLECTION_LIBRARIES;
    assert_eq!(collection, COLLECTION_LIBRARIES);
    // Should extract "langchain" from "_langchain"
    assert_eq!(tenant_id, "langchain");
}

#[test]
fn test_determine_collection_and_tenant_library_fallback_from_path() {
    let temp_dir = tempdir().unwrap();
    let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Library,
        temp_dir.path(),
        None, // No library_name
        "some_collection", // No underscore prefix
    );

    use wqm_common::constants::COLLECTION_LIBRARIES;
    assert_eq!(collection, COLLECTION_LIBRARIES);
    // Should use directory name from path
    assert!(!tenant_id.is_empty());
}

// Library watch ID format tests
#[test]
fn test_library_watch_id_format() {
    let library_name = "langchain";
    let id = format!("lib_{}", library_name);

    assert!(id.starts_with("lib_"));
    assert_eq!(id, "lib_langchain");

    // Test stripping prefix
    let extracted = id.strip_prefix("lib_").unwrap_or(&id);
    assert_eq!(extracted, "langchain");
}

#[test]
fn test_library_watch_config_creation() {
    use std::path::PathBuf;
    let library_name = "my_docs";
    let id = format!("lib_{}", library_name);

    let config = WatchConfig {
        id: id.clone(),
        path: PathBuf::from("/path/to/docs"),
        collection: format!("_{}", library_name),
        patterns: vec!["*.pdf".to_string(), "*.md".to_string()],
        ignore_patterns: vec![".git/*".to_string()],
        recursive: true,
        debounce_ms: 2000,
        enabled: true,
        watch_type: WatchType::Library,
        library_name: Some(library_name.to_string()),
    };

    assert_eq!(config.watch_type, WatchType::Library);
    assert_eq!(config.library_name, Some("my_docs".to_string()));
    assert_eq!(config.collection, "_my_docs");
}

#[test]
fn test_watch_type_routing_for_library() {
    let temp_dir = tempdir().unwrap();

    // Test library routing
    let (collection, tenant) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Library,
        temp_dir.path(),
        Some("langchain"),
        "_legacy",
    );

    // Now routes to canonical `libraries` collection
    assert_eq!(collection, "libraries");
    assert_eq!(tenant, "langchain");
}

#[test]
fn test_watch_type_routing_for_project() {
    let temp_dir = tempdir().unwrap();

    // Test project routing (should use tenant ID calculation)
    let (collection, tenant) = FileWatcherQueue::determine_collection_and_tenant(
        WatchType::Project,
        temp_dir.path(),
        None,
        "_legacy",
    );

    // Now routes to canonical `projects` collection
    assert_eq!(collection, "projects");
    // Tenant should be local_ prefixed hash since temp_dir is not a git repo
    assert!(tenant.starts_with("local_"));
}

// ========== Task 461: Watch Error State Tests ==========

#[test]
fn test_watch_error_state_new() {
    let state = WatchErrorState::new();
    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.total_errors, 0);
    assert_eq!(state.backoff_level, 0);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.last_error_time.is_none());
    assert!(state.can_process());
}

#[test]
fn test_watch_error_state_record_error() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // First error - should remain healthy
    let delay = state.record_error("test error 1", &config);
    assert_eq!(state.consecutive_errors, 1);
    assert_eq!(state.total_errors, 1);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert_eq!(delay, 0); // No backoff yet

    // Third error - should become degraded
    state.record_error("test error 2", &config);
    let delay = state.record_error("test error 3", &config);
    assert_eq!(state.consecutive_errors, 3);
    assert_eq!(state.health_status, WatchHealthStatus::Degraded);
    assert_eq!(delay, 0); // Degraded but no backoff yet
}

#[test]
fn test_watch_error_state_backoff_threshold() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record errors up to backoff threshold
    for i in 1..=5 {
        let delay = state.record_error(&format!("error {}", i), &config);
        if i >= 5 {
            // Should be in backoff now
            assert_eq!(state.health_status, WatchHealthStatus::Backoff);
            assert!(delay > 0, "Should have backoff delay");
        }
    }
}

#[test]
fn test_watch_error_state_disable_threshold() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record errors up to disable threshold (Task 461.15: threshold is now 20)
    for _ in 0..20 {
        state.record_error("repeated error", &config);
    }

    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert!(state.should_disable());
    assert!(!state.can_process());
}

#[test]
fn test_watch_error_state_record_success() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Get into degraded state
    for _ in 0..3 {
        state.record_error("error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Degraded);

    // Record successes to recover
    for _ in 0..3 {
        state.record_success(&config);
    }

    // Should be fully reset after success_reset_count successes
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.backoff_level, 0);
}

#[test]
fn test_watch_error_state_reset() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record some errors
    for _ in 0..5 {
        state.record_error("error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Backoff);

    // Reset
    state.reset();

    assert_eq!(state.consecutive_errors, 0);
    assert_eq!(state.backoff_level, 0);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    // Total errors should still be tracked
    assert_eq!(state.total_errors, 5);
}

#[test]
fn test_backoff_delay_calculation() {
    let config = BackoffConfig {
        base_delay_ms: 1000,
        max_delay_ms: 60_000,
        ..BackoffConfig::default()
    };
    let mut state = WatchErrorState::new();

    // Level 0 - no delay
    assert_eq!(state.calculate_backoff_delay(&config), 0);

    // Level 1 - base delay (~1000ms with jitter)
    state.backoff_level = 1;
    let delay1 = state.calculate_backoff_delay(&config);
    assert!(delay1 >= 900 && delay1 <= 1100, "Level 1 delay should be ~1000ms, got {}", delay1);

    // Level 2 - 2x base delay (~2000ms with jitter)
    state.backoff_level = 2;
    let delay2 = state.calculate_backoff_delay(&config);
    assert!(delay2 >= 1800 && delay2 <= 2200, "Level 2 delay should be ~2000ms, got {}", delay2);

    // Level 3 - 4x base delay (~4000ms with jitter)
    state.backoff_level = 3;
    let delay3 = state.calculate_backoff_delay(&config);
    assert!(delay3 >= 3600 && delay3 <= 4400, "Level 3 delay should be ~4000ms, got {}", delay3);
}

#[test]
fn test_backoff_delay_max_cap() {
    let config = BackoffConfig {
        base_delay_ms: 1000,
        max_delay_ms: 5000,
        ..BackoffConfig::default()
    };
    let mut state = WatchErrorState::new();

    // Very high level should be capped
    state.backoff_level = 20;
    let delay = state.calculate_backoff_delay(&config);
    assert!(delay <= 5500, "Delay should be capped at max_delay + jitter, got {}", delay);
}

#[test]
fn test_watch_health_status_as_str() {
    assert_eq!(WatchHealthStatus::Healthy.as_str(), "healthy");
    assert_eq!(WatchHealthStatus::Degraded.as_str(), "degraded");
    assert_eq!(WatchHealthStatus::Backoff.as_str(), "backoff");
    assert_eq!(WatchHealthStatus::Disabled.as_str(), "disabled");
}

#[test]
fn test_backoff_config_default() {
    let config = BackoffConfig::default();
    assert_eq!(config.base_delay_ms, 1000);
    assert_eq!(config.max_delay_ms, 300_000);
    assert_eq!(config.degraded_threshold, 3);
    assert_eq!(config.backoff_threshold, 5);
    assert_eq!(config.disable_threshold, 20);  // Task 461.15: updated threshold
    assert_eq!(config.success_reset_count, 3);
    // Circuit breaker settings (Task 461.15)
    assert_eq!(config.window_error_threshold, 50);
    assert_eq!(config.window_duration_secs, 3600);
    assert_eq!(config.cooldown_secs, 3600);
    assert_eq!(config.half_open_success_threshold, 3);
}

#[tokio::test]
async fn test_watch_error_tracker_basic() {
    let tracker = WatchErrorTracker::new();

    // Record error
    let delay = tracker.record_error("watch-1", "test error").await;
    assert_eq!(delay, 0); // First error, no backoff

    // Check status
    let status = tracker.get_health_status("watch-1").await;
    assert_eq!(status, WatchHealthStatus::Healthy);

    // Record success
    tracker.record_success("watch-1").await;

    // Should still be able to process
    assert!(tracker.can_process("watch-1").await);
}

#[tokio::test]
async fn test_watch_error_tracker_multiple_watches() {
    let tracker = WatchErrorTracker::new();

    // Record errors for multiple watches
    for _ in 0..5 {
        tracker.record_error("watch-bad", "error").await;
    }
    tracker.record_error("watch-good", "single error").await;

    // Check different states
    let bad_status = tracker.get_health_status("watch-bad").await;
    let good_status = tracker.get_health_status("watch-good").await;

    assert_eq!(bad_status, WatchHealthStatus::Backoff);
    assert_eq!(good_status, WatchHealthStatus::Healthy);
}

#[tokio::test]
async fn test_watch_error_tracker_get_error_summary() {
    let tracker = WatchErrorTracker::new();

    tracker.record_error("watch-1", "error 1").await;
    tracker.record_error("watch-2", "error 2").await;
    tracker.record_error("watch-2", "error 3").await;

    let summary = tracker.get_error_summary().await;
    assert_eq!(summary.len(), 2);

    let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1").unwrap();
    assert_eq!(watch1_summary.consecutive_errors, 1);

    let watch2_summary = summary.iter().find(|s| s.watch_id == "watch-2").unwrap();
    assert_eq!(watch2_summary.consecutive_errors, 2);
}

#[tokio::test]
async fn test_watch_error_tracker_reset_watch() {
    let tracker = WatchErrorTracker::new();

    // Get into bad state (Task 461.15: threshold is now 20)
    for _ in 0..20 {
        tracker.record_error("watch-1", "error").await;
    }
    assert_eq!(tracker.get_health_status("watch-1").await, WatchHealthStatus::Disabled);

    // Reset
    tracker.reset_watch("watch-1").await;

    // Should be healthy again
    assert_eq!(tracker.get_health_status("watch-1").await, WatchHealthStatus::Healthy);
    assert!(tracker.can_process("watch-1").await);
}

#[tokio::test]
async fn test_watch_error_tracker_remove_watch() {
    let tracker = WatchErrorTracker::new();

    tracker.record_error("watch-1", "error").await;
    assert_eq!(tracker.get_error_summary().await.len(), 1);

    tracker.remove_watch("watch-1").await;
    assert_eq!(tracker.get_error_summary().await.len(), 0);
}

// ========== Watch-Queue Coordinator Tests (Task 461.9) ==========

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

// ========== Circuit Breaker Tests (Task 461.15) ==========

#[test]
fn test_circuit_breaker_config_defaults() {
    let config = BackoffConfig::default();
    assert_eq!(config.disable_threshold, 20);
    assert_eq!(config.window_error_threshold, 50);
    assert_eq!(config.window_duration_secs, 3600);
    assert_eq!(config.cooldown_secs, 3600);
    assert_eq!(config.half_open_success_threshold, 3);
}

#[test]
fn test_circuit_breaker_opens_on_consecutive_errors() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record 19 errors - should not open circuit yet
    for _ in 0..19 {
        state.record_error("test error", &config);
    }
    assert_ne!(state.health_status, WatchHealthStatus::Disabled);

    // 20th error should open circuit
    state.record_error("test error", &config);
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert!(state.circuit_opened_at.is_some());
}

#[test]
fn test_circuit_breaker_state_info() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Initially closed
    let circuit_state = state.get_circuit_state();
    assert!(!circuit_state.is_open);
    assert!(!circuit_state.is_half_open);

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }

    let circuit_state = state.get_circuit_state();
    assert!(circuit_state.is_open);
    assert!(!circuit_state.is_half_open);
    assert!(circuit_state.opened_at.is_some());
    assert_eq!(circuit_state.errors_in_window, 20);
}

#[test]
fn test_half_open_state_transition() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0; // Immediate cooldown for testing
    let mut state = WatchErrorState::new();

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);

    // Should transition to half-open after cooldown
    assert!(state.should_attempt_half_open(&config));
    state.transition_to_half_open();
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

    let circuit_state = state.get_circuit_state();
    assert!(!circuit_state.is_open);
    assert!(circuit_state.is_half_open);
}

#[test]
fn test_half_open_error_reopens_circuit() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0;
    let mut state = WatchErrorState::new();

    // Open and transition to half-open
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    state.transition_to_half_open();
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

    // Error in half-open should reopen circuit
    state.record_error("retry failed", &config);
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);
    assert_eq!(state.half_open_attempts, 1);
}

#[test]
fn test_half_open_success_closes_circuit() {
    let mut config = BackoffConfig::default();
    config.cooldown_secs = 0;
    config.half_open_success_threshold = 2;
    let mut state = WatchErrorState::new();

    // Open and transition to half-open
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    state.transition_to_half_open();

    // First success - still half-open
    let changed = state.record_success(&config);
    assert!(!changed);
    assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);
    assert_eq!(state.half_open_successes, 1);

    // Second success - closes circuit
    let changed = state.record_success(&config);
    assert!(changed);
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.circuit_opened_at.is_none());
}

#[test]
fn test_manual_circuit_reset() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Open the circuit
    for _ in 0..20 {
        state.record_error("test error", &config);
    }
    assert_eq!(state.health_status, WatchHealthStatus::Disabled);

    // Manual reset
    state.manual_circuit_reset();
    assert_eq!(state.health_status, WatchHealthStatus::Healthy);
    assert!(state.circuit_opened_at.is_none());
    assert_eq!(state.consecutive_errors, 0);
    assert!(state.errors_in_window.is_empty());
}

#[test]
fn test_errors_in_window_tracking() {
    let config = BackoffConfig::default();
    let mut state = WatchErrorState::new();

    // Record some errors
    for _ in 0..10 {
        state.record_error("test error", &config);
    }

    let circuit_state = state.get_circuit_state();
    assert_eq!(circuit_state.errors_in_window, 10);
    assert_eq!(state.errors_in_window.len(), 10);
}

#[test]
fn test_watch_health_status_half_open_as_str() {
    assert_eq!(WatchHealthStatus::HalfOpen.as_str(), "half_open");
}

// ========== Processing Error Feedback Tests (Task 461.13) ==========

#[test]
fn test_processing_error_type_as_str() {
    assert_eq!(ProcessingErrorType::FileNotFound.as_str(), "file_not_found");
    assert_eq!(ProcessingErrorType::ParsingError.as_str(), "parsing_error");
    assert_eq!(ProcessingErrorType::QdrantError.as_str(), "qdrant_error");
    assert_eq!(ProcessingErrorType::EmbeddingError.as_str(), "embedding_error");
    assert_eq!(ProcessingErrorType::Unknown.as_str(), "unknown");
}

#[test]
fn test_processing_error_type_from_str() {
    assert_eq!(ProcessingErrorType::from_str("file_not_found"), ProcessingErrorType::FileNotFound);
    assert_eq!(ProcessingErrorType::from_str("parsing_error"), ProcessingErrorType::ParsingError);
    assert_eq!(ProcessingErrorType::from_str("qdrant_error"), ProcessingErrorType::QdrantError);
    assert_eq!(ProcessingErrorType::from_str("embedding_error"), ProcessingErrorType::EmbeddingError);
    assert_eq!(ProcessingErrorType::from_str("other"), ProcessingErrorType::Unknown);
}

#[test]
fn test_processing_error_type_should_skip_permanently() {
    assert!(ProcessingErrorType::FileNotFound.should_skip_permanently());
    assert!(!ProcessingErrorType::ParsingError.should_skip_permanently());
    assert!(!ProcessingErrorType::QdrantError.should_skip_permanently());
    assert!(!ProcessingErrorType::EmbeddingError.should_skip_permanently());
    assert!(!ProcessingErrorType::Unknown.should_skip_permanently());
}

#[test]
fn test_processing_error_feedback_new() {
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::ParsingError,
        "Failed to parse file"
    );

    assert_eq!(feedback.watch_id, "watch-1");
    assert_eq!(feedback.file_path, "/path/to/file.txt");
    assert_eq!(feedback.error_type, ProcessingErrorType::ParsingError);
    assert_eq!(feedback.error_message, "Failed to parse file");
    assert!(feedback.queue_item_id.is_none());
    assert!(feedback.context.is_empty());
}

#[test]
fn test_processing_error_feedback_with_context() {
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::EmbeddingError,
        "Embedding failed"
    )
    .with_queue_item_id("queue-123")
    .with_context("chunk_index", "5")
    .with_context("model", "all-MiniLM-L6-v2");

    assert_eq!(feedback.queue_item_id, Some("queue-123".to_string()));
    assert_eq!(feedback.context.get("chunk_index"), Some(&"5".to_string()));
    assert_eq!(feedback.context.get("model"), Some(&"all-MiniLM-L6-v2".to_string()));
}

#[tokio::test]
async fn test_error_feedback_manager_record_and_query() {
    let manager = ErrorFeedbackManager::new();

    // Record an error
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/path/to/file.txt",
        ProcessingErrorType::ParsingError,
        "Parse error"
    );
    manager.record_error(feedback).await;

    // Query recent errors
    let errors = manager.get_recent_errors("watch-1").await;
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].file_path, "/path/to/file.txt");
}

#[tokio::test]
async fn test_error_feedback_manager_permanent_skip() {
    let manager = ErrorFeedbackManager::new();

    // Record FileNotFound - should add to permanent skip
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/missing/file.txt",
        ProcessingErrorType::FileNotFound,
        "File not found"
    );
    manager.record_error(feedback).await;

    // Check if file is skipped
    assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);
    assert!(!manager.should_skip_file("watch-1", "/other/file.txt").await);
    assert!(!manager.should_skip_file("watch-2", "/missing/file.txt").await);
}

#[tokio::test]
async fn test_error_feedback_manager_error_counts() {
    let manager = ErrorFeedbackManager::new();

    // Record multiple errors of different types
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-1", "file1.txt", ProcessingErrorType::ParsingError, "error"
    )).await;
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-1", "file2.txt", ProcessingErrorType::ParsingError, "error"
    )).await;
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-1", "file3.txt", ProcessingErrorType::QdrantError, "error"
    )).await;

    let counts = manager.get_error_counts("watch-1").await;
    assert_eq!(counts.get(&ProcessingErrorType::ParsingError), Some(&2));
    assert_eq!(counts.get(&ProcessingErrorType::QdrantError), Some(&1));
    assert_eq!(counts.get(&ProcessingErrorType::FileNotFound), None);
}

#[tokio::test]
async fn test_error_feedback_manager_remove_skip() {
    let manager = ErrorFeedbackManager::new();

    // Add to skip list
    let feedback = ProcessingErrorFeedback::new(
        "watch-1",
        "/missing/file.txt",
        ProcessingErrorType::FileNotFound,
        "Not found"
    );
    manager.record_error(feedback).await;
    assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);

    // Remove from skip list
    let removed = manager.remove_skip("watch-1", "/missing/file.txt").await;
    assert!(removed);
    assert!(!manager.should_skip_file("watch-1", "/missing/file.txt").await);
}

#[tokio::test]
async fn test_error_feedback_manager_clear_skips() {
    let manager = ErrorFeedbackManager::new();

    // Add multiple files to skip list
    for i in 0..5 {
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            format!("/missing/file{}.txt", i),
            ProcessingErrorType::FileNotFound,
            "Not found"
        );
        manager.record_error(feedback).await;
    }

    let skipped = manager.get_skipped_files("watch-1").await;
    assert_eq!(skipped.len(), 5);

    // Clear all skips
    manager.clear_skips("watch-1").await;
    let skipped = manager.get_skipped_files("watch-1").await;
    assert!(skipped.is_empty());
}

#[tokio::test]
async fn test_error_feedback_manager_summary() {
    let manager = ErrorFeedbackManager::new();

    // Add errors for multiple watches
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-1", "file1.txt", ProcessingErrorType::ParsingError, "error"
    )).await;
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-1", "file2.txt", ProcessingErrorType::FileNotFound, "error"
    )).await;
    manager.record_error(ProcessingErrorFeedback::new(
        "watch-2", "file3.txt", ProcessingErrorType::QdrantError, "error"
    )).await;

    let summary = manager.get_processing_error_summary().await;
    assert_eq!(summary.len(), 2);

    let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1");
    assert!(watch1_summary.is_some());
    let watch1 = watch1_summary.unwrap();
    assert_eq!(watch1.recent_error_count, 2);
    assert_eq!(watch1.skipped_file_count, 1); // FileNotFound adds to skip
}

#[tokio::test]
async fn test_error_feedback_manager_max_recent() {
    let manager = ErrorFeedbackManager::new().with_max_recent(3);

    // Add more errors than max
    for i in 0..5 {
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1",
            format!("file{}.txt", i),
            ProcessingErrorType::ParsingError,
            format!("error {}", i)
        )).await;
    }

    let errors = manager.get_recent_errors("watch-1").await;
    assert_eq!(errors.len(), 3); // Should be capped at max
    // Should have the most recent 3 (indices 2, 3, 4)
    assert!(errors.iter().any(|e| e.file_path == "file2.txt"));
    assert!(errors.iter().any(|e| e.file_path == "file3.txt"));
    assert!(errors.iter().any(|e| e.file_path == "file4.txt"));
}
