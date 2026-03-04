//! Tests for file watching functionality
//!
//! Split into focused test modules:
//! - event_tests: File event detection and filtering
//! - lifecycle_tests: Project auto-watch lifecycle and restart behavior
//! - platform_tests: Platform-specific (macOS, Windows, Linux) watcher tests
//! - buffer_tests: PausedEventBuffer tests

use super::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};
use notify::EventKind;
use crate::processing::{Pipeline, TaskPriority, TaskSubmitter};

/// Create a test configuration for watching
pub(super) fn test_watcher_config() -> WatcherConfig {
    WatcherConfig {
        include_patterns: vec!["*.txt".to_string(), "*.md".to_string(), "*.rs".to_string()],
        exclude_patterns: vec!["*.tmp".to_string(), "*.swp".to_string()],
        recursive: true,
        max_depth: 3,
        debounce_ms: 100, // Short debounce for faster tests
        polling_interval_ms: 100,
        min_polling_interval_ms: 50,
        max_polling_interval_ms: 5000,
        max_queue_size: 1000,
        task_priority: TaskPriority::BackgroundWatching,
        default_collection: "test_collection".to_string(),
        process_existing: false,
        max_file_size: Some(1024 * 1024), // 1MB
        use_polling: true,
        batch_processing: BatchConfig {
            enabled: true,
            max_batch_size: 5,
            max_batch_wait_ms: 500,
            group_by_type: true,
        },
        max_debouncer_capacity: 10000,
        max_batcher_capacity: 5000,
        telemetry: TelemetryConfig {
            enabled: false, // Disable telemetry in tests
            history_retention: 10,
            collection_interval_secs: 60,
            cpu_usage: false,
            memory_usage: false,
            latency: false,
            queue_depth: false,
            throughput: false,
        },
    }
}

/// Create a test task submitter and keep the pipeline alive
pub(super) async fn create_test_task_submitter() -> (TaskSubmitter, Pipeline) {
    let mut pipeline = Pipeline::new(4);
    pipeline.start().await.expect("Failed to start test pipeline");
    let submitter = pipeline.task_submitter();
    (submitter, pipeline)
}

#[cfg(test)]
mod event_tests;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod platform_tests;
#[cfg(test)]
mod buffer_tests;
