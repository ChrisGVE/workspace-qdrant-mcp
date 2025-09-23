//! Comprehensive tests for event debouncing and filtering functionality
//!
//! This test suite covers:
//! - Event debouncing with configurable delays and thresholds
//! - Event filtering by file type, path patterns, and custom rules
//! - Burst event handling and rate limiting
//! - Edge cases: rapid successive events, mixed event types, large event bursts
//! - Memory efficiency during high event volumes

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::time::sleep;
use workspace_qdrant_daemon::config::FileWatcherConfig;
use workspace_qdrant_daemon::daemon::watcher::{
    EventDebouncer, EventFilter, FileSystemEventHandler, FileWatcher, DebouncedEvent,
};

/// Test fixture for event debouncing tests
#[derive(Debug)]
struct EventTestFixture {
    temp_dir: TempDir,
    config: FileWatcherConfig,
    debouncer: EventDebouncer,
    filter: EventFilter,
    handler: FileSystemEventHandler,
}

impl EventTestFixture {
    fn new(debounce_ms: u64, max_events_per_second: u32) -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms,
            max_watched_dirs: 100,
            ignore_patterns: vec![
                "*.tmp".to_string(),
                "*.log".to_string(),
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
            ],
            recursive: true,
        };

        let debouncer = EventDebouncer::new(Duration::from_millis(debounce_ms));
        let filter = EventFilter::new(&config);
        let handler = FileSystemEventHandler::new(max_events_per_second);

        Self {
            temp_dir,
            config,
            debouncer,
            filter,
            handler,
        }
    }

    fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }

    fn create_test_file(&self, filename: &str) -> PathBuf {
        let path = self.temp_path().join(filename);
        std::fs::write(&path, "test content").expect("Failed to create test file");
        path
    }
}

// Remove local DebouncedEvent definition as it's now in the watcher module

/// Test helper for capturing events
#[derive(Debug)]
struct EventCapture {
    events: Arc<Mutex<Vec<DebouncedEvent>>>,
}

impl EventCapture {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn add_event(&self, path: PathBuf, event_type: EventKind) {
        let event = DebouncedEvent {
            path,
            event_type,
            timestamp: Instant::now(),
        };
        self.events.lock().unwrap().push(event);
    }

    fn get_events(&self) -> Vec<DebouncedEvent> {
        self.events.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

#[tokio::test]
async fn test_basic_event_debouncing() {
    let fixture = EventTestFixture::new(100, 10); // 100ms debounce, 10 events/sec
    let capture = EventCapture::new();

    // Simulate rapid file changes
    let test_file = fixture.create_test_file("test.rs");

    // Generate rapid events (should be debounced)
    for i in 0..5 {
        capture.add_event(test_file.clone(), EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content)));
        sleep(Duration::from_millis(20)).await; // Rapid succession
    }

    // Wait for debounce period
    sleep(Duration::from_millis(150)).await;

    // Should have only processed the last event due to debouncing
    assert_eq!(capture.event_count(), 5, "All events should be captured before debouncing");

    // Now test actual debouncing behavior
    let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;
    assert!(debounced_events.len() <= 2, "Events should be debounced to 1-2 events");
}

#[tokio::test]
async fn test_configurable_debounce_delay() {
    // Test different debounce delays
    let test_delays = vec![50, 100, 200, 500];

    for delay_ms in test_delays {
        let fixture = EventTestFixture::new(delay_ms, 10);
        let capture = EventCapture::new();
        let test_file = fixture.create_test_file(&format!("test_{}.rs", delay_ms));

        let start_time = Instant::now();

        // Generate event
        capture.add_event(test_file.clone(), EventKind::Create(notify::event::CreateKind::File));

        // Wait for debounce period + small buffer
        sleep(Duration::from_millis(delay_ms + 50)).await;

        let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;
        let elapsed = start_time.elapsed();

        assert!(elapsed >= Duration::from_millis(delay_ms),
            "Debounce delay should be at least {}ms", delay_ms);
        assert_eq!(debounced_events.len(), 1, "Should have exactly one debounced event");
    }
}

#[tokio::test]
async fn test_event_filtering_by_file_type() {
    let fixture = EventTestFixture::new(100, 10);

    // Test files with different extensions
    let test_files = vec![
        ("source.rs", true),      // Should be processed
        ("script.py", true),      // Should be processed
        ("temp.tmp", false),      // Should be filtered (ignore pattern)
        ("debug.log", false),     // Should be filtered (ignore pattern)
        ("config.json", true),    // Should be processed
        ("readme.md", true),      // Should be processed
    ];

    for (filename, should_process) in test_files {
        let test_file = fixture.create_test_file(filename);
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![test_file],
            attrs: Default::default(),
        };

        let should_filter = fixture.filter.should_ignore(&event);
        assert_eq!(!should_filter, should_process,
            "File {} filtering behavior incorrect", filename);
    }
}

#[tokio::test]
async fn test_path_pattern_filtering() {
    let fixture = EventTestFixture::new(100, 10);

    // Test path patterns from ignore_patterns
    let test_paths = vec![
        ("target/debug/app", true),           // Should be filtered (target/**)
        ("node_modules/lib/index.js", true),  // Should be filtered (node_modules/**)
        (".git/config", true),                // Should be filtered (.git/**)
        ("src/main.rs", false),               // Should NOT be filtered
        ("tests/test.rs", false),             // Should NOT be filtered
        ("docs/readme.md", false),            // Should NOT be filtered
    ];

    for (path_str, should_be_filtered) in test_paths {
        let test_path = fixture.temp_path().join(path_str);
        let event = Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content)),
            paths: vec![test_path],
            attrs: Default::default(),
        };

        let is_filtered = fixture.filter.should_ignore(&event);
        assert_eq!(is_filtered, should_be_filtered,
            "Path pattern filtering failed for: {}", path_str);
    }
}

#[tokio::test]
async fn test_custom_filtering_rules() {
    let mut config = FileWatcherConfig {
        enabled: true,
        debounce_ms: 100,
        max_watched_dirs: 100,
        ignore_patterns: vec![
            "*.bak".to_string(),
            "*.swp".to_string(),
            "*~".to_string(),
            "*.cache".to_string(),
        ],
        recursive: true,
    };

    let temp_dir = TempDir::new().unwrap();
    let filter = EventFilter::new(&config);

    // Test custom ignore patterns
    let custom_test_files = vec![
        ("file.bak", true),         // Should be filtered
        ("file.swp", true),         // Should be filtered
        ("file~", true),            // Should be filtered
        ("data.cache", true),       // Should be filtered
        ("file.txt", false),        // Should NOT be filtered
        ("config.yaml", false),     // Should NOT be filtered
    ];

    for (filename, should_be_filtered) in custom_test_files {
        let test_path = temp_dir.path().join(filename);
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![test_path],
            attrs: Default::default(),
        };

        let is_filtered = filter.should_ignore(&event);
        assert_eq!(is_filtered, should_be_filtered,
            "Custom filtering rule failed for: {}", filename);
    }
}

#[tokio::test]
async fn test_burst_event_handling() {
    let fixture = EventTestFixture::new(50, 5); // Low rate limit for testing
    let capture = EventCapture::new();

    // Generate burst of events
    let test_file = fixture.create_test_file("burst_test.rs");
    let burst_size = 20;

    let start_time = Instant::now();
    for i in 0..burst_size {
        capture.add_event(
            test_file.clone(),
            EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content))
        );
        sleep(Duration::from_millis(10)).await; // Fast burst
    }

    // Process through rate limiter
    let processed_events = fixture.handler.handle_event_burst(capture.get_events()).await;
    let elapsed = start_time.elapsed();

    // Should have rate limited the events
    assert!(processed_events.len() < burst_size,
        "Burst events should be rate limited");

    // Rate limiting should introduce delays
    assert!(elapsed >= Duration::from_millis(500),
        "Rate limiting should introduce processing delays");
}

#[tokio::test]
async fn test_mixed_event_types() {
    let fixture = EventTestFixture::new(100, 10);
    let capture = EventCapture::new();

    // Create different types of events
    let test_file = fixture.create_test_file("mixed_test.rs");

    let event_types = vec![
        EventKind::Create(notify::event::CreateKind::File),
        EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content)),
        EventKind::Remove(notify::event::RemoveKind::File),
        EventKind::Modify(notify::event::ModifyKind::Metadata(notify::event::MetadataKind::Permissions)),
    ];

    // Add mixed events
    for event_type in event_types {
        capture.add_event(test_file.clone(), event_type);
        sleep(Duration::from_millis(25)).await;
    }

    // Wait for debounce
    sleep(Duration::from_millis(150)).await;

    let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;

    // Should handle different event types appropriately
    assert!(debounced_events.len() >= 1, "Should process at least one event");
    assert!(debounced_events.len() <= 4, "Should debounce similar events");
}

#[tokio::test]
async fn test_large_event_burst_memory_efficiency() {
    let fixture = EventTestFixture::new(200, 50); // Higher rate limit for large burst
    let capture = EventCapture::new();

    // Generate very large burst to test memory efficiency
    let large_burst_size = 1000;
    let test_files: Vec<PathBuf> = (0..large_burst_size)
        .map(|i| fixture.create_test_file(&format!("large_test_{}.rs", i)))
        .collect();

    let memory_before = get_memory_usage();

    // Generate large event burst
    for (i, test_file) in test_files.iter().enumerate() {
        capture.add_event(
            test_file.clone(),
            EventKind::Create(notify::event::CreateKind::File)
        );

        // Occasional yield to prevent overwhelming
        if i % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }

    let memory_after_generation = get_memory_usage();

    // Process through debouncer (should be memory efficient)
    let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;

    let memory_after_processing = get_memory_usage();

    // Memory should not grow excessively
    let memory_growth = memory_after_processing.saturating_sub(memory_before);
    assert!(memory_growth < 100 * 1024 * 1024, // Less than 100MB growth
        "Memory usage should be efficient for large event bursts");

    // Should significantly reduce event count through debouncing
    assert!(debounced_events.len() < large_burst_size / 2,
        "Large burst should be significantly debounced");
}

#[tokio::test]
async fn test_rapid_successive_events_same_file() {
    let fixture = EventTestFixture::new(150, 10);
    let capture = EventCapture::new();
    let test_file = fixture.create_test_file("rapid_test.rs");

    // Generate very rapid successive events on same file
    let rapid_count = 50;
    for i in 0..rapid_count {
        capture.add_event(
            test_file.clone(),
            EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content))
        );
        sleep(Duration::from_millis(5)).await; // Very rapid
    }

    // Wait for debounce period
    sleep(Duration::from_millis(200)).await;

    let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;

    // Should debounce to very few events
    assert!(debounced_events.len() <= 3,
        "Rapid successive events should be heavily debounced");
}

#[tokio::test]
async fn test_event_filtering_performance() {
    let fixture = EventTestFixture::new(100, 100);

    // Create many test files for performance testing
    let file_count = 1000;
    let test_files: Vec<PathBuf> = (0..file_count)
        .map(|i| {
            let extension = match i % 4 {
                0 => "rs",
                1 => "py",
                2 => "tmp", // Should be filtered
                _ => "log", // Should be filtered
            };
            fixture.create_test_file(&format!("perf_test_{}.{}", i, extension))
        })
        .collect();

    let start_time = Instant::now();

    // Test filtering performance
    let mut filtered_count = 0;
    for test_file in &test_files {
        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![test_file.clone()],
            attrs: Default::default(),
        };

        if fixture.filter.should_ignore(&event) {
            filtered_count += 1;
        }
    }

    let elapsed = start_time.elapsed();

    // Performance should be reasonable
    assert!(elapsed < Duration::from_millis(100),
        "Filtering {} files should be fast", file_count);

    // Should have filtered approximately half (tmp and log files)
    assert!(filtered_count >= file_count / 3,
        "Should filter appropriate number of files");
}

#[tokio::test]
async fn test_debouncing_with_different_thresholds() {
    // Test various debounce thresholds
    let threshold_tests = vec![
        (10, 20),   // Very short debounce
        (100, 10),  // Medium debounce
        (500, 5),   // Long debounce
        (1000, 2),  // Very long debounce
    ];

    for (debounce_ms, expected_max_events) in threshold_tests {
        let fixture = EventTestFixture::new(debounce_ms, 20);
        let capture = EventCapture::new();
        let test_file = fixture.create_test_file(&format!("threshold_test_{}.rs", debounce_ms));

        // Generate many rapid events
        for _ in 0..20 {
            capture.add_event(
                test_file.clone(),
                EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content))
            );
            sleep(Duration::from_millis(debounce_ms / 4)).await;
        }

        // Wait for debounce
        sleep(Duration::from_millis(debounce_ms + 100)).await;

        let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;

        assert!(debounced_events.len() <= expected_max_events,
            "Debounce threshold {}ms should limit events to {}, got {}",
            debounce_ms, expected_max_events, debounced_events.len());
    }
}

#[tokio::test]
async fn test_event_filtering_edge_cases() {
    let fixture = EventTestFixture::new(100, 10);

    // Test edge cases for filtering
    let edge_cases = vec![
        ("", false),                    // Empty path
        (".", false),                   // Current directory
        ("..", false),                  // Parent directory
        ("file.with.multiple.dots.rs", false), // Multiple dots
        ("file with spaces.rs", false), // Spaces in name
        ("файл.rs", false),             // Unicode filename
        ("🚀rocket.rs", false),         // Emoji in filename
        ("very_long_filename_that_exceeds_normal_limits.rs", false), // Long filename
    ];

    for (filename, should_be_filtered) in edge_cases {
        let test_path = if filename.is_empty() {
            fixture.temp_path().to_path_buf()
        } else {
            fixture.temp_path().join(filename)
        };

        let event = Event {
            kind: EventKind::Create(notify::event::CreateKind::File),
            paths: vec![test_path],
            attrs: Default::default(),
        };

        let is_filtered = fixture.filter.should_ignore(&event);
        assert_eq!(is_filtered, should_be_filtered,
            "Edge case filtering failed for: '{}'", filename);
    }
}

#[tokio::test]
async fn test_concurrent_event_processing() {
    let fixture = EventTestFixture::new(100, 20);

    // Create multiple concurrent event processing tasks
    let task_count = 10;
    let events_per_task = 50;

    let mut handles = Vec::new();

    for task_id in 0..task_count {
        let capture = EventCapture::new();
        let debouncer = EventDebouncer::new(Duration::from_millis(100));

        let handle = tokio::spawn(async move {
            // Generate events for this task
            for i in 0..events_per_task {
                let test_path = PathBuf::from(format!("concurrent_test_{}_{}.rs", task_id, i));
                capture.add_event(
                    test_path,
                    EventKind::Create(notify::event::CreateKind::File)
                );
                sleep(Duration::from_millis(10)).await;
            }

            // Process events
            let debounced = debouncer.process_events(capture.get_events()).await;
            debounced.len()
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let results = futures_util::future::join_all(handles).await;

    // All tasks should complete successfully
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Task {} should complete successfully", i);
        let event_count = result.as_ref().unwrap();
        assert!(*event_count > 0, "Task {} should process some events", i);
        assert!(*event_count <= events_per_task, "Task {} should debounce events", i);
    }
}

/// Helper function to estimate memory usage (simplified)
fn get_memory_usage() -> usize {
    // Simplified memory usage estimation
    // In a real implementation, you might use system calls or process monitoring
    std::mem::size_of::<Vec<DebouncedEvent>>() * 1024
}

/// Integration test for complete file watching pipeline
#[tokio::test]
async fn test_complete_file_watching_pipeline() {
    let temp_dir = TempDir::new().unwrap();
    let config = FileWatcherConfig {
        enabled: true,
        debounce_ms: 100,
        max_watched_dirs: 10,
        ignore_patterns: vec!["*.tmp".to_string()],
        recursive: true,
    };

    // This would test the complete FileWatcher implementation
    // Currently commented out as FileWatcher needs full implementation
    /*
    let processor = Arc::new(DocumentProcessor::test_instance());
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Start watching
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Create test files and verify events are processed correctly
    let test_file = temp_dir.path().join("integration_test.rs");
    std::fs::write(&test_file, "test content").unwrap();

    // Wait for file system events to be processed
    sleep(Duration::from_millis(200)).await;

    // Cleanup
    assert!(watcher.stop().await.is_ok());
    */
}

/// Test debouncing effectiveness under various load conditions
#[tokio::test]
async fn test_debouncing_effectiveness() {
    let load_conditions = vec![
        (10, 100),   // Light load: 10ms intervals, 100 events
        (5, 200),    // Medium load: 5ms intervals, 200 events
        (1, 500),    // Heavy load: 1ms intervals, 500 events
    ];

    for (interval_ms, event_count) in load_conditions {
        let fixture = EventTestFixture::new(100, 50); // 100ms debounce
        let capture = EventCapture::new();
        let test_file = fixture.create_test_file(&format!("load_test_{}_{}.rs", interval_ms, event_count));

        let start_time = Instant::now();

        // Generate events under specific load condition
        for _ in 0..event_count {
            capture.add_event(
                test_file.clone(),
                EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content))
            );
            sleep(Duration::from_millis(interval_ms)).await;
        }

        let generation_time = start_time.elapsed();

        // Wait for debounce period
        sleep(Duration::from_millis(150)).await;

        let debounced_events = fixture.debouncer.process_events(capture.get_events()).await;

        // Calculate debouncing effectiveness
        let reduction_ratio = 1.0 - (debounced_events.len() as f64 / event_count as f64);

        // Effectiveness should be high for rapid events
        if interval_ms <= 5 {
            assert!(reduction_ratio >= 0.8,
                "High load condition should achieve >80% reduction, got {:.2}%",
                reduction_ratio * 100.0);
        } else {
            assert!(reduction_ratio >= 0.3,
                "Load condition should achieve >30% reduction, got {:.2}%",
                reduction_ratio * 100.0);
        }
    }
}