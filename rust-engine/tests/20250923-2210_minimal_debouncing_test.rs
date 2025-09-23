//! Minimal working test for event debouncing functionality

use std::time::Duration;
use workspace_qdrant_daemon::daemon::watcher::EventDebouncer;

#[tokio::test]
async fn test_event_debouncer_creation() {
    // Test that we can create an EventDebouncer
    let debouncer = EventDebouncer::new(Duration::from_millis(100));

    // Verify it can be debugged (tests Debug trait)
    let debug_str = format!("{:?}", debouncer);
    assert!(debug_str.contains("EventDebouncer"));
}

#[tokio::test]
async fn test_event_filter_creation() {
    use workspace_qdrant_daemon::config::FileWatcherConfig;
    use workspace_qdrant_daemon::daemon::watcher::EventFilter;

    let config = FileWatcherConfig {
        enabled: true,
        debounce_ms: 100,
        max_watched_dirs: 10,
        ignore_patterns: vec!["*.tmp".to_string()],
        recursive: true,
    };

    let filter = EventFilter::new(&config);
    let debug_str = format!("{:?}", filter);
    assert!(debug_str.contains("EventFilter"));
}

#[tokio::test]
async fn test_file_system_event_handler_creation() {
    use workspace_qdrant_daemon::daemon::watcher::FileSystemEventHandler;

    let handler = FileSystemEventHandler::new(10);
    let debug_str = format!("{:?}", handler);
    assert!(debug_str.contains("FileSystemEventHandler"));
}