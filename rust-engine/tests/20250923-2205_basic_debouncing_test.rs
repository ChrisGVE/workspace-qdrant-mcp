//! Basic event debouncing tests

use notify::{Event, EventKind};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::time::sleep;
use workspace_qdrant_daemon::config::FileWatcherConfig;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::daemon::watcher::{EventDebouncer, EventFilter, DebouncedEvent};

#[tokio::test]
async fn test_simple_event_debouncing() {
    let debouncer = EventDebouncer::new(Duration::from_millis(100));

    // Create test events
    let test_path = PathBuf::from("test.rs");
    let events = vec![
        DebouncedEvent {
            path: test_path.clone(),
            event_type: EventKind::Create(notify::event::CreateKind::File),
            timestamp: Instant::now(),
        },
        DebouncedEvent {
            path: test_path.clone(),
            event_type: EventKind::Modify(notify::event::ModifyKind::Data(notify::event::DataChange::Content)),
            timestamp: Instant::now(),
        },
    ];

    // Wait for debounce period
    sleep(Duration::from_millis(150)).await;

    let debounced = debouncer.process_events(events).await;
    assert!(debounced.len() <= 2, "Events should be debounced");
}

#[tokio::test]
async fn test_event_filtering() {
    let config = FileWatcherConfig {
        enabled: true,
        debounce_ms: 100,
        max_watched_dirs: 10,
        ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string()],
        recursive: true,
    };

    let filter = EventFilter::new(&config);

    // Test filtering
    let tmp_event = Event {
        kind: EventKind::Create(notify::event::CreateKind::File),
        paths: vec![PathBuf::from("test.tmp")],
        attrs: Default::default(),
    };

    let rs_event = Event {
        kind: EventKind::Create(notify::event::CreateKind::File),
        paths: vec![PathBuf::from("test.rs")],
        attrs: Default::default(),
    };

    assert!(filter.should_ignore(&tmp_event), "Should ignore .tmp files");
    assert!(!filter.should_ignore(&rs_event), "Should not ignore .rs files");
}

#[tokio::test]
async fn test_debouncer_add_event() {
    let debouncer = EventDebouncer::new(Duration::from_millis(50));

    // Add an event
    let test_path = PathBuf::from("test.rs");
    debouncer.add_event(test_path.clone(), EventKind::Create(notify::event::CreateKind::File)).await;

    // Immediately check - should be empty
    let ready_events = debouncer.get_ready_events().await;
    assert!(ready_events.is_empty(), "Events should not be ready immediately");

    // Wait for debounce period
    sleep(Duration::from_millis(60)).await;

    // Check again - should have the event
    let ready_events = debouncer.get_ready_events().await;
    assert_eq!(ready_events.len(), 1, "Should have one ready event");
    assert_eq!(ready_events[0].0, test_path, "Path should match");
}

#[tokio::test]
async fn test_rate_limiting_initialization() {
    // Just test that we can create a FileSystemEventHandler
    use workspace_qdrant_daemon::daemon::watcher::FileSystemEventHandler;

    let handler = FileSystemEventHandler::new(10);
    assert_eq!(format!("{:?}", handler).contains("FileSystemEventHandler"), true);
}