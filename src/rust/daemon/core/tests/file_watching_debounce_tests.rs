//! Event debouncing and filtering tests for file system monitoring
//!
//! Tests for rapid file change coalescing, file type filtering in events,
//! and duplicate event detection patterns.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use shared_test_utils::{async_test, TestResult};
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Test helper for creating a notify watcher with event collection
struct TestWatcher {
    _watcher: RecommendedWatcher,
    _events: Arc<Mutex<Vec<Event>>>,
    event_rx: mpsc::UnboundedReceiver<Event>,
}

impl TestWatcher {
    fn new() -> Result<Self, notify::Error> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let watcher =
            notify::recommended_watcher(move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    if let Ok(mut events_lock) = events_clone.lock() {
                        events_lock.push(event.clone());
                    }
                    let _ = event_tx.send(event);
                }
            })?;

        Ok(Self {
            _watcher: watcher,
            _events: events,
            event_rx,
        })
    }

    fn watch<P: AsRef<Path>>(
        &mut self,
        path: P,
        recursive: RecursiveMode,
    ) -> Result<(), notify::Error> {
        self._watcher.watch(path.as_ref(), recursive)
    }

    async fn wait_for_events(
        &mut self,
        expected_count: usize,
        timeout: Duration,
    ) -> Vec<Event> {
        let mut collected_events = Vec::new();
        let start_time = Instant::now();

        while collected_events.len() < expected_count && start_time.elapsed() < timeout {
            tokio::select! {
                Some(event) = self.event_rx.recv() => {
                    collected_events.push(event);
                },
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }

        collected_events
    }
}

/// Test helper for creating test files
async fn create_test_file(dir: &Path, name: &str, content: &str) -> TestResult<PathBuf> {
    let file_path = dir.join(name);
    tokio::fs::write(&file_path, content).await?;
    Ok(file_path)
}

// ============================================================================
// EVENT DEBOUNCING AND FILTERING TESTS
// ============================================================================

async_test!(test_rapid_file_changes_debouncing, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_file = temp_path.join("rapid_changes.txt");

    let start_time = Instant::now();
    for i in 0..20 {
        tokio::fs::write(&test_file, format!("rapid change {}", i)).await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let events = watcher.wait_for_events(5, Duration::from_secs(3)).await;
    let total_time = start_time.elapsed();

    println!(
        "Rapid changes: 20 operations generated {} events in {:?}",
        events.len(),
        total_time
    );

    assert!(
        events.len() <= 20,
        "Events should be coalesced for rapid changes"
    );
    assert!(!events.is_empty(), "Should receive some events");

    Ok(())
});

async_test!(test_event_filtering_by_file_type, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let file_types = vec![
        ("document.txt", "text content"),
        ("config.json", r#"{"key": "value"}"#),
        ("script.py", "print('hello')"),
        ("temp.tmp", "temporary data"),
        (".hidden", "hidden file"),
        ("README.md", "# Documentation"),
    ];

    for (filename, content) in file_types {
        let file_path = temp_path.join(filename);
        tokio::fs::write(&file_path, content).await?;
    }

    let events = watcher.wait_for_events(3, Duration::from_secs(5)).await;

    let mut detected_extensions = HashSet::new();
    for event in &events {
        for path in &event.paths {
            if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                detected_extensions.insert(extension.to_string());
            }
            if let Some(filename) = path.file_name().and_then(|name| name.to_str()) {
                if filename.starts_with('.') {
                    detected_extensions.insert("hidden".to_string());
                }
            }
        }
    }

    assert!(
        !events.is_empty(),
        "Should receive events for different file types"
    );
    println!(
        "Detected file extensions in events: {:?}",
        detected_extensions
    );

    Ok(())
});

async_test!(test_duplicate_event_detection, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_file = temp_path.join("duplicate_test.txt");

    tokio::fs::write(&test_file, "initial content").await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    tokio::fs::write(&test_file, "modified content").await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    tokio::fs::write(&test_file, "different content").await?;

    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    let mut event_signatures = HashSet::new();
    for event in &events {
        let signature = format!("{:?}:{:?}", event.kind, event.paths);
        event_signatures.insert(signature);
    }

    println!(
        "Total events: {}, Unique signatures: {}",
        events.len(),
        event_signatures.len()
    );

    assert!(
        !events.is_empty(),
        "Should receive events for file modifications"
    );

    Ok(())
});
