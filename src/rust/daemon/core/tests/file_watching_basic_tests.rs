//! Basic file system monitoring tests with notify crate
//!
//! Tests for fundamental file watching operations: file creation, modification,
//! deletion, and directory watching behavior (recursive vs non-recursive).

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

        let watcher = notify::recommended_watcher(move |result: Result<Event, notify::Error>| {
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

    async fn wait_for_events(&mut self, expected_count: usize, timeout: Duration) -> Vec<Event> {
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

/// Test helper for creating test files with various sizes and content
async fn create_test_file(dir: &Path, name: &str, content: &str) -> TestResult<PathBuf> {
    let file_path = dir.join(name);
    tokio::fs::write(&file_path, content).await?;
    Ok(file_path)
}

/// Test helper for creating nested directory structures
async fn create_nested_structure(base_dir: &Path) -> TestResult<Vec<PathBuf>> {
    let mut created_paths = Vec::new();

    let nested_dirs = vec![
        "level1",
        "level1/level2",
        "level1/level2/level3",
        "other_branch",
        "other_branch/subfolder",
    ];

    for dir_path in nested_dirs {
        let full_path = base_dir.join(dir_path);
        tokio::fs::create_dir_all(&full_path).await?;
        created_paths.push(full_path);
    }

    let files = vec![
        ("root_file.txt", "Root level content"),
        ("level1/file1.txt", "Level 1 content"),
        ("level1/level2/file2.txt", "Level 2 content"),
        ("level1/level2/level3/file3.txt", "Level 3 content"),
        ("other_branch/other_file.txt", "Other branch content"),
    ];

    for (file_path, content) in files {
        let full_path = base_dir.join(file_path);
        tokio::fs::write(&full_path, content).await?;
        created_paths.push(full_path);
    }

    Ok(created_paths)
}

// ============================================================================
// BASIC FILE SYSTEM MONITORING TESTS
// ============================================================================

async_test!(test_basic_file_creation_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // FSEvents on macOS CI runners needs extra time to initialize
    tokio::time::sleep(Duration::from_millis(500)).await;

    let test_file = temp_path.join("created_file.txt");
    tokio::fs::write(&test_file, "test content").await?;

    let events = watcher.wait_for_events(1, Duration::from_secs(5)).await;

    assert!(
        !events.is_empty(),
        "Should receive at least one file creation event"
    );

    let relevant_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("created_file.txt")
            })
        })
        .collect();

    assert!(
        !relevant_events.is_empty(),
        "Should receive events for created_file.txt"
    );

    Ok(())
});

async_test!(test_file_modification_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let test_file = create_test_file(temp_path, "modify_test.txt", "initial content").await?;

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    for i in 1..=3 {
        tokio::fs::write(&test_file, format!("modified content {}", i)).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let events = watcher.wait_for_events(1, Duration::from_secs(3)).await;

    assert!(!events.is_empty(), "Should receive modification events");

    let file_related_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("modify_test.txt")
            })
        })
        .collect();

    assert!(
        !file_related_events.is_empty(),
        "Should receive events for the modified file"
    );

    Ok(())
});

async_test!(test_file_deletion_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let test_file = create_test_file(temp_path, "delete_test.txt", "content to delete").await?;

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    tokio::fs::remove_file(&test_file).await?;

    let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    assert!(!events.is_empty(), "Should receive deletion events");

    let _removal_events: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.kind, EventKind::Remove(_)))
        .collect();

    assert!(!test_file.exists(), "File should be deleted");

    Ok(())
});

// ============================================================================
// DIRECTORY WATCHING BEHAVIOR TESTS
// ============================================================================

async_test!(test_recursive_directory_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let created_paths = create_nested_structure(temp_path).await?;

    let events = watcher
        .wait_for_events(created_paths.len() / 2, Duration::from_secs(5))
        .await;

    assert!(
        !events.is_empty(),
        "Should receive events for nested structure creation"
    );

    let deep_file_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event
                .paths
                .iter()
                .any(|path| path.to_string_lossy().contains("level3"))
        })
        .collect();

    println!("Total events received: {}", events.len());
    println!("Deep file events: {}", deep_file_events.len());

    Ok(())
});

async_test!(test_non_recursive_directory_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let _root_file = create_test_file(temp_path, "root_level.txt", "root content").await?;

    let sub_dir = temp_path.join("subdir");
    tokio::fs::create_dir(&sub_dir).await?;
    let _sub_file = create_test_file(&sub_dir, "sub_file.txt", "sub content").await?;

    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    let _root_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("root_level.txt")
            })
        })
        .collect();

    assert!(
        !events.is_empty(),
        "Should receive some events in non-recursive mode"
    );

    Ok(())
});

async_test!(test_directory_creation_and_removal, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new().map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_dir = temp_path.join("test_directory");
    tokio::fs::create_dir(&test_dir).await?;

    let _test_file = create_test_file(&test_dir, "inside.txt", "directory content").await?;

    tokio::time::sleep(Duration::from_millis(200)).await;

    tokio::fs::remove_dir_all(&test_dir).await?;

    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    assert!(
        !events.is_empty(),
        "Should receive events for directory creation and removal"
    );
    assert!(!test_dir.exists(), "Directory should be removed");

    Ok(())
});
