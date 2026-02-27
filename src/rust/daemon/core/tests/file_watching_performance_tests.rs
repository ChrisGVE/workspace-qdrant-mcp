//! Memory usage and performance tests for file system monitoring
//!
//! Tests for watcher behavior under load: many files, performance baselines,
//! concurrent file operations, and large directory tree watching.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
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
// MEMORY USAGE AND PERFORMANCE TESTS
// ============================================================================

async_test!(test_memory_usage_with_many_files, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let file_count = 500;
    let start_time = Instant::now();

    for i in 0..file_count {
        let file_path = temp_path.join(format!("memory_test_{:04}.txt", i));
        tokio::fs::write(&file_path, format!("Content for file {}", i)).await?;

        if i % 50 == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    let creation_time = start_time.elapsed();

    let events = watcher
        .wait_for_events(file_count / 4, Duration::from_secs(10))
        .await;

    let mut read_dir = tokio::fs::read_dir(temp_path).await?;
    let mut entry_count = 0;
    while let Some(_entry) = read_dir.next_entry().await? {
        entry_count += 1;
    }

    assert_eq!(entry_count, file_count, "All files should be created");
    assert!(
        !events.is_empty(),
        "Should receive events for file creation"
    );

    println!(
        "Created {} files in {:?}, received {} events",
        file_count,
        creation_time,
        events.len()
    );

    let _additional_file =
        create_test_file(temp_path, "additional_test.txt", "additional content").await?;
    let additional_events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    assert!(
        !additional_events.is_empty() || events.len() > file_count / 8,
        "Watcher should remain responsive after processing many files"
    );

    Ok(())
});

async_test!(test_watcher_performance_baseline, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let setup_start = Instant::now();
    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;
    let setup_duration = setup_start.elapsed();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let operation_start = Instant::now();

    for i in 0..50 {
        let file_path = temp_path.join(format!("perf_test_{:03}.txt", i));
        tokio::fs::write(&file_path, format!("Performance test content {}", i)).await?;
    }

    let creation_duration = operation_start.elapsed();

    let detection_start = Instant::now();
    let events = watcher.wait_for_events(25, Duration::from_secs(5)).await;
    let detection_duration = detection_start.elapsed();

    assert!(
        setup_duration < Duration::from_secs(10),
        "Watcher setup should be fast: {:?}",
        setup_duration
    );
    assert!(
        creation_duration < Duration::from_secs(3),
        "File creation should be reasonable: {:?}",
        creation_duration
    );
    assert!(
        detection_duration < Duration::from_secs(5),
        "Event detection should be timely: {:?}",
        detection_duration
    );

    assert!(!events.is_empty(), "Should detect file creation events");
    assert!(
        events.len() >= 5,
        "Should detect multiple events (got {})",
        events.len()
    );

    println!("Performance metrics:");
    println!("  Setup time: {:?}", setup_duration);
    println!("  File creation time: {:?}", creation_duration);
    println!("  Event detection time: {:?}", detection_duration);
    println!(
        "  Events detected: {} out of 50 operations",
        events.len()
    );

    Ok(())
});

async_test!(test_concurrent_file_operations_with_watching, {
    use tokio::task::JoinSet;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().to_path_buf();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(&temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut join_set = JoinSet::new();

    let task_count = 5;
    let files_per_task = 8;

    for task_id in 0..task_count {
        let temp_path = temp_path.clone();
        join_set.spawn(async move {
            for file_id in 0..files_per_task {
                let file_path =
                    temp_path.join(format!("concurrent_{}_{}.txt", task_id, file_id));
                tokio::fs::write(
                    &file_path,
                    format!("Task {} File {}", task_id, file_id),
                )
                .await
                .unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            task_id
        });
    }

    let mut completed_tasks = Vec::new();
    while let Some(result) = join_set.join_next().await {
        completed_tasks.push(result?);
    }

    assert_eq!(completed_tasks.len(), task_count);

    let events = watcher
        .wait_for_events(task_count * files_per_task / 4, Duration::from_secs(10))
        .await;

    let mut total_files = 0;
    let mut entries = tokio::fs::read_dir(&temp_path).await?;
    while let Some(_entry) = entries.next_entry().await? {
        total_files += 1;
    }

    let expected_files = task_count * files_per_task;
    assert_eq!(
        total_files, expected_files,
        "Should create {} files",
        expected_files
    );
    assert!(
        !events.is_empty(),
        "Should receive events from concurrent operations"
    );

    println!(
        "Concurrent operations: {} tasks x {} files = {} total, {} events detected",
        task_count,
        files_per_task,
        expected_files,
        events.len()
    );

    Ok(())
});

// ============================================================================
// LARGE DIRECTORY TREE PERFORMANCE TESTS
// ============================================================================

async_test!(test_large_directory_tree_performance, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut created_dirs = Vec::new();
    for i in 0..10 {
        for j in 0..5 {
            let dir_path = temp_path
                .join(format!("level1_{}", i))
                .join(format!("level2_{}", j));
            tokio::fs::create_dir_all(&dir_path).await?;
            created_dirs.push(dir_path);
        }
    }

    let watch_start = Instant::now();
    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;
    let watch_setup_time = watch_start.elapsed();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let file_creation_start = Instant::now();
    for (i, dir_path) in created_dirs.iter().enumerate().take(20) {
        let file_path = dir_path.join(format!("file_{}.txt", i));
        tokio::fs::write(&file_path, format!("Content in directory {}", i)).await?;
    }
    let file_creation_time = file_creation_start.elapsed();

    let events = watcher.wait_for_events(10, Duration::from_secs(8)).await;

    assert!(
        watch_setup_time < Duration::from_secs(30),
        "Large tree setup should be reasonable: {:?}",
        watch_setup_time
    );
    assert!(
        file_creation_time < Duration::from_secs(2),
        "File creation in large tree should be fast: {:?}",
        file_creation_time
    );
    assert!(
        !events.is_empty(),
        "Should detect events in large directory tree"
    );

    println!("Large directory tree performance:");
    println!("  Directories created: {}", created_dirs.len());
    println!("  Recursive watch setup: {:?}", watch_setup_time);
    println!("  File creation time: {:?}", file_creation_time);
    println!("  Events detected: {}", events.len());

    Ok(())
});
