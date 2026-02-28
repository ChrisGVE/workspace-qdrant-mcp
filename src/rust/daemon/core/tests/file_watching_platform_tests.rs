//! Cross-platform file system monitoring tests
//!
//! Tests for platform-specific file watching behavior including path handling,
//! Linux inotify timing, macOS FSEvents batching, Windows Unicode support,
//! and Unix symlink handling (creation, broken links, circular links).

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
// CROSS-PLATFORM FILE SYSTEM TESTS
// ============================================================================

async_test!(test_cross_platform_path_handling, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_paths = vec![
        "simple.txt",
        "nested/file.txt",
        "deeply/nested/structure/file.txt",
        "with spaces/file name.txt",
        "unicode_\u{6587}\u{4ef6}.txt",
        "numbers_123.txt",
    ];

    for path_str in test_paths {
        let file_path = temp_path.join(path_str);

        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&file_path, "test content").await?;
        assert!(
            file_path.exists(),
            "File should exist: {}",
            file_path.display()
        );
    }

    let events = watcher.wait_for_events(3, Duration::from_secs(5)).await;

    assert!(
        !events.is_empty(),
        "Should receive events for various path formats"
    );

    Ok(())
});

#[cfg(target_os = "linux")]
async_test!(test_linux_inotify_event_timing, {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_file = temp_path.join("inotify_timing_test.txt");

    let start_time = Instant::now();
    tokio::fs::write(&test_file, "initial content").await?;

    let mut perms = tokio::fs::metadata(&test_file).await?.permissions();
    perms.set_mode(0o755);
    tokio::fs::set_permissions(&test_file, perms).await?;

    tokio::fs::write(&test_file, "modified content").await?;

    let events = watcher.wait_for_events(2, Duration::from_secs(2)).await;
    let event_time = start_time.elapsed();

    assert!(
        event_time < Duration::from_millis(500),
        "Events should be delivered quickly on Linux"
    );
    assert!(!events.is_empty(), "Should receive inotify events on Linux");

    println!("Linux inotify event timing: {:?}", event_time);

    Ok(())
});

#[cfg(target_os = "macos")]
async_test!(test_macos_fsevents_batch_behavior, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let start_time = Instant::now();
    for i in 0..10 {
        let test_file = temp_path.join(format!("fsevents_batch_{}.txt", i));
        tokio::fs::write(&test_file, format!("content {}", i)).await?;
    }

    let events = watcher.wait_for_events(5, Duration::from_secs(3)).await;
    let batch_time = start_time.elapsed();

    assert!(!events.is_empty(), "Should receive FSEvents on macOS");

    println!("macOS FSEvents batch processing time: {:?}", batch_time);
    println!(
        "Events received for 10 file operations: {}",
        events.len()
    );

    Ok(())
});

#[cfg(target_os = "windows")]
async_test!(test_windows_readdirectorychanges_unicode, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let unicode_files = vec![
        "test_\u{f1}\u{e1}m\u{e9}\u{e9}.txt",
        "\u{6d4b}\u{8bd5}\u{6587}\u{4ef6}.txt",
        "\u{444}\u{430}\u{439}\u{43b}_\u{442}\u{435}\u{441}\u{442}.txt",
        "\u{1f525}_emoji_file.txt",
    ];

    for filename in unicode_files {
        let test_file = temp_path.join(filename);
        tokio::fs::write(&test_file, "Unicode test content").await?;
    }

    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    assert!(
        !events.is_empty(),
        "Should receive ReadDirectoryChangesW events for Unicode files"
    );

    for event in &events {
        for path in &event.paths {
            assert!(
                !path.to_string_lossy().is_empty(),
                "Path should be properly decoded"
            );
        }
    }

    println!("Windows Unicode file events: {}", events.len());

    Ok(())
});

// ============================================================================
// SYMLINK HANDLING AND EDGE CASES
// ============================================================================

#[cfg(unix)]
async_test!(test_symlink_creation_and_following, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let target_file =
        create_test_file(temp_path, "target.txt", "target content").await?;

    let symlink_path = temp_path.join("link_to_target.txt");
    fs::symlink(&target_file, &symlink_path)
        .map_err(|e| format!("Failed to create symlink: {}", e))?;

    tokio::fs::write(&target_file, "modified target content").await?;
    tokio::fs::write(&symlink_path, "modified via symlink").await?;

    let events = watcher.wait_for_events(3, Duration::from_secs(3)).await;

    assert!(
        !events.is_empty(),
        "Should receive events for symlink operations"
    );

    let symlink_related_events: Vec<_> = events
        .iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                let path_str = path.to_string_lossy();
                path_str.contains("target.txt") || path_str.contains("link_to_target.txt")
            })
        })
        .collect();

    assert!(
        !symlink_related_events.is_empty(),
        "Should detect symlink-related events"
    );

    Ok(())
});

#[cfg(unix)]
async_test!(test_broken_symlink_detection, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher
        .watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let target_file =
        create_test_file(temp_path, "will_be_deleted.txt", "target content").await?;

    let symlink_path = temp_path.join("broken_link.txt");
    fs::symlink(&target_file, &symlink_path)
        .map_err(|e| format!("Failed to create symlink: {}", e))?;

    assert!(
        symlink_path.exists(),
        "Symlink should exist and point to valid target"
    );

    tokio::fs::remove_file(&target_file).await?;

    let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    assert!(
        symlink_path.is_symlink(),
        "Symlink should still exist as a symlink"
    );
    assert!(
        !symlink_path.exists(),
        "Broken symlink should not resolve"
    );

    assert!(
        !events.is_empty(),
        "Should receive events for target file deletion"
    );

    Ok(())
});

#[cfg(unix)]
async_test!(test_circular_symlink_prevention, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let dir_a = temp_path.join("dir_a");
    let dir_b = temp_path.join("dir_b");
    tokio::fs::create_dir(&dir_a).await?;
    tokio::fs::create_dir(&dir_b).await?;

    let link_a_to_b = dir_a.join("link_to_b");
    let link_b_to_a = dir_b.join("link_to_a");

    fs::symlink("../dir_b", &link_a_to_b)
        .map_err(|e| format!("Failed to create circular symlink A->B: {}", e))?;
    fs::symlink("../dir_a", &link_b_to_a)
        .map_err(|e| format!("Failed to create circular symlink B->A: {}", e))?;

    assert!(link_a_to_b.is_symlink(), "Link A->B should be a symlink");
    assert!(link_b_to_a.is_symlink(), "Link B->A should be a symlink");

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    let watch_result = watcher.watch(temp_path, RecursiveMode::Recursive);
    assert!(
        watch_result.is_ok(),
        "Should handle circular symlinks gracefully"
    );

    tokio::time::sleep(Duration::from_millis(500)).await;

    let _test_file =
        create_test_file(&dir_a, "test_in_circular.txt", "content").await?;

    let events = watcher.wait_for_events(1, Duration::from_secs(3)).await;

    println!(
        "Events detected with circular symlinks present: {}",
        events.len()
    );

    Ok(())
});
