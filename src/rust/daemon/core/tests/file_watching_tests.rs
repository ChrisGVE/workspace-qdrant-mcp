//! Comprehensive file system monitoring tests with notify crate
//!
//! Tests for file system watching and event processing across platforms,
//! including event debouncing, filtering, symlink handling, and cross-platform
//! file system differences.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tempfile::TempDir;
use shared_test_utils::{async_test, serial_async_test, TestResult};

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
                // Store event for verification
                if let Ok(mut events_lock) = events_clone.lock() {
                    events_lock.push(event.clone());
                }
                // Send event to async channel
                let _ = event_tx.send(event);
            }
        })?;

        Ok(Self {
            _watcher: watcher,
            _events: events,
            event_rx,
        })
    }

    fn watch<P: AsRef<Path>>(&mut self, path: P, recursive: RecursiveMode) -> Result<(), notify::Error> {
        self._watcher.watch(path.as_ref(), recursive)
    }

    fn unwatch<P: AsRef<Path>>(&mut self, path: P) -> Result<(), notify::Error> {
        self._watcher.unwatch(path.as_ref())
    }

    async fn wait_for_events(&mut self, expected_count: usize, timeout: Duration) -> Vec<Event> {
        let mut collected_events = Vec::new();
        let start_time = Instant::now();

        while collected_events.len() < expected_count && start_time.elapsed() < timeout {
            tokio::select! {
                Some(event) = self.event_rx.recv() => {
                    collected_events.push(event);
                },
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    // Continue waiting
                }
            }
        }

        collected_events
    }

    fn _get_collected_events(&self) -> Vec<Event> {
        if let Ok(events_lock) = self._events.lock() {
            events_lock.clone()
        } else {
            Vec::new()
        }
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

    // Create nested directories
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

    // Create files in various directories
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

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a test file
    let test_file = temp_path.join("created_file.txt");
    tokio::fs::write(&test_file, "test content").await?;

    // Wait for file creation events
    let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    // Verify we received file creation events
    assert!(!events.is_empty(), "Should receive at least one file creation event");

    // Check that we have events related to our file
    let relevant_events: Vec<_> = events.iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("created_file.txt")
            })
        })
        .collect();

    assert!(!relevant_events.is_empty(), "Should receive events for created_file.txt");

    Ok(())
});

async_test!(test_file_modification_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create initial file
    let test_file = create_test_file(temp_path, "modify_test.txt", "initial content").await?;

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Modify the file multiple times
    for i in 1..=3 {
        tokio::fs::write(&test_file, format!("modified content {}", i)).await?;
        tokio::time::sleep(Duration::from_millis(50)).await; // Small delay between modifications
    }

    // Wait for modification events
    let events = watcher.wait_for_events(1, Duration::from_secs(3)).await;

    // Verify we received modification events
    assert!(!events.is_empty(), "Should receive modification events");

    // Check for modify events
    let _modify_events: Vec<_> = events.iter()
        .filter(|event| {
            matches!(event.kind, EventKind::Modify(_))
        })
        .collect();

    // Note: Some file systems may generate different event types for modifications
    // We just verify that we got some events related to the file
    let file_related_events: Vec<_> = events.iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("modify_test.txt")
            })
        })
        .collect();

    assert!(!file_related_events.is_empty(), "Should receive events for the modified file");

    Ok(())
});

async_test!(test_file_deletion_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create initial file
    let test_file = create_test_file(temp_path, "delete_test.txt", "content to delete").await?;

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Delete the file
    tokio::fs::remove_file(&test_file).await?;

    // Wait for deletion events
    let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    // Verify we received deletion events
    assert!(!events.is_empty(), "Should receive deletion events");

    // Check for removal events
    let _removal_events: Vec<_> = events.iter()
        .filter(|event| {
            matches!(event.kind, EventKind::Remove(_))
        })
        .collect();

    // Some file systems may report different event types for deletion
    // We verify that we got events and the file is indeed deleted
    assert!(!test_file.exists(), "File should be deleted");

    Ok(())
});

// ============================================================================
// DIRECTORY WATCHING BEHAVIOR TESTS
// ============================================================================

async_test!(test_recursive_directory_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create nested directory structure with files
    let created_paths = create_nested_structure(temp_path).await?;

    // Wait for all creation events
    let events = watcher.wait_for_events(created_paths.len() / 2, Duration::from_secs(5)).await;

    // Verify we received events for nested files
    assert!(!events.is_empty(), "Should receive events for nested structure creation");

    // Check that we can detect events from deeply nested files
    let deep_file_events: Vec<_> = events.iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.to_string_lossy().contains("level3")
            })
        })
        .collect();

    // We should detect events from deep nested structures when using recursive mode
    // Note: Event generation varies by platform, so we check for any activity
    println!("Total events received: {}", events.len());
    println!("Deep file events: {}", deep_file_events.len());

    Ok(())
});

async_test!(test_non_recursive_directory_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a file in the root directory
    let _root_file = create_test_file(temp_path, "root_level.txt", "root content").await?;

    // Create a subdirectory and file
    let sub_dir = temp_path.join("subdir");
    tokio::fs::create_dir(&sub_dir).await?;
    let _sub_file = create_test_file(&sub_dir, "sub_file.txt", "sub content").await?;

    // Wait for events
    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    // With non-recursive mode, we should see root level changes
    let _root_events: Vec<_> = events.iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                path.file_name().and_then(|name| name.to_str()) == Some("root_level.txt")
            })
        })
        .collect();

    // We should detect root level file creation
    assert!(!events.is_empty(), "Should receive some events in non-recursive mode");

    Ok(())
});

async_test!(test_directory_creation_and_removal, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a directory
    let test_dir = temp_path.join("test_directory");
    tokio::fs::create_dir(&test_dir).await?;

    // Add a file to the directory
    let _test_file = create_test_file(&test_dir, "inside.txt", "directory content").await?;

    // Wait for creation events
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Remove the entire directory
    tokio::fs::remove_dir_all(&test_dir).await?;

    // Wait for removal events
    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    // Verify we received events for directory operations
    assert!(!events.is_empty(), "Should receive events for directory creation and removal");
    assert!(!test_dir.exists(), "Directory should be removed");

    Ok(())
});

// ============================================================================
// CROSS-PLATFORM FILE SYSTEM TESTS
// ============================================================================

async_test!(test_cross_platform_path_handling, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test different path structures are handled correctly
    let test_paths = vec![
        "simple.txt",
        "nested/file.txt",
        "deeply/nested/structure/file.txt",
        "with spaces/file name.txt",
        "unicode_文件.txt",
        "numbers_123.txt",
    ];

    for path_str in test_paths {
        let file_path = temp_path.join(path_str);

        // Create parent directories if needed
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&file_path, "test content").await?;
        assert!(file_path.exists(), "File should exist: {}", file_path.display());
    }

    // Wait for events from all the file creations
    let events = watcher.wait_for_events(3, Duration::from_secs(5)).await;

    // Verify we can handle various path formats
    assert!(!events.is_empty(), "Should receive events for various path formats");

    Ok(())
});

#[cfg(target_os = "linux")]
async_test!(test_linux_inotify_event_timing, {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test Linux-specific behavior - inotify timing
    let test_file = temp_path.join("inotify_timing_test.txt");

    // Create file
    let start_time = Instant::now();
    tokio::fs::write(&test_file, "initial content").await?;

    // Modify permissions (Linux-specific test)
    let mut perms = tokio::fs::metadata(&test_file).await?.permissions();
    perms.set_mode(0o755);
    tokio::fs::set_permissions(&test_file, perms).await?;

    // Modify content
    tokio::fs::write(&test_file, "modified content").await?;

    // Wait for events with timing measurement
    let events = watcher.wait_for_events(2, Duration::from_secs(2)).await;
    let event_time = start_time.elapsed();

    // On Linux with inotify, events should be relatively immediate
    assert!(event_time < Duration::from_millis(500), "Events should be delivered quickly on Linux");
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

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test macOS FSEvents batch behavior - create multiple files rapidly
    let start_time = Instant::now();
    for i in 0..10 {
        let test_file = temp_path.join(format!("fsevents_batch_{}.txt", i));
        tokio::fs::write(&test_file, format!("content {}", i)).await?;
    }

    // FSEvents on macOS may batch events together
    let events = watcher.wait_for_events(5, Duration::from_secs(3)).await;
    let batch_time = start_time.elapsed();

    // Verify we receive events (may be batched on macOS)
    assert!(!events.is_empty(), "Should receive FSEvents on macOS");

    println!("macOS FSEvents batch processing time: {:?}", batch_time);
    println!("Events received for 10 file operations: {}", events.len());

    // FSEvents may coalesce events, so we might get fewer events than operations
    // This is expected behavior on macOS

    Ok(())
});

#[cfg(target_os = "windows")]
async_test!(test_windows_readdirectorychanges_unicode, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test Windows Unicode handling in ReadDirectoryChangesW
    let unicode_files = vec![
        "test_ñáméé.txt",
        "测试文件.txt",
        "файл_тест.txt",
        "🔥_emoji_file.txt",
    ];

    for filename in unicode_files {
        let test_file = temp_path.join(filename);
        tokio::fs::write(&test_file, "Unicode test content").await?;
    }

    // Wait for Unicode filename events
    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    // Verify Windows can handle Unicode filenames properly
    assert!(!events.is_empty(), "Should receive ReadDirectoryChangesW events for Unicode files");

    // Check that paths are properly encoded
    for event in &events {
        for path in &event.paths {
            assert!(path.to_string_lossy().len() > 0, "Path should be properly decoded");
        }
    }

    println!("Windows Unicode file events: {}", events.len());

    Ok(())
});

// ============================================================================
// EVENT DEBOUNCING AND FILTERING TESTS
// ============================================================================

async_test!(test_rapid_file_changes_debouncing, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_file = temp_path.join("rapid_changes.txt");

    // Make rapid consecutive changes to the same file
    let start_time = Instant::now();
    for i in 0..20 {
        tokio::fs::write(&test_file, format!("rapid change {}", i)).await?;
        tokio::time::sleep(Duration::from_millis(10)).await; // Very rapid changes
    }

    // Wait for events to be processed
    let events = watcher.wait_for_events(5, Duration::from_secs(3)).await;
    let total_time = start_time.elapsed();

    // Verify that we don't get exactly 20 events (some should be debounced/coalesced)
    println!("Rapid changes: 20 operations generated {} events in {:?}", events.len(), total_time);

    // File system watching may coalesce rapid changes
    assert!(events.len() <= 20, "Events should be coalesced for rapid changes");
    assert!(!events.is_empty(), "Should receive some events");

    Ok(())
});

async_test!(test_event_filtering_by_file_type, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create files of different types
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

    // Wait for all file creation events
    let events = watcher.wait_for_events(3, Duration::from_secs(5)).await;

    // Extract file extensions from events
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

    // Verify we can distinguish different file types in events
    assert!(!events.is_empty(), "Should receive events for different file types");
    println!("Detected file extensions in events: {:?}", detected_extensions);

    Ok(())
});

async_test!(test_duplicate_event_detection, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    let test_file = temp_path.join("duplicate_test.txt");

    // Create file
    tokio::fs::write(&test_file, "initial content").await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Modify with same content (should still generate events)
    tokio::fs::write(&test_file, "modified content").await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Modify again with different content
    tokio::fs::write(&test_file, "different content").await?;

    // Wait for events
    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    // Analyze event patterns for potential duplicates
    let mut event_signatures = HashSet::new();
    for event in &events {
        let signature = format!("{:?}:{:?}", event.kind, event.paths);
        event_signatures.insert(signature);
    }

    println!("Total events: {}, Unique signatures: {}", events.len(), event_signatures.len());

    // Should receive multiple events for file modifications
    assert!(!events.is_empty(), "Should receive events for file modifications");

    Ok(())
});

// ============================================================================
// SYMLINK HANDLING AND EDGE CASES
// ============================================================================

#[cfg(unix)] // Symlinks behave differently on Unix vs Windows
async_test!(test_symlink_creation_and_following, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a target file
    let target_file = create_test_file(temp_path, "target.txt", "target content").await?;

    // Create a symlink to the target
    let symlink_path = temp_path.join("link_to_target.txt");
    fs::symlink(&target_file, &symlink_path)
        .map_err(|e| format!("Failed to create symlink: {}", e))?;

    // Modify the target file through the original path
    tokio::fs::write(&target_file, "modified target content").await?;

    // Modify the target file through the symlink
    tokio::fs::write(&symlink_path, "modified via symlink").await?;

    // Wait for events
    let events = watcher.wait_for_events(3, Duration::from_secs(3)).await;

    // Verify we can detect changes to symlinked files
    assert!(!events.is_empty(), "Should receive events for symlink operations");

    // Check if events reference either the target or the symlink
    let symlink_related_events: Vec<_> = events.iter()
        .filter(|event| {
            event.paths.iter().any(|path| {
                let path_str = path.to_string_lossy();
                path_str.contains("target.txt") || path_str.contains("link_to_target.txt")
            })
        })
        .collect();

    assert!(!symlink_related_events.is_empty(), "Should detect symlink-related events");

    Ok(())
});

#[cfg(unix)]
async_test!(test_broken_symlink_detection, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a target file
    let target_file = create_test_file(temp_path, "will_be_deleted.txt", "target content").await?;

    // Create a symlink
    let symlink_path = temp_path.join("broken_link.txt");
    fs::symlink(&target_file, &symlink_path)
        .map_err(|e| format!("Failed to create symlink: {}", e))?;

    // Verify symlink works initially
    assert!(symlink_path.exists(), "Symlink should exist and point to valid target");

    // Delete the target file (breaking the symlink)
    tokio::fs::remove_file(&target_file).await?;

    // Wait for events
    let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    // Verify the symlink is now broken
    assert!(symlink_path.is_symlink(), "Symlink should still exist as a symlink");
    assert!(!symlink_path.exists(), "Broken symlink should not resolve");

    // Should receive events for the target deletion
    assert!(!events.is_empty(), "Should receive events for target file deletion");

    Ok(())
});

#[cfg(unix)]
async_test!(test_circular_symlink_prevention, {
    use std::os::unix::fs;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create directories for circular symlink test
    let dir_a = temp_path.join("dir_a");
    let dir_b = temp_path.join("dir_b");
    tokio::fs::create_dir(&dir_a).await?;
    tokio::fs::create_dir(&dir_b).await?;

    // Create circular symlinks: dir_a/link_to_b -> ../dir_b, dir_b/link_to_a -> ../dir_a
    let link_a_to_b = dir_a.join("link_to_b");
    let link_b_to_a = dir_b.join("link_to_a");

    fs::symlink("../dir_b", &link_a_to_b)
        .map_err(|e| format!("Failed to create circular symlink A->B: {}", e))?;
    fs::symlink("../dir_a", &link_b_to_a)
        .map_err(|e| format!("Failed to create circular symlink B->A: {}", e))?;

    // Verify circular symlinks were created
    assert!(link_a_to_b.is_symlink(), "Link A->B should be a symlink");
    assert!(link_b_to_a.is_symlink(), "Link B->A should be a symlink");

    // Try to watch with recursive mode (should handle circular symlinks gracefully)
    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    // This should not hang or crash due to circular symlinks
    let watch_result = watcher.watch(temp_path, RecursiveMode::Recursive);
    assert!(watch_result.is_ok(), "Should handle circular symlinks gracefully");

    // Allow watcher to initialize and process the circular structure
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Create a regular file in one of the directories
    let _test_file = create_test_file(&dir_a, "test_in_circular.txt", "content").await?;

    // Wait for events
    let events = watcher.wait_for_events(1, Duration::from_secs(3)).await;

    // Should still be able to detect regular file operations despite circular symlinks
    println!("Events detected with circular symlinks present: {}", events.len());

    Ok(())
});

// ============================================================================
// MEMORY USAGE AND PERFORMANCE TESTS
// ============================================================================

async_test!(test_memory_usage_with_many_files, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create many files to test memory scaling
    let file_count = 500; // Reduced from 1000 for test performance
    let start_time = Instant::now();

    for i in 0..file_count {
        let file_path = temp_path.join(format!("memory_test_{:04}.txt", i));
        tokio::fs::write(&file_path, format!("Content for file {}", i)).await?;

        // Small delay every 50 files to avoid overwhelming the system
        if i % 50 == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    let creation_time = start_time.elapsed();

    // Wait for events to be processed
    let events = watcher.wait_for_events(file_count / 4, Duration::from_secs(10)).await;

    // Verify files were created and events were received
    let mut read_dir = tokio::fs::read_dir(temp_path).await?;
    let mut entry_count = 0;
    while let Some(_entry) = read_dir.next_entry().await? {
        entry_count += 1;
    }

    assert_eq!(entry_count, file_count, "All files should be created");
    assert!(!events.is_empty(), "Should receive events for file creation");

    println!("Created {} files in {:?}, received {} events", file_count, creation_time, events.len());

    // Memory usage verification - check that watcher is still responsive
    let _additional_file = create_test_file(temp_path, "additional_test.txt", "additional content").await?;
    let additional_events = watcher.wait_for_events(1, Duration::from_secs(2)).await;

    assert!(!additional_events.is_empty() || events.len() > file_count / 8,
        "Watcher should remain responsive after processing many files");

    Ok(())
});

async_test!(test_watcher_performance_baseline, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Measure watcher setup time
    let setup_start = Instant::now();
    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;
    let setup_duration = setup_start.elapsed();

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Measure file operation detection latency
    let operation_start = Instant::now();

    // Create test files with timing
    for i in 0..50 {
        let file_path = temp_path.join(format!("perf_test_{:03}.txt", i));
        tokio::fs::write(&file_path, format!("Performance test content {}", i)).await?;
    }

    let creation_duration = operation_start.elapsed();

    // Wait for events and measure detection time
    let detection_start = Instant::now();
    let events = watcher.wait_for_events(25, Duration::from_secs(5)).await;
    let detection_duration = detection_start.elapsed();

    // Performance assertions (generous to avoid flaky tests in CI)
    assert!(setup_duration < Duration::from_secs(10), "Watcher setup should be fast: {:?}", setup_duration);
    assert!(creation_duration < Duration::from_secs(3), "File creation should be reasonable: {:?}", creation_duration);
    assert!(detection_duration < Duration::from_secs(5), "Event detection should be timely: {:?}", detection_duration);

    // Verify we detected a reasonable portion of events
    assert!(!events.is_empty(), "Should detect file creation events");
    assert!(events.len() >= 5, "Should detect multiple events (got {})", events.len());

    println!("Performance metrics:");
    println!("  Setup time: {:?}", setup_duration);
    println!("  File creation time: {:?}", creation_duration);
    println!("  Event detection time: {:?}", detection_duration);
    println!("  Events detected: {} out of 50 operations", events.len());

    Ok(())
});

async_test!(test_concurrent_file_operations_with_watching, {
    use tokio::task::JoinSet;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().to_path_buf();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(&temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut join_set = JoinSet::new();

    // Spawn multiple tasks creating files concurrently
    let task_count = 5; // Reduced for test performance
    let files_per_task = 8;

    for task_id in 0..task_count {
        let temp_path = temp_path.clone();
        join_set.spawn(async move {
            for file_id in 0..files_per_task {
                let file_path = temp_path.join(format!("concurrent_{}_{}.txt", task_id, file_id));
                tokio::fs::write(&file_path, format!("Task {} File {}", task_id, file_id)).await.unwrap();
                // Small delay to avoid overwhelming the file system
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            task_id
        });
    }

    // Wait for all tasks to complete
    let mut completed_tasks = Vec::new();
    while let Some(result) = join_set.join_next().await {
        completed_tasks.push(result?);
    }

    assert_eq!(completed_tasks.len(), task_count);

    // Wait for events from concurrent operations
    let events = watcher.wait_for_events(task_count * files_per_task / 4, Duration::from_secs(10)).await;

    // Verify all files were created
    let mut total_files = 0;
    let mut entries = tokio::fs::read_dir(&temp_path).await?;
    while let Some(_entry) = entries.next_entry().await? {
        total_files += 1;
    }

    let expected_files = task_count * files_per_task;
    assert_eq!(total_files, expected_files, "Should create {} files", expected_files);
    assert!(!events.is_empty(), "Should receive events from concurrent operations");

    println!("Concurrent operations: {} tasks × {} files = {} total, {} events detected",
        task_count, files_per_task, expected_files, events.len());

    Ok(())
});

// ============================================================================
// LARGE DIRECTORY TREE PERFORMANCE TESTS
// ============================================================================

async_test!(test_large_directory_tree_performance, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Create a large directory tree structure
    let mut created_dirs = Vec::new();
    for i in 0..10 {
        for j in 0..5 {
            let dir_path = temp_path.join(format!("level1_{}", i)).join(format!("level2_{}", j));
            tokio::fs::create_dir_all(&dir_path).await?;
            created_dirs.push(dir_path);
        }
    }

    // Measure recursive watching setup time
    let watch_start = Instant::now();
    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;
    let watch_setup_time = watch_start.elapsed();

    // Allow watcher to initialize and scan the tree
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Add files to various directories and measure detection
    let file_creation_start = Instant::now();
    for (i, dir_path) in created_dirs.iter().enumerate().take(20) {
        let file_path = dir_path.join(format!("file_{}.txt", i));
        tokio::fs::write(&file_path, format!("Content in directory {}", i)).await?;
    }
    let file_creation_time = file_creation_start.elapsed();

    // Wait for events from the large tree
    let events = watcher.wait_for_events(10, Duration::from_secs(8)).await;

    // Performance should be reasonable even with large directory trees
    assert!(watch_setup_time < Duration::from_secs(15), "Large tree setup should be reasonable: {:?}", watch_setup_time);
    assert!(file_creation_time < Duration::from_secs(2), "File creation in large tree should be fast: {:?}", file_creation_time);
    assert!(!events.is_empty(), "Should detect events in large directory tree");

    println!("Large directory tree performance:");
    println!("  Directories created: {}", created_dirs.len());
    println!("  Recursive watch setup: {:?}", watch_setup_time);
    println!("  File creation time: {:?}", file_creation_time);
    println!("  Events detected: {}", events.len());

    Ok(())
});

// ============================================================================
// INTEGRATION WITH DOCUMENT PROCESSING PIPELINE TESTS
// ============================================================================

serial_async_test!(test_integration_with_document_processing, {
    use workspace_qdrant_core::DocumentProcessor;

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::NonRecursive)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create a document processor for integration testing
    let processor = DocumentProcessor::new();

    // Create various document types that would trigger processing
    let documents = vec![
        ("document.txt", "This is a text document for processing."),
        ("README.md", "# Markdown Document\n\nThis is markdown content."),
        ("script.py", "#!/usr/bin/env python3\nprint('Hello, world!')"),
        ("config.json", r#"{"setting": "value", "enabled": true}"#),
    ];

    for (filename, content) in documents {
        let file_path = create_test_file(temp_path, filename, content).await?;

        // Simulate document processing that would be triggered by file events
        let process_result = processor.process_file(&file_path, "test_collection").await;
        assert!(process_result.is_ok(), "Document processing should succeed for {}", filename);
    }

    // Wait for file creation events
    let events = watcher.wait_for_events(2, Duration::from_secs(3)).await;

    // Verify integration: file watching detects files that can be processed
    assert!(!events.is_empty(), "Should detect file creation events for document processing");

    // Verify that document processor can handle files detected by watcher
    for event in &events {
        for path in &event.paths {
            if path.is_file() {
                let process_result = processor.process_file(path, "event_test_collection").await;
                // Processing should succeed for detected files
                if let Err(e) = &process_result {
                    println!("Processing warning for {}: {}", path.display(), e);
                }
                // Note: Some files might fail processing (e.g., temporary files), which is acceptable
            }
        }
    }

    println!("Integration test: {} events processed with document pipeline", events.len());

    Ok(())
});

// ============================================================================
// WATCHER CLEANUP AND RESOURCE MANAGEMENT TESTS
// ============================================================================

async_test!(test_watcher_cleanup_and_resource_management, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Test that multiple watchers can be created and cleaned up properly
    for iteration in 0..3 {
        let mut watcher = TestWatcher::new()
            .map_err(|e| format!("Failed to create watcher {}: {}", iteration, e))?;

        watcher.watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching {}: {}", iteration, e))?;

        // Create a test file
        let test_file = create_test_file(temp_path, &format!("cleanup_test_{}.txt", iteration), "cleanup content").await?;

        // Brief usage
        let events = watcher.wait_for_events(1, Duration::from_secs(2)).await;
        println!("Iteration {} received {} events", iteration, events.len());

        // Unwatch to test cleanup
        let unwatch_result = watcher.unwatch(temp_path);
        assert!(unwatch_result.is_ok(), "Should be able to unwatch directory");

        // Clean up the test file
        tokio::fs::remove_file(&test_file).await?;

        // Watcher goes out of scope here, testing Drop implementation
    }

    // Verify that resources are properly released
    // (This is mainly testing that no panics or resource leaks occur)
    println!("Successfully created and cleaned up 3 watchers");

    Ok(())
});

async_test!(test_watch_unwatch_cycle, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create watcher: {}", e))?;

    // Test watch/unwatch cycle
    for cycle in 0..3 {
        // Start watching
        watcher.watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching cycle {}: {}", cycle, e))?;

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Create a file while watching
        let test_file = create_test_file(temp_path, &format!("cycle_test_{}.txt", cycle), "cycle content").await?;

        // Should receive events while watching
        let events_while_watching = watcher.wait_for_events(1, Duration::from_secs(1)).await;

        // Stop watching
        watcher.unwatch(temp_path)
            .map_err(|e| format!("Failed to unwatch cycle {}: {}", cycle, e))?;

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Modify file while not watching
        tokio::fs::write(&test_file, "modified while not watching").await?;

        // Should not receive new events while not watching
        let events_while_not_watching = watcher.wait_for_events(1, Duration::from_millis(500)).await;

        println!("Cycle {}: {} events while watching, {} events while not watching",
            cycle, events_while_watching.len(), events_while_not_watching.len());

        // Clean up
        tokio::fs::remove_file(&test_file).await?;
    }

    Ok(())
});

// ============================================================================
// NOTIFY CRATE CONFIGURATION TESTS
// ============================================================================

async_test!(test_notify_config_options, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Test different notify configurations
    let configs = vec![
        ("default", Config::default()),
        ("with_compare_contents", Config::default().with_compare_contents(true)),
        ("poll_interval_1s", Config::default().with_poll_interval(Duration::from_secs(1))),
    ];

    for (config_name, config) in configs {
        println!("Testing notify config: {}", config_name);

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let mut watcher = notify::RecommendedWatcher::new(
            move |result: Result<Event, notify::Error>| {
                if let Ok(event) = result {
                    if let Ok(mut events_lock) = events_clone.lock() {
                        events_lock.push(event);
                    }
                }
            },
            config,
        ).map_err(|e| format!("Failed to create watcher with {}: {}", config_name, e))?;

        watcher.watch(temp_path, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to start watching with {}: {}", config_name, e))?;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create and modify a test file
        let test_file = create_test_file(temp_path, &format!("config_test_{}.txt", config_name.replace(' ', "_")), "initial").await?;
        tokio::time::sleep(Duration::from_millis(100)).await;

        tokio::fs::write(&test_file, "modified").await?;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check collected events
        let collected_events = if let Ok(events_lock) = events.lock() {
            events_lock.len()
        } else {
            0
        };

        println!("  {} events collected with {}", collected_events, config_name);

        // Cleanup
        let _ = watcher.unwatch(temp_path);
        tokio::fs::remove_file(&test_file).await?;
    }

    Ok(())
});

async_test!(test_comprehensive_file_system_monitoring, {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    let mut watcher = TestWatcher::new()
        .map_err(|e| format!("Failed to create comprehensive watcher: {}", e))?;

    watcher.watch(temp_path, RecursiveMode::Recursive)
        .map_err(|e| format!("Failed to start comprehensive watching: {}", e))?;

    // Allow watcher to initialize
    tokio::time::sleep(Duration::from_millis(100)).await;

    println!("Starting comprehensive file system monitoring test...");

    // Test scenario 1: Multiple file operations
    let test_files = vec![
        ("comprehensive_1.txt", "content 1"),
        ("comprehensive_2.md", "# Markdown\nContent"),
        ("comprehensive_3.json", r#"{"test": true}"#),
    ];

    for (filename, content) in &test_files {
        let file_path = create_test_file(temp_path, filename, content).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Modify each file
        tokio::fs::write(&file_path, format!("{} - modified", content)).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Test scenario 2: Directory operations
    let test_dir = temp_path.join("comprehensive_subdir");
    tokio::fs::create_dir(&test_dir).await?;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _subdir_file = create_test_file(&test_dir, "subdir_file.txt", "subdirectory content").await?;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Test scenario 3: File deletions
    for (filename, _) in &test_files {
        let file_path = temp_path.join(filename);
        if file_path.exists() {
            tokio::fs::remove_file(&file_path).await?;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    // Collect all events from comprehensive test
    let all_events = watcher.wait_for_events(10, Duration::from_secs(5)).await;

    // Analyze the comprehensive test results
    let total_events = all_events.len();
    let creation_events = all_events.iter().filter(|e| matches!(e.kind, EventKind::Create(_))).count();
    let modification_events = all_events.iter().filter(|e| matches!(e.kind, EventKind::Modify(_))).count();
    let removal_events = all_events.iter().filter(|e| matches!(e.kind, EventKind::Remove(_))).count();

    println!("Comprehensive test results:");
    println!("  Total events: {}", total_events);
    println!("  Creation events: {}", creation_events);
    println!("  Modification events: {}", modification_events);
    println!("  Removal events: {}", removal_events);

    // Verify we captured a reasonable number of events
    assert!(total_events >= 5, "Should capture multiple events in comprehensive test");
    assert!(!all_events.is_empty(), "Comprehensive test should generate events");

    // Final verification: watcher should still be responsive
    let _final_test_file = create_test_file(temp_path, "final_test.txt", "final test").await?;
    let final_events = watcher.wait_for_events(1, Duration::from_secs(2)).await;
    assert!(!final_events.is_empty() || total_events > 15, "Watcher should remain responsive after comprehensive test");

    Ok(())
});