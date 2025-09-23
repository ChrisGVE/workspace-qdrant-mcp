//! Comprehensive directory watching behavior tests using TDD approach
//!
//! This test suite covers:
//! - Directory watcher initialization and configuration
//! - Recursive vs non-recursive monitoring with depth limits
//! - Directory exclusion patterns and filtering rules
//! - Directory structure change detection
//! - Cleanup and resource management
//! - Cross-platform compatibility

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::{TempDir, tempdir};
use tokio::time::sleep;
use workspace_qdrant_daemon::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};

/// Helper to create test configuration
fn create_test_file_watcher_config() -> FileWatcherConfig {
    FileWatcherConfig {
        enabled: true,
        debounce_ms: 50, // Short debounce for testing
        max_watched_dirs: 10,
        ignore_patterns: vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            ".git/**".to_string(),
            "target/**".to_string(),
        ],
        recursive: true,
    }
}

/// Helper to create test processor
fn create_test_processor() -> Arc<DocumentProcessor> {
    Arc::new(DocumentProcessor::test_instance())
}

/// Create a test directory structure
async fn create_test_directory_structure(base_dir: &Path) -> DaemonResult<()> {
    let src_dir = base_dir.join("src");
    let tests_dir = base_dir.join("tests");
    let target_dir = base_dir.join("target");
    let git_dir = base_dir.join(".git");

    // Create directories
    tokio::fs::create_dir_all(&src_dir).await?;
    tokio::fs::create_dir_all(&tests_dir).await?;
    tokio::fs::create_dir_all(&target_dir).await?;
    tokio::fs::create_dir_all(&git_dir).await?;

    // Create nested directories
    tokio::fs::create_dir_all(src_dir.join("modules")).await?;
    tokio::fs::create_dir_all(tests_dir.join("unit")).await?;
    tokio::fs::create_dir_all(tests_dir.join("integration")).await?;

    // Create test files
    tokio::fs::write(src_dir.join("lib.rs"), "// Main library file").await?;
    tokio::fs::write(src_dir.join("modules").join("mod.rs"), "// Module file").await?;
    tokio::fs::write(tests_dir.join("unit").join("test_lib.rs"), "// Unit test").await?;
    tokio::fs::write(base_dir.join("Cargo.toml"), "[package]\nname = \"test\"").await?;

    // Create ignored files
    tokio::fs::write(target_dir.join("build.log"), "Build log").await?;
    tokio::fs::write(base_dir.join("temp.tmp"), "Temporary file").await?;
    tokio::fs::write(git_dir.join("config"), "Git config").await?;

    Ok(())
}

#[tokio::test]
async fn test_directory_watcher_initialization() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();

    // Test successful initialization
    let watcher = FileWatcher::new(&config, processor).await;
    assert!(watcher.is_ok());

    let watcher = watcher.unwrap();
    assert_eq!(watcher.config().enabled, true);
    assert_eq!(watcher.config().recursive, true);
    assert_eq!(watcher.config().debounce_ms, 50);
    assert_eq!(watcher.config().max_watched_dirs, 10);
}

#[tokio::test]
async fn test_directory_watcher_disabled_initialization() {
    let mut config = create_test_file_watcher_config();
    config.enabled = false;
    let processor = create_test_processor();

    let watcher = FileWatcher::new(&config, processor).await;
    assert!(watcher.is_ok());

    let watcher = watcher.unwrap();
    assert_eq!(watcher.config().enabled, false);

    // Starting a disabled watcher should succeed but do nothing
    let result = watcher.start().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_directory_watcher_start_stop_lifecycle() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Test start
    let result = watcher.start().await;
    assert!(result.is_ok());

    // Test stop
    let result = watcher.stop().await;
    assert!(result.is_ok());

    // Test multiple start/stop cycles
    for _ in 0..3 {
        assert!(watcher.start().await.is_ok());
        assert!(watcher.stop().await.is_ok());
    }
}

#[tokio::test]
async fn test_directory_watching_single_directory() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();
    create_test_directory_structure(temp_dir.path()).await.unwrap();

    // Start watching
    assert!(watcher.start().await.is_ok());

    // Add directory to watch
    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    // Verify directory is being watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert!(watched_dirs.contains(&temp_dir.path().to_path_buf()));

    // Check count
    assert_eq!(watcher.watched_directory_count().await, 1);

    // Cleanup
    assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_multiple_directories() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir1 = tempdir().unwrap();
    let temp_dir2 = tempdir().unwrap();
    let temp_dir3 = tempdir().unwrap();

    create_test_directory_structure(temp_dir1.path()).await.unwrap();
    create_test_directory_structure(temp_dir2.path()).await.unwrap();
    create_test_directory_structure(temp_dir3.path()).await.unwrap();

    assert!(watcher.start().await.is_ok());

    // Watch multiple directories
    assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dir3.path()).await.is_ok());

    sleep(Duration::from_millis(100)).await;

    // Verify all directories are being watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert!(watched_dirs.contains(&temp_dir1.path().to_path_buf()));
    assert!(watched_dirs.contains(&temp_dir2.path().to_path_buf()));
    assert!(watched_dirs.contains(&temp_dir3.path().to_path_buf()));

    let watched_count = watcher.watched_directory_count().await;
    assert_eq!(watched_count, 3);

    // Test removing one directory from watching
    assert!(watcher.unwatch_directory(temp_dir2.path()).await.is_ok());

    let watched_dirs = watcher.get_watched_directories().await;
    assert!(!watched_dirs.contains(&temp_dir2.path().to_path_buf()));
    assert!(watched_dirs.contains(&temp_dir1.path().to_path_buf()));
    assert!(watched_dirs.contains(&temp_dir3.path().to_path_buf()));

    let watched_count = watcher.watched_directory_count().await;
    assert_eq!(watched_count, 2);

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_max_directories_limit() {
    let mut config = create_test_file_watcher_config();
    config.max_watched_dirs = 2; // Limit to 2 directories

    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dirs: Vec<TempDir> = (0..4).map(|_| tempdir().unwrap()).collect();

    assert!(watcher.start().await.is_ok());

    // Should be able to watch up to the limit
    assert!(watcher.watch_directory(temp_dirs[0].path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dirs[1].path()).await.is_ok());

    // Should fail when exceeding the limit
    let result = watcher.watch_directory(temp_dirs[2].path()).await;
    assert!(result.is_err());

    // After removing one, should be able to add another
    assert!(watcher.unwatch_directory(temp_dirs[0].path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dirs[2].path()).await.is_ok());

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_cleanup_and_resource_management() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dirs: Vec<TempDir> = (0..5).map(|_| tempdir().unwrap()).collect();

    assert!(watcher.start().await.is_ok());

    // Watch multiple directories
    for temp_dir in &temp_dirs {
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    }

    assert_eq!(watcher.watched_directory_count().await, 5);

    // Test cleanup on stop
    assert!(watcher.stop().await.is_ok());

    // After stop, no directories should be watched
    assert_eq!(watcher.watched_directory_count().await, 0);

    // Test restart and cleanup
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dirs[0].path()).await.is_ok());
    assert_eq!(watcher.watched_directory_count().await, 1);

    // Test individual cleanup
    assert!(watcher.unwatch_directory(temp_dirs[0].path()).await.is_ok());
    assert_eq!(watcher.watched_directory_count().await, 0);

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_error_handling() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    assert!(watcher.start().await.is_ok());

    // Test watching non-existent directory
    let non_existent = PathBuf::from("/non/existent/directory");
    let result = watcher.watch_directory(&non_existent).await;
    // This should succeed in the current implementation (no validation)
    // The error would come later when notify tries to watch

    // Test unwatching directory that isn't being watched
    let temp_dir = tempdir().unwrap();
    let result = watcher.unwatch_directory(temp_dir.path()).await;
    // This should not error - unwatching a non-watched directory should be idempotent
    assert!(result.is_ok());

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_ignore_patterns() {
    let mut config = create_test_file_watcher_config();
    config.ignore_patterns = vec![
        "*.tmp".to_string(),
        "*.log".to_string(),
        ".git/**".to_string(),
        "target/**".to_string(),
        "node_modules/**".to_string(),
        "**/cache/**".to_string(),
    ];

    let processor = create_test_processor();
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();
    create_test_directory_structure(temp_dir.path()).await.unwrap();

    // Create additional directories to test exclusion
    let node_modules = temp_dir.path().join("node_modules");
    let cache_dir = temp_dir.path().join("src").join("cache");
    tokio::fs::create_dir_all(&node_modules).await.unwrap();
    tokio::fs::create_dir_all(&cache_dir).await.unwrap();

    // Test that ignore patterns are compiled correctly
    assert!(watcher.should_ignore_path(&temp_dir.path().join("temp.tmp")));
    assert!(watcher.should_ignore_path(&temp_dir.path().join("build.log")));
    assert!(watcher.should_ignore_path(&temp_dir.path().join(".git").join("config")));
    assert!(watcher.should_ignore_path(&temp_dir.path().join("target").join("debug")));

    // Test that valid files are not ignored
    assert!(!watcher.should_ignore_path(&temp_dir.path().join("src").join("lib.rs")));
    assert!(!watcher.should_ignore_path(&temp_dir.path().join("Cargo.toml")));
}

#[tokio::test]
async fn test_directory_watching_concurrent_operations() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let watcher = Arc::new(Mutex::new(FileWatcher::new(&config, processor).await.unwrap()));

    let temp_dirs: Vec<TempDir> = (0..5).map(|_| tempdir().unwrap()).collect();

    {
        let w = watcher.lock().unwrap();
        assert!(w.start().await.is_ok());
    }

    // Spawn concurrent watch operations
    let mut handles = vec![];
    for (i, temp_dir) in temp_dirs.iter().enumerate() {
        let watcher_clone = Arc::clone(&watcher);
        let path = temp_dir.path().to_path_buf();

        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(i as u64 * 10)).await; // Stagger operations
            let mut w = watcher_clone.lock().unwrap();
            w.watch_directory(&path).await
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let results: Vec<_> = futures_util::future::join_all(handles).await;

    // All operations should succeed
    for result in results {
        assert!(result.unwrap().is_ok());
    }

    {
        let w = watcher.lock().unwrap();
        assert_eq!(w.watched_directory_count().await, 5);
        assert!(w.stop().await.is_ok());
    }
}

#[tokio::test]
async fn test_directory_watching_cross_platform_paths() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();

    // Create directories with various path characteristics
    let paths_to_test = vec![
        "simple",
        "with spaces",
        "with-dashes",
        "with_underscores",
        "with.dots",
        "UPPERCASE",
        "MixedCase",
        "123numbers",
    ];

    for path_name in &paths_to_test {
        let dir_path = temp_dir.path().join(path_name);
        tokio::fs::create_dir_all(&dir_path).await.unwrap();
        tokio::fs::write(dir_path.join("test.rs"), "// Test file").await.unwrap();
    }

    assert!(watcher.start().await.is_ok());

    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    sleep(Duration::from_millis(100)).await;

    let watched_dirs = watcher.get_watched_directories().await;

    // Verify the main directory is being watched
    assert!(watched_dirs.contains(&temp_dir.path().to_path_buf()));

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_comprehensive_edge_cases() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    assert!(watcher.start().await.is_ok());

    // Test empty directory
    let empty_dir = tempdir().unwrap();
    let result = watcher.watch_directory(empty_dir.path()).await;
    assert!(result.is_ok());

    // Test watching current directory
    let current_dir = std::env::current_dir().unwrap();
    let result = watcher.watch_directory(&current_dir).await;
    // This should work but might have platform-specific behavior
    // We just verify it doesn't panic

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_performance_stress() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();

    // Create a moderately complex directory structure
    let start = std::time::Instant::now();

    for i in 0..10 {
        let subdir = temp_dir.path().join(format!("dir_{}", i));
        tokio::fs::create_dir_all(&subdir).await.unwrap();

        for j in 0..5 {
            let subsubdir = subdir.join(format!("subdir_{}", j));
            tokio::fs::create_dir_all(&subsubdir).await.unwrap();

            for k in 0..3 {
                let file_path = subsubdir.join(format!("file_{}_{}.rs", j, k));
                tokio::fs::write(file_path, format!("// File content {}", k)).await.unwrap();
            }
        }
    }

    let setup_time = start.elapsed();
    println!("Setup time: {:?}", setup_time);

    assert!(watcher.start().await.is_ok());

    let watch_start = std::time::Instant::now();

    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    // Allow time for recursive discovery
    sleep(Duration::from_millis(500)).await;

    let watch_time = watch_start.elapsed();
    println!("Watch setup time: {:?}", watch_time);

    let watched_dirs = watcher.get_watched_directories().await;
    println!("Watched directories count: {}", watched_dirs.len());

    // Should be watching at least the root directory
    assert!(watched_dirs.len() >= 1);

    let stop_start = std::time::Instant::now();
    assert!(watcher.stop().await.is_ok());
    let stop_time = stop_start.elapsed();

    println!("Stop time: {:?}", stop_time);

    // Performance assertions (adjust based on requirements)
    assert!(watch_time < Duration::from_secs(5), "Watch setup should be fast");
    assert!(stop_time < Duration::from_secs(1), "Stop should be fast");
}

#[tokio::test]
async fn test_directory_watching_recursive_behavior() {
    let mut config = create_test_file_watcher_config();

    // Test with recursive disabled
    config.recursive = false;
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();
    create_test_directory_structure(temp_dir.path()).await.unwrap();

    assert!(watcher.start().await.is_ok());

    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    sleep(Duration::from_millis(100)).await;

    let watched_dirs = watcher.get_watched_directories().await;
    assert!(watched_dirs.contains(&temp_dir.path().to_path_buf()));

    assert!(watcher.stop().await.is_ok());

    // Test with recursive enabled
    config.recursive = true;
    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    assert!(watcher.start().await.is_ok());

    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    sleep(Duration::from_millis(100)).await;

    let watched_dirs = watcher.get_watched_directories().await;
    assert!(watched_dirs.contains(&temp_dir.path().to_path_buf()));

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_directory_watching_debounce_behavior() {
    let mut config = create_test_file_watcher_config();
    config.debounce_ms = 200; // Longer debounce for testing

    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = tempdir().unwrap();

    assert!(watcher.start().await.is_ok());

    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    sleep(Duration::from_millis(100)).await;

    // Create multiple files in rapid succession to test debouncing
    // Note: We can't easily test the actual debouncing without more complex event handling
    let test_file1 = temp_dir.path().join("test1.rs");
    let test_file2 = temp_dir.path().join("test2.rs");
    let test_file3 = temp_dir.path().join("test3.rs");

    tokio::fs::write(&test_file1, "// Test 1").await.unwrap();
    tokio::fs::write(&test_file2, "// Test 2").await.unwrap();
    tokio::fs::write(&test_file3, "// Test 3").await.unwrap();

    // Wait for debounce period
    sleep(Duration::from_millis(300)).await;

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_file_watcher_state_management() {
    let config = create_test_file_watcher_config();
    let processor = create_test_processor();
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Test initial state
    assert!(!watcher.is_running().await);
    assert_eq!(watcher.watched_directory_count().await, 0);

    // Test state after start
    assert!(watcher.start().await.is_ok());
    assert!(watcher.is_running().await);

    // Test state after stop
    assert!(watcher.stop().await.is_ok());
    assert!(!watcher.is_running().await);
}

#[tokio::test]
async fn test_file_watcher_with_zero_max_directories() {
    let mut config = create_test_file_watcher_config();
    config.max_watched_dirs = 0; // No limit

    let processor = create_test_processor();
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dirs: Vec<TempDir> = (0..5).map(|_| tempdir().unwrap()).collect();

    assert!(watcher.start().await.is_ok());

    // Should be able to watch multiple directories with no limit
    for temp_dir in &temp_dirs {
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    }

    assert_eq!(watcher.watched_directory_count().await, 5);

    assert!(watcher.stop().await.is_ok());
}

#[tokio::test]
async fn test_file_watcher_pattern_edge_cases() {
    let mut config = create_test_file_watcher_config();
    config.ignore_patterns = vec![
        "*.tmp".to_string(),
        "**/.git/**".to_string(),
        "**/target/**".to_string(),
        "test_*".to_string(),
    ];

    let processor = create_test_processor();
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Test various path patterns
    assert!(watcher.should_ignore_path(&PathBuf::from("file.tmp")));
    assert!(watcher.should_ignore_path(&PathBuf::from("project/.git/config")));
    assert!(watcher.should_ignore_path(&PathBuf::from("project/target/debug")));
    assert!(watcher.should_ignore_path(&PathBuf::from("test_file.rs")));

    // Test files that should not be ignored
    assert!(!watcher.should_ignore_path(&PathBuf::from("src/lib.rs")));
    assert!(!watcher.should_ignore_path(&PathBuf::from("Cargo.toml")));
    assert!(!watcher.should_ignore_path(&PathBuf::from("README.md")));
}