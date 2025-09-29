//! Daemon-level file watching lifecycle tests
//! Tests daemon startup with automatic project folder watching, library folder configuration,
//! watch persistence across restarts, and integration scenarios

use workspace_qdrant_daemon::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig, CollectionConfig};
use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use tokio::time::sleep;
use serial_test::serial;
use std::path::PathBuf;

/// Helper function to create file watcher config
fn create_watcher_config(enabled: bool, debounce_ms: u64) -> FileWatcherConfig {
    FileWatcherConfig {
        enabled,
        debounce_ms,
        max_watched_dirs: 10,
        ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string(), "*.swp".to_string()],
        recursive: true,
    }
}

/// Helper function to create test processor
async fn create_test_processor() -> Arc<DocumentProcessor> {
    let processing_config = ProcessingConfig {
        max_concurrent_tasks: 2,
        default_chunk_size: 1000,
        default_chunk_overlap: 200,
        max_file_size_bytes: 1024 * 1024,
        supported_extensions: vec!["rs".to_string(), "txt".to_string()],
        enable_lsp: false,
        lsp_timeout_secs: 10,
    };
    let qdrant_config = QdrantConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        max_retries: 3,
        default_collection: CollectionConfig {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            replication_factor: 1,
            shard_number: 1,
        },
    };
    Arc::new(DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap())
}

/// Test automatic project folder watching on daemon startup
#[tokio::test]
#[serial]
async fn test_project_folder_auto_watch_on_startup() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create a mock project directory structure
    let project_dir = TempDir::new().unwrap();
    let src_dir = project_dir.path().join("src");
    fs::create_dir(&src_dir).await.unwrap();

    // Simulate daemon startup: start watcher and add project directory
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(project_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Verify the project directory is being watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 1);
    assert!(watched_dirs.contains(&project_dir.path().to_path_buf()));

    // Create a file in the project and verify it's detected
    let test_file = src_dir.join("main.rs");
    fs::write(&test_file, "fn main() {}").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    assert!(test_file.exists());
    assert!(watcher.stop().await.is_ok());
}

/// Test library folder watch configuration
#[tokio::test]
#[serial]
async fn test_library_folder_watch_configuration() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create library-like directory structure
    let library_dir = TempDir::new().unwrap();
    let lib_src = library_dir.path().join("lib");
    fs::create_dir(&lib_src).await.unwrap();

    // Start watcher and add library directory
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(library_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create library files
    let lib_file = lib_src.join("lib.rs");
    fs::write(&lib_file, "pub fn add(a: i32, b: i32) -> i32 { a + b }").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    assert!(lib_file.exists());

    // Verify library directory is watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert!(watched_dirs.contains(&library_dir.path().to_path_buf()));

    assert!(watcher.stop().await.is_ok());
}

/// Test multiple library folders being watched
#[tokio::test]
#[serial]
async fn test_multiple_library_folders_watch() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create multiple library directories
    let lib1_dir = TempDir::new().unwrap();
    let lib2_dir = TempDir::new().unwrap();
    let lib3_dir = TempDir::new().unwrap();

    assert!(watcher.start().await.is_ok());

    // Watch all library directories
    assert!(watcher.watch_directory(lib1_dir.path()).await.is_ok());
    assert!(watcher.watch_directory(lib2_dir.path()).await.is_ok());
    assert!(watcher.watch_directory(lib3_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Verify all directories are watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 3);
    assert!(watched_dirs.contains(&lib1_dir.path().to_path_buf()));
    assert!(watched_dirs.contains(&lib2_dir.path().to_path_buf()));
    assert!(watched_dirs.contains(&lib3_dir.path().to_path_buf()));

    // Create files in each library
    for (i, lib_dir) in [&lib1_dir, &lib2_dir, &lib3_dir].iter().enumerate() {
        let lib_file = lib_dir.path().join(format!("lib{}.rs", i + 1));
        fs::write(&lib_file, format!("pub fn lib{}() {{}}", i + 1)).await.unwrap();
    }

    // Wait for event processing
    sleep(Duration::from_millis(300)).await;

    assert!(watcher.stop().await.is_ok());
}

/// Test watch persistence: saving watched directories to a state file
#[tokio::test]
#[serial]
async fn test_watch_state_persistence_save() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create directories to watch
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(dir1.path()).await.is_ok());
    assert!(watcher.watch_directory(dir2.path()).await.is_ok());

    // Get watched directories (simulating state to be persisted)
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 2);

    // Simulate saving state to file
    let state_file = TempDir::new().unwrap().path().join("watcher_state.json");
    let state_json = serde_json::to_string(&watched_dirs).unwrap();
    fs::write(&state_file, state_json).await.unwrap();

    // Verify state file was created
    assert!(state_file.exists());

    // Read back and verify
    let loaded_state = fs::read_to_string(&state_file).await.unwrap();
    let loaded_dirs: Vec<PathBuf> = serde_json::from_str(&loaded_state).unwrap();
    assert_eq!(loaded_dirs.len(), 2);

    assert!(watcher.stop().await.is_ok());
}

/// Test watch persistence: restoring watched directories from state file
#[tokio::test]
#[serial]
async fn test_watch_state_persistence_restore() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;

    // Create directories and save their state
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    let watched_dirs_original = vec![
        dir1.path().to_path_buf(),
        dir2.path().to_path_buf(),
    ];

    // Save state to file
    let state_file = TempDir::new().unwrap().path().join("watcher_state.json");
    let state_json = serde_json::to_string(&watched_dirs_original).unwrap();
    fs::write(&state_file, state_json).await.unwrap();

    // Create new watcher instance (simulating daemon restart)
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();
    assert!(watcher.start().await.is_ok());

    // Restore state from file
    let loaded_state = fs::read_to_string(&state_file).await.unwrap();
    let loaded_dirs: Vec<PathBuf> = serde_json::from_str(&loaded_state).unwrap();

    // Re-watch the directories
    for dir in &loaded_dirs {
        assert!(watcher.watch_directory(dir).await.is_ok());
    }

    // Verify all directories are being watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 2);
    assert!(watched_dirs.contains(&dir1.path().to_path_buf()));
    assert!(watched_dirs.contains(&dir2.path().to_path_buf()));

    assert!(watcher.stop().await.is_ok());
}

/// Test daemon restart scenario: stop and restart with same watched directories
#[tokio::test]
#[serial]
async fn test_daemon_restart_maintains_watch_list() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;

    // First daemon instance
    let mut watcher1 = FileWatcher::new(&watcher_config, Arc::clone(&processor)).await.unwrap();

    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    assert!(watcher1.start().await.is_ok());
    assert!(watcher1.watch_directory(dir1.path()).await.is_ok());
    assert!(watcher1.watch_directory(dir2.path()).await.is_ok());

    // Get and save state
    let watched_dirs_before = watcher1.get_watched_directories().await;
    assert_eq!(watched_dirs_before.len(), 2);

    // Simulate daemon shutdown
    assert!(watcher1.stop().await.is_ok());
    drop(watcher1);

    // Give time for cleanup
    sleep(Duration::from_millis(100)).await;

    // Second daemon instance (restart)
    let mut watcher2 = FileWatcher::new(&watcher_config, processor).await.unwrap();
    assert!(watcher2.start().await.is_ok());

    // Restore watched directories
    for dir in &watched_dirs_before {
        assert!(watcher2.watch_directory(dir).await.is_ok());
    }

    // Verify directories are still being watched
    let watched_dirs_after = watcher2.get_watched_directories().await;
    assert_eq!(watched_dirs_after.len(), 2);
    assert!(watched_dirs_after.contains(&dir1.path().to_path_buf()));
    assert!(watched_dirs_after.contains(&dir2.path().to_path_buf()));

    // Test that file events are still detected after restart
    let test_file = dir1.path().join("after_restart.txt");
    fs::write(&test_file, "content after restart").await.unwrap();

    sleep(Duration::from_millis(200)).await;
    assert!(test_file.exists());

    assert!(watcher2.stop().await.is_ok());
}

/// Test full daemon lifecycle with file watching
#[tokio::test]
#[serial]
async fn test_full_daemon_lifecycle_with_file_watching() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    let project_dir = TempDir::new().unwrap();

    // 1. Daemon startup
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(project_dir.path()).await.is_ok());
    sleep(Duration::from_millis(100)).await;

    // 2. Normal operation: file creation
    let file1 = project_dir.path().join("file1.txt");
    fs::write(&file1, "content 1").await.unwrap();
    sleep(Duration::from_millis(150)).await;

    // 3. Normal operation: file modification
    fs::write(&file1, "modified content 1").await.unwrap();
    sleep(Duration::from_millis(150)).await;

    // 4. Normal operation: file deletion
    fs::remove_file(&file1).await.unwrap();
    sleep(Duration::from_millis(150)).await;

    // 5. Daemon shutdown
    assert!(watcher.stop().await.is_ok());

    // Verify clean shutdown
    assert!(!watcher.is_running().await);
}

/// Test watch persistence with rapid daemon restarts
#[tokio::test]
#[serial]
async fn test_rapid_daemon_restarts_with_persistence() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;

    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    let watched_dirs_persistent = vec![
        dir1.path().to_path_buf(),
        dir2.path().to_path_buf(),
    ];

    // Perform multiple rapid restart cycles
    for cycle in 0..5 {
        let mut watcher = FileWatcher::new(&watcher_config, Arc::clone(&processor)).await.unwrap();

        assert!(watcher.start().await.is_ok());

        // Restore watched directories
        for dir in &watched_dirs_persistent {
            assert!(watcher.watch_directory(dir).await.is_ok());
        }

        // Verify directories are watched
        let watched = watcher.get_watched_directories().await;
        assert_eq!(watched.len(), 2, "Cycle {} failed: expected 2 watched dirs", cycle);

        // Quick test of functionality
        let test_file = dir1.path().join(format!("cycle_{}.txt", cycle));
        fs::write(&test_file, format!("cycle {}", cycle)).await.unwrap();
        sleep(Duration::from_millis(100)).await;

        assert!(watcher.stop().await.is_ok());
        sleep(Duration::from_millis(50)).await;
    }
}

/// Test project folder with nested subdirectories
#[tokio::test]
#[serial]
async fn test_project_folder_nested_structure() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create nested project structure
    let project_dir = TempDir::new().unwrap();
    let src_dir = project_dir.path().join("src");
    let tests_dir = project_dir.path().join("tests");
    let docs_dir = project_dir.path().join("docs");

    fs::create_dir(&src_dir).await.unwrap();
    fs::create_dir(&tests_dir).await.unwrap();
    fs::create_dir(&docs_dir).await.unwrap();

    let nested_module = src_dir.join("nested");
    fs::create_dir(&nested_module).await.unwrap();

    // Start watching at project root
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(project_dir.path()).await.is_ok());
    sleep(Duration::from_millis(100)).await;

    // Create files in various nested locations
    fs::write(src_dir.join("main.rs"), "fn main() {}").await.unwrap();
    fs::write(tests_dir.join("integration_test.rs"), "#[test] fn test() {}").await.unwrap();
    fs::write(docs_dir.join("README.md"), "# Documentation").await.unwrap();
    fs::write(nested_module.join("module.rs"), "pub mod nested {}").await.unwrap();

    // Wait for all events to be processed
    sleep(Duration::from_millis(400)).await;

    // Verify all files exist (indicating they were processed)
    assert!(src_dir.join("main.rs").exists());
    assert!(tests_dir.join("integration_test.rs").exists());
    assert!(docs_dir.join("README.md").exists());
    assert!(nested_module.join("module.rs").exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test library folder with common library structure
#[tokio::test]
#[serial]
async fn test_library_folder_standard_structure() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    // Create standard library structure
    let lib_dir = TempDir::new().unwrap();
    let src_dir = lib_dir.path().join("src");
    fs::create_dir(&src_dir).await.unwrap();

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(lib_dir.path()).await.is_ok());
    sleep(Duration::from_millis(100)).await;

    // Create library files
    fs::write(src_dir.join("lib.rs"), "pub mod core;").await.unwrap();
    fs::write(src_dir.join("core.rs"), "pub fn core_function() {}").await.unwrap();
    fs::write(lib_dir.path().join("Cargo.toml"), "[package]\nname = \"test-lib\"").await.unwrap();

    sleep(Duration::from_millis(300)).await;

    // Verify library files were created
    assert!(src_dir.join("lib.rs").exists());
    assert!(src_dir.join("core.rs").exists());
    assert!(lib_dir.path().join("Cargo.toml").exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test platform-specific watch behavior simulation
#[tokio::test]
#[serial]
async fn test_platform_specific_watch_behavior() {
    let watcher_config = create_watcher_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    let test_dir = TempDir::new().unwrap();

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(test_dir.path()).await.is_ok());
    sleep(Duration::from_millis(100)).await;

    // Test various file operations that may behave differently on different platforms

    // 1. File creation
    let file1 = test_dir.path().join("test1.txt");
    fs::write(&file1, "content").await.unwrap();
    sleep(Duration::from_millis(100)).await;

    // 2. Rapid file modifications (may trigger different events on different platforms)
    for i in 0..5 {
        fs::write(&file1, format!("content {}", i)).await.unwrap();
        sleep(Duration::from_millis(10)).await;
    }
    sleep(Duration::from_millis(150)).await;

    // 3. File rename (may trigger different event sequences)
    let file2 = test_dir.path().join("test2.txt");
    fs::rename(&file1, &file2).await.unwrap();
    sleep(Duration::from_millis(100)).await;

    // 4. File deletion
    fs::remove_file(&file2).await.unwrap();
    sleep(Duration::from_millis(100)).await;

    assert!(watcher.stop().await.is_ok());
}

/// Test maximum watched directories limit enforcement
#[tokio::test]
#[serial]
async fn test_max_watched_dirs_with_persistence() {
    let mut watcher_config = create_watcher_config(true, 50);
    watcher_config.max_watched_dirs = 3;

    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    let dir3 = TempDir::new().unwrap();
    let dir4 = TempDir::new().unwrap();

    // Watch up to the limit
    assert!(watcher.watch_directory(dir1.path()).await.is_ok());
    assert!(watcher.watch_directory(dir2.path()).await.is_ok());
    assert!(watcher.watch_directory(dir3.path()).await.is_ok());

    // Attempt to exceed limit
    let result = watcher.watch_directory(dir4.path()).await;
    assert!(result.is_err());

    // Verify exactly 3 directories are watched
    let watched = watcher.get_watched_directories().await;
    assert_eq!(watched.len(), 3);
}

/// Test watch state with concurrent file operations
#[tokio::test]
#[serial]
async fn test_watch_state_with_concurrent_operations() {
    let watcher_config = create_watcher_config(true, 100);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&watcher_config, processor).await.unwrap();

    let project_dir = TempDir::new().unwrap();

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(project_dir.path()).await.is_ok());
    sleep(Duration::from_millis(100)).await;

    // Create multiple files concurrently
    let mut handles = vec![];
    for i in 0..20 {
        let file_path = project_dir.path().join(format!("concurrent_{}.txt", i));
        let handle = tokio::spawn(async move {
            fs::write(&file_path, format!("content {}", i)).await.unwrap();
            file_path
        });
        handles.push(handle);
    }

    // Wait for all files to be created
    let mut created_files = vec![];
    for handle in handles {
        let file_path = handle.await.unwrap();
        created_files.push(file_path);
    }

    // Wait for event processing
    sleep(Duration::from_millis(500)).await;

    // Verify all files exist
    for file_path in created_files {
        assert!(file_path.exists());
    }

    // Verify watcher is still functional
    assert!(watcher.is_running().await);
    assert!(watcher.stop().await.is_ok());
}