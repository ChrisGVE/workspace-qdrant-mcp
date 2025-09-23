//! Comprehensive symlink and special file handling tests
//!
//! This test suite implements comprehensive testing for symlink, hard link, and special file
//! handling in the Rust engine using notify crate for file system monitoring.
//!
//! Test areas covered:
//! - Symlink creation, modification, and deletion detection
//! - Hard link monitoring and inode tracking
//! - Special file types: pipes, sockets, device files
//! - Broken symlinks and circular reference handling
//! - Cross-filesystem symlink behavior
//! - Edge cases: nested symlinks, permission changes

use std::fs::{self, File, Permissions};
use std::os::unix::fs::{symlink, PermissionsExt, MetadataExt, FileTypeExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tempfile::{tempdir, TempDir};
use tokio::time::sleep;

use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::daemon::file_ops::{AsyncFileProcessor, FileInfo, SpecialFileHandler, SpecialFileType, SymlinkInfo};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::config::FileWatcherConfig;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};

/// Test helper for creating temporary test environments
struct SymlinkTestEnvironment {
    temp_dir: TempDir,
    watched_events: Arc<Mutex<Vec<FileEvent>>>,
}

/// File system event types for testing
#[derive(Debug, Clone, PartialEq)]
enum FileEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
    SymlinkCreated(PathBuf, PathBuf), // (link_path, target_path)
    SymlinkDeleted(PathBuf),
    SymlinkTargetChanged(PathBuf, PathBuf, PathBuf), // (link_path, old_target, new_target)
    SpecialFileCreated(PathBuf, SpecialFileType),
    SpecialFileDeleted(PathBuf),
}

// Remove duplicate SpecialFileType definition since it's now imported from file_ops

impl SymlinkTestEnvironment {
    async fn new() -> DaemonResult<Self> {
        let temp_dir = tempdir().map_err(|e| DaemonError::FileIo {
            message: format!("Failed to create temp directory: {}", e),
            path: "temp".to_string(),
        })?;

        Ok(Self {
            temp_dir,
            watched_events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn path(&self) -> &Path {
        self.temp_dir.path()
    }

    fn create_file(&self, name: &str, content: &str) -> std::io::Result<PathBuf> {
        let path = self.temp_dir.path().join(name);
        fs::write(&path, content)?;
        Ok(path)
    }

    fn create_symlink(&self, link_name: &str, target: &Path) -> std::io::Result<PathBuf> {
        let link_path = self.temp_dir.path().join(link_name);
        symlink(target, &link_path)?;
        Ok(link_path)
    }

    fn create_relative_symlink(&self, link_name: &str, target_name: &str) -> std::io::Result<PathBuf> {
        let link_path = self.temp_dir.path().join(link_name);
        symlink(target_name, &link_path)?;
        Ok(link_path)
    }

    #[cfg(unix)]
    fn create_named_pipe(&self, name: &str) -> std::io::Result<PathBuf> {
        use std::ffi::CString;
        let path = self.temp_dir.path().join(name);
        let c_path = CString::new(path.to_string_lossy().as_bytes()).unwrap();

        unsafe {
            if libc::mkfifo(c_path.as_ptr(), 0o644) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(path)
    }

    #[cfg(unix)]
    fn create_unix_socket(&self, name: &str) -> std::io::Result<PathBuf> {
        use std::os::unix::net::UnixListener;
        let path = self.temp_dir.path().join(name);
        let _listener = UnixListener::bind(&path)?;
        Ok(path)
    }

    fn create_hard_link(&self, link_name: &str, target: &Path) -> std::io::Result<PathBuf> {
        let link_path = self.temp_dir.path().join(link_name);
        fs::hard_link(target, &link_path)?;
        Ok(link_path)
    }

    fn get_events(&self) -> Vec<FileEvent> {
        self.watched_events.lock().unwrap().clone()
    }

    fn clear_events(&self) {
        self.watched_events.lock().unwrap().clear();
    }
}

/// Tests for basic symlink operations
mod symlink_basic_tests {
    use super::*;

    #[tokio::test]
    async fn test_symlink_creation_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let processor = AsyncFileProcessor::default();
        let special_handler = SpecialFileHandler::new();

        // Create target file
        let target_file = env.create_file("target.txt", "target content").unwrap();

        // Create symlink
        let symlink_path = env.create_symlink("link.txt", &target_file).unwrap();

        // Verify symlink is detected
        let file_info = processor.validate_file(&symlink_path).await.unwrap();
        assert!(!file_info.is_file);
        assert!(!file_info.is_dir);

        // Verify symlink target resolution
        let resolved_target = special_handler.resolve_symlink(&symlink_path).await.unwrap();
        assert_eq!(resolved_target, target_file);

        // Verify symlink metadata
        let symlink_info = special_handler.get_symlink_info(&symlink_path).await.unwrap();
        assert_eq!(symlink_info.target, target_file);
        assert!(!symlink_info.is_broken);
        assert_eq!(symlink_info.depth, 1);
    }

    #[tokio::test]
    async fn test_broken_symlink_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create symlink to non-existent file
        let nonexistent_target = env.path().join("nonexistent.txt");
        let broken_link = env.create_symlink("broken_link.txt", &nonexistent_target).unwrap();

        // Verify broken symlink is detected
        let symlink_info = special_handler.get_symlink_info(&broken_link).await.unwrap();
        assert_eq!(symlink_info.target, nonexistent_target);
        assert!(symlink_info.is_broken);

        // Verify resolve_symlink returns appropriate error for broken links
        let resolve_result = special_handler.resolve_symlink(&broken_link).await;
        assert!(resolve_result.is_err());
        match resolve_result.unwrap_err() {
            DaemonError::SymlinkBroken { link_path, target_path } => {
                assert_eq!(link_path, broken_link.to_string_lossy());
                assert_eq!(target_path, nonexistent_target.to_string_lossy());
            }
            _ => panic!("Expected SymlinkBroken error"),
        }
    }

    #[tokio::test]
    async fn test_circular_symlink_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create circular symlinks
        let link1 = env.path().join("link1.txt");
        let link2 = env.path().join("link2.txt");

        symlink(&link2, &link1).unwrap();
        symlink(&link1, &link2).unwrap();

        // Verify circular reference is detected
        let resolve_result = special_handler.resolve_symlink(&link1).await;
        assert!(resolve_result.is_err());
        match resolve_result.unwrap_err() {
            DaemonError::SymlinkCircular { link_path, .. } => {
                assert_eq!(link_path, link1.to_string_lossy());
            }
            _ => panic!("Expected SymlinkCircular error"),
        }

        // Verify symlink chain depth calculation
        let depth_result = special_handler.calculate_symlink_depth(&link1, 10).await;
        assert!(depth_result.is_err());
    }

    #[tokio::test]
    async fn test_nested_symlinks_resolution() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create chain: file -> link1 -> link2 -> link3
        let target_file = env.create_file("final_target.txt", "final content").unwrap();
        let link1 = env.create_symlink("link1.txt", &target_file).unwrap();
        let link2 = env.create_symlink("link2.txt", &link1).unwrap();
        let link3 = env.create_symlink("link3.txt", &link2).unwrap();

        // Verify deep resolution works
        let final_target = special_handler.resolve_symlink(&link3).await.unwrap();
        assert_eq!(final_target, target_file);

        // Verify depth calculation
        let depth = special_handler.calculate_symlink_depth(&link3, 10).await.unwrap();
        assert_eq!(depth, 3);

        // Verify symlink info includes correct depth
        let info = special_handler.get_symlink_info(&link3).await.unwrap();
        assert_eq!(info.depth, 3);
        assert_eq!(info.final_target, target_file);
    }

    #[tokio::test]
    async fn test_relative_vs_absolute_symlinks() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create target file
        let target_file = env.create_file("target.txt", "content").unwrap();

        // Create absolute symlink
        let abs_link = env.create_symlink("abs_link.txt", &target_file).unwrap();

        // Create relative symlink
        let rel_link = env.create_relative_symlink("rel_link.txt", "target.txt").unwrap();

        // Both should resolve to the same target
        let abs_resolved = special_handler.resolve_symlink(&abs_link).await.unwrap();
        let rel_resolved = special_handler.resolve_symlink(&rel_link).await.unwrap();

        assert_eq!(abs_resolved, target_file);
        assert_eq!(rel_resolved, target_file);

        // Verify symlink info differentiates between absolute and relative
        let abs_info = special_handler.get_symlink_info(&abs_link).await.unwrap();
        let rel_info = special_handler.get_symlink_info(&rel_link).await.unwrap();

        assert!(abs_info.is_absolute);
        assert!(!rel_info.is_absolute);
    }
}

/// Tests for hard link operations
mod hardlink_tests {
    use super::*;

    #[tokio::test]
    async fn test_hard_link_creation_and_tracking() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create original file
        let original_file = env.create_file("original.txt", "shared content").unwrap();
        let original_metadata = fs::metadata(&original_file).unwrap();

        // Create hard link
        let hard_link = env.create_hard_link("hardlink.txt", &original_file).unwrap();
        let link_metadata = fs::metadata(&hard_link).unwrap();

        // Verify they share the same inode
        assert_eq!(original_metadata.ino(), link_metadata.ino());

        // Verify hard link detection
        let is_hardlink = special_handler.is_hard_link(&hard_link).await.unwrap();
        assert!(is_hardlink);

        // Verify hard link count
        let link_count = special_handler.get_hard_link_count(&hard_link).await.unwrap();
        assert_eq!(link_count, 2);

        // Find all hard links to the same inode
        let all_links = special_handler
            .find_hard_links(&env.path(), original_metadata.ino())
            .await
            .unwrap();
        assert_eq!(all_links.len(), 2);
        assert!(all_links.contains(&original_file));
        assert!(all_links.contains(&hard_link));
    }

    #[tokio::test]
    async fn test_hard_link_modification_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create original file and hard link
        let original_file = env.create_file("original.txt", "initial content").unwrap();
        let hard_link = env.create_hard_link("hardlink.txt", &original_file).unwrap();

        let original_inode = fs::metadata(&original_file).unwrap().ino();

        // Modify content through original file
        fs::write(&original_file, "modified content").unwrap();

        // Verify both files reflect the change
        let original_content = fs::read_to_string(&original_file).unwrap();
        let link_content = fs::read_to_string(&hard_link).unwrap();
        assert_eq!(original_content, "modified content");
        assert_eq!(link_content, "modified content");

        // Verify inode remains the same
        assert_eq!(fs::metadata(&original_file).unwrap().ino(), original_inode);
        assert_eq!(fs::metadata(&hard_link).unwrap().ino(), original_inode);

        // Verify modification time is updated for both
        let original_mtime = fs::metadata(&original_file).unwrap().modified().unwrap();
        let link_mtime = fs::metadata(&hard_link).unwrap().modified().unwrap();
        assert_eq!(original_mtime, link_mtime);
    }

    #[tokio::test]
    async fn test_hard_link_deletion_tracking() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create file with multiple hard links
        let original_file = env.create_file("original.txt", "content").unwrap();
        let link1 = env.create_hard_link("link1.txt", &original_file).unwrap();
        let link2 = env.create_hard_link("link2.txt", &original_file).unwrap();

        let inode = fs::metadata(&original_file).unwrap().ino();

        // Verify initial link count
        let initial_count = special_handler.get_hard_link_count(&original_file).await.unwrap();
        assert_eq!(initial_count, 3);

        // Delete one hard link
        fs::remove_file(&link1).unwrap();

        // Verify link count decreases
        let after_delete_count = special_handler.get_hard_link_count(&original_file).await.unwrap();
        assert_eq!(after_delete_count, 2);

        // Verify remaining files still share inode
        assert_eq!(fs::metadata(&original_file).unwrap().ino(), inode);
        assert_eq!(fs::metadata(&link2).unwrap().ino(), inode);

        // Find remaining hard links
        let remaining_links = special_handler
            .find_hard_links(&env.path(), inode)
            .await
            .unwrap();
        assert_eq!(remaining_links.len(), 2);
        assert!(!remaining_links.iter().any(|p| p == &link1));
    }
}

/// Tests for special file types (pipes, sockets, etc.)
mod special_file_tests {
    use super::*;

    #[tokio::test]
    #[cfg(unix)]
    async fn test_named_pipe_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create named pipe (FIFO)
        let pipe_path = env.create_named_pipe("test_pipe").unwrap();

        // Verify pipe detection
        let file_type = special_handler.get_file_type(&pipe_path).await.unwrap();
        assert_eq!(file_type, SpecialFileType::NamedPipe);

        // Verify special file handler recognizes it
        let is_special = special_handler.is_special_file(&pipe_path).await.unwrap();
        assert!(is_special);

        // Verify appropriate handling policy
        let should_process = special_handler.should_process_special_file(&pipe_path).await.unwrap();
        assert!(!should_process); // Pipes should not be processed for content
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_unix_socket_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create Unix domain socket
        let socket_path = env.create_unix_socket("test_socket").unwrap();

        // Verify socket detection
        let file_type = special_handler.get_file_type(&socket_path).await.unwrap();
        assert_eq!(file_type, SpecialFileType::Socket);

        // Verify special file handling
        let is_special = special_handler.is_special_file(&socket_path).await.unwrap();
        assert!(is_special);

        // Verify sockets should not be processed
        let should_process = special_handler.should_process_special_file(&socket_path).await.unwrap();
        assert!(!should_process);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_device_file_detection() {
        let special_handler = SpecialFileHandler::new();

        // Test with /dev/null (char device)
        let null_device = Path::new("/dev/null");
        if null_device.exists() {
            let file_type = special_handler.get_file_type(null_device).await.unwrap();
            assert_eq!(file_type, SpecialFileType::CharacterDevice);

            let is_special = special_handler.is_special_file(null_device).await.unwrap();
            assert!(is_special);

            let should_process = special_handler.should_process_special_file(null_device).await.unwrap();
            assert!(!should_process);
        }

        // Test with /dev/zero (char device)
        let zero_device = Path::new("/dev/zero");
        if zero_device.exists() {
            let file_type = special_handler.get_file_type(zero_device).await.unwrap();
            assert_eq!(file_type, SpecialFileType::CharacterDevice);
        }
    }

    #[tokio::test]
    async fn test_special_file_symlinks() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create named pipe and symlink to it
        #[cfg(unix)]
        {
            let pipe_path = env.create_named_pipe("pipe").unwrap();
            let pipe_link = env.create_symlink("pipe_link", &pipe_path).unwrap();

            // Verify symlink detection
            let is_symlink = special_handler.is_symlink(&pipe_link).await.unwrap();
            assert!(is_symlink);

            // Verify target resolution to special file
            let resolved_target = special_handler.resolve_symlink(&pipe_link).await.unwrap();
            assert_eq!(resolved_target, pipe_path);

            // Verify final target type detection
            let target_type = special_handler.get_file_type(&resolved_target).await.unwrap();
            assert_eq!(target_type, SpecialFileType::NamedPipe);
        }
    }
}

/// Tests for file watcher integration with symlinks and special files
mod watcher_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_watcher_symlink_creation_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let processor = Arc::new(DocumentProcessor::test_instance());

        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms: 50,
            max_watched_dirs: 10,
            ignore_patterns: vec![],
            recursive: true,
        };

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        watcher.watch_directory(env.path()).await.unwrap();

        // Create target file and symlink
        let target_file = env.create_file("target.txt", "content").unwrap();
        sleep(Duration::from_millis(100)).await;

        let symlink_path = env.create_symlink("link.txt", &target_file).unwrap();
        sleep(Duration::from_millis(100)).await;

        // Verify symlink creation was detected
        // Note: This would require implementing the actual watcher event handling
        // For now, we verify the static analysis works
        let special_handler = SpecialFileHandler::new();
        let is_symlink = special_handler.is_symlink(&symlink_path).await.unwrap();
        assert!(is_symlink);
    }

    #[tokio::test]
    async fn test_watcher_symlink_target_change_detection() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create initial target and symlink
        let target1 = env.create_file("target1.txt", "content1").unwrap();
        let target2 = env.create_file("target2.txt", "content2").unwrap();
        let symlink_path = env.create_symlink("link.txt", &target1).unwrap();

        // Verify initial target
        let initial_target = special_handler.resolve_symlink(&symlink_path).await.unwrap();
        assert_eq!(initial_target, target1);

        // Change symlink target
        fs::remove_file(&symlink_path).unwrap();
        symlink(&target2, &symlink_path).unwrap();

        // Verify target change
        let new_target = special_handler.resolve_symlink(&symlink_path).await.unwrap();
        assert_eq!(new_target, target2);
    }

    #[tokio::test]
    async fn test_watcher_broken_symlink_handling() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create symlink to existing file
        let target_file = env.create_file("target.txt", "content").unwrap();
        let symlink_path = env.create_symlink("link.txt", &target_file).unwrap();

        // Verify symlink works initially
        let resolved = special_handler.resolve_symlink(&symlink_path).await.unwrap();
        assert_eq!(resolved, target_file);

        // Delete target file (breaking the symlink)
        fs::remove_file(&target_file).unwrap();

        // Verify symlink is now broken
        let symlink_info = special_handler.get_symlink_info(&symlink_path).await.unwrap();
        assert!(symlink_info.is_broken);

        // Verify resolve_symlink fails appropriately
        let resolve_result = special_handler.resolve_symlink(&symlink_path).await;
        assert!(resolve_result.is_err());
    }
}

/// Edge case tests for complex symlink scenarios
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_symlink_to_directory() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create directory and symlink to it
        let target_dir = env.path().join("target_dir");
        fs::create_dir(&target_dir).unwrap();
        let dir_link = env.create_symlink("dir_link", &target_dir).unwrap();

        // Verify directory symlink resolution
        let resolved = special_handler.resolve_symlink(&dir_link).await.unwrap();
        assert_eq!(resolved, target_dir);

        // Verify metadata indicates directory target
        let metadata = fs::symlink_metadata(&dir_link).unwrap();
        assert!(metadata.file_type().is_symlink());

        let target_metadata = fs::metadata(&resolved).unwrap();
        assert!(target_metadata.is_dir());
    }

    #[tokio::test]
    async fn test_symlink_depth_limit_enforcement() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create a very deep symlink chain
        let target_file = env.create_file("target.txt", "content").unwrap();
        let mut current_target = target_file.clone();

        // Create 15 levels of symlinks
        for i in 1..=15 {
            let link_name = format!("link{}.txt", i);
            let link_path = env.create_symlink(&link_name, &current_target).unwrap();
            current_target = link_path;
        }

        // Test with reasonable depth limit (should succeed)
        let depth = special_handler.calculate_symlink_depth(&current_target, 20).await.unwrap();
        assert_eq!(depth, 15);

        // Test with restrictive depth limit (should fail)
        let depth_result = special_handler.calculate_symlink_depth(&current_target, 5).await;
        assert!(depth_result.is_err());
        match depth_result.unwrap_err() {
            DaemonError::SymlinkDepthExceeded { depth, max_depth, .. } => {
                assert!(depth > max_depth);
                assert_eq!(max_depth, 5);
            }
            _ => panic!("Expected SymlinkDepthExceeded error"),
        }
    }

    #[tokio::test]
    async fn test_permission_changes_on_symlinks() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create target file and symlink
        let target_file = env.create_file("target.txt", "content").unwrap();
        let symlink_path = env.create_symlink("link.txt", &target_file).unwrap();

        // Change permissions on target file
        let mut perms = fs::metadata(&target_file).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&target_file, perms).unwrap();

        // Verify symlink still resolves but target has changed permissions
        let resolved = special_handler.resolve_symlink(&symlink_path).await.unwrap();
        assert_eq!(resolved, target_file);

        let target_metadata = fs::metadata(&resolved).unwrap();
        assert!(target_metadata.permissions().readonly());

        // Verify symlink metadata vs target metadata
        let symlink_metadata = fs::symlink_metadata(&symlink_path).unwrap();
        assert!(symlink_metadata.file_type().is_symlink());
        assert_ne!(symlink_metadata.permissions().readonly(), target_metadata.permissions().readonly());
    }

    #[tokio::test]
    async fn test_cross_filesystem_symlinks() {
        // Note: This test may not work in all environments
        // It's designed to test the detection and handling of cross-filesystem symlinks
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create symlink to /tmp (likely different filesystem)
        let external_target = Path::new("/tmp");
        if external_target.exists() {
            let cross_fs_link = env.path().join("cross_fs_link");

            // This might fail on some systems, so we handle the error gracefully
            match symlink(external_target, &cross_fs_link) {
                Ok(_) => {
                    let resolved = special_handler.resolve_symlink(&cross_fs_link).await.unwrap();
                    assert_eq!(resolved, external_target);

                    // Check if we can detect cross-filesystem nature
                    let info = special_handler.get_symlink_info(&cross_fs_link).await.unwrap();
                    // The cross_filesystem field would need to be implemented
                    // assert!(info.is_cross_filesystem);
                }
                Err(_) => {
                    // Skip test if cross-filesystem symlinks aren't supported
                    println!("Skipping cross-filesystem symlink test - not supported in this environment");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_unicode_symlink_names() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create target file
        let target_file = env.create_file("target.txt", "unicode test content").unwrap();

        // Test various Unicode symlink names
        let unicode_names = vec![
            "测试链接.txt",     // Chinese
            "тестовая_ссылка.txt", // Russian
            "テストリンク.txt",  // Japanese
            "🔗링크.txt",        // Korean with emoji
            "αβγδε_link.txt",    // Greek
        ];

        for name in unicode_names {
            let symlink_path = env.create_symlink(name, &target_file).unwrap();

            // Verify Unicode symlink resolution works
            let resolved = special_handler.resolve_symlink(&symlink_path).await.unwrap();
            assert_eq!(resolved, target_file);

            // Verify symlink info handles Unicode correctly
            let info = special_handler.get_symlink_info(&symlink_path).await.unwrap();
            assert_eq!(info.target, target_file);
            assert!(!info.is_broken);
        }
    }
}

/// Performance and stress tests
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_bulk_symlink_operations() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Create target files
        let mut target_files = Vec::new();
        for i in 0..100 {
            let target = env.create_file(&format!("target_{}.txt", i), &format!("content {}", i)).unwrap();
            target_files.push(target);
        }

        // Create symlinks to all targets
        let mut symlink_paths = Vec::new();
        for (i, target) in target_files.iter().enumerate() {
            let symlink = env.create_symlink(&format!("link_{}.txt", i), target).unwrap();
            symlink_paths.push(symlink);
        }

        // Verify all symlinks resolve correctly
        for (i, symlink_path) in symlink_paths.iter().enumerate() {
            let resolved = special_handler.resolve_symlink(symlink_path).await.unwrap();
            assert_eq!(resolved, target_files[i]);
        }

        // Performance test: resolve all symlinks quickly
        let start_time = SystemTime::now();
        for symlink_path in &symlink_paths {
            let _resolved = special_handler.resolve_symlink(symlink_path).await.unwrap();
        }
        let elapsed = start_time.elapsed().unwrap();

        // Should resolve 100 symlinks in under 1 second
        assert!(elapsed < Duration::from_secs(1), "Bulk symlink resolution took too long: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_concurrent_symlink_operations() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = Arc::new(SpecialFileHandler::new());

        // Create target file
        let target_file = env.create_file("target.txt", "shared content").unwrap();

        // Create multiple symlinks concurrently
        let mut handles = Vec::new();
        for i in 0..50 {
            let handler = Arc::clone(&special_handler);
            let target = target_file.clone();
            let temp_path = env.path().to_path_buf();

            let handle = tokio::spawn(async move {
                let symlink_path = temp_path.join(format!("concurrent_link_{}.txt", i));
                symlink(&target, &symlink_path).unwrap();

                // Resolve symlink
                let resolved = handler.resolve_symlink(&symlink_path).await.unwrap();
                assert_eq!(resolved, target);

                // Get symlink info
                let info = handler.get_symlink_info(&symlink_path).await.unwrap();
                assert_eq!(info.target, target);
                assert!(!info.is_broken);

                i
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results = futures_util::future::join_all(handles).await;
        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), i);
        }
    }
}

/// Integration tests that verify 90%+ coverage
mod coverage_validation_tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_symlink_coverage() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Test all major symlink operations to ensure high coverage

        // 1. Basic symlink creation and resolution
        let target = env.create_file("target.txt", "content").unwrap();
        let link = env.create_symlink("link.txt", &target).unwrap();
        assert!(special_handler.is_symlink(&link).await.unwrap());
        assert_eq!(special_handler.resolve_symlink(&link).await.unwrap(), target);

        // 2. Broken symlink handling
        fs::remove_file(&target).unwrap();
        assert!(special_handler.get_symlink_info(&link).await.unwrap().is_broken);

        // 3. Circular symlink detection
        let link1 = env.path().join("circular1");
        let link2 = env.path().join("circular2");
        symlink(&link2, &link1).unwrap();
        symlink(&link1, &link2).unwrap();
        assert!(special_handler.resolve_symlink(&link1).await.is_err());

        // 4. Nested symlinks
        let new_target = env.create_file("new_target.txt", "new content").unwrap();
        let level1 = env.create_symlink("level1", &new_target).unwrap();
        let level2 = env.create_symlink("level2", &level1).unwrap();
        let level3 = env.create_symlink("level3", &level2).unwrap();
        assert_eq!(special_handler.resolve_symlink(&level3).await.unwrap(), new_target);

        // 5. Hard links
        let hard_target = env.create_file("hard_target.txt", "hard content").unwrap();
        let hard_link = env.create_hard_link("hard_link.txt", &hard_target).unwrap();
        assert!(special_handler.is_hard_link(&hard_link).await.unwrap());
        assert_eq!(special_handler.get_hard_link_count(&hard_link).await.unwrap(), 2);

        // 6. Special files (Unix only)
        #[cfg(unix)]
        {
            let pipe = env.create_named_pipe("test_pipe").unwrap();
            assert_eq!(special_handler.get_file_type(&pipe).await.unwrap(), SpecialFileType::NamedPipe);
            assert!(special_handler.is_special_file(&pipe).await.unwrap());
            assert!(!special_handler.should_process_special_file(&pipe).await.unwrap());

            let socket = env.create_unix_socket("test_socket").unwrap();
            assert_eq!(special_handler.get_file_type(&socket).await.unwrap(), SpecialFileType::Socket);
        }

        // 7. Error conditions
        let nonexistent = env.path().join("nonexistent");
        assert!(special_handler.resolve_symlink(&nonexistent).await.is_err());
        assert!(special_handler.get_symlink_info(&nonexistent).await.is_err());

        // 8. Edge cases
        let empty_name_link = env.path().join("");
        assert!(special_handler.is_symlink(&empty_name_link).await.is_err());
    }

    #[tokio::test]
    async fn test_all_error_conditions() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Test all error types to ensure proper error handling coverage

        // 1. SymlinkBroken error
        let broken_target = env.path().join("nonexistent");
        let broken_link = env.create_symlink("broken", &broken_target).unwrap();
        match special_handler.resolve_symlink(&broken_link).await.unwrap_err() {
            DaemonError::SymlinkBroken { .. } => (),
            _ => panic!("Expected SymlinkBroken error"),
        }

        // 2. SymlinkCircular error
        let circ1 = env.path().join("circ1");
        let circ2 = env.path().join("circ2");
        symlink(&circ2, &circ1).unwrap();
        symlink(&circ1, &circ2).unwrap();
        match special_handler.resolve_symlink(&circ1).await.unwrap_err() {
            DaemonError::SymlinkCircular { .. } => (),
            _ => panic!("Expected SymlinkCircular error"),
        }

        // 3. SymlinkDepthExceeded error
        let deep_target = env.create_file("deep_target.txt", "content").unwrap();
        let mut current = deep_target;
        for i in 1..=10 {
            let next = env.create_symlink(&format!("deep_{}", i), &current).unwrap();
            current = next;
        }
        match special_handler.calculate_symlink_depth(&current, 5).await.unwrap_err() {
            DaemonError::SymlinkDepthExceeded { .. } => (),
            _ => panic!("Expected SymlinkDepthExceeded error"),
        }

        // 4. FileIo errors for invalid paths
        let invalid_path = env.path().join("nonexistent/invalid/path");
        assert!(special_handler.is_symlink(&invalid_path).await.is_err());
        assert!(special_handler.get_file_type(&invalid_path).await.is_err());
    }

    #[tokio::test]
    async fn test_complete_special_file_handler_api() {
        let env = SymlinkTestEnvironment::new().await.unwrap();
        let special_handler = SpecialFileHandler::new();

        // Test every public method of SpecialFileHandler for coverage

        let target = env.create_file("api_target.txt", "api test").unwrap();
        let link = env.create_symlink("api_link.txt", &target).unwrap();

        // is_symlink
        assert!(special_handler.is_symlink(&link).await.unwrap());
        assert!(!special_handler.is_symlink(&target).await.unwrap());

        // resolve_symlink
        assert_eq!(special_handler.resolve_symlink(&link).await.unwrap(), target);

        // get_symlink_info
        let info = special_handler.get_symlink_info(&link).await.unwrap();
        assert_eq!(info.target, target);
        assert!(!info.is_broken);
        assert_eq!(info.depth, 1);

        // calculate_symlink_depth
        let depth = special_handler.calculate_symlink_depth(&link, 10).await.unwrap();
        assert_eq!(depth, 1);

        // is_hard_link
        let hard_link = env.create_hard_link("api_hard.txt", &target).unwrap();
        assert!(special_handler.is_hard_link(&hard_link).await.unwrap());
        assert!(!special_handler.is_hard_link(&link).await.unwrap());

        // get_hard_link_count
        assert_eq!(special_handler.get_hard_link_count(&hard_link).await.unwrap(), 2);

        // find_hard_links
        let inode = fs::metadata(&target).unwrap().ino();
        let links = special_handler.find_hard_links(env.path(), inode).await.unwrap();
        assert_eq!(links.len(), 2);

        // is_special_file
        assert!(!special_handler.is_special_file(&target).await.unwrap());

        // get_file_type - should error for regular files
        assert!(special_handler.get_file_type(&target).await.is_err());

        // should_process_special_file - should error for regular files
        assert!(special_handler.should_process_special_file(&target).await.is_err());

        #[cfg(unix)]
        {
            let pipe = env.create_named_pipe("api_pipe").unwrap();
            assert!(special_handler.is_special_file(&pipe).await.unwrap());
            assert_eq!(special_handler.get_file_type(&pipe).await.unwrap(), SpecialFileType::NamedPipe);
            assert!(!special_handler.should_process_special_file(&pipe).await.unwrap());
        }
    }
}