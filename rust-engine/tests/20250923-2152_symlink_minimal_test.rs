//! Minimal symlink and special file handling test
//!
//! This test verifies basic symlink functionality is working.

use workspace_qdrant_daemon::daemon::file_ops::{SpecialFileHandler, SpecialFileType};
use workspace_qdrant_daemon::error::DaemonError;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use tempfile::tempdir;

#[tokio::test]
async fn test_basic_symlink_creation() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create target file
    let target_file = temp_dir.path().join("target.txt");
    fs::write(&target_file, "target content").unwrap();

    // Create symlink
    let symlink_path = temp_dir.path().join("link.txt");
    symlink(&target_file, &symlink_path).unwrap();

    // Verify symlink is detected
    let is_symlink = special_handler.is_symlink(&symlink_path).await.unwrap();
    assert!(is_symlink);

    // Verify symlink target resolution
    let resolved_target = special_handler.resolve_symlink(&symlink_path).await.unwrap();
    assert_eq!(resolved_target, target_file);
}

#[tokio::test]
async fn test_broken_symlink() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create symlink to non-existent file
    let nonexistent_target = temp_dir.path().join("nonexistent.txt");
    let broken_link = temp_dir.path().join("broken_link.txt");
    symlink(&nonexistent_target, &broken_link).unwrap();

    // Verify broken symlink is detected
    let symlink_info = special_handler.get_symlink_info(&broken_link).await.unwrap();
    assert!(symlink_info.is_broken);

    // Verify resolve_symlink returns appropriate error for broken links
    let resolve_result = special_handler.resolve_symlink(&broken_link).await;
    assert!(resolve_result.is_err());
    match resolve_result.unwrap_err() {
        DaemonError::SymlinkBroken { .. } => (),
        _ => panic!("Expected SymlinkBroken error"),
    }
}

#[tokio::test]
async fn test_symlink_depth_calculation() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create chain: file -> link1 -> link2 -> link3
    let target_file = temp_dir.path().join("target.txt");
    fs::write(&target_file, "content").unwrap();

    let link1 = temp_dir.path().join("link1.txt");
    let link2 = temp_dir.path().join("link2.txt");
    let link3 = temp_dir.path().join("link3.txt");

    symlink(&target_file, &link1).unwrap();
    symlink(&link1, &link2).unwrap();
    symlink(&link2, &link3).unwrap();

    // Verify deep resolution works
    let final_target = special_handler.resolve_symlink(&link3).await.unwrap();
    assert_eq!(final_target, target_file);

    // Verify depth calculation
    let depth = special_handler.calculate_symlink_depth(&link3, 10).await.unwrap();
    assert_eq!(depth, 3);
}

#[tokio::test]
async fn test_hard_link_detection() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create original file
    let original_file = temp_dir.path().join("original.txt");
    fs::write(&original_file, "shared content").unwrap();

    // Create hard link
    let hard_link = temp_dir.path().join("hardlink.txt");
    fs::hard_link(&original_file, &hard_link).unwrap();

    // Verify hard link detection
    let is_hardlink = special_handler.is_hard_link(&hard_link).await.unwrap();
    assert!(is_hardlink);

    // Verify hard link count
    let link_count = special_handler.get_hard_link_count(&hard_link).await.unwrap();
    assert_eq!(link_count, 2);
}

#[cfg(unix)]
#[tokio::test]
async fn test_named_pipe_detection() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create named pipe (FIFO)
    let pipe_path = temp_dir.path().join("test_pipe");
    let pipe_path_str = pipe_path.to_str().unwrap();
    let c_path = std::ffi::CString::new(pipe_path_str).unwrap();

    unsafe {
        if libc::mkfifo(c_path.as_ptr(), 0o644) == 0 {
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
    }
}

#[tokio::test]
async fn test_circular_symlink_detection() {
    let temp_dir = tempdir().unwrap();
    let special_handler = SpecialFileHandler::new();

    // Create circular symlinks
    let link1 = temp_dir.path().join("link1.txt");
    let link2 = temp_dir.path().join("link2.txt");

    symlink(&link2, &link1).unwrap();
    symlink(&link1, &link2).unwrap();

    // Verify circular reference is detected
    let resolve_result = special_handler.resolve_symlink(&link1).await;
    assert!(resolve_result.is_err());
    match resolve_result.unwrap_err() {
        DaemonError::SymlinkCircular { .. } => (),
        _ => panic!("Expected SymlinkCircular error"),
    }
}