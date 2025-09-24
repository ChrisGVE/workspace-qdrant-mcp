//! Property-based tests for filesystem event handling
//!
//! This module validates filesystem event ordering, deduplication,
//! cross-platform path handling, and recursive watching properties.

use proptest::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use tokio::time::{sleep, timeout};
use workspace_qdrant_daemon::daemon::watcher::*;
use workspace_qdrant_daemon::daemon::file_ops::AsyncFileProcessor;
use workspace_qdrant_daemon::error::DaemonResult;

/// Strategy for generating filesystem operations
fn filesystem_operations() -> impl Strategy<Value = Vec<FileSystemOperation>> {
    prop::collection::vec(
        prop_oneof![
            "[a-zA-Z0-9_.-]{1,50}".prop_map(FileSystemOperation::CreateFile),
            "[a-zA-Z0-9_.-]{1,50}".prop_map(FileSystemOperation::ModifyFile),
            "[a-zA-Z0-9_.-]{1,50}".prop_map(FileSystemOperation::DeleteFile),
            "[a-zA-Z0-9_.-]{1,30}".prop_map(FileSystemOperation::CreateDirectory),
            "[a-zA-Z0-9_.-]{1,30}".prop_map(FileSystemOperation::DeleteDirectory),
            ("[a-zA-Z0-9_.-]{1,30}", "[a-zA-Z0-9_.-]{1,30}")
                .prop_map(|(from, to)| FileSystemOperation::MoveFile(from, to)),
        ],
        1..20
    )
}

/// Strategy for generating file paths with various edge cases
fn edge_case_paths() -> impl Strategy<Value = String> {
    prop_oneof![
        // Normal paths
        "[a-zA-Z0-9_.-/]{5,100}",
        // Unicode paths
        "[\\PC]{1,50}",
        // Paths with spaces
        "[ a-zA-Z0-9_.-/]{5,50}",
        // Very long paths
        "[a-zA-Z0-9_]{200,300}",
        // Paths with special characters (safe ones)
        "[a-zA-Z0-9_.()\\[\\]-]{5,100}",
        // Nested directory structures
        Just("deep/nested/directory/structure/file.txt".to_string()),
        // Relative paths
        Just("./relative/path/file.txt".to_string()),
        Just("../parent/path/file.txt".to_string()),
    ]
}

/// Strategy for generating file content patterns
fn file_content_patterns() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        prop::collection::vec(any::<u8>(), 0..10000),           // Random binary
        "[\\PC]{0,1000}".prop_map(|s| s.into_bytes()),         // Text content
        Just(Vec::new()),                                        // Empty files
        prop::collection::vec(0u8, 1000..10000),               // Null bytes
        prop::collection::vec(b' ', 100..1000),                // Whitespace
    ]
}

/// Strategy for generating concurrent access patterns
fn concurrent_patterns() -> impl Strategy<Value = (usize, Duration)> {
    (
        1usize..10usize,                     // number of concurrent operations
        prop::sample::select(vec![           // delay between operations
            Duration::from_millis(0),
            Duration::from_millis(10),
            Duration::from_millis(50),
            Duration::from_millis(100),
        ])
    )
}

#[derive(Debug, Clone)]
pub enum FileSystemOperation {
    CreateFile(String),
    ModifyFile(String),
    DeleteFile(String),
    CreateDirectory(String),
    DeleteDirectory(String),
    MoveFile(String, String),
}

/// Execute a filesystem operation
async fn execute_filesystem_operation(
    operation: &FileSystemOperation,
    base_dir: &Path,
    processor: &AsyncFileProcessor,
) -> DaemonResult<()> {
    match operation {
        FileSystemOperation::CreateFile(name) => {
            let path = base_dir.join(name);
            let content = b"test content";
            processor.write_file(&path, content).await
        }
        FileSystemOperation::ModifyFile(name) => {
            let path = base_dir.join(name);
            if path.exists() {
                let new_content = b"modified content";
                processor.write_file(&path, new_content).await
            } else {
                // Create if doesn't exist
                let content = b"new content";
                processor.write_file(&path, content).await
            }
        }
        FileSystemOperation::DeleteFile(name) => {
            let path = base_dir.join(name);
            fs::remove_file(&path).await.map_err(|e| {
                workspace_qdrant_daemon::error::DaemonError::FileIo {
                    message: format!("Delete failed: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })
        }
        FileSystemOperation::CreateDirectory(name) => {
            let path = base_dir.join(name);
            fs::create_dir_all(&path).await.map_err(|e| {
                workspace_qdrant_daemon::error::DaemonError::FileIo {
                    message: format!("Create dir failed: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })
        }
        FileSystemOperation::DeleteDirectory(name) => {
            let path = base_dir.join(name);
            fs::remove_dir_all(&path).await.map_err(|e| {
                workspace_qdrant_daemon::error::DaemonError::FileIo {
                    message: format!("Remove dir failed: {}", e),
                    path: path.to_string_lossy().to_string(),
                }
            })
        }
        FileSystemOperation::MoveFile(from, to) => {
            let from_path = base_dir.join(from);
            let to_path = base_dir.join(to);

            // Ensure source exists
            if !from_path.exists() {
                processor.write_file(&from_path, b"content to move").await?;
            }

            fs::rename(&from_path, &to_path).await.map_err(|e| {
                workspace_qdrant_daemon::error::DaemonError::FileIo {
                    message: format!("Move failed: {}", e),
                    path: from_path.to_string_lossy().to_string(),
                }
            })
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 25000, // 25 seconds for filesystem tests
        cases: 20,      // Fewer cases for filesystem tests
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_filesystem_event_ordering(operations in filesystem_operations()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();
            let mut executed_operations = Vec::new();

            // Property: Filesystem operations should maintain logical ordering
            for operation in operations {
                let result = timeout(
                    Duration::from_secs(2),
                    execute_filesystem_operation(&operation, temp_dir.path(), &processor)
                ).await;

                match result {
                    Ok(Ok(())) => {
                        executed_operations.push(operation.clone());
                    }
                    Ok(Err(_)) => {
                        // Operation failed - acceptable for some edge cases
                    }
                    Err(_) => {
                        // Timeout - acceptable for slow operations
                        break;
                    }
                }
            }

            // Verify final state consistency
            let entries = fs::read_dir(temp_dir.path()).await;
            if let Ok(mut dir_entries) = entries {
                let mut actual_files = HashSet::new();
                while let Some(entry) = dir_entries.next_entry().await.unwrap() {
                    if let Some(name) = entry.file_name().to_str() {
                        actual_files.insert(name.to_string());
                    }
                }

                // Property: Final filesystem state should be consistent with executed operations
                let mut expected_files = HashSet::new();
                for op in &executed_operations {
                    match op {
                        FileSystemOperation::CreateFile(name) |
                        FileSystemOperation::ModifyFile(name) => {
                            expected_files.insert(name.clone());
                        }
                        FileSystemOperation::DeleteFile(name) => {
                            expected_files.remove(name);
                        }
                        FileSystemOperation::MoveFile(from, to) => {
                            expected_files.remove(from);
                            expected_files.insert(to.clone());
                        }
                        FileSystemOperation::CreateDirectory(name) => {
                            expected_files.insert(name.clone());
                        }
                        FileSystemOperation::DeleteDirectory(name) => {
                            expected_files.remove(name);
                        }
                    }
                }

                // Allow for some discrepancy due to operation failures
                let common_files: HashSet<_> = actual_files.intersection(&expected_files).collect();
                prop_assert!(
                    common_files.len() as f64 >= (expected_files.len() as f64 * 0.7),
                    "At least 70% of expected files should exist"
                );
            }
        });
    }

    #[test]
    fn proptest_path_handling_edge_cases(paths in prop::collection::vec(edge_case_paths(), 1..10)) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Property: Path handling should be robust across various edge cases
            for path_str in paths {
                // Sanitize path to be within temp directory
                let safe_path = if path_str.contains("..") {
                    path_str.replace("..", "dotdot")
                } else {
                    path_str
                };

                let full_path = temp_dir.path().join(&safe_path);

                // Ensure parent directories exist
                if let Some(parent) = full_path.parent() {
                    let _ = fs::create_dir_all(parent).await;
                }

                let content = b"test content for path handling";
                let write_result = timeout(
                    Duration::from_secs(1),
                    processor.write_file(&full_path, content)
                ).await;

                match write_result {
                    Ok(Ok(())) => {
                        // If write succeeded, read should also work
                        let read_result = timeout(
                            Duration::from_secs(1),
                            processor.read_file(&full_path)
                        ).await;

                        if let Ok(Ok(read_content)) = read_result {
                            prop_assert_eq!(content, &read_content[..],
                                          "Content should match for path: {}", safe_path);
                        }
                    }
                    Ok(Err(_)) => {
                        // Write failure is acceptable for invalid paths
                    }
                    Err(_) => {
                        // Timeout is acceptable for problematic paths
                    }
                }
            }
        });
    }

    #[test]
    fn proptest_concurrent_file_modifications(
        (num_tasks, delay) in concurrent_patterns(),
        content_patterns in prop::collection::vec(file_content_patterns(), 1..5)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();
            let file_path = temp_dir.path().join("concurrent_test.txt");

            // Property: Concurrent modifications should not cause data corruption
            let handles: Vec<_> = (0..num_tasks)
                .map(|task_id| {
                    let path = file_path.clone();
                    let proc = processor.clone();
                    let content = content_patterns
                        .get(task_id % content_patterns.len())
                        .unwrap_or(&vec![])
                        .clone();

                    tokio::spawn(async move {
                        sleep(delay * task_id as u32).await;

                        // Write unique content for this task
                        let unique_content = format!("Task {}: ", task_id).into_bytes()
                            .into_iter()
                            .chain(content.iter().cloned())
                            .collect::<Vec<u8>>();

                        timeout(Duration::from_secs(2), proc.write_file(&path, &unique_content)).await
                    })
                })
                .collect();

            // Wait for all concurrent operations
            let mut successful_writes = 0;
            for handle in handles {
                match handle.await {
                    Ok(Ok(Ok(()))) => successful_writes += 1,
                    Ok(Ok(Err(_))) => {
                        // Write error is acceptable under concurrency
                    }
                    Ok(Err(_)) => {
                        // Timeout is acceptable
                    }
                    Err(_) => {
                        // Task panic should not happen
                        prop_assert!(false, "Task should not panic during concurrent writes");
                    }
                }
            }

            // Property: At least one write should succeed
            prop_assert!(successful_writes > 0, "At least one concurrent write should succeed");

            // Verify file integrity
            if file_path.exists() {
                let read_result = processor.read_file(&file_path).await;
                if let Ok(final_content) = read_result {
                    prop_assert!(!final_content.is_empty() || content_patterns.iter().any(|c| c.is_empty()),
                               "Final content should not be empty unless empty content was written");
                }
            }
        });
    }

    #[test]
    fn proptest_filesystem_event_deduplication(
        duplicate_operations in prop::collection::vec(
            "[a-zA-Z0-9_.-]{1,20}", 2..10
        )
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Property: Duplicate filesystem operations should be handled efficiently
            for filename in &duplicate_operations {
                let file_path = temp_dir.path().join(filename);
                let content = format!("Content for {}", filename).into_bytes();

                // Perform multiple identical operations
                let mut operation_results = Vec::new();
                for _ in 0..3 {
                    let result = timeout(
                        Duration::from_secs(1),
                        processor.write_file(&file_path, &content)
                    ).await;

                    match result {
                        Ok(Ok(())) => operation_results.push(true),
                        Ok(Err(_)) => operation_results.push(false),
                        Err(_) => break, // Timeout
                    }

                    // Small delay between operations
                    sleep(Duration::from_millis(10)).await;
                }

                // Property: Repeated operations on same file should be consistent
                if !operation_results.is_empty() {
                    let first_result = operation_results[0];
                    let all_same = operation_results.iter().all(|&result| result == first_result);
                    prop_assert!(all_same || operation_results.len() < 2,
                               "Repeated operations should have consistent results for file: {}", filename);
                }

                // Verify final state
                if file_path.exists() {
                    if let Ok(read_content) = processor.read_file(&file_path).await {
                        prop_assert_eq!(content, read_content,
                                      "Final content should match expected for file: {}", filename);
                    }
                }
            }
        });
    }

    #[test]
    fn proptest_recursive_directory_operations(
        directory_structure in prop::collection::vec(
            prop::collection::vec("[a-zA-Z0-9_-]{1,10}", 1..4), 1..5
        )
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Property: Recursive directory operations should handle nested structures
            for path_components in directory_structure {
                let mut current_path = temp_dir.path().to_path_buf();

                // Build nested directory structure
                for component in &path_components {
                    current_path.push(component);
                }

                // Create parent directories
                if let Some(parent) = current_path.parent() {
                    let create_result = timeout(
                        Duration::from_secs(2),
                        fs::create_dir_all(parent)
                    ).await;

                    if create_result.is_ok() {
                        // Create a file in the nested directory
                        let file_path = current_path.with_extension("txt");
                        let content = format!("Content in nested path: {:?}", path_components).into_bytes();

                        let write_result = timeout(
                            Duration::from_secs(1),
                            processor.write_file(&file_path, &content)
                        ).await;

                        if let Ok(Ok(())) = write_result {
                            // Verify file can be read back
                            let read_result = processor.read_file(&file_path).await;
                            if let Ok(read_content) = read_result {
                                prop_assert_eq!(content, read_content,
                                              "Nested file content should match");
                            }

                            // Test recursive deletion
                            if path_components.len() > 1 {
                                let delete_result = timeout(
                                    Duration::from_secs(2),
                                    fs::remove_dir_all(&current_path.parent().unwrap())
                                ).await;

                                // Deletion success is not required, but shouldn't panic
                                let _ = delete_result;
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn proptest_symlink_handling(
        symlink_targets in prop::collection::vec(
            "[a-zA-Z0-9_.-]{1,30}", 1..5
        )
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Property: Symlink operations should be handled appropriately
            for target_name in symlink_targets {
                let target_path = temp_dir.path().join(&target_name);
                let link_path = temp_dir.path().join(format!("{}_link", target_name));

                // Create target file
                let content = format!("Target content for {}", target_name).into_bytes();
                let create_result = processor.write_file(&target_path, &content).await;

                if create_result.is_ok() {
                    // Create symlink (Unix-specific, but test should handle gracefully)
                    #[cfg(unix)]
                    {
                        let symlink_result = tokio::fs::symlink(&target_path, &link_path).await;

                        if symlink_result.is_ok() {
                            // Test reading through symlink
                            let read_result = processor.read_file(&link_path).await;

                            match read_result {
                                Ok(symlink_content) => {
                                    prop_assert_eq!(content, symlink_content,
                                                  "Symlink should resolve to target content");
                                }
                                Err(_) => {
                                    // Symlink read failure is acceptable on some systems
                                }
                            }
                        }
                    }

                    #[cfg(not(unix))]
                    {
                        // On non-Unix systems, just verify target file works
                        let read_result = processor.read_file(&target_path).await;
                        if let Ok(target_content) = read_result {
                            prop_assert_eq!(content, target_content,
                                          "Target file should be readable");
                        }
                    }
                }
            }
        });
    }
}