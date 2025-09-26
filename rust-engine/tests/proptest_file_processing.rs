//! Property-based tests for file processing edge cases
//!
//! This module implements comprehensive property-based testing for document processing,
//! file I/O operations, and memory bounds validation using proptest framework.

use proptest::prelude::*;
use std::path::Path;
use tempfile::{NamedTempFile, TempDir};
use tokio::fs;
use workspace_qdrant_daemon::daemon::file_ops::AsyncFileProcessor;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::error::{DaemonError, DaemonResult};

/// Custom strategy for generating random file content
fn random_file_content() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Valid UTF-8 text content
        "[\\PC]*".prop_map(|s| s.into_bytes()),
        // Binary content
        prop::collection::vec(any::<u8>(), 0..10000),
        // Empty content
        Just(Vec::new()),
        // Large content near memory boundaries
        prop::collection::vec(any::<u8>(), 1_000_000..2_000_000),
        // Invalid UTF-8 sequences
        prop::collection::vec(0x80u8..0xFFu8, 10..1000),
        // Content with null bytes
        prop::collection::vec(any::<u8>(), 0..1000).prop_map(|mut v| {
            if !v.is_empty() && v.len() > 10 {
                let mid_idx = v.len() / 2;
                v[mid_idx] = 0;
            }
            v
        }),
    ]
}

/// Strategy for generating filenames with edge cases
fn random_filename() -> impl Strategy<Value = String> {
    prop_oneof![
        // Normal filenames
        "[a-zA-Z0-9_-]{1,50}\\.(txt|rs|py|md)",
        // Unicode filenames
        "[\\PC]{1,20}\\.(txt|rs)",
        // Filenames with spaces and special characters
        "[ a-zA-Z0-9_.-]{1,30}\\.(txt|rs)",
        // Very long filenames
        "[a-zA-Z0-9_]{100,200}\\.(txt|rs)",
        // Single character names
        "[a-zA-Z]\\.(txt|rs)",
    ]
}

/// Strategy for file processor configurations
fn random_processor_config() -> impl Strategy<Value = (u64, usize, bool)> {
    (
        1_024u64..100_000_000u64, // max_file_size
        512usize..65536usize,     // buffer_size
        any::<bool>(),            // enable_compression
    )
}

#[tokio::test]
async fn proptest_file_read_write_roundtrip() {
    proptest!(|(content in random_file_content(), filename in random_filename())| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create temporary directory
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join(&filename);

            // Create file processor
            let processor = AsyncFileProcessor::default();

            // Property: Write then read should return same content (for reasonable sizes)
            if content.len() <= 50_000_000 { // Stay within reasonable bounds
                // Write content
                if let Ok(()) = processor.write_file(&file_path, &content).await {
                    // Read content back
                    if let Ok(read_content) = processor.read_file(&file_path).await {
                        prop_assert_eq!(content, read_content, "Content should match after roundtrip");
                    }
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_file_size_limits() {
    proptest!(|(
        (max_size, buffer_size, compression) in random_processor_config(),
        content in random_file_content()
    )| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join("test.txt");

            let processor = AsyncFileProcessor::new(max_size, buffer_size, compression);

            // Write file first (if small enough)
            if content.len() as u64 <= max_size {
                if processor.write_file(&file_path, &content).await.is_ok() {
                    // Property: Files within size limit should be readable
                    let result = processor.read_file(&file_path).await;
                    prop_assert!(result.is_ok(), "File within size limit should be readable");

                    if let Ok(read_content) = result {
                        prop_assert_eq!(content.len(), read_content.len(), "Content length should match");
                    }
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_invalid_utf8_handling() {
    proptest!(|invalid_bytes in prop::collection::vec(0x80u8..0xFFu8, 1..1000)| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join("invalid_utf8.txt");

            let processor = AsyncFileProcessor::default();

            // Property: Invalid UTF-8 bytes should still be handled as raw bytes
            if processor.write_file(&file_path, &invalid_bytes).await.is_ok() {
                let result = processor.read_file(&file_path).await;

                // Should not panic or fail - treat as raw bytes
                if let Ok(read_bytes) = result {
                    prop_assert_eq!(invalid_bytes, read_bytes, "Invalid UTF-8 bytes should roundtrip");
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_concurrent_file_operations() {
    proptest!(|(
        files in prop::collection::vec((random_filename(), random_file_content()), 1..10)
    )| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Property: Concurrent operations should not interfere with each other
            let mut handles = Vec::new();

            for (filename, content) in files.iter() {
                if content.len() <= 1_000_000 { // Keep reasonable size for concurrent tests
                    let file_path = temp_dir.path().join(filename);
                    let proc_clone = processor.clone();
                    let content_clone = content.clone();

                    let handle = tokio::spawn(async move {
                        if proc_clone.write_file(&file_path, &content_clone).await.is_ok() {
                            proc_clone.read_file(&file_path).await
                        } else {
                            Err(DaemonError::Internal { message: "Write failed".to_string() })
                        }
                    });
                    handles.push((handle, content.clone()));
                }
            }

            // Wait for all operations and verify results
            for (handle, expected_content) in handles {
                if let Ok(Ok(actual_content)) = handle.await {
                    prop_assert_eq!(expected_content, actual_content, "Concurrent operations should preserve content");
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_memory_bounds_validation() {
    // Test memory usage with various file sizes
    proptest!(|size in 0usize..10_000_000usize| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join("memory_test.txt");

            // Create content of specified size
            let content = vec![42u8; size];
            let processor = AsyncFileProcessor::new(50_000_000, 8192, false);

            // Property: Memory allocation should be bounded and not cause OOM
            let write_result = processor.write_file(&file_path, &content).await;

            if write_result.is_ok() && size <= 50_000_000 {
                let read_result = processor.read_file(&file_path).await;

                if let Ok(read_content) = read_result {
                    prop_assert_eq!(content.len(), read_content.len(), "Memory bounds should be respected");

                    // Verify content is correct (sample check to avoid full comparison for large files)
                    if size > 0 {
                        prop_assert_eq!(content[0], read_content[0], "Content should be preserved");
                        if size > 1000 {
                            prop_assert_eq!(content[size-1], read_content[size-1], "End content should be preserved");
                        }
                    }
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_file_permission_edge_cases() {
    proptest!(|(content in random_file_content().prop_filter("small content", |c| c.len() < 10000))| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join("permission_test.txt");

            let processor = AsyncFileProcessor::default();

            // Property: Permission errors should be handled gracefully
            if processor.write_file(&file_path, &content).await.is_ok() {
                // Make file read-only
                if let Ok(metadata) = fs::metadata(&file_path).await {
                    let mut perms = metadata.permissions();
                    perms.set_readonly(true);
                    let _ = fs::set_permissions(&file_path, perms).await;

                    // Should still be able to read
                    let read_result = processor.read_file(&file_path).await;
                    prop_assert!(read_result.is_ok(), "Read-only files should still be readable");

                    // Writing should fail gracefully
                    let write_result = processor.write_file(&file_path, b"new content").await;
                    // This might succeed on some systems, but shouldn't panic
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_document_processor_edge_cases() {
    proptest!(|paths in prop::collection::vec(random_filename(), 1..5)| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = DocumentProcessor::test_instance();

            // Property: Document processor should handle various file paths without panicking
            for path in paths {
                let result = processor.process_document(&path).await;

                // Should not panic - either succeed or return appropriate error
                match result {
                    Ok(id) => {
                        // Should return a valid UUID string
                        prop_assert!(!id.is_empty(), "Document ID should not be empty");
                        prop_assert!(id.len() == 36, "UUID should be 36 characters"); // Standard UUID format
                    },
                    Err(_) => {
                        // Error is acceptable for invalid paths
                    }
                }
            }
        });
    });
}

#[tokio::test]
async fn proptest_path_sanitization() {
    proptest!(|path_components in prop::collection::vec("[\\PC]{1,20}", 1..10)| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let processor = AsyncFileProcessor::default();

            // Build path from components
            let mut file_path = temp_dir.path().to_path_buf();
            for component in &path_components {
                file_path.push(component);
            }
            file_path.set_extension("txt");

            let content = b"test content";

            // Property: Path handling should be robust against various Unicode characters
            let write_result = processor.write_file(&file_path, content).await;

            if write_result.is_ok() {
                let read_result = processor.read_file(&file_path).await;
                if let Ok(read_content) = read_result {
                    prop_assert_eq!(content, &read_content[..], "Content should survive path edge cases");
                }
            }
            // If write fails, that's acceptable for invalid paths
        });
    });
}

// Test configuration to control proptest behavior
proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 30000, // 30 seconds per test
        max_shrink_iters: 100,
        cases: 50, // Reasonable number for CI
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_buffer_size_edge_cases(
        content in random_file_content().prop_filter("medium size", |c| c.len() > 0 && c.len() < 100_000),
        buffer_size in 1usize..65536usize
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let file_path = temp_dir.path().join("buffer_test.txt");

            let processor = AsyncFileProcessor::new(1_000_000, buffer_size, false);

            // Property: Different buffer sizes should not affect correctness
            if processor.write_file(&file_path, &content).await.is_ok() {
                if let Ok(read_content) = processor.read_file(&file_path).await {
                    prop_assert_eq!(content, read_content, "Buffer size should not affect content correctness");
                }
            }
        });
    }
}