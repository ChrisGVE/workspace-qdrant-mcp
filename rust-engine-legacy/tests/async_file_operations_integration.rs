//! Integration tests for async file operations
//!
//! This test suite validates the async file operations in a more realistic scenario
//! where they're integrated with the document processing pipeline.

use workspace_qdrant_daemon::daemon::file_ops::{AsyncFileProcessor, AsyncFileStream};
use workspace_qdrant_daemon::error::DaemonError;
use tempfile::tempdir;
use std::fs;

/// Test basic async file operations integration
#[tokio::test]
async fn test_async_file_processor_integration() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("integration_test.txt");
    let content = b"Hello from async file processor integration test!";

    let processor = AsyncFileProcessor::default();

    // Write file
    processor.write_file(&file_path, content).await.unwrap();
    assert!(file_path.exists());

    // Read back and verify
    let read_content = processor.read_file(&file_path).await.unwrap();
    assert_eq!(read_content, content);

    // Calculate hash for integrity
    let hash = processor.calculate_hash(&file_path).await.unwrap();
    assert_eq!(hash.len(), 64); // BLAKE3 hash

    // Validate file info
    let file_info = processor.validate_file(&file_path).await.unwrap();
    assert_eq!(file_info.size, content.len() as u64);
    assert!(file_info.is_file);
}

/// Test async file operations with error handling
#[tokio::test]
async fn test_async_file_operations_error_scenarios() {
    let temp_dir = tempdir().unwrap();
    let processor = AsyncFileProcessor::default();

    // Test reading non-existent file
    let nonexistent_path = temp_dir.path().join("does_not_exist.txt");
    let result = processor.read_file(&nonexistent_path).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        DaemonError::FileIo { message, path } => {
            assert!(message.contains("Cannot access file metadata"));
            assert_eq!(path, nonexistent_path.to_string_lossy());
        }
        _ => panic!("Expected FileIo error"),
    }

    // Test file too large error
    let large_processor = AsyncFileProcessor::new(50, 1024, true);
    let large_file_path = temp_dir.path().join("large_file.txt");
    let large_content = vec![0u8; 100]; // 100 bytes > 50 byte limit

    fs::write(&large_file_path, &large_content).unwrap();

    let result = large_processor.read_file(&large_file_path).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        DaemonError::FileTooLarge { path, size, max_size } => {
            assert_eq!(path, large_file_path.to_string_lossy());
            assert_eq!(size, 100);
            assert_eq!(max_size, 50);
        }
        _ => panic!("Expected FileTooLarge error"),
    }
}

/// Test async file stream processing
#[tokio::test]
async fn test_async_file_stream_integration() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("stream_test.txt");
    let content = b"0123456789abcdefghijklmnopqrstuvwxyz"; // 36 bytes

    fs::write(&file_path, content).unwrap();

    let processor = AsyncFileProcessor::default();
    let stream_processor = AsyncFileStream::new(processor);

    let mut stream = stream_processor.process_stream(&file_path, 10).await.unwrap();

    let mut chunks = Vec::new();
    use futures_util::stream::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        chunks.push(chunk_result.unwrap());
    }

    // Verify we got the expected number of chunks
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].len(), 10);
    assert_eq!(chunks[1].len(), 10);
    assert_eq!(chunks[2].len(), 10);
    assert_eq!(chunks[3].len(), 6); // Last chunk with remainder

    // Verify content integrity
    let reconstructed: Vec<u8> = chunks.into_iter().flatten().collect();
    assert_eq!(reconstructed, content);
}

/// Test concurrent async file operations
#[tokio::test]
async fn test_concurrent_async_file_operations() {
    let temp_dir = tempdir().unwrap();
    let processor = AsyncFileProcessor::default();

    let mut handles = Vec::new();

    // Start 10 concurrent file operations
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("concurrent_test_{}.txt", i));
        let content = format!("Concurrent test content {}", i).into_bytes();
        let processor_clone = processor.clone();

        let handle = tokio::spawn(async move {
            // Write file
            processor_clone.write_file(&file_path, &content).await.unwrap();

            // Read back and verify
            let read_content = processor_clone.read_file(&file_path).await.unwrap();
            assert_eq!(read_content, content);

            // Calculate hash
            let hash = processor_clone.calculate_hash(&file_path).await.unwrap();
            assert_eq!(hash.len(), 64);

            i
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let results = futures_util::future::join_all(handles).await;

    // Verify all operations completed successfully
    for (i, result) in results.into_iter().enumerate() {
        assert_eq!(result.unwrap(), i);
    }
}

/// Test async file operations with atomic replacements
#[tokio::test]
async fn test_atomic_file_operations() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("atomic_test.txt");
    let original_content = b"Original content for atomic test";
    let new_content = b"New content after atomic replacement";

    let processor = AsyncFileProcessor::default();

    // Create original file
    processor.write_file(&file_path, original_content).await.unwrap();
    let original_read = processor.read_file(&file_path).await.unwrap();
    assert_eq!(original_read, original_content);

    // Perform atomic replacement
    processor.atomic_replace(&file_path, new_content).await.unwrap();

    // Verify replacement was successful
    let new_read = processor.read_file(&file_path).await.unwrap();
    assert_eq!(new_read, new_content);

    // Verify file info is updated
    let file_info = processor.validate_file(&file_path).await.unwrap();
    assert_eq!(file_info.size, new_content.len() as u64);
}

/// Test chunked processing with realistic document scenarios
#[tokio::test]
async fn test_chunked_document_processing() {
    let temp_dir = tempdir().unwrap();
    let document_path = temp_dir.path().join("document.txt");

    // Create a document with multiple paragraphs
    let document_content = r#"
Chapter 1: Introduction to Async Programming

Asynchronous programming is a programming paradigm that allows a program to perform
multiple operations simultaneously without blocking the execution thread.

Chapter 2: Tokio Runtime

Tokio is an asynchronous runtime for the Rust programming language. It provides
the building blocks needed for writing network applications.

Chapter 3: File Operations

When dealing with file I/O in async contexts, it's important to use proper
async file operations to avoid blocking the runtime.
    "#.trim().as_bytes();

    fs::write(&document_path, document_content).unwrap();

    let processor = AsyncFileProcessor::default();

    // Process document in chunks
    let chunks = processor.process_chunks(&document_path, 100, |chunk, index| {
        let _chunk_str = String::from_utf8_lossy(chunk);
        Ok(format!("Chunk {}: {} chars", index, chunk.len()))
    }).await.unwrap();

    // Verify we got multiple chunks
    assert!(chunks.len() > 5);

    // Verify each chunk is processed correctly
    for (i, chunk_info) in chunks.iter().enumerate() {
        assert!(chunk_info.starts_with(&format!("Chunk {}: ", i)));
        assert!(chunk_info.contains("chars"));
    }
}

/// Test memory-efficient processing of large files
#[tokio::test]
async fn test_memory_efficient_large_file_processing() {
    let temp_dir = tempdir().unwrap();
    let large_file_path = temp_dir.path().join("large_file.txt");

    // Create a larger file (10KB)
    let chunk_data = "A".repeat(1024); // 1KB chunk
    let mut large_content = String::new();
    for _ in 0..10 {
        large_content.push_str(&chunk_data);
    }

    fs::write(&large_file_path, large_content.as_bytes()).unwrap();

    let processor = AsyncFileProcessor::new(50 * 1024, 2048, true); // 50KB limit, 2KB buffer

    // Test reading large file
    let read_content = processor.read_file(&large_file_path).await.unwrap();
    assert_eq!(read_content.len(), large_content.len());

    // Test chunked processing for memory efficiency
    let chunk_sizes = processor.process_chunks(&large_file_path, 1024, |chunk, _index| {
        Ok(chunk.len())
    }).await.unwrap();

    let total_size: usize = chunk_sizes.iter().sum();
    assert_eq!(total_size, large_content.len());
}

/// Test file operations with various unicode filenames
#[tokio::test]
async fn test_unicode_filename_support() {
    let temp_dir = tempdir().unwrap();
    let processor = AsyncFileProcessor::default();

    let unicode_filenames = vec![
        "simple.txt",
        "файл_тест.txt",      // Russian
        "测试文件.txt",         // Chinese
        "テストファイル.txt",      // Japanese
        "🚀_rocket_file.txt",  // Emoji
        "café_résumé.txt",     // French accents
    ];

    for filename in unicode_filenames {
        let file_path = temp_dir.path().join(filename);
        let content = format!("Content for file: {}", filename).into_bytes();

        // Write file
        processor.write_file(&file_path, &content).await.unwrap();

        // Read back and verify
        let read_content = processor.read_file(&file_path).await.unwrap();
        assert_eq!(read_content, content);

        // Validate file info
        let file_info = processor.validate_file(&file_path).await.unwrap();
        assert_eq!(file_info.size, content.len() as u64);
        assert!(file_info.is_file);
    }
}