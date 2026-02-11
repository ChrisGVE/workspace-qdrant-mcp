//! File ingestion tests using Pipeline API
//!
//! These tests validate the Pipeline's handling of ProcessDocument tasks.
//! Note: The current implementation uses a placeholder document processor
//! (see processing.rs:1198-1213) that simulates processing without actual
//! file parsing, embedding generation, or metadata extraction.
//!
//! Current placeholder behavior:
//! - Sleeps for 100ms to simulate processing
//! - Always returns chunks_created: 1
//! - Logs file_path and collection but doesn't parse files
//!
//! These tests validate:
//! ✓ Pipeline task submission workflow
//! ✓ TaskPayload::ProcessDocument routing
//! ✓ TaskResultData::DocumentProcessing structure
//! ✓ Concurrent task execution
//! ✓ Task timeout behavior
//!
//! Future enhancement: When actual document processing is implemented,
//! these tests should be updated to validate real parsing, chunking, and
//! embedding generation behavior.

use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{
    classify_file_type, FileType, Pipeline, TaskPayload, TaskPriority, TaskResult,
    TaskResultData, TaskSource,
};

const TEST_COLLECTION: &str = "test_collection";
const TASK_TIMEOUT: Duration = Duration::from_secs(5);

/// Test helper to create a temporary file with content
async fn create_test_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("test_file.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

/// Test helper to create a document file with a non-test name
/// This is needed because files starting with "test_" are classified as test files
async fn create_document_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("document.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

//
// SINGLE FILE INGESTION TESTS
// Task 315.1: Rewrite single file ingestion tests using Pipeline API
//

#[tokio::test]
async fn test_text_file_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "This is a plain text document.\nWith multiple lines.\nAnd various content.";
    let (_temp_dir, file_path) = create_test_file(content, "txt").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success {
            execution_time_ms,
            data,
        } => {
            // Placeholder sleeps for 100ms
            assert!(execution_time_ms >= 100);

            if let TaskResultData::DocumentProcessing {
                chunks_created,
                collection,
                ..
            } = data
            {
                assert_eq!(collection, TEST_COLLECTION);
                // Placeholder always returns 1 chunk
                assert_eq!(chunks_created, 1);
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Task should succeed, got: {:?}", result),
    }

    Ok(())
}

#[tokio::test]
async fn test_markdown_file_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "# Test Document\n\nThis is a test **markdown** document with `code`.\n";
    let (_temp_dir, file_path) = create_test_file(content, "md").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing {
                chunks_created,
                collection,
                ..
            } = data
            {
                assert_eq!(collection, TEST_COLLECTION);
                assert_eq!(chunks_created, 1); // Placeholder behavior
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Task should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_code_file_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
fn main() {
    println!("Hello, world!");
}
"#;
    let (_temp_dir, file_path) = create_test_file(content, "rs").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing {
                chunks_created,
                collection,
                ..
            } = data
            {
                assert_eq!(collection, TEST_COLLECTION);
                assert_eq!(chunks_created, 1);
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Task should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_python_code_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
def hello_world():
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
"#;
    let (_temp_dir, file_path) = create_test_file(content, "py").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_javascript_code_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
function helloWorld() {
    console.log("Hello, world!");
}

helloWorld();
"#;
    let (_temp_dir, file_path) = create_test_file(content, "js").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_json_config_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"{
    "name": "test-config",
    "version": "1.0.0",
    "enabled": true
}"#;
    let (_temp_dir, file_path) = create_test_file(content, "json").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_empty_file_handling() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let (_temp_dir, file_path) = create_test_file("", "txt").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder still processes empty files
    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing { chunks_created, .. } = data {
                // Placeholder returns 1 even for empty files
                assert_eq!(chunks_created, 1);
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Task should succeed even with empty file"),
    }

    Ok(())
}

#[tokio::test]
async fn test_large_file_ingestion() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    // Create a larger file
    let content = "Lorem ipsum ".repeat(1000); // ~12KB
    let (_temp_dir, file_path) = create_test_file(&content, "txt").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_nonexistent_file() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let nonexistent_path = PathBuf::from("/nonexistent/path/to/file.txt");

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: nonexistent_path,
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder doesn't check file existence, so this will succeed
    // Real implementation should return an error
    // TODO: Update when real file processing is implemented
    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_file_with_unicode_content() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "Hello 世界! Привет мир! مرحبا بالعالم";
    let (_temp_dir, file_path) = create_test_file(content, "txt").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

//
// EDGE CASE TESTS
//

#[tokio::test]
async fn test_file_with_special_characters_in_name() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let file_path = temp_dir
        .path()
        .join("test file with spaces & special!chars.txt");
    fs::write(&file_path, "test content").await?;

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    assert!(matches!(result, TaskResult::Success { .. }));

    Ok(())
}

#[tokio::test]
async fn test_document_processing_execution_time() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let (_temp_dir, file_path) = create_test_file("test content", "txt").await?;

    let start = std::time::Instant::now();

    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    let elapsed = start.elapsed();

    // Should take at least 100ms due to placeholder sleep
    assert!(elapsed >= Duration::from_millis(100));
    assert!(elapsed < Duration::from_secs(1)); // But not too long

    match result {
        TaskResult::Success {
            execution_time_ms, ..
        } => {
            assert!(execution_time_ms >= 100);
            assert!(execution_time_ms < 1000);
        }
        _ => panic!("Task should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_collection_name_validation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let (_temp_dir, file_path) = create_test_file("test content", "txt").await?;

    let test_collections = vec![
        "simple_collection",
        "collection-with-hyphens",
        "collection123",
    ];

    for collection_name in test_collections {
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: collection_name.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        let result = handle.wait().await?;

        if let TaskResult::Success { data, .. } = result {
            if let TaskResultData::DocumentProcessing { collection, .. } = data {
                assert_eq!(collection, collection_name);
            }
        }
    }

    Ok(())
}

//
// BATCH FILE INGESTION TESTS
// Task 315.2: Batch file ingestion tests using Pipeline API
//

#[tokio::test]
async fn test_small_batch_ingestion_10_files() -> TestResult {
    let mut pipeline = Pipeline::new(4); // Allow concurrent processing
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create and submit 10 files
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("file_{}.txt", i));
        fs::write(&file_path, format!("Content of file {}", i)).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "batch_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut success_count = 0;
    for handle in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 10, "All 10 files should be processed successfully");

    Ok(())
}

#[tokio::test]
async fn test_medium_batch_ingestion_50_files() -> TestResult {
    let mut pipeline = Pipeline::new(8); // Higher concurrency
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create and submit 50 files
    for i in 0..50 {
        let file_path = temp_dir.path().join(format!("file_{}.txt", i));
        fs::write(&file_path, format!("Content of file {}", i)).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "batch_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut success_count = 0;
    for handle in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 50, "All 50 files should be processed successfully");

    Ok(())
}

#[tokio::test]
async fn test_large_batch_ingestion_100_files() -> TestResult {
    let mut pipeline = Pipeline::new(10); // Maximum concurrency for stress test
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    let start = std::time::Instant::now();

    // Create and submit 100 files
    for i in 0..100 {
        let file_path = temp_dir.path().join(format!("file_{}.txt", i));
        fs::write(&file_path, format!("Content of file {}", i)).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "batch_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut success_count = 0;
    for handle in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        }
    }

    let elapsed = start.elapsed();

    assert_eq!(success_count, 100, "All 100 files should be processed successfully");

    // With 10 concurrent workers and 100ms per file (placeholder sleep),
    // should complete in roughly 100/10 * 100ms = 1 second, allow 5 seconds max
    assert!(
        elapsed < Duration::from_secs(5),
        "Batch processing should complete within 5 seconds with concurrency, took {:?}",
        elapsed
    );

    Ok(())
}

#[tokio::test]
async fn test_mixed_file_types_batch() -> TestResult {
    let mut pipeline = Pipeline::new(6);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create mixed file types
    let files = vec![
        ("code.rs", "fn main() { println!(\"Hello\"); }"),
        ("code.py", "def main():\n    print('Hello')"),
        ("code.js", "function main() { console.log('Hello'); }"),
        ("doc.md", "# Hello\n\nThis is markdown"),
        ("doc.txt", "Plain text document"),
        ("config.json", r#"{"name": "test", "enabled": true}"#),
        ("data.xml", "<root><item>test</item></root>"),
        ("notes.txt", "Some notes"),
        ("README.md", "# Project\n\nDescription"),
        ("script.sh", "#!/bin/bash\necho 'Hello'"),
    ];

    for (filename, content) in &files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "mixed_batch".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut success_count = 0;
    for handle in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count,
        files.len(),
        "All mixed-type files should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_concurrent_batch_with_unique_document_ids() -> TestResult {
    let mut pipeline = Pipeline::new(5);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Submit 20 files concurrently
    for i in 0..20 {
        let file_path = temp_dir.path().join(format!("file_{}.txt", i));
        fs::write(&file_path, format!("Content {}", i)).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "concurrent_batch".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push(handle);
    }

    // Collect all document IDs to verify uniqueness
    let mut document_ids = Vec::new();
    for handle in handles {
        let result = handle.wait().await?;
        if let TaskResult::Success { data, .. } = result {
            if let TaskResultData::DocumentProcessing { document_id, .. } = data {
                document_ids.push(document_id);
            }
        }
    }

    assert_eq!(document_ids.len(), 20, "Should have 20 document IDs");

    // Verify all document IDs are unique (no duplicates)
    let unique_count = document_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(
        unique_count, 20,
        "All document IDs should be unique, found {} unique out of 20",
        unique_count
    );

    Ok(())
}

#[tokio::test]
async fn test_batch_processing_no_data_loss() -> TestResult {
    let mut pipeline = Pipeline::new(4);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let file_count = 30;
    let mut task_ids = Vec::new();

    // Submit batch and track task IDs
    for i in 0..file_count {
        let file_path = temp_dir.path().join(format!("file_{}.txt", i));
        fs::write(&file_path, format!("Content {}", i)).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "no_loss_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        task_ids.push((i, handle.task_id, handle));
    }

    // Verify all tasks complete
    let mut completed = 0;
    for (index, task_id, handle) in task_ids {
        let result = handle.wait().await;
        assert!(
            result.is_ok(),
            "Task {} (id: {}) should complete successfully",
            index,
            task_id
        );
        completed += 1;
    }

    assert_eq!(
        completed, file_count,
        "All {} tasks should complete, no data loss",
        file_count
    );

    Ok(())
}

#[tokio::test]
async fn test_batch_with_varying_file_sizes() -> TestResult {
    let mut pipeline = Pipeline::new(6);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create files of varying sizes
    let sizes = vec![
        (10, "Small"),          // ~10 bytes
        (1_000, "Medium"),      // ~1KB
        (10_000, "Large"),      // ~10KB
        (50_000, "Very Large"), // ~50KB
    ];

    for (size, label) in sizes {
        let content = "x".repeat(size);
        let file_path = temp_dir.path().join(format!("{}_file.txt", label));
        fs::write(&file_path, &content).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "varying_sizes".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((label, handle));
    }

    // All files should process successfully regardless of size
    for (label, handle) in handles {
        let result = handle.wait().await?;
        assert!(
            matches!(result, TaskResult::Success { .. }),
            "{} file should process successfully",
            label
        );
    }

    Ok(())
}

//
// DOCUMENT FORMAT PROCESSING TESTS
// Task 315.3: Rewrite document format processing tests using Pipeline API
//
// These tests validate file type classification and pipeline processing for various
// document formats. Note: The placeholder implementation doesn't parse file content,
// so we can only test format detection and pipeline acceptance.
//

#[tokio::test]
async fn test_markdown_document_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "# Markdown Document\n\nThis is a markdown file with **bold** and *italic* text.";
    let (_temp_dir, file_path) = create_document_file(content, "md").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Docs, "Markdown files should be classified as docs");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { execution_time_ms, data } => {
            assert!(execution_time_ms >= 100);
            if let TaskResultData::DocumentProcessing { collection, .. } = data {
                assert_eq!(collection, TEST_COLLECTION);
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Markdown file processing should succeed, got: {:?}", result),
    }

    Ok(())
}

#[tokio::test]
async fn test_pdf_document_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    // Create a mock PDF file (not a real PDF, but has .pdf extension)
    // The placeholder processor doesn't parse content anyway
    let content = "Mock PDF content";
    let (_temp_dir, file_path) = create_document_file(content, "pdf").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Docs, "PDF files should be classified as docs");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "PDF file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_docx_document_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    // Create a mock DOCX file
    let content = "Mock DOCX content";
    let (_temp_dir, file_path) = create_document_file(content, "docx").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Docs, "DOCX files should be classified as docs");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "DOCX file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_json_data_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"{"key": "value", "array": [1, 2, 3]}"#;
    let temp_dir = TempDir::new()?;
    // Place JSON in data directory to ensure it's classified as data, not config
    let data_dir = temp_dir.path().join("data");
    fs::create_dir(&data_dir).await?;
    let file_path = data_dir.join("records.json");
    fs::write(&file_path, content).await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type, FileType::Data,
        "JSON files in data directory should be classified as data"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "JSON data file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_json_config_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"{"setting": "value", "debug": true}"#;
    let temp_dir = TempDir::new()?;
    // Place JSON in config directory to ensure it's classified as config
    let config_dir = temp_dir.path().join("config");
    fs::create_dir(&config_dir).await?;
    let file_path = config_dir.join("app.json");
    fs::write(&file_path, content).await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type, FileType::Config,
        "JSON files in config directory should be classified as config"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "JSON config file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_xml_data_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"<?xml version="1.0"?><root><item>data</item></root>"#;
    let temp_dir = TempDir::new()?;
    // Place XML outside config directory to ensure it's classified as data
    let file_path = temp_dir.path().join("export.xml");
    fs::write(&file_path, content).await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type, FileType::Data,
        "XML files outside config directories should be classified as data"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "XML data file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_yaml_config_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "setting: value\ndebug: true\nport: 8080";
    let (_temp_dir, file_path) = create_document_file(content, "yaml").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Config, "YAML files should be classified as config");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "YAML config file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_csv_data_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "name,age,city\nAlice,30,NYC\nBob,25,LA";
    let (_temp_dir, file_path) = create_document_file(content, "csv").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Data, "CSV files should be classified as data");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "CSV data file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_code_file_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "fn main() {\n    println!(\"Hello, world!\");\n}";
    let (_temp_dir, file_path) = create_document_file(content, "rs").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Code, "Rust files should be classified as code");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "format_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_multiple_document_formats_concurrent() -> TestResult {
    let mut pipeline = Pipeline::new(6);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;

    // Create files of different formats
    let formats = vec![
        ("README.md", "# Documentation", FileType::Docs),
        ("data.csv", "col1,col2\n1,2", FileType::Data),
        ("config.yaml", "key: value", FileType::Config),
        ("script.py", "print('hello')", FileType::Code),
        ("notes.txt", "Plain text notes", FileType::Docs),
        ("export.json", r#"{"data": true}"#, FileType::Data),
    ];

    let mut handles = Vec::new();

    for (filename, content, expected_type) in formats.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        // Verify classification
        let file_type = classify_file_type(&file_path);
        assert_eq!(
            &file_type, expected_type,
            "{} should be classified as {:?}",
            filename, expected_type
        );

        // Submit for processing
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "multi_format_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((filename, handle));
    }

    // All formats should process successfully
    let mut success_count = 0;
    for (filename, handle) in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        } else {
            panic!("{} should process successfully, got: {:?}", filename, result);
        }
    }

    assert_eq!(
        success_count, 6,
        "All 6 different file formats should process successfully"
    );

    Ok(())
}

//
// LSP ANALYSIS INTEGRATION TESTS
// Task 315.4: Rewrite LSP analysis integration tests using Pipeline API
//
// These tests validate code file processing through the pipeline and prepare for
// future LSP integration. The placeholder implementation doesn't perform actual
// LSP analysis, but these tests ensure the pipeline correctly handles code files.
//
// Future enhancement: When LSP integration is implemented, these tests should
// validate:
// - Symbol extraction (functions, classes, methods)
// - Language-specific parsing
// - Dependency analysis
// - Code structure metadata
//

#[tokio::test]
async fn test_python_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, x: int, y: int) -> int:
        result = calculate_sum(x, y)
        self.history.append(result)
        return result
"#;
    let (_temp_dir, file_path) = create_document_file(content, "py").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Code, "Python files should be classified as code");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculate_sum, Calculator, add, __init__)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Python code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_javascript_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
function calculateSum(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(x, y) {
        const result = calculateSum(x, y);
        this.history.push(result);
        return result;
    }
}

export { Calculator, calculateSum };
"#;
    let (_temp_dir, file_path) = create_document_file(content, "js").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type, FileType::Code,
        "JavaScript files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculateSum, Calculator, add, constructor)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "JavaScript code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_typescript_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
interface CalculatorInterface {
    add(x: number, y: number): number;
    history: number[];
}

function calculateSum(a: number, b: number): number {
    return a + b;
}

class Calculator implements CalculatorInterface {
    history: number[] = [];

    add(x: number, y: number): number {
        const result = calculateSum(x, y);
        this.history.push(result);
        return result;
    }
}

export { Calculator, calculateSum, CalculatorInterface };
"#;
    let (_temp_dir, file_path) = create_document_file(content, "ts").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type, FileType::Code,
        "TypeScript files should be classified as code"
    );

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (CalculatorInterface, calculateSum, Calculator, add)
    // Future: Validate type information extraction
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "TypeScript code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_rust_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
pub fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Calculator {
    pub history: Vec<i32>,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    pub fn add(&mut self, x: i32, y: i32) -> i32 {
        let result = calculate_sum(x, y);
        self.history.push(result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mut calc = Calculator::new();
        assert_eq!(calc.add(2, 3), 5);
    }
}
"#;
    let (_temp_dir, file_path) = create_document_file(content, "rs").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Code, "Rust files should be classified as code");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (calculate_sum, Calculator, new, add)
    // Future: Validate module structure (tests module)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Rust code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_go_code_with_lsp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
package calculator

func CalculateSum(a int, b int) int {
    return a + b
}

type Calculator struct {
    History []int
}

func NewCalculator() *Calculator {
    return &Calculator{
        History: make([]int, 0),
    }
}

func (c *Calculator) Add(x int, y int) int {
    result := CalculateSum(x, y)
    c.History = append(c.History, result)
    return result
}
"#;
    let (_temp_dir, file_path) = create_document_file(content, "go").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(file_type, FileType::Code, "Go files should be classified as code");

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "lsp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate LSP extracted symbols (CalculateSum, Calculator, NewCalculator, Add)
    // Future: Validate package structure
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Go code file should be processed successfully"
    );

    Ok(())
}

#[tokio::test]
async fn test_multiple_code_languages_concurrent() -> TestResult {
    let mut pipeline = Pipeline::new(5);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;

    // Create code files in different languages
    let code_files = vec![
        (
            "script.py",
            r#"def hello(): print("Hello from Python")"#,
            "Python",
        ),
        (
            "script.js",
            r#"function hello() { console.log("Hello from JavaScript"); }"#,
            "JavaScript",
        ),
        (
            "main.rs",
            r#"fn main() { println!("Hello from Rust"); }"#,
            "Rust",
        ),
        (
            "main.go",
            r#"package main; func main() { println("Hello from Go") }"#,
            "Go",
        ),
        (
            "App.tsx",
            r#"export const App = () => <div>Hello from TypeScript</div>;"#,
            "TypeScript",
        ),
    ];

    let mut handles = Vec::new();

    for (filename, content, language) in code_files.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        // Verify classification
        let file_type = classify_file_type(&file_path);
        assert_eq!(
            file_type, FileType::Code,
            "{} files should be classified as code",
            language
        );

        // Submit for processing
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "multi_language_lsp_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((language, handle));
    }

    // All code files should process successfully
    let mut success_count = 0;
    for (language, handle) in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        } else {
            panic!(
                "{} code file should process successfully, got: {:?}",
                language, result
            );
        }
    }

    assert_eq!(
        success_count, 5,
        "All 5 code files in different languages should process successfully"
    );

    // Future: Validate each language's LSP analysis produced appropriate symbols
    // Future: Validate language-specific metadata (imports, exports, etc.)

    Ok(())
}

//
// METADATA EXTRACTION VERIFICATION TESTS
// Task 315.5: Rewrite metadata extraction verification tests using Pipeline API
//
// These tests validate metadata enrichment through the pipeline and prepare for
// future metadata extraction implementation. The placeholder implementation doesn't
// extract actual metadata, but these tests ensure the pipeline infrastructure is ready.
//
// Future enhancement: When metadata extraction is implemented, these tests should
// validate:
// - File size extraction
// - Creation/modification timestamps
// - File type classification (already validated in format tests)
// - Project context and root detection
// - Git branch information
// - File extension and MIME type
// - Custom metadata fields
//

#[tokio::test]
async fn test_file_metadata_extraction_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "Sample content for metadata extraction testing.\nMultiple lines of text.";
    let (_temp_dir, file_path) = create_document_file(content, "txt").await?;

    // Get file metadata before processing
    let file_metadata = tokio::fs::metadata(&file_path).await?;
    let file_size = file_metadata.len();

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "metadata_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate extracted metadata includes:
    // - file_size: file_size (current: ~58 bytes)
    // - file_extension: ".txt"
    // - file_type: "docs"
    // - created_at: file_metadata.created()
    // - modified_at: file_metadata.modified()
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "File with metadata should be processed successfully"
    );

    // Verify file size is correct
    assert!(file_size > 0, "File size should be greater than 0");
    assert_eq!(file_size, content.len() as u64, "File size should match content length");

    Ok(())
}

#[tokio::test]
async fn test_large_file_metadata_extraction() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    // Create a larger file (~100KB)
    let content = "x".repeat(100_000);
    let (_temp_dir, file_path) = create_document_file(&content, "txt").await?;

    let file_metadata = tokio::fs::metadata(&file_path).await?;
    let file_size = file_metadata.len();

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "large_file_metadata_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate metadata includes correct file_size (~100KB)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Large file metadata extraction should succeed"
    );

    assert_eq!(file_size, 100_000, "File size should be 100KB");

    Ok(())
}

#[tokio::test]
async fn test_file_extension_metadata_extraction() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;

    // Test multiple file extensions
    let extensions = vec![
        ("md", FileType::Docs),
        ("json", FileType::Data),
        ("py", FileType::Code),
        ("yaml", FileType::Config),
    ];

    for (ext, expected_type) in extensions {
        let file_path = temp_dir.path().join(format!("document.{}", ext));
        fs::write(&file_path, "content").await?;

        let file_type = classify_file_type(&file_path);
        assert_eq!(
            file_type, expected_type,
            "Extension .{} should be classified as {:?}",
            ext, expected_type
        );

        // Submit through pipeline
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "extension_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        let result = handle.wait().await?;

        // Placeholder implementation should succeed
        // Future: Validate metadata includes:
        // - file_extension: format!(".{}", ext)
        // - file_type: expected_type.as_str()
        assert!(
            matches!(result, TaskResult::Success { .. }),
            "File with extension .{} should process successfully",
            ext
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_timestamp_metadata_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "Content for timestamp testing";
    let (_temp_dir, file_path) = create_document_file(content, "txt").await?;

    // Get timestamps before processing
    let metadata = tokio::fs::metadata(&file_path).await?;
    let modified_time = metadata.modified()?;

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "timestamp_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate extracted metadata includes:
    // - modified_at: modified_time
    // - created_at: metadata.created() (platform-dependent)
    // - ingestion_timestamp: current time
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "File timestamp metadata should be extracted"
    );

    // Verify we can read the timestamp
    let _timestamp_nanos = modified_time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    Ok(())
}

#[tokio::test]
async fn test_collection_metadata_enrichment() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "Test content";
    let (_temp_dir, file_path) = create_document_file(content, "md").await?;

    let custom_collection = "custom_project_collection";

    // Submit through pipeline with custom collection
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "collection_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: custom_collection.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing { collection, .. } = data {
                // Verify collection name is preserved
                assert_eq!(
                    collection, custom_collection,
                    "Collection metadata should match submitted collection"
                );
                // Future: Validate additional collection-level metadata
                // - project_root
                // - project_id
                // - collection_type
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Collection metadata test should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_concurrent_metadata_extraction() -> TestResult {
    let mut pipeline = Pipeline::new(4);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create files with different characteristics
    let medium_content = "x".repeat(1000);
    let large_content = "y".repeat(10000);
    let test_files = vec![
        ("small.txt", "Small file", 10),
        ("medium.md", medium_content.as_str(), 1000),
        ("large.json", large_content.as_str(), 10000),
        ("code.py", "def main(): pass", 16), // 16 bytes
    ];

    for (filename, content, expected_size) in test_files.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        // Verify file size
        let metadata = tokio::fs::metadata(&file_path).await?;
        assert_eq!(
            metadata.len(),
            *expected_size as u64,
            "File {} should be {} bytes",
            filename,
            expected_size
        );

        // Submit for processing
        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "concurrent_metadata_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((filename, handle));
    }

    // All files should process successfully with correct metadata
    for (filename, handle) in handles {
        let result = handle.wait().await?;
        assert!(
            matches!(result, TaskResult::Success { .. }),
            "File {} metadata extraction should succeed",
            filename
        );
        // Future: Validate each file's metadata is correctly extracted and distinct
    }

    Ok(())
}

//
// EMBEDDING GENERATION VALIDATION TESTS
// Task 315.6: Rewrite embedding generation validation tests using Pipeline API
//
// These tests validate embedding generation through the pipeline and prepare for
// future embedding implementation. The placeholder implementation doesn't generate
// actual embeddings, but these tests ensure the pipeline infrastructure is ready.
//
// Future enhancement: When embedding generation is implemented, these tests should
// validate:
// - Embedding dimensionality (e.g., 384 for all-MiniLM-L6-v2)
// - Embedding values are normalized (unit vectors)
// - Similar content produces similar embeddings (cosine similarity)
// - Different content produces different embeddings
// - Embeddings are reproducible (same input → same output)
// - Batch embedding generation is efficient
// - Dense and sparse embeddings are both generated
//

#[tokio::test]
async fn test_text_embedding_generation_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "This is a sample text document for embedding generation testing.";
    let (_temp_dir, file_path) = create_document_file(content, "txt").await?;

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "embedding_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing { chunks_created, .. } = data {
                // Placeholder returns 1 chunk
                assert_eq!(chunks_created, 1, "Should create 1 chunk (placeholder)");
                // Future: Validate embedding was generated for the chunk
                // Future: Validate embedding dimensionality (e.g., 384)
                // Future: Validate embedding is normalized
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Embedding generation test should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_code_embedding_generation_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = r#"
def calculate_fibonacci(n):
    """Calculate Fibonacci numbers."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"#;
    let (_temp_dir, file_path) = create_document_file(content, "py").await?;

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "code_embedding_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    // Placeholder implementation should succeed
    // Future: Validate code-specific embedding generation
    // Future: Validate embeddings capture code semantics (functions, control flow)
    assert!(
        matches!(result, TaskResult::Success { .. }),
        "Code embedding generation should succeed"
    );

    Ok(())
}

#[tokio::test]
async fn test_batch_embedding_generation() -> TestResult {
    let mut pipeline = Pipeline::new(4);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;
    let mut handles = Vec::new();

    // Create multiple documents with different content
    let documents = vec![
        ("doc1.txt", "The quick brown fox jumps over the lazy dog."),
        ("doc2.txt", "Machine learning is transforming artificial intelligence."),
        ("doc3.txt", "Rust provides memory safety without garbage collection."),
        ("doc4.txt", "Vector databases enable semantic search capabilities."),
    ];

    for (filename, content) in documents.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "batch_embedding_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((filename, handle));
    }

    // All documents should generate embeddings successfully
    let mut success_count = 0;
    for (filename, handle) in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        } else {
            panic!(
                "Embedding generation for {} should succeed, got: {:?}",
                filename, result
            );
        }
    }

    assert_eq!(
        success_count, 4,
        "All 4 documents should generate embeddings successfully"
    );

    // Future: Validate each document has unique embedding
    // Future: Validate embeddings have correct dimensionality
    // Future: Validate batch efficiency (processing time)

    Ok(())
}

#[tokio::test]
async fn test_embedding_consistency_preparation() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "Consistent content for reproducibility testing.";

    // Create two identical files
    let (_temp_dir1, file_path1) = create_document_file(content, "txt").await?;
    let (_temp_dir2, file_path2) = create_document_file(content, "txt").await?;

    // Process first file
    let handle1 = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "consistency_test_1".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path1.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result1 = handle1.wait().await?;

    // Process second file
    let handle2 = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "consistency_test_2".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path2.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result2 = handle2.wait().await?;

    // Both should succeed
    assert!(
        matches!(result1, TaskResult::Success { .. }),
        "First embedding generation should succeed"
    );
    assert!(
        matches!(result2, TaskResult::Success { .. }),
        "Second embedding generation should succeed"
    );

    // Future: Validate embeddings are identical (same content → same embedding)
    // Future: Compare embedding vectors element-wise
    // Future: Validate reproducibility across runs

    Ok(())
}

#[tokio::test]
async fn test_large_text_chunking_and_embedding() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    // Create a large document that should be chunked
    let large_content = "This is a sentence. ".repeat(1000); // ~20,000 characters
    let (_temp_dir, file_path) = create_document_file(&large_content, "txt").await?;

    // Submit through pipeline
    let handle = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::Generic {
                operation: "chunking_test".to_string(),
            },
            TaskPayload::ProcessDocument {
                file_path: file_path.clone(),
                collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
            },
            Some(TASK_TIMEOUT),
        )
        .await?;

    let result = handle.wait().await?;

    match result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::DocumentProcessing { chunks_created, .. } = data {
                // Placeholder returns 1 chunk regardless of size
                assert_eq!(chunks_created, 1, "Placeholder returns 1 chunk");
                // Future: Validate multiple chunks were created (e.g., ~40 chunks for 20KB with 512-byte chunks)
                // Future: Validate each chunk has an embedding
                // Future: Validate chunk overlap is preserved
            } else {
                panic!("Expected DocumentProcessing result");
            }
        }
        _ => panic!("Large text chunking should succeed"),
    }

    Ok(())
}

#[tokio::test]
async fn test_mixed_content_embedding_generation() -> TestResult {
    let mut pipeline = Pipeline::new(5);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let temp_dir = TempDir::new()?;

    // Create files with different content types
    let mixed_content = vec![
        ("technical.md", "# Technical Documentation\n\nThis is technical content."),
        ("code.py", "def hello():\n    print('Hello, world!')"),
        ("data.json", r#"{"key": "value", "number": 42}"#),
        ("config.yaml", "setting: value\nenabled: true"),
        ("plain.txt", "Simple plain text content for testing."),
    ];

    let mut handles = Vec::new();

    for (filename, content) in mixed_content.iter() {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).await?;

        let handle = submitter
            .submit_task(
                TaskPriority::CliCommands,
                TaskSource::Generic {
                    operation: "mixed_content_test".to_string(),
                },
                TaskPayload::ProcessDocument {
                    file_path: file_path.clone(),
                    collection: TEST_COLLECTION.to_string(),
                branch: "main".to_string(),
                },
                Some(TASK_TIMEOUT),
            )
            .await?;

        handles.push((filename, handle));
    }

    // All content types should generate embeddings
    let mut success_count = 0;
    for (filename, handle) in handles {
        let result = handle.wait().await?;
        if matches!(result, TaskResult::Success { .. }) {
            success_count += 1;
        } else {
            panic!(
                "Embedding generation for {} should succeed, got: {:?}",
                filename, result
            );
        }
    }

    assert_eq!(
        success_count, 5,
        "All 5 mixed content types should generate embeddings"
    );

    // Future: Validate embeddings capture content-type specific semantics
    // Future: Validate code embeddings differ from prose embeddings
    // Future: Validate structured data (JSON/YAML) embeddings are meaningful

    Ok(())
}
