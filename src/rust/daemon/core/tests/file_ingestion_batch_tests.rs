//! Batch file ingestion tests using Pipeline API (Task 315.2)
//!
//! These tests validate concurrent batch processing of multiple files through
//! the Pipeline, including varying batch sizes, mixed file types, unique document
//! IDs, data loss prevention, and varying file sizes.

use shared_test_utils::TestResult;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{
    Pipeline, TaskPayload, TaskPriority, TaskResult, TaskResultData, TaskSource,
};

const TEST_COLLECTION: &str = "test_collection";
const TASK_TIMEOUT: Duration = Duration::from_secs(5);

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

    assert_eq!(
        success_count, 10,
        "All 10 files should be processed successfully"
    );

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

    assert_eq!(
        success_count, 50,
        "All 50 files should be processed successfully"
    );

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

    assert_eq!(
        success_count, 100,
        "All 100 files should be processed successfully"
    );

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
    let unique_count = document_ids
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
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
