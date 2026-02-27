//! File ingestion edge case tests using Pipeline API
//!
//! These tests validate edge cases: special characters in filenames,
//! execution time tracking, and collection name validation.

use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{Pipeline, TaskPayload, TaskPriority, TaskResult, TaskResultData, TaskSource};

const TEST_COLLECTION: &str = "test_collection";
const TASK_TIMEOUT: Duration = Duration::from_secs(5);

/// Test helper to create a temporary file with content
async fn create_test_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("test_file.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

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
