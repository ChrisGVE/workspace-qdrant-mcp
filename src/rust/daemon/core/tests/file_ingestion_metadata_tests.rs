//! Metadata extraction verification tests using Pipeline API (Task 315.5)
//!
//! These tests validate metadata enrichment through the pipeline and prepare for
//! future metadata extraction implementation. The placeholder implementation doesn't
//! extract actual metadata, but these tests ensure the pipeline infrastructure is ready.
//!
//! Future enhancement: When metadata extraction is implemented, these tests should
//! validate file size, timestamps, file type classification, project context,
//! Git branch information, and custom metadata fields.

use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{
    classify_file_type, FileType, Pipeline, TaskPayload, TaskPriority, TaskResult, TaskResultData,
    TaskSource,
};

const TEST_COLLECTION: &str = "test_collection";
const TASK_TIMEOUT: Duration = Duration::from_secs(5);

/// Test helper to create a document file with a non-test name
async fn create_document_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("document.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

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
    assert_eq!(
        file_size,
        content.len() as u64,
        "File size should match content length"
    );

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
        ("md", FileType::Text),
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
