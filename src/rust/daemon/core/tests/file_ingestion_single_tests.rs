//! Single file ingestion tests using Pipeline API (Task 315.1)
//!
//! These tests validate the Pipeline's handling of individual ProcessDocument tasks
//! for various file types and edge cases.

use shared_test_utils::TestResult;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{
    Pipeline, TaskPayload, TaskPriority, TaskResult, TaskResultData, TaskSource,
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
