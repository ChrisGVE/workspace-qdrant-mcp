//! Document format processing tests using Pipeline API (Task 315.3)
//!
//! These tests validate file type classification and pipeline processing for various
//! document formats. Note: The placeholder implementation doesn't parse file content,
//! so we can only test format detection and pipeline acceptance.

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
/// This is needed because files starting with "test_" are classified as test files
async fn create_document_file(content: &str, extension: &str) -> TestResult<(TempDir, PathBuf)> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(format!("document.{}", extension));
    fs::write(&file_path, content).await?;
    Ok((temp_dir, file_path))
}

#[tokio::test]
async fn test_markdown_document_classification_and_processing() -> TestResult {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    pipeline.start().await?;

    let content = "# Markdown Document\n\nThis is a markdown file with **bold** and *italic* text.";
    let (_temp_dir, file_path) = create_document_file(content, "md").await?;

    // Verify file classification
    let file_type = classify_file_type(&file_path);
    assert_eq!(
        file_type,
        FileType::Text,
        "Markdown files should be classified as text"
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

    match result {
        TaskResult::Success {
            execution_time_ms,
            data,
        } => {
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
    assert_eq!(
        file_type,
        FileType::Docs,
        "PDF files should be classified as docs"
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
    assert_eq!(
        file_type,
        FileType::Docs,
        "DOCX files should be classified as docs"
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
        file_type,
        FileType::Data,
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
        file_type,
        FileType::Config,
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
        file_type,
        FileType::Web,
        "XML files should be classified as web content"
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
    assert_eq!(
        file_type,
        FileType::Config,
        "YAML files should be classified as config"
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
    assert_eq!(
        file_type,
        FileType::Data,
        "CSV files should be classified as data"
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
    assert_eq!(
        file_type,
        FileType::Code,
        "Rust files should be classified as code"
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
        ("README.md", "# Documentation", FileType::Text),
        ("data.csv", "col1,col2\n1,2", FileType::Data),
        ("config.yaml", "key: value", FileType::Config),
        ("script.py", "print('hello')", FileType::Code),
        ("notes.txt", "Plain text notes", FileType::Text),
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
            panic!(
                "{} should process successfully, got: {:?}",
                filename, result
            );
        }
    }

    assert_eq!(
        success_count, 6,
        "All 6 different file formats should process successfully"
    );

    Ok(())
}
