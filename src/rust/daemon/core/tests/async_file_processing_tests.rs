//! Async file processing tests for the workspace-qdrant-mcp daemon.
//!
//! Tests async document processing with various content types, concurrent
//! processing, timeout handling, and error propagation in async contexts.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tempfile::NamedTempFile;
use tokio::time::timeout;

// Import core components
use workspace_qdrant_core::{DocumentProcessor, DocumentProcessorError};

// Import shared test utilities
use shared_test_utils::{async_test, test_helpers::init_test_tracing, TestResult};

/// Test helper for creating test documents with various content types
async fn create_test_document(content: &str, extension: &str) -> TestResult<NamedTempFile> {
    let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension))?;
    tokio::fs::write(temp_file.path(), content).await?;
    Ok(temp_file)
}

/// Test helper for verifying async operation timing
async fn verify_async_timing<F, T>(
    operation: F,
    expected_min: Duration,
    expected_max: Duration,
) -> TestResult<T>
where
    F: std::future::Future<Output = T>,
{
    let start = Instant::now();
    let result = operation.await;
    let elapsed = start.elapsed();

    assert!(
        elapsed >= expected_min,
        "Operation completed too quickly: {:?} < {:?}",
        elapsed,
        expected_min
    );
    assert!(
        elapsed <= expected_max,
        "Operation took too long: {:?} > {:?}",
        elapsed,
        expected_max
    );

    Ok(result)
}

// ============================================================================
// ASYNC FILE PROCESSING TESTS
// ============================================================================

async_test!(test_async_document_processor_creation, {
    init_test_tracing();

    // Test that DocumentProcessor can be created asynchronously
    let processor = tokio::task::spawn(async { DocumentProcessor::new() }).await?;

    // Verify processor is healthy
    assert!(processor.is_healthy().await);

    Ok(())
});

async_test!(test_async_file_processing_text, {
    init_test_tracing();

    let processor = DocumentProcessor::new();
    let test_content = "This is a test document for async processing.\nIt has multiple lines.";
    let temp_file = create_test_document(test_content, "txt").await?;

    // Test async file processing with timing verification
    // Note: No minimum time assertion as fast hardware can complete sub-millisecond
    let result = verify_async_timing(
        processor.process_file(temp_file.path(), "test_collection"),
        Duration::ZERO,
        Duration::from_secs(5),
    )
    .await?;

    let doc_result = result?;
    assert!(!doc_result.document_id.is_empty());
    assert_eq!(doc_result.collection, "test_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
    // Note: processing_time_ms can be 0 on fast hardware (sub-millisecond operations)

    Ok(())
});

async_test!(test_async_file_processing_markdown, {
    init_test_tracing();

    let processor = DocumentProcessor::new();
    let test_content = r#"# Async Test Document

## Section 1
This is a markdown document for testing async processing.

```rust
async fn example() {
    println!(\"Hello from async!\");
}
```

## Section 2
More content to ensure proper chunking."#;

    let temp_file = create_test_document(test_content, "md").await?;

    let result = processor
        .process_file(temp_file.path(), "markdown_test")
        .await?;

    assert!(!result.document_id.is_empty());
    assert_eq!(result.collection, "markdown_test");
    assert!(result.chunks_created.unwrap_or(0) > 0);

    Ok(())
});

async_test!(test_async_file_processing_code, {
    init_test_tracing();

    let processor = DocumentProcessor::new();
    let test_content = r#"//! Async test code

use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = process_async_data().await?;
    println!(\"Processed: {}\", result);
    Ok(())
}

async fn process_async_data() -> Result<String, &'static str> {
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(\"async data\".to_string())
}"#;

    let temp_file = create_test_document(test_content, "rs").await?;

    let result = processor
        .process_file(temp_file.path(), "code_test")
        .await?;

    assert!(!result.document_id.is_empty());
    assert_eq!(result.collection, "code_test");
    assert!(result.chunks_created.unwrap_or(0) > 0);

    Ok(())
});

async_test!(test_async_concurrent_file_processing, {
    init_test_tracing();

    let processor = Arc::new(DocumentProcessor::new());

    // Create multiple test files
    let files = vec![
        ("Test document 1", "txt"),
        ("# Test document 2", "md"),
        ("console.log('test 3');", "js"),
    ];

    let mut temp_files = Vec::new();
    for (content, ext) in files {
        let temp_file = create_test_document(content, ext).await?;
        temp_files.push(temp_file);
    }

    // Process files concurrently
    let mut handles = Vec::new();
    for (i, temp_file) in temp_files.iter().enumerate() {
        let processor_clone = Arc::clone(&processor);
        let path = temp_file.path().to_path_buf();
        let collection = format!("concurrent_test_{}", i);

        let handle =
            tokio::spawn(async move { processor_clone.process_file(&path, &collection).await });
        handles.push(handle);
    }

    // Wait for all to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await??;
        results.push(result);
    }

    // Verify all succeeded
    assert_eq!(results.len(), 3);
    for (i, result) in results.iter().enumerate() {
        assert!(!result.document_id.is_empty());
        assert_eq!(result.collection, format!("concurrent_test_{}", i));
    }

    Ok(())
});

async_test!(test_async_file_processing_error_handling, {
    init_test_tracing();

    let processor = DocumentProcessor::new();

    // Test processing non-existent file
    let non_existent_path = Path::new("/tmp/non_existent_file_12345.txt");
    let result = processor
        .process_file(non_existent_path, "error_test")
        .await;

    assert!(result.is_err());

    match result {
        Err(DocumentProcessorError::FileNotFound(_)) => {} // Expected
        Err(e) => panic!("Unexpected error type: {:?}", e),
        Ok(_) => panic!("Expected error but got success"),
    }

    Ok(())
});

// ============================================================================
// TIMEOUT HANDLING
// ============================================================================

async_test!(test_async_timeout_handling, {
    init_test_tracing();

    let processor = DocumentProcessor::new();

    // Create a task that would take too long
    let long_running_task = async {
        // Simulate a very slow operation
        tokio::time::sleep(Duration::from_secs(10)).await;
        processor
            .process_file(Path::new("/tmp/test.txt"), "timeout_test")
            .await
    };

    // Test that timeout works properly
    let result = timeout(Duration::from_millis(100), long_running_task).await;

    assert!(result.is_err()); // Should timeout

    Ok(())
});

// ============================================================================
// ERROR HANDLING IN ASYNC CONTEXTS
// ============================================================================

async_test!(test_async_error_propagation, {
    init_test_tracing();

    let processor = DocumentProcessor::new();

    // Test error propagation through async chains
    let result: TestResult<()> = async {
        let temp_file = create_test_document("test", "txt").await?;

        // Drop the temp file to cause an error
        drop(temp_file);

        // This should fail because file no longer exists
        let non_existent_path = Path::new("/tmp/dropped_file.txt");
        processor
            .process_file(non_existent_path, "error_prop_test")
            .await?;
        Ok(())
    }
    .await;

    assert!(result.is_err());

    Ok(())
});

async_test!(test_async_panic_handling, {
    init_test_tracing();

    // Test that panics in async tasks are properly caught
    let panic_task = tokio::spawn(async {
        panic!("Intentional panic for testing");
    });

    let result = panic_task.await;
    assert!(result.is_err());

    // Verify the panic was caught and the task failed
    let err = result.unwrap_err();
    assert!(err.is_panic());

    Ok(())
});
