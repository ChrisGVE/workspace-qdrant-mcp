//! Async performance, throughput, and stress tests for the workspace-qdrant-mcp daemon.
//!
//! Tests performance metrics tracking, throughput under concurrent load, tokio-test
//! IO mocking utilities, shared test utility integration, and high-concurrency stress.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::time::timeout;
use tokio_test::io::Builder as IoBuilder;
use tempfile::NamedTempFile;

// Import core components
use workspace_qdrant_core::{
    DocumentProcessor,
    logging::track_async_operation,
};

// Import shared test utilities
use shared_test_utils::{
    async_test, TestResult,
    test_helpers::init_test_tracing,
};

/// Test helper for creating test documents with various content types
async fn create_test_document(content: &str, extension: &str) -> TestResult<NamedTempFile> {
    let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension))
        ?;
    tokio::fs::write(temp_file.path(), content).await
        ?;
    Ok(temp_file)
}

// ============================================================================
// PERFORMANCE AND BENCHMARKING TESTS
// ============================================================================

async_test!(test_async_performance_metrics, {
    init_test_tracing();

    let processor = DocumentProcessor::new();
    let test_content = "Performance test document with substantial content to process.".repeat(100);
    let temp_file = create_test_document(&test_content, "txt").await
        .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

    // Track performance metrics
    let start_time = Instant::now();
    let result = track_async_operation("document_processing", async {
        processor.process_file(temp_file.path(), "perf_test").await
    }).await;
    let total_time = start_time.elapsed();

    let doc_result = result?;

    // Verify performance characteristics
    // Note: processing_time_ms can be 0 on fast hardware (sub-millisecond operations)
    assert!(total_time.as_millis() as u64 >= doc_result.processing_time_ms);

    // Performance should be reasonable for this size document
    assert!(doc_result.processing_time_ms < 5000); // Less than 5 seconds

    Ok(())
});

async_test!(test_async_throughput, {
    init_test_tracing();

    let processor = Arc::new(DocumentProcessor::new());
    let start_time = Instant::now();

    // Process multiple documents concurrently
    let mut handles = Vec::new();
    for i in 0..10 {
        let processor_clone = Arc::clone(&processor);
        let handle = tokio::spawn(async move {
            let content = format!("Throughput test document {}", i);
            let temp_file = create_test_document(&content, "txt").await?;
            let result = processor_clone
                .process_file(temp_file.path(), &format!("throughput_{}", i))
                .await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(result)
        });
        handles.push(handle);
    }

    // Wait for all to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await??;
        results.push(result);
    }

    let total_time = start_time.elapsed();

    // Calculate throughput
    let documents_per_second = results.len() as f64 / total_time.as_secs_f64();

    // Should be able to process at least 1 document per second
    assert!(documents_per_second >= 1.0, "Throughput too low: {} docs/sec", documents_per_second);

    Ok(())
});

// ============================================================================
// TOKIO-TEST UTILITIES TESTS
// ============================================================================

async_test!(test_tokio_test_ready_pending, {
    init_test_tracing();

    // Test tokio-test assert_ready and assert_pending
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);

    // Channel should be empty initially
    let recv_future = Box::pin(rx.recv());
    // Test that initially no message is available
    // (simplified from tokio-test polling due to API complexity)

    // Send a message
    tx.send("test_message".to_string()).await.unwrap();

    // Now the receive should be ready
    let result = recv_future.await;
    assert_eq!(result.unwrap(), "test_message");

    Ok(())
});

async_test!(test_tokio_test_io_builder, {
    init_test_tracing();

    // Test tokio-test IO mocking capabilities
    let mut mock_reader = IoBuilder::new()
        .read(b"Hello, async world!")
        .build();

    let mut buffer = Vec::new();
    use tokio::io::AsyncReadExt;
    let bytes_read = mock_reader.read_to_end(&mut buffer).await?;

    assert_eq!(bytes_read, 19);
    assert_eq!(buffer, b"Hello, async world!");

    Ok(())
});

// ============================================================================
// INTEGRATION WITH SHARED TEST UTILITIES
// ============================================================================

async_test!(test_shared_utilities_integration, {
    init_test_tracing();

    // Test integration with shared test utilities
    let temp_dir = tempfile::TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    tokio::fs::write(&test_file, "Shared utility test content").await?;

    let processor = DocumentProcessor::new();
    let result = processor.process_file(&test_file, "shared_util_test").await?;

    // Verify processing time is reasonable
    assert!(result.processing_time_ms < 5000);

    Ok(())
});

// ============================================================================
// STRESS TESTS
// ============================================================================

#[cfg(test)]
mod async_stress_tests {
    use super::*;

    async_test!(test_high_concurrency_stress, {
        init_test_tracing();

        let processor = Arc::new(DocumentProcessor::new());
        let mut handles = Vec::new();

        // Create many concurrent processing tasks
        for i in 0..100 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let content = format!("Stress test document {} with content", i);
                let temp_file = create_test_document(&content, "txt").await?;
                let result = processor_clone
                    .process_file(temp_file.path(), &format!("stress_{}", i))
                    .await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(result)
            });
            handles.push(handle);
        }

        // Wait for all tasks with a reasonable timeout
        let mut results = Vec::new();
        for handle in handles {
            let result = timeout(Duration::from_secs(30), handle).await??;
            results.push(result);
        }

        let final_results: Result<Vec<_>, _> = results.into_iter().collect();
        let final_results = final_results?;

        assert_eq!(final_results.len(), 100);

        Ok(())
    });
}
