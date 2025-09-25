# Rust Testing Guide: Engine Components

## Overview

This guide covers Rust testing best practices for the high-performance engine components in the workspace-qdrant-mcp project. The Rust engine provides file watching, document processing, and gRPC services with comprehensive test coverage.

## Testing Framework and Structure

### Core Technologies

- **cargo test**: Native Rust testing framework
- **tokio-test**: Asynchronous runtime testing
- **mockall**: Mock object generation and dependency injection
- **criterion**: Performance benchmarking
- **proptest**: Property-based testing for edge cases
- **tempfile**: Temporary file and directory management

### Test Directory Structure

```
rust-engine/
├── src/
│   ├── lib.rs
│   ├── processing.rs       # Document processing logic
│   ├── watching.rs        # File system monitoring
│   ├── daemon.rs          # Daemon service implementation
│   └── grpc.rs           # gRPC service layer
├── tests/
│   ├── integration_tests.rs
│   ├── processing_tests.rs
│   ├── watching_tests.rs
│   └── grpc_tests.rs
├── benches/
│   ├── processing_bench.rs
│   └── watching_bench.rs
└── Cargo.toml
```

## Test Writing Best Practices

### Test Naming and Organization

```rust
// Unit tests within the module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_document_with_valid_input_returns_success() {
        // Test implementation
    }

    #[tokio::test]
    async fn test_async_operation_with_timeout_completes_successfully() {
        // Async test implementation
    }
}

// Integration tests in separate files
// tests/processing_tests.rs
#[tokio::test]
async fn test_document_processing_pipeline_end_to_end() {
    // Full pipeline test
}
```

### Test Structure: Given-When-Then

```rust
#[tokio::test]
async fn test_file_watcher_detects_new_file() {
    // Given: Setup test environment
    let temp_dir = tempfile::tempdir().unwrap();
    let watcher_config = WatcherConfig {
        watch_path: temp_dir.path().to_path_buf(),
        debounce_ms: 100,
    };
    let mut watcher = FileWatcher::new(watcher_config).await.unwrap();

    // When: Create a new file
    let test_file = temp_dir.path().join("test.txt");
    tokio::fs::write(&test_file, "test content").await.unwrap();

    // Then: Verify file event is detected
    let event = timeout(Duration::from_secs(5), watcher.next_event())
        .await
        .expect("Timeout waiting for file event")
        .expect("Failed to receive file event");

    assert_eq!(event.path, test_file);
    assert_eq!(event.event_type, FileEventType::Created);
}
```

## Async Testing Patterns

### Tokio Runtime Testing

```rust
use tokio::time::{timeout, Duration};
use tokio_test::{assert_ok, assert_err, assert_pending, assert_ready};

#[tokio::test]
async fn test_concurrent_document_processing() {
    let processor = DocumentProcessor::new().await.unwrap();

    // Process multiple documents concurrently
    let tasks: Vec<_> = (0..10)
        .map(|i| {
            let processor = processor.clone();
            tokio::spawn(async move {
                processor.process_document(format!("doc_{}", i)).await
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Verify all succeeded
    for result in results {
        assert_ok!(result.unwrap());
    }
}

#[tokio::test]
async fn test_operation_with_timeout() {
    let processor = DocumentProcessor::new().await.unwrap();

    // Test operation completes within timeout
    let result = timeout(
        Duration::from_secs(10),
        processor.process_large_document("large_doc.pdf")
    ).await;

    assert_ok!(result);
}
```

### Stream Testing

```rust
use futures::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

#[tokio::test]
async fn test_document_stream_processing() {
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    let mut stream = ReceiverStream::new(rx);
    let processor = DocumentProcessor::new().await.unwrap();

    // Send test documents
    for i in 0..5 {
        tx.send(format!("document_{}", i)).await.unwrap();
    }
    drop(tx); // Close channel

    // Process stream
    let mut results = Vec::new();
    while let Some(doc) = stream.next().await {
        let result = processor.process_document(doc).await.unwrap();
        results.push(result);
    }

    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.is_success()));
}
```

## Mocking and Dependency Injection

### Using Mockall

```rust
use mockall::*;

#[automock]
pub trait DocumentStore {
    async fn store_document(&self, doc: Document) -> Result<String, StoreError>;
    async fn retrieve_document(&self, id: &str) -> Result<Document, StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_processor_stores_processed_document() {
        // Setup mock
        let mut mock_store = MockDocumentStore::new();
        mock_store
            .expect_store_document()
            .times(1)
            .returning(|_| Ok("doc_id_123".to_string()));

        // Test processor with mock
        let processor = DocumentProcessor::new_with_store(Box::new(mock_store));
        let doc = Document::new("test content");

        let result = processor.process_and_store(doc).await;

        assert_ok!(result);
        assert_eq!(result.unwrap(), "doc_id_123");
    }
}
```

### Dependency Injection Pattern

```rust
pub struct DocumentProcessor<S: DocumentStore> {
    store: S,
    config: ProcessorConfig,
}

impl<S: DocumentStore> DocumentProcessor<S> {
    pub fn new_with_store(store: S) -> Self {
        Self {
            store,
            config: ProcessorConfig::default(),
        }
    }

    pub async fn process_document(&self, content: String) -> Result<ProcessedDocument, ProcessError> {
        let processed = self.internal_process(content)?;
        let doc_id = self.store.store_document(processed.clone()).await?;
        Ok(processed.with_id(doc_id))
    }
}

#[cfg(test)]
mod tests {
    struct TestStore;

    impl DocumentStore for TestStore {
        async fn store_document(&self, doc: Document) -> Result<String, StoreError> {
            Ok(format!("test_id_{}", doc.hash()))
        }

        async fn retrieve_document(&self, id: &str) -> Result<Document, StoreError> {
            Ok(Document::new(format!("content for {}", id)))
        }
    }

    #[tokio::test]
    async fn test_with_test_implementation() {
        let processor = DocumentProcessor::new_with_store(TestStore);
        let result = processor.process_document("test".to_string()).await;
        assert_ok!(result);
    }
}
```

## File System Testing

### Temporary Directory Testing

```rust
use tempfile::{TempDir, NamedTempFile};

#[tokio::test]
async fn test_file_watcher_with_temp_directory() {
    let temp_dir = TempDir::new().unwrap();
    let watch_path = temp_dir.path();

    let mut watcher = FileWatcher::new(watch_path).await.unwrap();

    // Create nested directory structure
    let nested_dir = watch_path.join("nested");
    tokio::fs::create_dir_all(&nested_dir).await.unwrap();

    // Create test file
    let test_file = nested_dir.join("test.txt");
    tokio::fs::write(&test_file, "test content").await.unwrap();

    // Verify watcher detects the file
    let events = collect_events(&mut watcher, Duration::from_secs(2)).await;
    assert!(events.iter().any(|e| e.path == test_file));

    // Cleanup is automatic when temp_dir goes out of scope
}

async fn collect_events(watcher: &mut FileWatcher, duration: Duration) -> Vec<FileEvent> {
    let mut events = Vec::new();
    let deadline = tokio::time::Instant::now() + duration;

    while tokio::time::Instant::now() < deadline {
        match timeout(Duration::from_millis(100), watcher.next_event()).await {
            Ok(Ok(event)) => events.push(event),
            _ => break,
        }
    }

    events
}
```

### File Content Testing

```rust
#[tokio::test]
async fn test_document_parser_with_various_formats() {
    let test_files = [
        ("test.txt", "Plain text content"),
        ("test.md", "# Markdown Content\n\nSome text"),
        ("test.json", r#"{"key": "value", "nested": {"data": 123}}"#),
    ];

    for (filename, content) in test_files.iter() {
        let temp_file = NamedTempFile::new().unwrap();
        tokio::fs::write(temp_file.path(), content).await.unwrap();

        let parser = DocumentParser::new();
        let result = parser.parse_file(temp_file.path()).await;

        assert_ok!(&result);
        let parsed = result.unwrap();
        assert!(!parsed.content.is_empty());
        assert_eq!(parsed.file_type, detect_file_type(filename));
    }
}
```

## Error Handling and Edge Cases

### Error Propagation Testing

```rust
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum ProcessingError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    #[error("Invalid format: {format}")]
    InvalidFormat { format: String },
    #[error("Processing timeout after {seconds}s")]
    Timeout { seconds: u64 },
}

#[tokio::test]
async fn test_error_handling_chain() {
    let processor = DocumentProcessor::new().await.unwrap();

    // Test file not found error
    let result = processor.process_file("nonexistent.txt").await;
    assert_err!(&result);
    match result.unwrap_err() {
        ProcessingError::FileNotFound { path } => {
            assert_eq!(path, "nonexistent.txt");
        }
        _ => panic!("Expected FileNotFound error"),
    }

    // Test invalid format error
    let temp_file = create_invalid_file().await;
    let result = processor.process_file(temp_file.to_str().unwrap()).await;
    assert_err!(&result);
    assert!(matches!(result.unwrap_err(), ProcessingError::InvalidFormat { .. }));
}
```

### Boundary Condition Testing

```rust
#[tokio::test]
async fn test_large_file_processing() {
    let processor = DocumentProcessor::new().await.unwrap();

    // Create large test file (1MB)
    let large_content = "a".repeat(1024 * 1024);
    let temp_file = NamedTempFile::new().unwrap();
    tokio::fs::write(temp_file.path(), &large_content).await.unwrap();

    let result = processor.process_file(temp_file.path().to_str().unwrap()).await;
    assert_ok!(result);
}

#[tokio::test]
async fn test_empty_file_processing() {
    let processor = DocumentProcessor::new().await.unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    // File is empty by default

    let result = processor.process_file(temp_file.path().to_str().unwrap()).await;

    // Should handle empty files gracefully
    assert_ok!(result);
    let processed = result.unwrap();
    assert!(processed.content.is_empty());
}
```

## Performance Testing and Benchmarking

### Criterion Benchmarks

```rust
// benches/processing_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use workspace_daemon::processing::DocumentProcessor;

async fn process_document_benchmark() {
    let processor = DocumentProcessor::new().await.unwrap();
    let content = "Sample document content for benchmarking";

    processor.process_document(black_box(content.to_string())).await.unwrap();
}

fn processing_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("process_document", |b| {
        b.to_async(&rt).iter(|| process_document_benchmark())
    });
}

criterion_group!(benches, processing_benchmark);
criterion_main!(benches);
```

### Memory Usage Testing

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MemoryTracker;

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for MemoryTracker {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ret = System.alloc(layout);
            if !ret.is_null() {
                ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            }
            ret
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_memory_usage_during_processing() {
        let initial_memory = ALLOCATED.load(Ordering::SeqCst);

        {
            let processor = DocumentProcessor::new().await.unwrap();
            let _ = processor.process_document("test content".to_string()).await;
        } // processor goes out of scope

        tokio::time::sleep(Duration::from_millis(100)).await;
        let final_memory = ALLOCATED.load(Ordering::SeqCst);

        // Memory usage should return close to initial after cleanup
        let memory_diff = final_memory.saturating_sub(initial_memory);
        assert!(memory_diff < 1024 * 1024, "Memory leak detected: {} bytes", memory_diff);
    }
}
```

## gRPC Service Testing

### gRPC Client/Server Testing

```rust
use tonic::Request;
use workspace_daemon::proto::{DocumentRequest, ProcessingService};

#[tokio::test]
async fn test_grpc_document_processing() {
    // Start test server
    let addr = "127.0.0.1:0".parse().unwrap();
    let server = ProcessingService::new(DocumentProcessor::new().await.unwrap());
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create client and test
    let mut client = ProcessingServiceClient::connect(format!("http://{}", addr))
        .await
        .unwrap();

    let request = Request::new(DocumentRequest {
        content: "test document".to_string(),
        file_type: "text".to_string(),
    });

    let response = client.process_document(request).await.unwrap();
    let result = response.into_inner();

    assert!(result.success);
    assert!(!result.processed_content.is_empty());
}
```

## Property-Based Testing

### Using Proptest

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_document_processing_never_panics(
        content in ".*",
        file_type in prop::option::of("[a-z]{2,10}")
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let processor = DocumentProcessor::new().await.unwrap();

            // Should never panic, regardless of input
            let _ = processor.process_document_with_type(content, file_type).await;
        });
    }

    #[test]
    fn test_file_path_validation(path in ".*") {
        let validator = FilePathValidator::new();

        // Should not panic on any string input
        let _ = validator.validate_path(&path);
    }
}
```

## Integration Testing

### Full Pipeline Testing

```rust
// tests/integration_tests.rs
use workspace_daemon::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_full_document_processing_pipeline() {
    // Setup test environment
    let temp_dir = TempDir::new().unwrap();
    let config = create_test_config(temp_dir.path());

    // Initialize all components
    let watcher = FileWatcher::new(&config.watch_config).await.unwrap();
    let processor = DocumentProcessor::new(&config.processor_config).await.unwrap();
    let store = DocumentStore::new(&config.store_config).await.unwrap();

    // Create test document
    let test_file = temp_dir.path().join("test.md");
    tokio::fs::write(&test_file, "# Test Document\n\nContent here.").await.unwrap();

    // Wait for file system event
    let event = watcher.next_event().await.unwrap();
    assert_eq!(event.path, test_file);

    // Process the document
    let processed = processor.process_file(&test_file).await.unwrap();
    assert!(!processed.content.is_empty());

    // Store the result
    let doc_id = store.store_document(processed.clone()).await.unwrap();
    assert!(!doc_id.is_empty());

    // Verify retrieval
    let retrieved = store.retrieve_document(&doc_id).await.unwrap();
    assert_eq!(retrieved.content, processed.content);
}
```

## Test Configuration and Utilities

### Test Configuration

```rust
// tests/common/mod.rs
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub processor: ProcessorConfig,
    pub watcher: WatcherConfig,
    pub store: StoreConfig,
}

pub fn create_test_config(temp_dir: &Path) -> TestConfig {
    TestConfig {
        processor: ProcessorConfig {
            max_file_size: 10 * 1024 * 1024, // 10MB
            supported_formats: vec!["txt", "md", "json"].into_iter().map(String::from).collect(),
            timeout_seconds: 30,
        },
        watcher: WatcherConfig {
            watch_path: temp_dir.to_path_buf(),
            debounce_ms: 100,
            recursive: true,
        },
        store: StoreConfig {
            connection_string: "memory://".to_string(),
            max_connections: 10,
        },
    }
}
```

### Test Utilities

```rust
pub struct TestEnvironment {
    pub temp_dir: TempDir,
    pub config: TestConfig,
}

impl TestEnvironment {
    pub async fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path());

        Self { temp_dir, config }
    }

    pub async fn create_test_file(&self, name: &str, content: &str) -> std::path::PathBuf {
        let file_path = self.temp_dir.path().join(name);
        tokio::fs::write(&file_path, content).await.unwrap();
        file_path
    }

    pub async fn cleanup(&self) {
        // Explicit cleanup if needed
    }
}
```

## Running and Configuring Tests

### Cargo Test Configuration

```toml
# Cargo.toml
[dev-dependencies]
tokio-test = "0.4"
mockall = "0.12"
criterion = "0.5"
proptest = "1.0"
tempfile = "3.0"

[[bench]]
name = "processing_bench"
harness = false

[package.metadata.coverage]
exclude = ["tests/*", "benches/*"]
```

### Test Execution Commands

```bash
# Run all tests
cargo test

# Run specific test module
cargo test processing_tests

# Run tests with output
cargo test -- --nocapture

# Run integration tests only
cargo test --test integration_tests

# Run with coverage (requires cargo-llvm-cov)
cargo llvm-cov --html --output-dir target/coverage

# Run benchmarks
cargo bench

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test
```

This comprehensive Rust testing guide provides the foundation for maintaining high-quality, performant Rust components in the workspace-qdrant-mcp project. The patterns and examples shown here support the project's comprehensive testing coverage and enable robust concurrent and asynchronous operations.