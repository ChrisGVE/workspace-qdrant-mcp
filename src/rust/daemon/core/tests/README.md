# Rust Daemon Core Test Suite

This directory contains comprehensive tests for the workspace-qdrant-mcp Rust daemon core functionality.

## Test Organization

### Unit Tests
- `embedding_tests.rs` - Embedding generation and validation tests
- `document_processor_tests.rs` - Document processing pipeline tests
- `async_unit_tests.rs` - Async operation unit tests

### Integration Tests
- `file_ingestion_comprehensive_tests.rs` - **NEW: Comprehensive file ingestion tests**
- `integration_tests.rs` - System integration tests
- `hybrid_search_comprehensive_tests.rs` - Hybrid search functionality
- `qdrant_client_validation_tests.rs` - Qdrant client integration

### Functional Tests
- `functional_sample.rs` - End-to-end functional tests
- `file_watching_tests.rs` - File system monitoring tests
- `property_file_monitoring_tests.rs` - Property-based monitoring tests

### Performance Tests
- `ffi_performance_tests.rs` - FFI boundary performance
- `cross_platform_safety_tests.rs` - Cross-platform validation
- `benches/file_ingestion_benchmarks.rs` - **NEW: File ingestion benchmarks**

### Safety and Quality
- `unsafe_code_audit_tests.rs` - Unsafe code validation
- `valgrind_memory_tests.rs` - Memory leak detection
- `property_based_tests.rs` - Property-based testing with proptest

## File Ingestion Test Framework

The `file_ingestion_comprehensive_tests.rs` module provides extensive coverage for daemon file ingestion:

### Test Categories

#### 1. Format Ingestion (`format_ingestion`)
Tests for various file formats:
- PDF format detection and processing
- EPUB ebook parsing
- DOCX document extraction
- Markdown file handling
- Plain text file ingestion
- Empty file edge cases
- Large file chunking (100KB+ files)

#### 2. Code File Ingestion (`code_ingestion`)
Tests for code file parsing with LSP analysis:
- **Rust** - Full syntax tree analysis
- **Python** - AST parsing with type hints
- **JavaScript** - ES6+ syntax support
- **Go** - Package and type analysis
- **JSON** - Configuration file parsing
- Unicode support (🦀 日本語 Русский العربية)

#### 3. Metadata Extraction (`metadata_extraction`)
Validates metadata extraction:
- File system metadata (size, timestamps)
- Document type detection accuracy
- MIME type resolution
- Extension-based fallback

#### 4. Chunking Tests (`chunking_tests`)
Tests chunking configuration:
- Custom chunk sizes (10-1000 characters)
- Overlap configuration
- Paragraph preservation
- Boundary handling

#### 5. Edge Cases (`edge_cases`)
Error handling and special scenarios:
- Binary file handling
- Non-existent files
- Special characters in filenames
- Very long lines (1000+ words)
- Mixed line endings (CRLF/LF)
- Whitespace-only files

#### 6. Property-Based Tests (`property_based`)
Uses proptest for arbitrary input testing:
- Arbitrary text processing (1-1000 chars)
- Arbitrary chunk sizes (10-1000 bytes)
- Arbitrary file extensions
- Fuzz testing document processor

#### 7. Stress Tests (`stress_tests`)
High-load scenario testing:
- Concurrent file processing (50 files)
- Large batch processing
- Very large files (10MB+)
- Memory usage under load

**Note**: Stress tests are marked with `#[ignore]` and must be run explicitly:
```bash
cargo test --test file_ingestion_comprehensive_tests stress_tests -- --ignored
```

#### 8. Integration Tests (`integration_tests`)
Complete workflow validation:
- Full project ingestion
- Mixed format processing
- Multi-file batch operations

## Running Tests

### Run All Tests
```bash
cd src/rust/daemon/core
cargo test
```

### Run Specific Test Module
```bash
cargo test --test file_ingestion_comprehensive_tests
```

### Run Specific Test Category
```bash
cargo test file_ingestion_comprehensive_tests::format_ingestion
cargo test file_ingestion_comprehensive_tests::code_ingestion
cargo test file_ingestion_comprehensive_tests::edge_cases
```

### Run Property-Based Tests
```bash
cargo test file_ingestion_comprehensive_tests::property_based
```

### Run Stress Tests (Expensive)
```bash
cargo test --test file_ingestion_comprehensive_tests stress_tests -- --ignored
```

### Run with Output
```bash
cargo test -- --nocapture
```

### Run Single Test
```bash
cargo test file_ingestion_comprehensive_tests::format_ingestion::test_markdown_ingestion_with_metadata
```

## Benchmarks

### Run All Benchmarks
```bash
cargo bench --bench file_ingestion_benchmarks
```

### Run Specific Benchmark
```bash
cargo bench --bench file_ingestion_benchmarks -- bench_markdown_file_creation
cargo bench --bench file_ingestion_benchmarks -- bench_large_file_creation
```

### Benchmark Categories

1. **File Creation Benchmarks**
   - Markdown file creation
   - Python code file creation
   - Rust code file creation

2. **Large File Benchmarks**
   - Variable sizes: 10KB, 50KB, 100KB, 500KB, 1MB
   - Throughput measurement in bytes/second

3. **Project Creation Benchmark**
   - Complete project structure generation
   - Multiple file formats

4. **Embedding Benchmarks**
   - Single embedding generation
   - Batch embedding generation (10-500 embeddings)

5. **Concurrency Benchmarks**
   - Concurrent file operations (2-16 parallel)
   - Tokio async runtime performance

## Test Dependencies

The test suite relies on:

- `shared-test-utils` - Common test fixtures and helpers
- `proptest` - Property-based testing framework
- `criterion` - Benchmark framework
- `tempfile` - Temporary file creation
- `tokio-test` - Async testing utilities
- `serial_test` - Sequential test execution
- `testcontainers` - Container-based testing (for future Qdrant tests)

## Test Fixtures

Located in `shared-test-utils/src/fixtures.rs`:

- `DocumentFixtures::markdown_content()` - Sample Markdown
- `DocumentFixtures::python_content()` - Sample Python code
- `DocumentFixtures::rust_content()` - Sample Rust code
- `DocumentFixtures::json_config()` - Sample JSON configuration
- `TempFileFixtures::create_temp_file()` - Temporary file helper
- `TempFileFixtures::create_temp_project()` - Project structure
- `TempFileFixtures::create_large_temp_file()` - Large file generator
- `EmbeddingFixtures::random_embedding()` - Test embeddings

## Continuous Integration

The test suite integrates with CI/CD:

```bash
# Fast tests for PR validation
cargo test --test file_ingestion_comprehensive_tests

# Comprehensive tests for main branch
cargo test --test file_ingestion_comprehensive_tests -- --include-ignored

# Performance regression detection
cargo bench --bench file_ingestion_benchmarks
```

## Future Enhancements

Planned improvements:

1. **Real Qdrant Integration**
   - Use testcontainers-rs for isolated Qdrant instances
   - Test end-to-end embedding storage and retrieval
   - Validate collection creation and management

2. **LSP Server Integration**
   - Test real LSP server connections
   - Validate code analysis results
   - Test symbol extraction and indexing

3. **Additional File Formats**
   - RTF document parsing
   - ODT (LibreOffice) support
   - More code language parsers (C++, TypeScript, etc.)

4. **Performance Profiling**
   - Memory usage tracking
   - CPU profiling integration
   - I/O bottleneck identification

5. **Chaos Testing**
   - Random file corruption
   - Network failure simulation
   - Resource exhaustion scenarios

## Contributing

When adding new tests:

1. Follow existing test organization patterns
2. Use descriptive test names (`test_verb_noun_condition`)
3. Include docstrings explaining test purpose
4. Add appropriate test attributes (`#[tokio::test]`, `#[ignore]`, etc.)
5. Use fixtures from `shared-test-utils` when possible
6. Document expected behavior and edge cases
7. Run `cargo fmt` and `cargo clippy` before committing

## Known Issues

1. **Library Compilation Errors**: The core library currently has some compilation issues (`PipelineStats` missing). These are pre-existing and being addressed in parallel.

2. **PDF Extraction**: PDF text extraction is not yet implemented. Tests use placeholder implementation.

3. **Testcontainers**: Currently disabled due to dependency conflicts. Will be re-enabled when resolved.

## Resources

- [Proptest Documentation](https://docs.rs/proptest/)
- [Criterion Documentation](https://docs.rs/criterion/)
- [Tokio Test Documentation](https://docs.rs/tokio-test/)
- [Rust Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)