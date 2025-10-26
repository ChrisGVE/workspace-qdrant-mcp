# Testing Architecture

Comprehensive testing documentation for workspace-qdrant-mcp covering test organization, frameworks, execution, and contribution guidelines.

## Table of Contents

1. [Test Architecture Overview](#test-architecture-overview)
2. [Test Directory Structure](#test-directory-structure)
3. [Testing Frameworks](#testing-frameworks)
4. [Test Categories and Markers](#test-categories-and-markers)
5. [Fixture Organization](#fixture-organization)
6. [Mock Strategies](#mock-strategies)
7. [Test Data Management](#test-data-management)
8. [Running Tests](#running-tests)
9. [Adding New Tests](#adding-new-tests)
10. [Debugging Test Failures](#debugging-test-failures)

---

## Test Architecture Overview

workspace-qdrant-mcp employs a comprehensive multi-layered testing strategy that ensures reliability across:

- **Python components** (MCP server, CLI, core library)
- **Rust daemon** (file watching, document processing, gRPC services)
- **Integration points** (Python ↔ Rust, Client ↔ Server, Daemon ↔ Qdrant)
- **External dependencies** (Qdrant vector database, LSP services)

### Testing Pyramid

```
                    ┌─────────────┐
                    │     E2E     │  ← Full system tests (slowest, highest value)
                    │   Tests     │
                    └─────────────┘
                  ┌─────────────────┐
                  │  Integration    │  ← Component interaction tests
                  │     Tests       │
                  └─────────────────┘
              ┌───────────────────────┐
              │   Functional Tests    │  ← Feature-level tests
              └───────────────────────┘
          ┌─────────────────────────────┐
          │      Unit Tests             │  ← Component isolation tests
          └─────────────────────────────┘
```

### Test Execution Flow

```
pytest invocation
    │
    ├─> conftest.py (root) - Environment setup, marker registration
    │       │
    │       ├─> tests/shared/fixtures.py - Common fixtures
    │       ├─> Domain-specific conftest.py files
    │       └─> Test collection and marker application
    │
    ├─> Test Discovery (by markers, paths, patterns)
    │
    ├─> Test Execution
    │       │
    │       ├─> Setup fixtures (session → module → function scope)
    │       ├─> Run test
    │       └─> Teardown fixtures
    │
    └─> Result Reporting (coverage, benchmarks, reports)
```

---

## Test Directory Structure

### Python Tests (`tests/`)

```
tests/
├── conftest.py                    # Root pytest configuration
├── shared/                        # Shared test utilities
│   ├── fixtures.py               # Common fixtures
│   ├── test_data.py             # Sample data generators
│   └── testcontainers_utils.py  # Container management
│
├── unit/                          # Component isolation tests
│   ├── conftest.py
│   ├── core/                     # Core library tests
│   ├── mcp/                      # MCP server tests
│   ├── memory/                   # Memory management tests
│   └── utils/                    # Utility tests
│
├── integration/                   # Cross-component tests
│   ├── conftest.py
│   ├── test_server_integration.py
│   ├── test_daemon_integration.py
│   └── test_phase1_protocol_validation.py
│
├── functional/                    # Feature-level tests
│   ├── conftest.py
│   ├── test_search_workflow.py
│   ├── test_recall_precision.py
│   └── test_git_integration.py
│
├── e2e/                          # End-to-end system tests
│   ├── conftest.py
│   ├── features/                # BDD-style feature files
│   └── steps/                   # Step definitions
│
├── cli/                          # CLI command tests
│   ├── conftest.py
│   ├── nominal/                 # Happy path tests
│   ├── edge/                    # Error handling tests
│   └── stress/                  # Load tests
│
├── mcp_server/                   # MCP server-specific tests
│   ├── conftest.py
│   ├── nominal/                 # Tool success tests
│   ├── edge/                    # Tool error tests
│   └── stress/                  # Performance tests
│
├── daemon/                       # Daemon-specific tests
│   ├── conftest.py
│   └── test_daemon_lifecycle.py
│
├── performance/                  # Performance benchmarks
│   ├── conftest.py
│   ├── test_search_benchmarks.py
│   └── k6/                      # k6 load test scripts
│
├── security/                     # Security tests
│   ├── conftest.py
│   ├── test_secrets_management.py
│   ├── test_injection_prevention.py
│   └── fixtures/                # Security test data
│
├── stress/                       # Stress and load tests
│   ├── conftest.py
│   └── test_concurrent_operations.py
│
├── compatibility/                # Cross-platform/version tests
│   ├── test_os_platforms.py
│   ├── test_python_versions.py
│   └── test_qdrant_versions.py
│
├── mocks/                        # Mock implementations
│   ├── qdrant_mocks.py
│   ├── lsp_mocks.py
│   ├── grpc_mocks.py
│   └── embedding_mocks.py
│
└── utils/                        # Test utilities
    └── test_helpers.py
```

### Rust Tests (`src/rust/daemon/`)

```
src/rust/daemon/
├── core/
│   ├── tests/                    # Integration tests
│   │   ├── mod.rs
│   │   ├── async_unit_tests.rs
│   │   ├── embedding_tests.rs
│   │   ├── file_ingestion_comprehensive_tests.rs
│   │   ├── hybrid_search_comprehensive_tests.rs
│   │   ├── stress_tests.rs
│   │   └── valgrind_memory_tests.rs
│   └── src/
│       └── */tests.rs            # Unit tests (in-module)
│
├── grpc/
│   ├── tests/                    # gRPC integration tests
│   └── src/
│       └── */tests.rs
│
└── benches/                      # Criterion benchmarks
    ├── search_benchmark.rs
    └── ingestion_benchmark.rs
```

### Directory Purpose Guide

| Directory | Purpose | Test Speed | Example |
|-----------|---------|-----------|---------|
| `unit/` | Component isolation | Fast | Test HybridSearch class methods |
| `integration/` | Cross-component | Medium | Test MCP server → Qdrant flow |
| `functional/` | Feature workflows | Medium | Test complete search feature |
| `e2e/` | Full system | Slow | Test CLI → Daemon → MCP → Qdrant |
| `performance/` | Benchmarks | Slow | Measure search latency |
| `stress/` | Load testing | Very Slow | 1000 concurrent searches |
| `security/` | Security validation | Fast | Test SQL injection prevention |
| `compatibility/` | Cross-platform | Slow | Test on Ubuntu/macOS/Windows |

---

## Testing Frameworks

### Python Testing Stack

#### pytest (Primary Framework)

- **Version**: Latest stable
- **Configuration**: `pyproject.toml` → `[tool.pytest.ini_options]`
- **Key Features**:
  - Automatic test discovery
  - Fixture-based test organization
  - Parametrized testing
  - Marker-based test selection
  - Coverage reporting integration
  - Async test support (`asyncio_mode = "auto"`)

**Example Usage:**

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest -m nominal

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run integration tests only
uv run pytest tests/integration
```

#### testcontainers-python

- **Purpose**: Manage isolated Qdrant containers for testing
- **Implementation**: `tests/shared/testcontainers_utils.py`
- **Container Types**:
  - **Isolated Containers**: Fresh container per test (slow, complete isolation)
  - **Shared Containers**: Reused across tests (fast, reset between tests)

**Example Fixture:**

```python
@pytest.fixture
async def isolated_qdrant_container():
    """Provide isolated Qdrant container for individual test."""
    container = IsolatedQdrantContainer()
    container.start()
    yield container
    container.stop()
```

#### Coverage.py

- **Purpose**: Code coverage measurement
- **Configuration**: `.coveragerc`
- **Integration**: `pytest-cov` plugin
- **Target**: 80%+ coverage across all components

**Example:**

```bash
# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

#### pytest-benchmark

- **Purpose**: Performance regression testing
- **Usage**: Measure operation latency and throughput
- **Output**: Statistical analysis with comparison to baselines

**Example:**

```python
def test_search_performance(benchmark):
    result = benchmark(perform_search, query="test")
    assert result is not None
```

### Rust Testing Stack

#### Built-in Test Framework

- **Test Types**:
  - **Unit Tests**: In-module `#[cfg(test)]` blocks
  - **Integration Tests**: `tests/` directory
  - **Documentation Tests**: Code in doc comments

**Example Unit Test:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_processing() {
        let processor = DocumentProcessor::new();
        let result = processor.process("test content");
        assert!(result.is_ok());
    }
}
```

**Example Integration Test:**

```rust
// tests/file_ingestion_tests.rs
#[tokio::test]
async fn test_end_to_end_ingestion() {
    let daemon = setup_test_daemon().await;
    daemon.ingest_file("test.py").await.unwrap();
    // Assertions...
}
```

#### Criterion.rs (Benchmarks)

- **Purpose**: Statistical benchmarking with regression detection
- **Location**: `benches/`
- **Output**: HTML reports with charts and statistical analysis

**Example:**

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_search(c: &mut Criterion) {
    c.bench_function("hybrid_search", |b| {
        b.iter(|| perform_search(black_box("query")))
    });
}

criterion_group!(benches, benchmark_search);
criterion_main!(benches);
```

**Run benchmarks:**

```bash
cd src/rust/daemon
cargo bench
```

---

## Test Categories and Markers

### Pytest Markers

Markers are defined in `pyproject.toml` and applied automatically by `conftest.py` based on:
- Test file location (directory-based)
- Test name patterns
- Explicit marker decorators

#### Core Markers

| Marker | Purpose | Auto-Applied When | Example |
|--------|---------|-------------------|---------|
| `@pytest.mark.nominal` | Happy path tests | File in `nominal/` dir | Basic search success |
| `@pytest.mark.edge` | Edge cases/errors | File in `edge/` dir or test name contains "edge"/"invalid"/"error" | Empty query handling |
| `@pytest.mark.stress` | Load/stress tests | File in `stress/` dir or name contains "stress"/"load" | 1000 concurrent requests |
| `@pytest.mark.slow` | Time-intensive tests | Applied with `stress` marker | Full database scan |
| `@pytest.mark.integration` | Cross-component tests | Explicit only | MCP server + Qdrant |
| `@pytest.mark.requires_qdrant` | Needs Qdrant server | Name contains "container"/"qdrant" | Search integration test |
| `@pytest.mark.requires_git` | Needs Git repo | Explicit only | Project detection test |
| `@pytest.mark.requires_rust` | Needs Rust daemon | Name contains "daemon"/"grpc" | Daemon lifecycle test |
| `@pytest.mark.performance` | Performance benchmarks | Explicit only | Search latency test |
| `@pytest.mark.security` | Security tests | Explicit only | Injection prevention |

#### Domain Markers (Auto-Applied)

Based on test file location in top-level directories:

| Marker | Applied To | Example |
|--------|-----------|---------|
| `@pytest.mark.daemon` | `tests/daemon/` | `test_daemon_lifecycle.py` |
| `@pytest.mark.mcp_server` | `tests/mcp_server/` | `test_server_tools.py` |
| `@pytest.mark.cli` | `tests/cli/` | `test_add_command.py` |

#### Security Markers

| Marker | Test Type | Example |
|--------|-----------|---------|
| `@pytest.mark.sql_injection` | SQL injection prevention | Database query sanitization |
| `@pytest.mark.path_traversal` | Path traversal prevention | File access validation |
| `@pytest.mark.command_injection` | Command injection prevention | Shell command sanitization |
| `@pytest.mark.xss` | XSS prevention | Output sanitization |
| `@pytest.mark.redos` | ReDoS prevention | Regex validation |

### Test Selection Examples

```bash
# Run only nominal tests (fast, happy path)
uv run pytest -m nominal

# Run edge cases only
uv run pytest -m edge

# Skip slow tests
uv run pytest -m "not slow"

# Run security tests only
uv run pytest -m security

# Run MCP server nominal tests
uv run pytest -m "mcp_server and nominal"

# Run integration tests requiring Qdrant
uv run pytest -m "integration and requires_qdrant"

# Run all tests except stress tests
uv run pytest -m "not stress"
```

### Marker Auto-Application Logic

The `pytest_collection_modifyitems` hook in `tests/conftest.py` automatically applies markers:

```python
def pytest_collection_modifyitems(config, items):
    """Modify test collection to apply markers automatically."""
    for item in items:
        test_path = Path(item.fspath).relative_to(tests_root)
        parts = test_path.parts

        # Domain markers from directory
        if parts[0] == "daemon":
            item.add_marker(pytest.mark.daemon)

        # Category markers from subdirectory
        if len(parts) > 1 and parts[1] == "nominal":
            item.add_marker(pytest.mark.nominal)

        # Name-based markers
        if "stress" in item.name.lower():
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.slow)
```

---

## Fixture Organization

### Fixture Hierarchy

```
Session-scoped fixtures (setup once)
    │
    ├─> event_loop (async support)
    ├─> shared_qdrant_container (reusable container)
    └─> cleanup_test_containers (teardown)
    │
Module-scoped fixtures (per file)
    │
    └─> (Domain-specific fixtures)
    │
Function-scoped fixtures (per test)
    │
    ├─> isolated_qdrant_container
    ├─> test_data_generator
    ├─> sample_documents
    ├─> temp_test_workspace
    └─> cleanup_collections
```

### Core Fixtures Reference

#### Environment Fixtures

| Fixture | Scope | Purpose | Usage |
|---------|-------|---------|-------|
| `project_root` | function | Path to project root | `tests/../` |
| `tests_root` | function | Path to tests directory | `tests/` |
| `src_root` | function | Path to source directory | `src/python/` |
| `environment_vars` | function | Test environment variables | Override config |

#### Container Fixtures

| Fixture | Scope | Purpose | Usage |
|---------|-------|---------|-------|
| `isolated_qdrant_container` | function | Fresh Qdrant container per test | Complete isolation |
| `shared_qdrant_container` | session | Shared Qdrant container | Fast integration tests |
| `cleanup_test_containers` | session | Auto-cleanup at session end | Prevent container leaks |

#### Test Data Fixtures

| Fixture | Scope | Purpose | Returns |
|---------|-------|---------|---------|
| `test_data_generator` | function | Generate test data | `TestDataGenerator` instance |
| `sample_documents` | function | Sample documents | List of document dicts |
| `sample_queries` | function | Sample search queries | List of query strings |
| `edge_case_documents` | function | Edge case documents | List with edge cases |
| `test_embeddings` | function | Sample embedding vectors | List of 384-dim vectors |

#### Workspace Fixtures

| Fixture | Scope | Purpose | Returns |
|---------|-------|---------|---------|
| `temp_test_workspace` | function | Temporary project structure | Path to workspace |
| `mock_git_repo` | function | Mock Git repository | Path to mock repo |

#### Configuration Fixtures

| Fixture | Scope | Purpose | Returns |
|---------|-------|---------|---------|
| `performance_thresholds` | function | Performance expectations | Dict of operation → max ms |
| `test_timeout_config` | function | Timeout configuration | Dict of operation → timeout |
| `embedding_dimensions` | function | Standard embedding dimensions | 384 (MiniLM-L6-v2) |

### Fixture Example Usage

```python
def test_document_storage(
    isolated_qdrant_container,
    sample_documents,
    test_collection_name,
    performance_thresholds
):
    """Test document storage with performance validation."""
    import time

    # Use container URL
    qdrant_url = isolated_qdrant_container.get_http_url()

    # Store documents
    start = time.time()
    for doc in sample_documents:
        store_document(qdrant_url, test_collection_name, doc)
    duration_ms = (time.time() - start) * 1000

    # Verify performance
    assert duration_ms < performance_thresholds["document_storage"] * len(sample_documents)

    # Verify storage
    results = search_documents(qdrant_url, test_collection_name, sample_documents[0]["content"])
    assert len(results) > 0
```

### Creating Custom Fixtures

Add domain-specific fixtures in `tests/{domain}/conftest.py`:

```python
# tests/mcp_server/conftest.py
import pytest

@pytest.fixture
def mcp_client():
    """Provide MCP client for server tests."""
    from workspace_qdrant_mcp.server import app
    client = app.test_client()
    yield client
    client.close()

@pytest.fixture
def mock_fastembed():
    """Mock FastEmbed for unit tests."""
    from tests.mocks.embedding_mocks import MockFastEmbed
    return MockFastEmbed()
```

---

## Mock Strategies

### Qdrant Isolation Strategies

#### 1. Isolated Containers (Recommended for Unit/Functional Tests)

**Purpose**: Complete isolation with fresh Qdrant instance per test

**Usage:**

```python
@pytest.fixture
async def isolated_qdrant_container():
    """Fresh Qdrant container per test."""
    container = IsolatedQdrantContainer()
    container.start()
    yield container
    container.stop()

def test_with_isolation(isolated_qdrant_container):
    url = isolated_qdrant_container.get_http_url()
    # Test has exclusive access to Qdrant
```

**Pros:**
- Complete isolation (no cross-test pollution)
- Safe for parallel test execution
- Predictable state

**Cons:**
- Slower (container startup overhead ~2-5 seconds)
- Higher resource usage

#### 2. Shared Containers (Recommended for Integration Tests)

**Purpose**: Reuse Qdrant container across tests with reset between tests

**Usage:**

```python
@pytest.fixture(scope="session")
async def shared_qdrant_container():
    """Shared container with reset between tests."""
    container = await get_shared_qdrant_container()
    yield container
    await release_shared_qdrant_container(reset=True)

def test_with_shared(shared_qdrant_container):
    url = shared_qdrant_container.get_http_url()
    # Test uses shared Qdrant (faster startup)
```

**Pros:**
- Faster test execution (no container startup)
- Lower resource usage
- Good for integration test suites

**Cons:**
- Risk of cross-test pollution if reset fails
- Must be careful with test ordering
- Not safe for parallel execution

#### 3. Mock Qdrant (Recommended for Pure Unit Tests)

**Purpose**: In-memory mock for testing without real Qdrant

**Usage:**

```python
from tests.mocks.qdrant_mocks import MockQdrantClient

def test_with_mock():
    mock_client = MockQdrantClient()
    # Simulate Qdrant responses without running server
    result = mock_client.search(...)
    assert result == expected_mock_response
```

**Implementation:** See `tests/mocks/qdrant_mocks.py`

**Pros:**
- Extremely fast (no I/O)
- No external dependencies
- Perfect for unit testing logic

**Cons:**
- Doesn't validate actual Qdrant behavior
- Must maintain mock implementation
- Can drift from real API

### Other Mock Implementations

#### LSP Mocks (`tests/mocks/lsp_mocks.py`)

Mock LSP server responses for testing code intelligence features:

```python
from tests.mocks.lsp_mocks import MockLSPServer

def test_lsp_integration():
    lsp = MockLSPServer()
    symbols = lsp.get_symbols("test.py")
    assert len(symbols) > 0
```

#### gRPC Mocks (`tests/mocks/grpc_mocks.py`)

Mock gRPC daemon communication:

```python
from tests.mocks.grpc_mocks import MockDaemonClient

def test_daemon_communication():
    daemon = MockDaemonClient()
    response = daemon.ingest_text("content")
    assert response.success
```

#### Embedding Mocks (`tests/mocks/embedding_mocks.py`)

Mock FastEmbed for testing without model loading:

```python
from tests.mocks.embedding_mocks import MockFastEmbed

def test_embedding_generation():
    embedder = MockFastEmbed()
    vectors = embedder.embed(["test text"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384  # MiniLM-L6-v2 dimensions
```

---

## Test Data Management

### SampleData Class

Provides predefined test data for consistent testing:

```python
from tests.shared.test_data import SampleData

# Get sample documents
docs = SampleData.get_sample_documents()
# Returns: [
#   {"content": "Python is great", "metadata": {...}},
#   {"content": "Rust is fast", "metadata": {...}},
#   ...
# ]

# Get sample queries
queries = SampleData.get_sample_queries()
# Returns: ["python programming", "rust performance", ...]

# Get edge cases
edge_cases = SampleData.get_edge_case_documents()
# Returns: [
#   {"content": "", "metadata": {}},  # Empty
#   {"content": "x" * 100000, ...},   # Very long
#   ...
# ]
```

### TestDataGenerator Class

Generates dynamic test data:

```python
from tests.shared.test_data import TestDataGenerator

generator = TestDataGenerator()

# Generate unique collection name
collection = generator.generate_collection_name("test")
# Returns: "test_abc123def456"

# Generate random embedding vector
embedding = generator.generate_embedding(dimensions=384)
# Returns: [0.123, -0.456, ...] (384 values)

# Generate random document
doc = generator.generate_document()
# Returns: {"content": "...", "metadata": {...}}
```

### Test Data Best Practices

1. **Use Sample Data for Consistency**: Use `SampleData` when testing known behavior
2. **Use Generator for Uniqueness**: Use `TestDataGenerator` when tests need unique data
3. **Include Edge Cases**: Always test with `edge_case_documents()` for robustness
4. **Clean Up**: Use `cleanup_collections` fixture to remove test data after tests
5. **Isolate**: Use unique collection names to prevent cross-test pollution

**Example:**

```python
def test_search_recall(
    isolated_qdrant_container,
    sample_documents,
    sample_queries,
    test_collection_name,
    cleanup_collections
):
    """Test search recall with sample data."""
    # Register collection for cleanup
    cleanup_collections(test_collection_name)

    # Store sample documents
    for doc in sample_documents:
        store(test_collection_name, doc)

    # Test search recall
    for query in sample_queries:
        results = search(test_collection_name, query)
        assert len(results) > 0
```

---

## Running Tests

### Basic Test Execution

```bash
# All tests
uv run pytest

# Specific directory
uv run pytest tests/unit
uv run pytest tests/integration

# Specific file
uv run pytest tests/unit/test_hybrid_search.py

# Specific test function
uv run pytest tests/unit/test_hybrid_search.py::test_search_basic
```

### Test Selection by Markers

```bash
# Run nominal tests only (fast)
uv run pytest -m nominal

# Run integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"

# Run security tests
uv run pytest -m security

# Combined markers
uv run pytest -m "mcp_server and nominal"
```

### Coverage Analysis

```bash
# Run with coverage
uv run pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Terminal coverage report
uv run pytest --cov=src --cov-report=term-missing
```

### Performance Benchmarks

```bash
# Run benchmark tests only
uv run pytest --benchmark-only

# Run all tests including benchmarks
uv run pytest

# Save benchmark results
uv run pytest --benchmark-save=baseline

# Compare to baseline
uv run pytest --benchmark-compare=baseline
```

### Verbose and Debug Output

```bash
# Verbose output
uv run pytest -v

# Extra verbose with test output
uv run pytest -vv

# Show print statements
uv run pytest -s

# Show local variables on failure
uv run pytest -l

# Drop into debugger on failure
uv run pytest --pdb
```

### Parallel Execution

```bash
# Install pytest-xdist
uv add --dev pytest-xdist

# Run tests in parallel (4 workers)
uv run pytest -n 4

# Auto-detect CPU count
uv run pytest -n auto
```

### Rust Tests

```bash
# Run all Rust tests
cd src/rust/daemon
cargo test

# Run specific test
cargo test test_file_watching

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

---

## Adding New Tests

### Step 1: Determine Test Category

Choose the appropriate directory based on test type:

| Test Type | Directory | Example |
|-----------|-----------|---------|
| Component isolation | `tests/unit/` | Test `HybridSearch` class |
| Cross-component interaction | `tests/integration/` | Test MCP server → Qdrant |
| Feature workflow | `tests/functional/` | Test complete search feature |
| Full system | `tests/e2e/` | Test CLI → Daemon → Qdrant |
| Performance | `tests/performance/` | Benchmark search latency |
| Security | `tests/security/` | Test injection prevention |

### Step 2: Choose Test Category Subdirectory

For domain-specific tests (cli/, mcp_server/, daemon/), add to:

| Category | Subdirectory | Purpose |
|----------|-------------|---------|
| Happy path | `nominal/` | Success cases |
| Error handling | `edge/` | Edge cases, errors |
| Load testing | `stress/` | Performance under load |

### Step 3: Create Test File

**Naming Convention:** `test_{feature}.py`

**Template:**

```python
"""
Tests for {feature description}.

Test categories:
- Nominal: {happy path scenarios}
- Edge: {error cases}
- Stress: {load scenarios}
"""

import pytest


class TestFeatureNominal:
    """Nominal/happy path tests for {feature}."""

    def test_basic_usage(self, required_fixtures):
        """Test basic {feature} usage."""
        # Arrange
        input_data = setup_test_data()

        # Act
        result = perform_operation(input_data)

        # Assert
        assert result.success
        assert result.data == expected_output


class TestFeatureEdge:
    """Edge case tests for {feature}."""

    def test_empty_input(self):
        """Test {feature} with empty input."""
        with pytest.raises(ValueError):
            perform_operation("")

    def test_invalid_input(self):
        """Test {feature} with invalid input."""
        result = perform_operation(invalid_data)
        assert not result.success


@pytest.mark.slow
class TestFeatureStress:
    """Stress tests for {feature}."""

    def test_high_load(self, performance_thresholds):
        """Test {feature} under high load."""
        # Stress test implementation
        pass
```

### Step 4: Add Appropriate Fixtures

Use fixtures from `tests/shared/fixtures.py` or create domain-specific fixtures in `tests/{domain}/conftest.py`.

**Example custom fixture:**

```python
# tests/mcp_server/conftest.py
import pytest

@pytest.fixture
def mcp_tool_context():
    """Provide MCP tool execution context."""
    from workspace_qdrant_mcp.server import create_tool_context
    context = create_tool_context()
    yield context
    context.cleanup()
```

### Step 5: Add Markers

Add explicit markers if needed:

```python
@pytest.mark.requires_qdrant
@pytest.mark.integration
async def test_qdrant_integration(isolated_qdrant_container):
    """Test Qdrant integration."""
    # Test implementation
```

### Step 6: Run and Verify

```bash
# Run new test
uv run pytest tests/unit/test_new_feature.py -v

# Run with coverage
uv run pytest tests/unit/test_new_feature.py --cov=src.module --cov-report=term-missing

# Verify markers applied correctly
uv run pytest --collect-only tests/unit/test_new_feature.py
```

---

## Debugging Test Failures

### Common Debugging Techniques

#### 1. Verbose Output

```bash
# Show test names and results
uv run pytest -v

# Show all output (including print statements)
uv run pytest -s

# Show local variables on failure
uv run pytest -l

# Extra verbose
uv run pytest -vv
```

#### 2. Drop into Debugger

```bash
# Drop into pdb on first failure
uv run pytest --pdb

# Drop into pdb on specific test
uv run pytest tests/unit/test_foo.py::test_bar --pdb
```

#### 3. Run Single Test

```bash
# Run specific test to isolate issue
uv run pytest tests/unit/test_foo.py::test_bar -v
```

#### 4. Check Fixtures

```bash
# Show fixture setup
uv run pytest --setup-show tests/unit/test_foo.py::test_bar
```

### Common Test Failures

#### 1. Container Startup Failures

**Symptom:** `testcontainers` fails to start Qdrant

**Solutions:**

```bash
# Check Docker is running
docker ps

# Check Docker has resources
docker system info

# Clean up old containers
docker system prune

# Manually test container
docker run -p 6333:6333 qdrant/qdrant:latest
```

#### 2. Import Errors

**Symptom:** `ModuleNotFoundError`

**Solutions:**

```bash
# Verify src in path (check conftest.py)
# Reinstall dependencies
uv sync --dev

# Check import paths
python -c "import sys; print('\n'.join(sys.path))"
```

#### 3. Async Test Failures

**Symptom:** `RuntimeError: This event loop is already running`

**Solutions:**

```python
# Use pytest-asyncio with auto mode (in pyproject.toml)
[tool.pytest.ini_options]
asyncio_mode = "auto"

# Or use async fixtures properly
@pytest.fixture
async def async_setup():
    await some_async_operation()
```

#### 4. Fixture Not Found

**Symptom:** `fixture 'foo' not found`

**Solutions:**

```python
# Ensure fixture is imported
pytest_plugins = [
    "tests.shared.fixtures",
]

# Or add to conftest.py in test directory
# Check fixture scope matches test scope
```

#### 5. Flaky Tests

**Symptom:** Test passes sometimes, fails others

**Common Causes:**

- Timing issues (race conditions)
- Shared state between tests
- External dependencies (network, containers)

**Solutions:**

```python
# Add retries for flaky tests
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_flaky_operation():
    # Test implementation

# Add explicit waits
import time
time.sleep(1)  # Wait for async operation

# Use isolated fixtures instead of shared
# Add proper cleanup
```

### Debugging Qdrant Integration

```bash
# Check Qdrant logs in container
docker logs <container_id>

# Access Qdrant directly
curl http://localhost:6333/health

# Check collections
curl http://localhost:6333/collections
```

### Debugging Rust Tests

```bash
# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_name -- --exact

# Show backtrace on panic
RUST_BACKTRACE=1 cargo test
```

---

## Appendix: Testing Tools Reference

### Python Testing Tools

| Tool | Purpose | Documentation |
|------|---------|--------------|
| pytest | Test framework | https://pytest.org |
| pytest-cov | Coverage plugin | https://pytest-cov.readthedocs.io |
| pytest-asyncio | Async test support | https://pytest-asyncio.readthedocs.io |
| pytest-benchmark | Benchmarking | https://pytest-benchmark.readthedocs.io |
| testcontainers | Docker container management | https://testcontainers-python.readthedocs.io |
| coverage.py | Coverage measurement | https://coverage.readthedocs.io |

### Rust Testing Tools

| Tool | Purpose | Documentation |
|------|---------|--------------|
| cargo test | Built-in test runner | https://doc.rust-lang.org/book/ch11-00-testing.html |
| criterion | Benchmarking | https://bheisler.github.io/criterion.rs |
| tarpaulin | Code coverage | https://github.com/xd009642/tarpaulin |

### Continuous Integration

Tests are automatically run on:
- Every pull request
- Every commit to main branch
- Nightly builds

See `.github/workflows/` for CI configuration.

---

**Last Updated:** 2025-10-26
**Maintained By:** workspace-qdrant-mcp development team
