# Testing Architecture

Comprehensive testing documentation for workspace-qdrant-mcp covering test organization, frameworks, execution, and contribution guidelines.

## Table of Contents

1. [Local Testing Setup](#local-testing-setup)
2. [Test Architecture Overview](#test-architecture-overview)
3. [Test Directory Structure](#test-directory-structure)
4. [Testing Frameworks](#testing-frameworks)
5. [Test Categories and Markers](#test-categories-and-markers)
6. [Fixture Organization](#fixture-organization)
7. [Mock Strategies](#mock-strategies)
8. [Test Data Management](#test-data-management)
9. [Running Tests](#running-tests)
10. [Adding New Tests](#adding-new-tests)
11. [Debugging Test Failures](#debugging-test-failures)

---

## Local Testing Setup

This guide walks you through setting up a complete testing environment for workspace-qdrant-mcp on your local machine.

### Prerequisites

#### System Requirements

- **Python**: 3.11 or higher
- **Rust**: Latest stable (1.70+)
- **Docker**: For testcontainers (Qdrant isolation)
- **Git**: For repository operations

**Verify installations:**

```bash
# Python
python --version  # Should be 3.11+

# Rust
rustc --version   # Should be 1.70+

# Docker
docker --version
docker ps         # Should connect without errors

# Git
git --version
```

#### Required Tools

Install `uv` (Python package manager):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

### Step 1: Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/yourusername/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Install Python dependencies (including dev dependencies)
uv sync --dev

# Build Rust daemon (optional, needed for daemon tests)
cd src/rust/daemon
cargo build --release
cd ../../..

# Verify installation
uv run pytest --version
uv run wqm --version
```

### Step 2: Start Qdrant Server

**Option 1: Docker (Recommended)**

```bash
# Start Qdrant server
docker run -d --name qdrant-test \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest

# Verify Qdrant is running
curl http://localhost:6333/health
# Expected: {"title":"HealthChecker","version":"...","status":"ok"}

# Stop when done
docker stop qdrant-test
docker rm qdrant-test
```

**Option 2: Testcontainers (Automatic)**

Tests using `isolated_qdrant_container` or `shared_qdrant_container` fixtures will automatically start Qdrant in Docker. No manual setup needed.

**Option 3: Local Binary**

```bash
# Download and run Qdrant binary
# See: https://qdrant.tech/documentation/quick-start/

./qdrant --config-path ./config.yaml
```

### Step 3: Configure Environment

Create `.env` file in project root:

```bash
# Qdrant connection
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your_api_key  # Only if using Qdrant Cloud

# Testing configuration
WQM_TEST_MODE=true
WQM_LOG_LEVEL=WARNING
FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
FASTEMBED_CACHE_DIR=~/.cache/fastembed_test

# Optional: Rust daemon configuration (for daemon tests)
WQM_DAEMON_PORT=50051
```

### Step 4: Verify Test Environment

Run smoke test to verify everything is set up correctly:

```bash
# Run basic unit tests (no external dependencies)
uv run pytest tests/unit/test_hybrid_search.py -v

# Expected output:
# ========= test session starts =========
# ...
# tests/unit/test_hybrid_search.py::TestHybridSearch::test_basic ✓
# ...
# ========= X passed in Y.YYs =========
```

If this passes, your environment is ready!

### Step 5: Run Full Test Suite

```bash
# Run all tests (may take several minutes)
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Common Setup Issues

#### Issue: Docker Not Running

**Symptom:**

```
Cannot connect to the Docker daemon
```

**Solution:**

```bash
# macOS: Start Docker Desktop
open -a Docker

# Linux: Start Docker service
sudo systemctl start docker

# Verify
docker ps
```

#### Issue: Port Already in Use

**Symptom:**

```
Bind for 0.0.0.0:6333 failed: port is already allocated
```

**Solution:**

```bash
# Find process using port 6333
lsof -i :6333

# Kill existing Qdrant container
docker ps | grep qdrant
docker stop <container_id>

# Or use different port
docker run -p 6334:6333 qdrant/qdrant:latest
# Update QDRANT_URL=http://localhost:6334
```

#### Issue: Python Module Not Found

**Symptom:**

```
ModuleNotFoundError: No module named 'workspace_qdrant_mcp'
```

**Solution:**

```bash
# Reinstall dependencies
uv sync --dev

# Verify src is in Python path
python -c "import sys; print(sys.path)"

# Check conftest.py adds src to path
cat tests/conftest.py | grep sys.path
```

#### Issue: Rust Build Fails

**Symptom:**

```
error: could not compile `workspace-qdrant-daemon`
```

**Solution:**

```bash
# Update Rust
rustup update stable

# Clean and rebuild
cd src/rust/daemon
cargo clean
cargo build --release

# Check dependencies
cargo check
```

#### Issue: Test Hanging

**Symptom:**

Test runs but never completes

**Solution:**

```bash
# Run with timeout
uv run pytest --timeout=30

# Check for deadlocks in async code
# Add debug output
uv run pytest -s -v

# Run single test to isolate
uv run pytest tests/unit/test_foo.py::test_bar -v
```

### Testing Different Components

#### Python Tests Only

```bash
# Unit tests (fast, no external dependencies)
uv run pytest tests/unit -v

# Integration tests (requires Qdrant)
uv run pytest tests/integration -v

# MCP server tests
uv run pytest tests/mcp_server -v

# CLI tests
uv run pytest tests/cli -v
```

#### Rust Tests Only

```bash
cd src/rust/daemon

# All Rust tests
cargo test

# Specific test module
cargo test file_watching

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'
```

#### Combined Tests (Full System)

```bash
# End-to-end tests
uv run pytest tests/e2e -v

# Functional tests
uv run pytest tests/functional -v
```

### IDE Integration

#### Visual Studio Code

Install recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "rust-lang.rust-analyzer",
    "tamasfe.even-better-toml"
  ]
}
```

Configure pytest in VS Code:

```json
// .vscode/settings.json
{
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

#### PyCharm

1. Open **Settings** → **Tools** → **Python Integrated Tools**
2. Set **Default test runner** to **pytest**
3. Set **Project interpreter** to uv-managed venv
4. Right-click test file → **Run 'pytest in test_foo.py'**

#### Rust Analyzer

Add to `Cargo.toml`:

```toml
[workspace]
members = [
    "common",
    "common-node",
    "cli",
    "daemon/core",
    "daemon/grpc",
    "daemon/memexd",
    "daemon/shared-test-utils",
]
```

Configure in editor:
- VS Code: Install `rust-analyzer` extension
- IntelliJ/CLion: Install Rust plugin

### Performance Benchmarking Setup

#### Python Benchmarks

```bash
# Run benchmarks only
uv run pytest --benchmark-only

# Save baseline
uv run pytest --benchmark-save=my_baseline

# Compare to baseline
uv run pytest --benchmark-compare=my_baseline

# Generate histogram
uv run pytest --benchmark-histogram
```

#### Rust Benchmarks

```bash
cd src/rust/daemon

# Run benchmarks with Criterion
cargo bench

# View HTML reports
open target/criterion/report/index.html

# Benchmark specific function
cargo bench search
```

### Continuous Testing Workflow

#### Watch Mode (Auto-run on file changes)

Install `pytest-watch`:

```bash
uv add --dev pytest-watch
```

Run in watch mode:

```bash
# Watch all tests
uv run ptw

# Watch specific directory
uv run ptw tests/unit

# Watch with coverage
uv run ptw -- --cov=src
```

#### Pre-commit Hooks

Install pre-commit:

```bash
uv add --dev pre-commit
pre-commit install
```

Configure `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-quick
        name: Run Quick Tests
        entry: uv run pytest -m "not slow"
        language: system
        pass_filenames: false
        always_run: true
```

Now tests run automatically before each commit.

### Next Steps

- Read [Test Architecture Overview](#test-architecture-overview) to understand test organization
- See [Running Tests](#running-tests) for detailed execution commands
- Check [Adding New Tests](#adding-new-tests) when contributing tests
- Refer to [Debugging Test Failures](#debugging-test-failures) when tests fail

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

This section provides comprehensive guidelines for contributing new test cases to workspace-qdrant-mcp, ensuring consistency, quality, and maintainability.

### When to Add Tests

#### 1. New Features

Always add tests when implementing new features:

**Required Test Types:**
- **Unit tests**: Core logic in isolation
- **Integration tests**: Feature with external dependencies
- **Functional tests**: Complete feature workflow
- **Documentation tests**: Examples in docstrings

**Example:**

```
New feature: Add BM25 scoring to hybrid search

Required tests:
✓ Unit: BM25 algorithm calculation
✓ Unit: Score normalization
✓ Integration: BM25 with Qdrant storage
✓ Functional: Hybrid search with BM25 weighting
✓ Documentation: BM25 usage examples in docstring
```

#### 2. Bug Fixes

Add regression tests for every bug fix:

**Test Requirements:**
1. Reproduce the bug (test should fail initially)
2. Apply fix
3. Verify test passes
4. Add edge cases around the bug

**Example:**

```
Bug: Empty query causes NullPointerError

Regression test:
def test_empty_query_handling():
    """Regression test for issue #123: empty query crashes."""
    result = search(collection="test", query="")
    assert result.success is False
    assert "query cannot be empty" in result.error_message
```

#### 3. Edge Cases

Test boundary conditions and error paths:

**Common Edge Cases:**
- Empty inputs (strings, lists, dicts)
- Null/None values
- Maximum size inputs (very long strings, large lists)
- Invalid types
- Concurrent access
- Resource exhaustion

**Example:**

```python
class TestSearchEdgeCases:
    """Edge case tests for search functionality."""

    def test_empty_query(self):
        """Test search with empty query string."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            search(collection="test", query="")

    def test_very_long_query(self):
        """Test search with extremely long query."""
        long_query = "a" * 100_000
        result = search(collection="test", query=long_query)
        # Should either succeed or fail gracefully
        assert result.success in [True, False]
        if not result.success:
            assert "query too long" in result.error_message.lower()

    def test_null_collection(self):
        """Test search with None collection."""
        with pytest.raises(TypeError):
            search(collection=None, query="test")

    def test_invalid_limit(self):
        """Test search with invalid limit parameter."""
        with pytest.raises(ValueError):
            search(collection="test", query="test", limit=-1)
```

#### 4. Performance Regressions

Add benchmark tests for performance-critical code:

**When to Add:**
- New algorithm implementations
- Bulk operations
- Search and retrieval paths
- I/O operations

**Example:**

```python
@pytest.mark.performance
def test_search_latency(benchmark, sample_documents):
    """Benchmark search latency for performance regression detection."""
    # Setup: Pre-populate collection
    setup_collection("perf_test", sample_documents)

    # Benchmark the search operation
    result = benchmark(search, collection="perf_test", query="test query")

    # Assert performance thresholds
    stats = benchmark.stats
    assert stats.mean < 0.5  # Average < 500ms
    assert stats.stddev < 0.1  # Low variance
```

### Test Contribution Workflow

```
1. Identify what needs testing
   ↓
2. Determine test category and location
   ↓
3. Write test following standards
   ↓
4. Run test locally and verify it passes
   ↓
5. Run full test suite to ensure no regressions
   ↓
6. Update documentation if needed
   ↓
7. Submit PR with test coverage evidence
   ↓
8. CI/CD automatically validates tests
   ↓
9. Code review focuses on test quality
   ↓
10. Merge after approval
```

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

### Step 7: Write Effective Assertions

Good assertions make tests valuable and maintainable.

#### Assertion Best Practices

**1. Be Specific**

```python
# ❌ Bad: Vague assertion
assert result

# ✅ Good: Specific assertion
assert result.success is True
assert result.data == expected_data
assert len(result.items) == 5
```

**2. Use Descriptive Messages**

```python
# ❌ Bad: No context on failure
assert len(results) > 0

# ✅ Good: Clear failure message
assert len(results) > 0, f"Expected search results, got empty list for query: {query}"
```

**3. Test One Thing Per Test**

```python
# ❌ Bad: Testing multiple unrelated things
def test_everything():
    assert feature_a_works()
    assert feature_b_works()
    assert feature_c_works()

# ✅ Good: Focused tests
def test_feature_a():
    assert feature_a_works()

def test_feature_b():
    assert feature_b_works()

def test_feature_c():
    assert feature_c_works()
```

**4. Use pytest's Rich Assertions**

```python
# ❌ Bad: Manual exception handling
def test_invalid_input():
    try:
        process_data(None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()

# ✅ Good: pytest.raises context manager
def test_invalid_input():
    with pytest.raises(ValueError, match="invalid"):
        process_data(None)
```

**5. Assert on Behavior, Not Implementation**

```python
# ❌ Bad: Testing implementation details
def test_cache_internal_structure():
    cache = Cache()
    cache.set("key", "value")
    assert cache._internal_dict["key"] == "value"  # Implementation detail

# ✅ Good: Testing behavior
def test_cache_stores_and_retrieves():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"  # Public behavior
```

### Step 8: Create Test Data Properly

Follow these guidelines for test data creation:

#### 1. Use Fixtures for Reusable Data

```python
@pytest.fixture
def sample_user_data():
    """Provide sample user data for tests."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "role": "admin",
    }

def test_user_creation(sample_user_data):
    user = create_user(**sample_user_data)
    assert user.username == sample_user_data["username"]
```

#### 2. Use Generators for Large Datasets

```python
from tests.shared.test_data import TestDataGenerator

def test_bulk_insert(test_data_generator):
    """Test bulk insert with generated data."""
    # Generate 1000 unique documents
    documents = [
        test_data_generator.generate_document()
        for _ in range(1000)
    ]

    # Test bulk insert
    result = bulk_insert(documents)
    assert result.inserted_count == 1000
```

#### 3. Avoid Hardcoded Test Data in Tests

```python
# ❌ Bad: Hardcoded data in test
def test_search():
    doc1 = {"id": "123", "content": "Python is great", "metadata": {...}}
    doc2 = {"id": "456", "content": "Rust is fast", "metadata": {...}}
    # ... more hardcoded data

# ✅ Good: Use fixtures or sample data
def test_search(sample_documents):
    # sample_documents comes from fixture
    for doc in sample_documents:
        store(doc)
```

#### 4. Use Parametrize for Multiple Test Cases

```python
@pytest.mark.parametrize("query,expected_count", [
    ("python", 5),
    ("rust", 3),
    ("javascript", 0),
    ("", 0),  # Edge case: empty query
])
def test_search_counts(query, expected_count, search_collection):
    """Test search returns correct count for various queries."""
    results = search(collection=search_collection, query=query)
    assert len(results) == expected_count
```

### Step 9: Integrate with CI/CD

All tests are automatically run in CI/CD pipelines on:

1. **Pull Request Creation**: Full test suite runs
2. **Commits to Main**: Full test suite + coverage report
3. **Nightly Builds**: Full test suite + benchmarks + stress tests
4. **Release Tags**: All tests + compatibility matrix

#### CI/CD Test Workflow

```yaml
# .github/workflows/test.yml (simplified)
name: Tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev
      - name: Run tests with coverage
        run: |
          uv run pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Rust
        uses: actions-rs/toolchain@v1
      - name: Run Rust tests
        run: |
          cd src/rust/daemon
          cargo test
```

#### Local Pre-CI Validation

Before submitting PR, run the same checks CI will run:

```bash
# Full Python test suite with coverage
uv run pytest --cov=src --cov-report=term-missing

# Rust tests
cd src/rust/daemon && cargo test

# Linting (CI runs these too)
uv run ruff check src tests
uv run black --check src tests
uv run mypy src

# Type checking
uv run mypy src/
```

### Step 10: PR Requirements for Tests

When submitting a pull request with new code, ensure:

#### Required for All PRs:

- [ ] **Test coverage**: New code has corresponding tests
- [ ] **All tests pass**: Run `uv run pytest` locally
- [ ] **No coverage regression**: Coverage doesn't decrease
- [ ] **Tests are documented**: Clear docstrings explaining what's tested

#### Required for Feature PRs:

- [ ] **Unit tests**: Core logic tested in isolation
- [ ] **Integration tests**: Feature tested with dependencies
- [ ] **Edge case tests**: Boundary conditions covered
- [ ] **Documentation updated**: README/docs reflect new feature

#### Required for Bug Fix PRs:

- [ ] **Regression test**: Test reproduces the bug (fails before fix)
- [ ] **Test passes after fix**: Verify fix resolves issue
- [ ] **Related edge cases**: Test similar boundary conditions
- [ ] **Issue reference**: Test references GitHub issue number

#### PR Template Checklist

When creating PR, use this checklist:

```markdown
## Testing Checklist

- [ ] Added unit tests for new functionality
- [ ] Added integration tests where appropriate
- [ ] Added edge case tests
- [ ] All tests pass locally (`uv run pytest`)
- [ ] Test coverage is maintained or improved
- [ ] Tests are well-documented with clear docstrings
- [ ] Ran linters (`ruff check`, `black --check`)
- [ ] Updated documentation if needed

## Test Coverage

<!-- Paste coverage report showing new code is tested -->

## Test Execution

<!-- Paste pytest output showing all tests pass -->
```

### Best Practices for Test Quality

#### 1. Fast Tests

Keep unit tests fast (<100ms each):

```python
# ❌ Slow: Unnecessary I/O
def test_calculation():
    save_to_file("temp.txt", data)  # Slow I/O
    result = calculate(data)
    assert result == expected

# ✅ Fast: No I/O
def test_calculation():
    result = calculate(data)
    assert result == expected
```

#### 2. Isolated Tests

Tests should not depend on each other:

```python
# ❌ Bad: Tests depend on order
def test_create_user():
    global test_user
    test_user = create_user("alice")

def test_get_user():
    # Depends on test_create_user running first
    user = get_user(test_user.id)

# ✅ Good: Each test is independent
@pytest.fixture
def created_user():
    user = create_user("alice")
    yield user
    delete_user(user.id)

def test_create_user(created_user):
    assert created_user.username == "alice"

def test_get_user(created_user):
    user = get_user(created_user.id)
    assert user.id == created_user.id
```

#### 3. Descriptive Test Names

```python
# ❌ Bad: Unclear what's tested
def test_search():
    pass

def test_search2():
    pass

# ✅ Good: Clear test purpose
def test_search_returns_relevant_results():
    pass

def test_search_handles_empty_query():
    pass

def test_search_respects_limit_parameter():
    pass
```

#### 4. Arrange-Act-Assert Pattern

```python
def test_user_creation():
    # Arrange: Set up test data
    username = "testuser"
    email = "test@example.com"

    # Act: Perform the operation
    user = create_user(username=username, email=email)

    # Assert: Verify the outcome
    assert user.username == username
    assert user.email == email
    assert user.id is not None
```

#### 5. Clean Up Resources

```python
@pytest.fixture
def temp_collection():
    """Create temporary collection and clean up after test."""
    collection_name = f"test_{uuid.uuid4()}"

    # Setup
    create_collection(collection_name)

    yield collection_name

    # Teardown
    delete_collection(collection_name)

def test_with_temp_collection(temp_collection):
    # Test uses collection
    # Automatic cleanup after test
    pass
```

### Examples of Well-Written Tests

#### Example 1: Unit Test

```python
"""
Unit test for HybridSearch class.

Tests search scoring logic in isolation without external dependencies.
"""
import pytest
from workspace_qdrant_mcp.core.hybrid_search import HybridSearch


class TestHybridSearchScoring:
    """Test hybrid search scoring algorithms."""

    def test_rrf_score_calculation(self):
        """Test Reciprocal Rank Fusion score calculation."""
        # Arrange
        dense_scores = [(0, 0.9), (1, 0.7), (2, 0.5)]
        sparse_scores = [(1, 0.8), (0, 0.6), (3, 0.4)]
        k = 60

        # Act
        hybrid_search = HybridSearch()
        combined_scores = hybrid_search.rrf_score(
            dense_scores, sparse_scores, k=k
        )

        # Assert
        # Document 0 appears in both, should have highest combined score
        # Document 1 appears in both, second highest
        # Documents 2 and 3 appear in one each
        assert len(combined_scores) == 4
        assert combined_scores[0][0] in [0, 1]  # Top result is 0 or 1

    def test_empty_dense_scores(self):
        """Test RRF when dense search returns no results."""
        # Arrange
        dense_scores = []
        sparse_scores = [(0, 0.9), (1, 0.8)]

        # Act
        hybrid_search = HybridSearch()
        combined_scores = hybrid_search.rrf_score(dense_scores, sparse_scores)

        # Assert
        # Should return sparse scores only
        assert len(combined_scores) == 2
        assert combined_scores[0] == (0, pytest.approx(0.9, rel=0.01))
```

#### Example 2: Integration Test

```python
"""
Integration test for MCP server search tool.

Tests complete search flow from tool invocation through Qdrant to results.
"""
import pytest


@pytest.mark.requires_qdrant
@pytest.mark.integration
async def test_search_tool_integration(
    isolated_qdrant_container,
    mcp_client,
    sample_documents,
    test_collection_name
):
    """Test MCP search tool with real Qdrant instance."""
    # Arrange: Store sample documents
    for doc in sample_documents:
        await mcp_client.call_tool(
            "store",
            collection=test_collection_name,
            content=doc["content"],
            metadata=doc.get("metadata", {})
        )

    # Act: Search for documents
    search_result = await mcp_client.call_tool(
        "search",
        collection=test_collection_name,
        query="python programming"
    )

    # Assert: Verify search results
    assert search_result["success"] is True
    assert len(search_result["results"]) > 0

    # Verify result structure
    first_result = search_result["results"][0]
    assert "content" in first_result
    assert "score" in first_result
    assert "metadata" in first_result

    # Verify scoring (semantic search)
    # Documents about Python should score higher
    python_doc = next(
        r for r in search_result["results"]
        if "python" in r["content"].lower()
    )
    other_doc = next(
        r for r in search_result["results"]
        if "python" not in r["content"].lower()
    )
    assert python_doc["score"] > other_doc["score"]
```

#### Example 3: Functional Test

```python
"""
Functional test for complete search workflow.

Tests end-to-end user workflow from document storage to search retrieval.
"""
import pytest


@pytest.mark.functional
async def test_document_lifecycle_workflow(
    shared_qdrant_container,
    test_collection_name,
    cleanup_collections
):
    """Test complete document lifecycle: store → search → retrieve → delete."""
    # Register collection for cleanup
    cleanup_collections(test_collection_name)

    # Step 1: Store document
    doc_content = "Machine learning with Python and scikit-learn"
    doc_metadata = {"author": "Alice", "tags": ["ml", "python"]}

    store_result = await store_document(
        collection=test_collection_name,
        content=doc_content,
        metadata=doc_metadata
    )

    assert store_result.success
    doc_id = store_result.document_id

    # Step 2: Search for document
    search_results = await search_documents(
        collection=test_collection_name,
        query="machine learning python"
    )

    assert len(search_results) > 0
    assert any(r.id == doc_id for r in search_results)

    # Step 3: Retrieve specific document
    retrieved_doc = await get_document(
        collection=test_collection_name,
        document_id=doc_id
    )

    assert retrieved_doc.content == doc_content
    assert retrieved_doc.metadata == doc_metadata

    # Step 4: Delete document
    delete_result = await delete_document(
        collection=test_collection_name,
        document_id=doc_id
    )

    assert delete_result.success

    # Step 5: Verify deletion
    search_results = await search_documents(
        collection=test_collection_name,
        query="machine learning python"
    )

    assert all(r.id != doc_id for r in search_results)
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

### Advanced Debugging Techniques

#### IDE Debugging

**Visual Studio Code:**

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Debug pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/unit/test_foo.py::test_bar",
        "-v",
        "-s"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

**PyCharm:**

1. Right-click on test function
2. Select **Debug 'pytest in test_foo.py::test_bar'**
3. Set breakpoints in code
4. Use debugger panel to inspect variables

**Benefits:**
- Step through code line by line
- Inspect variable values at breakpoints
- Evaluate expressions in debug console
- View call stack

#### Analyzing Test Coverage Gaps

**Step 1: Generate Coverage Report**

```bash
# Generate detailed coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

**Step 2: Identify Coverage Gaps**

Look for:
- **Red lines**: Not executed by any test
- **Yellow lines**: Partially covered (e.g., one branch of if/else)
- **Low file coverage**: <80% for critical modules

**Example Output:**

```
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
src/core/hybrid_search.py     245     12    95%   127-130, 156
src/core/client.py            189     45    76%   45-67, 123-145
```

**Step 3: Write Tests for Gaps**

```python
# Example: Coverage gap on lines 127-130 (error handling)
def test_hybrid_search_handles_qdrant_error(mocker):
    """Test error handling when Qdrant connection fails."""
    # Mock Qdrant client to raise error
    mocker.patch(
        'src.core.client.QdrantClient.search',
        side_effect=QdrantConnectionError("Connection failed")
    )

    search = HybridSearch()
    result = search.query("test")

    # This should hit lines 127-130 (error handling path)
    assert result.success is False
    assert "connection failed" in result.error_message.lower()
```

#### Interpreting CI/CD Test Failures

**Common CI/CD Failure Scenarios:**

**1. Tests Pass Locally, Fail in CI**

**Possible Causes:**
- Timing differences (CI is slower/faster)
- Missing environment variables
- Different Python/Rust versions
- Resource constraints in CI

**Debugging Steps:**

```bash
# Check CI logs for environment differences
cat .github/workflows/test.yml | grep -A 5 "env:"

# Run tests with same Python version as CI
pyenv install 3.11.5
pyenv local 3.11.5
uv run pytest

# Increase timeouts for CI environment
@pytest.mark.timeout(60)  # Instead of 30
def test_slow_operation():
    pass
```

**2. Flaky Tests in CI**

```bash
# Run test multiple times locally to reproduce
for i in {1..50}; do
    uv run pytest tests/flaky_test.py::test_foo -v || break
done

# Check CI logs for patterns
gh run view <run_id> --log-failed
```

**3. Container Failures in CI**

```bash
# CI logs show: "docker: Cannot connect to Docker daemon"

# Solution: Check CI workflow has Docker service
jobs:
  test:
    services:
      docker:
        image: docker:dind
        options: --privileged
```

**4. Out of Memory Errors**

```
FAILED tests/stress/test_bulk_insert.py - MemoryError

# Solution: Reduce test data size in CI
@pytest.fixture
def sample_size():
    if os.getenv("CI"):
        return 100  # Smaller in CI
    return 1000  # Larger locally
```

#### Performance Test Analysis and Optimization

**1. Benchmark Test Failures**

```bash
# Run benchmark tests
uv run pytest --benchmark-only

# Compare to baseline
uv run pytest --benchmark-compare=baseline --benchmark-fail-threshold=1.5
```

**Example Failure:**

```
FAILED test_search_performance - Regression: Mean 1.2s exceeds threshold 0.5s
```

**Analysis Steps:**

```python
# Add profiling to identify bottleneck
import cProfile
import pstats

def test_search_performance_profiling():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run slow operation
    result = search(collection="test", query="query")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions
```

**2. Optimize Based on Profile**

```python
# Before optimization (slow)
def search(query):
    for doc in get_all_documents():  # O(n)
        if matches(doc, query):
            yield doc

# After optimization (fast)
def search(query):
    # Use index instead
    doc_ids = index.lookup(query)  # O(log n)
    return [get_document(id) for id in doc_ids]
```

**3. Stress Test Failure Analysis**

```bash
# Stress test fails: Too many open connections

# Check system limits
ulimit -n  # File descriptor limit

# Increase limits for test
ulimit -n 10000
uv run pytest tests/stress/
```

**Common Stress Test Issues:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| Connection pool exhaustion | "Too many connections" | Increase pool size or add connection reuse |
| Memory leak | OOM after N iterations | Profile memory, fix leaks, add cleanup |
| Rate limiting | "429 Too Many Requests" | Add delays, reduce concurrency |
| Resource deadlock | Test hangs indefinitely | Add timeouts, fix resource locking |

#### Log Analysis Techniques

**1. Enable Debug Logging in Tests**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_with_debug_logging():
    # Debug logs will appear in pytest output
    logger.debug("Starting test")
    result = perform_operation()
    logger.debug(f"Result: {result}")
```

**2. Capture Logs with pytest**

```bash
# Show log output for failed tests
uv run pytest --log-cli-level=DEBUG

# Save logs to file
uv run pytest --log-file=test.log --log-file-level=DEBUG
```

**3. Analyze Qdrant Logs**

```bash
# Get container logs
docker logs <qdrant_container_id> 2>&1 | tee qdrant.log

# Search for errors
grep ERROR qdrant.log

# Look for specific operation
grep "collection_name" qdrant.log | grep "search"
```

**4. Parse Structured Logs**

```python
import json

def analyze_test_logs(log_file):
    """Parse and analyze structured JSON logs."""
    errors = []
    warnings = []

    with open(log_file) as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get("level") == "ERROR":
                    errors.append(log_entry)
                elif log_entry.get("level") == "WARN":
                    warnings.append(log_entry)
            except json.JSONDecodeError:
                pass

    print(f"Found {len(errors)} errors, {len(warnings)} warnings")
    return errors, warnings
```

#### When to Use Different Debugging Approaches

**Decision Tree:**

```
Test Failure
    │
    ├─> Quick reproducible issue
    │   └─> Use: pytest -v -s --pdb
    │
    ├─> Intermittent/flaky
    │   └─> Use: Run in loop, enable debug logging
    │
    ├─> Performance regression
    │   └─> Use: pytest --benchmark + cProfile
    │
    ├─> CI-only failure
    │   └─> Use: Check CI logs, reproduce with CI environment
    │
    ├─> Complex interaction bug
    │   └─> Use: IDE debugger with breakpoints
    │
    └─> Unknown cause
        └─> Use: Enable all logging, run single test with -vv -s
```

**Debugging Approach Matrix:**

| Scenario | Best Approach | Commands |
|----------|---------------|----------|
| Quick syntax error | pytest -v | `uv run pytest -v` |
| Logic bug | IDE debugger | Set breakpoints, step through |
| Async race condition | Add logging | `pytest -v -s --log-cli-level=DEBUG` |
| Performance issue | Profiling | `python -m cProfile`, `pytest --benchmark` |
| Container issue | Docker logs | `docker logs <id>`, `docker exec -it <id> sh` |
| Memory leak | Memory profiler | `python -m memory_profiler` |
| Flaky test | Multiple runs | `pytest --count=50` |
| CI failure | CI logs | `gh run view --log-failed` |

### Troubleshooting Common Error Patterns

#### Pattern 1: "Connection Refused" Errors

**Symptom:**

```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=6333):
Max retries exceeded with url: /collections
```

**Root Causes & Solutions:**

```bash
# Cause 1: Qdrant not running
docker ps | grep qdrant  # Check if running
docker start qdrant-test  # Start if stopped

# Cause 2: Wrong port
# Check QDRANT_URL in .env
cat .env | grep QDRANT_URL

# Cause 3: Testcontainer not started
# Check test uses correct fixture
def test_search(isolated_qdrant_container):  # ✅ Correct
    url = isolated_qdrant_container.get_http_url()

def test_search():  # ❌ No container fixture
    url = "http://localhost:6333"  # May not be running
```

#### Pattern 2: "Collection Not Found" Errors

**Symptom:**

```
QdrantException: Collection 'test-collection' not found
```

**Debugging:**

```python
# Add collection existence check
def test_search_with_debug():
    # List all collections
    response = client.get_collections()
    print(f"Available collections: {response.collections}")

    # Check if test collection exists
    assert "test-collection" in [c.name for c.names in response.collections]
```

**Common Causes:**

- Collection not created in test setup
- Wrong collection name (typo)
- Test cleanup removed collection before assertion
- Using shared container without proper reset

#### Pattern 3: "Fixture Not Found" Errors

**Symptom:**

```
E   fixture 'isolated_qdrant_container' not found
```

**Solutions:**

```bash
# Check fixture is imported in conftest.py
cat tests/conftest.py | grep "pytest_plugins"

# Verify fixture exists in shared/fixtures.py
grep "def isolated_qdrant_container" tests/shared/fixtures.py

# Check fixture scope matches
# Function-scoped test can't use session-scoped fixture directly
```

#### Pattern 4: "Timeout" Errors in Async Tests

**Symptom:**

```
asyncio.exceptions.TimeoutError: Test exceeded timeout of 5 seconds
```

**Solutions:**

```python
# Increase timeout for specific test
@pytest.mark.timeout(30)
async def test_slow_operation():
    await slow_async_function()

# Check for deadlocks
import asyncio

async def test_with_timeout():
    try:
        result = await asyncio.wait_for(
            operation(),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        pytest.fail("Operation timed out - possible deadlock")
```

### Resource Constraint Debugging

#### Memory Constraints

**Detect Memory Issues:**

```bash
# Install memory profiler
uv add --dev memory-profiler

# Profile test
python -m memory_profiler tests/unit/test_foo.py

# Monitor memory during test
uv run pytest tests/stress/ --memray
```

**Example Output:**

```
Line #    Mem usage    Increment   Line Contents
================================================
   45   125.2 MiB    125.2 MiB   @profile
   46                             def test_bulk_insert():
   47   725.3 MiB    600.1 MiB       docs = [generate_doc() for _ in range(10000)]  # 🔴 Memory spike
   48   725.3 MiB      0.0 MiB       bulk_insert(docs)
```

**Fix:**

```python
# Use generator instead of list
def test_bulk_insert_optimized():
    docs = (generate_doc() for _ in range(10000))  # ✅ Generator
    bulk_insert(docs)
```

#### CPU Constraints

**Profile CPU Usage:**

```bash
# Profile with cProfile
python -m cProfile -o profile.stats -m pytest tests/performance/

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

**Identify Hotspots:**

```python
import cProfile
import pstats

def test_cpu_intensive():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run operation
    result = cpu_intensive_function()

    profiler.disable()

    # Print top 10 CPU-intensive functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### Disk I/O Constraints

**Symptoms:**
- Tests slow on CI but fast locally
- "Disk quota exceeded" errors
- Intermittent failures with temp files

**Solutions:**

```python
# Use in-memory filesystem for tests
import tempfile

@pytest.fixture
def temp_dir():
    # Create temp directory in memory (Linux)
    if os.path.exists('/dev/shm'):
        temp_dir = tempfile.mkdtemp(dir='/dev/shm')
    else:
        temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)
```

#### Network Constraints

**Symptoms:**
- Timeouts in CI
- Connection pool exhaustion
- Rate limiting errors

**Solutions:**

```python
# Add retry logic with backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def test_with_retry():
    result = await network_operation()
    assert result.success
```

### Debugging Checklist

When debugging a test failure, work through this checklist:

- [ ] **Read the error message** - What exactly failed?
- [ ] **Run single test** - Isolate the failure
- [ ] **Check recent changes** - What changed since it last passed?
- [ ] **Review test logs** - Enable debug logging
- [ ] **Check fixtures** - Are they set up correctly?
- [ ] **Verify environment** - Correct Python version, dependencies?
- [ ] **Test locally** - Does it fail locally or only in CI?
- [ ] **Check resources** - Memory, CPU, disk, network OK?
- [ ] **Profile if slow** - Use cProfile or benchmark
- [ ] **Reproduce consistently** - Can you make it fail reliably?
- [ ] **Minimal reproduction** - Simplify to smallest failing case
- [ ] **Fix root cause** - Don't just treat symptoms
- [ ] **Add regression test** - Prevent future occurrences
- [ ] **Document solution** - Help others who hit same issue

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
