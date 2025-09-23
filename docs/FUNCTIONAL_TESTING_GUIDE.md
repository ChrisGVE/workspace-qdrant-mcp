# Functional Testing Framework Guide

This guide provides comprehensive documentation for the functional testing frameworks installed in workspace-qdrant-mcp, covering both Python and Rust testing capabilities.

## Overview

The functional testing framework provides:

- **Python Frameworks**: pytest-playwright, testcontainers, httpx/respx, pytest-benchmark
- **Rust Frameworks**: cargo-nextest, testcontainers, criterion, proptest
- **Integration Testing**: Cross-language testing, MCP protocol compliance
- **Performance Testing**: Benchmarking and load testing capabilities
- **Web UI Testing**: Playwright-based browser automation
- **Service Isolation**: Testcontainers for isolated testing environments

## Table of Contents

1. [Python Functional Testing](#python-functional-testing)
2. [Rust Functional Testing](#rust-functional-testing)
3. [Integration Testing](#integration-testing)
4. [Performance Testing](#performance-testing)
5. [Web UI Testing](#web-ui-testing)
6. [MCP Protocol Testing](#mcp-protocol-testing)
7. [Test Execution Commands](#test-execution-commands)
8. [Configuration and Environment](#configuration-and-environment)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Python Functional Testing

### Framework Components

#### pytest-playwright
Web UI functional testing framework for browser automation.

**Key Features:**
- Cross-browser testing (Chromium, Firefox, WebKit)
- Automatic browser management
- Screenshot and video recording
- Mobile device emulation
- Network interception and mocking

**Usage Example:**
```python
import pytest
from playwright.async_api import Page

@pytest.mark.playwright
async def test_web_interface(page: Page):
    await page.goto("http://localhost:8000")
    await page.click("#search-button")
    await expect(page.locator("#results")).to_be_visible()
```

#### testcontainers
Service isolation framework for integration testing.

**Key Features:**
- Docker container management
- Service health checking
- Automatic cleanup
- Port mapping and networking
- Pre-configured service images

**Usage Example:**
```python
from testcontainers.qdrant import QdrantContainer

@pytest.mark.testcontainers
async def test_qdrant_integration():
    with QdrantContainer() as qdrant:
        host = qdrant.get_container_host_ip()
        port = qdrant.get_exposed_port(6333)
        # Test with isolated Qdrant instance
```

#### httpx/respx
HTTP client and mocking framework for API testing.

**Key Features:**
- Async HTTP client
- Request/response mocking
- Pattern-based route matching
- Request recording and playback
- JSON and form data handling

**Usage Example:**
```python
import httpx
import respx

@pytest.mark.api_testing
async def test_api_endpoint():
    with respx.mock:
        respx.get("/api/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        async with httpx.AsyncClient() as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
```

#### pytest-benchmark
Performance testing and benchmarking framework.

**Key Features:**
- Statistical analysis
- Performance regression detection
- Multiple measurement strategies
- HTML and JSON reporting
- Baseline comparison

**Usage Example:**
```python
@pytest.mark.benchmark
def test_search_performance(benchmark):
    result = benchmark(perform_search, "test query")
    assert len(result) > 0
```

### Test Organization

```
tests/functional/
├── conftest.py                 # Shared fixtures and configuration
├── pytest.ini                 # Pytest configuration
├── test_mcp_protocol_compliance.py    # MCP protocol tests
├── test_web_ui_functional.py   # Web UI tests
├── test_api_integration.py     # API integration tests
├── test_performance.py         # Performance tests
└── test_cross_language.py      # Cross-language integration
```

## Rust Functional Testing

### Framework Components

#### cargo-nextest
Enhanced test runner for Rust with better performance and features.

**Key Features:**
- Parallel test execution
- Test filtering and selection
- Retry mechanisms
- Multiple output formats
- Profile-based configuration

**Usage Commands:**
```bash
# Run all functional tests
cargo nextest run --profile functional

# Run only fast tests
cargo nextest run --profile fast

# Run integration tests
cargo nextest run --profile integration
```

#### testcontainers (Rust)
Container management for isolated service testing.

**Key Features:**
- Docker container lifecycle management
- Health check waiting
- Network isolation
- Port mapping
- Custom image support

**Usage Example:**
```rust
use testcontainers::*;

#[tokio::test]
async fn test_service_integration() {
    let docker = clients::Cli::default();
    let container = docker.run(images::qdrant::Qdrant);

    let host = container.get_host();
    let port = container.get_host_port_ipv4(6333);

    // Test with isolated service
}
```

#### criterion
Statistical benchmarking framework.

**Key Features:**
- Statistical analysis
- HTML report generation
- Baseline comparison
- Regression detection
- Parameterized benchmarks

**Usage Example:**
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_search(c: &mut Criterion) {
    c.bench_function("hybrid_search", |b| {
        b.iter(|| perform_hybrid_search("test query"))
    });
}

criterion_group!(benches, benchmark_search);
criterion_main!(benches);
```

#### proptest
Property-based testing framework.

**Key Features:**
- Automatic test case generation
- Shrinking on failure
- Custom generators
- Stateful testing
- Deterministic random testing

**Usage Example:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_search_properties(
        query in "\\PC{1,100}",
        limit in 1usize..1000
    ) {
        let results = search_documents(&query, limit);
        prop_assert!(results.len() <= limit);
    }
}
```

### Test Organization

```
rust-engine/
├── tests/
│   ├── functional/
│   │   ├── test_service_integration.rs
│   │   ├── test_performance.rs
│   │   └── test_property_based.rs
│   └── integration/
├── benches/
│   └── processing_benchmarks.rs
└── .cargo/
    └── config.toml            # Cargo configuration
```

## Integration Testing

### Cross-Language Testing

Tests that span both Python and Rust components:

```python
@pytest.mark.cross_language
async def test_python_rust_communication():
    # Test Python MCP server with Rust engine
    rust_response = await call_rust_engine(document)
    python_result = await process_rust_response(rust_response)
    assert python_result["success"] is True
```

### Service Integration

Tests that verify component interaction:

```python
@pytest.mark.integration
async def test_full_workflow():
    # Test complete document processing workflow
    document_id = await store_document(content)
    results = await search_documents(query)
    assert document_id in [r["id"] for r in results]
```

## Performance Testing

### Python Performance Tests

```python
@pytest.mark.performance
def test_bulk_processing_performance(benchmark):
    documents = generate_test_documents(1000)

    def process_bulk():
        return bulk_process_documents(documents)

    result = benchmark(process_bulk)
    assert len(result) == 1000
```

### Rust Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- hybrid_search

# Compare with baseline
cargo bench-compare
```

### Performance Monitoring

The framework includes automatic performance monitoring:

- Memory usage tracking
- CPU utilization monitoring
- Network request counting
- Test duration measurement

## Web UI Testing

### Playwright Configuration

```python
@pytest.fixture
async def browser_context():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        yield context
        await context.close()
        await browser.close()
```

### UI Test Patterns

```python
@pytest.mark.web_ui
async def test_search_interface(page: Page):
    # Navigate to search page
    await page.goto("http://localhost:8000/search")

    # Fill search form
    await page.fill("#search-input", "test query")
    await page.click("#search-button")

    # Verify results
    await page.wait_for_selector(".search-results")
    results = await page.locator(".result-item").count()
    assert results > 0
```

### Accessibility Testing

```python
@pytest.mark.accessibility
async def test_accessibility_compliance(page: Page):
    await page.goto("http://localhost:8000")

    # Check ARIA labels
    buttons = page.locator("button")
    for i in range(await buttons.count()):
        button = buttons.nth(i)
        aria_label = await button.get_attribute("aria-label")
        text_content = await button.text_content()
        assert aria_label or text_content
```

## MCP Protocol Testing

### Protocol Compliance Tests

```python
@pytest.mark.mcp_protocol
async def test_tools_list_compliance():
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }

    response = await mcp_client.call(request)

    # Validate JSON-RPC format
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "tools" in response["result"]
```

### Tool Call Testing

```python
@pytest.mark.mcp_protocol
async def test_tool_call_validation():
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "store_document",
            "arguments": {"content": "test content"}
        }
    }

    response = await mcp_client.call(request)
    validate_mcp_response(response, request["id"])
```

## Test Execution Commands

### Python Test Execution

```bash
# Run all functional tests
uv run pytest tests/functional/ -v

# Run specific test categories
uv run pytest -m "playwright" -v
uv run pytest -m "testcontainers" -v
uv run pytest -m "mcp_protocol" -v

# Run performance tests
uv run pytest -m "performance" --benchmark-only

# Run with coverage
uv run pytest tests/functional/ --cov=src/python --cov-report=html

# Run in parallel
uv run pytest tests/functional/ -n auto
```

### Rust Test Execution

```bash
# Navigate to Rust engine directory
cd rust-engine

# Run all tests with nextest
cargo nextest run

# Run specific test profiles
cargo nextest run --profile functional
cargo nextest run --profile integration
cargo nextest run --profile performance

# Run benchmarks
cargo bench

# Run property-based tests
cargo test --test property_tests

# Generate coverage report
cargo llvm-cov nextest --html
```

### Combined Testing

```bash
# Run both Python and Rust tests
make test-all

# Run only functional tests
make test-functional

# Run performance tests
make test-performance
```

## Configuration and Environment

### Environment Variables

```bash
# Test mode settings
export WQM_TEST_MODE=true
export RUST_TEST_MODE=true

# Service URLs
export QDRANT_URL=http://localhost:6333

# Test configuration
export RECORD_VIDEO=true        # Record Playwright videos
export HEADLESS=false          # Run browsers in headed mode
export SLOW_TESTS=true         # Include slow tests
export PARALLEL_TESTS=true     # Enable parallel execution
```

### Docker Requirements

Ensure Docker is running for testcontainer tests:

```bash
# Check Docker status
docker info

# Start Docker if needed
sudo systemctl start docker  # Linux
open -a Docker               # macOS
```

### Test Data Setup

```bash
# Create test data directory
mkdir -p test-data

# Generate sample documents
python scripts/generate_test_data.py

# Start test services
docker-compose -f docker-compose.test.yml up -d
```

## Best Practices

### Test Organization

1. **Use descriptive test names** that explain what is being tested
2. **Group related tests** in classes or modules
3. **Use appropriate markers** for test categorization
4. **Separate unit, integration, and functional tests**
5. **Mock external dependencies** where appropriate

### Performance Testing

1. **Establish baselines** before making changes
2. **Use statistical significance** for performance comparisons
3. **Control test environment** variables
4. **Run performance tests** in dedicated environments
5. **Monitor resource usage** during tests

### Test Data Management

1. **Use factories** for test data generation
2. **Clean up test data** after each test
3. **Use realistic data sizes** for performance tests
4. **Isolate test data** between test runs
5. **Version test datasets** for reproducibility

### Error Handling

1. **Test error conditions** explicitly
2. **Verify error messages** and codes
3. **Test recovery mechanisms**
4. **Validate error propagation**
5. **Test timeout scenarios**

## Troubleshooting

### Common Issues

#### Docker/Testcontainers Issues

```bash
# Check Docker daemon
docker info

# Clean up containers
docker system prune

# Reset testcontainers
rm -rf ~/.testcontainers
```

#### Playwright Issues

```bash
# Install browsers
playwright install

# Check browser status
playwright doctor

# Clear browser cache
rm -rf ~/.cache/ms-playwright
```

#### Performance Test Issues

```bash
# Check system resources
htop
iostat 1 5

# Disable CPU scaling
sudo cpupower frequency-set --governor performance

# Clear system caches
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

#### Network Issues

```bash
# Check port availability
netstat -tlnp | grep :6333

# Reset network namespace
sudo ip netns delete test-ns || true
```

### Debugging Tests

#### Enable Debug Logging

```bash
# Python tests
RUST_LOG=debug uv run pytest tests/functional/ -v -s

# Rust tests
RUST_LOG=debug cargo nextest run --nocapture
```

#### Generate Debug Reports

```bash
# Generate test reports
uv run pytest tests/functional/ --html=test-results/report.html

# Generate coverage reports
cargo llvm-cov nextest --html --open
```

#### Record Test Execution

```bash
# Record Playwright videos
RECORD_VIDEO=true uv run pytest tests/functional/ -m playwright

# Save screenshots on failure
uv run pytest tests/functional/ --screenshot=on-failure
```

### Performance Debugging

#### Profile Test Execution

```bash
# Profile Python tests
uv run pytest tests/functional/ --profile

# Profile Rust benchmarks
cargo bench -- --profile-time=10
```

#### Memory Analysis

```bash
# Python memory profiling
uv run pytest tests/functional/ --memray

# Rust memory analysis
cargo nextest run --profile performance
```

## Continuous Integration

### GitHub Actions Integration

The functional testing framework integrates with CI/CD:

```yaml
# .github/workflows/functional-tests.yml
name: Functional Tests

on: [push, pull_request]

jobs:
  functional-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable

    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
        cargo install cargo-nextest

    - name: Run Python functional tests
      run: uv run pytest tests/functional/ -v

    - name: Run Rust functional tests
      run: cargo nextest run --profile functional

    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test-results/
```

### Test Reporting

The framework generates comprehensive test reports:

- **JUnit XML**: For CI integration
- **HTML Coverage**: For detailed coverage analysis
- **Performance Reports**: For benchmark tracking
- **Screenshot/Video**: For UI test debugging

## Conclusion

This functional testing framework provides comprehensive testing capabilities for workspace-qdrant-mcp, covering:

- Web UI testing with Playwright
- Service isolation with testcontainers
- Performance benchmarking with criterion
- Property-based testing with proptest
- MCP protocol compliance validation
- Cross-language integration testing

Follow the patterns and practices outlined in this guide to ensure robust, maintainable, and effective functional tests for your application.

For additional support, refer to the framework documentation:

- [Playwright Documentation](https://playwright.dev/python/)
- [testcontainers Documentation](https://testcontainers-python.readthedocs.io/)
- [criterion Documentation](https://bheisler.github.io/criterion.rs/)
- [proptest Documentation](https://docs.rs/proptest/)
- [cargo-nextest Documentation](https://nexte.st/)