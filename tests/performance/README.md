# Performance Testing Suite

Comprehensive performance validation and testing framework for workspace-qdrant-mcp.

## Overview

This performance testing suite provides complete performance validation, benchmarking, and monitoring capabilities for the workspace-qdrant-mcp project. It includes:

- **Core Operation Benchmarking**: Document processing, vector search, hybrid search operations
- **MCP Tool Performance**: Response time validation across all 30+ MCP tools
- **Memory Profiling**: Memory usage patterns, leak detection, and efficiency analysis
- **Load Testing**: k6-based load testing with multiple scenarios (normal, stress, spike, soak, volume)
- **Concurrency Testing**: Performance under concurrent operations
- **Scaling Analysis**: Performance behavior with varying dataset sizes
- **Regression Detection**: Automated performance regression testing against baselines
- **Real-time Monitoring**: Continuous performance monitoring with alerting

## Quick Start

### Running All Performance Tests

```bash
# Run all performance test suites
python tests/performance/performance_runner.py --all

# Run specific test suites
python tests/performance/performance_runner.py --benchmark --memory

# Run with pytest directly
pytest tests/performance/ -m performance -v
```

### Test Categories

```bash
# Benchmark tests (pytest-benchmark)
pytest tests/performance/ -m benchmark --benchmark-only

# Load testing (k6 integration)
pytest tests/performance/ -m load_testing

# Memory profiling tests
pytest tests/performance/ -m memory_profiling

# Regression tests
pytest tests/performance/ -m regression

# Concurrency tests
pytest tests/performance/ -m concurrency

# Scaling tests
pytest tests/performance/ -m scaling
```

## Test Files

| File | Purpose | Key Features |
|------|---------|--------------|
| `test_performance_validation.py` | Core performance benchmarks | Document processing, vector search, MCP tools, memory analysis, concurrency, scaling |
| `test_load_testing.py` | k6 load testing integration | Normal load, stress testing, spike testing, soak testing, volume testing |
| `test_memory_performance.py` | Memory profiling and analysis | Leak detection, efficiency analysis, GC behavior, allocation patterns |
| `test_regression_validation.py` | Performance regression testing | Baseline comparison, trend analysis, anomaly detection, SLA monitoring |
| `performance_runner.py` | Test orchestration and monitoring | Automated execution, real-time monitoring, report generation |

## Performance Requirements

### Success Criteria

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| Document Processing | < 500ms average | Per document with embedding generation |
| Vector Search | < 100ms average | Standard similarity search |
| Hybrid Search | < 150ms average | Dense + sparse search with fusion |
| MCP Tool Response | < 200ms average | All 30+ tools |
| Memory Usage | < 50MB base overhead | Empty system baseline |
| Memory per Document | < 5KB average | Per document processed |
| Memory Leaks | < 1MB growth | Per 1000 operations |
| Concurrent Operations | Linear scaling | Up to 10 concurrent operations |
| Search Scaling | O(log n) behavior | Sub-linear with dataset size |

### Load Testing Thresholds

| Scenario | Success Criteria |
|----------|------------------|
| Normal Load (10 users) | P95 < 500ms, Error rate < 1% |
| Stress Load (200 users) | P95 < 1000ms, Error rate < 5% |
| Spike Recovery | < 30 seconds recovery time |
| Soak Testing | < 10% memory growth over 30 minutes |
| Volume Testing | Handle 10,000+ documents without degradation |

## Usage Examples

### Establishing Baselines

```bash
# Establish new performance baseline
python tests/performance/performance_runner.py --establish-baseline

# The baseline will be saved and used for future regression testing
```

### Regression Testing

```bash
# Test against latest baseline
python tests/performance/performance_runner.py --regression-test

# Test against specific baseline
python tests/performance/performance_runner.py --regression-test --baseline baseline_20240315_143022.json
```

### Real-time Monitoring

```bash
# Monitor for 1 hour
python tests/performance/performance_runner.py --monitor --duration 3600

# Monitor with verbose output
python tests/performance/performance_runner.py --monitor --duration 1800 --verbose
```

### CI/CD Integration

```bash
# Quick benchmark validation (for CI)
pytest tests/performance/test_performance_validation.py::TestCoreOperationPerformance -m benchmark --benchmark-disable

# Full performance validation (nightly CI)
python tests/performance/performance_runner.py --all

# Regression test in CI
python tests/performance/performance_runner.py --regression-test --baseline ci_baseline.json
```

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PERF_TEST_TIMEOUT` | Test timeout in seconds | 300 |
| `PERF_TEST_ITERATIONS` | Benchmark iterations | 20 |
| `PERF_TEST_QDRANT_URL` | Qdrant server URL | http://localhost:6333 |
| `PERF_TEST_K6_PATH` | Path to k6 binary | k6 |
| `PERF_TEST_REPORT_DIR` | Report output directory | performance_reports |

### Test Configuration

```python
# tests/performance/conftest.py contains configuration
PERFORMANCE_TEST_CONFIG = {
    "response_time_thresholds": {
        "document_processing_ms": 500,
        "vector_search_ms": 100,
        "hybrid_search_ms": 150,
        "mcp_tool_ms": 200,
    },
    "memory_thresholds": {
        "base_overhead_mb": 50,
        "per_document_mb": 0.005,
        "leak_detection_mb": 5,
    },
    "regression_thresholds": {
        "response_time_increase_percent": 20.0,
        "memory_increase_percent": 30.0,
        "throughput_decrease_percent": 15.0,
    }
}
```

## Dependencies

### Required

- `pytest>=7.0.0` - Test framework
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `pytest-asyncio>=0.21.0` - Async test support
- `psutil>=5.8.0` - System resource monitoring
- `tracemalloc` - Memory profiling (built-in)

### Optional

- `k6` - Load testing tool (falls back to mock if not available)
- `qdrant-client>=1.7.0` - For integration tests with real Qdrant

### Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or with uv
uv sync --dev
```

## Report Generation

### Automated Reports

Performance reports are automatically generated after test execution:

```bash
# Generate report from latest results
python tests/performance/performance_runner.py --report-only

# Generate report from specific results file
python tests/performance/performance_runner.py --report-only --results-file performance_test_results_20240315_143022.json
```

### Report Locations

- `performance_reports/` - Test results and reports
- `performance_baselines/` - Baseline files for regression testing
- `htmlcov/` - Coverage reports (if enabled)

### Report Types

| Report | Description | Location |
|--------|-------------|----------|
| Test Results | JSON test execution results | `performance_test_results_*.json` |
| Benchmark Report | pytest-benchmark JSON | `*_benchmark.json` |
| JUnit XML | CI-compatible test results | `*_junit.xml` |
| Regression Analysis | Baseline comparison | `regression_analysis_*.json` |
| Monitoring Data | Real-time monitoring | `monitoring_*.json` |

## Troubleshooting

### Common Issues

#### k6 Not Found

```bash
# Install k6 (macOS)
brew install k6

# Install k6 (Linux)
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Or run without k6 (uses mock results)
pytest tests/performance/test_load_testing.py --run-k6
```

#### Qdrant Connection Issues

```bash
# Start local Qdrant server
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or use test containers (automatic)
pytest tests/performance/ -m requires_qdrant
```

#### Memory Tests Failing

```bash
# Ensure sufficient memory available
free -h

# Run with relaxed thresholds for development
pytest tests/performance/test_memory_performance.py --tb=short
```

### Debug Mode

```bash
# Verbose output
python tests/performance/performance_runner.py --all --verbose

# Debug specific test
pytest tests/performance/test_performance_validation.py::test_specific_function -v -s

# Memory debugging
python -X tracemalloc=1 -m pytest tests/performance/test_memory_performance.py -v
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Nightly
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Install k6
      run: |
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6

    - name: Start Qdrant
      run: |
        docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
        sleep 10

    - name: Run performance tests
      run: |
        python tests/performance/performance_runner.py --all

    - name: Upload performance reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-reports
        path: performance_reports/
```

## Contributing

### Adding New Performance Tests

1. **Benchmark Tests**: Add to `test_performance_validation.py`
   ```python
   @pytest.mark.benchmark
   async def test_new_operation_performance(self, benchmark):
       # Your test implementation
   ```

2. **Memory Tests**: Add to `test_memory_performance.py`
   ```python
   @pytest.mark.memory_profiling
   async def test_new_memory_pattern(self, memory_analyzer):
       # Your test implementation
   ```

3. **Load Tests**: Add to `test_load_testing.py`
   ```python
   @pytest.mark.load_testing
   async def test_new_load_scenario(self, k6_load_tester):
       # Your test implementation
   ```

### Performance Test Guidelines

1. **Use Realistic Data**: Test with realistic document sizes and content
2. **Mock External Dependencies**: Use mocks for consistent performance measurement
3. **Set Clear Thresholds**: Define specific performance requirements
4. **Test Edge Cases**: Include performance tests for edge cases
5. **Document Expectations**: Clearly document what each test validates

### Test Markers

Use appropriate pytest markers for categorization:

```python
@pytest.mark.performance      # General performance test
@pytest.mark.benchmark        # Pytest-benchmark test
@pytest.mark.load_testing     # k6 load test
@pytest.mark.memory_profiling # Memory analysis test
@pytest.mark.regression       # Regression test
@pytest.mark.slow            # Long-running test
@pytest.mark.requires_qdrant  # Requires Qdrant server
@pytest.mark.requires_k6      # Requires k6 tool
```

## License

MIT License - see the main project LICENSE file.