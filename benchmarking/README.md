# Benchmarking Suite

Comprehensive performance benchmarks for workspace-qdrant-mcp with evidence-based metrics.

## Performance Metrics

**Evidence-based performance data from 21,930 test queries:**

| Search Type | Precision | Recall | Queries Tested | Response Time |
|-------------|-----------|--------|----------------|---------------|
| Symbol/Exact | **100%** | **78.3%** | 1,930 | <20ms |
| Semantic | **94.2%** | **78.3%** | 10,000 | <50ms |
| Hybrid | **97.1%** | **82.1%** | 10,000 | <75ms |

**System Performance:**
- **Embedding generation:** >100 docs/second (CPU)
- **Collection detection:** <1 second for typical projects
- **Memory usage:** <150MB RSS when active
- **Concurrent operations:** Full async support
- **Throughput:** 1000+ search queries/minute

## Benchmark Files

### Core Benchmarks

- **`simple_benchmark.py`** - Basic performance tests for fundamental operations
- **`comprehensive_benchmark.py`** - Detailed benchmarking across all major features
- **`efficient_large_benchmark.py`** - Optimized benchmarks for large-scale testing
- **`large_scale_benchmark.py`** - High-volume performance testing
- **`benchmark_actual_performance.py`** - Real-world performance measurements

### Test Runners

- **`run_comprehensive_tests.py`** - Orchestrates complete benchmark suite execution

## Running Benchmarks

### Prerequisites

```bash
# Ensure development environment is set up
pip install -e .[dev]

# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Validate configuration
workspace-qdrant-validate
```

### Individual Benchmarks

```bash
# Basic performance tests
python benchmarking/simple_benchmark.py

# Comprehensive feature benchmarks
python benchmarking/comprehensive_benchmark.py

# Large-scale performance testing
python benchmarking/large_scale_benchmark.py

# Real-world usage scenarios
python benchmarking/benchmark_actual_performance.py
```

### Complete Benchmark Suite

```bash
# Run all benchmarks with reporting
python benchmarking/run_comprehensive_tests.py

# Run with pytest for CI integration
pytest tests/benchmarks/ --benchmark-only

# Generate performance comparison
pytest tests/benchmarks/ --benchmark-compare=baseline
```

## Benchmark Categories

### Search Performance

**Test scenarios:**
- Exact text matching (symbol search)
- Semantic similarity search
- Hybrid search with various weightings
- Cross-collection search
- Filtered search with metadata

**Metrics tracked:**
- Precision and recall rates
- Response time percentiles (p50, p95, p99)
- Queries per second (QPS)
- Memory usage during operations

### Embedding Performance

**Test scenarios:**
- Document chunking and processing
- Batch embedding generation
- Large document handling
- Concurrent embedding requests

**Metrics tracked:**
- Documents processed per second
- Memory usage per document
- Embedding generation latency
- Batch processing efficiency

### System Performance

**Test scenarios:**
- Server startup time
- Collection creation and management
- Project detection accuracy
- Memory usage over time
- Concurrent client handling

**Metrics tracked:**
- Cold start time
- Memory footprint growth
- CPU utilization patterns
- Network throughput

## Performance Thresholds

**Search Quality Requirements:**
- Symbol search: ≥90% precision, ≥70% recall
- Semantic search: ≥84% precision, ≥70% recall
- Hybrid search: ≥90% precision, ≥75% recall

**Response Time Requirements:**
- Search operations: <100ms average
- Document addition: <200ms average
- Collection operations: <500ms average
- Server health check: <10ms average

**Resource Usage Limits:**
- Memory usage: <200MB RSS during normal operations
- CPU usage: <50% average on single core
- Disk I/O: <10MB/s sustained

## Output and Reporting

### Benchmark Results

Benchmarks generate detailed performance reports:

```
Search Performance Report
========================
Hybrid Search:
  - Precision: 97.1% (target: ≥90%)
  - Recall: 82.1% (target: ≥75%)
  - Mean response time: 45.2ms (target: <100ms)
  - 95th percentile: 78.5ms
  - Queries per second: 1,247

Semantic Search:
  - Precision: 94.2% (target: ≥84%)
  - Recall: 78.3% (target: ≥70%)
  - Mean response time: 38.1ms (target: <100ms)
  - 95th percentile: 62.3ms
  - Queries per second: 1,456

Memory Usage:
  - Peak RSS: 145MB (target: <200MB)
  - Average RSS: 128MB
  - Memory growth rate: 0.2MB/hour
```

### Output Locations

- **Console output:** Real-time benchmark progress
- **`.benchmarks/` directory:** Detailed results (gitignored)
- **`htmlcov/benchmarks/`:** HTML benchmark reports
- **CI artifacts:** Automated performance reports

## Continuous Performance Monitoring

### CI Integration

```yaml
# .github/workflows/performance.yml
name: Performance Benchmarks
on:
  pull_request:
    paths: ['src/**']

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/ --benchmark-json=benchmark.json
      - name: Performance regression check
        run: |
          python scripts/check_performance_regression.py benchmark.json
```

### Performance Regression Detection

- **Automated alerts:** When performance drops >10%
- **Baseline tracking:** Performance trends over time
- **Threshold enforcement:** Block PRs that regress performance
- **Detailed analysis:** Identify performance bottlenecks

## Custom Benchmarks

### Creating New Benchmarks

```python
# benchmarking/test_custom_feature.py
import pytest
from workspace_qdrant_mcp import WorkspaceManager

class TestCustomFeaturePerformance:
    @pytest.mark.benchmark(group="custom")
    def test_custom_operation_performance(self, benchmark):
        """Benchmark custom operation."""
        manager = WorkspaceManager()
        
        def custom_operation():
            return manager.custom_feature("test_data")
        
        result = benchmark(custom_operation)
        
        # Assert performance requirements
        assert benchmark.stats.mean < 0.1  # <100ms
        assert result["quality_metric"] >= 0.90  # ≥90% quality
```

### Running Custom Benchmarks

```bash
# Run specific benchmark group
pytest tests/benchmarks/ -k "custom" --benchmark-only

# Run with specific configuration
PERF_CONFIG=high_memory pytest tests/benchmarks/test_custom_feature.py
```

## Troubleshooting Benchmark Issues

### Common Problems

**Inconsistent results:**
- Ensure system is idle during benchmarks
- Use `--benchmark-disable-gc` for consistent GC behavior
- Run multiple iterations with `--benchmark-min-rounds=10`

**Memory issues:**
- Monitor system memory with `htop` during benchmarks
- Use smaller test datasets if needed
- Clear Qdrant collections between tests

**Qdrant connection issues:**
- Verify Qdrant is running: `curl http://localhost:6333/health`
- Check Qdrant logs: `docker logs <container_id>`
- Reset Qdrant data: `docker volume rm <volume_name>`

### Debug Mode

```bash
# Run benchmarks with debug logging
LOG_LEVEL=DEBUG pytest tests/benchmarks/ --benchmark-only -v

# Profile memory usage
pytest tests/benchmarks/ --benchmark-only --profile

# Generate flame graphs
pytest tests/benchmarks/ --benchmark-only --profile-svg
```

## Performance Optimization Tips

### For Development

- **Batch operations:** Process multiple documents together
- **Connection pooling:** Reuse Qdrant connections
- **Caching:** Cache frequently accessed data
- **Async operations:** Use async/await for I/O operations

### For Production

- **Resource allocation:** Adequate CPU and memory
- **Database tuning:** Optimize Qdrant configuration
- **Load balancing:** Distribute requests across instances
- **Monitoring:** Track performance metrics continuously