# Concurrent Operation Benchmarks

Comprehensive benchmarks measuring performance of concurrent operations including multiple simultaneous file ingestions, concurrent search operations, mixed workloads, and resource contention analysis.

## Overview

These benchmarks test the system's ability to handle concurrent operations efficiently, measuring throughput, latency under contention, and scalability across various concurrency levels (2, 5, 10, 20 operations).

## Benchmark Categories

### 1. Concurrent File Ingestion

Tests multiple file parsing operations running simultaneously.

#### Benchmarks:

- **`test_concurrent_ingestion_2_files`** - 2 small text files (10KB each)
- **`test_concurrent_ingestion_5_files`** - 5 medium Python files (50KB each)
- **`test_concurrent_ingestion_10_files`** - 10 mixed-type files (20KB each)
- **`test_concurrent_ingestion_20_files`** - 20 large text files (100KB each) - stress test

#### Metrics:

- Total execution time (wall-clock)
- Throughput (files/second)
- Individual file processing latency
- Resource utilization

### 2. Concurrent Search Operations

Tests multiple search queries executing simultaneously against Qdrant.

#### Benchmarks:

- **`test_concurrent_search_2_queries`** - 2 simultaneous searches
- **`test_concurrent_search_5_queries`** - 5 simultaneous searches
- **`test_concurrent_search_10_queries`** - 10 simultaneous searches
- **`test_concurrent_search_20_queries`** - 20 simultaneous searches - stress test

#### Metrics:

- Total query time
- Throughput (searches/second)
- Per-query latency
- Qdrant connection pool utilization

### 3. Mixed Workloads

Tests combinations of file ingestion and search operations running concurrently.

#### Benchmarks:

- **`test_mixed_workload_5_ingestion_5_search`** - 5 ingestions + 5 searches (10 total ops)
- **`test_mixed_workload_10_ingestion_10_search`** - 10 ingestions + 10 searches (20 total ops)

#### Metrics:

- Total workload time
- Overall throughput (ops/second)
- Individual operation type performance
- Resource contention indicators

### 4. Contention Analysis

Compares sequential vs concurrent execution to identify contention bottlenecks.

#### Benchmarks:

- **`test_contention_analysis_sequential_baseline`** - 10 files parsed sequentially
- **`test_contention_analysis_concurrent_comparison`** - 10 files parsed concurrently
- **`test_contention_search_sequential_baseline`** - 10 searches sequential
- **`test_contention_search_concurrent_comparison`** - 10 searches concurrent

#### Metrics:

- Sequential execution time
- Concurrent execution time
- Contention factor = concurrent_time / sequential_time
  - 1.0 = perfect linear scaling
  - \>1.0 = contention/overhead present
  - <1.0 = super-linear scaling (caching effects)

## Running Benchmarks

### All Concurrent Benchmarks

```bash
# Run all concurrent operation benchmarks
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only

# With verbose output
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only -v

# Save results to JSON
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only --benchmark-json=concurrent_results.json
```

### Specific Concurrency Levels

```bash
# Run only 2-concurrent benchmarks
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "concurrent_2" --benchmark-only

# Run only ingestion benchmarks
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "ingestion" --benchmark-only

# Run only search benchmarks
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "search" --benchmark-only

# Run only mixed workload benchmarks
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "mixed" --benchmark-only

# Run only contention analysis
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "contention" --benchmark-only
```

### Prerequisites

Some benchmarks require a running Qdrant instance:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Or if already running as service
wqm service start

# Run benchmarks requiring Qdrant
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -m "requires_qdrant" --benchmark-only
```

## Understanding Results

### Sample Output

```
========================== test session starts ===========================
tests/benchmarks/benchmark_concurrent_operations.py::test_concurrent_ingestion_10_files

Concurrent ingestion (10 files) - Throughput: 45.23 files/sec

--------------------------------- benchmark: 1 tests ---------------------------------
Name                                          Min      Max    Mean  Median  StdDev
test_concurrent_ingestion_10_files      220.89ms 245.12ms 221.00ms 220.50ms   3.21ms
--------------------------------------------------------------------------------------
```

### Key Metrics Explained

1. **Throughput**: Operations completed per second
   - Higher is better
   - Compare across concurrency levels to assess scalability

2. **Mean Time**: Average time to complete all concurrent operations
   - Lower is better
   - Should not increase linearly with operation count (indicates good parallelization)

3. **StdDev**: Standard deviation of timings
   - Lower indicates more consistent performance
   - High values may indicate resource contention

### Performance Targets

Based on typical hardware (4-core CPU, 16GB RAM):

| Benchmark Type | Concurrency | Target Throughput | Target Mean Time |
|---------------|-------------|-------------------|------------------|
| File Ingestion | 2 files | >100 files/sec | <20ms |
| File Ingestion | 5 files | >80 files/sec | <65ms |
| File Ingestion | 10 files | >40 files/sec | <250ms |
| Search Queries | 2 queries | >50 searches/sec | <40ms |
| Search Queries | 5 queries | >30 searches/sec | <170ms |
| Search Queries | 10 queries | >15 searches/sec | <700ms |
| Mixed Workload | 10 ops | >25 ops/sec | <400ms |

### Contention Factor Analysis

```python
# Calculate contention factor
sequential_time = 1.5  # seconds for 10 operations
concurrent_time = 0.22  # seconds for 10 operations

contention_factor = concurrent_time / (sequential_time / 10)
# contention_factor = 0.22 / 0.15 = 1.47

# Interpretation:
# 1.47 > 1.0 indicates some overhead/contention
# Ideally should be close to 1.0 for perfect scaling
```

**Ideal scaling**: `concurrent_time ≈ sequential_time / num_workers`

**Reality**: Some overhead from context switching, lock contention, resource sharing

## Integration with Other Benchmarks

Concurrent benchmarks build on infrastructure from:

- **`benchmark_file_ingestion.py`** - Parser performance baselines
- **`benchmark_search_latency.py`** - Search operation baselines
- **`benchmark_memory_usage.py`** - Memory profiling under load
- **`benchmark_grpc_overhead.py`** - gRPC communication overhead

### Cross-Benchmark Analysis

```bash
# Run file ingestion baseline + concurrent comparison
uv run pytest \
  tests/benchmarks/benchmark_file_ingestion.py \
  tests/benchmarks/benchmark_concurrent_operations.py \
  --benchmark-only \
  --benchmark-compare

# Run all benchmarks for comprehensive analysis
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=all_benchmarks.json
```

## Regression Detection

Use regression detection to track concurrent performance over time:

```bash
# Establish baseline
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py \
  --benchmark-only \
  --benchmark-save=concurrent_baseline

# Make changes...

# Compare for regressions
uv run pytest tests/benchmarks/benchmark_concurrent_operations.py \
  --benchmark-only \
  --benchmark-compare=concurrent_baseline \
  --benchmark-compare-fail=min:10%

# Or use the regression detection tool
uv run python tests/benchmarks/regression_detection.py \
  --baseline .benchmarks/concurrent_baseline.json \
  --current .benchmarks/current.json \
  --threshold 10.0 \
  --fail-on-regression
```

## CI/CD Integration

See `.github/workflows/benchmark-regression.yml` for automated regression detection in CI/CD pipelines.

### GitHub Actions Example

```yaml
- name: Run Concurrent Benchmarks
  run: |
    uv run pytest tests/benchmarks/benchmark_concurrent_operations.py \
      --benchmark-only \
      --benchmark-json=concurrent_results.json

- name: Detect Regressions
  run: |
    uv run python tests/benchmarks/regression_detection.py \
      --baseline baseline.json \
      --current concurrent_results.json \
      --threshold 10.0 \
      --fail-on-regression
```

## Troubleshooting

### High Variability in Results

If concurrent benchmarks show high standard deviation:

1. **Close other applications** - Reduce system load
2. **Increase iterations** - More samples for statistical significance
3. **Check system resources** - CPU/memory/disk utilization
4. **Disable power management** - Prevent CPU frequency scaling
5. **Run on dedicated hardware** - Avoid shared CI runners for baselines

### Qdrant Connection Errors

If search benchmarks fail:

```bash
# Check Qdrant status
curl http://localhost:6333/health

# Verify collections exist
curl http://localhost:6333/collections

# Check logs
docker logs <qdrant-container-id>
```

### Low Throughput

If throughput is unexpectedly low:

1. **Check parser performance** - Run `benchmark_file_ingestion.py` first
2. **Verify Qdrant performance** - Run `benchmark_search_latency.py` first
3. **Profile with memory benchmarks** - Check for memory leaks
4. **Monitor connection pooling** - gRPC connection overhead
5. **Review system limits** - File descriptors, network connections

### Unexpected Contention

If contention factor is high (>2.0):

1. **Profile with detailed tracing** - Use `py-spy` or similar
2. **Check for global locks** - GIL contention in CPU-bound code
3. **Review async implementation** - Ensure proper `async/await` usage
4. **Monitor I/O wait** - Disk or network bottlenecks
5. **Check resource pools** - Connection pool sizing

## Advanced Usage

### Custom Concurrency Levels

To test custom concurrency levels, modify the test:

```python
@pytest.mark.benchmark
def test_concurrent_ingestion_custom(benchmark, tmp_concurrent_dir):
    """Benchmark concurrent ingestion with custom file count."""
    num_files = 15  # Custom concurrency level
    files = ConcurrentTestDataGenerator.generate_test_files(
        num_files, 50, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    async def concurrent_parse():
        tasks = [parser.parse(f) for f in files]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == num_files
```

### Profiling Concurrent Operations

```bash
# Profile with py-spy
py-spy record -o profile.svg -- uv run pytest \
  tests/benchmarks/benchmark_concurrent_operations.py::test_concurrent_ingestion_10_files \
  --benchmark-only

# Profile with cProfile
uv run python -m cProfile -o profile.stats -m pytest \
  tests/benchmarks/benchmark_concurrent_operations.py::test_concurrent_search_10_queries \
  --benchmark-only

# Analyze profile
uv run python -m pstats profile.stats
```

### Memory Profiling Under Concurrent Load

```bash
# Combine with memory profiling
uv run pytest \
  tests/benchmarks/benchmark_concurrent_operations.py \
  tests/benchmarks/benchmark_memory_usage.py \
  --benchmark-only \
  -v
```

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [asyncio concurrency patterns](https://docs.python.org/3/library/asyncio-task.html)
- [Performance profiling best practices](https://docs.python.org/3/library/profile.html)
- [Regression detection guide](./README_REGRESSION_DETECTION.md)
