# Performance Benchmarks

This directory contains performance benchmarks for workspace-qdrant-mcp components using pytest-benchmark.

## Benchmark Categories

### File Ingestion Benchmarks (`benchmark_file_ingestion.py`)

Measures file parsing and ingestion throughput across various file sizes and types.

See: [File Ingestion Benchmarks Documentation](./README_FILE_INGESTION.md) (if exists)

### Search Latency Benchmarks (`benchmark_search_latency.py`)

Measures search operation latency with percentile metrics (p50, p95, p99).

See: [Search Latency Benchmarks Documentation](./README_SEARCH_LATENCY.md)

### Memory Usage Benchmarks (`benchmark_memory_usage.py`)

Profiles memory consumption and detects memory leaks across workload scenarios.

See: [Memory Profiling Documentation](./README_MEMORY_PROFILING.md)

### gRPC Overhead Benchmarks (`benchmark_grpc_overhead.py`)

Measures gRPC communication overhead between Python client and Rust daemon.

See: [gRPC Overhead Documentation](./README_GRPC_OVERHEAD.md)

### Context Injection Benchmarks (`benchmark_context_injection.py`)

Measures performance of the LLM Context Injection System:

1. **Rule Retrieval** (5 benchmarks)
   - Rule retrieval from vector database with varying limits (5, 10, 25)
   - Cache performance (cold start vs warm cache)

2. **Token Counting** (4 benchmarks)
   - Token counting with tiktoken for small/medium/large contexts
   - Estimation fallback performance

3. **Token Budget Allocation** (4 benchmarks)
   - Budget allocation across different context sizes
   - Tight budget constraint handling

4. **Context Formatting** (5 benchmarks)
   - Formatting for different LLM tools (Claude Code, GitHub Codex, Google Gemini)
   - Small/medium/large context formatting

5. **End-to-End Pipeline** (4 benchmarks)
   - Complete injection pipeline with varying context sizes
   - Tight budget scenarios

### Database Query Benchmarks (`benchmark_database_queries.py`)

Measures SQLite database query performance and optimization effectiveness.

See: [Database Benchmarks Documentation](./README_DATABASE_BENCHMARKS.md)

### Concurrent Operation Benchmarks (`benchmark_concurrent_operations.py`) ⭐ NEW

Measures performance under concurrent load with contention analysis:

1. **Concurrent File Ingestion** (4 benchmarks)
   - 2, 5, 10, 20 files parsed simultaneously
   - Throughput and scalability measurement

2. **Concurrent Search Operations** (4 benchmarks)
   - 2, 5, 10, 20 queries executing simultaneously
   - Connection pool and resource contention analysis

3. **Mixed Workloads** (2 benchmarks)
   - Combined ingestion + search operations
   - Real-world usage pattern simulation

4. **Contention Analysis** (4 benchmarks)
   - Sequential vs concurrent comparison
   - Contention factor calculation (perfect scaling = 1.0)

See: [Concurrent Benchmarks Documentation](./README_CONCURRENT_BENCHMARKS.md)

## Running Benchmarks

### All Benchmarks

```bash
# Run all benchmarks
uv run pytest tests/benchmarks/ --benchmark-only

# Run all benchmarks with table output
uv run pytest tests/benchmarks/ --benchmark-only -v

# Save benchmark results to JSON
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
```

### Specific Benchmark Files

```bash
# Run context injection benchmarks only
uv run pytest tests/benchmarks/benchmark_context_injection.py --benchmark-only

# Run search latency benchmarks only
uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only
```

### Specific Benchmark Tests

```bash
# Run a specific benchmark
uv run pytest tests/benchmarks/benchmark_context_injection.py::test_token_counting_small_context_tiktoken --benchmark-only

# Run all token counting benchmarks
uv run pytest tests/benchmarks/benchmark_context_injection.py -k "token_counting" --benchmark-only
```

### Benchmarks Requiring Qdrant

Some benchmarks require a running Qdrant instance:

```bash
# Start Qdrant (if not running)
docker run -p 6333:6333 qdrant/qdrant

# Run benchmarks that require Qdrant
uv run pytest tests/benchmarks/ -m "requires_qdrant" --benchmark-only
```

### Benchmark Output

Benchmark results include:

- **Min/Max/Mean/Median**: Timing statistics in microseconds
- **StdDev/IQR**: Variability measures
- **Outliers**: Statistical outliers detected
- **OPS**: Operations per second
- **Percentiles**: p50, p95, p99 latency percentiles (printed to stdout)

Example output:

```
Token counting (small, tiktoken) percentiles: {
  'p50_ms': 4.21,
  'p95_ms': 6.89,
  'p99_ms': 11.32,
  'min_ms': 4.04,
  'max_ms': 57.00,
  'mean_ms': 4.28,
  'median_ms': 4.21
}
```

## Benchmark Comparison

### Compare Across Runs

```bash
# Save baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Make changes...

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline

# Compare only regressions (>5% slower)
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=min:5%
```

### Historical Tracking

```bash
# Save results with timestamp
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

# View historical results
uv run pytest-benchmark list

# Compare specific saved results
uv run pytest-benchmark compare 0001 0002
```

## Regression Detection ⭐ NEW

Advanced regression detection with statistical significance testing:

```bash
# Establish baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Run current benchmarks
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=current.json

# Detect regressions with statistical tests
uv run python tests/benchmarks/regression_detection.py \
  --baseline .benchmarks/baseline.json \
  --current current.json \
  --threshold 5.0 \
  --fail-on-regression

# Export detailed report
uv run python tests/benchmarks/regression_detection.py \
  --baseline .benchmarks/baseline.json \
  --current current.json \
  --threshold 5.0 \
  --export regression_report.json
```

Features:
- Statistical significance testing (Welch's t-test, Mann-Whitney U test)
- Configurable regression thresholds (percentage-based)
- Comprehensive reporting (regressions, improvements, stable)
- JSON export for historical tracking
- CI/CD integration with exit codes

See: [Regression Detection Documentation](./README_REGRESSION_DETECTION.md)

## Interpreting Results

### Context Injection Latency Targets

- **Token Counting**: <10ms per rule (tiktoken), <1ms (estimation)
- **Budget Allocation**: <50ms for 25 rules
- **Context Formatting**: <100ms for 25 rules
- **End-to-End Pipeline**: <500ms for medium context (10 rules)

### Performance Considerations

1. **Caching Impact**: Warm cache should be 10-100x faster than cold start
2. **Context Size**: Latency should scale linearly with number of rules
3. **Token Budget**: Tight budgets add overhead for rule pruning
4. **Tokenizer Choice**: tiktoken is ~10x slower than estimation but more accurate

## Continuous Integration

Benchmarks can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Performance Benchmarks
  run: |
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=ci_results.json

- name: Check for Regressions
  run: |
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=min:10%
```

## Adding New Benchmarks

Follow the pattern in existing benchmark files:

```python
@pytest.mark.benchmark
@pytest.mark.requires_qdrant  # If Qdrant is needed
def test_my_operation(benchmark, fixture_name):
    """Benchmark description."""

    def operation_to_benchmark():
        # Your code here
        return result

    result = benchmark(operation_to_benchmark)

    # Calculate percentiles
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\\nMy operation percentiles: {percentiles}")

    # Assert correctness
    assert result is not None
```

## Troubleshooting

### Benchmarks Run Twice

If benchmarks appear to run twice, ensure you're using `--benchmark-only` flag to skip regular test execution.

### High Variability

If results show high standard deviation:
- Close other applications
- Run on dedicated hardware
- Increase `min_rounds` or `max_time` in benchmark configuration
- Check for resource contention (CPU/memory/disk)

### Qdrant Connection Errors

If rule retrieval or end-to-end benchmarks fail:
- Ensure Qdrant is running: `docker ps | grep qdrant`
- Check connection: `curl http://localhost:6333`
- Verify collection exists after fixture setup

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Benchmark best practices](https://pytest-benchmark.readthedocs.io/en/latest/usage.html#best-practices)
