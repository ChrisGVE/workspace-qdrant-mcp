# Memory Usage Profiling Benchmarks

Comprehensive memory profiling benchmarks for workspace-qdrant-mcp components.

## Overview

The memory profiling benchmarks measure:
- Memory consumption across various workload scenarios
- Memory leak detection in long-running operations
- Peak memory usage (RSS) and allocation patterns
- Memory efficiency of embedding models, search, and ingestion

## Running Benchmarks

### All Memory Benchmarks

```bash
uv run pytest tests/benchmarks/benchmark_memory_usage.py --benchmark-only
```

### Specific Test Categories

```bash
# Single document ingestion
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "single_document" --benchmark-only

# Batch ingestion (excluding slow tests)
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "batch_ingestion" -m "not slow" --benchmark-only

# Search operations
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "search" --benchmark-only

# Embedding model memory
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "embedding_model" --benchmark-only

# Leak detection tests (longer running)
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "leak" --benchmark-only

# Connection pooling
uv run pytest tests/benchmarks/benchmark_memory_usage.py -k "connection_pooling" --benchmark-only
```

### With Verbose Output

```bash
uv run pytest tests/benchmarks/benchmark_memory_usage.py --benchmark-only -v
```

## Memory Metrics

Each benchmark reports:

- **peak_rss_mb**: Peak resident set size (physical memory) in MB
- **current_rss_mb**: Current RSS after operation completes
- **rss_delta_mb**: Change in RSS during the operation (key metric)
- **tracemalloc_peak_mb**: Peak Python allocations tracked by tracemalloc
- **tracemalloc_current_mb**: Current Python allocations
- **allocation_count**: Number of distinct memory allocations

## Interpreting Results

### Good Performance Indicators

- **Low rss_delta_mb**: Operation is memory-efficient
- **Stable memory in leak tests**: No memory leaks detected
- **Reasonable peak_rss_mb**: Memory usage within expected bounds

### Warning Signs

- **High rss_delta_mb**: Operation may need optimization
- **Growing memory in leak tests**: Potential memory leak
- **Increasing allocation_count**: May indicate inefficient object creation

## Benchmark Categories

### 1. Single Document Ingestion

Tests memory usage for ingesting single documents of various sizes:
- Small (1KB)
- Medium (100KB)
- Large (1MB)

Each includes full embedding generation and Qdrant upload.

### 2. Batch Document Ingestion

Tests memory scaling with document count:
- 10 documents (10KB each)
- 100 documents (10KB each)
- 1000 documents (5KB each) - marked `slow`

### 3. Search Operations

Memory usage for search queries:
- Standard vector search (dense vectors)
- Hybrid search (dense + sparse vectors)
- Multiple search iterations to detect accumulation

### 4. Connection Pooling

Memory overhead of creating multiple Qdrant client connections (simulates connection pool behavior).

### 5. Embedding Model

Memory usage of the FastEmbed model:
- Single text embedding
- Batch embedding (20 texts)

### 6. Memory Leak Detection

Long-running operations to detect memory leaks:
- Ingestion loop (50 iterations)
- Search loop (100 iterations)

Both monitor memory growth rate and fail if:
- Ingestion growth exceeds 50%
- Search growth exceeds 30%

## Requirements

### Qdrant Server

Most benchmarks require a running Qdrant server:

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using local installation
qdrant --host localhost --port 6333
```

Tests marked with `@pytest.mark.requires_qdrant` will be skipped if Qdrant is unavailable.

### Embedding Model

Tests without Qdrant dependency (embedding-only tests) don't require Qdrant but still need the embedding model to be downloaded on first run.

## Memory Profiling Tools

The benchmarks use:

### Python Built-in Tools

- **tracemalloc**: Line-by-line Python allocation tracking
- **gc**: Garbage collection control for clean baselines

### External Libraries

- **psutil**: Process-level memory measurements (RSS, VmSize)
- **pytest-benchmark**: Consistent benchmark reporting

## Thresholds

Current memory usage thresholds (assertions):

| Test Category | Threshold |
|---------------|-----------|
| Single doc (1KB) | < 100MB |
| Single doc (100KB) | < 150MB |
| Single doc (1MB) | < 250MB |
| Batch 10 docs | < 200MB |
| Batch 100 docs | < 600MB |
| Batch 1000 docs | < 1.5GB |
| Search operations | < 100MB |
| Hybrid search | < 150MB |
| Connection pooling | < 200MB |
| Single embedding | < 150MB |
| Batch embedding | < 300MB |
| Long-running ops | < 400MB |

Thresholds are deliberately generous to account for:
- Embedding model loading (first-time overhead)
- Python runtime overhead
- Qdrant client overhead
- Operating system variations

## Future: Rust Daemon Profiling

For profiling the Rust daemon, see the documentation at the end of `benchmark_memory_usage.py` which covers:

1. **heaptrack** - Heap profiler with GUI
2. **Valgrind Massif** - Memory profiler
3. **cargo-flamegraph** - Flame graph generation
4. **/proc monitoring** - RSS tracking
5. **jemalloc** - Alternative allocator with profiling

## Continuous Integration

These benchmarks can be integrated into CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run memory benchmarks
  run: |
    uv run pytest tests/benchmarks/benchmark_memory_usage.py \
      --benchmark-only \
      --benchmark-json=memory_benchmark.json
```

## Troubleshooting

### High Memory Usage

If benchmarks fail due to high memory usage:

1. Check for other processes consuming memory
2. Ensure garbage collection is working (tests force `gc.collect()`)
3. Review recent code changes for memory leaks
4. Run leak detection tests in isolation

### Slow Performance

Slow benchmarks may indicate:

1. Embedding model not cached (first run downloads model)
2. Qdrant server under load
3. System memory pressure (swapping)
4. Background processes interfering

### Flaky Results

Memory measurements can vary based on:

1. System load
2. Python garbage collection timing
3. Operating system memory management
4. Other running processes

For more stable results, run benchmarks:
- On a quiet system
- Multiple times (`--benchmark-autosave`)
- With comparison to baseline (`--benchmark-compare`)

## Contributing

When adding new memory-sensitive features:

1. Add corresponding memory benchmarks
2. Set reasonable thresholds based on testing
3. Document expected memory usage
4. Consider adding leak detection if long-running

## References

- [Python tracemalloc documentation](https://docs.python.org/3/library/tracemalloc.html)
- [psutil documentation](https://psutil.readthedocs.io/)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
