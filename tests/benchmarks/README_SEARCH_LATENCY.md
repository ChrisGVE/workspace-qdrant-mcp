# Search Latency Benchmark Guide

## Overview

The search latency benchmark suite (`benchmark_search_latency.py`) measures the performance characteristics of the workspace-qdrant-mcp search system across different search types, query complexities, and result set sizes.

## Running the Benchmarks

### Run all search latency benchmarks

```bash
uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only
```

### Run specific benchmark category

```bash
# Dense search only
uv run pytest tests/benchmarks/benchmark_search_latency.py -k "dense" --benchmark-only

# Sparse search only
uv run pytest tests/benchmarks/benchmark_search_latency.py -k "sparse" --benchmark-only

# Hybrid search only
uv run pytest tests/benchmarks/benchmark_search_latency.py -k "hybrid" --benchmark-only

# Cold start vs warm cache
uv run pytest tests/benchmarks/benchmark_search_latency.py -k "cold or warm" --benchmark-only
```

### Run single benchmark

```bash
uv run pytest tests/benchmarks/benchmark_search_latency.py::test_dense_search_short_query_limit_5 --benchmark-only -v
```

## Benchmark Structure

### Search Types Tested

1. **Dense Search**: Semantic vector search using 384-dimensional embeddings
2. **Sparse Search**: Keyword-based BM25-style search
3. **Hybrid Search**: Combined dense + sparse with multiple fusion methods:
   - RRF (Reciprocal Rank Fusion)
   - Weighted Sum
   - Max Score

### Query Complexities

- **Short queries**: 1-3 words (e.g., "machine learning")
- **Medium queries**: 5-10 words (e.g., "best practices for database query optimization")
- **Long queries**: 20+ words (comprehensive technical descriptions)

### Result Set Sizes

Tested with `limit` values: 5, 10, 20, 50

### Test Collection

- 1000 sample documents with realistic embeddings
- Dense vectors: 384-dimensional (all-MiniLM-L6-v2 model)
- Sparse vectors: BM25-style keyword vectors
- Sample texts covering technical topics (ML, databases, cloud, etc.)

## Interpreting Results

### pytest-benchmark Output

```
Name (time in ms)                        Min     Max    Mean  StdDev  Median     IQR  Outliers     OPS
-------------------------------------------------------------------------------------------------------
test_dense_search_short_query_limit_5  2.67   5.38   3.44    0.67    3.28    0.81    16;4    290.88
```

**Key Metrics:**

- **Min**: Best case latency (milliseconds)
- **Max**: Worst case latency (milliseconds)
- **Mean**: Average latency across all rounds
- **StdDev**: Standard deviation (variability indicator)
- **Median**: Middle value (50th percentile / p50)
- **IQR**: Interquartile range (spread of middle 50%)
- **Outliers**: Tests significantly outside normal distribution
- **OPS**: Operations per second (1000 / Mean)

### Percentile Metrics

The benchmarks also calculate detailed percentiles:

- **p50 (Median)**: 50% of queries complete at or below this latency
- **p95**: 95% of queries complete at or below this latency (typical SLA target)
- **p99**: 99% of queries complete at or below this latency (tail latency)

**Example interpretation:**

```
p50: 3.28ms  - Half of all searches complete in 3.28ms or less
p95: 4.85ms  - 95% of searches complete in 4.85ms or less
p99: 5.20ms  - 99% of searches complete in 5.20ms or less
```

### Performance Baselines

Expected latency ranges on reference hardware (2020 MacBook Pro, local Qdrant):

| Search Type | Query Complexity | Limit | Expected p50 | Expected p95 |
|-------------|------------------|-------|--------------|--------------|
| Dense       | Short            | 5     | 2-4ms        | 4-6ms        |
| Dense       | Medium           | 20    | 3-5ms        | 5-8ms        |
| Dense       | Long             | 50    | 4-7ms        | 7-12ms       |
| Sparse      | Short            | 5     | 2-4ms        | 4-6ms        |
| Sparse      | Medium           | 10    | 3-5ms        | 5-8ms        |
| Hybrid RRF  | Short            | 10    | 4-6ms        | 6-10ms       |
| Hybrid RRF  | Medium           | 20    | 5-8ms        | 8-14ms       |

**Note**: Actual performance depends on hardware, network latency to Qdrant, collection size, and system load.

### Cold Start vs Warm Cache

- **Cold start**: Fresh engine instance, no caching, includes initialization overhead
- **Warm cache**: Reused engine instance, benefits from internal caching

Expected differences:
- Cold start: ~20-50% slower than warm cache
- After 2-3 queries, performance stabilizes to warm cache levels

## Performance Analysis

### What to Look For

#### 1. Latency Distribution

- **Low StdDev**: Consistent, predictable performance
- **High StdDev**: Variable performance, investigate outliers
- **Large IQR**: Wide spread in middle 50% of results

#### 2. Tail Latency (p95, p99)

- Critical for user experience (affects slowest requests)
- p99 > 2× p50 indicates potential optimization opportunities
- Monitor for degradation over time

#### 3. Fusion Method Comparison

Compare RRF, weighted_sum, and max_score for same query:

- RRF: Usually slower but better result quality
- Weighted Sum: Faster, good when score ranges similar
- Max Score: Fastest, best for high-precision scenarios

#### 4. Query Complexity Impact

Expected scaling:
- Short → Medium: ~20-40% increase
- Medium → Long: ~30-50% increase
- More than 100% increase suggests inefficiency

#### 5. Result Limit Scaling

Expected scaling:
- limit=5 → limit=10: ~10-20% increase
- limit=10 → limit=50: ~30-50% increase
- Linear or sublinear scaling is ideal

### Troubleshooting Slow Searches

If latencies exceed expected baselines:

1. **Check Qdrant connection**: Network latency to Qdrant server
2. **Collection size**: Larger collections = slower searches (consider sharding)
3. **System resources**: CPU/memory contention from other processes
4. **Qdrant configuration**: Check HNSW index parameters
5. **Embedding model**: Ensure models are properly loaded and cached

### Optimization Opportunities

Look for these patterns:

- **High p99/p50 ratio** (>2.5): Investigate outliers and caching
- **Cold start >> warm cache**: Optimize initialization or add warming phase
- **Hybrid >> Dense + Sparse**: Fusion overhead, consider simpler methods
- **Limit scaling non-linear**: Index optimization needed

## Continuous Monitoring

### Regression Detection

Run benchmarks regularly to detect performance regressions:

```bash
# Save baseline
uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only --benchmark-save=baseline

# Compare against baseline later
uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only --benchmark-compare=baseline
```

### CI/CD Integration

Add to CI pipeline:

```bash
# Fail if performance degrades >10%
uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

## Advanced Usage

### Custom Percentiles

Modify `calculate_percentiles()` function to add custom percentiles:

```python
"p90_ms": statistics.quantiles(data_ms, n=100)[89],  # 90th percentile
"p999_ms": statistics.quantiles(data_ms, n=1000)[998],  # 99.9th percentile
```

### Larger Test Collections

Modify fixture to test with more documents:

```python
asyncio.run(
    SearchBenchmarkFixtures.create_test_collection(
        qdrant_client,
        benchmark_collection_name,
        num_documents=10000,  # 10x larger
    )
)
```

### Different Query Patterns

Add fixtures for domain-specific queries:

```python
@pytest.fixture(scope="module")
def code_query_embeddings():
    """Embeddings for code-specific query."""
    return asyncio.run(
        SearchBenchmarkFixtures.generate_query_embeddings(
            "implement asynchronous error handling in python"
        )
    )
```

## Related Benchmarks

- `benchmark_file_ingestion.py`: Document parsing throughput
- `benchmark_example.py`: Simple benchmark examples

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Qdrant search performance](https://qdrant.tech/documentation/guides/search/)
- [HNSW index tuning](https://qdrant.tech/documentation/concepts/indexing/#hnsw-index)
