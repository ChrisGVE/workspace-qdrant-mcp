# Database Query Performance Benchmarks

This document describes the database query performance benchmarks and how to interpret results.

## Overview

The database benchmarks measure query performance for two critical database systems:

1. **SQLite State Manager** - Tracks ingestion progress, watch folders, and processing state
2. **Qdrant Vector Database** - Stores and searches vector embeddings

## Running Benchmarks

### Run All Database Benchmarks

```bash
uv run pytest tests/benchmarks/benchmark_database_queries.py --benchmark-only
```

### Run SQLite Benchmarks Only

```bash
uv run pytest tests/benchmarks/benchmark_database_queries.py::test_watch_folder -k "watch_folder" --benchmark-only
```

### Run Qdrant Benchmarks Only

```bash
uv run pytest tests/benchmarks/benchmark_database_queries.py::test_qdrant -k "qdrant" --benchmark-only
```

### Run Specific Category

```bash
# Watch folder operations
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "watch_folder" --benchmark-only

# Ingestion queue operations
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "ingestion_queue" --benchmark-only

# File processing operations
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "file_processing" --benchmark-only

# Collection operations
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "collection" --benchmark-only

# Search operations
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "search" --benchmark-only

# Comparison benchmarks
uv run pytest tests/benchmarks/benchmark_database_queries.py -k "comparison" --benchmark-only
```

### Generate HTML Report

```bash
uv run pytest tests/benchmarks/benchmark_database_queries.py --benchmark-only --benchmark-autosave --benchmark-save-data
```

View report:
```bash
uv run pytest-benchmark compare
```

## Benchmark Categories

### SQLite State Manager Benchmarks

#### Watch Folder Operations
- **save_single_small**: Single watch folder config save (10 records)
- **save_batch_small/medium**: Batch saves at different scales
- **get_single**: Retrieve single config by ID (indexed lookup)
- **list_all_small/medium/large**: List all configs (10/100/1000 records)
- **remove**: Delete watch folder config

**Key Metrics:**
- Single save latency should be < 5ms
- Batch operations should benefit from transaction reuse
- Indexed lookups should be < 1ms regardless of dataset size
- List operations scale linearly with dataset size

#### Ingestion Queue Operations
- **enqueue_single**: Add single file to queue
- **enqueue_batch_small/medium**: Batch enqueue operations
- **dequeue_small/medium**: Retrieve items from queue by priority
- **get_depth**: Count total items in queue
- **remove_single**: Remove specific item from queue

**Key Metrics:**
- Enqueue latency should be < 2ms per item
- Dequeue should use priority index effectively (< 5ms for batch of 50)
- Queue depth query should be < 1ms (uses COUNT with index)
- UNIQUE constraint handling should add minimal overhead

#### File Processing State Operations
- **start_single/batch**: Begin processing for files
- **complete_single**: Mark file processing complete
- **get_status**: Retrieve processing status for file
- **get_by_status**: Query files by processing status

**Key Metrics:**
- State transitions should be < 5ms
- Status queries should benefit from indexed status column
- Batch operations should outperform sequential single operations

#### Multi-Component Operations
- **record_event**: Log system events
- **get_events**: Retrieve events with filtering
- **record_search_operation**: Log search operations
- **get_search_history**: Retrieve search history

**Key Metrics:**
- Event recording should be < 3ms
- Filtered queries should use appropriate indexes
- History retrieval should support pagination efficiently

### Qdrant Vector Database Benchmarks

#### Collection Operations
- **collection_create**: Create new collection
- **collection_list**: List all collections
- **collection_info**: Get collection metadata

**Key Metrics:**
- Collection creation includes schema setup (expect 50-200ms)
- Listing should be < 10ms for < 100 collections
- Collection info should be < 5ms

#### Point Insertion Operations
- **insert_single**: Insert one point
- **insert_batch_small/medium/large**: Batch inserts (10/100/1000 points)

**Key Metrics:**
- Single insert: 5-20ms (includes network overhead)
- Batch small (10): 20-50ms total
- Batch medium (100): 100-300ms total
- Batch large (1000): 1-3 seconds total (chunked)
- Batch operations should show significant per-point improvement

#### Search Operations
- **search_dense_only_small/medium**: Dense vector search at different scales
- **search_with_filter**: Search with metadata filtering
- **search_large_limit**: Search returning 50 results

**Key Metrics:**
- Dense search on 10 points: < 10ms
- Dense search on 100 points: 10-30ms
- Dense search on 1000+ points: 30-100ms (HNSW index efficiency)
- Filtered search adds 5-15ms overhead
- Large result limits add minimal overhead (HNSW returns sorted)

#### Point Retrieval Operations
- **retrieve_single_point**: Get point by ID
- **retrieve_batch_points**: Get multiple points by ID
- **scroll_points**: Paginate through points

**Key Metrics:**
- Single retrieval: < 5ms
- Batch retrieval (20 points): 10-20ms
- Scroll operation: 10-30ms per page

## Data Volumes

Benchmarks use three standard dataset sizes:

- **Small**: 10 records (baseline performance)
- **Medium**: 100 records (typical workload)
- **Large**: 1000 records (stress testing)

## Interpreting Results

### Understanding Benchmark Output

```
test_watch_folder_save_single_small     5.2ms    (min: 4.8ms, max: 6.1ms)
test_watch_folder_save_batch_small     15.3ms    (min: 14.1ms, max: 17.2ms)
```

**Key columns:**
- **Mean**: Average execution time across all iterations
- **Min**: Best case performance
- **Max**: Worst case performance
- **StdDev**: Performance consistency (lower is better)
- **Iterations**: Number of times the benchmark ran

### Performance Targets

#### SQLite Operations
- **Single writes**: < 5ms
- **Batch writes (10)**: < 20ms (< 2ms per item)
- **Indexed lookups**: < 1ms
- **Full table scans (100 rows)**: < 10ms
- **Full table scans (1000 rows)**: < 50ms

#### Qdrant Operations
- **Collection management**: < 100ms
- **Single point insert**: < 20ms
- **Batch insert (100 points)**: < 300ms
- **Search (< 100 points)**: < 30ms
- **Search (1000+ points)**: < 100ms
- **Point retrieval**: < 10ms

### Red Flags

Watch for these performance issues:

1. **Non-linear scaling**: Operations that don't scale linearly with data size
2. **High variance**: Large differences between min/max times (indicates inconsistency)
3. **Batch slower than single**: Batch operations should always be faster per-item
4. **Index not helping**: Indexed lookups should be constant time regardless of dataset size
5. **Network overhead dominance**: Qdrant operations showing excessive latency

### Comparison Benchmarks

The comparison benchmarks specifically test:

- **Batch vs Single**: Verify batch operations outperform sequential single operations
- **Indexed vs Non-indexed**: Verify indexes improve query performance
- **Priority ordering**: Verify priority-based queries use indexes effectively

## Optimization Opportunities

### SQLite

1. **Slow single writes**: Check WAL mode enabled, review transaction handling
2. **Slow batch operations**: Ensure using transactions, increase cache_size pragma
3. **Slow lookups**: Verify indexes exist and are being used (EXPLAIN QUERY PLAN)
4. **Slow scans**: Consider adding indexes, review query filters

### Qdrant

1. **Slow inserts**: Use batch operations, increase batch size up to 100-200 points
2. **Slow searches**: Review HNSW parameters (ef_construct, m), check collection size
3. **Filter overhead**: Index frequently filtered fields in payload
4. **Network latency**: Consider connection pooling, request batching

## Running with Docker

If Qdrant benchmarks fail with "requires_qdrant" marker:

```bash
# Start Qdrant container
docker run -p 6333:6333 qdrant/qdrant

# Run benchmarks
uv run pytest tests/benchmarks/benchmark_database_queries.py --benchmark-only
```

Alternatively, use the isolated container fixtures (automatic):

```bash
# Benchmarks will start/stop containers automatically
uv run pytest tests/benchmarks/benchmark_database_queries.py --benchmark-only -m requires_qdrant
```

## Notes

### SQLite Considerations

- Benchmarks use temporary databases with WAL mode
- Each test gets isolated database instance
- Indexes are created during schema setup
- PRAGMA optimizations applied (cache_size, temp_store, mmap_size)

### Qdrant Considerations

- Benchmarks use isolated Qdrant containers per test
- Collections are created fresh for each benchmark
- Vector dimensions: 384 (sentence-transformers/all-MiniLM-L6-v2)
- Distance metric: COSINE
- Network latency included in measurements

### Benchmark Overhead

pytest-benchmark adds ~1-2ms overhead for each benchmark call. This is consistent across all tests and does not affect relative performance comparisons.

## See Also

- [Benchmark Infrastructure](./benchmark_example.py) - Example benchmark patterns
- [Search Latency Benchmarks](./benchmark_search_latency.py) - End-to-end search performance
- [Memory Usage Benchmarks](./benchmark_memory_usage.py) - Memory profiling
