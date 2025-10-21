# gRPC Communication Overhead Benchmarks

Comprehensive benchmarks measuring gRPC communication overhead between Python client and Rust daemon in the workspace-qdrant-mcp system.

## Overview

These benchmarks measure various aspects of gRPC performance:

1. **Connection Establishment** - Time to establish gRPC channel
2. **Unary RPC Latency** - Round-trip time for single request/response
3. **Batch Throughput** - Requests per second for sequential and concurrent operations
4. **Serialization Overhead** - Protobuf encoding/decoding performance
5. **Message Size Impact** - Latency scaling with payload and metadata size
6. **Retry/Circuit Breaker Overhead** - Cost of resilience infrastructure

## Prerequisites

### 1. Start Qdrant Server

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using local installation
qdrant --port 6333
```

### 2. Start Rust Daemon

```bash
# Install and start the daemon service
uv run wqm service install
uv run wqm service start

# Verify daemon is running
uv run wqm service status
```

### 3. Install Python Dependencies

```bash
# Install development dependencies
uv sync --dev
```

## Running Benchmarks

### Run All gRPC Benchmarks

```bash
# Basic run
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only

# With detailed statistics
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-verbose

# Save results for comparison
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-save=grpc_baseline
```

### Run Specific Benchmark Categories

```bash
# Connection benchmarks only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py::test_connection_establishment --benchmark-only

# Latency benchmarks only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "latency" --benchmark-only

# Throughput benchmarks only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "throughput" --benchmark-only

# Serialization benchmarks only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "serialization" --benchmark-only

# Specific payload size
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "1kb" --benchmark-only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "100kb" --benchmark-only
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "1mb" --benchmark-only
```

### Compare with Previous Results

```bash
# Compare with saved baseline
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-compare=grpc_baseline

# Compare and fail if performance regressed
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-compare=grpc_baseline --benchmark-compare-fail=mean:10%
```

### Generate Performance Report

```bash
# Generate histogram
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-histogram=grpc_histogram

# Export to JSON
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-json=grpc_results.json

# Export to CSV (requires plugin)
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-csv=grpc_results.csv
```

## Benchmark Categories Explained

### 1. Connection Establishment

**Test:** `test_connection_establishment`

**Measures:** Time to create new gRPC channel and verify it's ready.

**What it tells you:**
- One-time startup cost when client initializes
- Impact of connection pooling configuration
- Network latency to daemon

**Expected Performance:**
- Local daemon: 5-50ms
- Network daemon: 50-200ms (depends on network)

**Interpretation:**
- Higher values may indicate network issues or daemon startup problems
- Connection pooling amortizes this cost across requests

---

### 2. Unary RPC Latency

#### Health Check (`test_health_check_latency`)

**Measures:** Baseline gRPC overhead with minimal payload.

**What it tells you:**
- Minimum achievable latency for any RPC
- Network round-trip time + gRPC overhead
- Baseline for comparing other operations

**Expected Performance:**
- Local daemon: 0.5-5ms
- Network daemon: 10-50ms

#### Status & Metrics (`test_get_status_latency`, `test_get_metrics_latency`)

**Measures:** Latency for RPCs returning structured data.

**What it tells you:**
- Impact of response payload size
- Serialization overhead for complex types
- Daemon processing time

**Expected Performance:**
- Local daemon: 1-10ms
- Network daemon: 15-60ms

---

### 3. Text Ingestion by Payload Size

#### 1KB Payload (`test_ingest_text_1kb_latency`)

**Measures:** End-to-end latency for small documents.

**Expected Performance:** 10-100ms

**Use Cases:**
- Short notes or snippets
- Configuration files
- Small code files

#### 100KB Payload (`test_ingest_text_100kb_latency`)

**Measures:** Medium-sized document processing.

**Expected Performance:** 50-500ms

**Use Cases:**
- Medium code files
- Documentation pages
- Articles

#### 1MB Payload (`test_ingest_text_1mb_latency`)

**Measures:** Large document handling and throughput limits.

**Expected Performance:** 200ms-2s

**Use Cases:**
- Large source files
- Concatenated documentation
- Books or long articles

**What to watch for:**
- Non-linear scaling indicates serialization bottleneck
- Timeout errors may require adjusting gRPC message size limits
- Large variance suggests resource contention

---

### 4. Batch Throughput

#### Sequential Throughput (`test_sequential_health_checks_throughput`)

**Measures:** Requests per second for sequential unary calls.

**Expected Performance:** 50-200 RPS (local), 10-50 RPS (network)

**What it tells you:**
- Single-threaded client throughput
- Latency-bound performance ceiling

#### Concurrent Throughput (`test_concurrent_health_checks_throughput`)

**Measures:** Concurrent request handling capability.

**Expected Performance:** 200-1000 RPS (local), 50-200 RPS (network)

**What it tells you:**
- Benefit of async I/O and connection pooling
- Server's ability to handle parallel requests
- Connection pool effectiveness

**Comparison:**
- Concurrent should be 3-10x faster than sequential
- Lower ratio indicates connection pool or server limits

#### Batch Ingestion (`test_batch_ingest_small_documents_throughput`)

**Measures:** Documents per second for realistic batch workload.

**Expected Performance:** 10-100 documents/second

**What it tells you:**
- Real-world batch processing capability
- Impact of document processing on throughput

---

### 5. Serialization Overhead

**Tests:** `test_protobuf_serialization_{1kb,100kb,1mb}`

**Measures:** Pure protobuf encoding/decoding time (no network).

**Expected Performance:**
- 1KB: 0.01-0.1ms
- 100KB: 0.5-5ms
- 1MB: 5-50ms

**What it tells you:**
- CPU-bound serialization cost
- Scaling of protobuf with message size
- Proportion of latency attributable to serialization

**Analysis:**
- Compare serialization time to total RPC latency
- If serialization is >50% of latency, consider:
  - Reducing message sizes
  - Using compression
  - Batching smaller messages

---

### 6. Collection Management

**Tests:** `test_create_collection_latency`, `test_alias_operations_latency`

**Measures:** Latency for collection and alias operations.

**Expected Performance:**
- Collection creation: 50-500ms (creates Qdrant collection)
- Alias operations: 5-50ms (metadata only)

**What it tells you:**
- Cost of collection management operations
- Database operation overhead via gRPC

---

### 7. Metadata Size Impact

**Tests:** `test_metadata_size_impact_{small,large}`

**Measures:** Impact of metadata size on ingestion latency.

**Expected Performance:**
- Small metadata (5 fields): baseline + 1-5ms
- Large metadata (50 fields): baseline + 5-20ms

**What it tells you:**
- Metadata serialization overhead
- Whether to optimize metadata size

**Interpretation:**
- >20ms difference suggests metadata optimization needed
- Consider limiting metadata fields for performance-critical paths

---

### 8. Retry and Circuit Breaker

**Test:** `test_successful_request_with_retry_enabled`

**Measures:** Overhead of retry infrastructure on successful requests.

**Expected Performance:** Should match health check baseline (±10%)

**What it tells you:**
- Cost of resilience features when not triggered
- Whether retry logic adds measurable overhead

**Acceptable Overhead:** <10% of baseline latency

## Interpreting Results

### Understanding pytest-benchmark Output

```
--------------------------------------------------------------------------------------- benchmark: 18 tests ---------------------------------------------------------------------------------------
Name (time in ms)                                     Min                 Max                Mean            StdDev              Median               IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_health_check_latency                          1.2345           2.3456           1.4567        0.1234           1.4321        0.0987          12;5  686.4578           100           1
test_ingest_text_1kb_latency                      12.3456          23.4567          15.6789        1.2345          15.4321        0.9876           8;3   63.7834            50           1
```

#### Key Metrics

- **Min:** Best-case performance (ideal conditions)
- **Max:** Worst-case performance (may indicate GC, context switching)
- **Mean:** Average performance (primary metric for comparison)
- **StdDev:** Consistency (lower is better, <10% of mean is good)
- **Median:** Typical performance (less affected by outliers)
- **IQR:** Interquartile range (measures spread of middle 50%)
- **Outliers:** Number of extreme values (high count may indicate issues)
- **OPS:** Operations per second (1000 / mean_ms)

### Performance Targets

#### Excellent Performance
- Health check: <2ms
- 1KB ingestion: <20ms
- 100KB ingestion: <100ms
- 1MB ingestion: <500ms
- Sequential throughput: >100 RPS
- Concurrent throughput: >500 RPS

#### Acceptable Performance
- Health check: <5ms
- 1KB ingestion: <50ms
- 100KB ingestion: <250ms
- 1MB ingestion: <1000ms
- Sequential throughput: >50 RPS
- Concurrent throughput: >200 RPS

#### Needs Investigation
- Health check: >10ms
- High standard deviation (>20% of mean)
- Large outlier count (>10% of rounds)
- Non-linear scaling with payload size
- Concurrent throughput <2x sequential

### Common Issues and Solutions

#### High Connection Establishment Time (>100ms)

**Possible Causes:**
- Daemon not running locally
- Network latency
- DNS resolution delays
- Firewall/security software

**Solutions:**
- Verify daemon is running: `wqm service status`
- Use IP address instead of hostname
- Check firewall rules
- Run daemon locally for development

#### High RPC Latency (>50ms for health checks)

**Possible Causes:**
- Daemon under heavy load
- Resource contention (CPU, memory)
- Network congestion
- gRPC configuration issues

**Solutions:**
- Check daemon CPU/memory usage
- Reduce concurrent load
- Review connection pool configuration
- Increase daemon resources

#### Poor Concurrent Throughput (<3x sequential)

**Possible Causes:**
- Connection pool too small
- Server thread pool exhausted
- CPU bottleneck
- Lock contention

**Solutions:**
- Increase `pool_size` in ConnectionConfig
- Review daemon thread pool settings
- Profile daemon CPU usage
- Check for lock contention in logs

#### High Serialization Overhead (>50% of total latency)

**Possible Causes:**
- Very large messages
- Inefficient message structure
- CPU-bound serialization

**Solutions:**
- Reduce message sizes
- Enable compression
- Batch smaller messages
- Consider different serialization format for bulk data

#### Inconsistent Results (high StdDev)

**Possible Causes:**
- Garbage collection
- Background processes
- Thermal throttling
- Resource contention

**Solutions:**
- Run benchmarks on idle system
- Increase warmup rounds
- Use larger sample sizes
- Profile system resource usage

## Comparing with Direct Qdrant Calls

To understand gRPC overhead, compare these results with direct Qdrant client benchmarks:

```bash
# Run Qdrant direct benchmarks (if implemented)
uv run pytest tests/benchmarks/benchmark_qdrant_direct.py --benchmark-only
```

**Analysis:**
- Calculate overhead: `grpc_latency - direct_latency`
- Acceptable overhead: <10ms for local calls
- If overhead >20ms, investigate gRPC configuration

## Continuous Integration

Add to CI pipeline to catch performance regressions:

```yaml
# .github/workflows/benchmarks.yml
- name: Run gRPC benchmarks
  run: |
    # Start dependencies
    docker run -d -p 6333:6333 qdrant/qdrant
    uv run wqm service start

    # Run benchmarks with regression check
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py \
      --benchmark-only \
      --benchmark-compare=main \
      --benchmark-compare-fail=mean:10%
```

## Advanced Analysis

### Latency Percentiles

Use `--benchmark-histogram` to visualize latency distribution:

```bash
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py \
  --benchmark-only \
  --benchmark-histogram=grpc_histogram
```

Check P95 and P99 latencies for tail performance.

### Throughput vs. Latency Trade-off

Analyze the relationship between batch size and throughput:

```python
# Small batches: Lower latency, lower throughput
# Large batches: Higher latency, higher throughput
# Find the sweet spot for your workload
```

### Resource Utilization

Monitor system resources during benchmarks:

```bash
# Terminal 1: Run benchmarks
uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only

# Terminal 2: Monitor daemon
top -p $(pgrep -f workspace-qdrant-daemon)

# Or use htop for better visualization
htop -p $(pgrep -f workspace-qdrant-daemon)
```

## Troubleshooting

### Benchmarks Fail to Run

```
ERROR: Daemon not available
```

**Solution:**
1. Check daemon status: `wqm service status`
2. Start daemon: `wqm service start`
3. Check logs: `wqm service logs`
4. Verify Qdrant is running: `curl http://localhost:6333/healthz`

### Timeout Errors

```
DaemonTimeoutError: Operation timed out after 60s
```

**Solution:**
1. Increase timeout in benchmark code
2. Check daemon health
3. Reduce payload sizes
4. Review daemon resource limits

### Inconsistent Results

**Solution:**
1. Run benchmarks on idle system
2. Increase number of rounds: `--benchmark-min-rounds=100`
3. Use warmup: `--benchmark-warmup=on`
4. Disable CPU frequency scaling

## Future Enhancements

Potential additions to gRPC benchmarks:

1. **Streaming RPC benchmarks** (if implemented)
2. **Compression overhead** (gzip vs. none)
3. **TLS/SSL overhead** (secure vs. insecure channels)
4. **Different message sizes** (more granular size steps)
5. **Error handling overhead** (retry logic, timeouts)
6. **Connection pool scaling** (varying pool sizes)
7. **Multi-client scenarios** (concurrent clients)

## References

- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Protobuf Performance Guide](https://developers.google.com/protocol-buffers/docs/techniques)
- [Qdrant Performance Tuning](https://qdrant.tech/documentation/guides/administration/)
