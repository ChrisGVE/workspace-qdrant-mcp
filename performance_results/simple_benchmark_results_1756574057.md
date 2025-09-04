# Workspace-Qdrant-MCP Performance Baseline Report
============================================================

**Test Run:** 2025-08-30 19:14:17
**Duration:** 0.9 seconds
**Qdrant Host:** localhost:6333
**Collections Tested:** 3

## Collection Performance Results

### test-collection-5
- **Points:** 19
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 2.24ms avg (446.7 QPS)
- with_payload: 2.55ms avg (391.6 QPS)
- with_vectors: 3.48ms avg (287.1 QPS)
- large_limit: 1.88ms avg (531.6 QPS)

### bench_project_1000
- **Points:** 1,764
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 2.05ms avg (487.7 QPS)
- with_payload: 2.79ms avg (358.4 QPS)
- with_vectors: 5.31ms avg (188.3 QPS)
- large_limit: 2.22ms avg (451.2 QPS)

### quick-test
- **Points:** 1
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.81ms avg (551.2 QPS)
- with_payload: 2.11ms avg (475.0 QPS)
- with_vectors: 2.16ms avg (462.4 QPS)
- large_limit: 2.03ms avg (493.7 QPS)

## Concurrent Search Performance

### test-collection-5
- 1 concurrent: 300.4 QPS (100.0% success)
- 5 concurrent: 527.4 QPS (100.0% success)
- 10 concurrent: 575.6 QPS (100.0% success)
- 20 concurrent: 552.4 QPS (100.0% success)

### bench_project_1000
- 1 concurrent: 349.0 QPS (100.0% success)
- 5 concurrent: 627.7 QPS (100.0% success)
- 10 concurrent: 641.1 QPS (100.0% success)
- 20 concurrent: 649.6 QPS (100.0% success)

### quick-test
- 1 concurrent: 451.6 QPS (100.0% success)
- 5 concurrent: 705.9 QPS (100.0% success)
- 10 concurrent: 700.7 QPS (100.0% success)

## Performance Summary

- **Average Search Time:** 2.55ms
- **Fastest Search:** 1.81ms
- **Slowest Search:** 5.31ms
- **Average Throughput:** 427.1 QPS
- **Maximum Throughput:** 551.2 QPS
- **Fastest Collection:** quick-test (1.81ms)

## Status

**Overall Performance:** 🟢 Excellent (< 50ms average)

✅ Performance baseline established successfully!