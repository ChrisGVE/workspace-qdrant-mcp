# Workspace-Qdrant-MCP Performance Baseline Report
============================================================

**Test Run:** 2025-08-30 07:15:32
**Duration:** 0.4 seconds
**Qdrant Host:** localhost:6333
**Collections Tested:** 3

## Collection Performance Results

### test-collection-5
- **Points:** 19
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 3.62ms avg (276.3 QPS)
- with_payload: 2.75ms avg (363.9 QPS)
- with_vectors: 3.22ms avg (311.0 QPS)
- large_limit: 1.59ms avg (630.5 QPS)

### bench_project_1000
- **Points:** 1,764
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.70ms avg (587.2 QPS)
- with_payload: 2.03ms avg (491.7 QPS)
- with_vectors: 3.04ms avg (329.4 QPS)
- large_limit: 1.84ms avg (542.4 QPS)

### quick-test
- **Points:** 1
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.49ms avg (669.1 QPS)
- with_payload: 1.56ms avg (642.1 QPS)
- with_vectors: 1.65ms avg (605.5 QPS)
- large_limit: 1.71ms avg (584.5 QPS)

## Concurrent Search Performance

### test-collection-5

### quick-test

## Performance Summary

- **Average Search Time:** 2.18ms
- **Fastest Search:** 1.49ms
- **Slowest Search:** 3.62ms
- **Average Throughput:** 502.8 QPS
- **Maximum Throughput:** 669.1 QPS
- **Fastest Collection:** quick-test (1.49ms)

## Status

**Overall Performance:** 🟢 Excellent (< 50ms average)

✅ Performance baseline established successfully!