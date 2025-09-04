# Workspace-Qdrant-MCP Performance Baseline Report
============================================================

**Test Run:** 2025-08-30 19:07:55
**Duration:** 0.4 seconds
**Qdrant Host:** localhost:6333
**Collections Tested:** 3

## Collection Performance Results

### test-collection-5
- **Points:** 19
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 2.37ms avg (421.3 QPS)
- with_payload: 2.98ms avg (336.1 QPS)
- with_vectors: 4.01ms avg (249.3 QPS)
- large_limit: 1.96ms avg (511.4 QPS)

### bench_project_1000
- **Points:** 1,764
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 2.27ms avg (439.8 QPS)
- with_payload: 2.56ms avg (390.1 QPS)
- with_vectors: 3.67ms avg (272.8 QPS)
- large_limit: 2.39ms avg (418.5 QPS)

### quick-test
- **Points:** 1
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.84ms avg (544.3 QPS)
- with_payload: 1.75ms avg (571.6 QPS)
- with_vectors: 1.88ms avg (532.1 QPS)
- large_limit: 1.73ms avg (579.0 QPS)

## Concurrent Search Performance

### quick-test

## Performance Summary

- **Average Search Time:** 2.45ms
- **Fastest Search:** 1.73ms
- **Slowest Search:** 4.01ms
- **Average Throughput:** 438.9 QPS
- **Maximum Throughput:** 579.0 QPS
- **Fastest Collection:** quick-test (1.73ms)

## Status

**Overall Performance:** 🟢 Excellent (< 50ms average)

✅ Performance baseline established successfully!