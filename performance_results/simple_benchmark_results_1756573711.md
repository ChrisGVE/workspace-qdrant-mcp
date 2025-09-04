# Workspace-Qdrant-MCP Performance Baseline Report
============================================================

**Test Run:** 2025-08-30 19:08:31
**Duration:** 0.5 seconds
**Qdrant Host:** localhost:6333
**Collections Tested:** 3

## Collection Performance Results

### test-collection-5
- **Points:** 19
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.71ms avg (584.2 QPS)
- with_payload: 2.18ms avg (458.6 QPS)
- with_vectors: 2.98ms avg (335.7 QPS)
- large_limit: 1.72ms avg (582.6 QPS)

### bench_project_1000
- **Points:** 1,764
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.64ms avg (611.5 QPS)
- with_payload: 1.78ms avg (560.5 QPS)
- with_vectors: 3.01ms avg (331.9 QPS)
- large_limit: 1.85ms avg (541.4 QPS)

### quick-test
- **Points:** 1
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.67ms avg (597.9 QPS)
- with_payload: 1.44ms avg (693.4 QPS)
- with_vectors: 1.54ms avg (649.1 QPS)
- large_limit: 1.44ms avg (692.9 QPS)

## Concurrent Search Performance

### test-collection-5
- 1 concurrent: 461.1 QPS (100.0% success)
- 5 concurrent: 681.0 QPS (100.0% success)
- 10 concurrent: 680.9 QPS (100.0% success)
- 20 concurrent: 674.2 QPS (100.0% success)

## Performance Summary

- **Average Search Time:** 1.91ms
- **Fastest Search:** 1.44ms
- **Slowest Search:** 3.01ms
- **Average Throughput:** 553.3 QPS
- **Maximum Throughput:** 693.4 QPS
- **Fastest Collection:** quick-test (1.44ms)

## Status

**Overall Performance:** 🟢 Excellent (< 50ms average)

✅ Performance baseline established successfully!