# Workspace-Qdrant-MCP Performance Baseline Report
============================================================

**Test Run:** 2025-08-30 19:14:35
**Duration:** 0.7 seconds
**Qdrant Host:** localhost:6333
**Collections Tested:** 3

## Collection Performance Results

### test-collection-5
- **Points:** 19
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 2.13ms avg (469.8 QPS)
- with_payload: 2.34ms avg (427.5 QPS)
- with_vectors: 3.21ms avg (311.4 QPS)
- large_limit: 1.66ms avg (600.7 QPS)

### bench_project_1000
- **Points:** 1,764
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.69ms avg (590.3 QPS)
- with_payload: 1.96ms avg (510.1 QPS)
- with_vectors: 3.10ms avg (322.6 QPS)
- large_limit: 1.84ms avg (543.7 QPS)

### quick-test
- **Points:** 1
- **Has Vectors:** True
- **Vector Info:** {'dense': 384}

**Search Performance:**
- basic_search: 1.62ms avg (617.9 QPS)
- with_payload: 1.66ms avg (604.0 QPS)
- with_vectors: 1.90ms avg (525.8 QPS)
- large_limit: 1.72ms avg (582.0 QPS)

## Concurrent Search Performance

### test-collection-5
- 1 concurrent: 376.6 QPS (100.0% success)
- 5 concurrent: 602.8 QPS (100.0% success)
- 10 concurrent: 596.9 QPS (100.0% success)
- 20 concurrent: 604.8 QPS (100.0% success)

### bench_project_1000
- 1 concurrent: 450.5 QPS (100.0% success)
- 5 concurrent: 691.1 QPS (100.0% success)
- 10 concurrent: 678.3 QPS (100.0% success)
- 20 concurrent: 617.1 QPS (100.0% success)

## Performance Summary

- **Average Search Time:** 2.07ms
- **Fastest Search:** 1.62ms
- **Slowest Search:** 3.21ms
- **Average Throughput:** 508.8 QPS
- **Maximum Throughput:** 617.9 QPS
- **Fastest Collection:** quick-test (1.62ms)

## Status

**Overall Performance:** 🟢 Excellent (< 50ms average)

✅ Performance baseline established successfully!