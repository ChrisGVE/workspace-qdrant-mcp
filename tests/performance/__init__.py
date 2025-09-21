"""
Performance Testing Suite for workspace-qdrant-mcp.

This package provides comprehensive performance testing capabilities including:

- Core operation benchmarking (document processing, vector search, hybrid search)
- MCP tool response time validation across all 30+ tools
- Memory usage profiling and leak detection
- Concurrent operation performance testing
- Scaling behavior analysis with varying dataset sizes
- Load testing with k6 integration (normal, stress, spike, soak, volume testing)
- Performance regression detection and alerting
- Real-time performance monitoring and reporting

Test Execution:
    Run all performance tests:
    ```bash
    pytest tests/performance/ -m performance
    ```

    Run specific test categories:
    ```bash
    pytest tests/performance/ -m benchmark          # Pytest-benchmark tests
    pytest tests/performance/ -m load_testing       # K6 load tests
    pytest tests/performance/ -m slow               # Long-running tests
    ```

    Run with performance report:
    ```bash
    pytest tests/performance/ --benchmark-only --benchmark-sort=mean
    ```

    Generate performance comparison:
    ```bash
    pytest tests/performance/ --benchmark-compare=baseline.json
    ```

Performance Metrics:
    - Response Times: Document processing <500ms, Vector search <100ms, MCP tools <200ms
    - Memory Usage: <50MB base overhead, <5MB per 1000 documents
    - Concurrency: Linear scaling up to 10 concurrent operations
    - Error Rates: <1% under normal load, <5% under stress
    - Scaling: Sub-linear (O(log n)) for search operations

Configuration:
    Performance test configuration can be customized via environment variables:
    - PERF_TEST_TIMEOUT: Test timeout in seconds (default: 300)
    - PERF_TEST_ITERATIONS: Number of benchmark iterations (default: 20)
    - PERF_TEST_QDRANT_URL: Qdrant server URL for integration tests
    - PERF_TEST_K6_PATH: Path to k6 binary for load testing
    - PERF_TEST_REPORT_DIR: Directory for performance reports

Dependencies:
    - pytest-benchmark: For microbenchmarking and performance regression detection
    - psutil: For system resource monitoring
    - k6: For load testing (optional, mock results if not available)
    - tracemalloc: For detailed memory profiling
    - testcontainers: For isolated Qdrant instances in testing
"""

__version__ = "0.2.1dev1"
__author__ = "Christian C. Berclaz"

# Performance test configuration
PERFORMANCE_TEST_CONFIG = {
    "default_timeout": 300,  # 5 minutes
    "default_iterations": 20,
    "memory_snapshot_interval": 10,  # operations
    "gc_collection_interval": 50,   # operations
    "monitoring_sample_interval": 1,  # seconds
    "k6_default_vu": 10,  # virtual users
    "k6_default_duration": "60s",
    "response_time_thresholds": {
        "document_processing_ms": 500,
        "vector_search_ms": 100,
        "hybrid_search_ms": 150,
        "mcp_tool_ms": 200,
        "collection_management_ms": 50,
    },
    "memory_thresholds": {
        "base_overhead_mb": 50,
        "per_document_mb": 0.005,  # 5KB per document
        "leak_detection_mb": 5,    # 5MB total growth is concerning
        "growth_per_operation_mb": 0.01,  # 10KB per operation is concerning
    },
    "concurrency_thresholds": {
        "max_concurrent_operations": 10,
        "linear_scaling_tolerance": 1.5,  # 50% overhead acceptable
        "error_rate_under_load": 0.05,   # 5% error rate under stress
    },
    "load_testing_thresholds": {
        "normal_load": {
            "avg_response_ms": 200,
            "p95_response_ms": 500,
            "error_rate": 0.01,
        },
        "stress_load": {
            "avg_response_ms": 500,
            "p95_response_ms": 1000,
            "error_rate": 0.05,
        },
        "spike_recovery_s": 30,
        "soak_memory_growth_percent": 10,
    }
}

# Performance regression thresholds
REGRESSION_THRESHOLDS = {
    "response_time_increase_percent": 20.0,
    "memory_increase_percent": 30.0,
    "throughput_decrease_percent": 15.0,
    "error_rate_increase_percent": 100.0,  # Double error rate is concerning
}

# Test markers for categorization
TEST_MARKERS = {
    "performance": "Performance and benchmark tests",
    "benchmark": "Pytest-benchmark microbenchmarks",
    "load_testing": "K6 load testing scenarios",
    "memory_profiling": "Memory usage and leak detection tests",
    "concurrency": "Concurrent operation performance tests",
    "scaling": "Dataset scaling behavior tests",
    "regression": "Performance regression detection tests",
    "monitoring": "Real-time performance monitoring tests",
    "slow": "Long-running performance tests (>60s)",
    "requires_qdrant": "Tests requiring Qdrant server",
    "requires_k6": "Tests requiring k6 load testing tool",
}