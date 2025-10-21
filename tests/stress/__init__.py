"""
Stress Testing Suite for workspace-qdrant-mcp Daemon File Ingestion.

This package provides comprehensive stress testing for the daemon's file ingestion
capabilities under high load and resource constraints.

Test Categories (Task 317):
- 317.1: High-volume file processing (10,000+ files)
- 317.2: Rapid ingestion rate (100+ files/second)
- 317.3: Multiple folder watching (10+ folders)
- 317.4: Memory constraint testing (<512MB)
- 317.5: Disk I/O saturation
- 317.6: Network interruption handling
- 317.7: Resource monitoring integration
- 317.8: Performance degradation tracking

Test Execution:
    Run all stress tests (WARNING: may take 30+ minutes):
    ```bash
    pytest tests/stress/ -m stress -v
    ```

    Run specific stress test categories:
    ```bash
    pytest tests/stress/ -k "high_volume" -v
    pytest tests/stress/ -k "rapid_ingestion" -v
    pytest tests/stress/ -k "memory_constraint" -v
    ```

    Run with reduced scale for CI:
    ```bash
    STRESS_SCALE=ci pytest tests/stress/ -m stress -v
    ```

    Run with full stress scale:
    ```bash
    STRESS_SCALE=full pytest tests/stress/ -m stress -v --timeout=1800
    ```

Configuration:
    Stress test scale can be configured via environment variables:
    - STRESS_SCALE: Scale factor (ci/medium/full, default: ci)
    - STRESS_FILE_COUNT: Number of files for high-volume tests
    - STRESS_INGESTION_RATE: Target ingestion rate (files/second)
    - STRESS_FOLDER_COUNT: Number of watched folders
    - STRESS_MEMORY_LIMIT_MB: Memory limit for constraint tests
    - STRESS_TIMEOUT: Test timeout in seconds

Dependencies:
    - pytest: Test framework
    - psutil: System resource monitoring
    - tracemalloc: Memory profiling
    - asyncio: Concurrent operations
"""

import os
from typing import Dict, Any

__version__ = "0.3.0"

# Stress test scale configurations
STRESS_SCALE_CONFIGS = {
    "ci": {
        "file_count": 1000,
        "ingestion_rate": 50,
        "folder_count": 5,
        "memory_limit_mb": 512,
        "large_file_size_mb": 10,
        "timeout_seconds": 300,
        "concurrent_operations": 10,
    },
    "medium": {
        "file_count": 5000,
        "ingestion_rate": 100,
        "folder_count": 10,
        "memory_limit_mb": 256,
        "large_file_size_mb": 50,
        "timeout_seconds": 900,
        "concurrent_operations": 25,
    },
    "full": {
        "file_count": 10000,
        "ingestion_rate": 200,
        "folder_count": 15,
        "memory_limit_mb": 128,
        "large_file_size_mb": 100,
        "timeout_seconds": 1800,
        "concurrent_operations": 50,
    },
}

# Get current stress scale from environment
STRESS_SCALE = os.environ.get("STRESS_SCALE", "ci").lower()
if STRESS_SCALE not in STRESS_SCALE_CONFIGS:
    STRESS_SCALE = "ci"

# Active stress test configuration
STRESS_CONFIG = STRESS_SCALE_CONFIGS[STRESS_SCALE].copy()

# Allow individual overrides via environment variables
STRESS_CONFIG["file_count"] = int(os.environ.get("STRESS_FILE_COUNT", STRESS_CONFIG["file_count"]))
STRESS_CONFIG["ingestion_rate"] = int(os.environ.get("STRESS_INGESTION_RATE", STRESS_CONFIG["ingestion_rate"]))
STRESS_CONFIG["folder_count"] = int(os.environ.get("STRESS_FOLDER_COUNT", STRESS_CONFIG["folder_count"]))
STRESS_CONFIG["memory_limit_mb"] = int(os.environ.get("STRESS_MEMORY_LIMIT_MB", STRESS_CONFIG["memory_limit_mb"]))
STRESS_CONFIG["timeout_seconds"] = int(os.environ.get("STRESS_TIMEOUT", STRESS_CONFIG["timeout_seconds"]))

# Resource monitoring configuration
MONITORING_CONFIG = {
    "sample_interval_seconds": 1.0,
    "memory_warning_threshold_mb": 400,
    "memory_critical_threshold_mb": 480,
    "cpu_warning_threshold_percent": 80.0,
    "cpu_critical_threshold_percent": 95.0,
    "disk_io_warning_mb_per_sec": 50,
    "disk_io_critical_mb_per_sec": 100,
}

# Performance baseline thresholds
PERFORMANCE_THRESHOLDS = {
    "throughput_files_per_second": {
        "minimum": 50,
        "target": 100,
        "excellent": 200,
    },
    "memory_overhead_mb": {
        "maximum": 500,
        "target": 300,
        "excellent": 200,
    },
    "processing_latency_ms": {
        "p50": 100,
        "p95": 500,
        "p99": 1000,
    },
    "error_rate_percent": {
        "maximum": 5.0,
        "target": 1.0,
        "excellent": 0.1,
    },
}

# Degradation detection thresholds
DEGRADATION_THRESHOLDS = {
    "throughput_decrease_percent": 20.0,
    "memory_increase_percent": 30.0,
    "latency_increase_percent": 50.0,
    "error_rate_increase_percent": 100.0,
}
