# Stress Testing Suite

Comprehensive stress tests for daemon file ingestion system under high load and resource constraints.

## Overview

This test suite implements Task 317, providing 8 categories of stress tests to validate daemon performance, stability, and resilience under extreme conditions.

## Test Categories

### 317.1: High-Volume File Processing
- **Tests**: `TestHighVolumeFileProcessing`
- **Validates**: Processing 10,000+ files simultaneously
- **Metrics**: Throughput, memory usage, processing latency
- **Thresholds**: <500MB memory, >50 files/sec throughput

### 317.2: Rapid Ingestion Rate
- **Tests**: `TestRapidIngestionRate`
- **Validates**: 100+ files/second ingestion rate
- **Metrics**: Queue depth, processing latency, dropped files
- **Thresholds**: <1000ms p99 latency, 0% dropped files

### 317.3: Multiple Folder Watching
- **Tests**: `TestMultipleFolderWatching`
- **Validates**: 10+ folders with concurrent file changes
- **Metrics**: Resource allocation, isolation, interference
- **Thresholds**: Sub-linear memory growth, consistent performance

### 317.4: Memory Constraint
- **Tests**: `TestMemoryConstraint`
- **Validates**: Operation under <512MB memory limit
- **Metrics**: Memory leaks, graceful degradation
- **Thresholds**: <512MB max, <50MB leak growth
- **Note**: Memory limiting tests skipped on Windows

### 317.5: Disk I/O Saturation
- **Tests**: `TestDiskIOSaturation`
- **Validates**: Performance under I/O bottlenecks
- **Metrics**: I/O rates, throttling behavior
- **Thresholds**: Stable timing variance <100%

### 317.6: Network Interruption
- **Tests**: `TestNetworkInterruption`
- **Validates**: Recovery from network failures
- **Metrics**: Retry logic, data integrity, backoff
- **Thresholds**: Eventual consistency, exponential backoff

### 317.7: Resource Monitoring Integration
- **Tests**: `TestResourceMonitoringIntegration`
- **Validates**: Comprehensive resource monitoring
- **Metrics**: CPU, memory, disk I/O tracking
- **Thresholds**: Alert on 80% CPU, 400MB memory

### 317.8: Performance Degradation Tracking
- **Tests**: `TestPerformanceDegradationTracking`
- **Validates**: Baseline comparison and trend analysis
- **Metrics**: Throughput regression, memory growth
- **Thresholds**: <20% throughput decrease, <30% memory increase

## Usage

### Quick Start

```bash
# Run all stress tests (CI scale, ~5-10 minutes)
pytest tests/stress/ -m stress -v

# Run specific category
pytest tests/stress/ -k "high_volume" -v
pytest tests/stress/ -k "rapid_ingestion" -v
pytest tests/stress/ -k "memory_constraint" -v
```

### Scale Configuration

```bash
# CI scale (default, faster)
STRESS_SCALE=ci pytest tests/stress/ -m stress -v

# Medium scale (moderate)
STRESS_SCALE=medium pytest tests/stress/ -m stress -v

# Full scale (WARNING: 30+ minutes)
STRESS_SCALE=full pytest tests/stress/ -m stress -v --timeout=1800
```

### Custom Configuration

```bash
# Override specific parameters
STRESS_FILE_COUNT=5000 \
STRESS_INGESTION_RATE=150 \
STRESS_FOLDER_COUNT=12 \
STRESS_MEMORY_LIMIT_MB=256 \
pytest tests/stress/ -m stress -v
```

### Selective Testing

```bash
# Run only high-volume tests
pytest tests/stress/ -m high_volume -v

# Run only rapid ingestion tests
pytest tests/stress/ -m rapid_ingestion -v

# Run only memory constraint tests
pytest tests/stress/ -m memory_constraint -v

# Run only network interruption tests
pytest tests/stress/ -m network_interruption -v

# Exclude memory constraint tests (Windows)
pytest tests/stress/ -m "stress and not memory_constraint" -v
```

## Configuration

### Environment Variables

| Variable | Default (CI) | Medium | Full | Description |
|----------|--------------|--------|------|-------------|
| `STRESS_SCALE` | `ci` | `medium` | `full` | Overall scale factor |
| `STRESS_FILE_COUNT` | 1000 | 5000 | 10000 | Files for high-volume tests |
| `STRESS_INGESTION_RATE` | 50 | 100 | 200 | Target files/second |
| `STRESS_FOLDER_COUNT` | 5 | 10 | 15 | Watched folders |
| `STRESS_MEMORY_LIMIT_MB` | 512 | 256 | 128 | Memory constraint limit |
| `STRESS_TIMEOUT` | 300 | 900 | 1800 | Test timeout (seconds) |

### Performance Thresholds

#### Throughput
- **Minimum**: 50 files/second
- **Target**: 100 files/second
- **Excellent**: 200 files/second

#### Memory
- **Maximum**: 500MB overhead
- **Target**: 300MB overhead
- **Excellent**: 200MB overhead

#### Latency (milliseconds)
- **P50**: 100ms
- **P95**: 500ms
- **P99**: 1000ms

#### Error Rate
- **Maximum**: 5.0%
- **Target**: 1.0%
- **Excellent**: 0.1%

### Degradation Thresholds

| Metric | Threshold |
|--------|-----------|
| Throughput decrease | 20% |
| Memory increase | 30% |
| Latency increase | 50% |
| Error rate increase | 100% |

## Test Execution Time

### By Scale

- **CI**: 5-10 minutes
- **Medium**: 15-20 minutes
- **Full**: 30-60 minutes

### By Category

| Category | CI Time | Full Time |
|----------|---------|-----------|
| High-Volume | 1-2 min | 5-10 min |
| Rapid Ingestion | 30 sec | 2-3 min |
| Multi-Folder | 1 min | 3-5 min |
| Memory Constraint | 1-2 min | 5-8 min |
| Disk I/O | 1 min | 3-5 min |
| Network | 30 sec | 1-2 min |
| Monitoring | 1 min | 3-5 min |
| Degradation | 1 min | 5-10 min |

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Stress Tests (CI Scale)
  run: |
    STRESS_SCALE=ci pytest tests/stress/ -m stress -v --timeout=600
  timeout-minutes: 15

- name: Upload Stress Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: stress-test-reports
    path: |
      performance_baseline.json
      performance_comparison.json
      monitoring_report.json
```

### Nightly Full Scale Testing

```yaml
- name: Run Full Stress Tests (Nightly)
  run: |
    STRESS_SCALE=full pytest tests/stress/ -m stress -v --timeout=3600
  timeout-minutes: 90
  if: github.event_name == 'schedule'
```

## Hardware Requirements

### Minimum (CI Scale)
- **CPU**: 2 cores
- **RAM**: 2GB
- **Disk**: 5GB free space
- **I/O**: 50MB/s

### Recommended (Medium Scale)
- **CPU**: 4 cores
- **RAM**: 4GB
- **Disk**: 10GB free space
- **I/O**: 100MB/s

### Optimal (Full Scale)
- **CPU**: 8+ cores
- **RAM**: 8GB
- **Disk**: 20GB free space
- **I/O**: 200MB/s

## Output and Reports

### Generated Files

Tests generate performance reports in the test temp directory:

- `performance_baseline.json` - Baseline metrics
- `performance_comparison.json` - Comparison against baseline
- `monitoring_report.json` - Resource monitoring summary
- `degradation_alerts.json` - Performance degradation alerts
- `performance_trend.json` - Trend analysis over multiple runs

### Report Format

```json
{
  "monitoring": {
    "summary": {
      "cpu_percent": {"avg": 45.2, "min": 10.1, "max": 89.3},
      "memory_rss_mb": {"avg": 234.5, "min": 180.2, "max": 312.8},
      "memory_growth_mb": 42.3
    },
    "duration_seconds": 123.45,
    "sample_count": 120
  },
  "performance": {
    "throughput_files_per_second": 156.7,
    "processing_latency_ms": {"p50": 45, "p95": 234, "p99": 456}
  },
  "thresholds": {
    "warnings": ["Memory exceeded warning threshold: 410MB"],
    "criticals": []
  }
}
```

## Troubleshooting

### Tests Timing Out

```bash
# Increase timeout
STRESS_TIMEOUT=1200 pytest tests/stress/ -m stress -v
```

### Memory Errors

```bash
# Reduce scale
STRESS_SCALE=ci pytest tests/stress/ -m stress -v

# Or reduce specific parameters
STRESS_FILE_COUNT=500 pytest tests/stress/ -m stress -v
```

### Disk Space Issues

```bash
# Tests create temporary files - ensure adequate space
df -h /tmp

# Clean up manually if needed
rm -rf /tmp/stress_test_*
```

### Platform-Specific Issues

```bash
# Windows: Skip memory constraint tests
pytest tests/stress/ -m "stress and not memory_constraint" -v

# Linux: May need to adjust file descriptor limits
ulimit -n 4096
```

## Development

### Adding New Stress Tests

1. Add test to appropriate class in `test_daemon_stress.py`
2. Use fixtures: `resource_monitor`, `file_generator`, `performance_tracker`
3. Mark with appropriate markers: `@pytest.mark.stress`, `@pytest.mark.<category>`
4. Document expected behavior and thresholds
5. Update this README with new test details

### Modifying Thresholds

Edit `tests/stress/__init__.py`:

```python
PERFORMANCE_THRESHOLDS = {
    "throughput_files_per_second": {
        "minimum": 50,  # Adjust here
        ...
    },
    ...
}
```

### Custom Monitoring

```python
@pytest.mark.stress
async def test_custom_monitoring(resource_monitor):
    resource_monitor.start_monitoring()

    # Your test code here

    resource_monitor.record_snapshot()  # Manual snapshots
    resource_monitor.stop_monitoring()

    summary = resource_monitor.get_summary()
    # Validate metrics
```

## Best Practices

1. **Run CI scale locally** before committing
2. **Review monitoring reports** for anomalies
3. **Track trends over time** to detect regressions
4. **Use appropriate scale** for your environment
5. **Monitor system resources** during tests
6. **Clean up temporary files** after test failures
7. **Document threshold changes** with justification
8. **Run full scale tests** before releases

## Related Documentation

- [Performance Testing](../performance/README.md)
- [Benchmarking](../benchmarks/README.md)
- [Edge Case Testing](../unit/test_daemon_file_ingestion_edge_cases.py)
- [Testing Framework](../TESTING_FRAMEWORK.md)

## Support

For issues or questions about stress testing:

1. Check test output and monitoring reports
2. Review threshold configurations
3. Try reduced scale (STRESS_SCALE=ci)
4. Check hardware requirements
5. Review platform-specific notes

---

**Note**: Stress tests are designed to push the system to its limits. Failures may indicate areas for optimization rather than bugs.
