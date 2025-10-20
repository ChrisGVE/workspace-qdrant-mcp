# Stability Testing Guide

This document describes how to run extended stability tests for the workspace-qdrant-mcp system.

## Overview

Stability tests validate system behavior over extended periods (hours to days) with real-world usage patterns. These tests monitor for:

- Memory leaks
- Performance degradation
- Resource exhaustion
- System stability under load
- Recovery capabilities

## Test Durations

### 1-Hour Baseline (Recommended for CI)
Quick validation of basic stability patterns.

```bash
python scripts/run_stability_test.py --duration 1h
```

### 6-Hour Extended Test
Comprehensive validation with varying load patterns.

```bash
python scripts/run_stability_test.py --duration 6h
```

### 24-Hour Full Stability Test
Complete real-world simulation for production validation.

```bash
python scripts/run_stability_test.py --duration 24h
```

## Running Tests

### Basic Usage

```bash
# From project root
python scripts/run_stability_test.py --duration <duration>
```

### With Custom Output Directory

```bash
python scripts/run_stability_test.py --duration 6h --output ./my_results
```

### With Verbose Output

```bash
python scripts/run_stability_test.py --duration 1h --verbose
```

## Test Results

Results are saved to `./stability_test_results/` by default:

```
stability_test_results/
├── stability_test_1h_20250120_143022.log    # Test log
└── stability_test_1h_20250120_143022.xml    # JUnit XML report
```

## Running Individual Tests

You can also run stability tests directly with pytest:

```bash
# Run 1-hour baseline
uv run pytest tests/e2e/test_stability.py::TestShortStabilityBaseline::test_one_hour_stability_baseline -v

# Run 6-hour test
uv run pytest tests/e2e/test_stability.py::TestShortStabilityBaseline::test_six_hour_stability -v

# Run 24-hour test
uv run pytest tests/e2e/test_stability.py::TestExtendedStability::test_24_hour_stability -v
```

## Test Markers

Stability tests use special pytest markers:

- `@pytest.mark.stability` - All stability tests
- `@pytest.mark.extended` - Extended duration tests (24h+)
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration-level tests

### Running by Marker

```bash
# Run all stability tests
uv run pytest -m stability

# Run only short stability tests (exclude extended)
uv run pytest -m "stability and not extended"
```

## Monitoring During Tests

### Real-Time Monitoring

The stability test runner provides real-time output showing:
- Current test progress
- Operation counts
- Error rates
- Resource usage snapshots

### External Monitoring

For production-like monitoring, use system tools:

```bash
# Monitor process resources
watch -n 5 'ps aux | grep -E "memexd|workspace-qdrant-mcp"'

# Monitor system resources
htop  # or top

# Monitor memory specifically
watch -n 10 'free -h'
```

## What Tests Validate

### Memory Stability
- No unbounded memory growth (<2x growth for 1h, <3x for 24h)
- Proper resource cleanup
- Memory recovery after load

### Performance Consistency
- Response times remain stable
- No significant performance degradation (<2x slowdown)
- Throughput maintained under load

### Resource Management
- File descriptors don't leak
- Thread counts remain stable
- CPU usage recovers after load

### Error Handling
- Error rate stays low (<5%)
- System remains operational despite errors
- Graceful degradation under stress

## Test Patterns

### Real-World Simulation (24h test)
- Continuous file ingestion
- Periodic searches
- Varying load patterns (heavy every 4 hours, normal otherwise)
- Concurrent operations
- Status monitoring

### Multi-User Patterns
- 5 concurrent users
- Independent operations per user
- Resource isolation verification

### Load Patterns
- Burst loads (20 operations followed by idle)
- Sustained loads (consistent operations)
- Project switching (frequent context changes)

## Interpreting Results

### Success Criteria

A passing test indicates:
- All operations completed successfully
- Memory growth within acceptable bounds
- Performance remained consistent
- Error rate below threshold
- System operational throughout test

### Common Issues

#### Memory Leaks
Symptom: Memory growth ratio >2x (1h) or >3x (24h)
Investigation: Check for unclosed resources, caching issues

#### Performance Degradation
Symptom: Operations taking significantly longer over time
Investigation: Check for database bloat, index fragmentation

#### Resource Exhaustion
Symptom: File descriptor or thread count growing unbounded
Investigation: Check for connection leaks, improper cleanup

## Prerequisites

Before running stability tests:

1. Ensure Qdrant is running:
   ```bash
   docker compose up -d
   ```

2. Install test dependencies:
   ```bash
   uv sync --dev
   ```

3. Ensure sufficient disk space:
   - 1h test: ~1GB
   - 6h test: ~5GB
   - 24h test: ~20GB

4. For 24h tests, consider using `screen` or `tmux`:
   ```bash
   screen -S stability
   python scripts/run_stability_test.py --duration 24h
   # Detach: Ctrl+A, D
   # Reattach: screen -r stability
   ```

## CI Integration

For CI/CD pipelines, use the 1-hour baseline test:

```yaml
# Example GitHub Actions workflow
stability-test:
  runs-on: ubuntu-latest
  timeout-minutes: 90
  steps:
    - name: Run 1-hour stability test
      run: python scripts/run_stability_test.py --duration 1h
```

## Troubleshooting

### Test Hangs
- Check daemon is responsive: `uv run wqm service status`
- Check Qdrant is running: `curl http://localhost:6333/health`
- Review logs in output directory

### Out of Memory
- Reduce concurrent operations
- Increase system swap space
- Run on machine with more RAM

### Disk Space Issues
- Clean up old test artifacts
- Use external storage for test workspaces
- Monitor disk usage: `df -h`

## Best Practices

1. **Run shorter tests first** - Validate with 1h before attempting 24h
2. **Monitor externally** - Don't rely solely on test output
3. **Save results** - Archive test logs and metrics for comparison
4. **Baseline regularly** - Run stability tests before major releases
5. **Automate when possible** - Integrate 1h tests into CI/CD

## Support

For issues or questions about stability testing:
- Check logs in `./stability_test_results/`
- Review test implementation in `tests/e2e/test_stability.py`
- Check GitHub issues for similar problems
