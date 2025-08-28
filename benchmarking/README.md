# Benchmarking Suite

This directory contains performance benchmarks for the workspace-qdrant-mcp project.

## Files

### Core Benchmarks

- **`simple_benchmark.py`** - Basic performance tests for fundamental operations
- **`comprehensive_benchmark.py`** - Detailed benchmarking across all major features
- **`efficient_large_benchmark.py`** - Optimized benchmarks for large-scale testing
- **`large_scale_benchmark.py`** - High-volume performance testing
- **`benchmark_actual_performance.py`** - Real-world performance measurements

### Test Runners

- **`run_comprehensive_tests.py`** - Orchestrates complete benchmark suite execution

## Usage

Run individual benchmarks:
```bash
python benchmarking/simple_benchmark.py
python benchmarking/comprehensive_benchmark.py
```

Run complete benchmark suite:
```bash
python benchmarking/run_comprehensive_tests.py
```

## Output

Benchmarks generate performance metrics including:
- Execution times
- Memory usage
- Throughput measurements
- Latency statistics

Results are typically saved to `.benchmarks/` directory (gitignored).

## Requirements

Ensure the project environment is set up with all dependencies before running benchmarks:
```bash
pip install -e .
```

Some benchmarks may require additional test data or specific Qdrant server configurations.