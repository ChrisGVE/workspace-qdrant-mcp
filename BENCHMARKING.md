# Benchmarking Guide

This document describes the benchmarking infrastructure and how to run performance benchmarks for workspace-qdrant-mcp.

## Overview

The project uses two benchmarking frameworks:

- **Python**: `pytest-benchmark` for Python component benchmarks
- **Rust**: `criterion` for Rust daemon benchmarks

Both frameworks provide statistical analysis, regression detection, and report generation.

## Python Benchmarks

### Location

Python benchmarks are located in:
```
tests/benchmarks/
├── __init__.py
└── benchmark_example.py
```

### Running Python Benchmarks

```bash
# Run all Python benchmarks
uv run pytest tests/benchmarks/ --benchmark-only

# Run with verbose output
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-verbose

# Run specific benchmark file
uv run pytest tests/benchmarks/benchmark_example.py --benchmark-only

# Save baseline for comparison
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline

# Generate histogram
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-histogram

# Skip slow benchmarks
uv run pytest tests/benchmarks/ -m "not slow" --benchmark-only
```

### Writing Python Benchmarks

```python
import pytest

@pytest.mark.benchmark
def test_my_function(benchmark):
    """Benchmark my_function performance."""
    # The benchmark fixture will run the function multiple times
    result = benchmark(my_function, arg1, arg2)

    # Add assertions to verify correctness
    assert result == expected_value

# For setup/teardown operations
@pytest.mark.benchmark
def test_with_setup(benchmark):
    """Benchmark with setup phase."""
    def setup():
        # Expensive setup that should not be benchmarked
        return expensive_setup()

    def run(data):
        # Code to benchmark
        return process(data)

    result = benchmark.pedantic(run, setup=setup, rounds=100)
    assert result is not None
```

### Python Benchmark Output

```
-------------------------------- benchmark: 4 tests --------------------------------
Name (time in ns)              Min       Max      Mean    StdDev    Median
------------------------------------------------------------------------------------
test_fibonacci_iterative   416.71   5,236.00   439.36    64.04   436.21
test_string_concat       8,624.05  59,115.00  8,976.44   765.54  8,945.00
```

## Rust Benchmarks

### Location

Rust benchmarks are located in:
```
src/rust/daemon/core/benches/
├── example_benchmark.rs
├── cross_platform_benchmarks.rs
├── processing_benchmarks.rs
├── queue_processor_bench.rs
└── stress_benchmarks.rs
```

### Running Rust Benchmarks

```bash
# Navigate to Rust daemon core
cd src/rust/daemon/core

# Run all Rust benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench example_benchmark

# Run benchmarks matching a pattern
cargo bench -- fibonacci

# Run with verbose output
cargo bench -- --verbose

# Save baseline for comparison
cargo bench -- --save-baseline my-baseline

# Compare against baseline
cargo bench -- --baseline my-baseline

# Generate plots (requires gnuplot)
cargo bench --bench example_benchmark
# Plots saved to: target/criterion/<benchmark_name>/report/index.html
```

### Writing Rust Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_function(n: u64) -> u64 {
    // Function to benchmark
    n * 2
}

fn bench_my_function(c: &mut Criterion) {
    c.bench_function("my_function_20", |b| {
        b.iter(|| my_function(black_box(20)))
    });
}

// For parameterized benchmarks
fn bench_parameterized(c: &mut Criterion) {
    for size in [10, 100, 1000].iter() {
        c.bench_function(&format!("process_{}", size), |b| {
            b.iter(|| process_data(black_box(*size)))
        });
    }
}

criterion_group!(benches, bench_my_function, bench_parameterized);
criterion_main!(benches);
```

### Rust Benchmark Output

```
fibonacci_recursive_20  time:   [13.694 µs 13.730 µs 13.768 µs]
fibonacci_iterative_20  time:   [3.2037 ns 3.2057 ns 3.2080 ns]
                        change: [-2.1% +0.5% +3.2%] (p = 0.42 > 0.05)
```

## Benchmark Configuration

### Python Configuration

Configure pytest-benchmark in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "benchmark: marks tests as benchmarks",
    "slow: marks tests as slow benchmarks",
]
```

### Rust Configuration

Configure criterion in `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports", "async_tokio"] }

[[bench]]
name = "example_benchmark"
harness = false  # Required for criterion
```

## Best Practices

### General

1. **Isolate benchmarks**: Run on a quiet system without other intensive processes
2. **Warm up**: Frameworks automatically warm up, but ensure consistent state
3. **Statistical significance**: Run enough iterations for statistical validity
4. **Baseline comparisons**: Save baselines before major changes
5. **Document assumptions**: Note hardware specs and configuration in commit messages

### Python-Specific

1. Use `black_box` equivalent (function parameters) to prevent optimization
2. Mark slow benchmarks with `@pytest.mark.slow`
3. Use `benchmark.pedantic` for precise control over iterations
4. Separate setup/teardown from measured code

### Rust-Specific

1. Always use `black_box()` to prevent compiler optimizations
2. Set `harness = false` in Cargo.toml for benchmark targets
3. Use `criterion_group!` and `criterion_main!` macros
4. Consider async benchmarks with `async_tokio` feature

## Performance Regression Detection

### Automated Detection

Both frameworks support regression detection:

```bash
# Python: Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%

# Rust: Compare against baseline (fails if regression > 10%)
cargo bench -- --baseline my-baseline
```

### CI Integration

```yaml
# Example GitHub Actions workflow
- name: Run Python Benchmarks
  run: |
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

- name: Run Rust Benchmarks
  run: |
    cd src/rust/daemon/core
    cargo bench --bench example_benchmark -- --save-baseline ci-baseline
```

## Interpreting Results

### Key Metrics

- **Mean**: Average time per iteration
- **StdDev**: Standard deviation (lower is better for consistency)
- **Median**: Middle value (less affected by outliers)
- **Min/Max**: Best and worst times observed
- **IQR**: Interquartile range (spread of middle 50% of data)
- **Outliers**: Values far from the mean (indicates inconsistency)

### What to Look For

- **Regressions**: Mean time increases significantly between runs
- **High variance**: Large StdDev indicates inconsistent performance
- **Outliers**: Many outliers suggest system interference or measurement issues
- **Comparison**: Changes should be >5% to be considered significant

## Benchmark Targets

### Current Benchmarks

**Python** (`tests/benchmarks/`):
- `benchmark_example.py`: Basic Python operations (fibonacci, strings, lists)

**Rust** (`src/rust/daemon/core/benches/`):
- `example_benchmark.rs`: Basic Rust operations (fibonacci, HashMap, String, Vec)
- `cross_platform_benchmarks.rs`: Platform-specific performance tests
- `processing_benchmarks.rs`: Document processing performance
- `queue_processor_bench.rs`: Queue processing performance
- `stress_benchmarks.rs`: Stress and load testing

### Planned Benchmarks (Task 304)

Future subtasks will add benchmarks for:
- Hybrid search performance
- LSP integration overhead
- File watching and ingestion
- Cross-platform file operations
- Memory usage profiling

## Troubleshooting

### Python Issues

**Issue**: `ModuleNotFoundError: No module named 'pytest_benchmark'`
```bash
# Solution: Install dev dependencies
uv pip install pytest-benchmark
```

**Issue**: Benchmarks are too slow
```bash
# Solution: Reduce iterations or mark as slow
@pytest.mark.slow
def test_expensive_operation(benchmark):
    ...
```

### Rust Issues

**Issue**: `gnuplot not found, using plotters backend`
```bash
# Solution: Install gnuplot for better plots
brew install gnuplot  # macOS
apt-get install gnuplot  # Linux
```

**Issue**: Benchmarks show high variance
```bash
# Solution: Close other applications and try again
# Or increase measurement time in the benchmark:
c.measurement_time(Duration::from_secs(10));
```

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [criterion.rs documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust performance book](https://nnethercote.github.io/perf-book/)

---

**Note**: Always run benchmarks in a consistent environment and save baselines before making performance-critical changes.
