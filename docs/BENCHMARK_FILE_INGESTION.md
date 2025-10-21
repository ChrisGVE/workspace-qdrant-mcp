# File Ingestion Throughput Benchmark Guide

This document explains how to run and interpret the file ingestion throughput benchmarks for workspace-qdrant-mcp.

## Overview

The file ingestion benchmarks measure parsing and processing performance across:
- **File sizes**: Small (1KB), Medium (100KB), Large (1MB), Very Large (10MB)
- **File types**: `.txt`, `.py`, `.md`, `.json`
- **Scenarios**: Single-file and batch ingestion
- **Metrics**: Files/second and MB/second throughput

## Python Benchmarks

### Location
```
tests/benchmarks/benchmark_file_ingestion.py
```

### Running Python Benchmarks

**Run all file ingestion benchmarks:**
```bash
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only
```

**Run specific benchmark:**
```bash
uv run pytest tests/benchmarks/benchmark_file_ingestion.py::test_parse_small_text_file --benchmark-only
```

**Save benchmark results:**
```bash
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only --benchmark-save=ingestion_baseline
```

**Compare with baseline:**
```bash
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only --benchmark-compare=ingestion_baseline
```

**Generate HTML report:**
```bash
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only --benchmark-autosave --benchmark-histogram
```

### Python Benchmark Categories

**Single File Parsing:**
- `test_parse_small_text_file` - 1KB text file
- `test_parse_medium_text_file` - 100KB text file
- `test_parse_large_text_file` - 1MB text file
- `test_parse_very_large_text_file` - 10MB text file
- `test_parse_small_python_file` - 1KB Python file
- `test_parse_medium_python_file` - 100KB Python file
- `test_parse_large_python_file` - 1MB Python file
- `test_parse_small_markdown_file` - 1KB Markdown file
- `test_parse_medium_markdown_file` - 100KB Markdown file
- `test_parse_large_markdown_file` - 1MB Markdown file
- `test_parse_small_json_file` - 1KB JSON file
- `test_parse_medium_json_file` - 100KB JSON file
- `test_parse_large_json_file` - 1MB JSON file

**Batch Parsing:**
- `test_parse_batch_10_small_files` - 10 files @ 1KB each
- `test_parse_batch_10_medium_files` - 10 files @ 100KB each
- `test_parse_batch_50_small_files` - 50 files @ 1KB each
- `test_parse_batch_mixed_types` - 20 mixed type files

**Throughput Measurements:**
- `test_throughput_small_files` - 100 files @ 1KB each (files/sec)
- `test_throughput_medium_files` - 20 files @ 100KB each (MB/sec)

### Interpreting Python Results

pytest-benchmark output includes:

| Metric | Description |
|--------|-------------|
| Min | Fastest iteration time |
| Max | Slowest iteration time |
| Mean | Average time per iteration |
| StdDev | Standard deviation (consistency) |
| Median | Middle value (50th percentile) |
| IQR | Interquartile range (variability) |
| Outliers | Number of outlier measurements |
| OPS | Operations per second (1/Mean) |

**Example output:**
```
Name (time in ms)                Min      Max     Mean   StdDev   Median     IQR  Outliers     OPS
test_parse_medium_text_file    12.34    15.67   13.45    0.89    13.21    1.23      12;5   74.35
```

**Interpreting:**
- Mean of 13.45ms = ~74 files/second (OPS)
- Low StdDev (0.89) = consistent performance
- Few outliers = stable measurement
- For throughput tests, OPS directly shows files/second

## Rust Benchmarks

### Location
```
src/rust/daemon/core/benches/file_ingestion_benchmarks.rs
```

### Running Rust Benchmarks

**Run all file ingestion benchmarks:**
```bash
cd src/rust/daemon/core
cargo bench --bench file_ingestion_benchmarks
```

**Run specific benchmark group:**
```bash
cargo bench --bench file_ingestion_benchmarks -- ingest_small
```

**Save baseline for comparison:**
```bash
cargo bench --bench file_ingestion_benchmarks -- --save-baseline ingestion_baseline
```

**Compare with baseline:**
```bash
cargo bench --bench file_ingestion_benchmarks -- --baseline ingestion_baseline
```

### Rust Benchmark Categories

**Single File Ingestion (ingestion_benches group):**
- `bench_ingest_small_files` - 1KB files (.txt, .py, .md, .json)
- `bench_ingest_medium_files` - 100KB files (all types)
- `bench_ingest_large_files` - 1MB files (.txt, .py, .md)
- `bench_ingest_very_large_files` - 10MB files (.txt, .py)

**Batch Ingestion:**
- `bench_batch_ingestion_10_files` - 10 x 1KB files
- `bench_batch_ingestion_50_files` - 50 x 1KB files
- `bench_batch_mixed_types` - 20 mixed files @ 10KB each

**Throughput Measurements:**
- `bench_throughput_small_files` - 100 x 1KB files
- `bench_throughput_medium_files` - 20 x 100KB files (2MB total)

**Legacy Benchmarks (legacy_benches group):**
- File creation benchmarks (for compatibility)

### Interpreting Rust Results

Criterion output includes:

| Metric | Description |
|--------|-------------|
| time | Mean execution time per iteration |
| thrpt | Throughput (elements/sec or bytes/sec) |
| change | % change from baseline (if comparing) |

**Example output:**
```
ingest_medium_100kb/txt time:   [2.1234 ms 2.1567 ms 2.1901 ms]
                        thrpt:  [46.356 MiB/s 47.092 MiB/s 47.829 MiB/s]
```

**Interpreting:**
- time: Mean of 2.16ms per 100KB file
- thrpt: ~47 MB/s throughput
- Range shows 95% confidence interval
- Lower is better for time, higher is better for throughput

**Regression detection:**
```
change: [-5.2341% +1.2345% +7.8901%] (p = 0.42 > 0.05)
```
- Central estimate: +1.23% slower
- 95% confidence: between 5.23% faster and 7.89% slower
- p-value > 0.05 = not statistically significant

## Performance Baselines

### Expected Performance (Reference System: M1 Mac)

**Python Parser Throughput:**
| File Size | File Type | Expected OPS | Expected Time |
|-----------|-----------|--------------|---------------|
| 1KB | .txt | 800-1000/sec | ~1ms |
| 100KB | .txt | 80-100/sec | ~10ms |
| 1MB | .txt | 8-10/sec | ~100ms |
| 10MB | .txt | 0.8-1/sec | ~1s |

**Rust File Reading:**
| File Size | Expected Throughput | Expected Time |
|-----------|---------------------|---------------|
| 1KB | N/A | ~10-50μs |
| 100KB | 40-60 MB/s | ~2ms |
| 1MB | 50-100 MB/s | ~15-20ms |
| 10MB | 50-100 MB/s | ~150-200ms |

**Batch Processing:**
| Scenario | Expected Throughput |
|----------|---------------------|
| 10 x 1KB | 700-900 files/sec |
| 50 x 1KB | 700-900 files/sec |
| 100 x 1KB | 700-900 files/sec |

*Note: Actual performance depends on hardware, filesystem, and system load.*

## Comparing Python vs Rust

**When to use each:**
- **Python benchmarks**: Measure end-to-end parser performance (TextParser, MarkdownParser, CodeParser)
- **Rust benchmarks**: Measure low-level file I/O and basic processing

**Key differences:**
- Python includes full parsing logic (encoding detection, content extraction, metadata)
- Rust measures file reading + simple line counting
- Python async overhead vs Rust sync I/O
- Different test data (Python uses fixtures, Rust generates content)

## Continuous Performance Monitoring

### Recommended Workflow

**1. Establish baseline:**
```bash
# Python
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only --benchmark-save=baseline

# Rust
cd src/rust/daemon/core && cargo bench --bench file_ingestion_benchmarks -- --save-baseline baseline
```

**2. After code changes:**
```bash
# Python
uv run pytest tests/benchmarks/benchmark_file_ingestion.py --benchmark-only --benchmark-compare=baseline

# Rust
cargo bench --bench file_ingestion_benchmarks -- --baseline baseline
```

**3. Acceptable regression thresholds:**
- < 5% change: Acceptable noise
- 5-10% change: Investigate if consistent
- > 10% change: Requires investigation/optimization

### CI/CD Integration

**Python benchmark in CI:**
```yaml
- name: Run benchmarks
  run: |
    uv run pytest tests/benchmarks/benchmark_file_ingestion.py \
      --benchmark-only \
      --benchmark-json=benchmark_results.json

- name: Store benchmark results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark_results.json
```

**Rust benchmark in CI:**
```yaml
- name: Run benchmarks
  run: |
    cd src/rust/daemon/core
    cargo bench --bench file_ingestion_benchmarks -- --output-format bencher | tee benchmark_results.txt

- name: Store benchmark results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'cargo'
    output-file-path: benchmark_results.txt
```

## Troubleshooting

### Python Benchmarks

**Issue: Inconsistent results**
- Reduce system load (close other applications)
- Increase warmup iterations: `--benchmark-warmup=on`
- Increase sample size: `--benchmark-min-rounds=10`

**Issue: OutOfMemory for large files**
- Run large file tests separately
- Reduce batch sizes in `TestDataGenerator`

**Issue: Slow benchmark execution**
- Run subset: `pytest tests/benchmarks/benchmark_file_ingestion.py::test_parse_small_text_file`
- Skip slow tests: `pytest -m "not slow"`

### Rust Benchmarks

**Issue: Benchmarks not detected**
- Ensure `criterion_group!` and `criterion_main!` are properly configured
- Check benchmark function names match group definition

**Issue: Long execution time**
- Reduce sample size in benchmark: `group.sample_size(10)`
- Run specific benchmark: `cargo bench -- bench_name`

**Issue: Compilation warnings**
- Run `cargo fix --bench file_ingestion_benchmarks`
- Address unused imports and variables

## Related Benchmarks

This file ingestion benchmark suite is part of a larger performance testing infrastructure:

- `tests/benchmarks/benchmark_example.py` - Example Python benchmarks
- `src/rust/daemon/core/benches/example_benchmark.rs` - Example Rust benchmarks
- `src/rust/daemon/core/benches/processing_benchmarks.rs` - Processing pipeline benchmarks
- Future: Search benchmarks, memory benchmarks, end-to-end benchmarks

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Criterion.rs documentation](https://bheisler.github.io/criterion.rs/book/)
- [workspace-qdrant-mcp architecture](../ARCHITECTURE.md)
- [Task 304.2 - File Ingestion Throughput Benchmarks](/.taskmaster/tasks/)
