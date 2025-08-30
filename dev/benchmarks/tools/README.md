# Benchmark Tools

Performance benchmarking tools for workspace-qdrant-mcp with realistic, end-to-end testing.

> **Note**: This directory contains benchmark tools only. Results are stored in `../results/` and are gitignored to keep the repository clean.

## Overview

The benchmarking suite has been consolidated into a single, comprehensive tool that provides realistic performance measurements using actual Qdrant operations, the workspace-qdrant-ingest CLI, and large open-source projects for mixed-environment testing.

## Performance Methodology

**Realistic Testing Approach:**
- ✅ **Real Qdrant Integration:** Actual search operations, not simulations
- ✅ **CLI Integration:** Uses workspace-qdrant-ingest for realistic data loading
- ✅ **Large OSS Projects:** Tests with neovim, rust, and go codebases
- ✅ **Chunk Size Optimization:** Compares performance across chunk sizes
- ✅ **Statistical Analysis:** Confidence intervals and proper sampling
- ✅ **End-to-End Testing:** Complete workflow from ingestion to search

## Benchmark Tools

### Authoritative Benchmark

- **`authoritative_benchmark.py`** - Comprehensive, realistic performance testing
  - Real Qdrant search operations
  - Large OSS project integration (neovim, rust, go)
  - Chunk size optimization testing (500, 1000, 2000, 4000 chars)
  - Project-only vs mixed environment comparisons
  - Statistical analysis with confidence intervals
  - End-to-end workflow testing

### Test Orchestration

- **`run_comprehensive_tests.py`** - CI/CD test suite orchestration
  - Pytest integration for functional tests
  - Different purpose from benchmark (test validation vs performance measurement)

## Available Tools

### performance_baseline_test.py
- **Purpose**: Real-world baseline performance testing
- **Features**: Tests against actual Qdrant instances with real collections
- **Output**: Performance metrics and regression detection data
- **Usage**: `python performance_baseline_test.py`

### simple_performance_benchmark.py  
- **Purpose**: Quick performance testing using existing collections
- **Features**: Lightweight testing for rapid feedback loops
- **Output**: Simple performance metrics and timing data
- **Usage**: `python simple_performance_benchmark.py`

### authoritative_benchmark.py
- **Purpose**: Comprehensive benchmark suite with large OSS projects
- **Features**: Full ingestion and search testing at scale
- **Output**: Detailed performance analysis and optimization recommendations
- **Usage**: `python authoritative_benchmark.py`

### run_comprehensive_tests.py
- **Purpose**: Test orchestration and CI/CD integration
- **Features**: Coordinates multiple test scenarios and reporting
- **Output**: Comprehensive test reports across configurations
- **Usage**: `python run_comprehensive_tests.py`

## Running Benchmarks

### Prerequisites

```bash
# From project root, ensure development environment is set up
pip install -e .[dev]

# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Validate configuration (optional)
workspace-qdrant-validate
```

### Quick Start

```bash
# From project root
cd dev/benchmarks/tools

# Run simple benchmark
python simple_performance_benchmark.py

# Run comprehensive baseline test
python performance_baseline_test.py

# Results will be automatically saved to ../results/
```

### Authoritative Benchmark

```bash
# Full benchmark with OSS projects and all chunk sizes
python benchmarking/authoritative_benchmark.py

# Skip OSS project download (faster, less realistic)
python benchmarking/authoritative_benchmark.py --skip-oss

# Test specific chunk sizes only
python benchmarking/authoritative_benchmark.py --chunk-sizes 1000 2000

# Example output location
ls benchmark_results/
```

### Test Suite Orchestration

```bash
# Run functional test suite (different from benchmarking)
python benchmarking/run_comprehensive_tests.py

# Run specific test categories
python benchmarking/run_comprehensive_tests.py --categories performance recall_precision
```

## Benchmark Test Scenarios

### Project-Only vs Mixed Environment Testing

**Scenario A: Project-Only**
- Ingest workspace-qdrant-mcp codebase only
- Test search quality in clean environment
- Baseline performance measurements
- Optimal precision/recall expected

**Scenario B: Mixed Environment**
- Ingest project + large OSS codebases (neovim, rust, go)
- Test search precision with "noise" data
- Realistic production-like conditions
- Measures performance degradation in mixed environments

### Chunk Size Optimization

**Chunk sizes tested:** 500, 1000, 2000, 4000 characters

**Metrics compared:**
- Search precision and recall
- Response time performance
- Index size and memory usage
- Embedding generation time

**Optimization goals:**
- Identify optimal chunk size for search quality
- Balance performance vs accuracy
- Provide actionable configuration recommendations

### Search Type Performance

**Search types benchmarked:**
- **Semantic Search:** Vector similarity matching
- **Hybrid Search:** Combined vector and keyword search
- **Exact Search:** Precise text matching

**Metrics tracked:**
- Precision, recall, and F1 scores
- Response time percentiles (p50, p95, p99)  
- Queries per second throughput
- Statistical confidence intervals

## Expected Performance Characteristics

**Realistic Performance Targets:**
- Performance thresholds are determined empirically from actual benchmark runs
- Confidence intervals provided for statistical reliability
- Separate baselines for project-only vs mixed environments
- Chunk size recommendations based on measured trade-offs

**Key Insights Provided:**
- **Precision Degradation:** Quantified impact of mixed OSS projects on search quality
- **Chunk Size Optimization:** Optimal chunk size for your use case
- **Response Time Analysis:** P50, P95, P99 response time distributions
- **Scalability Characteristics:** Performance impact of index size growth

**Data-Driven Recommendations:**
- Chunk size selection based on quality vs speed trade-offs
- Expected precision/recall ranges for realistic workloads
- Configuration tuning guidance for production deployments

## Output and Reporting

### Benchmark Results

The authoritative benchmark generates comprehensive reports with statistical analysis:

```
🎯 Benchmark Results Summary
============================
Scenario              | Chunk Size | Documents | Chunks   | Precision | Recall
project_only_1000     | 1000       | 156       | 2,341    | 0.892     | 0.734
mixed_projects_1000   | 1000       | 8,432     | 125,678  | 0.743     | 0.681
project_only_2000     | 2000       | 156       | 1,284    | 0.915     | 0.701
mixed_projects_2000   | 2000       | 8,432     | 68,432   | 0.761     | 0.663

📈 Performance Comparison: Project-Only vs Mixed
================================================
Chunk Size | Project Precision | Mixed Precision | Precision Drop
1000       | 0.892            | 0.743           | 16.7%
2000       | 0.915            | 0.761           | 16.8%

📏 Chunk Size Optimization Analysis
====================================
Chunk Size | Precision | Recall | Response Time | Recommendation
1000       | 0.892     | 0.734  | 0.045s       | ✅ Best Quality
2000       | 0.915     | 0.701  | 0.038s       | ⚡ Best Speed
```

### Output Locations

- **Console output:** Real-time benchmark progress with rich formatting
- **`benchmark_results/`:** Detailed JSON results with timestamps
- **`test_data/`:** Downloaded OSS projects (cached for future runs)
- **Benchmark logs:** Comprehensive logging for debugging

## Integration with Development Workflow

### CI Integration

The authoritative benchmark can be integrated into CI/CD pipelines:

```yaml
# Example CI integration
name: Performance Benchmarks
on:
  pull_request:
    paths: ['src/**', 'benchmarking/**']

jobs:
  benchmark:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    steps:
      - uses: actions/checkout@v3
      - name: Run authoritative benchmark
        run: |
          python benchmarking/authoritative_benchmark.py --skip-oss
      - name: Archive benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

### Custom Benchmark Scenarios

To add custom test scenarios, modify the `authoritative_benchmark.py`:

```python
# Add to _generate_scenarios() method
custom_scenario = BenchmarkScenario(
    name="custom_test",
    collection_name="custom_collection", 
    chunk_size=1500,
    includes_oss=True
)
self.scenarios.append(custom_scenario)

# Add to _generate_test_queries() method
custom_queries = [
    {"text": "your custom query", "type": "semantic", "expected_relevance": "high"}
]
queries.extend(custom_queries)
```

## Troubleshooting

### Common Issues

**Qdrant Connection Problems:**
```bash
# Verify Qdrant is running
curl http://localhost:6333/health

# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant

# Check Qdrant logs
docker logs $(docker ps -q --filter ancestor=qdrant/qdrant)
```

**OSS Project Download Failures:**
- Network connectivity issues may cause download failures
- Use `--skip-oss` flag for faster testing without external dependencies
- Downloaded projects are cached in `test_data/` directory

**Memory Issues:**
- Large OSS projects require significant memory for processing
- Monitor system resources during benchmark execution
- Consider testing fewer chunk sizes or projects if memory limited

**Performance Inconsistencies:**
- Ensure system is idle during benchmarking
- Run multiple benchmark iterations for statistical reliability
- Check for background processes affecting performance

### Debug Mode

```bash
# Enable debug logging
python benchmarking/authoritative_benchmark.py --debug

# Check benchmark logs
tail -f benchmark_results/benchmark.log

# Inspect intermediate results
ls -la benchmark_results/
cat benchmark_results/benchmark_results_*.json | jq .
```

## Migration from Old Benchmark Tools

**If you were using the old benchmark files:**

- `simple_benchmark.py` → Use `authoritative_benchmark.py --skip-oss --chunk-sizes 1000`
- `comprehensive_benchmark.py` → Use `authoritative_benchmark.py --chunk-sizes 1000 2000`  
- `efficient_large_benchmark.py` → Use `authoritative_benchmark.py` (full suite)
- `large_scale_benchmark.py` → Use `authoritative_benchmark.py` (full suite)
- `benchmark_actual_performance.py` → Use `authoritative_benchmark.py` (enhanced version)

**Benefits of migration:**
- Real Qdrant integration instead of simulation
- Large OSS project testing for realistic scenarios
- Chunk size optimization recommendations
- Statistical analysis with confidence intervals
- End-to-end workflow validation

---

*The authoritative benchmark provides comprehensive, realistic performance measurement to replace all previous simulation-based tools.*