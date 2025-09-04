# Benchmark Organization

This directory contains all benchmark and performance testing tools, results, and data for the workspace-qdrant-mcp project.

## Directory Structure

```
dev/benchmarks/
├── README.md                    # This documentation
├── tools/                       # Benchmark scripts and utilities
│   ├── performance_baseline_test.py         # Real-world baseline testing
│   ├── simple_performance_benchmark.py     # Simple performance testing
│   ├── authoritative_benchmark.py          # Comprehensive benchmark suite
│   ├── run_comprehensive_tests.py          # Test runner
│   └── README.md                           # Tool documentation
├── results/                     # All benchmark results (gitignored)
│   ├── benchmark_results/       # Legacy benchmark results
│   └── performance_results/     # Performance test results
└── data/                        # Benchmark data and cache (gitignored)
    └── .benchmarks/             # Pytest-benchmark cache
```

## Usage

### Running Benchmarks

From project root:

```bash
# Simple performance benchmark using existing collections
python dev/benchmarks/tools/simple_performance_benchmark.py

# Comprehensive baseline performance test
python dev/benchmarks/tools/performance_baseline_test.py

# Authoritative benchmark suite (requires large OSS projects)
python dev/benchmarks/tools/authoritative_benchmark.py
```

### Understanding Results

- **results/** - All benchmark output files are automatically stored here
- Results are excluded from git to prevent repository clutter
- Each benchmark tool generates timestamped result files
- JSON files contain raw data, MD files contain human-readable reports

## Development vs Production Separation

### Development Benchmarks
- **Location**: `dev/benchmarks/`
- **Purpose**: Performance testing, regression detection, optimization
- **Git Status**: Tools are tracked, results/data are gitignored
- **Audience**: Developers and CI/CD systems

### Production Code
- **Location**: Main project directories (`src/`, `tests/`, etc.)
- **Purpose**: Actual MCP server functionality
- **Git Status**: Fully tracked
- **Audience**: End users and production deployments

## Key Principles

1. **Clean Repository**: All benchmark artifacts (results, data, cache) are gitignored
2. **Organized Tools**: All benchmark scripts are centralized in `tools/`
3. **Preserved Functionality**: Existing benchmarks work unchanged
4. **Clear Separation**: Development testing is isolated from production code
5. **Documented Structure**: Clear documentation for all components

## Benchmark Tools Overview

### performance_baseline_test.py
- Real-world performance testing against actual Qdrant instance
- Establishes baseline metrics for regression detection
- Tests with actual collections and realistic operations

### simple_performance_benchmark.py
- Quick performance testing using existing collections
- Lightweight testing for rapid feedback
- Focuses on search performance with real data

### authoritative_benchmark.py
- Comprehensive benchmark suite with large OSS projects
- Tests ingestion and search performance at scale
- Includes chunk size optimization and statistical analysis

### run_comprehensive_tests.py
- Test runner for executing multiple benchmark scenarios
- Coordination and reporting for complex test suites
- Integration testing across different configurations

## Best Practices

1. **Run benchmarks before/after major changes** to detect regressions
2. **Use timestamped result files** to track performance over time
3. **Focus on realistic scenarios** that match production usage
4. **Document benchmark methodology** for reproducible results
5. **Keep tools updated** as the codebase evolves

## Git Workflow

```bash
# Benchmark artifacts are automatically gitignored
git status  # Should not show results/ or data/ directories

# Only tool changes should be committed
git add dev/benchmarks/tools/new_benchmark.py
git commit -m "feat: add new benchmark tool"

# Results stay local for analysis
ls dev/benchmarks/results/  # View results locally
```

This organization ensures clean repository management while providing comprehensive performance testing capabilities.