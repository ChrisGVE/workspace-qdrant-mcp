# Performance Regression Detection

Automated system for detecting performance regressions through statistical analysis of benchmark results, with support for CI/CD integration and historical trend tracking.

## Overview

The regression detection system compares current benchmark results against baseline measurements to identify statistically significant performance changes. It uses advanced statistical tests to distinguish real regressions from random variation.

## Features

- **Statistical Significance Testing**: Welch's t-test and Mann-Whitney U test
- **Configurable Thresholds**: Percentage-based regression detection
- **Historical Tracking**: JSON export/import for trend analysis
- **CI/CD Integration**: Command-line tool with exit codes
- **Comprehensive Reporting**: Regressions, improvements, and stable benchmarks

## Quick Start

### 1. Establish Baseline

```bash
# Run benchmarks and save baseline
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=baseline
```

This creates `.benchmarks/baseline.json` with current performance measurements.

### 2. Make Changes

Modify code, optimize algorithms, or update dependencies.

### 3. Run Regression Detection

```bash
# Run new benchmarks
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-json=current.json

# Detect regressions
uv run python tests/benchmarks/regression_detection.py \
  --baseline .benchmarks/baseline.json \
  --current current.json \
  --threshold 5.0 \
  --fail-on-regression
```

## Regression Detection Tool

### Command-Line Interface

```bash
python tests/benchmarks/regression_detection.py \
  --baseline BASELINE_PATH \
  --current CURRENT_PATH \
  [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--baseline PATH` | Path to baseline benchmark JSON | Required |
| `--current PATH` | Path to current benchmark JSON | Required |
| `--threshold PERCENT` | Regression threshold percentage | 5.0 |
| `--significance ALPHA` | Statistical significance level | 0.05 |
| `--test-method METHOD` | Statistical test (`welch`, `mann_whitney`) | `welch` |
| `--no-stats` | Disable statistical testing | False |
| `--export PATH` | Export report to JSON file | None |
| `--show-stable` | Show stable benchmarks in report | False |
| `--fail-on-regression` | Exit with error if regressions found | False |

### Examples

#### Basic Usage

```bash
# Detect regressions with 5% threshold
python tests/benchmarks/regression_detection.py \
  --baseline baseline.json \
  --current current.json \
  --threshold 5.0
```

#### CI/CD Integration

```bash
# Fail build if regressions > 10%
python tests/benchmarks/regression_detection.py \
  --baseline baseline.json \
  --current current.json \
  --threshold 10.0 \
  --fail-on-regression
```

#### Export Report

```bash
# Save detailed report to JSON
python tests/benchmarks/regression_detection.py \
  --baseline baseline.json \
  --current current.json \
  --threshold 5.0 \
  --export regression_report.json
```

#### Use Different Statistical Test

```bash
# Use Mann-Whitney U test (non-parametric)
python tests/benchmarks/regression_detection.py \
  --baseline baseline.json \
  --current current.json \
  --test-method mann_whitney
```

## Statistical Methods

### Welch's T-Test (Default)

**When to use**: Normal distribution, different variances between samples

**Advantages**:
- Robust to unequal variances
- Good for normally distributed data
- Widely understood and accepted

**Interpretation**:
- p-value < 0.05: Statistically significant difference
- p-value ≥ 0.05: No significant difference (random variation)

**Formula**:
```
t = (mean1 - mean2) / sqrt((var1/n1) + (var2/n2))
```

### Mann-Whitney U Test

**When to use**: Non-normal distribution, outliers present

**Advantages**:
- Non-parametric (no distribution assumptions)
- Robust to outliers
- Works with small sample sizes

**Interpretation**:
- p-value < 0.05: Statistically significant rank difference
- p-value ≥ 0.05: No significant rank difference

**Formula**:
```
U = n1*n2 + (n1*(n1+1))/2 - R1
```
Where R1 is sum of ranks for first sample.

## Report Format

### Console Output

```
================================================================================
PERFORMANCE REGRESSION DETECTION REPORT
================================================================================
Timestamp: 2024-10-21 10:30:45
Baseline: .benchmarks/baseline.json
Threshold: 5.0%
Total Benchmarks: 45

Regressions: 2
Improvements: 5
Stable: 38

REGRESSIONS (slower than baseline):
--------------------------------------------------------------------------------
  test_concurrent_ingestion_10_files
    Baseline: 221.50 ms
    Current:  245.80 ms
    Change:   +10.98%
    P-value:  0.0234

  test_hybrid_search_medium_query
    Baseline: 45.20 ms
    Current:  51.30 ms
    Change:   +13.50%
    P-value:  0.0156

IMPROVEMENTS (faster than baseline):
--------------------------------------------------------------------------------
  test_parse_large_python_file
    Baseline: 125.30 ms
    Current:  98.45 ms
    Change:   -21.43%
    P-value:  0.0089

  [Additional improvements...]

================================================================================
```

### JSON Export Format

```json
{
  "timestamp": "2024-10-21T10:30:45.123456",
  "baseline_path": ".benchmarks/baseline.json",
  "current_path": "current.json",
  "threshold_percent": 5.0,
  "total_benchmarks": 45,
  "summary": {
    "regressions": 2,
    "improvements": 5,
    "stable": 38
  },
  "regressions": [
    {
      "name": "test_concurrent_ingestion_10_files",
      "baseline_mean": 0.22150,
      "current_mean": 0.24580,
      "percent_change": 10.98,
      "p_value": 0.0234
    }
  ],
  "improvements": [
    {
      "name": "test_parse_large_python_file",
      "baseline_mean": 0.12530,
      "current_mean": 0.09845,
      "percent_change": -21.43,
      "p_value": 0.0089
    }
  ]
}
```

## Threshold Guidelines

### Conservative (Production)

```bash
--threshold 10.0  # Only flag >10% regressions
```

**Use when**:
- Production deployments
- Release decisions
- Customer-facing changes

### Moderate (Development)

```bash
--threshold 5.0  # Flag >5% regressions (default)
```

**Use when**:
- Feature development
- PR reviews
- Continuous integration

### Aggressive (Optimization)

```bash
--threshold 2.0  # Flag >2% regressions
```

**Use when**:
- Performance optimization work
- Benchmarking experiments
- Detailed performance analysis

## CI/CD Integration

### GitHub Actions

Complete workflow in `.github/workflows/benchmark-regression.yml`:

```yaml
- name: Run benchmarks
  run: |
    uv run pytest tests/benchmarks/ \
      --benchmark-only \
      --benchmark-json=current.json

- name: Detect regressions
  run: |
    uv run python tests/benchmarks/regression_detection.py \
      --baseline .benchmarks/baseline.json \
      --current current.json \
      --threshold 10.0 \
      --export regression_report.json \
      --fail-on-regression

- name: Upload report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: regression-report
    path: regression_report.json
```

### GitLab CI

```yaml
regression_check:
  stage: test
  script:
    - uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=current.json
    - |
      uv run python tests/benchmarks/regression_detection.py \
        --baseline baseline.json \
        --current current.json \
        --threshold 10.0 \
        --fail-on-regression
  artifacts:
    when: always
    paths:
      - current.json
      - regression_report.json
```

### Jenkins

```groovy
stage('Regression Detection') {
  steps {
    sh 'uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=current.json'
    sh '''
      uv run python tests/benchmarks/regression_detection.py \
        --baseline ${WORKSPACE}/baseline.json \
        --current current.json \
        --threshold 10.0 \
        --export regression_report.json \
        --fail-on-regression
    '''
  }
  post {
    always {
      archiveArtifacts artifacts: 'current.json,regression_report.json'
    }
  }
}
```

## Baseline Management

### Creating Baselines

```bash
# Create initial baseline
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=baseline

# This creates: .benchmarks/<timestamp>_baseline.json
```

### Updating Baselines

```bash
# After verifying changes are improvements, update baseline
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=baseline  # Overwrites previous baseline

# Or manually copy
cp .benchmarks/current.json .benchmarks/baseline.json
```

### Multiple Baselines

```bash
# Create named baselines for different scenarios
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=baseline_python311

uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=baseline_python312

# Compare against specific baseline
python tests/benchmarks/regression_detection.py \
  --baseline .benchmarks/baseline_python311.json \
  --current current.json
```

### Baseline Storage Best Practices

1. **Version Control**: Store baseline in git for team consistency
2. **Per-Branch**: Maintain separate baselines for main/dev branches
3. **Platform-Specific**: Create baselines for different OS/hardware
4. **Periodic Updates**: Refresh baselines after validated improvements
5. **Documentation**: Note baseline creation date and conditions

## Historical Trend Tracking

### Collecting Historical Data

```bash
# Run benchmarks regularly and save with timestamps
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-save=benchmark_$(date +%Y%m%d)

# Or use autosave
uv run pytest tests/benchmarks/ \
  --benchmark-only \
  --benchmark-autosave
```

### Analyzing Trends

```bash
# List all saved benchmarks
ls -la .benchmarks/

# Compare across time
pytest-benchmark compare 0001 0002 0003 --group-by=name
```

### Visualization

Use the exported JSON data for custom visualizations:

```python
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load multiple reports
reports = []
for file in sorted(Path('.benchmarks/reports/').glob('*.json')):
    with open(file) as f:
        reports.append(json.load(f))

# Extract trend data
benchmark_name = "test_concurrent_ingestion_10_files"
timestamps = [datetime.fromisoformat(r['timestamp']) for r in reports]
means = [next(
    (b['current_mean'] for b in r['regressions'] + r['improvements']
     if b['name'] == benchmark_name),
    None
) for r in reports]

# Plot trend
plt.plot(timestamps, means)
plt.xlabel('Time')
plt.ylabel('Execution Time (s)')
plt.title(f'Performance Trend: {benchmark_name}')
plt.savefig('performance_trend.png')
```

## Python API

### Programmatic Usage

```python
from regression_detection import RegressionDetector

# Initialize detector
detector = RegressionDetector(
    baseline_path="baseline.json",
    use_statistical_test=True,
    test_method="welch"
)

# Load data
detector.load_baseline()
detector.load_current_results("current.json")

# Detect regressions
regressions = detector.detect_regressions(
    threshold_percent=5.0,
    significance_level=0.05
)

# Generate full report
report = detector.generate_report(
    threshold_percent=5.0,
    significance_level=0.05
)

# Print report
detector.print_report(
    report,
    show_stable=False,
    show_improvements=True
)

# Export to JSON
detector.export_report_json(report, "regression_report.json")

# Check for failures
if report.regressions:
    print(f"Found {len(report.regressions)} regressions!")
    for regression in report.regressions:
        print(f"  - {regression.name}: {regression.percent_change:+.2f}%")
```

### Custom Analysis

```python
from regression_detection import StatisticalTests

# Perform custom statistical tests
baseline_times = [0.221, 0.225, 0.219, 0.223, 0.220]
current_times = [0.245, 0.248, 0.242, 0.250, 0.246]

# Welch's t-test
is_significant, p_value = StatisticalTests.welch_t_test(
    baseline_times,
    current_times,
    alpha=0.05
)

print(f"Significant: {is_significant}, p-value: {p_value:.4f}")

# Mann-Whitney U test
is_significant, p_value = StatisticalTests.mann_whitney_u_test(
    baseline_times,
    current_times,
    alpha=0.05
)

print(f"Significant (MWU): {is_significant}, p-value: {p_value:.4f}")
```

## Troubleshooting

### False Positives

**Symptom**: Many benchmarks flagged as regressions despite no code changes

**Solutions**:
1. Increase threshold: `--threshold 10.0`
2. Use statistical testing: Default enabled
3. Reduce system noise: Close applications, disable power management
4. Increase benchmark iterations: More samples for better statistics
5. Check baseline validity: Ensure baseline was created under similar conditions

### False Negatives

**Symptom**: Known regressions not detected

**Solutions**:
1. Decrease threshold: `--threshold 2.0`
2. Check statistical power: Need more benchmark iterations
3. Verify baseline freshness: Update if too old
4. Review regression magnitude: Small changes may not reach threshold

### P-Value Interpretation

**Symptom**: Confused about p-value meaning

**Interpretation**:
- p < 0.05: Less than 5% chance difference is random
- p < 0.01: Less than 1% chance difference is random
- p ≥ 0.05: Likely due to random variation, not real change

**Note**: p-value measures confidence, not magnitude. A small regression (2%) can be highly significant (p=0.001) with enough samples.

### Baseline Compatibility

**Symptom**: Error loading baseline or current results

**Solutions**:
1. Verify JSON format: Must be pytest-benchmark format
2. Check file paths: Use absolute paths or check working directory
3. Validate JSON syntax: Use `python -m json.tool baseline.json`
4. Ensure benchmark names match: Compare benchmark names in both files

## Best Practices

### 1. Establish Clean Baselines

- Run on dedicated hardware
- Close unnecessary applications
- Use consistent environment (Python version, dependencies)
- Run multiple times and average if needed
- Document baseline conditions

### 2. Regular Baseline Updates

- Update after validated performance improvements
- Don't update for every change
- Maintain history of baselines
- Document reasons for updates

### 3. Threshold Selection

- Start conservative (10%)
- Tighten for optimization work (2-5%)
- Consider business impact
- Balance false positives vs false negatives

### 4. Statistical Testing

- Always use statistical tests in CI/CD
- Use Welch's t-test for normal distributions
- Use Mann-Whitney for non-normal data or outliers
- Understand p-value interpretation

### 5. Historical Tracking

- Export reports to JSON regularly
- Store in version control or artifact storage
- Visualize trends over time
- Track both regressions and improvements

### 6. CI/CD Integration

- Fail builds on significant regressions
- Post reports to PRs for visibility
- Archive benchmark results
- Update baselines on main branch only

## References

- [Statistical Testing in Performance Analysis](https://en.wikipedia.org/wiki/Student%27s_t-test)
- [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
- [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance Testing Best Practices](https://martinfowler.com/articles/performance-testing.html)
