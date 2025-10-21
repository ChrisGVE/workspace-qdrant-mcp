# Test Coverage Tracking

Comprehensive code coverage tracking and reporting for workspace-qdrant-mcp.

## Overview

The coverage tracking system integrates with the test reporting infrastructure to provide:

- **Python Coverage**: Uses `coverage.py` with pytest
- **Rust Coverage**: Uses `cargo-tarpaulin` for Rust code
- **Coverage Thresholds**: Configurable pass/fail criteria
- **Visualization**: Charts and tables in HTML/PDF reports
- **CI/CD Integration**: Exit codes and threshold checking

## Quick Start

### Run Tests with Coverage

```bash
# Run all tests with coverage (default thresholds)
python tests/reporting/run_tests_with_coverage.py

# Use strict thresholds
python tests/reporting/run_tests_with_coverage.py --threshold=strict

# Python tests only
python tests/reporting/run_tests_with_coverage.py --python-only

# Custom output location
python tests/reporting/run_tests_with_coverage.py --output=my_report.html
```

### Manual Coverage Generation

**Python:**
```bash
# Run pytest with coverage
pytest --cov=src --cov-report=xml:coverage.xml --cov-report=html:htmlcov

# Parse coverage data
python -c "
from tests.reporting.parsers.coverage_py_parser import parse_coverage_xml
from pathlib import Path
coverage = parse_coverage_xml(Path('coverage.xml'))
print(f'Coverage: {coverage.line_coverage_percent:.2f}%')
"
```

**Rust:**
```bash
# Generate coverage with tarpaulin
cd src/rust/daemon
cargo tarpaulin --out Json --output-dir ../../../coverage_reports

# Or use LCOV format
cargo tarpaulin --out Lcov --output-dir ../../../coverage_reports

# Parse coverage data
python -c "
from tests.reporting.parsers.tarpaulin_parser import parse_tarpaulin_json
from pathlib import Path
coverage = parse_tarpaulin_json(Path('coverage_reports/rust_coverage.json'))
print(f'Coverage: {coverage.line_coverage_percent:.2f}%')
"
```

## Coverage Thresholds

### Default Thresholds

```python
from tests.reporting.coverage_checker import CoverageThresholds

thresholds = CoverageThresholds.default()
# Line coverage: 80% minimum, 90% warning
# Function coverage: 70% minimum, 80% warning
# Branch coverage: 60% minimum, 70% warning
# Per-file coverage: 70% minimum
```

### Strict Thresholds

```python
thresholds = CoverageThresholds.strict()
# Line coverage: 90% minimum, 95% warning
# Function coverage: 85% minimum, 90% warning
# Branch coverage: 80% minimum, 85% warning
# Per-file coverage: 85% minimum
```

### Custom Thresholds

```python
from tests.reporting.coverage_checker import CoverageThresholds, CoverageChecker

thresholds = CoverageThresholds(
    line_coverage_min=85.0,
    line_coverage_warning=92.0,
    function_coverage_min=75.0,
    file_line_coverage_min=80.0,
    exclude_files=["tests/", "__init__.py"]
)

checker = CoverageChecker(thresholds)
result = checker.check(coverage)

if not result.passed:
    for violation in result.violations:
        print(violation.message)
    sys.exit(1)
```

## Programmatic Usage

### Parse Coverage Data

```python
from pathlib import Path
from tests.reporting.parsers.coverage_py_parser import parse_coverage_xml
from tests.reporting.parsers.tarpaulin_parser import parse_tarpaulin_json

# Parse Python coverage
python_coverage = parse_coverage_xml(
    Path("coverage.xml"),
    source_root=Path.cwd()
)

# Parse Rust coverage
rust_coverage = parse_tarpaulin_json(
    Path("cobertura.json"),
    source_root=Path.cwd()
)

# Access metrics
print(f"Line coverage: {python_coverage.line_coverage_percent:.2f}%")
print(f"Lines covered: {python_coverage.lines_covered}/{python_coverage.lines_total}")

# Per-file coverage
for file_cov in python_coverage.file_coverage:
    print(f"{file_cov.file_path}: {file_cov.line_coverage_percent:.2f}%")
    if file_cov.uncovered_lines:
        print(f"  Uncovered lines: {file_cov.uncovered_lines}")
```

### Store Coverage with Test Run

```python
from tests.reporting.storage import TestResultStorage
from tests.reporting.models import TestRun, TestSource

storage = TestResultStorage()

# Create test run with coverage
test_run = TestRun.create(source=TestSource.PYTEST)
test_run.coverage = python_coverage

# Save to database
storage.save_test_run(test_run)

# Retrieve later
retrieved_run = storage.get_test_run(test_run.run_id)
if retrieved_run.coverage:
    print(f"Coverage: {retrieved_run.coverage.line_coverage_percent:.2f}%")
```

### Generate Report with Coverage

```python
from tests.reporting.report_generator import ReportGenerator

generator = ReportGenerator(storage)

# Generate HTML report with coverage visualization
html_path = generator.generate_html_report(
    run_id=test_run.run_id,
    output_path="report.html",
    include_charts=True,  # Include coverage charts
    include_trends=False
)

# Generate PDF report
pdf_path = generator.generate_pdf_report(
    run_id=test_run.run_id,
    output_path="report.pdf"
)
```

### Check Thresholds

```python
from tests.reporting.coverage_checker import (
    check_coverage_thresholds,
    CoverageThresholds
)

# Quick check with defaults
result = check_coverage_thresholds(coverage)

# With custom thresholds
thresholds = CoverageThresholds(
    line_coverage_min=90.0,
    file_line_coverage_min=85.0
)
result = check_coverage_thresholds(coverage, thresholds)

# Handle results
if result.passed:
    print("All thresholds passed!")
else:
    print(f"Threshold check failed: {result.message}")
    for violation in result.violations:
        print(f"  FAIL: {violation.message}")
    for warning in result.warnings:
        print(f"  WARN: {warning.message}")
```

## Report Visualization

Coverage data is visualized in HTML/PDF reports:

### Coverage Metrics Section

- **Overall Coverage**: Line/function/branch percentages
- **Covered vs Uncovered**: Visual breakdown of coverage
- **Coverage Trends**: Historical coverage over time

### Coverage Charts

1. **Coverage Gauge**: Doughnut chart showing covered vs uncovered lines
2. **Coverage Breakdown**: Bar chart comparing line/function/branch coverage
3. **File Coverage**: Horizontal bar chart of top 20 files by size
   - Green (>=80%), Amber (>=60%), Red (<60%)

### Coverage Tables

- **Lowest Coverage Files**: Top 10 files needing attention
- **Fully Covered Files**: List of 100% covered files
- **Uncovered Lines**: Per-file list of line numbers not covered

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests with Coverage

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest pytest-cov

      - name: Run tests with coverage
        run: |
          python tests/reporting/run_tests_with_coverage.py \
            --threshold=default \
            --output=coverage_report.html

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: coverage_report.html

      - name: Check coverage thresholds
        run: |
          # Exit code 1 if thresholds not met
          python tests/reporting/run_tests_with_coverage.py --threshold=strict
```

### Local Development

```bash
# Run with relaxed thresholds during development
python tests/reporting/run_tests_with_coverage.py --skip-threshold-check

# This always exits 0, even if coverage is low
# Use for local testing without CI failures
```

## Data Models

### CoverageMetrics

```python
@dataclass
class CoverageMetrics:
    line_coverage_percent: float
    lines_covered: int
    lines_total: int
    function_coverage_percent: Optional[float] = None
    functions_covered: Optional[int] = None
    functions_total: Optional[int] = None
    branch_coverage_percent: Optional[float] = None
    branches_covered: Optional[int] = None
    branches_total: Optional[int] = None
    file_coverage: List[FileCoverage] = field(default_factory=list)
    coverage_tool: Optional[str] = None
```

### FileCoverage

```python
@dataclass
class FileCoverage:
    file_path: str
    lines_covered: int
    lines_total: int
    line_coverage_percent: float
    uncovered_lines: List[int] = field(default_factory=list)
    functions_covered: Optional[int] = None
    functions_total: Optional[int] = None
    branches_covered: Optional[int] = None
    branches_total: Optional[int] = None
```

## Output Files

Running coverage tests generates:

```
coverage_reports/
├── python_coverage.xml          # Coverage.py XML output
├── python_htmlcov/              # HTML coverage browser
│   ├── index.html
│   └── ...
├── rust_coverage.json           # Tarpaulin JSON output
├── rust_lcov.info              # Tarpaulin LCOV output
└── test_report_YYYYMMDD_HHMMSS.html  # Integrated report
```

Test results database:

```
tests/reporting/test_results.db
├── test_runs
├── coverage_metrics             # Overall coverage per run
├── file_coverage               # Per-file coverage details
└── ...
```

## Troubleshooting

### Python Coverage Issues

**Issue**: `coverage.xml` not generated

```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Run with explicit paths
pytest --cov=src/python --cov-report=xml:coverage.xml
```

**Issue**: No coverage data collected

```bash
# Check .coveragerc or pyproject.toml configuration
# Ensure source paths are correct
pytest --cov=src --cov-report=term  # Debug output
```

### Rust Coverage Issues

**Issue**: `cargo-tarpaulin` not found

```bash
# Install tarpaulin
cargo install cargo-tarpaulin
```

**Issue**: Tarpaulin fails on macOS

```bash
# Tarpaulin has limited macOS support
# Use llvm-cov instead:
cargo install cargo-llvm-cov
cargo llvm-cov --lcov --output-path lcov.info
```

### Threshold Check Failures

**Issue**: Threshold failures on new code

```python
# Exclude new files temporarily
thresholds = CoverageThresholds(
    line_coverage_min=80.0,
    exclude_files=["src/new_feature/"]
)
```

**Issue**: Legacy code has low coverage

```python
# Use per-file thresholds
thresholds = CoverageThresholds(
    line_coverage_min=80.0,        # Overall must be 80%
    file_line_coverage_min=50.0,   # But files can be 50%
)
```

## Best Practices

1. **Run coverage locally before CI**: Catch issues early
2. **Use strict thresholds for new code**: Maintain quality
3. **Exclude test files**: Don't measure test coverage
4. **Track coverage trends**: Monitor over time
5. **Review uncovered lines**: Understand gaps
6. **Set realistic thresholds**: Balance coverage and development speed

## See Also

- [Test Aggregation](../README.md) - Overall test reporting system
- [Report Generation](report_generator.py) - HTML/PDF report creation
- [Coverage.py Docs](https://coverage.readthedocs.io/)
- [Tarpaulin Docs](https://github.com/xd009642/tarpaulin)
