# Executive Performance Dashboard - Usage Guide

This guide explains how to use the performance trend visualization and executive dashboard features for workspace-qdrant-mcp test reporting.

## Overview

The executive dashboard provides comprehensive performance trend analysis and health monitoring for your test suite. It integrates data from multiple test runs to identify trends, detect issues, and provide actionable insights.

## Features

### 1. Executive Summary
- Key metrics at-a-glance (total tests, success rate, coverage, execution time)
- Overall health status (excellent/good/warning/critical)
- Trend indicators for key metrics (↑/→/↓)
- Action items based on current health and trends

### 2. Health Indicators
- **Success Rate**: Test pass percentage with trend
- **Coverage**: Line/function/branch coverage with trend
- **Failed Tests**: Count of failing tests
- **Flakiness**: Number of flaky tests detected

### 3. Performance Trend Charts
- **Success Rate Trends**: 7, 30, 90-day views
- **Execution Time Trends**: Total test execution time over time
- **Coverage Trends**: Line, function, and branch coverage trends
- **Flakiness Trends**: Flaky test counts and failure patterns
- **Benchmark Performance**: Individual benchmark trends

### 4. Interactive Features
- Chart.js interactive charts with zoom/pan
- Time range selection (7/30/90 days)
- Drill-down capabilities
- Export to PDF/PNG

## Quick Start

### Generate Dashboard from Python

```python
from tests.reporting.report_generator import ReportGenerator
from tests.reporting.storage import TestResultStorage

# Initialize
storage = TestResultStorage()
generator = ReportGenerator(storage)

# Get latest test run ID
runs = storage.list_test_runs(limit=1)
latest_run_id = runs[0]["run_id"]

# Generate dashboard
html_content = generator.generate_dashboard_report(
    run_id=latest_run_id,
    output_path="dashboard.html",
    time_windows=[7, 30, 90],  # Days to analyze
)

print(f"Dashboard generated: dashboard.html")
```

### Generate Dashboard from CLI

```bash
# Using the report generator directly
python -m tests.reporting.report_generator \
    --run-id <run_id> \
    --output dashboard.html \
    --type dashboard \
    --time-windows 7,30,90
```

## API Reference

### TrendAnalyzer

Analyzes historical trends in test results.

```python
from tests.reporting.trend_analyzer import TrendAnalyzer
from tests.reporting.storage import TestResultStorage

storage = TestResultStorage()
analyzer = TrendAnalyzer(storage)

# Analyze success rate trend
success_trend = analyzer.analyze_success_rate_trend(days=30)
print(f"Trend direction: {success_trend['trend_direction']}")
print(f"Current: {success_trend['statistics']['current']:.2f}%")

# Analyze coverage trend
coverage_trend = analyzer.analyze_coverage_trend(days=30)
print(f"Coverage trend: {coverage_trend['trend_direction']}")

# Analyze execution time trend
exec_trend = analyzer.analyze_execution_time_trend(days=30)
print(f"Execution time trend: {exec_trend['trend_direction']}")

# Analyze flakiness
flakiness_trend = analyzer.analyze_flakiness_trend(days=30)
print(f"Flaky tests: {flakiness_trend['statistics']['current']}")

# Analyze benchmark performance
benchmark_trend = analyzer.analyze_benchmark_performance_trend(
    benchmark_name="search_latency",
    metric="avg_ms",
    days=30
)
print(f"Benchmark trend: {benchmark_trend['trend_direction']}")

# Calculate health status
from tests.reporting.models import TestRun

test_run = storage.get_test_run(run_id)
health = analyzer.calculate_health_status(test_run)
print(f"Health status: {health}")

# Get action items
trends = {
    "success_rate": success_trend,
    "coverage": coverage_trend,
    "execution_time": exec_trend,
    "flakiness": flakiness_trend,
}
actions = analyzer.get_action_items(health, trends)
for action in actions:
    print(f"- {action}")
```

### PerformanceDashboard

Generates executive dashboards with visualizations.

```python
from tests.reporting.performance_dashboard import PerformanceDashboard
from tests.reporting.storage import TestResultStorage

storage = TestResultStorage()
dashboard = PerformanceDashboard(storage)

# Load test run
test_run = storage.get_test_run(run_id)

# Generate executive summary
summary = dashboard.generate_executive_summary(
    test_run,
    include_trends=True
)

print(f"Health: {summary['health_status']}")
print(f"Success rate: {summary['key_metrics']['success_rate']:.2f}%")
print(f"Coverage: {summary['key_metrics']['line_coverage']:.2f}%")

for action in summary.get("action_items", []):
    print(f"Action: {action}")

# Generate health indicators
indicators = dashboard.generate_health_indicators(test_run)

for name, indicator in indicators.items():
    print(f"{indicator['label']}: {indicator['value']}{indicator['unit']}")
    print(f"  Status: {indicator['status']}")
    print(f"  Trend: {indicator['trend']}")

# Generate dashboard charts
charts = dashboard.generate_dashboard_charts(
    test_run,
    time_windows=[7, 30, 90]
)

# Charts are in Chart.js format, ready to render
print(f"Generated {len(charts)} charts")
```

### ReportGenerator

Enhanced report generator with dashboard support.

```python
from tests.reporting.report_generator import ReportGenerator

generator = ReportGenerator()

# Generate comprehensive HTML report with dashboard
html = generator.generate_html_report(
    run_id=run_id,
    output_path="report.html",
    include_charts=True,
    include_trends=True,  # Includes dashboard data
    include_failure_analysis=True,
)

# Generate dedicated dashboard report
dashboard_html = generator.generate_dashboard_report(
    run_id=run_id,
    output_path="dashboard.html",
    time_windows=[7, 30, 90],
)

# Generate PDF version
pdf_path = generator.generate_pdf_report(
    run_id=run_id,
    output_path="dashboard.pdf",
    include_charts=True,
    include_trends=True,
)
```

## Understanding Health Status

Health status is calculated based on success rate and coverage:

| Status | Success Rate | Coverage | Color |
|--------|-------------|----------|-------|
| Excellent | ≥ 95% | ≥ 80% | Green |
| Good | ≥ 90% | ≥ 70% | Light Green |
| Warning | ≥ 75% | Any | Yellow |
| Critical | < 75% | Any | Red |

## Understanding Trend Direction

Trend direction is calculated using linear regression on historical data:

- **Improving (↑)**: Metric is getting better over time
  - Success rate increasing
  - Coverage increasing
  - Execution time decreasing (lower is better)
  - Flakiness decreasing (lower is better)

- **Stable (→)**: No significant change (< 2% of average)

- **Declining (↓)**: Metric is getting worse over time
  - Success rate decreasing
  - Coverage decreasing
  - Execution time increasing
  - Flakiness increasing

## Time Windows

The dashboard supports multiple time windows for trend analysis:

- **7 Days**: Recent short-term trends
- **30 Days**: Medium-term trends (default)
- **90 Days**: Long-term trends

You can customize time windows:

```python
dashboard_charts = dashboard.generate_dashboard_charts(
    test_run,
    time_windows=[3, 7, 14, 30]  # Custom windows
)
```

## Action Items

The dashboard automatically generates action items based on:

- Current health status
- Trend directions
- Specific thresholds (coverage < 80%, flaky tests > 0, etc.)

Example action items:

- "URGENT: Investigate and fix failing tests immediately" (Critical health)
- "Success rate declining - review recent test failures" (Declining success trend)
- "Code coverage declining - add tests for new code" (Declining coverage trend)
- "Test execution time increasing - optimize slow tests" (Declining execution time trend)
- "Fix 5 flaky tests to improve reliability" (Flaky tests detected)
- "All metrics healthy - maintain current quality standards" (Everything good)

## Sample Dashboard Generator

A sample dashboard generator is provided for demonstration:

```bash
# Run sample generator
uv run python tmp/20251021-0000_sample_dashboard_generator.py
```

This creates:
- 30 historical test runs with realistic trends
- SQLite database with all data
- Interactive HTML dashboard
- Executive summary in console

The generated dashboard demonstrates:
- Success rate improving from 85% to 95%
- Coverage improving from 75% to 85%
- Multiple benchmark trends
- Health indicators and action items

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Generate Test Dashboard
  run: |
    # Run tests and store results
    uv run pytest --json-report --json-report-file=test_results.json

    # Import to storage and generate dashboard
    uv run python -m tests.reporting.import_results test_results.json
    uv run python -m tests.reporting.generate_dashboard --output dashboard.html

- name: Upload Dashboard
  uses: actions/upload-artifact@v3
  with:
    name: test-dashboard
    path: dashboard.html
```

### GitLab CI

```yaml
test_and_dashboard:
  script:
    - uv run pytest --json-report
    - uv run python -m tests.reporting.generate_dashboard
  artifacts:
    paths:
      - dashboard.html
    reports:
      junit: test-results.xml
```

## Exporting Charts

Charts can be exported to PNG for inclusion in reports:

```python
# Using Chart.js Node.js canvas export (requires canvas and chart.js dependencies)
from tests.reporting.performance_dashboard import PerformanceDashboard

dashboard = PerformanceDashboard()
charts = dashboard.generate_dashboard_charts(test_run)

# Charts are in Chart.js JSON format
# Use a headless browser or chart.js-node-canvas for PNG export
# Example with playwright:
# - Load dashboard HTML
# - Screenshot specific chart canvas elements
# - Save as PNG files
```

## Troubleshooting

### Insufficient Data

If you see "insufficient_data" messages, you need more test runs:

- Success/coverage/execution trends: minimum 3 runs
- Flakiness trends: minimum 5 runs

### Empty Charts

If charts are empty:
- Verify test runs exist in the database: `storage.list_test_runs()`
- Check time window overlaps with your test run dates
- Ensure coverage data is being collected (for coverage charts)

### Trend Direction Always Stable

If all trends show "stable":
- Increase time window (use 30 or 90 days instead of 7)
- Ensure sufficient variation in data (not all identical runs)
- Check significance threshold (2% of average value)

## Best Practices

1. **Regular Test Runs**: Run tests regularly (daily) for meaningful trends
2. **Consistent Environment**: Use same environment for comparable results
3. **Historical Data**: Keep at least 90 days of historical data
4. **Review Frequency**: Review dashboard weekly for trends, daily for health
5. **Action Items**: Address critical and warning items before next release
6. **Benchmark Baseline**: Establish baseline performance for benchmarks
7. **Coverage Goals**: Set and maintain minimum coverage thresholds (80%+)
8. **Flakiness**: Fix flaky tests immediately to maintain reliability

## Advanced Usage

### Custom Thresholds

Customize health status thresholds:

```python
# Subclass TrendAnalyzer to customize
class CustomTrendAnalyzer(TrendAnalyzer):
    def calculate_health_status(self, test_run):
        success_rate = test_run.success_rate
        coverage = test_run.coverage.line_coverage_percent if test_run.coverage else 0

        # Custom thresholds
        if success_rate >= 98 and coverage >= 85:
            return HealthStatus.EXCELLENT
        elif success_rate >= 95 and coverage >= 80:
            return HealthStatus.GOOD
        elif success_rate >= 90:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

analyzer = CustomTrendAnalyzer(storage)
dashboard = PerformanceDashboard(storage, analyzer)
```

### Custom Metrics

Track custom metrics in trends:

```python
# Add custom metrics to PerformanceMetrics
perf_metrics = PerformanceMetrics(
    avg_ms=100,
    custom_metrics={
        "memory_usage_mb": 256,
        "cpu_percent": 45,
        "cache_hit_rate": 0.92,
    }
)

# Analyze custom metric trend
custom_trend = analyzer.analyze_benchmark_performance_trend(
    benchmark_name="my_benchmark",
    metric="custom_metrics.cache_hit_rate",
    days=30
)
```

### Dashboard Templates

Create custom dashboard templates using Jinja2:

```html
<!-- templates/my_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Custom Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Executive Dashboard</h1>

    <!-- Health Status -->
    <div class="health-status {{ executive_summary.health_status }}">
        {{ executive_summary.health_status }}
    </div>

    <!-- Charts -->
    {% for chart_id, chart_config in dashboard_charts.items() %}
    <div class="chart-container">
        <canvas id="{{ chart_id }}"></canvas>
        <script>
            new Chart(document.getElementById('{{ chart_id }}'),
                {{ chart_config|tojson }}
            );
        </script>
    </div>
    {% endfor %}
</body>
</html>
```

Use custom template:

```python
html = generator.generate_dashboard_report(
    run_id=run_id,
    template_name="my_dashboard.html"
)
```

## Support

For issues or questions:
- Check test logs: `tests/reporting/test_results.db`
- Verify data: `sqlite3 tests/reporting/test_results.db "SELECT * FROM test_runs LIMIT 5;"`
- Review generated HTML: Inspect browser console for JavaScript errors
- File bug reports with sample data

## Future Enhancements

Planned features:
- Real-time dashboard updates (WebSocket)
- Comparison view (compare two test runs side-by-side)
- Email alerts for critical health status
- Slack/Teams integration for notifications
- Custom dashboard widgets
- Machine learning anomaly detection
- Historical baseline comparison
