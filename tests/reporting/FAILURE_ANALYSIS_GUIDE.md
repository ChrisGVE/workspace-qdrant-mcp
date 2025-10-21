# Failure Pattern Analysis and Flakiness Detection Guide

This guide explains how to interpret and use the failure analysis features in the test reporting system.

## Overview

The failure analysis system provides:
- **Flakiness Detection**: Identifies tests with inconsistent results
- **Failure Categorization**: Classifies failures by type
- **Pattern Grouping**: Groups similar errors together
- **Trend Analysis**: Tracks failure rates over time

## Flakiness Scores

### What is Flakiness?

A test is "flaky" if it produces inconsistent results across multiple runs without code changes. The flakiness score measures this inconsistency on a 0-100 scale.

### Score Interpretation

| Score Range | Severity | Meaning | Action Required |
|------------|----------|---------|-----------------|
| 0 | None | Test consistently passes or fails | No action needed |
| 1-10 | Low | Occasional flakiness | Monitor, may be environmental |
| 11-20 | Moderate | Regular inconsistency | Investigate and fix |
| 21-40 | High | Frequent flakiness | Priority fix required |
| 41+ | Critical | Extremely unreliable | Disable or rewrite immediately |

### Score Calculation

```
Flakiness Score = (min(pass_count, fail_count + error_count) / total_runs) × 100
```

**Examples:**
- 10 passes, 0 failures: Score = 0 (consistent)
- 5 passes, 5 failures: Score = 50 (highly flaky)
- 9 passes, 1 failure: Score = 10 (slightly flaky)
- 0 passes, 10 failures: Score = 0 (consistently fails)

### Why Flaky Tests Matter

Flaky tests:
- Reduce confidence in test suite
- Waste developer time investigating false failures
- May mask real bugs
- Slow down CI/CD pipelines with retries

## Failure Categories

The system automatically categorizes failures into six types:

### 1. ASSERTION
**What**: Assertion errors, value mismatches, logical errors

**Patterns Detected**:
- `AssertionError`
- `expected X got Y`
- `should be X but was Y`
- `assert` statements

**Common Causes**:
- Logic bugs in code under test
- Incorrect test expectations
- Data-dependent failures
- Timing-sensitive comparisons

**Investigation Steps**:
1. Review the assertion and expected vs actual values
2. Check if test data is correct and deterministic
3. Verify business logic in code under test
4. Look for race conditions in async code

### 2. TIMEOUT
**What**: Tests that exceed time limits or hang

**Patterns Detected**:
- `TimeoutError`
- `timed out`
- `deadline exceeded`
- `took too long`

**Common Causes**:
- Slow external dependencies
- Infinite loops or deadlocks
- Insufficient timeout values
- Performance regressions

**Investigation Steps**:
1. Check if timeouts are realistic for the operation
2. Look for blocking operations (network, I/O)
3. Profile code to find bottlenecks
4. Mock slow external services

### 3. SETUP_TEARDOWN
**What**: Failures in test fixtures, setup, or cleanup

**Patterns Detected**:
- `fixture` errors
- `setUp`, `tearDown` failures
- `@pytest.fixture` issues
- `cleanup` errors

**Common Causes**:
- Missing test dependencies
- Improper resource cleanup
- Shared state between tests
- Fixture ordering issues

**Investigation Steps**:
1. Verify all fixtures are properly defined
2. Check fixture dependency order
3. Ensure proper cleanup in teardown
4. Run test in isolation to rule out state issues

### 4. EXTERNAL_DEPENDENCY
**What**: Failures due to external services, databases, APIs

**Patterns Detected**:
- `ConnectionError`
- `connection refused`
- `network` errors
- `database` errors
- `service unavailable`

**Common Causes**:
- Service unavailability
- Network issues
- Authentication failures
- API rate limits

**Investigation Steps**:
1. Verify service is running and accessible
2. Check credentials and permissions
3. Review service logs
4. Consider mocking external dependencies
5. Add retry logic with exponential backoff

### 5. RESOURCE_EXHAUSTION
**What**: System resource limitations

**Patterns Detected**:
- `MemoryError`, `OutOfMemory`
- `disk full`
- `too many open files`
- `resource exhausted`

**Common Causes**:
- Memory leaks
- File handle leaks
- Insufficient system resources
- Inefficient algorithms

**Investigation Steps**:
1. Monitor resource usage during tests
2. Check for resource leaks (memory, files)
3. Reduce test data size if possible
4. Increase resource limits or optimize code

### 6. UNKNOWN
**What**: Failures that don't match known patterns

**Investigation Steps**:
1. Review error message and traceback
2. Search for similar issues
3. Add pattern to categorization if common
4. Create specific test for reproduction

## Failure Patterns

### What are Failure Patterns?

Failure patterns group similar errors across multiple tests and runs. This helps identify:
- Systemic issues affecting multiple tests
- Common error signatures
- Root causes impacting many tests

### Pattern Analysis

Each pattern shows:
- **Error Signature**: Normalized error message (paths/values removed)
- **Category**: Type of failure
- **Occurrences**: How many times this pattern appeared
- **Affected Tests**: Which tests have this error
- **Timeline**: First and last occurrence

### Using Patterns to Find Root Causes

**Example: Multiple tests failing with same error**
```
Pattern: "ConnectionError: database connection refused"
Occurrences: 15
Affected Tests: test_user_login, test_create_post, test_update_profile, ...
```

**Diagnosis**: Database service is down or unreachable, affecting all DB-dependent tests.

**Action**: Fix database service, not individual tests.

## Using the Failure Analysis

### In HTML Reports

Failure analysis sections include:
1. **Flaky Tests Table**: Sorted by flakiness score (highest first)
2. **Failure Patterns Table**: Sorted by occurrence count
3. **Category Distribution Chart**: Pie chart showing failure types
4. **Flaky Tests Chart**: Bar chart of top 10 flaky tests
5. **Failure Patterns Chart**: Bar chart of most common patterns

### Programmatic Access

```python
from tests.reporting.failure_analyzer import FailureAnalyzer
from tests.reporting.storage import TestResultStorage

# Initialize
storage = TestResultStorage()
analyzer = FailureAnalyzer(storage)

# Analyze recent test runs (last 30 days)
report = analyzer.analyze_test_runs(days=30, min_flakiness_score=5.0)

# Access results
print(f"Found {report.total_flaky_tests} flaky tests")
print(f"Found {report.total_failure_patterns} failure patterns")

# Top flaky tests
for metrics in report.flaky_tests[:5]:
    print(f"{metrics.test_case_name}: {metrics.flakiness_score:.1f}% flaky")
    print(f"  Pass rate: {metrics.pass_rate:.1f}%")
    print(f"  Total runs: {metrics.total_runs}")

# Common failure patterns
for pattern in report.failure_patterns[:5]:
    print(f"{pattern.error_signature}")
    print(f"  Category: {pattern.category.value}")
    print(f"  Occurred: {pattern.occurrences} times")
    print(f"  Affected: {len(pattern.affected_tests)} tests")
```

### Analyzing Specific Runs

```python
# Analyze specific test runs
run_ids = ["run-1", "run-2", "run-3"]
report = analyzer.analyze_test_runs(run_ids=run_ids)

# Save report for future reference
storage.save_failure_analysis_report(report)

# Retrieve later
retrieved = storage.get_failure_analysis_report(report.report_id)
```

## Workflow: Investigating Failures

### 1. Identify Problematic Tests

Start with the flaky tests table in your report:
- Focus on tests with flakiness score > 20
- Prioritize by test importance and score
- Check how long the test has been flaky

### 2. Review Failure Patterns

Look at common failure patterns:
- Are multiple tests failing with the same error?
- Is there a systemic issue (e.g., service down)?
- Can you fix one root cause instead of many tests?

### 3. Categorize and Prioritize

Use failure categories to guide investigation:
- **Timeouts**: Performance issue or unrealistic timeout?
- **External Dependencies**: Service availability or mocking needed?
- **Resource Exhaustion**: Memory leak or insufficient resources?
- **Assertions**: Logic bug or incorrect expectations?

### 4. Examine Trends

Check the failure trend:
- **Increasing**: Recent code change introduced instability
- **Decreasing**: Fixes are working, continue monitoring
- **Stable**: Long-standing issue needing focused attention

### 5. Take Action

Based on findings:
- Fix flaky tests (add waits, fix race conditions, mock dependencies)
- Address root causes (fix services, increase timeouts, add resources)
- Quarantine extremely flaky tests (skip until fixed)
- Update test expectations if requirements changed

## Best Practices

### 1. Regular Monitoring
- Review failure analysis reports weekly
- Track flakiness trends over time
- Set alerts for sudden increases in flakiness

### 2. Flakiness Thresholds
- Set team standards (e.g., no tests with score > 10)
- Fail CI builds if average flakiness exceeds threshold
- Require fixes before merging if introducing flaky tests

### 3. Pattern Analysis
- Review top 10 failure patterns monthly
- Look for recurring categories (e.g., many timeouts = performance issue)
- Create reusable solutions for common patterns

### 4. Test Hygiene
- Write deterministic tests (avoid randomness, fixed test data)
- Properly mock external dependencies
- Use appropriate timeouts and retries
- Clean up resources in teardown
- Avoid shared state between tests

### 5. Documentation
- Document known flaky tests and their causes
- Add comments explaining timing-sensitive assertions
- Create runbooks for common failure patterns

## Example Report Interpretation

### Scenario: High Flakiness After Deployment

**Observations from Report**:
- 12 flaky tests (was 2 last week)
- Flakiness trend: "increasing"
- Top pattern: "ConnectionError: service timeout" (25 occurrences)
- Category distribution: 60% External Dependency, 30% Timeout

**Diagnosis**:
Recent deployment caused service performance degradation. Multiple tests timing out when calling the service.

**Actions**:
1. Investigate service performance
2. Check for resource issues (CPU, memory)
3. Review recent code changes
4. Consider increasing timeouts temporarily
5. Add more comprehensive mocking

**Long-term Fix**:
Improve service performance and reduce test dependencies on real service.

## Integration with CI/CD

### Fail Build on High Flakiness

```yaml
# GitHub Actions example
- name: Analyze Test Failures
  run: |
    python -m tests.reporting.check_flakiness --max-score 15 --max-tests 5
```

### Generate Report After Test Run

```yaml
- name: Run Tests
  run: pytest

- name: Generate Failure Analysis
  run: |
    python -m tests.reporting.generate_report --with-failure-analysis

- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: test-report
    path: reports/test-report.html
```

## Troubleshooting

### False Positives

**Issue**: Test marked as flaky but seems stable

**Causes**:
- Test was fixed recently, old failures still in analysis window
- Environmental differences between test runs
- Data cleanup issues

**Solutions**:
- Reduce analysis time window (analyze last 7 days instead of 30)
- Clear old test results: `storage.delete_test_run(old_run_id)`
- Run test multiple times to verify stability

### Missing Patterns

**Issue**: Expected pattern not detected

**Causes**:
- Error signature normalization too aggressive
- Pattern occurs less than minimum threshold
- Category patterns incomplete

**Solutions**:
- Check raw error messages in failed_tests
- Adjust min_flakiness_score parameter
- Add new patterns to FailureAnalyzer class

## Further Reading

- [Test Reporting Documentation](./README.md)
- [Test Result Storage](./storage.py)
- [Failure Analyzer Source](./failure_analyzer.py)
- [Report Generator](./report_generator.py)
