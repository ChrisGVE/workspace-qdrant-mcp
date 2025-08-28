# Comprehensive Functional Testing for workspace-qdrant-mcp

This test suite provides comprehensive functional testing with real workspace data, measuring recall/precision, performance benchmarks, and MCP integration quality.

## Test Architecture

The functional test suite consists of five main categories:

### 1. Data Ingestion Tests (`test_data_ingestion.py`)
- **Real Data Collection**: Uses actual workspace-qdrant-mcp source code as test corpus
- **Symbol Extraction**: Extracts Python functions, classes, methods with signatures and docstrings  
- **Chunking Strategy**: Tests document chunking with symbol preservation
- **Metadata Integrity**: Verifies metadata preservation during ingestion
- **Performance Benchmarks**: Measures ingestion throughput and batch processing efficiency

### 2. Search Functionality Tests (`test_search_functionality.py`)  
- **Multi-Modal Search**: Tests semantic, hybrid, and exact search modes
- **Quality Measurement**: Comprehensive recall/precision analysis with ground truth
- **Content-Based Simulation**: Advanced mock search using realistic content matching
- **Ranking Quality**: Validates search result ordering and relevance scoring
- **Cross-Collection Search**: Tests search across multiple collections

### 3. MCP Integration Tests (`test_mcp_integration.py`)
- **End-to-End Tool Testing**: Tests all FastMCP server endpoints
- **Realistic Data Flow**: Uses collected test data in MCP operations  
- **Error Handling**: Comprehensive edge case and error scenario testing
- **Data Consistency**: Verifies data integrity across MCP operations
- **Concurrent Operations**: Tests MCP tool performance under concurrent load

### 4. Performance Tests (`test_performance.py`)
- **Response Time Benchmarks**: Search latency across query types and complexities
- **Concurrent Load Testing**: Performance under concurrent operations
- **Memory Profiling**: Memory usage analysis and leak detection
- **Throughput Measurement**: Operations per second across different scenarios
- **Stress Testing**: Sustained high-load performance validation

### 5. Recall/Precision Tests (`test_recall_precision.py`)
- **Ground Truth Validation**: Uses real codebase content for quality measurement
- **Comprehensive Metrics**: Precision, recall, F1, AP, NDCG calculations
- **Query Type Analysis**: Performance across symbol, semantic, and exact queries
- **Cross-Validation**: Quality consistency validation across data splits
- **Ranking Quality**: Search result relevance and ordering assessment

## Test Data Collection

The test suite automatically collects real data from the workspace-qdrant-mcp codebase:

- **Python Source Files**: Functions, classes, methods with full metadata
- **Documentation**: README files, docstrings, comments
- **Symbol Index**: Complete symbol mapping with relationships
- **Ground Truth**: Automatically generated test cases with expected results

## Running the Tests

### Quick Start
```bash
# Run all functional tests
python run_comprehensive_tests.py

# Run specific categories
python run_comprehensive_tests.py --categories data_ingestion search_functionality

# Quiet mode
python run_comprehensive_tests.py --quiet

# Custom output directory
python run_comprehensive_tests.py --output-dir my_test_results
```

### Individual Test Categories
```bash
# Data ingestion tests
pytest tests/functional/test_data_ingestion.py -v -m integration

# Search functionality with performance
pytest tests/functional/test_search_functionality.py -v -m "integration or performance"

# MCP integration end-to-end
pytest tests/functional/test_mcp_integration.py -v -m e2e

# Performance benchmarking
pytest tests/functional/test_performance.py -v -m performance

# Quality measurement
pytest tests/functional/test_recall_precision.py -v -m integration
```

### Advanced Options
```bash
# With coverage and benchmarks
pytest tests/functional/ -v \
  --cov=src/workspace_qdrant_mcp \
  --cov-report=html \
  --benchmark-json=benchmarks.json \
  --timeout=300

# Parallel execution
pytest tests/functional/ -v -n auto

# Specific test patterns
pytest tests/functional/ -v -k "symbol_search"
```

## Test Output and Reports

The comprehensive test runner generates detailed reports:

### Generated Files
- **`comprehensive_test_summary.json`**: Machine-readable test summary
- **`TEST_RESULTS.md`**: Human-readable results overview  
- **`*_junit.xml`**: JUnit XML reports for CI/CD integration
- **`*_report.html`**: HTML test reports with details
- **`*_coverage/`**: Code coverage analysis
- **`*_benchmarks.json`**: Performance benchmark data
- **`*_quality_report.json`**: Search quality metrics

### Key Metrics Tracked
- **Code Coverage**: Line and branch coverage percentages
- **Search Quality**: Precision, recall, F1 scores by query type
- **Performance**: Response times, throughput, memory usage
- **Test Success**: Pass/fail rates across categories
- **Data Quality**: Test corpus statistics and ground truth coverage

## Quality Targets

The test suite validates against these quality targets:

### Search Quality
- **Symbol Search Precision**: ≥ 80% for exact symbol matches (measured baseline: 100%)
- **Semantic Search Recall**: ≥ 80% for conceptual queries (measured baseline: 100%)  
- **Overall F1 Score**: ≥ 70% across all query types (measured baseline: 100%)
- **Precision@1**: ≥ 80% for top search result relevance (measured baseline: 100%)

### Performance
- **Search Response Time**: < 200ms average, < 500ms P95
- **Ingestion Throughput**: > 10 documents/second
- **Concurrent Performance**: > 2x throughput improvement
- **Memory Efficiency**: < 1MB per operation

### Reliability  
- **Test Success Rate**: ≥ 95% under normal conditions
- **Error Rate**: < 5% under stress conditions
- **Coverage**: ≥ 80% code coverage across core modules
- **Consistency**: < 20% quality variation across test runs

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run Comprehensive Tests
  run: |
    python run_comprehensive_tests.py --quiet
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_results/
    
- name: Publish Test Report
  uses: mikepenz/action-junit-report@v3
  with:
    report_paths: 'test_results/*_junit.xml'
```

### Performance Regression Detection
The benchmark data can be used to detect performance regressions:

```python
# Compare with baseline benchmarks
python -c "
import json
with open('test_results/performance_benchmarks.json') as f:
    current = json.load(f)
with open('baseline_benchmarks.json') as f:
    baseline = json.load(f)
    
for metric, current_val in current.items():
    baseline_val = baseline.get(metric, {}).get('mean_time_ms', 0)
    if current_val.get('mean_time_ms', 0) > baseline_val * 1.2:
        print(f'REGRESSION: {metric} is 20% slower than baseline')
        exit(1)
"
```

## Troubleshooting

### Common Issues

**Test Data Collection Fails**
- Ensure source code is accessible in the expected directory structure
- Check that Python AST parsing succeeds for all source files

**Search Quality Low**
- Verify ground truth generation logic matches search implementation
- Check that test data includes diverse content types (code, docs, symbols)

**Performance Tests Timeout**
- Increase timeout values for slower systems
- Reduce iteration counts in benchmark configurations

**Memory Tests Fail**
- Close other memory-intensive applications
- Verify system has sufficient RAM (recommended: 4GB+)

### Debug Mode
Enable detailed logging for troubleshooting:

```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
pytest tests/functional/test_data_ingestion.py::TestRealDataIngestion::test_collected_data_quality -v -s
```

## Contributing

When adding new functional tests:

1. **Use Real Data**: Leverage the TestDataCollector for realistic test scenarios
2. **Measure Quality**: Include recall/precision measurements for search features  
3. **Benchmark Performance**: Add timing measurements for new operations
4. **Document Expected Behavior**: Clear assertions with meaningful thresholds
5. **Update Test Runner**: Add new test categories to the comprehensive runner

### Test Data Maintenance
- Periodically review ground truth generation logic
- Update quality targets based on system improvements  
- Ensure test corpus remains representative of real usage

This comprehensive test suite ensures workspace-qdrant-mcp maintains high quality, performance, and reliability standards throughout development.