# Integration Test Docker Environment

This directory contains Docker configurations for running integration tests in isolated environments.

## Quick Start

### Basic Integration Tests
```bash
# Start Qdrant service
docker-compose up -d qdrant

# Wait for service to be ready
docker-compose logs -f qdrant

# Run integration tests
docker-compose --profile test-runner run --rm test-runner
```

### Performance Testing
```bash
# Run performance benchmarks
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories performance --verbose
```

### Multi-Instance Testing
```bash
# Start multiple Qdrant instances
docker-compose --profile multi-instance up -d

# Run tests against multiple instances
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories integration --verbose
```

### Continuous Monitoring
```bash
# Start monitoring service
docker-compose --profile monitoring up -d performance-monitor

# View monitoring logs
docker-compose logs -f performance-monitor
```

## Services

### qdrant
- **Purpose**: Primary Qdrant instance for testing
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Health Check**: HTTP endpoint monitoring
- **Data**: Persistent storage in Docker volume

### qdrant-secondary
- **Purpose**: Secondary instance for multi-instance testing
- **Ports**: 6335 (HTTP), 6336 (gRPC)
- **Profile**: `multi-instance`
- **Use Case**: Testing daemon coordination and resource isolation

### test-runner
- **Purpose**: Execute integration test suite
- **Profile**: `test-runner`
- **Features**:
  - Testcontainers support via Docker socket mounting
  - Coverage report generation
  - Performance benchmark execution
  - Test result artifacts

### performance-monitor
- **Purpose**: Continuous performance monitoring
- **Profile**: `monitoring`
- **Features**:
  - Real-time performance metrics
  - Regression detection
  - Historical trend analysis

## Volumes

- `qdrant_storage`: Primary Qdrant data persistence
- `qdrant_storage_secondary`: Secondary Qdrant data persistence
- `test_results`: Test execution results and reports
- `coverage_reports`: HTML coverage reports
- `performance_data`: Performance benchmarks and metrics

## Network

- **Name**: `integration-test`
- **Type**: Bridge network with custom subnet
- **Isolation**: Services can communicate but are isolated from host network

## Usage Examples

### Run Specific Test Categories
```bash
# Smoke tests only
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories smoke

# Performance tests with coverage disabled
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories performance --no-coverage

# All tests with parallel execution
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories all --parallel
```

### Access Test Results
```bash
# Copy coverage reports from container
docker-compose --profile test-runner run --rm test-runner bash -c "
  python scripts/run_integration_tests.py --categories integration &&
  cp -r htmlcov/* /app/coverage_reports/
"

# View results
docker-compose run --rm -v $(pwd)/htmlcov:/host_htmlcov test-runner \
  cp -r /app/htmlcov/* /host_htmlcov/
```

### Debug Test Failures
```bash
# Interactive shell in test environment
docker-compose --profile test-runner run --rm test-runner bash

# Run specific test file
docker-compose --profile test-runner run --rm test-runner \
  python -m pytest tests/integration/test_document_ingestion_pipeline.py -v

# Run with debugging output
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories integration --verbose
```

### Performance Monitoring
```bash
# Start monitoring and view real-time logs
docker-compose --profile monitoring up performance-monitor

# Extract performance data
docker-compose run --rm -v $(pwd)/perf_results:/host_perf performance-monitor \
  cp -r /app/performance_results/* /host_perf/
```

### Cleanup
```bash
# Stop all services
docker-compose down

# Remove volumes (data loss!)
docker-compose down -v

# Clean up everything including images
docker-compose down -v --rmi all
```

## Environment Variables

### Test Configuration
- `QDRANT_HOST`: Qdrant service hostname (default: `qdrant`)
- `QDRANT_PORT`: Qdrant HTTP port (default: `6333`)
- `QDRANT_GRPC_PORT`: Qdrant gRPC port (default: `6334`)
- `TEST_ENV`: Test environment identifier (default: `integration`)
- `INTEGRATION_TESTING`: Flag for integration test mode (default: `1`)

### Performance Monitoring
- `MONITOR_INTERVAL`: Monitoring interval in seconds (default: `60`)
- `PERFORMANCE_THRESHOLD`: Regression threshold percentage (default: `20`)

## Troubleshooting

### Service Won't Start
```bash
# Check service logs
docker-compose logs qdrant

# Verify health status
docker-compose ps

# Test connectivity
docker-compose exec qdrant curl http://localhost:6333/health
```

### Test Failures
```bash
# Run tests with detailed output
docker-compose --profile test-runner run --rm test-runner \
  python -m pytest tests/integration/ -v -s --tb=long

# Check test environment
docker-compose --profile test-runner run --rm test-runner env
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check Qdrant performance
docker-compose exec qdrant curl http://localhost:6333/metrics

# View performance monitoring data
docker-compose logs performance-monitor
```

### Docker Socket Issues (Linux)
```bash
# Ensure Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# Alternative: Run with Docker group
docker-compose --profile test-runner run --rm --group-add $(getent group docker | cut -d: -f3) test-runner
```

## CI/CD Integration

This Docker setup is designed to work seamlessly with the GitHub Actions workflows:

- `integration-tests.yml`: Uses similar container setup
- `performance-monitoring.yml`: Leverages performance monitoring patterns
- Local development mirrors CI environment exactly

## Security Considerations

- Docker socket mounting provides container access - use only in trusted environments
- Network isolation prevents external access during testing
- Volumes contain temporary test data - clean up regularly
- Use specific image tags in production environments