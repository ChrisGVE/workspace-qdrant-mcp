# End-to-End (E2E) Test Framework

Comprehensive end-to-end testing framework for workspace-qdrant-mcp using pytest-bdd scenario-based testing.

## Overview

This E2E test framework validates complete system workflows from component startup through ingestion, search, and administration. Built with pytest-bdd for behavior-driven development (BDD) style testing.

**Key Features:**
- pytest-bdd scenario-based testing with Gherkin syntax
- Docker Compose orchestration for all services
- Component lifecycle management with health checks
- Temporary Git project creation for realistic testing
- Test isolation and automatic cleanup
- Resource usage tracking and performance monitoring
- Test orchestration for complex multi-step workflows

## Quick Start

### Run All E2E Tests

```bash
# Run all E2E tests
uv run pytest tests/e2e/ -v

# Run specific scenario
uv run pytest tests/e2e/test_system_startup_scenarios.py -v

# Run with specific markers
uv run pytest tests/e2e/ -m "workflow" -v
uv run pytest tests/e2e/ -m "e2e and not slow" -v
```

### Run with Docker Compose

```bash
# Start services manually
cd docker/integration-tests
docker-compose up -d

# Run E2E tests
uv run pytest tests/e2e/ -v --docker-compose-running

# Stop services
docker-compose down -v
```

## Framework Components

### 1. conftest.py - Core Infrastructure

**Fixtures Provided:**
- `docker_compose_project` - Docker Compose lifecycle management (session scope)
- `e2e_services` - Service URLs and configuration (function scope)
- `temp_project_workspace` - Temporary Git project creation
- `component_lifecycle_manager` - Component startup/shutdown/health checks
- `resource_tracker` - CPU, memory, disk usage monitoring
- `test_orchestrator` - Multi-step workflow coordination
- `scenario_context` - Shared state between pytest-bdd steps

**Configuration:**
```python
E2E_TEST_CONFIG = {
    "docker_compose": {
        "project_name": "wqm-e2e-tests",
        "startup_timeout": 60,
        "shutdown_timeout": 30
    },
    "services": {
        "qdrant": {"http_port": 6333, "grpc_port": 6334},
        "daemon": {"grpc_port": 50051},
        "mcp_server": {"http_port": 8000}
    },
    "timeouts": {
        "short": 10,
        "medium": 30,
        "long": 60,
        "workflow": 120
    }
}
```

### 2. Feature Files - Test Scenarios

**Location:** `tests/e2e/features/*.feature`

**Example Feature:**
```gherkin
Feature: Complete System Workflow
  As a user
  I want to perform complete workflows from ingestion to search
  So that I can validate end-to-end functionality

  Scenario: Complete document ingestion workflow
    Given the system is fully operational
    And I have a test project workspace
    When I create a new Python file "src/main.py"
    Then the daemon should detect the file within 5 seconds
    And the file should be ingested to Qdrant within 10 seconds
    And I should be able to search for the file content
```

**Available Features:**
- `system_startup.feature` - Component startup and initialization
- `complete_workflow.feature` - End-to-end workflows

### 3. Step Definitions

**Location:** `tests/e2e/steps/*.py`

**Common Steps (common_steps.py):**
- Background steps: `the system is not running`, `all previous test data is cleaned up`
- Component startup: `I start {component} service`
- Health checks: `{component} should be healthy within {N} seconds`
- Project setup: `I have a test project workspace`
- File operations: `I create a new Python file "{filename}"`
- Validation: `the daemon should detect the file within {N} seconds`

**Creating Custom Steps:**
```python
from pytest_bdd import given, when, then, parsers

@given("I have a custom setup")
def custom_setup(scenario_context):
    scenario_context.set("custom_key", "custom_value")

@when(parsers.parse("I perform action with {param}"))
def perform_action(param, scenario_context):
    result = do_something(param)
    scenario_context.set("result", result)

@then("the result should be valid")
def validate_result(scenario_context):
    result = scenario_context.get("result")
    assert result is not None
```

### 4. Test Files

**test_system_startup_scenarios.py:**
- Sequential component startup
- Component dependency validation
- Parallel startup
- Startup with missing configuration
- Recovery from partial startup

**test_complete_workflow_scenarios.py:**
- Complete document ingestion workflow
- Multi-file ingestion and search
- Real-time file modification tracking
- Project switching workflow
- Collection management workflow

### 5. Utilities (utils.py)

**HealthChecker:**
```python
from tests.e2e.utils import HealthChecker

# Wait for HTTP endpoint
is_ready = await HealthChecker.wait_for_http_endpoint(
    "http://localhost:8000/health",
    timeout=30
)

# Check gRPC service
is_available = await HealthChecker.check_grpc_service(
    "localhost",
    50051,
    timeout=10
)
```

**WorkflowTimer:**
```python
from tests.e2e.utils import WorkflowTimer

timer = WorkflowTimer()
timer.start()

# ... perform step 1 ...
timer.checkpoint("step1_complete")

# ... perform step 2 ...
timer.checkpoint("step2_complete")

summary = timer.get_summary()
# {"total_duration": 5.2, "checkpoints": [...]}
```

**TestDataGenerator:**
```python
from tests.e2e.utils import TestDataGenerator

# Generate Python module
python_code = TestDataGenerator.create_python_module(
    name="test_module",
    functions=3,
    classes=2
)

# Generate Markdown document
markdown = TestDataGenerator.create_markdown_document(
    title="API Guide",
    sections=5,
    content_per_section=100
)

# Generate config file
yaml_config = TestDataGenerator.create_config_file(
    format="yaml",
    complexity="complex"
)
```

**QdrantTestHelper:**
```python
from tests.e2e.utils import QdrantTestHelper

helper = QdrantTestHelper("http://localhost:6333")

# Create test collection
await helper.create_test_collection("test_collection", vector_size=384)

# Verify document count
is_correct = await helper.verify_document_count("test_collection", expected_count=10)

# Search query
results = await helper.search_test_query("test_collection", "test query")

# Cleanup
await helper.cleanup_test_collections(prefix="test_")
```

## Test Markers

```python
@pytest.mark.e2e              # All E2E tests
@pytest.mark.workflow         # Complete workflow tests
@pytest.mark.stability        # Long-running stability tests
@pytest.mark.performance      # Performance regression tests
@pytest.mark.slow             # Slow tests (>60s)
```

**Running with Markers:**
```bash
# Run only workflow tests
uv run pytest tests/e2e/ -m workflow

# Exclude slow tests
uv run pytest tests/e2e/ -m "e2e and not slow"

# Run stability tests (long-running)
uv run pytest tests/e2e/ -m stability --timeout=3600
```

## Writing New E2E Tests

### 1. Create Feature File

Create `tests/e2e/features/my_feature.feature`:

```gherkin
Feature: My New Feature
  As a user
  I want to test my feature
  So that I can validate it works correctly

  Scenario: Basic feature test
    Given the system is running
    When I perform my action
    Then I should see expected result
```

### 2. Create Step Definitions

Create `tests/e2e/steps/my_feature_steps.py`:

```python
from pytest_bdd import given, when, then, parsers

@when("I perform my action")
def perform_my_action(scenario_context):
    result = my_action()
    scenario_context.set("action_result", result)

@then("I should see expected result")
def validate_result(scenario_context):
    result = scenario_context.get("action_result")
    assert result == "expected"
```

### 3. Create Test File

Create `tests/e2e/test_my_feature.py`:

```python
import pytest
from pytest_bdd import scenarios

scenarios('features/my_feature.feature')

@pytest.mark.e2e
class TestMyFeature:
    """E2E tests for my feature."""
    pass  # Tests defined in feature file
```

## Test Execution Workflow

### Component Lifecycle

1. **Session Setup** (once per test session):
   - Docker Compose project initialized
   - Services available for all tests

2. **Function Setup** (per test):
   - Services ensured running
   - Temporary workspace created
   - Component lifecycle manager initialized
   - Resource tracking started

3. **Scenario Execution**:
   - Background steps run (Given)
   - When steps executed
   - Then steps validated

4. **Function Teardown**:
   - Workspace cleaned up
   - Resources tracked and reported
   - Components stopped if needed

5. **Session Teardown**:
   - Docker Compose services stopped
   - Final cleanup

### Using Component Lifecycle Manager

```python
async def test_custom_workflow(component_lifecycle_manager):
    # Start specific component
    await component_lifecycle_manager.start_component("qdrant")

    # Check health
    health = await component_lifecycle_manager.check_health("qdrant")
    assert health["healthy"]

    # Start all components
    await component_lifecycle_manager.start_all()

    # Wait for ready
    ready = await component_lifecycle_manager.wait_for_ready(timeout=60)
    assert ready

    # Stop specific component
    await component_lifecycle_manager.stop_component("qdrant")
```

### Using Test Orchestrator

```python
async def test_complex_workflow(test_orchestrator):
    async def step1():
        return {"status": "success"}

    def step2():
        return {"status": "success"}

    test_orchestrator.add_step("Step 1", "First step description", step1)
    test_orchestrator.add_step("Step 2", "Second step description", step2)

    results = await test_orchestrator.execute_workflow()

    assert results["successful_steps"] == 2
    print(test_orchestrator.get_summary())
```

## Debugging

### View Test Output

```bash
# Verbose output
uv run pytest tests/e2e/ -v -s

# Show local variables on failure
uv run pytest tests/e2e/ -v -l

# Stop on first failure
uv run pytest tests/e2e/ -x
```

### Check Docker Services

```bash
# Check running containers
docker ps

# View logs
docker-compose -f docker/integration-tests/docker-compose.yml logs qdrant
docker-compose -f docker/integration-tests/docker-compose.yml logs daemon
docker-compose -f docker/integration-tests/docker-compose.yml logs mcp-server

# Check health
curl http://localhost:6333/health
curl http://localhost:8000/health
```

### Resource Monitoring

```bash
# Run with resource tracking
uv run pytest tests/e2e/ -v

# Resource warnings will be printed:
# Resource usage warnings:
#   - Memory usage exceeded: 512.3 MB
```

## Test Data Management

### Temporary Workspaces

Each test gets isolated temporary workspace:

```python
def test_with_workspace(temp_project_workspace):
    workspace_path = temp_project_workspace["path"]

    # Create files
    (workspace_path / "src/new_file.py").write_text("content")

    # Use Git
    from tests.e2e.utils import run_git_command
    run_git_command(["add", "."], cwd=workspace_path)
    run_git_command(["commit", "-m", "test"], cwd=workspace_path)

    # Workspace auto-cleaned after test
```

### Test Collections

Use prefixed collection names for easy cleanup:

```python
COLLECTION_NAME = "test_e2e_workflow"

# Create test collection
await helper.create_test_collection(COLLECTION_NAME)

# Cleanup at end
await helper.cleanup_test_collections(prefix="test_e2e_")
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Start services
        run: |
          cd docker/integration-tests
          docker-compose up -d
          sleep 30

      - name: Run E2E tests
        run: uv run pytest tests/e2e/ -v --junitxml=e2e-results.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: e2e-test-results
          path: e2e-results.xml
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Resource Cleanup**: Always clean up test data
3. **Timeouts**: Use appropriate timeouts for async operations
4. **Markers**: Use markers to categorize tests
5. **Descriptive Scenarios**: Write clear Gherkin scenarios
6. **Reusable Steps**: Create reusable step definitions
7. **Health Checks**: Verify component health before operations
8. **Error Handling**: Handle and test error scenarios

## Troubleshooting

### Tests Hang

- Check component health: `await component_lifecycle_manager.check_health("qdrant")`
- Verify Docker services: `docker ps`
- Increase timeouts in config

### Tests Fail Intermittently

- Check for race conditions
- Add health checks between steps
- Increase debounce/wait times

### Resource Issues

- Check Docker resource limits
- Monitor memory usage with `resource_tracker`
- Clean up test collections

## Performance Benchmarks

Expected performance for E2E tests:

- **Startup Tests**: ~10-30s per scenario
- **Workflow Tests**: ~30-60s per scenario
- **Stability Tests**: 30min - 24hr
- **Full Suite**: ~5-15 minutes (excluding stability)

## Future Enhancements

Planned improvements:
- Distributed tracing integration
- Performance regression detection
- Chaos engineering scenarios
- Multi-region testing
- Load testing scenarios
- Blue-green deployment testing

---

For questions or issues, see the project's main README or create an issue on GitHub.
