# CLI Component Tests

Test suite for the `wqm` command-line interface.

## Overview

The CLI provides commands for:
- Document ingestion and management
- Search operations
- Collection management
- Service status and diagnostics
- Configuration management
- Admin operations

## Test Structure

```
cli/
├── nominal/       # Happy path tests
├── edge/          # Edge case tests
├── stress/        # Performance and load tests
├── parsers/       # Document parser tests
└── conftest.py    # CLI-specific fixtures
```

## Test Categories

### Nominal Tests (`nominal/`)
Normal CLI operations:
- Command execution and output
- Document ingestion workflows
- Search operations
- Collection listing and management
- Status and health checks
- Configuration file handling
- Output formatting (JSON, table, YAML)

### Edge Tests (`edge/`)
Edge cases and error handling:
- Invalid command arguments
- Missing configuration files
- Non-existent file paths
- Malformed input files
- Permission errors
- Network failures
- Empty result sets
- Invalid output formats

### Stress Tests (`stress/`)
Performance and scalability:
- Large directory ingestion
- High-volume search operations
- Concurrent command execution
- Large file processing
- Memory usage under load
- Command response times

## Running Tests

```bash
# Run all CLI tests
uv run pytest tests/cli/ -m cli

# Run nominal tests only
uv run pytest tests/cli/nominal/ -m "cli and nominal"

# Run edge case tests
uv run pytest tests/cli/edge/ -m "cli and edge"

# Run stress tests
uv run pytest tests/cli/stress/ -m "cli and stress"

# Run parser tests
uv run pytest tests/cli/parsers/ -m cli

# Run fast tests (exclude slow stress tests)
uv run pytest tests/cli/ -m "cli and not slow"
```

## Markers

Apply these markers to CLI tests:
- `@pytest.mark.cli`: All CLI component tests
- `@pytest.mark.nominal`: Normal operation tests
- `@pytest.mark.edge`: Edge case tests
- `@pytest.mark.stress`: Performance and load tests
- `@pytest.mark.requires_qdrant`: Requires Qdrant server
- `@pytest.mark.slow`: Long-running tests (>10s)

## Fixtures

### Available Fixtures

- `cli_runner`: Typer CLI test runner
- `cli_config_dir`: Temporary configuration directory
- `cli_test_environment`: Environment variables for testing
- `cli_config_file`: Test configuration file
- `sample_cli_commands`: Sample CLI commands
- `mock_cli_output`: Output capture utilities
- `test_documents_dir`: Temporary directory with test documents
- `cli_ingestion_config`: Ingestion configuration
- `cli_search_config`: Search configuration
- `cli_output_formatter`: Output formatting utilities
- `mock_cli_services`: Mock external services
- `cli_performance_monitor`: Performance tracking

### Example Test

```python
import pytest
from typer.testing import CliRunner

@pytest.mark.cli
@pytest.mark.nominal
def test_status_command(cli_runner, cli_test_environment):
    """Test wqm status command shows server status."""
    from wqm_cli.cli_wrapper import app

    result = cli_runner.invoke(
        app,
        ["status"],
        env=cli_test_environment
    )

    assert result.exit_code == 0
    assert "Server Status" in result.stdout
    assert "healthy" in result.stdout.lower() or "running" in result.stdout.lower()
```

## Command Coverage

Tests should cover all CLI commands:
- `wqm status`: Server status
- `wqm collections list`: List collections
- `wqm collections create`: Create collection
- `wqm collections delete`: Delete collection
- `wqm ingest`: Ingest documents
- `wqm search`: Search documents
- `wqm admin`: Admin operations
- `wqm config`: Configuration management

## Output Validation

Tests validate:
- Exit codes (0 for success, non-zero for errors)
- stdout content and formatting
- stderr for error messages
- Progress indicators
- Table formatting
- JSON output structure
- YAML output validity

## Performance Targets

Nominal CLI performance:
- Simple commands: < 500ms
- Status checks: < 1s
- Directory ingestion: < 5s for 100 files
- Search operations: < 2s
- Collection operations: < 1s

## Test Data

CLI tests use:
- Sample configuration files
- Test document directories
- Mock service responses
- Various file formats (MD, TXT, PDF, DOCX)
- Different directory structures