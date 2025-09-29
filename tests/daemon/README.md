# Daemon Component Tests

Test suite for the Rust daemon component of workspace-qdrant-mcp.

## Overview

The daemon component handles:
- File system watching and change detection
- Document processing and chunking
- gRPC communication with Python MCP server
- Asynchronous task queue management

## Test Structure

```
daemon/
├── nominal/       # Happy path tests
├── edge/          # Edge case tests
├── stress/        # Performance and load tests
└── conftest.py    # Daemon-specific fixtures
```

## Test Categories

### Nominal Tests
Normal operation scenarios:
- Daemon startup and shutdown
- File watch event handling
- Document processing pipeline
- gRPC communication
- Status and health checks

### Edge Tests
Edge cases and error handling:
- Invalid file paths
- Corrupted documents
- Connection failures
- Resource exhaustion
- Race conditions
- Signal handling

### Stress Tests
Performance and scalability:
- High-volume file watching
- Concurrent processing
- Large file handling
- Memory pressure
- CPU saturation
- gRPC throughput

## Running Tests

```bash
# Run all daemon tests
uv run pytest tests/daemon/ -m daemon

# Run nominal tests only
uv run pytest tests/daemon/ -m "daemon and nominal"

# Run edge case tests
uv run pytest tests/daemon/ -m "daemon and edge"

# Run stress tests
uv run pytest tests/daemon/ -m "daemon and stress"

# Run without Rust daemon (mocked)
uv run pytest tests/daemon/ -m "daemon and not requires_rust"
```

## Markers

Apply these markers to daemon tests:
- `@pytest.mark.daemon`: All daemon component tests
- `@pytest.mark.nominal`: Normal operation tests
- `@pytest.mark.edge`: Edge case tests
- `@pytest.mark.stress`: Performance tests
- `@pytest.mark.requires_rust`: Requires compiled Rust daemon
- `@pytest.mark.slow`: Long-running tests (>10s)

## Fixtures

### Available Fixtures

- `daemon_config`: Test configuration for daemon
- `mock_daemon_client`: Mocked daemon client (no Rust required)
- `daemon_process_manager`: Daemon lifecycle management
- `grpc_channel_config`: gRPC channel configuration
- `sample_watch_events`: Sample file system events

### Example Test

```python
import pytest

@pytest.mark.daemon
@pytest.mark.nominal
async def test_daemon_startup(daemon_process_manager, daemon_config):
    """Test daemon starts successfully with valid configuration."""
    success = await daemon_process_manager.start(daemon_config)
    assert success
    assert daemon_process_manager.is_running
    assert daemon_process_manager.get_pid() is not None
```

## Current Status

**Note**: This test directory is a placeholder for future daemon tests.
The Rust daemon is currently under development. Tests use mocked daemon
clients until the actual Rust implementation is complete.

## Future Work

- [ ] Implement actual Rust daemon process management
- [ ] Add gRPC protocol tests
- [ ] Add file watching integration tests
- [ ] Add document processing pipeline tests
- [ ] Add performance benchmarks
- [ ] Add concurrent processing tests