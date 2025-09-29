# MCP Server Component Tests

Test suite for the MCP (Model Context Protocol) server component.

## Overview

The MCP server component provides:
- FastMCP-based tool implementations
- Document storage and retrieval
- Hybrid search capabilities
- Collection management
- Scratchbook functionality
- Protocol compliance with MCP specification

## Test Structure

```
mcp_server/
├── nominal/       # Happy path tests
├── edge/          # Edge case tests
├── stress/        # Performance and load tests
└── conftest.py    # MCP server-specific fixtures
```

## Test Categories

### Nominal Tests (`nominal/`)
Normal operation scenarios:
- Tool registration and discovery
- Document storage and retrieval
- Search functionality
- Collection operations
- Scratchbook operations
- Protocol request/response handling
- Client interactions

### Edge Tests (`edge/`)
Edge cases and error handling:
- Invalid tool arguments
- Missing required parameters
- Malformed protocol requests
- Connection failures
- Large document handling
- Special characters in content
- Collection name conflicts
- Empty search results

### Stress Tests (`stress/`)
Performance and scalability:
- Concurrent tool calls
- High-volume document ingestion
- Large search result sets
- Memory pressure scenarios
- Connection pool exhaustion
- Protocol throughput limits

## Running Tests

```bash
# Run all MCP server tests
uv run pytest tests/mcp_server/ -m mcp_server

# Run nominal tests only
uv run pytest tests/mcp_server/nominal/ -m "mcp_server and nominal"

# Run edge case tests
uv run pytest tests/mcp_server/edge/ -m "mcp_server and edge"

# Run stress tests
uv run pytest tests/mcp_server/stress/ -m "mcp_server and stress"

# Run without Qdrant server (using mocks)
uv run pytest tests/mcp_server/ -m "mcp_server and not requires_qdrant"

# Run fast tests only (exclude slow stress tests)
uv run pytest tests/mcp_server/ -m "mcp_server and not slow"
```

## Markers

Apply these markers to MCP server tests:
- `@pytest.mark.mcp_server`: All MCP server component tests
- `@pytest.mark.nominal`: Normal operation tests
- `@pytest.mark.edge`: Edge case tests
- `@pytest.mark.stress`: Performance and load tests
- `@pytest.mark.requires_qdrant`: Requires Qdrant server
- `@pytest.mark.slow`: Long-running tests (>10s)

## Fixtures

### Available Fixtures

- `mcp_server_config`: Test configuration
- `mock_qdrant_client`: Mocked Qdrant client
- `mock_fastmcp_server`: Mocked FastMCP server
- `sample_mcp_tools`: Sample tool definitions
- `sample_mcp_requests`: Sample protocol requests
- `mcp_protocol_validator`: Protocol compliance validation
- `test_workspace_dir`: Temporary workspace directory
- `performance_metrics`: Performance tracking utilities

### Example Test

```python
import pytest

@pytest.mark.mcp_server
@pytest.mark.nominal
async def test_store_document(mock_fastmcp_server, mock_qdrant_client):
    """Test storing a document via MCP tool."""
    # Register tool
    @mock_fastmcp_server.tool("store_document")
    async def store_document(content: str, metadata: dict = None):
        await mock_qdrant_client.upsert(
            collection_name="test",
            points=[{"content": content, "metadata": metadata or {}}]
        )
        return {"success": True, "id": "doc_123"}

    # Call tool
    result = await mock_fastmcp_server.call_tool(
        "store_document",
        {"content": "Test document", "metadata": {"source": "test"}}
    )

    assert result["success"]
    assert "id" in result
```

## Protocol Compliance

Tests ensure compliance with MCP specification:
- JSON-RPC 2.0 message format
- Tool discovery and listing
- Tool execution and result format
- Error handling and reporting
- Resource management
- Prompt templates

## Performance Targets

Nominal performance expectations:
- Tool call latency: < 100ms (p95)
- Document storage: < 500ms per document
- Search operations: < 1000ms (p95)
- Concurrent requests: 100+ req/s
- Memory usage: < 512MB under load

## Test Data

Tests use:
- Sample documents of varying sizes
- Realistic metadata structures
- Common search queries
- Edge case inputs (empty, very long, special characters)
- Protocol-compliant requests and responses