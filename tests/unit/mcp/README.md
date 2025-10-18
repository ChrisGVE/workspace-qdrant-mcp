# MCP Protocol Compliance Test Suite

This directory contains comprehensive tests verifying that workspace-qdrant-mcp's MCP server implementation complies with the official Model Context Protocol specification.

## Overview

**Status:** ✅ **183/183 tests passing (100% compliance)**

All tests use the **official `fastmcp.Client` SDK** following MCP team's recommended testing approach. Tests verify protocol compliance, not business logic - all external dependencies (Qdrant, daemon) are mocked.

## Test Files

### 1. Tool Call Handling Validation (`test_mcp_tool_call_validation.py`)
**64 tests** - Verifies all 4 MCP tools comply with protocol specifications:

- **Store Tool** (13 tests)
  - Schema validation and MCP compliance
  - Required/optional parameter handling
  - Response format (CallToolResult)
  - JSON serialization
  - Invalid parameter type handling

- **Search Tool** (15 tests)
  - Query parameter validation
  - Limit, score_threshold, filters parameters
  - Mode enum validation (hybrid/semantic/keyword)
  - Response structure verification
  - Cross-parameter validation

- **Manage Tool** (17 tests)
  - All 7 action types (create, delete, list, info, backup, restore, stats)
  - Action-specific parameter requirements
  - Collection management operations
  - Error response formats

- **Retrieve Tool** (15 tests)
  - Document ID and metadata retrieval
  - Filter parameters (branch, file_type, project_name)
  - Limit parameter validation
  - Complex metadata filters

- **Cross-Tool Compliance** (4 tests)
  - All tools properly registered
  - Consistent response structures
  - JSON serialization across tools
  - Protocol compliance verification

### 2. Parameter Validation & Error Handling (`test_mcp_parameter_validation.py`)
**67 tests** - Comprehensive error handling and parameter validation:

- **Store Tool** (11 tests)
  - Missing required content parameter
  - Type mismatches (int, list, dict, None)
  - Metadata validation
  - Extra/unexpected parameters
  - Collection parameter validation

- **Search Tool** (18 tests)
  - Missing query parameter
  - Query type validation
  - Limit boundary conditions (negative, zero, extreme values)
  - Score threshold validation (negative, > 1.0)
  - Filters type validation
  - Mode enum validation
  - File type validation

- **Manage Tool** (15 tests)
  - Missing action parameter
  - Action type validation
  - Empty/unknown action values
  - Action-specific missing parameters
  - Name/config parameter validation

- **Retrieve Tool** (15 tests)
  - Missing both ID and metadata
  - Document ID validation
  - Metadata parameter validation
  - Limit boundary conditions
  - Collection/branch/file_type validation

- **Cross-Tool Error Handling** (8 tests)
  - All tools handle None parameters
  - JSON serializable error responses
  - Extra parameter handling
  - Informative error messages
  - No sensitive information leakage
  - Consistent error response structure

### 3. Capability Negotiation (`test_mcp_capability_negotiation.py`)
**43 tests** - Verifies capability exchange and tool discovery:

- **Server Initialization** (4 tests)
  - Server startup and initialization
  - FastMCP app instance validation
  - Implementation info (name, version)
  - Protocol version accessibility

- **Capability Advertisement** (12 tests)
  - Tool manager presence
  - Tool enumeration
  - All 4 tools registered (store, search, manage, retrieve)
  - Unique tool names
  - Complete tool schemas with parameters
  - Type annotations on parameters
  - Function docstrings
  - Required vs optional parameter distinction
  - JSON serializable schemas

- **Version Negotiation** (3 tests)
  - MCP SDK version availability
  - Server accepts current protocol
  - FastMCP version information

- **Feature Detection** (6 tests)
  - Client can list tools
  - Client can retrieve tool schemas
  - Tool parameter inspection
  - Tool descriptions available
  - Parameter types discoverable
  - All tools discoverable via iteration

- **Successful Negotiation** (6 tests)
  - Complete initialization handshake
  - All tools available after init
  - Tool calls work after init
  - Server ready for requests
  - Multiple tools callable
  - Advertised capabilities match functionality

- **Failed Negotiation** (8 tests)
  - Nonexistent tool calls fail gracefully
  - Malformed tool names handled
  - Error response format compliance
  - Invalid parameters handled
  - Server continues after failed calls
  - Empty/null parameter handling

- **Cross-Capability Compliance** (4 tests)
  - Consistent schema structure
  - JSON serializable responses
  - Capability negotiation preserves functionality
  - Server capabilities match registered tools

### 4. Protocol Version Handling (`test_mcp_protocol_version_handling.py`)
**9 tests** - Validates MCP protocol version 2024-11-05 compliance:

- Current protocol version support
- Version format validation (YYYY-MM-DD)
- FastMCP SDK version availability
- Version negotiation validation
- Backward compatibility
- Version mismatch handling
- Future version recognition
- Version downgrade scenarios
- Edge cases (invalid formats)

## Running the Tests

### Run All MCP Compliance Tests
```bash
pytest tests/unit/mcp/ -v
```

### Run with FastMCP Marker
```bash
pytest -m fastmcp -v
```

### Run Specific Test File
```bash
pytest tests/unit/mcp/test_mcp_tool_call_validation.py -v
pytest tests/unit/mcp/test_mcp_parameter_validation.py -v
pytest tests/unit/mcp/test_mcp_capability_negotiation.py -v
pytest tests/unit/mcp/test_mcp_protocol_version_handling.py -v
```

### Generate Compliance Report
```bash
pytest tests/unit/mcp/ \
  -v \
  --html=mcp-compliance-report.html \
  --self-contained-html \
  --junitxml=mcp-compliance-results.xml
```

## Test Infrastructure

### Official SDK Testing Pattern
All tests use the official `fastmcp.Client` SDK:

```python
from fastmcp import Client
from fastmcp.client.client import CallToolResult

@pytest.fixture
async def mcp_client():
    """Official SDK fixture using fastmcp.Client."""
    from workspace_qdrant_mcp.server import app

    async with Client(app) as client:
        await client.initialize()
        yield client
```

### Response Validation
```python
# Tool call returns official CallToolResult type
result = await client.call_tool("store", {"content": "test"})

assert isinstance(result, CallToolResult)
assert not result.isError  # Success indicator
assert result.content  # List of TextContent or other content types
```

### Mock Configuration
All external dependencies are mocked via `conftest.py`:
- **Qdrant client:** MockQdrantClient (in-memory operations)
- **Daemon client:** Mock gRPC client (no actual daemon required)
- **Git operations:** Mock project detection

## CI/CD Integration

### Dedicated Workflow
`.github/workflows/mcp-compliance.yml` runs on:
- Every push to `main`/`develop` affecting MCP code
- Pull requests touching MCP server or tests
- Daily at 2 AM UTC (regression detection)
- Manual workflow dispatch

### Quality Gates
- **Compliance Threshold:** 100% test pass rate required
- **Python Versions:** 3.10, 3.11, 3.12
- **Test Execution:** All 183 tests must pass
- **Reporting:** Automated compliance reports and PR comments

### Artifacts
- JUnit XML test results
- HTML compliance report
- Test execution logs
- Compliance summary markdown

## MCP Specification Compliance

### Protocol Version
**2024-11-05** (current MCP specification)

### Implemented Features
✅ Tool registration and discovery
✅ Tool call handling with CallToolResult
✅ Parameter validation (required/optional)
✅ Error response format
✅ Capability negotiation
✅ JSON-RPC message format
✅ Protocol version handling

### Not Implemented (by design)
❌ Resources (server is tools-only)
❌ Prompts (not needed for our use case)
❌ Sampling (not applicable)

## Development Guidelines

### Adding New MCP Tests
1. Use official `fastmcp.Client` SDK
2. Mock all external dependencies
3. Test protocol compliance, not business logic
4. Mark tests with `@pytest.mark.fastmcp`
5. Verify JSON serialization of responses
6. Test both success and error cases

### Test Naming Convention
- `test_<tool>_<aspect>_<scenario>`
- Example: `test_store_parameter_validation`

### Assertions to Include
```python
# Response type
assert isinstance(result, CallToolResult)

# Error status
assert not result.isError  # or: assert result.isError

# Response content
assert result.content
assert len(result.content) > 0

# JSON serialization
json.dumps(result.content, default=str)  # Must not raise
```

## Maintenance

### When to Update Tests
- MCP specification version changes
- New tools added to server
- Tool parameters modified
- FastMCP SDK major version upgrade

### Regression Prevention
- Daily CI runs catch breaking changes
- 100% pass rate requirement
- Cross-Python version testing
- Automated PR compliance reports

## References

- [MCP Specification](https://modelcontextprotocol.io/docs/spec)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Task #325 - MCP Protocol Compliance Tests](../../.taskmaster/tasks/task-325.md)

## Support

For issues or questions about MCP compliance tests:
1. Check test failure logs in CI artifacts
2. Review this README for testing patterns
3. Consult MCP specification for protocol details
4. Open an issue with compliance report attached
