"""
MCP server test fixtures and configuration.

Provides fixtures specific to MCP server testing including
FastMCP server instances, tool mocking, and protocol testing utilities.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path


@pytest.fixture
def mcp_server_config() -> Dict[str, Any]:
    """Provide test configuration for MCP server."""
    return {
        "server_name": "workspace-qdrant-mcp-test",
        "version": "0.2.1dev1",
        "qdrant_url": "http://localhost:6333",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "collections": ["test-project"],
        "global_collections": ["test-global"],
    }


@pytest.fixture
async def mock_qdrant_client():
    """Provide mock Qdrant client for MCP server testing."""

    class MockQdrantClient:
        """Mock Qdrant client for testing without actual Qdrant server."""

        def __init__(self):
            self.collections = {}
            self.documents = {}
            self.connected = True

        async def create_collection(
            self, collection_name: str, vectors_config: Any
        ) -> bool:
            """Mock collection creation."""
            self.collections[collection_name] = {
                "vectors_config": vectors_config,
                "points_count": 0,
            }
            return True

        async def upsert(
            self, collection_name: str, points: List[Any]
        ) -> Dict[str, Any]:
            """Mock document upsert."""
            if collection_name not in self.documents:
                self.documents[collection_name] = []
            self.documents[collection_name].extend(points)
            return {"status": "completed", "points_count": len(points)}

        async def search(
            self, collection_name: str, query_vector: List[float], limit: int = 10
        ) -> List[Dict[str, Any]]:
            """Mock search."""
            docs = self.documents.get(collection_name, [])
            return docs[:limit]

        async def delete(
            self, collection_name: str, points_selector: Any
        ) -> Dict[str, Any]:
            """Mock document deletion."""
            return {"status": "completed"}

        async def get_collections(self) -> List[str]:
            """Mock collection listing."""
            return list(self.collections.keys())

        async def collection_exists(self, collection_name: str) -> bool:
            """Check if collection exists."""
            return collection_name in self.collections

        async def close(self):
            """Mock close connection."""
            self.connected = False

    client = MockQdrantClient()
    yield client

    if client.connected:
        await client.close()


@pytest.fixture
async def mock_fastmcp_server():
    """Provide mock FastMCP server instance."""

    class MockFastMCPServer:
        """Mock FastMCP server for testing."""

        def __init__(self):
            self.tools = {}
            self.resources = {}
            self.prompts = {}
            self.running = False

        def tool(self, name: str):
            """Decorator to register tool."""

            def decorator(func):
                self.tools[name] = func
                return func

            return decorator

        def resource(self, uri: str):
            """Decorator to register resource."""

            def decorator(func):
                self.resources[uri] = func
                return func

            return decorator

        def prompt(self, name: str):
            """Decorator to register prompt."""

            def decorator(func):
                self.prompts[name] = func
                return func

            return decorator

        async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
            """Mock tool call."""
            if name not in self.tools:
                raise ValueError(f"Tool '{name}' not found")
            return await self.tools[name](**arguments)

        async def start(self):
            """Mock server start."""
            self.running = True

        async def stop(self):
            """Mock server stop."""
            self.running = False

    server = MockFastMCPServer()
    yield server

    if server.running:
        await server.stop()


@pytest.fixture
def sample_mcp_tools() -> List[Dict[str, Any]]:
    """Provide sample MCP tool definitions."""
    return [
        {
            "name": "store_document",
            "description": "Store a document in the project collection",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                    "collection": {"type": "string"},
                },
                "required": ["content"],
            },
        },
        {
            "name": "search_documents",
            "description": "Search for documents using hybrid search",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_collections",
            "description": "List all available collections",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


@pytest.fixture
def sample_mcp_requests() -> List[Dict[str, Any]]:
    """Provide sample MCP protocol requests."""
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "store_document",
                "arguments": {
                    "content": "Test document content",
                    "metadata": {"source": "test"},
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {"query": "test query", "limit": 5},
            },
        },
    ]


@pytest.fixture
def mcp_protocol_validator():
    """Provide MCP protocol validation utilities."""

    class MCPProtocolValidator:
        """Validates MCP protocol compliance."""

        @staticmethod
        def validate_request(request: Dict[str, Any]) -> bool:
            """Validate MCP request format."""
            required_fields = ["jsonrpc", "id", "method"]
            return all(field in request for field in required_fields)

        @staticmethod
        def validate_response(response: Dict[str, Any]) -> bool:
            """Validate MCP response format."""
            required_fields = ["jsonrpc", "id"]
            has_result_or_error = "result" in response or "error" in response
            return (
                all(field in response for field in required_fields)
                and has_result_or_error
            )

        @staticmethod
        def validate_tool_definition(tool: Dict[str, Any]) -> bool:
            """Validate tool definition format."""
            required_fields = ["name", "description", "inputSchema"]
            return all(field in tool for field in required_fields)

    return MCPProtocolValidator()


@pytest.fixture
async def test_workspace_dir(tmp_path: Path) -> Path:
    """Provide temporary workspace directory for testing."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()

    # Create basic project structure
    (workspace / "src").mkdir()
    (workspace / "docs").mkdir()
    (workspace / ".git").mkdir()

    # Create sample files
    (workspace / "README.md").write_text("# Test Project")
    (workspace / "src" / "main.py").write_text("def main():\n    pass")
    (workspace / "docs" / "api.md").write_text("# API Documentation")

    yield workspace


@pytest.fixture
def performance_metrics():
    """Provide performance metrics tracking."""

    class PerformanceMetrics:
        """Track MCP server performance metrics."""

        def __init__(self):
            self.request_times = []
            self.memory_usage = []
            self.tool_call_counts = {}

        def record_request_time(self, method: str, duration_ms: float):
            """Record request processing time."""
            self.request_times.append({"method": method, "duration_ms": duration_ms})

        def record_tool_call(self, tool_name: str):
            """Record tool call."""
            self.tool_call_counts[tool_name] = (
                self.tool_call_counts.get(tool_name, 0) + 1
            )

        def get_average_time(self, method: Optional[str] = None) -> float:
            """Get average request time."""
            times = self.request_times
            if method:
                times = [t for t in times if t["method"] == method]
            if not times:
                return 0.0
            return sum(t["duration_ms"] for t in times) / len(times)

        def get_p95_time(self, method: Optional[str] = None) -> float:
            """Get 95th percentile request time."""
            times = self.request_times
            if method:
                times = [t for t in times if t["method"] == method]
            if not times:
                return 0.0
            sorted_times = sorted(t["duration_ms"] for t in times)
            idx = int(len(sorted_times) * 0.95)
            return sorted_times[min(idx, len(sorted_times) - 1)]

    return PerformanceMetrics()