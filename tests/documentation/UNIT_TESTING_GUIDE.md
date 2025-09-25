# Unit Testing Guide: Python Components

## Overview

This guide covers unit testing best practices for Python components in the workspace-qdrant-mcp project. The project has achieved 100% unit test coverage through systematic implementation of comprehensive test suites.

## Testing Framework and Structure

### Core Technologies

- **pytest**: Primary testing framework
- **pytest-asyncio**: Async/await testing support
- **unittest.mock**: Comprehensive mocking capabilities
- **coverage.py**: Coverage measurement and reporting

### Test Directory Structure

```
tests/unit/
├── __init__.py
├── test_core_client.py          # QdrantWorkspaceClient tests
├── test_core_embeddings.py      # Embedding service tests
├── test_core_hybrid_search.py   # Hybrid search engine tests
├── test_core_memory.py          # Memory management tests
├── test_tools_memory.py         # MCP memory tools tests
├── test_tools_state_management.py # State management tests
├── test_cli_main.py             # CLI interface tests
├── test_utils_project_detection.py # Project detection tests
└── ...                          # Additional component tests
```

## Test Writing Best Practices

### Test Naming Convention

Follow the pattern: `test_<component>_<action>_<condition>_<expected_result>`

```python
def test_client_search_with_empty_query_returns_empty_results():
    """Test that empty query returns empty search results."""
    pass

def test_embeddings_initialize_with_invalid_model_raises_error():
    """Test that invalid model raises appropriate error during initialization."""
    pass

def test_hybrid_search_fusion_with_equal_weights_combines_scores():
    """Test that RRF fusion with equal weights properly combines scores."""
    pass
```

### Test Structure: Arrange-Act-Assert

```python
@pytest.mark.asyncio
async def test_client_create_collection_success():
    """Test successful collection creation with valid parameters."""
    # Arrange
    mock_qdrant = AsyncMock()
    mock_config = MockConfig()
    client = QdrantWorkspaceClient(mock_config, mock_qdrant)
    collection_name = "test-collection"

    # Act
    result = await client.create_collection(collection_name)

    # Assert
    assert result is True
    mock_qdrant.create_collection.assert_called_once()
    call_args = mock_qdrant.create_collection.call_args
    assert call_args[1]["collection_name"] == collection_name
```

## Mocking Strategies

### Lightweight Mocking Pattern

The project uses lightweight mocking to avoid external dependencies while maintaining realistic test scenarios.

```python
class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.environment = "development"
        self.qdrant = MagicMock()
        self.qdrant.url = "http://localhost:6333"
        self.embedding = MagicMock()
        self.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.workspace = MagicMock()
        self.workspace.github_user = "testuser"

class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def encode_documents(self, texts):
        return [[0.1] * 384 for _ in texts]  # Mock embeddings
```

### Async Testing Patterns

```python
@pytest.mark.asyncio
async def test_async_operation_with_mock():
    """Test asynchronous operations with proper mocking."""
    # Setup async mocks
    mock_service = AsyncMock()
    mock_service.process.return_value = {"status": "success"}

    # Test async operation
    result = await mock_service.process("test_data")

    # Verify async call
    mock_service.process.assert_awaited_once_with("test_data")
    assert result["status"] == "success"
```

## Component-Specific Testing

### Core Client Testing

```python
class TestQdrantWorkspaceClient:
    """Comprehensive tests for QdrantWorkspaceClient."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        mock_config = MockConfig()
        mock_qdrant = AsyncMock()
        return QdrantWorkspaceClient(mock_config, mock_qdrant)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_client):
        """Test search operation with metadata filters."""
        # Setup test data
        query = "test query"
        filters = {"project": "test-project"}

        # Mock search results
        mock_points = [
            Mock(id=1, score=0.9, payload={"content": "result 1"}),
            Mock(id=2, score=0.8, payload={"content": "result 2"})
        ]
        mock_client._qdrant_client.search.return_value = mock_points

        # Execute search
        results = await mock_client.search("collection", query, filters)

        # Verify results
        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[0]["content"] == "result 1"
```

### Hybrid Search Testing

```python
class TestHybridSearchEngine:
    """Test hybrid search functionality with fusion algorithms."""

    def test_rrf_fusion_algorithm(self):
        """Test Reciprocal Rank Fusion algorithm implementation."""
        # Arrange
        dense_results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
        sparse_results = [{"id": 2, "score": 0.95}, {"id": 3, "score": 0.85}]

        # Act
        fused_results = rrf_fusion(dense_results, sparse_results, k=60)

        # Assert
        assert len(fused_results) == 3
        assert fused_results[0]["id"] == 2  # Appears in both, should rank highest

    def test_fusion_with_empty_results(self):
        """Test fusion behavior with empty result sets."""
        empty_results = []
        normal_results = [{"id": 1, "score": 0.9}]

        fused = rrf_fusion(empty_results, normal_results)

        assert len(fused) == 1
        assert fused[0]["id"] == 1
```

## Edge Case Testing

### Input Validation Testing

```python
class TestInputValidation:
    """Test input validation and boundary conditions."""

    @pytest.mark.parametrize("invalid_input", [
        None,
        "",
        "   ",
        {"invalid": "type"},
        ["list", "input"]
    ])
    def test_search_with_invalid_query(self, invalid_input):
        """Test search with various invalid query inputs."""
        with pytest.raises(ValueError, match="Invalid query"):
            validate_search_query(invalid_input)

    def test_large_query_handling(self):
        """Test handling of extremely large queries."""
        large_query = "test " * 10000  # 50KB query

        with pytest.raises(ValueError, match="Query too large"):
            validate_search_query(large_query)
```

### Error Condition Testing

```python
class TestErrorHandling:
    """Test error conditions and exception handling."""

    @pytest.mark.asyncio
    async def test_qdrant_connection_failure(self):
        """Test handling of Qdrant connection failures."""
        mock_qdrant = AsyncMock()
        mock_qdrant.search.side_effect = ConnectionError("Connection failed")

        client = QdrantWorkspaceClient(MockConfig(), mock_qdrant)

        with pytest.raises(ConnectionError):
            await client.search("collection", "query")

    @pytest.mark.asyncio
    async def test_embedding_service_failure(self):
        """Test handling of embedding service failures."""
        mock_embeddings = AsyncMock()
        mock_embeddings.encode_documents.side_effect = RuntimeError("Model error")

        search_engine = HybridSearchEngine(mock_embeddings)

        with pytest.raises(RuntimeError, match="Model error"):
            await search_engine.search("test query")
```

## Memory and Resource Testing

### Memory Leak Testing

```python
import gc
import sys

def test_client_memory_cleanup():
    """Test that client properly cleans up resources."""
    initial_objects = len(gc.get_objects())

    # Create and destroy multiple clients
    for i in range(100):
        client = QdrantWorkspaceClient(MockConfig(), AsyncMock())
        del client

    gc.collect()
    final_objects = len(gc.get_objects())

    # Allow some variance for test framework objects
    assert final_objects - initial_objects < 50
```

### Resource Cleanup Testing

```python
@pytest.mark.asyncio
async def test_connection_cleanup():
    """Test that connections are properly closed."""
    mock_qdrant = AsyncMock()
    client = QdrantWorkspaceClient(MockConfig(), mock_qdrant)

    await client.close()

    mock_qdrant.close.assert_called_once()
```

## Performance Unit Testing

### Response Time Testing

```python
import time
import pytest

@pytest.mark.benchmark
def test_search_performance():
    """Test that search operations meet performance requirements."""
    client = create_test_client()

    start_time = time.time()
    results = client.search_sync("test query")
    duration = time.time() - start_time

    assert duration < 0.1  # Should complete within 100ms
    assert len(results) > 0
```

### Concurrency Testing

```python
import asyncio

@pytest.mark.asyncio
async def test_concurrent_searches():
    """Test handling of concurrent search operations."""
    client = create_test_client()

    # Execute multiple concurrent searches
    tasks = [
        client.search("collection", f"query {i}")
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    # Verify all searches completed successfully
    assert len(results) == 10
    for result in results:
        assert isinstance(result, list)
```

## Test Configuration and Fixtures

### Global Configuration

```python
# conftest.py
import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    return MockConfig()

@pytest.fixture
async def test_client(mock_config):
    """Provide test client instance."""
    mock_qdrant = AsyncMock()
    client = QdrantWorkspaceClient(mock_config, mock_qdrant)
    yield client
    await client.close()
```

### Component-Specific Fixtures

```python
@pytest.fixture
def search_engine():
    """Provide configured search engine for testing."""
    mock_embeddings = MockEmbeddingService()
    engine = HybridSearchEngine(mock_embeddings)
    return engine

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {"id": 1, "content": "Python testing guide", "metadata": {"type": "doc"}},
        {"id": 2, "content": "Unit testing best practices", "metadata": {"type": "guide"}},
        {"id": 3, "content": "Async testing patterns", "metadata": {"type": "example"}}
    ]
```

## Coverage and Quality Metrics

### Running Tests with Coverage

```bash
# Run all unit tests with coverage
uv run pytest tests/unit/ --cov=src --cov-report=html --cov-report=term

# Run specific component tests
uv run pytest tests/unit/test_core_client.py -v

# Run tests with performance benchmarks
uv run pytest tests/unit/ --benchmark-only
```

### Coverage Requirements

- **Line Coverage**: 90% minimum for all modules
- **Branch Coverage**: 85% minimum for critical components
- **Function Coverage**: 95% for public APIs
- **Missing Coverage**: Document any uncovered lines with justification

### Quality Checks

```python
def test_function_has_docstring():
    """Ensure all public functions have docstrings."""
    from src.common.core.client import QdrantWorkspaceClient

    for name in dir(QdrantWorkspaceClient):
        if not name.startswith('_'):
            method = getattr(QdrantWorkspaceClient, name)
            if callable(method):
                assert method.__doc__ is not None, f"Method {name} missing docstring"
```

## Common Testing Patterns

### Parameterized Testing

```python
@pytest.mark.parametrize("query,expected_count", [
    ("simple query", 5),
    ("complex query with filters", 3),
    ("empty query", 0),
    ("special characters !@#$%", 2)
])
def test_search_variations(query, expected_count):
    """Test search with various query types."""
    results = search_function(query)
    assert len(results) == expected_count
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_query_processing_with_arbitrary_text(query_text):
    """Test query processing with arbitrary text input."""
    # Should not raise exceptions for any valid string input
    result = process_query(query_text)
    assert isinstance(result, str)
    assert len(result) > 0
```

## Debugging and Troubleshooting

### Test Debugging

```python
import logging

# Enable debug logging in tests
logging.basicConfig(level=logging.DEBUG)

def test_with_debug_output():
    """Example test with debug output."""
    logger = logging.getLogger(__name__)
    logger.debug("Starting test execution")

    result = function_under_test()
    logger.debug(f"Function returned: {result}")

    assert result == expected_value
```

### Test Isolation

```python
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset any global variables or singletons
    GlobalCache.clear()
    ConfigManager.reset()
    yield
    # Cleanup after test
    GlobalCache.clear()
```

This unit testing guide provides comprehensive coverage of Python testing best practices for the workspace-qdrant-mcp project. Follow these patterns to maintain the 100% coverage achievement and ensure robust, reliable software development.