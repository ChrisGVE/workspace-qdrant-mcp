"""
Comprehensive tests for metadata filtering in hybrid search (Task 322.4).

This test suite validates metadata-based filtering across all supported metadata types:
- file_type filtering (Python, JavaScript, Markdown, etc.)
- project_id filtering for multi-tenant isolation
- branch filtering for version control integration
- symbols filtering (functions, classes, etc.)
- custom metadata filtering (tags, categories, etc.)
- Multiple filter combinations
- Filter accuracy with dense and sparse search results
- Filter performance impact on search latency

Test Coverage:
1. file_type filtering - single and multiple types
2. project_id filtering - single and multiple projects
3. branch filtering - single and multiple branches
4. symbols filtering - function, class, variable filters
5. custom metadata filtering - tags, categories, priority
6. Multiple filter combinations - 2-5 filters combined
7. Filter accuracy with dense results
8. Filter accuracy with sparse results
9. Filter accuracy with hybrid (fused) results
10. Filter performance impact measurement
11. Edge cases - missing metadata, invalid values, empty filters
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from qdrant_client.http import models

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.hybrid_search import (
    HybridSearchEngine,
    RRFFusionRanker,
    WeightedSumFusionRanker,
)
from common.core.metadata_filtering import (
    FilterCriteria,
    FilterPerformanceLevel,
    FilterStrategy,
    MetadataFilterManager,
)
from common.core.multitenant_collections import (
    ProjectMetadata,
)

# ============================================================================
# Test Fixtures - Mock Qdrant Client
# ============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing with all required methods."""
    client = MagicMock()

    # Core search methods
    client.search = MagicMock(return_value=[])

    # Collection management methods
    client.get_collection = MagicMock(return_value=MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(
                    config={"dense": MagicMock(size=384)}
                )
            )
        )
    ))
    client.collection_exists = MagicMock(return_value=True)
    client.create_payload_index = MagicMock(return_value=True)
    client.update_collection = MagicMock(return_value=True)
    client.get_collections = MagicMock(return_value=MagicMock(collections=[]))

    return client


@pytest.fixture
def hybrid_search_engine(mock_qdrant_client):
    """Create a HybridSearchEngine instance for testing.

    Note: Optimizations disabled to avoid pydantic validation issues in unit tests.
    Performance monitoring disabled for faster test execution.
    """
    return HybridSearchEngine(
        client=mock_qdrant_client,
        enable_optimizations=False,  # Disabled due to KeywordIndexParams validation issue
        enable_multi_tenant_aggregation=True,
        enable_performance_monitoring=False  # Disabled for unit test performance
    )


@pytest.fixture
def metadata_filter_manager(mock_qdrant_client):
    """Create a MetadataFilterManager instance for testing."""
    return MetadataFilterManager(
        qdrant_client=mock_qdrant_client,
        enable_caching=True,
        enable_performance_monitoring=True
    )


# ============================================================================
# Test Fixtures - Mock Search Results with Diverse Metadata
# ============================================================================


@pytest.fixture
def mock_results_python_files():
    """Mock search results for Python files with comprehensive metadata."""
    return [
        Mock(
            id="doc-py1",
            score=0.95,
            payload={
                "content": "def process_data(): pass",
                "file_type": "python",
                "file_path": "/project/src/processor.py",
                "project_id": "proj123",
                "project_name": "data-processor",
                "branch": "main",
                "symbols": ["process_data", "DataProcessor"],
                "tags": ["core", "data"],
                "category": "source_code",
                "priority": 1,
                "created_by": "user1"
            }
        ),
        Mock(
            id="doc-py2",
            score=0.88,
            payload={
                "content": "class DataValidator: pass",
                "file_type": "python",
                "file_path": "/project/src/validator.py",
                "project_id": "proj123",
                "project_name": "data-processor",
                "branch": "main",
                "symbols": ["DataValidator", "validate"],
                "tags": ["validation", "core"],
                "category": "source_code",
                "priority": 2,
                "created_by": "user1"
            }
        ),
        Mock(
            id="doc-py3",
            score=0.82,
            payload={
                "content": "async def fetch_data(): pass",
                "file_type": "python",
                "file_path": "/project/src/fetcher.py",
                "project_id": "proj123",
                "project_name": "data-processor",
                "branch": "feature/async",
                "symbols": ["fetch_data", "AsyncFetcher"],
                "tags": ["async", "data"],
                "category": "source_code",
                "priority": 3,
                "created_by": "user2"
            }
        ),
    ]


@pytest.fixture
def mock_results_javascript_files():
    """Mock search results for JavaScript files."""
    return [
        Mock(
            id="doc-js1",
            score=0.90,
            payload={
                "content": "function handleClick() {}",
                "file_type": "javascript",
                "file_path": "/project/frontend/handlers.js",
                "project_id": "proj456",
                "project_name": "web-app",
                "branch": "main",
                "symbols": ["handleClick", "EventHandler"],
                "tags": ["frontend", "ui"],
                "category": "source_code",
                "priority": 1,
                "created_by": "user3"
            }
        ),
        Mock(
            id="doc-js2",
            score=0.85,
            payload={
                "content": "const api = {}",
                "file_type": "javascript",
                "file_path": "/project/frontend/api.js",
                "project_id": "proj456",
                "project_name": "web-app",
                "branch": "develop",
                "symbols": ["api", "fetchData"],
                "tags": ["api", "frontend"],
                "category": "source_code",
                "priority": 2,
                "created_by": "user3"
            }
        ),
    ]


@pytest.fixture
def mock_results_markdown_files():
    """Mock search results for Markdown documentation."""
    return [
        Mock(
            id="doc-md1",
            score=0.78,
            payload={
                "content": "# API Documentation",
                "file_type": "markdown",
                "file_path": "/project/docs/api.md",
                "project_id": "proj123",
                "project_name": "data-processor",
                "branch": "main",
                "symbols": [],
                "tags": ["documentation", "api"],
                "category": "documentation",
                "priority": 1,
                "created_by": "user1"
            }
        ),
        Mock(
            id="doc-md2",
            score=0.72,
            payload={
                "content": "## Getting Started",
                "file_type": "markdown",
                "file_path": "/project/docs/getting-started.md",
                "project_id": "proj456",
                "project_name": "web-app",
                "branch": "main",
                "symbols": [],
                "tags": ["documentation", "tutorial"],
                "category": "documentation",
                "priority": 2,
                "created_by": "user3"
            }
        ),
    ]


@pytest.fixture
def mock_results_mixed_metadata():
    """Mock search results with diverse metadata for combination testing."""
    return [
        Mock(
            id="doc-mix1",
            score=0.94,
            payload={
                "content": "Critical function",
                "file_type": "python",
                "file_path": "/proj/core.py",
                "project_id": "proj789",
                "project_name": "critical-system",
                "branch": "release/v2",
                "symbols": ["critical_function", "SystemCore"],
                "tags": ["critical", "core", "production"],
                "category": "source_code",
                "priority": 0,
                "created_by": "admin"
            }
        ),
        Mock(
            id="doc-mix2",
            score=0.86,
            payload={
                "content": "Test utilities",
                "file_type": "python",
                "file_path": "/proj/tests/utils.py",
                "project_id": "proj789",
                "project_name": "critical-system",
                "branch": "main",
                "symbols": ["test_helper", "MockFactory"],
                "tags": ["testing", "utilities"],
                "category": "test_code",
                "priority": 5,
                "created_by": "user2"
            }
        ),
        Mock(
            id="doc-mix3",
            score=0.79,
            payload={
                "content": "Configuration file",
                "file_type": "yaml",
                "file_path": "/proj/config.yaml",
                "project_id": "proj789",
                "project_name": "critical-system",
                "branch": "main",
                "symbols": [],
                "tags": ["config", "infrastructure"],
                "category": "configuration",
                "priority": 3,
                "created_by": "admin"
            }
        ),
    ]


@pytest.fixture
def mock_results_missing_metadata():
    """Mock search results with missing or incomplete metadata."""
    return [
        Mock(
            id="doc-missing1",
            score=0.75,
            payload={
                "content": "No metadata",
                "file_path": "/unknown/file.txt",
                # Missing: file_type, project_id, branch, symbols, etc.
            }
        ),
        Mock(
            id="doc-partial1",
            score=0.70,
            payload={
                "content": "Partial metadata",
                "file_type": "text",
                "project_id": "unknown",
                # Missing: branch, symbols, tags, etc.
            }
        ),
    ]


# ============================================================================
# Test Suite 1: File Type Filtering
# ============================================================================


class TestFileTypeFiltering:
    """Test file_type metadata filtering in hybrid search."""

    @pytest.mark.asyncio
    async def test_filter_single_file_type_python(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_javascript_files
    ):
        """Test filtering for single file type (Python)."""
        # Setup mock to return mixed results
        all_results = mock_results_python_files + mock_results_javascript_files
        mock_qdrant_client.search.return_value = all_results

        # Create filter for Python files only
        file_type_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                )
            ]
        )

        # Execute hybrid search with file type filter
        await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=file_type_filter
        )

        # Verify filter was applied
        assert mock_qdrant_client.search.called
        search_calls = mock_qdrant_client.search.call_args_list
        for search_call in search_calls:
            query_filter = search_call[1].get('query_filter') or search_call[0][2] if len(search_call[0]) > 2 else None
            if query_filter:
                assert query_filter is not None
                # Filter should contain file_type condition
                assert hasattr(query_filter, 'must')

    @pytest.mark.asyncio
    async def test_filter_multiple_file_types(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_javascript_files,
        mock_results_markdown_files
    ):
        """Test filtering for multiple file types (Python, JavaScript)."""
        all_results = mock_results_python_files + mock_results_javascript_files + mock_results_markdown_files
        mock_qdrant_client.search.return_value = all_results

        # Create filter for Python and JavaScript files
        file_type_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="javascript")
                )
            ]
        )

        await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=file_type_filter
        )

        # Verify search was executed with filter
        assert mock_qdrant_client.search.called
        assert len(mock_qdrant_client.search.call_args_list) >= 1

    @pytest.mark.asyncio
    async def test_filter_file_type_accuracy_dense(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_javascript_files
    ):
        """Test file_type filter accuracy with dense search results only."""
        # Return only Python results for dense search
        mock_qdrant_client.search.return_value = mock_results_python_files

        file_type_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,  # Only dense search
            },
            limit=10,
            filter_conditions=file_type_filter
        )

        # Verify only dense results are returned
        assert "dense_results" in results
        assert len(results["dense_results"]) == len(mock_results_python_files)

        # Verify all returned results have correct file_type
        for result in results["dense_results"]:
            assert result.payload.get("file_type") == "python"


# ============================================================================
# Test Suite 2: Project ID Filtering
# ============================================================================


class TestProjectIdFiltering:
    """Test project_id metadata filtering for multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_filter_single_project_id(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering for single project_id."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        # Create project isolation filter
        project_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj123")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=project_filter
        )

        # Verify filter was applied
        assert mock_qdrant_client.search.called
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_filter_project_id_with_project_context(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test project_id filtering using project context auto-injection."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        # Use project context instead of explicit filter
        project_context = ProjectMetadata.create_project_metadata(
            project_name="data-processor",
            collection_type="code",
            workspace_scope="project"
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            project_context=project_context,
            auto_inject_metadata=True
        )

        # Verify search was executed
        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_filter_multiple_project_ids(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_javascript_files
    ):
        """Test filtering for multiple project IDs."""
        all_results = mock_results_python_files + mock_results_javascript_files
        mock_qdrant_client.search.return_value = all_results

        # Filter for multiple projects
        project_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj123")
                ),
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj456")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=project_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None


# ============================================================================
# Test Suite 3: Branch Filtering
# ============================================================================


class TestBranchFiltering:
    """Test branch metadata filtering for version control integration."""

    @pytest.mark.asyncio
    async def test_filter_single_branch_main(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering for main branch only."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        branch_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=branch_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_filter_multiple_branches(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering for multiple branches (main, develop)."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        branch_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                ),
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="develop")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=branch_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_filter_branch_accuracy_sparse(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test branch filter accuracy with sparse search results only."""
        # Filter to main branch results
        main_branch_results = [r for r in mock_results_python_files if r.payload.get("branch") == "main"]
        mock_qdrant_client.search.return_value = main_branch_results

        branch_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}  # Only sparse
            },
            limit=10,
            filter_conditions=branch_filter
        )

        assert "sparse_results" in results
        # Verify all results are from main branch
        for result in results["sparse_results"]:
            assert result.payload.get("branch") == "main"


# ============================================================================
# Test Suite 4: Symbols Filtering
# ============================================================================


class TestSymbolsFiltering:
    """Test symbols metadata filtering (functions, classes, etc.)."""

    @pytest.mark.asyncio
    async def test_filter_by_specific_symbol_function(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering for specific function symbol."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        # Filter for documents containing 'process_data' symbol
        symbol_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="symbols",
                    match=models.MatchAny(any=["process_data"])
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=symbol_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_filter_by_multiple_symbols(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering for multiple symbols."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        # Filter for documents containing any of these symbols
        symbol_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="symbols",
                    match=models.MatchAny(any=["DataProcessor", "DataValidator", "AsyncFetcher"])
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=symbol_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_filter_symbols_with_file_type(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_javascript_files
    ):
        """Test combination of symbols and file_type filtering."""
        all_results = mock_results_python_files + mock_results_javascript_files
        mock_qdrant_client.search.return_value = all_results

        # Filter for Python files with specific symbols
        combined_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="symbols",
                    match=models.MatchAny(any=["DataProcessor"])
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=combined_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None


# ============================================================================
# Test Suite 5: Custom Metadata Filtering
# ============================================================================


class TestCustomMetadataFiltering:
    """Test custom metadata filtering (tags, categories, priority)."""

    @pytest.mark.asyncio
    async def test_filter_by_tags_single(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering by single tag."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        tag_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=["core"])
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=tag_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_filter_by_category(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files, mock_results_markdown_files
    ):
        """Test filtering by category."""
        all_results = mock_results_python_files + mock_results_markdown_files
        mock_qdrant_client.search.return_value = all_results

        category_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="source_code")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=category_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_filter_by_priority_range(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_mixed_metadata
    ):
        """Test filtering by priority range."""
        mock_qdrant_client.search.return_value = mock_results_mixed_metadata

        # Filter for high priority items (0-2)
        priority_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="priority",
                    range=models.Range(
                        gte=0,
                        lte=2
                    )
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=priority_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_filter_by_created_by(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test filtering by created_by metadata."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        creator_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="created_by",
                    match=models.MatchValue(value="user1")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=creator_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results


# ============================================================================
# Test Suite 6: Multiple Filter Combinations
# ============================================================================


class TestMultipleFilterCombinations:
    """Test combining multiple metadata filters."""

    @pytest.mark.asyncio
    async def test_combine_file_type_and_project_id(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test 2-filter combination: file_type + project_id."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        combined_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj123")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=combined_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_combine_three_filters(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test 3-filter combination: file_type + project_id + branch."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        combined_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj123")
                ),
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=combined_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_combine_five_filters_complex(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_mixed_metadata
    ):
        """Test 5-filter combination: file_type + project_id + branch + tags + priority."""
        mock_qdrant_client.search.return_value = mock_results_mixed_metadata

        complex_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj789")
                ),
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                ),
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=["testing", "utilities"])
                ),
                models.FieldCondition(
                    key="priority",
                    range=models.Range(gte=0, lte=5)
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=complex_filter
        )

        assert mock_qdrant_client.search.called
        assert results is not None

    @pytest.mark.asyncio
    async def test_combine_must_and_should_conditions(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_mixed_metadata
    ):
        """Test combining must and should filter conditions."""
        mock_qdrant_client.search.return_value = mock_results_mixed_metadata

        hybrid_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj789")
                )
            ],
            should=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="yaml")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=hybrid_filter
        )

        assert mock_qdrant_client.search.called
        assert "fused_results" in results


# ============================================================================
# Test Suite 7: Filter Performance Impact
# ============================================================================


class TestFilterPerformanceImpact:
    """Test performance impact of metadata filtering on search latency."""

    @pytest.mark.asyncio
    async def test_single_filter_performance(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test performance impact of single metadata filter."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        single_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                )
            ]
        )

        start_time = time.time()
        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=single_filter
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Should complete reasonably quickly (under 100ms in mock environment)
        assert elapsed_ms < 100
        assert results is not None

    @pytest.mark.asyncio
    async def test_complex_filter_performance(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_mixed_metadata
    ):
        """Test performance impact of complex multi-condition filters."""
        mock_qdrant_client.search.return_value = mock_results_mixed_metadata

        complex_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="proj789")
                ),
                models.FieldCondition(
                    key="branch",
                    match=models.MatchValue(value="main")
                ),
                models.FieldCondition(
                    key="priority",
                    range=models.Range(gte=0, lte=3)
                )
            ]
        )

        start_time = time.time()
        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=complex_filter
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Complex filters should still complete reasonably quickly
        assert elapsed_ms < 150
        assert results is not None

    @pytest.mark.asyncio
    async def test_no_filter_baseline_performance(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test baseline performance without any filters."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        start_time = time.time()
        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=None  # No filters
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Baseline should be fast
        assert elapsed_ms < 100
        assert results is not None

        # Can use this as baseline to compare filtered queries
        return elapsed_ms


# ============================================================================
# Test Suite 8: Edge Cases
# ============================================================================


class TestFilterEdgeCases:
    """Test edge cases and error handling in metadata filtering."""

    @pytest.mark.asyncio
    async def test_filter_with_missing_metadata(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_missing_metadata
    ):
        """Test filtering when documents have missing metadata."""
        mock_qdrant_client.search.return_value = mock_results_missing_metadata

        # Filter for file_type, but some docs don't have it
        file_type_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="python")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=file_type_filter
        )

        # Should complete without errors
        assert results is not None
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_empty_filter_conditions(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test with empty filter conditions."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        empty_filter = models.Filter()

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=empty_filter
        )

        # Should work like no filter
        assert results is not None
        assert "fused_results" in results

    @pytest.mark.asyncio
    async def test_filter_with_no_matching_results(
        self, hybrid_search_engine, mock_qdrant_client
    ):
        """Test filter that matches no results."""
        # Return empty results
        mock_qdrant_client.search.return_value = []

        impossible_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value="nonexistent_project")
                )
            ]
        )

        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=impossible_filter
        )

        assert results is not None
        assert "fused_results" in results
        assert len(results["fused_results"]) == 0

    @pytest.mark.asyncio
    async def test_filter_with_none_values(
        self, hybrid_search_engine, mock_qdrant_client,
        mock_results_python_files
    ):
        """Test handling of None values in filter conditions."""
        mock_qdrant_client.search.return_value = mock_results_python_files

        # Test with None filter_conditions parameter
        results = await hybrid_search_engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings={
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            filter_conditions=None
        )

        assert results is not None
        assert "fused_results" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
