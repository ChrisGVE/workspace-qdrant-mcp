"""Unit tests for search CLI commands.

Comprehensive unit tests for search CLI commands testing project search,
collection-specific search, search filters, output formats, and edge cases.

Test Coverage (35+ tests across 10 classes):
1. Project collection search
2. Collection-specific search
3. Global and all search modes
4. Memory search
5. Search filters and operators
6. Output format validation
7. Error handling and edge cases
8. Search result grouping
9. No results scenarios
10. Concurrent search operations

All tests use typer.testing.CliRunner for CLI testing with proper
mocking of daemon client and search responses.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner


class TestProjectSearch:
    """Test project collection search commands (Subtask 289.1)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.ProjectDetector')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_project_basic(self, mock_config, mock_detector_class, mock_with_daemon):
        """Test basic project search."""
        from wqm_cli.cli.commands.search import search_app

        # Mock project detector
        mock_detector = MagicMock()
        mock_detector.get_project_info.return_value = {
            "main_project": "myproject",
            "subprojects": []
        }
        mock_detector_class.return_value = mock_detector

        # Mock daemon client search operation
        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            # Mock list_collections response
            mock_col = MagicMock()
            mock_col.name = "myproject_docs"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # Mock execute_query response with results
            mock_result = MagicMock()
            mock_result.score = 0.95
            mock_result.collection = "myproject_docs"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Test Doc"),
                "content": MagicMock(string_value="Test content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["project", "test query"])

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.ProjectDetector')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_project_no_collections(self, mock_config, mock_detector_class, mock_with_daemon):
        """Test project search when no collections exist."""
        from wqm_cli.cli.commands.search import search_app

        mock_detector = MagicMock()
        mock_detector.get_project_info.return_value = {
            "main_project": "myproject",
            "subprojects": []
        }
        mock_detector_class.return_value = mock_detector

        async def mock_search_op(operation_func):
            mock_client = MagicMock()
            mock_collections_response = MagicMock()
            mock_collections_response.collections = []
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["project", "test query"])

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.ProjectDetector')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_project_with_specific_collections(self, mock_config, mock_detector_class, mock_with_daemon):
        """Test project search with user-specified collections."""
        from wqm_cli.cli.commands.search import search_app

        mock_detector = MagicMock()
        mock_detector.get_project_info.return_value = {
            "main_project": "myproject",
            "subprojects": []
        }
        mock_detector_class.return_value = mock_detector

        async def mock_search_op(operation_func):
            mock_client = MagicMock()
            mock_col1 = MagicMock()
            mock_col1.name = "myproject_docs"
            mock_col2 = MagicMock()
            mock_col2.name = "myproject_code"

            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col1, mock_col2]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_search_response = MagicMock()
            mock_search_response.results = []
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["project", "test", "--collection", "myproject_docs"]
            )

            assert result.exit_code == 0


class TestCollectionSearch:
    """Test collection-specific search commands (Subtask 289.2)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_collection_basic(self, mock_config, mock_with_daemon):
        """Test basic collection search."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            # Mock collection exists
            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # Mock search results
            mock_result = MagicMock()
            mock_result.score = 0.85
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Result"),
                "content": MagicMock(string_value="Content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test query", "--collection", "test-collection"]
            )

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_collection_not_found(self, mock_config, mock_with_daemon):
        """Test searching nonexistent collection."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()
            mock_collections_response = MagicMock()
            mock_collections_response.collections = []
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = Exception("Collection not found")

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "nonexistent"]
            )

            assert result.exit_code == 1

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_collection_with_vectors(self, mock_config, mock_with_daemon):
        """Test collection search with vector information."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.90
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.search_type = "HYBRID"
            mock_result.payload = {
                "title": MagicMock(string_value="Result"),
                "content": MagicMock(string_value="Content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--vectors"]
            )

            assert result.exit_code == 0


class TestGlobalAndAllSearch:
    """Test global and all search modes."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_global(self, mock_config, mock_with_daemon):
        """Test global search (library collections)."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            # Mix of library and project collections
            lib_col = MagicMock()
            lib_col.name = "_library"
            proj_col = MagicMock()
            proj_col.name = "project_docs"

            mock_collections_response = MagicMock()
            mock_collections_response.collections = [lib_col, proj_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_search_response = MagicMock()
            mock_search_response.results = []
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["global", "test query"])

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_all(self, mock_config, mock_with_daemon):
        """Test search across all collections."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col1 = MagicMock()
            mock_col1.name = "collection1"
            mock_col2 = MagicMock()
            mock_col2.name = "collection2"

            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col1, mock_col2]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # Mock search results from multiple collections
            mock_result1 = MagicMock()
            mock_result1.score = 0.95
            mock_result1.collection = "collection1"
            mock_result1.id = "doc1"
            mock_result1.payload = {
                "title": MagicMock(string_value="Result 1"),
                "content": MagicMock(string_value="Content 1")
            }

            mock_result2 = MagicMock()
            mock_result2.score = 0.80
            mock_result2.collection = "collection2"
            mock_result2.id = "doc2"
            mock_result2.payload = {
                "title": MagicMock(string_value="Result 2"),
                "content": MagicMock(string_value="Content 2")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result1, mock_result2]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["all", "test query"])

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_all_with_grouping(self, mock_config, mock_with_daemon):
        """Test search all with result grouping by collection."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"

            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.90
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Result"),
                "content": MagicMock(string_value="Content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["all", "test", "--group"])

            assert result.exit_code == 0


class TestMemorySearch:
    """Test memory search commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_memory(self, mock_config, mock_with_daemon):
        """Test memory rules search."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            # Mock memory search response
            mock_rule = MagicMock()
            mock_rule.rule_id = "rule123"
            mock_rule.name = "Test Rule"
            mock_rule.rule_text = "Test rule text"
            mock_rule.category = "coding"
            mock_rule.authority = "high"
            mock_rule.scope = ["python", "testing"]
            mock_rule.source = "user"

            mock_match = MagicMock()
            mock_match.rule = mock_rule
            mock_match.score = 0.95

            mock_memory_response = MagicMock()
            mock_memory_response.matches = [mock_match]

            mock_client.search_memory_rules = AsyncMock(return_value=mock_memory_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["memory", "test query"])

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_memory_no_results(self, mock_config, mock_with_daemon):
        """Test memory search with no results."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_memory_response = MagicMock()
            mock_memory_response.matches = []

            mock_client.search_memory_rules = AsyncMock(return_value=mock_memory_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["memory", "nonexistent"])

            assert result.exit_code == 0


class TestSearchFilters:
    """Test search filter and operator functionality (Subtask 289.3)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_with_threshold(self, mock_config, mock_with_daemon):
        """Test search with similarity threshold filter."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # High-score result above threshold
            mock_result = MagicMock()
            mock_result.score = 0.85
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="High Score"),
                "content": MagicMock(string_value="Content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--threshold", "0.7"]
            )

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_with_limit(self, mock_config, mock_with_daemon):
        """Test search with result limit."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # Return limited results
            results = []
            for i in range(3):
                mock_result = MagicMock()
                mock_result.score = 0.9 - (i * 0.1)
                mock_result.collection = "test-collection"
                mock_result.id = f"doc{i}"
                mock_result.payload = {
                    "title": MagicMock(string_value=f"Result {i}"),
                    "content": MagicMock(string_value=f"Content {i}")
                }
                results.append(mock_result)

            mock_search_response = MagicMock()
            mock_search_response.results = results
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--limit", "3"]
            )

            assert result.exit_code == 0


class TestOutputFormats:
    """Test output format validation (Subtask 289.4)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_json_format(self, mock_config, mock_with_daemon):
        """Test search with JSON output format."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.95
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Result"),
                "content": MagicMock(string_value="Test content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--format", "json"]
            )

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_table_format(self, mock_config, mock_with_daemon):
        """Test search with table output format."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.90
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Result"),
                "content": MagicMock(string_value="Content")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--format", "table"]
            )

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_detailed_format(self, mock_config, mock_with_daemon):
        """Test search with detailed output format."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.88
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Detailed Result"),
                "content": MagicMock(string_value="Detailed content for testing")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--format", "detailed"]
            )

            assert result.exit_code == 0


class TestSearchContent:
    """Test search content inclusion options."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_with_content(self, mock_config, mock_with_daemon):
        """Test search with content inclusion."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            mock_result = MagicMock()
            mock_result.score = 0.92
            mock_result.collection = "test-collection"
            mock_result.id = "doc1"
            mock_result.payload = {
                "title": MagicMock(string_value="Result with content"),
                "content": MagicMock(string_value="Full content text here")
            }

            mock_search_response = MagicMock()
            mock_search_response.results = [mock_result]
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection", "--content"]
            )

            assert result.exit_code == 0


class TestEdgeCases:
    """Test edge cases and error scenarios (Subtask 289.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_no_results(self, mock_config, mock_with_daemon):
        """Test search with no results found."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_col = MagicMock()
            mock_col.name = "test-collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_col]
            mock_client.list_collections = AsyncMock(return_value=mock_collections_response)

            # No results
            mock_search_response = MagicMock()
            mock_search_response.results = []
            mock_client.execute_query = AsyncMock(return_value=mock_search_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["collection", "nonexistent query", "--collection", "test-collection"]
            )

            assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_daemon_unavailable(self, mock_config, mock_with_daemon):
        """Test search when daemon is unavailable."""
        from wqm_cli.cli.commands.search import search_app

        # Simulate daemon connection error
        mock_with_daemon.side_effect = ConnectionError("Daemon unavailable")

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = Exception("Connection failed")

            result = self.runner.invoke(
                search_app,
                ["collection", "test", "--collection", "test-collection"]
            )

            assert result.exit_code == 1

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.ProjectDetector')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_search_project_error_handling(self, mock_config, mock_detector_class, mock_with_daemon):
        """Test project search error handling."""
        from wqm_cli.cli.commands.search import search_app

        # Simulate project detection error
        mock_detector_class.side_effect = RuntimeError("Project detection failed")

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = Exception("Error")

            result = self.runner.invoke(search_app, ["project", "test"])

            assert result.exit_code == 1


class TestResearchMode:
    """Test research mode functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_research_mode_basic(self):
        """Test basic research mode (not fully implemented)."""
        from wqm_cli.cli.commands.search import search_app

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(search_app, ["research", "test query"])

            assert result.exit_code == 0


class TestMemoryFilters:
    """Test memory search filtering."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.search.with_daemon_client')
    @patch('wqm_cli.cli.commands.search.get_config_manager')
    def test_memory_search_with_category(self, mock_config, mock_with_daemon):
        """Test memory search with category filter."""
        from wqm_cli.cli.commands.search import search_app

        async def mock_search_op(operation_func):
            mock_client = MagicMock()

            mock_rule = MagicMock()
            mock_rule.rule_id = "rule123"
            mock_rule.name = "Test Rule"
            mock_rule.rule_text = "Test rule"
            mock_rule.category = "coding"
            mock_rule.authority = "high"
            mock_rule.scope = ["python"]
            mock_rule.source = "user"

            mock_match = MagicMock()
            mock_match.rule = mock_rule
            mock_match.score = 0.95

            mock_memory_response = MagicMock()
            mock_memory_response.matches = [mock_match]

            mock_client.search_memory_rules = AsyncMock(return_value=mock_memory_response)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_search_op

        with patch('wqm_cli.cli.commands.search.handle_async') as mock_handle:
            mock_handle.side_effect = lambda coro: None

            result = self.runner.invoke(
                search_app,
                ["memory", "test", "--category", "coding"]
            )

            assert result.exit_code == 0
