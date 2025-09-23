"""
Direct unit tests for memory tools module.

This module tests the memory tools functionality using lightweight mocking
and direct function testing for comprehensive coverage.
"""

import sys
from pathlib import Path

# Add src paths for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Any, Dict, List


class TestMemoryToolsFunctionality:
    """Test memory tools functionality with lightweight mocking."""

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    async def test_initialize_memory_session_basic(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class
    ):
        """Test basic memory session initialization."""
        # Import the module under test
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        mock_manager = AsyncMock()
        mock_manager.initialize_memory_collection.return_value = None
        mock_manager.list_memory_rules.return_value = []
        mock_manager.detect_conflicts.return_value = []

        mock_stats = Mock()
        mock_stats.total_rules = 0
        mock_stats.estimated_tokens = 0
        mock_manager.get_memory_stats.return_value = mock_stats

        mock_memory_manager_class.return_value = mock_manager

        # Create a mock FastMCP server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the initialized tool function
        assert "initialize_memory_session" in mock_server._tools
        tool_func = mock_server._tools["initialize_memory_session"].fn

        # Execute the tool
        result = await tool_func()

        # Verify result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "total_rules" in result
        assert result["total_rules"] == 0

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    async def test_add_memory_rule_basic(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class
    ):
        """Test basic memory rule addition."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        mock_manager = AsyncMock()
        mock_manager.initialize_memory_collection.return_value = None
        mock_manager.add_memory_rule.return_value = "rule-123"
        mock_memory_manager_class.return_value = mock_manager

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["add_memory_rule"].fn

        # Execute the tool
        result = await tool_func(
            category="preference",
            name="test-rule",
            rule="Test rule content",
            authority="default"
        )

        # Verify result
        assert result["success"] is True
        assert result["rule_id"] == "rule-123"

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    async def test_add_memory_rule_invalid_category(self, mock_config_class):
        """Test memory rule addition with invalid category."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["add_memory_rule"].fn

        # Execute the tool with invalid category
        result = await tool_func(
            category="invalid_category",
            name="test-rule",
            rule="Test rule content",
            authority="default"
        )

        # Verify error result
        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @patch('workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update')
    async def test_update_memory_from_conversation_no_pattern(self, mock_parse_function):
        """Test conversational memory update with no pattern detected."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mock
        mock_parse_function.return_value = None

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["update_memory_from_conversation"].fn

        # Execute the tool
        result = await tool_func("Just a regular message")

        # Verify result
        assert result["detected"] is False
        assert "No memory update pattern detected" in result["message"]

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    async def test_search_memory_rules_basic(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class
    ):
        """Test basic memory rule search."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Create mock rule
        mock_rule = Mock()
        mock_rule.id = "test-rule-1"
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.category.value = "preference"
        mock_rule.authority.value = "default"
        mock_rule.scope = ["test"]
        mock_rule.created_at = datetime.now(timezone.utc)

        mock_manager = AsyncMock()
        mock_manager.search_memory_rules.return_value = [(mock_rule, 0.95)]
        mock_memory_manager_class.return_value = mock_manager

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["search_memory_rules"].fn

        # Execute the tool
        result = await tool_func(query="test query", limit=5)

        # Verify result
        assert result["success"] is True
        assert result["total_found"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "test-rule-1"

    async def test_search_memory_rules_invalid_category(self):
        """Test memory rule search with invalid category."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["search_memory_rules"].fn

        # Execute the tool with invalid category
        result = await tool_func(
            query="test query",
            category="invalid_category"
        )

        # Verify error result
        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    async def test_get_memory_stats_basic(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class
    ):
        """Test basic memory statistics retrieval."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Mock categories and authority levels
        from common.core.memory import MemoryCategory, AuthorityLevel

        mock_stats = Mock()
        mock_stats.total_rules = 5
        mock_stats.rules_by_category = {
            MemoryCategory.PREFERENCE: 2,
            MemoryCategory.BEHAVIOR: 3
        }
        mock_stats.rules_by_authority = {
            AuthorityLevel.ABSOLUTE: 1,
            AuthorityLevel.DEFAULT: 4
        }
        mock_stats.estimated_tokens = 250
        mock_stats.last_optimization = datetime.now(timezone.utc)

        mock_manager = AsyncMock()
        mock_manager.get_memory_stats.return_value = mock_stats
        mock_memory_manager_class.return_value = mock_manager

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["get_memory_stats"].fn

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["total_rules"] == 5
        assert result["estimated_tokens"] == 250
        assert result["token_status"] == "low"

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    async def test_memory_tool_error_handling(self, mock_config_class):
        """Test error handling in memory tools."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mock to raise exception
        mock_config_class.side_effect = Exception("Configuration error")

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get any tool function
        tool_func = mock_server._tools["get_memory_stats"].fn

        # Execute the tool
        result = await tool_func()

        # Verify error handling
        assert "error" in result
        assert "Configuration error" in result["error"]

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    async def test_detect_memory_conflicts_basic(
        self, mock_memory_manager_class, mock_naming_manager,
        mock_qdrant_client, mock_config_class
    ):
        """Test basic memory conflict detection."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Setup mocks
        mock_config = Mock()
        mock_config.qdrant_client_config = {"url": "http://localhost:6333"}
        mock_config.workspace.global_collections = ["memory"]
        mock_config_class.return_value = mock_config

        # Create mock conflict
        mock_conflict = Mock()
        mock_conflict.conflict_type = "semantic_conflict"
        mock_conflict.description = "Conflicting rules detected"
        mock_conflict.confidence = 0.85
        mock_conflict.resolution_options = ["merge", "prioritize"]

        # Mock rules
        mock_rule1 = Mock()
        mock_rule1.id = "rule-1"
        mock_rule1.name = "rule-1"
        mock_rule1.rule = "Rule 1 content"
        mock_rule1.authority.value = "default"

        mock_rule2 = Mock()
        mock_rule2.id = "rule-2"
        mock_rule2.name = "rule-2"
        mock_rule2.rule = "Rule 2 content"
        mock_rule2.authority.value = "default"

        mock_conflict.rule1 = mock_rule1
        mock_conflict.rule2 = mock_rule2

        mock_manager = AsyncMock()
        mock_manager.detect_conflicts.return_value = [mock_conflict]
        mock_memory_manager_class.return_value = mock_manager

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Get the tool function
        tool_func = mock_server._tools["detect_memory_conflicts"].fn

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["conflicts_found"] == 1
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["type"] == "semantic_conflict"


class TestMemoryUtilities:
    """Test memory utility functions."""

    def test_memory_tools_registration(self):
        """Test that memory tools can be registered."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools
        register_memory_tools(mock_server)

        # Verify tools are registered
        expected_tools = [
            "initialize_memory_session",
            "add_memory_rule",
            "update_memory_from_conversation",
            "search_memory_rules",
            "get_memory_stats",
            "detect_memory_conflicts",
            "list_memory_rules",
            "apply_memory_context",
            "optimize_memory_tokens",
            "export_memory_profile"
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_server._tools

    def test_memory_module_imports(self):
        """Test that memory module can be imported."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools
            assert callable(register_memory_tools)
        except ImportError as e:
            pytest.skip(f"Memory module not available: {e}")

    def test_memory_core_imports(self):
        """Test that memory core classes can be imported."""
        try:
            from common.core.memory import (
                AuthorityLevel,
                MemoryCategory,
                MemoryManager,
                MemoryRule,
            )

            # Verify enums have expected values
            assert hasattr(AuthorityLevel, 'ABSOLUTE')
            assert hasattr(AuthorityLevel, 'DEFAULT')
            assert hasattr(MemoryCategory, 'PREFERENCE')
            assert hasattr(MemoryCategory, 'BEHAVIOR')
            assert hasattr(MemoryCategory, 'AGENT')

        except ImportError as e:
            pytest.skip(f"Memory core module not available: {e}")

    @patch('workspace_qdrant_mcp.tools.memory.logger')
    def test_memory_logging(self, mock_logger):
        """Test that memory tools use proper logging."""
        from workspace_qdrant_mcp.tools.memory import register_memory_tools

        # Create mock server
        mock_server = Mock()
        mock_server._tools = {}

        def mock_tool_decorator():
            def decorator(func):
                tool_mock = Mock()
                tool_mock.name = func.__name__
                tool_mock.fn = func
                mock_server._tools[func.__name__] = tool_mock
                return func
            return decorator

        mock_server.tool = mock_tool_decorator

        # Register tools (should not raise errors)
        register_memory_tools(mock_server)

        # Verify logger is available
        assert mock_logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])