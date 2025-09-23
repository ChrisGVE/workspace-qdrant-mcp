"""
Comprehensive unit tests for workspace_qdrant_mcp.tools.memory.

This module tests the MCP memory tools functionality including:
- Memory session initialization
- Adding and updating memory rules
- Searching and listing memory rules
- Memory statistics and optimization
- Conflict detection and resolution
- Conversational memory updates
- Memory profile export
"""

import sys
from pathlib import Path

# Add src/python to path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

try:
    from workspace_qdrant_mcp.tools.memory import register_memory_tools
    from common.core.memory import (
        AuthorityLevel,
        MemoryCategory,
        MemoryManager,
        MemoryRule,
    )
except ImportError as e:
    pytest.skip(f"Unable to import required modules: {e}", allow_module_level=True)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.qdrant_client_config = {"url": "http://localhost:6333"}
    config.workspace.global_collections = ["memory"]
    return config


@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager for testing."""
    manager = AsyncMock(spec=MemoryManager)
    manager.memory_collection_name = "test-memory"
    return manager


@pytest.fixture
def mock_memory_rule():
    """Create a mock memory rule for testing."""
    return MemoryRule(
        id="test-rule-1",
        category=MemoryCategory.PREFERENCE,
        name="python-tool",
        rule="Use uv for Python package management",
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "development"],
        source="test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"test": True}
    )


@pytest.fixture
def mock_memory_stats():
    """Mock memory statistics."""
    stats = Mock()
    stats.total_rules = 10
    stats.rules_by_category = {
        MemoryCategory.PREFERENCE: 3,
        MemoryCategory.BEHAVIOR: 5,
        MemoryCategory.AGENT: 2
    }
    stats.rules_by_authority = {
        AuthorityLevel.ABSOLUTE: 2,
        AuthorityLevel.DEFAULT: 8
    }
    stats.estimated_tokens = 500
    stats.last_optimization = datetime.now(timezone.utc)
    return stats


@pytest.fixture
def mock_memory_conflict():
    """Mock memory conflict for testing."""
    rule1 = MemoryRule(
        id="rule-1",
        category=MemoryCategory.BEHAVIOR,
        name="commit-style",
        rule="Use conventional commits",
        authority=AuthorityLevel.DEFAULT,
        scope=["git"]
    )
    rule2 = MemoryRule(
        id="rule-2",
        category=MemoryCategory.BEHAVIOR,
        name="commit-format",
        rule="Use semantic commit messages",
        authority=AuthorityLevel.DEFAULT,
        scope=["git"]
    )

    conflict = Mock()
    conflict.conflict_type = "semantic_conflict"
    conflict.description = "Conflicting commit message formats"
    conflict.rule1 = rule1
    conflict.rule2 = rule2
    conflict.confidence = 0.85
    conflict.resolution_options = ["merge", "prioritize_rule1", "prioritize_rule2"]
    return conflict


@pytest.fixture
def mcp_server():
    """FastMCP server for testing."""
    if FastMCP is None:
        pytest.skip("FastMCP not available")
    server = FastMCP("test-memory-server")
    register_memory_tools(server)
    return server


class TestMemoryToolsInitialization:
    """Test memory session initialization functionality."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_initialize_memory_session_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule, mock_memory_stats
    ):
        """Test successful memory session initialization."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]
        mock_memory_manager.detect_conflicts.return_value = []
        mock_memory_manager.get_memory_stats.return_value = mock_memory_stats

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "initialize_memory_session":
                tool_func = tool.fn
                break

        assert tool_func is not None, "initialize_memory_session tool not found"

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["status"] == "ready"
        assert result["total_rules"] == 1
        assert result["absolute_rules"] == 0
        assert result["default_rules"] == 1
        assert result["conflicts_detected"] == 0
        assert result["estimated_tokens"] == 500
        assert "rules_for_injection" in result
        assert "absolute" in result["rules_for_injection"]
        assert "default" in result["rules_for_injection"]

        # Verify manager calls
        mock_memory_manager.initialize_memory_collection.assert_called_once()
        mock_memory_manager.list_memory_rules.assert_called_once()
        mock_memory_manager.detect_conflicts.assert_called_once()
        mock_memory_manager.get_memory_stats.assert_called_once()

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_initialize_memory_session_with_conflicts(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule,
        mock_memory_stats, mock_memory_conflict
    ):
        """Test memory session initialization with conflicts detected."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]
        mock_memory_manager.detect_conflicts.return_value = [mock_memory_conflict]
        mock_memory_manager.get_memory_stats.return_value = mock_memory_stats

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "initialize_memory_session":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["status"] == "conflicts_require_resolution"
        assert result["conflicts_detected"] == 1
        assert "conflicts" in result
        assert len(result["conflicts"]) == 1

        conflict = result["conflicts"][0]
        assert conflict["type"] == "semantic_conflict"
        assert conflict["confidence"] == 0.85
        assert "resolution_options" in conflict

    @patch('common.core.config.Config')
    async def test_initialize_memory_session_error(self, mock_config_class, mcp_server):
        """Test memory session initialization error handling."""
        # Setup mock to raise exception
        mock_config_class.side_effect = Exception("Configuration error")

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "initialize_memory_session":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify error result
        assert result["status"] == "error"
        assert "error" in result
        assert result["total_rules"] == 0


class TestMemoryRuleManagement:
    """Test memory rule creation and management."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_add_memory_rule_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager
    ):
        """Test successful memory rule addition."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.add_memory_rule.return_value = "rule-123"

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "add_memory_rule":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(
            category="preference",
            name="test-rule",
            rule="Always use TypeScript for new projects",
            authority="default",
            scope=["typescript", "frontend"],
            source="test_user"
        )

        # Verify result
        assert result["success"] is True
        assert result["rule_id"] == "rule-123"
        assert "message" in result

        # Verify manager calls
        mock_memory_manager.initialize_memory_collection.assert_called_once()
        mock_memory_manager.add_memory_rule.assert_called_once()

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_add_memory_rule_invalid_category(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager
    ):
        """Test memory rule addition with invalid category."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "add_memory_rule":
                tool_func = tool.fn
                break

        # Execute the tool with invalid category
        result = await tool_func(
            category="invalid_category",
            name="test-rule",
            rule="Test rule",
            authority="default"
        )

        # Verify error result
        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @patch('common.core.memory.parse_conversational_memory_update')
    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_update_memory_from_conversation_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mock_parse_function,
        mcp_server, mock_config, mock_memory_manager
    ):
        """Test successful conversational memory update."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_parse_function.return_value = {
            "category": MemoryCategory.PREFERENCE,
            "rule": "Call me Chris",
            "authority": AuthorityLevel.DEFAULT,
            "source": "conversational"
        }

        mock_memory_manager.initialize_memory_collection.return_value = None
        mock_memory_manager.add_memory_rule.return_value = "conv-rule-123"

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "update_memory_from_conversation":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func("Note: call me Chris")

        # Verify result
        assert result["detected"] is True
        assert result["rule_added"] is True
        assert result["rule_id"] == "conv-rule-123"
        assert result["category"] == "preference"
        assert result["rule"] == "Call me Chris"

    @patch('common.core.memory.parse_conversational_memory_update')
    async def test_update_memory_from_conversation_no_pattern(
        self, mock_parse_function, mcp_server
    ):
        """Test conversational memory update with no pattern detected."""
        # Setup mock
        mock_parse_function.return_value = None

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "update_memory_from_conversation":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func("Just a regular message")

        # Verify result
        assert result["detected"] is False
        assert "No memory update pattern detected" in result["message"]


class TestMemorySearch:
    """Test memory search and retrieval functionality."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_search_memory_rules_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule
    ):
        """Test successful memory rule search."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        search_results = [(mock_memory_rule, 0.95)]
        mock_memory_manager.search_memory_rules.return_value = search_results

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "search_memory_rules":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(
            query="python development",
            category="preference",
            authority="default",
            limit=5
        )

        # Verify result
        assert result["success"] is True
        assert result["query"] == "python development"
        assert result["total_found"] == 1
        assert len(result["results"]) == 1

        rule_result = result["results"][0]
        assert rule_result["id"] == "test-rule-1"
        assert rule_result["name"] == "python-tool"
        assert rule_result["relevance_score"] == 0.95

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_list_memory_rules_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule
    ):
        """Test successful memory rule listing."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "list_memory_rules":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(
            category="preference",
            authority="default",
            limit=10
        )

        # Verify result
        assert result["success"] is True
        assert result["total_returned"] == 1
        assert len(result["rules"]) == 1

        rule_result = result["rules"][0]
        assert rule_result["id"] == "test-rule-1"
        assert rule_result["category"] == "preference"
        assert rule_result["authority"] == "default"


class TestMemoryStatistics:
    """Test memory statistics and optimization functionality."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_get_memory_stats_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_stats
    ):
        """Test successful memory statistics retrieval."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.get_memory_stats.return_value = mock_memory_stats

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "get_memory_stats":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["total_rules"] == 10
        assert result["estimated_tokens"] == 500
        assert result["token_status"] == "low"
        assert "rules_by_category" in result
        assert "rules_by_authority" in result

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_optimize_memory_tokens_no_optimization_needed(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_stats
    ):
        """Test memory optimization when no optimization is needed."""
        # Setup mocks - tokens under limit
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_stats.estimated_tokens = 500
        mock_memory_manager.get_memory_stats.return_value = mock_memory_stats

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "optimize_memory_tokens":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(max_tokens=2000)

        # Verify result
        assert result["optimization_needed"] is False
        assert result["current_tokens"] == 500
        assert result["token_limit"] == 2000

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_optimize_memory_tokens_optimization_performed(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_stats
    ):
        """Test memory optimization when optimization is needed."""
        # Setup mocks - tokens over limit
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        # Initial stats show high token usage
        high_token_stats = Mock()
        high_token_stats.estimated_tokens = 3000

        # Optimized stats show reduced token usage
        optimized_stats = Mock()
        optimized_stats.estimated_tokens = 1500

        mock_memory_manager.get_memory_stats.side_effect = [
            high_token_stats, optimized_stats
        ]
        mock_memory_manager.optimize_memory.return_value = (1500, ["removed_duplicate_rules"])

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "optimize_memory_tokens":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(max_tokens=2000)

        # Verify result
        assert result["optimization_needed"] is True
        assert result["optimization_completed"] is True
        assert result["tokens_before"] == 3000
        assert result["tokens_after"] == 1500
        assert result["tokens_saved"] == 1500


class TestMemoryConflicts:
    """Test memory conflict detection and resolution."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_detect_memory_conflicts_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_conflict
    ):
        """Test successful memory conflict detection."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.detect_conflicts.return_value = [mock_memory_conflict]

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "detect_memory_conflicts":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["conflicts_found"] == 1
        assert len(result["conflicts"]) == 1

        conflict = result["conflicts"][0]
        assert conflict["type"] == "semantic_conflict"
        assert conflict["confidence"] == 0.85
        assert "rule1" in conflict
        assert "rule2" in conflict


class TestMemoryProfile:
    """Test memory profile export functionality."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_export_memory_profile_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule,
        mock_memory_stats, mock_memory_conflict
    ):
        """Test successful memory profile export."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        mock_memory_manager.memory_collection_name = "test-memory"
        mock_memory_manager.list_memory_rules.return_value = [mock_memory_rule]
        mock_memory_manager.get_memory_stats.return_value = mock_memory_stats
        mock_memory_manager.detect_conflicts.return_value = [mock_memory_conflict]

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "export_memory_profile":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify result
        assert result["success"] is True
        assert "memory_profile" in result
        assert "export_size" in result

        profile = result["memory_profile"]
        assert "export_timestamp" in profile
        assert profile["memory_collection_name"] == "test-memory"
        assert "statistics" in profile
        assert "rules" in profile
        assert "conflicts" in profile
        assert len(profile["rules"]) == 1
        assert len(profile["conflicts"]) == 1


class TestMemoryContextApplication:
    """Test memory context application functionality."""

    @patch('common.core.client.create_qdrant_client')
    @patch('common.core.collection_naming.create_naming_manager')
    @patch('common.core.config.Config')
    @patch('common.core.memory.MemoryManager')
    async def test_apply_memory_context_success(
        self, mock_memory_manager_class, mock_config_class,
        mock_naming_manager, mock_qdrant_client, mcp_server,
        mock_config, mock_memory_manager, mock_memory_rule
    ):
        """Test successful memory context application."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_memory_manager_class.return_value = mock_memory_manager

        # Create absolute rule
        absolute_rule = MemoryRule(
            id="abs-rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="always-commit",
            rule="Always make atomic commits",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["git"]
        )

        search_results = [(mock_memory_rule, 0.85)]
        mock_memory_manager.search_memory_rules.return_value = search_results
        mock_memory_manager.list_memory_rules.return_value = [absolute_rule]

        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "apply_memory_context":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func(
            task_description="Implement a new Python feature",
            project_context="Web development project"
        )

        # Verify result
        assert result["success"] is True
        assert result["memory_applied"] is True
        assert "applicable_rules" in result
        assert "behavioral_guidance" in result
        assert result["total_applicable_rules"] >= 1


class TestErrorHandling:
    """Test error handling in memory tools."""

    @patch('common.core.config.Config')
    async def test_memory_tool_generic_error(self, mock_config_class, mcp_server):
        """Test generic error handling in memory tools."""
        # Setup mock to raise exception
        mock_config_class.side_effect = Exception("Database connection failed")

        # Get any tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "get_memory_stats":
                tool_func = tool.fn
                break

        # Execute the tool
        result = await tool_func()

        # Verify error handling
        assert "error" in result
        assert "Database connection failed" in result["error"]

    async def test_search_memory_rules_invalid_params(self, mcp_server):
        """Test search with invalid parameters."""
        # Get the tool function
        tool_func = None
        for tool in mcp_server._tools.values():
            if tool.name == "search_memory_rules":
                tool_func = tool.fn
                break

        # Execute with invalid category
        result = await tool_func(
            query="test query",
            category="invalid_category"
        )

        # Verify error handling
        assert result["success"] is False
        assert "Invalid category" in result["error"]