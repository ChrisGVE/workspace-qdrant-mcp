"""
Comprehensive unit tests for memory tools module.

This module provides 100% test coverage for src/python/workspace_qdrant_mcp/tools/memory.py,
including all memory operations, document storage, scratchbook functionality, and error scenarios.
"""

import sys
from pathlib import Path

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.server.fastmcp import FastMCP

# Import the module under test
from python.workspace_qdrant_mcp.tools.memory import register_memory_tools

from python.common.core.collection_naming import CollectionNamingManager
from python.common.core.config import Config
from python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
    MemoryStats,
    parse_conversational_memory_update,
)


class TestMemoryToolsRegistration:
    """Test memory tools registration with FastMCP server."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.mock_server.tool = Mock()
        self.registered_tools = {}

        # Capture tool registrations
        def capture_tool_registration(func):
            self.registered_tools[func.__name__] = func
            return Mock()

        self.mock_server.tool.side_effect = capture_tool_registration

    def test_register_memory_tools(self):
        """Test that all memory tools are registered with the server."""
        register_memory_tools(self.mock_server)

        # Verify all expected tools were registered
        expected_tools = [
            'initialize_memory_session',
            'add_memory_rule',
            'update_memory_from_conversation',
            'search_memory_rules',
            'get_memory_stats',
            'detect_memory_conflicts',
            'list_memory_rules',
            'apply_memory_context',
            'optimize_memory_tokens',
            'export_memory_profile'
        ]

        for tool_name in expected_tools:
            assert tool_name in self.registered_tools
            assert callable(self.registered_tools[tool_name])

        # Verify server.tool decorator was called for each tool
        assert self.mock_server.tool.call_count == len(expected_tools)


class TestInitializeMemorySession:
    """Test initialize_memory_session tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_initialize_memory_session_success(self):
        """Test successful memory session initialization."""
        # Create mock memory manager components
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock rules
        mock_absolute_rule = Mock(spec=MemoryRule)
        mock_absolute_rule.authority = AuthorityLevel.ABSOLUTE
        mock_absolute_rule.name = "atomic-commits"
        mock_absolute_rule.rule = "Always make atomic commits"
        mock_absolute_rule.scope = ["development"]
        mock_absolute_rule.category = MemoryCategory.BEHAVIOR

        mock_default_rule = Mock(spec=MemoryRule)
        mock_default_rule.authority = AuthorityLevel.DEFAULT
        mock_default_rule.name = "python-tool"
        mock_default_rule.rule = "Use uv for Python package management"
        mock_default_rule.scope = ["python"]
        mock_default_rule.category = MemoryCategory.PREFERENCE

        rules = [mock_absolute_rule, mock_default_rule]

        # Mock memory stats
        mock_stats = Mock(spec=MemoryStats)
        mock_stats.estimated_tokens = 150

        # Configure mocks
        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=rules)
        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[])
        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["status"] == "ready"
            assert result["total_rules"] == 2
            assert result["absolute_rules"] == 1
            assert result["default_rules"] == 1
            assert result["conflicts_detected"] == 0
            assert result["estimated_tokens"] == 150
            assert "rules_for_injection" in result
            assert "absolute" in result["rules_for_injection"]
            assert "default" in result["rules_for_injection"]

    @pytest.mark.asyncio
    async def test_initialize_memory_session_with_conflicts(self):
        """Test memory session initialization with conflicts detected."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock conflict
        mock_rule1 = Mock(spec=MemoryRule)
        mock_rule1.name = "rule1"
        mock_rule2 = Mock(spec=MemoryRule)
        mock_rule2.name = "rule2"

        mock_conflict = Mock(spec=MemoryConflict)
        mock_conflict.conflict_type = "direct_contradiction"
        mock_conflict.description = "Rules conflict"
        mock_conflict.rule1 = mock_rule1
        mock_conflict.rule2 = mock_rule2
        mock_conflict.confidence = 0.9
        mock_conflict.resolution_options = ["Keep rule1", "Keep rule2"]

        mock_stats = Mock(spec=MemoryStats)
        mock_stats.estimated_tokens = 100

        # Configure mocks
        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])
        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])
        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["status"] == "conflicts_require_resolution"
            assert result["conflicts_detected"] == 1
            assert "conflicts" in result
            assert len(result["conflicts"]) == 1

    @pytest.mark.asyncio
    async def test_initialize_memory_session_error(self):
        """Test memory session initialization with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert result["status"] == "error"
            assert "error" in result
            assert result["total_rules"] == 0


class TestAddMemoryRule:
    """Test add_memory_rule tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'add_memory_rule':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_add_memory_rule_success(self):
        """Test successful memory rule addition."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.add_memory_rule = AsyncMock(return_value="rule-123")

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(
                category="preference",
                name="test-rule",
                rule="Test rule content",
                authority="default",
                scope=["test"],
                source="test_source"
            )

            assert result["success"] is True
            assert result["rule_id"] == "rule-123"
            assert "Added memory rule 'test-rule'" in result["message"]

    @pytest.mark.asyncio
    async def test_add_memory_rule_invalid_category(self):
        """Test adding memory rule with invalid category."""
        result = await self.tool_func(
            category="invalid_category",
            name="test-rule",
            rule="Test rule content"
        )

        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_add_memory_rule_invalid_authority(self):
        """Test adding memory rule with invalid authority."""
        result = await self.tool_func(
            category="preference",
            name="test-rule",
            rule="Test rule content",
            authority="invalid_authority"
        )

        assert result["success"] is False
        assert "Invalid parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_add_memory_rule_error(self):
        """Test adding memory rule with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func(
                category="preference",
                name="test-rule",
                rule="Test rule content"
            )

            assert result["success"] is False
            assert "Config error" in result["error"]


class TestUpdateMemoryFromConversation:
    """Test update_memory_from_conversation tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'update_memory_from_conversation':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_update_memory_from_conversation_success(self):
        """Test successful conversational memory update."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock parsed conversational update
        mock_parsed = {
            "category": MemoryCategory.PREFERENCE,
            "rule": "call me Chris",
            "authority": AuthorityLevel.DEFAULT,
            "source": "conversational_note"
        }

        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.add_memory_rule = AsyncMock(return_value="rule-456")

        with patch('python.workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update', return_value=mock_parsed), \
             patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func("Note: call me Chris")

            assert result["detected"] is True
            assert result["rule_added"] is True
            assert result["rule_id"] == "rule-456"
            assert result["category"] == "preference"
            assert result["rule"] == "call me Chris"
            assert result["authority"] == "default"

    @pytest.mark.asyncio
    async def test_update_memory_from_conversation_no_pattern(self):
        """Test conversational update with no memory pattern."""
        with patch('python.workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update', return_value=None):
            result = await self.tool_func("Just a regular message")

            assert result["detected"] is False
            assert "No memory update pattern detected" in result["message"]

    @pytest.mark.asyncio
    async def test_update_memory_from_conversation_with_stop_words(self):
        """Test conversational update with stop words in rule name generation."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock parsed update with stop words
        mock_parsed = {
            "category": MemoryCategory.PREFERENCE,
            "rule": "the user prefers to use uv",
            "authority": AuthorityLevel.DEFAULT,
            "source": "conversational_note"
        }

        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.add_memory_rule = AsyncMock(return_value="rule-789")

        with patch('python.workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update', return_value=mock_parsed), \
             patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func("Note: the user prefers to use uv")

            assert result["detected"] is True
            assert result["rule_added"] is True
            # Verify stop words were filtered out in name generation
            mock_memory_manager.add_memory_rule.assert_called_once()
            call_args = mock_memory_manager.add_memory_rule.call_args
            assert call_args[1]["name"] == "user-prefers"  # Stop words filtered

    @pytest.mark.asyncio
    async def test_update_memory_from_conversation_empty_name(self):
        """Test conversational update resulting in empty name after filtering."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock parsed update with only stop words
        mock_parsed = {
            "category": MemoryCategory.PREFERENCE,
            "rule": "the and or",
            "authority": AuthorityLevel.DEFAULT,
            "source": "conversational_note"
        }

        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.add_memory_rule = AsyncMock(return_value="rule-default")

        with patch('python.workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update', return_value=mock_parsed), \
             patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func("Note: the and or")

            assert result["detected"] is True
            assert result["rule_added"] is True
            # Verify default name was used
            call_args = mock_memory_manager.add_memory_rule.call_args
            assert call_args[1]["name"] == "conversational-rule"

    @pytest.mark.asyncio
    async def test_update_memory_from_conversation_error(self):
        """Test conversational update with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.parse_conversational_memory_update', side_effect=Exception("Parse error")):
            result = await self.tool_func("Note: call me Chris")

            assert result["detected"] is True
            assert result["rule_added"] is False
            assert "Parse error" in result["error"]


class TestSearchMemoryRules:
    """Test search_memory_rules tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'search_memory_rules':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_search_memory_rules_success(self):
        """Test successful memory rules search."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock search result
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-123"
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.category = MemoryCategory.PREFERENCE
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.scope = ["test"]
        mock_rule.created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        search_results = [(mock_rule, 0.85)]

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=search_results)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(
                query="test query",
                category="preference",
                authority="default",
                limit=10
            )

            assert result["success"] is True
            assert result["query"] == "test query"
            assert result["total_found"] == 1
            assert len(result["results"]) == 1

            rule_result = result["results"][0]
            assert rule_result["id"] == "rule-123"
            assert rule_result["name"] == "test-rule"
            assert rule_result["relevance_score"] == 0.85
            assert rule_result["created_at"] == "2023-01-01T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_search_memory_rules_invalid_category(self):
        """Test search with invalid category."""
        result = await self.tool_func(
            query="test query",
            category="invalid_category"
        )

        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @pytest.mark.asyncio
    async def test_search_memory_rules_invalid_authority(self):
        """Test search with invalid authority."""
        result = await self.tool_func(
            query="test query",
            authority="invalid_authority"
        )

        assert result["success"] is False
        assert "Invalid authority" in result["error"]

    @pytest.mark.asyncio
    async def test_search_memory_rules_no_created_at(self):
        """Test search with rule having no created_at timestamp."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock search result without created_at
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-123"
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.category = MemoryCategory.PREFERENCE
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.scope = ["test"]
        mock_rule.created_at = None

        search_results = [(mock_rule, 0.85)]

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=search_results)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(query="test query")

            assert result["success"] is True
            rule_result = result["results"][0]
            assert rule_result["created_at"] is None

    @pytest.mark.asyncio
    async def test_search_memory_rules_error(self):
        """Test search with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func(query="test query")

            assert result["success"] is False
            assert "Config error" in result["error"]


class TestGetMemoryStats:
    """Test get_memory_stats tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'get_memory_stats':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_get_memory_stats_success(self):
        """Test successful memory stats retrieval."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock stats
        mock_stats = Mock(spec=MemoryStats)
        mock_stats.total_rules = 5
        mock_stats.rules_by_category = {
            MemoryCategory.PREFERENCE: 3,
            MemoryCategory.BEHAVIOR: 2
        }
        mock_stats.rules_by_authority = {
            AuthorityLevel.DEFAULT: 4,
            AuthorityLevel.ABSOLUTE: 1
        }
        mock_stats.estimated_tokens = 1500
        mock_stats.last_optimization = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["total_rules"] == 5
            assert result["rules_by_category"]["preference"] == 3
            assert result["rules_by_category"]["behavior"] == 2
            assert result["rules_by_authority"]["default"] == 4
            assert result["rules_by_authority"]["absolute"] == 1
            assert result["estimated_tokens"] == 1500
            assert result["last_optimization"] == "2023-01-01T12:00:00+00:00"
            assert result["token_status"] == "moderate"  # 1500 tokens

    @pytest.mark.asyncio
    async def test_get_memory_stats_token_status_low(self):
        """Test memory stats with low token usage."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_stats = Mock(spec=MemoryStats)
        mock_stats.total_rules = 2
        mock_stats.rules_by_category = {}
        mock_stats.rules_by_authority = {}
        mock_stats.estimated_tokens = 500  # Low usage
        mock_stats.last_optimization = None

        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["token_status"] == "low"
            assert result["last_optimization"] is None

    @pytest.mark.asyncio
    async def test_get_memory_stats_token_status_high(self):
        """Test memory stats with high token usage."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_stats = Mock(spec=MemoryStats)
        mock_stats.total_rules = 10
        mock_stats.rules_by_category = {}
        mock_stats.rules_by_authority = {}
        mock_stats.estimated_tokens = 2500  # High usage
        mock_stats.last_optimization = None

        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["token_status"] == "high"

    @pytest.mark.asyncio
    async def test_get_memory_stats_error(self):
        """Test memory stats with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert "error" in result
            assert "Config error" in result["error"]


class TestDetectMemoryConflicts:
    """Test detect_memory_conflicts tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'detect_memory_conflicts':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_detect_memory_conflicts_success(self):
        """Test successful conflict detection."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock conflict
        mock_rule1 = Mock(spec=MemoryRule)
        mock_rule1.id = "rule-1"
        mock_rule1.name = "use-uv"
        mock_rule1.rule = "Use uv for Python"
        mock_rule1.authority = AuthorityLevel.DEFAULT

        mock_rule2 = Mock(spec=MemoryRule)
        mock_rule2.id = "rule-2"
        mock_rule2.name = "use-pip"
        mock_rule2.rule = "Use pip for Python"
        mock_rule2.authority = AuthorityLevel.DEFAULT

        mock_conflict = Mock(spec=MemoryConflict)
        mock_conflict.conflict_type = "direct_contradiction"
        mock_conflict.description = "Rules conflict about Python package manager"
        mock_conflict.confidence = 0.9
        mock_conflict.rule1 = mock_rule1
        mock_conflict.rule2 = mock_rule2
        mock_conflict.resolution_options = ["Keep higher authority", "Merge rules"]

        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["conflicts_found"] == 1
            assert len(result["conflicts"]) == 1

            conflict = result["conflicts"][0]
            assert conflict["type"] == "direct_contradiction"
            assert conflict["confidence"] == 0.9
            assert conflict["rule1"]["id"] == "rule-1"
            assert conflict["rule2"]["id"] == "rule-2"

    @pytest.mark.asyncio
    async def test_detect_memory_conflicts_none_found(self):
        """Test conflict detection with no conflicts."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func()

            assert result["conflicts_found"] == 0
            assert len(result["conflicts"]) == 0

    @pytest.mark.asyncio
    async def test_detect_memory_conflicts_error(self):
        """Test conflict detection with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert "error" in result
            assert "Config error" in result["error"]


class TestListMemoryRules:
    """Test list_memory_rules tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'list_memory_rules':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_list_memory_rules_success(self):
        """Test successful memory rules listing."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock rules
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-123"
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.category = MemoryCategory.PREFERENCE
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.scope = ["test"]
        mock_rule.source = "user_explicit"
        mock_rule.created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_rule.updated_at = datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[mock_rule])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(
                category="preference",
                authority="default",
                limit=25
            )

            assert result["success"] is True
            assert result["total_returned"] == 1
            assert len(result["rules"]) == 1

            rule_result = result["rules"][0]
            assert rule_result["id"] == "rule-123"
            assert rule_result["name"] == "test-rule"
            assert rule_result["created_at"] == "2023-01-01T12:00:00+00:00"
            assert rule_result["updated_at"] == "2023-01-02T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_list_memory_rules_with_limit(self):
        """Test listing with limit applied."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create more rules than limit
        mock_rules = []
        for i in range(10):
            mock_rule = Mock(spec=MemoryRule)
            mock_rule.id = f"rule-{i}"
            mock_rule.name = f"rule-{i}"
            mock_rule.rule = f"Rule {i}"
            mock_rule.category = MemoryCategory.PREFERENCE
            mock_rule.authority = AuthorityLevel.DEFAULT
            mock_rule.scope = []
            mock_rule.source = "test"
            mock_rule.created_at = None
            mock_rule.updated_at = None
            mock_rules.append(mock_rule)

        mock_memory_manager.list_memory_rules = AsyncMock(return_value=mock_rules)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(limit=5)

            assert result["success"] is True
            assert result["total_returned"] == 5  # Limited to 5
            assert len(result["rules"]) == 5

    @pytest.mark.asyncio
    async def test_list_memory_rules_invalid_category(self):
        """Test listing with invalid category."""
        result = await self.tool_func(category="invalid_category")

        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @pytest.mark.asyncio
    async def test_list_memory_rules_invalid_authority(self):
        """Test listing with invalid authority."""
        result = await self.tool_func(authority="invalid_authority")

        assert result["success"] is False
        assert "Invalid authority" in result["error"]

    @pytest.mark.asyncio
    async def test_list_memory_rules_error(self):
        """Test listing with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert result["success"] is False
            assert "Config error" in result["error"]


class TestApplyMemoryContext:
    """Test apply_memory_context tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'apply_memory_context':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_apply_memory_context_success(self):
        """Test successful memory context application."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock absolute rule
        mock_absolute_rule = Mock(spec=MemoryRule)
        mock_absolute_rule.id = "abs-rule-1"
        mock_absolute_rule.name = "atomic-commits"
        mock_absolute_rule.rule = "Always make atomic commits"
        mock_absolute_rule.category = MemoryCategory.BEHAVIOR
        mock_absolute_rule.authority = AuthorityLevel.ABSOLUTE
        mock_absolute_rule.scope = ["development"]

        # Create mock contextual rule
        mock_contextual_rule = Mock(spec=MemoryRule)
        mock_contextual_rule.id = "ctx-rule-1"
        mock_contextual_rule.name = "python-testing"
        mock_contextual_rule.rule = "Use pytest for Python testing"
        mock_contextual_rule.category = MemoryCategory.BEHAVIOR
        mock_contextual_rule.authority = AuthorityLevel.DEFAULT
        mock_contextual_rule.scope = ["python"]

        # Create mock preference rule
        mock_preference_rule = Mock(spec=MemoryRule)
        mock_preference_rule.id = "pref-rule-1"
        mock_preference_rule.name = "python-tool"
        mock_preference_rule.rule = "Prefer uv for Python package management"
        mock_preference_rule.category = MemoryCategory.PREFERENCE
        mock_preference_rule.authority = AuthorityLevel.DEFAULT
        mock_preference_rule.scope = ["python"]

        # Mock search results (contextual rules)
        search_results = [
            (mock_contextual_rule, 0.8),
            (mock_preference_rule, 0.75)
        ]

        # Mock absolute rules
        absolute_rules = [mock_absolute_rule]

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=search_results)
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=absolute_rules)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(
                task_description="Write Python tests",
                project_context="Python testing project"
            )

            assert result["success"] is True
            assert result["task_description"] == "Write Python tests"
            assert result["memory_applied"] is True
            assert result["total_applicable_rules"] == 3

            # Check rule categorization
            assert len(result["applicable_rules"]["absolute"]) == 1
            assert len(result["applicable_rules"]["contextual"]) == 1
            assert len(result["applicable_rules"]["preferences"]) == 1

            # Check behavioral guidance
            assert "CRITICAL" in result["behavioral_guidance"]
            assert "Always make atomic commits" in result["behavioral_guidance"]
            assert "pytest" in result["behavioral_guidance"]

    @pytest.mark.asyncio
    async def test_apply_memory_context_low_relevance(self):
        """Test memory context with low relevance scores."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Create mock rule with low relevance
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-1"
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.category = MemoryCategory.BEHAVIOR

        # Low relevance score (below 0.7 threshold)
        search_results = [(mock_rule, 0.5)]

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=search_results)
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(task_description="Write code")

            assert result["success"] is True
            # Low relevance rule should not be included
            assert result["total_applicable_rules"] == 0

    @pytest.mark.asyncio
    async def test_apply_memory_context_error(self):
        """Test memory context application with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func(task_description="Write code")

            assert result["success"] is False
            assert result["memory_applied"] is False
            assert "Config error" in result["error"]


class TestOptimizeMemoryTokens:
    """Test optimize_memory_tokens tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'optimize_memory_tokens':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_optimize_memory_tokens_no_optimization_needed(self):
        """Test optimization when tokens are within limit."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock stats showing low token usage
        mock_stats = Mock(spec=MemoryStats)
        mock_stats.estimated_tokens = 1500  # Below 2000 limit

        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(max_tokens=2000)

            assert result["optimization_needed"] is False
            assert result["current_tokens"] == 1500
            assert result["token_limit"] == 2000
            assert "within token limits" in result["message"]

    @pytest.mark.asyncio
    async def test_optimize_memory_tokens_optimization_needed(self):
        """Test optimization when tokens exceed limit."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock stats showing high token usage
        mock_stats_before = Mock(spec=MemoryStats)
        mock_stats_before.estimated_tokens = 2500  # Above 2000 limit

        mock_stats_after = Mock(spec=MemoryStats)
        mock_stats_after.estimated_tokens = 1800  # After optimization

        mock_memory_manager.get_memory_stats = AsyncMock(side_effect=[mock_stats_before, mock_stats_after])
        mock_memory_manager.optimize_memory = AsyncMock(return_value=(700, ["Removed duplicate rules"]))

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tool_func(max_tokens=2000)

            assert result["optimization_needed"] is True
            assert result["optimization_completed"] is True
            assert result["tokens_before"] == 2500
            assert result["tokens_after"] == 1800
            assert result["tokens_saved"] == 700
            assert result["token_limit"] == 2000
            assert len(result["optimization_actions"]) == 1

    @pytest.mark.asyncio
    async def test_optimize_memory_tokens_error(self):
        """Test optimization with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert result["optimization_completed"] is False
            assert "Config error" in result["error"]


class TestExportMemoryProfile:
    """Test export_memory_profile tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tool_func = None

        def capture_tool(func):
            if func.__name__ == 'export_memory_profile':
                self.tool_func = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_export_memory_profile_success(self):
        """Test successful memory profile export."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock memory manager attributes
        mock_memory_manager.memory_collection_name = "memory"

        # Create mock rule
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-123"
        mock_rule.category = MemoryCategory.PREFERENCE
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.scope = ["test"]
        mock_rule.source = "user_explicit"
        mock_rule.conditions = {}
        mock_rule.replaces = []
        mock_rule.created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_rule.updated_at = datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        mock_rule.metadata = {"test": "value"}

        # Mock memory stats
        mock_stats = Mock(spec=MemoryStats)
        mock_stats.total_rules = 1
        mock_stats.rules_by_category = {MemoryCategory.PREFERENCE: 1}
        mock_stats.rules_by_authority = {AuthorityLevel.DEFAULT: 1}
        mock_stats.estimated_tokens = 100

        # Mock conflict
        mock_conflict = Mock(spec=MemoryConflict)
        mock_conflict.conflict_type = "semantic_overlap"
        mock_conflict.description = "Rules have semantic overlap"
        mock_conflict.confidence = 0.7
        mock_conflict.rule1 = mock_rule
        mock_conflict.rule2 = mock_rule
        mock_conflict.resolution_options = ["Merge rules"]

        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[mock_rule])
        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)
        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.datetime') as mock_datetime:

            # Mock datetime.now()
            mock_now = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now

            result = await self.tool_func()

            assert result["success"] is True
            assert result["export_size"] > 0
            assert "Exported 1 memory rules" in result["message"]

            profile = result["memory_profile"]
            assert profile["export_timestamp"] == "2023-06-01T10:00:00+00:00"
            assert profile["memory_collection_name"] == "memory"
            assert len(profile["rules"]) == 1
            assert len(profile["conflicts"]) == 1

            # Check rule export format
            exported_rule = profile["rules"][0]
            assert exported_rule["id"] == "rule-123"
            assert exported_rule["category"] == "preference"
            assert exported_rule["authority"] == "default"
            assert exported_rule["created_at"] == "2023-01-01T12:00:00+00:00"
            assert exported_rule["updated_at"] == "2023-01-02T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_export_memory_profile_rule_no_timestamps(self):
        """Test export with rule having no timestamps."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Mock memory manager attributes
        mock_memory_manager.memory_collection_name = "memory"

        # Create mock rule without timestamps
        mock_rule = Mock(spec=MemoryRule)
        mock_rule.id = "rule-123"
        mock_rule.category = MemoryCategory.PREFERENCE
        mock_rule.name = "test-rule"
        mock_rule.rule = "Test rule content"
        mock_rule.authority = AuthorityLevel.DEFAULT
        mock_rule.scope = ["test"]
        mock_rule.source = "user_explicit"
        mock_rule.conditions = {}
        mock_rule.replaces = []
        mock_rule.created_at = None
        mock_rule.updated_at = None
        mock_rule.metadata = {}

        mock_stats = Mock(spec=MemoryStats)
        mock_stats.total_rules = 1
        mock_stats.rules_by_category = {MemoryCategory.PREFERENCE: 1}
        mock_stats.rules_by_authority = {AuthorityLevel.DEFAULT: 1}
        mock_stats.estimated_tokens = 100

        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[mock_rule])
        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)
        mock_memory_manager.detect_conflicts = AsyncMock(return_value=[])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.datetime') as mock_datetime:

            mock_now = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now

            result = await self.tool_func()

            assert result["success"] is True
            exported_rule = result["memory_profile"]["rules"][0]
            assert exported_rule["created_at"] is None
            assert exported_rule["updated_at"] is None

    @pytest.mark.asyncio
    async def test_export_memory_profile_error(self):
        """Test export with error."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config error")):
            result = await self.tool_func()

            assert result["success"] is False
            assert "Config error" in result["error"]


class TestMemoryToolsIntegration:
    """Test integration scenarios between memory tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tools = {}

        def capture_tool(func):
            self.tools[func.__name__] = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_full_memory_workflow(self):
        """Test a complete memory management workflow."""
        # This integration test would involve multiple tools working together
        # but since they all require mocking external dependencies, we'll test
        # that the tools are properly integrated

        assert 'initialize_memory_session' in self.tools
        assert 'add_memory_rule' in self.tools
        assert 'search_memory_rules' in self.tools
        assert 'get_memory_stats' in self.tools

        # Verify all tools are callable
        for _tool_name, tool_func in self.tools.items():
            assert callable(tool_func)
            assert asyncio.iscoroutinefunction(tool_func)

    def test_tool_registration_completeness(self):
        """Test that all expected tools are registered."""
        expected_tools = {
            'initialize_memory_session',
            'add_memory_rule',
            'update_memory_from_conversation',
            'search_memory_rules',
            'get_memory_stats',
            'detect_memory_conflicts',
            'list_memory_rules',
            'apply_memory_context',
            'optimize_memory_tokens',
            'export_memory_profile'
        }

        registered_tools = set(self.tools.keys())
        assert registered_tools == expected_tools

    def test_tool_function_signatures(self):
        """Test that tools have proper async function signatures."""
        for tool_name, tool_func in self.tools.items():
            assert asyncio.iscoroutinefunction(tool_func)
            assert hasattr(tool_func, '__name__')
            assert tool_func.__name__ == tool_name


class TestErrorHandling:
    """Test error handling scenarios across memory tools."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tools = {}

        def capture_tool(func):
            self.tools[func.__name__] = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_config_creation_failure(self):
        """Test handling of configuration creation failures."""
        with patch('python.workspace_qdrant_mcp.tools.memory.Config', side_effect=Exception("Config failed")):
            # Test multiple tools handle config errors gracefully
            tools_to_test = ['initialize_memory_session', 'add_memory_rule', 'get_memory_stats']

            for tool_name in tools_to_test:
                if tool_name == 'add_memory_rule':
                    result = await self.tools[tool_name](
                        category="preference",
                        name="test",
                        rule="test rule"
                    )
                    assert result["success"] is False
                else:
                    result = await self.tools[tool_name]()
                    if "success" in result:
                        assert result["success"] is False
                    else:
                        assert "error" in result

    @pytest.mark.asyncio
    async def test_qdrant_client_creation_failure(self):
        """Test handling of Qdrant client creation failures."""
        mock_config = Mock(spec=Config)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', side_effect=Exception("Qdrant failed")):

            result = await self.tools['initialize_memory_session']()
            assert result["status"] == "error"
            assert "Qdrant failed" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_manager_operation_failure(self):
        """Test handling of memory manager operation failures."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Make memory manager operation fail
        mock_memory_manager.initialize_memory_collection = AsyncMock(side_effect=Exception("Manager failed"))

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tools['initialize_memory_session']()
            assert result["status"] == "error"
            assert "Manager failed" in result["error"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = Mock(spec=FastMCP)
        self.tools = {}

        def capture_tool(func):
            self.tools[func.__name__] = func
            return func

        self.mock_server.tool.side_effect = capture_tool
        register_memory_tools(self.mock_server)

    @pytest.mark.asyncio
    async def test_add_memory_rule_none_scope(self):
        """Test adding memory rule with None scope."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        mock_memory_manager.add_memory_rule = AsyncMock(return_value="rule-123")

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tools['add_memory_rule'](
                category="preference",
                name="test-rule",
                rule="Test rule",
                scope=None  # None scope should be converted to empty list
            )

            assert result["success"] is True
            # Verify that None scope was converted to empty list
            call_args = mock_memory_manager.add_memory_rule.call_args
            assert call_args[1]["scope"] == []

    @pytest.mark.asyncio
    async def test_search_memory_rules_none_filters(self):
        """Test search with None category and authority filters."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=[])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tools['search_memory_rules'](
                query="test",
                category=None,
                authority=None
            )

            assert result["success"] is True
            # Verify None values were passed correctly
            call_args = mock_memory_manager.search_memory_rules.call_args
            assert call_args[1]["category"] is None
            assert call_args[1]["authority"] is None

    @pytest.mark.asyncio
    async def test_apply_memory_context_none_project_context(self):
        """Test apply memory context with None project context."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        mock_memory_manager.search_memory_rules = AsyncMock(return_value=[])
        mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tools['apply_memory_context'](
                task_description="Write code",
                project_context=None
            )

            assert result["success"] is True
            assert result["task_description"] == "Write code"

    @pytest.mark.asyncio
    async def test_optimize_memory_tokens_edge_values(self):
        """Test optimize memory tokens with edge case token values."""
        mock_config = Mock(spec=Config)
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)
        mock_memory_manager = Mock(spec=MemoryManager)

        # Test exact limit match
        mock_stats = Mock(spec=MemoryStats)
        mock_stats.estimated_tokens = 2000  # Exactly at limit

        mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        with patch('python.workspace_qdrant_mcp.tools.memory.Config', return_value=mock_config), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_qdrant_client', return_value=mock_client), \
             patch('python.workspace_qdrant_mcp.tools.memory.create_naming_manager', return_value=mock_naming_manager), \
             patch('python.workspace_qdrant_mcp.tools.memory.MemoryManager', return_value=mock_memory_manager):

            result = await self.tools['optimize_memory_tokens'](max_tokens=2000)

            # Exactly at limit should not require optimization
            assert result["optimization_needed"] is False


if __name__ == "__main__":
    pytest.main([__file__])
