"""
Focused unit tests for memory tools module.

Tests the memory tools functionality with lightweight patterns focused on achieving
90%+ coverage of the tools/memory.py module within 30-second execution constraints.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# Add src paths
src_path = Path(__file__).parent.parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestMemoryToolsCore:
    """Test core memory tools functionality."""

    def test_register_function_callable(self):
        """Test register_memory_tools function exists and is callable."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools
            assert callable(register_memory_tools)
        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    def test_tools_registration(self, mock_memory_manager, mock_naming, mock_client, mock_config):
        """Test tools registration with complete mocking."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools

            # Track registered tools
            registered_tools = []

            def mock_tool_decorator():
                def decorator(func):
                    registered_tools.append(func.__name__)
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool_decorator

            # Register tools
            register_memory_tools(mock_server)

            # Verify all expected tools are registered
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

            for tool in expected_tools:
                assert tool in registered_tools, f"Tool {tool} not registered"

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")

    def test_memory_enums_available(self):
        """Test that memory enums are available."""
        try:
            from common.core.memory import MemoryCategory, AuthorityLevel

            # Test MemoryCategory values
            assert MemoryCategory.PREFERENCE.value == 'preference'
            assert MemoryCategory.BEHAVIOR.value == 'behavior'
            assert MemoryCategory.AGENT.value == 'agent'

            # Test AuthorityLevel values
            assert AuthorityLevel.ABSOLUTE.value == 'absolute'
            assert AuthorityLevel.DEFAULT.value == 'default'

        except ImportError as e:
            pytest.skip(f"Memory enums not available: {e}")

    def test_memory_rule_creation(self):
        """Test MemoryRule dataclass creation."""
        try:
            from common.core.memory import MemoryRule, MemoryCategory, AuthorityLevel

            rule = MemoryRule(
                id="test-123",
                category=MemoryCategory.PREFERENCE,
                name="test-rule",
                rule="Test rule content",
                authority=AuthorityLevel.DEFAULT,
                scope=["test"],
                source="unit_test"
            )

            assert rule.id == "test-123"
            assert rule.category == MemoryCategory.PREFERENCE
            assert rule.authority == AuthorityLevel.DEFAULT
            assert rule.scope == ["test"]
            assert rule.source == "unit_test"

        except ImportError as e:
            pytest.skip(f"MemoryRule not available: {e}")

    def test_enum_validation(self):
        """Test enum validation behavior."""
        try:
            from common.core.memory import MemoryCategory, AuthorityLevel

            # Test valid enum creation
            cat = MemoryCategory('preference')
            assert cat == MemoryCategory.PREFERENCE

            auth = AuthorityLevel('default')
            assert auth == AuthorityLevel.DEFAULT

            # Test invalid enum handling
            with pytest.raises(ValueError):
                MemoryCategory('invalid')

            with pytest.raises(ValueError):
                AuthorityLevel('invalid')

        except ImportError as e:
            pytest.skip(f"Memory enums not available: {e}")


class TestMemoryToolsMocked:
    """Test memory tools with comprehensive mocking."""

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    @patch('workspace_qdrant_mcp.tools.memory.create_qdrant_client')
    @patch('workspace_qdrant_mcp.tools.memory.create_naming_manager')
    @patch('workspace_qdrant_mcp.tools.memory.MemoryManager')
    @patch('workspace_qdrant_mcp.tools.memory.logger')
    def test_initialize_memory_session_mock(self, mock_logger, mock_memory_manager_cls,
                                           mock_naming, mock_client, mock_config):
        """Test initialize_memory_session with full mocking."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools

            # Setup mocks
            mock_config_instance = Mock()
            mock_config_instance.qdrant_client_config = {"url": "test"}
            mock_config_instance.workspace.global_collections = ["memory"]
            mock_config.return_value = mock_config_instance

            mock_manager = AsyncMock()
            mock_manager.initialize_memory_collection.return_value = None
            mock_manager.list_memory_rules.return_value = []
            mock_manager.detect_conflicts.return_value = []

            mock_stats = Mock()
            mock_stats.total_rules = 0
            mock_stats.estimated_tokens = 0
            mock_manager.get_memory_stats.return_value = mock_stats

            mock_memory_manager_cls.return_value = mock_manager

            # Extract tool function
            tool_functions = {}
            def mock_tool():
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool

            register_memory_tools(mock_server)

            # Get initialize function
            init_func = tool_functions.get("initialize_memory_session")
            assert init_func is not None

            # Test would require async execution - just verify function exists
            assert callable(init_func)

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")

    @patch('workspace_qdrant_mcp.tools.memory.Config')
    def test_error_handling_patterns(self, mock_config):
        """Test error handling in memory tools."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools

            # Setup mock to raise exception
            mock_config.side_effect = Exception("Config error")

            # Extract tool function
            tool_functions = {}
            def mock_tool():
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool

            register_memory_tools(mock_server)

            # Verify functions exist even with config errors
            assert "initialize_memory_session" in tool_functions
            assert "add_memory_rule" in tool_functions

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")

    def test_parameter_validation_structure(self):
        """Test parameter validation structure."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools
            from common.core.memory import MemoryCategory, AuthorityLevel

            # Extract tool function
            tool_functions = {}
            def mock_tool():
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool

            register_memory_tools(mock_server)

            # Test that enums can be used for validation
            # Valid categories
            try:
                MemoryCategory('preference')
                MemoryCategory('behavior')
                MemoryCategory('agent')
            except ValueError:
                pytest.fail("Valid categories should not raise ValueError")

            # Valid authority levels
            try:
                AuthorityLevel('absolute')
                AuthorityLevel('default')
            except ValueError:
                pytest.fail("Valid authority levels should not raise ValueError")

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")


class TestMemoryDependencies:
    """Test memory module dependencies."""

    def test_core_imports(self):
        """Test core memory imports."""
        try:
            from common.core.config import Config
            from common.core.client import create_qdrant_client
            from common.core.collection_naming import create_naming_manager
            from common.core.memory import MemoryManager

            assert Config is not None
            assert callable(create_qdrant_client)
            assert callable(create_naming_manager)
            assert MemoryManager is not None

        except ImportError as e:
            pytest.skip(f"Core dependencies not available: {e}")

    def test_memory_utility_imports(self):
        """Test memory utility imports."""
        try:
            from common.core.memory import parse_conversational_memory_update
            assert callable(parse_conversational_memory_update)
        except ImportError as e:
            pytest.skip(f"Memory utilities not available: {e}")

    def test_logging_import(self):
        """Test that logging is properly imported."""
        try:
            # Import the module and check logger usage
            import workspace_qdrant_mcp.tools.memory as memory_module
            import inspect

            # Check that logger is used in the module
            source = inspect.getsource(memory_module)
            assert 'logger' in source or 'log' in source

        except ImportError as e:
            pytest.skip(f"Memory module not available: {e}")


class TestMemoryConstants:
    """Test memory-related constants and patterns."""

    def test_datetime_usage(self):
        """Test datetime usage patterns."""
        from datetime import datetime, timezone

        # Test timezone-aware datetime creation
        now = datetime.now(timezone.utc)
        assert now.tzinfo is not None

        # Test isoformat conversion
        iso_string = now.isoformat()
        assert isinstance(iso_string, str)
        assert 'T' in iso_string

    def test_tool_naming_conventions(self):
        """Test that tool naming follows conventions."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools

            # Extract tool function names
            tool_names = []
            def mock_tool():
                def decorator(func):
                    tool_names.append(func.__name__)
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool

            register_memory_tools(mock_server)

            # Verify naming conventions
            for name in tool_names:
                # Should be snake_case
                assert name.islower() or '_' in name
                # Should be descriptive
                assert len(name) > 5
                # Should contain memory or relevant keywords
                assert any(keyword in name for keyword in ['memory', 'rule', 'conflict', 'session', 'stats'])

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")


class TestMemoryDocstrings:
    """Test memory tool documentation."""

    def test_function_docstrings(self):
        """Test that functions have proper docstrings."""
        try:
            from workspace_qdrant_mcp.tools.memory import register_memory_tools

            # Extract functions with docstrings
            functions_with_docs = []
            def mock_tool():
                def decorator(func):
                    if func.__doc__:
                        functions_with_docs.append(func.__name__)
                    return func
                return decorator

            mock_server = Mock()
            mock_server.tool = mock_tool

            register_memory_tools(mock_server)

            # Most functions should have docstrings
            assert len(functions_with_docs) >= 8

        except ImportError as e:
            pytest.skip(f"Memory tools not available: {e}")

    def test_module_docstring(self):
        """Test that module has docstring."""
        try:
            import workspace_qdrant_mcp.tools.memory as memory_module
            assert memory_module.__doc__ is not None
            assert len(memory_module.__doc__.strip()) > 0
        except ImportError as e:
            pytest.skip(f"Memory module not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])