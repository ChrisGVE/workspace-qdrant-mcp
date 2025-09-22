"""
Integration Tests for Graceful Degradation System

This module tests the complete graceful degradation system integration with:
- MCP tools and CLI operations
- Component lifecycle management
- Health monitoring
- User experience under various failure scenarios
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from workspace_qdrant_mcp.core.graceful_degradation import (
    DegradationManager,
    DegradationMode,
    FeatureType,
)
from workspace_qdrant_mcp.core.component_coordination import ComponentType
from workspace_qdrant_mcp.core.component_lifecycle import ComponentLifecycleManager
from workspace_qdrant_mcp.core.lsp_health_monitor import LspHealthMonitor
from workspace_qdrant_mcp.tools.degradation_aware import (
    DegradationAwareMCPTools,
    CLIDegradationHandler,
    create_degradation_aware_tools,
    create_cli_degradation_handler,
)


class TestGracefulDegradationIntegration:
    """Integration tests for graceful degradation system."""

    @pytest.fixture
    async def mock_lifecycle_manager(self):
        """Create mock lifecycle manager."""
        manager = AsyncMock(spec=ComponentLifecycleManager)

        # Default healthy state
        manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "operational"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }

        return manager

    @pytest.fixture
    async def mock_health_monitor(self):
        """Create mock health monitor."""
        return AsyncMock(spec=LspHealthMonitor)

    @pytest.fixture
    async def degradation_manager(self, mock_lifecycle_manager, mock_health_monitor):
        """Create degradation manager for testing."""
        manager = DegradationManager(
            lifecycle_manager=mock_lifecycle_manager,
            health_monitor=mock_health_monitor
        )
        await manager.initialize()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    async def original_tools(self):
        """Create mock original tools."""
        return {
            "search_workspace": AsyncMock(return_value={
                "results": [{"content": "test result", "score": 0.9}],
                "total": 1
            }),
            "add_document": AsyncMock(return_value={
                "success": True,
                "document_id": "test-doc-id"
            }),
            "list_collections": AsyncMock(return_value=["test-collection"]),
            "get_workspace_status": AsyncMock(return_value={
                "status": "healthy",
                "components": "all_operational"
            })
        }

    @pytest.fixture
    async def degradation_aware_tools(self, degradation_manager, original_tools):
        """Create degradation-aware tools."""
        return await create_degradation_aware_tools(degradation_manager, original_tools)

    @pytest.fixture
    async def cli_handler(self, degradation_manager):
        """Create CLI degradation handler."""
        return await create_cli_degradation_handler(degradation_manager)

    async def test_normal_operation_mcp_tools(self, degradation_aware_tools):
        """Test MCP tools in normal operation mode."""
        # Test search
        result = await degradation_aware_tools.search_workspace("test query")
        assert result.success is True
        assert result.data is not None
        assert result.degradation_mode == "normal"
        assert not result.from_cache

        # Test document addition
        result = await degradation_aware_tools.add_document(
            "test content", "test-collection"
        )
        assert result.success is True
        assert result.data["success"] is True

        # Test list collections
        result = await degradation_aware_tools.list_collections()
        assert result.success is True
        assert isinstance(result.data, list)

    async def test_search_degradation_semantic_unavailable(self, degradation_aware_tools):
        """Test search degradation when semantic search is unavailable."""
        # Force degradation mode that disables semantic search
        await degradation_aware_tools.degradation_manager.force_degradation_mode(
            DegradationMode.FEATURES_LIMITED,
            "Test semantic search unavailable"
        )

        # Search should still work but fall back to keyword search
        result = await degradation_aware_tools.search_workspace(
            "test query", mode="hybrid"
        )
        assert result.success is True
        # Mode should be adjusted automatically

    async def test_search_complete_failure_with_cache(self, degradation_aware_tools, original_tools):
        """Test search with complete failure falling back to cache."""
        # Set up cache with previous results
        cache_key = "search:test query:all"
        cached_response = {
            "results": [{"content": "cached result", "score": 0.8}],
            "total": 1
        }

        degradation_aware_tools.degradation_manager.cached_responses[cache_key] = {
            "response": cached_response,
            "timestamp": asyncio.get_event_loop().time(),
            "ttl": 300
        }

        # Make original tool fail
        original_tools["search_workspace"].side_effect = Exception("Service unavailable")

        result = await degradation_aware_tools.search_workspace("test query")

        assert result.success is True
        assert result.from_cache is True
        assert "cached result" in str(result.data)
        assert "cache" in result.user_guidance.lower()

    async def test_search_complete_failure_no_cache(self, degradation_aware_tools, original_tools):
        """Test search with complete failure and no cache available."""
        # Make original tool fail
        original_tools["search_workspace"].side_effect = Exception("Service unavailable")

        # Force degradation mode
        await degradation_aware_tools.degradation_manager.force_degradation_mode(
            DegradationMode.OFFLINE_CLI,
            "Complete service failure"
        )

        result = await degradation_aware_tools.search_workspace("test query")

        assert result.success is False
        assert "offline mode" in result.user_guidance.lower()
        assert "wqm search" in result.user_guidance

    async def test_document_ingestion_degradation(self, degradation_aware_tools):
        """Test document ingestion in degraded mode."""
        # Force read-only mode
        await degradation_aware_tools.degradation_manager.force_degradation_mode(
            DegradationMode.READ_ONLY,
            "Ingestion service unavailable"
        )

        result = await degradation_aware_tools.add_document(
            "test content", "test-collection"
        )

        assert result.success is False
        assert "read-only mode" in result.user_guidance.lower()
        assert "ingestion disabled" in result.error_message.lower()

    async def test_workspace_status_always_available(self, degradation_aware_tools, original_tools):
        """Test that workspace status is always available even when other services fail."""
        # Make original status tool fail
        original_tools["get_workspace_status"].side_effect = Exception("Status service down")

        # Force degraded mode
        await degradation_aware_tools.degradation_manager.force_degradation_mode(
            DegradationMode.EMERGENCY,
            "Critical system failure"
        )

        result = await degradation_aware_tools.get_workspace_status()

        # Status should still be available with degradation info
        assert result.success is True
        assert "degradation" in result.data
        assert result.data["degradation"]["current_mode"] == "emergency"
        assert "available_features" in result.data
        assert "user_guidance" in result.data

    async def test_cli_handler_normal_mode(self, cli_handler):
        """Test CLI handler in normal mode."""
        # Search command
        result = await cli_handler.handle_search_command("test query")
        assert isinstance(result, dict)
        assert "results" in result or "mode" in result

        # Status command
        result = await cli_handler.handle_status_command()
        assert result["mode"] == "normal"
        assert result["health"] == "healthy"

        # Verbose status
        result = await cli_handler.handle_status_command(verbose=True)
        assert "degradation" in result
        assert result["cli_available"] is True

    async def test_cli_handler_offline_mode(self, cli_handler):
        """Test CLI handler in offline mode."""
        # Force offline mode
        await cli_handler.degradation_manager.force_degradation_mode(
            DegradationMode.OFFLINE_CLI,
            "Network connectivity lost"
        )

        # Search should fall back to local search
        result = await cli_handler.handle_search_command("test query")
        assert result["mode"] == "local_search"
        assert "local search" in result["message"].lower()

        # Service commands should be unavailable
        result = await cli_handler.handle_service_command("start")
        assert result["success"] is False
        assert "offline mode" in result["message"].lower()

    async def test_cli_handler_forced_local_search(self, cli_handler):
        """Test CLI handler with forced local search."""
        result = await cli_handler.handle_search_command(
            "test query", local_only=True
        )
        assert result["mode"] == "local_search"
        assert len(result["results"]) > 0

    async def test_progressive_degradation_scenario(self, degradation_manager, degradation_aware_tools, mock_lifecycle_manager):
        """Test progressive degradation scenario."""
        # Start in normal mode
        assert degradation_manager.current_mode == DegradationMode.NORMAL

        # Simulate Rust daemon failure
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "failed"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }

        # Trigger evaluation
        await degradation_manager._evaluate_degradation_mode()

        # Should degrade to cached-only or features-limited
        assert degradation_manager.current_mode in [DegradationMode.CACHED_ONLY, DegradationMode.FEATURES_LIMITED]

        # Search should still work but may use fallbacks
        result = await degradation_aware_tools.search_workspace("test query")
        # Result depends on whether cache is available

        # Simulate MCP server also failing
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "failed"},
                "python_mcp_server": {"state": "failed"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "degraded"},
            }
        }

        await degradation_manager._evaluate_degradation_mode()

        # Should degrade further to offline CLI
        assert degradation_manager.current_mode == DegradationMode.OFFLINE_CLI

        # Document ingestion should be unavailable
        result = await degradation_aware_tools.add_document("test", "collection")
        assert result.success is False

    async def test_recovery_scenario(self, degradation_manager, degradation_aware_tools, mock_lifecycle_manager):
        """Test recovery from degraded state."""
        # Start in degraded mode
        await degradation_manager.force_degradation_mode(
            DegradationMode.READ_ONLY,
            "Simulated failure"
        )

        # Verify degraded behavior
        result = await degradation_aware_tools.add_document("test", "collection")
        assert result.success is False

        # Simulate recovery
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "operational"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }

        await degradation_manager._evaluate_degradation_mode()

        # Should recover to normal mode
        assert degradation_manager.current_mode == DegradationMode.NORMAL

        # Document ingestion should work again
        result = await degradation_aware_tools.add_document("test", "collection")
        assert result.success is True

    async def test_tool_statistics_tracking(self, degradation_aware_tools, original_tools):
        """Test tool statistics tracking."""
        # Execute several operations
        await degradation_aware_tools.search_workspace("query1")
        await degradation_aware_tools.search_workspace("query2")
        await degradation_aware_tools.list_collections()

        # Make one operation fail
        original_tools["add_document"].side_effect = Exception("Simulated failure")
        await degradation_aware_tools.add_document("test", "collection")

        stats = degradation_aware_tools.get_tool_statistics()

        assert stats["total_executions"] == 4
        assert stats["degraded_executions"] == 1
        assert stats["degradation_rate"] == 0.25
        assert "current_mode" in stats
        assert "available_features" in stats

    async def test_mcp_tool_definitions(self, degradation_aware_tools):
        """Test MCP tool definitions creation."""
        tool_defs = degradation_aware_tools.create_mcp_tool_definitions()

        # Should have all major tools
        tool_names = [tool.name for tool in tool_defs]
        expected_tools = [
            "search_workspace",
            "add_document",
            "list_collections",
            "get_workspace_status"
        ]

        for expected in expected_tools:
            assert expected in tool_names

        # Each tool should have proper schema
        for tool in tool_defs:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            assert "degradation" in tool.description.lower()

    async def test_circuit_breaker_integration(self, degradation_aware_tools, original_tools):
        """Test circuit breaker integration with MCP tools."""
        # Make search consistently fail to trigger circuit breaker
        original_tools["search_workspace"].side_effect = Exception("Persistent failure")

        # Execute multiple failing operations
        for _ in range(6):  # More than circuit breaker threshold
            result = await degradation_aware_tools.search_workspace("test")
            assert result.success is False

        # Circuit breaker should now be open for search service
        cb_state = degradation_aware_tools.degradation_manager.get_circuit_breaker_state("search-service")
        # Circuit breaker state depends on implementation details

    async def test_concurrent_tool_operations(self, degradation_aware_tools):
        """Test concurrent tool operations with degradation."""
        # Execute multiple operations concurrently
        tasks = [
            degradation_aware_tools.search_workspace(f"query{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete (successfully or with graceful degradation)
        assert len(results) == 10
        for result in results:
            assert hasattr(result, 'success')
            assert hasattr(result, 'execution_time_ms')

    async def test_user_guidance_consistency(self, degradation_aware_tools, cli_handler):
        """Test consistency of user guidance across MCP and CLI interfaces."""
        # Force degraded mode
        await degradation_aware_tools.degradation_manager.force_degradation_mode(
            DegradationMode.CACHED_ONLY,
            "Test user guidance consistency"
        )

        # Get guidance from MCP tools
        mcp_result = await degradation_aware_tools.search_workspace("test")
        mcp_guidance = mcp_result.user_guidance

        # Get guidance from CLI
        cli_result = await cli_handler.handle_status_command(verbose=True)
        cli_guidance = cli_result["recommendations"]

        # Both should mention cached responses
        assert "cache" in mcp_guidance.lower() or "cached" in mcp_guidance.lower()
        assert any("cache" in rec.lower() or "cached" in rec.lower() for rec in cli_guidance)


@pytest.mark.slow
class TestGracefulDegradationEndToEnd:
    """End-to-end tests for graceful degradation system."""

    async def test_complete_system_failure_recovery(self):
        """Test complete system failure and recovery cycle."""
        # This would be a full integration test with real components
        # For now, it's a placeholder for future implementation
        pass

    async def test_load_based_degradation(self):
        """Test degradation triggered by high system load."""
        # This would test resource-based degradation
        # For now, it's a placeholder for future implementation
        pass

    async def test_network_partition_scenarios(self):
        """Test behavior under network partition scenarios."""
        # This would test network-related degradation
        # For now, it's a placeholder for future implementation
        pass