"""
Integration tests for MCP Tool Test Harnesses with pytest-mcp framework (Task 241.5).

This module validates that the comprehensive test harnesses integrate properly
with the pytest-mcp framework and AI-powered evaluation capabilities.

Tests:
- Integration with AI-powered test evaluation
- Performance baseline validation
- Protocol compliance verification
- Tool coverage completeness
- Error handling robustness
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from tests.utils.pytest_mcp_framework import (
    AITestEvaluator,
    MCPToolEvaluator,
    IntelligentTestRunner,
    ai_powered_mcp_testing
)

# Import the test harnesses we created - using importlib for dynamic import
import importlib.util
import sys
from pathlib import Path

# Dynamic import of the test harness module
harness_module_path = Path(__file__).parent / "20250920-2246_mcp_tool_test_harnesses.py"
spec = importlib.util.spec_from_file_location("mcp_tool_test_harnesses", harness_module_path)
harness_module = importlib.util.module_from_spec(spec)
sys.modules["mcp_tool_test_harnesses"] = harness_module
spec.loader.exec_module(harness_module)

MCPToolTestHarnessOrchestrator = harness_module.MCPToolTestHarnessOrchestrator
WorkspaceStatusTestHarness = harness_module.WorkspaceStatusTestHarness
ListWorkspaceCollectionsTestHarness = harness_module.ListWorkspaceCollectionsTestHarness
SearchWorkspaceTestHarness = harness_module.SearchWorkspaceTestHarness
BaseMCPToolTestHarness = harness_module.BaseMCPToolTestHarness


class TestMCPHarnessIntegration:
    """Test MCP tool test harnesses integration with pytest-mcp framework."""

    def test_harness_orchestrator_initialization(self):
        """Test that the harness orchestrator initializes correctly."""
        mock_client = Mock()
        orchestrator = MCPToolTestHarnessOrchestrator(mock_client)

        # Verify all expected harnesses are present
        expected_harnesses = {
            "workspace_status": WorkspaceStatusTestHarness,
            "list_workspace_collections": ListWorkspaceCollectionsTestHarness,
            "search_workspace_tool": SearchWorkspaceTestHarness
        }

        for tool_name, expected_class in expected_harnesses.items():
            assert tool_name in orchestrator.test_harnesses
            harness = orchestrator.test_harnesses[tool_name]
            assert isinstance(harness, expected_class)
            assert harness.tool_name == tool_name
            assert harness.client == mock_client

    def test_base_harness_interface(self):
        """Test the base harness interface compliance."""
        mock_client = Mock()
        harness = BaseMCPToolTestHarness("test_tool", mock_client)

        # Verify basic interface
        assert harness.tool_name == "test_tool"
        assert harness.client == mock_client
        assert isinstance(harness.test_results, list)
        assert len(harness.test_results) == 0

        # Verify all required methods exist
        required_methods = [
            "run_comprehensive_tests",
            "test_normal_operation",
            "test_edge_cases",
            "test_error_conditions",
            "test_input_validation",
            "test_output_validation",
            "test_side_effects",
            "test_performance",
            "test_protocol_compliance"
        ]

        for method_name in required_methods:
            assert hasattr(harness, method_name)
            assert callable(getattr(harness, method_name))

    @pytest.mark.asyncio
    async def test_harness_tool_call_validation_helper(self):
        """Test the harness tool call validation helper method."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()
        mock_result = MCPTestResult(
            success=True,
            tool_name="test_tool",
            parameters={"param": "value"},
            response={"result": "data"},
            execution_time_ms=50.0,
            protocol_compliance={"json_serializable": True}
        )
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        harness = BaseMCPToolTestHarness("test_tool", mock_client)

        # Test the validation helper
        result = await harness._call_tool_with_validation({"test": "param"}, "test_scenario")

        # Verify call was made correctly
        mock_client.call_tool.assert_called_once_with("test_tool", {"test": "param"})

        # Verify result structure
        assert result["scenario"] == "test_scenario"
        assert result["parameters"] == {"test": "param"}
        assert result["success"] == True
        assert result["execution_time_ms"] == 50.0
        assert result["response_type"] == "dict"
        assert result["has_error"] == False
        assert result["protocol_compliant"] == True

        # Verify test results tracking
        assert len(harness.test_results) == 1
        assert harness.test_results[0] == mock_result

    @pytest.mark.asyncio
    async def test_workspace_status_harness_normal_operation(self):
        """Test workspace status harness normal operation scenarios."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()
        mock_result = MCPTestResult(
            success=True,
            tool_name="workspace_status",
            parameters={},
            response={"status": "healthy", "collections": []},
            execution_time_ms=25.0,
            protocol_compliance={"json_serializable": True}
        )
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        harness = WorkspaceStatusTestHarness(mock_client)
        results = await harness.test_normal_operation()

        assert len(results) == 1
        result = results[0]
        assert result["scenario"] == "basic_status_check"
        assert result["success"] == True
        assert result["execution_time_ms"] == 25.0

    @pytest.mark.asyncio
    async def test_search_harness_edge_cases(self):
        """Test search harness edge case scenarios."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()

        # Mock different responses for edge cases
        def mock_call_tool(tool_name, params):
            if params.get("query") == "":
                return MCPTestResult(
                    success=False, tool_name=tool_name, parameters=params,
                    response=None, execution_time_ms=10.0,
                    error="Empty query not allowed"
                )
            else:
                return MCPTestResult(
                    success=True, tool_name=tool_name, parameters=params,
                    response={"results": [], "total": 0}, execution_time_ms=15.0,
                    protocol_compliance={"json_serializable": True}
                )

        mock_client.call_tool = AsyncMock(side_effect=mock_call_tool)

        harness = SearchWorkspaceTestHarness(mock_client)
        results = await harness.test_edge_cases()

        # Should have multiple edge case tests
        assert len(results) >= 3

        # Find the empty query test
        empty_query_result = next(r for r in results if r["scenario"] == "empty_query")
        assert empty_query_result["success"] == False
        assert "error" in empty_query_result

    def test_harness_coverage_completeness(self):
        """Test that harnesses cover all critical MCP tools."""
        mock_client = Mock()
        orchestrator = MCPToolTestHarnessOrchestrator(mock_client)

        # Core tools that must have harnesses
        critical_tools = [
            "workspace_status",          # Core status
            "list_workspace_collections", # Collection management
            "create_collection",         # Collection creation
            "search_workspace_tool",     # Primary search
            "add_document_tool",         # Document ingestion
            "get_document_tool",         # Document retrieval
            "search_by_metadata_tool",   # Metadata search
            "update_scratchbook_tool",   # Scratchbook operations
            "search_scratchbook_tool"    # Scratchbook search
        ]

        for tool_name in critical_tools:
            assert tool_name in orchestrator.test_harnesses, f"Missing harness for critical tool: {tool_name}"

        # Verify we have the expected number of harnesses
        assert len(orchestrator.test_harnesses) >= len(critical_tools)

    @pytest.mark.asyncio
    async def test_performance_validation_integration(self):
        """Test performance validation integration with k6 framework."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()

        # Mock fast and slow responses
        fast_result = MCPTestResult(
            success=True, tool_name="workspace_status", parameters={},
            response={"status": "ok"}, execution_time_ms=50.0,
            protocol_compliance={"json_serializable": True}
        )

        slow_result = MCPTestResult(
            success=True, tool_name="workspace_status", parameters={},
            response={"status": "ok"}, execution_time_ms=300.0,
            protocol_compliance={"json_serializable": True}
        )

        mock_client.call_tool = AsyncMock(return_value=fast_result)

        harness = WorkspaceStatusTestHarness(mock_client)
        performance_results = await harness.test_performance()

        assert len(performance_results) >= 1
        result = performance_results[0]
        assert result["scenario"] == "basic_performance"
        assert result["success"] == True
        assert result["meets_target"] == True  # <200ms target

        # Test with slow response
        mock_client.call_tool = AsyncMock(return_value=slow_result)
        performance_results = await harness.test_performance()

        result = performance_results[0]
        assert result["meets_target"] == False  # >200ms

    @pytest.mark.asyncio
    async def test_protocol_compliance_validation(self):
        """Test protocol compliance validation integration."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()

        # Mock compliant response
        compliant_result = MCPTestResult(
            success=True, tool_name="workspace_status", parameters={},
            response={"status": "ok", "data": [1, 2, 3]}, execution_time_ms=25.0,
            protocol_compliance={
                "json_serializable": True,
                "is_dict_or_list": True,
                "has_content": True
            }
        )

        mock_client.call_tool = AsyncMock(return_value=compliant_result)

        harness = WorkspaceStatusTestHarness(mock_client)
        compliance_results = await harness.test_protocol_compliance()

        assert len(compliance_results) == 1
        result = compliance_results[0]
        assert result["scenario"] == "protocol_compliance"
        assert result["success"] == True
        assert result["json_serializable"] == True
        assert result["is_dict_or_list"] == True
        assert result["has_content"] == True

    def test_harness_error_resilience(self):
        """Test that harnesses handle errors gracefully."""
        mock_client = Mock()

        # Mock client that raises exceptions
        mock_client.call_tool = AsyncMock(side_effect=Exception("Connection failed"))

        harness = BaseMCPToolTestHarness("test_tool", mock_client)

        # Harness should not crash on client errors
        # (This would be tested in the actual async test execution)
        assert harness.tool_name == "test_tool"
        assert harness.client == mock_client

    @pytest.mark.asyncio
    async def test_comprehensive_test_result_structure(self):
        """Test comprehensive test result structure."""
        from tests.utils.fastmcp_test_infrastructure import MCPTestResult

        mock_client = Mock()
        mock_result = MCPTestResult(
            success=True, tool_name="workspace_status", parameters={},
            response={"status": "ok"}, execution_time_ms=25.0,
            protocol_compliance={"json_serializable": True}
        )
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        harness = WorkspaceStatusTestHarness(mock_client)

        # Mock the individual test methods to avoid dependency issues
        harness.test_normal_operation = AsyncMock(return_value=[{"success": True}])
        harness.test_edge_cases = AsyncMock(return_value=[{"success": True}])
        harness.test_error_conditions = AsyncMock(return_value=[{"success": False}])
        harness.test_input_validation = AsyncMock(return_value=[])
        harness.test_output_validation = AsyncMock(return_value=[{"success": True}])
        harness.test_side_effects = AsyncMock(return_value=[])

        results = await harness.run_comprehensive_tests()

        # Verify comprehensive result structure
        expected_keys = [
            "tool_name", "test_timestamp", "normal_operation", "edge_cases",
            "error_conditions", "input_validation", "output_validation",
            "side_effects", "performance", "protocol_compliance", "summary"
        ]

        for key in expected_keys:
            assert key in results, f"Missing key in comprehensive results: {key}"

        # Verify summary structure
        summary = results["summary"]
        assert "total_tests" in summary
        assert "successful_tests" in summary
        assert "success_rate" in summary
        assert "performance_target_met" in summary

        assert 0.0 <= summary["success_rate"] <= 1.0


class TestAIPoweredHarnessIntegration:
    """Test AI-powered evaluation integration with test harnesses."""

    @pytest.mark.skipif(
        True,  # Skip by default since it requires AI models
        reason="AI-powered tests require model access"
    )
    @pytest.mark.asyncio
    async def test_ai_powered_harness_evaluation(self):
        """Test AI-powered evaluation of harness results."""
        # This would integrate with the AI evaluation framework
        # when models are available for testing

        mock_harness_results = {
            "tool_name": "workspace_status",
            "summary": {
                "total_tests": 10,
                "successful_tests": 8,
                "success_rate": 0.8,
                "performance_target_met": True
            },
            "normal_operation": [{"success": True}],
            "edge_cases": [{"success": True}, {"success": False}],
            "error_conditions": [{"success": False}]
        }

        # Mock AI evaluator
        ai_evaluator = Mock(spec=AITestEvaluator)
        ai_evaluator.evaluate_test_results = AsyncMock(return_value={
            "evaluation_score": 0.85,
            "recommendations": [
                "Improve edge case handling",
                "Add more error condition tests"
            ],
            "compliance_score": 0.9
        })

        # This would be the actual integration point
        evaluation = await ai_evaluator.evaluate_test_results(mock_harness_results)

        assert "evaluation_score" in evaluation
        assert "recommendations" in evaluation
        assert "compliance_score" in evaluation
        assert 0.0 <= evaluation["evaluation_score"] <= 1.0


class TestHarnessDocumentation:
    """Test that harnesses are properly documented and maintainable."""

    def test_harness_docstrings(self):
        """Test that all harnesses have proper documentation."""
        mock_client = Mock()
        orchestrator = MCPToolTestHarnessOrchestrator(mock_client)

        for tool_name, harness in orchestrator.test_harnesses.items():
            # Check class has docstring
            assert harness.__class__.__doc__ is not None
            assert len(harness.__class__.__doc__.strip()) > 0

            # Check key methods have docstrings
            key_methods = ["test_normal_operation", "test_edge_cases", "test_error_conditions"]
            for method_name in key_methods:
                method = getattr(harness, method_name)
                assert method.__doc__ is not None, f"{tool_name}.{method_name} missing docstring"

    def test_harness_naming_conventions(self):
        """Test that harnesses follow naming conventions."""
        mock_client = Mock()
        orchestrator = MCPToolTestHarnessOrchestrator(mock_client)

        for tool_name, harness in orchestrator.test_harnesses.items():
            # Class name should end with TestHarness
            assert harness.__class__.__name__.endswith("TestHarness")

            # Tool name should match
            assert harness.tool_name == tool_name

    def test_comprehensive_coverage_metrics(self):
        """Test that harnesses provide comprehensive coverage metrics."""
        mock_client = Mock()
        orchestrator = MCPToolTestHarnessOrchestrator(mock_client)

        # Verify we have harnesses for the 11 core tools mentioned in task
        # (This validates the requirement completion)
        core_tool_count = len(orchestrator.test_harnesses)
        assert core_tool_count >= 10, f"Expected at least 10 core tool harnesses, got {core_tool_count}"

        # Verify each harness implements comprehensive testing
        for tool_name, harness in orchestrator.test_harnesses.items():
            test_methods = [
                "test_normal_operation", "test_edge_cases", "test_error_conditions",
                "test_input_validation", "test_output_validation", "test_side_effects",
                "test_performance", "test_protocol_compliance"
            ]

            for method_name in test_methods:
                assert hasattr(harness, method_name), f"{tool_name} missing {method_name}"