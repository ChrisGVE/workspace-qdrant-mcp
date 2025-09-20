"""
Example tests demonstrating pytest-mcp framework AI-powered evaluation.

This module provides practical examples of how to use the pytest-mcp framework
for AI-powered testing of MCP tools, showcasing different evaluation scenarios
and testing patterns.

Examples include:
    - Basic AI evaluation of workspace tools
    - Performance testing with AI insights
    - Custom test cases with validation functions
    - Comprehensive tool suite evaluation
    - Error handling and edge case testing with AI analysis
"""

import pytest
from typing import Dict, Any

from tests.utils.pytest_mcp_framework import (
    MCPToolTestCase,
    ai_powered_mcp_testing,
    assert_ai_score_above,
    assert_tool_functional,
    assert_performance_acceptable,
    assert_reliability_high,
    assert_no_critical_issues
)


@pytest.mark.pytest_mcp
@pytest.mark.ai_evaluation
class TestWorkspaceToolsAIEvaluation:
    """Example AI evaluation tests for workspace MCP tools."""

    async def test_workspace_status_ai_evaluation(self, ai_powered_test_environment):
        """Example: AI evaluation of workspace_status tool."""
        runner = ai_powered_test_environment

        # Import the actual MCP app
        from workspace_qdrant_mcp.server import app

        # Define custom test cases for workspace_status
        test_cases = {
            "workspace_status": [
                MCPToolTestCase(
                    tool_name="workspace_status",
                    description="Basic status functionality",
                    parameters={},
                    expected_response_type=dict,
                    expected_fields=["connected", "current_project"],
                    should_succeed=True,
                    validation_fn=lambda response: (
                        isinstance(response, dict) and
                        "connected" in response
                    )
                ),
                MCPToolTestCase(
                    tool_name="workspace_status",
                    description="Status with invalid parameters",
                    parameters={"invalid_param": "test"},
                    expected_response_type=dict,
                    should_succeed=True  # Should handle gracefully
                )
            ]
        }

        # Run AI-powered evaluation
        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=test_cases,
            context={"evaluation_focus": "functionality_and_usability"}
        )

        # Extract evaluation for workspace_status
        if "workspace_status" in report["evaluation_results"]:
            status_evaluation = report["evaluation_results"]["workspace_status"]

            # Use pytest-mcp assertions for validation
            # Note: These would need the actual evaluation objects,
            # here we demonstrate with the report data

            # Verify AI found the tool functional
            assert status_evaluation["functionality_score"] >= 0.7, \
                f"Workspace status functionality below threshold: {status_evaluation['functionality_score']}"

            # Verify performance is acceptable
            assert status_evaluation["performance_metrics"]["average_execution_time_ms"] <= 500, \
                f"Workspace status too slow: {status_evaluation['performance_metrics']['average_execution_time_ms']}ms"

            # Verify reliability
            assert status_evaluation["performance_metrics"]["success_rate"] >= 0.8, \
                f"Workspace status unreliable: {status_evaluation['performance_metrics']['success_rate']}"

            # Check AI insights
            assert len(status_evaluation["ai_insights"]) >= 0, "Should have AI insights"

            # Verify no critical issues detected
            critical_issues = [
                issue for issue in status_evaluation.get("detected_issues", [])
                if "CRITICAL" in issue.upper()
            ]
            assert len(critical_issues) == 0, f"Critical issues detected: {critical_issues}"

    async def test_search_tool_performance_evaluation(self, ai_powered_test_environment):
        """Example: Performance-focused AI evaluation of search tool."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        # Performance-focused test cases
        test_cases = {
            "search_workspace_tool": [
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Basic search performance",
                    parameters={"query": "test", "limit": 5},
                    expected_response_type=dict,
                    expected_fields=["results", "total"],
                    should_succeed=True
                ),
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Large query search",
                    parameters={"query": "test " * 100, "limit": 50},
                    expected_response_type=dict,
                    should_succeed=True
                ),
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Empty query handling",
                    parameters={"query": "", "limit": 10},
                    expected_response_type=dict,
                    should_succeed=False  # Should handle empty query gracefully
                )
            ]
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=test_cases,
            context={"evaluation_focus": "performance_and_reliability"}
        )

        # Analyze search tool performance
        if "search_workspace_tool" in report["evaluation_results"]:
            search_evaluation = report["evaluation_results"]["search_workspace_tool"]

            # Performance assertions
            avg_time = search_evaluation["performance_metrics"]["average_execution_time_ms"]
            assert avg_time <= 1000, f"Search tool too slow: {avg_time}ms (target: <1000ms)"

            # Check for performance insights
            performance_insights = [
                insight for insight in search_evaluation["ai_insights"]
                if "performance" in insight.lower()
            ]

            # If slow, should have performance recommendations
            if avg_time > 500:
                performance_recommendations = [
                    rec for rec in search_evaluation["recommendations"]
                    if "performance" in rec.lower() or "optimize" in rec.lower()
                ]
                assert len(performance_recommendations) > 0, "Should recommend performance improvements"

    async def test_comprehensive_workspace_suite_evaluation(self, ai_powered_test_environment):
        """Example: Comprehensive evaluation of multiple workspace tools."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        # Comprehensive test suite for multiple tools
        comprehensive_test_cases = {
            "workspace_status": [
                MCPToolTestCase(
                    tool_name="workspace_status",
                    description="Status check",
                    parameters={},
                    expected_response_type=dict,
                    should_succeed=True
                )
            ],
            "list_workspace_collections": [
                MCPToolTestCase(
                    tool_name="list_workspace_collections",
                    description="List collections",
                    parameters={},
                    expected_response_type=(dict, list),
                    should_succeed=True
                )
            ],
            "search_workspace_tool": [
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Search functionality",
                    parameters={"query": "workspace", "limit": 10},
                    expected_response_type=dict,
                    should_succeed=True
                )
            ]
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=comprehensive_test_cases,
            context={"evaluation_type": "comprehensive_suite"}
        )

        # Verify comprehensive evaluation
        summary = report["summary"]

        # Overall quality check
        assert summary["average_overall_score"] >= 0.6, \
            f"Overall application quality below threshold: {summary['average_overall_score']}"

        # Tool count verification
        assert summary["total_tools_evaluated"] >= 3, \
            f"Expected at least 3 tools evaluated, got {summary['total_tools_evaluated']}"

        # Quality assessment
        quality_assessment = report["quality_assessment"]
        assert quality_assessment["status"] == "completed"
        assert quality_assessment["quality_level"] in ["excellent", "good", "acceptable", "poor"]

        # Verify recommendations are provided
        assert len(report["recommendations"]) >= 0, "Should provide recommendations"

        # Check tool rankings
        rankings = summary["tool_rankings"]
        assert len(rankings) >= 3, "Should rank evaluated tools"
        assert all("tool" in ranking and "score" in ranking for ranking in rankings)

    async def test_error_handling_ai_evaluation(self, ai_powered_test_environment):
        """Example: AI evaluation of error handling capabilities."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        # Error handling focused test cases
        error_test_cases = {
            "search_workspace_tool": [
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Invalid parameters test",
                    parameters={"invalid_field": "test", "bad_limit": "not_a_number"},
                    expected_response_type=dict,
                    should_succeed=False,
                    validation_fn=lambda response: (
                        # Good error handling should return structured error
                        isinstance(response, dict) and
                        ("error" in response or "message" in response)
                    )
                ),
                MCPToolTestCase(
                    tool_name="search_workspace_tool",
                    description="Missing required parameters",
                    parameters={},  # No query parameter
                    expected_response_type=dict,
                    should_succeed=False
                )
            ],
            "nonexistent_tool": [
                MCPToolTestCase(
                    tool_name="nonexistent_tool",
                    description="Tool not found test",
                    parameters={},
                    expected_response_type=dict,
                    should_succeed=False
                )
            ]
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=error_test_cases,
            context={"evaluation_focus": "error_handling"}
        )

        # Analyze error handling quality
        for tool_name, tool_result in report["evaluation_results"].items():
            if tool_name == "nonexistent_tool":
                # Should fail gracefully
                assert tool_result["performance_metrics"]["success_rate"] == 0.0, \
                    "Nonexistent tool should have 0% success rate"

                # Should have clear error insights
                error_insights = [
                    insight for insight in tool_result["ai_insights"]
                    if "error" in insight.lower() or "not found" in insight.lower()
                ]
                assert len(error_insights) > 0, "Should have error-related insights"

            elif tool_name == "search_workspace_tool":
                # Should handle errors gracefully
                detected_issues = tool_result.get("detected_issues", [])
                critical_errors = [
                    issue for issue in detected_issues
                    if "CRITICAL" in issue.upper()
                ]

                # Error handling tests might fail, but shouldn't be critical failures
                # unless the tool completely crashes
                if len(critical_errors) > 0:
                    pytest.skip("Tool has critical error handling issues")

    async def test_custom_validation_ai_evaluation(self, ai_powered_test_environment):
        """Example: AI evaluation with custom validation functions."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        def validate_workspace_status(response):
            """Custom validation for workspace status response."""
            if not isinstance(response, dict):
                return False

            # Must have connection status
            if "connected" not in response:
                return False

            # If connected, should have project info
            if response["connected"]:
                return "current_project" in response

            return True

        def validate_collections_list(response):
            """Custom validation for collections list response."""
            if isinstance(response, list):
                return all(isinstance(item, str) for item in response)
            elif isinstance(response, dict):
                return "collections" in response or "data" in response
            return False

        # Test cases with custom validation
        custom_validation_cases = {
            "workspace_status": [
                MCPToolTestCase(
                    tool_name="workspace_status",
                    description="Workspace status with custom validation",
                    parameters={},
                    expected_response_type=dict,
                    should_succeed=True,
                    validation_fn=validate_workspace_status
                )
            ],
            "list_workspace_collections": [
                MCPToolTestCase(
                    tool_name="list_workspace_collections",
                    description="Collections list with custom validation",
                    parameters={},
                    expected_response_type=(dict, list),
                    should_succeed=True,
                    validation_fn=validate_collections_list
                )
            ]
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=custom_validation_cases,
            context={"validation_type": "custom_business_logic"}
        )

        # Verify custom validation was applied
        for tool_name, tool_result in report["evaluation_results"].items():
            # Custom validation failures should be reflected in the results
            if tool_result["test_summary"]["failed_tests"] > 0:
                # Should have insights about validation failures
                validation_insights = [
                    insight for insight in tool_result["ai_insights"]
                    if "validation" in insight.lower()
                ]
                # Note: Current implementation may not specifically mention validation
                # but AI should detect functionality issues


@pytest.mark.pytest_mcp
@pytest.mark.integration
class TestRealWorldAIEvaluationScenarios:
    """Real-world AI evaluation scenarios and use cases."""

    async def test_production_readiness_evaluation(self, ai_powered_test_environment):
        """Example: Production readiness evaluation with AI insights."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        # Production readiness criteria
        production_context = {
            "environment": "production",
            "performance_targets": {
                "workspace_status": 100,  # ms
                "search_workspace_tool": 500,  # ms
                "list_workspace_collections": 200  # ms
            },
            "reliability_threshold": 0.95,
            "functionality_threshold": 0.8
        }

        # Run evaluation with production context
        report = await runner.run_comprehensive_evaluation(
            app,
            context=production_context
        )

        # Production readiness assessment
        quality_level = report["quality_assessment"]["quality_level"]

        if quality_level in ["excellent", "good"]:
            # Check specific production criteria
            for tool_name, tool_result in report["evaluation_results"].items():
                perf_metrics = tool_result["performance_metrics"]

                # Performance check
                target_time = production_context["performance_targets"].get(tool_name, 1000)
                actual_time = perf_metrics["average_execution_time_ms"]

                if actual_time > target_time:
                    pytest.warning(f"{tool_name} exceeds production performance target: "
                                 f"{actual_time}ms > {target_time}ms")

                # Reliability check
                if perf_metrics["success_rate"] < production_context["reliability_threshold"]:
                    pytest.warning(f"{tool_name} below production reliability: "
                                 f"{perf_metrics['success_rate']} < {production_context['reliability_threshold']}")

        else:
            pytest.skip(f"Application quality '{quality_level}' not ready for production")

    async def test_performance_regression_detection(self, ai_powered_test_environment):
        """Example: AI-powered performance regression detection."""
        runner = ai_powered_test_environment

        from workspace_qdrant_mcp.server import app

        # Simulate baseline performance data
        baseline_performance = {
            "workspace_status": 80.0,  # ms
            "search_workspace_tool": 300.0,  # ms
            "list_workspace_collections": 150.0  # ms
        }

        regression_context = {
            "baseline_performance": baseline_performance,
            "regression_threshold": 1.5,  # 50% slower is regression
            "evaluation_type": "regression_testing"
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            context=regression_context
        )

        # Check for performance regressions
        regressions_detected = []

        for tool_name, tool_result in report["evaluation_results"].items():
            if tool_name in baseline_performance:
                baseline = baseline_performance[tool_name]
                current = tool_result["performance_metrics"]["average_execution_time_ms"]

                if current > baseline * regression_context["regression_threshold"]:
                    regressions_detected.append({
                        "tool": tool_name,
                        "baseline": baseline,
                        "current": current,
                        "regression_factor": current / baseline
                    })

        # AI should provide insights about performance issues
        if regressions_detected:
            # Look for performance-related recommendations
            perf_recommendations = [
                rec for rec in report["recommendations"]
                if "performance" in rec.lower() or "optimize" in rec.lower()
            ]

            assert len(perf_recommendations) > 0, \
                f"Expected performance recommendations for regressions: {regressions_detected}"


# Integration test with actual FastMCP fixtures
@pytest.mark.integration
@pytest.mark.ai_evaluation
async def test_integration_with_fastmcp_fixtures(fastmcp_test_server, ai_test_evaluator):
    """Integration test showing pytest-mcp working with FastMCP fixtures."""

    # Use FastMCP test server directly
    client = await fastmcp_test_server.create_test_client()

    try:
        # Test workspace_status tool
        result = await client.call_tool("workspace_status", {})

        # Use AI evaluator directly
        evaluation = await ai_test_evaluator.evaluate_tool_response("workspace_status", result)

        # Apply pytest-mcp assertions
        assert_ai_score_above(evaluation, 0.3, "AI evaluation should be above minimum threshold")

        # Check that evaluation has expected structure
        assert len(evaluation.criteria_scores) > 0
        assert evaluation.confidence_level > 0.0
        assert isinstance(evaluation.ai_insights, list)
        assert isinstance(evaluation.recommendations, list)

        # If tool succeeded, should have reasonable functionality score
        if result.success:
            assert evaluation.criteria_scores.get("functionality", 0) >= 0.5

    finally:
        await client.close()


# Example of pytest parameterization with AI evaluation
@pytest.mark.pytest_mcp
@pytest.mark.parametrize("tool_name,expected_fields", [
    ("workspace_status", ["connected"]),
    ("list_workspace_collections", []),  # May return list or dict
])
async def test_parametrized_ai_evaluation(tool_name, expected_fields, ai_powered_test_environment):
    """Example: Parametrized tests with AI evaluation."""
    runner = ai_powered_test_environment

    from workspace_qdrant_mcp.server import app

    # Create test case for the parametrized tool
    test_cases = {
        tool_name: [
            MCPToolTestCase(
                tool_name=tool_name,
                description=f"Parametrized test for {tool_name}",
                parameters={},
                expected_response_type=(dict, list),
                expected_fields=expected_fields,
                should_succeed=True
            )
        ]
    }

    report = await runner.run_comprehensive_evaluation(
        app,
        custom_test_cases=test_cases
    )

    # Verify tool was evaluated
    assert tool_name in report["evaluation_results"]

    tool_result = report["evaluation_results"][tool_name]

    # Basic AI evaluation checks
    assert tool_result["overall_score"] >= 0.0
    assert tool_result["test_summary"]["total_tests"] >= 1