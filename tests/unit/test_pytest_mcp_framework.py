"""
Unit tests for pytest-mcp Framework AI-Powered Evaluation (Task 241.3).

This module tests the pytest-mcp framework components to ensure reliable
AI-powered evaluation capabilities for MCP tool testing.

Validates:
    - AITestEvaluator evaluation criteria and scoring
    - MCPToolEvaluator comprehensive tool evaluation
    - IntelligentTestRunner automated test execution
    - AI insights generation and recommendations
    - Performance analysis and issue detection
    - Integration with FastMCP testing infrastructure
"""

import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from tests.utils.pytest_mcp_framework import (
    AITestEvaluator,
    MCPToolEvaluator,
    IntelligentTestRunner,
    AIEvaluationResult,
    MCPToolEvaluation,
    MCPToolTestCase,
    EvaluationCriteria,
    ai_powered_mcp_testing,
    assert_ai_score_above,
    assert_tool_functional,
    assert_performance_acceptable,
    assert_reliability_high,
    assert_no_critical_issues
)
from tests.utils.fastmcp_test_infrastructure import (
    MCPTestResult,
    FastMCPTestServer,
    FastMCPTestClient
)


class TestAITestEvaluator:
    """Test AI test evaluator functionality."""

    @pytest.mark.ai_evaluation
    async def test_ai_evaluator_initialization(self):
        """Test AI evaluator initialization with default settings."""
        evaluator = AITestEvaluator()

        assert evaluator.domain_knowledge == {}
        assert evaluator.evaluation_history == []
        assert len(evaluator.criteria_weights) > 0
        assert EvaluationCriteria.FUNCTIONALITY in evaluator.criteria_weights
        assert EvaluationCriteria.PERFORMANCE in evaluator.criteria_weights

        # Verify weights sum to approximately 1.0
        total_weight = sum(evaluator.criteria_weights.values())
        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.ai_evaluation
    async def test_ai_evaluator_with_domain_knowledge(self):
        """Test AI evaluator initialization with custom domain knowledge."""
        domain_knowledge = {
            "workspace_tools": ["workspace_status", "search_workspace_tool"],
            "performance_thresholds": {"search": 500, "status": 100}
        }

        evaluator = AITestEvaluator(domain_knowledge)
        assert evaluator.domain_knowledge == domain_knowledge

    @pytest.mark.ai_evaluation
    async def test_evaluate_successful_tool_response(self):
        """Test AI evaluation of successful tool response."""
        evaluator = AITestEvaluator()

        # Create successful test result
        test_result = MCPTestResult(
            success=True,
            tool_name="workspace_status",
            parameters={},
            response={"connected": True, "current_project": "test-project", "collections": ["test_collection"]},
            execution_time_ms=50.0,
            protocol_compliance={
                "json_serializable": True,
                "is_dict_or_list": True,
                "has_content": True,
                "error_format_valid": True
            }
        )

        evaluation = await evaluator.evaluate_tool_response("workspace_status", test_result)

        # Verify evaluation structure
        assert isinstance(evaluation, AIEvaluationResult)
        assert 0.0 <= evaluation.overall_score <= 1.0
        assert len(evaluation.criteria_scores) > 0
        assert evaluation.confidence_level > 0.0

        # Verify criteria scores
        assert EvaluationCriteria.FUNCTIONALITY in evaluation.criteria_scores
        assert EvaluationCriteria.PERFORMANCE in evaluation.criteria_scores
        assert EvaluationCriteria.RELIABILITY in evaluation.criteria_scores

        # Should have high scores for successful result
        assert evaluation.criteria_scores[EvaluationCriteria.FUNCTIONALITY] >= 0.7
        assert evaluation.criteria_scores[EvaluationCriteria.RELIABILITY] >= 0.8

    @pytest.mark.ai_evaluation
    async def test_evaluate_failed_tool_response(self):
        """Test AI evaluation of failed tool response."""
        evaluator = AITestEvaluator()

        # Create failed test result
        test_result = MCPTestResult(
            success=False,
            tool_name="nonexistent_tool",
            parameters={},
            response=None,
            execution_time_ms=0.0,
            error="Tool 'nonexistent_tool' not found"
        )

        evaluation = await evaluator.evaluate_tool_response("nonexistent_tool", test_result)

        # Verify evaluation structure
        assert isinstance(evaluation, AIEvaluationResult)
        assert evaluation.overall_score < 0.5  # Should be low for failed tool
        assert len(evaluation.detected_issues) > 0

        # Should have low functionality score
        assert evaluation.criteria_scores[EvaluationCriteria.FUNCTIONALITY] < 0.5

        # Should have some reliability points for error message
        assert evaluation.criteria_scores[EvaluationCriteria.RELIABILITY] > 0.0

    @pytest.mark.ai_evaluation
    async def test_evaluate_performance_scenarios(self):
        """Test AI evaluation of different performance scenarios."""
        evaluator = AITestEvaluator()

        # Fast performance test
        fast_result = MCPTestResult(
            success=True,
            tool_name="workspace_status",
            parameters={},
            response={"status": "ok"},
            execution_time_ms=25.0,
            protocol_compliance={"json_serializable": True}
        )

        fast_evaluation = await evaluator.evaluate_tool_response("workspace_status", fast_result)
        fast_performance = fast_evaluation.criteria_scores[EvaluationCriteria.PERFORMANCE]

        # Slow performance test
        slow_result = MCPTestResult(
            success=True,
            tool_name="workspace_status",
            parameters={},
            response={"status": "ok"},
            execution_time_ms=2000.0,  # 2 seconds
            protocol_compliance={"json_serializable": True}
        )

        slow_evaluation = await evaluator.evaluate_tool_response("workspace_status", slow_result)
        slow_performance = slow_evaluation.criteria_scores[EvaluationCriteria.PERFORMANCE]

        # Fast should score higher than slow
        assert fast_performance > slow_performance
        assert fast_performance >= 0.8  # Should be excellent
        assert slow_performance <= 0.4  # Should be poor

    @pytest.mark.ai_evaluation
    async def test_generate_insights_and_recommendations(self):
        """Test AI insights and recommendations generation."""
        evaluator = AITestEvaluator()

        # Create test result with performance issue
        slow_result = MCPTestResult(
            success=True,
            tool_name="search_workspace_tool",
            parameters={"query": "test"},
            response={"results": []},
            execution_time_ms=1500.0,  # Slow
            protocol_compliance={"json_serializable": True}
        )

        evaluation = await evaluator.evaluate_tool_response("search_workspace_tool", slow_result)

        # Should have insights about performance
        performance_insights = [insight for insight in evaluation.ai_insights if "performance" in insight.lower()]
        assert len(performance_insights) > 0

        # Should have recommendations for optimization
        optimization_recommendations = [rec for rec in evaluation.recommendations if "optimize" in rec.lower()]
        assert len(optimization_recommendations) > 0

    @pytest.mark.ai_evaluation
    async def test_evaluation_history_tracking(self):
        """Test evaluation history tracking."""
        evaluator = AITestEvaluator()

        # Perform multiple evaluations
        test_results = [
            MCPTestResult(
                success=True,
                tool_name="tool1",
                parameters={},
                response={"data": "test"},
                execution_time_ms=100.0
            ),
            MCPTestResult(
                success=False,
                tool_name="tool2",
                parameters={},
                response=None,
                execution_time_ms=0.0,
                error="Test error"
            )
        ]

        for i, result in enumerate(test_results):
            await evaluator.evaluate_tool_response(f"tool{i+1}", result)

        # Verify history tracking
        assert len(evaluator.evaluation_history) == 2
        assert all(isinstance(eval_result, AIEvaluationResult) for eval_result in evaluator.evaluation_history)


class TestMCPToolEvaluator:
    """Test MCP tool evaluator functionality."""

    @pytest.mark.ai_evaluation
    async def test_mcp_tool_evaluator_initialization(self, ai_test_evaluator):
        """Test MCP tool evaluator initialization."""
        evaluator = MCPToolEvaluator(ai_test_evaluator)

        assert evaluator.ai_evaluator == ai_test_evaluator
        assert evaluator.evaluation_results == {}

    @pytest.mark.ai_evaluation
    async def test_evaluate_single_tool_with_test_cases(self, fastmcp_test_server, ai_test_evaluator):
        """Test evaluation of single tool with custom test cases."""
        evaluator = MCPToolEvaluator(ai_test_evaluator)

        # Create test cases for workspace_status tool
        test_cases = [
            MCPToolTestCase(
                tool_name="workspace_status",
                description="Basic status check",
                parameters={},
                expected_response_type=dict,
                should_succeed=True
            ),
            MCPToolTestCase(
                tool_name="workspace_status",
                description="Status with invalid parameter",
                parameters={"invalid": "param"},
                expected_response_type=dict,
                should_succeed=False  # May succeed but should handle gracefully
            )
        ]

        evaluation = await evaluator.evaluate_tool("workspace_status", test_cases)

        # Verify evaluation structure
        assert isinstance(evaluation, MCPToolEvaluation)
        assert evaluation.tool_name == "workspace_status"
        assert len(evaluation.test_results) == 2
        assert isinstance(evaluation.ai_evaluation, AIEvaluationResult)

        # Verify metrics calculation
        assert evaluation.total_tests == 2
        assert evaluation.passed_tests >= 0
        assert evaluation.failed_tests >= 0
        assert evaluation.passed_tests + evaluation.failed_tests == evaluation.total_tests
        assert 0.0 <= evaluation.success_rate <= 1.0
        assert evaluation.average_execution_time >= 0.0

    @pytest.mark.ai_evaluation
    async def test_evaluate_tool_with_default_test_cases(self, fastmcp_test_server, ai_test_evaluator):
        """Test evaluation of tool with default generated test cases."""
        evaluator = MCPToolEvaluator(ai_test_evaluator)

        # Evaluate without providing test cases - should generate defaults
        evaluation = await evaluator.evaluate_tool("workspace_status", [])

        # Should have generated default test cases
        assert evaluation.total_tests >= 2  # Basic + error handling tests
        assert len(evaluation.test_results) >= 2

    @pytest.mark.ai_evaluation
    async def test_evaluate_multiple_tools(self, fastmcp_test_server, ai_test_evaluator):
        """Test evaluation of multiple tools."""
        evaluator = MCPToolEvaluator(ai_test_evaluator)

        # Create test suite for multiple tools
        tool_test_suite = {
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
            ]
        }

        results = await evaluator.evaluate_multiple_tools(tool_test_suite)

        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "workspace_status" in results
        assert "list_workspace_collections" in results

        for tool_name, evaluation in results.items():
            assert isinstance(evaluation, MCPToolEvaluation)
            assert evaluation.tool_name == tool_name

    @pytest.mark.ai_evaluation
    async def test_evaluation_summary_generation(self, ai_test_evaluator):
        """Test evaluation summary generation."""
        evaluator = MCPToolEvaluator(ai_test_evaluator)

        # Mock some evaluation results
        mock_ai_eval = AIEvaluationResult(overall_score=0.8)
        mock_tool_eval = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=mock_ai_eval,
            functionality_score=0.85,
            average_execution_time=150.0,
            success_rate=0.9
        )

        evaluator.evaluation_results["test_tool"] = mock_tool_eval

        summary = evaluator.get_evaluation_summary()

        # Verify summary structure
        assert "total_tools_evaluated" in summary
        assert "average_overall_score" in summary
        assert "average_functionality_score" in summary
        assert "average_execution_time_ms" in summary
        assert "average_success_rate" in summary
        assert "tool_rankings" in summary

        assert summary["total_tools_evaluated"] == 1
        assert summary["average_overall_score"] == 0.8


class TestIntelligentTestRunner:
    """Test intelligent test runner functionality."""

    @pytest.mark.intelligent_testing
    async def test_intelligent_test_runner_initialization(self):
        """Test intelligent test runner initialization."""
        runner = IntelligentTestRunner()

        assert isinstance(runner.ai_evaluator, AITestEvaluator)
        assert isinstance(runner.tool_evaluator, MCPToolEvaluator)
        assert runner.test_history == []

    @pytest.mark.intelligent_testing
    async def test_intelligent_test_runner_with_custom_evaluator(self):
        """Test intelligent test runner with custom AI evaluator."""
        custom_evaluator = AITestEvaluator({"custom": "knowledge"})
        runner = IntelligentTestRunner(custom_evaluator)

        assert runner.ai_evaluator == custom_evaluator
        assert runner.ai_evaluator.domain_knowledge == {"custom": "knowledge"}

    @pytest.mark.intelligent_testing
    async def test_generate_default_test_cases(self):
        """Test default test case generation."""
        runner = IntelligentTestRunner()

        available_tools = ["workspace_status", "search_workspace_tool", "add_document_tool"]
        test_cases = await runner._generate_default_test_cases(available_tools)

        # Verify test cases generated for all tools
        assert len(test_cases) == 3
        assert all(tool in test_cases for tool in available_tools)

        # Verify search tool has specific test cases
        search_cases = test_cases["search_workspace_tool"]
        assert len(search_cases) >= 3  # Basic, search with query, empty search, error handling

        # Verify all tools have error handling test
        for tool_name, cases in test_cases.items():
            error_cases = [case for case in cases if "error" in case.description.lower()]
            assert len(error_cases) >= 1

    @pytest.mark.intelligent_testing
    async def test_comprehensive_evaluation_execution(self, fastmcp_test_server):
        """Test comprehensive evaluation execution."""
        from workspace_qdrant_mcp.server import app

        runner = IntelligentTestRunner()

        # Run comprehensive evaluation
        report = await runner.run_comprehensive_evaluation(app)

        # Verify report structure
        assert "session_id" in report
        assert "timestamp" in report
        assert "application_info" in report
        assert "evaluation_results" in report
        assert "summary" in report
        assert "recommendations" in report
        assert "quality_assessment" in report

        # Verify application info
        app_info = report["application_info"]
        assert "total_tools" in app_info
        assert "evaluated_tools" in app_info
        assert "available_tools" in app_info

        # Verify evaluation results structure
        eval_results = report["evaluation_results"]
        assert isinstance(eval_results, dict)

        for tool_name, tool_result in eval_results.items():
            assert "overall_score" in tool_result
            assert "functionality_score" in tool_result
            assert "performance_metrics" in tool_result
            assert "ai_insights" in tool_result
            assert "recommendations" in tool_result

    @pytest.mark.intelligent_testing
    async def test_overall_recommendations_generation(self):
        """Test overall recommendations generation."""
        runner = IntelligentTestRunner()

        # Mock evaluation results with various issues
        mock_evaluations = {
            "slow_tool": MCPToolEvaluation(
                tool_name="slow_tool",
                test_results=[],
                ai_evaluation=AIEvaluationResult(overall_score=0.6),
                average_execution_time=1500.0,  # Slow
                success_rate=0.9,
                functionality_score=0.8
            ),
            "unreliable_tool": MCPToolEvaluation(
                tool_name="unreliable_tool",
                test_results=[],
                ai_evaluation=AIEvaluationResult(overall_score=0.5),
                average_execution_time=200.0,
                success_rate=0.6,  # Unreliable
                functionality_score=0.4  # Low functionality
            )
        }

        recommendations = await runner._generate_overall_recommendations(mock_evaluations)

        # Should identify performance issues
        perf_recommendations = [rec for rec in recommendations if "performance" in rec.lower() or "slow" in rec.lower()]
        assert len(perf_recommendations) > 0

        # Should identify reliability issues
        reliability_recommendations = [rec for rec in recommendations if "reliability" in rec.lower() or "unreliable" in rec.lower()]
        assert len(reliability_recommendations) > 0

    @pytest.mark.intelligent_testing
    async def test_overall_quality_assessment(self):
        """Test overall quality assessment."""
        runner = IntelligentTestRunner()

        # Test high quality scenario
        high_quality_evaluations = {
            "excellent_tool": MCPToolEvaluation(
                tool_name="excellent_tool",
                test_results=[],
                ai_evaluation=AIEvaluationResult(overall_score=0.95, detected_issues=[]),
                functionality_score=0.9,
                success_rate=0.95,
                average_execution_time=100.0
            )
        }

        quality_assessment = await runner._assess_overall_quality(high_quality_evaluations)
        assert quality_assessment["quality_level"] == "excellent"
        assert quality_assessment["metrics"]["total_detected_issues"] == 0

        # Test poor quality scenario
        poor_quality_evaluations = {
            "poor_tool": MCPToolEvaluation(
                tool_name="poor_tool",
                test_results=[],
                ai_evaluation=AIEvaluationResult(overall_score=0.3, detected_issues=["Critical issue"]),
                functionality_score=0.2,
                success_rate=0.4,
                average_execution_time=2000.0
            )
        }

        quality_assessment = await runner._assess_overall_quality(poor_quality_evaluations)
        assert quality_assessment["quality_level"] == "poor"
        assert quality_assessment["metrics"]["total_detected_issues"] >= 1


class TestAIPoweredMCPTesting:
    """Test AI-powered MCP testing context manager."""

    @pytest.mark.pytest_mcp
    async def test_ai_powered_mcp_testing_context_manager(self):
        """Test AI-powered MCP testing context manager."""
        from workspace_qdrant_mcp.server import app

        async with ai_powered_mcp_testing(app, "test-context") as runner:
            assert isinstance(runner, IntelligentTestRunner)
            assert isinstance(runner.ai_evaluator, AITestEvaluator)
            assert isinstance(runner.tool_evaluator, MCPToolEvaluator)

    @pytest.mark.pytest_mcp
    async def test_ai_powered_testing_with_custom_evaluator(self):
        """Test AI-powered testing with custom evaluator."""
        from workspace_qdrant_mcp.server import app

        custom_evaluator = AITestEvaluator({"domain": "custom"})

        async with ai_powered_mcp_testing(app, "custom-test", custom_evaluator) as runner:
            assert runner.ai_evaluator == custom_evaluator
            assert runner.ai_evaluator.domain_knowledge == {"domain": "custom"}


class TestPytestMCPAssertions:
    """Test pytest-mcp assertion functions."""

    @pytest.mark.pytest_mcp
    def test_assert_ai_score_above_success(self):
        """Test successful AI score assertion."""
        evaluation = AIEvaluationResult(overall_score=0.85)

        # Should not raise
        assert_ai_score_above(evaluation, 0.8)
        assert_ai_score_above(evaluation, 0.85)

    @pytest.mark.pytest_mcp
    def test_assert_ai_score_above_failure(self):
        """Test failed AI score assertion."""
        evaluation = AIEvaluationResult(overall_score=0.7)

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            assert_ai_score_above(evaluation, 0.8)

    @pytest.mark.pytest_mcp
    def test_assert_tool_functional_success(self):
        """Test successful tool functionality assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.8),
            functionality_score=0.85
        )

        # Should not raise
        assert_tool_functional(evaluation, 0.8)

    @pytest.mark.pytest_mcp
    def test_assert_tool_functional_failure(self):
        """Test failed tool functionality assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.6),
            functionality_score=0.6
        )

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            assert_tool_functional(evaluation, 0.8)

    @pytest.mark.pytest_mcp
    def test_assert_performance_acceptable_success(self):
        """Test successful performance assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.8),
            average_execution_time=500.0
        )

        # Should not raise
        assert_performance_acceptable(evaluation, 1000.0)

    @pytest.mark.pytest_mcp
    def test_assert_performance_acceptable_failure(self):
        """Test failed performance assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.8),
            average_execution_time=1500.0
        )

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            assert_performance_acceptable(evaluation, 1000.0)

    @pytest.mark.pytest_mcp
    def test_assert_reliability_high_success(self):
        """Test successful reliability assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.8),
            success_rate=0.9
        )

        # Should not raise
        assert_reliability_high(evaluation, 0.8)

    @pytest.mark.pytest_mcp
    def test_assert_reliability_high_failure(self):
        """Test failed reliability assertion."""
        evaluation = MCPToolEvaluation(
            tool_name="test_tool",
            test_results=[],
            ai_evaluation=AIEvaluationResult(overall_score=0.8),
            success_rate=0.7
        )

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            assert_reliability_high(evaluation, 0.8)

    @pytest.mark.pytest_mcp
    def test_assert_no_critical_issues_success(self):
        """Test successful no critical issues assertion."""
        evaluation = AIEvaluationResult(
            overall_score=0.8,
            detected_issues=["Minor issue", "Performance concern"]
        )

        # Should not raise
        assert_no_critical_issues(evaluation)

    @pytest.mark.pytest_mcp
    def test_assert_no_critical_issues_failure(self):
        """Test failed no critical issues assertion."""
        evaluation = AIEvaluationResult(
            overall_score=0.5,
            detected_issues=["CRITICAL: Tool failure", "Minor issue"]
        )

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            assert_no_critical_issues(evaluation)


@pytest.mark.ai_evaluation
class TestIntegrationWithFastMCP:
    """Test integration between pytest-mcp framework and FastMCP infrastructure."""

    async def test_end_to_end_ai_evaluation(self, ai_powered_test_environment):
        """Test end-to-end AI evaluation using existing FastMCP infrastructure."""
        runner = ai_powered_test_environment

        # Import the actual MCP app
        from workspace_qdrant_mcp.server import app

        # Run a focused evaluation on workspace_status tool
        custom_test_cases = {
            "workspace_status": [
                MCPToolTestCase(
                    tool_name="workspace_status",
                    description="Basic workspace status check",
                    parameters={},
                    expected_response_type=dict,
                    should_succeed=True
                )
            ]
        }

        report = await runner.run_comprehensive_evaluation(
            app,
            custom_test_cases=custom_test_cases
        )

        # Verify AI evaluation was performed
        assert "evaluation_results" in report
        if "workspace_status" in report["evaluation_results"]:
            status_result = report["evaluation_results"]["workspace_status"]
            assert "overall_score" in status_result
            assert "ai_insights" in status_result
            assert "recommendations" in status_result

        # Verify quality assessment
        assert "quality_assessment" in report
        assert report["quality_assessment"]["status"] == "completed"

    async def test_ai_evaluation_with_fastmcp_fixtures(self, fastmcp_test_server, ai_test_evaluator):
        """Test AI evaluation using FastMCP test fixtures."""
        # Create test client
        client = await fastmcp_test_server.create_test_client()

        try:
            # Call a tool
            result = await client.call_tool("workspace_status", {})

            # Evaluate with AI
            evaluation = await ai_test_evaluator.evaluate_tool_response("workspace_status", result)

            # Verify AI evaluation
            assert isinstance(evaluation, AIEvaluationResult)
            assert 0.0 <= evaluation.overall_score <= 1.0
            assert len(evaluation.criteria_scores) > 0

            # Verify evaluation makes sense for the result
            if result.success:
                assert evaluation.criteria_scores[EvaluationCriteria.FUNCTIONALITY] > 0.5
            else:
                assert evaluation.criteria_scores[EvaluationCriteria.FUNCTIONALITY] < 0.5

        finally:
            await client.close()