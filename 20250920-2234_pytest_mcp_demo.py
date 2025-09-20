#!/usr/bin/env python3
"""
Demonstration of pytest-mcp framework AI-powered evaluation capabilities.

This script shows how the pytest-mcp framework can be used to evaluate MCP tools
with AI-powered insights and recommendations.
"""

import asyncio
from tests.utils.pytest_mcp_framework import (
    AITestEvaluator,
    MCPToolEvaluator,
    IntelligentTestRunner,
    MCPToolTestCase,
    assert_ai_score_above,
    assert_tool_functional,
    assert_performance_acceptable
)
from tests.utils.fastmcp_test_infrastructure import MCPTestResult
from tests.pytest_mcp_config import DEFAULT_PYTEST_MCP_CONFIG, PytestMCPPresets


async def demo_ai_evaluator():
    """Demonstrate AI evaluation capabilities."""
    print("=== AI Evaluator Demo ===")

    evaluator = AITestEvaluator()

    # Simulate a successful tool response
    successful_result = MCPTestResult(
        success=True,
        tool_name="workspace_status",
        parameters={},
        response={
            "connected": True,
            "current_project": "workspace-qdrant-mcp",
            "collections": ["project_docs", "project_scratchbook"],
            "qdrant_url": "http://localhost:6333"
        },
        execution_time_ms=75.0,
        protocol_compliance={
            "json_serializable": True,
            "is_dict_or_list": True,
            "has_content": True,
            "error_format_valid": True
        }
    )

    evaluation = await evaluator.evaluate_tool_response("workspace_status", successful_result)

    print(f"Overall AI Score: {evaluation.overall_score:.3f}")
    print(f"Confidence Level: {evaluation.confidence_level:.3f}")
    print(f"Criteria Scores:")
    for criteria, score in evaluation.criteria_scores.items():
        print(f"  {criteria.value}: {score:.3f}")

    print(f"\nAI Insights ({len(evaluation.ai_insights)}):")
    for insight in evaluation.ai_insights:
        print(f"  • {insight}")

    print(f"\nRecommendations ({len(evaluation.recommendations)}):")
    for rec in evaluation.recommendations:
        print(f"  • {rec}")

    print(f"\nStrengths ({len(evaluation.strengths)}):")
    for strength in evaluation.strengths:
        print(f"  • {strength}")

    if evaluation.detected_issues:
        print(f"\nDetected Issues ({len(evaluation.detected_issues)}):")
        for issue in evaluation.detected_issues:
            print(f"  ⚠ {issue}")

    # Test AI assertions
    try:
        assert_ai_score_above(evaluation, 0.7)
        print("\n✓ AI score assertion passed")
    except AssertionError as e:
        print(f"\n✗ AI score assertion failed: {e}")


async def demo_failed_tool_evaluation():
    """Demonstrate evaluation of a failed tool."""
    print("\n=== Failed Tool Evaluation Demo ===")

    evaluator = AITestEvaluator()

    # Simulate a failed tool response
    failed_result = MCPTestResult(
        success=False,
        tool_name="nonexistent_tool",
        parameters={"invalid": "params"},
        response=None,
        execution_time_ms=0.0,
        error="Tool 'nonexistent_tool' not found in registered tools"
    )

    evaluation = await evaluator.evaluate_tool_response("nonexistent_tool", failed_result)

    print(f"Overall AI Score: {evaluation.overall_score:.3f}")
    print(f"Error Analysis: {evaluation.error_analysis}")

    print(f"\nDetected Issues ({len(evaluation.detected_issues)}):")
    for issue in evaluation.detected_issues:
        print(f"  ⚠ {issue}")

    print(f"\nAI Insights:")
    for insight in evaluation.ai_insights:
        print(f"  • {insight}")


async def demo_performance_analysis():
    """Demonstrate performance analysis capabilities."""
    print("\n=== Performance Analysis Demo ===")

    evaluator = AITestEvaluator()

    # Simulate slow tool response
    slow_result = MCPTestResult(
        success=True,
        tool_name="search_workspace_tool",
        parameters={"query": "test", "limit": 10},
        response={
            "results": [],
            "total": 0,
            "query": "test",
            "execution_time": "1.5s"
        },
        execution_time_ms=1500.0,  # Slow
        protocol_compliance={"json_serializable": True}
    )

    evaluation = await evaluator.evaluate_tool_response("search_workspace_tool", slow_result)

    print(f"Performance Score: {evaluation.criteria_scores.get('performance', 0):.3f}")
    print(f"Performance Analysis: {evaluation.performance_analysis}")

    # Check for performance insights
    perf_insights = [insight for insight in evaluation.ai_insights if "performance" in insight.lower()]
    if perf_insights:
        print(f"\nPerformance Insights:")
        for insight in perf_insights:
            print(f"  • {insight}")

    # Check for optimization recommendations
    perf_recs = [rec for rec in evaluation.recommendations if "optimize" in rec.lower()]
    if perf_recs:
        print(f"\nOptimization Recommendations:")
        for rec in perf_recs:
            print(f"  • {rec}")


def demo_configuration():
    """Demonstrate configuration system."""
    print("\n=== Configuration System Demo ===")

    # Default configuration
    config = DEFAULT_PYTEST_MCP_CONFIG
    print(f"Default AI evaluation enabled: {config.ai_evaluation_enabled}")
    print(f"workspace_status performance threshold: {config.get_tool_performance_threshold('workspace_status')}ms")
    print(f"workspace_status reliability threshold: {config.get_tool_reliability_threshold('workspace_status')}")

    # Production configuration
    prod_config = PytestMCPPresets.production_validation_config()
    print(f"\nProduction reliability threshold: {prod_config.get_tool_reliability_threshold('workspace_status')}")
    print(f"Production performance threshold: {prod_config.get_tool_performance_threshold('workspace_status')}ms")

    # Performance testing configuration
    perf_config = PytestMCPPresets.performance_testing_config()
    print(f"\nPerformance testing threshold: {perf_config.get_tool_performance_threshold('workspace_status')}ms")
    print(f"Performance criteria weight: {perf_config.criteria_weights.get('performance', 0)}")


async def demo_test_case_generation():
    """Demonstrate intelligent test case generation."""
    print("\n=== Test Case Generation Demo ===")

    runner = IntelligentTestRunner()

    # Simulate available tools
    available_tools = [
        "workspace_status",
        "search_workspace_tool",
        "add_document_tool",
        "list_workspace_collections"
    ]

    test_cases = await runner._generate_default_test_cases(available_tools)

    print(f"Generated test cases for {len(test_cases)} tools:")
    for tool_name, cases in test_cases.items():
        print(f"\n{tool_name} ({len(cases)} test cases):")
        for case in cases:
            print(f"  • {case.description}")
            if case.parameters:
                print(f"    Parameters: {case.parameters}")
            print(f"    Should succeed: {case.should_succeed}")


async def demo_assertion_functions():
    """Demonstrate pytest-mcp assertion functions."""
    print("\n=== Assertion Functions Demo ===")

    # Create mock evaluation results for testing
    from tests.utils.pytest_mcp_framework import AIEvaluationResult, MCPToolEvaluation

    good_ai_eval = AIEvaluationResult(overall_score=0.85)
    good_tool_eval = MCPToolEvaluation(
        tool_name="test_tool",
        test_results=[],
        ai_evaluation=good_ai_eval,
        functionality_score=0.9,
        average_execution_time=150.0,
        success_rate=0.95
    )

    print("Testing assertions with good evaluation:")
    try:
        assert_ai_score_above(good_ai_eval, 0.8)
        print("✓ AI score assertion passed")

        assert_tool_functional(good_tool_eval, 0.8)
        print("✓ Tool functionality assertion passed")

        assert_performance_acceptable(good_tool_eval, 500.0)
        print("✓ Performance assertion passed")

    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")

    # Test with poor evaluation
    poor_ai_eval = AIEvaluationResult(overall_score=0.4)
    poor_tool_eval = MCPToolEvaluation(
        tool_name="poor_tool",
        test_results=[],
        ai_evaluation=poor_ai_eval,
        functionality_score=0.3,
        average_execution_time=2000.0,
        success_rate=0.6
    )

    print("\nTesting assertions with poor evaluation (should fail):")
    try:
        assert_ai_score_above(poor_ai_eval, 0.8)
        print("✗ AI score assertion unexpectedly passed")
    except AssertionError:
        print("✓ AI score assertion correctly failed")


async def main():
    """Run all demonstration functions."""
    print("pytest-mcp Framework AI-Powered Evaluation Demo")
    print("=" * 50)

    await demo_ai_evaluator()
    await demo_failed_tool_evaluation()
    await demo_performance_analysis()
    demo_configuration()
    await demo_test_case_generation()
    await demo_assertion_functions()

    print("\n" + "=" * 50)
    print("pytest-mcp Framework Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• AI-powered evaluation with intelligent scoring")
    print("• Performance analysis and optimization recommendations")
    print("• Error detection and issue identification")
    print("• Configurable evaluation criteria and thresholds")
    print("• Automatic test case generation")
    print("• pytest-style assertion functions")
    print("• Integration with FastMCP infrastructure")


if __name__ == "__main__":
    asyncio.run(main())