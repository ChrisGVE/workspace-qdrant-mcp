"""
Pytest-MCP Framework for AI-Powered Test Evaluation (Task 241.3).

This module provides a comprehensive AI-powered testing framework for MCP tool validation,
building on the FastMCP testing infrastructure to add intelligent evaluation capabilities.

Key Features:
    - AI-powered evaluation of MCP tool responses and behavior
    - Intelligent scoring mechanisms for tool functionality
    - Context-aware evaluation criteria with domain-specific knowledge
    - Integration with existing pytest infrastructure and FastMCP utilities
    - Comprehensive reporting with AI insights and recommendations
    - Automated test case generation based on tool signatures
    - Performance analysis with AI-driven optimization suggestions

Architecture:
    - AITestEvaluator: Core AI evaluation engine
    - MCPToolEvaluator: Tool-specific evaluation with AI insights
    - IntelligentTestRunner: Automated test execution with AI guidance
    - EvaluationCriteria: Configurable scoring and validation rules
    - AIInsightsReporter: Intelligent reporting and recommendations

Example:
    ```python
    async with ai_powered_mcp_testing(app) as evaluator:
        result = await evaluator.evaluate_tool("workspace_status", {})
        assert result.ai_score >= 0.8
        assert result.functionality_score >= 0.9
    ```
"""

import asyncio
import json
import logging
import re
import traceback
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

import pytest
from fastmcp import FastMCP
from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestClient,
    FastMCPTestServer,
    MCPTestResult,
    MCPToolTestCase,
    fastmcp_test_environment,
)

# Suppress logging during AI evaluation for clean output
logging.getLogger("pytest_mcp").setLevel(logging.WARNING)


class EvaluationCriteria(Enum):
    """AI evaluation criteria for MCP tool responses."""

    FUNCTIONALITY = "functionality"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    ERROR_HANDLING = "error_handling"


@dataclass
class AIEvaluationResult:
    """Result of AI-powered evaluation of MCP tool behavior."""

    # Core evaluation results
    overall_score: float  # 0.0 to 1.0
    criteria_scores: dict[EvaluationCriteria, float] = field(default_factory=dict)

    # AI insights and analysis
    ai_insights: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    detected_issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    # Detailed analysis
    response_quality: float | None = None
    error_analysis: str | None = None
    performance_analysis: str | None = None
    usability_assessment: str | None = None

    # Metadata
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluator_version: str = "1.0.0"
    confidence_level: float = 1.0  # AI confidence in evaluation


@dataclass
class MCPToolEvaluation:
    """Comprehensive evaluation result for a single MCP tool."""

    tool_name: str
    test_results: list[MCPTestResult]
    ai_evaluation: AIEvaluationResult

    # Performance metrics
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0

    # Protocol compliance
    protocol_compliance_score: float = 0.0
    format_validation_score: float = 0.0

    # AI-specific metrics
    functionality_score: float = 0.0
    user_experience_score: float = 0.0
    reliability_score: float = 0.0

    # Testing metadata
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0


class AITestEvaluator:
    """
    AI-powered evaluation engine for MCP tool testing.

    Provides intelligent evaluation of MCP tool responses using rule-based
    AI evaluation criteria and heuristics designed for MCP protocol compliance.
    """

    def __init__(self, domain_knowledge: dict[str, Any] | None = None):
        """
        Initialize AI test evaluator.

        Args:
            domain_knowledge: Optional domain-specific knowledge for evaluation
        """
        self.domain_knowledge = domain_knowledge or {}
        self.evaluation_history: list[AIEvaluationResult] = []

        # Initialize evaluation rules and criteria
        self._initialize_evaluation_criteria()

    def _initialize_evaluation_criteria(self) -> None:
        """Initialize AI evaluation criteria and scoring rules."""
        self.criteria_weights = {
            EvaluationCriteria.FUNCTIONALITY: 0.25,
            EvaluationCriteria.USABILITY: 0.15,
            EvaluationCriteria.PERFORMANCE: 0.15,
            EvaluationCriteria.RELIABILITY: 0.20,
            EvaluationCriteria.COMPLETENESS: 0.10,
            EvaluationCriteria.ACCURACY: 0.10,
            EvaluationCriteria.CONSISTENCY: 0.05,
        }

        # Tool-specific evaluation patterns
        self.tool_patterns = {
            "workspace_status": {
                "expected_fields": ["connected", "current_project", "collections"],
                "performance_threshold_ms": 100,
                "criticality": "high"
            },
            "search_workspace_tool": {
                "expected_fields": ["results", "total", "query"],
                "performance_threshold_ms": 500,
                "criticality": "critical"
            },
            "list_workspace_collections": {
                "expected_fields": [],
                "performance_threshold_ms": 200,
                "criticality": "medium"
            },
            "add_document_tool": {
                "expected_fields": ["success", "document_id"],
                "performance_threshold_ms": 1000,
                "criticality": "high"
            },
            "get_document_tool": {
                "expected_fields": ["content", "metadata"],
                "performance_threshold_ms": 300,
                "criticality": "high"
            }
        }

    async def evaluate_tool_response(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any] | None = None
    ) -> AIEvaluationResult:
        """
        Evaluate a tool response using AI-powered analysis.

        Args:
            tool_name: Name of the MCP tool being evaluated
            test_result: Test result to evaluate
            context: Optional context for evaluation

        Returns:
            AI evaluation result with scores and insights
        """
        context = context or {}

        # Initialize evaluation result
        evaluation = AIEvaluationResult(overall_score=0.0)

        try:
            # Evaluate each criteria
            criteria_scores = {}

            # Functionality evaluation
            functionality_score = await self._evaluate_functionality(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.FUNCTIONALITY] = functionality_score

            # Performance evaluation
            performance_score = await self._evaluate_performance(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.PERFORMANCE] = performance_score

            # Reliability evaluation
            reliability_score = await self._evaluate_reliability(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.RELIABILITY] = reliability_score

            # Usability evaluation
            usability_score = await self._evaluate_usability(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.USABILITY] = usability_score

            # Completeness evaluation
            completeness_score = await self._evaluate_completeness(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.COMPLETENESS] = completeness_score

            # Accuracy evaluation
            accuracy_score = await self._evaluate_accuracy(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.ACCURACY] = accuracy_score

            # Consistency evaluation
            consistency_score = await self._evaluate_consistency(tool_name, test_result, context)
            criteria_scores[EvaluationCriteria.CONSISTENCY] = consistency_score

            # Calculate weighted overall score
            overall_score = sum(
                score * self.criteria_weights.get(criteria, 0.0)
                for criteria, score in criteria_scores.items()
            )

            # Generate AI insights
            insights = await self._generate_insights(tool_name, test_result, criteria_scores, context)
            recommendations = await self._generate_recommendations(tool_name, test_result, criteria_scores, context)
            issues = await self._detect_issues(tool_name, test_result, criteria_scores, context)
            strengths = await self._identify_strengths(tool_name, test_result, criteria_scores, context)

            # Update evaluation result
            evaluation.overall_score = overall_score
            evaluation.criteria_scores = criteria_scores
            evaluation.ai_insights = insights
            evaluation.recommendations = recommendations
            evaluation.detected_issues = issues
            evaluation.strengths = strengths
            evaluation.confidence_level = self._calculate_confidence(test_result, criteria_scores)

            # Detailed analysis
            evaluation.response_quality = self._assess_response_quality(test_result)
            evaluation.error_analysis = self._analyze_errors(test_result) if not test_result.success else None
            evaluation.performance_analysis = self._analyze_performance(tool_name, test_result)
            evaluation.usability_assessment = self._assess_usability(tool_name, test_result)

        except Exception as e:
            evaluation.detected_issues.append(f"Evaluation error: {str(e)}")
            evaluation.overall_score = 0.0
            evaluation.confidence_level = 0.0

        # Store evaluation history
        self.evaluation_history.append(evaluation)

        return evaluation

    async def _evaluate_functionality(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate tool functionality based on response and expected behavior."""
        if not test_result.success:
            return 0.1  # Some points for graceful failure

        score = 0.7  # Base score for successful execution

        # Check tool-specific patterns
        tool_pattern = self.tool_patterns.get(tool_name, {})
        expected_fields = tool_pattern.get("expected_fields", [])

        if expected_fields and isinstance(test_result.response, dict):
            present_fields = sum(1 for field in expected_fields if field in test_result.response)
            field_score = present_fields / len(expected_fields) if expected_fields else 1.0
            score += 0.3 * field_score
        else:
            score += 0.2  # Partial credit if no specific expectations

        # Check response structure validity
        if test_result.protocol_compliance:
            compliance_bonus = sum(test_result.protocol_compliance.values()) / len(test_result.protocol_compliance)
            score = min(1.0, score + 0.1 * compliance_bonus)

        return min(1.0, score)

    async def _evaluate_performance(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate tool performance based on execution time and efficiency."""
        tool_pattern = self.tool_patterns.get(tool_name, {})
        threshold = tool_pattern.get("performance_threshold_ms", 1000)

        exec_time = test_result.execution_time_ms

        if exec_time <= threshold * 0.5:
            return 1.0  # Excellent performance
        elif exec_time <= threshold:
            return 0.8  # Good performance
        elif exec_time <= threshold * 2:
            return 0.6  # Acceptable performance
        elif exec_time <= threshold * 5:
            return 0.3  # Poor performance
        else:
            return 0.1  # Very poor performance

    async def _evaluate_reliability(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate tool reliability based on success rate and error handling."""
        if test_result.success:
            base_score = 0.8

            # Check for proper response structure
            if test_result.response is not None:
                base_score += 0.1

            # Check protocol compliance
            if test_result.protocol_compliance:
                compliance_rate = sum(test_result.protocol_compliance.values()) / len(test_result.protocol_compliance)
                base_score += 0.1 * compliance_rate

            return min(1.0, base_score)
        else:
            # Evaluate error handling quality
            if test_result.error and len(test_result.error) > 0:
                # Graceful error with message
                return 0.4
            else:
                # Poor error handling
                return 0.1

    async def _evaluate_usability(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate tool usability based on response clarity and structure."""
        if not test_result.success:
            # Check if error message is clear and helpful
            if test_result.error:
                if any(keyword in test_result.error.lower() for keyword in ["not found", "invalid", "missing"]):
                    return 0.6  # Clear error message
                else:
                    return 0.3  # Unclear error message
            return 0.1

        score = 0.6  # Base usability score

        # Check response structure
        if isinstance(test_result.response, dict):
            score += 0.2

            # Check for helpful fields
            helpful_fields = ["success", "message", "status", "error", "data"]
            present_helpful = sum(1 for field in helpful_fields if field in test_result.response)
            score += 0.2 * (present_helpful / len(helpful_fields))
        elif isinstance(test_result.response, list):
            score += 0.15  # Lists are generally usable

        return min(1.0, score)

    async def _evaluate_completeness(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate completeness of tool response."""
        if not test_result.success:
            return 0.2  # Incomplete due to failure

        tool_pattern = self.tool_patterns.get(tool_name, {})
        expected_fields = tool_pattern.get("expected_fields", [])

        if not expected_fields:
            return 0.8  # No specific expectations, assume reasonable completeness

        if isinstance(test_result.response, dict):
            present_fields = sum(1 for field in expected_fields if field in test_result.response)
            completeness = present_fields / len(expected_fields)
            return completeness

        return 0.5  # Partial completeness for non-dict responses

    async def _evaluate_accuracy(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate accuracy of tool response."""
        if not test_result.success:
            return 0.1

        # Basic accuracy checks
        score = 0.7  # Base accuracy score

        # Check JSON serialization (accuracy in format)
        if test_result.protocol_compliance and test_result.protocol_compliance.get("json_serializable", False):
            score += 0.2

        # Check for data type consistency
        if isinstance(test_result.response, (dict, list)):
            score += 0.1

        return min(1.0, score)

    async def _evaluate_consistency(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        context: dict[str, Any]
    ) -> float:
        """Evaluate consistency of tool behavior."""
        # For single test, evaluate internal consistency
        if not test_result.success:
            # Check if error handling is consistent
            if test_result.error:
                return 0.6  # Consistent error handling
            return 0.2

        # Check response format consistency
        score = 0.8  # Base consistency score

        if test_result.protocol_compliance:
            consistency_indicators = ["json_serializable", "is_dict_or_list", "has_content"]
            passed_checks = sum(1 for key in consistency_indicators
                              if test_result.protocol_compliance.get(key, False))
            score += 0.2 * (passed_checks / len(consistency_indicators))

        return min(1.0, score)

    async def _generate_insights(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        criteria_scores: dict[EvaluationCriteria, float],
        context: dict[str, Any]
    ) -> list[str]:
        """Generate AI insights about tool performance."""
        insights = []

        # Performance insights
        if criteria_scores.get(EvaluationCriteria.PERFORMANCE, 0) < 0.6:
            insights.append(f"Tool '{tool_name}' shows performance issues with {test_result.execution_time_ms:.1f}ms execution time")
        elif criteria_scores.get(EvaluationCriteria.PERFORMANCE, 0) > 0.9:
            insights.append(f"Tool '{tool_name}' demonstrates excellent performance at {test_result.execution_time_ms:.1f}ms")

        # Functionality insights
        if criteria_scores.get(EvaluationCriteria.FUNCTIONALITY, 0) < 0.7:
            insights.append(f"Tool '{tool_name}' functionality may be incomplete or not working as expected")

        # Reliability insights
        if not test_result.success:
            insights.append(f"Tool '{tool_name}' failed to execute successfully: {test_result.error}")
        elif criteria_scores.get(EvaluationCriteria.RELIABILITY, 0) > 0.8:
            insights.append(f"Tool '{tool_name}' demonstrates high reliability and proper error handling")

        # Protocol compliance insights
        if test_result.protocol_compliance:
            failed_checks = [k for k, v in test_result.protocol_compliance.items() if not v]
            if failed_checks:
                insights.append(f"Tool '{tool_name}' has protocol compliance issues: {', '.join(failed_checks)}")

        return insights

    async def _generate_recommendations(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        criteria_scores: dict[EvaluationCriteria, float],
        context: dict[str, Any]
    ) -> list[str]:
        """Generate AI recommendations for improvement."""
        recommendations = []

        # Performance recommendations
        if criteria_scores.get(EvaluationCriteria.PERFORMANCE, 0) < 0.6:
            recommendations.append(f"Optimize '{tool_name}' performance to reduce execution time below recommended threshold")

        # Functionality recommendations
        if criteria_scores.get(EvaluationCriteria.FUNCTIONALITY, 0) < 0.8:
            tool_pattern = self.tool_patterns.get(tool_name, {})
            expected_fields = tool_pattern.get("expected_fields", [])
            if expected_fields:
                recommendations.append(f"Ensure '{tool_name}' returns all expected fields: {', '.join(expected_fields)}")

        # Error handling recommendations
        if not test_result.success and not test_result.error:
            recommendations.append(f"Add proper error messaging to '{tool_name}' for better debugging")

        # Usability recommendations
        if criteria_scores.get(EvaluationCriteria.USABILITY, 0) < 0.7:
            recommendations.append(f"Improve '{tool_name}' response structure for better usability")

        return recommendations

    async def _detect_issues(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        criteria_scores: dict[EvaluationCriteria, float],
        context: dict[str, Any]
    ) -> list[str]:
        """Detect potential issues with tool behavior."""
        issues = []

        # Critical functionality issues
        if not test_result.success:
            issues.append(f"CRITICAL: Tool '{tool_name}' failed to execute")

        # Performance issues
        tool_pattern = self.tool_patterns.get(tool_name, {})
        threshold = tool_pattern.get("performance_threshold_ms", 1000)
        if test_result.execution_time_ms > threshold * 3:
            issues.append(f"PERFORMANCE: Tool '{tool_name}' exceeds performance threshold by 3x")

        # Protocol compliance issues
        if test_result.protocol_compliance:
            if not test_result.protocol_compliance.get("json_serializable", True):
                issues.append(f"PROTOCOL: Tool '{tool_name}' response is not JSON serializable")
            if not test_result.protocol_compliance.get("has_content", True):
                issues.append(f"PROTOCOL: Tool '{tool_name}' returned empty response")

        # Response format issues
        if test_result.success and test_result.response is None:
            issues.append(f"FORMAT: Tool '{tool_name}' successful but returned None response")

        return issues

    async def _identify_strengths(
        self,
        tool_name: str,
        test_result: MCPTestResult,
        criteria_scores: dict[EvaluationCriteria, float],
        context: dict[str, Any]
    ) -> list[str]:
        """Identify strengths in tool behavior."""
        strengths = []

        # Performance strengths
        if criteria_scores.get(EvaluationCriteria.PERFORMANCE, 0) > 0.9:
            strengths.append(f"Excellent performance: {test_result.execution_time_ms:.1f}ms execution time")

        # Reliability strengths
        if criteria_scores.get(EvaluationCriteria.RELIABILITY, 0) > 0.8:
            strengths.append("High reliability and proper error handling")

        # Protocol compliance strengths
        if test_result.protocol_compliance:
            passed_checks = sum(1 for v in test_result.protocol_compliance.values() if v)
            total_checks = len(test_result.protocol_compliance)
            if passed_checks == total_checks:
                strengths.append("Full MCP protocol compliance")

        # Functionality strengths
        if criteria_scores.get(EvaluationCriteria.FUNCTIONALITY, 0) > 0.8:
            strengths.append("Robust functionality implementation")

        return strengths

    def _calculate_confidence(
        self,
        test_result: MCPTestResult,
        criteria_scores: dict[EvaluationCriteria, float]
    ) -> float:
        """Calculate AI confidence in evaluation."""
        base_confidence = 0.8

        # Higher confidence for successful tests
        if test_result.success:
            base_confidence += 0.1

        # Higher confidence when we have protocol compliance data
        if test_result.protocol_compliance:
            base_confidence += 0.1

        # Lower confidence for very low or very high scores (potential outliers)
        avg_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0
        if avg_score < 0.2 or avg_score > 0.95:
            base_confidence -= 0.2

        return max(0.1, min(1.0, base_confidence))

    def _assess_response_quality(self, test_result: MCPTestResult) -> float:
        """Assess overall response quality."""
        if not test_result.success:
            return 0.2 if test_result.error else 0.1

        quality = 0.6  # Base quality

        if isinstance(test_result.response, dict):
            quality += 0.2
        elif isinstance(test_result.response, list):
            quality += 0.15

        if test_result.protocol_compliance:
            compliance_rate = sum(test_result.protocol_compliance.values()) / len(test_result.protocol_compliance)
            quality += 0.2 * compliance_rate

        return min(1.0, quality)

    def _analyze_errors(self, test_result: MCPTestResult) -> str:
        """Analyze error patterns and provide insights."""
        if not test_result.error:
            return "No error to analyze"

        error_msg = test_result.error.lower()

        if "not found" in error_msg:
            return "Error indicates missing resource or tool - check tool registration and parameters"
        elif "invalid" in error_msg:
            return "Error indicates invalid input - review parameter validation and types"
        elif "timeout" in error_msg:
            return "Error indicates performance issue - investigate execution time optimization"
        elif "connection" in error_msg:
            return "Error indicates connectivity issue - check service dependencies"
        else:
            return f"Generic error pattern detected: {test_result.error[:100]}..."

    def _analyze_performance(self, tool_name: str, test_result: MCPTestResult) -> str:
        """Analyze performance characteristics."""
        exec_time = test_result.execution_time_ms
        tool_pattern = self.tool_patterns.get(tool_name, {})
        threshold = tool_pattern.get("performance_threshold_ms", 1000)

        if exec_time <= threshold * 0.5:
            return f"Excellent performance: {exec_time:.1f}ms (well below {threshold}ms threshold)"
        elif exec_time <= threshold:
            return f"Good performance: {exec_time:.1f}ms (within {threshold}ms threshold)"
        elif exec_time <= threshold * 2:
            return f"Acceptable performance: {exec_time:.1f}ms (slightly above {threshold}ms threshold)"
        else:
            return f"Poor performance: {exec_time:.1f}ms (significantly above {threshold}ms threshold)"

    def _assess_usability(self, tool_name: str, test_result: MCPTestResult) -> str:
        """Assess usability characteristics."""
        if not test_result.success:
            if test_result.error:
                return f"Error message provided: '{test_result.error}' - helps with debugging"
            else:
                return "No error message provided - makes debugging difficult"

        if isinstance(test_result.response, dict):
            field_count = len(test_result.response)
            return f"Structured response with {field_count} fields - good for programmatic use"
        elif isinstance(test_result.response, list):
            item_count = len(test_result.response)
            return f"List response with {item_count} items - appropriate for collections"
        else:
            return f"Simple response type: {type(test_result.response).__name__} - may limit usability"


class MCPToolEvaluator:
    """
    Tool-specific evaluator that combines FastMCP testing with AI-powered analysis.

    Provides comprehensive evaluation of individual MCP tools including functionality,
    performance, reliability, and user experience assessment.
    """

    def __init__(self, ai_evaluator: AITestEvaluator):
        """
        Initialize MCP tool evaluator.

        Args:
            ai_evaluator: AI evaluation engine for intelligent analysis
        """
        self.ai_evaluator = ai_evaluator
        self.evaluation_results: dict[str, MCPToolEvaluation] = {}

    async def evaluate_tool(
        self,
        server: FastMCPTestServer,
        tool_name: str,
        test_cases: list[MCPToolTestCase],
        context: dict[str, Any] | None = None
    ) -> MCPToolEvaluation:
        """
        Evaluate a specific MCP tool using multiple test cases.

        Args:
            server: FastMCP test server
            tool_name: Name of the tool to evaluate
            test_cases: List of test cases to execute
            context: Optional evaluation context

        Returns:
            Comprehensive tool evaluation result
        """
        context = context or {}

        # Create test client
        client = await server.create_test_client()
        test_results = []

        try:
            # Execute all test cases
            for test_case in test_cases:
                result = await client.call_tool(test_case.tool_name, test_case.parameters)
                test_results.append(result)

                # Apply custom validation if provided
                if test_case.validation_fn and result.success:
                    try:
                        validation_passed = test_case.validation_fn(result.response)
                        if not validation_passed:
                            result.success = False
                            result.error = "Custom validation failed"
                    except Exception as e:
                        result.success = False
                        result.error = f"Validation error: {str(e)}"

            # If no specific test cases provided, create default ones
            if not test_cases:
                # Basic functionality test
                basic_result = await client.call_tool(tool_name, {})
                test_results.append(basic_result)

                # Error handling test
                error_result = await client.call_tool(tool_name, {"invalid_param": "test"})
                test_results.append(error_result)

            # Calculate basic metrics
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.success)
            failed_tests = total_tests - passed_tests
            success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            error_rate = failed_tests / total_tests if total_tests > 0 else 0.0

            # Calculate performance metrics
            successful_times = [r.execution_time_ms for r in test_results if r.success]
            avg_execution_time = sum(successful_times) / len(successful_times) if successful_times else 0.0

            # Calculate protocol compliance
            compliance_scores = []
            format_scores = []

            for result in test_results:
                if result.protocol_compliance:
                    compliance_score = sum(result.protocol_compliance.values()) / len(result.protocol_compliance)
                    compliance_scores.append(compliance_score)

                    format_score = result.protocol_compliance.get("json_serializable", 0.0)
                    format_scores.append(format_score)

            protocol_compliance_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0
            format_validation_score = sum(format_scores) / len(format_scores) if format_scores else 0.0

            # Get AI evaluation for overall tool behavior
            # Use the most representative test result for AI analysis
            representative_result = test_results[0] if test_results else None
            if len(test_results) > 1:
                # Prefer successful results for AI analysis
                successful_results = [r for r in test_results if r.success]
                if successful_results:
                    representative_result = successful_results[0]

            ai_evaluation = AIEvaluationResult(overall_score=0.0)
            if representative_result:
                ai_evaluation = await self.ai_evaluator.evaluate_tool_response(
                    tool_name, representative_result, context
                )

            # Calculate AI-specific scores
            functionality_score = ai_evaluation.criteria_scores.get(EvaluationCriteria.FUNCTIONALITY, 0.0)
            user_experience_score = (
                ai_evaluation.criteria_scores.get(EvaluationCriteria.USABILITY, 0.0) +
                ai_evaluation.criteria_scores.get(EvaluationCriteria.COMPLETENESS, 0.0)
            ) / 2
            reliability_score = ai_evaluation.criteria_scores.get(EvaluationCriteria.RELIABILITY, 0.0)

            # Create comprehensive evaluation
            evaluation = MCPToolEvaluation(
                tool_name=tool_name,
                test_results=test_results,
                ai_evaluation=ai_evaluation,
                average_execution_time=avg_execution_time,
                success_rate=success_rate,
                error_rate=error_rate,
                protocol_compliance_score=protocol_compliance_score,
                format_validation_score=format_validation_score,
                functionality_score=functionality_score,
                user_experience_score=user_experience_score,
                reliability_score=reliability_score,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests
            )

            # Store evaluation result
            self.evaluation_results[tool_name] = evaluation

            return evaluation

        finally:
            await client.close()

    async def evaluate_multiple_tools(
        self,
        server: FastMCPTestServer,
        tool_test_suite: dict[str, list[MCPToolTestCase]],
        context: dict[str, Any] | None = None
    ) -> dict[str, MCPToolEvaluation]:
        """
        Evaluate multiple MCP tools with their respective test suites.

        Args:
            server: FastMCP test server
            tool_test_suite: Dictionary mapping tool names to test cases
            context: Optional evaluation context

        Returns:
            Dictionary mapping tool names to evaluation results
        """
        results = {}

        for tool_name, test_cases in tool_test_suite.items():
            try:
                evaluation = await self.evaluate_tool(server, tool_name, test_cases, context)
                results[tool_name] = evaluation
            except Exception as e:
                # Create minimal evaluation for failed tools
                ai_evaluation = AIEvaluationResult(
                    overall_score=0.0,
                    detected_issues=[f"Evaluation failed: {str(e)}"]
                )
                results[tool_name] = MCPToolEvaluation(
                    tool_name=tool_name,
                    test_results=[],
                    ai_evaluation=ai_evaluation,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1
                )

        return results

    def get_evaluation_summary(self) -> dict[str, Any]:
        """Get summary of all tool evaluations."""
        if not self.evaluation_results:
            return {"message": "No evaluations performed"}

        total_tools = len(self.evaluation_results)
        overall_scores = [eval_result.ai_evaluation.overall_score for eval_result in self.evaluation_results.values()]
        avg_overall_score = sum(overall_scores) / len(overall_scores)

        functionality_scores = [eval_result.functionality_score for eval_result in self.evaluation_results.values()]
        avg_functionality = sum(functionality_scores) / len(functionality_scores)

        performance_times = [eval_result.average_execution_time for eval_result in self.evaluation_results.values()]
        avg_performance = sum(performance_times) / len(performance_times)

        success_rates = [eval_result.success_rate for eval_result in self.evaluation_results.values()]
        avg_success_rate = sum(success_rates) / len(success_rates)

        return {
            "total_tools_evaluated": total_tools,
            "average_overall_score": avg_overall_score,
            "average_functionality_score": avg_functionality,
            "average_execution_time_ms": avg_performance,
            "average_success_rate": avg_success_rate,
            "tool_rankings": sorted(
                [
                    {"tool": name, "score": eval_result.ai_evaluation.overall_score}
                    for name, eval_result in self.evaluation_results.items()
                ],
                key=lambda x: x["score"],
                reverse=True
            )
        }


class IntelligentTestRunner:
    """
    Intelligent test runner that coordinates FastMCP testing with AI evaluation.

    Provides automated test execution with AI-powered insights and recommendations.
    """

    def __init__(self, ai_evaluator: AITestEvaluator | None = None):
        """
        Initialize intelligent test runner.

        Args:
            ai_evaluator: Optional AI evaluator instance
        """
        self.ai_evaluator = ai_evaluator or AITestEvaluator()
        self.tool_evaluator = MCPToolEvaluator(self.ai_evaluator)
        self.test_history: list[dict[str, Any]] = []

    async def run_comprehensive_evaluation(
        self,
        app: FastMCP,
        custom_test_cases: dict[str, list[MCPToolTestCase]] | None = None,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Run comprehensive evaluation of all MCP tools in the application.

        Args:
            app: FastMCP application to evaluate
            custom_test_cases: Optional custom test cases for specific tools
            context: Optional evaluation context

        Returns:
            Comprehensive evaluation report
        """
        context = context or {}
        test_session_id = datetime.now(timezone.utc).isoformat()

        async with FastMCPTestServer(app, f"evaluation-{test_session_id[:10]}") as server:
            # Discover available tools
            available_tools = server.get_available_tools()

            # Generate default test cases if not provided
            if custom_test_cases is None:
                custom_test_cases = await self._generate_default_test_cases(available_tools)

            # Add any missing tools with basic test cases
            for tool_name in available_tools:
                if tool_name not in custom_test_cases:
                    custom_test_cases[tool_name] = [
                        MCPToolTestCase(
                            tool_name=tool_name,
                            description=f"Basic functionality test for {tool_name}",
                            parameters={},
                            expected_response_type=dict,
                            should_succeed=True
                        )
                    ]

            # Evaluate all tools
            evaluation_results = await self.tool_evaluator.evaluate_multiple_tools(
                server, custom_test_cases, context
            )

            # Generate comprehensive report
            report = {
                "session_id": test_session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "application_info": {
                    "total_tools": len(available_tools),
                    "evaluated_tools": len(evaluation_results),
                    "available_tools": available_tools
                },
                "evaluation_results": {
                    tool_name: {
                        "overall_score": eval_result.ai_evaluation.overall_score,
                        "functionality_score": eval_result.functionality_score,
                        "user_experience_score": eval_result.user_experience_score,
                        "reliability_score": eval_result.reliability_score,
                        "performance_metrics": {
                            "average_execution_time_ms": eval_result.average_execution_time,
                            "success_rate": eval_result.success_rate,
                            "error_rate": eval_result.error_rate
                        },
                        "ai_insights": eval_result.ai_evaluation.ai_insights,
                        "recommendations": eval_result.ai_evaluation.recommendations,
                        "detected_issues": eval_result.ai_evaluation.detected_issues,
                        "strengths": eval_result.ai_evaluation.strengths,
                        "test_summary": {
                            "total_tests": eval_result.total_tests,
                            "passed_tests": eval_result.passed_tests,
                            "failed_tests": eval_result.failed_tests
                        }
                    }
                    for tool_name, eval_result in evaluation_results.items()
                },
                "summary": self.tool_evaluator.get_evaluation_summary(),
                "recommendations": await self._generate_overall_recommendations(evaluation_results),
                "quality_assessment": await self._assess_overall_quality(evaluation_results)
            }

            # Store test history
            self.test_history.append(report)

            return report

    async def _generate_default_test_cases(self, available_tools: list[str]) -> dict[str, list[MCPToolTestCase]]:
        """Generate default test cases for available tools."""
        test_cases = {}

        for tool_name in available_tools:
            cases = []

            # Basic functionality test
            cases.append(MCPToolTestCase(
                tool_name=tool_name,
                description=f"Basic functionality test for {tool_name}",
                parameters={},
                expected_response_type=dict,
                should_succeed=True
            ))

            # Tool-specific test cases
            if "search" in tool_name:
                cases.append(MCPToolTestCase(
                    tool_name=tool_name,
                    description=f"Search with query test for {tool_name}",
                    parameters={"query": "test", "limit": 5},
                    expected_response_type=dict,
                    expected_fields=["results", "total"],
                    should_succeed=True
                ))

                cases.append(MCPToolTestCase(
                    tool_name=tool_name,
                    description=f"Empty search test for {tool_name}",
                    parameters={"query": ""},
                    expected_response_type=dict,
                    should_succeed=False  # Empty query should fail or return empty results
                ))

            elif "add" in tool_name or "store" in tool_name:
                cases.append(MCPToolTestCase(
                    tool_name=tool_name,
                    description=f"Add document test for {tool_name}",
                    parameters={"content": "Test document content", "collection": "test"},
                    expected_response_type=dict,
                    expected_fields=["success"],
                    should_succeed=True
                ))

            elif "get" in tool_name or "retrieve" in tool_name:
                cases.append(MCPToolTestCase(
                    tool_name=tool_name,
                    description=f"Get document test for {tool_name}",
                    parameters={"document_id": "test_id"},
                    expected_response_type=dict,
                    should_succeed=False  # Likely to fail without actual document
                ))

            # Error handling test for all tools
            cases.append(MCPToolTestCase(
                tool_name=tool_name,
                description=f"Error handling test for {tool_name}",
                parameters={"invalid_parameter": "test_value"},
                expected_response_type=dict,
                should_succeed=False  # Should fail gracefully
            ))

            test_cases[tool_name] = cases

        return test_cases

    async def _generate_overall_recommendations(
        self,
        evaluation_results: dict[str, MCPToolEvaluation]
    ) -> list[str]:
        """Generate overall recommendations for the MCP application."""
        recommendations = []

        # Performance recommendations
        slow_tools = [
            name for name, eval_result in evaluation_results.items()
            if eval_result.average_execution_time > 500
        ]
        if slow_tools:
            recommendations.append(f"Optimize performance for slow tools: {', '.join(slow_tools)}")

        # Reliability recommendations
        unreliable_tools = [
            name for name, eval_result in evaluation_results.items()
            if eval_result.success_rate < 0.8
        ]
        if unreliable_tools:
            recommendations.append(f"Improve reliability for tools: {', '.join(unreliable_tools)}")

        # Functionality recommendations
        low_functionality_tools = [
            name for name, eval_result in evaluation_results.items()
            if eval_result.functionality_score < 0.7
        ]
        if low_functionality_tools:
            recommendations.append(f"Enhance functionality for tools: {', '.join(low_functionality_tools)}")

        # Overall recommendations
        overall_scores = [eval_result.ai_evaluation.overall_score for eval_result in evaluation_results.values()]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        if avg_score < 0.6:
            recommendations.append("Overall application quality is below acceptable threshold - consider comprehensive review")
        elif avg_score > 0.8:
            recommendations.append("Application demonstrates high quality - focus on performance optimization and advanced features")

        return recommendations

    async def _assess_overall_quality(
        self,
        evaluation_results: dict[str, MCPToolEvaluation]
    ) -> dict[str, Any]:
        """Assess overall quality of the MCP application."""
        if not evaluation_results:
            return {"status": "no_data", "assessment": "No evaluation data available"}

        # Calculate aggregate metrics
        total_tools = len(evaluation_results)
        overall_scores = [eval_result.ai_evaluation.overall_score for eval_result in evaluation_results.values()]
        functionality_scores = [eval_result.functionality_score for eval_result in evaluation_results.values()]
        success_rates = [eval_result.success_rate for eval_result in evaluation_results.values()]
        avg_execution_times = [eval_result.average_execution_time for eval_result in evaluation_results.values()]

        avg_overall = sum(overall_scores) / len(overall_scores)
        avg_functionality = sum(functionality_scores) / len(functionality_scores)
        avg_success_rate = sum(success_rates) / len(success_rates)
        avg_execution_time = sum(avg_execution_times) / len(avg_execution_times)

        # Determine quality level
        if avg_overall >= 0.8 and avg_success_rate >= 0.9:
            quality_level = "excellent"
        elif avg_overall >= 0.7 and avg_success_rate >= 0.8:
            quality_level = "good"
        elif avg_overall >= 0.5 and avg_success_rate >= 0.6:
            quality_level = "acceptable"
        else:
            quality_level = "poor"

        # Count issues
        total_issues = sum(
            len(eval_result.ai_evaluation.detected_issues)
            for eval_result in evaluation_results.values()
        )

        return {
            "status": "completed",
            "quality_level": quality_level,
            "metrics": {
                "average_overall_score": avg_overall,
                "average_functionality_score": avg_functionality,
                "average_success_rate": avg_success_rate,
                "average_execution_time_ms": avg_execution_time,
                "total_tools": total_tools,
                "total_detected_issues": total_issues
            },
            "assessment": f"Application quality is {quality_level} with {total_tools} tools evaluated"
        }


@asynccontextmanager
async def ai_powered_mcp_testing(
    app: FastMCP,
    name: str = "ai-test-env",
    ai_evaluator: AITestEvaluator | None = None
) -> AsyncGenerator[IntelligentTestRunner, None]:
    """
    Async context manager for AI-powered MCP testing environment.

    Provides a complete AI-powered testing environment with intelligent evaluation,
    insights generation, and comprehensive reporting capabilities.

    Args:
        app: FastMCP application instance
        name: Environment name for identification
        ai_evaluator: Optional custom AI evaluator

    Yields:
        IntelligentTestRunner ready for AI-powered evaluation

    Example:
        ```python
        async with ai_powered_mcp_testing(app) as runner:
            report = await runner.run_comprehensive_evaluation(app)
            assert report["summary"]["average_overall_score"] >= 0.8
        ```
    """
    ai_eval = ai_evaluator or AITestEvaluator()
    runner = IntelligentTestRunner(ai_eval)

    try:
        yield runner
    finally:
        # Cleanup if needed
        pass


# Pytest integration utilities
class PytestMCPAssertions:
    """Pytest-style assertions for MCP testing with AI evaluation."""

    @staticmethod
    def assert_ai_score_above(evaluation: AIEvaluationResult, threshold: float, message: str = ""):
        """Assert that AI evaluation score is above threshold."""
        msg = message or f"AI score {evaluation.overall_score} is not above threshold {threshold}"
        assert evaluation.overall_score >= threshold, msg

    @staticmethod
    def assert_tool_functional(evaluation: MCPToolEvaluation, threshold: float = 0.7, message: str = ""):
        """Assert that tool functionality meets threshold."""
        msg = message or f"Tool functionality score {evaluation.functionality_score} is below threshold {threshold}"
        assert evaluation.functionality_score >= threshold, msg

    @staticmethod
    def assert_performance_acceptable(evaluation: MCPToolEvaluation, max_time_ms: float = 1000, message: str = ""):
        """Assert that tool performance is acceptable."""
        msg = message or f"Tool execution time {evaluation.average_execution_time}ms exceeds maximum {max_time_ms}ms"
        assert evaluation.average_execution_time <= max_time_ms, msg

    @staticmethod
    def assert_reliability_high(evaluation: MCPToolEvaluation, min_success_rate: float = 0.8, message: str = ""):
        """Assert that tool reliability meets minimum success rate."""
        msg = message or f"Tool success rate {evaluation.success_rate} is below minimum {min_success_rate}"
        assert evaluation.success_rate >= min_success_rate, msg

    @staticmethod
    def assert_no_critical_issues(evaluation: AIEvaluationResult, message: str = ""):
        """Assert that no critical issues were detected."""
        critical_issues = [issue for issue in evaluation.detected_issues if "CRITICAL" in issue.upper()]
        msg = message or f"Critical issues detected: {critical_issues}"
        assert len(critical_issues) == 0, msg


# Make assertions available at module level
assert_ai_score_above = PytestMCPAssertions.assert_ai_score_above
assert_tool_functional = PytestMCPAssertions.assert_tool_functional
assert_performance_acceptable = PytestMCPAssertions.assert_performance_acceptable
assert_reliability_high = PytestMCPAssertions.assert_reliability_high
assert_no_critical_issues = PytestMCPAssertions.assert_no_critical_issues
