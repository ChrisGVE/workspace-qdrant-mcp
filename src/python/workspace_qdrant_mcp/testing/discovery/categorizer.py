"""
Test Categorizer

Categorizes tests and identifies coverage gaps to suggest new test cases.
Provides intelligent analysis of test patterns and missing scenarios.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import logging

from ..documentation.parser import TestFileInfo, TestMetadata, TestType

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Categories for test classification."""
    UNIT_CORE = "unit_core"
    UNIT_EDGE_CASE = "unit_edge_case"
    UNIT_ERROR_HANDLING = "unit_error_handling"
    INTEGRATION_API = "integration_api"
    INTEGRATION_DATABASE = "integration_database"
    INTEGRATION_EXTERNAL = "integration_external"
    FUNCTIONAL_WORKFLOW = "functional_workflow"
    FUNCTIONAL_USER_JOURNEY = "functional_user_journey"
    PERFORMANCE_LOAD = "performance_load"
    PERFORMANCE_STRESS = "performance_stress"
    SECURITY_AUTH = "security_auth"
    SECURITY_VALIDATION = "security_validation"
    REGRESSION_BUG_FIX = "regression_bug_fix"
    REGRESSION_COMPATIBILITY = "regression_compatibility"
    UI_COMPONENT = "ui_component"
    UI_INTERACTION = "ui_interaction"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    UNKNOWN = "unknown"


@dataclass
class CoverageGap:
    """Represents a gap in test coverage."""
    category: TestCategory
    description: str
    priority: int  # 1-10, higher is more important
    suggested_tests: List[str] = field(default_factory=list)
    affected_modules: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical
    effort_estimate: str = "medium"  # low, medium, high


@dataclass
class TestPattern:
    """Pattern detected in test code."""
    pattern_type: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class CoverageAnalysis:
    """Complete analysis of test coverage and patterns."""
    total_tests: int
    categories: Dict[TestCategory, int] = field(default_factory=dict)
    patterns: List[TestPattern] = field(default_factory=list)
    gaps: List[CoverageGap] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    coverage_score: float = 0.0
    completeness_score: float = 0.0


class TestCategorizer:
    """
    Categorizes tests and identifies coverage gaps.

    Analyzes test patterns, detects missing coverage areas,
    and provides recommendations for improving test suites.
    """

    def __init__(self):
        """Initialize test categorizer with pattern matching rules."""
        self._category_patterns = self._build_category_patterns()
        self._gap_detection_rules = self._build_gap_detection_rules()

    def categorize_tests(self, file_infos: List[TestFileInfo]) -> CoverageAnalysis:
        """
        Categorize all tests and analyze coverage gaps.

        Args:
            file_infos: List of parsed test file information

        Returns:
            Complete coverage analysis
        """
        analysis = CoverageAnalysis(total_tests=0)

        # Collect all tests
        all_tests = []
        for file_info in file_infos:
            all_tests.extend(file_info.tests)

        analysis.total_tests = len(all_tests)

        if not all_tests:
            analysis.gaps.append(CoverageGap(
                category=TestCategory.UNKNOWN,
                description="No tests found in project",
                priority=10,
                risk_level="critical",
                suggested_tests=["Create basic unit tests for core functionality"]
            ))
            return analysis

        # Categorize individual tests
        for test in all_tests:
            category = self._categorize_single_test(test)
            analysis.categories[category] = analysis.categories.get(category, 0) + 1

        # Detect patterns
        analysis.patterns = self._detect_patterns(all_tests)

        # Identify gaps
        analysis.gaps = self._identify_gaps(analysis, file_infos)

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        # Calculate scores
        analysis.coverage_score = self._calculate_coverage_score(analysis)
        analysis.completeness_score = self._calculate_completeness_score(analysis)

        return analysis

    def _categorize_single_test(self, test: TestMetadata) -> TestCategory:
        """Categorize a single test based on patterns."""
        # Check name patterns first
        test_name = test.name.lower()
        docstring = (test.docstring or "").lower()
        combined_text = f"{test_name} {docstring}"

        # Score each category
        category_scores = {}
        for category, patterns in self._category_patterns.items():
            score = 0
            for pattern, weight in patterns.items():
                if re.search(pattern, combined_text, re.IGNORECASE):
                    score += weight

            if score > 0:
                category_scores[category] = score

        # Return highest scoring category
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        # Fallback based on test type
        if test.test_type == TestType.UNIT:
            return TestCategory.UNIT_CORE
        elif test.test_type == TestType.INTEGRATION:
            return TestCategory.INTEGRATION_API
        elif test.test_type == TestType.FUNCTIONAL:
            return TestCategory.FUNCTIONAL_WORKFLOW
        elif test.test_type == TestType.PERFORMANCE:
            return TestCategory.PERFORMANCE_LOAD
        else:
            return TestCategory.UNKNOWN

    def _detect_patterns(self, tests: List[TestMetadata]) -> List[TestPattern]:
        """Detect common patterns in test code."""
        patterns = []

        # Pattern: Mock usage
        mock_count = sum(1 for test in tests if 'mock' in (test.docstring or "").lower() or
                        any('mock' in d.name.lower() for d in test.decorators))
        if mock_count > 0:
            patterns.append(TestPattern(
                pattern_type="mock_usage",
                frequency=mock_count,
                confidence=0.9,
                examples=[t.name for t in tests if 'mock' in (t.docstring or "").lower()][:3]
            ))

        # Pattern: Exception testing
        exception_count = sum(1 for test in tests if
                            'exception' in (test.docstring or "").lower() or
                            'error' in (test.docstring or "").lower() or
                            'raise' in (test.docstring or "").lower())
        if exception_count > 0:
            patterns.append(TestPattern(
                pattern_type="exception_testing",
                frequency=exception_count,
                confidence=0.8,
                examples=[t.name for t in tests if 'exception' in (t.docstring or "").lower()][:3]
            ))

        # Pattern: Parametrized tests
        param_count = sum(1 for test in tests if test.is_parametrized)
        if param_count > 0:
            patterns.append(TestPattern(
                pattern_type="parametrized_tests",
                frequency=param_count,
                confidence=1.0,
                examples=[t.name for t in tests if t.is_parametrized][:3]
            ))

        # Pattern: Async testing
        async_count = sum(1 for test in tests if test.is_async)
        if async_count > 0:
            patterns.append(TestPattern(
                pattern_type="async_testing",
                frequency=async_count,
                confidence=1.0,
                examples=[t.name for t in tests if t.is_async][:3]
            ))

        # Pattern: Fixture usage
        fixture_pattern_count = sum(1 for test in tests if 'fixture' in (test.docstring or "").lower())
        if fixture_pattern_count > 0:
            patterns.append(TestPattern(
                pattern_type="fixture_usage",
                frequency=fixture_pattern_count,
                confidence=0.7,
                examples=[t.name for t in tests if 'fixture' in (t.docstring or "").lower()][:3]
            ))

        return patterns

    def _identify_gaps(self, analysis: CoverageAnalysis, file_infos: List[TestFileInfo]) -> List[CoverageGap]:
        """Identify coverage gaps based on analysis."""
        gaps = []

        # Check for missing categories
        missing_categories = []
        essential_categories = [
            TestCategory.UNIT_CORE,
            TestCategory.UNIT_ERROR_HANDLING,
            TestCategory.INTEGRATION_API
        ]

        for category in essential_categories:
            if category not in analysis.categories or analysis.categories[category] == 0:
                missing_categories.append(category)

        for category in missing_categories:
            gap = CoverageGap(
                category=category,
                description=f"Missing {category.value} tests",
                priority=8 if category == TestCategory.UNIT_CORE else 6,
                risk_level="high" if category == TestCategory.UNIT_CORE else "medium",
                suggested_tests=self._suggest_tests_for_category(category)
            )
            gaps.append(gap)

        # Check for edge case coverage
        edge_case_count = analysis.categories.get(TestCategory.UNIT_EDGE_CASE, 0)
        core_count = analysis.categories.get(TestCategory.UNIT_CORE, 0)

        if core_count > 0 and edge_case_count < core_count * 0.3:  # Less than 30% edge cases
            gaps.append(CoverageGap(
                category=TestCategory.UNIT_EDGE_CASE,
                description="Insufficient edge case testing",
                priority=7,
                risk_level="medium",
                suggested_tests=[
                    "Test with null/empty inputs",
                    "Test boundary values",
                    "Test invalid input handling",
                    "Test concurrent access scenarios"
                ]
            ))

        # Check for error handling coverage
        error_count = analysis.categories.get(TestCategory.UNIT_ERROR_HANDLING, 0)
        if error_count < max(1, core_count * 0.2):  # Less than 20% error handling
            gaps.append(CoverageGap(
                category=TestCategory.UNIT_ERROR_HANDLING,
                description="Insufficient error handling tests",
                priority=8,
                risk_level="high",
                suggested_tests=[
                    "Test exception scenarios",
                    "Test error recovery",
                    "Test timeout handling",
                    "Test resource cleanup on errors"
                ]
            ))

        # Check for integration testing
        integration_count = sum(analysis.categories.get(cat, 0) for cat in [
            TestCategory.INTEGRATION_API,
            TestCategory.INTEGRATION_DATABASE,
            TestCategory.INTEGRATION_EXTERNAL
        ])

        if integration_count == 0 and analysis.total_tests > 5:
            gaps.append(CoverageGap(
                category=TestCategory.INTEGRATION_API,
                description="Missing integration tests",
                priority=6,
                risk_level="medium",
                suggested_tests=[
                    "Test API endpoint integration",
                    "Test database connectivity",
                    "Test external service integration"
                ]
            ))

        # Check for performance testing
        perf_count = sum(analysis.categories.get(cat, 0) for cat in [
            TestCategory.PERFORMANCE_LOAD,
            TestCategory.PERFORMANCE_STRESS
        ])

        if perf_count == 0 and analysis.total_tests > 10:
            gaps.append(CoverageGap(
                category=TestCategory.PERFORMANCE_LOAD,
                description="Missing performance tests",
                priority=5,
                risk_level="low",
                effort_estimate="high",
                suggested_tests=[
                    "Test response time under normal load",
                    "Test memory usage patterns",
                    "Test concurrent user scenarios"
                ]
            ))

        # Check for security testing
        security_count = sum(analysis.categories.get(cat, 0) for cat in [
            TestCategory.SECURITY_AUTH,
            TestCategory.SECURITY_VALIDATION
        ])

        if security_count == 0 and self._has_security_concerns(file_infos):
            gaps.append(CoverageGap(
                category=TestCategory.SECURITY_VALIDATION,
                description="Missing security validation tests",
                priority=9,
                risk_level="high",
                suggested_tests=[
                    "Test input validation and sanitization",
                    "Test authentication and authorization",
                    "Test SQL injection prevention",
                    "Test XSS protection"
                ]
            ))

        return gaps

    def _has_security_concerns(self, file_infos: List[TestFileInfo]) -> bool:
        """Check if codebase has security-related functionality."""
        security_indicators = ['auth', 'login', 'password', 'token', 'session', 'validate', 'sanitize']

        for file_info in file_infos:
            file_content = str(file_info.file_path).lower()
            for test in file_info.tests:
                test_content = f"{test.name} {test.docstring or ''}".lower()
                if any(indicator in file_content or indicator in test_content
                      for indicator in security_indicators):
                    return True
        return False

    def _suggest_tests_for_category(self, category: TestCategory) -> List[str]:
        """Suggest specific tests for a category."""
        suggestions = {
            TestCategory.UNIT_CORE: [
                "Test core functionality with valid inputs",
                "Test method return values",
                "Test state changes after operations"
            ],
            TestCategory.UNIT_ERROR_HANDLING: [
                "Test with invalid inputs",
                "Test exception handling",
                "Test error messages and codes"
            ],
            TestCategory.INTEGRATION_API: [
                "Test API endpoints with valid requests",
                "Test API error responses",
                "Test API authentication"
            ],
            TestCategory.PERFORMANCE_LOAD: [
                "Test response times under load",
                "Test memory usage patterns",
                "Test concurrent access handling"
            ],
            TestCategory.SECURITY_VALIDATION: [
                "Test input validation",
                "Test authentication mechanisms",
                "Test authorization checks"
            ]
        }
        return suggestions.get(category, ["Add appropriate tests for this category"])

    def _generate_recommendations(self, analysis: CoverageAnalysis) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Coverage recommendations
        if analysis.coverage_score < 0.7:
            recommendations.append("Increase overall test coverage - aim for 80% minimum")

        # Pattern-based recommendations
        async_pattern = next((p for p in analysis.patterns if p.pattern_type == "async_testing"), None)
        if async_pattern and async_pattern.frequency < analysis.total_tests * 0.1:
            recommendations.append("Consider adding more async/await test scenarios")

        mock_pattern = next((p for p in analysis.patterns if p.pattern_type == "mock_usage"), None)
        if not mock_pattern or mock_pattern.frequency < 3:
            recommendations.append("Use mocking to isolate units under test")

        param_pattern = next((p for p in analysis.patterns if p.pattern_type == "parametrized_tests"), None)
        if not param_pattern or param_pattern.frequency < analysis.total_tests * 0.2:
            recommendations.append("Use parametrized tests to reduce code duplication")

        # Gap-based recommendations
        high_priority_gaps = [g for g in analysis.gaps if g.priority >= 8]
        if high_priority_gaps:
            recommendations.append("Address high-priority coverage gaps immediately")

        # Complexity recommendations
        if analysis.completeness_score < 0.6:
            recommendations.append("Improve test comprehensiveness with edge cases and error scenarios")

        return recommendations

    def _calculate_coverage_score(self, analysis: CoverageAnalysis) -> float:
        """Calculate coverage adequacy score (0-1)."""
        if analysis.total_tests == 0:
            return 0.0

        # Base score from test count (diminishing returns)
        base_score = min(1.0, analysis.total_tests / 50.0)

        # Diversity bonus
        diversity_score = len(analysis.categories) / len(TestCategory)

        # Pattern bonus
        pattern_score = min(1.0, len(analysis.patterns) / 5.0)

        # Gap penalty
        gap_penalty = min(0.5, len([g for g in analysis.gaps if g.priority >= 7]) * 0.1)

        return max(0.0, (base_score * 0.5 + diversity_score * 0.3 + pattern_score * 0.2) - gap_penalty)

    def _calculate_completeness_score(self, analysis: CoverageAnalysis) -> float:
        """Calculate test completeness score (0-1)."""
        if analysis.total_tests == 0:
            return 0.0

        # Essential categories coverage
        essential_coverage = 0
        essential_categories = [
            TestCategory.UNIT_CORE,
            TestCategory.UNIT_ERROR_HANDLING,
            TestCategory.UNIT_EDGE_CASE
        ]

        for category in essential_categories:
            if category in analysis.categories and analysis.categories[category] > 0:
                essential_coverage += 1

        essential_score = essential_coverage / len(essential_categories)

        # Breadth score (how many different types of tests)
        breadth_score = len(analysis.categories) / max(1, analysis.total_tests / 5)
        breadth_score = min(1.0, breadth_score)

        # High-priority gap penalty
        critical_gaps = len([g for g in analysis.gaps if g.priority >= 8])
        gap_penalty = min(0.4, critical_gaps * 0.1)

        return max(0.0, essential_score * 0.6 + breadth_score * 0.4 - gap_penalty)

    def _build_category_patterns(self) -> Dict[TestCategory, Dict[str, int]]:
        """Build pattern matching rules for test categorization."""
        return {
            TestCategory.UNIT_CORE: {
                r'\btest.*core\b': 3,
                r'\btest.*basic\b': 2,
                r'\btest.*simple\b': 2,
                r'\btest.*main\b': 2,
                r'\btest.*primary\b': 2,
                r'\breturn\b': 1,
                r'\bassert\b': 1
            },
            TestCategory.UNIT_EDGE_CASE: {
                r'\bedge\b': 5,
                r'\bboundary\b': 5,
                r'\blimit\b': 4,
                r'\bmax\b': 3,
                r'\bmin\b': 3,
                r'\bempty\b': 3,
                r'\bnull\b': 3,
                r'\bzero\b': 2,
                r'\bnegative\b': 2
            },
            TestCategory.UNIT_ERROR_HANDLING: {
                r'\berror\b': 5,
                r'\bexception\b': 5,
                r'\braise\b': 4,
                r'\bfail\b': 3,
                r'\binvalid\b': 3,
                r'\bbad\b': 2,
                r'\bwrong\b': 2,
                r'with.*raises': 4
            },
            TestCategory.INTEGRATION_API: {
                r'\bapi\b': 5,
                r'\bendpoint\b': 4,
                r'\brequest\b': 3,
                r'\bresponse\b': 3,
                r'\bhttp\b': 3,
                r'\brest\b': 3,
                r'\bpost\b': 2,
                r'\bget\b': 2
            },
            TestCategory.INTEGRATION_DATABASE: {
                r'\bdatabase\b': 5,
                r'\bdb\b': 4,
                r'\bsql\b': 4,
                r'\bquery\b': 3,
                r'\btable\b': 3,
                r'\binsert\b': 3,
                r'\bselect\b': 3,
                r'\bupdate\b': 3
            },
            TestCategory.INTEGRATION_EXTERNAL: {
                r'\bexternal\b': 5,
                r'\bservice\b': 4,
                r'\bthird.?party\b': 4,
                r'\bclient\b': 3,
                r'\bremote\b': 3,
                r'\bapi.*call\b': 3
            },
            TestCategory.FUNCTIONAL_WORKFLOW: {
                r'\bworkflow\b': 5,
                r'\bprocess\b': 4,
                r'\bflow\b': 3,
                r'\bstep\b': 3,
                r'\bsequence\b': 3,
                r'\bpipeline\b': 3
            },
            TestCategory.PERFORMANCE_LOAD: {
                r'\bperformance\b': 5,
                r'\bload\b': 5,
                r'\bspeed\b': 4,
                r'\btime\b': 3,
                r'\bbenchmark\b': 4,
                r'\bfast\b': 2,
                r'\bslow\b': 2
            },
            TestCategory.PERFORMANCE_STRESS: {
                r'\bstress\b': 5,
                r'\bconcurrent\b': 4,
                r'\bparallel\b': 4,
                r'\bheavy\b': 3,
                r'\bmemory\b': 3,
                r'\bresource\b': 3
            },
            TestCategory.SECURITY_AUTH: {
                r'\bauth\b': 5,
                r'\blogin\b': 4,
                r'\bpassword\b': 4,
                r'\btoken\b': 4,
                r'\bsession\b': 3,
                r'\bpermission\b': 3
            },
            TestCategory.SECURITY_VALIDATION: {
                r'\bvalidat\b': 5,
                r'\bsanitiz\b': 4,
                r'\bsecurity\b': 4,
                r'\binjection\b': 5,
                r'\bxss\b': 5,
                r'\binput.*clean\b': 3
            }
        }

    def _build_gap_detection_rules(self) -> Dict[str, Any]:
        """Build rules for gap detection."""
        return {
            'min_edge_case_ratio': 0.3,
            'min_error_handling_ratio': 0.2,
            'essential_categories': [
                TestCategory.UNIT_CORE,
                TestCategory.UNIT_ERROR_HANDLING
            ],
            'security_indicators': [
                'auth', 'login', 'password', 'token', 'session',
                'validate', 'sanitize', 'input', 'user'
            ]
        }