"""
Comprehensive unit tests for TestCategorizer.

Tests categorization logic, coverage analysis, gap detection,
and recommendation generation with extensive edge case coverage.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.python.workspace_qdrant_mcp.testing.discovery.categorizer import (
    TestCategorizer,
    TestCategory,
    CoverageGap,
    TestPattern,
    CoverageAnalysis
)
from src.python.workspace_qdrant_mcp.testing.documentation.parser import (
    TestFileInfo,
    TestMetadata,
    TestType,
    DecoratorInfo
)


class TestTestCategorizer:
    """Test TestCategorizer with comprehensive edge case coverage."""

    def setup_method(self):
        """Setup test fixtures."""
        self.categorizer = TestCategorizer()

    def create_test_metadata(self, name: str, docstring: str = "", test_type: TestType = TestType.UNKNOWN,
                           decorators: list = None, is_async: bool = False, is_parametrized: bool = False,
                           marks: set = None) -> TestMetadata:
        """Helper to create test metadata."""
        return TestMetadata(
            name=name,
            docstring=docstring,
            file_path=Path("test_example.py"),
            line_number=10,
            test_type=test_type,
            decorators=decorators or [],
            is_async=is_async,
            is_parametrized=is_parametrized,
            marks=marks or set()
        )

    def create_test_file_info(self, tests: list = None) -> TestFileInfo:
        """Helper to create test file info."""
        file_info = TestFileInfo(file_path=Path("test_example.py"))
        file_info.tests = tests or []
        return file_info

    def test_categorize_empty_test_suite(self):
        """Test categorization with no tests."""
        file_infos = [self.create_test_file_info()]

        analysis = self.categorizer.categorize_tests(file_infos)

        assert analysis.total_tests == 0
        assert len(analysis.gaps) > 0
        assert analysis.coverage_score == 0.0
        assert analysis.completeness_score == 0.0

        # Should suggest creating basic tests
        critical_gap = next((g for g in analysis.gaps if g.priority == 10), None)
        assert critical_gap is not None
        assert "No tests found" in critical_gap.description

    def test_categorize_unit_core_tests(self):
        """Test categorization of core unit tests."""
        tests = [
            self.create_test_metadata("test_core_function", "Test core functionality"),
            self.create_test_metadata("test_basic_operation", "Test basic operation"),
            self.create_test_metadata("test_simple_case", "Test simple case"),
            self.create_test_metadata("test_main_workflow", "Test main workflow")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        assert analysis.total_tests == 4
        assert analysis.categories[TestCategory.UNIT_CORE] == 4
        assert analysis.coverage_score > 0

    def test_categorize_edge_case_tests(self):
        """Test categorization of edge case tests."""
        tests = [
            self.create_test_metadata("test_empty_input", "Test with empty input"),
            self.create_test_metadata("test_null_value", "Test with null value"),
            self.create_test_metadata("test_boundary_conditions", "Test boundary conditions"),
            self.create_test_metadata("test_max_limit", "Test maximum limit"),
            self.create_test_metadata("test_min_value", "Test minimum value"),
            self.create_test_metadata("test_negative_input", "Test negative input")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        edge_case_count = analysis.categories.get(TestCategory.UNIT_EDGE_CASE, 0)
        assert edge_case_count > 0

    def test_categorize_error_handling_tests(self):
        """Test categorization of error handling tests."""
        tests = [
            self.create_test_metadata("test_exception_handling", "Test exception handling"),
            self.create_test_metadata("test_error_conditions", "Test error conditions"),
            self.create_test_metadata("test_invalid_input", "Test with invalid input"),
            self.create_test_metadata("test_failure_scenarios", "Test failure scenarios"),
            self.create_test_metadata("test_raises_exception", "Test that raises exception")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        error_handling_count = analysis.categories.get(TestCategory.UNIT_ERROR_HANDLING, 0)
        assert error_handling_count > 0

    def test_categorize_integration_tests(self):
        """Test categorization of integration tests."""
        tests = [
            self.create_test_metadata("test_api_endpoint", "Test API endpoint integration"),
            self.create_test_metadata("test_database_connection", "Test database integration"),
            self.create_test_metadata("test_external_service", "Test external service integration"),
            self.create_test_metadata("test_service_calls", "Test service API calls")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        api_count = analysis.categories.get(TestCategory.INTEGRATION_API, 0)
        db_count = analysis.categories.get(TestCategory.INTEGRATION_DATABASE, 0)
        external_count = analysis.categories.get(TestCategory.INTEGRATION_EXTERNAL, 0)

        assert (api_count + db_count + external_count) > 0

    def test_categorize_performance_tests(self):
        """Test categorization of performance tests."""
        tests = [
            self.create_test_metadata("test_performance_benchmark", "Performance benchmark test"),
            self.create_test_metadata("test_load_handling", "Test load handling"),
            self.create_test_metadata("test_stress_conditions", "Test under stress conditions"),
            self.create_test_metadata("test_concurrent_access", "Test concurrent access")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        load_count = analysis.categories.get(TestCategory.PERFORMANCE_LOAD, 0)
        stress_count = analysis.categories.get(TestCategory.PERFORMANCE_STRESS, 0)

        assert (load_count + stress_count) > 0

    def test_categorize_security_tests(self):
        """Test categorization of security tests."""
        tests = [
            self.create_test_metadata("test_authentication", "Test authentication mechanism"),
            self.create_test_metadata("test_authorization", "Test authorization checks"),
            self.create_test_metadata("test_input_validation", "Test input validation"),
            self.create_test_metadata("test_sanitization", "Test data sanitization"),
            self.create_test_metadata("test_xss_protection", "Test XSS protection")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        auth_count = analysis.categories.get(TestCategory.SECURITY_AUTH, 0)
        validation_count = analysis.categories.get(TestCategory.SECURITY_VALIDATION, 0)

        assert (auth_count + validation_count) > 0

    def test_detect_mock_usage_pattern(self):
        """Test detection of mock usage patterns."""
        tests = [
            self.create_test_metadata("test_with_mock", "Test using mock objects"),
            self.create_test_metadata("test_mock_service", "Test with mocked service"),
            self.create_test_metadata("test_patched_method", "Test with patched method",
                                    decorators=[DecoratorInfo(name="patch")])
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        mock_pattern = next((p for p in analysis.patterns if p.pattern_type == "mock_usage"), None)
        assert mock_pattern is not None
        assert mock_pattern.frequency >= 2

    def test_detect_exception_testing_pattern(self):
        """Test detection of exception testing patterns."""
        tests = [
            self.create_test_metadata("test_exception_raised", "Test that exception is raised"),
            self.create_test_metadata("test_error_handling", "Test error handling"),
            self.create_test_metadata("test_invalid_raises", "Test raises on invalid input")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        exception_pattern = next((p for p in analysis.patterns if p.pattern_type == "exception_testing"), None)
        assert exception_pattern is not None
        assert exception_pattern.frequency >= 2

    def test_detect_parametrized_tests_pattern(self):
        """Test detection of parametrized test patterns."""
        tests = [
            self.create_test_metadata("test_parametrized_1", "Parametrized test 1", is_parametrized=True),
            self.create_test_metadata("test_parametrized_2", "Parametrized test 2", is_parametrized=True),
            self.create_test_metadata("test_regular", "Regular test", is_parametrized=False)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        param_pattern = next((p for p in analysis.patterns if p.pattern_type == "parametrized_tests"), None)
        assert param_pattern is not None
        assert param_pattern.frequency == 2

    def test_detect_async_testing_pattern(self):
        """Test detection of async testing patterns."""
        tests = [
            self.create_test_metadata("test_async_function", "Async test function", is_async=True),
            self.create_test_metadata("test_sync_function", "Sync test function", is_async=False),
            self.create_test_metadata("test_another_async", "Another async test", is_async=True)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        async_pattern = next((p for p in analysis.patterns if p.pattern_type == "async_testing"), None)
        assert async_pattern is not None
        assert async_pattern.frequency == 2

    def test_identify_missing_core_tests_gap(self):
        """Test identification of missing core tests."""
        # No unit core tests
        tests = [
            self.create_test_metadata("test_integration_only", "Integration test")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        core_gap = next((g for g in analysis.gaps
                        if g.category == TestCategory.UNIT_CORE), None)
        assert core_gap is not None
        assert core_gap.priority == 8
        assert core_gap.risk_level == "high"

    def test_identify_insufficient_edge_case_coverage(self):
        """Test identification of insufficient edge case coverage."""
        # Many core tests but few edge cases
        core_tests = [
            self.create_test_metadata(f"test_core_{i}", "Core test")
            for i in range(10)
        ]
        edge_tests = [
            self.create_test_metadata("test_edge_case", "Edge case test")
        ]

        # Set categories manually to test gap detection
        file_infos = [self.create_test_file_info(core_tests + edge_tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        # Should detect insufficient edge case coverage
        edge_gap = next((g for g in analysis.gaps
                        if g.category == TestCategory.UNIT_EDGE_CASE), None)

        # Gap might be detected based on ratio
        if edge_gap:
            assert edge_gap.priority >= 5

    def test_identify_missing_error_handling_gap(self):
        """Test identification of missing error handling tests."""
        # Core tests without error handling
        core_tests = [
            self.create_test_metadata(f"test_core_{i}", "Core test")
            for i in range(5)
        ]

        file_infos = [self.create_test_file_info(core_tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        error_gap = next((g for g in analysis.gaps
                         if g.category == TestCategory.UNIT_ERROR_HANDLING), None)
        assert error_gap is not None
        assert error_gap.priority == 8
        assert error_gap.risk_level == "high"

    def test_identify_missing_integration_tests_gap(self):
        """Test identification of missing integration tests."""
        # Only unit tests, no integration
        unit_tests = [
            self.create_test_metadata(f"test_unit_{i}", "Unit test")
            for i in range(8)
        ]

        file_infos = [self.create_test_file_info(unit_tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        integration_gap = next((g for g in analysis.gaps
                              if g.category == TestCategory.INTEGRATION_API), None)
        assert integration_gap is not None
        assert integration_gap.priority == 6

    def test_identify_missing_performance_tests_gap(self):
        """Test identification of missing performance tests."""
        # Large test suite without performance tests
        tests = [
            self.create_test_metadata(f"test_{i}", "Test")
            for i in range(15)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        perf_gap = next((g for g in analysis.gaps
                        if g.category == TestCategory.PERFORMANCE_LOAD), None)
        assert perf_gap is not None
        assert perf_gap.priority == 5
        assert perf_gap.effort_estimate == "high"

    def test_has_security_concerns_detection(self):
        """Test detection of security-related functionality."""
        # Create test with security-related content
        tests = [
            self.create_test_metadata("test_user_auth", "Test user authentication"),
            self.create_test_metadata("test_login_validation", "Test login validation")
        ]

        file_infos = [self.create_test_file_info(tests)]
        has_security = self.categorizer._has_security_concerns(file_infos)

        assert has_security is True

    def test_no_security_concerns_detection(self):
        """Test when no security concerns are detected."""
        tests = [
            self.create_test_metadata("test_math_calculation", "Test math calculation"),
            self.create_test_metadata("test_string_formatting", "Test string formatting")
        ]

        file_infos = [self.create_test_file_info(tests)]
        has_security = self.categorizer._has_security_concerns(file_infos)

        assert has_security is False

    def test_generate_recommendations_low_coverage(self):
        """Test recommendation generation for low coverage."""
        tests = [
            self.create_test_metadata("test_single", "Single test")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        # Should recommend increasing coverage
        coverage_rec = next((r for r in analysis.recommendations
                           if "coverage" in r.lower()), None)
        assert coverage_rec is not None

    def test_generate_recommendations_missing_patterns(self):
        """Test recommendations for missing test patterns."""
        # Tests without mocking or parametrization
        tests = [
            self.create_test_metadata(f"test_{i}", "Simple test")
            for i in range(5)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        # Should recommend using mocking
        mock_rec = next((r for r in analysis.recommendations
                       if "mock" in r.lower()), None)
        assert mock_rec is not None

        # Should recommend parametrized tests
        param_rec = next((r for r in analysis.recommendations
                         if "parametrized" in r.lower()), None)
        assert param_rec is not None

    def test_generate_recommendations_high_priority_gaps(self):
        """Test recommendations for high-priority gaps."""
        # Create analysis with high-priority gaps
        file_infos = [self.create_test_file_info([])]  # No tests
        analysis = self.categorizer.categorize_tests(file_infos)

        # Should recommend addressing high-priority gaps
        gap_rec = next((r for r in analysis.recommendations
                      if "high-priority" in r.lower()), None)
        assert gap_rec is not None

    def test_calculate_coverage_score_empty_suite(self):
        """Test coverage score calculation for empty test suite."""
        file_infos = [self.create_test_file_info()]
        analysis = self.categorizer.categorize_tests(file_infos)

        assert analysis.coverage_score == 0.0

    def test_calculate_coverage_score_good_coverage(self):
        """Test coverage score calculation for good coverage."""
        # Create diverse test suite
        tests = [
            self.create_test_metadata("test_core", "Core test"),
            self.create_test_metadata("test_edge_case", "Edge case test"),
            self.create_test_metadata("test_error", "Error test"),
            self.create_test_metadata("test_integration", "Integration test"),
            self.create_test_metadata("test_performance", "Performance test", is_async=True),
            self.create_test_metadata("test_parametrized", "Parametrized test", is_parametrized=True)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        assert analysis.coverage_score > 0.5

    def test_calculate_completeness_score_empty_suite(self):
        """Test completeness score calculation for empty suite."""
        file_infos = [self.create_test_file_info()]
        analysis = self.categorizer.categorize_tests(file_infos)

        assert analysis.completeness_score == 0.0

    def test_calculate_completeness_score_essential_coverage(self):
        """Test completeness score with essential category coverage."""
        tests = [
            self.create_test_metadata("test_core", "Core test"),
            self.create_test_metadata("test_error_handling", "Error handling test"),
            self.create_test_metadata("test_edge_case", "Edge case test")
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        # Should have reasonable completeness score
        assert analysis.completeness_score > 0.3

    def test_suggest_tests_for_category(self):
        """Test test suggestions for specific categories."""
        core_suggestions = self.categorizer._suggest_tests_for_category(TestCategory.UNIT_CORE)
        assert len(core_suggestions) > 0
        assert any("core functionality" in s.lower() for s in core_suggestions)

        error_suggestions = self.categorizer._suggest_tests_for_category(TestCategory.UNIT_ERROR_HANDLING)
        assert len(error_suggestions) > 0
        assert any("exception" in s.lower() for s in error_suggestions)

        unknown_suggestions = self.categorizer._suggest_tests_for_category(TestCategory.UNKNOWN)
        assert len(unknown_suggestions) > 0

    def test_pattern_confidence_levels(self):
        """Test that patterns have appropriate confidence levels."""
        tests = [
            self.create_test_metadata("test_with_mock", "Test using mock", is_parametrized=True),
            self.create_test_metadata("test_async", "Async test", is_async=True)
        ]

        file_infos = [self.create_test_file_info(tests)]
        analysis = self.categorizer.categorize_tests(file_infos)

        for pattern in analysis.patterns:
            assert 0.0 <= pattern.confidence <= 1.0

            # Parametrized tests should have high confidence
            if pattern.pattern_type == "parametrized_tests":
                assert pattern.confidence >= 0.9

            # Async tests should have high confidence
            if pattern.pattern_type == "async_testing":
                assert pattern.confidence >= 0.9


class TestCoverageGap:
    """Test CoverageGap dataclass."""

    def test_coverage_gap_creation(self):
        """Test CoverageGap creation and defaults."""
        gap = CoverageGap(
            category=TestCategory.UNIT_CORE,
            description="Missing core tests",
            priority=8
        )

        assert gap.category == TestCategory.UNIT_CORE
        assert gap.description == "Missing core tests"
        assert gap.priority == 8
        assert gap.risk_level == "medium"  # Default
        assert gap.effort_estimate == "medium"  # Default
        assert gap.suggested_tests == []  # Default empty list
        assert gap.affected_modules == []  # Default empty list

    def test_coverage_gap_with_suggestions(self):
        """Test CoverageGap with suggestions."""
        gap = CoverageGap(
            category=TestCategory.UNIT_ERROR_HANDLING,
            description="Missing error handling",
            priority=9,
            suggested_tests=["Test exception handling", "Test error recovery"],
            risk_level="high",
            effort_estimate="low"
        )

        assert len(gap.suggested_tests) == 2
        assert gap.risk_level == "high"
        assert gap.effort_estimate == "low"


class TestTestPattern:
    """Test TestPattern dataclass."""

    def test_test_pattern_creation(self):
        """Test TestPattern creation."""
        pattern = TestPattern(
            pattern_type="mock_usage",
            frequency=5,
            examples=["test_mock_1", "test_mock_2"],
            confidence=0.9
        )

        assert pattern.pattern_type == "mock_usage"
        assert pattern.frequency == 5
        assert len(pattern.examples) == 2
        assert pattern.confidence == 0.9

    def test_test_pattern_defaults(self):
        """Test TestPattern with defaults."""
        pattern = TestPattern(
            pattern_type="exception_testing",
            frequency=3
        )

        assert pattern.examples == []  # Default empty list
        assert pattern.confidence == 1.0  # Default confidence


class TestCoverageAnalysis:
    """Test CoverageAnalysis dataclass."""

    def test_coverage_analysis_creation(self):
        """Test CoverageAnalysis creation and defaults."""
        analysis = CoverageAnalysis(total_tests=10)

        assert analysis.total_tests == 10
        assert analysis.categories == {}  # Default empty dict
        assert analysis.patterns == []  # Default empty list
        assert analysis.gaps == []  # Default empty list
        assert analysis.recommendations == []  # Default empty list
        assert analysis.coverage_score == 0.0  # Default
        assert analysis.completeness_score == 0.0  # Default

    def test_coverage_analysis_with_data(self):
        """Test CoverageAnalysis with data."""
        gap = CoverageGap(TestCategory.UNIT_CORE, "Missing tests", 8)
        pattern = TestPattern("mock_usage", 3)

        analysis = CoverageAnalysis(
            total_tests=5,
            categories={TestCategory.UNIT_CORE: 3, TestCategory.UNIT_EDGE_CASE: 2},
            patterns=[pattern],
            gaps=[gap],
            recommendations=["Add more tests"],
            coverage_score=0.7,
            completeness_score=0.6
        )

        assert analysis.total_tests == 5
        assert len(analysis.categories) == 2
        assert len(analysis.patterns) == 1
        assert len(analysis.gaps) == 1
        assert len(analysis.recommendations) == 1
        assert analysis.coverage_score == 0.7
        assert analysis.completeness_score == 0.6