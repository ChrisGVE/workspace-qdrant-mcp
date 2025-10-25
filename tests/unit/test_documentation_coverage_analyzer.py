"""Unit tests for the documentation coverage analyzer."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the docs framework to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../docs/framework'))

from generators.ast_parser import MemberType
from validation.coverage_analyzer import (
    CoverageStats,
    DocumentationCoverageAnalyzer,
    FileCoverage,
    MemberCoverage,
    ProjectCoverage,
)


class TestCoverageStats:
    """Test the CoverageStats class."""

    def test_calculate_percentage_normal(self):
        """Test percentage calculation with normal values."""
        stats = CoverageStats(total_items=10, documented_items=7)
        stats.calculate_percentage()
        assert stats.coverage_percentage == 70.0

    def test_calculate_percentage_zero_total(self):
        """Test percentage calculation with zero total items."""
        stats = CoverageStats(total_items=0, documented_items=0)
        stats.calculate_percentage()
        assert stats.coverage_percentage == 100.0

    def test_calculate_percentage_perfect(self):
        """Test percentage calculation with 100% coverage."""
        stats = CoverageStats(total_items=5, documented_items=5)
        stats.calculate_percentage()
        assert stats.coverage_percentage == 100.0

    def test_calculate_percentage_zero_documented(self):
        """Test percentage calculation with no documented items."""
        stats = CoverageStats(total_items=3, documented_items=0)
        stats.calculate_percentage()
        assert stats.coverage_percentage == 0.0


class TestMemberCoverage:
    """Test the MemberCoverage class."""

    def test_is_fully_documented_true(self):
        """Test fully documented member."""
        member = MemberCoverage(
            name="test_func",
            member_type=MemberType.FUNCTION,
            has_docstring=True,
            has_parameters_documented=True,
            has_return_documented=True
        )
        assert member.is_fully_documented is True

    def test_is_fully_documented_false_no_docstring(self):
        """Test member missing docstring."""
        member = MemberCoverage(
            name="test_func",
            member_type=MemberType.FUNCTION,
            has_docstring=False,
            has_parameters_documented=True,
            has_return_documented=True
        )
        assert member.is_fully_documented is False

    def test_is_fully_documented_false_no_params(self):
        """Test member missing parameter documentation."""
        member = MemberCoverage(
            name="test_func",
            member_type=MemberType.FUNCTION,
            has_docstring=True,
            has_parameters_documented=False,
            has_return_documented=True
        )
        assert member.is_fully_documented is False

    def test_is_fully_documented_false_no_return(self):
        """Test member missing return documentation."""
        member = MemberCoverage(
            name="test_func",
            member_type=MemberType.FUNCTION,
            has_docstring=True,
            has_parameters_documented=True,
            has_return_documented=False
        )
        assert member.is_fully_documented is False


class TestFileCoverage:
    """Test the FileCoverage class."""

    def test_calculate_stats(self):
        """Test calculation of file coverage statistics."""
        file_coverage = FileCoverage(file_path="test.py")

        # Add some test members
        file_coverage.members = [
            MemberCoverage(
                name="documented_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True,
                has_parameters_documented=True,
                has_return_documented=True,
                has_examples=True
            ),
            MemberCoverage(
                name="undocumented_func",
                member_type=MemberType.FUNCTION,
                has_docstring=False,
                has_parameters_documented=False,
                has_return_documented=False,
                has_examples=False
            ),
            MemberCoverage(
                name="partially_documented_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True,
                has_parameters_documented=False,
                has_return_documented=True,
                has_examples=False
            )
        ]

        file_coverage.calculate_stats()

        assert file_coverage.stats.total_items == 3
        assert file_coverage.stats.documented_items == 2
        assert file_coverage.stats.missing_docstring == 1
        assert file_coverage.stats.missing_parameters == 2
        assert file_coverage.stats.missing_returns == 1
        assert file_coverage.stats.missing_examples == 3
        assert file_coverage.stats.coverage_percentage == pytest.approx(66.67, rel=1e-2)


class TestProjectCoverage:
    """Test the ProjectCoverage class."""

    def test_calculate_overall_stats(self):
        """Test calculation of overall project statistics."""
        project_coverage = ProjectCoverage(project_path="/test/project")

        # Create file coverages
        file1 = FileCoverage(file_path="file1.py")
        file1.stats = CoverageStats(total_items=5, documented_items=3)
        file1.stats.missing_docstring = 2
        file1.stats.missing_parameters = 1

        file2 = FileCoverage(file_path="file2.py")
        file2.stats = CoverageStats(total_items=3, documented_items=2)
        file2.stats.missing_docstring = 1
        file2.stats.missing_returns = 1

        project_coverage.files = [file1, file2]
        project_coverage.calculate_overall_stats()

        assert project_coverage.overall_stats.total_items == 8
        assert project_coverage.overall_stats.documented_items == 5
        assert project_coverage.overall_stats.missing_docstring == 3
        assert project_coverage.overall_stats.missing_parameters == 1
        assert project_coverage.overall_stats.missing_returns == 1
        assert project_coverage.overall_stats.coverage_percentage == 62.5


class TestDocumentationCoverageAnalyzer:
    """Test the DocumentationCoverageAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DocumentationCoverageAnalyzer()
        self.strict_analyzer = DocumentationCoverageAnalyzer(
            require_examples=True,
            require_return_docs=True,
            require_param_docs=True
        )

    def test_analyze_well_documented_file(self):
        """Test analyzing a well-documented Python file."""
        code = '''
"""A well-documented module."""

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number to add
        b: Second number to add

    Returns:
        The sum of a and b

    Examples:
        >>> add_numbers(2, 3)
        5
    """
    return a + b

class Calculator:
    """A simple calculator class."""

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            Product of x and y
        """
        return x * y
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage = self.analyzer.analyze_file(f.name)

                assert coverage.file_path == f.name
                assert coverage.stats.total_items >= 3  # module, function, class, method
                assert coverage.stats.coverage_percentage > 80

                # Find the add_numbers function
                add_func = next((m for m in coverage.members if m.name == "add_numbers"), None)
                assert add_func is not None
                assert add_func.has_docstring is True
                assert add_func.has_parameters_documented is True
                assert add_func.has_return_documented is True

            finally:
                os.unlink(f.name)

    def test_analyze_poorly_documented_file(self):
        """Test analyzing a poorly documented Python file."""
        code = '''
def undocumented_function(param1, param2):
    return param1 + param2

class UndocumentedClass:
    def method_without_docs(self, arg):
        pass

CONSTANT_VALUE = 42
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage = self.analyzer.analyze_file(f.name)

                assert coverage.stats.coverage_percentage < 50
                assert coverage.stats.missing_docstring > 0

                # All functions should have documentation issues
                undoc_func = next((m for m in coverage.members if m.name == "undocumented_function"), None)
                assert undoc_func is not None
                assert undoc_func.has_docstring is False
                assert len(undoc_func.issues) > 0

            finally:
                os.unlink(f.name)

    def test_analyze_file_not_found(self):
        """Test analyzing non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze_file("/nonexistent/file.py")

    def test_analyze_file_syntax_error(self):
        """Test analyzing file with syntax errors."""
        code = '''
def broken_function(
    # Missing closing parenthesis and other syntax errors
    return "broken"
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                with pytest.raises(SyntaxError):
                    self.analyzer.analyze_file(f.name)
            finally:
                os.unlink(f.name)

    def test_analyze_directory(self):
        """Test analyzing a directory of Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "good_module.py").write_text('''
"""Well documented module."""

def documented_func(param: str) -> str:
    """A documented function.

    Args:
        param: Input parameter

    Returns:
        Processed string
    """
    return param.upper()
''')

            (temp_path / "bad_module.py").write_text('''
def undocumented_func(param):
    return param * 2

class UndocumentedClass:
    def method(self):
        pass
''')

            # Create subdirectory
            subdir = temp_path / "subpackage"
            subdir.mkdir()
            (subdir / "sub_module.py").write_text('''
"""Subpackage module."""
VALUE = 42
''')

            coverage = self.analyzer.analyze_directory(temp_path, recursive=True)

            assert coverage.project_path == str(temp_path)
            assert len(coverage.files) == 3
            assert coverage.overall_stats.total_items > 0

            # Check that files were processed
            file_paths = [fc.file_path for fc in coverage.files]
            assert any("good_module.py" in path for path in file_paths)
            assert any("bad_module.py" in path for path in file_paths)
            assert any("sub_module.py" in path for path in file_paths)

    def test_analyze_directory_non_recursive(self):
        """Test analyzing directory without recursion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Root file
            (temp_path / "root_module.py").write_text('def func(): pass')

            # Subdirectory file (should be ignored)
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "sub_module.py").write_text('def sub_func(): pass')

            coverage = self.analyzer.analyze_directory(temp_path, recursive=False)

            assert len(coverage.files) == 1
            assert "root_module.py" in coverage.files[0].file_path

    def test_analyze_directory_not_found(self):
        """Test analyzing non-existent directory."""
        with pytest.raises(NotADirectoryError):
            self.analyzer.analyze_directory("/nonexistent/directory")

    def test_strict_requirements(self):
        """Test analyzer with strict requirements."""
        code = '''
def partially_documented(param: str) -> str:
    """Function with minimal documentation."""
    return param.upper()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                # Regular analyzer might be more lenient
                regular_coverage = self.analyzer.analyze_file(f.name)

                # Strict analyzer should find more issues
                strict_coverage = self.strict_analyzer.analyze_file(f.name)

                func_regular = next((m for m in regular_coverage.members if m.name == "partially_documented"), None)
                func_strict = next((m for m in strict_coverage.members if m.name == "partially_documented"), None)

                # Strict analyzer should find more issues
                assert len(func_strict.issues) >= len(func_regular.issues)

            finally:
                os.unlink(f.name)

    def test_private_members_inclusion(self):
        """Test inclusion/exclusion of private members."""
        code = '''
def public_function():
    """Public function."""
    pass

def _private_function():
    """Private function."""
    pass

class _PrivateClass:
    """Private class."""
    pass
'''

        analyzer_no_private = DocumentationCoverageAnalyzer(include_private=False)
        analyzer_with_private = DocumentationCoverageAnalyzer(include_private=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage_no_private = analyzer_no_private.analyze_file(f.name)
                coverage_with_private = analyzer_with_private.analyze_file(f.name)

                # Should have fewer members without private
                assert len(coverage_no_private.members) < len(coverage_with_private.members)

                # Public function should be in both
                public_names_no_private = [m.name for m in coverage_no_private.members]
                public_names_with_private = [m.name for m in coverage_with_private.members]

                assert "public_function" in public_names_no_private
                assert "public_function" in public_names_with_private

                # Private members should only be in the inclusive analysis
                assert "_private_function" not in public_names_no_private
                assert "_private_function" in public_names_with_private

            finally:
                os.unlink(f.name)

    def test_generate_text_report_file(self):
        """Test generating text report for file coverage."""
        file_coverage = FileCoverage(file_path="test.py")
        file_coverage.members = [
            MemberCoverage(
                name="good_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True,
                has_parameters_documented=True,
                has_return_documented=True
            ),
            MemberCoverage(
                name="bad_func",
                member_type=MemberType.FUNCTION,
                has_docstring=False,
                issues=["Missing docstring", "Missing parameters"]
            )
        ]
        file_coverage.calculate_stats()

        report = self.analyzer.generate_report(file_coverage, 'text')

        assert "test.py" in report
        assert "Coverage:" in report
        assert "good_func" in report
        assert "bad_func" in report
        assert "Missing docstring" in report

    def test_generate_json_report_project(self):
        """Test generating JSON report for project coverage."""
        project_coverage = ProjectCoverage(project_path="/test/project")

        file_coverage = FileCoverage(file_path="test.py")
        file_coverage.members = [
            MemberCoverage(
                name="test_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True,
                line_number=10
            )
        ]
        file_coverage.calculate_stats()

        project_coverage.files = [file_coverage]
        project_coverage.calculate_overall_stats()

        report = self.analyzer.generate_report(project_coverage, 'json')
        data = json.loads(report)

        assert data['project_path'] == "/test/project"
        assert 'overall_stats' in data
        assert len(data['files']) == 1
        assert data['files'][0]['file_path'] == "test.py"
        assert len(data['files'][0]['members']) == 1

    def test_generate_html_report_file(self):
        """Test generating HTML report for file coverage."""
        file_coverage = FileCoverage(file_path="test.py")
        file_coverage.members = [
            MemberCoverage(
                name="test_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True
            )
        ]
        file_coverage.calculate_stats()

        report = self.analyzer.generate_report(file_coverage, 'html')

        assert "<!DOCTYPE html>" in report
        assert "test.py" in report
        assert "test_func" in report
        assert "Coverage:" in report

    def test_find_undocumented_members_file(self):
        """Test finding undocumented members in file coverage."""
        file_coverage = FileCoverage(file_path="test.py")
        file_coverage.members = [
            MemberCoverage(
                name="good_func",
                member_type=MemberType.FUNCTION,
                has_docstring=True,
                has_parameters_documented=True,
                has_return_documented=True
            ),
            MemberCoverage(
                name="bad_func",
                member_type=MemberType.FUNCTION,
                has_docstring=False
            )
        ]

        undocumented = self.analyzer.find_undocumented_members(file_coverage)

        assert len(undocumented) == 1
        assert undocumented[0].name == "bad_func"

    def test_find_undocumented_members_project(self):
        """Test finding undocumented members in project coverage."""
        project_coverage = ProjectCoverage(project_path="/test")

        file1 = FileCoverage(file_path="file1.py")
        file1.members = [
            MemberCoverage(name="good1", member_type=MemberType.FUNCTION, has_docstring=True),
            MemberCoverage(name="bad1", member_type=MemberType.FUNCTION, has_docstring=False)
        ]

        file2 = FileCoverage(file_path="file2.py")
        file2.members = [
            MemberCoverage(name="bad2", member_type=MemberType.FUNCTION, has_docstring=False)
        ]

        project_coverage.files = [file1, file2]

        undocumented = self.analyzer.find_undocumented_members(project_coverage)

        assert len(undocumented) == 2
        names = [m.name for m in undocumented]
        assert "bad1" in names
        assert "bad2" in names

    def test_meets_threshold_file(self):
        """Test threshold checking for file coverage."""
        file_coverage = FileCoverage(file_path="test.py")
        file_coverage.stats = CoverageStats(total_items=10, documented_items=8)
        file_coverage.stats.calculate_percentage()

        assert self.analyzer.meets_threshold(file_coverage, 75.0) is True
        assert self.analyzer.meets_threshold(file_coverage, 85.0) is False

    def test_meets_threshold_project(self):
        """Test threshold checking for project coverage."""
        project_coverage = ProjectCoverage(project_path="/test")
        project_coverage.overall_stats = CoverageStats(total_items=20, documented_items=15)
        project_coverage.overall_stats.calculate_percentage()

        assert self.analyzer.meets_threshold(project_coverage, 70.0) is True
        assert self.analyzer.meets_threshold(project_coverage, 80.0) is False

    def test_edge_case_empty_file(self):
        """Test analyzing empty Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            try:
                coverage = self.analyzer.analyze_file(f.name)
                assert coverage.stats.total_items == 0
                assert coverage.stats.coverage_percentage == 100.0

            finally:
                os.unlink(f.name)

    def test_edge_case_only_imports(self):
        """Test analyzing file with only imports."""
        code = '''
import os
import sys
from pathlib import Path
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage = self.analyzer.analyze_file(f.name)
                # Should handle files with no documentable members
                assert coverage.stats.total_items >= 0

            finally:
                os.unlink(f.name)

    def test_complex_class_analysis(self):
        """Test analyzing complex class with various member types."""
        code = '''
class ComplexClass:
    """A complex class for testing."""

    CLASS_CONSTANT = "value"

    def __init__(self, value: int):
        """Initialize the class.

        Args:
            value: Initial value
        """
        self.value = value

    @property
    def property_method(self) -> int:
        """Get the value as property."""
        return self.value

    @staticmethod
    def static_method(param: str) -> str:
        """Static method without documentation for parameters."""
        return param.upper()

    def _private_method(self):
        """Private method."""
        pass
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage = self.analyzer.analyze_file(f.name)

                # Should find various member types
                member_types = [m.member_type for m in coverage.members]
                assert MemberType.CLASS in member_types
                assert MemberType.METHOD in member_types

                # Find the class
                complex_class = next((m for m in coverage.members if m.name == "ComplexClass"), None)
                assert complex_class is not None
                assert complex_class.has_docstring is True

            finally:
                os.unlink(f.name)

    def test_parameter_documentation_checking(self):
        """Test checking of parameter documentation."""
        code = '''
def well_documented(param1: str, param2: int = 5) -> str:
    """Well documented function.

    Args:
        param1: First parameter description
        param2: Second parameter description

    Returns:
        Combined result
    """
    return f"{param1}_{param2}"

def poorly_documented(param1: str, param2: int) -> str:
    """Function missing parameter docs."""
    return f"{param1}_{param2}"

def no_params() -> str:
    """Function with no parameters."""
    return "hello"
'''

        analyzer_strict = DocumentationCoverageAnalyzer(require_param_docs=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                coverage = analyzer_strict.analyze_file(f.name)

                well_doc = next((m for m in coverage.members if m.name == "well_documented"), None)
                poorly_doc = next((m for m in coverage.members if m.name == "poorly_documented"), None)
                no_params = next((m for m in coverage.members if m.name == "no_params"), None)

                assert well_doc.has_parameters_documented is True
                assert poorly_doc.has_parameters_documented is False
                assert no_params.has_parameters_documented is True  # No params to document

            finally:
                os.unlink(f.name)
