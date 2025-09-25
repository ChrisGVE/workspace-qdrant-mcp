"""
Comprehensive Tests for Coverage Validation Framework

This module provides extensive testing for the 100% coverage validation system,
covering AST analysis, coverage reporting, gap detection, and improvement planning.
"""

import ast
import pytest
import tempfile
import time
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Import the framework components
import sys
src_path = Path(__file__).parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

framework_path = Path(__file__).parent / "tests" / "framework"
if str(framework_path) not in sys.path:
    sys.path.insert(0, str(framework_path))

from tests.framework.coverage_validator import (
    CoverageValidator, CoverageReport, CoverageLevel, CoverageType,
    CoverageGap, CoverageAnalyzer, ValidationSeverity
)


class TestCoverageGap:
    """Test coverage gap representation."""

    def test_coverage_gap_creation(self):
        """Test coverage gap creation."""
        gap = CoverageGap(
            file_path="src/test.py",
            line_number=42,
            line_content="if x > 0:",
            coverage_type=CoverageType.BRANCH,
            function_name="test_function",
            class_name="TestClass",
            complexity=3,
            is_edge_case=True,
            suggested_test="def test_edge_case(): pass"
        )

        assert gap.file_path == "src/test.py"
        assert gap.line_number == 42
        assert gap.line_content == "if x > 0:"
        assert gap.coverage_type == CoverageType.BRANCH
        assert gap.function_name == "test_function"
        assert gap.class_name == "TestClass"
        assert gap.complexity == 3
        assert gap.is_edge_case is True
        assert gap.suggested_test == "def test_edge_case(): pass"

    def test_coverage_gap_defaults(self):
        """Test coverage gap default values."""
        gap = CoverageGap(
            file_path="src/simple.py",
            line_number=10,
            line_content="return True",
            coverage_type=CoverageType.LINE
        )

        assert gap.function_name is None
        assert gap.class_name is None
        assert gap.complexity == 1
        assert gap.is_edge_case is False
        assert gap.suggested_test is None


class TestCoverageReport:
    """Test coverage report functionality."""

    def test_coverage_report_creation(self):
        """Test coverage report creation."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=80,
            missed_lines=20,
            line_coverage_percentage=80.0,
            branch_coverage_percentage=75.0,
            function_coverage_percentage=90.0
        )

        assert report.total_lines == 100
        assert report.covered_lines == 80
        assert report.missed_lines == 20
        assert report.line_coverage_percentage == 80.0
        assert report.branch_coverage_percentage == 75.0
        assert report.function_coverage_percentage == 90.0

    def test_overall_coverage_calculation(self):
        """Test overall coverage percentage calculation."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=90,
            missed_lines=10,
            line_coverage_percentage=90.0,
            branch_coverage_percentage=80.0,
            function_coverage_percentage=85.0
        )

        # Weighted average: 90*0.4 + 80*0.3 + 85*0.3 = 36 + 24 + 25.5 = 85.5
        expected = 90.0 * 0.4 + 80.0 * 0.3 + 85.0 * 0.3
        assert abs(report.overall_coverage_percentage - expected) < 0.01

    def test_100_percent_threshold_met(self):
        """Test 100% threshold detection when met."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=100,
            missed_lines=0,
            line_coverage_percentage=100.0,
            branch_coverage_percentage=100.0,
            function_coverage_percentage=100.0
        )

        assert report.meets_100_percent_threshold is True

    def test_100_percent_threshold_not_met_line(self):
        """Test 100% threshold detection when line coverage insufficient."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=95,
            missed_lines=5,
            line_coverage_percentage=95.0,
            branch_coverage_percentage=100.0,
            function_coverage_percentage=100.0
        )

        assert report.meets_100_percent_threshold is False

    def test_100_percent_threshold_not_met_edge_cases(self):
        """Test 100% threshold detection when edge cases missing."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=100,
            missed_lines=0,
            line_coverage_percentage=100.0,
            branch_coverage_percentage=100.0,
            function_coverage_percentage=100.0,
            edge_cases_missing=["edge_case_1", "edge_case_2"]
        )

        assert report.meets_100_percent_threshold is False


class TestCoverageAnalyzer:
    """Test AST-based coverage analyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = CoverageAnalyzer()

        assert isinstance(analyzer.functions, list)
        assert isinstance(analyzer.classes, list)
        assert isinstance(analyzer.branches, list)
        assert isinstance(analyzer.complexity_points, list)
        assert isinstance(analyzer.edge_cases, list)
        assert analyzer.current_class is None
        assert analyzer.current_function is None

    def test_visit_class_def(self):
        """Test class definition analysis."""
        code = '''
class TestClass:
    """Test class."""

    def method1(self):
        pass

    def method2(self):
        return True
'''

        tree = ast.parse(code)
        analyzer = CoverageAnalyzer()
        analyzer.visit(tree)

        assert len(analyzer.classes) == 1
        class_info = analyzer.classes[0]
        assert class_info['name'] == 'TestClass'
        assert class_info['lineno'] == 2
        assert isinstance(class_info['methods'], list)
        assert isinstance(class_info['decorators'], list)
        assert isinstance(class_info['bases'], list)

    def test_visit_function_def(self):
        """Test function definition analysis."""
        code = '''
def simple_function(a, b):
    """Simple function."""
    return a + b

@staticmethod
def complex_function(x, y, z):
    """Complex function with branches."""
    if x > 0:
        return x + y
    else:
        return z
'''

        tree = ast.parse(code)
        analyzer = CoverageAnalyzer()
        analyzer.visit(tree)

        assert len(analyzer.functions) >= 2

        # Find simple function
        simple_func = next(f for f in analyzer.functions if f['name'] == 'simple_function')
        assert simple_func['args'] == 2
        assert simple_func['class'] is None
        assert len(simple_func['decorators']) == 0
        assert simple_func['has_return'] is True

        # Find complex function
        complex_func = next(f for f in analyzer.functions if f['name'] == 'complex_function')
        assert complex_func['args'] == 3
        assert 'staticmethod' in [str(d) for d in complex_func['decorators']]
        assert complex_func['complexity'] > 1

    def test_visit_if_statement(self):
        """Test if statement branch analysis."""
        code = '''
def test_branches(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"

    if x == 0:  # Edge case
        handle_zero()
'''

        tree = ast.parse(code)
        analyzer = CoverageAnalyzer()
        analyzer.visit(tree)

        # Should detect branches
        assert len(analyzer.branches) >= 2

        # Check for if statements
        if_branches = [b for b in analyzer.branches if b.get('type') == 'if']
        assert len(if_branches) >= 2

        # Check for edge cases (x == 0)
        edge_cases = [ec for ec in analyzer.edge_cases if ec.get('type') == 'boundary_condition']
        assert len(edge_cases) >= 1

    def test_visit_for_loop(self):
        """Test for loop analysis."""
        code = '''
def process_items(items):
    for item in items:
        process(item)

    for i in range(0):  # Empty range - edge case
        pass
'''

        tree = ast.parse(code)
        analyzer = CoverageAnalyzer()
        analyzer.visit(tree)

        # Should detect complexity points for loops
        loop_points = [cp for cp in analyzer.complexity_points if cp.get('type') == 'loop']
        assert len(loop_points) >= 2

        # Should detect empty loop edge case
        empty_loops = [ec for ec in analyzer.edge_cases if ec.get('type') == 'empty_loop']
        assert len(empty_loops) >= 1

    def test_visit_try_except(self):
        """Test try-except block analysis."""
        code = '''
def risky_operation():
    try:
        dangerous_call()
    except ValueError as e:
        handle_value_error(e)
    except TypeError:
        handle_type_error()
    except Exception:
        handle_generic_error()
    finally:
        cleanup()
'''

        tree = ast.parse(code)
        analyzer = CoverageAnalyzer()
        analyzer.visit(tree)

        # Should detect exception branches
        exception_branches = [b for b in analyzer.branches if 'except_count' in b]
        assert len(exception_branches) >= 1

        exception_branch = exception_branches[0]
        assert exception_branch['except_count'] == 3
        assert exception_branch['has_finally'] is True

        # Should detect exception handler edge cases
        exception_handlers = [ec for ec in analyzer.edge_cases if ec.get('type') == 'exception_handler']
        assert len(exception_handlers) == 3  # One for each except block

    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation."""
        analyzer = CoverageAnalyzer()

        # Simple function (complexity = 1)
        simple_code = '''
def simple():
    return True
'''
        simple_tree = ast.parse(simple_code)
        simple_func = simple_tree.body[0]
        complexity = analyzer._calculate_complexity(simple_func)
        assert complexity == 1

        # Complex function with branches
        complex_code = '''
def complex(x, y):
    if x > 0 and y > 0:
        for i in range(x):
            if i % 2 == 0:
                continue
        return True
    elif x < 0:
        while y > 0:
            y -= 1
        return False
    else:
        return None
'''
        complex_tree = ast.parse(complex_code)
        complex_func = complex_tree.body[0]
        complexity = analyzer._calculate_complexity(complex_func)
        assert complexity > 5  # Should have high complexity

    def test_edge_case_detection(self):
        """Test edge case condition detection."""
        analyzer = CoverageAnalyzer()

        # Test boundary conditions
        boundary_cases = [
            "x == 0",
            "y < 1",
            "z != None",
            "len(items) == 0",
            "value == ''"
        ]

        for case in boundary_cases:
            test_node = ast.parse(case, mode='eval').body
            if analyzer._is_edge_case_condition(test_node):
                # At least some should be detected as edge cases
                pass

        # Test negation
        not_test = ast.parse("not condition", mode='eval').body
        assert analyzer._is_edge_case_condition(not_test) is True


class TestCoverageValidator:
    """Test the main coverage validation system."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            src_dir = project_root / "src"
            test_dir = project_root / "tests"
            src_dir.mkdir()
            test_dir.mkdir()

            # Create sample source file
            (src_dir / "calculator.py").write_text('''
"""Simple calculator module."""

class Calculator:
    """Basic calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:  # Edge case
            raise ValueError("Division by zero")
        return a / b

    def factorial(self, n):
        """Calculate factorial."""
        if n < 0:
            raise ValueError("Negative input")
        if n == 0 or n == 1:  # Edge cases
            return 1

        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

def utility_function(items):
    """Process items."""
    if not items:  # Edge case
        return []

    processed = []
    try:
        for item in items:
            processed.append(str(item).upper())
    except Exception as e:
        print(f"Error: {e}")
        return []

    return processed
''')

            # Create sample test file
            (test_dir / "test_calculator.py").write_text('''
import pytest
from src.calculator import Calculator, utility_function

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5

def test_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ValueError):
        calc.divide(10, 0)
''')

            yield project_root, src_dir, test_dir

    @pytest.fixture
    def validator(self, temp_project):
        """Create test coverage validator."""
        project_root, src_dir, test_dir = temp_project
        return CoverageValidator(project_root, src_dir, test_dir)

    def test_validator_initialization(self, temp_project):
        """Test coverage validator initialization."""
        project_root, src_dir, test_dir = temp_project

        validator = CoverageValidator(project_root, src_dir, test_dir)

        assert validator.project_root == project_root
        assert validator.source_directory == src_dir
        assert validator.test_directory == test_dir
        assert validator.coverage_level == CoverageLevel.COMPREHENSIVE
        assert validator.database_path.exists()

    def test_validator_with_custom_level(self, temp_project):
        """Test validator with custom coverage level."""
        project_root, src_dir, test_dir = temp_project

        validator = CoverageValidator(
            project_root, src_dir, test_dir,
            coverage_level=CoverageLevel.EXHAUSTIVE
        )

        assert validator.coverage_level == CoverageLevel.EXHAUSTIVE

    def test_database_initialization(self, validator):
        """Test database initialization."""
        # Check that database file exists
        assert validator.database_path.exists()

        # Check that tables were created
        with sqlite3.connect(validator.database_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'coverage_reports', 'coverage_gaps', 'validation_history'
                )
            """)
            tables = [row[0] for row in cursor.fetchall()]

            assert 'coverage_reports' in tables
            assert 'coverage_gaps' in tables
            assert 'validation_history' in tables

    def test_coverage_config(self, validator):
        """Test coverage configuration."""
        config = validator.coverage_config

        assert str(validator.source_directory) in config['source']
        assert '*/tests/*' in config['omit']
        assert '*.py' in config['include']
        assert config['precision'] == 2
        assert config['show_missing'] is True

    def test_ast_analysis(self, validator):
        """Test AST analysis of source files."""
        analyzers = validator._perform_ast_analysis(None)

        assert len(analyzers) >= 1

        # Check calculator.py analysis
        calc_analyzer = None
        for path, analyzer in analyzers.items():
            if 'calculator.py' in path:
                calc_analyzer = analyzer
                break

        assert calc_analyzer is not None
        assert len(calc_analyzer.functions) >= 4  # Calculator methods + utility_function
        assert len(calc_analyzer.classes) >= 1   # Calculator class
        assert len(calc_analyzer.branches) > 0   # if statements
        assert len(calc_analyzer.edge_cases) > 0 # Edge case conditions

    def test_ast_analysis_with_target_files(self, validator):
        """Test AST analysis with specific target files."""
        target_files = ["src/calculator.py"]
        analyzers = validator._perform_ast_analysis(target_files)

        assert len(analyzers) == 1
        assert any('calculator.py' in path for path in analyzers.keys())

    def test_load_coverage_data_mock(self, validator):
        """Test loading coverage data with mocked files."""
        # Mock coverage files
        mock_json_data = {
            'totals': {
                'num_statements': 100,
                'covered_lines': 80,
                'missing_lines': 20,
                'percent_covered': 80.0,
                'num_branches': 30,
                'covered_branches': 24
            },
            'files': {
                'src/calculator.py': {
                    'summary': {'percent_covered': 75.0},
                    'missing_lines': [15, 25, 30]
                }
            }
        }

        mock_xml_content = '''<?xml version="1.0"?>
        <coverage version="1.0">
            <sources><source>src</source></sources>
        </coverage>'''

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('xml.etree.ElementTree.parse') as mock_parse:
                    mock_parse.return_value = Mock()
                    coverage_data = validator._load_coverage_data()

                    assert 'json' in coverage_data
                    assert coverage_data['json']['totals']['num_statements'] == 100

    def test_generate_coverage_report(self, validator):
        """Test coverage report generation."""
        # Mock coverage data
        coverage_data = {
            'json': {
                'totals': {
                    'num_statements': 50,
                    'covered_lines': 40,
                    'missing_lines': 10,
                    'percent_covered': 80.0,
                    'num_branches': 20,
                    'covered_branches': 16
                },
                'files': {
                    'src/calculator.py': {
                        'summary': {'percent_covered': 80.0},
                        'missing_lines': [15, 25]
                    }
                }
            }
        }

        # Mock AST analysis
        ast_analysis = validator._perform_ast_analysis(None)

        with patch.object(validator, '_get_line_content', return_value="mock line"):
            report = validator._generate_coverage_report(coverage_data, ast_analysis)

            assert report.total_lines == 50
            assert report.covered_lines == 40
            assert report.missed_lines == 10
            assert report.line_coverage_percentage == 80.0
            assert report.branch_coverage_percentage == 80.0  # (16/20)*100
            assert len(report.coverage_gaps) == 2  # Two missing lines

    def test_validate_edge_cases(self, validator):
        """Test edge case validation."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=50,
            covered_lines=45,
            missed_lines=5,
            line_coverage_percentage=90.0,
            branch_coverage_percentage=85.0,
            function_coverage_percentage=88.0
        )

        # Mock AST analysis with edge cases
        ast_analysis = {}
        mock_analyzer = Mock()
        mock_analyzer.edge_cases = [
            {
                'type': 'boundary_condition',
                'lineno': 10,
                'function': 'test_func',
                'class': 'TestClass'
            },
            {
                'type': 'exception_handler',
                'lineno': 20,
                'function': 'handle_error',
                'class': None
            }
        ]
        ast_analysis['src/test.py'] = mock_analyzer

        with patch.object(validator, '_is_edge_case_covered', return_value=False):
            validator._validate_edge_cases(report, ast_analysis)

            assert len(report.edge_cases_missing) == 2
            assert len(report.coverage_gaps) == 2

            # Check gap properties
            gap = report.coverage_gaps[0]
            assert gap.is_edge_case is True
            assert gap.coverage_type == CoverageType.CONDITION

    def test_generate_test_suggestions(self, validator):
        """Test test suggestion generation."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=50,
            covered_lines=45,
            missed_lines=5,
            line_coverage_percentage=90.0,
            branch_coverage_percentage=85.0,
            function_coverage_percentage=88.0
        )

        # Add coverage gaps
        report.coverage_gaps = [
            CoverageGap(
                file_path="src/test.py",
                line_number=10,
                line_content="boundary_condition check",
                coverage_type=CoverageType.CONDITION,
                function_name="test_function",
                is_edge_case=True
            ),
            CoverageGap(
                file_path="src/test.py",
                line_number=15,
                line_content="return value",
                coverage_type=CoverageType.LINE,
                function_name="another_function",
                is_edge_case=False
            )
        ]

        validator._generate_test_suggestions(report)

        # Check that suggestions were generated
        for gap in report.coverage_gaps:
            assert gap.suggested_test is not None
            assert "def test_" in gap.suggested_test

    def test_get_line_content(self, validator):
        """Test getting line content from file."""
        # Create a test file
        test_file = validator.project_root / "test_lines.py"
        test_content = "line 1\nline 2\nline 3\n"
        test_file.write_text(test_content)

        # Test getting various lines
        assert validator._get_line_content(str(test_file), 1) == "line 1"
        assert validator._get_line_content(str(test_file), 2) == "line 2"
        assert validator._get_line_content(str(test_file), 3) == "line 3"

        # Test out of bounds
        assert validator._get_line_content(str(test_file), 10) == ""

        # Test non-existent file
        assert validator._get_line_content("nonexistent.py", 1) == ""

    def test_save_coverage_report(self, validator):
        """Test saving coverage report to database."""
        report = CoverageReport(
            timestamp=time.time(),
            total_lines=100,
            covered_lines=90,
            missed_lines=10,
            line_coverage_percentage=90.0,
            branch_coverage_percentage=85.0,
            function_coverage_percentage=95.0
        )

        # Add a coverage gap
        report.coverage_gaps = [
            CoverageGap(
                file_path="src/test.py",
                line_number=42,
                line_content="uncovered line",
                coverage_type=CoverageType.LINE,
                function_name="test_func",
                complexity=2,
                is_edge_case=True,
                suggested_test="def test_suggestion(): pass"
            )
        ]

        validator._save_coverage_report(report)

        # Verify data was saved
        with sqlite3.connect(validator.database_path) as conn:
            # Check report was saved
            cursor = conn.execute("""
                SELECT line_coverage, branch_coverage, function_coverage,
                       overall_coverage, total_gaps, meets_threshold
                FROM coverage_reports
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            assert row[0] == 90.0  # line_coverage
            assert row[1] == 85.0  # branch_coverage
            assert row[2] == 95.0  # function_coverage
            assert row[4] == 1     # total_gaps
            assert row[5] == 0     # meets_threshold (False)

            # Check coverage gap was saved
            cursor = conn.execute("SELECT COUNT(*) FROM coverage_gaps")
            gap_count = cursor.fetchone()[0]
            assert gap_count == 1

    def test_get_coverage_trends(self, validator):
        """Test getting coverage trends over time."""
        # Add test data to database
        current_time = time.time()
        with sqlite3.connect(validator.database_path) as conn:
            for i in range(5):
                timestamp = current_time - (i * 86400)  # Daily intervals
                conn.execute("""
                    INSERT INTO coverage_reports
                    (timestamp, line_coverage, branch_coverage, function_coverage,
                     overall_coverage, total_gaps, meets_threshold, report_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    80 + i * 2,  # Increasing coverage
                    75 + i * 3,
                    85 + i * 1,
                    80 + i * 2,
                    5 - i,       # Decreasing gaps
                    0,
                    '{}'
                ))

        trends = validator.get_coverage_trends(days=7)

        assert len(trends['timestamps']) == 5
        assert len(trends['line_coverage']) == 5
        assert len(trends['branch_coverage']) == 5
        assert len(trends['function_coverage']) == 5
        assert len(trends['overall_coverage']) == 5

        # Check trend direction (should be increasing)
        assert trends['line_coverage'][-1] > trends['line_coverage'][0]

    def test_get_persistent_coverage_gaps(self, validator):
        """Test getting persistent coverage gaps."""
        # Add test data
        with sqlite3.connect(validator.database_path) as conn:
            # Create reports
            for i in range(3):
                cursor = conn.execute("""
                    INSERT INTO coverage_reports
                    (timestamp, line_coverage, branch_coverage, function_coverage,
                     overall_coverage, total_gaps, meets_threshold, report_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (time.time() - i * 3600, 80, 75, 85, 80, 1, 0, '{}'))

                report_id = cursor.lastrowid

                # Add persistent gap
                conn.execute("""
                    INSERT INTO coverage_gaps
                    (report_id, file_path, line_number, coverage_type,
                     function_name, suggested_test, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (report_id, 'src/persistent.py', 42, 'line', 'test_func', 'test suggestion', 0))

        gaps = validator.get_persistent_coverage_gaps(occurrences_threshold=2)

        assert len(gaps) == 1
        gap = gaps[0]
        assert gap['file_path'] == 'src/persistent.py'
        assert gap['line_number'] == 42
        assert gap['coverage_type'] == 'line'
        assert gap['occurrences'] == 3

    def test_generate_coverage_improvement_plan(self, validator):
        """Test generating coverage improvement plan."""
        # Add test report to database
        with sqlite3.connect(validator.database_path) as conn:
            cursor = conn.execute("""
                INSERT INTO coverage_reports
                (timestamp, line_coverage, branch_coverage, function_coverage,
                 overall_coverage, total_gaps, meets_threshold, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), 85, 80, 90, 85, 5, 0, '{}'))

            report_id = cursor.lastrowid

            # Add gaps with different priorities
            for i in range(6):
                conn.execute("""
                    INSERT INTO coverage_gaps
                    (report_id, file_path, line_number, coverage_type, resolved)
                    VALUES (?, ?, ?, ?, ?)
                """, (report_id, f'src/test{i}.py', i * 10, 'line', 0))

        with patch.object(validator, 'get_persistent_coverage_gaps') as mock_gaps:
            mock_gaps.return_value = [
                {'file_path': 'high.py', 'occurrences': 6},  # High priority
                {'file_path': 'medium.py', 'occurrences': 4}, # Medium priority
                {'file_path': 'low.py', 'occurrences': 2}     # Low priority
            ]

            plan = validator.generate_coverage_improvement_plan()

            assert 'current_coverage' in plan
            assert 'target_coverage' in plan
            assert 'improvement_needed' in plan
            assert 'action_items' in plan
            assert 'estimated_effort' in plan

            # Check coverage values
            assert plan['current_coverage']['line'] == 85
            assert plan['target_coverage']['line'] == 100.0
            assert plan['improvement_needed']['line'] == 15.0

            # Check prioritization
            assert len(plan['action_items']['high_priority']) == 1
            assert len(plan['action_items']['medium_priority']) == 1
            assert len(plan['action_items']['low_priority']) == 1

    def test_export_coverage_report_json(self, validator):
        """Test exporting coverage report as JSON."""
        # Add test data
        with sqlite3.connect(validator.database_path) as conn:
            conn.execute("""
                INSERT INTO coverage_reports
                (timestamp, line_coverage, branch_coverage, function_coverage,
                 overall_coverage, total_gaps, meets_threshold, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), 95, 90, 98, 94.2, 2, 0, '{}'))

        json_report = validator.export_coverage_report(format_type='json')

        assert json_report != ""
        data = json.loads(json_report)
        assert data['line_coverage'] == 95
        assert data['branch_coverage'] == 90
        assert data['function_coverage'] == 98
        assert data['overall_coverage'] == 94.2

    def test_export_coverage_report_html(self, validator):
        """Test exporting coverage report as HTML."""
        # Add test data
        with sqlite3.connect(validator.database_path) as conn:
            conn.execute("""
                INSERT INTO coverage_reports
                (timestamp, line_coverage, branch_coverage, function_coverage,
                 overall_coverage, total_gaps, meets_threshold, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), 88, 82, 95, 88.6, 3, 0, '{}'))

        html_report = validator.export_coverage_report(format_type='html')

        assert html_report != ""
        assert "Coverage Validation Report" in html_report
        assert "88.00%" in html_report  # Line coverage
        assert "82.00%" in html_report  # Branch coverage
        assert "95.00%" in html_report  # Function coverage

    def test_coverage_validator_integration(self, validator):
        """Test complete coverage validation workflow."""
        # Mock the test execution and coverage data loading
        with patch.object(validator, '_run_tests_with_coverage') as mock_run_tests:
            with patch.object(validator, '_load_coverage_data') as mock_load_data:
                mock_load_data.return_value = {
                    'json': {
                        'totals': {
                            'num_statements': 100,
                            'covered_lines': 95,
                            'missing_lines': 5,
                            'percent_covered': 95.0,
                            'num_branches': 30,
                            'covered_branches': 28
                        },
                        'files': {
                            'src/calculator.py': {
                                'summary': {'percent_covered': 95.0},
                                'missing_lines': [25]
                            }
                        }
                    }
                }

                report = validator.validate_coverage(run_tests=True)

                assert report is not None
                assert report.line_coverage_percentage == 95.0
                assert report.branch_coverage_percentage > 90.0
                assert len(report.coverage_gaps) >= 1
                mock_run_tests.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])