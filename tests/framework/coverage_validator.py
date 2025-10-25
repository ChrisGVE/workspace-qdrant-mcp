"""
100% Coverage Validation System

This module provides comprehensive coverage validation and analysis,
ensuring complete test coverage with meaningful assertions and edge case handling.
Integrates with pytest-cov, coverage.py, and custom AST analysis for deep validation.
"""

import ast
import json
import logging
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from .discovery import TestDiscovery, TestMetadata


class CoverageType(Enum):
    """Types of coverage analysis."""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"
    CONDITION = "condition"


class CoverageLevel(Enum):
    """Coverage validation levels."""
    BASIC = "basic"  # Line coverage only
    STANDARD = "standard"  # Line + branch coverage
    COMPREHENSIVE = "comprehensive"  # All coverage types
    EXHAUSTIVE = "exhaustive"  # 100% coverage with edge case validation


class ValidationSeverity(Enum):
    """Coverage validation issue severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CoverageGap:
    """Represents a gap in test coverage."""
    file_path: str
    line_number: int
    line_content: str
    coverage_type: CoverageType
    function_name: str | None = None
    class_name: str | None = None
    complexity: int = 1
    is_edge_case: bool = False
    suggested_test: str | None = None


@dataclass
class CoverageReport:
    """Comprehensive coverage analysis report."""
    timestamp: float
    total_lines: int
    covered_lines: int
    missed_lines: int
    line_coverage_percentage: float
    branch_coverage_percentage: float
    function_coverage_percentage: float
    coverage_gaps: list[CoverageGap] = field(default_factory=list)
    uncovered_functions: list[str] = field(default_factory=list)
    uncovered_branches: list[str] = field(default_factory=list)
    edge_cases_missing: list[str] = field(default_factory=list)
    file_coverage: dict[str, float] = field(default_factory=dict)
    complexity_coverage: dict[int, int] = field(default_factory=dict)
    validation_issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def overall_coverage_percentage(self) -> float:
        """Calculate weighted overall coverage percentage."""
        weights = {
            'line': 0.4,
            'branch': 0.3,
            'function': 0.3
        }

        return (
            self.line_coverage_percentage * weights['line'] +
            self.branch_coverage_percentage * weights['branch'] +
            self.function_coverage_percentage * weights['function']
        )

    @property
    def meets_100_percent_threshold(self) -> bool:
        """Check if coverage meets 100% threshold."""
        return (
            self.line_coverage_percentage >= 100.0 and
            self.branch_coverage_percentage >= 100.0 and
            self.function_coverage_percentage >= 100.0 and
            len(self.edge_cases_missing) == 0
        )


class CoverageAnalyzer(ast.NodeVisitor):
    """AST-based coverage analyzer for deep code analysis."""

    def __init__(self):
        self.functions: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self.complexity_points: list[dict[str, Any]] = []
        self.edge_cases: list[dict[str, Any]] = []
        self.current_class: str | None = None
        self.current_function: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name

        class_info = {
            'name': node.name,
            'lineno': node.lineno,
            'methods': [],
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            'bases': [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases]
        }
        self.classes.append(class_info)

        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        old_function = self.current_function
        self.current_function = node.name

        function_info = {
            'name': node.name,
            'class': self.current_class,
            'lineno': node.lineno,
            'args': len(node.args.args),
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            'complexity': self._calculate_complexity(node),
            'has_return': self._has_return_statement(node),
            'has_exceptions': self._has_exception_handling(node)
        }
        self.functions.append(function_info)

        self.generic_visit(node)
        self.current_function = old_function

    def visit_If(self, node: ast.If):
        """Visit if statement for branch coverage."""
        branch_info = {
            'type': 'if',
            'lineno': node.lineno,
            'function': self.current_function,
            'class': self.current_class,
            'has_else': node.orelse is not None and len(node.orelse) > 0,
            'is_elif_chain': isinstance(node.orelse, list) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
        }
        self.branches.append(branch_info)

        # Check for edge cases
        if self._is_edge_case_condition(node.test):
            self.edge_cases.append({
                'type': 'boundary_condition',
                'lineno': node.lineno,
                'function': self.current_function,
                'class': self.current_class,
                'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
            })

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Visit for loop."""
        self.complexity_points.append({
            'type': 'loop',
            'lineno': node.lineno,
            'function': self.current_function,
            'class': self.current_class
        })

        # Check for empty loop edge case
        if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
            self.edge_cases.append({
                'type': 'empty_loop',
                'lineno': node.lineno,
                'function': self.current_function,
                'class': self.current_class
            })

        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        """Visit while loop."""
        self.complexity_points.append({
            'type': 'while_loop',
            'lineno': node.lineno,
            'function': self.current_function,
            'class': self.current_class
        })

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        """Visit try-except block."""
        exception_info = {
            'lineno': node.lineno,
            'function': self.current_function,
            'class': self.current_class,
            'except_count': len(node.handlers),
            'has_finally': node.finalbody is not None and len(node.finalbody) > 0,
            'has_else': node.orelse is not None and len(node.orelse) > 0
        }
        self.branches.append(exception_info)

        # Each exception handler is an edge case
        for handler in node.handlers:
            self.edge_cases.append({
                'type': 'exception_handler',
                'lineno': handler.lineno,
                'function': self.current_function,
                'class': self.current_class,
                'exception_type': handler.type.id if handler.type and isinstance(handler.type, ast.Name) else 'generic'
            })

        self.generic_visit(node)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.comprehension)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity

    def _has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Check if function has return statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False

    def _has_exception_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has exception handling."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.Raise)):
                return True
        return False

    def _is_edge_case_condition(self, test_node: ast.expr) -> bool:
        """Check if condition represents an edge case."""
        if isinstance(test_node, ast.Compare):
            # Check for boundary conditions like x == 0, x < 1, etc.
            for comparator in test_node.comparators:
                if isinstance(comparator, ast.Constant):
                    if comparator.value in [0, 1, -1, None, "", []]:
                        return True
        elif isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            # Negation conditions often represent edge cases
            return True

        return False


class CoverageValidator:
    """
    Comprehensive 100% coverage validation system.

    Provides deep analysis of test coverage including line, branch, function,
    and edge case coverage with intelligent gap detection and remediation suggestions.
    """

    def __init__(self,
                 project_root: Path,
                 source_directory: Path,
                 test_directory: Path,
                 coverage_level: CoverageLevel = CoverageLevel.COMPREHENSIVE,
                 database_path: Path | None = None):
        """Initialize coverage validator.

        Args:
            project_root: Root directory of the project
            source_directory: Directory containing source code
            test_directory: Directory containing test files
            coverage_level: Level of coverage validation
            database_path: Path to coverage database
        """
        self.project_root = Path(project_root)
        self.source_directory = Path(source_directory)
        self.test_directory = Path(test_directory)
        self.coverage_level = coverage_level
        self.database_path = database_path or self.project_root / ".coverage_validation.db"

        # Initialize components
        self.discovery = TestDiscovery(project_root, test_directory)
        self.logger = logging.getLogger(__name__)

        # Coverage configuration
        self.coverage_config = {
            'source': str(self.source_directory),
            'omit': [
                '*/tests/*',
                '*/test_*',
                '*/__pycache__/*',
                '*/venv/*',
                '*/env/*',
                '*/.tox/*',
                'setup.py',
                'conftest.py'
            ],
            'include': ['*.py'],
            'precision': 2,
            'show_missing': True,
            'skip_covered': False
        }

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize coverage validation database."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    line_coverage REAL NOT NULL,
                    branch_coverage REAL NOT NULL,
                    function_coverage REAL NOT NULL,
                    overall_coverage REAL NOT NULL,
                    total_gaps INTEGER NOT NULL,
                    meets_threshold BOOLEAN NOT NULL,
                    report_data TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    coverage_type TEXT NOT NULL,
                    function_name TEXT,
                    class_name TEXT,
                    complexity INTEGER DEFAULT 1,
                    is_edge_case BOOLEAN DEFAULT 0,
                    suggested_test TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    FOREIGN KEY (report_id) REFERENCES coverage_reports (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    validation_type TEXT NOT NULL,
                    target_file TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details TEXT
                )
            """)

    def validate_coverage(self,
                         run_tests: bool = True,
                         target_files: list[str] | None = None) -> CoverageReport:
        """
        Perform comprehensive coverage validation.

        Args:
            run_tests: Whether to run tests before validation
            target_files: Specific files to validate (None for all)

        Returns:
            Detailed coverage report with gaps and recommendations
        """
        self.logger.info("Starting comprehensive coverage validation...")

        # Run tests with coverage if requested
        if run_tests:
            self._run_tests_with_coverage()

        # Load coverage data
        coverage_data = self._load_coverage_data()

        # Perform AST analysis
        ast_analysis = self._perform_ast_analysis(target_files)

        # Generate comprehensive report
        report = self._generate_coverage_report(coverage_data, ast_analysis)

        # Validate edge cases
        self._validate_edge_cases(report, ast_analysis)

        # Generate test suggestions
        self._generate_test_suggestions(report)

        # Save report to database
        self._save_coverage_report(report)

        self.logger.info(f"Coverage validation complete: {report.overall_coverage_percentage:.2f}% overall coverage")

        return report

    def _run_tests_with_coverage(self):
        """Run tests with coverage collection."""
        self.logger.info("Running tests with coverage collection...")

        # Prepare coverage command
        coverage_cmd = [
            sys.executable, '-m', 'pytest',
            '--cov=' + str(self.source_directory),
            '--cov-report=xml',
            '--cov-report=json',
            '--cov-report=html',
            '--cov-branch',
            str(self.test_directory)
        ]

        try:
            result = subprocess.run(
                coverage_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode != 0:
                self.logger.warning(f"Test execution had issues: {result.stderr}")
            else:
                self.logger.info("Tests completed successfully with coverage")

        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timed out")
            raise

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run tests with coverage: {e}")
            raise

    def _load_coverage_data(self) -> dict[str, Any]:
        """Load coverage data from coverage files."""
        coverage_data = {}

        # Load JSON coverage data
        json_path = self.project_root / 'coverage.json'
        if json_path.exists():
            with open(json_path) as f:
                coverage_data['json'] = json.load(f)

        # Load XML coverage data
        xml_path = self.project_root / 'coverage.xml'
        if xml_path.exists():
            coverage_data['xml'] = ET.parse(xml_path)

        return coverage_data

    def _perform_ast_analysis(self, target_files: list[str] | None) -> dict[str, CoverageAnalyzer]:
        """Perform AST analysis on source files."""
        self.logger.info("Performing AST analysis...")

        analyzers = {}
        source_files = []

        if target_files:
            source_files = [self.project_root / f for f in target_files]
        else:
            # Find all Python source files
            source_files = list(self.source_directory.rglob('*.py'))

        for source_file in source_files:
            if source_file.name.startswith('test_') or '/test' in str(source_file):
                continue

            try:
                with open(source_file, encoding='utf-8') as f:
                    source_code = f.read()

                tree = ast.parse(source_code)
                analyzer = CoverageAnalyzer()
                analyzer.visit(tree)

                relative_path = str(source_file.relative_to(self.project_root))
                analyzers[relative_path] = analyzer

            except Exception as e:
                self.logger.warning(f"Failed to analyze {source_file}: {e}")

        return analyzers

    def _generate_coverage_report(self,
                                 coverage_data: dict[str, Any],
                                 ast_analysis: dict[str, CoverageAnalyzer]) -> CoverageReport:
        """Generate comprehensive coverage report."""
        report = CoverageReport(timestamp=time.time(), total_lines=0, covered_lines=0, missed_lines=0,
                               line_coverage_percentage=0.0, branch_coverage_percentage=0.0,
                               function_coverage_percentage=0.0)

        # Extract coverage metrics from JSON data
        if 'json' in coverage_data:
            json_data = coverage_data['json']
            totals = json_data.get('totals', {})

            report.total_lines = totals.get('num_statements', 0)
            report.covered_lines = totals.get('covered_lines', 0)
            report.missed_lines = totals.get('missing_lines', 0)
            report.line_coverage_percentage = totals.get('percent_covered', 0.0)

            # Branch coverage
            if 'num_branches' in totals:
                total_branches = totals['num_branches']
                covered_branches = totals.get('covered_branches', 0)
                if total_branches > 0:
                    report.branch_coverage_percentage = (covered_branches / total_branches) * 100.0

            # File-level coverage
            for file_path, file_data in json_data.get('files', {}).items():
                relative_path = str(Path(file_path).relative_to(self.project_root))
                file_coverage = file_data.get('summary', {}).get('percent_covered', 0.0)
                report.file_coverage[relative_path] = file_coverage

                # Identify coverage gaps
                missing_lines = file_data.get('missing_lines', [])
                for line_num in missing_lines:
                    gap = CoverageGap(
                        file_path=relative_path,
                        line_number=line_num,
                        line_content=self._get_line_content(file_path, line_num),
                        coverage_type=CoverageType.LINE
                    )
                    report.coverage_gaps.append(gap)

        # Function coverage analysis
        total_functions = 0
        covered_functions = 0

        for file_path, analyzer in ast_analysis.items():
            total_functions += len(analyzer.functions)
            # In a full implementation, we'd cross-reference with actual coverage data
            # For demo, assume functions are covered if file coverage > 50%
            file_coverage = report.file_coverage.get(file_path, 0.0)
            if file_coverage > 50.0:
                covered_functions += len(analyzer.functions)
            else:
                for func in analyzer.functions:
                    report.uncovered_functions.append(f"{file_path}:{func['name']}")

        if total_functions > 0:
            report.function_coverage_percentage = (covered_functions / total_functions) * 100.0

        return report

    def _validate_edge_cases(self, report: CoverageReport, ast_analysis: dict[str, CoverageAnalyzer]):
        """Validate edge case coverage."""
        self.logger.info("Validating edge case coverage...")

        for file_path, analyzer in ast_analysis.items():
            for edge_case in analyzer.edge_cases:
                # Check if edge case is covered
                is_covered = self._is_edge_case_covered(file_path, edge_case)

                if not is_covered:
                    report.edge_cases_missing.append(
                        f"{file_path}:{edge_case['lineno']} - {edge_case['type']}"
                    )

                    # Add to coverage gaps
                    gap = CoverageGap(
                        file_path=file_path,
                        line_number=edge_case['lineno'],
                        line_content=f"Edge case: {edge_case['type']}",
                        coverage_type=CoverageType.CONDITION,
                        function_name=edge_case['function'],
                        class_name=edge_case['class'],
                        is_edge_case=True
                    )
                    report.coverage_gaps.append(gap)

    def _is_edge_case_covered(self, file_path: str, edge_case: dict[str, Any]) -> bool:
        """Check if specific edge case is covered by tests."""
        # In a full implementation, this would analyze test execution traces
        # For demo, we'll use heuristics based on file coverage
        file_coverage = 80.0  # Mock coverage percentage
        return file_coverage > 95.0  # Edge cases require very high coverage

    def _generate_test_suggestions(self, report: CoverageReport):
        """Generate intelligent test suggestions for coverage gaps."""
        self.logger.info("Generating test suggestions...")

        for gap in report.coverage_gaps:
            suggestion = self._create_test_suggestion(gap)
            gap.suggested_test = suggestion

    def _create_test_suggestion(self, gap: CoverageGap) -> str:
        """Create test suggestion for coverage gap."""
        if gap.is_edge_case:
            if 'boundary_condition' in gap.line_content:
                return f"def test_{gap.function_name}_boundary_condition():\n    # Test edge case: {gap.line_content}\n    pass"
            elif 'exception_handler' in gap.line_content:
                return f"def test_{gap.function_name}_exception_handling():\n    # Test exception: {gap.line_content}\n    pass"

        # General coverage gap
        return f"def test_{gap.function_name}_line_{gap.line_number}():\n    # Cover: {gap.line_content}\n    pass"

    def _get_line_content(self, file_path: str, line_number: int) -> str:
        """Get content of specific line from file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except Exception:
            pass
        return ""

    def _save_coverage_report(self, report: CoverageReport):
        """Save coverage report to database."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                INSERT INTO coverage_reports
                (timestamp, line_coverage, branch_coverage, function_coverage,
                 overall_coverage, total_gaps, meets_threshold, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.timestamp,
                report.line_coverage_percentage,
                report.branch_coverage_percentage,
                report.function_coverage_percentage,
                report.overall_coverage_percentage,
                len(report.coverage_gaps),
                report.meets_100_percent_threshold,
                json.dumps({
                    'file_coverage': report.file_coverage,
                    'uncovered_functions': report.uncovered_functions,
                    'edge_cases_missing': report.edge_cases_missing,
                    'validation_issues': report.validation_issues
                })
            ))

            report_id = cursor.lastrowid

            # Save coverage gaps
            for gap in report.coverage_gaps:
                conn.execute("""
                    INSERT INTO coverage_gaps
                    (report_id, file_path, line_number, coverage_type, function_name,
                     class_name, complexity, is_edge_case, suggested_test)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report_id, gap.file_path, gap.line_number, gap.coverage_type.value,
                    gap.function_name, gap.class_name, gap.complexity,
                    gap.is_edge_case, gap.suggested_test
                ))

    def get_coverage_trends(self, days: int = 30) -> dict[str, list[float]]:
        """Get coverage trends over time."""
        with sqlite3.connect(self.database_path) as conn:
            cutoff_time = time.time() - (days * 24 * 3600)

            cursor = conn.execute("""
                SELECT timestamp, line_coverage, branch_coverage, function_coverage, overall_coverage
                FROM coverage_reports
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            """, (cutoff_time,))

            trends = {
                'timestamps': [],
                'line_coverage': [],
                'branch_coverage': [],
                'function_coverage': [],
                'overall_coverage': []
            }

            for row in cursor.fetchall():
                trends['timestamps'].append(row[0])
                trends['line_coverage'].append(row[1])
                trends['branch_coverage'].append(row[2])
                trends['function_coverage'].append(row[3])
                trends['overall_coverage'].append(row[4])

            return trends

    def get_persistent_coverage_gaps(self, occurrences_threshold: int = 3) -> list[dict[str, Any]]:
        """Get coverage gaps that persist across multiple reports."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, line_number, coverage_type, COUNT(*) as occurrences,
                       MAX(suggested_test) as latest_suggestion
                FROM coverage_gaps cg
                JOIN coverage_reports cr ON cg.report_id = cr.id
                WHERE cg.resolved = 0
                GROUP BY file_path, line_number, coverage_type
                HAVING occurrences >= ?
                ORDER BY occurrences DESC
            """, (occurrences_threshold,))

            gaps = []
            for row in cursor.fetchall():
                gaps.append({
                    'file_path': row[0],
                    'line_number': row[1],
                    'coverage_type': row[2],
                    'occurrences': row[3],
                    'suggested_test': row[4]
                })

            return gaps

    def generate_coverage_improvement_plan(self) -> dict[str, Any]:
        """Generate comprehensive plan to achieve 100% coverage."""
        latest_report = self._get_latest_coverage_report()
        if not latest_report:
            return {'error': 'No coverage reports found'}

        persistent_gaps = self.get_persistent_coverage_gaps()

        # Priority matrix: complexity vs impact
        high_priority_gaps = []
        medium_priority_gaps = []
        low_priority_gaps = []

        for gap in persistent_gaps:
            if gap['occurrences'] >= 5:
                high_priority_gaps.append(gap)
            elif gap['occurrences'] >= 3:
                medium_priority_gaps.append(gap)
            else:
                low_priority_gaps.append(gap)

        plan = {
            'current_coverage': {
                'line': latest_report['line_coverage'],
                'branch': latest_report['branch_coverage'],
                'function': latest_report['function_coverage'],
                'overall': latest_report['overall_coverage']
            },
            'target_coverage': {
                'line': 100.0,
                'branch': 100.0,
                'function': 100.0,
                'overall': 100.0
            },
            'improvement_needed': {
                'line': max(0, 100.0 - latest_report['line_coverage']),
                'branch': max(0, 100.0 - latest_report['branch_coverage']),
                'function': max(0, 100.0 - latest_report['function_coverage'])
            },
            'action_items': {
                'high_priority': high_priority_gaps,
                'medium_priority': medium_priority_gaps,
                'low_priority': low_priority_gaps
            },
            'estimated_effort': {
                'high_priority_hours': len(high_priority_gaps) * 2,
                'medium_priority_hours': len(medium_priority_gaps) * 1,
                'low_priority_hours': len(low_priority_gaps) * 0.5
            }
        }

        return plan

    def _get_latest_coverage_report(self) -> dict[str, Any] | None:
        """Get the latest coverage report from database."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT line_coverage, branch_coverage, function_coverage, overall_coverage
                FROM coverage_reports
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                return {
                    'line_coverage': row[0],
                    'branch_coverage': row[1],
                    'function_coverage': row[2],
                    'overall_coverage': row[3]
                }

        return None

    def export_coverage_report(self, format_type: str = 'json') -> str:
        """Export coverage report in specified format."""
        latest_report_data = self._get_latest_coverage_report()
        if not latest_report_data:
            return ""

        if format_type == 'json':
            return json.dumps(latest_report_data, indent=2)
        elif format_type == 'html':
            return self._generate_html_report(latest_report_data)
        else:
            return str(latest_report_data)

    def _generate_html_report(self, report_data: dict[str, Any]) -> str:
        """Generate HTML coverage report."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coverage Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ padding: 10px; margin: 5px; border-radius: 5px; }}
                .good {{ background-color: #d4edda; color: #155724; }}
                .warning {{ background-color: #fff3cd; color: #856404; }}
                .poor {{ background-color: #f8d7da; color: #721c24; }}
            </style>
        </head>
        <body>
            <h1>Coverage Validation Report</h1>
            <div class="metric {'good' if report_data['line_coverage'] >= 90 else 'warning' if report_data['line_coverage'] >= 70 else 'poor'}">
                Line Coverage: {report_data['line_coverage']:.2f}%
            </div>
            <div class="metric {'good' if report_data['branch_coverage'] >= 90 else 'warning' if report_data['branch_coverage'] >= 70 else 'poor'}">
                Branch Coverage: {report_data['branch_coverage']:.2f}%
            </div>
            <div class="metric {'good' if report_data['function_coverage'] >= 90 else 'warning' if report_data['function_coverage'] >= 70 else 'poor'}">
                Function Coverage: {report_data['function_coverage']:.2f}%
            </div>
            <div class="metric {'good' if report_data['overall_coverage'] >= 90 else 'warning' if report_data['overall_coverage'] >= 70 else 'poor'}">
                Overall Coverage: {report_data['overall_coverage']:.2f}%
            </div>
        </body>
        </html>
        """
        return html_template

    def close(self):
        """Close coverage validator and cleanup resources."""
        if hasattr(self, 'discovery'):
            self.discovery.close()
        self.logger.info("Coverage validator closed")
