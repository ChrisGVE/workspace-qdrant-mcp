"""
Test Discovery Engine

Automatically discovers test cases across a codebase, categorizes them,
identifies coverage gaps, and suggests new test scenarios with intelligent
false positive filtering and discovery failure handling.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import re
import concurrent.futures
from collections import defaultdict
import json
import sqlite3
from datetime import datetime

from ..documentation.parser import TestFileParser, TestFileInfo, TestMetadata, TestType
from .categorizer import TestCategorizer, CoverageAnalysis, TestCategory, CoverageGap

logger = logging.getLogger(__name__)


@dataclass
class SourceCodeAnalysis:
    """Analysis of source code for test generation suggestions."""
    file_path: Path
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    public_methods: List[str] = field(default_factory=list)
    complexity_score: int = 1
    dependencies: Set[str] = field(default_factory=set)
    potential_edge_cases: List[str] = field(default_factory=list)
    error_conditions: List[str] = field(default_factory=list)


@dataclass
class TestSuggestion:
    """Suggestion for a new test case."""
    name: str
    description: str
    category: TestCategory
    priority: int  # 1-10
    source_file: Optional[Path] = None
    target_function: Optional[str] = None
    test_type: str = "unit"
    estimated_effort: str = "medium"  # low, medium, high
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    confidence: float = 0.8


@dataclass
class DiscoveryResult:
    """Complete test discovery results."""
    discovered_tests: List[TestMetadata] = field(default_factory=list)
    coverage_analysis: Optional[CoverageAnalysis] = None
    test_suggestions: List[TestSuggestion] = field(default_factory=list)
    source_analysis: List[SourceCodeAnalysis] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    discovery_errors: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class SourceCodeAnalyzer:
    """Analyzes source code to suggest test cases."""

    def __init__(self):
        """Initialize source code analyzer."""
        self.complexity_patterns = [
            (r'\bif\b', 1),
            (r'\belse\b', 1),
            (r'\belif\b', 1),
            (r'\bfor\b', 2),
            (r'\bwhile\b', 2),
            (r'\btry\b', 2),
            (r'\bexcept\b', 1),
            (r'\bwith\b', 1),
            (r'\braise\b', 1)
        ]

        self.edge_case_patterns = [
            r'\bempty\b.*\blist\b',
            r'\bnull\b',
            r'\bNone\b',
            r'\b0\b',
            r'\b-\d+\b',
            r'\blen\(',
            r'\.split\(',
            r'\.strip\(',
            r'\bmax\b',
            r'\bmin\b'
        ]

        self.error_patterns = [
            r'\braise\b',
            r'\bexcept\b',
            r'\bassert\b',
            r'\bvalidat\b',
            r'\bcheck\b',
            r'if.*not\b',
            r'if.*is None\b'
        ]

    def analyze_source_file(self, file_path: Path) -> Optional[SourceCodeAnalysis]:
        """Analyze a source code file for testing opportunities."""
        if not file_path.exists() or not file_path.suffix == '.py':
            return None

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)
            analysis = SourceCodeAnalysis(file_path=file_path)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis.classes.append(node.name)

                    # Extract public methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not item.name.startswith('_'):
                                analysis.public_methods.append(f"{node.name}.{item.name}")

                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        analysis.functions.append(node.name)

            # Calculate complexity
            analysis.complexity_score = self._calculate_complexity(content)

            # Find potential edge cases
            analysis.potential_edge_cases = self._find_edge_cases(content)

            # Find error conditions
            analysis.error_conditions = self._find_error_conditions(content)

            # Extract dependencies
            analysis.dependencies = self._extract_dependencies(tree)

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze source file {file_path}: {e}")
            return None

    def _calculate_complexity(self, content: str) -> int:
        """Calculate complexity score based on patterns."""
        complexity = 1
        for pattern, weight in self.complexity_patterns:
            matches = len(re.findall(pattern, content))
            complexity += matches * weight
        return min(complexity, 20)  # Cap at 20

    def _find_edge_cases(self, content: str) -> List[str]:
        """Find potential edge cases in code."""
        edge_cases = []
        for pattern in self.edge_case_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:3]:  # Limit to avoid noise
                edge_cases.append(f"Test edge case: {match}")
        return edge_cases

    def _find_error_conditions(self, content: str) -> List[str]:
        """Find error handling opportunities."""
        errors = []
        for pattern in self.error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                errors.append(f"Test error condition: {pattern}")
        return errors[:5]  # Limit to avoid noise

    def _extract_dependencies(self, tree: ast.AST) -> Set[str]:
        """Extract dependencies from imports."""
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module)
        return dependencies


class TestSuggestionEngine:
    """Generates intelligent test suggestions based on analysis."""

    def __init__(self):
        """Initialize suggestion engine."""
        self.suggestion_templates = self._build_suggestion_templates()

    def generate_suggestions(self,
                           coverage_analysis: CoverageAnalysis,
                           source_analyses: List[SourceCodeAnalysis],
                           existing_tests: List[TestMetadata]) -> List[TestSuggestion]:
        """Generate test suggestions based on analysis."""
        suggestions = []

        # Generate suggestions from coverage gaps
        for gap in coverage_analysis.gaps:
            gap_suggestions = self._suggest_for_gap(gap, source_analyses)
            suggestions.extend(gap_suggestions)

        # Generate suggestions from source code analysis
        for source_analysis in source_analyses:
            source_suggestions = self._suggest_for_source(source_analysis, existing_tests)
            suggestions.extend(source_suggestions)

        # Filter and rank suggestions
        suggestions = self._filter_suggestions(suggestions, existing_tests)
        suggestions = self._rank_suggestions(suggestions)

        return suggestions[:20]  # Limit to top 20

    def _suggest_for_gap(self, gap: CoverageGap, source_analyses: List[SourceCodeAnalysis]) -> List[TestSuggestion]:
        """Generate suggestions for coverage gaps."""
        suggestions = []

        template = self.suggestion_templates.get(gap.category)
        if not template:
            return suggestions

        for i, suggested_test in enumerate(gap.suggested_tests[:3]):
            suggestion = TestSuggestion(
                name=f"test_{gap.category.value}_{i+1}",
                description=suggested_test,
                category=gap.category,
                priority=gap.priority,
                estimated_effort=gap.effort_estimate,
                confidence=0.7
            )
            suggestions.append(suggestion)

        return suggestions

    def _suggest_for_source(self, source_analysis: SourceCodeAnalysis,
                           existing_tests: List[TestMetadata]) -> List[TestSuggestion]:
        """Generate suggestions based on source code analysis."""
        suggestions = []

        # Get existing test names for filtering
        existing_names = {test.name.lower() for test in existing_tests}

        # Suggest tests for public functions
        for func in source_analysis.functions[:5]:  # Limit to avoid noise
            test_name = f"test_{func.lower()}"
            if test_name not in existing_names:
                suggestions.append(TestSuggestion(
                    name=test_name,
                    description=f"Test {func} function with valid inputs",
                    category=TestCategory.UNIT_CORE,
                    priority=6,
                    source_file=source_analysis.file_path,
                    target_function=func,
                    confidence=0.8
                ))

        # Suggest tests for public methods
        for method in source_analysis.public_methods[:3]:
            class_name, method_name = method.split('.', 1)
            test_name = f"test_{class_name.lower()}_{method_name.lower()}"
            if test_name not in existing_names:
                suggestions.append(TestSuggestion(
                    name=test_name,
                    description=f"Test {method} method functionality",
                    category=TestCategory.UNIT_CORE,
                    priority=5,
                    source_file=source_analysis.file_path,
                    target_function=method,
                    confidence=0.7
                ))

        # Suggest edge case tests
        if source_analysis.potential_edge_cases:
            suggestions.append(TestSuggestion(
                name=f"test_{source_analysis.file_path.stem}_edge_cases",
                description=f"Test edge cases for {source_analysis.file_path.name}",
                category=TestCategory.UNIT_EDGE_CASE,
                priority=7,
                source_file=source_analysis.file_path,
                confidence=0.6,
                preconditions=source_analysis.potential_edge_cases[:3]
            ))

        # Suggest error handling tests
        if source_analysis.error_conditions:
            suggestions.append(TestSuggestion(
                name=f"test_{source_analysis.file_path.stem}_error_handling",
                description=f"Test error handling for {source_analysis.file_path.name}",
                category=TestCategory.UNIT_ERROR_HANDLING,
                priority=8,
                source_file=source_analysis.file_path,
                confidence=0.7,
                preconditions=source_analysis.error_conditions[:3]
            ))

        return suggestions

    def _filter_suggestions(self, suggestions: List[TestSuggestion],
                           existing_tests: List[TestMetadata]) -> List[TestSuggestion]:
        """Filter out low-quality suggestions."""
        existing_names = {test.name.lower() for test in existing_tests}

        filtered = []
        for suggestion in suggestions:
            # Skip if test already exists
            if suggestion.name.lower() in existing_names:
                continue

            # Skip if confidence too low
            if suggestion.confidence < 0.5:
                continue

            # Skip if description too generic
            if len(suggestion.description.split()) < 3:
                continue

            filtered.append(suggestion)

        return filtered

    def _rank_suggestions(self, suggestions: List[TestSuggestion]) -> List[TestSuggestion]:
        """Rank suggestions by priority and confidence."""
        return sorted(suggestions,
                     key=lambda s: (s.priority * s.confidence, s.priority),
                     reverse=True)

    def _build_suggestion_templates(self) -> Dict[TestCategory, Dict[str, Any]]:
        """Build templates for test suggestions."""
        return {
            TestCategory.UNIT_CORE: {
                'priority': 8,
                'effort': 'low',
                'examples': [
                    'Test function with valid inputs',
                    'Test method return values',
                    'Test object state changes'
                ]
            },
            TestCategory.UNIT_ERROR_HANDLING: {
                'priority': 9,
                'effort': 'medium',
                'examples': [
                    'Test exception handling',
                    'Test invalid input handling',
                    'Test error recovery'
                ]
            },
            TestCategory.INTEGRATION_API: {
                'priority': 6,
                'effort': 'medium',
                'examples': [
                    'Test API endpoint responses',
                    'Test API error handling',
                    'Test API authentication'
                ]
            }
        }


class TestDiscoveryEngine:
    """
    Main test discovery engine.

    Automatically discovers existing tests, analyzes coverage patterns,
    identifies gaps, and suggests new test cases with comprehensive
    error handling and false positive filtering.
    """

    def __init__(self,
                 cache_db: Optional[Path] = None,
                 max_workers: int = 4,
                 enable_source_analysis: bool = True):
        """
        Initialize test discovery engine.

        Args:
            cache_db: Path to SQLite cache database
            max_workers: Maximum parallel workers
            enable_source_analysis: Whether to analyze source code for suggestions
        """
        self.parser = TestFileParser()
        self.categorizer = TestCategorizer()
        self.source_analyzer = SourceCodeAnalyzer() if enable_source_analysis else None
        self.suggestion_engine = TestSuggestionEngine()
        self.max_workers = max_workers
        self.cache_db = cache_db

        if cache_db:
            self._initialize_cache_db()

    def discover_tests(self,
                      project_root: Union[str, Path],
                      test_patterns: List[str] = None,
                      source_patterns: List[str] = None,
                      exclude_patterns: List[str] = None) -> DiscoveryResult:
        """
        Discover tests and analyze coverage across a project.

        Args:
            project_root: Root directory of the project
            test_patterns: Patterns to match test files
            source_patterns: Patterns to match source files
            exclude_patterns: Patterns to exclude

        Returns:
            Complete discovery results with suggestions

        Raises:
            ValueError: If project root is invalid
        """
        project_root = Path(project_root)
        if not project_root.exists() or not project_root.is_dir():
            raise ValueError(f"Invalid project root: {project_root}")

        result = DiscoveryResult()

        try:
            # Set default patterns
            if test_patterns is None:
                test_patterns = ['test_*.py', '*_test.py', 'tests.py']
            if source_patterns is None:
                source_patterns = ['*.py']
            if exclude_patterns is None:
                exclude_patterns = ['__pycache__', '*.pyc', 'venv', '.env']

            # Discover test files
            logger.info(f"Discovering tests in {project_root}")
            test_files = self._find_test_files(project_root, test_patterns, exclude_patterns)
            result.statistics['test_files_found'] = len(test_files)

            # Parse test files
            file_infos = self._parse_test_files(test_files, result)

            # Collect all discovered tests
            for file_info in file_infos:
                result.discovered_tests.extend(file_info.tests)

            result.statistics['total_tests'] = len(result.discovered_tests)

            # Analyze coverage
            logger.info("Analyzing test coverage patterns")
            result.coverage_analysis = self.categorizer.categorize_tests(file_infos)

            # Analyze source code if enabled
            source_analyses = []
            if self.source_analyzer:
                logger.info("Analyzing source code for test suggestions")
                source_files = self._find_source_files(project_root, source_patterns, exclude_patterns)
                source_analyses = self._analyze_source_files(source_files, result)

            result.source_analysis = source_analyses
            result.statistics['source_files_analyzed'] = len(source_analyses)

            # Generate test suggestions
            logger.info("Generating test suggestions")
            result.test_suggestions = self.suggestion_engine.generate_suggestions(
                result.coverage_analysis,
                source_analyses,
                result.discovered_tests
            )

            result.statistics['suggestions_generated'] = len(result.test_suggestions)

            # Cache results if enabled
            if self.cache_db:
                self._cache_discovery_results(project_root, result)

            logger.info(f"Discovery complete: {len(result.discovered_tests)} tests, "
                       f"{len(result.test_suggestions)} suggestions")

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            result.discovery_errors.append(f"Discovery failed: {e}")

        return result

    def _find_test_files(self, root: Path, patterns: List[str], excludes: List[str]) -> List[Path]:
        """Find test files matching patterns."""
        test_files = []

        for pattern in patterns:
            files = root.rglob(pattern)
            for file_path in files:
                # Check exclusions
                if any(exclude in str(file_path) for exclude in excludes):
                    continue

                # Additional validation
                if file_path.is_file() and file_path.suffix == '.py':
                    test_files.append(file_path)

        # Remove duplicates and sort
        return sorted(set(test_files))

    def _find_source_files(self, root: Path, patterns: List[str], excludes: List[str]) -> List[Path]:
        """Find source files for analysis."""
        source_files = []
        excludes = excludes + ['test_', '_test.py', 'tests.py']  # Exclude test files

        for pattern in patterns:
            files = root.rglob(pattern)
            for file_path in files:
                # Check exclusions
                if any(exclude in str(file_path) for exclude in excludes):
                    continue

                if file_path.is_file() and file_path.suffix == '.py':
                    source_files.append(file_path)

        return sorted(set(source_files))

    def _parse_test_files(self, test_files: List[Path], result: DiscoveryResult) -> List[TestFileInfo]:
        """Parse test files with error handling."""
        file_infos = []

        if len(test_files) > 1 and self.max_workers > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._safe_parse_file, file_path): file_path
                    for file_path in test_files
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_info = future.result()
                        if file_info:
                            file_infos.append(file_info)
                    except Exception as e:
                        error_msg = f"Failed to parse {file_path}: {e}"
                        logger.error(error_msg)
                        result.discovery_errors.append(error_msg)
        else:
            # Sequential processing
            for file_path in test_files:
                try:
                    file_info = self._safe_parse_file(file_path)
                    if file_info:
                        file_infos.append(file_info)
                except Exception as e:
                    error_msg = f"Failed to parse {file_path}: {e}"
                    logger.error(error_msg)
                    result.discovery_errors.append(error_msg)

        return file_infos

    def _safe_parse_file(self, file_path: Path) -> Optional[TestFileInfo]:
        """Safely parse a test file with error handling."""
        try:
            return self.parser.parse_file(file_path)
        except Exception as e:
            logger.warning(f"Parse failed for {file_path}: {e}")
            # Return minimal info for failed files
            return TestFileInfo(
                file_path=file_path,
                parse_errors=[f"Parse failed: {e}"]
            )

    def _analyze_source_files(self, source_files: List[Path], result: DiscoveryResult) -> List[SourceCodeAnalysis]:
        """Analyze source files for test suggestions."""
        if not self.source_analyzer:
            return []

        analyses = []

        for file_path in source_files[:50]:  # Limit to avoid overwhelming analysis
            try:
                analysis = self.source_analyzer.analyze_source_file(file_path)
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                error_msg = f"Source analysis failed for {file_path}: {e}"
                logger.warning(error_msg)
                result.discovery_errors.append(error_msg)

        return analyses

    def _initialize_cache_db(self) -> None:
        """Initialize cache database."""
        if not self.cache_db:
            return

        conn = sqlite3.connect(self.cache_db)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS discovery_cache (
                    project_root TEXT PRIMARY KEY,
                    discovery_data TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def _cache_discovery_results(self, project_root: Path, result: DiscoveryResult) -> None:
        """Cache discovery results."""
        if not self.cache_db:
            return

        try:
            # Serialize result (simplified)
            cache_data = {
                'test_count': len(result.discovered_tests),
                'suggestion_count': len(result.test_suggestions),
                'statistics': result.statistics,
                'cached_at': datetime.now().isoformat()
            }

            conn = sqlite3.connect(self.cache_db)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO discovery_cache
                    (project_root, discovery_data, updated_at)
                    VALUES (?, ?, julianday('now'))
                ''', (str(project_root), json.dumps(cache_data)))
                conn.commit()
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to cache discovery results: {e}")

    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics from cache."""
        if not self.cache_db or not Path(self.cache_db).exists():
            return {}

        conn = sqlite3.connect(self.cache_db)
        try:
            cursor = conn.execute('''
                SELECT COUNT(*) as total_discoveries,
                       MAX(updated_at) as last_discovery
                FROM discovery_cache
            ''')
            stats = cursor.fetchone()

            return {
                'total_discoveries': stats[0] or 0,
                'last_discovery': stats[1],
                'cache_db_path': str(self.cache_db)
            }

        except Exception as e:
            logger.error(f"Failed to get discovery statistics: {e}")
            return {}
        finally:
            conn.close()