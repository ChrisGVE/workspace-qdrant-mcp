"""
Intelligent Test Discovery and Categorization System

This module provides advanced test discovery capabilities that go beyond pytest's
built-in discovery. It analyzes test files, categorizes them by complexity and type,
identifies dependencies, and creates optimal execution plans.

Features:
- Automated test discovery with pattern matching
- Complexity analysis based on code structure and dependencies
- Test categorization by type (unit, integration, e2e, performance)
- Dependency analysis for execution ordering
- Resource requirement detection
- Flaky test identification through historical data
"""

import ast
import inspect
import importlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict,
    List,
    Set,
    Optional,
    Tuple,
    Any,
    Union,
    Callable,
    NamedTuple
)
import concurrent.futures
import hashlib

import pytest


class TestCategory(Enum):
    """Test category classification."""
    UNIT = auto()
    INTEGRATION = auto()
    E2E = auto()
    FUNCTIONAL = auto()
    PERFORMANCE = auto()
    SMOKE = auto()
    REGRESSION = auto()
    SECURITY = auto()
    STRESS = auto()
    COMPATIBILITY = auto()
    UNKNOWN = auto()


class TestComplexity(Enum):
    """Test complexity levels."""
    TRIVIAL = auto()    # Simple unit tests, no external dependencies
    LOW = auto()        # Basic unit tests with minimal mocking
    MEDIUM = auto()     # Integration tests, moderate setup/teardown
    HIGH = auto()       # Complex integration, multiple components
    EXTREME = auto()    # E2E tests, full system integration


class ResourceRequirement(Enum):
    """Types of resources required by tests."""
    NONE = auto()
    FILESYSTEM = auto()
    NETWORK = auto()
    DATABASE = auto()
    EXTERNAL_SERVICE = auto()
    GPU = auto()
    HIGH_MEMORY = auto()
    LONG_RUNNING = auto()


@dataclass
class TestMetadata:
    """Comprehensive test metadata."""
    name: str
    file_path: Path
    category: TestCategory
    complexity: TestComplexity
    resources: Set[ResourceRequirement] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0
    markers: Set[str] = field(default_factory=set)
    fixtures: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    mocks: Set[str] = field(default_factory=set)
    async_test: bool = False
    parametrized: bool = False
    expected_failures: Set[str] = field(default_factory=set)
    flaky_history: List[Dict[str, Any]] = field(default_factory=list)
    coverage_impact: float = 0.0
    last_modified: float = 0.0
    source_hash: str = ""


class TestDiscoveryError(Exception):
    """Custom exception for test discovery errors."""
    pass


class TestASTVisitor(ast.NodeVisitor):
    """AST visitor for analyzing test code structure."""

    def __init__(self):
        self.imports = set()
        self.mocks = set()
        self.fixtures = set()
        self.async_functions = set()
        self.parametrized = set()
        self.complexity_score = 0
        self.resource_requirements = set()

    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
            self._analyze_import_for_resources(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from-import statements."""
        if node.module:
            self.imports.add(node.module)
            self._analyze_import_for_resources(node.module)

            # Track specific imports
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports.add(full_name)

                # Detect mocking frameworks
                if "mock" in node.module.lower():
                    self.mocks.add(alias.name)

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze function definitions."""
        if node.name.startswith('test_') or any(
            d.id == 'pytest' for d in node.decorator_list
            if isinstance(d, ast.Name)
        ):
            # Check for async tests
            if isinstance(node, ast.AsyncFunctionDef):
                self.async_functions.add(node.name)

            # Check for parametrization
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and
                    hasattr(decorator.func, 'attr') and
                    decorator.func.attr == 'parametrize'):
                    self.parametrized.add(node.name)

            # Analyze complexity
            self.complexity_score += self._calculate_function_complexity(node)

        # Check for fixture definitions
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and
                hasattr(decorator.func, 'attr') and
                decorator.func.attr == 'fixture'):
                self.fixtures.add(node.name)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Handle async function definitions."""
        if node.name.startswith('test_'):
            self.async_functions.add(node.name)
        self.visit_FunctionDef(node)  # Reuse regular function logic

    def _analyze_import_for_resources(self, module_name: str):
        """Analyze imports to determine resource requirements."""
        resource_patterns = {
            ResourceRequirement.FILESYSTEM: ['pathlib', 'os', 'tempfile', 'shutil'],
            ResourceRequirement.NETWORK: ['requests', 'httpx', 'aiohttp', 'urllib'],
            ResourceRequirement.DATABASE: ['sqlite3', 'sqlalchemy', 'qdrant', 'redis'],
            ResourceRequirement.EXTERNAL_SERVICE: ['testcontainers', 'docker'],
            ResourceRequirement.HIGH_MEMORY: ['numpy', 'pandas', 'torch'],
        }

        for requirement, patterns in resource_patterns.items():
            if any(pattern in module_name.lower() for pattern in patterns):
                self.resource_requirements.add(requirement)

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity score for a function."""
        complexity = 0

        # Count control flow statements
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While)):
                complexity += 2
            elif isinstance(child, ast.Try):
                complexity += 3
            elif isinstance(child, ast.With):
                complexity += 1

        # Count nested functions/classes
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child != node:  # Don't count self
                    complexity += 2

        return complexity


class TestDiscovery:
    """Advanced test discovery and categorization system."""

    def __init__(self,
                 project_root: Path,
                 test_dir: Optional[Path] = None,
                 cache_file: Optional[Path] = None):
        """
        Initialize test discovery system.

        Args:
            project_root: Root directory of the project
            test_dir: Directory containing tests (default: project_root/tests)
            cache_file: File to cache discovery results
        """
        self.project_root = Path(project_root)
        self.test_dir = test_dir or self.project_root / "tests"
        self.cache_file = cache_file or self.project_root / ".test_discovery_cache.db"

        self._discovered_tests: Dict[str, TestMetadata] = {}
        self._discovery_lock = threading.RLock()
        self._cache_db: Optional[sqlite3.Connection] = None

        # Category detection patterns
        self._category_patterns = {
            TestCategory.UNIT: [r"unit/", r"test_.*unit", r".*_unit_test"],
            TestCategory.INTEGRATION: [r"integration/", r"test_.*integration", r".*_integration_test"],
            TestCategory.E2E: [r"e2e/", r"end.*to.*end", r"test_.*e2e"],
            TestCategory.FUNCTIONAL: [r"functional/", r"test_.*functional"],
            TestCategory.PERFORMANCE: [r"performance/", r"benchmark", r"test_.*perf"],
            TestCategory.SMOKE: [r"smoke/", r"test_.*smoke"],
            TestCategory.REGRESSION: [r"regression/", r"test_.*regression"],
            TestCategory.SECURITY: [r"security/", r"test_.*security"],
            TestCategory.STRESS: [r"stress/", r"test_.*stress"],
            TestCategory.COMPATIBILITY: [r"compat", r"test_.*compat"],
        }

        # Complexity thresholds
        self._complexity_thresholds = {
            TestComplexity.TRIVIAL: (0, 5),
            TestComplexity.LOW: (6, 15),
            TestComplexity.MEDIUM: (16, 35),
            TestComplexity.HIGH: (36, 60),
            TestComplexity.EXTREME: (61, float('inf')),
        }

        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        try:
            self._cache_db = sqlite3.connect(str(self.cache_file), check_same_thread=False)
            self._cache_db.execute("""
                CREATE TABLE IF NOT EXISTS test_metadata (
                    test_name TEXT PRIMARY KEY,
                    file_path TEXT,
                    category TEXT,
                    complexity TEXT,
                    resources TEXT,
                    dependencies TEXT,
                    estimated_duration REAL,
                    markers TEXT,
                    fixtures TEXT,
                    imports TEXT,
                    mocks TEXT,
                    async_test INTEGER,
                    parametrized INTEGER,
                    expected_failures TEXT,
                    flaky_history TEXT,
                    coverage_impact REAL,
                    last_modified REAL,
                    source_hash TEXT,
                    discovery_timestamp REAL
                )
            """)

            self._cache_db.execute("""
                CREATE TABLE IF NOT EXISTS test_execution_history (
                    test_name TEXT,
                    execution_time REAL,
                    result TEXT,
                    timestamp REAL,
                    error_message TEXT
                )
            """)

            self._cache_db.commit()
        except Exception as e:
            print(f"Warning: Could not initialize cache database: {e}")
            self._cache_db = None

    def discover_tests(self,
                      force_refresh: bool = False,
                      parallel: bool = True,
                      max_workers: Optional[int] = None) -> Dict[str, TestMetadata]:
        """
        Discover and categorize all tests in the project.

        Args:
            force_refresh: Force rediscovery even if cache is valid
            parallel: Use parallel processing for discovery
            max_workers: Maximum number of worker threads

        Returns:
            Dictionary mapping test names to metadata

        Raises:
            TestDiscoveryError: If discovery fails
        """
        with self._discovery_lock:
            if not force_refresh and self._load_from_cache():
                return self._discovered_tests

            try:
                if not self.test_dir.exists():
                    raise TestDiscoveryError(f"Test directory not found: {self.test_dir}")

                # Find all Python test files
                test_files = list(self._find_test_files())

                if not test_files:
                    raise TestDiscoveryError(f"No test files found in {self.test_dir}")

                # Analyze test files
                if parallel and len(test_files) > 1:
                    self._analyze_files_parallel(test_files, max_workers)
                else:
                    self._analyze_files_sequential(test_files)

                # Post-process and calculate relationships
                self._calculate_dependencies()
                self._estimate_durations()
                self._analyze_flaky_patterns()

                # Cache results
                self._save_to_cache()

                return self._discovered_tests.copy()

            except Exception as e:
                raise TestDiscoveryError(f"Test discovery failed: {e}") from e

    def _find_test_files(self) -> List[Path]:
        """Find all Python test files."""
        patterns = [
            "test_*.py",
            "*_test.py",
            "tests.py"
        ]

        test_files = []
        for pattern in patterns:
            test_files.extend(self.test_dir.rglob(pattern))

        # Filter out __pycache__ and other irrelevant files
        return [f for f in test_files
                if "__pycache__" not in str(f) and f.is_file()]

    def _analyze_files_sequential(self, test_files: List[Path]):
        """Analyze test files sequentially."""
        for file_path in test_files:
            try:
                self._analyze_single_file(file_path)
            except Exception as e:
                print(f"Warning: Failed to analyze {file_path}: {e}")

    def _analyze_files_parallel(self, test_files: List[Path], max_workers: Optional[int]):
        """Analyze test files in parallel."""
        max_workers = max_workers or min(len(test_files), 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._analyze_single_file, file_path)
                for file_path in test_files
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Warning: Parallel analysis failed: {e}")

    def _analyze_single_file(self, file_path: Path):
        """Analyze a single test file."""
        try:
            # Read and parse file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            source_hash = hashlib.md5(source_code.encode()).hexdigest()
            last_modified = file_path.stat().st_mtime

            # Parse AST
            try:
                tree = ast.parse(source_code, filename=str(file_path))
            except SyntaxError as e:
                print(f"Warning: Syntax error in {file_path}: {e}")
                return

            # Visit AST to extract information
            visitor = TestASTVisitor()
            visitor.visit(tree)

            # Extract test functions
            test_functions = self._extract_test_functions(tree, file_path)

            # Create metadata for each test
            for test_name, test_node in test_functions:
                full_test_name = f"{file_path.stem}::{test_name}"

                metadata = TestMetadata(
                    name=full_test_name,
                    file_path=file_path,
                    category=self._categorize_test(file_path, test_name),
                    complexity=self._determine_complexity(visitor.complexity_score, visitor),
                    resources=visitor.resource_requirements,
                    dependencies=set(),  # Will be calculated later
                    estimated_duration=0.0,  # Will be calculated later
                    markers=self._extract_markers(test_node),
                    fixtures=visitor.fixtures,
                    imports=visitor.imports,
                    mocks=visitor.mocks,
                    async_test=test_name in visitor.async_functions,
                    parametrized=test_name in visitor.parametrized,
                    expected_failures=set(),  # TODO: Extract from markers
                    flaky_history=[],
                    coverage_impact=0.0,  # Will be calculated later
                    last_modified=last_modified,
                    source_hash=source_hash
                )

                self._discovered_tests[full_test_name] = metadata

        except Exception as e:
            print(f"Warning: Failed to analyze file {file_path}: {e}")

    def _extract_test_functions(self, tree: ast.AST, file_path: Path) -> List[Tuple[str, ast.FunctionDef]]:
        """Extract test functions from AST."""
        test_functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's a test function
                if (node.name.startswith('test_') or
                    any(self._is_pytest_marker(d) for d in node.decorator_list)):
                    test_functions.append((node.name, node))

        return test_functions

    def _is_pytest_marker(self, decorator) -> bool:
        """Check if decorator is a pytest marker."""
        if isinstance(decorator, ast.Call):
            if hasattr(decorator.func, 'attr'):
                return decorator.func.attr in ['mark', 'fixture', 'parametrize']
            if hasattr(decorator.func, 'id'):
                return decorator.func.id in ['pytest']
        return False

    def _extract_markers(self, test_node: ast.FunctionDef) -> Set[str]:
        """Extract pytest markers from test function."""
        markers = set()

        for decorator in test_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if hasattr(decorator.func, 'attr'):
                    if decorator.func.attr == 'mark':
                        # pytest.mark.something
                        if hasattr(decorator.func.value, 'attr'):
                            markers.add(decorator.func.value.attr)
                    elif decorator.func.attr in ['parametrize', 'fixture']:
                        markers.add(decorator.func.attr)
            elif isinstance(decorator, ast.Name):
                markers.add(decorator.id)

        return markers

    def _categorize_test(self, file_path: Path, test_name: str) -> TestCategory:
        """Categorize test based on file path and name patterns."""
        file_str = str(file_path).lower()
        test_name_lower = test_name.lower()

        for category, patterns in self._category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_str) or re.search(pattern, test_name_lower):
                    return category

        return TestCategory.UNKNOWN

    def _determine_complexity(self, base_score: int, visitor: TestASTVisitor) -> TestComplexity:
        """Determine test complexity based on various factors."""
        # Adjust score based on additional factors
        adjusted_score = base_score

        # Resource requirements increase complexity
        adjusted_score += len(visitor.resource_requirements) * 5

        # Async tests are more complex
        adjusted_score += len(visitor.async_functions) * 3

        # Parametrized tests add complexity
        adjusted_score += len(visitor.parametrized) * 2

        # Many imports suggest complexity
        if len(visitor.imports) > 20:
            adjusted_score += 10
        elif len(visitor.imports) > 10:
            adjusted_score += 5

        # Find matching complexity level
        for complexity, (min_score, max_score) in self._complexity_thresholds.items():
            if min_score <= adjusted_score <= max_score:
                return complexity

        return TestComplexity.EXTREME

    def _calculate_dependencies(self):
        """Calculate test dependencies based on fixtures and imports."""
        for test_name, metadata in self._discovered_tests.items():
            dependencies = set()

            # Find tests that provide fixtures this test needs
            for other_name, other_metadata in self._discovered_tests.items():
                if other_name == test_name:
                    continue

                # Check for fixture dependencies
                if metadata.fixtures.intersection(other_metadata.fixtures):
                    dependencies.add(other_name)

                # Check for import dependencies
                if metadata.imports.intersection(other_metadata.imports):
                    # Only add as dependency if significantly overlapping
                    overlap = len(metadata.imports.intersection(other_metadata.imports))
                    if overlap > 3:
                        dependencies.add(other_name)

            metadata.dependencies = dependencies

    def _estimate_durations(self):
        """Estimate test execution durations based on complexity and history."""
        # Base duration estimates by complexity
        base_durations = {
            TestComplexity.TRIVIAL: 0.1,
            TestComplexity.LOW: 0.5,
            TestComplexity.MEDIUM: 2.0,
            TestComplexity.HIGH: 10.0,
            TestComplexity.EXTREME: 30.0,
        }

        for test_name, metadata in self._discovered_tests.items():
            base_duration = base_durations.get(metadata.complexity, 1.0)

            # Adjust for resource requirements
            multiplier = 1.0
            if ResourceRequirement.EXTERNAL_SERVICE in metadata.resources:
                multiplier *= 3.0
            if ResourceRequirement.DATABASE in metadata.resources:
                multiplier *= 2.0
            if ResourceRequirement.NETWORK in metadata.resources:
                multiplier *= 1.5

            # Adjust for async tests (often longer due to I/O)
            if metadata.async_test:
                multiplier *= 1.5

            # Adjust for parametrized tests
            if metadata.parametrized:
                multiplier *= 2.0

            metadata.estimated_duration = base_duration * multiplier

            # Use historical data if available
            if self._cache_db:
                historical_duration = self._get_historical_duration(test_name)
                if historical_duration:
                    # Weighted average of estimate and history
                    metadata.estimated_duration = (
                        0.3 * metadata.estimated_duration +
                        0.7 * historical_duration
                    )

    def _get_historical_duration(self, test_name: str) -> Optional[float]:
        """Get historical average duration for a test."""
        if not self._cache_db:
            return None

        try:
            cursor = self._cache_db.execute("""
                SELECT AVG(execution_time)
                FROM test_execution_history
                WHERE test_name = ? AND result = 'passed'
                AND timestamp > ?
            """, (test_name, time.time() - 30 * 24 * 3600))  # Last 30 days

            result = cursor.fetchone()
            return result[0] if result and result[0] else None
        except Exception:
            return None

    def _analyze_flaky_patterns(self):
        """Analyze historical data to identify flaky tests."""
        if not self._cache_db:
            return

        for test_name, metadata in self._discovered_tests.items():
            try:
                # Get recent execution history
                cursor = self._cache_db.execute("""
                    SELECT result, execution_time, timestamp, error_message
                    FROM test_execution_history
                    WHERE test_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                """, (test_name,))

                history = cursor.fetchall()
                if len(history) < 10:  # Need minimum history
                    continue

                # Calculate flakiness metrics
                total_runs = len(history)
                failures = sum(1 for result, _, _, _ in history if result != 'passed')
                failure_rate = failures / total_runs

                # Check for intermittent failures (flaky pattern)
                consecutive_states = []
                current_state = None
                state_count = 0

                for result, _, _, _ in reversed(history):
                    if result != current_state:
                        if current_state is not None:
                            consecutive_states.append((current_state, state_count))
                        current_state = result
                        state_count = 1
                    else:
                        state_count += 1

                if current_state is not None:
                    consecutive_states.append((current_state, state_count))

                # Flaky if multiple state changes and moderate failure rate
                state_changes = len(consecutive_states)
                is_flaky = (state_changes > 4 and 0.1 <= failure_rate <= 0.8)

                if is_flaky:
                    flaky_info = {
                        'failure_rate': failure_rate,
                        'total_runs': total_runs,
                        'state_changes': state_changes,
                        'last_failure': max((timestamp for result, _, timestamp, _
                                           in history if result != 'passed'), default=0)
                    }
                    metadata.flaky_history.append(flaky_info)

            except Exception as e:
                print(f"Warning: Failed to analyze flaky patterns for {test_name}: {e}")

    def _load_from_cache(self) -> bool:
        """Load test metadata from cache if valid."""
        if not self._cache_db:
            return False

        try:
            cursor = self._cache_db.execute("""
                SELECT * FROM test_metadata ORDER BY discovery_timestamp DESC
            """)

            cached_tests = cursor.fetchall()
            if not cached_tests:
                return False

            # Check if cache is still valid (files not modified)
            current_tests = {}
            for row in cached_tests:
                test_name = row[0]
                file_path = Path(row[1])
                cached_hash = row[17]

                if not file_path.exists():
                    return False  # File deleted, need rediscovery

                # Check if file was modified
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_hash = hashlib.md5(f.read().encode()).hexdigest()
                    if current_hash != cached_hash:
                        return False  # File modified, need rediscovery
                except Exception:
                    return False

                # Reconstruct metadata
                metadata = TestMetadata(
                    name=test_name,
                    file_path=file_path,
                    category=TestCategory[row[2]],
                    complexity=TestComplexity[row[3]],
                    resources=set(ResourceRequirement[r] for r in json.loads(row[4])),
                    dependencies=set(json.loads(row[5])),
                    estimated_duration=row[6],
                    markers=set(json.loads(row[7])),
                    fixtures=set(json.loads(row[8])),
                    imports=set(json.loads(row[9])),
                    mocks=set(json.loads(row[10])),
                    async_test=bool(row[11]),
                    parametrized=bool(row[12]),
                    expected_failures=set(json.loads(row[13])),
                    flaky_history=json.loads(row[14]),
                    coverage_impact=row[15],
                    last_modified=row[16],
                    source_hash=row[17]
                )
                current_tests[test_name] = metadata

            self._discovered_tests = current_tests
            return True

        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
            return False

    def _save_to_cache(self):
        """Save test metadata to cache."""
        if not self._cache_db:
            return

        try:
            # Clear old cache
            self._cache_db.execute("DELETE FROM test_metadata")

            # Save current metadata
            for test_name, metadata in self._discovered_tests.items():
                self._cache_db.execute("""
                    INSERT INTO test_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_name,
                    str(metadata.file_path),
                    metadata.category.name,
                    metadata.complexity.name,
                    json.dumps([r.name for r in metadata.resources]),
                    json.dumps(list(metadata.dependencies)),
                    metadata.estimated_duration,
                    json.dumps(list(metadata.markers)),
                    json.dumps(list(metadata.fixtures)),
                    json.dumps(list(metadata.imports)),
                    json.dumps(list(metadata.mocks)),
                    int(metadata.async_test),
                    int(metadata.parametrized),
                    json.dumps(list(metadata.expected_failures)),
                    json.dumps(metadata.flaky_history),
                    metadata.coverage_impact,
                    metadata.last_modified,
                    metadata.source_hash,
                    time.time()
                ))

            self._cache_db.commit()

        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")

    def get_tests_by_category(self, category: TestCategory) -> Dict[str, TestMetadata]:
        """Get all tests of a specific category."""
        return {name: metadata for name, metadata in self._discovered_tests.items()
                if metadata.category == category}

    def get_tests_by_complexity(self, complexity: TestComplexity) -> Dict[str, TestMetadata]:
        """Get all tests of a specific complexity level."""
        return {name: metadata for name, metadata in self._discovered_tests.items()
                if metadata.complexity == complexity}

    def get_flaky_tests(self) -> Dict[str, TestMetadata]:
        """Get tests identified as flaky."""
        return {name: metadata for name, metadata in self._discovered_tests.items()
                if metadata.flaky_history}

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test discovery statistics."""
        if not self._discovered_tests:
            return {}

        total_tests = len(self._discovered_tests)

        # Category breakdown
        category_counts = defaultdict(int)
        for metadata in self._discovered_tests.values():
            category_counts[metadata.category] += 1

        # Complexity breakdown
        complexity_counts = defaultdict(int)
        for metadata in self._discovered_tests.values():
            complexity_counts[metadata.complexity] += 1

        # Resource requirements
        resource_counts = defaultdict(int)
        for metadata in self._discovered_tests.values():
            for resource in metadata.resources:
                resource_counts[resource] += 1

        # Duration estimates
        total_duration = sum(m.estimated_duration for m in self._discovered_tests.values())
        avg_duration = total_duration / total_tests if total_tests > 0 else 0

        return {
            'total_tests': total_tests,
            'categories': dict(category_counts),
            'complexities': dict(complexity_counts),
            'resource_requirements': dict(resource_counts),
            'total_estimated_duration': total_duration,
            'average_duration': avg_duration,
            'flaky_tests': len(self.get_flaky_tests()),
            'async_tests': sum(1 for m in self._discovered_tests.values() if m.async_test),
            'parametrized_tests': sum(1 for m in self._discovered_tests.values() if m.parametrized),
        }

    def record_test_execution(self,
                            test_name: str,
                            execution_time: float,
                            result: str,
                            error_message: Optional[str] = None):
        """Record test execution for historical analysis."""
        if not self._cache_db:
            return

        try:
            self._cache_db.execute("""
                INSERT INTO test_execution_history VALUES (?, ?, ?, ?, ?)
            """, (test_name, execution_time, result, time.time(), error_message))
            self._cache_db.commit()
        except Exception as e:
            print(f"Warning: Failed to record test execution: {e}")

    def close(self):
        """Close the discovery system and cleanup resources."""
        if self._cache_db:
            self._cache_db.close()
            self._cache_db = None