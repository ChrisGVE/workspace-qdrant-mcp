"""
Test Lifecycle Manager

Manages the complete lifecycle of tests including aging analysis,
maintenance recommendations, obsolete test detection, and automated
cleanup suggestions with comprehensive health monitoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import sqlite3
import json
import ast
import re
from collections import defaultdict

from ..documentation.parser import TestFileInfo, TestMetadata, TestFileParser
from .scheduler import MaintenanceScheduler, MaintenanceTask, TaskType, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class TestHealth:
    """Health metrics for a test."""
    test_id: str
    file_path: Path
    test_name: str
    age_days: int
    last_modified: datetime
    complexity_score: int
    failure_rate: float = 0.0
    execution_time_ms: float = 0.0
    coverage_contribution: float = 0.0
    maintenance_score: float = 1.0  # 0-1, lower means needs maintenance
    health_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ObsoleteTestCandidate:
    """Candidate test for removal."""
    test_metadata: TestMetadata
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    impact_assessment: str = "low"  # low, medium, high
    suggested_action: str = "review"  # review, refactor, remove


@dataclass
class RefactoringSuggestion:
    """Suggestion for test refactoring."""
    target_tests: List[TestMetadata]
    refactoring_type: str  # extract_common, reduce_duplication, simplify, modernize
    description: str
    estimated_effort: str = "medium"  # low, medium, high
    expected_benefit: str = "maintenance"  # maintenance, performance, readability
    code_examples: List[str] = field(default_factory=list)


@dataclass
class LifecycleReport:
    """Complete lifecycle analysis report."""
    analysis_date: datetime
    total_tests: int
    test_health: List[TestHealth] = field(default_factory=list)
    obsolete_candidates: List[ObsoleteTestCandidate] = field(default_factory=list)
    refactoring_suggestions: List[RefactoringSuggestion] = field(default_factory=list)
    maintenance_tasks: List[MaintenanceTask] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)


class TestLifecycleManager:
    """
    Manages test lifecycle including aging, maintenance, and cleanup.

    Provides comprehensive test health monitoring, obsolete test detection,
    refactoring recommendations, and automated maintenance scheduling.
    """

    def __init__(self,
                 db_path: Path,
                 scheduler: Optional[MaintenanceScheduler] = None):
        """
        Initialize test lifecycle manager.

        Args:
            db_path: Path to SQLite database for persistence
            scheduler: Optional maintenance scheduler for task automation
        """
        self.db_path = db_path
        self.scheduler = scheduler
        self.parser = TestFileParser()

        self._initialize_db()

    def analyze_test_lifecycle(self,
                             project_root: Path,
                             test_patterns: List[str] = None) -> LifecycleReport:
        """
        Perform comprehensive test lifecycle analysis.

        Args:
            project_root: Root directory of project
            test_patterns: Patterns to match test files

        Returns:
            Complete lifecycle analysis report
        """
        logger.info("Starting test lifecycle analysis")

        if test_patterns is None:
            test_patterns = ['test_*.py', '*_test.py']

        report = LifecycleReport(
            analysis_date=datetime.now(),
            total_tests=0
        )

        try:
            # Discover and parse test files
            test_files = self._discover_test_files(project_root, test_patterns)
            file_infos = []

            for test_file in test_files:
                try:
                    file_info = self.parser.parse_file(test_file)
                    file_infos.append(file_info)
                except Exception as e:
                    logger.warning(f"Failed to parse {test_file}: {e}")

            # Collect all tests
            all_tests = []
            for file_info in file_infos:
                all_tests.extend(file_info.tests)

            report.total_tests = len(all_tests)

            # Analyze test health
            report.test_health = self._analyze_test_health(all_tests, file_infos)

            # Identify obsolete test candidates
            report.obsolete_candidates = self._identify_obsolete_tests(all_tests, file_infos)

            # Generate refactoring suggestions
            report.refactoring_suggestions = self._generate_refactoring_suggestions(all_tests, file_infos)

            # Create maintenance tasks
            report.maintenance_tasks = self._create_maintenance_tasks(report)

            # Generate summary statistics
            report.summary_stats = self._generate_summary_stats(report)

            # Save results
            self._save_analysis_results(report)

            logger.info(f"Lifecycle analysis complete: {report.total_tests} tests analyzed")

        except Exception as e:
            logger.error(f"Lifecycle analysis failed: {e}")
            raise

        return report

    def schedule_maintenance_tasks(self, report: LifecycleReport) -> int:
        """
        Schedule maintenance tasks based on lifecycle analysis.

        Args:
            report: Lifecycle analysis report

        Returns:
            Number of tasks scheduled
        """
        if not self.scheduler:
            logger.warning("No scheduler available for task scheduling")
            return 0

        scheduled_count = 0

        for task in report.maintenance_tasks:
            try:
                if self.scheduler.schedule_task(task):
                    scheduled_count += 1
            except Exception as e:
                logger.error(f"Failed to schedule task {task.task_id}: {e}")

        logger.info(f"Scheduled {scheduled_count} maintenance tasks")
        return scheduled_count

    def get_test_health_summary(self) -> Dict[str, Any]:
        """Get summary of test health across the project."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_tests,
                    AVG(maintenance_score) as avg_maintenance_score,
                    SUM(CASE WHEN maintenance_score < 0.5 THEN 1 ELSE 0 END) as unhealthy_tests,
                    AVG(age_days) as avg_age_days,
                    MAX(age_days) as oldest_test_days
                FROM test_health
                WHERE analysis_date = (SELECT MAX(analysis_date) FROM test_health)
            ''')

            result = cursor.fetchone()
            if result:
                return {
                    'total_tests': result[0] or 0,
                    'avg_maintenance_score': round(result[1] or 0, 2),
                    'unhealthy_tests': result[2] or 0,
                    'avg_age_days': round(result[3] or 0, 1),
                    'oldest_test_days': result[4] or 0
                }

        finally:
            conn.close()

        return {}

    def _discover_test_files(self, project_root: Path, patterns: List[str]) -> List[Path]:
        """Discover test files matching patterns."""
        test_files = []
        for pattern in patterns:
            files = project_root.rglob(pattern)
            test_files.extend([f for f in files if f.is_file()])
        return sorted(set(test_files))

    def _analyze_test_health(self,
                           tests: List[TestMetadata],
                           file_infos: List[TestFileInfo]) -> List[TestHealth]:
        """Analyze health of individual tests."""
        health_results = []

        file_modification_times = {}
        for file_info in file_infos:
            try:
                stat = file_info.file_path.stat()
                file_modification_times[file_info.file_path] = datetime.fromtimestamp(stat.st_mtime)
            except:
                file_modification_times[file_info.file_path] = datetime.now()

        for test in tests:
            health = TestHealth(
                test_id=f"{test.file_path}::{test.name}",
                file_path=test.file_path,
                test_name=test.name,
                age_days=0,
                last_modified=datetime.now(),
                complexity_score=test.complexity_score
            )

            # Calculate age
            if test.file_path in file_modification_times:
                health.last_modified = file_modification_times[test.file_path]
                health.age_days = (datetime.now() - health.last_modified).days

            # Analyze health issues
            health.health_issues = self._identify_health_issues(test)

            # Generate recommendations
            health.recommendations = self._generate_health_recommendations(test, health)

            # Calculate maintenance score
            health.maintenance_score = self._calculate_maintenance_score(test, health)

            health_results.append(health)

        return health_results

    def _identify_health_issues(self, test: TestMetadata) -> List[str]:
        """Identify health issues with a test."""
        issues = []

        # Check for missing docstring
        if not test.docstring or test.docstring.strip() == "":
            issues.append("Missing or empty docstring")

        # Check for high complexity
        if test.complexity_score > 7:
            issues.append(f"High complexity score: {test.complexity_score}")

        # Check for potential naming issues
        if len(test.name.split('_')) < 2:
            issues.append("Test name may not be descriptive enough")

        # Check for too many decorators (potential over-engineering)
        if len(test.decorators) > 5:
            issues.append("Too many decorators - may indicate over-engineering")

        # Check for expected failures that might be outdated
        if test.expected_to_fail and not any('todo' in d.name.lower() or 'fixme' in d.name.lower()
                                           for d in test.decorators):
            issues.append("Expected failure without clear TODO/FIXME indication")

        return issues

    def _generate_health_recommendations(self, test: TestMetadata, health: TestHealth) -> List[str]:
        """Generate recommendations for improving test health."""
        recommendations = []

        # Docstring recommendations
        if "Missing or empty docstring" in health.health_issues:
            recommendations.append("Add descriptive docstring explaining test purpose")

        # Complexity recommendations
        if test.complexity_score > 7:
            recommendations.append("Consider breaking test into smaller, focused tests")
            recommendations.append("Extract complex setup into fixtures")

        # Age-based recommendations
        if health.age_days > 365:
            recommendations.append("Review test relevance - test is over 1 year old")

        if health.age_days > 90 and not test.docstring:
            recommendations.append("Add documentation for older test to preserve knowledge")

        # Modernization recommendations
        if not test.is_async and 'async' in test.name.lower():
            recommendations.append("Consider converting to async test if testing async code")

        if not test.is_parametrized and 'multiple' in (test.docstring or '').lower():
            recommendations.append("Consider parametrizing test for multiple scenarios")

        return recommendations

    def _calculate_maintenance_score(self, test: TestMetadata, health: TestHealth) -> float:
        """Calculate maintenance score (0-1, higher is better)."""
        score = 1.0

        # Penalty for health issues
        score -= len(health.health_issues) * 0.1

        # Penalty for high complexity
        if test.complexity_score > 5:
            score -= (test.complexity_score - 5) * 0.05

        # Penalty for age without documentation
        if health.age_days > 180 and not test.docstring:
            score -= 0.2

        # Bonus for good practices
        if test.docstring and len(test.docstring.strip()) > 20:
            score += 0.1

        if test.is_parametrized:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _identify_obsolete_tests(self,
                               tests: List[TestMetadata],
                               file_infos: List[TestFileInfo]) -> List[ObsoleteTestCandidate]:
        """Identify tests that may be obsolete."""
        candidates = []

        # Group tests by file for analysis
        tests_by_file = defaultdict(list)
        for test in tests:
            tests_by_file[test.file_path].append(test)

        for file_path, file_tests in tests_by_file.items():
            # Look for duplicate test patterns
            name_patterns = defaultdict(list)
            for test in file_tests:
                # Extract base pattern from test name
                pattern = re.sub(r'_\d+$', '', test.name)  # Remove trailing numbers
                pattern = re.sub(r'_v\d+$', '', pattern)   # Remove version suffixes
                name_patterns[pattern].append(test)

            # Identify potential duplicates
            for pattern, pattern_tests in name_patterns.items():
                if len(pattern_tests) > 1:
                    # Analyze for actual duplication
                    for test in pattern_tests[1:]:  # Keep first, mark others as candidates
                        candidate = ObsoleteTestCandidate(
                            test_metadata=test,
                            reasons=["Potentially duplicate test pattern"],
                            confidence=0.6,
                            impact_assessment="medium",
                            suggested_action="review"
                        )
                        candidates.append(candidate)

            # Look for very old tests with no recent modifications
            for test in file_tests:
                age_days = self._calculate_test_age(test)
                if age_days > 730:  # 2 years
                    reasons = [f"Very old test (${age_days} days)"]

                    # Additional checks for obsolescence indicators
                    if test.expected_to_fail:
                        reasons.append("Marked as expected failure for extended period")

                    if 'deprecated' in (test.docstring or '').lower():
                        reasons.append("Contains 'deprecated' in documentation")

                    if len(reasons) > 1:
                        candidate = ObsoleteTestCandidate(
                            test_metadata=test,
                            reasons=reasons,
                            confidence=0.4,
                            impact_assessment="low",
                            suggested_action="review"
                        )
                        candidates.append(candidate)

        return candidates

    def _generate_refactoring_suggestions(self,
                                        tests: List[TestMetadata],
                                        file_infos: List[TestFileInfo]) -> List[RefactoringSuggestion]:
        """Generate refactoring suggestions for tests."""
        suggestions = []

        # Group tests by file
        tests_by_file = defaultdict(list)
        for test in tests:
            tests_by_file[test.file_path].append(test)

        for file_path, file_tests in tests_by_file.items():
            # Look for common setup patterns that could be extracted
            high_complexity_tests = [t for t in file_tests if t.complexity_score > 6]
            if len(high_complexity_tests) > 2:
                suggestions.append(RefactoringSuggestion(
                    target_tests=high_complexity_tests,
                    refactoring_type="extract_common",
                    description="Extract common setup/teardown patterns from complex tests",
                    estimated_effort="medium",
                    expected_benefit="maintenance",
                    code_examples=["@pytest.fixture", "def setup_complex_scenario():"]
                ))

            # Look for tests that could benefit from parametrization
            similar_tests = self._find_similar_tests(file_tests)
            for test_group in similar_tests:
                if len(test_group) >= 3:
                    suggestions.append(RefactoringSuggestion(
                        target_tests=test_group,
                        refactoring_type="reduce_duplication",
                        description="Combine similar tests using parametrization",
                        estimated_effort="low",
                        expected_benefit="maintenance",
                        code_examples=["@pytest.mark.parametrize"]
                    ))

            # Look for tests that could be simplified
            over_decorated_tests = [t for t in file_tests if len(t.decorators) > 4]
            if over_decorated_tests:
                suggestions.append(RefactoringSuggestion(
                    target_tests=over_decorated_tests,
                    refactoring_type="simplify",
                    description="Simplify tests with excessive decorators",
                    estimated_effort="low",
                    expected_benefit="readability"
                ))

        return suggestions

    def _find_similar_tests(self, tests: List[TestMetadata]) -> List[List[TestMetadata]]:
        """Find groups of similar tests that could be parametrized."""
        similar_groups = []

        # Group by name pattern
        pattern_groups = defaultdict(list)
        for test in tests:
            # Extract base pattern from test name
            base_name = re.sub(r'_\w+$', '', test.name)  # Remove last word
            if len(base_name) < len(test.name):  # Only if we actually removed something
                pattern_groups[base_name].append(test)

        for pattern, group_tests in pattern_groups.items():
            if len(group_tests) >= 3:
                similar_groups.append(group_tests)

        return similar_groups

    def _create_maintenance_tasks(self, report: LifecycleReport) -> List[MaintenanceTask]:
        """Create maintenance tasks based on analysis results."""
        tasks = []

        # Create tasks for unhealthy tests
        unhealthy_tests = [h for h in report.test_health if h.maintenance_score < 0.5]
        if len(unhealthy_tests) > 5:
            task = MaintenanceTask(
                task_id=f"health_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                task_type=TaskType.TEST_REFACTOR,
                priority=TaskPriority.MEDIUM,
                title="Improve unhealthy test health",
                description=f"Address {len(unhealthy_tests)} tests with low maintenance scores",
                estimated_duration=timedelta(hours=4),
                target_files=list(set(h.file_path for h in unhealthy_tests)),
                metadata={
                    'unhealthy_test_count': len(unhealthy_tests),
                    'avg_maintenance_score': sum(h.maintenance_score for h in unhealthy_tests) / len(unhealthy_tests)
                }
            )
            tasks.append(task)

        # Create tasks for obsolete test candidates
        high_confidence_obsolete = [c for c in report.obsolete_candidates if c.confidence > 0.7]
        if high_confidence_obsolete:
            task = MaintenanceTask(
                task_id=f"obsolete_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                task_type=TaskType.OBSOLETE_TEST_REMOVAL,
                priority=TaskPriority.LOW,
                title="Review and remove obsolete tests",
                description=f"Review {len(high_confidence_obsolete)} potentially obsolete tests",
                estimated_duration=timedelta(hours=2),
                target_files=list(set(c.test_metadata.file_path for c in high_confidence_obsolete)),
                metadata={'candidate_count': len(high_confidence_obsolete)}
            )
            tasks.append(task)

        # Create tasks for refactoring suggestions
        for suggestion in report.refactoring_suggestions[:3]:  # Limit to top 3
            effort_hours = {'low': 1, 'medium': 3, 'high': 8}.get(suggestion.estimated_effort, 3)
            task = MaintenanceTask(
                task_id=f"refactor_{suggestion.refactoring_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                task_type=TaskType.TEST_REFACTOR,
                priority=TaskPriority.MEDIUM if suggestion.expected_benefit == "maintenance" else TaskPriority.LOW,
                title=f"Refactor tests: {suggestion.refactoring_type}",
                description=suggestion.description,
                estimated_duration=timedelta(hours=effort_hours),
                target_files=list(set(t.file_path for t in suggestion.target_tests)),
                metadata={
                    'refactoring_type': suggestion.refactoring_type,
                    'expected_benefit': suggestion.expected_benefit,
                    'target_test_count': len(suggestion.target_tests)
                }
            )
            tasks.append(task)

        return tasks

    def _generate_summary_stats(self, report: LifecycleReport) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        stats = {}

        if report.test_health:
            stats['health'] = {
                'avg_maintenance_score': sum(h.maintenance_score for h in report.test_health) / len(report.test_health),
                'unhealthy_count': len([h for h in report.test_health if h.maintenance_score < 0.5]),
                'avg_age_days': sum(h.age_days for h in report.test_health) / len(report.test_health),
                'avg_complexity': sum(h.complexity_score for h in report.test_health) / len(report.test_health)
            }

        stats['obsolete'] = {
            'total_candidates': len(report.obsolete_candidates),
            'high_confidence': len([c for c in report.obsolete_candidates if c.confidence > 0.7]),
            'removal_suggestions': len([c for c in report.obsolete_candidates if c.suggested_action == "remove"])
        }

        stats['refactoring'] = {
            'total_suggestions': len(report.refactoring_suggestions),
            'high_effort': len([s for s in report.refactoring_suggestions if s.estimated_effort == "high"]),
            'maintenance_focused': len([s for s in report.refactoring_suggestions if s.expected_benefit == "maintenance"])
        }

        stats['maintenance_tasks'] = {
            'total_tasks': len(report.maintenance_tasks),
            'high_priority': len([t for t in report.maintenance_tasks if t.priority.value >= 3]),
            'estimated_total_hours': sum(t.estimated_duration.total_seconds() / 3600 for t in report.maintenance_tasks)
        }

        return stats

    def _calculate_test_age(self, test: TestMetadata) -> int:
        """Calculate age of test in days."""
        try:
            stat = test.file_path.stat()
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            return (datetime.now() - modified_time).days
        except:
            return 0

    def _save_analysis_results(self, report: LifecycleReport) -> None:
        """Save analysis results to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Save test health data
            for health in report.test_health:
                conn.execute('''
                    INSERT OR REPLACE INTO test_health (
                        test_id, file_path, test_name, age_days, last_modified,
                        complexity_score, maintenance_score, health_issues,
                        recommendations, analysis_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    health.test_id,
                    str(health.file_path),
                    health.test_name,
                    health.age_days,
                    health.last_modified.timestamp(),
                    health.complexity_score,
                    health.maintenance_score,
                    json.dumps(health.health_issues),
                    json.dumps(health.recommendations),
                    report.analysis_date.timestamp()
                ))

            # Save summary stats
            conn.execute('''
                INSERT INTO lifecycle_reports (
                    analysis_date, total_tests, summary_stats
                ) VALUES (?, ?, ?)
            ''', (
                report.analysis_date.timestamp(),
                report.total_tests,
                json.dumps(report.summary_stats)
            ))

            conn.commit()
        finally:
            conn.close()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_health (
                    test_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    age_days INTEGER,
                    last_modified REAL,
                    complexity_score INTEGER,
                    failure_rate REAL DEFAULT 0.0,
                    execution_time_ms REAL DEFAULT 0.0,
                    coverage_contribution REAL DEFAULT 0.0,
                    maintenance_score REAL,
                    health_issues TEXT,
                    recommendations TEXT,
                    analysis_date REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS lifecycle_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date REAL NOT NULL,
                    total_tests INTEGER,
                    summary_stats TEXT
                )
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_health_analysis_date
                ON test_health(analysis_date)
            ''')

            conn.commit()
        finally:
            conn.close()