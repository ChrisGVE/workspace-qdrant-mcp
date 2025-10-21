"""
Coverage threshold checking and validation.

Provides functionality to check if coverage metrics meet specified thresholds
and generate warnings/failures for CI/CD integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .models import CoverageMetrics, FileCoverage


class ThresholdStatus(str, Enum):
    """Status of threshold check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class CoverageThresholds:
    """Coverage thresholds configuration."""

    # Overall thresholds
    line_coverage_min: Optional[float] = None  # Minimum line coverage %
    function_coverage_min: Optional[float] = None  # Minimum function coverage %
    branch_coverage_min: Optional[float] = None  # Minimum branch coverage %

    # Per-file thresholds
    file_line_coverage_min: Optional[float] = None  # Minimum per-file line coverage %

    # Warning thresholds (if not met, warn but don't fail)
    line_coverage_warning: Optional[float] = None
    function_coverage_warning: Optional[float] = None
    branch_coverage_warning: Optional[float] = None

    # Allow specific files to be excluded from threshold checks
    exclude_files: List[str] = field(default_factory=list)  # File path patterns

    @classmethod
    def default(cls) -> "CoverageThresholds":
        """
        Create default thresholds.

        Returns:
            CoverageThresholds with reasonable defaults
        """
        return cls(
            line_coverage_min=80.0,
            line_coverage_warning=90.0,
            function_coverage_min=70.0,
            function_coverage_warning=80.0,
            branch_coverage_min=60.0,
            branch_coverage_warning=70.0,
            file_line_coverage_min=70.0,
        )

    @classmethod
    def strict(cls) -> "CoverageThresholds":
        """
        Create strict thresholds for high-quality projects.

        Returns:
            CoverageThresholds with strict requirements
        """
        return cls(
            line_coverage_min=90.0,
            line_coverage_warning=95.0,
            function_coverage_min=85.0,
            function_coverage_warning=90.0,
            branch_coverage_min=80.0,
            branch_coverage_warning=85.0,
            file_line_coverage_min=85.0,
        )


@dataclass
class ThresholdViolation:
    """A single threshold violation."""

    metric: str  # e.g., "line_coverage", "function_coverage"
    actual: float  # Actual coverage percentage
    threshold: float  # Required threshold
    status: ThresholdStatus  # FAILED or WARNING
    file_path: Optional[str] = None  # If violation is for specific file

    @property
    def message(self) -> str:
        """Get human-readable violation message."""
        if self.file_path:
            return (
                f"{self.status.value.upper()}: {self.file_path} "
                f"{self.metric} is {self.actual:.2f}% (threshold: {self.threshold:.2f}%)"
            )
        else:
            return (
                f"{self.status.value.upper()}: Overall {self.metric} is "
                f"{self.actual:.2f}% (threshold: {self.threshold:.2f}%)"
            )


@dataclass
class CoverageCheckResult:
    """Result of coverage threshold check."""

    status: ThresholdStatus
    violations: List[ThresholdViolation] = field(default_factory=list)
    warnings: List[ThresholdViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all thresholds passed (warnings allowed)."""
        return self.status != ThresholdStatus.FAILED

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def message(self) -> str:
        """Get summary message."""
        if self.status == ThresholdStatus.PASSED:
            return "All coverage thresholds passed"
        elif self.status == ThresholdStatus.WARNING:
            return f"Coverage thresholds passed with {len(self.warnings)} warning(s)"
        else:
            return f"Coverage thresholds failed with {len(self.violations)} violation(s)"

    def get_all_issues(self) -> List[ThresholdViolation]:
        """Get all violations and warnings combined."""
        return self.violations + self.warnings


class CoverageChecker:
    """Check coverage metrics against thresholds."""

    def __init__(self, thresholds: Optional[CoverageThresholds] = None):
        """
        Initialize coverage checker.

        Args:
            thresholds: Coverage thresholds to check against (uses default if None)
        """
        self.thresholds = thresholds or CoverageThresholds.default()

    def check(self, coverage: CoverageMetrics) -> CoverageCheckResult:
        """
        Check coverage metrics against thresholds.

        Args:
            coverage: CoverageMetrics to check

        Returns:
            CoverageCheckResult with status and violations
        """
        violations = []
        warnings = []

        # Check overall line coverage
        if self.thresholds.line_coverage_min is not None:
            if coverage.line_coverage_percent < self.thresholds.line_coverage_min:
                violations.append(
                    ThresholdViolation(
                        metric="line_coverage",
                        actual=coverage.line_coverage_percent,
                        threshold=self.thresholds.line_coverage_min,
                        status=ThresholdStatus.FAILED,
                    )
                )
            elif (
                self.thresholds.line_coverage_warning is not None
                and coverage.line_coverage_percent < self.thresholds.line_coverage_warning
            ):
                warnings.append(
                    ThresholdViolation(
                        metric="line_coverage",
                        actual=coverage.line_coverage_percent,
                        threshold=self.thresholds.line_coverage_warning,
                        status=ThresholdStatus.WARNING,
                    )
                )

        # Check function coverage
        if (
            self.thresholds.function_coverage_min is not None
            and coverage.function_coverage_percent is not None
        ):
            if coverage.function_coverage_percent < self.thresholds.function_coverage_min:
                violations.append(
                    ThresholdViolation(
                        metric="function_coverage",
                        actual=coverage.function_coverage_percent,
                        threshold=self.thresholds.function_coverage_min,
                        status=ThresholdStatus.FAILED,
                    )
                )
            elif (
                self.thresholds.function_coverage_warning is not None
                and coverage.function_coverage_percent
                < self.thresholds.function_coverage_warning
            ):
                warnings.append(
                    ThresholdViolation(
                        metric="function_coverage",
                        actual=coverage.function_coverage_percent,
                        threshold=self.thresholds.function_coverage_warning,
                        status=ThresholdStatus.WARNING,
                    )
                )

        # Check branch coverage
        if (
            self.thresholds.branch_coverage_min is not None
            and coverage.branch_coverage_percent is not None
        ):
            if coverage.branch_coverage_percent < self.thresholds.branch_coverage_min:
                violations.append(
                    ThresholdViolation(
                        metric="branch_coverage",
                        actual=coverage.branch_coverage_percent,
                        threshold=self.thresholds.branch_coverage_min,
                        status=ThresholdStatus.FAILED,
                    )
                )
            elif (
                self.thresholds.branch_coverage_warning is not None
                and coverage.branch_coverage_percent < self.thresholds.branch_coverage_warning
            ):
                warnings.append(
                    ThresholdViolation(
                        metric="branch_coverage",
                        actual=coverage.branch_coverage_percent,
                        threshold=self.thresholds.branch_coverage_warning,
                        status=ThresholdStatus.WARNING,
                    )
                )

        # Check per-file coverage
        if self.thresholds.file_line_coverage_min is not None:
            for file_cov in coverage.file_coverage:
                # Skip excluded files
                if self._is_excluded(file_cov.file_path):
                    continue

                if file_cov.line_coverage_percent < self.thresholds.file_line_coverage_min:
                    violations.append(
                        ThresholdViolation(
                            metric="file_line_coverage",
                            actual=file_cov.line_coverage_percent,
                            threshold=self.thresholds.file_line_coverage_min,
                            status=ThresholdStatus.FAILED,
                            file_path=file_cov.file_path,
                        )
                    )

        # Determine overall status
        if violations:
            status = ThresholdStatus.FAILED
        elif warnings:
            status = ThresholdStatus.WARNING
        else:
            status = ThresholdStatus.PASSED

        return CoverageCheckResult(
            status=status, violations=violations, warnings=warnings
        )

    def _is_excluded(self, file_path: str) -> bool:
        """
        Check if a file should be excluded from threshold checks.

        Args:
            file_path: File path to check

        Returns:
            True if file should be excluded
        """
        for pattern in self.thresholds.exclude_files:
            # Simple glob-style matching
            if pattern in file_path:
                return True
            # Exact match
            if file_path == pattern:
                return True
        return False


# Convenience function
def check_coverage_thresholds(
    coverage: CoverageMetrics,
    thresholds: Optional[CoverageThresholds] = None,
) -> CoverageCheckResult:
    """
    Quick helper to check coverage against thresholds.

    Args:
        coverage: CoverageMetrics to check
        thresholds: Optional thresholds (uses default if None)

    Returns:
        CoverageCheckResult

    Example:
        >>> from tests.reporting.coverage_checker import check_coverage_thresholds
        >>> result = check_coverage_thresholds(coverage)
        >>> if not result.passed:
        >>>     for violation in result.violations:
        >>>         print(violation.message)
        >>>     sys.exit(1)
    """
    checker = CoverageChecker(thresholds)
    return checker.check(coverage)
