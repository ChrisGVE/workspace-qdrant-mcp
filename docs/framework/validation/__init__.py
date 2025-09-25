"""Documentation validation tools for quality assurance and coverage analysis."""

from .coverage_analyzer import DocumentationCoverageAnalyzer, CoverageStats, MemberCoverage, ProjectCoverage
from .quality_checker import DocumentationQualityChecker, QualityIssue, QualityReport, ProjectQualityReport
from .link_validator import LinkValidator, LinkStatus, ValidationResult as LinkValidationResult
from .cross_reference import CrossReferenceValidator, ReferenceType, ReferenceLink, ValidationResult as CrossRefValidationResult
from .consistency_checker import ConsistencyChecker, ConsistencyRule, ConsistencyViolation, ConsistencyLevel

__all__ = [
    # Coverage analysis
    "DocumentationCoverageAnalyzer",
    "CoverageStats",
    "MemberCoverage",
    "ProjectCoverage",

    # Quality checking
    "DocumentationQualityChecker",
    "QualityIssue",
    "QualityReport",
    "ProjectQualityReport",

    # Link validation
    "LinkValidator",
    "LinkStatus",
    "LinkValidationResult",

    # Cross-reference validation
    "CrossReferenceValidator",
    "ReferenceType",
    "ReferenceLink",
    "CrossRefValidationResult",

    # Consistency checking
    "ConsistencyChecker",
    "ConsistencyRule",
    "ConsistencyViolation",
    "ConsistencyLevel",
]