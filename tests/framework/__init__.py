"""
Comprehensive Testing Framework for workspace-qdrant-mcp

This package provides advanced testing orchestration, discovery, execution,
and analytics capabilities that extend beyond standard pytest functionality.

Components:
- discovery: Intelligent test discovery and categorization
- execution: Parallel test execution with dependency management
- analytics: Advanced result aggregation and reporting
- integration: Cross-component integration testing coordination
- orchestration: Central test orchestration and coordination
- coverage_validator: 100% coverage validation system
"""

from .analytics import TestAnalytics, TestMetrics
from .coverage_validator import CoverageLevel, CoverageReport, CoverageValidator
from .discovery import TestCategory, TestComplexity, TestDiscovery
from .execution import ExecutionStrategy, ParallelTestExecutor
from .integration import IntegrationTestCoordinator
from .orchestration import OrchestrationConfig, OrchestrationMode, TestOrchestrator

__all__ = [
    "TestDiscovery",
    "TestCategory",
    "TestComplexity",
    "ParallelTestExecutor",
    "ExecutionStrategy",
    "TestAnalytics",
    "TestMetrics",
    "IntegrationTestCoordinator",
    "TestOrchestrator",
    "OrchestrationConfig",
    "OrchestrationMode",
    "CoverageValidator",
    "CoverageReport",
    "CoverageLevel",
]
