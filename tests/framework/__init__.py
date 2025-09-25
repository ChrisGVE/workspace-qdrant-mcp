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

from .discovery import TestDiscovery, TestCategory, TestComplexity
from .execution import ParallelTestExecutor, ExecutionStrategy
from .analytics import TestAnalytics, TestMetrics
from .integration import IntegrationTestCoordinator
from .orchestration import TestOrchestrator, OrchestrationConfig, OrchestrationMode
from .coverage_validator import CoverageValidator, CoverageReport, CoverageLevel

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