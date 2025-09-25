"""
Test Execution Framework

Comprehensive test execution system providing automated scheduling, resource management,
parallel execution, and result collection with robust error handling and monitoring.

Components:
- TestExecutionScheduler: Core execution scheduling with resource monitoring
- TestAutomationSystem: High-level automation with CI/CD integration
"""

from .scheduler import (
    TestExecutionScheduler,
    TestExecution,
    ExecutionResult,
    ExecutionStatus,
    ExecutionPriority,
    ExecutionConstraints
)

from .automation import (
    TestAutomationSystem,
    AutomationConfig,
    AutomationMode,
    AutomationTrigger,
    AutomationRun,
    TriggerType
)

__all__ = [
    "TestExecutionScheduler",
    "TestExecution",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionPriority",
    "ExecutionConstraints",
    "TestAutomationSystem",
    "AutomationConfig",
    "AutomationMode",
    "AutomationTrigger",
    "AutomationRun",
    "TriggerType"
]

__version__ = "1.0.0"