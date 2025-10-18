"""
E2E Complete Workflow Scenarios (Task 292.1).

Tests complete system workflows using pytest-bdd scenarios.
"""

import pytest
from pytest_bdd import scenarios

# Load all scenarios from complete_workflow.feature
scenarios('features/complete_workflow.feature')


@pytest.mark.e2e
@pytest.mark.workflow
class TestCompleteWorkflowScenarios:
    """
    E2E tests for complete system workflows.

    Tests validated:
    - Complete document ingestion workflow
    - Multi-file ingestion and search
    - Real-time file modification tracking
    - Project switching workflow
    - Collection management workflow
    """

    pass  # Tests defined in feature file and step definitions
