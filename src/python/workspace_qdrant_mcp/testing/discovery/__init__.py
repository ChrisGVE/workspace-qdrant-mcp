"""
Test Discovery Module

Provides automated test case discovery, categorization, and gap analysis
to identify missing test coverage and suggest new test scenarios.
"""

from .engine import TestDiscoveryEngine
from .categorizer import TestCategorizer, TestCategory, CoverageGap

__all__ = [
    "TestDiscoveryEngine",
    "TestCategorizer",
    "TestCategory",
    "CoverageGap"
]