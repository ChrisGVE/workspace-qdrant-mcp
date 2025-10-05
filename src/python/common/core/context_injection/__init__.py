"""
Context injection system for LLM tools.

This module provides automatic context injection into LLM tool sessions using
memory rules stored in Qdrant.
"""

from .authority_filter import AuthorityFilter, FilteredRules, RuleHierarchy
from .formatters import (
    ClaudeCodeAdapter,
    FormatManager,
    FormatType,
    FormattedContext,
    GitHubCodexAdapter,
    GoogleGeminiAdapter,
    LLMToolAdapter,
    ToolCapabilities,
)
from .project_context import (
    ProjectContext,
    ProjectContextDetector,
    ProjectRuleApplicator,
)
from .rule_retrieval import RuleFilter, RuleRetrieval

__all__ = [
    "RuleRetrieval",
    "RuleFilter",
    "AuthorityFilter",
    "FilteredRules",
    "RuleHierarchy",
    "ProjectContext",
    "ProjectContextDetector",
    "ProjectRuleApplicator",
    # Formatters
    "FormatManager",
    "LLMToolAdapter",
    "ToolCapabilities",
    "FormattedContext",
    "FormatType",
    "ClaudeCodeAdapter",
    "GitHubCodexAdapter",
    "GoogleGeminiAdapter",
]
