"""
Context injection system for LLM tools.

This module provides automatic context injection into LLM tool sessions using
memory rules stored in Qdrant.
"""

from .authority_filter import AuthorityFilter, FilteredRules, RuleHierarchy
from .claude_code_detector import (
    ClaudeCodeDetector,
    ClaudeCodeSession,
    get_claude_code_session,
    is_claude_code_session,
)
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
from .performance_optimization import (
    BatchProcessor,
    PerformanceMetrics,
    RuleCache,
    RuleIndex,
)
from .rule_retrieval import RuleFilter, RuleRetrieval, RuleRetrievalResult
from .token_budget import (
    AllocationStrategy,
    BudgetAllocation,
    CompressionStrategy,
    TokenBudgetManager,
    TokenCounter,
)

__all__ = [
    # Rule Retrieval
    "RuleRetrieval",
    "RuleFilter",
    "RuleRetrievalResult",
    # Authority Filtering
    "AuthorityFilter",
    "FilteredRules",
    "RuleHierarchy",
    # Project Context
    "ProjectContext",
    "ProjectContextDetector",
    "ProjectRuleApplicator",
    # Performance Optimization
    "PerformanceMetrics",
    "RuleCache",
    "RuleIndex",
    "BatchProcessor",
    # Formatters
    "FormatManager",
    "LLMToolAdapter",
    "ToolCapabilities",
    "FormattedContext",
    "FormatType",
    "ClaudeCodeAdapter",
    "GitHubCodexAdapter",
    "GoogleGeminiAdapter",
    # Token Budget
    "TokenBudgetManager",
    "BudgetAllocation",
    "AllocationStrategy",
    "CompressionStrategy",
    "TokenCounter",
    # Claude Code Detection
    "ClaudeCodeDetector",
    "ClaudeCodeSession",
    "is_claude_code_session",
    "get_claude_code_session",
]
