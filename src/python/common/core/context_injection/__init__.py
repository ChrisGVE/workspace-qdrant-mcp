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
from .claude_md_injector import (
    ClaudeMdInjector,
    ClaudeMdLocation,
    inject_claude_md_content,
)
from .system_prompt_injector import (
    SystemPromptInjector,
    SystemPromptConfig,
    InjectionMode,
    generate_mcp_context,
    generate_api_system_prompt,
)
from .session_trigger import (
    SessionTrigger,
    TriggerPhase,
    TriggerPriority,
    TriggerResult,
    TriggerContext,
    TriggerManager,
    ClaudeMdFileTrigger,
    SystemPromptTrigger,
    CleanupTrigger,
    CustomCallbackTrigger,
    prepare_claude_code_session,
    cleanup_claude_code_session,
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
from .claude_budget_manager import (
    ClaudeBudgetManager,
    ClaudeBudgetAllocation,
    ClaudeModel,
    SessionUsageStats,
)
from .live_refresh import (
    LiveRefreshManager,
    RefreshMode,
    RefreshThrottleConfig,
    RefreshState,
    RefreshResult,
    start_live_refresh,
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
    # Claude Budget Manager
    "ClaudeBudgetManager",
    "ClaudeBudgetAllocation",
    "ClaudeModel",
    "SessionUsageStats",
    # Claude Code Detection
    "ClaudeCodeDetector",
    "ClaudeCodeSession",
    "is_claude_code_session",
    "get_claude_code_session",
    # CLAUDE.md Injection
    "ClaudeMdInjector",
    "ClaudeMdLocation",
    "inject_claude_md_content",
    # System Prompt Injection
    "SystemPromptInjector",
    "SystemPromptConfig",
    "InjectionMode",
    "generate_mcp_context",
    "generate_api_system_prompt",
    # Session Triggers
    "SessionTrigger",
    "TriggerPhase",
    "TriggerPriority",
    "TriggerResult",
    "TriggerContext",
    "TriggerManager",
    "ClaudeMdFileTrigger",
    "SystemPromptTrigger",
    "CleanupTrigger",
    "CustomCallbackTrigger",
    "prepare_claude_code_session",
    "cleanup_claude_code_session",
    # Live Refresh
    "LiveRefreshManager",
    "RefreshMode",
    "RefreshThrottleConfig",
    "RefreshState",
    "RefreshResult",
    "start_live_refresh",
]
