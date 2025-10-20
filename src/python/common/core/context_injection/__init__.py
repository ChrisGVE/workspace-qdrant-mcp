"""
Context injection system for LLM tools.

This module provides automatic context injection into LLM tool sessions using
memory rules stored in Qdrant.
"""

from .authority_filter import AuthorityFilter, FilteredRules, RuleHierarchy
from .claude_code_detector import (
    ClaudeCodeDetector,
    ClaudeCodeSession,
    ClaudeSessionMonitor,
    ProjectContextMetadata,
    SessionEvent,
    get_claude_code_session,
    is_claude_code_session,
)
from .copilot_detector import (
    CopilotDetector,
    CopilotSession,
    CopilotSessionType,
    get_copilot_session,
    is_copilot_session,
    get_code_comment_prefix,
)
from .llm_tool_detector import (
    LLMToolDetector,
    LLMToolType,
    UnifiedLLMSession,
    get_active_llm_tool,
    is_llm_tool_active,
    get_llm_formatter,
)
from .comment_injector import (
    CommentInjector,
    PlacementStrategy,
    ConflictResolution,
    ConflictType,
    RuleConflict,
    InjectedComment,
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
    CacheStatistics,
    CompressionStrategy,
    TokenBudgetManager,
    TokenCountCache,
    TokenCounter,
    TokenizerFactory,
    TokenizerType,
)
from .claude_budget_manager import (
    ClaudeBudgetManager,
    ClaudeBudgetAllocation,
    ClaudeModel,
    SessionUsageStats,
)
from .token_usage_tracker import (
    GlobalUsageTracker,
    OperationType,
    OperationUsage,
    SessionUsageSnapshot,
    TokenUsageTracker,
    ToolUsageStats,
)
from .budget_warning_system import (
    BudgetWarningSystem,
    WarningLevel,
    WarningEvent,
    BudgetThreshold,
    ThrottleConfig,
)
from .live_refresh import (
    LiveRefreshManager,
    RefreshMode,
    RefreshThrottleConfig,
    RefreshState,
    RefreshResult,
    start_live_refresh,
)
from .rule_prioritizer import (
    PrioritizationResult,
    PrioritizationStrategy,
    RulePrioritizer,
    RulePriorityScore,
)
from .interactive_trimmer import (
    BudgetVisualization,
    InteractiveTrimmer,
    RuleDisplay,
    TrimDecision,
    TrimDecisionType,
    TrimSession,
)
from .budget_config import (
    BudgetConfig,
    BudgetConfigManager,
    BudgetScope,
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
    "TokenCountCache",
    "CacheStatistics",
    "TokenizerFactory",
    "TokenizerType",
    # Claude Budget Manager
    "ClaudeBudgetManager",
    "ClaudeBudgetAllocation",
    "ClaudeModel",
    "SessionUsageStats",
    # Token Usage Tracker
    "TokenUsageTracker",
    "OperationType",
    "OperationUsage",
    "ToolUsageStats",
    "SessionUsageSnapshot",
    "GlobalUsageTracker",
    # Budget Warning System
    "BudgetWarningSystem",
    "WarningLevel",
    "WarningEvent",
    "BudgetThreshold",
    "ThrottleConfig",
    # Budget Config Manager
    "BudgetConfig",
    "BudgetConfigManager",
    "BudgetScope",
    # Rule Prioritizer
    "RulePrioritizer",
    "PrioritizationStrategy",
    "RulePriorityScore",
    "PrioritizationResult",
    # Interactive Trimmer
    "InteractiveTrimmer",
    "TrimDecision",
    "TrimDecisionType",
    "TrimSession",
    "BudgetVisualization",
    "RuleDisplay",
    # Claude Code Detection
    "ClaudeCodeDetector",
    "ClaudeCodeSession",
    "ClaudeSessionMonitor",
    "ProjectContextMetadata",
    "SessionEvent",
    "is_claude_code_session",
    "get_claude_code_session",
    # Copilot/Codex Detection
    "CopilotDetector",
    "CopilotSession",
    "CopilotSessionType",
    "is_copilot_session",
    "get_copilot_session",
    "get_code_comment_prefix",
    # Unified LLM Detection
    "LLMToolDetector",
    "LLMToolType",
    "UnifiedLLMSession",
    "get_active_llm_tool",
    "is_llm_tool_active",
    "get_llm_formatter",
    # Code Comment Injection
    "CommentInjector",
    "PlacementStrategy",
    "ConflictResolution",
    "ConflictType",
    "RuleConflict",
    "InjectedComment",
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
