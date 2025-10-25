"""
Context injection system for LLM tools.

This module provides automatic context injection into LLM tool sessions using
memory rules stored in Qdrant.
"""

from .authority_filter import AuthorityFilter, FilteredRules, RuleHierarchy
from .budget_config import (
    BudgetConfig,
    BudgetConfigManager,
    BudgetScope,
)
from .budget_warning_system import (
    BudgetThreshold,
    BudgetWarningSystem,
    ThrottleConfig,
    WarningEvent,
    WarningLevel,
)
from .claude_budget_manager import (
    ClaudeBudgetAllocation,
    ClaudeBudgetManager,
    ClaudeModel,
    SessionUsageStats,
)
from .claude_code_detector import (
    ClaudeCodeDetector,
    ClaudeCodeSession,
    ClaudeSessionMonitor,
    ProjectContextMetadata,
    SessionEvent,
    get_claude_code_session,
    is_claude_code_session,
)
from .claude_md_injector import (
    ClaudeMdInjector,
    ClaudeMdLocation,
    inject_claude_md_content,
)
from .comment_injector import (
    CommentInjector,
    ConflictResolution,
    ConflictType,
    InjectedComment,
    PlacementStrategy,
    RuleConflict,
)
from .context_switcher import (
    ContextSwitcher,
    SwitchValidationResult,
)
from .copilot_detector import (
    CopilotDetector,
    CopilotSession,
    CopilotSessionType,
    get_code_comment_prefix,
    get_copilot_session,
    is_copilot_session,
)
from .formatters import (
    ClaudeCodeAdapter,
    FormatManager,
    FormattedContext,
    FormatType,
    GitHubCodexAdapter,
    GoogleGeminiAdapter,
    LLMToolAdapter,
    ToolCapabilities,
)
from .interactive_trimmer import (
    BudgetVisualization,
    InteractiveTrimmer,
    RuleDisplay,
    TrimDecision,
    TrimDecisionType,
    TrimSession,
)
from .live_refresh import (
    LiveRefreshManager,
    RefreshMode,
    RefreshResult,
    RefreshState,
    RefreshThrottleConfig,
    start_live_refresh,
)
from .llm_override_config import (
    LLMOverrideConfig,
    LLMOverrideManager,
    clear_llm_override_cli,
    set_llm_override_cli,
    show_llm_override_cli,
)
from .llm_tool_detector import (
    LLMToolDetector,
    LLMToolType,
    UnifiedLLMSession,
    get_active_llm_tool,
    get_llm_formatter,
    is_llm_tool_active,
)
from .performance_optimization import (
    BatchProcessor,
    PerformanceMetrics,
    RuleCache,
    RuleIndex,
)
from .project_context import (
    ProjectContext,
    ProjectContextDetector,
    ProjectRuleApplicator,
)
from .rule_prioritizer import (
    PrioritizationResult,
    PrioritizationStrategy,
    RulePrioritizer,
    RulePriorityScore,
)
from .rule_retrieval import RuleFilter, RuleRetrieval, RuleRetrievalResult
from .session_trigger import (
    ClaudeMdFileTrigger,
    CleanupTrigger,
    CustomCallbackTrigger,
    OnDemandRefreshTrigger,
    PostUpdateTrigger,
    SessionTrigger,
    SystemPromptTrigger,
    ToolAwareTrigger,
    TriggerContext,
    TriggerEvent,
    TriggerEventLogger,
    TriggerHealthMetrics,
    TriggerHealthMonitor,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerResult,
    TriggerRetryPolicy,
    cleanup_claude_code_session,
    prepare_claude_code_session,
    refresh_claude_code_context,
)
from .system_prompt_injector import (
    InjectionMode,
    SystemPromptConfig,
    SystemPromptInjector,
    generate_api_system_prompt,
    generate_mcp_context,
)
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
from .token_usage_tracker import (
    GlobalUsageTracker,
    OperationType,
    OperationUsage,
    SessionUsageSnapshot,
    TokenUsageTracker,
    ToolUsageStats,
)
from .tool_token_manager import (
    ToolTokenLimits,
    ToolTokenManager,
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
    # Tool Token Manager
    "ToolTokenLimits",
    "ToolTokenManager",
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
    # LLM Override Configuration
    "LLMOverrideConfig",
    "LLMOverrideManager",
    "set_llm_override_cli",
    "clear_llm_override_cli",
    "show_llm_override_cli",
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
    "OnDemandRefreshTrigger",
    "PostUpdateTrigger",
    "ToolAwareTrigger",
    "TriggerEvent",
    "TriggerHealthMetrics",
    "TriggerRetryPolicy",
    "TriggerEventLogger",
    "TriggerHealthMonitor",
    "prepare_claude_code_session",
    "cleanup_claude_code_session",
    "refresh_claude_code_context",
    # Live Refresh
    "LiveRefreshManager",
    "RefreshMode",
    "RefreshThrottleConfig",
    "RefreshState",
    "RefreshResult",
    "start_live_refresh",
    # Context Switcher
    "ContextSwitcher",
    "SwitchValidationResult",
]
