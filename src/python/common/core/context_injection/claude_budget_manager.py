"""
Claude-specific token budget management for context injection.

This module provides Claude model-specific token budget management with session
tracking, model-specific limits, and usage monitoring for Claude Code integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger

from ..memory import MemoryRule
from .token_budget import (
    AllocationStrategy,
    BudgetAllocation,
    CompressionStrategy,
    TokenBudgetManager,
    TokenCounter,
)


class ClaudeModel(Enum):
    """Claude model variants with token limits."""

    # Claude 3.5 models (200K context)
    SONNET_3_5 = ("claude-3-5-sonnet-20241022", 200000)
    SONNET_3_5_OLD = ("claude-3-5-sonnet-20240620", 200000)

    # Claude 3 Opus (200K context)
    OPUS_3 = ("claude-3-opus-20240229", 200000)

    # Claude 3 Sonnet (200K context)
    SONNET_3 = ("claude-3-sonnet-20240229", 200000)

    # Claude 3 Haiku (200K context)
    HAIKU_3 = ("claude-3-haiku-20240307", 200000)

    # Default/unknown (conservative 200K)
    DEFAULT = ("claude-default", 200000)

    def __init__(self, model_id: str, token_limit: int):
        """
        Initialize Claude model enum.

        Args:
            model_id: Model identifier string
            token_limit: Maximum context tokens for this model
        """
        self.model_id = model_id
        self.token_limit = token_limit

    @classmethod
    def from_model_id(cls, model_id: str) -> "ClaudeModel":
        """
        Get Claude model variant from model ID string.

        Args:
            model_id: Model identifier (e.g., "claude-3-5-sonnet-20241022")

        Returns:
            ClaudeModel enum variant, or DEFAULT if not recognized
        """
        for model in cls:
            if model.model_id == model_id:
                return model

        # Try partial matching for flexibility
        model_id_lower = model_id.lower()
        if "3-5-sonnet" in model_id_lower or "3.5-sonnet" in model_id_lower:
            return cls.SONNET_3_5
        elif "3-opus" in model_id_lower or "3.opus" in model_id_lower:
            return cls.OPUS_3
        elif "3-sonnet" in model_id_lower or "3.sonnet" in model_id_lower:
            return cls.SONNET_3
        elif "3-haiku" in model_id_lower or "3.haiku" in model_id_lower:
            return cls.HAIKU_3

        logger.warning(f"Unknown Claude model ID: {model_id}, using DEFAULT")
        return cls.DEFAULT


@dataclass
class SessionUsageStats:
    """
    Token usage statistics for a Claude Code session.

    Tracks cumulative token usage across multiple interactions within
    a single Claude Code session.

    Attributes:
        session_id: Unique session identifier
        model: Claude model variant
        total_tokens_used: Cumulative tokens used in session
        total_context_tokens: Cumulative tokens from context injection
        total_user_tokens: Cumulative tokens from user queries
        interaction_count: Number of interactions in session
        started_at: Session start timestamp
        last_interaction_at: Most recent interaction timestamp
        budget_limit: Token budget limit for the session
        warnings_triggered: List of warning threshold levels hit
    """

    session_id: str
    model: ClaudeModel
    total_tokens_used: int = 0
    total_context_tokens: int = 0
    total_user_tokens: int = 0
    interaction_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    budget_limit: int = 200000
    warnings_triggered: List[int] = field(default_factory=list)

    def utilization_percentage(self) -> float:
        """
        Calculate current budget utilization percentage.

        Returns:
            Percentage of budget used (0-100+)
        """
        if self.budget_limit <= 0:
            return 0.0
        return (self.total_tokens_used / self.budget_limit) * 100.0

    def tokens_remaining(self) -> int:
        """
        Calculate remaining tokens in budget.

        Returns:
            Number of tokens remaining (can be negative if over budget)
        """
        return self.budget_limit - self.total_tokens_used

    def is_over_budget(self) -> bool:
        """
        Check if session has exceeded budget limit.

        Returns:
            True if over budget, False otherwise
        """
        return self.total_tokens_used > self.budget_limit


@dataclass
class ClaudeBudgetAllocation:
    """
    Extended budget allocation with Claude-specific session tracking.

    Wraps BudgetAllocation with additional session-specific metadata
    and usage tracking.

    Attributes:
        base_allocation: Base budget allocation from TokenBudgetManager
        session_stats: Session usage statistics
        context_overhead_tokens: Tokens used for context injection overhead
        model: Claude model variant
        warning_level: Warning threshold triggered (0-100), None if no warning
    """

    base_allocation: BudgetAllocation
    session_stats: SessionUsageStats
    context_overhead_tokens: int
    model: ClaudeModel
    warning_level: Optional[int] = None

    @property
    def total_budget(self) -> int:
        """Total available tokens."""
        return self.base_allocation.total_budget

    @property
    def rules_included(self) -> List[MemoryRule]:
        """Rules included in allocation."""
        return self.base_allocation.rules_included

    @property
    def rules_skipped(self) -> List[MemoryRule]:
        """Rules skipped due to budget constraints."""
        return self.base_allocation.rules_skipped


class ClaudeBudgetManager:
    """
    Claude-specific token budget manager with session tracking.

    Extends TokenBudgetManager with Claude model-specific features:
    - Model-specific token limits (Opus, Sonnet, Haiku)
    - Session token accumulation across interactions
    - Context injection overhead tracking
    - Warning thresholds (80%, 90%, 95%)
    - Usage analytics and reporting
    """

    # Warning threshold percentages
    WARNING_THRESHOLDS = [80, 90, 95]

    def __init__(
        self,
        model: ClaudeModel = ClaudeModel.DEFAULT,
        session_id: Optional[str] = None,
        allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED,
        compression_strategy: CompressionStrategy = CompressionStrategy.NONE,
        absolute_rules_protected: bool = True,
        overhead_percentage: float = 0.05,
        custom_budget_limit: Optional[int] = None,
    ):
        """
        Initialize Claude budget manager.

        Args:
            model: Claude model variant to use
            session_id: Optional session identifier (generated if not provided)
            allocation_strategy: Strategy for allocating budget to rules
            compression_strategy: Strategy for compressing rules if needed
            absolute_rules_protected: If True, absolute rules always included
            overhead_percentage: Percentage reserved for formatting overhead
            custom_budget_limit: Optional custom budget limit (overrides model default)
        """
        self.model = model
        self.session_id = session_id or self._generate_session_id()

        # Initialize base token budget manager
        self.base_manager = TokenBudgetManager(
            allocation_strategy=allocation_strategy,
            compression_strategy=compression_strategy,
            absolute_rules_protected=absolute_rules_protected,
            overhead_percentage=overhead_percentage,
        )

        # Initialize session stats
        budget_limit = custom_budget_limit or model.token_limit
        self.session_stats = SessionUsageStats(
            session_id=self.session_id,
            model=model,
            budget_limit=budget_limit,
        )

        logger.info(
            f"Initialized Claude budget manager for {model.model_id} "
            f"(session: {self.session_id}, limit: {budget_limit} tokens)"
        )

    def allocate_budget(
        self,
        rules: List[MemoryRule],
        user_query_tokens: int = 0,
        user_query: Optional[str] = None,
    ) -> ClaudeBudgetAllocation:
        """
        Allocate token budget for memory rules with session tracking.

        Accounts for:
        1. Cumulative session token usage
        2. User query token cost (current interaction)
        3. Context injection overhead
        4. Model-specific token limits
        5. Warning thresholds

        Args:
            rules: Memory rules to allocate budget for
            user_query_tokens: Pre-counted tokens for user query (optional)
            user_query: User query text for token counting (optional)

        Returns:
            ClaudeBudgetAllocation with session tracking and warnings
        """
        # Count user query tokens if not provided
        if user_query and user_query_tokens == 0:
            user_query_tokens = TokenCounter.count_claude_tokens(user_query)

        # Calculate available budget for this interaction
        # Total budget - session usage - user query = available for context
        available_for_context = (
            self.session_stats.budget_limit
            - self.session_stats.total_tokens_used
            - user_query_tokens
        )

        # Ensure non-negative budget
        available_for_context = max(0, available_for_context)

        # Allocate budget using base manager
        base_allocation = self.base_manager.allocate_budget(
            rules=rules,
            total_budget=available_for_context,
            tool_name="claude",
        )

        # Calculate context overhead (formatting, headers, etc.)
        context_overhead = base_allocation.overhead_tokens

        # Update session stats
        context_tokens = (
            base_allocation.absolute_tokens
            + base_allocation.default_tokens
            + context_overhead
        )
        total_interaction_tokens = context_tokens + user_query_tokens

        self.session_stats.total_tokens_used += total_interaction_tokens
        self.session_stats.total_context_tokens += context_tokens
        self.session_stats.total_user_tokens += user_query_tokens
        self.session_stats.interaction_count += 1
        self.session_stats.last_interaction_at = datetime.now(timezone.utc)

        # Check warning thresholds
        warning_level = self._check_warning_thresholds()

        # Create Claude-specific allocation
        claude_allocation = ClaudeBudgetAllocation(
            base_allocation=base_allocation,
            session_stats=self.session_stats,
            context_overhead_tokens=context_overhead,
            model=self.model,
            warning_level=warning_level,
        )

        # Log allocation details
        logger.debug(
            f"Budget allocation: {context_tokens} context + {user_query_tokens} query = "
            f"{total_interaction_tokens} tokens (session total: {self.session_stats.total_tokens_used})"
        )

        if warning_level:
            logger.warning(
                f"Budget utilization: {self.session_stats.utilization_percentage():.1f}% "
                f"(warning threshold: {warning_level}%)"
            )

        return claude_allocation

    def reset_session(self, new_session_id: Optional[str] = None) -> None:
        """
        Reset session tracking for a new Claude Code session.

        Args:
            new_session_id: Optional new session ID (generated if not provided)
        """
        old_session_id = self.session_id
        self.session_id = new_session_id or self._generate_session_id()

        # Reset session stats
        self.session_stats = SessionUsageStats(
            session_id=self.session_id,
            model=self.model,
            budget_limit=self.session_stats.budget_limit,
        )

        logger.info(
            f"Reset session tracking: {old_session_id} -> {self.session_id}"
        )

    def get_session_report(self) -> Dict[str, any]:
        """
        Generate comprehensive session usage report.

        Returns:
            Dictionary with session statistics and analytics
        """
        utilization = self.session_stats.utilization_percentage()
        remaining = self.session_stats.tokens_remaining()

        return {
            "session_id": self.session_stats.session_id,
            "model": self.model.model_id,
            "budget_limit": self.session_stats.budget_limit,
            "tokens_used": self.session_stats.total_tokens_used,
            "context_tokens": self.session_stats.total_context_tokens,
            "user_tokens": self.session_stats.total_user_tokens,
            "tokens_remaining": remaining,
            "utilization_percentage": utilization,
            "interaction_count": self.session_stats.interaction_count,
            "started_at": self.session_stats.started_at.isoformat(),
            "last_interaction_at": self.session_stats.last_interaction_at.isoformat(),
            "warnings_triggered": self.session_stats.warnings_triggered,
            "is_over_budget": self.session_stats.is_over_budget(),
            "average_tokens_per_interaction": (
                self.session_stats.total_tokens_used
                / self.session_stats.interaction_count
                if self.session_stats.interaction_count > 0
                else 0
            ),
        }

    def _check_warning_thresholds(self) -> Optional[int]:
        """
        Check if budget utilization crosses warning thresholds.

        Tracks which thresholds have been triggered and returns the
        highest active threshold.

        Returns:
            Warning level (80, 90, 95) if threshold crossed, None otherwise
        """
        utilization = self.session_stats.utilization_percentage()
        triggered_level = None

        for threshold in self.WARNING_THRESHOLDS:
            if utilization >= threshold:
                triggered_level = threshold
                if threshold not in self.session_stats.warnings_triggered:
                    self.session_stats.warnings_triggered.append(threshold)
                    logger.warning(
                        f"Budget warning: {utilization:.1f}% utilization "
                        f"(threshold: {threshold}%)"
                    )

        return triggered_level

    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate unique session identifier.

        Returns:
            Session ID string with timestamp
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return f"claude_session_{timestamp}"

    @staticmethod
    def detect_model_from_environment() -> ClaudeModel:
        """
        Detect Claude model from environment variables.

        Checks common environment variables that might indicate the
        Claude model being used.

        Returns:
            Detected ClaudeModel or DEFAULT if not detected
        """
        import os

        # Check for model environment variables
        model_var_names = [
            "CLAUDE_MODEL",
            "ANTHROPIC_MODEL",
            "CLAUDE_CODE_MODEL",
        ]

        for var_name in model_var_names:
            model_id = os.environ.get(var_name)
            if model_id:
                logger.debug(f"Detected Claude model from {var_name}: {model_id}")
                return ClaudeModel.from_model_id(model_id)

        logger.debug("No Claude model detected in environment, using DEFAULT")
        return ClaudeModel.DEFAULT
