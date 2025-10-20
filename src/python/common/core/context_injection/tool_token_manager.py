"""
Tool-specific token limit management for context injection.

This module provides per-tool token limit validation and enforcement to ensure
that formatted context respects the token limits of different LLM tools.

Each tool has different token limits:
- Claude Code: 200,000 tokens (context window)
- GitHub Copilot: 8,192 tokens (typical Codex limit)
- Google Gemini: 1,000,000 tokens (extended context)
- Cursor: 8,192 tokens (uses Codex-style limits)
- JetBrains AI: 8,192 tokens (conservative estimate)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger

from .llm_tool_detector import LLMToolDetector, LLMToolType


@dataclass
class ToolTokenLimits:
    """
    Per-tool token limit configuration.

    Attributes:
        tool_type: Type of LLM tool
        max_context_tokens: Maximum context window size in tokens
        recommended_budget: Recommended budget (80% of max for safety margin)
        warning_threshold: Usage percentage that triggers warnings (0-1)
        critical_threshold: Usage percentage that triggers critical warnings (0-1)
    """

    tool_type: LLMToolType
    max_context_tokens: int
    recommended_budget: int
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%

    def __post_init__(self):
        """Validate threshold values."""
        if not 0 <= self.warning_threshold <= 1:
            raise ValueError(
                f"warning_threshold must be 0-1, got {self.warning_threshold}"
            )
        if not 0 <= self.critical_threshold <= 1:
            raise ValueError(
                f"critical_threshold must be 0-1, got {self.critical_threshold}"
            )
        if self.warning_threshold > self.critical_threshold:
            raise ValueError(
                "warning_threshold cannot exceed critical_threshold "
                f"({self.warning_threshold} > {self.critical_threshold})"
            )


class ToolTokenManager:
    """
    Manages token limits per LLM tool.

    Provides:
    - Per-tool token limit lookup with conservative defaults
    - Token count validation against tool limits
    - Warning and critical threshold detection
    - Recommended budget calculation (80% of max)
    - Usage percentage calculation for progress tracking

    All methods are class methods as there is no instance state.
    """

    # Static token limit mapping
    # NOTE: These values come from ToolCapabilities in formatter adapters
    # and represent the actual context windows of each tool
    _TOOL_LIMITS = {
        LLMToolType.CLAUDE_CODE: ToolTokenLimits(
            tool_type=LLMToolType.CLAUDE_CODE,
            max_context_tokens=200_000,  # Claude 3.5 Sonnet/Opus context window
            recommended_budget=160_000,  # 80% for safety
        ),
        LLMToolType.GITHUB_COPILOT: ToolTokenLimits(
            tool_type=LLMToolType.GITHUB_COPILOT,
            max_context_tokens=8_192,  # Codex context limit
            recommended_budget=6_553,  # 80% for safety
        ),
        LLMToolType.CODEX_API: ToolTokenLimits(
            tool_type=LLMToolType.CODEX_API,
            max_context_tokens=8_192,  # Codex API limit
            recommended_budget=6_553,  # 80% for safety
        ),
        LLMToolType.CURSOR: ToolTokenLimits(
            tool_type=LLMToolType.CURSOR,
            max_context_tokens=8_192,  # Uses Codex-style limits
            recommended_budget=6_553,  # 80% for safety
        ),
        LLMToolType.JETBRAINS_AI: ToolTokenLimits(
            tool_type=LLMToolType.JETBRAINS_AI,
            max_context_tokens=8_192,  # Conservative estimate
            recommended_budget=6_553,  # 80% for safety
        ),
        LLMToolType.GOOGLE_GEMINI: ToolTokenLimits(
            tool_type=LLMToolType.GOOGLE_GEMINI,
            max_context_tokens=1_000_000,  # Gemini 1.5 Pro extended context
            recommended_budget=800_000,  # 80% for safety
        ),
        LLMToolType.TABNINE: ToolTokenLimits(
            tool_type=LLMToolType.TABNINE,
            max_context_tokens=4_096,  # Conservative estimate
            recommended_budget=3_276,  # 80% for safety
        ),
    }

    # Conservative default for unknown tools
    _DEFAULT_LIMITS = ToolTokenLimits(
        tool_type=LLMToolType.UNKNOWN,
        max_context_tokens=4_096,  # Very conservative (smallest common limit)
        recommended_budget=3_276,  # 80% for safety
    )

    @classmethod
    def get_limits(cls, tool_type: LLMToolType) -> ToolTokenLimits:
        """
        Get token limits for a specific LLM tool.

        Args:
            tool_type: Type of LLM tool

        Returns:
            ToolTokenLimits configuration for the tool.
            Returns conservative defaults if tool type is unknown.

        Example:
            >>> from context_injection import ToolTokenManager, LLMToolType
            >>> limits = ToolTokenManager.get_limits(LLMToolType.CLAUDE_CODE)
            >>> print(f"Max tokens: {limits.max_context_tokens}")
            Max tokens: 200000
        """
        limits = cls._TOOL_LIMITS.get(tool_type, cls._DEFAULT_LIMITS)

        # Log warning for unknown tools
        if tool_type not in cls._TOOL_LIMITS and tool_type != LLMToolType.UNKNOWN:
            logger.warning(
                f"Unknown tool type: {tool_type.value}. "
                f"Using conservative default limit of {cls._DEFAULT_LIMITS.max_context_tokens} tokens"
            )

        return limits

    @classmethod
    def validate_token_count(
        cls, tool_type: LLMToolType, token_count: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate token count against tool's limit.

        Returns validation status and optional warning/error message.

        Args:
            tool_type: Type of LLM tool
            token_count: Number of tokens to validate

        Returns:
            Tuple of (is_valid, message):
            - is_valid: False if count exceeds max, True otherwise
            - message: None if valid, warning at 80%, error at 100%+

        Example:
            >>> is_valid, msg = ToolTokenManager.validate_token_count(
            ...     LLMToolType.CODEX_API, 7000
            ... )
            >>> if msg:
            ...     print(msg)
            WARNING: Token count approaching limit (7000/8192, 85.4%)
        """
        limits = cls.get_limits(tool_type)

        # Check for invalid token counts
        if token_count < 0:
            return False, f"ERROR: Token count cannot be negative: {token_count}"

        if token_count == 0:
            return True, None  # Zero tokens is valid (empty context)

        # Calculate usage percentage
        usage_pct = cls.calculate_usage_percentage(tool_type, token_count)

        # Check if exceeds maximum
        if token_count > limits.max_context_tokens:
            return (
                False,
                f"ERROR: Token count exceeds {tool_type.value} limit "
                f"({token_count:,} / {limits.max_context_tokens:,} tokens, {usage_pct:.1f}%). "
                f"Reduce context or use rule prioritization.",
            )

        # Check critical threshold (95%)
        if usage_pct >= limits.critical_threshold * 100:
            return (
                True,
                f"CRITICAL: Token count near limit "
                f"({token_count:,} / {limits.max_context_tokens:,} tokens, {usage_pct:.1f}%). "
                f"Remaining: {limits.max_context_tokens - token_count:,} tokens.",
            )

        # Check warning threshold (80%)
        if usage_pct >= limits.warning_threshold * 100:
            return (
                True,
                f"WARNING: Token count approaching limit "
                f"({token_count:,} / {limits.max_context_tokens:,} tokens, {usage_pct:.1f}%). "
                f"Recommended budget: {limits.recommended_budget:,} tokens.",
            )

        # Valid and under thresholds
        return True, None

    @classmethod
    def get_recommended_budget(cls, tool_type: Optional[LLMToolType] = None) -> int:
        """
        Get recommended token budget for a tool (80% of max).

        Auto-detects tool if not specified using LLMToolDetector.

        Args:
            tool_type: Type of LLM tool (auto-detect if None)

        Returns:
            Recommended token budget (80% of max context tokens)

        Example:
            >>> # Auto-detect current tool
            >>> budget = ToolTokenManager.get_recommended_budget()
            >>> print(f"Recommended budget: {budget:,} tokens")

            >>> # Explicit tool type
            >>> budget = ToolTokenManager.get_recommended_budget(LLMToolType.CLAUDE_CODE)
            >>> print(f"Claude budget: {budget:,} tokens")
            Claude budget: 160,000 tokens
        """
        # Auto-detect tool if not specified
        if tool_type is None:
            session = LLMToolDetector.detect()
            if session.is_active:
                tool_type = session.tool_type
                logger.debug(f"Auto-detected tool type: {tool_type.value}")
            else:
                # No active session - use conservative default
                logger.debug(
                    "No active LLM tool detected, using conservative default budget"
                )
                return cls._DEFAULT_LIMITS.recommended_budget

        limits = cls.get_limits(tool_type)
        return limits.recommended_budget

    @classmethod
    def calculate_usage_percentage(
        cls, tool_type: LLMToolType, token_count: int
    ) -> float:
        """
        Calculate token usage as percentage of tool's maximum.

        Useful for progress indicators and usage tracking.

        Args:
            tool_type: Type of LLM tool
            token_count: Current token count

        Returns:
            Usage percentage (0-100+, can exceed 100 if over limit)

        Example:
            >>> pct = ToolTokenManager.calculate_usage_percentage(
            ...     LLMToolType.CODEX_API, 4096
            ... )
            >>> print(f"Usage: {pct:.1f}%")
            Usage: 50.0%
        """
        limits = cls.get_limits(tool_type)

        if limits.max_context_tokens <= 0:
            logger.warning(
                f"Invalid max_context_tokens for {tool_type.value}: "
                f"{limits.max_context_tokens}"
            )
            return 0.0

        return (token_count / limits.max_context_tokens) * 100.0

    @classmethod
    def get_all_limits(cls) -> dict[LLMToolType, ToolTokenLimits]:
        """
        Get token limits for all known tools.

        Returns:
            Dictionary mapping tool types to their limits

        Example:
            >>> limits = ToolTokenManager.get_all_limits()
            >>> for tool_type, limits in limits.items():
            ...     print(f"{tool_type.value}: {limits.max_context_tokens:,} tokens")
        """
        return cls._TOOL_LIMITS.copy()
