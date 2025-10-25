"""
LLM-specific formatters for memory rule context injection.

This module provides adapters for formatting memory rules into
LLM-specific formats (Claude Code, GitHub Codex, Google Gemini).
"""

from .base import (
    FormattedContext,
    FormatType,
    LLMToolAdapter,
    ToolCapabilities,
)
from .claude_code import ClaudeCodeAdapter
from .github_codex import GitHubCodexAdapter
from .google_gemini import GoogleGeminiAdapter
from .manager import FormatManager

__all__ = [
    # Base classes and dataclasses
    "FormatType",
    "FormattedContext",
    "LLMToolAdapter",
    "ToolCapabilities",
    # Adapters
    "ClaudeCodeAdapter",
    "GitHubCodexAdapter",
    "GoogleGeminiAdapter",
    # Manager
    "FormatManager",
]
