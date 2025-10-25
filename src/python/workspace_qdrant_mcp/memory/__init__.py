"""
Memory system facade for workspace-qdrant-mcp.

This module provides a facade to the actual memory implementation
in common.memory, maintaining backward compatibility with test imports.
"""

# Re-export all components from common.memory
from common.memory import (
    AuthorityLevel,
    ClaudeCodeIntegration,
    ConflictDetector,
    MemoryCategory,
    MemoryCollectionSchema,
    MemoryManager,
    MemoryRule,
    MemoryRuleConflict,
    TokenCounter,
    TokenUsage,
)

__all__ = [
    "AuthorityLevel",
    "ClaudeCodeIntegration",
    "ConflictDetector",
    "MemoryCategory",
    "MemoryCollectionSchema",
    "MemoryManager",
    "MemoryRule",
    "MemoryRuleConflict",
    "TokenCounter",
    "TokenUsage",
]
