"""
Memory system type definitions.

This module defines the core data structures for the memory-driven LLM behavior system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AuthorityLevel(Enum):
    """Authority levels for memory rules."""

    ABSOLUTE = "absolute"  # Non-negotiable, always follow
    DEFAULT = "default"  # Follow unless explicitly overridden by user/PRD


class MemoryCategory(Enum):
    """Categories of memory rules."""

    PREFERENCE = "preference"  # User preferences (e.g., "Use uv for Python")
    BEHAVIOR = "behavior"  # LLM behavioral rules (e.g., "Always make atomic commits")
    FORMAT = "format"  # Formatting and style rules
    AGENT_LIBRARY = "agent_library"  # Agent definitions and capabilities
    KNOWLEDGE = "knowledge"  # Factual knowledge and context
    CONTEXT = "context"  # Session and project context


@dataclass
class MemoryRule:
    """A single memory rule for LLM behavior management."""

    # Rule content (required fields first)
    rule: str  # The actual rule text
    category: MemoryCategory  # Rule category
    authority: AuthorityLevel  # Authority level
    name: str | None = None  # Optional short identifier for the rule

    # Optional fields with defaults
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: list[str] = field(
        default_factory=list
    )  # Contexts where rule applies (empty = global)
    tags: list[str] = field(default_factory=list)  # Tags for organization and search
    source: str = "user_cli"  # Where rule came from (user_cli, conversation, etc.)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    last_used: datetime | None = None
    use_count: int = 0  # How many times rule has been applied
    metadata: dict[str, Any] = field(default_factory=dict)  # Extended metadata

    def __post_init__(self):
        """Validate rule after initialization."""
        if not self.rule or not self.rule.strip():
            raise ValueError("Rule text cannot be empty")
        if not isinstance(self.category, MemoryCategory):
            raise ValueError(f"Invalid category: {self.category}")
        if not isinstance(self.authority, AuthorityLevel):
            raise ValueError(f"Invalid authority level: {self.authority}")

    def update_usage(self):
        """Update usage statistics."""
        self.use_count += 1
        self.last_used = lambda: datetime.now(timezone.utc)()
        self.updated_at = lambda: datetime.now(timezone.utc)()

    def matches_scope(self, context_scope: list[str]) -> bool:
        """Check if this rule applies to the given context scope."""
        if not self.scope:  # Empty scope means global
            return True
        return any(scope_item in context_scope for scope_item in self.scope)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "rule": self.rule,
            "name": self.name,
            "category": self.category.value,
            "authority": self.authority.value,
            "scope": self.scope,
            "tags": self.tags,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRule":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            rule=data["rule"],
            name=data.get("name"),
            category=MemoryCategory(data["category"]),
            authority=AuthorityLevel(data["authority"]),
            scope=data.get("scope", []),
            tags=data.get("tags", []),
            source=data.get("source", "user_cli"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            last_used=datetime.fromisoformat(data["last_used"])
            if data.get("last_used")
            else None,
            use_count=data.get("use_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryRuleConflict:
    """Represents a conflict between memory rules."""

    rule1: MemoryRule
    rule2: MemoryRule
    conflict_type: str  # semantic, scope, direct, etc.
    confidence: float  # Confidence level of conflict (0.0 to 1.0)
    description: str  # Human-readable conflict description
    severity: str  # low, medium, high, critical
    resolution_suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule1_id": self.rule1.id,
            "rule2_id": self.rule2.id,
            "rule1_text": self.rule1.rule,
            "rule2_text": self.rule2.rule,
            "conflict_type": self.conflict_type,
            "confidence": self.confidence,
            "description": self.description,
            "severity": self.severity,
            "resolution_suggestion": self.resolution_suggestion,
        }


@dataclass
class MemoryContext:
    """Context information for memory rule application."""

    session_id: str
    project_name: str | None = None
    project_path: str | None = None
    user_name: str | None = None
    agent_type: str | None = None
    conversation_context: list[str] = field(default_factory=list)
    active_scopes: list[str] = field(default_factory=list)

    def to_scope_list(self) -> list[str]:
        """Convert context to list of scope strings for rule matching."""
        scopes = []

        if self.project_name:
            scopes.append(f"project:{self.project_name}")
        if self.agent_type:
            scopes.append(f"agent:{self.agent_type}")
        if self.user_name:
            scopes.append(f"user:{self.user_name}")

        # Add conversation context items
        scopes.extend(self.conversation_context)

        # Add explicitly active scopes
        scopes.extend(self.active_scopes)

        return scopes


@dataclass
class ConversationalUpdate:
    """Represents a conversational memory update."""

    text: str  # The conversational text containing the update
    extracted_rule: str | None = None  # Extracted rule text
    category: MemoryCategory | None = None
    authority: AuthorityLevel | None = None
    scope: list[str] = field(default_factory=list)
    confidence: float = 0.0  # Confidence in the extraction

    def is_valid(self) -> bool:
        """Check if this update contains a valid extractable rule."""
        return (
            self.extracted_rule is not None
            and self.extracted_rule.strip()
            and self.category is not None
            and self.confidence >= 0.5
        )


# Agent Library Types


@dataclass
class AgentCapability:
    """Represents a capability of an agent."""

    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
            "limitations": self.limitations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCapability":
        return cls(
            name=data["name"],
            description=data["description"],
            examples=data.get("examples", []),
            limitations=data.get("limitations", []),
        )


@dataclass
class AgentDefinition:
    """Represents an agent in the agent library."""

    name: str  # Agent identifier (e.g., "python-pro")
    display_name: str  # Human-readable name
    description: str  # Brief description
    capabilities: list[AgentCapability] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    personality_traits: list[str] = field(default_factory=list)

    # Agent-specific rules and preferences
    default_rules: list[str] = field(default_factory=list)
    interaction_patterns: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "specializations": self.specializations,
            "tools": self.tools,
            "frameworks": self.frameworks,
            "personality_traits": self.personality_traits,
            "default_rules": self.default_rules,
            "interaction_patterns": self.interaction_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentDefinition":
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            capabilities=[
                AgentCapability.from_dict(cap) for cap in data.get("capabilities", [])
            ],
            specializations=data.get("specializations", []),
            tools=data.get("tools", []),
            frameworks=data.get("frameworks", []),
            personality_traits=data.get("personality_traits", []),
            default_rules=data.get("default_rules", []),
            interaction_patterns=data.get("interaction_patterns", {}),
        )


# Claude Code SDK Integration Types


@dataclass
class ClaudeCodeSession:
    """Information about a Claude Code session."""

    session_id: str
    workspace_path: str
    user_name: str | None = None
    project_name: str | None = None
    active_files: list[str] = field(default_factory=list)
    context_window_size: int = 200000  # Default Claude context window

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "user_name": self.user_name,
            "project_name": self.project_name,
            "active_files": self.active_files,
            "context_window_size": self.context_window_size,
        }


@dataclass
class MemoryInjectionResult:
    """Result of injecting memory rules into Claude Code session."""

    success: bool
    rules_injected: int
    total_tokens_used: int
    remaining_context_tokens: int
    skipped_rules: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rules_injected": self.rules_injected,
            "total_tokens_used": self.total_tokens_used,
            "remaining_context_tokens": self.remaining_context_tokens,
            "skipped_rules": self.skipped_rules,
            "errors": self.errors,
        }
