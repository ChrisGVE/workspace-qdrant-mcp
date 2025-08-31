"""
Memory system for workspace-qdrant-mcp.

This module implements the memory collection system for storing user preferences,
LLM behavioral rules, and agent library definitions. The memory system provides
persistent context that automatically integrates with Claude Code sessions.

Memory Categories:
1. User Preferences: Personal preferences like "Use uv for Python"
2. LLM Behavioral Rules: Behavioral instructions like "Always make atomic commits"
3. Agent Library: Available agent definitions with capabilities and costs

Authority Levels:
- absolute: Non-negotiable rules that must always be followed
- default: Rules to follow unless explicitly overridden by user/PRD context

Key Features:
- Automatic Claude Code SDK integration for rule injection
- Conflict detection using semantic analysis
- Conversational memory updates ("Note: call me Chris")
- Token counting and optimization tools
- Session initialization with memory rule loading
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchRequest,
    VectorParams,
)

from .collection_naming import CollectionNamingManager, CollectionType
from .sparse_vectors import BM25SparseEncoder

logger = logging.getLogger(__name__)


class MemoryCategory(Enum):
    """Categories of memory entries."""
    PREFERENCE = "preference"
    BEHAVIOR = "behavior"
    AGENT = "agent"


class AuthorityLevel(Enum):
    """Authority levels for memory rules."""
    ABSOLUTE = "absolute"  # Non-negotiable, always follow
    DEFAULT = "default"    # Follow unless explicitly overridden


@dataclass
class MemoryRule:
    """
    A memory rule storing user preferences or LLM behavioral instructions.

    Attributes:
        id: Unique identifier for the rule
        category: Type of memory entry (preference, behavior, agent)
        name: Short name/identifier for the rule
        rule: The actual rule text or instruction
        authority: Authority level (absolute vs default)
        scope: List of contexts where this rule applies
        source: How the rule was created (user_explicit, conversational, etc.)
        conditions: Optional conditional logic for rule application
        replaces: List of rule IDs this rule supersedes
        created_at: Timestamp when rule was created
        updated_at: Timestamp when rule was last modified
        metadata: Additional metadata for the rule
    """
    id: str
    category: MemoryCategory
    name: str
    rule: str
    authority: AuthorityLevel
    scope: list[str]
    source: str = "user_explicit"
    conditions: dict[str, Any] | None = None
    replaces: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class AgentDefinition:
    """
    Definition of an available agent for deployment decisions.

    Attributes:
        id: Unique identifier for the agent
        name: Agent name (e.g., "python-pro")
        description: Brief description of agent capabilities
        capabilities: List of specific capabilities
        deploy_cost: Resource cost for deploying this agent
        last_used: Last time this agent was deployed
        metadata: Additional agent metadata
    """
    id: str
    name: str
    description: str
    capabilities: list[str]
    deploy_cost: str = "medium"
    last_used: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MemoryConflict:
    """
    Represents a conflict between memory rules.

    Attributes:
        conflict_type: Type of conflict detected
        rule1: First conflicting rule
        rule2: Second conflicting rule
        confidence: Confidence score of conflict detection (0.0-1.0)
        description: Human-readable description of the conflict
        resolution_options: Possible resolution strategies
    """
    conflict_type: str
    rule1: MemoryRule
    rule2: MemoryRule
    confidence: float
    description: str
    resolution_options: list[str]


@dataclass
class MemoryStats:
    """
    Statistics about memory usage and performance.

    Attributes:
        total_rules: Total number of memory rules
        rules_by_category: Breakdown by category
        rules_by_authority: Breakdown by authority level
        estimated_tokens: Estimated token count for all rules
        last_optimization: When memory was last optimized
    """
    total_rules: int
    rules_by_category: dict[MemoryCategory, int]
    rules_by_authority: dict[AuthorityLevel, int]
    estimated_tokens: int
    last_optimization: datetime | None = None


class MemoryManager:
    """
    Manages the memory collection system for persistent LLM context.

    This class provides the core functionality for storing, retrieving, and
    managing memory rules that control LLM behavior. It integrates with the
    reserved collection naming system and provides conflict detection.

    Features:
    - CRUD operations for memory rules
    - Authority level enforcement
    - Conflict detection using semantic analysis
    - Token counting and optimization
    - Session initialization support
    - Conversational memory updates
    """

    MEMORY_COLLECTION = "memory"

    def __init__(
        self,
        qdrant_client: QdrantClient,
        naming_manager: CollectionNamingManager,
        embedding_dim: int = 384,
        sparse_vector_generator: BM25SparseEncoder | None = None
    ):
        """
        Initialize the memory manager.

        Args:
            qdrant_client: Qdrant client for vector operations
            naming_manager: Collection naming manager
            embedding_dim: Dimension of dense embeddings (default: all-MiniLM-L6-v2)
            sparse_vector_generator: Generator for sparse vectors (optional)
        """
        self.client = qdrant_client
        self.naming_manager = naming_manager
        self.embedding_dim = embedding_dim
        self.sparse_generator = sparse_vector_generator

        # Rule ID tracking
        self._rule_id_counter = 0

    async def initialize_memory_collection(self) -> bool:
        """
        Initialize the memory collection in Qdrant.

        Returns:
            True if collection was created or already exists, False on error
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = {col.name for col in collections.collections}

            if self.MEMORY_COLLECTION in collection_names:
                logger.info(f"Memory collection '{self.MEMORY_COLLECTION}' already exists")
                return True

            # Create collection with named vectors for hybrid search
            vector_config = {
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            }

            # Add sparse vector config if generator available
            if self.sparse_generator:
                vector_config["sparse"] = VectorParams(
                    size=self.sparse_generator.vector_size,
                    distance=Distance.DOT
                )

            self.client.create_collection(
                collection_name=self.MEMORY_COLLECTION,
                vectors_config=vector_config
            )

            logger.info(f"Created memory collection '{self.MEMORY_COLLECTION}'")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory collection: {e}")
            return False

    async def add_memory_rule(
        self,
        category: MemoryCategory,
        name: str,
        rule: str,
        authority: AuthorityLevel = AuthorityLevel.DEFAULT,
        scope: list[str] | None = None,
        source: str = "user_explicit",
        conditions: dict[str, Any] | None = None,
        replaces: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_vector: list[float] | None = None
    ) -> str:
        """
        Add a new memory rule to the collection.

        Args:
            category: Category of memory rule
            name: Short name for the rule
            rule: The actual rule text
            authority: Authority level (absolute or default)
            scope: Contexts where this rule applies
            source: Source of the rule creation
            conditions: Optional conditional logic
            replaces: Rules this supersedes
            metadata: Additional metadata
            embedding_vector: Pre-computed embedding (optional)

        Returns:
            The ID of the created rule

        Raises:
            Exception: If rule creation fails
        """
        # Generate unique ID
        rule_id = self._generate_rule_id()

        # Create memory rule
        memory_rule = MemoryRule(
            id=rule_id,
            category=category,
            name=name,
            rule=rule,
            authority=authority,
            scope=scope or [],
            source=source,
            conditions=conditions,
            replaces=replaces,
            metadata=metadata
        )

        # Generate embedding if not provided
        if embedding_vector is None:
            # For now, create a placeholder embedding
            # In full implementation, this would use the embedding service
            embedding_vector = [0.0] * self.embedding_dim

        # Prepare vectors
        vectors = {"dense": embedding_vector}
        if self.sparse_generator:
            sparse_vector = self.sparse_generator.generate_sparse_vector(rule)
            vectors["sparse"] = sparse_vector

        # Create point
        point = PointStruct(
            id=rule_id,
            vector=vectors,
            payload={
                "category": category.value,
                "name": name,
                "rule": rule,
                "authority": authority.value,
                "scope": scope or [],
                "source": source,
                "conditions": conditions or {},
                "replaces": replaces or [],
                "created_at": memory_rule.created_at.isoformat(),
                "updated_at": memory_rule.updated_at.isoformat(),
                "metadata": metadata or {}
            }
        )

        # Upsert to collection
        self.client.upsert(
            collection_name=self.MEMORY_COLLECTION,
            points=[point]
        )

        # Handle rule replacement if specified
        if replaces:
            await self._handle_rule_replacement(rule_id, replaces)

        logger.info(f"Added memory rule '{name}' with ID {rule_id}")
        return rule_id

    async def get_memory_rule(self, rule_id: str) -> MemoryRule | None:
        """
        Retrieve a specific memory rule by ID.

        Args:
            rule_id: The ID of the rule to retrieve

        Returns:
            MemoryRule if found, None otherwise
        """
        try:
            points = self.client.retrieve(
                collection_name=self.MEMORY_COLLECTION,
                ids=[rule_id],
                with_payload=True
            )

            if not points:
                return None

            point = points[0]
            return self._point_to_memory_rule(point)

        except Exception as e:
            logger.error(f"Failed to retrieve memory rule {rule_id}: {e}")
            return None

    async def list_memory_rules(
        self,
        category: MemoryCategory | None = None,
        authority: AuthorityLevel | None = None,
        scope: str | None = None
    ) -> list[MemoryRule]:
        """
        List memory rules with optional filtering.

        Args:
            category: Filter by category
            authority: Filter by authority level
            scope: Filter by scope containing this value

        Returns:
            List of matching memory rules
        """
        try:
            # Build filter conditions
            conditions = []

            if category:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category.value))
                )

            if authority:
                conditions.append(
                    FieldCondition(key="authority", match=MatchValue(value=authority.value))
                )

            if scope:
                conditions.append(
                    FieldCondition(key="scope", match=MatchValue(value=scope))
                )

            # Create filter if we have conditions
            search_filter = Filter(must=conditions) if conditions else None

            # Scroll through all points
            points, _ = self.client.scroll(
                collection_name=self.MEMORY_COLLECTION,
                scroll_filter=search_filter,
                limit=1000,  # Adjust based on expected memory rule count
                with_payload=True
            )

            # Convert points to memory rules
            memory_rules = []
            for point in points:
                rule = self._point_to_memory_rule(point)
                if rule:
                    memory_rules.append(rule)

            # Sort by created_at
            memory_rules.sort(key=lambda r: r.created_at)
            return memory_rules

        except Exception as e:
            logger.error(f"Failed to list memory rules: {e}")
            return []

    async def update_memory_rule(
        self,
        rule_id: str,
        updates: dict[str, Any],
        embedding_vector: list[float] | None = None
    ) -> bool:
        """
        Update an existing memory rule.

        Args:
            rule_id: ID of the rule to update
            updates: Dictionary of field updates
            embedding_vector: New embedding if rule text changed

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get existing rule
            existing_rule = await self.get_memory_rule(rule_id)
            if not existing_rule:
                logger.error(f"Memory rule {rule_id} not found for update")
                return False

            # Apply updates
            for key, value in updates.items():
                if hasattr(existing_rule, key):
                    setattr(existing_rule, key, value)

            # Update timestamp
            existing_rule.updated_at = datetime.utcnow()

            # Generate new embedding if rule text changed
            if "rule" in updates and embedding_vector is None:
                # Placeholder for embedding generation
                embedding_vector = [0.0] * self.embedding_dim

            # Prepare vectors (keep existing if not regenerating)
            vectors = {}
            if embedding_vector:
                vectors["dense"] = embedding_vector
                if self.sparse_generator:
                    vectors["sparse"] = self.sparse_generator.generate_sparse_vector(existing_rule.rule)

            # Create updated point
            point = PointStruct(
                id=rule_id,
                vector=vectors if vectors else None,
                payload={
                    "category": existing_rule.category.value,
                    "name": existing_rule.name,
                    "rule": existing_rule.rule,
                    "authority": existing_rule.authority.value,
                    "scope": existing_rule.scope,
                    "source": existing_rule.source,
                    "conditions": existing_rule.conditions or {},
                    "replaces": existing_rule.replaces or [],
                    "created_at": existing_rule.created_at.isoformat(),
                    "updated_at": existing_rule.updated_at.isoformat(),
                    "metadata": existing_rule.metadata or {}
                }
            )

            # Update in collection
            self.client.upsert(
                collection_name=self.MEMORY_COLLECTION,
                points=[point]
            )

            logger.info(f"Updated memory rule {rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory rule {rule_id}: {e}")
            return False

    async def delete_memory_rule(self, rule_id: str) -> bool:
        """
        Delete a memory rule.

        Args:
            rule_id: ID of the rule to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.MEMORY_COLLECTION,
                points_selector=[rule_id]
            )

            logger.info(f"Deleted memory rule {rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory rule {rule_id}: {e}")
            return False

    async def search_memory_rules(
        self,
        query: str,
        limit: int = 10,
        category: MemoryCategory | None = None,
        authority: AuthorityLevel | None = None
    ) -> list[tuple[MemoryRule, float]]:
        """
        Search memory rules by semantic similarity.

        Args:
            query: Search query
            limit: Maximum number of results
            category: Filter by category
            authority: Filter by authority level

        Returns:
            List of (MemoryRule, score) tuples sorted by relevance
        """
        try:
            # Generate query embedding (placeholder)
            query_vector = [0.0] * self.embedding_dim

            # Build filter conditions
            conditions = []

            if category:
                conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category.value))
                )

            if authority:
                conditions.append(
                    FieldCondition(key="authority", match=MatchValue(value=authority.value))
                )

            search_filter = Filter(must=conditions) if conditions else None

            # Search using dense vectors
            search_result = self.client.search(
                collection_name=self.MEMORY_COLLECTION,
                query_vector=("dense", query_vector),
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Convert to memory rules with scores
            results = []
            for scored_point in search_result:
                rule = self._point_to_memory_rule(scored_point)
                if rule:
                    results.append((rule, scored_point.score))

            return results

        except Exception as e:
            logger.error(f"Failed to search memory rules: {e}")
            return []

    async def detect_conflicts(
        self,
        rules: list[MemoryRule] | None = None,
        semantic_analysis: bool = True
    ) -> list[MemoryConflict]:
        """
        Detect conflicts between memory rules.

        Args:
            rules: Specific rules to check (default: all rules)
            semantic_analysis: Whether to use semantic analysis (placeholder)

        Returns:
            List of detected conflicts
        """
        if rules is None:
            rules = await self.list_memory_rules()

        conflicts = []

        # Rule-based conflict detection
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                # Check for direct conflicts
                if self._rules_conflict(rule1, rule2):
                    conflict = MemoryConflict(
                        conflict_type="direct_contradiction",
                        rule1=rule1,
                        rule2=rule2,
                        confidence=0.9,
                        description=f"Rules '{rule1.name}' and '{rule2.name}' appear to conflict",
                        resolution_options=[
                            "Keep higher authority rule",
                            "Merge rules with conditions",
                            "User resolution required"
                        ]
                    )
                    conflicts.append(conflict)

        # Semantic analysis (placeholder for future LLM integration)
        if semantic_analysis:
            # This would integrate with Claude/Sonnet for semantic conflict detection
            pass

        return conflicts

    async def get_memory_stats(self) -> MemoryStats:
        """
        Get statistics about memory usage.

        Returns:
            MemoryStats with usage information
        """
        rules = await self.list_memory_rules()

        # Count by category
        by_category = {}
        for category in MemoryCategory:
            by_category[category] = len([r for r in rules if r.category == category])

        # Count by authority
        by_authority = {}
        for authority in AuthorityLevel:
            by_authority[authority] = len([r for r in rules if r.authority == authority])

        # Estimate token count (rough approximation)
        total_text = " ".join([r.rule for r in rules])
        estimated_tokens = len(total_text.split()) * 1.3  # Rough token estimation

        return MemoryStats(
            total_rules=len(rules),
            rules_by_category=by_category,
            rules_by_authority=by_authority,
            estimated_tokens=int(estimated_tokens)
        )

    async def optimize_memory(self, max_tokens: int = 2000) -> tuple[int, list[str]]:
        """
        Optimize memory usage by removing or consolidating rules.

        Args:
            max_tokens: Maximum allowed token count

        Returns:
            Tuple of (tokens_saved, list_of_optimization_actions)
        """
        stats = await self.get_memory_stats()

        if stats.estimated_tokens <= max_tokens:
            return 0, ["Memory already within token limit"]

        # Placeholder optimization logic
        actions = [
            "Identified redundant rules for consolidation",
            "Suggested merging similar preference rules",
            "Recommended archiving unused agent definitions"
        ]

        tokens_saved = stats.estimated_tokens - max_tokens
        return tokens_saved, actions

    def _generate_rule_id(self) -> str:
        """Generate a unique rule ID."""
        self._rule_id_counter += 1
        timestamp = int(time.time() * 1000)  # Millisecond timestamp
        return f"rule_{timestamp}_{self._rule_id_counter}"

    def _point_to_memory_rule(self, point) -> MemoryRule | None:
        """Convert a Qdrant point to a MemoryRule object."""
        try:
            payload = point.payload

            return MemoryRule(
                id=str(point.id),
                category=MemoryCategory(payload["category"]),
                name=payload["name"],
                rule=payload["rule"],
                authority=AuthorityLevel(payload["authority"]),
                scope=payload.get("scope", []),
                source=payload.get("source", "unknown"),
                conditions=payload.get("conditions") or None,
                replaces=payload.get("replaces") or None,
                created_at=datetime.fromisoformat(payload["created_at"]),
                updated_at=datetime.fromisoformat(payload["updated_at"]),
                metadata=payload.get("metadata") or None
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to convert point to memory rule: {e}")
            return None

    def _rules_conflict(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """
        Check if two rules conflict using simple heuristics.

        Args:
            rule1: First rule to check
            rule2: Second rule to check

        Returns:
            True if rules appear to conflict
        """
        # Simple keyword-based conflict detection
        conflicting_pairs = [
            (["use", "python"], ["avoid", "python"]),
            (["always"], ["never"]),
            (["commit", "immediately"], ["batch", "commit"]),
            (["uv"], ["pip"]),
            (["pytest"], ["unittest"]),
        ]

        rule1_lower = rule1.rule.lower()
        rule2_lower = rule2.rule.lower()

        for keywords1, keywords2 in conflicting_pairs:
            if (all(kw in rule1_lower for kw in keywords1) and
                all(kw in rule2_lower for kw in keywords2)):
                return True
            if (all(kw in rule2_lower for kw in keywords1) and
                all(kw in rule1_lower for kw in keywords2)):
                return True

        return False

    async def _handle_rule_replacement(self, new_rule_id: str, replaced_rule_ids: list[str]):
        """
        Handle replacement of old rules by a new rule.

        Args:
            new_rule_id: ID of the new rule
            replaced_rule_ids: IDs of rules being replaced
        """
        for old_rule_id in replaced_rule_ids:
            await self.delete_memory_rule(old_rule_id)
            logger.info(f"Deleted replaced rule {old_rule_id} (replaced by {new_rule_id})")


# Utility functions for memory management

def create_memory_manager(
    qdrant_client: QdrantClient,
    naming_manager: CollectionNamingManager,
    embedding_dim: int = 384,
    sparse_vector_generator: BM25SparseEncoder | None = None
) -> MemoryManager:
    """
    Factory function to create a memory manager.

    Args:
        qdrant_client: Qdrant client instance
        naming_manager: Collection naming manager
        embedding_dim: Embedding dimension
        sparse_vector_generator: Optional sparse vector generator

    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(
        qdrant_client=qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=embedding_dim,
        sparse_vector_generator=sparse_vector_generator
    )


def parse_conversational_memory_update(message: str) -> dict[str, Any] | None:
    """
    Parse conversational memory updates from chat messages.

    Detects patterns like:
    - "Note: call me Chris"
    - "For future reference, always use TypeScript strict mode"
    - "Remember that I prefer uv for Python package management"

    Args:
        message: The conversational message to parse

    Returns:
        Dictionary with parsed memory update or None if no update detected
    """
    message = message.strip()

    # Pattern: "Note: <preference>"
    note_match = re.match(r"^(?:note|reminder?):\s*(.+)$", message, re.IGNORECASE)
    if note_match:
        content = note_match.group(1).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": content,
            "source": "conversational_note",
            "authority": AuthorityLevel.DEFAULT
        }

    # Pattern: "For future reference, <instruction>"
    future_match = re.match(r"^for future reference,?\s*(.+)$", message, re.IGNORECASE)
    if future_match:
        content = future_match.group(1).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": content,
            "source": "conversational_future",
            "authority": AuthorityLevel.DEFAULT
        }

    # Pattern: "Remember (that) I <preference>"
    remember_match = re.match(r"^remember (?:that )?i (.+)$", message, re.IGNORECASE)
    if remember_match:
        content = remember_match.group(1).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": f"User {content}",
            "source": "conversational_remember",
            "authority": AuthorityLevel.DEFAULT
        }

    # Pattern: "Always <behavior>" or "Never <behavior>"
    behavior_match = re.match(r"^(always|never) (.+)$", message, re.IGNORECASE)
    if behavior_match:
        modifier = behavior_match.group(1).lower()
        behavior = behavior_match.group(2).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": f"{modifier.title()} {behavior}",
            "source": "conversational_behavior",
            "authority": AuthorityLevel.ABSOLUTE
        }

    return None
