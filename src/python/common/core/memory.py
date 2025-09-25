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
import re
import statistics
from collections import defaultdict, Counter

from loguru import logger
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
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

# logger imported from loguru


class MemoryCategory(Enum):
    """Categories of memory entries."""

    PREFERENCE = "preference"
    BEHAVIOR = "behavior"
    AGENT = "agent"


class AuthorityLevel(Enum):
    """Authority levels for memory rules."""

    ABSOLUTE = "absolute"  # Non-negotiable, always follow
    DEFAULT = "default"  # Follow unless explicitly overridden


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
            self.created_at = datetime.now(timezone.utc)
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


@dataclass
class ConversationalContext:
    """
    Context extracted from conversational memory updates.

    Attributes:
        intent: Detected intent (preference, behavior, identity, etc.)
        confidence: Confidence score (0.0-1.0)
        project_scope: Detected project or domain scope
        temporal_context: Time-based context (immediate, future, conditional)
        urgency_level: Extracted urgency or priority level
        conditions: Conditional logic detected in the message
        authority_signals: Language patterns indicating authority level
        extracted_entities: Named entities found (tools, libraries, people, etc.)
    """

    intent: str
    confidence: float
    project_scope: list[str] | None = None
    temporal_context: str | None = None
    urgency_level: str = "normal"  # low, normal, high, critical
    conditions: dict[str, Any] | None = None
    authority_signals: list[str] | None = None
    extracted_entities: dict[str, list[str]] | None = None


@dataclass
class BehavioralDecision:
    """
    Decision made by the behavioral control system.

    Attributes:
        decision_id: Unique identifier for this decision
        context: The context that triggered the decision
        applicable_rules: Rules that influenced the decision
        decision: The actual decision or recommendation
        confidence: Confidence score for the decision
        reasoning: Explanation of the decision logic
        conflicts_resolved: Any rule conflicts that were resolved
        fallback_used: Whether a fallback decision was used
    """

    decision_id: str
    context: str
    applicable_rules: list[str]
    decision: str
    confidence: float
    reasoning: str
    conflicts_resolved: list[str] | None = None
    fallback_used: bool = False


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
    - Configurable memory collection names with '__' prefix support
    """

    # Default memory collection - can be overridden via configuration
    DEFAULT_MEMORY_COLLECTION = "memory"

    def __init__(
        self,
        qdrant_client: QdrantClient,
        naming_manager: CollectionNamingManager,
        embedding_dim: int = 384,
        sparse_vector_generator: BM25SparseEncoder | None = None,
        memory_collection_name: str | None = None,
    ):
        """
        Initialize the memory manager.

        Args:
            qdrant_client: Qdrant client for vector operations
            naming_manager: Collection naming manager
            embedding_dim: Dimension of dense embeddings (default: all-MiniLM-L6-v2)
            sparse_vector_generator: Generator for sparse vectors (optional)
            memory_collection_name: Custom memory collection name (defaults to 'memory')
                                  Supports '__' prefix for system memory collections
        """
        self.client = qdrant_client
        self.naming_manager = naming_manager
        self.embedding_dim = embedding_dim
        self.sparse_generator = sparse_vector_generator

        # Set memory collection name - support configurable names including '__' prefix
        self.memory_collection_name = memory_collection_name or self.DEFAULT_MEMORY_COLLECTION

        # Validate the memory collection name using the naming manager
        if self.memory_collection_name != self.DEFAULT_MEMORY_COLLECTION:
            validation_result = naming_manager.validate_collection_name(self.memory_collection_name)
            if not validation_result.is_valid:
                logger.warning(f"Invalid memory collection name '{self.memory_collection_name}': {validation_result.error_message}")
                logger.warning(f"Falling back to default memory collection: {self.DEFAULT_MEMORY_COLLECTION}")
                self.memory_collection_name = self.DEFAULT_MEMORY_COLLECTION

        # Rule ID tracking
        self._rule_id_counter = 0

        logger.info(f"MemoryManager initialized with collection: {self.memory_collection_name}")

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

            if self.memory_collection_name in collection_names:
                logger.info(
                    f"Memory collection '{self.memory_collection_name}' already exists"
                )
                return True

            # Create collection with named vectors for hybrid search
            vector_config = {
                "dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            }

            # Add sparse vector config if generator available
            if self.sparse_generator:
                vector_config["sparse"] = VectorParams(
                    size=self.sparse_generator.vector_size, distance=Distance.DOT
                )

            self.client.create_collection(
                collection_name=self.memory_collection_name, vectors_config=vector_config
            )

            logger.info(f"Created memory collection '{self.memory_collection_name}'")
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
        embedding_vector: list[float] | None = None,
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
        # Check if memory collection exists
        collections = self.client.get_collections()
        collection_names = {col.name for col in collections.collections}

        if self.memory_collection_name not in collection_names:
            # Collection doesn't exist, need to initialize first
            raise RuntimeError(
                f"Memory collection '{self.memory_collection_name}' doesn't exist. "
                "Call initialize_memory_collection() first."
            )

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
            metadata=metadata,
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
                "metadata": metadata or {},
            },
        )

        # Upsert to collection
        self.client.upsert(collection_name=self.memory_collection_name, points=[point])

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
            # Check if memory collection exists
            collections = self.client.get_collections()
            collection_names = {col.name for col in collections.collections}

            if self.memory_collection_name not in collection_names:
                # Collection doesn't exist yet, rule cannot be found
                logger.debug(
                    f"Memory collection '{self.memory_collection_name}' doesn't exist yet"
                )
                return None

            points = self.client.retrieve(
                collection_name=self.memory_collection_name, ids=[rule_id], with_payload=True
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
        scope: str | None = None,
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
            # Check if memory collection exists
            collections = self.client.get_collections()
            collection_names = {col.name for col in collections.collections}

            if self.memory_collection_name not in collection_names:
                # Collection doesn't exist yet, return empty list
                logger.debug(
                    f"Memory collection '{self.memory_collection_name}' doesn't exist yet"
                )
                return []

            # Build filter conditions
            conditions = []

            if category:
                conditions.append(
                    FieldCondition(
                        key="category", match=MatchValue(value=category.value)
                    )
                )

            if authority:
                conditions.append(
                    FieldCondition(
                        key="authority", match=MatchValue(value=authority.value)
                    )
                )

            if scope:
                conditions.append(
                    FieldCondition(key="scope", match=MatchValue(value=scope))
                )

            # Create filter if we have conditions
            search_filter = Filter(must=conditions) if conditions else None

            # Scroll through all points
            points, _ = self.client.scroll(
                collection_name=self.memory_collection_name,
                scroll_filter=search_filter,
                limit=1000,  # Adjust based on expected memory rule count
                with_payload=True,
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
        embedding_vector: list[float] | None = None,
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
            existing_rule.updated_at = datetime.now(timezone.utc)

            # Generate new embedding if rule text changed
            if "rule" in updates and embedding_vector is None:
                # Placeholder for embedding generation
                embedding_vector = [0.0] * self.embedding_dim

            # Prepare vectors (keep existing if not regenerating)
            vectors = {}
            if embedding_vector:
                vectors["dense"] = embedding_vector
                if self.sparse_generator:
                    vectors["sparse"] = self.sparse_generator.generate_sparse_vector(
                        existing_rule.rule
                    )

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
                    "metadata": existing_rule.metadata or {},
                },
            )

            # Update in collection
            self.client.upsert(collection_name=self.memory_collection_name, points=[point])

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
            # Check if memory collection exists
            collections = self.client.get_collections()
            collection_names = {col.name for col in collections.collections}

            if self.memory_collection_name not in collection_names:
                # Collection doesn't exist, rule cannot be deleted
                logger.debug(
                    f"Memory collection '{self.memory_collection_name}' doesn't exist yet"
                )
                return False

            self.client.delete(
                collection_name=self.memory_collection_name, points_selector=[rule_id]
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
        authority: AuthorityLevel | None = None,
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
            # Check if memory collection exists
            collections = self.client.get_collections()
            collection_names = {col.name for col in collections.collections}

            if self.memory_collection_name not in collection_names:
                # Collection doesn't exist yet, return empty list
                logger.debug(
                    f"Memory collection '{self.memory_collection_name}' doesn't exist yet"
                )
                return []

            # Generate query embedding (placeholder)
            query_vector = [0.0] * self.embedding_dim

            # Build filter conditions
            conditions = []

            if category:
                conditions.append(
                    FieldCondition(
                        key="category", match=MatchValue(value=category.value)
                    )
                )

            if authority:
                conditions.append(
                    FieldCondition(
                        key="authority", match=MatchValue(value=authority.value)
                    )
                )

            search_filter = Filter(must=conditions) if conditions else None

            # Search using dense vectors
            search_result = self.client.search(
                collection_name=self.memory_collection_name,
                query_vector=("dense", query_vector),
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
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
        self, rules: list[MemoryRule] | None = None, semantic_analysis: bool = True
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
            for rule2 in rules[i + 1 :]:
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
                            "User resolution required",
                        ],
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
            by_authority[authority] = len(
                [r for r in rules if r.authority == authority]
            )

        # Estimate token count using improved estimation
        total_text = " ".join([r.rule for r in rules])
        estimated_tokens = estimate_token_count(total_text)

        return MemoryStats(
            total_rules=len(rules),
            rules_by_category=by_category,
            rules_by_authority=by_authority,
            estimated_tokens=int(estimated_tokens),
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
            "Recommended archiving unused agent definitions",
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
                metadata=payload.get("metadata") or None,
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
            if all(kw in rule1_lower for kw in keywords1) and all(
                kw in rule2_lower for kw in keywords2
            ):
                return True
            if all(kw in rule2_lower for kw in keywords1) and all(
                kw in rule1_lower for kw in keywords2
            ):
                return True

        return False

    async def _handle_rule_replacement(
        self, new_rule_id: str, replaced_rule_ids: list[str]
    ):
        """
        Handle replacement of old rules by a new rule.

        Args:
            new_rule_id: ID of the new rule
            replaced_rule_ids: IDs of rules being replaced
        """
        for old_rule_id in replaced_rule_ids:
            await self.delete_memory_rule(old_rule_id)
            logger.info(
                f"Deleted replaced rule {old_rule_id} (replaced by {new_rule_id})"
            )


class ConversationalMemoryProcessor:
    """
    Advanced conversational memory update processor with NLP capabilities.

    This class provides sophisticated natural language processing for extracting
    memory rules from conversational messages, including context analysis,
    intent classification, and confidence scoring.
    """

    # Known tools, libraries, and frameworks for entity recognition
    KNOWN_ENTITIES = {
        "tools": {
            "uv", "pip", "poetry", "conda", "npm", "yarn", "pnpm", "docker", "kubernetes",
            "git", "svn", "mercurial", "pytest", "unittest", "jest", "vitest", "cypress"
        },
        "languages": {
            "python", "javascript", "typescript", "rust", "go", "java", "c++", "c#",
            "ruby", "php", "swift", "kotlin", "scala", "clojure", "haskell"
        },
        "frameworks": {
            "react", "vue", "angular", "django", "flask", "fastapi", "express", "nextjs",
            "spring", "rails", "laravel", "asp.net", "gin", "echo", "fiber"
        }
    }

    # Authority level indicators in language
    HIGH_AUTHORITY_SIGNALS = [
        "always", "never", "must", "required", "mandatory", "critical", "essential",
        "from now on", "absolutely", "without exception", "under no circumstances"
    ]

    MEDIUM_AUTHORITY_SIGNALS = [
        "should", "recommend", "suggest", "prefer", "typically", "usually",
        "make sure", "ensure", "remember", "important"
    ]

    # Urgency level indicators
    URGENCY_PATTERNS = {
        "critical": ["urgent", "critical", "immediately", "asap", "emergency"],
        "high": ["important", "priority", "soon", "quickly", "right away"],
        "normal": ["when possible", "eventually", "at some point"],
        "low": ["if convenient", "low priority", "not urgent", "sometime"]
    }

    def __init__(self):
        """Initialize the conversational processor."""
        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {
            # Enhanced pattern matching with context capture
            "note": re.compile(r"^(?:note|reminder?|fyi):\s*(.+)$", re.IGNORECASE),
            "future_reference": re.compile(r"^for future reference,?\s*(.+)$", re.IGNORECASE),
            "from_now_on": re.compile(r"^from now on,?\s*(.+)$", re.IGNORECASE),
            "make_sure": re.compile(r"^(?:make sure|ensure) (?:to\s+)?(.+)$", re.IGNORECASE),
            "remember": re.compile(r"^remember (?:that\s+)?(.+)$", re.IGNORECASE),
            "prefer": re.compile(r"^i prefer (.+)$", re.IGNORECASE),
            "always_never": re.compile(r"^(always|never) (.+)$", re.IGNORECASE),
            "instead_of": re.compile(r"^use (.+) instead of (.+)$", re.IGNORECASE),
            "preference_over": re.compile(r"^(.+) over (.+)$", re.IGNORECASE),
            "when_doing": re.compile(r"^when (?:working (?:on|with)|doing) (.+), (.+)$", re.IGNORECASE),
            "for_project": re.compile(r"^(?:for|in) (?:the\s+)?(.+) project,?\s*(.+)$", re.IGNORECASE),
            "conditional": re.compile(r"^(?:if|when) (.+), (?:then\s+)?(.+)$", re.IGNORECASE),
            "identity": re.compile(r"^(?:my name is|call me|i am|i'm)\s+(.+)$", re.IGNORECASE),
            "please_request": re.compile(r"^please (.+)(?:\s+going forward|\s+from now on)?$", re.IGNORECASE),
            "should_behavior": re.compile(r"^(?:you should|we should|i should) (.+)$", re.IGNORECASE),
        }

    def process_conversational_update(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """
        Process a conversational message for memory updates with advanced NLP.

        Args:
            message: The conversational message to process
            context: Optional context about the conversation (project, task, etc.)

        Returns:
            Dictionary with extracted memory rule information or None if no pattern matched
        """
        if not message or not message.strip():
            return None

        message = message.strip()

        # Extract conversational context first
        conv_context = self._extract_context(message, context)

        if conv_context.confidence < 0.3:  # Too ambiguous
            return None

        # Try pattern matching with context awareness
        result = self._match_patterns(message, conv_context)

        if result:
            # Add extracted context to the result
            result["context"] = conv_context
            result["confidence"] = conv_context.confidence

        return result

    def _extract_context(self, message: str, external_context: dict[str, Any] | None = None) -> ConversationalContext:
        """
        Extract conversational context from the message.

        Args:
            message: The message to analyze
            external_context: External context from the conversation

        Returns:
            ConversationalContext with extracted information
        """
        message_lower = message.lower()

        # Detect intent
        intent = self._classify_intent(message)

        # Extract entities (tools, languages, etc.)
        entities = self._extract_entities(message_lower)

        # Detect authority signals
        authority_signals = self._detect_authority_signals(message_lower)

        # Detect urgency level
        urgency_level = self._detect_urgency(message_lower)

        # Extract project scope
        project_scope = self._extract_project_scope(message, external_context)

        # Detect temporal context
        temporal_context = self._detect_temporal_context(message_lower)

        # Extract conditions
        conditions = self._extract_conditions(message)

        # Calculate confidence based on signal strength
        confidence = self._calculate_confidence(
            intent, entities, authority_signals, project_scope
        )

        return ConversationalContext(
            intent=intent,
            confidence=confidence,
            project_scope=project_scope,
            temporal_context=temporal_context,
            urgency_level=urgency_level,
            conditions=conditions,
            authority_signals=authority_signals,
            extracted_entities=entities
        )

    def _classify_intent(self, message: str) -> str:
        """Classify the intent of the message."""
        message_lower = message.lower()

        # Identity patterns
        if any(pattern in message_lower for pattern in ["my name", "call me", "i am", "i'm"]):
            return "identity"

        # Preference patterns
        if any(pattern in message_lower for pattern in ["prefer", "like", "favor", "choose"]):
            return "preference"

        # Behavior patterns
        if any(pattern in message_lower for pattern in ["always", "never", "make sure", "ensure", "should"]):
            return "behavior"

        # Tool/technology choice
        if any(pattern in message_lower for pattern in ["use", "avoid", "instead", "over", "better than"]):
            return "tool_choice"

        # Process/workflow
        if any(pattern in message_lower for pattern in ["when", "process", "workflow", "procedure", "method"]):
            return "process"

        # Default
        return "general"

    def _extract_entities(self, message_lower: str) -> dict[str, list[str]]:
        """Extract known entities from the message."""
        entities = {}

        for entity_type, entity_set in self.KNOWN_ENTITIES.items():
            found_entities = [entity for entity in entity_set if entity in message_lower]
            if found_entities:
                entities[entity_type] = found_entities

        return entities

    def _detect_authority_signals(self, message_lower: str) -> list[str]:
        """Detect authority level indicators in the message."""
        signals = []

        for signal in self.HIGH_AUTHORITY_SIGNALS:
            if signal.lower() in message_lower:
                signals.append(f"high:{signal}")

        for signal in self.MEDIUM_AUTHORITY_SIGNALS:
            if signal.lower() in message_lower:
                signals.append(f"medium:{signal}")

        return signals

    def _detect_urgency(self, message_lower: str) -> str:
        """Detect urgency level from the message."""
        for urgency_level, indicators in self.URGENCY_PATTERNS.items():
            if any(indicator in message_lower for indicator in indicators):
                return urgency_level
        return "normal"

    def _extract_project_scope(self, message: str, external_context: dict[str, Any] | None = None) -> list[str] | None:
        """Extract project or domain scope from the message."""
        scope = []

        # Check external context first
        if external_context:
            if "project" in external_context:
                scope.append(external_context["project"])
            if "domain" in external_context:
                scope.append(external_context["domain"])

        # Extract from message patterns
        project_match = re.search(r"(?:for|in) (?:the\s+)?(\w+) project", message, re.IGNORECASE)
        if project_match:
            scope.append(project_match.group(1))

        domain_match = re.search(r"when (?:working (?:on|with)|doing) ([\w\s]+)", message, re.IGNORECASE)
        if domain_match:
            domain = domain_match.group(1).strip()
            if len(domain.split()) <= 3:  # Reasonable domain length
                scope.append(domain)

        return scope if scope else None

    def _detect_temporal_context(self, message_lower: str) -> str | None:
        """Detect temporal context from the message."""
        if any(word in message_lower for word in ["now", "immediately", "right away"]):
            return "immediate"
        elif any(word in message_lower for word in ["future", "going forward", "from now on"]):
            return "future"
        elif any(word in message_lower for word in ["when", "if", "whenever"]):
            return "conditional"
        return None

    def _extract_conditions(self, message: str) -> dict[str, Any] | None:
        """Extract conditional logic from the message."""
        conditions = {}

        # Conditional patterns
        conditional_match = re.search(r"^(?:if|when) (.+), (?:then\s+)?(.+)$", message, re.IGNORECASE)
        if conditional_match:
            conditions["condition"] = conditional_match.group(1).strip()
            conditions["action"] = conditional_match.group(2).strip()
            return conditions

        # Context-specific patterns
        context_match = re.search(r"when (?:working (?:on|with)|doing) (.+), (.+)", message, re.IGNORECASE)
        if context_match:
            conditions["context"] = context_match.group(1).strip()
            conditions["behavior"] = context_match.group(2).strip()
            return conditions

        return None

    def _calculate_confidence(
        self,
        intent: str,
        entities: dict[str, list[str]],
        authority_signals: list[str],
        project_scope: list[str] | None
    ) -> float:
        """Calculate confidence score for the extracted information."""
        confidence = 0.0

        # Base confidence for intent recognition
        if intent != "general":
            confidence += 0.3
        else:
            confidence += 0.1

        # Entity recognition boosts confidence
        if entities:
            confidence += 0.3 * min(len(entities), 2) / 2

        # Clear authority signals boost confidence
        if authority_signals:
            confidence += 0.2

        # Project scope adds context confidence
        if project_scope:
            confidence += 0.2

        return min(confidence, 1.0)

    def _match_patterns(self, message: str, context: ConversationalContext) -> dict[str, Any] | None:
        """
        Match message against patterns with context awareness.

        Args:
            message: The message to match
            context: Extracted conversational context

        Returns:
            Memory rule information dictionary or None
        """
        # Enhanced pattern matching using compiled patterns

        # Note pattern
        if match := self.patterns["note"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.PREFERENCE,
                rule=match.group(1).strip(),
                source="conversational_note",
                authority=self._determine_authority(context),
                context=context
            )

        # Future reference pattern
        if match := self.patterns["future_reference"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=match.group(1).strip(),
                source="conversational_future",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        # From now on pattern (high authority)
        if match := self.patterns["from_now_on"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=match.group(1).strip(),
                source="conversational_directive",
                authority=AuthorityLevel.ABSOLUTE,
                context=context
            )

        # Make sure pattern (high authority)
        if match := self.patterns["make_sure"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=f"Always {match.group(1).strip()}",
                source="conversational_instruction",
                authority=AuthorityLevel.ABSOLUTE,
                context=context
            )

        # Remember pattern
        if match := self.patterns["remember"].match(message):
            content = match.group(1).strip()
            if content.lower().startswith("i "):
                content = f"User {content[2:]}"
            return self._create_rule_dict(
                category=MemoryCategory.PREFERENCE,
                rule=content,
                source="conversational_remember",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        # Preference pattern
        if match := self.patterns["prefer"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.PREFERENCE,
                rule=f"User prefers {match.group(1).strip()}",
                source="conversational_preference",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        # Always/Never pattern (high authority)
        if match := self.patterns["always_never"].match(message):
            modifier = match.group(1).lower()
            behavior = match.group(2).strip()
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=f"{modifier.title()} {behavior}",
                source="conversational_behavior",
                authority=AuthorityLevel.ABSOLUTE,
                context=context
            )

        # Use instead of pattern
        if match := self.patterns["instead_of"].match(message):
            preferred = match.group(1).strip()
            avoided = match.group(2).strip()
            return self._create_rule_dict(
                category=MemoryCategory.PREFERENCE,
                rule=f"Use {preferred} instead of {avoided}",
                source="conversational_substitution",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        # Preference over pattern
        if match := self.patterns["preference_over"].match(message):
            preferred = match.group(1).strip()
            avoided = match.group(2).strip()
            # Only match if both are reasonable tool/library names
            if len(preferred.split()) <= 2 and len(avoided.split()) <= 2:
                return self._create_rule_dict(
                    category=MemoryCategory.PREFERENCE,
                    rule=f"Prefer {preferred} over {avoided}",
                    source="conversational_preference_comparison",
                    authority=AuthorityLevel.DEFAULT,
                    context=context
                )

        # Identity pattern
        if match := self.patterns["identity"].match(message):
            name = match.group(1).strip()
            return self._create_rule_dict(
                category=MemoryCategory.PREFERENCE,
                rule=f"User's name is {name}",
                source="conversational_identity",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        # Conditional pattern
        if match := self.patterns["conditional"].match(message):
            condition = match.group(1).strip()
            action = match.group(2).strip()
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=f"When {condition}, {action}",
                source="conversational_conditional",
                authority=self._determine_authority(context),
                context=context,
                conditions={"condition": condition, "action": action}
            )

        # When doing pattern (context-specific)
        if match := self.patterns["when_doing"].match(message):
            context_desc = match.group(1).strip()
            behavior = match.group(2).strip()
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=f"When working on {context_desc}, {behavior}",
                source="conversational_context_specific",
                authority=AuthorityLevel.DEFAULT,
                context=context,
                conditions={"context": context_desc, "behavior": behavior}
            )

        # Project-specific pattern
        if match := self.patterns["for_project"].match(message):
            project = match.group(1).strip()
            rule_content = match.group(2).strip()
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=rule_content,
                source="conversational_project_specific",
                authority=AuthorityLevel.DEFAULT,
                context=context,
                scope=[project]
            )

        # Please request pattern
        if match := self.patterns["please_request"].match(message):
            return self._create_rule_dict(
                category=MemoryCategory.BEHAVIOR,
                rule=match.group(1).strip(),
                source="conversational_request",
                authority=AuthorityLevel.DEFAULT,
                context=context
            )

        return None

    def _create_rule_dict(
        self,
        category: MemoryCategory,
        rule: str,
        source: str,
        authority: AuthorityLevel,
        context: ConversationalContext,
        conditions: dict[str, Any] | None = None,
        scope: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a standardized rule dictionary."""
        return {
            "category": category,
            "rule": rule,
            "source": source,
            "authority": authority,
            "scope": scope or context.project_scope or [],
            "conditions": conditions or context.conditions,
            "urgency_level": context.urgency_level,
            "temporal_context": context.temporal_context,
            "extracted_entities": context.extracted_entities,
        }

    def _determine_authority(self, context: ConversationalContext) -> AuthorityLevel:
        """Determine authority level based on conversational context."""
        if not context.authority_signals:
            return AuthorityLevel.DEFAULT

        high_signals = [s for s in context.authority_signals if s.startswith("high:")]
        if high_signals:
            return AuthorityLevel.ABSOLUTE

        return AuthorityLevel.DEFAULT


# Utility functions for memory management


def create_memory_manager(
    qdrant_client: QdrantClient,
    naming_manager: CollectionNamingManager,
    embedding_dim: int = 384,
    sparse_vector_generator: BM25SparseEncoder | None = None,
    memory_collection_name: str | None = None,
) -> MemoryManager:
    """
    Create a memory manager instance.

    Args:
        qdrant_client: Qdrant client instance
        naming_manager: Collection naming manager
        embedding_dim: Embedding dimension
        sparse_vector_generator: Optional sparse vector generator
        memory_collection_name: Custom memory collection name (supports '__' prefix)

    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(
        qdrant_client=qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=embedding_dim,
        sparse_vector_generator=sparse_vector_generator,
        memory_collection_name=memory_collection_name,
    )


def parse_conversational_memory_update(message: str) -> dict[str, Any] | None:
    """
    Parse conversational memory updates from chat messages.

    Detects patterns like:
    - "Note: call me Chris"
    - "For future reference, always use TypeScript strict mode"
    - "Remember that I prefer uv for Python package management"
    - "I prefer using pytest over unittest"
    - "Make sure to always commit after each change"
    - "Please use atomic commits going forward"
    - "From now on, use uv instead of pip"
    - "My name is Chris"
    - "Call me Chris"

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
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "For future reference, <instruction>"
    future_match = re.match(r"^for future reference,?\s*(.+)$", message, re.IGNORECASE)
    if future_match:
        content = future_match.group(1).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": content,
            "source": "conversational_future",
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "From now on, <instruction>"
    from_now_match = re.match(r"^from now on,?\s*(.+)$", message, re.IGNORECASE)
    if from_now_match:
        content = from_now_match.group(1).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": content,
            "source": "conversational_directive",
            "authority": AuthorityLevel.ABSOLUTE,
        }

    # Pattern: "Please <behavior> going forward"
    please_match = re.match(r"^please (.+) (?:going forward|from now on)$", message, re.IGNORECASE)
    if please_match:
        content = please_match.group(1).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": content,
            "source": "conversational_request",
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "Make sure to <behavior>"
    make_sure_match = re.match(r"^make sure to (.+)$", message, re.IGNORECASE)
    if make_sure_match:
        content = make_sure_match.group(1).strip()
        return {
            "category": MemoryCategory.BEHAVIOR,
            "rule": f"Always {content}",
            "source": "conversational_instruction",
            "authority": AuthorityLevel.ABSOLUTE,
        }

    # Pattern: "Remember (that) I <preference>"
    remember_match = re.match(r"^remember (?:that )?i (.+)$", message, re.IGNORECASE)
    if remember_match:
        content = remember_match.group(1).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": f"User {content}",
            "source": "conversational_remember",
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "I prefer <preference>"
    prefer_match = re.match(r"^i prefer (.+)$", message, re.IGNORECASE)
    if prefer_match:
        content = prefer_match.group(1).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": f"User prefers {content}",
            "source": "conversational_preference",
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "My name is <name>" or "Call me <name>"
    name_match = re.match(r"^(?:my name is|call me)\s+(.+)$", message, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": f"User's name is {name}",
            "source": "conversational_identity",
            "authority": AuthorityLevel.DEFAULT,
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
            "authority": AuthorityLevel.ABSOLUTE,
        }

    # Pattern: "Use <tool> instead of <other_tool>"
    instead_match = re.match(r"^use (.+) instead of (.+)$", message, re.IGNORECASE)
    if instead_match:
        preferred = instead_match.group(1).strip()
        avoided = instead_match.group(2).strip()
        return {
            "category": MemoryCategory.PREFERENCE,
            "rule": f"Use {preferred} instead of {avoided}",
            "source": "conversational_substitution",
            "authority": AuthorityLevel.DEFAULT,
        }

    # Pattern: "<tool> over <other_tool>" (e.g., "pytest over unittest")
    over_match = re.match(r"^(.+) over (.+)$", message, re.IGNORECASE)
    if over_match:
        preferred = over_match.group(1).strip()
        avoided = over_match.group(2).strip()
        # Only match if both are recognizable tool/library names
        if len(preferred.split()) <= 2 and len(avoided.split()) <= 2:
            return {
                "category": MemoryCategory.PREFERENCE,
                "rule": f"Prefer {preferred} over {avoided}",
                "source": "conversational_preference_comparison",
                "authority": AuthorityLevel.DEFAULT,
            }

    return None


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text using simple heuristics.

    This provides a rough approximation since we don't have access to
    the exact tokenizer. More accurate counting would require integration
    with the specific LLM's tokenizer.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple approximation:
    # - Split by whitespace and punctuation
    # - Apply a multiplier for subword tokenization
    words = len(text.split())

    # Account for punctuation and subword tokens
    # Most modern tokenizers split words into subwords
    punctuation_chars = len([c for c in text if c in '.,!?;:()[]{}"\'-'])

    # Rough approximation: 1.3x words + 0.5x punctuation
    estimated_tokens = int(words * 1.3 + punctuation_chars * 0.5)

    return max(1, estimated_tokens)  # Minimum 1 token for non-empty text


def format_memory_rules_for_injection(rules: list[MemoryRule]) -> str:
    """
    Format memory rules for injection into Claude Code sessions.

    Creates a formatted string of memory rules that can be injected
    into system prompts or context.

    Args:
        rules: List of memory rules to format

    Returns:
        Formatted string ready for injection
    """
    if not rules:
        return "No memory rules to apply."

    # Separate by authority level
    absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
    default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

    sections = []

    if absolute_rules:
        sections.append("## CRITICAL RULES (Always Follow)")
        for rule in absolute_rules:
            sections.append(f"- **{rule.name}**: {rule.rule}")
            if rule.scope:
                sections.append(f"  - Scope: {', '.join(rule.scope)}")
        sections.append("")

    if default_rules:
        sections.append("## DEFAULT GUIDELINES (Unless Overridden)")

        # Group by category
        by_category = {}
        for rule in default_rules:
            category = rule.category.value.title()
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(rule)

        for category, category_rules in by_category.items():
            sections.append(f"### {category}")
            for rule in category_rules:
                sections.append(f"- **{rule.name}**: {rule.rule}")
                if rule.scope:
                    sections.append(f"  - Scope: {', '.join(rule.scope)}")
            sections.append("")

    return "\n".join(sections)


def create_memory_session_context(
    memory_manager: MemoryManager,
    task_context: str | None = None
) -> dict[str, Any]:
    """
    Create session context with memory rules for Claude Code integration.

    Args:
        memory_manager: The memory manager instance
        task_context: Optional task context for relevance filtering

    Returns:
        Dictionary with session context and memory integration
    """
    async def _create_context():
        try:
            # Get all memory rules
            all_rules = await memory_manager.list_memory_rules()

            # Get memory statistics
            stats = await memory_manager.get_memory_stats()

            # Search for task-relevant rules if context provided
            relevant_rules = []
            if task_context:
                search_results = await memory_manager.search_memory_rules(
                    query=task_context, limit=10
                )
                relevant_rules = [rule for rule, score in search_results if score > 0.6]

            # Detect conflicts
            conflicts = await memory_manager.detect_conflicts(all_rules)

            # Format rules for injection
            injection_text = format_memory_rules_for_injection(all_rules)

            return {
                "status": "ready",
                "total_rules": len(all_rules),
                "estimated_tokens": stats.estimated_tokens,
                "conflicts_detected": len(conflicts),
                "relevant_rules": len(relevant_rules),
                "memory_injection": injection_text,
                "rules_by_authority": {
                    k.value: v for k, v in stats.rules_by_authority.items()
                },
                "rules_by_category": {
                    k.value: v for k, v in stats.rules_by_category.items()
                },
                "conflicts": [
                    {
                        "type": c.conflict_type,
                        "description": c.description,
                        "confidence": c.confidence
                    } for c in conflicts
                ] if conflicts else []
            }

        except Exception as e:
            logger.error(f"Failed to create memory session context: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_rules": 0,
                "memory_injection": "Memory system unavailable."
            }

    # For synchronous usage, this would need to be called with asyncio.run()
    return _create_context
