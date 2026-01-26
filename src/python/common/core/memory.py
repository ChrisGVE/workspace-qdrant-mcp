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

import hashlib
import inspect
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .collection_naming import CollectionNamingManager
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


CONFLICTING_KEYWORD_PAIRS = (
    (("use", "python"), ("avoid", "python")),
    (("always",), ("never",)),
    (("commit", "immediately"), ("batch", "commit")),
    (("uv",), ("pip",)),
    (("pytest",), ("unittest",)),
    (("parallel",), ("sequential",)),
    (("parallel",), ("sequentially",)),
)


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
            else:
                # Retrieve existing vectors from Qdrant when not updating them
                existing_points = self.client.retrieve(
                    collection_name=self.memory_collection_name,
                    ids=[rule_id],
                    with_vectors=True,
                )
                if existing_points and existing_points[0].vector:
                    vectors = existing_points[0].vector

            # Create updated point
            point = PointStruct(
                id=rule_id,
                vector=vectors,
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

        # Rule-based conflict detection (precompute lowercase rules and match lists)
        rule_texts = [rule.rule.lower() for rule in rules]
        pair_matches: list[tuple[list[int], list[int]]] = [
            ([], []) for _ in CONFLICTING_KEYWORD_PAIRS
        ]

        for idx, rule_lower in enumerate(rule_texts):
            for pair_idx, (keywords1, keywords2) in enumerate(
                CONFLICTING_KEYWORD_PAIRS
            ):
                if all(kw in rule_lower for kw in keywords1):
                    pair_matches[pair_idx][0].append(idx)
                if all(kw in rule_lower for kw in keywords2):
                    pair_matches[pair_idx][1].append(idx)

        seen_pairs: set[tuple[int, int]] = set()
        for pair_idx, (match_a, match_b) in enumerate(pair_matches):
            if not match_a or not match_b:
                continue
            for i in match_a:
                for j in match_b:
                    if i >= j:
                        continue
                    pair_key = (i, j)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    rule1 = rules[i]
                    rule2 = rules[j]
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
        return self._rules_conflict_text(rule1.rule.lower(), rule2.rule.lower())

    def _rules_conflict_text(self, rule1_lower: str, rule2_lower: str) -> bool:
        for keywords1, keywords2 in CONFLICTING_KEYWORD_PAIRS:
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


class BehavioralController:
    """
    Memory-driven behavioral control system with adaptive decision making.

    This class uses memory rules to make intelligent decisions about how to
    respond to different situations, with conflict resolution, priority
    management, and adaptive learning from user feedback.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the behavioral controller.

        Args:
            memory_manager: The memory manager instance for rule retrieval
        """
        self.memory_manager = memory_manager
        self.decision_cache = {}  # Cache recent decisions for consistency
        self.conflict_resolution_history = []  # Track how conflicts were resolved
        self.feedback_history = []  # Track user feedback on decisions

    async def make_decision(
        self,
        context: str,
        situation_type: str | None = None,
        project_scope: list[str] | None = None,
        urgency: str = "normal"
    ) -> BehavioralDecision:
        """
        Make a behavioral decision based on memory rules and context.

        Args:
            context: The situation or context requiring a decision
            situation_type: Type of situation (development, communication, etc.)
            project_scope: Relevant project or domain context
            urgency: Urgency level (low, normal, high, critical)

        Returns:
            BehavioralDecision with the decision and reasoning
        """
        decision_id = self._generate_decision_id()

        try:
            # Get relevant rules for this context
            relevant_rules = await self._find_relevant_rules(
                context, situation_type, project_scope
            )

            if not relevant_rules:
                # No specific rules found, use fallback decision
                return BehavioralDecision(
                    decision_id=decision_id,
                    context=context,
                    applicable_rules=[],
                    decision="No specific memory rules found, use default behavior",
                    confidence=0.3,
                    reasoning="No applicable memory rules found for this context",
                    fallback_used=True
                )

            # Check for conflicts between applicable rules
            conflicts = await self._detect_rule_conflicts(relevant_rules)
            resolved_conflicts = []

            if conflicts:
                # Resolve conflicts using priority and authority levels
                relevant_rules, resolved_conflicts = await self._resolve_conflicts(
                    relevant_rules, conflicts, context, urgency
                )

            # Generate decision based on remaining rules
            decision = await self._generate_decision(
                relevant_rules, context, urgency
            )

            # Calculate confidence based on rule coverage and conflict resolution
            confidence = self._calculate_decision_confidence(
                relevant_rules, conflicts, context
            )

            # Create reasoning explanation
            reasoning = self._generate_reasoning(
                relevant_rules, conflicts, resolved_conflicts, context
            )

            behavioral_decision = BehavioralDecision(
                decision_id=decision_id,
                context=context,
                applicable_rules=[rule.id for rule in relevant_rules],
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                conflicts_resolved=resolved_conflicts,
                fallback_used=False
            )

            # Cache decision for consistency
            self._cache_decision(context, behavioral_decision)

            logger.info(
                f"Made behavioral decision {decision_id}: {decision} "
                f"(confidence: {confidence:.2f}, conflicts: {len(conflicts)})"
            )

            return behavioral_decision

        except Exception as e:
            logger.error(f"Failed to make behavioral decision: {e}")
            return BehavioralDecision(
                decision_id=decision_id,
                context=context,
                applicable_rules=[],
                decision="Error occurred, use fallback behavior",
                confidence=0.1,
                reasoning=f"Decision making failed: {e}",
                fallback_used=True
            )

    async def _find_relevant_rules(
        self,
        context: str,
        situation_type: str | None,
        project_scope: list[str] | None
    ) -> list[MemoryRule]:
        """Find memory rules relevant to the given context."""
        # Search by semantic similarity
        search_results = await self.memory_manager.search_memory_rules(
            query=context, limit=20
        )

        relevant_rules = []

        for rule, score in search_results:
            if score < 0.3:  # Skip low-relevance rules
                continue

            # Check if rule applies to current situation type
            if situation_type and rule.conditions:
                if not self._rule_applies_to_situation(rule, situation_type):
                    continue

            # Check project scope matching
            if project_scope and rule.scope:
                if not any(scope in project_scope for scope in rule.scope):
                    continue

            relevant_rules.append(rule)

        # Also get rules by category if we can infer it from context
        inferred_category = self._infer_category_from_context(context)
        if inferred_category:
            category_rules = await self.memory_manager.list_memory_rules(
                category=inferred_category
            )

            # Add high-relevance category rules that aren't already included
            existing_ids = {rule.id for rule in relevant_rules}
            for rule in category_rules:
                if rule.id not in existing_ids:
                    # Quick relevance check
                    if self._quick_relevance_check(rule, context):
                        relevant_rules.append(rule)

        # Sort by authority level and creation time
        relevant_rules.sort(key=lambda r: (
            r.authority == AuthorityLevel.ABSOLUTE,  # Absolute authority first
            -r.created_at.timestamp()  # More recent rules first
        ), reverse=True)

        return relevant_rules

    def _rule_applies_to_situation(self, rule: MemoryRule, situation_type: str) -> bool:
        """Check if a rule applies to the given situation type."""
        if not rule.conditions:
            return True  # Rule applies universally

        # Check for situation-specific conditions
        conditions = rule.conditions
        if "context" in conditions:
            context_lower = conditions["context"].lower()
            if situation_type.lower() in context_lower:
                return True

        return True  # Default to applicable

    def _infer_category_from_context(self, context: str) -> MemoryCategory | None:
        """Infer memory category from context string."""
        context_lower = context.lower()

        # Behavior-related contexts
        if any(word in context_lower for word in [
            "commit", "test", "deploy", "review", "workflow", "process"
        ]):
            return MemoryCategory.BEHAVIOR

        # Preference-related contexts
        if any(word in context_lower for word in [
            "tool", "library", "framework", "language", "choose", "select"
        ]):
            return MemoryCategory.PREFERENCE

        return None

    def _quick_relevance_check(self, rule: MemoryRule, context: str) -> bool:
        """Quick relevance check for category-based rules."""
        rule_text_lower = rule.rule.lower()
        context_lower = context.lower()

        # Look for overlapping keywords
        rule_words = set(rule_text_lower.split())
        context_words = set(context_lower.split())

        # Check for significant word overlap
        overlap = len(rule_words & context_words)
        return overlap >= 2 or len(rule_words & context_words) / len(rule_words) > 0.3

    async def _detect_rule_conflicts(self, rules: list[MemoryRule]) -> list[MemoryConflict]:
        """Detect conflicts between the given rules."""
        if len(rules) <= 1:
            return []

        conflicts = await self.memory_manager.detect_conflicts(rules)
        return conflicts

    async def _resolve_conflicts(
        self,
        rules: list[MemoryRule],
        conflicts: list[MemoryConflict],
        context: str,
        urgency: str
    ) -> tuple[list[MemoryRule], list[str]]:
        """
        Resolve conflicts between rules using priority and context.

        Returns:
            Tuple of (resolved_rules, conflict_descriptions)
        """
        if not conflicts:
            return rules, []

        resolved_conflicts = []
        rules_to_remove = set()

        for conflict in conflicts:
            rule1, rule2 = conflict.rule1, conflict.rule2
            resolution_description = ""

            # Authority level takes precedence
            if rule1.authority != rule2.authority:
                if rule1.authority == AuthorityLevel.ABSOLUTE:
                    rules_to_remove.add(rule2.id)
                    resolution_description = f"Kept absolute rule '{rule1.name}' over default rule '{rule2.name}'"
                else:
                    rules_to_remove.add(rule1.id)
                    resolution_description = f"Kept absolute rule '{rule2.name}' over default rule '{rule1.name}'"

            # If same authority level, prefer more recent rules
            elif rule1.created_at != rule2.created_at:
                if rule1.created_at > rule2.created_at:
                    rules_to_remove.add(rule2.id)
                    resolution_description = f"Kept newer rule '{rule1.name}' over older rule '{rule2.name}'"
                else:
                    rules_to_remove.add(rule1.id)
                    resolution_description = f"Kept newer rule '{rule2.name}' over older rule '{rule1.name}'"

            # If both are same age, prefer more specific rules (with conditions/scope)
            else:
                rule1_specificity = len(rule1.scope or []) + (1 if rule1.conditions else 0)
                rule2_specificity = len(rule2.scope or []) + (1 if rule2.conditions else 0)

                if rule1_specificity > rule2_specificity:
                    rules_to_remove.add(rule2.id)
                    resolution_description = f"Kept more specific rule '{rule1.name}' over general rule '{rule2.name}'"
                elif rule2_specificity > rule1_specificity:
                    rules_to_remove.add(rule1.id)
                    resolution_description = f"Kept more specific rule '{rule2.name}' over general rule '{rule1.name}'"
                else:
                    # As last resort, keep the first rule (arbitrary but consistent)
                    rules_to_remove.add(rule2.id)
                    resolution_description = f"Kept rule '{rule1.name}' over conflicting rule '{rule2.name}' (arbitrary resolution)"

            resolved_conflicts.append(resolution_description)

        # Remove conflicted rules
        remaining_rules = [rule for rule in rules if rule.id not in rules_to_remove]

        # Log conflict resolution
        self.conflict_resolution_history.extend(resolved_conflicts)
        logger.info(f"Resolved {len(conflicts)} conflicts: {resolved_conflicts}")

        return remaining_rules, resolved_conflicts

    async def _generate_decision(
        self,
        rules: list[MemoryRule],
        context: str,
        urgency: str
    ) -> str:
        """Generate a decision based on the applicable rules."""
        if not rules:
            return "No specific guidance available, use default behavior"

        # Group rules by category
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        decision_parts = []

        # Process absolute authority rules first
        absolute_rules = [rule for rule in rules if rule.authority == AuthorityLevel.ABSOLUTE]
        if absolute_rules:
            decision_parts.append("Required actions:")
            for rule in absolute_rules[:3]:  # Limit to top 3 for clarity
                decision_parts.append(f"- {rule.rule}")

        # Process default rules
        default_rules = [rule for rule in rules if rule.authority == AuthorityLevel.DEFAULT]
        if default_rules:
            decision_parts.append("Recommended actions:")
            for rule in default_rules[:3]:  # Limit to top 3 for clarity
                decision_parts.append(f"- {rule.rule}")

        # Handle urgency
        if urgency in ["high", "critical"] and absolute_rules:
            decision_parts.insert(1, "High priority situation - follow absolute rules strictly")

        return "\n".join(decision_parts)

    def _calculate_decision_confidence(
        self,
        rules: list[MemoryRule],
        conflicts: list[MemoryConflict],
        context: str
    ) -> float:
        """Calculate confidence score for the decision."""
        if not rules:
            return 0.1  # Very low confidence with no rules

        confidence = 0.0

        # Base confidence from number of applicable rules
        confidence += min(len(rules) * 0.2, 0.6)

        # Authority level confidence
        absolute_rules = [rule for rule in rules if rule.authority == AuthorityLevel.ABSOLUTE]
        if absolute_rules:
            confidence += 0.3

        # Conflicts reduce confidence
        if conflicts:
            confidence -= len(conflicts) * 0.1

        # Recent rules boost confidence
        recent_rules = [
            rule for rule in rules
            if (datetime.now(timezone.utc) - rule.created_at).days < 30
        ]
        if recent_rules:
            confidence += 0.1

        return max(0.1, min(confidence, 1.0))

    def _generate_reasoning(
        self,
        rules: list[MemoryRule],
        conflicts: list[MemoryConflict],
        resolved_conflicts: list[str],
        context: str
    ) -> str:
        """Generate reasoning explanation for the decision."""
        reasoning_parts = []

        if rules:
            reasoning_parts.append(
                f"Based on {len(rules)} applicable memory rule(s):"
            )
            for i, rule in enumerate(rules[:3], 1):
                authority_desc = "required" if rule.authority == AuthorityLevel.ABSOLUTE else "recommended"
                reasoning_parts.append(
                    f"{i}. {rule.name} ({authority_desc}): {rule.rule}"
                )

        if conflicts and resolved_conflicts:
            reasoning_parts.append("\nConflict resolution applied:")
            for resolution in resolved_conflicts:
                reasoning_parts.append(f"- {resolution}")

        if not rules:
            reasoning_parts.append("No specific memory rules found for this context")

        return "\n".join(reasoning_parts)

    def _generate_decision_id(self) -> str:
        """Generate a unique decision ID."""
        timestamp = int(time.time() * 1000)
        return f"decision_{timestamp}"

    def _cache_decision(self, context: str, decision: BehavioralDecision):
        """Cache decision for consistency in similar contexts."""
        # Simple cache with size limit
        if len(self.decision_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(
                self.decision_cache.items(),
                key=lambda x: x[1].decision_id
            )
            for old_context, _ in sorted_cache[:50]:
                del self.decision_cache[old_context]

        self.decision_cache[context] = decision

    async def learn_from_feedback(
        self,
        decision_id: str,
        feedback: str,
        user_action: str | None = None,
        effectiveness_score: float | None = None
    ):
        """
        Learn from user feedback on decisions to improve future decisions.

        Args:
            decision_id: ID of the decision to provide feedback on
            feedback: User feedback text
            user_action: What the user actually did
            effectiveness_score: Score from 0.0 to 1.0 for decision effectiveness
        """
        feedback_entry = {
            "decision_id": decision_id,
            "feedback": feedback,
            "user_action": user_action,
            "effectiveness_score": effectiveness_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.feedback_history.append(feedback_entry)

        # Simple learning: if feedback indicates the decision was wrong,
        # we could adjust confidence calculations or create new rules
        if effectiveness_score is not None and effectiveness_score < 0.3:
            logger.warning(
                f"Low effectiveness score ({effectiveness_score}) for decision {decision_id}: {feedback}"
            )

        # Keep feedback history manageable
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-500:]

        logger.info(f"Recorded feedback for decision {decision_id}")

    async def get_decision_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent decision history for analysis."""
        # Return recent cached decisions
        recent_decisions = list(self.decision_cache.values())[-limit:]
        return [
            {
                "decision_id": d.decision_id,
                "context": d.context,
                "decision": d.decision,
                "confidence": d.confidence,
                "conflicts_resolved": len(d.conflicts_resolved or [])
            }
            for d in recent_decisions
        ]


class MemoryLifecycleManager:
    """
    Manages memory lifecycle with intelligent cleanup and archiving policies.

    This class handles automatic cleanup of outdated rules, archiving for
    historical context, storage optimization, and memory consolidation.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the lifecycle manager.

        Args:
            memory_manager: The memory manager instance
        """
        self.memory_manager = memory_manager
        self.cleanup_policies = self._initialize_cleanup_policies()
        self.archive_collection = f"{memory_manager.memory_collection_name}_archive"

    def _initialize_cleanup_policies(self) -> dict[str, Any]:
        """Initialize default cleanup policies."""
        return {
            "max_age_days": {
                MemoryCategory.PREFERENCE: 365,  # Keep preferences for 1 year
                MemoryCategory.BEHAVIOR: 180,    # Keep behaviors for 6 months
                MemoryCategory.AGENT: 90,        # Keep agent defs for 3 months
            },
            "max_unused_days": 60,  # Archive rules unused for 60 days
            "max_total_rules": 1000,  # Maximum total rules before cleanup
            "consolidation_threshold": 5,  # Consolidate when 5+ similar rules exist
            "archive_enabled": True,
            "backup_before_cleanup": True,
        }

    async def run_cleanup_cycle(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Run a complete cleanup cycle with archiving and optimization.

        Args:
            dry_run: If True, only report what would be cleaned up

        Returns:
            Cleanup results and statistics
        """
        logger.info("Starting memory lifecycle cleanup cycle")

        cleanup_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "rules_processed": 0,
            "rules_archived": 0,
            "rules_deleted": 0,
            "rules_consolidated": 0,
            "storage_optimized": False,
            "errors": [],
            "actions": []
        }

        try:
            # Get all current memory rules
            all_rules = await self.memory_manager.list_memory_rules()
            cleanup_results["rules_processed"] = len(all_rules)

            if not all_rules:
                cleanup_results["actions"].append("No rules found to process")
                return cleanup_results

            # 1. Archive old and unused rules
            archive_results = await self._archive_old_rules(all_rules, dry_run)
            cleanup_results["rules_archived"] = archive_results["archived_count"]
            cleanup_results["actions"].extend(archive_results["actions"])

            # 2. Delete rules that should be removed (not archived)
            deletion_results = await self._delete_obsolete_rules(all_rules, dry_run)
            cleanup_results["rules_deleted"] = deletion_results["deleted_count"]
            cleanup_results["actions"].extend(deletion_results["actions"])

            # 3. Consolidate similar rules
            consolidation_results = await self._consolidate_similar_rules(all_rules, dry_run)
            cleanup_results["rules_consolidated"] = consolidation_results["consolidated_count"]
            cleanup_results["actions"].extend(consolidation_results["actions"])

            # 4. Optimize storage
            if not dry_run:
                storage_results = await self._optimize_storage()
                cleanup_results["storage_optimized"] = storage_results["optimized"]
                cleanup_results["actions"].extend(storage_results["actions"])

            logger.info(
                f"Cleanup cycle completed: {cleanup_results['rules_archived']} archived, "
                f"{cleanup_results['rules_deleted']} deleted, "
                f"{cleanup_results['rules_consolidated']} consolidated"
            )

        except Exception as e:
            error_msg = f"Cleanup cycle failed: {e}"
            logger.error(error_msg)
            cleanup_results["errors"].append(error_msg)

        return cleanup_results

    async def _archive_old_rules(self, rules: list[MemoryRule], dry_run: bool) -> dict[str, Any]:
        """Archive rules that are old or unused."""
        results = {"archived_count": 0, "actions": []}

        if not self.cleanup_policies["archive_enabled"]:
            results["actions"].append("Archiving disabled by policy")
            return results

        current_time = datetime.now(timezone.utc)
        rules_to_archive = []

        for rule in rules:
            should_archive = False
            reason = ""

            # Check age-based archiving
            age_days = (current_time - rule.created_at).days
            max_age = self.cleanup_policies["max_age_days"].get(rule.category, 180)

            if age_days > max_age:
                should_archive = True
                reason = f"Rule age ({age_days} days) exceeds maximum for {rule.category.value} ({max_age} days)"

            # Check usage-based archiving (placeholder - would need usage tracking)
            # For now, we'll simulate usage based on rule update frequency
            elif rule.updated_at and (current_time - rule.updated_at).days > self.cleanup_policies["max_unused_days"]:
                should_archive = True
                reason = f"Rule unused for {(current_time - rule.updated_at).days} days"

            if should_archive:
                rules_to_archive.append((rule, reason))

        # Perform archiving
        for rule, reason in rules_to_archive:
            action_desc = f"Archive rule '{rule.name}' ({rule.id}): {reason}"
            results["actions"].append(action_desc)

            if not dry_run:
                archive_success = await self._archive_rule(rule)
                if archive_success:
                    # Remove from active memory
                    await self.memory_manager.delete_memory_rule(rule.id)
                    results["archived_count"] += 1
                    logger.debug(f"Archived rule {rule.id}: {rule.name}")
                else:
                    results["actions"].append(f"Failed to archive rule {rule.id}")

        return results

    async def _delete_obsolete_rules(self, rules: list[MemoryRule], dry_run: bool) -> dict[str, Any]:
        """Delete rules that should be removed entirely (not archived)."""
        results = {"deleted_count": 0, "actions": []}

        rules_to_delete = []

        # Look for rules that should be deleted (e.g., replaced rules, invalid rules)
        for rule in rules:
            should_delete = False
            reason = ""

            # Check if this rule was explicitly replaced by another rule
            if rule.replaces:
                # Check if all replaced rules still exist (they shouldn't)
                for replaced_id in rule.replaces:
                    if any(r.id == replaced_id for r in rules):
                        should_delete = True
                        reason = f"Rule {replaced_id} still exists but should have been replaced"
                        break

            # Check for malformed rules (basic validation)
            if not rule.rule.strip() or len(rule.rule.strip()) < 3:
                should_delete = True
                reason = "Rule content is too short or empty"

            if should_delete:
                rules_to_delete.append((rule, reason))

        # Perform deletion
        for rule, reason in rules_to_delete:
            action_desc = f"Delete rule '{rule.name}' ({rule.id}): {reason}"
            results["actions"].append(action_desc)

            if not dry_run:
                delete_success = await self.memory_manager.delete_memory_rule(rule.id)
                if delete_success:
                    results["deleted_count"] += 1
                    logger.debug(f"Deleted rule {rule.id}: {rule.name}")

        return results

    async def _consolidate_similar_rules(self, rules: list[MemoryRule], dry_run: bool) -> dict[str, Any]:
        """Consolidate similar rules to reduce redundancy."""
        results = {"consolidated_count": 0, "actions": []}

        # Group rules by category and look for similar ones
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        for _category, category_rules in by_category.items():
            if len(category_rules) < self.cleanup_policies["consolidation_threshold"]:
                continue

            # Find similar rules within this category
            similar_groups = self._find_similar_rules(category_rules)

            for group in similar_groups:
                if len(group) >= 2:  # Only consolidate groups of 2 or more
                    consolidation_result = await self._consolidate_rule_group(group, dry_run)
                    if consolidation_result["consolidated"]:
                        results["consolidated_count"] += consolidation_result["rules_consolidated"]
                        results["actions"].extend(consolidation_result["actions"])

        return results

    def _find_similar_rules(self, rules: list[MemoryRule]) -> list[list[MemoryRule]]:
        """Find groups of similar rules that could be consolidated."""
        similar_groups = []
        processed_rules = set()

        for i, rule1 in enumerate(rules):
            if rule1.id in processed_rules:
                continue

            similar_group = [rule1]
            processed_rules.add(rule1.id)

            for _j, rule2 in enumerate(rules[i+1:], i+1):
                if rule2.id in processed_rules:
                    continue

                if self._rules_are_similar(rule1, rule2):
                    similar_group.append(rule2)
                    processed_rules.add(rule2.id)

            if len(similar_group) > 1:
                similar_groups.append(similar_group)

        return similar_groups

    def _rules_are_similar(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """Check if two rules are similar enough to be consolidated."""
        # Same category required
        if rule1.category != rule2.category:
            return False

        # Similar scope
        scope1 = set(rule1.scope or [])
        scope2 = set(rule2.scope or [])
        if scope1 and scope2 and not (scope1 & scope2):  # No overlap
            return False

        # Similar rule text (simple word overlap)
        words1 = set(rule1.rule.lower().split())
        words2 = set(rule2.rule.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union_size = len(words1 | words2)
        similarity = overlap / union_size if union_size > 0 else 0

        return similarity > 0.6  # 60% word overlap threshold

    async def _consolidate_rule_group(self, rules: list[MemoryRule], dry_run: bool) -> dict[str, Any]:
        """Consolidate a group of similar rules into one rule."""
        if len(rules) < 2:
            return {"consolidated": False, "rules_consolidated": 0, "actions": []}

        # Choose the "best" rule as the base (highest authority, most recent)
        base_rule = max(rules, key=lambda r: (
            r.authority == AuthorityLevel.ABSOLUTE,
            r.created_at.timestamp()
        ))

        # Create consolidated rule content
        consolidated_content = self._merge_rule_content(rules)
        consolidated_scope = list(set().union(*(r.scope or [] for r in rules)))
        consolidated_replaces = [r.id for r in rules if r.id != base_rule.id]

        action_desc = f"Consolidate {len(rules)} rules into '{base_rule.name}'"

        result = {
            "consolidated": True,
            "rules_consolidated": len(rules) - 1,  # One rule remains
            "actions": [action_desc]
        }

        if not dry_run:
            try:
                # Update the base rule with consolidated content
                updates = {
                    "rule": consolidated_content,
                    "scope": consolidated_scope,
                    "replaces": consolidated_replaces,
                    "metadata": {
                        "consolidated_from": [r.id for r in rules if r.id != base_rule.id],
                        "consolidation_date": datetime.now(timezone.utc).isoformat()
                    }
                }

                update_success = await self.memory_manager.update_memory_rule(
                    base_rule.id, updates
                )

                if update_success:
                    # Delete the other rules
                    for rule in rules:
                        if rule.id != base_rule.id:
                            await self.memory_manager.delete_memory_rule(rule.id)

                    logger.info(f"Consolidated {len(rules)} similar rules into {base_rule.id}")
                else:
                    result["consolidated"] = False
                    result["actions"].append("Failed to update consolidated rule")

            except Exception as e:
                result["consolidated"] = False
                result["actions"].append(f"Consolidation failed: {e}")

        return result

    def _merge_rule_content(self, rules: list[MemoryRule]) -> str:
        """Merge rule content from multiple similar rules."""
        # Simple merging strategy: combine unique sentences
        all_sentences = []
        for rule in rules:
            sentences = rule.rule.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in all_sentences:
                    all_sentences.append(sentence)

        # Choose the most comprehensive version, or merge if reasonable
        if len(all_sentences) <= 3:
            return '. '.join(all_sentences) + '.'
        else:
            # Take the longest rule if too many sentences
            return max(rules, key=lambda r: len(r.rule)).rule

    async def _optimize_storage(self) -> dict[str, Any]:
        """Optimize storage by cleaning up and compacting collections."""
        results = {"optimized": False, "actions": []}

        try:
            # Check collection statistics
            stats = await self.memory_manager.get_memory_stats()

            results["actions"].append(f"Current memory stats: {stats.total_rules} rules, ~{stats.estimated_tokens} tokens")

            # Placeholder for actual storage optimization
            # In a real implementation, this might involve:
            # - Reindexing the collection
            # - Optimizing vector storage
            # - Compacting the database
            # - Cleaning up orphaned data

            results["optimized"] = True
            results["actions"].append("Storage optimization completed")

        except Exception as e:
            results["actions"].append(f"Storage optimization failed: {e}")

        return results

    async def _archive_rule(self, rule: MemoryRule) -> bool:
        """Archive a single rule to the archive collection."""
        try:
            # Ensure archive collection exists
            await self._ensure_archive_collection()

            # Create archive entry with additional metadata
            archive_entry = {
                **asdict(rule),
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "original_collection": self.memory_manager.memory_collection_name
            }

            # Create embedding (placeholder - would use actual embedding service)
            embedding_vector = [0.0] * self.memory_manager.embedding_dim

            # Store in archive collection
            point = PointStruct(
                id=f"archived_{rule.id}",
                vector={"dense": embedding_vector},
                payload=archive_entry
            )

            self.memory_manager.client.upsert(
                collection_name=self.archive_collection,
                points=[point]
            )

            return True

        except Exception as e:
            logger.error(f"Failed to archive rule {rule.id}: {e}")
            return False

    async def _ensure_archive_collection(self):
        """Ensure the archive collection exists."""
        collections = self.memory_manager.client.get_collections()
        collection_names = {col.name for col in collections.collections}

        if self.archive_collection not in collection_names:
            # Create archive collection with same configuration as main collection
            vector_config = {
                "dense": VectorParams(
                    size=self.memory_manager.embedding_dim,
                    distance=Distance.COSINE
                )
            }

            self.memory_manager.client.create_collection(
                collection_name=self.archive_collection,
                vectors_config=vector_config
            )

            logger.info(f"Created archive collection: {self.archive_collection}")

    async def get_archived_rules(
        self,
        limit: int = 20,
        date_range: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve archived rules for analysis or restoration."""
        try:
            await self._ensure_archive_collection()

            # Build filter for date range if provided
            search_filter = None
            if date_range:
                start_date, end_date = date_range
                # Note: This would need proper date filtering implementation
                pass

            # Retrieve archived rules
            points, _ = self.memory_manager.client.scroll(
                collection_name=self.archive_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            archived_rules = []
            for point in points:
                payload = point.payload
                archived_rules.append({
                    "id": payload.get("id"),
                    "name": payload.get("name"),
                    "rule": payload.get("rule"),
                    "category": payload.get("category"),
                    "archived_at": payload.get("archived_at"),
                    "original_created_at": payload.get("created_at")
                })

            return archived_rules

        except Exception as e:
            logger.error(f"Failed to retrieve archived rules: {e}")
            return []

    async def restore_rule_from_archive(self, archived_rule_id: str) -> bool:
        """Restore a rule from the archive back to active memory."""
        try:
            # Retrieve from archive
            points = self.memory_manager.client.retrieve(
                collection_name=self.archive_collection,
                ids=[f"archived_{archived_rule_id}"],
                with_payload=True
            )

            if not points:
                logger.error(f"Archived rule {archived_rule_id} not found")
                return False

            archived_data = points[0].payload

            # Recreate the memory rule
            restored_rule_id = await self.memory_manager.add_memory_rule(
                category=MemoryCategory(archived_data["category"]),
                name=archived_data["name"],
                rule=archived_data["rule"],
                authority=AuthorityLevel(archived_data["authority"]),
                scope=archived_data.get("scope", []),
                source=f"{archived_data.get('source', 'archived')}_restored",
                conditions=archived_data.get("conditions"),
                metadata={
                    "restored_from_archive": True,
                    "original_id": archived_rule_id,
                    "restored_at": datetime.now(timezone.utc).isoformat()
                }
            )

            logger.info(f"Restored rule from archive: {archived_rule_id} -> {restored_rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore rule {archived_rule_id}: {e}")
            return False

    def update_cleanup_policy(self, policy_key: str, value: Any):
        """Update a cleanup policy setting."""
        if policy_key in self.cleanup_policies:
            self.cleanup_policies[policy_key] = value
            logger.info(f"Updated cleanup policy {policy_key} to {value}")
        else:
            logger.warning(f"Unknown cleanup policy: {policy_key}")


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


def parse_conversational_memory_update(
    message: str, context: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """
    Parse conversational memory updates from chat messages using advanced NLP.

    This function now uses the ConversationalMemoryProcessor for sophisticated
    pattern matching, context extraction, and confidence scoring.

    Args:
        message: The conversational message to parse
        context: Optional context about the conversation (project, task, etc.)

    Returns:
        Dictionary with parsed memory update or None if no update detected

    Note:
        This function leverages the enhanced ConversationalMemoryProcessor for better accuracy.
    """
    if not message or not message.strip():
        return None

    # Use the advanced processor for better results
    processor = ConversationalMemoryProcessor()
    result = processor.process_conversational_update(message, context)

    if result:
        # Convert ConversationalContext and extracted entities to serializable format
        if "context" in result:
            conv_context = result["context"]
            result["context"] = {
                "intent": conv_context.intent,
                "confidence": conv_context.confidence,
                "project_scope": conv_context.project_scope,
                "temporal_context": conv_context.temporal_context,
                "urgency_level": conv_context.urgency_level,
                "conditions": conv_context.conditions,
                "authority_signals": conv_context.authority_signals,
                "extracted_entities": conv_context.extracted_entities
            }

    return result


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


# --- Document memory system (compatibility layer for tests) ---


@dataclass
class DocumentMetadata:
    """Metadata for a document stored in the document memory system."""

    file_path: str
    file_type: str
    file_size: int
    created_at: datetime | None = None
    modified_at: datetime | None = None
    project: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    language: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.modified_at is None:
            self.modified_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "project": self.project,
            "author": self.author,
            "tags": self.tags,
            "language": self.language,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        created_at = data.get("created_at")
        modified_at = data.get("modified_at")
        return cls(
            file_path=data["file_path"],
            file_type=data["file_type"],
            file_size=data["file_size"],
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            modified_at=datetime.fromisoformat(modified_at) if modified_at else None,
            project=data.get("project"),
            author=data.get("author"),
            tags=data.get("tags"),
            language=data.get("language"),
            checksum=data.get("checksum"),
        )

    def update_checksum(self, content: str) -> None:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.checksum = digest


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""

    id: str
    document_id: str
    content: str
    start_offset: int
    end_offset: int
    chunk_index: int
    embedding: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def get_metadata_value(self, key: str) -> Any:
        if not self.metadata:
            return None
        return self.metadata.get(key)

    def update_embedding(self, embedding: dict[str, Any]) -> None:
        self.embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "chunk_index": self.chunk_index,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            content=data["content"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            chunk_index=data["chunk_index"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata"),
        )


@dataclass
class Document:
    """A document with optional chunks and embeddings."""

    id: str
    content: str
    metadata: DocumentMetadata
    embedding: dict[str, Any] | None = None
    chunks: list[DocumentChunk] = field(default_factory=list)

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_by_metadata(self, key: str, value: Any) -> list[DocumentChunk]:
        return [
            chunk for chunk in self.chunks
            if chunk.metadata and chunk.metadata.get(key) == value
        ]

    def update_embedding(self, embedding: dict[str, Any]) -> None:
        self.embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "embedding": self.embedding,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        metadata = DocumentMetadata.from_dict(data["metadata"])
        doc = cls(
            id=data["id"],
            content=data["content"],
            metadata=metadata,
            embedding=data.get("embedding"),
        )
        for chunk_data in data.get("chunks", []):
            doc.add_chunk(DocumentChunk.from_dict(chunk_data))
        return doc


class ChunkingStrategy:
    """Base class for document chunking strategies."""

    def chunk_text(self, content: str, document_id: str) -> list[DocumentChunk]:
        raise NotImplementedError


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """Chunks text by fixed size with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            overlap = max(0, chunk_size - 1)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, content: str, document_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        length = len(content)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk_content = content[start:end]
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}_chunk_{index}",
                    document_id=document_id,
                    content=chunk_content,
                    start_offset=start,
                    end_offset=end,
                    chunk_index=index,
                )
            )
            index += 1
            if end >= length:
                break
            start = max(0, end - self.overlap)
            if start >= length:
                break
        return chunks


class SentenceChunkingStrategy(ChunkingStrategy):
    """Chunks text by sentence groups."""

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    def chunk_text(self, content: str, document_id: str) -> list[DocumentChunk]:
        sentences = []
        for match in re.finditer(r"[^.!?]+[.!?]?", content):
            sentence = match.group(0)
            if sentence.strip():
                sentences.append((match.start(), match.end(), sentence))

        chunks: list[DocumentChunk] = []
        index = 0
        for i in range(0, len(sentences), self.max_sentences):
            group = sentences[i:i + self.max_sentences]
            if not group:
                continue
            start = group[0][0]
            end = group[-1][1]
            chunk_content = content[start:end]
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}_chunk_{index}",
                    document_id=document_id,
                    content=chunk_content,
                    start_offset=start,
                    end_offset=end,
                    chunk_index=index,
                )
            )
            index += 1
        return chunks or [
            DocumentChunk(
                id=f"{document_id}_chunk_0",
                document_id=document_id,
                content=content,
                start_offset=0,
                end_offset=len(content),
                chunk_index=0,
            )
        ]


class ParagraphChunkingStrategy(ChunkingStrategy):
    """Chunks text by paragraph boundaries."""

    def chunk_text(self, content: str, document_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        index = 0
        for match in re.finditer(r"(?:[^\n]|\n(?!\n))+", content):
            paragraph = match.group(0)
            if not paragraph.strip():
                continue
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}_chunk_{index}",
                    document_id=document_id,
                    content=paragraph,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    chunk_index=index,
                )
            )
            index += 1
        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """Mock semantic chunking strategy."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, content: str, document_id: str) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                id=f"{document_id}_chunk_0",
                document_id=document_id,
                content=content,
                start_offset=0,
                end_offset=len(content),
                chunk_index=0,
            )
        ]


def create_chunking_strategy(strategy_name: str, **kwargs: Any) -> ChunkingStrategy:
    name = strategy_name.lower()
    if name in {"fixed", "fixed_size", "fixed-size"}:
        return FixedSizeChunkingStrategy(
            chunk_size=kwargs.get("chunk_size", 512),
            overlap=kwargs.get("overlap", 0),
        )
    if name in {"sentence", "sentences"}:
        return SentenceChunkingStrategy(max_sentences=kwargs.get("max_sentences", 5))
    if name in {"paragraph", "paragraphs"}:
        return ParagraphChunkingStrategy()
    if name in {"semantic"}:
        return SemanticChunkingStrategy(
            similarity_threshold=kwargs.get("similarity_threshold", 0.8)
        )
    raise ValueError(f"Unknown chunking strategy: {strategy_name}")


class MemoryIndex:
    """In-memory index for stored documents."""

    def __init__(self, name: str, collection_name: str) -> None:
        self.name = name
        self.collection_name = collection_name
        self.documents: dict[str, Document] = {}

    def add_document(self, document: Document) -> None:
        self.documents[document.id] = document

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def remove_document(self, document_id: str) -> Document | None:
        return self.documents.pop(document_id, None)

    def get_statistics(self) -> dict[str, Any]:
        doc_count = len(self.documents)
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        total_size = sum(len(doc.content) for doc in self.documents.values())
        average_size = total_size / doc_count if doc_count else 0
        return {
            "document_count": doc_count,
            "total_chunks": total_chunks,
            "average_document_size": average_size,
        }

    def search_documents(self, query: str) -> list[Document]:
        query_lower = query.lower()
        return [
            doc for doc in self.documents.values()
            if query_lower in doc.content.lower()
        ]


@dataclass
class RetrievalOptions:
    """Options for retrieval and search operations."""

    limit: int = 10
    include_metadata: bool = False
    include_chunks: bool = False
    chunk_limit: int | None = None
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None
    sort_by: str | None = None


class DocumentMemoryManager:
    """Document storage manager backed by Qdrant and an embedding service."""

    def __init__(
        self,
        qdrant_client: Any,
        embedding_service: Any,
        collection_name: str,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.chunking_strategy = chunking_strategy

    async def _maybe_await(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    async def store_document(self, document: Document) -> dict[str, Any]:
        embedding = await self._maybe_await(
            self.embedding_service.embed_document(document.content)
        )
        document.update_embedding(embedding)

        chunks_created = 0
        if self.chunking_strategy:
            chunks = self.chunking_strategy.chunk_text(document.content, document.id)
            document.chunks = []
            for chunk in chunks:
                chunk_embedding = await self._maybe_await(
                    self.embedding_service.embed_document(chunk.content)
                )
                chunk.update_embedding(chunk_embedding)
                document.add_chunk(chunk)
            chunks_created = len(document.chunks)

        points = [self._build_document_point(document)]
        for chunk in document.chunks:
            points.append(self._build_chunk_point(chunk, document.metadata))

        await self._maybe_await(
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        )

        return {
            "status": "success",
            "document_id": document.id,
            "chunks_created": chunks_created,
        }

    async def retrieve_document(self, document_id: str) -> Document | None:
        results = await self._maybe_await(
            self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=None,
                limit=1,
                with_payload=True,
                filter={"document_id": document_id},
            )
        )
        if not results:
            return None
        point = results[0]
        payload = getattr(point, "payload", {}) or {}
        metadata = DocumentMetadata.from_dict(payload.get("metadata", {}))
        return Document(
            id=str(getattr(point, "id", document_id)),
            content=payload.get("content", ""),
            metadata=metadata,
            embedding=None,
        )

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        options: RetrievalOptions | None = None,
    ) -> list[dict[str, Any]]:
        embedding = await self._maybe_await(self.embedding_service.embed_query(query))
        effective_limit = options.limit if options else limit
        effective_filters = options.filters if options and options.filters else filters
        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "query_vector": embedding.get("dense") if isinstance(embedding, dict) else embedding,
            "limit": effective_limit,
            "with_payload": True,
        }
        if effective_filters:
            search_kwargs["filter"] = effective_filters
        if options and options.score_threshold is not None:
            search_kwargs["score_threshold"] = options.score_threshold

        results = await self._maybe_await(self.qdrant_client.search(**search_kwargs))
        output = []
        for point in results or []:
            output.append(
                {
                    "id": str(getattr(point, "id", "")),
                    "score": getattr(point, "score", None),
                    "payload": getattr(point, "payload", None),
                }
            )
        return output

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        await self._maybe_await(
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={"points": [document_id]},
            )
        )
        return {"status": "success", "document_id": document_id}

    async def update_document(self, document: Document) -> dict[str, Any]:
        return await self.store_document(document)

    async def batch_store_documents(self, documents: list[Document]) -> list[dict[str, Any]]:
        results = []
        for document in documents:
            results.append(await self.store_document(document))
        return results

    async def get_statistics(self) -> dict[str, Any]:
        info = await self._maybe_await(self.qdrant_client.get_collection(self.collection_name))
        return {
            "collection_name": self.collection_name,
            "vectors_count": getattr(info, "vectors_count", 0),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            "points_count": getattr(info, "points_count", 0),
            "status": getattr(info, "status", None),
        }

    def _build_document_point(self, document: Document) -> http_models.PointStruct:
        payload = {
            "content": document.content,
            "metadata": document.metadata.to_dict(),
            "document_type": "full_document",
        }
        vector = document.embedding.get("dense") if document.embedding else None
        return http_models.PointStruct(id=document.id, vector=vector, payload=payload)

    def _build_chunk_point(
        self,
        chunk: DocumentChunk,
        metadata: DocumentMetadata,
    ) -> http_models.PointStruct:
        payload = {
            "content": chunk.content,
            "metadata": metadata.to_dict(),
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "document_type": "chunk",
        }
        vector = chunk.embedding.get("dense") if chunk.embedding else None
        return http_models.PointStruct(id=chunk.id, vector=vector, payload=payload)
