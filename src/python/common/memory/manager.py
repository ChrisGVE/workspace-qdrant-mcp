"""
Memory system manager.

This module provides the main interface for the memory-driven LLM behavior system,
coordinating all memory operations including storage, conflict detection, and integration.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ..core.client import QdrantWorkspaceClient
from ..core.config import ConfigManager, get_config
from ..core.embeddings import EmbeddingService
from .claude_integration import ClaudeCodeIntegration
from .conflict_detector import ConflictDetector
from .schema import MemoryCollectionSchema
from .token_counter import TokenCounter, TokenUsage
from .types import (
    AuthorityLevel,
    ClaudeCodeSession,
    MemoryCategory,
    MemoryContext,
    MemoryInjectionResult,
    MemoryRule,
    MemoryRuleConflict,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Main memory system manager.

    Coordinates all memory operations including:
    - Rule storage and retrieval from Qdrant
    - Conflict detection and resolution
    - Token counting and optimization
    - Claude Code integration
    - Conversational memory updates

    This is the primary interface for all memory system functionality.
    """

    def __init__(
        self,
        config: ConfigManager | None = None,
        qdrant_client: QdrantWorkspaceClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """
        Initialize memory manager.

        Args:
            config: Configuration object
            qdrant_client: Qdrant client instance
            embedding_service: Embedding service instance
        """
        self.config = config or get_config()

        # Initialize core components
        if qdrant_client:
            self.qdrant_client = qdrant_client
        else:
            # QdrantWorkspaceClient uses get_config() internally, no args needed
            self.qdrant_client = QdrantWorkspaceClient()

        if embedding_service:
            self.embedding_service = embedding_service
        else:
            # EmbeddingService uses get_config() internally, no args needed
            self.embedding_service = EmbeddingService()

        # Initialize memory components
        self.schema = MemoryCollectionSchema(self.qdrant_client, self.embedding_service)
        self.conflict_detector = ConflictDetector(
            enable_ai_analysis=getattr(self.config, "enable_memory_ai_analysis", True)
        )
        self.token_counter = TokenCounter(
            context_window_size=200000
        )  # Claude context window
        self.claude_integration = ClaudeCodeIntegration(
            token_counter=self.token_counter,
            max_memory_tokens=getattr(self.config, "max_memory_tokens", 5000),
        )

        # Internal state
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the memory system.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure memory collection exists
            if not await self.schema.ensure_collection_exists():
                logger.error("Failed to ensure memory collection exists")
                return False

            # Initialize embedding service
            if hasattr(self.embedding_service, "initialize"):
                await self.embedding_service.initialize()

            self._initialized = True
            logger.info("Memory system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            return False

    async def add_rule(
        self, rule: MemoryRule, check_conflicts: bool = True
    ) -> tuple[str, list[MemoryRuleConflict]]:
        """
        Add a new memory rule.

        Args:
            rule: MemoryRule to add
            check_conflicts: Whether to check for conflicts before adding

        Returns:
            Tuple of (rule_id, list_of_conflicts)
        """
        if not self._initialized:
            await self.initialize()

        conflicts = []

        try:
            # Check for conflicts if requested
            if check_conflicts:
                existing_rules = await self.list_rules()
                conflicts = await self.conflict_detector.detect_conflicts(
                    rule, existing_rules
                )

                # Log conflicts but don't prevent addition (user decision)
                if conflicts:
                    logger.warning(
                        f"Found {len(conflicts)} potential conflicts for new rule"
                    )
                    for conflict in conflicts:
                        logger.warning(
                            f"  - {conflict.severity}: {conflict.description}"
                        )

            # Store the rule
            success = await self.schema.store_rule(rule)

            if success:
                logger.info(f"Added memory rule: {rule.id}")
                return rule.id, conflicts
            else:
                raise Exception("Failed to store rule in Qdrant")

        except Exception as e:
            logger.error(f"Failed to add memory rule: {e}")
            raise

    async def update_rule(self, rule: MemoryRule) -> bool:
        """
        Update an existing memory rule.

        Args:
            rule: Updated MemoryRule

        Returns:
            True if updated successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            rule.updated_at = datetime.now(timezone.utc)
            success = await self.schema.update_rule(rule)

            if success:
                logger.info(f"Updated memory rule: {rule.id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update memory rule {rule.id}: {e}")
            return False

    async def get_rule(self, rule_id: str) -> MemoryRule | None:
        """
        Get a memory rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            MemoryRule if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        return await self.schema.get_rule(rule_id)

    async def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a memory rule.

        Args:
            rule_id: Rule identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            success = await self.schema.delete_rule(rule_id)

            if success:
                logger.info(f"Deleted memory rule: {rule_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete memory rule {rule_id}: {e}")
            return False

    async def list_rules(
        self,
        category_filter: MemoryCategory | None = None,
        authority_filter: AuthorityLevel | None = None,
        source_filter: str | None = None,
    ) -> list[MemoryRule]:
        """
        List memory rules with optional filtering.

        Args:
            category_filter: Filter by category
            authority_filter: Filter by authority level
            source_filter: Filter by source

        Returns:
            List of MemoryRule objects
        """
        if not self._initialized:
            await self.initialize()

        return await self.schema.list_all_rules(
            category_filter=category_filter,
            authority_filter=authority_filter,
            source_filter=source_filter,
        )

    async def search_rules(
        self, query: str, limit: int = 10, **filters
    ) -> list[tuple[MemoryRule, float]]:
        """
        Search memory rules using semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            **filters: Additional filters (category_filter, authority_filter, etc.)

        Returns:
            List of (MemoryRule, score) tuples
        """
        if not self._initialized:
            await self.initialize()

        return await self.schema.search_rules(query, limit, **filters)

    async def check_conflicts(self, rule: MemoryRule) -> list[MemoryRuleConflict]:
        """
        Check for conflicts with a rule against all existing rules.

        Args:
            rule: MemoryRule to check

        Returns:
            List of detected conflicts
        """
        if not self._initialized:
            await self.initialize()

        existing_rules = await self.list_rules()
        return await self.conflict_detector.detect_conflicts(rule, existing_rules)

    async def analyze_all_conflicts(self) -> list[MemoryRuleConflict]:
        """
        Analyze conflicts among all rules.

        Returns:
            List of all detected conflicts
        """
        if not self._initialized:
            await self.initialize()

        rules = await self.list_rules()
        return await self.conflict_detector.analyze_all_conflicts(rules)

    async def get_token_usage(self) -> TokenUsage:
        """
        Get current token usage for all memory rules.

        Returns:
            TokenUsage object with statistics
        """
        if not self._initialized:
            await self.initialize()

        rules = await self.list_rules()
        return self.token_counter.count_rules_tokens(rules)

    async def optimize_rules_for_context(
        self, context: MemoryContext, max_tokens: int = 5000
    ) -> tuple[list[MemoryRule], TokenUsage]:
        """
        Optimize rule selection for a specific context.

        Args:
            context: Memory context (session, project, etc.)
            max_tokens: Maximum tokens to use

        Returns:
            Tuple of (selected_rules, token_usage)
        """
        if not self._initialized:
            await self.initialize()

        # Get all rules and filter by context
        all_rules = await self.list_rules()
        context_scopes = context.to_scope_list()

        relevant_rules = [
            rule for rule in all_rules if rule.matches_scope(context_scopes)
        ]

        return self.token_counter.optimize_rules_for_context(
            relevant_rules, max_tokens, preserve_absolute=True
        )

    async def suggest_optimizations(self, target_tokens: int = 3000) -> dict[str, Any]:
        """
        Suggest optimizations to reduce memory token usage.

        Args:
            target_tokens: Target token count

        Returns:
            Dictionary with optimization suggestions
        """
        if not self._initialized:
            await self.initialize()

        rules = await self.list_rules()
        return self.token_counter.suggest_memory_optimizations(rules, target_tokens)

    async def process_conversational_text(
        self, text: str, session_context: MemoryContext | None = None
    ) -> list[MemoryRule]:
        """
        Process conversational text for memory updates.

        Args:
            text: Conversation text to analyze
            session_context: Current session context

        Returns:
            List of new MemoryRule objects created from conversation
        """
        if not self._initialized:
            await self.initialize()

        # Detect conversational updates
        updates = self.claude_integration.detect_conversational_updates(
            text, session_context
        )

        new_rules = []
        for update in updates:
            # Convert update to rule
            rule = self.claude_integration.process_conversational_update(update)

            if rule:
                # Add the rule (with conflict checking)
                try:
                    rule_id, conflicts = await self.add_rule(rule, check_conflicts=True)

                    if conflicts:
                        logger.info(
                            f"Added conversational rule with {len(conflicts)} conflicts: {rule.rule}"
                        )
                    else:
                        logger.info(f"Added conversational rule: {rule.rule}")

                    new_rules.append(rule)

                except Exception as e:
                    logger.error(f"Failed to add conversational rule: {e}")

        return new_rules

    async def initialize_claude_session(
        self, session: ClaudeCodeSession
    ) -> MemoryInjectionResult:
        """
        Initialize a Claude Code session with memory rules.

        Args:
            session: Claude Code session information

        Returns:
            Results of memory injection
        """
        if not self._initialized:
            await self.initialize()

        all_rules = await self.list_rules()
        return await self.claude_integration.initialize_session(session, all_rules)

    async def get_memory_stats(self) -> dict[str, Any]:
        """
        Get comprehensive memory system statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        # Collection stats from Qdrant
        collection_stats = await self.schema.get_collection_stats()

        # Token usage
        token_usage = await self.get_token_usage()

        # Conflict analysis
        all_conflicts = await self.analyze_all_conflicts()
        conflict_summary = self.conflict_detector.get_conflict_summary(all_conflicts)

        return {
            "collection": collection_stats,
            "token_usage": token_usage.to_dict(),
            "conflicts": conflict_summary,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def export_rules(self) -> list[dict[str, Any]]:
        """
        Export all memory rules to a serializable format.

        Returns:
            List of rule dictionaries
        """
        if not self._initialized:
            await self.initialize()

        rules = await self.list_rules()
        return [rule.to_dict() for rule in rules]

    async def import_rules(
        self, rule_dicts: list[dict[str, Any]], overwrite_existing: bool = False
    ) -> tuple[int, int, list[str]]:
        """
        Import memory rules from serialized format.

        Args:
            rule_dicts: List of rule dictionaries
            overwrite_existing: Whether to overwrite existing rules

        Returns:
            Tuple of (imported_count, skipped_count, error_messages)
        """
        if not self._initialized:
            await self.initialize()

        imported_count = 0
        skipped_count = 0
        errors = []

        for rule_dict in rule_dicts:
            try:
                rule = MemoryRule.from_dict(rule_dict)

                # Check if rule already exists
                existing_rule = await self.get_rule(rule.id)

                if existing_rule and not overwrite_existing:
                    skipped_count += 1
                    continue

                # Add or update rule
                if existing_rule:
                    success = await self.update_rule(rule)
                else:
                    _, conflicts = await self.add_rule(rule, check_conflicts=False)
                    success = True  # add_rule raises exception on failure

                if success:
                    imported_count += 1
                else:
                    skipped_count += 1
                    errors.append(f"Failed to import rule: {rule.id}")

            except Exception as e:
                skipped_count += 1
                errors.append(f"Error importing rule: {str(e)}")

        return imported_count, skipped_count, errors
