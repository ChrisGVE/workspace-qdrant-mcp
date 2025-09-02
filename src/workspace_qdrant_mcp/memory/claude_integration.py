
"""
Claude Code SDK integration for memory system.

This module provides integration with Claude Code for automatic memory rule injection
and session initialization with memory-driven LLM behavior.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .token_counter import TokenCounter, TokenUsage
from .types import (
    AuthorityLevel,
    ClaudeCodeSession,
    ConversationalUpdate,
    MemoryCategory,
    MemoryContext,
    MemoryInjectionResult,
    MemoryRule,
)

logger = logging.getLogger(__name__)


class ClaudeCodeIntegration:
    """
    Integrates memory system with Claude Code SDK.

    Provides functionality for:
    - Session initialization with memory rule injection
    - Conversational memory updates detection and processing
    - Context-aware rule selection based on project and scope
    - Token budget management for optimal rule injection
    """

    def __init__(self,
                 token_counter: TokenCounter,
                 max_memory_tokens: int = 5000,
                 claude_config_path: str | None = None):
        """
        Initialize Claude Code integration.

        Args:
            token_counter: Token counter for optimization
            max_memory_tokens: Maximum tokens to use for memory rules
            claude_config_path: Path to Claude Code configuration
        """
        self.token_counter = token_counter
        self.max_memory_tokens = max_memory_tokens
        self.claude_config_path = claude_config_path or self._find_claude_config()

        # Cache for session contexts
        self._session_contexts: dict[str, MemoryContext] = {}

    async def initialize_session(self,
                                session: ClaudeCodeSession,
                                available_rules: list[MemoryRule]) -> MemoryInjectionResult:
        """
        Initialize a Claude Code session with memory rules.

        Args:
            session: Claude Code session information
            available_rules: List of all available memory rules

        Returns:
            Results of memory injection
        """
        try:
            # Create memory context for the session
            context = await self._create_memory_context(session)
            self._session_contexts[session.session_id] = context

            # Filter rules relevant to this session
            relevant_rules = self._filter_rules_by_context(available_rules, context)

            # Optimize rules for token budget
            selected_rules, token_usage = self.token_counter.optimize_rules_for_context(
                relevant_rules,
                self.max_memory_tokens,
                preserve_absolute=True
            )

            if not selected_rules:
                return MemoryInjectionResult(
                    success=True,
                    rules_injected=0,
                    total_tokens_used=0,
                    remaining_context_tokens=session.context_window_size,
                    skipped_rules=[],
                    errors=[]
                )

            # Generate memory injection content
            injection_content = self._generate_injection_content(selected_rules, context)

            # Inject into Claude Code session (via configuration or system prompt)
            success = await self._inject_into_session(session, injection_content)

            if success:
                # Update usage statistics for injected rules
                for rule in selected_rules:
                    rule.update_usage()

                skipped_rules = [
                    f"{rule.rule[:50]}..." for rule in relevant_rules
                    if rule not in selected_rules
                ]

                return MemoryInjectionResult(
                    success=True,
                    rules_injected=len(selected_rules),
                    total_tokens_used=token_usage.total_tokens,
                    remaining_context_tokens=session.context_window_size - token_usage.total_tokens,
                    skipped_rules=skipped_rules,
                    errors=[]
                )
            else:
                return MemoryInjectionResult(
                    success=False,
                    rules_injected=0,
                    total_tokens_used=0,
                    remaining_context_tokens=session.context_window_size,
                    skipped_rules=[],
                    errors=["Failed to inject memory rules into session"]
                )

        except Exception as e:
            logger.error(f"Failed to initialize session with memory rules: {e}")
            return MemoryInjectionResult(
                success=False,
                rules_injected=0,
                total_tokens_used=0,
                remaining_context_tokens=session.context_window_size,
                skipped_rules=[],
                errors=[str(e)]
            )

    def detect_conversational_updates(self,
                                    conversation_text: str,
                                    session_context: MemoryContext | None = None) -> list[ConversationalUpdate]:
        """
        Detect memory rule updates from conversational text.

        Args:
            conversation_text: Text from conversation to analyze
            session_context: Current session context

        Returns:
            List of detected conversational updates
        """
        updates = []

        # Pattern-based detection for common conversational updates
        patterns = [
            # "Note: call me Chris"
            (r'note:?\s+call\s+me\s+(\w+)', lambda m: ConversationalUpdate(
                text=conversation_text,
                extracted_rule=f"User name is {m.group(1)}",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["user_preference"],
                confidence=0.9
            )),

            # "Remember: I prefer X"
            (r'remember:?\s+i\s+prefer\s+(.+?)(?:\.|$)', lambda m: ConversationalUpdate(
                text=conversation_text,
                extracted_rule=f"User prefers {m.group(1).strip()}",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["user_preference"],
                confidence=0.8
            )),

            # "Always use X for Y"
            (r'always\s+use\s+(.+?)\s+for\s+(.+?)(?:\.|$)', lambda m: ConversationalUpdate(
                text=conversation_text,
                extracted_rule=f"Always use {m.group(1).strip()} for {m.group(2).strip()}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["behavior"],
                confidence=0.9
            )),

            # "Don't use X" or "Avoid X"
            (r'(?:don\'?t\s+use|avoid)\s+(.+?)(?:\.|$)', lambda m: ConversationalUpdate(
                text=conversation_text,
                extracted_rule=f"Avoid using {m.group(1).strip()}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["behavior"],
                confidence=0.8
            )),

            # "I work on project X"
            (r'i\s+work\s+on\s+(?:project\s+)?(.+?)(?:\.|$)', lambda m: ConversationalUpdate(
                text=conversation_text,
                extracted_rule=f"User works on project: {m.group(1).strip()}",
                category=MemoryCategory.CONTEXT,
                authority=AuthorityLevel.DEFAULT,
                scope=["project_context"],
                confidence=0.7
            )),
        ]

        import re
        text_lower = conversation_text.lower()

        for pattern, update_factory in patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                try:
                    update = update_factory(match)
                    if update.is_valid():
                        # Add session context if available
                        if session_context:
                            if session_context.project_name:
                                update.scope.append(f"project:{session_context.project_name}")
                            if session_context.user_name:
                                update.scope.append(f"user:{session_context.user_name}")

                        updates.append(update)
                except Exception as e:
                    logger.error(f"Error creating conversational update: {e}")
                    continue

        return updates

    def process_conversational_update(self, update: ConversationalUpdate) -> MemoryRule | None:
        """
        Convert a conversational update into a memory rule.

        Args:
            update: Conversational update to process

        Returns:
            MemoryRule if conversion successful, None otherwise
        """
        if not update.is_valid():
            return None

        try:
            return MemoryRule(
                rule=update.extracted_rule,
                category=update.category,
                authority=update.authority,
                scope=update.scope,
                source="conversation",
                metadata={
                    "original_text": update.text,
                    "confidence": update.confidence,
                    "extracted_at": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to convert conversational update to rule: {e}")
            return None

    def get_session_context(self, session_id: str) -> MemoryContext | None:
        """
        Get the memory context for a session.

        Args:
            session_id: Session identifier

        Returns:
            MemoryContext if found, None otherwise
        """
        return self._session_contexts.get(session_id)

    def update_session_context(self,
                             session_id: str,
                             context_updates: dict[str, Any]) -> bool:
        """
        Update the memory context for a session.

        Args:
            session_id: Session identifier
            context_updates: Dictionary of context updates

        Returns:
            True if updated successfully, False otherwise
        """
        if session_id not in self._session_contexts:
            return False

        try:
            context = self._session_contexts[session_id]

            for key, value in context_updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)

            return True
        except Exception as e:
            logger.error(f"Failed to update session context: {e}")
            return False

    async def _create_memory_context(self, session: ClaudeCodeSession) -> MemoryContext:
        """
        Create memory context from Claude Code session.

        Args:
            session: Claude Code session

        Returns:
            MemoryContext for the session
        """
        # Detect project name from workspace path
        project_name = session.project_name
        if not project_name and session.workspace_path:
            project_name = Path(session.workspace_path).name

        # Build active scopes based on session
        active_scopes = []
        if project_name:
            active_scopes.append(f"project:{project_name}")

        # Add file-type scopes based on active files
        if session.active_files:
            file_extensions = set()
            for file_path in session.active_files:
                ext = Path(file_path).suffix.lower()
                if ext:
                    file_extensions.add(ext[1:])  # Remove the dot

            for ext in file_extensions:
                active_scopes.append(f"filetype:{ext}")

        return MemoryContext(
            session_id=session.session_id,
            project_name=project_name,
            project_path=session.workspace_path,
            user_name=session.user_name,
            conversation_context=[],
            active_scopes=active_scopes
        )

    def _filter_rules_by_context(self,
                                rules: list[MemoryRule],
                                context: MemoryContext) -> list[MemoryRule]:
        """
        Filter rules based on session context relevance.

        Args:
            rules: All available rules
            context: Session context

        Returns:
            List of rules relevant to the context
        """
        relevant_rules = []
        context_scopes = context.to_scope_list()

        for rule in rules:
            if rule.matches_scope(context_scopes):
                relevant_rules.append(rule)

        return relevant_rules

    def _generate_injection_content(self,
                                  rules: list[MemoryRule],
                                  context: MemoryContext) -> str:
        """
        Generate content for injecting into Claude Code session.

        Args:
            rules: Rules to inject
            context: Session context

        Returns:
            Formatted injection content
        """
        content_parts = []

        # Header
        content_parts.append("# Memory-Driven Behavior Rules")
        content_parts.append("")

        if context.user_name:
            content_parts.append(f"User: {context.user_name}")
        if context.project_name:
            content_parts.append(f"Project: {context.project_name}")

        content_parts.append("")

        # Group rules by category and authority
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        if absolute_rules:
            content_parts.append("## Absolute Rules (Non-negotiable)")
            content_parts.append("")

            for rule in absolute_rules:
                scope_info = f" (Scope: {', '.join(rule.scope)})" if rule.scope else ""
                content_parts.append(f"- {rule.rule}{scope_info}")

            content_parts.append("")

        if default_rules:
            content_parts.append("## Default Rules (Override if needed)")
            content_parts.append("")

            for rule in default_rules:
                scope_info = f" (Scope: {', '.join(rule.scope)})" if rule.scope else ""
                content_parts.append(f"- {rule.rule}{scope_info}")

            content_parts.append("")

        content_parts.append("---")
        content_parts.append("")

        return "\n".join(content_parts)

    async def _inject_into_session(self,
                                 session: ClaudeCodeSession,
                                 content: str) -> bool:
        """
        Inject memory rules into Claude Code session.

        This could work through:
        1. Writing to Claude Code configuration
        2. System prompt injection
        3. Session initialization hooks

        Args:
            session: Claude Code session
            content: Content to inject

        Returns:
            True if injection successful, False otherwise
        """
        try:
            # Method 1: Write to session-specific configuration
            session_config_path = Path(session.workspace_path) / ".claude" / "memory.md"

            if session_config_path.parent.exists() or session_config_path.parent.mkdir(parents=True, exist_ok=True):
                session_config_path.write_text(content, encoding="utf-8")
                logger.info(f"Injected memory rules to {session_config_path}")
                return True

            # Method 2: Environment variable injection (fallback)
            os.environ["CLAUDE_MEMORY_RULES"] = content
            logger.info("Injected memory rules via environment variable")
            return True

        except Exception as e:
            logger.error(f"Failed to inject memory rules into session: {e}")
            return False

    def _find_claude_config(self) -> str | None:
        """
        Find Claude Code configuration directory.

        Returns:
            Path to Claude Code config directory or None
        """
        # Common locations for Claude Code configuration
        possible_paths = [
            Path.home() / ".claude",
            Path.home() / ".config" / "claude",
            Path("/etc/claude"),
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)

        return None

    def create_system_prompt_injection(self,
                                     rules: list[MemoryRule],
                                     context: MemoryContext) -> str:
        """
        Create a system prompt injection for memory rules.

        Args:
            rules: Memory rules to include
            context: Session context

        Returns:
            System prompt text with memory rules
        """
        prompt_parts = []

        prompt_parts.append("You are Claude Code with memory-driven behavior. Follow these rules:")
        prompt_parts.append("")

        # Absolute rules first
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        if absolute_rules:
            prompt_parts.append("ABSOLUTE RULES (Always follow):")
            for rule in absolute_rules:
                prompt_parts.append(f"- {rule.rule}")
            prompt_parts.append("")

        # Default rules
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]
        if default_rules:
            prompt_parts.append("DEFAULT RULES (Follow unless overridden):")
            for rule in default_rules:
                prompt_parts.append(f"- {rule.rule}")
            prompt_parts.append("")

        # Context information
        if context.user_name or context.project_name:
            prompt_parts.append("CONTEXT:")
            if context.user_name:
                prompt_parts.append(f"- User: {context.user_name}")
            if context.project_name:
                prompt_parts.append(f"- Project: {context.project_name}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)
