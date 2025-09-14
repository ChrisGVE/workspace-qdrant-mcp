"""
Claude Code SDK integration for memory system.

This module provides integration with Claude Code for automatic memory rule
injection and session initialization. It handles the workflow of loading
memory rules, detecting conflicts, and formatting them for system context.
"""

from loguru import logger
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
)

# logger imported from loguru


class ClaudeIntegrationManager:
    """
    Manages integration between memory system and Claude Code SDK.

    This class handles the session initialization workflow, conflict resolution,
    and system context formatting for Claude Code integration.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the Claude integration manager.

        Args:
            memory_manager: The memory manager instance
        """
        self.memory_manager = memory_manager

    async def initialize_session(self) -> dict[str, Any]:
        """
        Initialize Claude Code session with memory rules.

        This is the main session initialization workflow that:
        1. Loads all memory rules
        2. Detects conflicts
        3. Prepares system context injection
        4. Returns session status

        Returns:
            Session initialization data with rules and conflict status
        """
        try:
            # Ensure memory collection exists
            await self.memory_manager.initialize_memory_collection()

            # Load all memory rules
            rules = await self.memory_manager.list_memory_rules()

            # Detect conflicts
            conflicts = await self.memory_manager.detect_conflicts(rules)

            # Get memory statistics
            stats = await self.memory_manager.get_memory_stats()

            # Prepare session data
            session_data = {
                "status": "ready" if not conflicts else "conflicts_detected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_stats": {
                    "total_rules": stats.total_rules,
                    "estimated_tokens": stats.estimated_tokens,
                    "rules_by_category": {
                        k.value: v for k, v in stats.rules_by_category.items()
                    },
                    "rules_by_authority": {
                        k.value: v for k, v in stats.rules_by_authority.items()
                    },
                },
                "conflicts_detected": len(conflicts),
                "system_context": await self._format_system_context(rules),
                "rules_summary": self._create_rules_summary(rules),
            }

            # Include conflict details if any
            if conflicts:
                session_data["conflicts"] = [
                    self._format_conflict(conflict) for conflict in conflicts
                ]
                session_data["conflict_resolution_prompt"] = (
                    self._generate_conflict_resolution_prompt(conflicts)
                )

            logger.info(
                f"Claude session initialized: {len(rules)} rules, {len(conflicts)} conflicts"
            )
            return session_data

        except Exception as e:
            logger.error(f"Failed to initialize Claude session: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def format_system_rules_for_injection(
        self, rules: list[MemoryRule] | None = None
    ) -> str:
        """
        Format memory rules for injection into Claude Code system context.

        Args:
            rules: Specific rules to format (default: all rules)

        Returns:
            Formatted string ready for system context injection
        """
        if rules is None:
            rules = await self.memory_manager.list_memory_rules()

        return await self._format_system_context(rules)

    async def handle_conversational_update(self, message: str) -> dict[str, Any]:
        """
        Handle conversational memory updates and apply them immediately.

        Args:
            message: The conversational message to parse

        Returns:
            Result of processing the conversational update
        """
        from .memory import parse_conversational_memory_update

        try:
            # Parse the message
            parsed = parse_conversational_memory_update(message)

            if not parsed:
                return {
                    "detected": False,
                    "message": "No memory update pattern detected in message",
                }

            # Generate a name for the rule
            name = self._generate_rule_name(parsed["rule"])

            # Add the rule to memory
            rule_id = await self.memory_manager.add_memory_rule(
                category=parsed["category"],
                name=name,
                rule=parsed["rule"],
                authority=parsed["authority"],
                source=parsed["source"],
            )

            # Get updated system context
            updated_context = await self.format_system_rules_for_injection()

            logger.info(f"Applied conversational memory update: {rule_id}")
            return {
                "detected": True,
                "rule_added": True,
                "rule_id": rule_id,
                "rule_name": name,
                "category": parsed["category"].value,
                "authority": parsed["authority"].value,
                "rule_text": parsed["rule"],
                "updated_system_context": updated_context,
                "message": f"Added memory rule: {parsed['rule']}",
            }

        except Exception as e:
            logger.error(f"Failed to handle conversational update: {e}")
            return {"detected": True, "rule_added": False, "error": str(e)}

    async def resolve_conflicts(
        self, conflict_resolutions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Apply conflict resolutions and update memory rules.

        Args:
            conflict_resolutions: List of resolution decisions

        Returns:
            Result of conflict resolution process
        """
        try:
            resolved_count = 0
            failed_resolutions = []

            for resolution in conflict_resolutions:
                try:
                    await self._apply_conflict_resolution(resolution)
                    resolved_count += 1
                except Exception as e:
                    failed_resolutions.append(
                        {"resolution": resolution, "error": str(e)}
                    )
                    logger.error(f"Failed to apply conflict resolution: {e}")

            # Get updated system context
            updated_context = await self.format_system_rules_for_injection()

            return {
                "success": True,
                "resolved_conflicts": resolved_count,
                "failed_resolutions": len(failed_resolutions),
                "failures": failed_resolutions,
                "updated_system_context": updated_context,
            }

        except Exception as e:
            logger.error(f"Failed to resolve conflicts: {e}")
            return {"success": False, "error": str(e)}

    def _create_rules_summary(self, rules: list[MemoryRule]) -> dict[str, Any]:
        """Create a summary of memory rules for session initialization."""
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        return {
            "total_rules": len(rules),
            "absolute_rules": len(absolute_rules),
            "default_rules": len(default_rules),
            "categories": {
                category.value: len([r for r in rules if r.category == category])
                for category in MemoryCategory
            },
            "recent_rules": [
                {
                    "name": rule.name,
                    "rule": rule.rule,
                    "category": rule.category.value,
                    "authority": rule.authority.value,
                }
                for rule in sorted(
                    rules, key=lambda r: r.created_at or datetime.min, reverse=True
                )[:5]
            ],
        }

    async def _format_system_context(self, rules: list[MemoryRule]) -> str:
        """
        Format memory rules for Claude Code system context injection.

        Args:
            rules: Memory rules to format

        Returns:
            Formatted system context string
        """
        if not rules:
            return """# User Memory Rules (workspace-qdrant-mcp)
## No memory rules configured
The memory collection is empty. Memory rules can be added via conversational updates or the wqm memory CLI."""

        # Separate by authority level
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Separate by category for organization
        preferences = [r for r in rules if r.category == MemoryCategory.PREFERENCE]
        behaviors = [r for r in rules if r.category == MemoryCategory.BEHAVIOR]
        agents = [r for r in rules if r.category == MemoryCategory.AGENT]

        context = f"""# User Memory Rules (workspace-qdrant-mcp)
## Loaded {len(rules)} rules from memory collection

### Rule Categories
- **Preferences**: {len(preferences)} rules (user preferences and settings)
- **Behaviors**: {len(behaviors)} rules (LLM behavioral instructions)
- **Agents**: {len(agents)} rules (available agent definitions)

### Authority Levels
- **Absolute Rules**: {len(absolute_rules)} rules (always follow, non-negotiable)
- **Default Rules**: {len(default_rules)} rules (follow unless explicitly overridden)

"""

        # Add absolute rules
        if absolute_rules:
            context += "## Absolute Rules (ALWAYS FOLLOW)\n"
            context += "These rules are non-negotiable and must always be followed:\n\n"

            for rule in absolute_rules:
                scope_text = f" (scope: {', '.join(rule.scope)})" if rule.scope else ""
                context += f"**{rule.name}** [{rule.category.value}]{scope_text}:\n"
                context += f"{rule.rule}\n\n"

        # Add default rules
        if default_rules:
            context += "## Default Rules (follow unless explicitly overridden)\n"
            context += "These rules should be followed unless explicitly overridden by user instructions or PRD context:\n\n"

            for rule in default_rules:
                scope_text = f" (scope: {', '.join(rule.scope)})" if rule.scope else ""
                context += f"**{rule.name}** [{rule.category.value}]{scope_text}:\n"
                context += f"{rule.rule}\n\n"

        # Add agent library if any agent rules exist
        if agents:
            context += "## Available Agents\n"
            context += "Agent definitions available for deployment decisions:\n\n"

            for agent_rule in agents:
                context += f"**{agent_rule.name}**:\n"
                context += f"{agent_rule.rule}\n\n"

        context += """## Conflict Resolution
If conflicting instructions arise during this session:
1. Alert the user about the conflict
2. Reference these memory rules for clarification
3. Ask for guidance on which instruction takes priority

## Memory Updates
New memory rules can be added through conversational patterns:
- "Note: [preference]"
- "For future reference, [instruction]"
- "Remember that I [preference]"
- "Always [behavior]" or "Never [behavior]"

Session initialized with memory-driven context."""

        return context

    def _format_conflict(self, conflict: MemoryConflict) -> dict[str, Any]:
        """Format a memory conflict for session data."""
        return {
            "type": conflict.conflict_type,
            "description": conflict.description,
            "confidence": conflict.confidence,
            "rule1": {
                "id": conflict.rule1.id,
                "name": conflict.rule1.name,
                "rule": conflict.rule1.rule,
                "authority": conflict.rule1.authority.value,
                "category": conflict.rule1.category.value,
            },
            "rule2": {
                "id": conflict.rule2.id,
                "name": conflict.rule2.name,
                "rule": conflict.rule2.rule,
                "authority": conflict.rule2.authority.value,
                "category": conflict.rule2.category.value,
            },
            "resolution_options": conflict.resolution_options,
        }

    def _generate_conflict_resolution_prompt(
        self, conflicts: list[MemoryConflict]
    ) -> str:
        """Generate a user-friendly conflict resolution prompt."""
        if not conflicts:
            return ""

        prompt = f"# Memory Rule Conflicts Detected ({len(conflicts)})\n\n"
        prompt += "The following memory rules appear to conflict with each other. Please resolve these conflicts before proceeding:\n\n"

        for i, conflict in enumerate(conflicts, 1):
            prompt += f"## Conflict {i}: {conflict.conflict_type}\n"
            prompt += f"**Confidence**: {conflict.confidence:.1%}\n"
            prompt += f"**Description**: {conflict.description}\n\n"

            prompt += f"**Rule 1**: {conflict.rule1.name} ({conflict.rule1.authority.value})\n"
            prompt += f"```\n{conflict.rule1.rule}\n```\n\n"

            prompt += f"**Rule 2**: {conflict.rule2.name} ({conflict.rule2.authority.value})\n"
            prompt += f"```\n{conflict.rule2.rule}\n```\n\n"

            prompt += "**Resolution Options**:\n"
            for j, option in enumerate(conflict.resolution_options, 1):
                prompt += f"{j}. {option}\n"
            prompt += "\n"

        prompt += "Please choose a resolution for each conflict, or use `wqm memory conflicts` to resolve them interactively."

        return prompt

    def _generate_rule_name(self, rule_text: str) -> str:
        """Generate a short name from rule text."""
        words = rule_text.lower().split()[:3]
        # Remove common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "always",
            "never",
            "i",
            "me",
            "my",
        }
        words = [w for w in words if w not in stop_words]
        # Take first 2 meaningful words
        name_words = words[:2] if len(words) >= 2 else words
        return "-".join(name_words) if name_words else "conversational-rule"

    async def _apply_conflict_resolution(self, resolution: dict[str, Any]):
        """Apply a single conflict resolution."""
        resolution_type = resolution.get("type")

        if resolution_type == "keep_higher_authority":
            # Keep the rule with higher authority, remove the other
            rule1_id = resolution["rule1_id"]
            rule2_id = resolution["rule2_id"]
            rule1_authority = resolution["rule1_authority"]
            rule2_authority = resolution["rule2_authority"]

            if rule1_authority == "absolute" and rule2_authority != "absolute":
                await self.memory_manager.delete_memory_rule(rule2_id)
            elif rule2_authority == "absolute" and rule1_authority != "absolute":
                await self.memory_manager.delete_memory_rule(rule1_id)
            else:
                raise ValueError("Cannot determine which rule has higher authority")

        elif resolution_type == "merge_rules":
            # Merge two rules into one
            merged_rule = resolution["merged_rule"]
            rule1_id = resolution["rule1_id"]
            rule2_id = resolution["rule2_id"]

            # Delete both original rules
            await self.memory_manager.delete_memory_rule(rule1_id)
            await self.memory_manager.delete_memory_rule(rule2_id)

            # Add the merged rule
            await self.memory_manager.add_memory_rule(
                category=MemoryCategory(merged_rule["category"]),
                name=merged_rule["name"],
                rule=merged_rule["rule"],
                authority=AuthorityLevel(merged_rule["authority"]),
                source="conflict_resolution",
                replaces=[rule1_id, rule2_id],
            )

        elif resolution_type == "delete_rule":
            # Delete the specified rule
            rule_id = resolution["rule_id"]
            await self.memory_manager.delete_memory_rule(rule_id)

        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")


def create_claude_integration(
    memory_manager: MemoryManager,
) -> ClaudeIntegrationManager:
    """
    Factory function to create a Claude integration manager.

    Args:
        memory_manager: The memory manager instance

    Returns:
        Configured ClaudeIntegrationManager instance
    """
    return ClaudeIntegrationManager(memory_manager)
