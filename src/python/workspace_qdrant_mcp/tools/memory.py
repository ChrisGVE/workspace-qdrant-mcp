"""
MCP tools for memory management.

This module provides MCP tools for interacting with the memory collection system,
including session initialization, conversational updates, and memory queries.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from mcp.server.fastmcp import FastMCP

from python.common.core.client import create_qdrant_client
from python.common.core.collection_naming import create_naming_manager
from python.common.core.config import Config
from python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
    parse_conversational_memory_update,
)


# Function imported from common.core.memory


# logger imported from loguru


def register_memory_tools(server: FastMCP):
    """Register memory management tools with the MCP server."""

    @server.tool()
    async def initialize_memory_session() -> dict[str, Any]:
        """
        Initialize memory system for Claude Code session.

        Loads all memory rules, detects conflicts, and prepares rule injection.
        This is typically called at Claude Code startup.

        Returns:
            Dictionary with session initialization status and memory rules
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()

            # Load all memory rules
            rules = await memory_manager.list_memory_rules()

            # Detect conflicts
            conflicts = await memory_manager.detect_conflicts(rules)

            # Get memory statistics
            stats = await memory_manager.get_memory_stats()

            # Format rules for Claude Code injection
            absolute_rules = [
                r for r in rules if r.authority == AuthorityLevel.ABSOLUTE
            ]
            default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

            # Prepare session data
            session_data = {
                "status": "ready",
                "total_rules": len(rules),
                "absolute_rules": len(absolute_rules),
                "default_rules": len(default_rules),
                "conflicts_detected": len(conflicts),
                "estimated_tokens": stats.estimated_tokens,
                "rules_for_injection": {
                    "absolute": [
                        {
                            "name": rule.name,
                            "rule": rule.rule,
                            "scope": rule.scope,
                            "category": rule.category.value,
                        }
                        for rule in absolute_rules
                    ],
                    "default": [
                        {
                            "name": rule.name,
                            "rule": rule.rule,
                            "scope": rule.scope,
                            "category": rule.category.value,
                        }
                        for rule in default_rules
                    ],
                },
            }

            # Include conflict information if any
            if conflicts:
                session_data["conflicts"] = [
                    {
                        "type": conflict.conflict_type,
                        "description": conflict.description,
                        "rule1_name": conflict.rule1.name,
                        "rule2_name": conflict.rule2.name,
                        "confidence": conflict.confidence,
                        "resolution_options": conflict.resolution_options,
                    }
                    for conflict in conflicts
                ]
                session_data["status"] = "conflicts_require_resolution"

            logger.info(
                f"Memory session initialized: {len(rules)} rules loaded, {len(conflicts)} conflicts"
            )
            return session_data

        except Exception as e:
            logger.error(f"Failed to initialize memory session: {e}")
            return {"status": "error", "error": str(e), "total_rules": 0}

    @server.tool()
    async def add_memory_rule(
        category: str,
        name: str,
        rule: str,
        authority: str = "default",
        scope: list[str] | None = None,
        source: str = "mcp_user",
    ) -> dict[str, Any]:
        """
        Add a new memory rule.

        Args:
            category: Memory category (preference, behavior, agent)
            name: Short name for the rule
            rule: The actual rule text
            authority: Authority level (absolute, default)
            scope: List of contexts where rule applies
            source: Source of the rule creation

        Returns:
            Result of rule creation with ID
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Validate inputs
            try:
                category_enum = MemoryCategory(category)
                authority_enum = AuthorityLevel(authority)
            except ValueError as e:
                return {"success": False, "error": f"Invalid parameter: {e}"}

            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()

            # Add the rule
            rule_id = await memory_manager.add_memory_rule(
                category=category_enum,
                name=name,
                rule=rule,
                authority=authority_enum,
                scope=scope or [],
                source=source,
            )

            logger.info(f"Added memory rule via MCP: {rule_id}")
            return {
                "success": True,
                "rule_id": rule_id,
                "message": f"Added memory rule '{name}'",
            }

        except Exception as e:
            logger.error(f"Failed to add memory rule: {e}")
            return {"success": False, "error": str(e)}

    @server.tool()
    async def update_memory_from_conversation(message: str) -> dict[str, Any]:
        """
        Parse conversational message for memory updates.

        Detects patterns like "Note: call me Chris" and automatically
        adds them as memory rules.

        Args:
            message: The conversational message to parse

        Returns:
            Result of parsing and optional rule creation
        """
        try:
            # Parse the conversational update
            parsed = parse_conversational_memory_update(message)

            if not parsed:
                return {
                    "detected": False,
                    "message": "No memory update pattern detected",
                }

            # If update detected, add it as a memory rule
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()

            # Generate name from rule
            name_words = parsed["rule"].lower().split()[:2]
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
            }
            name_words = [w for w in name_words if w not in stop_words]
            name = "-".join(name_words) if name_words else "conversational-rule"

            # Add the rule
            rule_id = await memory_manager.add_memory_rule(
                category=parsed["category"],
                name=name,
                rule=parsed["rule"],
                authority=parsed["authority"],
                source=parsed["source"],
            )

            logger.info(f"Added conversational memory rule: {rule_id}")
            return {
                "detected": True,
                "rule_added": True,
                "rule_id": rule_id,
                "category": parsed["category"].value,
                "rule": parsed["rule"],
                "authority": parsed["authority"].value,
                "message": f"Added memory rule from conversation: '{parsed['rule']}'",
            }

        except Exception as e:
            logger.error(f"Failed to process conversational memory update: {e}")
            return {"detected": True, "rule_added": False, "error": str(e)}

    @server.tool()
    async def search_memory_rules(
        query: str,
        category: str | None = None,
        authority: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Search memory rules by semantic similarity.

        Args:
            query: Search query
            category: Filter by category (optional)
            authority: Filter by authority level (optional)
            limit: Maximum number of results

        Returns:
            List of matching memory rules with relevance scores
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Validate optional parameters
            category_enum = None
            authority_enum = None

            if category:
                try:
                    category_enum = MemoryCategory(category)
                except ValueError:
                    return {"success": False, "error": f"Invalid category: {category}"}

            if authority:
                try:
                    authority_enum = AuthorityLevel(authority)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid authority: {authority}",
                    }

            # Search memory rules
            results = await memory_manager.search_memory_rules(
                query=query,
                limit=limit,
                category=category_enum,
                authority=authority_enum,
            )

            # Format results
            formatted_results = []
            for rule, score in results:
                formatted_results.append(
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "rule": rule.rule,
                        "category": rule.category.value,
                        "authority": rule.authority.value,
                        "scope": rule.scope,
                        "relevance_score": score,
                        "created_at": rule.created_at.isoformat()
                        if rule.created_at
                        else None,
                    }
                )

            logger.info(
                f"Memory search returned {len(results)} results for query: {query}"
            )
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_found": len(results),
            }

        except Exception as e:
            logger.error(f"Failed to search memory rules: {e}")
            return {"success": False, "error": str(e)}

    @server.tool()
    async def get_memory_stats() -> dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Statistics about memory rules and token usage
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            stats = await memory_manager.get_memory_stats()

            return {
                "total_rules": stats.total_rules,
                "rules_by_category": {
                    k.value: v for k, v in stats.rules_by_category.items()
                },
                "rules_by_authority": {
                    k.value: v for k, v in stats.rules_by_authority.items()
                },
                "estimated_tokens": stats.estimated_tokens,
                "last_optimization": stats.last_optimization.isoformat()
                if stats.last_optimization
                else None,
                "token_status": (
                    "low"
                    if stats.estimated_tokens < 1000
                    else "moderate"
                    if stats.estimated_tokens < 2000
                    else "high"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    @server.tool()
    async def detect_memory_conflicts() -> dict[str, Any]:
        """
        Detect conflicts between memory rules.

        Returns:
            List of detected conflicts with resolution options
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            conflicts = await memory_manager.detect_conflicts()

            formatted_conflicts = []
            for conflict in conflicts:
                formatted_conflicts.append(
                    {
                        "type": conflict.conflict_type,
                        "description": conflict.description,
                        "confidence": conflict.confidence,
                        "rule1": {
                            "id": conflict.rule1.id,
                            "name": conflict.rule1.name,
                            "rule": conflict.rule1.rule,
                            "authority": conflict.rule1.authority.value,
                        },
                        "rule2": {
                            "id": conflict.rule2.id,
                            "name": conflict.rule2.name,
                            "rule": conflict.rule2.rule,
                            "authority": conflict.rule2.authority.value,
                        },
                        "resolution_options": conflict.resolution_options,
                    }
                )

            logger.info(f"Detected {len(conflicts)} memory conflicts")
            return {"conflicts_found": len(conflicts), "conflicts": formatted_conflicts}

        except Exception as e:
            logger.error(f"Failed to detect memory conflicts: {e}")
            return {"error": str(e)}

    @server.tool()
    async def list_memory_rules(
        category: str | None = None, authority: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        """
        List memory rules with optional filtering.

        Args:
            category: Filter by category (optional)
            authority: Filter by authority level (optional)
            limit: Maximum number of rules to return

        Returns:
            List of memory rules matching the criteria
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Validate optional parameters
            category_enum = None
            authority_enum = None

            if category:
                try:
                    category_enum = MemoryCategory(category)
                except ValueError:
                    return {"success": False, "error": f"Invalid category: {category}"}

            if authority:
                try:
                    authority_enum = AuthorityLevel(authority)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid authority: {authority}",
                    }

            # List memory rules
            rules = await memory_manager.list_memory_rules(
                category=category_enum, authority=authority_enum
            )

            # Apply limit
            rules = rules[:limit]

            # Format results
            formatted_rules = []
            for rule in rules:
                formatted_rules.append(
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "rule": rule.rule,
                        "category": rule.category.value,
                        "authority": rule.authority.value,
                        "scope": rule.scope,
                        "source": rule.source,
                        "created_at": rule.created_at.isoformat()
                        if rule.created_at
                        else None,
                        "updated_at": rule.updated_at.isoformat()
                        if rule.updated_at
                        else None,
                    }
                )

            logger.info(f"Listed {len(rules)} memory rules")
            return {
                "success": True,
                "total_returned": len(rules),
                "rules": formatted_rules,
            }

        except Exception as e:
            logger.error(f"Failed to list memory rules: {e}")
            return {"success": False, "error": str(e)}

    @server.tool()
    async def apply_memory_context(
        task_description: str, project_context: str | None = None
    ) -> dict[str, Any]:
        """
        Apply memory context to a task for behavioral adaptation.

        Analyzes the task description and project context against memory rules
        to provide relevant behavioral guidance and rule application.

        Args:
            task_description: Description of the task being performed
            project_context: Optional project-specific context

        Returns:
            Applicable memory rules and behavioral guidance
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Search for relevant memory rules based on task description
            relevant_rules = await memory_manager.search_memory_rules(
                query=task_description, limit=10
            )

            # Get all absolute authority rules (always apply)
            absolute_rules = await memory_manager.list_memory_rules(
                authority="absolute"
            )

            # Categorize applicable rules
            applicable_rules = {
                "absolute": [],
                "contextual": [],
                "preferences": []
            }

            # Add absolute rules
            for rule in absolute_rules:
                applicable_rules["absolute"].append({
                    "id": rule.id,
                    "name": rule.name,
                    "rule": rule.rule,
                    "category": rule.category.value,
                    "scope": rule.scope
                })

            # Add contextually relevant rules
            for rule, score in relevant_rules:
                if rule.authority == AuthorityLevel.DEFAULT and score > 0.7:
                    category_key = "preferences" if rule.category == MemoryCategory.PREFERENCE else "contextual"
                    applicable_rules[category_key].append({
                        "id": rule.id,
                        "name": rule.name,
                        "rule": rule.rule,
                        "category": rule.category.value,
                        "scope": rule.scope,
                        "relevance_score": score
                    })

            # Generate behavioral guidance
            guidance = []
            if applicable_rules["absolute"]:
                guidance.append("CRITICAL: The following rules must always be followed:")
                for rule in applicable_rules["absolute"][:3]:  # Top 3 absolute rules
                    guidance.append(f"- {rule['rule']}")

            if applicable_rules["contextual"]:
                guidance.append("\nFor this task, consider these contextual guidelines:")
                for rule in applicable_rules["contextual"][:5]:  # Top 5 contextual rules
                    guidance.append(f"- {rule['rule']}")

            if applicable_rules["preferences"]:
                guidance.append("\nUser preferences to keep in mind:")
                for rule in applicable_rules["preferences"][:3]:  # Top 3 preferences
                    guidance.append(f"- {rule['rule']}")

            logger.info(f"Applied memory context for task: {len(sum(applicable_rules.values(), []))} rules")
            return {
                "success": True,
                "task_description": task_description,
                "applicable_rules": applicable_rules,
                "behavioral_guidance": "\n".join(guidance),
                "total_applicable_rules": len(sum(applicable_rules.values(), [])),
                "memory_applied": True
            }

        except Exception as e:
            logger.error(f"Failed to apply memory context: {e}")
            return {"success": False, "error": str(e), "memory_applied": False}

    @server.tool()
    async def optimize_memory_tokens(max_tokens: int = 2000) -> dict[str, Any]:
        """
        Optimize memory usage to stay within token limits.

        Analyzes current memory usage and suggests optimizations to reduce
        token consumption while preserving important rules.

        Args:
            max_tokens: Maximum allowed token count for memory rules

        Returns:
            Optimization results and actions taken
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Get current memory statistics
            stats = await memory_manager.get_memory_stats()

            if stats.estimated_tokens <= max_tokens:
                return {
                    "optimization_needed": False,
                    "current_tokens": stats.estimated_tokens,
                    "token_limit": max_tokens,
                    "message": "Memory usage is within token limits"
                }

            # Perform optimization
            tokens_saved, actions = await memory_manager.optimize_memory(max_tokens)

            # Get updated statistics
            new_stats = await memory_manager.get_memory_stats()

            logger.info(f"Memory optimization completed: {tokens_saved} tokens saved")
            return {
                "optimization_needed": True,
                "optimization_completed": True,
                "tokens_before": stats.estimated_tokens,
                "tokens_after": new_stats.estimated_tokens,
                "tokens_saved": tokens_saved,
                "token_limit": max_tokens,
                "optimization_actions": actions,
                "message": f"Optimized memory usage: saved {tokens_saved} tokens"
            }

        except Exception as e:
            logger.error(f"Failed to optimize memory tokens: {e}")
            return {"optimization_completed": False, "error": str(e)}

    @server.tool()
    async def export_memory_profile() -> dict[str, Any]:
        """
        Export complete memory profile for backup or transfer.

        Creates a comprehensive export of all memory rules, statistics,
        and configuration for backup or transfer to another system.

        Returns:
            Complete memory profile data
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = MemoryManager(
                qdrant_client=client,
                naming_manager=naming_manager
            )

            # Get all memory rules
            all_rules = await memory_manager.list_memory_rules()

            # Get memory statistics
            stats = await memory_manager.get_memory_stats()

            # Detect any conflicts
            conflicts = await memory_manager.detect_conflicts()

            # Create exportable profile
            memory_profile = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_collection_name": memory_manager.memory_collection_name,
                "statistics": {
                    "total_rules": stats.total_rules,
                    "rules_by_category": {k.value: v for k, v in stats.rules_by_category.items()},
                    "rules_by_authority": {k.value: v for k, v in stats.rules_by_authority.items()},
                    "estimated_tokens": stats.estimated_tokens
                },
                "rules": [
                    {
                        "id": rule.id,
                        "category": rule.category.value,
                        "name": rule.name,
                        "rule": rule.rule,
                        "authority": rule.authority.value,
                        "scope": rule.scope,
                        "source": rule.source,
                        "conditions": rule.conditions,
                        "replaces": rule.replaces,
                        "created_at": rule.created_at.isoformat() if rule.created_at else None,
                        "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
                        "metadata": rule.metadata
                    }
                    for rule in all_rules
                ],
                "conflicts": [
                    {
                        "type": conflict.conflict_type,
                        "description": conflict.description,
                        "confidence": conflict.confidence,
                        "rule1_id": conflict.rule1.id,
                        "rule2_id": conflict.rule2.id,
                        "resolution_options": conflict.resolution_options
                    }
                    for conflict in conflicts
                ]
            }

            logger.info(f"Exported memory profile with {len(all_rules)} rules")
            return {
                "success": True,
                "export_size": len(str(memory_profile)),
                "memory_profile": memory_profile,
                "message": f"Exported {len(all_rules)} memory rules"
            }

        except Exception as e:
            logger.error(f"Failed to export memory profile: {e}")
            return {"success": False, "error": str(e)}
