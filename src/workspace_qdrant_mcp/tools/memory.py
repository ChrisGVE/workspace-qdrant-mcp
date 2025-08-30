"""
MCP tools for memory management.

This module provides MCP tools for interacting with the memory collection system,
including session initialization, conversational updates, and memory queries.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ..core.config import Config
from ..core.client import create_qdrant_client
from ..core.collection_naming import create_naming_manager
from ..core.memory import (
    MemoryManager,
    MemoryCategory,
    AuthorityLevel,
    MemoryRule,
    create_memory_manager,
    parse_conversational_memory_update
)

logger = logging.getLogger(__name__)


def register_memory_tools(server: FastMCP):
    """Register memory management tools with the MCP server."""

    @server.tool()
    async def initialize_memory_session() -> Dict[str, Any]:
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
            memory_manager = create_memory_manager(client, naming_manager)
            
            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()
            
            # Load all memory rules
            rules = await memory_manager.list_memory_rules()
            
            # Detect conflicts
            conflicts = await memory_manager.detect_conflicts(rules)
            
            # Get memory statistics
            stats = await memory_manager.get_memory_stats()
            
            # Format rules for Claude Code injection
            absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
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
                            "category": rule.category.value
                        }
                        for rule in absolute_rules
                    ],
                    "default": [
                        {
                            "name": rule.name,
                            "rule": rule.rule,
                            "scope": rule.scope,
                            "category": rule.category.value
                        }
                        for rule in default_rules
                    ]
                }
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
                        "resolution_options": conflict.resolution_options
                    }
                    for conflict in conflicts
                ]
                session_data["status"] = "conflicts_require_resolution"
            
            logger.info(f"Memory session initialized: {len(rules)} rules loaded, {len(conflicts)} conflicts")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to initialize memory session: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_rules": 0
            }

    @server.tool()
    async def add_memory_rule(
        category: str,
        name: str,
        rule: str,
        authority: str = "default",
        scope: Optional[List[str]] = None,
        source: str = "mcp_user"
    ) -> Dict[str, Any]:
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
            memory_manager = create_memory_manager(client, naming_manager)
            
            # Validate inputs
            try:
                category_enum = MemoryCategory(category)
                authority_enum = AuthorityLevel(authority)
            except ValueError as e:
                return {
                    "success": False,
                    "error": f"Invalid parameter: {e}"
                }
            
            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()
            
            # Add the rule
            rule_id = await memory_manager.add_memory_rule(
                category=category_enum,
                name=name,
                rule=rule,
                authority=authority_enum,
                scope=scope or [],
                source=source
            )
            
            logger.info(f"Added memory rule via MCP: {rule_id}")
            return {
                "success": True,
                "rule_id": rule_id,
                "message": f"Added memory rule '{name}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to add memory rule: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @server.tool()
    async def update_memory_from_conversation(message: str) -> Dict[str, Any]:
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
                    "message": "No memory update pattern detected"
                }
            
            # If update detected, add it as a memory rule
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = create_memory_manager(client, naming_manager)
            
            # Ensure memory collection exists
            await memory_manager.initialize_memory_collection()
            
            # Generate name from rule
            name_words = parsed["rule"].lower().split()[:2]
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            name_words = [w for w in name_words if w not in stop_words]
            name = "-".join(name_words) if name_words else "conversational-rule"
            
            # Add the rule
            rule_id = await memory_manager.add_memory_rule(
                category=parsed["category"],
                name=name,
                rule=parsed["rule"],
                authority=parsed["authority"],
                source=parsed["source"]
            )
            
            logger.info(f"Added conversational memory rule: {rule_id}")
            return {
                "detected": True,
                "rule_added": True,
                "rule_id": rule_id,
                "category": parsed["category"].value,
                "rule": parsed["rule"],
                "authority": parsed["authority"].value,
                "message": f"Added memory rule from conversation: '{parsed['rule']}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to process conversational memory update: {e}")
            return {
                "detected": True,
                "rule_added": False,
                "error": str(e)
            }

    @server.tool()
    async def search_memory_rules(
        query: str,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
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
            memory_manager = create_memory_manager(client, naming_manager)
            
            # Validate optional parameters
            category_enum = None
            authority_enum = None
            
            if category:
                try:
                    category_enum = MemoryCategory(category)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid category: {category}"
                    }
            
            if authority:
                try:
                    authority_enum = AuthorityLevel(authority)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid authority: {authority}"
                    }
            
            # Search memory rules
            results = await memory_manager.search_memory_rules(
                query=query,
                limit=limit,
                category=category_enum,
                authority=authority_enum
            )
            
            # Format results
            formatted_results = []
            for rule, score in results:
                formatted_results.append({
                    "id": rule.id,
                    "name": rule.name,
                    "rule": rule.rule,
                    "category": rule.category.value,
                    "authority": rule.authority.value,
                    "scope": rule.scope,
                    "relevance_score": score,
                    "created_at": rule.created_at.isoformat() if rule.created_at else None
                })
            
            logger.info(f"Memory search returned {len(results)} results for query: {query}")
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to search memory rules: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @server.tool()
    async def get_memory_stats() -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Statistics about memory rules and token usage
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = create_memory_manager(client, naming_manager)
            
            stats = await memory_manager.get_memory_stats()
            
            return {
                "total_rules": stats.total_rules,
                "rules_by_category": {k.value: v for k, v in stats.rules_by_category.items()},
                "rules_by_authority": {k.value: v for k, v in stats.rules_by_authority.items()},
                "estimated_tokens": stats.estimated_tokens,
                "last_optimization": stats.last_optimization.isoformat() if stats.last_optimization else None,
                "token_status": (
                    "low" if stats.estimated_tokens < 1000
                    else "moderate" if stats.estimated_tokens < 2000
                    else "high"
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "error": str(e)
            }

    @server.tool()
    async def detect_memory_conflicts() -> Dict[str, Any]:
        """
        Detect conflicts between memory rules.
        
        Returns:
            List of detected conflicts with resolution options
        """
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            naming_manager = create_naming_manager(config.workspace.global_collections)
            memory_manager = create_memory_manager(client, naming_manager)
            
            conflicts = await memory_manager.detect_conflicts()
            
            formatted_conflicts = []
            for conflict in conflicts:
                formatted_conflicts.append({
                    "type": conflict.conflict_type,
                    "description": conflict.description,
                    "confidence": conflict.confidence,
                    "rule1": {
                        "id": conflict.rule1.id,
                        "name": conflict.rule1.name,
                        "rule": conflict.rule1.rule,
                        "authority": conflict.rule1.authority.value
                    },
                    "rule2": {
                        "id": conflict.rule2.id,
                        "name": conflict.rule2.name,
                        "rule": conflict.rule2.rule,
                        "authority": conflict.rule2.authority.value
                    },
                    "resolution_options": conflict.resolution_options
                })
            
            logger.info(f"Detected {len(conflicts)} memory conflicts")
            return {
                "conflicts_found": len(conflicts),
                "conflicts": formatted_conflicts
            }
            
        except Exception as e:
            logger.error(f"Failed to detect memory conflicts: {e}")
            return {
                "error": str(e)
            }

    @server.tool()
    async def list_memory_rules(
        category: Optional[str] = None,
        authority: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
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
            memory_manager = create_memory_manager(client, naming_manager)
            
            # Validate optional parameters
            category_enum = None
            authority_enum = None
            
            if category:
                try:
                    category_enum = MemoryCategory(category)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid category: {category}"
                    }
            
            if authority:
                try:
                    authority_enum = AuthorityLevel(authority)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid authority: {authority}"
                    }
            
            # List memory rules
            rules = await memory_manager.list_memory_rules(
                category=category_enum,
                authority=authority_enum
            )
            
            # Apply limit
            rules = rules[:limit]
            
            # Format results
            formatted_rules = []
            for rule in rules:
                formatted_rules.append({
                    "id": rule.id,
                    "name": rule.name,
                    "rule": rule.rule,
                    "category": rule.category.value,
                    "authority": rule.authority.value,
                    "scope": rule.scope,
                    "source": rule.source,
                    "created_at": rule.created_at.isoformat() if rule.created_at else None,
                    "updated_at": rule.updated_at.isoformat() if rule.updated_at else None
                })
            
            logger.info(f"Listed {len(rules)} memory rules")
            return {
                "success": True,
                "total_returned": len(rules),
                "rules": formatted_rules
            }
            
        except Exception as e:
            logger.error(f"Failed to list memory rules: {e}")
            return {
                "success": False,
                "error": str(e)
            }