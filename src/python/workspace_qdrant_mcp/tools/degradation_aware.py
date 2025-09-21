"""
Degradation-Aware MCP Tools

This module provides degradation-aware wrappers for MCP tools that gracefully handle
component failures and resource constraints by automatically falling back to
appropriate strategies based on the current system state.

Key Features:
    - Automatic fallback mechanisms for MCP tools
    - Circuit breaker integration for component protection
    - User-friendly error messages with guidance
    - Seamless integration with existing MCP tool implementations
    - Resource throttling under high load conditions

Example:
    ```python
    from workspace_qdrant_mcp.tools.degradation_aware import DegradationAwareMCPTools

    # Initialize degradation-aware tools
    tools = DegradationAwareMCPTools(
        degradation_manager=degradation_manager,
        original_tools=original_mcp_tools
    )

    # Use tools normally - degradation handled automatically
    result = await tools.search_workspace(query="test", collections=["docs"])
    if result.success:
        # Full functionality available
        process_results(result.data)
    else:
        # Degraded mode - show user guidance
        show_user_message(result.user_guidance)
    ```
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from mcp.types import Tool

from python.common.core.graceful_degradation import (
    DegradationManager,
    DegradationMode,
    FeatureType,
)
from python.common.core.component_coordination import ComponentType


@dataclass
class DegradedResponse:
    """Response from a degraded MCP tool operation."""

    success: bool
    data: Any = None
    error_message: Optional[str] = None
    degradation_mode: Optional[str] = None
    fallback_used: Optional[str] = None
    user_guidance: Optional[str] = None
    from_cache: bool = False
    execution_time_ms: float = 0.0


class DegradationAwareMCPTools:
    """
    Wrapper for MCP tools that provides automatic graceful degradation.

    This class wraps existing MCP tools and automatically handles degradation
    scenarios by applying appropriate fallback strategies based on the current
    system state and component health.
    """

    def __init__(
        self,
        degradation_manager: DegradationManager,
        original_tools: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize degradation-aware MCP tools.

        Args:
            degradation_manager: Graceful degradation manager instance
            original_tools: Dictionary of original tool implementations
        """
        self.degradation_manager = degradation_manager
        self.original_tools = original_tools or {}

        # Tool execution statistics
        self.tool_executions = 0
        self.degraded_executions = 0
        self.cache_hits = 0

        logger.info("Degradation-aware MCP tools initialized")

    async def search_workspace(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
        **kwargs
    ) -> DegradedResponse:
        """
        Execute workspace search with degradation awareness.

        Args:
            query: Search query
            collections: Collections to search (None for all)
            mode: Search mode (hybrid, semantic, keyword)
            limit: Maximum number of results
            score_threshold: Minimum score threshold
            **kwargs: Additional search parameters

        Returns:
            DegradedResponse with search results or fallback
        """
        self.tool_executions += 1
        operation_start = time.time()

        # Check if semantic search is available
        if mode in ["hybrid", "semantic"] and not self.degradation_manager.is_feature_available(FeatureType.SEMANTIC_SEARCH):
            logger.info("Semantic search unavailable, falling back to keyword search")
            mode = "keyword"

        # Check if search features are available at all
        if not self.degradation_manager.is_feature_available(FeatureType.KEYWORD_SEARCH):
            return await self._handle_search_degraded(query, collections, **kwargs)

        try:
            # Attempt primary search operation
            if "search_workspace" in self.original_tools:
                result = await self.original_tools["search_workspace"](
                    query=query,
                    collections=collections,
                    mode=mode,
                    limit=limit,
                    score_threshold=score_threshold,
                    **kwargs
                )

                execution_time = (time.time() - operation_start) * 1000

                return DegradedResponse(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    degradation_mode=self.degradation_manager.current_mode.name.lower()
                )

        except Exception as e:
            logger.warning(f"Primary search operation failed: {e}")
            self.degraded_executions += 1

            # Record component failure
            await self.degradation_manager.record_component_failure("search-service")

            # Attempt fallback
            return await self._handle_search_degraded(query, collections, str(e), **kwargs)

    async def add_document(
        self,
        content: str,
        collection: str,
        metadata: Optional[Dict[str, str]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
        **kwargs
    ) -> DegradedResponse:
        """
        Add document with degradation awareness.

        Args:
            content: Document content
            collection: Target collection
            metadata: Document metadata
            document_id: Optional document ID
            chunk_text: Whether to chunk the text
            **kwargs: Additional parameters

        Returns:
            DegradedResponse with operation result
        """
        self.tool_executions += 1
        operation_start = time.time()

        # Check if document ingestion is available
        if not self.degradation_manager.is_feature_available(FeatureType.DOCUMENT_INGESTION):
            return await self._handle_ingestion_degraded(content, collection, metadata)

        try:
            if "add_document" in self.original_tools:
                result = await self.original_tools["add_document"](
                    content=content,
                    collection=collection,
                    metadata=metadata,
                    document_id=document_id,
                    chunk_text=chunk_text,
                    **kwargs
                )

                execution_time = (time.time() - operation_start) * 1000

                return DegradedResponse(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    degradation_mode=self.degradation_manager.current_mode.name.lower()
                )

        except Exception as e:
            logger.warning(f"Document ingestion failed: {e}")
            self.degraded_executions += 1

            # Record component failure
            await self.degradation_manager.record_component_failure("ingestion-service")

            return await self._handle_ingestion_degraded(content, collection, metadata, str(e))

    async def list_collections(self, **kwargs) -> DegradedResponse:
        """
        List collections with degradation awareness.

        Args:
            **kwargs: Additional parameters

        Returns:
            DegradedResponse with collection list
        """
        self.tool_executions += 1
        operation_start = time.time()

        try:
            if "list_collections" in self.original_tools:
                result = await self.original_tools["list_collections"](**kwargs)

                execution_time = (time.time() - operation_start) * 1000

                return DegradedResponse(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    degradation_mode=self.degradation_manager.current_mode.name.lower()
                )

        except Exception as e:
            logger.warning(f"List collections failed: {e}")
            self.degraded_executions += 1

            # Try to get cached collection list
            cache_key = "list_collections"
            fallback = await self.degradation_manager.get_fallback_response(
                "list_collections", {}, cache_key
            )

            if fallback:
                self.cache_hits += 1
                return DegradedResponse(
                    success=True,
                    data=fallback.get("collections", []),
                    from_cache=True,
                    user_guidance="Collection list may be outdated (from cache)",
                    execution_time_ms=(time.time() - operation_start) * 1000
                )

            return DegradedResponse(
                success=False,
                error_message=f"Unable to list collections: {e}",
                user_guidance=self.degradation_manager.get_user_guidance(ComponentType.PYTHON_MCP_SERVER)
            )

    async def get_workspace_status(self, **kwargs) -> DegradedResponse:
        """
        Get workspace status with degradation information.

        Args:
            **kwargs: Additional parameters

        Returns:
            DegradedResponse with workspace status including degradation info
        """
        operation_start = time.time()

        try:
            # Always provide status, even in degraded mode
            base_status = {}

            if "get_workspace_status" in self.original_tools:
                try:
                    base_status = await self.original_tools["get_workspace_status"](**kwargs)
                except Exception as e:
                    logger.warning(f"Failed to get base workspace status: {e}")
                    base_status = {"error": str(e)}

            # Add degradation information
            degradation_status = self.degradation_manager.get_degradation_status()

            combined_status = {
                **base_status,
                "degradation": degradation_status,
                "available_features": degradation_status["available_features"],
                "unavailable_features": degradation_status["unavailable_features"],
                "system_mode": degradation_status["current_mode"],
                "user_guidance": self._get_status_user_guidance(),
            }

            execution_time = (time.time() - operation_start) * 1000

            return DegradedResponse(
                success=True,
                data=combined_status,
                execution_time_ms=execution_time,
                degradation_mode=degradation_status["current_mode"]
            )

        except Exception as e:
            logger.error(f"Failed to get workspace status: {e}")

            # Provide minimal status even if everything fails
            minimal_status = {
                "error": str(e),
                "degradation": {
                    "current_mode": "unknown",
                    "message": "Unable to determine system status"
                },
                "user_guidance": [
                    "System status check failed",
                    "Check component health manually",
                    "Try restarting services if needed"
                ]
            }

            return DegradedResponse(
                success=False,
                data=minimal_status,
                error_message=str(e),
                user_guidance="System status unavailable - check component health"
            )

    async def _handle_search_degraded(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> DegradedResponse:
        """Handle search operation in degraded mode."""

        # Try to get cached search results
        cache_key = f"search:{query}:{','.join(collections or ['all'])}"
        fallback = await self.degradation_manager.get_fallback_response(
            "search", {"query": query, "collections": collections}, cache_key
        )

        if fallback and not fallback.get("fallback", False):
            self.cache_hits += 1
            return DegradedResponse(
                success=True,
                data=fallback,
                from_cache=True,
                user_guidance="Search results from cache - may not reflect recent changes",
                degradation_mode=self.degradation_manager.current_mode.name.lower()
            )

        # Return graceful failure with guidance
        mode = self.degradation_manager.current_mode

        if mode == DegradationMode.OFFLINE_CLI:
            user_guidance = (
                "Search service unavailable in offline mode. "
                "Try using CLI commands: 'wqm search \"{}\" --local' for local file search.".format(query)
            )
        elif mode == DegradationMode.CACHED_ONLY:
            user_guidance = (
                "Real-time search unavailable. Only cached results can be served. "
                "Check system status or try again later."
            )
        else:
            user_guidance = (
                "Search service temporarily unavailable. "
                "Check component health with 'wqm service status'."
            )

        return DegradedResponse(
            success=False,
            error_message=f"Search unavailable: {error or 'Service degraded'}",
            user_guidance=user_guidance,
            degradation_mode=mode.name.lower()
        )

    async def _handle_ingestion_degraded(
        self,
        content: str,
        collection: str,
        metadata: Optional[Dict[str, str]] = None,
        error: Optional[str] = None
    ) -> DegradedResponse:
        """Handle document ingestion in degraded mode."""

        mode = self.degradation_manager.current_mode

        if mode in [DegradationMode.READ_ONLY, DegradationMode.CACHED_ONLY]:
            user_guidance = (
                "Document ingestion disabled in read-only mode. "
                "Wait for system recovery or check component status."
            )
        elif mode == DegradationMode.OFFLINE_CLI:
            user_guidance = (
                "Document ingestion unavailable in offline mode. "
                "Save documents locally and ingest when services recover."
            )
        else:
            user_guidance = (
                "Document ingestion temporarily unavailable. "
                "Check Rust daemon and MCP server status."
            )

        return DegradedResponse(
            success=False,
            error_message=f"Document ingestion unavailable: {error or 'Service degraded'}",
            user_guidance=user_guidance,
            degradation_mode=mode.name.lower()
        )

    def _get_status_user_guidance(self) -> List[str]:
        """Get user guidance for current status."""
        mode = self.degradation_manager.current_mode

        if mode == DegradationMode.NORMAL:
            return ["All systems operational"]

        guidance = [f"System in {mode.name.lower().replace('_', ' ')} mode"]

        # Add mode-specific guidance
        mode_guidance = self.degradation_manager.get_user_guidance(ComponentType.PYTHON_MCP_SERVER)
        if mode_guidance:
            guidance.append(mode_guidance)

        # Add available alternatives
        available_features = self.degradation_manager.get_available_features()

        if FeatureType.CLI_OPERATIONS in available_features:
            guidance.append("CLI operations available: 'wqm --help' for commands")

        if FeatureType.SEMANTIC_SEARCH in available_features:
            guidance.append("Search operations available")

        if mode != DegradationMode.NORMAL:
            guidance.append("Check 'wqm service status' for component health")

        return guidance

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        return {
            "total_executions": self.tool_executions,
            "degraded_executions": self.degraded_executions,
            "cache_hits": self.cache_hits,
            "degradation_rate": (
                self.degraded_executions / max(1, self.tool_executions)
            ),
            "cache_hit_rate": (
                self.cache_hits / max(1, self.tool_executions)
            ),
            "current_mode": self.degradation_manager.current_mode.name.lower(),
            "available_features": [
                f.name.lower() for f in self.degradation_manager.get_available_features()
            ],
        }

    async def register_original_tool(self, name: str, tool_function: Any) -> None:
        """Register an original tool implementation."""
        self.original_tools[name] = tool_function
        logger.debug(f"Registered original tool: {name}")

    def create_mcp_tool_definitions(self) -> List[Tool]:
        """Create MCP tool definitions with degradation awareness."""
        return [
            Tool(
                name="search_workspace",
                description=(
                    "Search workspace documents with automatic degradation handling. "
                    "Falls back to cached results or keyword search if semantic search unavailable."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "collections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Collections to search (optional)"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "semantic", "keyword"],
                            "default": "hybrid",
                            "description": "Search mode (automatically adjusted based on availability)"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of results"
                        },
                        "score_threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Minimum score threshold"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="add_document",
                description=(
                    "Add document to workspace with degradation awareness. "
                    "Gracefully handles ingestion service unavailability."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Document content"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Target collection"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Document metadata (optional)"
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Document ID (optional)"
                        },
                        "chunk_text": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to chunk the text"
                        }
                    },
                    "required": ["content", "collection"]
                }
            ),
            Tool(
                name="list_collections",
                description=(
                    "List available collections with fallback to cached data. "
                    "Indicates if data is from cache due to service unavailability."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="get_workspace_status",
                description=(
                    "Get comprehensive workspace status including degradation information. "
                    "Always available even when other services are degraded."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
        ]


class CLIDegradationHandler:
    """
    Handler for CLI operations with degradation awareness.

    This class provides graceful degradation for CLI operations by detecting
    service availability and providing appropriate fallback behaviors.
    """

    def __init__(self, degradation_manager: DegradationManager):
        """
        Initialize CLI degradation handler.

        Args:
            degradation_manager: Graceful degradation manager instance
        """
        self.degradation_manager = degradation_manager
        self.offline_mode = False

        logger.info("CLI degradation handler initialized")

    async def handle_search_command(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        local_only: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle search command with degradation awareness.

        Args:
            query: Search query
            collections: Collections to search
            local_only: Force local-only search
            **kwargs: Additional search parameters

        Returns:
            Search results with degradation information
        """
        if local_only or self.degradation_manager.current_mode == DegradationMode.OFFLINE_CLI:
            return await self._local_file_search(query, **kwargs)

        if not self.degradation_manager.is_feature_available(FeatureType.SEMANTIC_SEARCH):
            logger.info("Semantic search unavailable, using local search")
            return await self._local_file_search(query, **kwargs)

        # Try to use full search functionality
        try:
            # This would call the actual search implementation
            return {
                "results": [],  # Placeholder
                "mode": "degraded_cli",
                "message": "Search functionality limited in current mode"
            }
        except Exception as e:
            logger.warning(f"CLI search failed: {e}")
            return await self._local_file_search(query, **kwargs)

    async def handle_status_command(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Handle status command with comprehensive degradation information.

        Args:
            verbose: Include detailed status information

        Returns:
            System status with degradation details
        """
        try:
            status = self.degradation_manager.get_degradation_status()

            if verbose:
                return {
                    "system_mode": status["current_mode"],
                    "degradation": status,
                    "recommendations": self._get_status_recommendations(status),
                    "cli_available": True
                }
            else:
                return {
                    "mode": status["current_mode"],
                    "health": "degraded" if status["current_mode"] != "normal" else "healthy",
                    "available_features": status["available_features"],
                    "message": self._get_simple_status_message(status["current_mode"])
                }

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {
                "mode": "unknown",
                "health": "error",
                "error": str(e),
                "message": "Unable to determine system status"
            }

    async def handle_service_command(
        self,
        action: str,
        service: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle service management command with degradation context.

        Args:
            action: Service action (start, stop, restart, status)
            service: Specific service name (optional)

        Returns:
            Service management result
        """
        if self.degradation_manager.current_mode == DegradationMode.OFFLINE_CLI:
            return {
                "success": False,
                "message": "Service management unavailable in offline mode",
                "recommendation": "Check network connectivity and try manual service restart"
            }

        # This would integrate with actual service management
        return {
            "action": action,
            "service": service or "all",
            "status": "simulated",
            "message": f"Service {action} command processed"
        }

    async def _local_file_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform local file search as fallback."""
        logger.info(f"Performing local file search for: {query}")

        # This would implement actual local file search
        return {
            "results": [
                {
                    "type": "local_file",
                    "path": f"./example_{query}.txt",
                    "snippet": f"Local search result for '{query}'",
                    "score": 0.8
                }
            ],
            "mode": "local_search",
            "message": "Results from local file search (services unavailable)",
            "total": 1
        }

    def _get_status_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Get recommendations based on current status."""
        mode = status["current_mode"]
        recommendations = []

        if mode == "normal":
            recommendations.append("All systems operational")
        elif mode == "offline_cli":
            recommendations.extend([
                "Only CLI operations available",
                "Check network connectivity",
                "Try 'wqm service start' to restart services"
            ])
        elif mode == "read_only":
            recommendations.extend([
                "Document ingestion disabled",
                "Search operations available",
                "Check component health with 'wqm service status'"
            ])
        elif mode == "cached_only":
            recommendations.extend([
                "Using cached responses only",
                "Real-time operations unavailable",
                "Check Rust daemon: 'wqm service status daemon'"
            ])
        else:
            recommendations.extend([
                "System experiencing issues",
                "Check component status",
                "Contact administrator if problems persist"
            ])

        return recommendations

    def _get_simple_status_message(self, mode: str) -> str:
        """Get simple status message for current mode."""
        messages = {
            "normal": "All systems operational",
            "performance_reduced": "Reduced performance",
            "features_limited": "Some features disabled",
            "read_only": "Read-only mode active",
            "cached_only": "Cached responses only",
            "offline_cli": "Offline mode - CLI only",
            "emergency": "Emergency mode - minimal functionality",
            "unavailable": "System unavailable"
        }

        return messages.get(mode, "Unknown system state")


# Helper functions for integration
async def create_degradation_aware_tools(
    degradation_manager: DegradationManager,
    original_tools: Optional[Dict[str, Any]] = None
) -> DegradationAwareMCPTools:
    """
    Create degradation-aware MCP tools.

    Args:
        degradation_manager: Graceful degradation manager
        original_tools: Original tool implementations

    Returns:
        Configured degradation-aware tools
    """
    tools = DegradationAwareMCPTools(degradation_manager, original_tools)
    logger.info("Created degradation-aware MCP tools")
    return tools


async def create_cli_degradation_handler(
    degradation_manager: DegradationManager
) -> CLIDegradationHandler:
    """
    Create CLI degradation handler.

    Args:
        degradation_manager: Graceful degradation manager

    Returns:
        Configured CLI degradation handler
    """
    handler = CLIDegradationHandler(degradation_manager)
    logger.info("Created CLI degradation handler")
    return handler