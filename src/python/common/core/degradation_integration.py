"""
Graceful Degradation Integration for MCP Tools and CLI Operations.

This module provides integration points for graceful degradation with existing
MCP server tools and CLI operations, ensuring seamless fallback behavior
when components are unavailable or degraded.

Key Features:
    - MCP tool wrappers with degradation-aware behavior
    - CLI operation fallbacks for offline modes
    - Automatic feature detection and graceful degradation
    - User guidance for degraded operations
    - Status reporting with degradation context

Integration Points:
    - MCP server tools (search, ingestion, health)
    - CLI commands (file operations, status checks)
    - gRPC client operations with circuit breaker patterns
    - Health monitoring integration
    - Context injection with fallback modes

Example:
    ```python
    from workspace_qdrant_mcp.core.degradation_integration import DegradationAwareMCPServer

    # Create degradation-aware MCP server
    mcp_server = DegradationAwareMCPServer(
        degradation_manager=degradation_manager,
        base_server=existing_mcp_server
    )

    # Search with automatic fallback
    result = await mcp_server.search_with_degradation(query, options)
    if result.get("fallback"):
        # Handle degraded response
        pass
    ```
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from .graceful_degradation import DegradationManager, DegradationMode, FeatureType


@dataclass
class DegradedOperation:
    """Result of a degraded operation with fallback information."""

    success: bool
    result: Any
    is_fallback: bool = False
    degradation_mode: DegradationMode | None = None
    available_features: list[FeatureType] = None
    user_message: str | None = None
    suggested_actions: list[str] = None

    def __post_init__(self):
        if self.available_features is None:
            self.available_features = []
        if self.suggested_actions is None:
            self.suggested_actions = []


class DegradationAwareMCPServer:
    """
    MCP Server wrapper that provides graceful degradation for all operations.

    This class wraps existing MCP server functionality and adds automatic
    degradation detection and fallback behavior when components are unavailable.
    """

    def __init__(
        self,
        degradation_manager: DegradationManager,
        base_server: Any = None,
        config: dict[str, Any] | None = None
    ):
        """
        Initialize degradation-aware MCP server.

        Args:
            degradation_manager: Degradation manager instance
            base_server: Base MCP server instance to wrap
            config: Configuration options
        """
        self.degradation_manager = degradation_manager
        self.base_server = base_server
        self.config = config or {}

        # Operation counters
        self.successful_operations = 0
        self.fallback_operations = 0
        self.failed_operations = 0

        logger.info("Degradation-aware MCP server initialized")

    async def search_with_degradation(
        self,
        query: str,
        options: dict[str, Any] | None = None
    ) -> DegradedOperation:
        """
        Perform search with automatic degradation handling.

        Args:
            query: Search query
            options: Search options

        Returns:
            DegradedOperation with search results and degradation info
        """
        options = options or {}

        try:
            # Check if search features are available
            semantic_available = self.degradation_manager.is_feature_available(
                FeatureType.SEMANTIC_SEARCH
            )
            keyword_available = self.degradation_manager.is_feature_available(
                FeatureType.KEYWORD_SEARCH
            )
            hybrid_available = self.degradation_manager.is_feature_available(
                FeatureType.HYBRID_SEARCH
            )

            # Determine best available search method
            if hybrid_available and self.base_server:
                # Try full hybrid search
                if self._should_attempt_operation("search_hybrid"):
                    result = await self._attempt_hybrid_search(query, options)
                    if result:
                        await self.degradation_manager.record_component_success("python_mcp_server")
                        self.successful_operations += 1
                        return DegradedOperation(
                            success=True,
                            result=result,
                            is_fallback=False,
                            degradation_mode=self.degradation_manager.current_mode
                        )

            # Fall back to semantic search only
            if semantic_available and self.base_server:
                if self._should_attempt_operation("search_semantic"):
                    result = await self._attempt_semantic_search(query, options)
                    if result:
                        await self.degradation_manager.record_component_success("python_mcp_server")
                        self.successful_operations += 1
                        return DegradedOperation(
                            success=True,
                            result=result,
                            is_fallback=True,
                            degradation_mode=self.degradation_manager.current_mode,
                            user_message="Using semantic search only due to system limitations",
                            suggested_actions=["Check system status for full search capabilities"]
                        )

            # Fall back to keyword search only
            if keyword_available and self.base_server:
                if self._should_attempt_operation("search_keyword"):
                    result = await self._attempt_keyword_search(query, options)
                    if result:
                        await self.degradation_manager.record_component_success("python_mcp_server")
                        self.successful_operations += 1
                        return DegradedOperation(
                            success=True,
                            result=result,
                            is_fallback=True,
                            degradation_mode=self.degradation_manager.current_mode,
                            user_message="Using keyword search only due to system limitations",
                            suggested_actions=["Check Rust daemon status for semantic search"]
                        )

            # Use cached response if available
            cached_result = await self.degradation_manager.get_fallback_response(
                "search",
                {"query": query, "options": options},
                cache_key=f"search:{hash(query)}"
            )

            if cached_result:
                self.fallback_operations += 1
                return DegradedOperation(
                    success=True,
                    result=cached_result,
                    is_fallback=True,
                    degradation_mode=self.degradation_manager.current_mode,
                    user_message="Serving cached search results due to service unavailability",
                    suggested_actions=[
                        "Results may be outdated",
                        "Check system status and restart services if needed"
                    ]
                )

            # No search capabilities available
            self.failed_operations += 1
            return DegradedOperation(
                success=False,
                result={"error": "Search services are currently unavailable"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode,
                user_message="Search services are currently unavailable",
                suggested_actions=[
                    "Check if Rust daemon and MCP server are running",
                    "Verify Qdrant database connectivity",
                    "Review system logs for error details"
                ]
            )

        except Exception as e:
            logger.error(f"Error in search_with_degradation: {e}")
            await self.degradation_manager.record_component_failure("python_mcp_server")
            self.failed_operations += 1

            return DegradedOperation(
                success=False,
                result={"error": f"Search operation failed: {str(e)}"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode,
                user_message="Search operation encountered an error",
                suggested_actions=[
                    "Try again in a few moments",
                    "Check system logs for detailed error information"
                ]
            )

    async def ingest_document_with_degradation(
        self,
        file_path: str,
        options: dict[str, Any] | None = None
    ) -> DegradedOperation:
        """
        Perform document ingestion with degradation handling.

        Args:
            file_path: Path to document to ingest
            options: Ingestion options

        Returns:
            DegradedOperation with ingestion results
        """
        options = options or {}

        try:
            # Check if ingestion is available
            if not self.degradation_manager.is_feature_available(FeatureType.DOCUMENT_INGESTION):
                return DegradedOperation(
                    success=False,
                    result={"error": "Document ingestion is currently disabled"},
                    is_fallback=True,
                    degradation_mode=self.degradation_manager.current_mode,
                    user_message="Document ingestion is temporarily unavailable",
                    suggested_actions=[
                        "System is in read-only mode",
                        "Wait for services to recover",
                        "Check system health status"
                    ]
                )

            # Check if this is a high load situation requiring throttling
            if self.degradation_manager.should_throttle_request(options.get("priority", 5)):
                return DegradedOperation(
                    success=False,
                    result={"error": "Request throttled due to high system load"},
                    is_fallback=True,
                    degradation_mode=self.degradation_manager.current_mode,
                    user_message="Request was throttled due to high system load",
                    suggested_actions=[
                        "Try again in a few moments",
                        "Consider reducing concurrent operations",
                        "Use lower priority for non-urgent requests"
                    ]
                )

            # Attempt ingestion through base server
            if self.base_server and self._should_attempt_operation("ingest_document"):
                result = await self._attempt_document_ingestion(file_path, options)
                if result:
                    await self.degradation_manager.record_component_success("python_mcp_server")
                    self.successful_operations += 1
                    return DegradedOperation(
                        success=True,
                        result=result,
                        is_fallback=False,
                        degradation_mode=self.degradation_manager.current_mode
                    )

            # Ingestion failed
            await self.degradation_manager.record_component_failure("python_mcp_server")
            self.failed_operations += 1

            return DegradedOperation(
                success=False,
                result={"error": "Document ingestion service is unavailable"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode,
                user_message="Document ingestion service is currently unavailable",
                suggested_actions=[
                    "Check if Rust daemon is running and responsive",
                    "Verify file permissions and accessibility",
                    "Review ingestion queue status"
                ]
            )

        except Exception as e:
            logger.error(f"Error in ingest_document_with_degradation: {e}")
            await self.degradation_manager.record_component_failure("python_mcp_server")
            self.failed_operations += 1

            return DegradedOperation(
                success=False,
                result={"error": f"Document ingestion failed: {str(e)}"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode,
                user_message="Document ingestion encountered an error",
                suggested_actions=[
                    "Check file format and accessibility",
                    "Verify sufficient disk space",
                    "Review system logs for details"
                ]
            )

    async def get_health_status_with_degradation(self) -> DegradedOperation:
        """
        Get system health status with degradation context.

        Returns:
            DegradedOperation with health status
        """
        try:
            # Always available operation - get degradation status
            degradation_status = self.degradation_manager.get_degradation_status()

            # Get component health if available
            component_health = {}
            if self.degradation_manager.lifecycle_manager:
                component_status = await self.degradation_manager.lifecycle_manager.get_component_status()
                component_health = component_status.get("components", {})

            # Combine health information
            health_result = {
                "degradation_status": degradation_status,
                "component_health": component_health,
                "operation_statistics": {
                    "successful_operations": self.successful_operations,
                    "fallback_operations": self.fallback_operations,
                    "failed_operations": self.failed_operations,
                    "total_operations": (
                        self.successful_operations +
                        self.fallback_operations +
                        self.failed_operations
                    )
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.successful_operations += 1

            return DegradedOperation(
                success=True,
                result=health_result,
                is_fallback=False,
                degradation_mode=self.degradation_manager.current_mode,
                available_features=list(self.degradation_manager.get_available_features())
            )

        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            self.failed_operations += 1

            return DegradedOperation(
                success=False,
                result={"error": f"Health status check failed: {str(e)}"},
                is_fallback=True,
                degradation_mode=DegradationMode.UNKNOWN
            )

    def _should_attempt_operation(self, operation_type: str) -> bool:
        """Check if operation should be attempted based on circuit breaker state."""
        # Map operation types to component IDs
        component_mapping = {
            "search_hybrid": "rust_daemon-default",
            "search_semantic": "python_mcp_server-default",
            "search_keyword": "rust_daemon-default",
            "ingest_document": "rust_daemon-default"
        }

        component_id = component_mapping.get(operation_type, "python_mcp_server-default")
        return self.degradation_manager.is_component_available(component_id)

    async def _attempt_hybrid_search(
        self,
        query: str,
        options: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Attempt hybrid search through base server."""
        try:
            if hasattr(self.base_server, 'hybrid_search'):
                return await self.base_server.hybrid_search(query, **options)
            elif hasattr(self.base_server, 'search'):
                # Fallback to generic search
                return await self.base_server.search(query, **options)
            return None
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return None

    async def _attempt_semantic_search(
        self,
        query: str,
        options: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Attempt semantic search through base server."""
        try:
            if hasattr(self.base_server, 'semantic_search'):
                return await self.base_server.semantic_search(query, **options)
            elif hasattr(self.base_server, 'search'):
                # Use generic search with semantic flag
                options['search_type'] = 'semantic'
                return await self.base_server.search(query, **options)
            return None
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return None

    async def _attempt_keyword_search(
        self,
        query: str,
        options: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Attempt keyword search through base server."""
        try:
            if hasattr(self.base_server, 'keyword_search'):
                return await self.base_server.keyword_search(query, **options)
            elif hasattr(self.base_server, 'search'):
                # Use generic search with keyword flag
                options['search_type'] = 'keyword'
                return await self.base_server.search(query, **options)
            return None
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return None

    async def _attempt_document_ingestion(
        self,
        file_path: str,
        options: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Attempt document ingestion through base server."""
        try:
            if hasattr(self.base_server, 'ingest_document'):
                return await self.base_server.ingest_document(file_path, **options)
            elif hasattr(self.base_server, 'add_document'):
                return await self.base_server.add_document(file_path, **options)
            return None
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return None


class DegradationAwareCLI:
    """
    CLI wrapper that provides graceful degradation for command-line operations.

    This class wraps CLI functionality and provides fallback behavior
    when daemon services are unavailable.
    """

    def __init__(
        self,
        degradation_manager: DegradationManager,
        base_cli: Any = None
    ):
        """
        Initialize degradation-aware CLI.

        Args:
            degradation_manager: Degradation manager instance
            base_cli: Base CLI instance to wrap
        """
        self.degradation_manager = degradation_manager
        self.base_cli = base_cli

        logger.info("Degradation-aware CLI initialized")

    async def execute_command_with_degradation(
        self,
        command: str,
        args: list[str],
        options: dict[str, Any] | None = None
    ) -> DegradedOperation:
        """
        Execute CLI command with degradation handling.

        Args:
            command: Command name
            args: Command arguments
            options: Command options

        Returns:
            DegradedOperation with command results
        """
        options = options or {}

        try:
            # Check if CLI operations are available
            if not self.degradation_manager.is_feature_available(FeatureType.CLI_OPERATIONS):
                return DegradedOperation(
                    success=False,
                    result={"error": "CLI operations are currently unavailable"},
                    is_fallback=True,
                    degradation_mode=self.degradation_manager.current_mode,
                    user_message="CLI operations are currently unavailable",
                    suggested_actions=[
                        "System is in unavailable mode",
                        "Check if core services are running",
                        "Try basic file operations only"
                    ]
                )

            # Handle different degradation modes
            current_mode = self.degradation_manager.current_mode

            if current_mode == DegradationMode.OFFLINE_CLI:
                return await self._execute_offline_command(command, args, options)
            elif current_mode in {DegradationMode.CACHED_ONLY, DegradationMode.READ_ONLY}:
                return await self._execute_limited_command(command, args, options)
            else:
                return await self._execute_full_command(command, args, options)

        except Exception as e:
            logger.error(f"Error executing CLI command: {e}")
            return DegradedOperation(
                success=False,
                result={"error": f"CLI command failed: {str(e)}"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode
            )

    async def _execute_offline_command(
        self,
        command: str,
        args: list[str],
        options: dict[str, Any]
    ) -> DegradedOperation:
        """Execute command in offline CLI mode."""
        # Only allow file-based operations
        offline_commands = {
            "status", "health", "list", "info", "config", "help",
            "ls", "cat", "find", "grep"
        }

        if command not in offline_commands:
            return DegradedOperation(
                success=False,
                result={"error": f"Command '{command}' not available in offline mode"},
                is_fallback=True,
                degradation_mode=DegradationMode.OFFLINE_CLI,
                user_message=f"Command '{command}' requires network services",
                suggested_actions=[
                    "Use offline commands: " + ", ".join(offline_commands),
                    "Check daemon service status",
                    "Wait for services to recover"
                ]
            )

        # Execute offline command
        result = await self._execute_basic_command(command, args, options)
        result.user_message = "Operating in offline CLI mode"
        result.suggested_actions = [
            "Limited functionality available",
            "Network services are unavailable",
            "Use 'status' command to check service health"
        ]

        return result

    async def _execute_limited_command(
        self,
        command: str,
        args: list[str],
        options: dict[str, Any]
    ) -> DegradedOperation:
        """Execute command in limited functionality mode."""
        # Read-only operations allowed
        read_only_commands = {
            "search", "status", "health", "list", "info", "config", "help",
            "ls", "cat", "find", "grep", "show"
        }

        if command not in read_only_commands:
            return DegradedOperation(
                success=False,
                result={"error": f"Command '{command}' not available in read-only mode"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode,
                user_message="Write operations are disabled",
                suggested_actions=[
                    "System is in read-only mode",
                    "Use read operations only",
                    "Wait for full service recovery"
                ]
            )

        # Execute read-only command
        result = await self._execute_basic_command(command, args, options)
        result.user_message = "Operating in read-only mode"
        return result

    async def _execute_full_command(
        self,
        command: str,
        args: list[str],
        options: dict[str, Any]
    ) -> DegradedOperation:
        """Execute command with full functionality."""
        return await self._execute_basic_command(command, args, options)

    async def _execute_basic_command(
        self,
        command: str,
        args: list[str],
        options: dict[str, Any]
    ) -> DegradedOperation:
        """Execute basic command implementation."""
        try:
            # Use base CLI if available
            if self.base_cli and hasattr(self.base_cli, 'execute'):
                result = await self.base_cli.execute(command, args, **options)
                return DegradedOperation(
                    success=True,
                    result=result,
                    is_fallback=False,
                    degradation_mode=self.degradation_manager.current_mode
                )

            # Provide basic fallback implementations
            if command == "status":
                return await self._status_command()
            elif command == "health":
                return await self._health_command()
            elif command == "help":
                return self._help_command()
            else:
                return DegradedOperation(
                    success=False,
                    result={"error": f"Command '{command}' not implemented in fallback mode"},
                    is_fallback=True,
                    degradation_mode=self.degradation_manager.current_mode
                )

        except Exception as e:
            logger.error(f"Basic command execution failed: {e}")
            return DegradedOperation(
                success=False,
                result={"error": f"Command execution failed: {str(e)}"},
                is_fallback=True,
                degradation_mode=self.degradation_manager.current_mode
            )

    async def _status_command(self) -> DegradedOperation:
        """Fallback status command implementation."""
        status = self.degradation_manager.get_degradation_status()
        return DegradedOperation(
            success=True,
            result=status,
            is_fallback=True,
            degradation_mode=self.degradation_manager.current_mode,
            user_message="Status from degradation manager"
        )

    async def _health_command(self) -> DegradedOperation:
        """Fallback health command implementation."""
        health_info = {
            "degradation_mode": self.degradation_manager.current_mode.value,
            "available_features": [f.value for f in self.degradation_manager.get_available_features()],
            "unavailable_features": [f.value for f in self.degradation_manager.get_unavailable_features()],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return DegradedOperation(
            success=True,
            result=health_info,
            is_fallback=True,
            degradation_mode=self.degradation_manager.current_mode,
            user_message="Health check from degradation manager"
        )

    def _help_command(self) -> DegradedOperation:
        """Fallback help command implementation."""
        help_text = {
            "available_commands": {
                "status": "Show system status",
                "health": "Show health information",
                "help": "Show this help message"
            },
            "degradation_mode": self.degradation_manager.current_mode.value,
            "note": "Limited commands available due to system degradation"
        }

        return DegradedOperation(
            success=True,
            result=help_text,
            is_fallback=True,
            degradation_mode=self.degradation_manager.current_mode
        )


class DegradationMiddleware:
    """
    Middleware for adding degradation awareness to any operation.

    This can be used to wrap any function or method with automatic
    degradation detection and fallback behavior.
    """

    def __init__(self, degradation_manager: DegradationManager):
        """
        Initialize degradation middleware.

        Args:
            degradation_manager: Degradation manager instance
        """
        self.degradation_manager = degradation_manager

    def degradation_aware(
        self,
        required_features: list[FeatureType],
        fallback_function: Callable | None = None,
        component_id: str | None = None
    ):
        """
        Decorator to make a function degradation-aware.

        Args:
            required_features: Features required for operation
            fallback_function: Function to call if features unavailable
            component_id: Component ID for circuit breaker tracking

        Returns:
            Decorated function with degradation awareness
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    # Check if required features are available
                    missing_features = []
                    for feature in required_features:
                        if not self.degradation_manager.is_feature_available(feature):
                            missing_features.append(feature)

                    if missing_features:
                        if fallback_function:
                            return await fallback_function(*args, **kwargs)
                        else:
                            return DegradedOperation(
                                success=False,
                                result={"error": f"Required features unavailable: {missing_features}"},
                                is_fallback=True,
                                degradation_mode=self.degradation_manager.current_mode,
                                user_message=f"Operation requires features: {[f.value for f in missing_features]}",
                                suggested_actions=["Check system health and wait for recovery"]
                            )

                    # Check circuit breaker if component_id provided
                    if component_id and not self.degradation_manager.is_component_available(component_id):
                        return DegradedOperation(
                            success=False,
                            result={"error": f"Component {component_id} is not available"},
                            is_fallback=True,
                            degradation_mode=self.degradation_manager.current_mode,
                            user_message=f"Component {component_id} is temporarily unavailable",
                            suggested_actions=["Wait for component recovery"]
                        )

                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Record success for circuit breaker
                    if component_id:
                        await self.degradation_manager.record_component_success(component_id)

                    return result

                except Exception as e:
                    # Record failure for circuit breaker
                    if component_id:
                        await self.degradation_manager.record_component_failure(component_id)

                    logger.error(f"Error in degradation-aware function {func.__name__}: {e}")
                    return DegradedOperation(
                        success=False,
                        result={"error": f"Operation failed: {str(e)}"},
                        is_fallback=True,
                        degradation_mode=self.degradation_manager.current_mode
                    )

            return wrapper
        return decorator


# Utility functions for integration
def create_degradation_aware_mcp_tools(
    degradation_manager: DegradationManager,
    base_tools: dict[str, Callable]
) -> dict[str, Callable]:
    """
    Create degradation-aware versions of MCP tools.

    Args:
        degradation_manager: Degradation manager instance
        base_tools: Dictionary of base MCP tool functions

    Returns:
        Dictionary of degradation-aware tool functions
    """
    middleware = DegradationMiddleware(degradation_manager)
    degraded_tools = {}

    # Define feature requirements for each tool type
    tool_requirements = {
        "search": [FeatureType.SEMANTIC_SEARCH, FeatureType.KEYWORD_SEARCH],
        "ingest": [FeatureType.DOCUMENT_INGESTION],
        "health": [],  # Always available
        "status": [],  # Always available
        "list": [FeatureType.MCP_SERVER],
    }

    for tool_name, tool_func in base_tools.items():
        required_features = tool_requirements.get(tool_name, [])
        component_id = "python_mcp_server-default"

        # Apply degradation awareness
        degraded_tools[tool_name] = middleware.degradation_aware(
            required_features=required_features,
            component_id=component_id
        )(tool_func)

    return degraded_tools


async def create_degradation_status_report(
    degradation_manager: DegradationManager
) -> dict[str, Any]:
    """
    Create comprehensive degradation status report.

    Args:
        degradation_manager: Degradation manager instance

    Returns:
        Comprehensive status report
    """
    status = degradation_manager.get_degradation_status()

    # Add user-friendly interpretations
    status["user_friendly"] = {
        "current_status": _get_user_friendly_status(degradation_manager.current_mode),
        "impact_summary": _get_impact_summary(degradation_manager),
        "recommended_actions": _get_recommended_actions(degradation_manager.current_mode),
        "estimated_recovery_time": _estimate_recovery_time(degradation_manager)
    }

    return status


def _get_user_friendly_status(mode: DegradationMode) -> str:
    """Get user-friendly status description."""
    status_map = {
        DegradationMode.NORMAL: "System operating normally",
        DegradationMode.PERFORMANCE_REDUCED: "System running with reduced performance",
        DegradationMode.FEATURES_LIMITED: "Some features temporarily unavailable",
        DegradationMode.READ_ONLY: "System in read-only mode",
        DegradationMode.CACHED_ONLY: "Serving cached responses only",
        DegradationMode.OFFLINE_CLI: "Only command-line operations available",
        DegradationMode.EMERGENCY: "Emergency mode - minimal functionality",
        DegradationMode.UNAVAILABLE: "System currently unavailable"
    }
    return status_map.get(mode, "Unknown status")


def _get_impact_summary(degradation_manager: DegradationManager) -> str:
    """Get impact summary for current degradation mode."""
    unavailable_features = degradation_manager.get_unavailable_features()

    if not unavailable_features:
        return "No functionality impact"
    elif len(unavailable_features) <= 2:
        return f"Limited impact: {len(unavailable_features)} features unavailable"
    elif len(unavailable_features) <= 5:
        return f"Moderate impact: {len(unavailable_features)} features unavailable"
    else:
        return f"Significant impact: {len(unavailable_features)} features unavailable"


def _get_recommended_actions(mode: DegradationMode) -> list[str]:
    """Get recommended actions for degradation mode."""
    actions_map = {
        DegradationMode.NORMAL: ["No action required", "Monitor system health"],
        DegradationMode.PERFORMANCE_REDUCED: [
            "Monitor resource usage",
            "Consider reducing concurrent operations",
            "Check for resource constraints"
        ],
        DegradationMode.FEATURES_LIMITED: [
            "Check component health status",
            "Review system logs for errors",
            "Consider restarting affected services"
        ],
        DegradationMode.READ_ONLY: [
            "Check Rust daemon status",
            "Verify database connectivity",
            "Review ingestion queue for issues"
        ],
        DegradationMode.CACHED_ONLY: [
            "Restart Rust daemon if needed",
            "Check MCP server connectivity",
            "Verify Qdrant database status"
        ],
        DegradationMode.OFFLINE_CLI: [
            "Check all services are running",
            "Verify network connectivity",
            "Review daemon startup logs"
        ],
        DegradationMode.EMERGENCY: [
            "Immediate system administrator attention required",
            "Check all component processes",
            "Review system resources and logs"
        ],
        DegradationMode.UNAVAILABLE: [
            "System restart may be required",
            "Check all services and dependencies",
            "Contact system administrator"
        ]
    }
    return actions_map.get(mode, ["Check system status"])


def _estimate_recovery_time(degradation_manager: DegradationManager) -> str:
    """Estimate recovery time based on current state."""
    mode = degradation_manager.current_mode

    if mode == DegradationMode.NORMAL:
        return "N/A - System operating normally"
    elif mode in {DegradationMode.PERFORMANCE_REDUCED, DegradationMode.FEATURES_LIMITED}:
        return "1-5 minutes (automatic recovery expected)"
    elif mode in {DegradationMode.READ_ONLY, DegradationMode.CACHED_ONLY}:
        return "5-15 minutes (service restart may be needed)"
    elif mode == DegradationMode.OFFLINE_CLI:
        return "10-30 minutes (full system restart likely needed)"
    else:
        return "30+ minutes (manual intervention required)"
