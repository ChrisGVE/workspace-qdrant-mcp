"""
FastMCP server for workspace-qdrant-mcp.

Streamlined 4-tool implementation that provides all the functionality of the original
36-tool system through intelligent content-based routing and parameter analysis.

The server automatically detects project structure, initializes workspace-specific collections,
and provides hybrid search combining dense (semantic) and sparse (keyword) vectors.

Key Features:
    - 4 comprehensive tools: store, search, manage, retrieve
    - Content-based routing - parameters determine specific actions
    - Unified multi-tenant collections with tenant_id-based filtering
    - Branch-aware querying with automatic Git branch detection
    - File type filtering via metadata (code, test, docs, config, data, build, other)
    - Hybrid search combining dense (semantic) and sparse (keyword) vectors
    - Evidence-based performance: 100% precision for symbol/exact search, 94.2% for semantic
    - Comprehensive scratchbook for cross-project note management
    - Production-ready async architecture with comprehensive error handling

Architecture (ADR-001 Canonical Collections):
    - Canonical projects collection: projects (ALL projects in one collection)
    - Canonical libraries collection: libraries (ALL libraries in one collection)
    - Canonical memory collection: memory (LLM rules and preferences)
    - Tenant isolation via tenant_id payload filtering (indexed for O(1) lookup)
    - Branch-scoped queries: All queries filter by Git branch (default: current branch)
    - File type differentiation via metadata: code, test, docs, config, data, build, other
    - User collections: {basename}-{type} for user notes (auto-enriched with project_id)
    - NO underscore prefix on canonical collection names per ADR-001

Tools:
    1. store - Store any content (documents, notes, code, web content)
    2. search - Hybrid semantic + keyword search with branch and file_type filtering
    3. manage - Collection management, system status, configuration
    4. retrieve - Direct document retrieval by ID or metadata with branch filtering

Example Usage:
    # Store different content types (all go to 'projects' collection with tenant_id)
    store(content="user notes", source="scratchbook")  # metadata: file_type="other"
    store(file_path="main.py", content="code")         # metadata: file_type="code"
    store(url="https://docs.com", content="docs")      # metadata: file_type="docs"

    # Search with scope, branch and file_type filtering
    search(query="authentication", scope="project")              # Current project only
    search(query="def login", mode="exact", file_type="code")    # Current project, code only
    search(query="notes", scope="global")                        # All projects
    search(query="numpy", include_libraries=True)                # Projects + libraries

    # Management operations
    manage(action="list_collections")                  # List all collections
    manage(action="workspace_status")                  # System status
    manage(action="init_project")                      # Register project in 'projects'

    # Direct retrieval with branch filtering
    retrieve(document_id="uuid-123")                              # Current branch
    retrieve(metadata={"file_type": "test"}, branch="develop")    # develop branch, tests
    retrieve(scope="all")                                         # All collections

Write Path Architecture (First Principle 10, ADR-001, ADR-002):
    DAEMON-ONLY WRITES: All Qdrant write operations MUST route through the daemon

    Collection Types (ADR-001 Canonical Names - NO underscore prefix):
        - PROJECT: projects - Canonical collection, tenant isolation via tenant_id filter
        - LIBRARY: libraries - Canonical collection, isolation via library_name filter
        - USER: {basename}-{type} - User collections, enriched with project_id
        - MEMORY: memory - LLM rules and preferences (routes through daemon per Task 30)

    Write Priority (Task 37/ADR-002 - NO DIRECT QDRANT WRITES):
        1. PRIMARY: DaemonClient.ingest_text() / create_collection_v2() / delete_collection_v2()
        2. FALLBACK: Queue to unified_queue via state_manager.enqueue_unified()
        3. LAST RESORT: Direct Qdrant writes only when both daemon and queue unavailable (logged)

    Fallback to unified_queue (ADR-002):
        - When daemon is unavailable, content is queued to SQLite unified_queue
        - Daemon processes queued items when it becomes available
        - No direct Qdrant writes from MCP server (queue fallback ensures data persistence)
        - All fallback paths log warnings and include "fallback_mode": "unified_queue"

    Backward Compatibility Note (v0.4.0):
        - Legacy direct writes removed; all collections use daemon-only or queue-fallback paths
        - MEMORY collections previously used direct writes; now route through daemon (Task 30)

    See: FIRST-PRINCIPLES.md (Principle 10), ADR-002, Task 30, Task 37
"""

import asyncio
import inspect
import logging
import os
import random
import subprocess
from contextlib import asynccontextmanager
from enum import Enum

# CRITICAL: Complete stdio silence must be set up before ANY other imports
# This prevents ALL console output in MCP stdio mode for protocol compliance
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _detect_stdio_mode() -> bool:
    """Detect MCP stdio mode with comprehensive checks."""
    # Explicit environment variables
    if os.getenv("WQM_STDIO_MODE", "").lower() == "true":
        return True
    if os.getenv("WQM_CLI_MODE", "").lower() == "true":
        return False

    # Check if stdin/stdout are connected to pipes (MCP stdio mode)
    try:
        import stat
        mode = os.fstat(sys.stdin.fileno()).st_mode
        if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
            return True
    except (OSError, AttributeError):
        pass

    # Check for MCP-related environment or argv patterns
    if any(arg in ['stdio', 'mcp'] for arg in sys.argv):
        return True

    return False

# Apply stdio mode silencing if detected
if _detect_stdio_mode():
    # Redirect all console output to devnull in stdio mode
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    # Disable all logging to prevent protocol contamination
    logging.disable(logging.CRITICAL)

# Import project detection and branch utilities after stdio setup
from common.core.collection_aliases import AliasManager
from common.core.collection_naming import build_project_collection_name
from common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ContentIngestionStatus,
    UnifiedQueueItemType,
    UnifiedQueueOperation,
)
from common.grpc.daemon_client import DaemonClient, DaemonConnectionError
from common.observability.metrics import (
    track_tool, record_search_scope, record_search_results
)
from common.utils.git_utils import get_current_branch
from common.utils.project_detection import calculate_tenant_id

# Memory system imports for Claude Code context injection (Task 5: code audit)
from common.memory import MemoryManager
from common.core.context_injection import (
    generate_mcp_context,
    RuleFilter,
)

# Global components
qdrant_client: AsyncQdrantClient | None = None
embedding_model = None
daemon_client: DaemonClient | None = None
alias_manager: AliasManager | None = None
state_manager: SQLiteStateManager | None = None
memory_manager: MemoryManager | None = None  # Memory system for context injection
project_cache = {}

# Session lifecycle state
_session_project_id: str | None = None
_session_project_path: str | None = None
_session_heartbeat: "SessionHeartbeat | None" = None

# ============================================================================
# Task 452: Stability and Reliability Improvements
# ============================================================================
# Constants for connection retry with exponential backoff
_RETRY_BASE_DELAY_SECS = 1.0  # Base delay for exponential backoff
_RETRY_MAX_DELAY_SECS = 30.0  # Maximum delay cap
_RETRY_JITTER_FACTOR = 0.1  # Random jitter factor (±10%)

# Cache management constants
_CACHE_MAX_SIZE = 1000  # Maximum entries in project_cache
_CACHE_TTL_SECS = 3600  # Cache TTL in seconds (1 hour)

# Health monitoring state
_last_health_check: datetime | None = None
_consecutive_failures: int = 0
_total_operations: int = 0
_successful_operations: int = 0


def is_project_activated() -> bool:
    """
    Check if a project is currently activated in this session.

    Task 457: Explicit project activation flow
    Projects must be activated via manage(action="activate_project") before
    project-scoped operations have full daemon priority support.

    Returns:
        True if project is activated (session has project_id and heartbeat running),
        False otherwise.
    """
    return _session_project_id is not None and (
        _session_heartbeat is not None and _session_heartbeat.is_running
    )


def get_activation_warning() -> str:
    """
    Get warning message for operations performed without project activation.

    Task 457: Explicit project activation flow
    """
    return (
        "Project not activated. Use manage(action='activate_project') to enable "
        "high-priority daemon processing and session heartbeat. Operations will "
        "still work but may have lower priority."
    )


# ============================================================================
# Task 458: Daemon Unresponsive State Machine
# ============================================================================
# Tracks daemon availability with 2-attempt 10s deadline policy to handle
# daemon unavailability gracefully.
# ============================================================================


class DaemonState(Enum):
    """Daemon availability state."""
    AVAILABLE = "available"
    UNRESPONSIVE = "unresponsive"


# Daemon state machine constants
_DAEMON_CHECK_TIMEOUT_SECS = 10  # Timeout for each health check attempt
_DAEMON_CHECK_MAX_ATTEMPTS = 2   # Max attempts before marking as unresponsive

# Global daemon state (protected by _daemon_state_lock for thread safety)
_daemon_state: DaemonState = DaemonState.AVAILABLE
_daemon_state_lock = asyncio.Lock()


async def check_daemon_availability() -> bool:
    """
    Check if daemon is available using 2-attempt 10s deadline policy.

    Task 458: Daemon unresponsive state machine
    Performs up to 2 health checks with 10s timeout each before
    marking daemon as unresponsive.

    Returns:
        True if daemon is available, False if unresponsive
    """
    global _daemon_state

    if not daemon_client:
        async with _daemon_state_lock:
            _daemon_state = DaemonState.UNRESPONSIVE
        return False

    logger = logging.getLogger(__name__)

    for attempt in range(1, _DAEMON_CHECK_MAX_ATTEMPTS + 1):
        try:
            # Try health check with timeout
            health = await asyncio.wait_for(
                daemon_client.health_check(),
                timeout=_DAEMON_CHECK_TIMEOUT_SECS
            )
            if health.healthy:
                async with _daemon_state_lock:
                    _daemon_state = DaemonState.AVAILABLE
                logger.debug(f"Daemon health check passed (attempt {attempt})")
                return True
            else:
                logger.warning(f"Daemon health check failed (attempt {attempt}): not healthy")
        except asyncio.TimeoutError:
            logger.warning(
                f"Daemon health check timeout after {_DAEMON_CHECK_TIMEOUT_SECS}s "
                f"(attempt {attempt}/{_DAEMON_CHECK_MAX_ATTEMPTS})"
            )
        except Exception as e:
            logger.warning(
                f"Daemon health check error (attempt {attempt}): {e}"
            )

    # All attempts failed - mark as unresponsive
    async with _daemon_state_lock:
        _daemon_state = DaemonState.UNRESPONSIVE
    logger.error(
        f"Daemon marked as UNRESPONSIVE after {_DAEMON_CHECK_MAX_ATTEMPTS} failed attempts"
    )
    return False


async def update_daemon_state(success: bool) -> None:
    """
    Update daemon state based on operation result.

    Task 458: Daemon unresponsive state machine
    Called after daemon operations to update the state machine.

    Args:
        success: True if daemon operation succeeded, False if failed
    """
    global _daemon_state

    async with _daemon_state_lock:
        if success:
            _daemon_state = DaemonState.AVAILABLE
        else:
            _daemon_state = DaemonState.UNRESPONSIVE


def get_daemon_state() -> DaemonState:
    """
    Get current daemon state.

    Task 458: Daemon unresponsive state machine

    Returns:
        Current DaemonState (AVAILABLE or UNRESPONSIVE)
    """
    return _daemon_state


def is_daemon_available() -> bool:
    """
    Check if daemon is currently marked as available.

    Task 458: Daemon unresponsive state machine

    Returns:
        True if daemon state is AVAILABLE, False otherwise
    """
    return _daemon_state == DaemonState.AVAILABLE


def get_daemon_unavailable_message() -> str:
    """
    Get message explaining daemon unavailability and suggesting fallback.

    Task 458: Daemon unresponsive state machine

    Returns:
        User-friendly message with fallback suggestion
    """
    return (
        "Daemon is currently unresponsive. Project-scoped operations require "
        "an available daemon for high-priority processing. "
        "Suggested fallback: Use scope='global' with include_libraries=True "
        "to search across all collections without daemon dependency."
    )


# ============================================================================
# Task 452: Stability and Reliability Helper Functions
# ============================================================================


def calculate_retry_delay(attempt: int) -> float:
    """
    Calculate retry delay with exponential backoff and jitter.

    Task 452: Connection retry logic with exponential backoff

    Uses exponential backoff formula: base_delay * (2 ^ attempt)
    with jitter to prevent thundering herd problems.

    Args:
        attempt: Current attempt number (0-indexed)

    Returns:
        Delay in seconds before next retry
    """
    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, ...
    delay = _RETRY_BASE_DELAY_SECS * (2 ** attempt)

    # Cap at maximum delay
    delay = min(delay, _RETRY_MAX_DELAY_SECS)

    # Add jitter (±10%) to prevent thundering herd
    jitter = delay * _RETRY_JITTER_FACTOR * (2 * random.random() - 1)
    delay += jitter

    return max(0.1, delay)  # Minimum 100ms delay


async def retry_with_backoff(
    operation,
    max_attempts: int = 3,
    operation_name: str = "operation",
    on_retry=None,
) -> Any:
    """
    Execute operation with exponential backoff retry.

    Task 452: Connection retry logic

    Args:
        operation: Async callable to execute
        max_attempts: Maximum number of attempts
        operation_name: Name for logging
        on_retry: Optional callback called on retry with (attempt, exception)

    Returns:
        Result of successful operation

    Raises:
        Last exception if all attempts fail
    """
    logger = logging.getLogger(__name__)
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            remaining = max_attempts - attempt - 1

            if remaining > 0:
                delay = calculate_retry_delay(attempt)
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {delay:.1f}s ({remaining} attempts left)"
                )

                if on_retry:
                    on_retry(attempt, e)

                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"{operation_name} failed after {max_attempts} attempts: {e}"
                )

    raise last_exception


def manage_cache_size():
    """
    Manage project_cache size to prevent memory leaks.

    Task 452: Memory leak prevention

    Evicts oldest entries when cache exceeds maximum size.
    """
    global project_cache

    if len(project_cache) > _CACHE_MAX_SIZE:
        # Evict oldest half of entries
        entries_to_remove = len(project_cache) - (_CACHE_MAX_SIZE // 2)
        keys_to_remove = list(project_cache.keys())[:entries_to_remove]
        for key in keys_to_remove:
            del project_cache[key]

        logger = logging.getLogger(__name__)
        logger.info(f"Cache cleanup: removed {len(keys_to_remove)} entries")


def record_operation_result(success: bool) -> None:
    """
    Record operation result for health monitoring.

    Task 452: Health monitoring

    Args:
        success: Whether the operation succeeded
    """
    global _total_operations, _successful_operations, _consecutive_failures

    _total_operations += 1
    if success:
        _successful_operations += 1
        _consecutive_failures = 0
    else:
        _consecutive_failures += 1


def get_health_metrics() -> dict[str, Any]:
    """
    Get health metrics for monitoring.

    Task 452: Health monitoring endpoints

    Returns:
        Dict with health metrics including success rate and failure counts
    """
    global _total_operations, _successful_operations, _consecutive_failures
    global _last_health_check

    success_rate = (
        (_successful_operations / _total_operations * 100)
        if _total_operations > 0
        else 100.0
    )

    return {
        "total_operations": _total_operations,
        "successful_operations": _successful_operations,
        "failed_operations": _total_operations - _successful_operations,
        "success_rate_percent": round(success_rate, 2),
        "consecutive_failures": _consecutive_failures,
        "last_health_check": _last_health_check.isoformat() if _last_health_check else None,
        "daemon_state": _daemon_state.value,
        "cache_size": len(project_cache),
        "cache_max_size": _CACHE_MAX_SIZE,
    }


async def cleanup_resources() -> None:
    """
    Cleanup resources for graceful shutdown or error recovery.

    Task 452: Resource cleanup on errors

    Performs:
    - Cache cleanup
    - Daemon client disconnection
    - State manager cleanup
    - Memory manager cleanup (Task 5: code audit)
    """
    global qdrant_client, daemon_client, alias_manager, state_manager
    global embedding_model, project_cache, memory_manager

    logger = logging.getLogger(__name__)
    logger.info("Starting resource cleanup...")

    # Clear caches
    project_cache.clear()

    # Cleanup daemon client
    if daemon_client:
        try:
            await daemon_client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting daemon client: {e}")

    # Cleanup state manager
    if state_manager:
        try:
            await state_manager.close()
        except Exception as e:
            logger.warning(f"Error closing state manager: {e}")

    # Close Qdrant client
    if qdrant_client:
        try:
            await _maybe_await(qdrant_client.close())
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

    # Clear embedding model (release memory)
    embedding_model = None

    # Clear memory manager (Task 5: code audit)
    memory_manager = None

    logger.info("Resource cleanup completed")


class SessionHeartbeat:
    """
    Background heartbeat task for MCP server session lifecycle.

    Task 407: Implements periodic heartbeat to keep session alive with daemon.
    Without heartbeat, sessions are considered orphaned after 60 seconds and
    demoted by the daemon's SessionMonitor.

    The heartbeat runs every 30 seconds (well within 60s timeout) to ensure
    the daemon knows the session is still active.
    """

    # Heartbeat interval in seconds (should be < 60s daemon timeout)
    HEARTBEAT_INTERVAL_SECS = 30

    def __init__(self, daemon_client: DaemonClient, project_id: str):
        """
        Initialize heartbeat task.

        Args:
            daemon_client: Connected daemon client for sending heartbeats
            project_id: 12-char hex project identifier
        """
        self._daemon_client = daemon_client
        self._project_id = project_id
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._running = False
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the background heartbeat task."""
        if self._running:
            self._logger.warning("SessionHeartbeat already running")
            return

        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        self._logger.info(
            f"SessionHeartbeat started for project {self._project_id} "
            f"(interval: {self.HEARTBEAT_INTERVAL_SECS}s)"
        )

    async def stop(self) -> None:
        """Stop the background heartbeat task."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._task:
            # Wait for task to complete with timeout
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._logger.warning("SessionHeartbeat task did not stop cleanly, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        self._logger.info(f"SessionHeartbeat stopped for project {self._project_id}")

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop - sends periodic heartbeats to daemon."""
        consecutive_failures = 0
        max_consecutive_failures = 3

        while self._running:
            try:
                # Wait for interval or stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.HEARTBEAT_INTERVAL_SECS
                    )
                    # Stop event was set - exit loop
                    break
                except asyncio.TimeoutError:
                    # Normal timeout - send heartbeat
                    pass

                # Send heartbeat to daemon
                try:
                    response = await self._daemon_client.heartbeat(self._project_id)
                    if response.acknowledged:
                        consecutive_failures = 0
                        self._logger.debug(
                            f"Heartbeat acknowledged for project {self._project_id}"
                        )
                    else:
                        self._logger.warning(
                            f"Heartbeat not acknowledged for project {self._project_id}"
                        )
                        consecutive_failures += 1

                except Exception as e:
                    consecutive_failures += 1
                    self._logger.warning(
                        f"Heartbeat failed for project {self._project_id}: {e} "
                        f"(failures: {consecutive_failures}/{max_consecutive_failures})"
                    )

                # If too many consecutive failures, log error but continue
                # The daemon will eventually detect the orphaned session
                if consecutive_failures >= max_consecutive_failures:
                    self._logger.error(
                        f"Heartbeat consistently failing for project {self._project_id}, "
                        f"daemon may consider session orphaned"
                    )
                    # Reset counter to avoid log spam
                    consecutive_failures = 0

            except asyncio.CancelledError:
                self._logger.debug("Heartbeat loop cancelled")
                break
            except Exception as e:
                self._logger.error(f"Unexpected error in heartbeat loop: {e}")
                # Continue loop - try to recover

    @property
    def is_running(self) -> bool:
        """Check if heartbeat task is running."""
        return self._running


async def _get_git_remote() -> str | None:
    """Get git remote URL for the current repository."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "remote", "get-url", "origin",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.cwd()
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0 and stdout:
                return stdout.decode().strip()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    except Exception:
        pass
    return None


@asynccontextmanager
async def lifespan(app):
    """
    FastMCP lifespan context manager for session lifecycle management.

    Task 395: Multi-tenant session lifecycle
    Task 407: Heartbeat mechanism for session liveness
    Task 457: Explicit project activation flow (ADR-001)

    Changes (Task 457):
    - On startup: Initialize daemon client only (NO auto-registration)
    - User must explicitly call manage(action="activate_project") to register
    - On shutdown: Stop heartbeat (if running), deprioritize project (if activated)

    The daemon uses session registration to:
    - Set HIGH priority for actively-edited projects
    - Track active sessions for crash recovery (via heartbeat)
    - Optimize file watcher resources based on activity

    The heartbeat mechanism (Task 407):
    - Sends periodic heartbeat every 30 seconds
    - Daemon timeout is 60 seconds for orphaned session detection
    - Without heartbeat, crashed sessions are detected and demoted

    Project activation (Task 457):
    - Projects must be explicitly activated via manage(action="activate_project")
    - Using scope="project" without activation generates warnings
    - Deactivation via manage(action="deactivate_project")
    """
    global daemon_client, _session_project_id, _session_project_path, _session_heartbeat

    logger = logging.getLogger(__name__)

    # =========================================================================
    # STARTUP: Initialize daemon client only (Task 457: NO auto-registration)
    # =========================================================================
    try:
        # Initialize daemon client if not already done
        if daemon_client is None:
            daemon_client = DaemonClient()
            try:
                await daemon_client.connect()
                logger.info("Daemon client connected (project activation required)")
            except DaemonConnectionError:
                # Daemon connection is optional - server works without it
                daemon_client = None
                logger.warning("Daemon not available - session lifecycle disabled")

        # Task 457: NO auto-registration on startup
        # User must explicitly call manage(action="activate_project")
        # This ensures projects are only prioritized when actively being worked on
        logger.debug("Server started. Use manage(action='activate_project') to register project.")

        # =====================================================================
        # Task 5 (code audit): Initialize memory system and inject context
        # =====================================================================
        # Initialize memory manager for Claude Code context injection
        # This enables memory rules to be retrieved and injected on MCP startup
        global memory_manager
        try:
            memory_manager = MemoryManager()
            if await memory_manager.initialize():
                logger.info("Memory manager initialized for context injection")

                # Query memory collection for rules and generate MCP context
                try:
                    # Generate context from memory rules (no filter = get all rules)
                    mcp_context = await generate_mcp_context(
                        memory_manager=memory_manager,
                        token_budget=15000,  # Default MCP context budget
                        filter=None,  # Get all applicable rules
                    )

                    if mcp_context:
                        # Store context in global for access by tools if needed
                        # The context is automatically available through memory manager
                        logger.info(
                            f"Memory context injected: {len(mcp_context)} chars "
                            f"from 'memory' collection"
                        )
                    else:
                        logger.debug("No memory rules found for context injection")

                except Exception as inject_err:
                    logger.warning(f"Memory context injection failed: {inject_err}")
            else:
                logger.warning("Memory manager initialization failed - context injection disabled")
                memory_manager = None
        except Exception as mem_err:
            logger.warning(f"Failed to initialize memory manager: {mem_err}")
            memory_manager = None

    except Exception as e:
        logger.warning(f"Lifespan startup error: {e}")

    # Yield control to the application
    yield

    # =========================================================================
    # SHUTDOWN: Stop heartbeat and deprioritize project with daemon
    # Task 452: Graceful shutdown handling with comprehensive cleanup
    # =========================================================================
    try:
        # Task 407: Stop heartbeat first to prevent orphan detection race
        if _session_heartbeat and _session_heartbeat.is_running:
            try:
                await _session_heartbeat.stop()
            except Exception as e:
                logger.warning(f"Failed to stop heartbeat: {e}")

        if daemon_client and _session_project_id:
            try:
                response = await daemon_client.deprioritize_project(
                    project_id=_session_project_id
                )
                logger.info(
                    f"Project deprioritized: {_session_project_id}, "
                    f"remaining_sessions={response.remaining_sessions}, "
                    f"new_priority={response.new_priority}"
                )
            except Exception as e:
                logger.warning(f"Failed to deprioritize project: {e}")

        # Task 452: Comprehensive resource cleanup
        await cleanup_resources()
    except Exception as e:
        logger.warning(f"Lifespan shutdown error: {e}")
        # Task 452: Ensure cleanup happens even on error
        try:
            await cleanup_resources()
        except Exception as cleanup_error:
            logger.error(f"Cleanup during shutdown error also failed: {cleanup_error}")


# Initialize the FastMCP app with lifespan management
app = FastMCP("Workspace Qdrant MCP", lifespan=lifespan)

# Collection basename mapping for Rust daemon validation
# Maps collection types to valid basenames (non-empty strings)
BASENAME_MAP = {
    "project": "code",      # PROJECT collections: _{project_id}
    "user": "notes",        # USER collections: {basename}-{type}
    "library": "lib",       # LIBRARY collections: _{library_name}
    "memory": "memory",     # MEMORY collections: memory (canonical per ADR-001)
}

# Canonical collection names per ADR-001
# These collections store data from all projects/libraries with tenant_id filtering
# NO underscore prefix - canonical names only
CANONICAL_COLLECTIONS = {
    "projects": "projects",    # All project code/documents
    "libraries": "libraries",  # All library documentation
    "memory": "memory",        # Agent memory and rules
}

# Deprecated patterns - will be rejected with helpful error messages
DEPRECATED_COLLECTION_PATTERNS = [
    "_projects", "_libraries", "_memory", "_agent_memory"
]


def validate_collection_name(name: str) -> tuple[bool, str | None]:
    """Validate collection name is canonical (not deprecated) per ADR-001.

    Args:
        name: Collection name to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if name in DEPRECATED_COLLECTION_PATTERNS:
        canonical = name.lstrip("_").replace("agent_", "")
        return False, f"Collection '{name}' is deprecated per ADR-001. Use '{canonical}' instead."
    if name.startswith("__"):
        return False, f"Double-underscore prefix is deprecated per ADR-001: {name}"
    return True, None


def get_canonical_collection(collection_type: str) -> str:
    """Get canonical collection name for a type per ADR-001.

    Args:
        collection_type: One of "projects", "libraries", "memory"

    Returns:
        Canonical collection name

    Raises:
        ValueError: If collection_type is unknown
    """
    if collection_type not in CANONICAL_COLLECTIONS:
        raise ValueError(f"Unknown collection type: {collection_type}. "
                        f"Valid types: {list(CANONICAL_COLLECTIONS.keys())}")
    return CANONICAL_COLLECTIONS[collection_type]


def get_collection_type(collection_name: str) -> str:
    """Determine collection type from collection name per ADR-001.

    Supports both canonical names (preferred) and deprecated patterns (for migration).

    Args:
        collection_name: Collection name to analyze

    Returns:
        One of: "project", "user", "library", "memory"
    """
    # Canonical collection names (ADR-001)
    if collection_name == "memory":
        return "memory"
    if collection_name == "projects":
        return "project"
    if collection_name == "libraries":
        return "library"

    # Deprecated patterns (logged as warnings, still supported for migration)
    if collection_name in ("_memory", "_agent_memory"):
        return "memory"
    elif collection_name.startswith("_"):
        # Could be project or library - check for library patterns
        # Libraries typically have recognizable names (e.g., _numpy, _pandas)
        # Projects are hex hashes (e.g., _a1b2c3d4e5f6)
        if len(collection_name) == 13:  # _{12-char-hash}
            return "project"
        else:
            return "library"
    else:
        # No underscore prefix = user collection
        return "user"


# ============================================================================
# Task 459: Dot-Separated Tag Hierarchy (ADR-001)
# ============================================================================
# Tag format: main_tag.sub_tag
# - Projects: project_id.branch (e.g., workspace-qdrant-mcp.feature-auth)
# - Libraries: library_name.version (e.g., numpy.1.24.0)
#
# Benefits:
# - Hierarchical filtering (prefix matching)
# - Clear organization within collections
# - Enables branch-scoped and version-scoped queries
# ============================================================================


def generate_main_tag(
    project_id: str | None = None,
    library_name: str | None = None
) -> str | None:
    """
    Generate main_tag from project_id or library_name.

    Task 459: Dot-separated tag hierarchy (ADR-001)

    Args:
        project_id: Project identifier (12-char hex)
        library_name: Library name (e.g., "numpy", "react")

    Returns:
        Main tag string or None if neither provided
    """
    if project_id:
        return project_id
    if library_name:
        return library_name
    return None


def generate_full_tag(
    main_tag: str,
    sub_tag: str | None = None
) -> str:
    """
    Generate full dot-separated tag from main_tag and sub_tag.

    Task 459: Dot-separated tag hierarchy (ADR-001)

    Format: main_tag.sub_tag
    Examples:
        - project_id.branch: "a1b2c3d4e5f6.feature-auth"
        - library_name.version: "numpy.1.24.0"

    Args:
        main_tag: Primary identifier (project_id or library_name)
        sub_tag: Secondary identifier (branch or version), optional

    Returns:
        Full tag string (main_tag alone if no sub_tag)
    """
    if sub_tag:
        return f"{main_tag}.{sub_tag}"
    return main_tag


def parse_tag(tag: str) -> tuple[str, str | None]:
    """
    Parse dot-separated tag into main_tag and sub_tag components.

    Task 459: Dot-separated tag hierarchy (ADR-001)

    Args:
        tag: Full tag string (e.g., "myproject.feature-auth")

    Returns:
        Tuple of (main_tag, sub_tag). sub_tag is None if no dot separator.
    """
    if "." in tag:
        parts = tag.split(".", 1)
        return parts[0], parts[1]
    return tag, None


def validate_tag(tag: str) -> tuple[bool, str | None]:
    """
    Validate tag format per ADR-001 specification.

    Task 459: Dot-separated tag hierarchy (ADR-001)

    Valid tag rules:
    - main_tag: alphanumeric, hyphens, underscores
    - sub_tag: alphanumeric, hyphens, underscores, dots (for versions)
    - Cannot start with dot or end with dot
    - Cannot have consecutive dots

    Args:
        tag: Tag string to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not tag:
        return False, "Tag cannot be empty"
    if tag.startswith("."):
        return False, "Tag cannot start with a dot"
    if tag.endswith("."):
        return False, "Tag cannot end with a dot"
    if ".." in tag:
        return False, "Tag cannot contain consecutive dots"

    # Parse and validate components
    main_tag, sub_tag = parse_tag(tag)

    # Validate main_tag (alphanumeric, hyphens, underscores)
    import re
    main_tag_pattern = r'^[a-zA-Z0-9_-]+$'
    if not re.match(main_tag_pattern, main_tag):
        return False, f"Invalid main_tag '{main_tag}': must be alphanumeric with hyphens/underscores"

    # Validate sub_tag if present (alphanumeric, hyphens, underscores, dots for versions)
    if sub_tag:
        sub_tag_pattern = r'^[a-zA-Z0-9._-]+$'
        if not re.match(sub_tag_pattern, sub_tag):
            return False, f"Invalid sub_tag '{sub_tag}': must be alphanumeric with hyphens/underscores/dots"

    return True, None


def matches_tag_prefix(full_tag: str, prefix: str) -> bool:
    """
    Check if full_tag matches or starts with the given prefix.

    Task 459: Dot-separated tag hierarchy (ADR-001)
    Enables hierarchical filtering by prefix matching.

    Examples:
        - matches_tag_prefix("project.feature", "project") -> True
        - matches_tag_prefix("project.feature", "project.feature") -> True
        - matches_tag_prefix("project.feature", "project.other") -> False
        - matches_tag_prefix("numpy.1.24.0", "numpy") -> True

    Args:
        full_tag: The complete tag to check
        prefix: The prefix to match against

    Returns:
        True if full_tag equals prefix or starts with prefix followed by dot
    """
    if full_tag == prefix:
        return True
    return full_tag.startswith(f"{prefix}.")


# Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_CONFIG = {
    "distance": Distance.COSINE,
    "vector_size": 384,  # all-MiniLM-L6-v2 embedding size
}

async def get_project_name() -> str:
    """Detect current project name from git or directory using async subprocess."""
    try:
        # Try to get from git remote URL using async subprocess
        proc = await asyncio.create_subprocess_exec(
            "git", "remote", "get-url", "origin",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.cwd()
        )

        # Wait for subprocess with timeout
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            if proc.returncode == 0 and stdout:
                url = stdout.decode().strip()
                # Extract repo name from URL
                if url.endswith('.git'):
                    url = url[:-4]
                return url.split('/')[-1]
        except asyncio.TimeoutError:
            # Kill subprocess if timeout
            proc.kill()
            await proc.wait()
    except Exception:
        pass

    # Fallback to directory name
    return Path.cwd().name

def get_project_collection(project_path: Path | None = None) -> str:
    """
    Get the project collection name for a given project path.

    Uses Task 374.6 architecture: _{project_id} where project_id is
    12-char hex hash from calculate_tenant_id().

    Args:
        project_path: Path to project root. Defaults to current directory.

    Returns:
        Collection name in format _{project_id}
    """
    if project_path is None:
        project_path = Path.cwd()

    # Generate project ID using calculate_tenant_id from project_detection
    project_id = calculate_tenant_id(str(project_path))

    # Build collection name using collection_naming module
    return build_project_collection_name(project_id)

async def initialize_components():
    """Initialize Qdrant client, daemon client, embedding model, and alias manager.

    Note: daemon_client may already be initialized by the lifespan context manager.
    This function ensures all components are ready for tool operations.

    Task 405: Adds alias_manager initialization for backward compatibility
    when old collection names (_{project_id}) are used.
    """
    global qdrant_client, embedding_model, daemon_client, alias_manager, state_manager

    if qdrant_client is None:
        # Connect to Qdrant with async client
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        qdrant_client = AsyncQdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

    if embedding_model is None:
        # Lazy import to avoid slow module-level imports
        from fastembed import TextEmbedding
        model_name = os.getenv("FASTEMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
        embedding_model = TextEmbedding(model_name)

    # Daemon client may be initialized by lifespan, only create if not present
    if daemon_client is None:
        # Initialize daemon client for write operations
        daemon_client = DaemonClient()
        try:
            await daemon_client.connect()
        except DaemonConnectionError:
            # Daemon connection is optional - fall back to direct writes if unavailable
            daemon_client = None

    # Initialize alias manager for backward compatibility (Task 405)
    if alias_manager is None:
        try:
            # Initialize state manager for SQLite persistence
            if state_manager is None:
                state_manager = SQLiteStateManager()
                await state_manager.initialize()

            # Create sync Qdrant client for alias manager (uses sync API)
            from qdrant_client import QdrantClient
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            sync_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

            alias_manager = AliasManager(sync_client, state_manager)
            await alias_manager.initialize()
        except Exception as e:
            # Alias manager is optional - proceed without it
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize alias manager: {e}")


async def resolve_collection_alias(collection_name: str) -> tuple[str, bool]:
    """
    Resolve a collection name, handling aliases transparently.

    Task 405: Backward compatibility for old collection names (_{project_id}).
    If an alias exists, resolves to the actual collection name and logs a
    deprecation warning.

    Args:
        collection_name: Collection name or alias to resolve

    Returns:
        Tuple of (resolved_collection_name, was_alias)
    """
    if alias_manager is None:
        return collection_name, False

    try:
        resolved = await alias_manager.resolve_collection_name(collection_name)

        if resolved != collection_name:
            # Collection was an alias - log deprecation warning
            logger = logging.getLogger(__name__)
            logger.warning(
                f"DEPRECATION: Collection alias '{collection_name}' resolved to '{resolved}'. "
                f"Old collection names will be removed in a future version. "
                f"Please use the new collection name directly."
            )
            return resolved, True

        return collection_name, False

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Alias resolution failed for '{collection_name}': {e}")
        return collection_name, False


async def ensure_collection_exists(collection_name: str, project_id: str | None = None) -> bool:
    """
    Ensure a collection exists, create if it doesn't.

    REFACTORED (Task 375.4): Now uses DaemonClient.create_collection_v2() for writes.
    REFACTORED (Task 382.9): Now uses async qdrant_client operations.
    Falls back to direct qdrant_client if daemon unavailable.

    Args:
        collection_name: Name of the collection to ensure exists
        project_id: Project identifier for daemon. Defaults to current project.

    Returns:
        True if collection exists or was created successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    # First check if collection exists (read-only, async call)
    try:
        await _maybe_await(qdrant_client.get_collection(collection_name))
        return True
    except Exception:
        # Collection doesn't exist, need to create it
        pass

    # Get project_id if not provided
    if project_id is None:
        project_id = calculate_tenant_id(str(Path.cwd()))

    # Create collection via daemon (First Principle 10: daemon-only writes)
    if not daemon_client:
        logger.error(f"Cannot create collection '{collection_name}': daemon not connected")
        return False

    vector_size = DEFAULT_COLLECTION_CONFIG["vector_size"]
    distance_metric = DEFAULT_COLLECTION_CONFIG["distance"]

    async def _daemon_create_collection_v2() -> Any:
        try:
            return await daemon_client.create_collection_v2(
                collection_name=collection_name,
                project_id=project_id,
                # config=None uses daemon defaults (384 vectors, Cosine, indexing enabled)
            )
        except TypeError as e:
            message = str(e)
            # Backward compatibility for daemon clients without project_id parameter.
            if "project_id" in message:
                try:
                    return await daemon_client.create_collection_v2(
                        collection_name=collection_name,
                    )
                except TypeError as inner:
                    if "vector_size" in str(inner) or "distance_metric" in str(inner):
                        return await daemon_client.create_collection_v2(
                            collection_name, vector_size, distance_metric
                        )
                    raise
            if "vector_size" in message or "distance_metric" in message:
                return await daemon_client.create_collection_v2(
                    collection_name, vector_size, distance_metric
                )
            raise

    try:
        response = await _daemon_create_collection_v2()
        if response.success:
            logger.info(f"Collection '{collection_name}' created via daemon")
            return True
        else:
            logger.error(
                f"Daemon failed to create collection '{collection_name}': {response.error_message}"
            )
            return False
    except DaemonConnectionError as e:
        logger.error(f"Daemon connection error creating collection '{collection_name}': {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to create collection {collection_name}: {e}")
        return False

def determine_collection_name(
    content: str = "",
    source: str = "user_input",
    file_path: str = None,
    url: str = None,
    collection: str = None,
    project_name: str = None
) -> str:
    """
    Determine appropriate collection name based on content and context.

    DEPRECATED: This function maintains backwards compatibility but the new
    architecture (Task 374.6) uses a single _{project_id} collection per project.

    All files now go to the same collection with differentiation via metadata fields:
    - file_type: "code", "test", "docs", "config", "data", "build", "other"
    - branch: Current Git branch name
    - project_id: 12-char hex project identifier

    For MCP server operations, prefer get_project_collection() instead.
    """
    if collection:
        return collection

    # Use new single-collection architecture
    return get_project_collection()

async def generate_embeddings(text: str) -> list[float]:
    """Generate embeddings for text using non-blocking async execution."""
    if not embedding_model:
        await initialize_components()

    # FastEmbed embed() is CPU-intensive and synchronous - run in thread pool
    # to avoid blocking the event loop
    embeddings = await asyncio.to_thread(lambda: list(embedding_model.embed([text])))
    return embeddings[0].tolist()

def build_metadata_filters(
    filters: dict[str, Any] = None,
    branch: str = None,
    file_type: str = None,
    project_id: str = None,
    tag: str = None,
    exclude_deleted: bool = False
) -> Filter | None:
    """
    Build Qdrant filter with branch, file_type, project_id, tag, and deletion conditions.

    Task 459: Added tag parameter for dot-separated tag hierarchy filtering.
    Task 460: Added exclude_deleted parameter for additive library deletion policy.

    Args:
        filters: User-provided metadata filters
        branch: Git branch to filter by (None = current branch, "*" = all branches)
        file_type: File type to filter by ("code", "test", "docs", etc.)
        project_id: Project ID to filter by (for multi-tenant unified collections)
        tag: Tag filter (supports main_tag or full_tag matching)
        exclude_deleted: If True, exclude documents with deleted=true (Task 460)

    Returns:
        Qdrant Filter object or None if no filters
    """
    from qdrant_client.models import MatchText

    conditions = []

    # Add project_id filter for multi-tenant collections (Task 396)
    if project_id:
        conditions.append(FieldCondition(key="project_id", match=MatchValue(value=project_id)))

    # Add branch filter (always include unless branch="*")
    if branch != "*":
        if branch is None:
            # Detect current branch
            branch = get_current_branch(Path.cwd())
        conditions.append(FieldCondition(key="branch", match=MatchValue(value=branch)))

    # Add file_type filter if specified
    if file_type:
        conditions.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))

    # Task 459: Add tag filter for dot-separated tag hierarchy
    # Supports filtering by main_tag (prefix match) or exact full_tag match
    if tag:
        # Check if tag is a prefix or exact match
        # If tag contains a dot, it's likely a full_tag (exact match)
        # If no dot, it's a main_tag (can use prefix matching on full_tag)
        if "." in tag:
            # Full tag - exact match on full_tag field
            conditions.append(FieldCondition(key="full_tag", match=MatchValue(value=tag)))
        else:
            # Main tag only - match on main_tag field for prefix behavior
            conditions.append(FieldCondition(key="main_tag", match=MatchValue(value=tag)))

    # Task 460: Add deletion filter for additive library deletion policy
    # When exclude_deleted=True, filter out documents marked as deleted
    if exclude_deleted:
        conditions.append(FieldCondition(key="deleted", match=MatchValue(value=False)))

    # Add user-provided filters
    if filters:
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=conditions) if conditions else None


def _detect_file_type(file_path: str) -> str:
    """
    Detect file type from file path for metadata tagging.

    Args:
        file_path: Path to the file

    Returns:
        File type: "code", "test", "docs", "config", "data", "build", or "other"
    """
    path = Path(file_path)
    name = path.name.lower()
    suffix = path.suffix.lower()

    # Test files
    if "test" in name or "spec" in name or name.startswith("test_"):
        return "test"

    # Documentation
    if suffix in (".md", ".rst", ".txt", ".adoc", ".asciidoc"):
        return "docs"
    if name in ("readme", "changelog", "contributing", "license"):
        return "docs"

    # Configuration
    if suffix in (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"):
        return "config"
    if name in ("dockerfile", ".dockerignore", ".gitignore", ".env", "makefile"):
        return "config"

    # Build files
    if suffix in (".lock", ".sum"):
        return "build"
    if name in ("cargo.toml", "pyproject.toml", "package.json", "go.mod", "build.gradle"):
        return "build"

    # Data files
    if suffix in (".csv", ".parquet", ".arrow", ".avro", ".db", ".sqlite"):
        return "data"

    # Code files (default for programming language extensions)
    code_extensions = {
        ".py", ".rs", ".go", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
        ".h", ".hpp", ".swift", ".kt", ".rb", ".php", ".cs", ".lua", ".scala",
        ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss", ".less"
    }
    if suffix in code_extensions:
        return "code"

    return "other"


@app.tool()
@track_tool("store")
async def store(
    content: str,
    title: str = None,
    metadata: dict[str, Any] = None,
    collection: str = None,
    source: str = "user_input",
    document_type: str = "text",
    file_path: str = None,
    url: str = None,
    project_name: str = None,
    file_type: str = None
) -> dict[str, Any]:
    """
    Store any type of content in the unified multi-tenant vector database.

    NEW: Task 397 - Multi-tenant storage with automatic project_id tagging
    - All content stored in unified 'projects' collection (canonical per ADR-001)
    - Automatic project_id tagging from session context
    - File type differentiation via metadata
    - Enables cross-project search while maintaining project isolation

    Storage location:
    - All project content → 'projects' collection
    - project_id automatically set from current session
    - Differentiation via metadata: file_type, branch, source

    DAEMON WRITES (First Principle 10):
    Routes through DaemonClient.ingest_text() for all writes.
    Daemon handles embedding, collection creation, and metadata enrichment.

    Args:
        content: The text content to store
        title: Optional title for the document
        metadata: Additional metadata to attach
        collection: Override collection (for library/memory storage)
        source: Source type (user_input, scratchbook, file, web, etc.)
        document_type: Type of document (text, code, note, etc.)
        file_path: Path to source file (for file_type detection)
        url: Source URL (for web content)
        project_name: Override automatic project name detection
        file_type: Explicit file type (code, test, docs, config, data, build, other)

    Returns:
        Dict with document_id, collection, project_id, and storage confirmation
    """
    await initialize_components()

    # Get project_id from session context or compute from current path (Task 397)
    global _session_project_id
    project_id = _session_project_id or calculate_tenant_id(str(Path.cwd()))

    # Task 457: Track if project activation warning should be shown
    # Warning applies when storing to project-scoped collection without activation
    activation_warning = None

    # Determine target collection based on override or default to unified projects
    if collection:
        # Explicit collection override (e.g., for libraries or memory)
        target_collection = collection
    else:
        # Default: use unified 'projects' collection (Task 397, ADR-001)
        target_collection = CANONICAL_COLLECTIONS["projects"]
        # Task 457: Warn if storing to project collection without activation
        if not is_project_activated():
            activation_warning = get_activation_warning()

    # Detect file_type from file_path if not explicitly provided
    if not file_type and file_path:
        file_type = _detect_file_type(file_path)
    elif not file_type:
        # Default based on source
        source_to_type = {
            "user_input": "other",
            "scratchbook": "other",
            "file": "code",
            "web": "docs",
            "note": "other",
        }
        file_type = source_to_type.get(source, "other")

    # Get current branch for metadata
    current_branch = get_current_branch(Path.cwd())

    # Task 459: Generate dot-separated tag hierarchy for project content
    # Format: main_tag.sub_tag where main_tag=project_id, sub_tag=branch
    main_tag = generate_main_tag(project_id=project_id)
    full_tag = generate_full_tag(main_tag, current_branch) if main_tag else None

    # Prepare metadata with project_id for multi-tenant filtering
    doc_metadata = {
        "title": title or f"Document {uuid.uuid4().hex[:8]}",
        "project_id": project_id,  # Critical for multi-tenant filtering
        "source": source,
        "document_type": document_type,
        "file_type": file_type,  # For file type filtering
        "branch": current_branch,  # For branch filtering
        "main_tag": main_tag,  # Task 459: Hierarchical tag (project_id)
        "full_tag": full_tag,  # Task 459: Full dot-separated tag (project_id.branch)
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": project_name or await get_project_name(),
        "content_preview": content[:200] + "..." if len(content) > 200 else content
    }

    if file_path:
        doc_metadata["file_path"] = file_path
        doc_metadata["file_name"] = Path(file_path).name
    if url:
        doc_metadata["url"] = url
        doc_metadata["domain"] = urlparse(url).netloc
    if metadata:
        doc_metadata.update(metadata)

    # Collection basename for daemon - always "projects" for unified collection
    collection_basename = "projects"
    tenant_id = project_id

    # ============================================================================
    # DAEMON WRITE BOUNDARY (First Principle 10)
    # ============================================================================
    # All Qdrant writes MUST go through daemon. Fallback to direct writes only
    # when daemon is unavailable (logged as warning with fallback_mode flag).
    # See module docstring "Write Path Architecture" for complete documentation.
    # ============================================================================

    # Use DaemonClient for ingestion if available
    if daemon_client:
        try:
            response = await daemon_client.ingest_text(
                content=content,
                collection_basename=collection_basename,
                tenant_id=tenant_id,
                metadata=doc_metadata,
                chunk_text=True
            )

            # Task 458: Update daemon state to AVAILABLE on successful write
            await update_daemon_state(success=True)

            # Task 452: Record successful operation for health monitoring
            record_operation_result(success=True)
            manage_cache_size()  # Task 452: Prevent memory leaks

            result = {
                "success": True,
                "document_id": response.document_id,
                "collection": target_collection,
                "project_id": project_id,  # Task 397: Include for multi-tenant reference
                "title": doc_metadata["title"],
                "content_length": len(content),
                "chunks_created": response.chunks_created,
                "file_type": file_type,
                "branch": current_branch,
                "main_tag": main_tag,  # Task 459: Tag hierarchy
                "full_tag": full_tag,  # Task 459: Full dot-separated tag
                "metadata": doc_metadata
            }
            # Task 457: Include activation warning if applicable
            if activation_warning:
                result["project_activation_warning"] = activation_warning
            return result
        except DaemonConnectionError as e:
            # Task 458: Update daemon state to UNRESPONSIVE on connection error
            await update_daemon_state(success=False)

            # Task 452: Record failed operation for health monitoring
            record_operation_result(success=False)

            return {
                "success": False,
                "error": f"Failed to store document via daemon: {str(e)}",
                "daemon_available": False,
                "suggestion": "Content will be queued for later processing if state_manager is available."
            }
    else:
        # Fallback: Queue content to unified_queue for later daemon processing (Task 37/ADR-002)
        # Instead of direct Qdrant writes (which violate First Principle 10),
        # queue content in unified_queue for daemon to process when available.
        try:
            if state_manager:
                # Queue content for daemon to process later via unified_queue
                # Task 37: Use enqueue_unified() for consolidated queue system
                queue_id, is_new = await state_manager.enqueue_unified(
                    item_type=UnifiedQueueItemType.CONTENT,
                    op=UnifiedQueueOperation.INGEST,
                    tenant_id=project_id,
                    collection=target_collection,
                    payload={
                        "content": content,
                        "source_type": source,
                        "main_tag": main_tag,
                        "full_tag": full_tag,
                    },
                    priority=8,  # Default priority for user content
                    branch=current_branch,
                    metadata=doc_metadata,
                )

                if is_new:
                    logger.warning(
                        f"Daemon unavailable - content queued to unified_queue: {queue_id}"
                    )
                else:
                    logger.debug(f"Content already queued (idempotency): {queue_id}")

                result = {
                    "success": True,
                    "queue_id": queue_id,
                    "queued": True,
                    "collection": target_collection,
                    "project_id": project_id,
                    "title": doc_metadata["title"],
                    "content_length": len(content),
                    "file_type": file_type,
                    "branch": current_branch,
                    "main_tag": main_tag,  # Task 459: Tag hierarchy
                    "full_tag": full_tag,  # Task 459: Full dot-separated tag
                    "fallback_mode": "unified_queue",
                    "message": "Content queued for daemon processing. Will be ingested when daemon is available."
                }
                # Task 457: Include activation warning if applicable
                if activation_warning:
                    result["project_activation_warning"] = activation_warning
                return result
            else:
                # No state_manager available - cannot proceed (ADR-002)
                # Direct Qdrant writes are PROHIBITED per daemon-only write policy.
                # Both daemon and SQLite queue must be unavailable for this to happen.
                logger.error(
                    "Neither daemon nor state_manager available - "
                    "cannot store content (ADR-002 prohibits direct Qdrant writes)"
                )
                return {
                    "success": False,
                    "error": "storage_unavailable",
                    "message": (
                        "Cannot store content: both daemon and SQLite queue are unavailable. "
                        "Start the daemon with: wqm service start"
                    ),
                    "suggestion": "Ensure daemon is running or SQLite state database is accessible.",
                    "adr_reference": "ADR-002: Daemon-Only Write Policy"
                }
        except Exception as e:
            logger.error(f"Failed to store/queue document: {e}")
            # Task 452: Record failed operation
            record_operation_result(success=False)
            return {
                "success": False,
                "error": f"Failed to store document: {str(e)}"
            }

def _merge_with_rrf(
    results_lists: list[list[dict[str, Any]]],
    k: int = 60
) -> list[dict[str, Any]]:
    """
    Merge multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective method for combining ranked lists.
    Formula: RRF(d) = Σ 1 / (k + rank(d))

    Args:
        results_lists: List of result lists, each sorted by score
        k: Constant to prevent high-ranked items from dominating (default: 60)

    Returns:
        Merged results sorted by RRF score
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, dict[str, Any]] = {}

    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            result_id = str(result["id"])
            rrf_scores[result_id] = rrf_scores.get(result_id, 0) + 1 / (k + rank)
            # Keep the result with highest original score
            if result_id not in result_map or result["score"] > result_map[result_id]["score"]:
                result_map[result_id] = result

    # Sort by RRF score and return
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [
        {**result_map[rid], "rrf_score": rrf_scores[rid]}
        for rid in sorted_ids
    ]


@app.tool()
@track_tool("search")
async def search(
    query: str,
    collection: str = None,
    project_name: str = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.3,
    filters: dict[str, Any] = None,
    branch: str = None,
    file_type: str = None,
    workspace_type: str = None,
    scope: str = "project",
    include_libraries: bool = False,
    tag: str = None,
    include_deleted: bool = False
) -> dict[str, Any]:
    """
    Search across collections with hybrid semantic + keyword matching.

    NEW: Task 396 - Multi-tenant filtering with scope parameter
    - scope="project": Filter by current project_id (default, most focused)
    - scope="global": Search all projects (no project_id filter)
    - scope="all": Search projects + libraries collections (broadest)
    - include_libraries: Also search 'libraries' collection

    NEW: Task 459 - Dot-separated tag hierarchy filtering
    - tag="project_id": Filter by main tag (all branches of a project)
    - tag="project_id.branch": Filter by full tag (specific project+branch)
    - tag="numpy": Filter library by name
    - tag="numpy.1.24.0": Filter library by specific version

    NEW: Task 460 - Additive library deletion policy (ADR-001)
    - Libraries marked as deleted are excluded from search by default
    - Set include_deleted=True to search deleted libraries
    - Use manage(action="restore_deleted_library") to restore deleted libraries

    Architecture:
    - Searches unified 'projects' collection with project_id filtering
    - Optional parallel search in 'libraries' collection
    - Results merged using Reciprocal Rank Fusion (RRF)
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified
    - Filters by tag for hierarchical organization (Task 459)
    - Excludes deleted libraries by default (Task 460)

    Search modes:
    - mode="hybrid" -> combines semantic and keyword search
    - mode="semantic" -> pure vector similarity search
    - mode="exact" -> keyword/symbol exact matching

    Examples:
        # Search current project only (default)
        search(query="authentication", scope="project")

        # Search all projects
        search(query="login", scope="global")

        # Search everything including libraries
        search(query="numpy array", scope="all")

        # Search project + include libraries
        search(query="datetime", include_libraries=True)

        # Search by tag (Task 459)
        search(query="config", tag="a1b2c3d4e5f6")              # All branches
        search(query="config", tag="a1b2c3d4e5f6.main")         # main branch only
        search(query="array", tag="numpy", include_libraries=True)  # numpy library

        # Include deleted libraries in search (Task 460)
        search(query="old_lib", include_libraries=True, include_deleted=True)

    Args:
        query: Search query text
        collection: Specific collection to search (overrides scope-based selection)
        project_name: Search within specific project collections
        mode: Search mode - "hybrid", "semantic", "exact", or "keyword"
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0.0-1.0)
        filters: Additional metadata filters
        branch: Git branch to search (None=current, "*"=all branches)
        file_type: File type filter ("code", "test", "docs", "config", "data", "build", "other")
        workspace_type: DEPRECATED - use file_type instead
        scope: Search scope - "project" (default), "global", or "all"
        include_libraries: Also search libraries collection (merged with RRF)
        tag: Tag filter for hierarchical filtering (Task 459, e.g., "project_id" or "project_id.branch")
        include_deleted: If True, include deleted libraries in search (Task 460, default: False)

    Returns:
        Dict with search results, metadata, and performance info
    """
    await initialize_components()

    # Handle deprecated workspace_type parameter
    if workspace_type and not file_type:
        # Map workspace_type to file_type
        workspace_to_file_type = {
            "code": "code",
            "docs": "docs",
            "notes": "other",
            "scratchbook": "other",
            "memory": "other",
        }
        file_type = workspace_to_file_type.get(workspace_type, "other")

    # Validate scope parameter (Task 396)
    valid_scopes = ("project", "global", "all")
    if scope not in valid_scopes:
        return {
            "success": False,
            "error": f"Invalid scope: {scope}. Must be one of: {', '.join(valid_scopes)}",
            "results": []
        }

    # Task 457: Track if project activation warning should be shown
    # Warning applies when using project scope without activation
    activation_warning = None
    if scope == "project" and not is_project_activated():
        activation_warning = get_activation_warning()

    # Task 458: Check daemon availability for project-scoped searches
    # When daemon is marked as unresponsive, verify and potentially return error
    if scope == "project" and not is_daemon_available():
        # Try to verify daemon is actually unresponsive
        daemon_is_available = await check_daemon_availability()
        if not daemon_is_available:
            return {
                "success": False,
                "error": get_daemon_unavailable_message(),
                "daemon_available": False,
                "scope": scope,
                "suggestion": "Use scope='global' with include_libraries=True to search without daemon dependency.",
                "results": []
            }

    # Determine current project_id for filtering
    current_project_id = calculate_tenant_id(str(Path.cwd()))

    # Track if alias was used for deprecation warning in response
    alias_used = False

    # Determine search collections based on scope (Task 396)
    # If explicit collection is provided, resolve any alias first (Task 405)
    if collection:
        resolved_collection, alias_used = await resolve_collection_alias(collection)
        search_collections = [resolved_collection]
        # Don't apply project_id filter for explicit collections
        project_filter_id = None
    else:
        # Use unified collections based on scope
        if scope == "project":
            # Search only current project in unified collection
            search_collections = [CANONICAL_COLLECTIONS["projects"]]
            project_filter_id = current_project_id
        elif scope == "global":
            # Search all projects (no project_id filter)
            search_collections = [CANONICAL_COLLECTIONS["projects"]]
            project_filter_id = None
        else:  # scope == "all"
            # Search both projects and libraries
            search_collections = [CANONICAL_COLLECTIONS["projects"], CANONICAL_COLLECTIONS["libraries"]]
            project_filter_id = None

        # Add libraries collection if requested
        if include_libraries and CANONICAL_COLLECTIONS["libraries"] not in search_collections:
            search_collections.append(CANONICAL_COLLECTIONS["libraries"])

    # Build metadata filters with branch, file_type, project_id, and tag (Task 459)
    search_filter = build_metadata_filters(
        filters=filters,
        branch=branch,
        file_type=file_type,
        project_id=project_filter_id,
        tag=tag
    )

    # For libraries, we don't filter by branch (they're external documentation)
    # Task 459: Library tag filtering uses library_name.version format
    # Task 460: Exclude deleted libraries by default (additive deletion policy)
    library_filter = build_metadata_filters(
        filters=filters,
        branch="*",  # Don't filter libraries by branch
        file_type=file_type,
        project_id=None,  # Libraries have library_name, not project_id
        tag=tag,  # Task 459: Allow tag filtering for libraries (e.g., "numpy.1.24.0")
        exclude_deleted=not include_deleted  # Task 460: Exclude deleted unless explicitly requested
    )

    # Execute search based on mode
    search_start = datetime.now()
    all_collection_results = []

    async def search_single_collection(
        coll: str,
        filter_to_use: Filter | None
    ) -> list[dict[str, Any]]:
        """Search a single collection and return formatted results."""
        results = []

        try:
            # Check collection exists
            if not await ensure_collection_exists(coll):
                return results

            if mode in ["semantic", "hybrid"]:
                # Generate query embeddings for semantic search
                query_embeddings = await generate_embeddings(query)

                # Perform vector search (async)
                search_results = await _maybe_await(
                    qdrant_client.search(
                        collection_name=coll,
                        query_vector=query_embeddings,
                        query_filter=filter_to_use,
                        limit=limit,
                        score_threshold=score_threshold,
                    )
                )

                # Convert results
                for hit in search_results:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "collection": coll,
                        "content": hit.payload.get("content", ""),
                        "title": hit.payload.get("title", ""),
                        "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
                    }
                    results.append(result)

            if mode in ["exact", "keyword", "hybrid"]:
                # For keyword/exact search, use scroll to find text matches (async)
                scroll_results = await _maybe_await(
                    qdrant_client.scroll(
                        collection_name=coll,
                        scroll_filter=filter_to_use,
                        limit=limit * 2,  # Get more for filtering
                    )
                )

                # Filter results by keyword match
                query_lower = query.lower()
                for point in scroll_results[0]:  # scroll returns (points, next_page_offset)
                    content = point.payload.get("content", "").lower()
                    if query_lower in content:
                        # Simple relevance scoring based on keyword frequency
                        keyword_score = content.count(query_lower) / len(content.split()) if content else 0

                        result = {
                            "id": point.id,
                            "score": min(keyword_score * 10, 1.0),  # Normalize to 0-1
                            "collection": coll,
                            "content": point.payload.get("content", ""),
                            "title": point.payload.get("title", ""),
                            "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                        }
                        results.append(result)

        except Exception as e:
            # Log but don't fail the entire search if one collection fails
            pass

        return results

    try:
        # Execute searches in parallel across collections (Task 396)
        search_tasks = []
        for coll in search_collections:
            # Use library_filter for libraries collection, search_filter for others
            if coll == CANONICAL_COLLECTIONS["libraries"]:
                search_tasks.append(search_single_collection(coll, library_filter))
            else:
                search_tasks.append(search_single_collection(coll, search_filter))

        # Run all searches concurrently
        collection_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect results from each collection
        for result in collection_results:
            if isinstance(result, list):
                all_collection_results.append(result)

        # Merge results using RRF if multiple collections searched
        if len(all_collection_results) > 1:
            merged_results = _merge_with_rrf(all_collection_results)
        elif all_collection_results:
            # Single collection - just flatten and sort
            merged_results = sorted(
                all_collection_results[0],
                key=lambda x: x["score"],
                reverse=True
            )
        else:
            merged_results = []

        # Deduplicate by ID (in case same doc appears in multiple collections)
        seen_ids = set()
        unique_results = []
        for result in merged_results:
            result_id = str(result["id"])
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        # Limit final results
        final_results = unique_results[:limit]

        search_duration = (datetime.now() - search_start).total_seconds()

        # Record search metrics (Task 412.10)
        record_search_scope(scope)
        record_search_results(scope, len(final_results), search_duration)

        # Task 452: Record successful operation for health monitoring
        record_operation_result(success=True)

        response = {
            "success": True,
            "query": query,
            "mode": mode,
            "scope": scope,
            "collections_searched": search_collections,
            "total_results": len(final_results),
            "results": final_results,
            "search_time_ms": round(search_duration * 1000, 2),
            "filters_applied": {
                "project_id": project_filter_id,
                "branch": branch if branch != "*" else (get_current_branch(Path.cwd()) if scope == "project" else "*"),
                "file_type": file_type,
                "custom": filters or {}
            }
        }

        # Add deprecation warning if alias was used (Task 405)
        if alias_used:
            response["_deprecation_warning"] = (
                f"Collection '{collection}' is an alias. "
                f"Old collection names will be removed in a future version. "
                f"Please update your code to use the new collection name."
            )

        # Task 457: Include activation warning if applicable
        if activation_warning:
            response["project_activation_warning"] = activation_warning

        return response

    except Exception as e:
        # Task 452: Record failed operation for health monitoring
        record_operation_result(success=False)
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": []
        }

@app.tool()
@track_tool("manage")
async def manage(
    action: str,
    collection: str = None,
    name: str = None,
    project_name: str = None,
    config: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Manage collections, system status, and configuration.

    Actions determined by the 'action' parameter:
    - "list_collections" -> list all collections with stats
    - "create_collection" -> create new collection (name required)
    - "delete_collection" -> delete collection (name required)
    - "workspace_status" -> system status and health check
    - "collection_info" -> detailed info about specific collection
    - "init_project" -> register project in unified 'projects' collection
    - "cleanup" -> remove empty collections and optimize
    - "activate_project" -> register project with daemon, start heartbeat (Task 457)
    - "deactivate_project" -> stop heartbeat, deprioritize project (Task 457)
    - "mark_library_deleted" -> mark library as deleted without removing (Task 460)
    - "restore_deleted_library" -> restore previously deleted library (Task 460)
    - "list_deleted_libraries" -> list all libraries marked as deleted (Task 460)

    Project Activation (Task 457 / ADR-001):
    - Projects must be explicitly activated for high-priority daemon processing
    - activate_project: Register project with daemon, start heartbeat
    - deactivate_project: Stop heartbeat, deprioritize project, clear session state
    - Using scope="project" without activation generates warnings

    Additive Library Deletion (Task 460 / ADR-001):
    - Libraries are NEVER physically deleted from Qdrant
    - mark_library_deleted: Sets deleted=true, deleted_at=timestamp
    - restore_deleted_library: Clears deletion markers, restores searchability
    - list_deleted_libraries: Lists all libraries currently marked as deleted
    - Deleted libraries are excluded from search by default (use include_deleted=True)
    - Re-ingestion of a library automatically clears deletion markers

    Multi-Tenant Architecture:
    - init_project registers the current project's tenant_id in 'projects' collection
    - The 'projects' and 'libraries' collections are unified (multi-tenant per ADR-001)
    - create_collection can create additional user collections

    Args:
        action: Management action to perform
        collection: Target collection name (for collection-specific actions)
        name: Name for new collections or operations (library name for deletion actions)
        project_name: Project context for workspace operations
        config: Additional configuration for operations

    Returns:
        Dict with action results and status information
    """
    await initialize_components()
    logger = logging.getLogger(__name__)

    # Task 457: Declare globals for session state modification in activate/deactivate actions
    global _session_project_id, _session_project_path, _session_heartbeat

    try:
        if action == "list_collections":
            collections_response = await _maybe_await(qdrant_client.get_collections())
            collections_info = []

            for col in collections_response.collections:
                try:
                    col_info = await _maybe_await(qdrant_client.get_collection(col.name))
                    collections_info.append({
                        "name": col.name,
                        "points_count": col_info.points_count,
                        "segments_count": col_info.segments_count,
                        "status": col_info.status.value,
                        "vector_size": col_info.config.params.vectors.size,
                        "distance": col_info.config.params.vectors.distance.value
                    })
                except Exception:
                    collections_info.append({
                        "name": col.name,
                        "status": "error_getting_info"
                    })

            return {
                "success": True,
                "action": action,
                "collections": collections_info,
                "total_collections": len(collections_info)
            }

        elif action == "create_collection":
            if not name:
                return {"success": False, "error": "Collection name required for create action"}

            collection_config = config or DEFAULT_COLLECTION_CONFIG
            vector_size = collection_config.get(
                "vector_size", DEFAULT_COLLECTION_CONFIG["vector_size"]
            )
            distance_metric = collection_config.get(
                "distance", DEFAULT_COLLECTION_CONFIG["distance"]
            )

            # Get project_id from project_name parameter or current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10, Task 37/ADR-002)
            # ============================================================================
            # Collection creation MUST go through daemon. When daemon unavailable,
            # queue to unified_queue instead of direct Qdrant writes.
            # ============================================================================

            if not daemon_client:
                # Task 37: Queue to unified_queue instead of direct Qdrant writes
                if state_manager:
                    try:
                        queue_id, is_new = await state_manager.enqueue_unified(
                            item_type=UnifiedQueueItemType.PROJECT,
                            op=UnifiedQueueOperation.INGEST,
                            tenant_id=project_id,
                            collection=name,
                            payload={
                                "action": "create_collection",
                                "collection_name": name,
                                "vector_size": vector_size,
                                "distance_metric": str(distance_metric),
                            },
                            priority=9,  # High priority for collection operations
                        )

                        if is_new:
                            logger.warning(
                                f"Daemon unavailable - collection creation queued to unified_queue: {queue_id}"
                            )
                        else:
                            logger.debug(f"Collection creation already queued (idempotency): {queue_id}")

                        return {
                            "success": True,
                            "action": action,
                            "queue_id": queue_id,
                            "queued": True,
                            "collection_name": name,
                            "message": f"Collection '{name}' creation queued for daemon processing",
                            "fallback_mode": "unified_queue",
                        }
                    except Exception as e:
                        logger.error(f"Failed to queue collection creation: {e}")
                        return {
                            "success": False,
                            "action": action,
                            "error": f"Failed to queue collection creation: {e}",
                        }
                else:
                    # No state_manager available - cannot proceed (ADR-002)
                    logger.error(
                        "Neither daemon nor state_manager available - "
                        "cannot create collection (ADR-002 prohibits direct Qdrant writes)"
                    )
                    return {
                        "success": False,
                        "action": action,
                        "error": "storage_unavailable",
                        "message": (
                            "Cannot create collection: both daemon and SQLite queue are unavailable. "
                            "Start the daemon with: wqm service start"
                        ),
                        "suggestion": "Ensure daemon is running or SQLite state database is accessible.",
                        "adr_reference": "ADR-002: Daemon-Only Write Policy"
                    }

            try:
                try:
                    response = await daemon_client.create_collection_v2(
                        collection_name=name,
                        project_id=project_id,
                        # config=None uses daemon defaults (384 vectors, Cosine, indexing enabled)
                    )
                except TypeError as e:
                    message = str(e)
                    # Backward compatibility for daemon clients without project_id parameter.
                    if "project_id" in message:
                        try:
                            response = await daemon_client.create_collection_v2(
                                collection_name=name,
                            )
                        except TypeError as inner:
                            if "vector_size" in str(inner) or "distance_metric" in str(inner):
                                response = await daemon_client.create_collection_v2(
                                    name, vector_size, distance_metric
                                )
                            else:
                                raise
                    elif "vector_size" in message or "distance_metric" in message:
                        response = await daemon_client.create_collection_v2(
                            name, vector_size, distance_metric
                        )
                    else:
                        raise

                if response.success:
                    return {
                        "success": True,
                        "action": action,
                        "collection_name": name,
                        "message": f"Collection '{name}' created successfully via daemon"
                    }
                else:
                    return {
                        "success": False,
                        "action": action,
                        "error": f"Daemon failed to create collection: {response.error_message}"
                    }
            except DaemonConnectionError as e:
                return {
                    "success": False,
                    "action": action,
                    "error": f"Daemon connection error: {e}"
                }

        elif action == "delete_collection":
            if not name and not collection:
                return {"success": False, "error": "Collection name required for delete action"}

            target_collection = name or collection

            # Get project_id from current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10, Task 37/ADR-002)
            # ============================================================================
            # Collection deletion MUST go through daemon. When daemon unavailable,
            # queue to unified_queue instead of direct Qdrant writes.
            # ============================================================================

            if not daemon_client:
                # Task 37: Queue to unified_queue instead of direct Qdrant writes
                if state_manager:
                    try:
                        queue_id, is_new = await state_manager.enqueue_unified(
                            item_type=UnifiedQueueItemType.DELETE_TENANT,
                            op=UnifiedQueueOperation.DELETE,
                            tenant_id=project_id,
                            collection=target_collection,
                            payload={
                                "action": "delete_collection",
                                "collection_name": target_collection,
                            },
                            priority=9,  # High priority for collection operations
                        )

                        if is_new:
                            logger.warning(
                                f"Daemon unavailable - collection deletion queued to unified_queue: {queue_id}"
                            )
                        else:
                            logger.debug(f"Collection deletion already queued (idempotency): {queue_id}")

                        return {
                            "success": True,
                            "action": action,
                            "queue_id": queue_id,
                            "queued": True,
                            "collection_name": target_collection,
                            "message": f"Collection '{target_collection}' deletion queued for daemon processing",
                            "fallback_mode": "unified_queue",
                        }
                    except Exception as e:
                        logger.error(f"Failed to queue collection deletion: {e}")
                        return {
                            "success": False,
                            "action": action,
                            "error": f"Failed to queue collection deletion: {e}",
                        }
                else:
                    # No state_manager available - cannot proceed (ADR-002)
                    logger.error(
                        "Neither daemon nor state_manager available - "
                        "cannot delete collection (ADR-002 prohibits direct Qdrant writes)"
                    )
                    return {
                        "success": False,
                        "action": action,
                        "error": "storage_unavailable",
                        "message": (
                            "Cannot delete collection: both daemon and SQLite queue are unavailable. "
                            "Start the daemon with: wqm service start"
                        ),
                        "suggestion": "Ensure daemon is running or SQLite state database is accessible.",
                        "adr_reference": "ADR-002: Daemon-Only Write Policy"
                    }

            try:
                await daemon_client.delete_collection_v2(
                    collection_name=target_collection,
                    project_id=project_id,
                )

                return {
                    "success": True,
                    "action": action,
                    "collection_name": target_collection,
                    "message": f"Collection '{target_collection}' deleted successfully via daemon"
                }
            except DaemonConnectionError as e:
                return {
                    "success": False,
                    "action": action,
                    "error": f"Daemon connection error: {e}"
                }

        elif action == "collection_info":
            if not name and not collection:
                return {"success": False, "error": "Collection name required for info action"}

            target_collection = name or collection
            col_info = await _maybe_await(qdrant_client.get_collection(target_collection))

            info = {
                "points_count": col_info.points_count,
                "segments_count": col_info.segments_count,
                "status": col_info.status.value,
                "vector_size": col_info.config.params.vectors.size,
                "distance": col_info.config.params.vectors.distance.value,
                "indexed": col_info.indexed_vectors_count,
                "optimizer_status": col_info.optimizer_status,
            }

            return {
                "success": True,
                "action": action,
                "collection_name": target_collection,
                "collection": {
                    "name": target_collection,
                    **info,
                },
                "info": info,
            }

        elif action == "workspace_status":
            # System health check (async)
            # Task 452: Enhanced health monitoring
            global _last_health_check
            _last_health_check = datetime.now(timezone.utc)

            current_project = project_name or await get_project_name()
            project_collection = get_project_collection()

            # Get collections info (async)
            collections_response = await _maybe_await(qdrant_client.get_collections())

            # Check for project collection (new architecture: single _{project_id})
            project_collections = []
            for col in collections_response.collections:
                if col.name == project_collection:
                    project_collections.append(col.name)
                # Also include legacy collections for backwards compatibility
                elif col.name.startswith(f"{current_project}-"):
                    project_collections.append(col.name)

            # Get Qdrant cluster info (async)
            cluster_info = await _maybe_await(qdrant_client.get_cluster_info())

            # Task 452: Check daemon health
            daemon_health = None
            daemon_available = False
            if daemon_client:
                try:
                    health = await asyncio.wait_for(
                        daemon_client.health_check(),
                        timeout=_DAEMON_CHECK_TIMEOUT_SECS
                    )
                    daemon_health = {
                        "healthy": health.healthy,
                        "version": getattr(health, 'version', 'unknown'),
                        "uptime_seconds": getattr(health, 'uptime_seconds', None),
                    }
                    daemon_available = health.healthy
                except asyncio.TimeoutError:
                    daemon_health = {"healthy": False, "error": "timeout"}
                except Exception as e:
                    daemon_health = {"healthy": False, "error": str(e)}

            # Task 452: Get health metrics
            health_metrics = get_health_metrics()

            # Task 452: Get memory usage info
            import sys
            memory_info = {
                "project_cache_entries": len(project_cache),
                "project_cache_max": _CACHE_MAX_SIZE,
            }
            collection_names = [col.name for col in collections_response.collections]

            # Task 37: Get unified_queue stats for monitoring
            unified_queue_stats = None
            if state_manager:
                try:
                    unified_queue_stats = await state_manager.get_unified_queue_stats()
                except Exception as e:
                    logger.warning(f"Failed to get unified_queue stats: {e}")
                    unified_queue_stats = {"error": str(e)}

            return {
                "success": True,
                "action": action,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_project": current_project,
                "project_collection": project_collection,
                "branch": get_current_branch(Path.cwd()),
                "qdrant_status": "connected",
                "cluster_info": {
                    "peer_id": cluster_info.peer_id,
                    "raft_info": cluster_info.raft_info
                },
                "collections": collection_names,
                "project_collections": project_collections,
                "total_collections": len(collections_response.collections),
                "embedding_model": os.getenv("FASTEMBED_MODEL", DEFAULT_EMBEDDING_MODEL),
                "health_status": "ok",
                # Task 452: Enhanced health monitoring
                "daemon_health": daemon_health,
                "daemon_available": daemon_available,
                "daemon_state": _daemon_state.value,
                "session_activated": is_project_activated(),
                "session_project_id": _session_project_id,
                "health_metrics": health_metrics,
                "memory_info": memory_info,
                # Task 37: Unified queue monitoring
                "unified_queue": unified_queue_stats,
            }

        elif action == "init_project":
            # Initialize project collection (new architecture: single _{project_id}, async)
            target_project = project_name or await get_project_name()
            project_collection = get_project_collection()

            created_collections = []
            if await ensure_collection_exists(project_collection):
                created_collections.append(project_collection)

            return {
                "success": True,
                "action": action,
                "project": target_project,
                "project_collection": project_collection,
                "collections_created": created_collections,
                "message": f"Initialized collection '{project_collection}' for project '{target_project}'"
            }

        elif action == "cleanup":
            # Remove empty collections and optimize (async)
            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection deletion should go through daemon. Fallback to direct writes
            # is allowed when daemon is unavailable (compatibility during rollout).
            # ============================================================================

            use_daemon = daemon_client is not None
            if not use_daemon and not qdrant_client:
                return {
                    "success": False,
                    "action": action,
                    "error": "Daemon not connected and Qdrant client unavailable.",
                }

            collections_response = await _maybe_await(qdrant_client.get_collections())
            cleaned_collections = []
            failed_collections = []

            # Get project_id from current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            for col in collections_response.collections:
                try:
                    col_info = await _maybe_await(qdrant_client.get_collection(col.name))
                    if col_info.points_count == 0:
                        try:
                            if use_daemon:
                                await daemon_client.delete_collection_v2(
                                    collection_name=col.name,
                                    project_id=project_id,
                                )
                            else:
                                await _maybe_await(qdrant_client.delete_collection(col.name))
                            cleaned_collections.append(col.name)
                            logger.info(
                                f"Deleted empty collection '{col.name}'"
                                + (" via daemon" if use_daemon else " directly")
                            )
                        except DaemonConnectionError as e:
                            failed_collections.append(col.name)
                            logger.error(f"Failed to delete collection '{col.name}': {e}")
                        except Exception as e:
                            failed_collections.append(col.name)
                            logger.error(f"Failed to delete collection '{col.name}': {e}")
                except Exception:
                    continue

            return {
                "success": True,
                "action": action,
                "cleaned_collections": cleaned_collections,
                "failed_collections": failed_collections,
                "message": f"Cleaned up {len(cleaned_collections)} empty collections"
            }

        elif action == "activate_project":
            # Task 457: Explicit project activation flow
            # Register project with daemon, start heartbeat, validate project signature

            # Check if already activated
            if is_project_activated():
                return {
                    "success": True,
                    "action": action,
                    "already_activated": True,
                    "project_id": _session_project_id,
                    "project_path": _session_project_path,
                    "message": "Project already activated"
                }

            # Require daemon client for activation
            if not daemon_client:
                return {
                    "success": False,
                    "action": action,
                    "error": "Daemon not connected. Project activation requires daemon."
                }

            # Detect current project
            project_path = str(Path.cwd())
            project_id = calculate_tenant_id(project_path)
            detected_project_name = project_name or await get_project_name()
            git_remote = await _get_git_remote()

            # Register project with daemon
            try:
                response = await daemon_client.register_project(
                    path=project_path,
                    project_id=project_id,
                    name=detected_project_name,
                    git_remote=git_remote
                )

                # Store session state
                _session_project_id = project_id
                _session_project_path = project_path

                # Start heartbeat after successful registration
                _session_heartbeat = SessionHeartbeat(daemon_client, project_id)
                await _session_heartbeat.start()

                logger.info(
                    f"Project activated: {detected_project_name} ({project_id}), "
                    f"priority={response.priority}, sessions={response.active_sessions}"
                )

                return {
                    "success": True,
                    "action": action,
                    "project_id": project_id,
                    "project_name": detected_project_name,
                    "project_path": project_path,
                    "priority": response.priority,
                    "active_sessions": response.active_sessions,
                    "heartbeat_started": True,
                    "message": f"Project '{detected_project_name}' activated with high priority"
                }

            except Exception as e:
                logger.error(f"Failed to activate project: {e}")
                return {
                    "success": False,
                    "action": action,
                    "error": f"Failed to activate project: {str(e)}"
                }

        elif action == "deactivate_project":
            # Task 457: Explicit project deactivation flow
            # Stop heartbeat, deprioritize project, clear session state

            # Check if project is activated
            if not _session_project_id:
                return {
                    "success": True,
                    "action": action,
                    "already_deactivated": True,
                    "message": "No project currently activated"
                }

            deactivated_project_id = _session_project_id
            deactivated_project_path = _session_project_path

            # Stop heartbeat first
            if _session_heartbeat and _session_heartbeat.is_running:
                try:
                    await _session_heartbeat.stop()
                    logger.info(f"Heartbeat stopped for project {deactivated_project_id}")
                except Exception as e:
                    logger.warning(f"Failed to stop heartbeat: {e}")

            # Deprioritize project with daemon
            deprioritize_result = None
            if daemon_client:
                try:
                    response = await daemon_client.deprioritize_project(
                        project_id=deactivated_project_id
                    )
                    deprioritize_result = {
                        "remaining_sessions": response.remaining_sessions,
                        "new_priority": response.new_priority
                    }
                    logger.info(
                        f"Project deprioritized: {deactivated_project_id}, "
                        f"remaining_sessions={response.remaining_sessions}, "
                        f"new_priority={response.new_priority}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to deprioritize project: {e}")

            # Clear session state
            _session_project_id = None
            _session_project_path = None
            _session_heartbeat = None

            return {
                "success": True,
                "action": action,
                "deactivated_project_id": deactivated_project_id,
                "deactivated_project_path": deactivated_project_path,
                "heartbeat_stopped": True,
                "deprioritize_result": deprioritize_result,
                "message": f"Project '{deactivated_project_id}' deactivated"
            }

        elif action == "mark_library_deleted":
            # Task 460: Additive library deletion policy (ADR-001)
            # Mark library documents as deleted rather than physically removing them
            # Preserves historical context, enables undo, and maintains audit trail

            if not name:
                return {
                    "success": False,
                    "action": action,
                    "error": "Library name required for mark_library_deleted action"
                }

            library_name = name
            library_collection = CANONICAL_COLLECTIONS["libraries"]

            try:
                # Check if collection exists
                if not await ensure_collection_exists(library_collection):
                    return {
                        "success": False,
                        "action": action,
                        "error": f"Libraries collection '{library_collection}' does not exist"
                    }

                # Find all documents with matching library_name
                # Use scroll to get all matching documents
                scroll_result = await _maybe_await(
                    qdrant_client.scroll(
                        collection_name=library_collection,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="library_name",
                                    match=MatchValue(value=library_name),
                                ),
                                # Only mark non-deleted documents
                                FieldCondition(
                                    key="deleted",
                                    match=MatchValue(value=False),
                                ),
                            ]
                        ),
                        limit=1000,  # Reasonable batch size
                    )
                )

                documents, _ = scroll_result
                if not documents:
                    # Try without the deleted filter in case deleted field doesn't exist
                    scroll_result = await _maybe_await(
                        qdrant_client.scroll(
                            collection_name=library_collection,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="library_name",
                                        match=MatchValue(value=library_name),
                                    )
                                ]
                            ),
                            limit=1000,
                        )
                    )
                    documents, _ = scroll_result

                if not documents:
                    return {
                        "success": False,
                        "action": action,
                        "error": f"No documents found for library '{library_name}'"
                    }

                # Mark each document as deleted with timestamp
                deletion_timestamp = datetime.now(timezone.utc).isoformat()
                marked_count = 0

                for doc in documents:
                    # Update payload with deletion markers
                    payload_update = {
                        "deleted": True,
                        "deleted_at": deletion_timestamp
                    }

                    await _maybe_await(
                        qdrant_client.set_payload(
                            collection_name=library_collection,
                            payload=payload_update,
                            points=[doc.id],
                        )
                    )
                    marked_count += 1

                logger.info(
                    f"Marked {marked_count} documents as deleted for library '{library_name}'"
                )

                return {
                    "success": True,
                    "action": action,
                    "library_name": library_name,
                    "documents_marked": marked_count,
                    "deleted_at": deletion_timestamp,
                    "message": f"Library '{library_name}' marked as deleted ({marked_count} documents). Use restore_deleted_library to undo."
                }

            except Exception as e:
                logger.error(f"Failed to mark library as deleted: {e}")
                return {
                    "success": False,
                    "action": action,
                    "error": f"Failed to mark library as deleted: {str(e)}"
                }

        elif action == "restore_deleted_library":
            # Task 460: Restore a previously deleted library (ADR-001)
            # Clears deletion markers, making library searchable again

            if not name:
                return {
                    "success": False,
                    "action": action,
                    "error": "Library name required for restore_deleted_library action"
                }

            library_name = name
            library_collection = CANONICAL_COLLECTIONS["libraries"]

            try:
                # Check if collection exists
                if not await ensure_collection_exists(library_collection):
                    return {
                        "success": False,
                        "action": action,
                        "error": f"Libraries collection '{library_collection}' does not exist"
                    }

                # Find all deleted documents with matching library_name
                scroll_result = await _maybe_await(
                    qdrant_client.scroll(
                        collection_name=library_collection,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="library_name",
                                    match=MatchValue(value=library_name),
                                ),
                                FieldCondition(
                                    key="deleted",
                                    match=MatchValue(value=True),
                                ),
                            ]
                        ),
                        limit=1000,
                    )
                )

                documents, _ = scroll_result
                if not documents:
                    return {
                        "success": False,
                        "action": action,
                        "error": f"No deleted documents found for library '{library_name}'"
                    }

                # Clear deletion markers from each document
                restored_count = 0
                restore_timestamp = datetime.now(timezone.utc).isoformat()

                for doc in documents:
                    # Update payload to clear deletion markers
                    payload_update = {
                        "deleted": False,
                        "deleted_at": None,
                        "restored_at": restore_timestamp
                    }

                    await _maybe_await(
                        qdrant_client.set_payload(
                            collection_name=library_collection,
                            payload=payload_update,
                            points=[doc.id],
                        )
                    )
                    restored_count += 1

                logger.info(
                    f"Restored {restored_count} documents for library '{library_name}'"
                )

                return {
                    "success": True,
                    "action": action,
                    "library_name": library_name,
                    "documents_restored": restored_count,
                    "restored_at": restore_timestamp,
                    "message": f"Library '{library_name}' restored ({restored_count} documents)"
                }

            except Exception as e:
                logger.error(f"Failed to restore library: {e}")
                return {
                    "success": False,
                    "action": action,
                    "error": f"Failed to restore library: {str(e)}"
                }

        elif action == "list_deleted_libraries":
            # Task 460: List all libraries marked as deleted (ADR-001)

            library_collection = CANONICAL_COLLECTIONS["libraries"]

            try:
                # Check if collection exists
                if not await ensure_collection_exists(library_collection):
                    return {
                        "success": True,
                        "action": action,
                        "deleted_libraries": [],
                        "message": "Libraries collection does not exist"
                    }

                # Find all deleted documents
                scroll_result = await _maybe_await(
                    qdrant_client.scroll(
                        collection_name=library_collection,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="deleted",
                                    match=MatchValue(value=True),
                                )
                            ]
                        ),
                        limit=1000,
                    )
                )

                documents, _ = scroll_result

                # Group by library_name
                deleted_libraries = {}
                for doc in documents:
                    lib_name = doc.payload.get("library_name", "unknown")
                    deleted_at = doc.payload.get("deleted_at")
                    if lib_name not in deleted_libraries:
                        deleted_libraries[lib_name] = {
                            "library_name": lib_name,
                            "document_count": 0,
                            "deleted_at": deleted_at
                        }
                    deleted_libraries[lib_name]["document_count"] += 1

                return {
                    "success": True,
                    "action": action,
                    "deleted_libraries": list(deleted_libraries.values()),
                    "total_deleted_documents": len(documents),
                    "message": f"Found {len(deleted_libraries)} deleted libraries"
                }

            except Exception as e:
                logger.error(f"Failed to list deleted libraries: {e}")
                return {
                    "success": False,
                    "action": action,
                    "error": f"Failed to list deleted libraries: {str(e)}"
                }

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "available_actions": [
                    "list_collections", "create_collection", "delete_collection",
                    "collection_info", "workspace_status", "init_project", "cleanup",
                    "activate_project", "deactivate_project",
                    "mark_library_deleted", "restore_deleted_library", "list_deleted_libraries"
                ]
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Management action '{action}' failed: {str(e)}"
        }

@app.tool()
@track_tool("retrieve")
async def retrieve(
    document_id: str = None,
    collection: str = None,
    metadata: dict[str, Any] = None,
    limit: int = 10,
    project_name: str = None,
    branch: str = None,
    file_type: str = None,
    scope: str = "project"
) -> dict[str, Any]:
    """
    Retrieve documents directly by ID or metadata without search ranking.

    NEW: Task 398 - Multi-tenant scope parameter with branch filtering
    - scope="project": Filter by current project_id (default, most focused)
    - scope="global": Retrieve from all projects (no project_id filter)
    - scope="all": Retrieve from projects + libraries collections (broadest)
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified

    Retrieval methods determined by parameters:
    - document_id specified -> direct ID lookup
    - metadata specified -> filter-based retrieval
    - collection specified -> limits retrieval to specific collection (overrides scope)
    - branch -> filters by Git branch
    - file_type -> filters by file type
    - scope -> controls multi-tenant collection and filtering

    Args:
        document_id: Direct document ID to retrieve
        collection: Specific collection to retrieve from (overrides scope)
        metadata: Metadata filters for document selection
        limit: Maximum number of documents to retrieve
        project_name: Limit retrieval to project collections (deprecated, use scope)
        branch: Git branch to filter by (None=current, "*"=all branches)
        file_type: File type filter ("code", "test", "docs", etc.)
        scope: Retrieval scope - "project" (default), "global", or "all"

    Returns:
        Dict with retrieved documents and metadata

    Examples:
        # Retrieve by ID from current project
        retrieve(document_id="uuid-123")

        # Retrieve by ID from all projects
        retrieve(document_id="uuid-123", scope="global")

        # Retrieve by metadata from current project
        retrieve(metadata={"file_type": "test"}, branch="develop")

        # Retrieve from all collections (projects + libraries)
        retrieve(metadata={"author": "john"}, scope="all")
    """
    await initialize_components()

    if not document_id and not metadata:
        return {
            "success": False,
            "error": "Either document_id or metadata filters must be provided"
        }

    # Validate scope parameter (Task 398)
    valid_scopes = ("project", "global", "all")
    if scope not in valid_scopes:
        return {
            "success": False,
            "error": f"Invalid scope: {scope}. Must be one of: {', '.join(valid_scopes)}",
            "results": []
        }

    # Task 457: Track if project activation warning should be shown
    # Warning applies when using project scope without activation
    activation_warning = None
    if scope == "project" and not is_project_activated():
        activation_warning = get_activation_warning()

    try:
        results = []

        # Determine current project_id for filtering
        current_project_id = calculate_tenant_id(str(Path.cwd()))

        # Track if alias was used for deprecation warning in response
        alias_used = False

        # Determine retrieval collections based on scope (Task 398)
        project_filter_id = None
        if collection:
            # Resolve any alias first (Task 405)
            resolved_collection, alias_used = await resolve_collection_alias(collection)
            retrieve_collections = [resolved_collection]
            # Don't apply project_id filter for explicit collections
        else:
            # Use unified collections based on scope
            if scope == "project":
                # Retrieve from projects with project_id filter
                retrieve_collections = [CANONICAL_COLLECTIONS["projects"]]
                project_filter_id = current_project_id
            elif scope == "global":
                # Retrieve from all projects (no project_id filter)
                retrieve_collections = [CANONICAL_COLLECTIONS["projects"]]
                project_filter_id = None
            else:  # scope == "all"
                # Retrieve from projects + libraries
                retrieve_collections = [CANONICAL_COLLECTIONS["projects"], CANONICAL_COLLECTIONS["libraries"]]
                project_filter_id = None

        if document_id:
            # Direct ID retrieval - search across all target collections
            for search_collection in retrieve_collections:
                try:
                    points = await _maybe_await(
                        qdrant_client.retrieve(
                            collection_name=search_collection,
                            ids=[document_id],
                        )
                    )

                    if points:
                        point = points[0]

                        # Apply project_id filter for multi-tenant (scope="project")
                        if project_filter_id:
                            doc_project_id = point.payload.get("project_id")
                            if doc_project_id != project_filter_id:
                                continue  # Document not in current project

                        # Apply branch filter to retrieved document
                        if branch != "*":
                            effective_branch = branch if branch else get_current_branch(Path.cwd())
                            doc_branch = point.payload.get("branch")
                            if doc_branch and doc_branch != effective_branch:
                                # Document not on requested branch
                                continue

                        # Apply file_type filter if specified
                        if file_type:
                            doc_file_type = point.payload.get("file_type")
                            if doc_file_type and doc_file_type != file_type:
                                continue

                        result = {
                            "id": point.id,
                            "collection": search_collection,
                            "content": point.payload.get("content", ""),
                            "title": point.payload.get("title", ""),
                            "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                        }
                        results.append(result)
                        break  # Found document, stop searching

                except Exception:
                    pass  # Collection might not exist or ID not found

            if not results:
                # Document not found in any collection
                return {
                    "success": True,
                    "total_results": 0,
                    "results": [],
                    "query_type": "id_lookup",
                    "scope": scope,
                    "collections_searched": retrieve_collections,
                    "message": f"Document not found or filtered out (branch/file_type/project_id)"
                }

        elif metadata:
            # Metadata-based retrieval with branch, file_type, and project_id filters
            for search_collection in retrieve_collections:
                # Build filter conditions including branch, file_type, and project_id
                # Use project_id for projects collection, but not for libraries
                collection_project_id = project_filter_id if search_collection == CANONICAL_COLLECTIONS["projects"] else None

                search_filter = build_metadata_filters(
                    filters=metadata,
                    branch=branch,
                    file_type=file_type,
                    project_id=collection_project_id
                )

                # Retrieve from collection (async)
                try:
                    scroll_result = await _maybe_await(
                        qdrant_client.scroll(
                            collection_name=search_collection,
                            scroll_filter=search_filter,
                            limit=limit - len(results),  # Respect total limit
                        )
                    )

                    points = scroll_result[0]  # scroll returns (points, next_page_offset)

                    for point in points:
                        result = {
                            "id": point.id,
                            "collection": search_collection,
                            "content": point.payload.get("content", ""),
                            "title": point.payload.get("title", ""),
                            "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                        }
                        results.append(result)

                        if len(results) >= limit:
                            break

                except Exception:
                    pass  # Collection might not exist

                if len(results) >= limit:
                    break

        response = {
            "success": True,
            "total_results": len(results),
            "results": results,
            "query_type": "id_lookup" if document_id else "metadata_filter",
            "scope": scope,
            "collections_searched": retrieve_collections,
            "filters_applied": {
                "project_id": project_filter_id,
                "branch": branch if branch == "*" else (branch or get_current_branch(Path.cwd())),
                "file_type": file_type,
                "metadata": metadata or {}
            }
        }

        # Add deprecation warning if alias was used (Task 405)
        if alias_used:
            response["_deprecation_warning"] = (
                f"Collection '{collection}' is an alias. "
                f"Old collection names will be removed in a future version. "
                f"Please update your code to use the new collection name."
            )

        # Task 457: Include activation warning if applicable
        if activation_warning:
            response["project_activation_warning"] = activation_warning

        return response

    except Exception as e:
        return {
            "success": False,
            "error": f"Retrieval failed: {str(e)}",
            "results": []
        }


# Compatibility: expose tool callables for direct invocation in tests.
for _tool_name in ("manage", "store", "search", "retrieve"):
    _tool = globals().get(_tool_name)
    if hasattr(_tool, "fn"):
        globals()[f"{_tool_name}_tool"] = _tool
        globals()[_tool_name] = _tool.fn

def run_server(
    transport: str = typer.Option(
        "stdio", help="Transport protocol (stdio, http, sse, streamable-http)"
    ),
    host: str = typer.Option("127.0.0.1", help="Server host for non-stdio transports"),
    port: int = typer.Option(8000, help="Server port for non-stdio transports"),
) -> None:
    """
    Run the Workspace Qdrant MCP server with specified transport.

    Supports multiple transport protocols for different integration scenarios:
    - stdio: For Claude Desktop and MCP clients (default)
    - http: Standard HTTP REST API
    - sse: Server-Sent Events for streaming
    - streamable-http: HTTP with streaming support
    """
    # Configure server based on transport
    if transport == "stdio":
        # MCP stdio mode - ensure complete silence
        os.environ["WQM_STDIO_MODE"] = "true"
        _detect_stdio_mode()  # Re-apply stdio silencing

    # Run the FastMCP app with specified transport
    if transport == "stdio":
        app.run(transport=transport)
    else:
        app.run(transport=transport, host=host, port=port)

def main() -> None:
    """Console script entry point for UV tool installation and direct execution."""
    typer.run(run_server)

if __name__ == "__main__":
    main()
