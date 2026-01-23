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

Architecture (ADR-001 Multi-Tenant):
    - Canonical collections: projects, libraries, memory (NO underscore prefix)
    - Tenant isolation via project_id/library_name payload filtering (indexed)
    - Branch-scoped queries: All queries filter by Git branch (default: current branch)
    - File type differentiation via metadata: code, test, docs, config, data, build, other
    - Memory collection: global behavioral rules (project-specific via tags)
    - Deprecated: _projects, _libraries, _memory, _{project_id} patterns

Tools:
    1. store - Store any content (documents, notes, code, web content)
    2. search - Hybrid semantic + keyword search with branch and file_type filtering
    3. manage - Collection management, system status, configuration
    4. retrieve - Direct document retrieval by ID or metadata with branch filtering

Example Usage:
    # Store different content types (all go to 'projects' collection with project_id)
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
    manage(action="init_project")                      # Register project in _projects

    # Direct retrieval with branch filtering
    retrieve(document_id="uuid-123")                              # Current branch
    retrieve(metadata={"file_type": "test"}, branch="develop")    # develop branch, tests
    retrieve(scope="all")                                         # All collections

Write Path Architecture (First Principle 10):
    DAEMON-ONLY WRITES: All Qdrant write operations MUST route through the daemon

    Collection Types (ADR-001):
        - projects: Canonical collection, tenant isolation via project_id filter
        - libraries: Canonical collection, isolation via library_name filter
        - memory: Global behavioral rules, project-specific via tags

    Write Priority:
        1. PRIMARY: DaemonClient.ingest_text() / create_collection_v2() / delete_collection_v2()
        2. QUEUE FALLBACK: SQLite content_ingestion_queue (when daemon unavailable)
        3. EXCEPTION: MEMORY collections use direct writes (architectural decision)

    Queue Fallback Architecture (Task 428):
        When daemon is unavailable, writes are queued to SQLite instead of direct Qdrant:
        - Content queued via SQLiteStateManager.enqueue_ingestion()
        - Daemon polls content_ingestion_queue table periodically (5s interval)
        - Idempotency key prevents duplicate ingestion
        - Returns "queued_for_processing: true" in response
        - Status transitions: pending → in_progress → done/failed

    See: FIRST-PRINCIPLES.md (Principle 10), Task 375.6, Task 428
"""

import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager

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
    MatchText,
    MatchValue,
    PointStruct,
    VectorParams,
)


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
from common.core.sqlite_state_manager import SQLiteStateManager, WatchFolderConfig
from common.core.queue_client import QueueOperation
from common.grpc.daemon_client import DaemonClient, DaemonConnectionError
from common.observability.metrics import (
    track_tool, record_search_scope, record_search_results
)
from common.utils.git_utils import get_current_branch
from common.utils.project_detection import calculate_tenant_id

# Error response utilities (Task 449)
from workspace_qdrant_mcp.error_responses import (
    create_error_response,
    create_simple_error_response,
    handle_tool_error,
    validation_error,
    not_found_error,
    daemon_unavailable_error,
    ErrorCode,
    ErrorType,
)

# Global components
qdrant_client: AsyncQdrantClient | None = None
embedding_model = None
daemon_client: DaemonClient | None = None
alias_manager: AliasManager | None = None
state_manager: SQLiteStateManager | None = None
project_cache = {}

# Session lifecycle state
_session_project_id: str | None = None
_session_project_path: str | None = None
_session_heartbeat: "SessionHeartbeat | None" = None

# Daemon availability state (Task 430)
# States: "AVAILABLE" (daemon responding) or "UNRESPONSIVE" (2x 10s attempts failed)
_daemon_state: str = "AVAILABLE"
_DAEMON_CHECK_TIMEOUT_SECS: float = 10.0
_DAEMON_CHECK_MAX_ATTEMPTS: int = 2

# Memory injection state (Task 435)
# Caches injected memory rules to avoid redundant injections
_memory_rules_cache: list[dict[str, Any]] | None = None
_memory_last_injected: datetime | None = None
_MEMORY_INJECTION_STALE_SECS: float = 300.0  # Re-inject after 5 minutes


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


async def check_daemon_availability() -> bool:
    """
    Check if daemon is available with 2-attempt, 10s deadline policy.

    Task 430: Implements daemon availability state machine.

    Attempts up to _DAEMON_CHECK_MAX_ATTEMPTS (2) health checks,
    each with _DAEMON_CHECK_TIMEOUT_SECS (10s) deadline.

    Returns:
        True if daemon responds, False otherwise.

    Side effects:
        Updates global _daemon_state to "AVAILABLE" or "UNRESPONSIVE".
    """
    global _daemon_state, daemon_client

    if daemon_client is None:
        _daemon_state = "UNRESPONSIVE"
        return False

    logger = logging.getLogger(__name__)

    for attempt in range(_DAEMON_CHECK_MAX_ATTEMPTS):
        try:
            # Attempt health check with timeout
            health_response = await asyncio.wait_for(
                daemon_client.health_check(),
                timeout=_DAEMON_CHECK_TIMEOUT_SECS
            )

            # Check if response indicates healthy status
            if hasattr(health_response, 'status') and health_response.status:
                _daemon_state = "AVAILABLE"
                logger.debug(f"Daemon available (attempt {attempt + 1})")
                return True

        except asyncio.TimeoutError:
            logger.warning(
                f"Daemon health check timed out (attempt {attempt + 1}/{_DAEMON_CHECK_MAX_ATTEMPTS})"
            )
        except Exception as e:
            logger.warning(
                f"Daemon health check failed (attempt {attempt + 1}/{_DAEMON_CHECK_MAX_ATTEMPTS}): {e}"
            )

    # All attempts failed
    _daemon_state = "UNRESPONSIVE"
    logger.warning("Daemon marked as UNRESPONSIVE after failed health checks")
    return False


def update_daemon_state(available: bool) -> None:
    """
    Update daemon state based on operation success/failure.

    Task 430: Called after daemon operations to track availability.

    Args:
        available: True if daemon operation succeeded, False otherwise.
    """
    global _daemon_state
    _daemon_state = "AVAILABLE" if available else "UNRESPONSIVE"


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


async def _load_memory_rules(force_refresh: bool = False) -> list[dict[str, Any]]:
    """
    Load memory rules from the memory collection for LLM context injection.

    Task 435: Memory injection on MCP startup and compaction.

    Queries the 'memory' collection for all rules and parses them for
    LLM behavioral context. Rules are sorted by priority (absolute first)
    and cached to avoid redundant queries.

    Args:
        force_refresh: If True, bypass cache and reload from Qdrant

    Returns:
        List of memory rules with metadata for context injection:
        [
            {
                "id": str,
                "rule": str,
                "name": str,
                "category": str,  # preference, behavior, agent
                "authority": str,  # absolute, default
                "scope": list[str],
                "source": str
            }
        ]
    """
    global _memory_rules_cache, _memory_last_injected, qdrant_client

    logger = logging.getLogger(__name__)

    # Check cache validity (unless force refresh)
    if not force_refresh and _memory_rules_cache is not None and _memory_last_injected is not None:
        age = (datetime.now(timezone.utc) - _memory_last_injected).total_seconds()
        if age < _MEMORY_INJECTION_STALE_SECS:
            logger.debug(f"Using cached memory rules (age: {age:.1f}s)")
            return _memory_rules_cache

    # Ensure Qdrant client is initialized
    if qdrant_client is None:
        logger.warning("Qdrant client not initialized, cannot load memory rules")
        return []

    try:
        # Check if memory collection exists
        collections_response = await qdrant_client.get_collections()
        collection_names = {col.name for col in collections_response.collections}

        # Use canonical 'memory' collection (ADR-001)
        memory_collection = CANONICAL_COLLECTIONS["memory"]

        if memory_collection not in collection_names:
            logger.debug(f"Memory collection '{memory_collection}' does not exist yet")
            return []

        # Query all memory rules (limit 1000 for safety)
        # Using scroll for efficiency with large collections
        points = []
        offset = None

        while True:
            scroll_result = await qdrant_client.scroll(
                collection_name=memory_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for injection
            )

            batch_points, next_offset = scroll_result
            points.extend(batch_points)

            if next_offset is None or len(points) >= 1000:
                break
            offset = next_offset

        # Parse rules from points
        rules = []
        for point in points:
            payload = point.payload or {}
            rule_entry = {
                "id": str(point.id) if point.id else None,
                "rule": payload.get("rule", ""),
                "name": payload.get("name", ""),
                "category": payload.get("category", "behavior"),
                "authority": payload.get("authority", "default"),
                "scope": payload.get("scope", []),
                "source": payload.get("source", "unknown"),
            }
            rules.append(rule_entry)

        # Sort rules: absolute authority first, then by name for stability
        authority_order = {"absolute": 0, "default": 1}
        rules.sort(key=lambda r: (authority_order.get(r["authority"], 2), r["name"]))

        # Update cache
        _memory_rules_cache = rules
        _memory_last_injected = datetime.now(timezone.utc)

        logger.info(f"Loaded {len(rules)} memory rules for context injection")
        return rules

    except Exception as e:
        logger.warning(f"Failed to load memory rules: {e}")
        return []


def _format_memory_rules_for_llm(rules: list[dict[str, Any]]) -> str:
    """
    Format memory rules into a string suitable for LLM context injection.

    Task 435: Formats rules for injection into session context.

    Args:
        rules: List of memory rule dictionaries from _load_memory_rules()

    Returns:
        Formatted string with rules organized by authority level
    """
    if not rules:
        return ""

    lines = []
    lines.append("# Memory Rules (Auto-injected)")
    lines.append("")

    # Group by authority level
    absolute_rules = [r for r in rules if r["authority"] == "absolute"]
    default_rules = [r for r in rules if r["authority"] == "default"]

    if absolute_rules:
        lines.append("## Absolute Rules (Non-negotiable)")
        for rule in absolute_rules:
            name = rule.get("name", "")
            text = rule.get("rule", "")
            scope = rule.get("scope", [])
            scope_str = f" [{', '.join(scope)}]" if scope else ""
            lines.append(f"- **{name}**{scope_str}: {text}")
        lines.append("")

    if default_rules:
        lines.append("## Default Rules (Override when explicitly requested)")
        for rule in default_rules:
            name = rule.get("name", "")
            text = rule.get("rule", "")
            scope = rule.get("scope", [])
            scope_str = f" [{', '.join(scope)}]" if scope else ""
            lines.append(f"- **{name}**{scope_str}: {text}")
        lines.append("")

    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app):
    """
    FastMCP lifespan context manager for session lifecycle management.

    Task 395: Multi-tenant session lifecycle
    Task 407: Heartbeat mechanism for session liveness
    Task 429: Explicit project activation (no auto-registration)
    Task 435: Memory injection on startup and compaction

    Architecture (ADR-001):
    - On startup: Initialize daemon client, inject memory rules
    - Memory rules loaded from 'memory' collection and cached (5 min TTL)
    - Projects require explicit activation via manage(action="activate_project")
    - On shutdown: Stop heartbeat, deprioritize project (if activated)

    The daemon uses session registration to:
    - Set HIGH priority for actively-edited projects
    - Track active sessions for crash recovery (via heartbeat)
    - Optimize file watcher resources based on activity

    The heartbeat mechanism (Task 407):
    - Sends periodic heartbeat every 30 seconds
    - Daemon timeout is 60 seconds for orphaned session detection
    - Without heartbeat, crashed sessions are detected and demoted

    Explicit Activation (Task 429):
    - MCP server does NOT auto-register projects on startup
    - Projects require manage(action="activate_project") to register
    - Project-scoped operations (store/search with scope="project") require activation
    - Provides clear error messages with activation instructions if not activated
    """
    global daemon_client, _session_project_id, _session_project_path, _session_heartbeat

    logger = logging.getLogger(__name__)

    # =========================================================================
    # STARTUP: Initialize daemon client only (NO automatic project registration)
    # Task 429: Explicit project activation - no auto-registration
    # =========================================================================
    try:
        # Initialize daemon client if not already done
        if daemon_client is None:
            daemon_client = DaemonClient()
            try:
                await daemon_client.connect()
                logger.info("Daemon client connected - use manage(action='activate_project') to register project")
            except DaemonConnectionError:
                # Daemon connection is optional - server works without it
                daemon_client = None
                logger.warning("Daemon not available - session lifecycle disabled")

        # NOTE: No automatic project registration (Task 429)
        # Projects must be explicitly activated via manage(action="activate_project")
        # This enables:
        # - Clear user intent for which projects to track
        # - Avoids accidental registration of transient directories
        # - Matches ADR-001 explicit activation policy

        # =========================================================================
        # Task 435: Memory injection on startup
        # Load memory rules from the 'memory' collection and inject into session context
        # =========================================================================
        try:
            # Initialize components to ensure Qdrant client is ready
            await initialize_components()

            # Load memory rules
            memory_rules = await _load_memory_rules(force_refresh=True)

            if memory_rules:
                logger.info(
                    f"Memory injection: Loaded {len(memory_rules)} rules "
                    f"(absolute: {len([r for r in memory_rules if r['authority'] == 'absolute'])}, "
                    f"default: {len([r for r in memory_rules if r['authority'] == 'default'])})"
                )
            else:
                logger.debug("Memory injection: No memory rules found (memory collection may be empty)")

        except Exception as mem_error:
            # Memory injection failure is non-fatal - server continues without rules
            logger.warning(f"Memory injection failed (non-fatal): {mem_error}")

    except Exception as e:
        logger.warning(f"Lifespan startup error: {e}")

    # Yield control to the application
    yield

    # =========================================================================
    # SHUTDOWN: Stop heartbeat and deprioritize project with daemon
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

        # Clean up daemon client
        if daemon_client:
            await daemon_client.disconnect()
    except Exception as e:
        logger.warning(f"Lifespan shutdown error: {e}")


# Initialize the FastMCP app with lifespan management
app = FastMCP("Workspace Qdrant MCP", lifespan=lifespan)

# Collection basename mapping for Rust daemon validation
# Maps collection types to valid basenames (non-empty strings)
BASENAME_MAP = {
    "project": "code",      # PROJECT collections: _{project_id}
    "user": "notes",        # USER collections: {basename}-{type}
    "library": "lib",       # LIBRARY collections: _{library_name}
    "memory": "memory",     # MEMORY collections: _memory, _agent_memory
}

# Canonical multi-tenant collection names (ADR-001)
# These are the ONLY canonical collections - see docs/decisions/ADR-001-multi-tenant-architecture.md
# All data stored with tenant isolation via metadata filtering (project_id, library_name)
CANONICAL_COLLECTIONS = {
    "projects": "projects",    # All project code/documents (tenant_id: project_id)
    "libraries": "libraries",  # All library documentation (tenant_id: library_name)
    "memory": "memory",        # Global behavioral rules and instructions
}

# Deprecated collection patterns - will be rejected
# See ADR-001 for migration path
DEPRECATED_PATTERNS = [
    "_projects", "_libraries", "_memory", "_agent_memory",  # Underscore-prefixed
    "__memory", "__system",                                  # Double-underscore system
]


def validate_collection_name(collection_name: str) -> tuple[bool, str | None]:
    """
    Validate collection name against canonical architecture (ADR-001).

    Args:
        collection_name: Collection name to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Check for deprecated patterns
    if collection_name in DEPRECATED_PATTERNS:
        return False, (
            f"Collection '{collection_name}' uses deprecated naming. "
            f"Use canonical names: {list(CANONICAL_COLLECTIONS.values())}. "
            f"See ADR-001 for migration guidance."
        )

    # Check for underscore-prefix patterns (deprecated per ADR-001)
    if collection_name.startswith("_"):
        return False, (
            f"Underscore-prefixed collection '{collection_name}' is deprecated. "
            f"Use canonical names: {list(CANONICAL_COLLECTIONS.values())}. "
            f"See ADR-001 for migration guidance."
        )

    # Check for double-underscore patterns (deprecated per ADR-001)
    if collection_name.startswith("__"):
        return False, (
            f"Double-underscore collection '{collection_name}' is deprecated. "
            f"Use canonical names: {list(CANONICAL_COLLECTIONS.values())}. "
            f"See ADR-001 for migration guidance."
        )

    return True, None


def get_canonical_collection(collection_type: str) -> str:
    """
    Get canonical collection name for a collection type.

    Args:
        collection_type: Type of collection ("projects", "libraries", "memory")

    Returns:
        Canonical collection name

    Raises:
        ValueError: If collection_type is not recognized
    """
    if collection_type not in CANONICAL_COLLECTIONS:
        raise ValueError(
            f"Unknown collection type: {collection_type}. "
            f"Valid types: {list(CANONICAL_COLLECTIONS.keys())}"
        )
    return CANONICAL_COLLECTIONS[collection_type]


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
        await qdrant_client.get_collection(collection_name)
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

    try:
        response = await daemon_client.create_collection_v2(
            collection_name=collection_name,
            project_id=project_id,
            # config=None uses daemon defaults (384 vectors, Cosine, indexing enabled)
        )
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
    tag_prefix: bool = False
) -> Filter | None:
    """
    Build Qdrant filter with branch, file_type, project_id, and tag conditions.

    Args:
        filters: User-provided metadata filters
        branch: Git branch to filter by (None = current branch, "*" = all branches)
        file_type: File type to filter by ("code", "test", "docs", etc.)
        project_id: Project ID to filter by (for multi-tenant unified collections)
        tag: Tag value to filter by (exact match or prefix based on tag_prefix)
        tag_prefix: If True, match tags starting with the given tag (e.g., "myproject." matches "myproject.main")

    Returns:
        Qdrant Filter object or None if no filters

    Task 431: Tag filtering uses the full_tag field:
    - Projects: "project_id.branch" (e.g., "myapp.main")
    - Libraries: "folder1.folder2" (e.g., "docs.api.reference")
    - Memory: "project_id" (no branch hierarchy)
    """
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

    # Task 431: Add tag filter for dot-separated hierarchy
    if tag:
        if tag_prefix:
            # Match tags starting with the given prefix (e.g., "myproject." matches "myproject.main")
            # Use text match for prefix matching
            conditions.append(FieldCondition(key="full_tag", match=MatchText(text=tag)))
        else:
            # Exact tag match
            conditions.append(FieldCondition(key="full_tag", match=MatchValue(value=tag)))

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
    Store any type of content in the canonical multi-tenant vector database.

    ADR-001: Multi-tenant storage with automatic project_id tagging
    - All content stored in canonical 'projects' collection
    - Automatic project_id tagging from session context
    - File type differentiation via metadata
    - Enables cross-project search while maintaining project isolation

    Storage location:
    - All project content → 'projects' collection (canonical)
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
    logger = logging.getLogger(__name__)

    # Task 429: Check for explicit project activation
    # For project-scoped storage, activation is recommended but not blocking
    # (to maintain backwards compatibility during migration)
    if not _session_project_id and (collection is None or collection == CANONICAL_COLLECTIONS["projects"]):
        # Compute project_id from current path as fallback
        project_id = calculate_tenant_id(str(Path.cwd()))
        logger.warning(
            f"Storing to 'projects' without explicit activation. "
            f"Use manage(action='activate_project') for proper session tracking. "
            f"Fallback project_id={project_id}"
        )
    else:
        project_id = _session_project_id or calculate_tenant_id(str(Path.cwd()))

    # Determine target collection based on override or default to canonical projects
    if collection:
        # Validate collection name against ADR-001
        is_valid, error_msg = validate_collection_name(collection)
        if not is_valid:
            return create_error_response(
                ErrorCode.INVALID_COLLECTION_NAME,
                message_override=error_msg,
                context={"collection": collection},
            )
        target_collection = collection
    else:
        # Default: use canonical 'projects' collection (ADR-001)
        target_collection = CANONICAL_COLLECTIONS["projects"]

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

    # Prepare metadata with project_id for multi-tenant filtering
    doc_metadata = {
        "title": title or f"Document {uuid.uuid4().hex[:8]}",
        "project_id": project_id,  # Critical for multi-tenant filtering
        "source": source,
        "document_type": document_type,
        "file_type": file_type,  # For file type filtering
        "branch": current_branch,  # For branch filtering
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

    # ============================================================================
    # Task 431: Compute dot-separated tag hierarchy
    # ============================================================================
    # Projects: main_tag = project_id, full_tag = "project_id.branch_name"
    # Libraries: main_tag = library_name, full_tag = "library_name.subfolder1.subfolder2"
    # Memory: main_tag = project_id or "global", full_tag = main_tag
    main_tag = project_id
    full_tag = f"{project_id}.{current_branch}" if project_id and current_branch else project_id

    # Determine collection type for tag_type
    tag_type = "project"  # Default for projects collection
    if target_collection == CANONICAL_COLLECTIONS.get("libraries"):
        tag_type = "library"
        # For libraries, compute full_tag from file_path subfolder hierarchy
        if file_path:
            # Extract relative path within library
            path_parts = Path(file_path).parts
            if len(path_parts) > 1:
                # Use subfolder hierarchy: library_name.subfolder1.subfolder2
                full_tag = ".".join(path_parts[:-1])  # Exclude filename
                main_tag = path_parts[0] if path_parts else project_id
    elif target_collection == CANONICAL_COLLECTIONS.get("memory"):
        tag_type = "memory"
        full_tag = main_tag  # Memory tags are flat

    # Add tags to metadata
    doc_metadata["main_tag"] = main_tag
    doc_metadata["full_tag"] = full_tag
    doc_metadata["tag_type"] = tag_type

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

            # Task 430: Update daemon state on success
            update_daemon_state(available=True)

            return {
                "success": True,
                "document_id": response.document_id,
                "collection": target_collection,
                "project_id": project_id,  # Task 397: Include for multi-tenant reference
                "title": doc_metadata["title"],
                "content_length": len(content),
                "chunks_created": response.chunks_created,
                "file_type": file_type,
                "branch": current_branch,
                "metadata": doc_metadata,
                "daemon_available": True
            }
        except DaemonConnectionError as e:
            # Task 430: Update daemon state on failure
            update_daemon_state(available=False)
            return create_error_response(
                ErrorCode.DAEMON_CONNECTION_ERROR,
                context={"operation": "store"},
            )
    else:
        # Queue-based fallback when daemon unavailable (Task 428)
        # Content is queued to SQLite for processing when daemon becomes available
        # This replaces the direct Qdrant write fallback to maintain First Principle 10
        try:
            # Determine priority based on active project
            # HIGH (8) for active projects, NORMAL (5) for background
            priority = 8  # Assume active context when called from MCP

            # Determine main_tag from project_id
            main_tag = project_id if project_id else None
            full_tag = f"{project_id}.{current_branch}" if project_id else None

            # Enqueue content for later daemon processing
            queue_id = await state_manager.enqueue_ingestion(
                content=content,
                target_collection=target_collection,
                source_type=file_type or "text",
                operation="create",
                source_id=file_path or None,
                main_tag=main_tag,
                full_tag=full_tag,
                metadata=doc_metadata,
                priority=priority,
            )

            logger.warning(
                f"Daemon unavailable - content queued for processing: "
                f"queue_id={queue_id}, collection={target_collection}"
            )

            return {
                "success": True,
                "queue_id": queue_id,
                "collection": target_collection,
                "project_id": project_id,
                "title": doc_metadata["title"],
                "content_length": len(content),
                "file_type": file_type,
                "branch": current_branch,
                "metadata": doc_metadata,
                "queued_for_processing": True,
                "message": "Content queued for ingestion when daemon becomes available"
            }
        except Exception as e:
            logger.error(f"Failed to queue content for ingestion: {e}")
            return create_error_response(
                ErrorCode.QUEUE_ERROR,
                context={"operation": "store"},
            )

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
    tag_prefix: bool = False
) -> dict[str, Any]:
    """
    Search across collections with hybrid semantic + keyword matching.

    NEW: Task 396 - Multi-tenant filtering with scope parameter
    - scope="project": Filter by current project_id (default, most focused)
    - scope="global": Search all projects (no project_id filter)
    - scope="all": Search projects + libraries collections (broadest)
    - include_libraries: Also search _libraries collection

    NEW: Task 431 - Tag filtering with dot-separated hierarchy
    - tag: Filter by exact tag value (e.g., "myproject.main")
    - tag_prefix: If True, filter by tag prefix (e.g., "myproject." matches "myproject.main", "myproject.dev")

    Architecture:
    - Searches unified _projects collection with project_id filtering
    - Optional parallel search in _libraries collection
    - Results merged using Reciprocal Rank Fusion (RRF)
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified

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

        # Task 431: Search by tag (exact match)
        search(query="auth", tag="myproject.main")

        # Task 431: Search by tag prefix (matches all branches)
        search(query="auth", tag="myproject.", tag_prefix=True)

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
        tag: Filter by tag value (exact match or prefix based on tag_prefix)
        tag_prefix: If True, match tags starting with tag value

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
        return create_error_response(
            ErrorCode.INVALID_SCOPE,
            message_override=f"Invalid scope: {scope}",
            context={"scope": scope, "valid_scopes": valid_scopes},
        )

    # Task 430: Check daemon availability for project-scoped searches
    global _session_project_id, _daemon_state
    logger = logging.getLogger(__name__)

    if scope == "project" and _daemon_state == "UNRESPONSIVE":
        # Attempt to re-check daemon availability
        if not await check_daemon_availability():
            return create_error_response(
                ErrorCode.DAEMON_UNAVAILABLE,
                message_override="Daemon is unresponsive. Project-scoped searches require daemon for session priority management.",
                suggestion_override="Use scope='all' with include_libraries=True to search libraries without daemon dependency.",
                context={"operation": "search", "scope": scope},
            )
        # Daemon recovered - continue with search

    if scope == "project" and not _session_project_id:
        # Compute project_id from current path as fallback
        current_project_id = calculate_tenant_id(str(Path.cwd()))
        logger.warning(
            f"Searching with scope='project' without explicit activation. "
            f"Use manage(action='activate_project') for proper session tracking. "
            f"Fallback project_id={current_project_id}"
        )
    else:
        # Use session project_id if activated, otherwise compute from path
        current_project_id = _session_project_id or calculate_tenant_id(str(Path.cwd()))

    # Track if alias was used for deprecation warning in response
    alias_used = False

    # Determine search collections based on scope (ADR-001)
    # If explicit collection is provided, validate and resolve any alias first
    if collection:
        # Validate collection name against ADR-001
        is_valid, error_msg = validate_collection_name(collection)
        if not is_valid:
            return create_error_response(
                ErrorCode.INVALID_COLLECTION_NAME,
                message_override=error_msg,
                context={"collection": collection, "operation": "search"},
            )
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

    # Build metadata filters with branch, file_type, project_id, and tag (Task 431)
    search_filter = build_metadata_filters(
        filters=filters,
        branch=branch,
        file_type=file_type,
        project_id=project_filter_id,
        tag=tag,
        tag_prefix=tag_prefix
    )

    # For libraries, we don't filter by branch (they're external documentation)
    library_filter = build_metadata_filters(
        filters=filters,
        branch="*",  # Don't filter libraries by branch
        file_type=file_type,
        project_id=None,  # Libraries have library_name, not project_id
        tag=tag,  # Task 431: Tags work for libraries too (folder-based hierarchy)
        tag_prefix=tag_prefix
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
                search_results = await qdrant_client.search(
                    collection_name=coll,
                    query_vector=query_embeddings,
                    query_filter=filter_to_use,
                    limit=limit,
                    score_threshold=score_threshold
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
                scroll_results = await qdrant_client.scroll(
                    collection_name=coll,
                    scroll_filter=filter_to_use,
                    limit=limit * 2  # Get more for filtering
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
                "tag": tag,  # Task 431
                "tag_prefix": tag_prefix,  # Task 431
                "custom": filters or {}
            },
            # Task 430: Include daemon availability for debugging
            "daemon_available": _daemon_state == "AVAILABLE"
        }

        # Add deprecation warning if alias was used (Task 405)
        if alias_used:
            response["_deprecation_warning"] = (
                f"Collection '{collection}' is an alias. "
                f"Old collection names will be removed in a future version. "
                f"Please update your code to use the new collection name."
            )

        return response

    except Exception as e:
        return handle_tool_error(e, "search", context={"query": query[:50] if query else None})

@app.tool()
@track_tool("manage")
async def manage(
    action: str,
    collection: str = None,
    name: str = None,
    project_name: str = None,
    project_path: str = None,
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
    - "init_project" -> register project in unified _projects collection
    - "activate_project" -> REQUIRED: activate project for session (Task 429)
    - "deactivate_project" -> deactivate current project session
    - "cleanup" -> remove empty collections and optimize

    Watch Folder Management (Task 433):
    - "add_watch" -> add folder to watch with transactional handshake
        config: {path, patterns, ignore_patterns, recursive, debounce_seconds, watch_type, library_name}
    - "list_watches" -> list all configured watch folders
    - "remove_watch" -> remove a watch folder (config: {watch_id})
    - "update_watch" -> update watch configuration (config: {watch_id, enabled, patterns, ...})

    Tag Management (Task 431 - Dot-Separated Tag Hierarchy):
    - "list_tags" -> list tags with optional filtering
        config: {tag_type, parent_tag, watched_only, include_hierarchy}
        tag_type: "project" | "library" | "memory"
        parent_tag: filter to children of this tag (e.g., "myproject" returns "myproject.main", "myproject.dev")
        watched_only: only return tags marked as watched
        include_hierarchy: include parent tags in response
    - "set_tag_watched" -> mark a tag for daemon monitoring
        config: {tag_value, watched, collection}
    - "get_watched_tags" -> get all tags marked as watched

    Tag Hierarchy Format:
    - Projects: "project_id.branch_name" (e.g., "myapp.main", "myapp.feature-x")
    - Libraries: "folder1.folder2.folder3" (e.g., "docs.api.reference")
    - Memory: "project_id" (no branch for memory)

    Library Deletion Management (Task 432 - Additive Deletion Policy):
    - "mark_library_deleted" -> mark a library file as deleted (vectors preserved in Qdrant)
        config: {library_name, file_path, metadata}
        File will be skipped during future ingestion runs.
    - "re_ingest_deleted" -> clear deletion mark to allow re-ingestion
        config: {library_name, file_path}
    - "list_library_deletions" -> list deleted library files
        config: {library_name, include_re_ingested}
    - "check_library_deletion" -> check if a file is marked as deleted
        config: {library_name, file_path}

    Additive Deletion Policy:
    - Deleted files are tracked but vectors remain in Qdrant
    - Files marked as deleted are skipped during re-ingestion
    - Use re_ingest_deleted to explicitly allow re-ingestion

    Multi-Tenant Architecture (ADR-001):
    - init_project registers the current project's project_id in 'projects' collection
    - Canonical collections: projects, libraries, memory
    - Additional user collections can be created for special use cases

    Explicit Project Activation (Task 429):
    - Projects MUST be activated before project-scoped operations (store/search with scope="project")
    - Use manage(action="activate_project") or manage(action="activate_project", project_path="/path")
    - Without activation, scope="project" operations return activation instructions
    - Active project persists for the session lifetime

    Memory Rule Management (Task 435 - Memory Injection):
    - "refresh_memory" -> force re-injection of memory rules from 'memory' collection
        Returns: rules_loaded, rules_by_authority, rules_by_category, formatted_context
        Use after adding/updating memory rules to refresh session context
    - "get_memory_status" -> get current memory injection status
        Returns: rules_cached, last_injected, cache_age_seconds, cache_is_stale

    Memory Injection Architecture:
    - Memory rules are loaded on MCP startup from 'memory' collection
    - Rules are cached with 5-minute TTL to avoid redundant queries
    - Use refresh_memory after context compaction to re-inject rules
    - Rules sorted by authority: absolute (non-negotiable) before default

    Args:
        action: Management action to perform
        collection: Target collection name (for collection-specific actions)
        name: Name for new collections or operations
        project_name: Project context for workspace operations
        project_path: Project path for activate_project action (default: current directory)
        config: Additional configuration for operations

    Returns:
        Dict with action results and status information
    """
    await initialize_components()
    logger = logging.getLogger(__name__)

    # Global variables for session state (Task 429) and memory injection (Task 435)
    global _session_project_id, _session_project_path, _session_heartbeat
    global _memory_rules_cache, _memory_last_injected

    try:
        if action == "list_collections":
            collections_response = await qdrant_client.get_collections()
            collections_info = []

            for col in collections_response.collections:
                try:
                    col_info = await qdrant_client.get_collection(col.name)
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
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="Collection name required for create action",
                    suggestion_override="Provide 'name' parameter with a valid collection name.",
                    context={"action": action},
                )

            # Validate collection name against ADR-001
            is_valid, error_msg = validate_collection_name(name)
            if not is_valid:
                return create_error_response(
                    ErrorCode.INVALID_COLLECTION_NAME,
                    message_override=error_msg,
                    context={"collection": name, "action": action},
                )

            collection_config = config or DEFAULT_COLLECTION_CONFIG

            # Get project_id from project_name parameter or current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection creation must go through daemon. No fallback to direct writes.
            # ============================================================================

            if not daemon_client:
                return create_error_response(
                    ErrorCode.DAEMON_UNAVAILABLE,
                    message_override="Daemon not connected. Collection creation requires daemon (First Principle 10).",
                    suggestion_override="Start the daemon with 'wqm service start' and retry.",
                    context={"action": action},
                )

            try:
                response = await daemon_client.create_collection_v2(
                    collection_name=name,
                    project_id=project_id,
                    # config=None uses daemon defaults (384 vectors, Cosine, indexing enabled)
                )

                if response.success:
                    return {
                        "success": True,
                        "action": action,
                        "collection_name": name,
                        "message": f"Collection '{name}' created successfully via daemon"
                    }
                else:
                    return create_error_response(
                        ErrorCode.STORE_FAILED,
                        message_override=f"Daemon failed to create collection: {response.error_message}",
                        context={"action": action, "collection": name},
                    )
            except DaemonConnectionError as e:
                return create_error_response(
                    ErrorCode.DAEMON_CONNECTION_ERROR,
                    context={"action": action, "collection": name},
                )

        elif action == "delete_collection":
            if not name and not collection:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="Collection name required for delete action",
                    suggestion_override="Provide 'name' or 'collection' parameter.",
                    context={"action": action},
                )

            target_collection = name or collection

            # Validate collection name against ADR-001
            is_valid, error_msg = validate_collection_name(target_collection)
            if not is_valid:
                return create_error_response(
                    ErrorCode.INVALID_COLLECTION_NAME,
                    message_override=error_msg,
                    context={"collection": target_collection, "action": action},
                )

            # Get project_id from current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection deletion must go through daemon. No fallback to direct writes.
            # ============================================================================

            if not daemon_client:
                return create_error_response(
                    ErrorCode.DAEMON_UNAVAILABLE,
                    message_override="Daemon not connected. Collection deletion requires daemon (First Principle 10).",
                    suggestion_override="Start the daemon with 'wqm service start' and retry.",
                    context={"action": action, "collection": target_collection},
                )

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
                return create_error_response(
                    ErrorCode.DAEMON_CONNECTION_ERROR,
                    context={"action": action, "collection": target_collection},
                )

        elif action == "collection_info":
            if not name and not collection:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="Collection name required for info action",
                    suggestion_override="Provide 'name' or 'collection' parameter.",
                    context={"action": action},
                )

            target_collection = name or collection
            col_info = await qdrant_client.get_collection(target_collection)

            return {
                "success": True,
                "action": action,
                "collection_name": target_collection,
                "info": {
                    "points_count": col_info.points_count,
                    "segments_count": col_info.segments_count,
                    "status": col_info.status.value,
                    "vector_size": col_info.config.params.vectors.size,
                    "distance": col_info.config.params.vectors.distance.value,
                    "indexed": col_info.indexed_vectors_count,
                    "optimizer_status": col_info.optimizer_status
                }
            }

        elif action == "workspace_status":
            # System health check (async)
            current_project = project_name or await get_project_name()
            project_collection = get_project_collection()

            # Get collections info (async)
            collections_response = await qdrant_client.get_collections()

            # Check for project collection (new architecture: single _{project_id})
            project_collections = []
            for col in collections_response.collections:
                if col.name == project_collection:
                    project_collections.append(col.name)
                # Also include legacy collections for backwards compatibility
                elif col.name.startswith(f"{current_project}-"):
                    project_collections.append(col.name)

            # Get Qdrant cluster info (async)
            cluster_info = await qdrant_client.get_cluster_info()

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
                "project_collections": project_collections,
                "total_collections": len(collections_response.collections),
                "embedding_model": os.getenv("FASTEMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
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
            # Collection deletion must go through daemon. No fallback to direct writes.
            # ============================================================================

            if not daemon_client:
                return create_error_response(
                    ErrorCode.DAEMON_UNAVAILABLE,
                    message_override="Daemon not connected. Cleanup requires daemon (First Principle 10).",
                    suggestion_override="Start the daemon with 'wqm service start' and retry.",
                    context={"action": action},
                )

            collections_response = await qdrant_client.get_collections()
            cleaned_collections = []
            failed_collections = []

            # Get project_id from current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            for col in collections_response.collections:
                try:
                    col_info = await qdrant_client.get_collection(col.name)
                    if col_info.points_count == 0:
                        try:
                            await daemon_client.delete_collection_v2(
                                collection_name=col.name,
                                project_id=project_id,
                            )
                            cleaned_collections.append(col.name)
                            logger.info(f"Deleted empty collection '{col.name}' via daemon")
                        except DaemonConnectionError as e:
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
            # Task 429: Explicit project activation
            # Validates project, registers with daemon, starts heartbeat

            # Determine project path
            target_path = project_path or str(Path.cwd())
            target_path = str(Path(target_path).resolve())

            # Validate project signature (git repo or .taskmaster marker)
            is_git_repo = (Path(target_path) / ".git").exists()
            has_taskmaster = (Path(target_path) / ".taskmaster").exists()

            if not is_git_repo and not has_taskmaster:
                return create_error_response(
                    ErrorCode.INVALID_PROJECT_PATH,
                    message_override=f"Invalid project: path is not a git repository and has no .taskmaster directory",
                    suggestion_override="Projects must be git repositories or have a .taskmaster directory marker.",
                    context={"action": action},
                )

            # Compute project identifiers
            project_id = calculate_tenant_id(target_path)
            target_project_name = project_name or Path(target_path).name
            git_remote = await _get_git_remote() if is_git_repo else None

            # Register with daemon if available
            if daemon_client:
                try:
                    response = await daemon_client.register_project(
                        path=target_path,
                        project_id=project_id,
                        name=target_project_name,
                        git_remote=git_remote
                    )

                    # Start heartbeat for session tracking
                    if _session_heartbeat:
                        try:
                            await _session_heartbeat.stop()
                        except Exception:
                            pass

                    _session_heartbeat = SessionHeartbeat(daemon_client, project_id)
                    await _session_heartbeat.start()

                    logger.info(
                        f"Project activated: {target_project_name} ({project_id}), "
                        f"priority={response.priority}, sessions={response.active_sessions}"
                    )

                except DaemonConnectionError as e:
                    logger.warning(f"Daemon registration failed: {e}")
                    # Continue with local activation
            else:
                logger.warning("Daemon not available - project activated locally only")

            # Store in session state
            _session_project_id = project_id
            _session_project_path = target_path

            return {
                "success": True,
                "action": action,
                "project_id": project_id,
                "project_path": target_path,
                "project_name": target_project_name,
                "is_git_repo": is_git_repo,
                "has_taskmaster": has_taskmaster,
                "daemon_registered": daemon_client is not None,
                "message": f"Project '{target_project_name}' activated for this session"
            }

        elif action == "deactivate_project":
            # Task 429: Deactivate current project session

            if not _session_project_id:
                return create_error_response(
                    ErrorCode.PROJECT_NOT_ACTIVATED,
                    context={"action": action},
                )

            old_project_id = _session_project_id

            # Stop heartbeat
            if _session_heartbeat and _session_heartbeat.is_running:
                try:
                    await _session_heartbeat.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop heartbeat: {e}")

            # Deprioritize with daemon
            if daemon_client:
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

            # Clear session state
            _session_project_id = None
            _session_project_path = None
            _session_heartbeat = None

            return {
                "success": True,
                "action": action,
                "deactivated_project_id": old_project_id,
                "message": "Project deactivated. Use manage(action='activate_project') to activate a project."
            }

        # ============================================================================
        # WATCH FOLDER MANAGEMENT (Task 433: Queue/Watch Handshake)
        # ============================================================================
        elif action == "add_watch":
            # Add a folder to watch with transactional handshake
            # This creates both the watch config AND enqueues initial scan
            if not config or "path" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.path required for add_watch action",
                    suggestion_override="Provide config={'path': '/path/to/folder'} to add a watch folder.",
                    context={"action": action},
                )

            folder_path = config["path"]
            if not Path(folder_path).exists():
                return create_error_response(
                    ErrorCode.PATH_NOT_FOUND,
                    context={"action": action},
                )
            if not Path(folder_path).is_dir():
                return create_error_response(
                    ErrorCode.PATH_NOT_DIRECTORY,
                    context={"action": action},
                )

            # Generate watch_id if not provided
            watch_id = config.get("watch_id") or f"watch-{uuid.uuid4().hex[:8]}"

            # Determine watch type and target collection
            watch_type = config.get("watch_type", "project")  # "project" or "library"
            library_name = config.get("library_name")

            if watch_type == "library" and not library_name:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="library_name required when watch_type='library'",
                    suggestion_override="Provide config={'library_name': 'name'} when using watch_type='library'.",
                    context={"action": action, "watch_type": watch_type},
                )

            # Default collection based on watch type
            target_collection = config.get("collection")
            if not target_collection:
                target_collection = "_libraries" if watch_type == "library" else "_projects"

            # Create watch folder config
            watch_config = WatchFolderConfig(
                watch_id=watch_id,
                path=str(Path(folder_path).resolve()),
                collection=target_collection,
                patterns=config.get("patterns", ["*"]),
                ignore_patterns=config.get("ignore_patterns", [".*", "__pycache__/*", "*.pyc"]),
                auto_ingest=config.get("auto_ingest", True),
                recursive=config.get("recursive", True),
                recursive_depth=config.get("recursive_depth", 10),
                debounce_seconds=config.get("debounce_seconds", 2.0),
                enabled=True,
                watch_type=watch_type,
                library_name=library_name,
            )

            # TRANSACTIONAL HANDSHAKE (Task 433):
            # 1. Save watch config to SQLite (daemon will detect via polling)
            # 2. Enqueue scan_folder operation for initial file discovery
            await state_manager.save_watch_folder_config(watch_id, watch_config)

            # Enqueue scan_folder operation with the folder path
            # The daemon will process this and discover all matching files
            await state_manager.enqueue_ingestion(
                content=f"scan:{folder_path}",
                target_collection=target_collection,
                source_type="scan_folder",
                operation="create",  # Using 'create' since scan_folder isn't a valid operation
                metadata={
                    "watch_id": watch_id,
                    "patterns": watch_config.patterns,
                    "ignore_patterns": watch_config.ignore_patterns,
                    "recursive": watch_config.recursive,
                    "recursive_depth": watch_config.recursive_depth,
                    "watch_type": watch_type,
                    "library_name": library_name,
                },
                priority=7,  # High priority for initial scan
            )

            logger.info(f"Added watch folder: {watch_id} at {folder_path}")

            return {
                "success": True,
                "action": action,
                "watch_id": watch_id,
                "path": str(Path(folder_path).resolve()),
                "collection": target_collection,
                "watch_type": watch_type,
                "patterns": watch_config.patterns,
                "message": f"Watch folder added. Daemon will scan and ingest matching files."
            }

        elif action == "list_watches":
            # List all configured watch folders
            watches = await state_manager.list_watch_folders()
            return {
                "success": True,
                "action": action,
                "watches": [
                    {
                        "watch_id": w.watch_id,
                        "path": w.path,
                        "collection": w.collection,
                        "patterns": w.patterns,
                        "enabled": w.enabled,
                        "watch_type": w.watch_type,
                        "library_name": w.library_name,
                        "last_scan": w.last_scan.isoformat() if w.last_scan else None,
                    }
                    for w in watches
                ],
                "total_watches": len(watches)
            }

        elif action == "remove_watch":
            # Remove a watch folder configuration
            if not config or "watch_id" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.watch_id required for remove_watch action",
                    suggestion_override="Provide config={'watch_id': 'watch-id'} to remove a watch folder.",
                    context={"action": action},
                )

            watch_id = config["watch_id"]

            # Get current config before removing (for response)
            current = await state_manager.get_watch_folder_config(watch_id)
            if not current:
                return create_error_response(
                    ErrorCode.WATCH_NOT_FOUND,
                    message_override=f"Watch not found: {watch_id}",
                    context={"action": action, "watch_id": watch_id},
                )

            await state_manager.remove_watch_folder_config(watch_id)

            logger.info(f"Removed watch folder: {watch_id}")

            return {
                "success": True,
                "action": action,
                "watch_id": watch_id,
                "removed_path": current.path,
                "message": "Watch folder removed. Daemon will stop watching on next poll."
            }

        elif action == "update_watch":
            # Update a watch folder configuration (enable/disable, change patterns, etc.)
            if not config or "watch_id" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.watch_id required for update_watch action",
                    suggestion_override="Provide config={'watch_id': 'watch-id', ...} to update a watch folder.",
                    context={"action": action},
                )

            watch_id = config["watch_id"]

            # Get current config
            current = await state_manager.get_watch_folder_config(watch_id)
            if not current:
                return create_error_response(
                    ErrorCode.WATCH_NOT_FOUND,
                    message_override=f"Watch not found: {watch_id}",
                    context={"action": action, "watch_id": watch_id},
                )

            # Update fields that were provided
            if "enabled" in config:
                current.enabled = config["enabled"]
            if "patterns" in config:
                current.patterns = config["patterns"]
            if "ignore_patterns" in config:
                current.ignore_patterns = config["ignore_patterns"]
            if "recursive" in config:
                current.recursive = config["recursive"]
            if "recursive_depth" in config:
                current.recursive_depth = config["recursive_depth"]
            if "debounce_seconds" in config:
                current.debounce_seconds = config["debounce_seconds"]

            await state_manager.save_watch_folder_config(watch_id, current)

            logger.info(f"Updated watch folder: {watch_id}")

            return {
                "success": True,
                "action": action,
                "watch_id": watch_id,
                "enabled": current.enabled,
                "patterns": current.patterns,
                "message": "Watch folder updated. Changes take effect on next daemon poll."
            }

        elif action == "list_tags":
            # Task 431: List tags with optional filtering
            # Supports dot-separated tag hierarchy (project.branch, library.folder1.folder2)
            tag_type = config.get("tag_type") if config else None
            parent_tag = config.get("parent_tag") if config else None
            watched_only = config.get("watched_only", False) if config else False
            include_hierarchy = config.get("include_hierarchy", False) if config else False

            tags = await state_manager.list_tags(
                collection=collection,
                tag_type=tag_type,
                parent_tag=parent_tag,
                watched_only=watched_only,
                include_hierarchy=include_hierarchy,
            )

            return {
                "success": True,
                "action": action,
                "tags": tags,
                "count": len(tags),
                "filters": {
                    "collection": collection,
                    "tag_type": tag_type,
                    "parent_tag": parent_tag,
                    "watched_only": watched_only,
                }
            }

        elif action == "set_tag_watched":
            # Task 431: Mark a tag as watched for daemon monitoring
            if not config or "tag_value" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.tag_value required for set_tag_watched action",
                    suggestion_override="Provide config={'tag_value': 'tag'} to mark a tag as watched.",
                    context={"action": action},
                )

            tag_value = config["tag_value"]
            watched = config.get("watched", True)
            tag_collection = config.get("collection") or collection

            if not tag_collection:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="collection required for set_tag_watched action",
                    suggestion_override="Provide 'collection' parameter or config={'collection': 'name'}.",
                    context={"action": action},
                )

            await state_manager.set_tag_watched(tag_value, tag_collection, watched)

            return {
                "success": True,
                "action": action,
                "tag_value": tag_value,
                "collection": tag_collection,
                "watched": watched,
                "message": f"Tag '{tag_value}' {'marked as watched' if watched else 'unmarked from watched'}. Daemon will {'monitor' if watched else 'stop monitoring'} this tag."
            }

        elif action == "get_watched_tags":
            # Task 431: Get all tags marked as watched
            watched_tags = await state_manager.get_watched_tags()

            return {
                "success": True,
                "action": action,
                "watched_tags": watched_tags,
                "count": len(watched_tags)
            }

        elif action == "mark_library_deleted":
            # Task 432: Mark a library file as deleted (additive deletion policy)
            # Vectors remain in Qdrant but file won't be re-ingested
            if not config or "library_name" not in config or "file_path" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.library_name and config.file_path required for mark_library_deleted action",
                    suggestion_override="Provide config={'library_name': 'name', 'file_path': '/path'}.",
                    context={"action": action},
                )

            library_name = config["library_name"]
            file_path = config["file_path"]
            metadata_info = config.get("metadata")

            success = await state_manager.mark_library_deleted(
                library_name=library_name,
                file_path=file_path,
                metadata=metadata_info,
            )

            return {
                "success": success,
                "action": action,
                "library_name": library_name,
                "file_path": file_path,
                "message": f"Library file marked as deleted. Vectors preserved in Qdrant but file will be skipped during re-ingestion."
                if success else "Failed to mark library file as deleted"
            }

        elif action == "re_ingest_deleted":
            # Task 432: Clear deletion mark to allow re-ingestion
            if not config or "library_name" not in config or "file_path" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.library_name and config.file_path required for re_ingest_deleted action",
                    suggestion_override="Provide config={'library_name': 'name', 'file_path': '/path'}.",
                    context={"action": action},
                )

            library_name = config["library_name"]
            file_path = config["file_path"]

            success = await state_manager.re_ingest_library_file(
                library_name=library_name,
                file_path=file_path,
            )

            return {
                "success": success,
                "action": action,
                "library_name": library_name,
                "file_path": file_path,
                "message": "Library file cleared for re-ingestion."
                if success else "File not found in deletion list or already cleared"
            }

        elif action == "list_library_deletions":
            # Task 432: List deleted library files
            library_name = config.get("library_name") if config else None
            include_re_ingested = config.get("include_re_ingested", False) if config else False

            deletions = await state_manager.list_library_deletions(
                library_name=library_name,
                include_re_ingested=include_re_ingested,
            )

            return {
                "success": True,
                "action": action,
                "deletions": deletions,
                "count": len(deletions),
                "filters": {
                    "library_name": library_name,
                    "include_re_ingested": include_re_ingested,
                }
            }

        elif action == "check_library_deletion":
            # Task 432: Check if a specific file is marked as deleted
            if not config or "library_name" not in config or "file_path" not in config:
                return create_error_response(
                    ErrorCode.MISSING_REQUIRED_PARAMETER,
                    message_override="config.library_name and config.file_path required for check_library_deletion action",
                    suggestion_override="Provide config={'library_name': 'name', 'file_path': '/path'}.",
                    context={"action": action},
                )

            library_name = config["library_name"]
            file_path = config["file_path"]

            is_deleted = await state_manager.is_library_file_deleted(
                library_name=library_name,
                file_path=file_path,
            )

            return {
                "success": True,
                "action": action,
                "library_name": library_name,
                "file_path": file_path,
                "is_deleted": is_deleted,
                "message": "File is marked as deleted (will be skipped during ingestion)"
                if is_deleted else "File is not marked as deleted (will be ingested normally)"
            }

        elif action == "refresh_memory":
            # Task 435: Force re-injection of memory rules
            # Use after adding/updating memory rules to refresh session context
            try:
                # Force reload from memory collection
                memory_rules = await _load_memory_rules(force_refresh=True)

                # Format for display
                formatted_rules = _format_memory_rules_for_llm(memory_rules)

                return {
                    "success": True,
                    "action": action,
                    "rules_loaded": len(memory_rules),
                    "rules_by_authority": {
                        "absolute": len([r for r in memory_rules if r["authority"] == "absolute"]),
                        "default": len([r for r in memory_rules if r["authority"] == "default"]),
                    },
                    "rules_by_category": {
                        "behavior": len([r for r in memory_rules if r["category"] == "behavior"]),
                        "preference": len([r for r in memory_rules if r["category"] == "preference"]),
                        "agent": len([r for r in memory_rules if r["category"] == "agent"]),
                    },
                    "last_injected": _memory_last_injected.isoformat() if _memory_last_injected else None,
                    "formatted_context": formatted_rules,
                    "message": f"Memory rules refreshed: {len(memory_rules)} rules loaded"
                }
            except Exception as e:
                return handle_tool_error(e, "refresh_memory", context={"action": action})

        elif action == "get_memory_status":
            # Task 435: Get current memory injection status
            age_secs = None
            if _memory_last_injected:
                age_secs = (datetime.now(timezone.utc) - _memory_last_injected).total_seconds()

            return {
                "success": True,
                "action": action,
                "rules_cached": len(_memory_rules_cache) if _memory_rules_cache else 0,
                "last_injected": _memory_last_injected.isoformat() if _memory_last_injected else None,
                "cache_age_seconds": age_secs,
                "cache_stale_threshold_seconds": _MEMORY_INJECTION_STALE_SECS,
                "cache_is_stale": age_secs > _MEMORY_INJECTION_STALE_SECS if age_secs else True,
                "message": "Memory cache status retrieved"
            }

        else:
            available_actions = [
                "list_collections", "create_collection", "delete_collection",
                "collection_info", "workspace_status", "init_project",
                "activate_project", "deactivate_project", "cleanup",
                "add_watch", "list_watches", "remove_watch", "update_watch",
                "list_tags", "set_tag_watched", "get_watched_tags",
                "mark_library_deleted", "re_ingest_deleted", "list_library_deletions", "check_library_deletion",
                "refresh_memory", "get_memory_status"
            ]
            return create_error_response(
                ErrorCode.INVALID_ACTION,
                message_override=f"Unknown action: {action}",
                suggestion_override=f"Available actions: {', '.join(available_actions)}",
                context={"action": action, "available_actions": available_actions},
            )

    except Exception as e:
        return handle_tool_error(e, f"manage({action})", context={"action": action})

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
    scope: str = "project",
    tag: str = None,
    tag_prefix: bool = False
) -> dict[str, Any]:
    """
    Retrieve documents directly by ID or metadata without search ranking.

    NEW: Task 398 - Multi-tenant scope parameter with branch filtering
    - scope="project": Filter by current project_id (default, most focused)
    - scope="global": Retrieve from all projects (no project_id filter)
    - scope="all": Retrieve from projects + libraries collections (broadest)
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified

    NEW: Task 431 - Tag filtering with dot-separated hierarchy
    - tag: Filter by exact tag value (e.g., "myproject.main")
    - tag_prefix: If True, filter by tag prefix (e.g., "myproject." matches all branches)

    Retrieval methods determined by parameters:
    - document_id specified -> direct ID lookup
    - metadata specified -> filter-based retrieval
    - collection specified -> limits retrieval to specific collection (overrides scope)
    - branch -> filters by Git branch
    - file_type -> filters by file type
    - scope -> controls multi-tenant collection and filtering
    - tag -> filters by tag value

    Args:
        document_id: Direct document ID to retrieve
        collection: Specific collection to retrieve from (overrides scope)
        metadata: Metadata filters for document selection
        limit: Maximum number of documents to retrieve
        project_name: Limit retrieval to project collections (deprecated, use scope)
        branch: Git branch to filter by (None=current, "*"=all branches)
        file_type: File type filter ("code", "test", "docs", etc.)
        scope: Retrieval scope - "project" (default), "global", or "all"
        tag: Filter by tag value (exact match or prefix based on tag_prefix)
        tag_prefix: If True, match tags starting with tag value

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

        # Task 431: Retrieve by tag (exact match)
        retrieve(metadata={}, tag="myproject.main")

        # Task 431: Retrieve by tag prefix (matches all branches)
        retrieve(metadata={}, tag="myproject.", tag_prefix=True)
    """
    await initialize_components()

    if not document_id and not metadata:
        return create_error_response(
            ErrorCode.MISSING_REQUIRED_PARAMETER,
            message_override="Either document_id or metadata filters must be provided",
            suggestion_override="Provide a document_id for exact retrieval, or metadata filters for filtered retrieval.",
            context={"operation": "retrieve"},
        )

    # Validate scope parameter (Task 398)
    valid_scopes = ("project", "global", "all")
    if scope not in valid_scopes:
        return create_error_response(
            ErrorCode.INVALID_SCOPE,
            message_override=f"Invalid scope: {scope}",
            context={"scope": scope, "valid_scopes": valid_scopes},
        )

    # Task 429: Check for explicit project activation when using scope="project"
    global _session_project_id
    logger = logging.getLogger(__name__)

    if scope == "project" and not _session_project_id:
        # Compute project_id from current path as fallback
        current_project_id = calculate_tenant_id(str(Path.cwd()))
        logger.warning(
            f"Retrieving with scope='project' without explicit activation. "
            f"Use manage(action='activate_project') for proper session tracking. "
            f"Fallback project_id={current_project_id}"
        )
    else:
        # Use session project_id if activated, otherwise compute from path
        current_project_id = _session_project_id or calculate_tenant_id(str(Path.cwd()))

    try:
        results = []

        # Track if alias was used for deprecation warning in response
        alias_used = False

        # Determine retrieval collections based on scope (ADR-001)
        project_filter_id = None
        if collection:
            # Validate collection name against ADR-001
            is_valid, error_msg = validate_collection_name(collection)
            if not is_valid:
                return create_error_response(
                    ErrorCode.INVALID_COLLECTION_NAME,
                    message_override=error_msg,
                    context={"collection": collection, "operation": "retrieve"},
                )
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
                    points = await qdrant_client.retrieve(
                        collection_name=search_collection,
                        ids=[document_id]
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

                        # Task 431: Apply tag filter if specified
                        if tag:
                            doc_tag = point.payload.get("full_tag", "")
                            if tag_prefix:
                                # Prefix match: doc_tag should start with tag
                                if not doc_tag.startswith(tag):
                                    continue
                            else:
                                # Exact match
                                if doc_tag != tag:
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
                    project_id=collection_project_id,
                    tag=tag,  # Task 431
                    tag_prefix=tag_prefix
                )

                # Retrieve from collection (async)
                try:
                    scroll_result = await qdrant_client.scroll(
                        collection_name=search_collection,
                        scroll_filter=search_filter,
                        limit=limit - len(results)  # Respect total limit
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
                "tag": tag,  # Task 431
                "tag_prefix": tag_prefix,  # Task 431
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

        return response

    except Exception as e:
        return handle_tool_error(e, "retrieve", context={"document_id": document_id, "collection": collection})

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
    app.run(transport=transport, host=host, port=port)

def main() -> None:
    """Console script entry point for UV tool installation and direct execution."""
    typer.run(run_server)

if __name__ == "__main__":
    main()
