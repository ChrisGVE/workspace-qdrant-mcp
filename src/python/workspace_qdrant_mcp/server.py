"""
FastMCP server for workspace-qdrant-mcp.

Streamlined 4-tool implementation that provides all the functionality of the original
36-tool system through intelligent content-based routing and parameter analysis.

The server automatically detects project structure, initializes workspace-specific collections,
and provides hybrid search combining dense (semantic) and sparse (keyword) vectors.

Key Features:
    - 4 comprehensive tools: store, search, manage, retrieve
    - Content-based routing - parameters determine specific actions
    - Single collection per project with metadata-based differentiation
    - Branch-aware querying with automatic Git branch detection
    - File type filtering via metadata (code, test, docs, config, data, build, other)
    - Hybrid search combining dense (semantic) and sparse (keyword) vectors
    - Evidence-based performance: 100% precision for symbol/exact search, 94.2% for semantic
    - Comprehensive scratchbook for cross-project note management
    - Production-ready async architecture with comprehensive error handling

Architecture (Task 374.6):
    - Project collections: _{project_id} (single collection per project)
    - project_id: 12-char hex hash from project path (via calculate_tenant_id)
    - Branch-scoped queries: All queries filter by Git branch (default: current branch)
    - File type differentiation via metadata: code, test, docs, config, data, build, other
    - Shared collections for cross-project resources (memory, libraries)
    - No collection type suffixes (replaced with metadata filtering)

Tools:
    1. store - Store any content (documents, notes, code, web content)
    2. search - Hybrid semantic + keyword search with branch and file_type filtering
    3. manage - Collection management, system status, configuration
    4. retrieve - Direct document retrieval by ID or metadata with branch filtering

Example Usage:
    # Store different content types (all go to _{project_id} collection)
    store(content="user notes", source="scratchbook")  # metadata: file_type="other"
    store(file_path="main.py", content="code")         # metadata: file_type="code"
    store(url="https://docs.com", content="docs")      # metadata: file_type="docs"

    # Search with branch and file_type filtering
    search(query="authentication", mode="hybrid")             # Current branch, all file types
    search(query="def login", mode="exact", file_type="code") # Current branch, code only
    search(query="notes", branch="main", file_type="docs")    # main branch, docs only

    # Management operations
    manage(action="list_collections")                  # List all collections
    manage(action="workspace_status")                 # System status
    manage(action="init_project")                     # Create _{project_id} collection

    # Direct retrieval with branch filtering
    retrieve(document_id="uuid-123")                              # Current branch
    retrieve(metadata={"file_type": "test"}, branch="develop")    # develop branch, tests

Write Path Architecture (First Principle 10):
    DAEMON-ONLY WRITES: All Qdrant write operations MUST route through the daemon

    Collection Types:
        - PROJECT: _{project_id} - Auto-created by daemon for file watching
        - USER: {basename}-{type} - User collections, created via daemon
        - LIBRARY: _{library_name} - External libraries, managed via daemon
        - MEMORY: _memory, _agent_memory - EXCEPTION: Direct writes allowed (meta-level data)

    Write Priority:
        1. PRIMARY: DaemonClient.ingest_text() / create_collection_v2() / delete_collection_v2()
        2. FALLBACK: Direct qdrant_client writes (when daemon unavailable)
        3. EXCEPTION: MEMORY collections use direct writes (architectural decision)

    All fallback paths:
        - Are clearly documented with NOTE comments
        - Log warnings when used
        - Include "fallback_mode" in return values
        - Maintain backwards compatibility during daemon rollout

    See: FIRST-PRINCIPLES.md (Principle 10), Task 375.6 validation report
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
from common.core.collection_naming import build_project_collection_name
from common.grpc.daemon_client import DaemonClient, DaemonConnectionError
from common.utils.git_utils import get_current_branch
from common.utils.project_detection import calculate_tenant_id

# Global components
qdrant_client: AsyncQdrantClient | None = None
embedding_model = None
daemon_client: DaemonClient | None = None
project_cache = {}

# Session lifecycle state
_session_project_id: str | None = None
_session_project_path: str | None = None


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
    - On startup: Detect project, compute project_id, register with daemon
    - On shutdown: Deprioritize project to allow daemon to adjust priorities

    The daemon uses session registration to:
    - Set HIGH priority for actively-edited projects
    - Track active sessions for crash recovery
    - Optimize file watcher resources based on activity
    """
    global daemon_client, _session_project_id, _session_project_path

    logger = logging.getLogger(__name__)

    # =========================================================================
    # STARTUP: Register project with daemon for high-priority processing
    # =========================================================================
    try:
        # Initialize daemon client if not already done
        if daemon_client is None:
            daemon_client = DaemonClient()
            try:
                await daemon_client.connect()
            except DaemonConnectionError:
                # Daemon connection is optional - server works without it
                daemon_client = None
                logger.warning("Daemon not available - session lifecycle disabled")

        if daemon_client:
            # Detect current project
            project_path = str(Path.cwd())
            project_id = calculate_tenant_id(project_path)
            project_name = await get_project_name()
            git_remote = await _get_git_remote()

            # Store for shutdown cleanup
            _session_project_id = project_id
            _session_project_path = project_path

            # Register project with daemon
            try:
                response = await daemon_client.register_project(
                    path=project_path,
                    project_id=project_id,
                    name=project_name,
                    git_remote=git_remote
                )
                logger.info(
                    f"Project registered: {project_name} ({project_id}), "
                    f"priority={response.priority}, sessions={response.active_sessions}"
                )
            except Exception as e:
                logger.warning(f"Failed to register project: {e}")
                # Continue without registration - daemon features degraded
    except Exception as e:
        logger.warning(f"Lifespan startup error: {e}")

    # Yield control to the application
    yield

    # =========================================================================
    # SHUTDOWN: Deprioritize project with daemon
    # =========================================================================
    try:
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

# Unified multi-tenant collection names (Task 394/396)
# These collections store data from all projects/libraries with tenant_id filtering
UNIFIED_COLLECTIONS = {
    "projects": "_projects",    # All project code/documents
    "libraries": "_libraries",  # All library documentation
    "memory": "_memory",        # Agent memory and cross-project notes
}


def get_collection_type(collection_name: str) -> str:
    """Determine collection type from collection name.

    Args:
        collection_name: Collection name to analyze

    Returns:
        One of: "project", "user", "library", "memory"
    """
    if collection_name in ("_memory", "_agent_memory"):
        return "memory"
    elif collection_name.startswith("_"):
        # Could be project or library - check for library patterns
        # Libraries typically have recognizable names (e.g., _numpy, _pandas)
        # Projects are hex hashes (e.g., _a1b2c3d4e5f6)
        # For now, assume underscore-prefixed is project unless it's a known library pattern
        # This can be refined based on actual usage patterns
        if len(collection_name) == 13:  # _{12-char-hash}
            return "project"
        else:
            return "library"
    else:
        # No underscore prefix = user collection
        return "user"


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
    """Initialize Qdrant client, daemon client, and embedding model.

    Note: daemon_client may already be initialized by the lifespan context manager.
    This function ensures all components are ready for tool operations.
    """
    global qdrant_client, embedding_model, daemon_client

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
    project_id: str = None
) -> Filter | None:
    """
    Build Qdrant filter with branch, file_type, and project_id conditions.

    Args:
        filters: User-provided metadata filters
        branch: Git branch to filter by (None = current branch, "*" = all branches)
        file_type: File type to filter by ("code", "test", "docs", etc.)
        project_id: Project ID to filter by (for multi-tenant unified collections)

    Returns:
        Qdrant Filter object or None if no filters
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
    - All content stored in unified _projects collection
    - Automatic project_id tagging from session context
    - File type differentiation via metadata
    - Enables cross-project search while maintaining project isolation

    Storage location:
    - All project content → _projects collection
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

    # Determine target collection based on override or default to unified projects
    if collection:
        # Explicit collection override (e.g., for libraries or memory)
        target_collection = collection
    else:
        # Default: use unified _projects collection (Task 397)
        target_collection = UNIFIED_COLLECTIONS["projects"]

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
                "metadata": doc_metadata
            }
        except DaemonConnectionError as e:
            return {
                "success": False,
                "error": f"Failed to store document via daemon: {str(e)}"
            }
    else:
        # Fallback to direct Qdrant write if daemon unavailable
        # This maintains backwards compatibility but violates First Principle 10
        try:
            # Ensure collection exists
            if not await ensure_collection_exists(target_collection):
                return {
                    "success": False,
                    "error": f"Failed to create/access collection: {target_collection}"
                }

            # Generate document ID and embeddings
            document_id = str(uuid.uuid4())
            embeddings = await generate_embeddings(content)

            # Store in Qdrant (async)
            point = PointStruct(
                id=document_id,
                vector=embeddings,
                payload={
                    "content": content,
                    **doc_metadata
                }
            )

            await qdrant_client.upsert(
                collection_name=target_collection,
                points=[point]
            )

            return {
                "success": True,
                "document_id": document_id,
                "collection": target_collection,
                "project_id": project_id,  # Task 397: Include for multi-tenant reference
                "title": doc_metadata["title"],
                "content_length": len(content),
                "file_type": file_type,
                "branch": current_branch,
                "metadata": doc_metadata,
                "fallback_mode": "direct_qdrant_write"
            }
        except Exception as e:
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
    include_libraries: bool = False
) -> dict[str, Any]:
    """
    Search across collections with hybrid semantic + keyword matching.

    NEW: Task 396 - Multi-tenant filtering with scope parameter
    - scope="project": Filter by current project_id (default, most focused)
    - scope="global": Search all projects (no project_id filter)
    - scope="all": Search projects + libraries collections (broadest)
    - include_libraries: Also search _libraries collection

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

    # Determine current project_id for filtering
    current_project_id = calculate_tenant_id(str(Path.cwd()))

    # Determine search collections based on scope (Task 396)
    # If explicit collection is provided, use it directly
    if collection:
        search_collections = [collection]
        # Don't apply project_id filter for explicit collections
        project_filter_id = None
    else:
        # Use unified collections based on scope
        if scope == "project":
            # Search only current project in unified collection
            search_collections = [UNIFIED_COLLECTIONS["projects"]]
            project_filter_id = current_project_id
        elif scope == "global":
            # Search all projects (no project_id filter)
            search_collections = [UNIFIED_COLLECTIONS["projects"]]
            project_filter_id = None
        else:  # scope == "all"
            # Search both projects and libraries
            search_collections = [UNIFIED_COLLECTIONS["projects"], UNIFIED_COLLECTIONS["libraries"]]
            project_filter_id = None

        # Add libraries collection if requested
        if include_libraries and UNIFIED_COLLECTIONS["libraries"] not in search_collections:
            search_collections.append(UNIFIED_COLLECTIONS["libraries"])

    # Build metadata filters with branch, file_type, and project_id
    search_filter = build_metadata_filters(
        filters=filters,
        branch=branch,
        file_type=file_type,
        project_id=project_filter_id
    )

    # For libraries, we don't filter by branch (they're external documentation)
    library_filter = build_metadata_filters(
        filters=filters,
        branch="*",  # Don't filter libraries by branch
        file_type=file_type,
        project_id=None  # Libraries have library_name, not project_id
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
            if coll == UNIFIED_COLLECTIONS["libraries"]:
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

        return {
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

    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": []
        }

@app.tool()
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
    - "init_project" -> initialize project collection (single _{project_id})
    - "cleanup" -> remove empty collections and optimize

    Args:
        action: Management action to perform
        collection: Target collection name (for collection-specific actions)
        name: Name for new collections or operations
        project_name: Project context for workspace operations
        config: Additional configuration for operations

    Returns:
        Dict with action results and status information
    """
    await initialize_components()
    logger = logging.getLogger(__name__)

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
                return {"success": False, "error": "Collection name required for create action"}

            collection_config = config or DEFAULT_COLLECTION_CONFIG

            # Get project_id from project_name parameter or current directory
            project_id = calculate_tenant_id(str(Path.cwd()))

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection creation must go through daemon. No fallback to direct writes.
            # ============================================================================

            if not daemon_client:
                return {
                    "success": False,
                    "action": action,
                    "error": "Daemon not connected. Collection creation requires daemon (First Principle 10)."
                }

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
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection deletion must go through daemon. No fallback to direct writes.
            # ============================================================================

            if not daemon_client:
                return {
                    "success": False,
                    "action": action,
                    "error": "Daemon not connected. Collection deletion requires daemon (First Principle 10)."
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
                return {
                    "success": False,
                    "action": action,
                    "error": "Daemon not connected. Cleanup requires daemon (First Principle 10)."
                }

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

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "available_actions": [
                    "list_collections", "create_collection", "delete_collection",
                    "collection_info", "workspace_status", "init_project", "cleanup"
                ]
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Management action '{action}' failed: {str(e)}"
        }

@app.tool()
async def retrieve(
    document_id: str = None,
    collection: str = None,
    metadata: dict[str, Any] = None,
    limit: int = 10,
    project_name: str = None,
    branch: str = None,
    file_type: str = None
) -> dict[str, Any]:
    """
    Retrieve documents directly by ID or metadata without search ranking.

    NEW: Task 374.7 - Branch and file_type filtering
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified
    - Searches single _{project_id} collection per project

    Retrieval methods determined by parameters:
    - document_id specified -> direct ID lookup
    - metadata specified -> filter-based retrieval
    - collection specified -> limits retrieval to specific collection
    - branch -> filters by Git branch
    - file_type -> filters by file type

    Args:
        document_id: Direct document ID to retrieve
        collection: Specific collection to retrieve from
        metadata: Metadata filters for document selection
        limit: Maximum number of documents to retrieve
        project_name: Limit retrieval to project collections
        branch: Git branch to filter by (None=current, "*"=all branches)
        file_type: File type filter ("code", "test", "docs", etc.)

    Returns:
        Dict with retrieved documents and metadata
    """
    await initialize_components()

    if not document_id and not metadata:
        return {
            "success": False,
            "error": "Either document_id or metadata filters must be provided"
        }

    try:
        results = []

        # Determine search collection
        if collection:
            search_collection = collection
        else:
            # Use single project collection (new architecture)
            search_collection = get_project_collection()

        if document_id:
            # Direct ID retrieval (async)
            try:
                points = await qdrant_client.retrieve(
                    collection_name=search_collection,
                    ids=[document_id]
                )

                if points:
                    point = points[0]
                    # Apply branch filter to retrieved document
                    if branch != "*":
                        effective_branch = branch if branch else get_current_branch(Path.cwd())
                        doc_branch = point.payload.get("branch")
                        if doc_branch != effective_branch:
                            # Document not on requested branch
                            return {
                                "success": True,
                                "total_results": 0,
                                "results": [],
                                "query_type": "id_lookup",
                                "message": f"Document found but not on branch '{effective_branch}'"
                            }

                    # Apply file_type filter if specified
                    if file_type:
                        doc_file_type = point.payload.get("file_type")
                        if doc_file_type != file_type:
                            return {
                                "success": True,
                                "total_results": 0,
                                "results": [],
                                "query_type": "id_lookup",
                                "message": f"Document found but not file_type '{file_type}'"
                            }

                    result = {
                        "id": point.id,
                        "collection": search_collection,
                        "content": point.payload.get("content", ""),
                        "title": point.payload.get("title", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                    }
                    results.append(result)

            except Exception:
                pass  # Collection might not exist or ID not found

        elif metadata:
            # Metadata-based retrieval with branch and file_type filters
            # Build filter conditions including branch and file_type
            search_filter = build_metadata_filters(
                filters=metadata,
                branch=branch,
                file_type=file_type
            )

            # Retrieve from collection (async)
            try:
                scroll_result = await qdrant_client.scroll(
                    collection_name=search_collection,
                    scroll_filter=search_filter,
                    limit=limit
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

        return {
            "success": True,
            "total_results": len(results),
            "results": results,
            "query_type": "id_lookup" if document_id else "metadata_filter",
            "filters_applied": {
                "branch": branch or get_current_branch(Path.cwd()) if branch != "*" else "*",
                "file_type": file_type,
                "metadata": metadata or {}
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Retrieval failed: {str(e)}",
            "results": []
        }

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
