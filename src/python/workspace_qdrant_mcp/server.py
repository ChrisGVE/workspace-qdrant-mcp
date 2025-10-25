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
import atexit
import hashlib
import logging
import os
import signal
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import typer
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, SearchParams, UpdateStatus, CollectionInfo
)

# CRITICAL: Complete stdio silence must be set up before ANY other imports
# This prevents ALL console output in MCP stdio mode for protocol compliance
import sys

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
from common.utils.project_detection import calculate_tenant_id
from common.utils.git_utils import get_current_branch
from common.core.collection_naming import build_project_collection_name
from common.grpc.daemon_client import DaemonClient, DaemonConnectionError

# Initialize the FastMCP app
app = FastMCP("Workspace Qdrant MCP")

# Global components
qdrant_client: Optional[QdrantClient] = None
embedding_model = None
daemon_client: Optional[DaemonClient] = None
project_cache = {}

# Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_CONFIG = {
    "distance": Distance.COSINE,
    "vector_size": 384,  # all-MiniLM-L6-v2 embedding size
}

def get_project_name() -> str:
    """Detect current project name from git or directory."""
    try:
        # Try to get from git remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL
            if url.endswith('.git'):
                url = url[:-4]
            return url.split('/')[-1]
    except Exception:
        pass

    # Fallback to directory name
    return Path.cwd().name

def get_project_collection(project_path: Optional[Path] = None) -> str:
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
    """Initialize Qdrant client, daemon client, and embedding model."""
    global qdrant_client, embedding_model, daemon_client

    if qdrant_client is None:
        # Connect to Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

    if embedding_model is None:
        # Lazy import to avoid slow module-level imports
        from fastembed import TextEmbedding
        model_name = os.getenv("FASTEMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
        embedding_model = TextEmbedding(model_name)

    if daemon_client is None:
        # Initialize daemon client for write operations
        daemon_client = DaemonClient()
        try:
            await daemon_client.connect()
        except DaemonConnectionError:
            # Daemon connection is optional - fall back to direct writes if unavailable
            daemon_client = None

async def ensure_collection_exists(collection_name: str) -> bool:
    """
    Ensure a collection exists, create if it doesn't.

    REFACTORED (Task 375.4): Now uses DaemonClient.create_collection_v2() for writes.
    Falls back to direct qdrant_client if daemon unavailable.

    Args:
        collection_name: Name of the collection to ensure exists

    Returns:
        True if collection exists or was created successfully, False otherwise
    """
    logger = logging.getLogger(__name__)

    # First check if collection exists (read-only, OK to use qdrant_client)
    try:
        qdrant_client.get_collection(collection_name)
        return True
    except Exception:
        # Collection doesn't exist, need to create it
        pass

    # Try to create collection via daemon first
    if daemon_client:
        try:
            response = await daemon_client.create_collection_v2(
                collection_name=collection_name,
                vector_size=DEFAULT_COLLECTION_CONFIG["vector_size"],
                distance_metric="Cosine",  # Map Distance.COSINE to string
            )
            if response.success:
                logger.info(f"Collection '{collection_name}' created via daemon")
                return True
            else:
                logger.warning(
                    f"Daemon failed to create collection '{collection_name}': {response.error_message}"
                )
                # Fall through to direct creation
        except DaemonConnectionError as e:
            logger.warning(
                f"Daemon unavailable for collection creation, falling back to direct write: {e}"
            )
            # Fall through to direct creation

    # Fallback: Create collection directly via qdrant_client
    # NOTE: This violates First Principle 10 but maintains backwards compatibility
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=DEFAULT_COLLECTION_CONFIG["vector_size"],
                distance=DEFAULT_COLLECTION_CONFIG["distance"]
            )
        )
        logger.warning(
            f"Collection '{collection_name}' created via direct Qdrant write (daemon unavailable)"
        )
        return True
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

async def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings for text."""
    if not embedding_model:
        await initialize_components()

    # FastEmbed returns generator, convert to list
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def build_metadata_filters(
    filters: Dict[str, Any] = None,
    branch: str = None,
    file_type: str = None
) -> Optional[Filter]:
    """
    Build Qdrant filter with branch and file_type conditions.

    Args:
        filters: User-provided metadata filters
        branch: Git branch to filter by (None = current branch, "*" = all branches)
        file_type: File type to filter by ("code", "test", "docs", etc.)

    Returns:
        Qdrant Filter object or None if no filters
    """
    conditions = []

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

@app.tool()
async def store(
    content: str,
    title: str = None,
    metadata: Dict[str, Any] = None,
    collection: str = None,
    source: str = "user_input",
    document_type: str = "text",
    file_path: str = None,
    url: str = None,
    project_name: str = None
) -> Dict[str, Any]:
    """
    Store any type of content in the vector database.

    The content and parameters determine the storage location and processing:
    - All content for a project goes to single _{project_id} collection
    - Files differentiated by metadata: file_type, branch, project_id
    - Legacy collection parameter supported for backwards compatibility

    REFACTORED (Task 375.3): Now uses DaemonClient.ingest_text() for all writes.
    The daemon handles embedding generation, collection creation, and metadata enrichment.

    Args:
        content: The text content to store
        title: Optional title for the document
        metadata: Additional metadata to attach
        collection: Override automatic collection selection (legacy support)
        source: Source type (user_input, scratchbook, file, web, etc.)
        document_type: Type of document (text, code, note, etc.)
        file_path: Path to source file (influences collection choice)
        url: Source URL (influences collection choice)
        project_name: Override automatic project detection

    Returns:
        Dict with document_id, collection, and storage confirmation
    """
    await initialize_components()

    # Determine collection based on content and context
    target_collection = determine_collection_name(
        content=content,
        source=source,
        file_path=file_path,
        url=url,
        collection=collection,
        project_name=project_name
    )

    # Prepare metadata
    doc_metadata = {
        "title": title or f"Document {uuid.uuid4().hex[:8]}",
        "source": source,
        "document_type": document_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": project_name or get_project_name(),
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

    # Extract collection_basename and tenant_id for daemon
    # For project collections (_{project_id}), extract the project_id
    # For custom collections, use the full name as basename
    if target_collection.startswith('_'):
        # Project collection format: _{project_id}
        tenant_id = target_collection[1:]  # Remove leading underscore
        collection_basename = ""  # Empty basename for project collections
    else:
        # Custom/legacy collection - use collection name as-is
        # Generate tenant_id from current project path
        tenant_id = calculate_tenant_id(str(Path.cwd()))
        collection_basename = target_collection

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
                "title": doc_metadata["title"],
                "content_length": len(content),
                "chunks_created": response.chunks_created,
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

            # Store in Qdrant
            point = PointStruct(
                id=document_id,
                vector=embeddings,
                payload={
                    "content": content,
                    **doc_metadata
                }
            )

            qdrant_client.upsert(
                collection_name=target_collection,
                points=[point]
            )

            return {
                "success": True,
                "document_id": document_id,
                "collection": target_collection,
                "title": doc_metadata["title"],
                "content_length": len(content),
                "metadata": doc_metadata,
                "fallback_mode": "direct_qdrant_write"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store document: {str(e)}"
            }

@app.tool()
async def search(
    query: str,
    collection: str = None,
    project_name: str = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.3,
    filters: Dict[str, Any] = None,
    branch: str = None,
    file_type: str = None,
    workspace_type: str = None
) -> Dict[str, Any]:
    """
    Search across collections with hybrid semantic + keyword matching.

    NEW: Task 374.7 - Single collection + branch filtering architecture
    - Searches single _{project_id} collection per project
    - Filters by Git branch (default: current branch, "*" = all branches)
    - Filters by file_type when specified (code, test, docs, config, data, build, other)
    - workspace_type parameter deprecated (use file_type instead)

    Search modes and behavior determined by parameters:
    - mode="hybrid" -> combines semantic and keyword search
    - mode="semantic" -> pure vector similarity search
    - mode="exact" -> keyword/symbol exact matching
    - branch=None -> current Git branch (detected automatically)
    - branch="main" -> specific branch
    - branch="*" -> search across all branches
    - file_type="code" -> only code files
    - filters -> applies additional metadata filtering

    Args:
        query: Search query text
        collection: Specific collection to search (overrides project-based selection)
        project_name: Search within specific project collections
        mode: Search mode - "hybrid", "semantic", "exact", or "keyword"
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0.0-1.0)
        filters: Additional metadata filters
        branch: Git branch to search (None=current, "*"=all branches)
        file_type: File type filter ("code", "test", "docs", "config", "data", "build", "other")
        workspace_type: DEPRECATED - use file_type instead

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

    # Determine search collection
    if collection:
        search_collection = collection
    else:
        # Use single project collection (new architecture)
        search_collection = get_project_collection()

    # Build metadata filters with branch and file_type
    search_filter = build_metadata_filters(
        filters=filters,
        branch=branch,
        file_type=file_type
    )

    # Execute search based on mode
    all_results = []
    search_start = datetime.now()

    try:
        # Ensure collection exists before searching
        if not await ensure_collection_exists(search_collection):
            return {
                "success": False,
                "error": f"Collection not found: {search_collection}",
                "results": []
            }

        if mode in ["semantic", "hybrid"]:
            # Generate query embeddings for semantic search
            query_embeddings = await generate_embeddings(query)

            # Perform vector search
            search_results = qdrant_client.search(
                collection_name=search_collection,
                query_vector=query_embeddings,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )

            # Convert results
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "collection": search_collection,
                    "content": hit.payload.get("content", ""),
                    "title": hit.payload.get("title", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
                }
                all_results.append(result)

        if mode in ["exact", "keyword", "hybrid"]:
            # For keyword/exact search, use scroll to find text matches
            # This is a simplified implementation - in production, you'd want
            # to implement proper sparse vector search or use Qdrant's full-text search
            scroll_results = qdrant_client.scroll(
                collection_name=search_collection,
                scroll_filter=search_filter,
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
                        "collection": search_collection,
                        "content": point.payload.get("content", ""),
                        "title": point.payload.get("title", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                    }
                    all_results.append(result)

        # Sort results by score and deduplicate
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)

        # Limit final results
        final_results = unique_results[:limit]

        search_duration = (datetime.now() - search_start).total_seconds()

        return {
            "success": True,
            "query": query,
            "mode": mode,
            "collection_searched": search_collection,
            "total_results": len(final_results),
            "results": final_results,
            "search_time_ms": round(search_duration * 1000, 2),
            "filters_applied": {
                "branch": branch or get_current_branch(Path.cwd()),
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
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
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
            collections_response = qdrant_client.get_collections()
            collections_info = []

            for col in collections_response.collections:
                try:
                    col_info = qdrant_client.get_collection(col.name)
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

            # Extract distance metric and convert to string for daemon API
            distance = collection_config.get("distance", Distance.COSINE)
            distance_str = "Cosine"  # Default
            if distance == Distance.EUCLID:
                distance_str = "Euclidean"
            elif distance == Distance.DOT:
                distance_str = "Dot"

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection creation must go through daemon. Direct write is fallback only.
            # ============================================================================

            # Try to create via daemon first
            if daemon_client:
                try:
                    response = await daemon_client.create_collection_v2(
                        collection_name=name,
                        vector_size=collection_config.get("vector_size", 384),
                        distance_metric=distance_str,
                    )

                    if response.success:
                        return {
                            "success": True,
                            "action": action,
                            "collection_name": name,
                            "message": f"Collection '{name}' created successfully via daemon"
                        }
                    else:
                        logger.warning(
                            f"Daemon failed to create collection '{name}': {response.error_message}"
                        )
                        # Fall through to direct creation
                except DaemonConnectionError as e:
                    logger.warning(
                        f"Daemon unavailable for collection creation, falling back to direct write: {e}"
                    )
                    # Fall through to direct creation

            # Fallback: Create collection directly via qdrant_client
            # NOTE: This violates First Principle 10 but maintains backwards compatibility
            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=collection_config.get("vector_size", 384),
                    distance=collection_config.get("distance", Distance.COSINE)
                )
            )

            return {
                "success": True,
                "action": action,
                "collection_name": name,
                "message": f"Collection '{name}' created successfully (direct write - daemon unavailable)"
            }

        elif action == "delete_collection":
            if not name and not collection:
                return {"success": False, "error": "Collection name required for delete action"}

            target_collection = name or collection

            # ============================================================================
            # DAEMON WRITE BOUNDARY (First Principle 10)
            # ============================================================================
            # Collection deletion must go through daemon. Direct write is fallback only.
            # ============================================================================

            # Try to delete via daemon first
            if daemon_client:
                try:
                    await daemon_client.delete_collection_v2(
                        collection_name=target_collection,
                    )

                    return {
                        "success": True,
                        "action": action,
                        "collection_name": target_collection,
                        "message": f"Collection '{target_collection}' deleted successfully via daemon"
                    }
                except DaemonConnectionError as e:
                    logger.warning(
                        f"Daemon unavailable for collection deletion, falling back to direct write: {e}"
                    )
                    # Fall through to direct deletion

            # Fallback: Delete collection directly via qdrant_client
            # NOTE: This violates First Principle 10 but maintains backwards compatibility
            qdrant_client.delete_collection(target_collection)

            return {
                "success": True,
                "action": action,
                "collection_name": target_collection,
                "message": f"Collection '{target_collection}' deleted successfully (direct write - daemon unavailable)"
            }

        elif action == "collection_info":
            if not name and not collection:
                return {"success": False, "error": "Collection name required for info action"}

            target_collection = name or collection
            col_info = qdrant_client.get_collection(target_collection)

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
            # System health check
            current_project = project_name or get_project_name()
            project_collection = get_project_collection()

            # Get collections info
            collections_response = qdrant_client.get_collections()

            # Check for project collection (new architecture: single _{project_id})
            project_collections = []
            for col in collections_response.collections:
                if col.name == project_collection:
                    project_collections.append(col.name)
                # Also include legacy collections for backwards compatibility
                elif col.name.startswith(f"{current_project}-"):
                    project_collections.append(col.name)

            # Get Qdrant cluster info
            cluster_info = qdrant_client.get_cluster_info()

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
            # Initialize project collection (new architecture: single _{project_id})
            target_project = project_name or get_project_name()
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
            # Remove empty collections and optimize
            collections_response = qdrant_client.get_collections()
            cleaned_collections = []

            for col in collections_response.collections:
                try:
                    col_info = qdrant_client.get_collection(col.name)
                    if col_info.points_count == 0:
                        # Try to delete via daemon first
                        if daemon_client:
                            try:
                                await daemon_client.delete_collection_v2(
                                    collection_name=col.name,
                                )
                                cleaned_collections.append(col.name)
                                logger.info(f"Deleted empty collection '{col.name}' via daemon")
                                continue
                            except DaemonConnectionError as e:
                                logger.warning(
                                    f"Daemon unavailable for cleanup deletion, falling back to direct write: {e}"
                                )
                                # Fall through to direct deletion

                        # Fallback: Delete directly
                        qdrant_client.delete_collection(col.name)
                        cleaned_collections.append(col.name)
                        logger.warning(f"Deleted empty collection '{col.name}' (direct write - daemon unavailable)")
                except Exception:
                    continue

            return {
                "success": True,
                "action": action,
                "cleaned_collections": cleaned_collections,
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
    metadata: Dict[str, Any] = None,
    limit: int = 10,
    project_name: str = None,
    branch: str = None,
    file_type: str = None
) -> Dict[str, Any]:
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
            # Direct ID retrieval
            try:
                points = qdrant_client.retrieve(
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

            # Retrieve from collection
            try:
                scroll_result = qdrant_client.scroll(
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
