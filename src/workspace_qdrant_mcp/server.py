"""
FastMCP server for workspace-qdrant-mcp.

Provides project-scoped Qdrant collections with scratchbook functionality.
"""

import asyncio
import logging
import os
from typing import Optional

import typer
from fastmcp import FastMCP
from pydantic import BaseModel

from .core.config import Config
from .core.client import QdrantWorkspaceClient
from .utils.config_validator import ConfigValidator
from .tools.search import search_workspace, search_collection_by_metadata
from .core.hybrid_search import HybridSearchEngine
from .tools.documents import add_document, update_document, delete_document, get_document
from .tools.scratchbook import update_scratchbook, ScratchbookManager

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize FastMCP application
app = FastMCP("workspace-qdrant-mcp")

# Global client instance
workspace_client: Optional[QdrantWorkspaceClient] = None


class ServerInfo(BaseModel):
    """Server information model."""
    
    name: str = "workspace-qdrant-mcp"
    version: str = "0.1.0"
    description: str = "Project-scoped Qdrant MCP server with scratchbook functionality"


@app.tool()
async def workspace_status() -> dict:
    """Get workspace and collection status information."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await workspace_client.get_status()


@app.tool()
async def list_workspace_collections() -> list[str]:
    """List all available workspace collections."""
    if not workspace_client:
        return []
    
    return await workspace_client.list_collections()


@app.tool()
async def search_workspace_tool(
    query: str,
    collections: list[str] = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7
) -> dict:
    """Search across workspace collections with hybrid search."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await search_workspace(
        workspace_client,
        query,
        collections,
        mode,
        limit,
        score_threshold
    )


@app.tool()
async def add_document_tool(
    content: str,
    collection: str,
    metadata: dict = None,
    document_id: str = None,
    chunk_text: bool = True
) -> dict:
    """Add document to specified collection."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await add_document(
        workspace_client,
        content,
        collection,
        metadata,
        document_id,
        chunk_text
    )


@app.tool()
async def get_document_tool(
    document_id: str,
    collection: str,
    include_vectors: bool = False
) -> dict:
    """Retrieve a document from collection."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await get_document(
        workspace_client,
        document_id,
        collection,
        include_vectors
    )


@app.tool()
async def search_by_metadata_tool(
    collection: str,
    metadata_filter: dict,
    limit: int = 10
) -> dict:
    """Search collection by metadata filter."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await search_collection_by_metadata(
        workspace_client,
        collection,
        metadata_filter,
        limit
    )


@app.tool()
async def update_scratchbook_tool(
    content: str,
    note_id: str = None,
    title: str = None,
    tags: list[str] = None,
    note_type: str = "note"
) -> dict:
    """Add or update a scratchbook note."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    return await update_scratchbook(
        workspace_client,
        content,
        note_id,
        title,
        tags,
        note_type
    )


@app.tool()
async def search_scratchbook_tool(
    query: str,
    note_types: list[str] = None,
    tags: list[str] = None,
    project_name: str = None,
    limit: int = 10,
    mode: str = "hybrid"
) -> dict:
    """Search scratchbook notes with specialized filtering."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    manager = ScratchbookManager(workspace_client)
    return await manager.search_notes(
        query,
        note_types,
        tags,
        project_name,
        limit,
        mode
    )


@app.tool()
async def list_scratchbook_notes_tool(
    project_name: str = None,
    note_type: str = None,
    tags: list[str] = None,
    limit: int = 50
) -> dict:
    """List notes in scratchbook with optional filtering."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    manager = ScratchbookManager(workspace_client)
    return await manager.list_notes(
        project_name,
        note_type,
        tags,
        limit
    )


@app.tool()
async def delete_scratchbook_note_tool(
    note_id: str,
    project_name: str = None
) -> dict:
    """Delete a note from the scratchbook."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    manager = ScratchbookManager(workspace_client)
    return await manager.delete_note(note_id, project_name)


@app.tool()
async def hybrid_search_advanced_tool(
    query: str,
    collection: str,
    fusion_method: str = "rrf",
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    limit: int = 10,
    score_threshold: float = 0.0
) -> dict:
    """Advanced hybrid search with configurable fusion methods."""
    if not workspace_client:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Validate collection exists
        available_collections = await workspace_client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Generate embeddings
        embedding_service = workspace_client.get_embedding_service()
        embeddings = await embedding_service.generate_embeddings(query, include_sparse=True)
        
        # Perform hybrid search
        hybrid_engine = HybridSearchEngine(workspace_client.client)
        result = await hybrid_engine.hybrid_search(
            collection_name=collection,
            query_embeddings=embeddings,
            limit=limit,
            score_threshold=score_threshold,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            fusion_method=fusion_method
        )
        
        return result
        
    except Exception as e:
        logger.error("Advanced hybrid search failed: %s", e)
        return {"error": f"Advanced hybrid search failed: {e}"}


async def initialize_workspace() -> None:
    """Initialize the workspace client and collections."""
    global workspace_client
    
    # Load configuration
    config = Config()
    
    # Validate configuration
    validator = ConfigValidator(config)
    is_valid, validation_results = validator.validate_all()
    
    if not is_valid:
        print("❌ Configuration validation failed:")
        for issue in validation_results["issues"]:
            print(f"  • {issue}")
        raise RuntimeError("Configuration validation failed")
    
    # Show warnings if any
    if validation_results["warnings"]:
        print("⚠️  Configuration warnings:")
        for warning in validation_results["warnings"]:
            print(f"  • {warning}")
    
    # Initialize Qdrant workspace client
    workspace_client = QdrantWorkspaceClient(config)
    
    # Initialize collections for current project
    await workspace_client.initialize()


def run_server(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
) -> None:
    """Run the workspace-qdrant-mcp server."""
    
    # Set configuration file if provided
    if config_file:
        os.environ["CONFIG_FILE"] = config_file
    
    # Initialize workspace before running server
    asyncio.run(initialize_workspace())
    
    # Run FastMCP server
    app.run(host=host, port=port)


def main() -> None:
    """Console script entry point for uv tool installation."""
    typer.run(run_server)


if __name__ == "__main__":
    main()