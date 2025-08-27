"""
FastMCP server for workspace-qdrant-mcp.

Provides project-scoped Qdrant collections with scratchbook functionality.
"""

import asyncio
import os
from typing import Optional

import typer
from fastmcp import FastMCP
from pydantic import BaseModel

from .core.config import Config
from .core.client import QdrantWorkspaceClient
from .utils.config_validator import ConfigValidator
from .tools.search import search_workspace, search_collection_by_metadata
from .tools.documents import add_document, update_document, delete_document, get_document

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


def main(
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


if __name__ == "__main__":
    typer.run(main)