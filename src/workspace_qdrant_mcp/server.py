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


async def initialize_workspace() -> None:
    """Initialize the workspace client and collections."""
    global workspace_client
    
    # Load configuration
    config = Config()
    
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