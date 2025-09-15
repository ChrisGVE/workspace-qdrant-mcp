"""
Multi-tenant document memory tools for workspace-qdrant-mcp.

This module provides MCP tools for document storage operations with multi-tenant
collection support. Documents are stored with project-specific metadata filtering
to ensure proper isolation between different projects.

Key Features:
- Project-based metadata filtering for multi-tenant support
- Document CRUD operations (add, update, delete, get)
- Automatic project context detection and injection
- Validation for cross-project access prevention
- Integration with multi-tenant collection architecture

The tools provided:
- add_memory: Add documents with project isolation
- update_memory: Update existing documents with validation
- delete_memory: Delete documents with project context
- get_memory: Retrieve documents with metadata filtering
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from common.core.client import QdrantWorkspaceClient
from common.core.multitenant import (
    MultiTenantWorkspaceCollectionManager,
    WorkspaceCollectionRegistry,
)
from common.observability import monitor_async, record_operation
from common.core.error_handling import (
    ErrorRecoveryStrategy,
    with_error_handling,
    error_context,
)

# Import document operations from existing tools
from .documents import add_document, get_document, update_document, delete_document
from .multitenant_tools import add_document_with_project_context


def register_document_memory_tools(app: FastMCP, workspace_client: QdrantWorkspaceClient):
    """Register multi-tenant document memory tools with the FastMCP app."""

    @app.tool()
    @monitor_async("add_memory", timeout_warning=10.0, slow_threshold=5.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "add_memory")
    async def add_memory(
        content: str,
        collection: str,
        project_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
        document_type: str = "memory"
    ) -> Dict[str, Any]:
        """
        Add a document to multi-tenant collection with project isolation.

        This tool adds documents to collections with automatic project metadata
        injection for multi-tenant support. Documents are tagged with project
        context to ensure proper isolation between projects.

        Args:
            content: Document text content to be stored
            collection: Target collection name
            project_name: Project context (auto-detected if None)
            metadata: Additional metadata (enriched with project context)
            document_id: Custom document identifier (generated if None)
            chunk_text: Whether to split large documents into chunks
            document_type: Type classification for the document

        Returns:
            Dict containing success status, document_id, and metadata

        Example:
            ```python
            result = await add_memory(
                content="User authentication flow notes",
                collection="my-project-docs",
                metadata={"category": "security", "priority": "high"}
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized", "success": False}

        # Validate required parameters
        if not content or not content.strip():
            return {"error": "Content cannot be empty", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # Prepare enhanced metadata with project context
            base_metadata = metadata or {}
            base_metadata["document_type"] = document_type
            base_metadata["source"] = "memory_tool"
            base_metadata["created_at"] = datetime.now(timezone.utc).isoformat()

            # Use existing multi-tenant document addition
            result = await add_document_with_project_context(
                content=content,
                collection=collection,
                project_name=project_name,
                metadata=base_metadata,
                document_id=document_id,
                chunk_text=chunk_text,
                creator="memory_tool"
            )

            if result.get("success"):
                logger.info(
                    "Memory document added with project context",
                    document_id=result.get("document_id"),
                    project_name=project_name,
                    collection=collection
                )

            return result

        except Exception as e:
            logger.error(f"Failed to add memory document: {e}", exc_info=True)
            return {"error": f"Document addition failed: {str(e)}", "success": False}

    @app.tool()
    @monitor_async("get_memory", timeout_warning=5.0, slow_threshold=2.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "get_memory")
    async def get_memory(
        document_id: str,
        collection: str,
        project_name: Optional[str] = None,
        include_vectors: bool = False,
        validate_project_access: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve a document from multi-tenant collection with project validation.

        This tool retrieves documents with project-based access validation to
        ensure proper multi-tenant isolation. Only documents belonging to the
        specified or auto-detected project are accessible.

        Args:
            document_id: Unique identifier of the document to retrieve
            collection: Collection name containing the document
            project_name: Project context for validation (auto-detected if None)
            include_vectors: Whether to include embedding vectors in response
            validate_project_access: Whether to validate project-based access

        Returns:
            Dict containing document data with metadata validation

        Example:
            ```python
            doc = await get_memory(
                document_id="doc-123",
                collection="my-project-docs"
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized", "success": False}

        # Validate required parameters
        if not document_id or not document_id.strip():
            return {"error": "Document ID is required", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # Get document using existing function
            result = await get_document(
                client=workspace_client,
                document_id=document_id,
                collection=collection,
                include_vectors=include_vectors
            )

            # Validate project access if enabled
            if validate_project_access and project_name and result.get("success", False):
                payload = result.get("payload", {})
                doc_project = payload.get("project_name")

                if doc_project and doc_project != project_name:
                    logger.warning(
                        "Cross-project document access denied",
                        requested_doc_id=document_id,
                        doc_project=doc_project,
                        current_project=project_name
                    )
                    return {
                        "error": f"Access denied: Document belongs to project '{doc_project}'",
                        "success": False,
                        "access_violation": True
                    }

            if result.get("success"):
                logger.info(
                    "Memory document retrieved with project validation",
                    document_id=document_id,
                    project_name=project_name,
                    collection=collection
                )

            return result

        except Exception as e:
            logger.error(f"Failed to retrieve memory document: {e}", exc_info=True)
            return {"error": f"Document retrieval failed: {str(e)}", "success": False}

    @app.tool()
    @monitor_async("update_memory", timeout_warning=10.0, slow_threshold=5.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "update_memory")
    async def update_memory(
        document_id: str,
        collection: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        project_name: Optional[str] = None,
        validate_project_access: bool = True,
        preserve_project_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Update a document in multi-tenant collection with project validation.

        This tool updates documents while maintaining project-based isolation.
        It validates that the document belongs to the current project before
        allowing modifications and preserves project metadata.

        Args:
            document_id: Unique identifier of the document to update
            collection: Collection name containing the document
            content: New content for the document (optional)
            metadata: Updated metadata (merged with existing)
            project_name: Project context for validation (auto-detected if None)
            validate_project_access: Whether to validate project-based access
            preserve_project_metadata: Whether to preserve existing project metadata

        Returns:
            Dict containing update result and validation status

        Example:
            ```python
            result = await update_memory(
                document_id="doc-123",
                collection="my-project-docs",
                content="Updated authentication flow notes",
                metadata={"updated_reason": "security review"}
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized", "success": False}

        # Validate required parameters
        if not document_id or not document_id.strip():
            return {"error": "Document ID is required", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}
        if not content and not metadata:
            return {"error": "Either content or metadata must be provided for update", "success": False}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # First, get the existing document to validate project access
            if validate_project_access:
                existing_doc = await get_document(
                    client=workspace_client,
                    document_id=document_id,
                    collection=collection,
                    include_vectors=False
                )

                if not existing_doc.get("success"):
                    return {
                        "error": "Document not found or inaccessible",
                        "success": False
                    }

                # Validate project ownership
                payload = existing_doc.get("payload", {})
                doc_project = payload.get("project_name")

                if project_name and doc_project and doc_project != project_name:
                    logger.warning(
                        "Cross-project document update denied",
                        document_id=document_id,
                        doc_project=doc_project,
                        current_project=project_name
                    )
                    return {
                        "error": f"Access denied: Document belongs to project '{doc_project}'",
                        "success": False,
                        "access_violation": True
                    }

                # Prepare merged metadata preserving project context
                if preserve_project_metadata:
                    existing_metadata = payload.copy()
                    if metadata:
                        existing_metadata.update(metadata)
                    # Add update timestamp
                    existing_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
                    existing_metadata["updated_by"] = "memory_tool"
                    final_metadata = existing_metadata
                else:
                    final_metadata = metadata or {}

            else:
                # Use provided metadata directly
                final_metadata = metadata or {}
                if final_metadata:
                    final_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
                    final_metadata["updated_by"] = "memory_tool"

            # Perform the update using existing function
            result = await update_document(
                client=workspace_client,
                document_id=document_id,
                collection=collection,
                content=content,
                metadata=final_metadata
            )

            if result.get("success"):
                logger.info(
                    "Memory document updated with project validation",
                    document_id=document_id,
                    project_name=project_name,
                    collection=collection,
                    content_updated=content is not None,
                    metadata_updated=metadata is not None
                )

            return result

        except Exception as e:
            logger.error(f"Failed to update memory document: {e}", exc_info=True)
            return {"error": f"Document update failed: {str(e)}", "success": False}

    @app.tool()
    @monitor_async("delete_memory", timeout_warning=5.0, slow_threshold=2.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "delete_memory")
    async def delete_memory(
        document_id: str,
        collection: str,
        project_name: Optional[str] = None,
        validate_project_access: bool = True,
        cascade_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a document from multi-tenant collection with project validation.

        This tool deletes documents while enforcing project-based access control.
        It validates that the document belongs to the current project before
        allowing deletion to prevent accidental cross-project modifications.

        Args:
            document_id: Unique identifier of the document to delete
            collection: Collection name containing the document
            project_name: Project context for validation (auto-detected if None)
            validate_project_access: Whether to validate project-based access
            cascade_delete: Whether to delete related chunks (if applicable)

        Returns:
            Dict containing deletion result and validation status

        Example:
            ```python
            result = await delete_memory(
                document_id="doc-123",
                collection="my-project-docs"
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized", "success": False}

        # Validate required parameters
        if not document_id or not document_id.strip():
            return {"error": "Document ID is required", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # Validate project access before deletion
            if validate_project_access:
                existing_doc = await get_document(
                    client=workspace_client,
                    document_id=document_id,
                    collection=collection,
                    include_vectors=False
                )

                if not existing_doc.get("success"):
                    return {
                        "error": "Document not found or inaccessible",
                        "success": False
                    }

                # Validate project ownership
                payload = existing_doc.get("payload", {})
                doc_project = payload.get("project_name")

                if project_name and doc_project and doc_project != project_name:
                    logger.warning(
                        "Cross-project document deletion denied",
                        document_id=document_id,
                        doc_project=doc_project,
                        current_project=project_name
                    )
                    return {
                        "error": f"Access denied: Document belongs to project '{doc_project}'",
                        "success": False,
                        "access_violation": True
                    }

            # Perform the deletion using existing function
            result = await delete_document(
                client=workspace_client,
                document_id=document_id,
                collection=collection,
                cascade_delete=cascade_delete
            )

            if result.get("success"):
                logger.info(
                    "Memory document deleted with project validation",
                    document_id=document_id,
                    project_name=project_name,
                    collection=collection
                )

            return result

        except Exception as e:
            logger.error(f"Failed to delete memory document: {e}", exc_info=True)
            return {"error": f"Document deletion failed: {str(e)}", "success": False}

    logger.info("Multi-tenant document memory tools registered successfully")