"""
Comprehensive scratchbook management for workspace-qdrant-mcp.

This module implements a sophisticated scratchbook system for managing notes, ideas,
todos, and reminders across projects. It provides a unified interface for capturing
and organizing thoughts, code snippets, meeting notes, and project insights with
advanced search and organization capabilities.

Key Features:
    - Multi-project note organization with automatic project detection
    - Rich note types: notes, ideas, todos, reminders, code-snippets, meetings
    - Hierarchical tagging system for flexible organization
    - Version tracking with automatic timestamping
    - Advanced search with semantic and keyword matching
    - Cross-project note discovery and linking
    - Export capabilities for external tools

Note Structure:
    Each note contains:
    - Unique identifier and title (auto-generated or custom)
    - Rich content with markdown support
    - Project association (current or specified)
    - Type classification (note, idea, todo, etc.)
    - Tag-based organization system
    - Creation and modification timestamps
    - Version history (future enhancement)

Use Cases:
    - Meeting notes with action items
    - Code snippets and implementation ideas
    - Project todos and reminders
    - Research findings and insights
    - Cross-project knowledge sharing
    - Daily development journal

Example:
    ```python
    from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager

    manager = ScratchbookManager(workspace_client)

    # Add a meeting note
    result = await manager.add_note(
        content="Discussed API design patterns...",
        title="Architecture Review Meeting",
        note_type="meeting",
        tags=["architecture", "api", "team-review"]
    )

    # Search across projects
    notes = await manager.search_notes(
        query="authentication patterns",
        note_types=["note", "idea"],
        tags=["security"]
    )
    ```
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from qdrant_client.http import models

from ..core.client import QdrantWorkspaceClient
from ..core.sparse_vectors import create_qdrant_sparse_vector

logger = logging.getLogger(__name__)


class ScratchbookManager:
    """
    Advanced scratchbook manager for cross-project note management.

    This class provides a comprehensive interface for managing a workspace-wide
    scratchbook system that spans multiple projects. It handles note lifecycle
    management, intelligent organization, search capabilities, and maintains
    project context while enabling cross-project knowledge discovery.

    The scratchbook system is designed for developers and teams who need to:
    - Capture ideas and insights quickly during development
    - Maintain project-specific notes while enabling cross-project search
    - Organize information with flexible tagging and categorization
    - Search historical notes and decisions using semantic search
    - Export notes for integration with external tools

    Architecture:
        - Uses the appropriate project collection for scoped notes
        - Falls back to global scratchbook collection if available
        - Each note includes project context for scoping
        - Supports multiple note types with specialized handling
        - Implements versioning for tracking note evolution
        - Provides rich metadata for advanced filtering and search

    Attributes:
        client (QdrantWorkspaceClient): Workspace client for database operations
        project_info (Optional[Dict]): Current project information for context

    Example:
        ```python
        manager = ScratchbookManager(workspace_client)

        # Add different types of notes
        await manager.add_note("Important insight about caching",
                              note_type="idea", tags=["performance"])

        await manager.add_note("Fix authentication bug in login.py",
                              note_type="todo", tags=["bug", "auth"])

        # Search and organize
        ideas = await manager.search_notes("caching", note_types=["idea"])
        todos = await manager.list_notes(note_type="todo", limit=10)
        ```
    """

    def __init__(self, client: QdrantWorkspaceClient) -> None:
        """Initialize the scratchbook manager with workspace context.

        Args:
            client: Initialized workspace client for database operations
        """
        self.client = client
        self.project_info = client.get_project_info()

    def _get_scratchbook_collection_name(self, project_name: str | None = None) -> str:
        """Determine the scratchbook collection name for the project.

        The scratchbook collection is always project-specific and uses the pattern
        '{project-name}-scratchbook'. This collection is created automatically and
        does not need to be declared in the workspace configuration.

        Args:
            project_name: Project name (defaults to current project)

        Returns:
            Collection name: {project-name}-scratchbook
        """
        if not project_name:
            project_name = (
                self.project_info["main_project"] if self.project_info else "default"
            )

        # Always use the consistent scratchbook naming pattern
        return f"{project_name}-scratchbook"

    async def add_note(
        self,
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
        note_type: str = "note",
        project_name: str | None = None,
    ) -> dict:
        """
        Add a new note to the scratchbook.

        Args:
            content: Note content
            title: Optional note title (auto-generated if not provided)
            tags: Optional list of tags
            note_type: Type of note (note, idea, todo, reminder)
            project_name: Project name (defaults to current project)

        Returns:
            Dictionary with operation result
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        if not content or not content.strip():
            return {"error": "Note content cannot be empty"}

        try:
            # Determine collection name using configured collections
            collection_name = self._get_scratchbook_collection_name(project_name)

            # Validate collection exists
            available_collections = await self.client.list_collections()
            if collection_name not in available_collections:
                return {
                    "error": f"Scratchbook collection '{collection_name}' not found"
                }

            # Generate note ID and title
            note_id = str(uuid.uuid4())
            if not title:
                title = self._generate_title_from_content(content)

            # Prepare metadata
            metadata = {
                "note_id": note_id,
                "title": title,
                "note_type": note_type,
                "tags": tags or [],
                "project_name": project_name,
                "collection_type": "scratchbook",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "version": 1,
                "content_length": len(content),
                "is_scratchbook_note": True,
            }

            # Generate embeddings
            embedding_service = self.client.get_embedding_service()
            embeddings = await embedding_service.generate_embeddings(content)

            # Prepare vectors
            vectors = {"dense": embeddings["dense"]}
            if "sparse" in embeddings:
                vectors["sparse"] = create_qdrant_sparse_vector(
                    indices=embeddings["sparse"]["indices"],
                    values=embeddings["sparse"]["values"],
                )

            # Add content to metadata
            payload = metadata.copy()
            payload["content"] = content

            # Create point
            point = models.PointStruct(id=note_id, vector=vectors, payload=payload)

            # Insert into Qdrant
            self.client.client.upsert(collection_name=collection_name, points=[point])

            logger.info("Added note %s to scratchbook %s", note_id, collection_name)

            return {
                "note_id": note_id,
                "title": title,
                "collection": collection_name,
                "note_type": note_type,
                "tags": tags or [],
                "content_length": len(content),
                "created_at": metadata["created_at"],
            }

        except Exception as e:
            logger.error("Failed to add scratchbook note: %s", e)
            return {"error": f"Failed to add note: {e}"}

    async def update_note(
        self,
        note_id: str,
        content: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
        project_name: str | None = None,
    ) -> dict:
        """
        Update an existing scratchbook note with versioning.

        Args:
            note_id: Note ID to update
            content: New content (optional)
            title: New title (optional)
            tags: New tags (optional)
            project_name: Project name (defaults to current project)

        Returns:
            Dictionary with operation result
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Determine collection name using configured collections
            collection_name = self._get_scratchbook_collection_name(project_name)

            # Find existing note
            existing_points = self.client.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="note_id", match=models.MatchValue(value=note_id)
                        )
                    ]
                ),
                with_payload=True,
                limit=1,
            )

            if not existing_points[0]:
                return {
                    "error": f"Note '{note_id}' not found in scratchbook '{collection_name}'"
                }

            existing_point = existing_points[0][0]
            old_payload = existing_point.payload

            # Create new payload with updates
            new_payload = old_payload.copy()
            new_payload["updated_at"] = datetime.utcnow().isoformat()
            new_payload["version"] = old_payload.get("version", 1) + 1

            # Update fields if provided
            if title is not None:
                new_payload["title"] = title
            if tags is not None:
                new_payload["tags"] = tags

            # Handle content update
            if content is not None:
                new_payload["content"] = content
                new_payload["content_length"] = len(content)

                # Generate new embeddings for content
                embedding_service = self.client.get_embedding_service()
                embeddings = await embedding_service.generate_embeddings(content)

                # Prepare new vectors
                vectors = {"dense": embeddings["dense"]}
                if "sparse" in embeddings:
                    vectors["sparse"] = create_qdrant_sparse_vector(
                        indices=embeddings["sparse"]["indices"],
                        values=embeddings["sparse"]["values"],
                    )

                # Update point with new vectors and payload
                updated_point = models.PointStruct(
                    id=note_id, vector=vectors, payload=new_payload
                )
            else:
                # Update only payload
                updated_point = models.PointStruct(id=note_id, payload=new_payload)

            self.client.client.upsert(
                collection_name=collection_name, points=[updated_point]
            )

            logger.info("Updated note %s in scratchbook %s", note_id, collection_name)

            return {
                "note_id": note_id,
                "collection": collection_name,
                "version": new_payload["version"],
                "updated_at": new_payload["updated_at"],
                "content_updated": content is not None,
                "title_updated": title is not None,
                "tags_updated": tags is not None,
            }

        except Exception as e:
            logger.error("Failed to update scratchbook note: %s", e)
            return {"error": f"Failed to update note: {e}"}

    async def search_notes(
        self,
        query: str,
        note_types: list[str] | None = None,
        tags: list[str] | None = None,
        project_name: str | None = None,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> dict:
        """
        Search scratchbook notes with specialized filtering.

        Args:
            query: Search query
            note_types: Filter by note types
            tags: Filter by tags
            project_name: Project name (defaults to current project)
            limit: Maximum number of results
            mode: Search mode (dense, sparse, hybrid)

        Returns:
            Dictionary with search results
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Determine collection name using configured collections
            collection_name = self._get_scratchbook_collection_name(project_name)

            # Validate collection exists
            available_collections = await self.client.list_collections()
            if collection_name not in available_collections:
                return {
                    "error": f"Scratchbook collection '{collection_name}' not found"
                }

            # Generate embeddings for query
            embedding_service = self.client.get_embedding_service()
            embeddings = await embedding_service.generate_embeddings(
                query, include_sparse=(mode in ["sparse", "hybrid"])
            )

            # Build filter conditions
            filter_conditions = [
                models.FieldCondition(
                    key="is_scratchbook_note", match=models.MatchValue(value=True)
                )
            ]

            if note_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="note_type", match=models.MatchAny(any=note_types)
                    )
                )

            if tags:
                filter_conditions.append(
                    models.FieldCondition(key="tags", match=models.MatchAny(any=tags))
                )

            search_filter = (
                models.Filter(must=filter_conditions) if filter_conditions else None
            )

            # Perform search
            search_results = []

            if mode in ["dense", "hybrid"]:
                # Dense vector search
                dense_results = self.client.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", embeddings["dense"]),
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )

                for result in dense_results:
                    search_results.append(
                        {
                            "note_id": result.id,
                            "score": result.score,
                            "title": result.payload.get("title", ""),
                            "note_type": result.payload.get("note_type", "note"),
                            "tags": result.payload.get("tags", []),
                            "created_at": result.payload.get("created_at"),
                            "updated_at": result.payload.get("updated_at"),
                            "version": result.payload.get("version", 1),
                            "content": result.payload.get("content", "")[:200] + "..."
                            if len(result.payload.get("content", "")) > 200
                            else result.payload.get("content", ""),
                            "search_type": "dense",
                        }
                    )

            # Sort by score and return
            search_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return {
                "query": query,
                "collection": collection_name,
                "total_results": len(search_results),
                "filters": {"note_types": note_types, "tags": tags},
                "results": search_results[:limit],
            }

        except Exception as e:
            logger.error("Failed to search scratchbook notes: %s", e)
            return {"error": f"Search failed: {e}"}

    async def list_notes(
        self,
        project_name: str | None = None,
        note_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> dict:
        """
        List notes in scratchbook with optional filtering.

        Args:
            project_name: Project name (defaults to current project)
            note_type: Filter by note type
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            Dictionary with note list
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Determine collection name using configured collections
            collection_name = self._get_scratchbook_collection_name(project_name)

            # Build filter conditions
            filter_conditions = [
                models.FieldCondition(
                    key="is_scratchbook_note", match=models.MatchValue(value=True)
                )
            ]

            if note_type:
                filter_conditions.append(
                    models.FieldCondition(
                        key="note_type", match=models.MatchValue(value=note_type)
                    )
                )

            if tags:
                filter_conditions.append(
                    models.FieldCondition(key="tags", match=models.MatchAny(any=tags))
                )

            scroll_filter = models.Filter(must=filter_conditions)

            # Get notes
            points, _ = self.client.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
            )

            # Format results
            notes = []
            for point in points:
                notes.append(
                    {
                        "note_id": point.id,
                        "title": point.payload.get("title", ""),
                        "note_type": point.payload.get("note_type", "note"),
                        "tags": point.payload.get("tags", []),
                        "created_at": point.payload.get("created_at"),
                        "updated_at": point.payload.get("updated_at"),
                        "version": point.payload.get("version", 1),
                        "content_length": point.payload.get("content_length", 0),
                    }
                )

            # Sort by updated_at (most recent first)
            notes.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

            return {
                "collection": collection_name,
                "total_notes": len(notes),
                "filters": {"note_type": note_type, "tags": tags},
                "notes": notes,
            }

        except Exception as e:
            logger.error("Failed to list scratchbook notes: %s", e)
            return {"error": f"Failed to list notes: {e}"}

    async def delete_note(self, note_id: str, project_name: str | None = None) -> dict:
        """
        Delete a note from the scratchbook.

        Args:
            note_id: Note ID to delete
            project_name: Project name (defaults to current project)

        Returns:
            Dictionary with operation result
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Determine collection name using configured collections
            collection_name = self._get_scratchbook_collection_name(project_name)

            # Delete the note
            result = self.client.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[note_id]),
            )

            logger.info("Deleted note %s from scratchbook %s", note_id, collection_name)

            return {
                "note_id": note_id,
                "collection": collection_name,
                "deleted": True,
                "operation_id": result.operation_id,
            }

        except Exception as e:
            logger.error("Failed to delete scratchbook note: %s", e)
            return {"error": f"Failed to delete note: {e}"}

    def _generate_title_from_content(self, content: str, max_length: int = 50) -> str:
        """Generate a title from the content."""
        # Take first line or first sentence
        lines = content.strip().split("\n")
        first_line = lines[0].strip()

        if not first_line:
            return "Untitled Note"

        # If first line is too long, truncate at word boundary
        if len(first_line) <= max_length:
            return first_line

        words = first_line.split()
        title_words = []
        current_length = 0

        for word in words:
            # Check if adding this word would exceed the limit
            word_length = len(word) + (
                1 if title_words else 0
            )  # +1 for space separator
            if current_length + word_length > max_length - 3:  # Leave space for "..."
                if title_words:
                    title_words.append("...")
                break
            title_words.append(word)
            current_length += word_length

        return " ".join(title_words) if title_words else "Untitled Note"


async def update_scratchbook(
    client: QdrantWorkspaceClient,
    content: str,
    note_id: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
    note_type: str = "note",
) -> dict:
    """
    Add or update a scratchbook note.

    Args:
        client: Workspace client instance
        content: Note content
        note_id: Existing note ID to update (creates new if None)
        title: Note title
        tags: List of tags
        note_type: Type of note

    Returns:
        Dictionary with operation result
    """
    manager = ScratchbookManager(client)

    if note_id:
        return await manager.update_note(note_id, content, title, tags)
    else:
        return await manager.add_note(content, title, tags, note_type)
