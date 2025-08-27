"""
Scratchbook-specific functionality for workspace-qdrant-mcp.

Provides specialized tools for managing scratchbook collections with notes and ideas.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.client import QdrantWorkspaceClient
from ..core.sparse_vectors import create_qdrant_sparse_vector

logger = logging.getLogger(__name__)


class ScratchbookManager:
    """
    Manages scratchbook collections with specialized note functionality.
    
    Provides note versioning, organization, and search capabilities.
    """
    
    def __init__(self, client: QdrantWorkspaceClient):
        self.client = client
        self.project_info = client.get_project_info()
    
    async def add_note(
        self,
        content: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        note_type: str = "note",
        project_name: Optional[str] = None
    ) -> Dict:
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
            # Determine collection name
            if not project_name:
                project_name = self.project_info["main_project"] if self.project_info else "default"
            
            collection_name = f"{project_name}-scratchbook"
            
            # Validate collection exists
            available_collections = await self.client.list_collections()
            if collection_name not in available_collections:
                return {"error": f"Scratchbook collection '{collection_name}' not found"}
            
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
                "is_scratchbook_note": True
            }
            
            # Generate embeddings
            embedding_service = self.client.get_embedding_service()
            embeddings = await embedding_service.generate_embeddings(content)
            
            # Prepare vectors
            vectors = {"dense": embeddings["dense"]}
            if "sparse" in embeddings:
                vectors["sparse"] = create_qdrant_sparse_vector(
                    indices=embeddings["sparse"]["indices"],
                    values=embeddings["sparse"]["values"]
                )
            
            # Add content to metadata
            payload = metadata.copy()
            payload["content"] = content
            
            # Create point
            point = models.PointStruct(
                id=note_id,
                vector=vectors,
                payload=payload
            )
            
            # Insert into Qdrant
            self.client.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.info("Added note %s to scratchbook %s", note_id, collection_name)
            
            return {
                "note_id": note_id,
                "title": title,
                "collection": collection_name,
                "note_type": note_type,
                "tags": tags or [],
                "content_length": len(content),
                "created_at": metadata["created_at"]
            }
            
        except Exception as e:
            logger.error("Failed to add scratchbook note: %s", e)
            return {"error": f"Failed to add note: {e}"}
    
    async def update_note(
        self,
        note_id: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        project_name: Optional[str] = None
    ) -> Dict:
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
            # Determine collection name
            if not project_name:
                project_name = self.project_info["main_project"] if self.project_info else "default"
            
            collection_name = f"{project_name}-scratchbook"
            
            # Find existing note
            existing_points = self.client.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="note_id",
                            match=models.MatchValue(value=note_id)
                        )
                    ]
                ),
                with_payload=True,
                limit=1
            )
            
            if not existing_points[0]:
                return {"error": f"Note '{note_id}' not found in scratchbook '{collection_name}'"}
            
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
                        values=embeddings["sparse"]["values"]
                    )
                
                # Update point with new vectors and payload
                updated_point = models.PointStruct(
                    id=note_id,
                    vector=vectors,
                    payload=new_payload
                )
            else:
                # Update only payload
                updated_point = models.PointStruct(
                    id=note_id,
                    payload=new_payload
                )
            
            self.client.client.upsert(
                collection_name=collection_name,
                points=[updated_point]
            )
            
            logger.info("Updated note %s in scratchbook %s", note_id, collection_name)
            
            return {
                "note_id": note_id,
                "collection": collection_name,
                "version": new_payload["version"],
                "updated_at": new_payload["updated_at"],
                "content_updated": content is not None,
                "title_updated": title is not None,
                "tags_updated": tags is not None
            }
            
        except Exception as e:
            logger.error("Failed to update scratchbook note: %s", e)
            return {"error": f"Failed to update note: {e}"}
    
    async def search_notes(
        self,
        query: str,
        note_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        limit: int = 10,
        mode: str = "hybrid"
    ) -> Dict:
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
            # Determine collection name
            if not project_name:
                project_name = self.project_info["main_project"] if self.project_info else "default"
            
            collection_name = f"{project_name}-scratchbook"
            
            # Validate collection exists
            available_collections = await self.client.list_collections()
            if collection_name not in available_collections:
                return {"error": f"Scratchbook collection '{collection_name}' not found"}
            
            # Generate embeddings for query
            embedding_service = self.client.get_embedding_service()
            embeddings = await embedding_service.generate_embeddings(
                query, 
                include_sparse=(mode in ["sparse", "hybrid"])
            )
            
            # Build filter conditions
            filter_conditions = [
                models.FieldCondition(
                    key="is_scratchbook_note",
                    match=models.MatchValue(value=True)
                )
            ]
            
            if note_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="note_type",
                        match=models.MatchAny(any=note_types)
                    )
                )
            
            if tags:
                filter_conditions.append(
                    models.FieldCondition(
                        key="tags",
                        match=models.MatchAny(any=tags)
                    )
                )
            
            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            search_results = []
            
            if mode in ["dense", "hybrid"]:
                # Dense vector search
                dense_results = self.client.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", embeddings["dense"]),
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
                
                for result in dense_results:
                    search_results.append({
                        "note_id": result.id,
                        "score": result.score,
                        "title": result.payload.get("title", ""),
                        "note_type": result.payload.get("note_type", "note"),
                        "tags": result.payload.get("tags", []),
                        "created_at": result.payload.get("created_at"),
                        "updated_at": result.payload.get("updated_at"),
                        "version": result.payload.get("version", 1),
                        "content": result.payload.get("content", "")[:200] + "..." if len(result.payload.get("content", "")) > 200 else result.payload.get("content", ""),
                        "search_type": "dense"
                    })
            
            # Sort by score and return
            search_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return {
                "query": query,
                "collection": collection_name,
                "total_results": len(search_results),
                "filters": {
                    "note_types": note_types,
                    "tags": tags
                },
                "results": search_results[:limit]
            }
            
        except Exception as e:
            logger.error("Failed to search scratchbook notes: %s", e)
            return {"error": f"Search failed: {e}"}
    
    async def list_notes(
        self,
        project_name: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> Dict:
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
            # Determine collection name
            if not project_name:
                project_name = self.project_info["main_project"] if self.project_info else "default"
            
            collection_name = f"{project_name}-scratchbook"
            
            # Build filter conditions
            filter_conditions = [
                models.FieldCondition(
                    key="is_scratchbook_note",
                    match=models.MatchValue(value=True)
                )
            ]
            
            if note_type:
                filter_conditions.append(
                    models.FieldCondition(
                        key="note_type",
                        match=models.MatchValue(value=note_type)
                    )
                )
            
            if tags:
                filter_conditions.append(
                    models.FieldCondition(
                        key="tags",
                        match=models.MatchAny(any=tags)
                    )
                )
            
            scroll_filter = models.Filter(must=filter_conditions)
            
            # Get notes
            points, _ = self.client.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            notes = []
            for point in points:
                notes.append({
                    "note_id": point.id,
                    "title": point.payload.get("title", ""),
                    "note_type": point.payload.get("note_type", "note"),
                    "tags": point.payload.get("tags", []),
                    "created_at": point.payload.get("created_at"),
                    "updated_at": point.payload.get("updated_at"),
                    "version": point.payload.get("version", 1),
                    "content_length": point.payload.get("content_length", 0)
                })
            
            # Sort by updated_at (most recent first)
            notes.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            
            return {
                "collection": collection_name,
                "total_notes": len(notes),
                "filters": {
                    "note_type": note_type,
                    "tags": tags
                },
                "notes": notes
            }
            
        except Exception as e:
            logger.error("Failed to list scratchbook notes: %s", e)
            return {"error": f"Failed to list notes: {e}"}
    
    async def delete_note(
        self,
        note_id: str,
        project_name: Optional[str] = None
    ) -> Dict:
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
            # Determine collection name
            if not project_name:
                project_name = self.project_info["main_project"] if self.project_info else "default"
            
            collection_name = f"{project_name}-scratchbook"
            
            # Delete the note
            result = self.client.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[note_id])
            )
            
            logger.info("Deleted note %s from scratchbook %s", note_id, collection_name)
            
            return {
                "note_id": note_id,
                "collection": collection_name,
                "deleted": True,
                "operation_id": result.operation_id
            }
            
        except Exception as e:
            logger.error("Failed to delete scratchbook note: %s", e)
            return {"error": f"Failed to delete note: {e}"}
    
    def _generate_title_from_content(self, content: str, max_length: int = 50) -> str:
        """Generate a title from the content."""
        # Take first line or first sentence
        lines = content.strip().split('\n')
        first_line = lines[0].strip()
        
        if not first_line:
            return "Untitled Note"
        
        # If first line is too long, truncate at word boundary
        if len(first_line) <= max_length:
            return first_line
        
        words = first_line.split()
        title = ""
        for word in words:
            if len(title + " " + word) > max_length - 3:  # Leave space for "..."
                title += "..."
                break
            title = (title + " " + word).strip()
        
        return title or "Untitled Note"


async def update_scratchbook(
    client: QdrantWorkspaceClient,
    content: str,
    note_id: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    note_type: str = "note"
) -> Dict:
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