"""
Type definitions and data classes for gRPC communication.

This module provides Python-friendly wrappers and type definitions
for the protobuf-generated classes.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .ingestion_pb2 import (
    ProcessDocumentRequest as PbProcessDocumentRequest,
    ProcessDocumentResponse as PbProcessDocumentResponse,
    ExecuteQueryRequest as PbExecuteQueryRequest,
    ExecuteQueryResponse as PbExecuteQueryResponse,
    HealthResponse as PbHealthResponse,
    SearchMode,
    WatchEventType,
    WatchStatus,
    HealthStatus,
)


@dataclass
class ProcessDocumentRequest:
    """Request to process a document for ingestion."""
    file_path: str
    collection: str
    metadata: Optional[Dict[str, str]] = None
    document_id: Optional[str] = None
    chunk_text: bool = True
    
    def to_pb(self) -> PbProcessDocumentRequest:
        """Convert to protobuf message."""
        req = PbProcessDocumentRequest()
        req.file_path = self.file_path
        req.collection = self.collection
        req.chunk_text = self.chunk_text
        
        if self.document_id:
            req.document_id = self.document_id
        
        if self.metadata:
            for key, value in self.metadata.items():
                req.metadata[key] = value
        
        return req


@dataclass
class ProcessDocumentResponse:
    """Response from document processing."""
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_added: int = 0
    applied_metadata: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_pb(cls, pb_response: PbProcessDocumentResponse) -> 'ProcessDocumentResponse':
        """Create from protobuf message."""
        return cls(
            success=pb_response.success,
            message=pb_response.message,
            document_id=pb_response.document_id if pb_response.HasField('document_id') else None,
            chunks_added=pb_response.chunks_added,
            applied_metadata=dict(pb_response.applied_metadata) if pb_response.applied_metadata else None
        )


@dataclass  
class ExecuteQueryRequest:
    """Request to execute a search query."""
    query: str
    collections: Optional[List[str]] = None
    mode: str = "hybrid"  # "hybrid", "dense", "sparse"
    limit: int = 10
    score_threshold: float = 0.7
    
    def to_pb(self) -> PbExecuteQueryRequest:
        """Convert to protobuf message."""
        req = PbExecuteQueryRequest()
        req.query = self.query
        req.limit = self.limit
        req.score_threshold = self.score_threshold
        
        # Convert mode string to protobuf enum
        mode_mapping = {
            "hybrid": SearchMode.SEARCH_MODE_HYBRID,
            "dense": SearchMode.SEARCH_MODE_DENSE, 
            "sparse": SearchMode.SEARCH_MODE_SPARSE,
        }
        req.mode = mode_mapping.get(self.mode.lower(), SearchMode.SEARCH_MODE_HYBRID)
        
        if self.collections:
            req.collections.extend(self.collections)
        
        return req


@dataclass
class SearchResult:
    """Individual search result."""
    id: str
    score: float
    payload: Dict[str, Any]
    collection: str
    search_type: str
    
    @classmethod
    def from_pb(cls, pb_result) -> 'SearchResult':
        """Create from protobuf search result."""
        # Convert protobuf Any payload to dict
        payload = {}
        if pb_result.payload:
            # This is a simplified conversion - in practice you'd need to handle
            # the Any type properly based on the actual data types
            for key, any_value in pb_result.payload.items():
                # For now, just convert to string - you'd need proper Any unpacking here
                payload[key] = str(any_value.value) if hasattr(any_value, 'value') else str(any_value)
        
        return cls(
            id=pb_result.id,
            score=pb_result.score,
            payload=payload,
            collection=pb_result.collection,
            search_type=pb_result.search_type
        )


@dataclass
class ExecuteQueryResponse:
    """Response from query execution."""
    query: str
    mode: str
    collections_searched: List[str]
    total_results: int
    results: List[SearchResult]
    
    @classmethod  
    def from_pb(cls, pb_response: PbExecuteQueryResponse) -> 'ExecuteQueryResponse':
        """Create from protobuf message."""
        # Convert mode enum back to string
        mode_mapping = {
            SearchMode.SEARCH_MODE_HYBRID: "hybrid",
            SearchMode.SEARCH_MODE_DENSE: "dense",
            SearchMode.SEARCH_MODE_SPARSE: "sparse",
        }
        mode_str = mode_mapping.get(pb_response.mode, "hybrid")
        
        results = [SearchResult.from_pb(r) for r in pb_response.results]
        
        return cls(
            query=pb_response.query,
            mode=mode_str,
            collections_searched=list(pb_response.collections_searched),
            total_results=pb_response.total_results,
            results=results
        )


@dataclass
class HealthCheckRequest:
    """Health check request (uses Empty protobuf)."""
    pass
    
    def to_pb(self):
        """Convert to protobuf Empty message."""
        from google.protobuf.empty_pb2 import Empty
        return Empty()


@dataclass  
class HealthCheckResponse:
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    services: List[Dict[str, Any]]
    
    @classmethod
    def from_pb(cls, pb_response: PbHealthResponse) -> 'HealthCheckResponse':
        """Create from protobuf HealthResponse message."""
        # Convert status enum to string
        status_mapping = {
            HealthStatus.HEALTH_STATUS_HEALTHY: "healthy",
            HealthStatus.HEALTH_STATUS_DEGRADED: "degraded", 
            HealthStatus.HEALTH_STATUS_UNHEALTHY: "unhealthy",
        }
        status_str = status_mapping.get(pb_response.status, "unknown")
        
        services = []
        for service in pb_response.services:
            services.append({
                "name": service.name,
                "status": status_mapping.get(service.status, "unknown"),
                "message": service.message
            })
        
        return cls(
            status=status_str,
            message=pb_response.message,
            services=services
        )


# Utility functions for type conversions
def dict_to_metadata_map(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Convert dict to protobuf string map."""
    return metadata or {}


def timestamp_to_datetime(pb_timestamp) -> datetime:
    """Convert protobuf timestamp to Python datetime."""
    return datetime.fromtimestamp(pb_timestamp.seconds + pb_timestamp.nanos / 1e9)


def datetime_to_timestamp(dt: datetime):
    """Convert Python datetime to protobuf timestamp."""
    from google.protobuf.timestamp_pb2 import Timestamp
    timestamp = Timestamp()
    timestamp.FromDatetime(dt)
    return timestamp