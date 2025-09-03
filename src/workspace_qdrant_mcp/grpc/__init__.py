"""
gRPC client module for Python-Rust communication.

This module provides async gRPC client wrappers for communicating with
the Rust-based ingestion engine daemon.
"""

from .client import AsyncIngestClient
from .connection_manager import GrpcConnectionManager
from .types import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    HealthCheckRequest,
    HealthCheckResponse,
)

__all__ = [
    "AsyncIngestClient",
    "GrpcConnectionManager", 
    "ProcessDocumentRequest",
    "ProcessDocumentResponse",
    "ExecuteQueryRequest",
    "ExecuteQueryResponse",
    "HealthCheckRequest",
    "HealthCheckResponse",
]