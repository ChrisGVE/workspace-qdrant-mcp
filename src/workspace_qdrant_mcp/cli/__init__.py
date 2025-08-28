"""
Command-line interface for workspace-qdrant-mcp batch operations.

This module provides CLI tools for batch document ingestion and management
operations that complement the MCP server functionality.

Available commands:
    - workspace-qdrant-ingest: Batch document ingestion from directories
    - Format detection and processing for various document types
    - Progress tracking and error handling for large ingestion operations

The CLI tools integrate with the existing workspace collection system and
provide the missing batch processing capabilities identified in the feature gap analysis.
"""

from .ingest import main as ingest_main
from .ingestion_engine import DocumentIngestionEngine, IngestionResult, IngestionStats
from . import parsers

__all__ = [
    "ingest_main",
    "DocumentIngestionEngine", 
    "IngestionResult",
    "IngestionStats",
    "parsers",
]