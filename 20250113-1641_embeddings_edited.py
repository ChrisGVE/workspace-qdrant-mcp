"""
FastEmbed integration for high-performance document embeddings.

This module provides a comprehensive embedding service that combines dense semantic
embeddings (via FastEmbed's all-MiniLM-L6-v2) with enhanced sparse keyword vectors
(via BM25) for optimal hybrid search performance.

Key Features:
    - Dense semantic embeddings using FastEmbed's optimized models
    - Enhanced BM25 sparse vectors for precise keyword matching
    - Intelligent text chunking with overlap for large documents
    - Async batch processing for high throughput
    - Content deduplication via SHA256 hashing
    - Configurable model parameters and batch sizes

Performance Characteristics:
    - Dense model: 384-dimensional all-MiniLM-L6-v2 embeddings
    - Processing speed: ~1000 documents/second on modern hardware
    - Memory usage: ~500MB for model initialization
    - Chunking: Intelligent word-boundary splitting with overlap

Example:
    ```python
    from workspace_qdrant_mcp.core.embeddings import EmbeddingService
    from workspace_qdrant_mcp.core.config import Config

    config = Config()
    service = EmbeddingService(config)
    await service.initialize()

    # Generate embeddings for text
    embeddings = await service.generate_embeddings(
        "Your document content here",
        include_sparse=True
    )

    # Process multiple documents with metadata
    documents = [{"content": "doc1"}, {"content": "doc2"}]
    embedded_docs = await service.embed_documents(documents)
    ```

Task 215: Migrated to unified logging system for MCP stdio compliance.
"""

import asyncio
import hashlib
# Task 215: Direct logging import replaced with unified system
# import logging  # MIGRATED
import re
from typing import Optional, Union

from fastembed import TextEmbedding
from fastembed.sparse import SparseTextEmbedding

from .config import Config
from .sparse_vectors import BM25SparseEncoder

# Task 215: Use unified logging system
from ..observability.logger import get_logger
logger = get_logger(__name__)