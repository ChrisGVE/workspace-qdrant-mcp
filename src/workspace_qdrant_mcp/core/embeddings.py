"""
FastEmbed integration for document embeddings.

Provides embedding generation with all-MiniLM-L6-v2 and sparse vector support.
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Union

from fastembed import TextEmbedding
from fastembed.sparse import SparseTextEmbedding

from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating dense and sparse embeddings using FastEmbed.
    
    Handles model initialization, batch processing, and embedding generation.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.dense_model: Optional[TextEmbedding] = None
        self.sparse_model: Optional[SparseTextEmbedding] = None
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the embedding models."""
        if self.initialized:
            return
            
        try:
            # Initialize dense embedding model
            logger.info("Initializing dense embedding model: %s", self.config.embedding.model)
            self.dense_model = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: TextEmbedding(
                    model_name=self.config.embedding.model,
                    max_length=512  # Reasonable limit for document chunks
                )
            )
            
            # Initialize sparse embedding model if enabled
            if self.config.embedding.enable_sparse_vectors:
                logger.info("Initializing sparse embedding model")
                self.sparse_model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: SparseTextEmbedding(model_name="Qdrant/bm25")
                )
                
            self.initialized = True
            logger.info("Embedding models initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize embedding models: %s", e)
            raise
    
    async def generate_embeddings(
        self, 
        texts: Union[str, List[str]],
        include_sparse: bool = None
    ) -> Dict[str, Union[List[float], List[List[float]], Dict]]:
        """
        Generate dense and optionally sparse embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            include_sparse: Whether to include sparse vectors (defaults to config setting)
            
        Returns:
            Dictionary with 'dense' and optionally 'sparse' embeddings
        """
        if not self.initialized:
            await self.initialize()
            
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
            
        if include_sparse is None:
            include_sparse = self.config.embedding.enable_sparse_vectors
            
        result = {}
        
        try:
            # Generate dense embeddings
            dense_embeddings = await self._generate_dense_embeddings(texts)
            result["dense"] = dense_embeddings[0] if single_text else dense_embeddings
            
            # Generate sparse embeddings if requested
            if include_sparse and self.sparse_model:
                sparse_embeddings = await self._generate_sparse_embeddings(texts)
                result["sparse"] = sparse_embeddings[0] if single_text else sparse_embeddings
                
            return result
            
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", e)
            raise
    
    async def _generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings using FastEmbed."""
        try:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(self.dense_model.embed(texts))
            )
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error("Failed to generate dense embeddings: %s", e)
            raise
    
    async def _generate_sparse_embeddings(self, texts: List[str]) -> List[Dict]:
        """Generate sparse embeddings using BM25."""
        try:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(self.sparse_model.embed(texts))
            )
            
            # Convert to Qdrant-compatible sparse vector format
            sparse_vectors = []
            for embedding in embeddings:
                indices = embedding.indices.tolist()
                values = embedding.values.tolist()
                sparse_vectors.append({
                    "indices": indices,
                    "values": values
                })
                
            return sparse_vectors
            
        except Exception as e:
            logger.error("Failed to generate sparse embeddings: %s", e)
            raise
    
    async def embed_documents(
        self, 
        documents: List[Dict[str, str]],
        content_field: str = "content",
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Embed a list of documents with metadata.
        
        Args:
            documents: List of document dictionaries
            content_field: Field containing the text content
            batch_size: Batch size for processing (defaults to config)
            
        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []
            
        batch_size = batch_size or self.config.embedding.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.get(content_field, "") for doc in batch]
            
            # Generate embeddings for batch
            embeddings = await self.generate_embeddings(batch_texts)
            
            # Add embeddings to documents
            for j, doc in enumerate(batch):
                embedded_doc = doc.copy()
                embedded_doc["dense_vector"] = embeddings["dense"][j]
                
                if "sparse" in embeddings:
                    embedded_doc["sparse_vector"] = embeddings["sparse"][j]
                    
                # Add embedding metadata
                embedded_doc["embedding_model"] = self.config.embedding.model
                embedded_doc["embedding_timestamp"] = asyncio.get_event_loop().time()
                embedded_doc["content_hash"] = self._hash_content(doc.get(content_field, ""))
                
                results.append(embedded_doc)
        
        return results
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.embedding.chunk_size
        chunk_overlap = chunk_overlap or self.config.embedding.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Find the last space before the limit
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # No space found, force break
                    end = start + chunk_size
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move start position with overlap
            start = end - chunk_overlap
            if start <= 0:
                start = end
                
        return chunks
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "dense_model": {
                "name": self.config.embedding.model,
                "loaded": self.dense_model is not None,
                "dimensions": 384 if "all-MiniLM-L6-v2" in self.config.embedding.model else "unknown"
            },
            "sparse_model": {
                "name": "Qdrant/bm25" if self.config.embedding.enable_sparse_vectors else None,
                "loaded": self.sparse_model is not None,
                "enabled": self.config.embedding.enable_sparse_vectors
            },
            "config": {
                "chunk_size": self.config.embedding.chunk_size,
                "chunk_overlap": self.config.embedding.chunk_overlap,
                "batch_size": self.config.embedding.batch_size
            },
            "initialized": self.initialized
        }
    
    async def close(self) -> None:
        """Clean up embedding models."""
        # FastEmbed models don't need explicit cleanup
        self.dense_model = None
        self.sparse_model = None
        self.initialized = False