"""
Enhanced sparse vector support with BM25 implementation.

Provides robust BM25 sparse vectors for hybrid search capabilities.
"""

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from fastembed.sparse import SparseTextEmbedding
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class BM25SparseEncoder:
    """
    Enhanced BM25 sparse vector encoder with improved performance.
    
    Provides both FastEmbed-based encoding and custom BM25 implementation.
    """
    
    def __init__(self, 
                 use_fastembed: bool = True,
                 k1: float = 1.2, 
                 b: float = 0.75,
                 min_df: int = 1,
                 max_df: float = 0.95):
        """
        Initialize BM25 encoder.
        
        Args:
            use_fastembed: Whether to use FastEmbed for encoding
            k1: BM25 parameter controlling term frequency normalization
            b: BM25 parameter controlling length normalization
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency ratio for terms
        """
        self.use_fastembed = use_fastembed
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        
        self.fastembed_model: Optional[SparseTextEmbedding] = None
        self.vocab: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.corpus_size: int = 0
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the sparse vector encoder."""
        if self.initialized:
            return
            
        if self.use_fastembed:
            try:
                import asyncio
                self.fastembed_model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: SparseTextEmbedding(model_name="Qdrant/bm25")
                )
                logger.info("FastEmbed BM25 model initialized")
            except Exception as e:
                logger.warning("Failed to initialize FastEmbed BM25, falling back to custom: %s", e)
                self.use_fastembed = False
        
        self.initialized = True
    
    async def encode_single(self, text: str) -> Dict:
        """
        Encode a single text into sparse vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Dictionary with indices and values for sparse vector
        """
        if not self.initialized:
            await self.initialize()
        
        if self.use_fastembed and self.fastembed_model:
            return await self._encode_with_fastembed([text])
        else:
            return self._encode_with_custom_bm25([text])[0]
    
    async def encode_batch(self, texts: List[str]) -> List[Dict]:
        """
        Encode multiple texts into sparse vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of sparse vector dictionaries
        """
        if not self.initialized:
            await self.initialize()
        
        if self.use_fastembed and self.fastembed_model:
            return await self._encode_with_fastembed(texts)
        else:
            return self._encode_with_custom_bm25(texts)
    
    async def _encode_with_fastembed(self, texts: List[str]) -> List[Dict]:
        """Encode using FastEmbed BM25 model."""
        try:
            import asyncio
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(self.fastembed_model.embed(texts))
            )
            
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
            logger.error("FastEmbed encoding failed: %s", e)
            # Fall back to custom BM25
            return self._encode_with_custom_bm25(texts)
    
    def _encode_with_custom_bm25(self, texts: List[str]) -> List[Dict]:
        """Encode using custom BM25 implementation."""
        if not self.vocab:
            # Build vocabulary from corpus
            self._build_vocabulary(texts)
        
        sparse_vectors = []
        for text in texts:
            sparse_vector = self._compute_bm25_scores(text)
            sparse_vectors.append(sparse_vector)
            
        return sparse_vectors
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary and compute IDF scores."""
        # Tokenize all documents
        tokenized_docs = [self._tokenize(text) for text in texts]
        
        # Count document frequencies
        doc_freq = defaultdict(int)
        self.doc_lengths = []
        
        for tokens in tokenized_docs:
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        self.corpus_size = len(texts)
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Filter terms by document frequency
        filtered_terms = {}
        max_df_threshold = self.max_df * self.corpus_size
        
        for term, freq in doc_freq.items():
            if freq >= self.min_df and freq <= max_df_threshold:
                filtered_terms[term] = freq
        
        # Build vocabulary and compute IDF
        self.vocab = {term: idx for idx, term in enumerate(sorted(filtered_terms.keys()))}
        
        for term, doc_freq_val in filtered_terms.items():
            # BM25 IDF formula
            idf = math.log((self.corpus_size - doc_freq_val + 0.5) / (doc_freq_val + 0.5))
            self.idf_scores[term] = max(idf, 0.01)  # Ensure positive IDF
    
    def _compute_bm25_scores(self, text: str, doc_length: Optional[int] = None) -> Dict:
        """Compute BM25 scores for a document."""
        tokens = self._tokenize(text)
        if doc_length is None:
            doc_length = len(tokens)
        
        # Count term frequencies
        term_freq = Counter(tokens)
        
        # Compute BM25 scores
        indices = []
        values = []
        
        for term, tf in term_freq.items():
            if term in self.vocab:
                # BM25 score calculation
                idf = self.idf_scores.get(term, 0.01)
                
                # Term frequency component
                tf_component = tf * (self.k1 + 1)
                tf_denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                
                bm25_score = idf * (tf_component / tf_denominator)
                
                if bm25_score > 0:
                    indices.append(self.vocab[term])
                    values.append(float(bm25_score))
        
        # Sort by indices (required for Qdrant)
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            indices = list(indices)
            values = list(values)
        
        return {
            "indices": indices,
            "values": values
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        
        # Convert to lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_model_info(self) -> Dict:
        """Get information about the sparse vector model."""
        return {
            "encoder_type": "fastembed" if (self.use_fastembed and self.fastembed_model) else "custom_bm25",
            "vocab_size": len(self.vocab),
            "corpus_size": self.corpus_size,
            "avg_doc_length": self.avg_doc_length,
            "parameters": {
                "k1": self.k1,
                "b": self.b,
                "min_df": self.min_df,
                "max_df": self.max_df
            },
            "initialized": self.initialized
        }


def create_qdrant_sparse_vector(indices: List[int], values: List[float]) -> models.SparseVector:
    """
    Create a Qdrant SparseVector from indices and values.
    
    Args:
        indices: List of term indices
        values: List of corresponding values
        
    Returns:
        Qdrant SparseVector model
    """
    return models.SparseVector(
        indices=indices,
        values=values
    )


def create_named_sparse_vector(indices: List[int], values: List[float], name: str = "sparse") -> models.NamedSparseVector:
    """
    Create a Qdrant NamedSparseVector for search queries.
    
    Args:
        indices: List of term indices
        values: List of corresponding values
        name: Name of the sparse vector (default: "sparse")
        
    Returns:
        Qdrant NamedSparseVector model
    """
    return models.NamedSparseVector(
        name=name,
        vector=models.SparseVector(indices=indices, values=values)
    )