"""
Enhanced embedding service mocking for testing vector operations.

Provides sophisticated mocking for embedding generation including FastEmbed integration,
vector operations, and various embedding-related error scenarios.
"""

import asyncio
import random
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock

import numpy as np

from .error_injection import ErrorInjector, FailureScenarios


class EmbeddingErrorInjector(ErrorInjector):
    """Specialized error injector for embedding operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "model_not_found": {"probability": 0.0, "error": "Embedding model not found"},
            "model_load_failed": {"probability": 0.0, "error": "Failed to load embedding model"},
            "encoding_error": {"probability": 0.0, "error": "Text encoding error"},
            "vector_dimension_mismatch": {"probability": 0.0, "error": "Vector dimension mismatch"},
            "memory_exhausted": {"probability": 0.0, "error": "Insufficient memory for embeddings"},
            "gpu_error": {"probability": 0.0, "error": "GPU processing error"},
            "tokenization_failed": {"probability": 0.0, "error": "Text tokenization failed"},
            "batch_size_exceeded": {"probability": 0.0, "error": "Batch size too large"},
            "timeout": {"probability": 0.0, "timeout_seconds": 30.0},
            "invalid_input": {"probability": 0.0, "error": "Invalid input text"},
        }

    def configure_model_issues(self, probability: float = 0.1):
        """Configure model-related failures."""
        self.failure_modes["model_not_found"]["probability"] = probability
        self.failure_modes["model_load_failed"]["probability"] = probability / 2

    def configure_processing_issues(self, probability: float = 0.1):
        """Configure processing-related failures."""
        self.failure_modes["encoding_error"]["probability"] = probability
        self.failure_modes["tokenization_failed"]["probability"] = probability / 2
        self.failure_modes["invalid_input"]["probability"] = probability / 3

    def configure_resource_issues(self, probability: float = 0.1):
        """Configure resource-related failures."""
        self.failure_modes["memory_exhausted"]["probability"] = probability
        self.failure_modes["gpu_error"]["probability"] = probability / 2
        self.failure_modes["batch_size_exceeded"]["probability"] = probability / 3


class EnhancedEmbeddingServiceMock:
    """Enhanced mock embedding service with realistic behavior."""

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_dim: int = 384,
                 error_injector: EmbeddingErrorInjector | None = None):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.error_injector = error_injector or EmbeddingErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.initialized = False
        self.batch_cache: dict[str, list[float]] = {}

        # Performance characteristics
        self.performance_delays = {
            "small_batch": 0.1,    # < 10 texts
            "medium_batch": 0.5,   # 10-100 texts
            "large_batch": 2.0,    # > 100 texts
        }

        # Setup method mocks
        self._setup_embedding_methods()

    def _setup_embedding_methods(self):
        """Setup embedding service method mocks."""
        self.initialize = AsyncMock(side_effect=self._mock_initialize)
        self.close = AsyncMock(side_effect=self._mock_close)
        self.generate_embeddings = AsyncMock(side_effect=self._mock_generate_embeddings)
        self.generate_batch_embeddings = AsyncMock(side_effect=self._mock_generate_batch_embeddings)
        self.get_model_info = Mock(side_effect=self._mock_get_model_info)
        self.encode_text = AsyncMock(side_effect=self._mock_encode_text)
        self.similarity = Mock(side_effect=self._mock_similarity)

    async def _inject_embedding_error(self, operation: str, text_count: int = 1) -> None:
        """Inject embedding errors based on configuration."""
        # Add realistic processing delay based on batch size
        if text_count < 10:
            await asyncio.sleep(self.performance_delays["small_batch"])
        elif text_count < 100:
            await asyncio.sleep(self.performance_delays["medium_batch"])
        else:
            await asyncio.sleep(self.performance_delays["large_batch"])

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_embedding_error(error_type)

    async def _raise_embedding_error(self, error_type: str) -> None:
        """Raise appropriate embedding error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})

        if error_type == "model_not_found":
            raise FileNotFoundError(f"Embedding model not found: {self.model_name}")
        elif error_type == "model_load_failed":
            raise RuntimeError(f"Failed to load embedding model: {self.model_name}")
        elif error_type == "encoding_error":
            raise UnicodeError("Text encoding error")
        elif error_type == "vector_dimension_mismatch":
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}")
        elif error_type == "memory_exhausted":
            raise MemoryError("Insufficient memory for embedding generation")
        elif error_type == "gpu_error":
            raise RuntimeError("GPU processing error")
        elif error_type == "tokenization_failed":
            raise ValueError("Text tokenization failed")
        elif error_type == "batch_size_exceeded":
            raise ValueError("Batch size too large for processing")
        elif error_type == "timeout":
            timeout = error_config.get("timeout_seconds", 30.0)
            await asyncio.sleep(timeout)
            raise TimeoutError("Embedding generation timeout")
        elif error_type == "invalid_input":
            raise ValueError("Invalid input text for embedding")

    async def _mock_initialize(self, model_name: str | None = None) -> None:
        """Mock embedding service initialization."""
        await self._inject_embedding_error("initialize")

        if model_name:
            self.model_name = model_name

        self.initialized = True

        self.operation_history.append({
            "operation": "initialize",
            "model_name": self.model_name,
            "vector_dim": self.vector_dim
        })

    async def _mock_close(self) -> None:
        """Mock embedding service cleanup."""
        self.initialized = False
        self.batch_cache.clear()

        self.operation_history.append({
            "operation": "close"
        })

    async def _mock_generate_embeddings(self, text: str) -> dict[str, Any]:
        """Mock single text embedding generation."""
        await self._inject_embedding_error("generate_embeddings", 1)

        if not self.initialized:
            raise RuntimeError("Embedding service not initialized")

        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided for embedding")

        self.operation_history.append({
            "operation": "generate_embeddings",
            "text_length": len(text),
            "model": self.model_name
        })

        # Generate realistic dense embedding
        dense_vector = self._generate_realistic_vector(text)

        # Generate sparse embedding (keyword-based)
        sparse_vector = self._generate_sparse_vector(text)

        return {
            "dense": dense_vector,
            "sparse": sparse_vector,
            "model": self.model_name,
            "vector_dim": self.vector_dim,
            "processing_time_ms": random.randint(50, 200)
        }

    async def _mock_generate_batch_embeddings(self, texts: list[str]) -> list[dict[str, Any]]:
        """Mock batch embedding generation."""
        await self._inject_embedding_error("generate_batch_embeddings", len(texts))

        if not self.initialized:
            raise RuntimeError("Embedding service not initialized")

        if not texts:
            raise ValueError("Empty text list provided")

        self.operation_history.append({
            "operation": "generate_batch_embeddings",
            "batch_size": len(texts),
            "total_length": sum(len(text) for text in texts),
            "model": self.model_name
        })

        embeddings = []
        for text in texts:
            if text in self.batch_cache:
                # Simulate cache hit
                dense_vector = self.batch_cache[text]
            else:
                dense_vector = self._generate_realistic_vector(text)
                self.batch_cache[text] = dense_vector

            embeddings.append({
                "dense": dense_vector,
                "sparse": self._generate_sparse_vector(text),
                "text": text[:50] + "..." if len(text) > 50 else text,
                "cached": text in self.batch_cache
            })

        return embeddings

    def _mock_get_model_info(self) -> dict[str, Any]:
        """Mock getting model information."""
        self.operation_history.append({
            "operation": "get_model_info"
        })

        return {
            "model_name": self.model_name,
            "vector_dimension": self.vector_dim,
            "max_sequence_length": 512,
            "model_size_mb": random.randint(100, 500),
            "supported_languages": ["en", "es", "fr", "de", "it"],
            "architecture": "sentence-transformer",
            "initialized": self.initialized
        }

    async def _mock_encode_text(self, text: str, normalize: bool = True) -> list[float]:
        """Mock direct text encoding to vector."""
        await self._inject_embedding_error("encode_text", 1)

        if not self.initialized:
            raise RuntimeError("Embedding service not initialized")

        self.operation_history.append({
            "operation": "encode_text",
            "text_length": len(text),
            "normalize": normalize
        })

        vector = self._generate_realistic_vector(text)

        if normalize:
            # Normalize vector to unit length
            norm = sum(x * x for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]

        return vector

    def _mock_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        """Mock vector similarity calculation."""
        if len(vector1) != len(vector2):
            raise ValueError("Vector dimension mismatch")

        self.operation_history.append({
            "operation": "similarity",
            "vector_dim": len(vector1)
        })

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2, strict=False))
        norm1 = sum(a * a for a in vector1) ** 0.5
        norm2 = sum(b * b for b in vector2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))  # Clamp to [-1, 1]

    def _generate_realistic_vector(self, text: str) -> list[float]:
        """Generate realistic embedding vector based on text content."""
        # Use text hash as seed for reproducible vectors
        seed = hash(text) % (2**32)
        random.seed(seed)

        # Generate vector with some patterns based on text characteristics
        vector = []

        # Base random vector
        for _i in range(self.vector_dim):
            vector.append(random.gauss(0, 0.3))

        # Add patterns based on text features
        text_lower = text.lower()

        # Length influence
        length_factor = min(1.0, len(text) / 100.0)
        for i in range(min(50, self.vector_dim)):
            vector[i] += length_factor * 0.1

        # Word count influence
        word_count = len(text.split())
        word_factor = min(1.0, word_count / 20.0)
        for i in range(50, min(100, self.vector_dim)):
            vector[i] += word_factor * 0.1

        # Common word patterns
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for"]
        for word in common_words:
            if word in text_lower:
                idx = hash(word) % self.vector_dim
                vector[idx] += 0.2

        # Technical terms
        tech_words = ["function", "class", "method", "variable", "api", "data", "system"]
        for word in tech_words:
            if word in text_lower:
                idx = hash(word) % self.vector_dim
                vector[idx] += 0.3

        # Reset random seed
        random.seed()

        return vector

    def _generate_sparse_vector(self, text: str) -> dict[str, list[int] | list[float]]:
        """Generate sparse vector representation (BM25-style)."""
        words = text.lower().split()
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Select top words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:min(20, len(sorted_words))]

        indices = []
        values = []

        for word, count in top_words:
            # Use word hash as index
            idx = hash(word) % 10000  # Sparse vector space
            indices.append(idx)

            # TF-IDF-like score
            tf = count / len(words)
            idf = 2.0  # Mock IDF
            score = tf * idf
            values.append(score)

        return {
            "indices": indices,
            "values": values
        }

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Get history of embedding operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset embedding service state."""
        self.operation_history.clear()
        self.initialized = False
        self.batch_cache.clear()
        self.error_injector.reset()


class EmbeddingGeneratorMock:
    """Mock for direct embedding generation without service wrapper."""

    def __init__(self, vector_dim: int = 384, error_injector: EmbeddingErrorInjector | None = None):
        self.vector_dim = vector_dim
        self.error_injector = error_injector or EmbeddingErrorInjector()
        self.operation_history: list[dict[str, Any]] = []

        # Setup method mocks
        self.generate = Mock(side_effect=self._mock_generate)
        self.generate_batch = Mock(side_effect=self._mock_generate_batch)

    def _mock_generate(self, text: str) -> list[float]:
        """Mock single vector generation."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type == "invalid_input":
                raise ValueError("Invalid input text")

        self.operation_history.append({
            "operation": "generate",
            "text_length": len(text)
        })

        # Generate deterministic vector based on text
        seed = hash(text) % (2**32)
        random.seed(seed)
        vector = [random.gauss(0, 0.3) for _ in range(self.vector_dim)]
        random.seed()  # Reset seed

        return vector

    def _mock_generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Mock batch vector generation."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type == "batch_size_exceeded":
                raise ValueError("Batch size too large")

        self.operation_history.append({
            "operation": "generate_batch",
            "batch_size": len(texts)
        })

        return [self._mock_generate(text) for text in texts]

    def reset_state(self) -> None:
        """Reset generator state."""
        self.operation_history.clear()
        self.error_injector.reset()


class FastEmbedMock:
    """Mock for FastEmbed library integration."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", error_injector: EmbeddingErrorInjector | None = None):
        self.model_name = model_name
        self.error_injector = error_injector or EmbeddingErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.model_loaded = False

        # Setup method mocks
        self.load_model = Mock(side_effect=self._mock_load_model)
        self.embed = Mock(side_effect=self._mock_embed)
        self.embed_batch = Mock(side_effect=self._mock_embed_batch)
        self.get_available_models = Mock(side_effect=self._mock_get_available_models)

    def _mock_load_model(self) -> None:
        """Mock FastEmbed model loading."""
        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type == "model_load_failed":
                raise RuntimeError(f"Failed to load FastEmbed model: {self.model_name}")

        self.model_loaded = True
        self.operation_history.append({
            "operation": "load_model",
            "model_name": self.model_name
        })

    def _mock_embed(self, text: str) -> np.ndarray:
        """Mock single text embedding with FastEmbed."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type == "encoding_error":
                raise UnicodeError("Text encoding error")

        self.operation_history.append({
            "operation": "embed",
            "text_length": len(text)
        })

        # Generate numpy array
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        vector = np.random.normal(0, 0.3, 384)  # Common FastEmbed dimension
        np.random.seed()  # Reset seed

        return vector

    def _mock_embed_batch(self, texts: list[str]) -> np.ndarray:
        """Mock batch embedding with FastEmbed."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            if error_type == "batch_size_exceeded":
                raise ValueError("Batch too large for FastEmbed")

        self.operation_history.append({
            "operation": "embed_batch",
            "batch_size": len(texts)
        })

        # Generate batch of vectors
        vectors = []
        for text in texts:
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            vector = np.random.normal(0, 0.3, 384)
            vectors.append(vector)

        np.random.seed()  # Reset seed
        return np.array(vectors)

    def _mock_get_available_models(self) -> list[dict[str, Any]]:
        """Mock getting available FastEmbed models."""
        return [
            {
                "model": "BAAI/bge-small-en-v1.5",
                "dim": 384,
                "description": "BGE small model",
                "size_in_GB": 0.13
            },
            {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dim": 384,
                "description": "All MiniLM L6 v2",
                "size_in_GB": 0.09
            },
            {
                "model": "BAAI/bge-base-en-v1.5",
                "dim": 768,
                "description": "BGE base model",
                "size_in_GB": 0.44
            }
        ]

    def reset_state(self) -> None:
        """Reset FastEmbed mock state."""
        self.operation_history.clear()
        self.model_loaded = False
        self.error_injector.reset()


def create_embedding_mock(
    mock_type: str = "service",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    vector_dim: int = 384,
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> EnhancedEmbeddingServiceMock | EmbeddingGeneratorMock | FastEmbedMock:
    """
    Create an embedding mock with optional error injection.

    Args:
        mock_type: Type of mock ("service", "generator", "fastembed")
        model_name: Name of the embedding model
        vector_dim: Vector dimension
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured embedding mock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = EmbeddingErrorInjector()
        error_injector.configure_model_issues(error_probability)
        error_injector.configure_processing_issues(error_probability)
        error_injector.configure_resource_issues(error_probability)

    if mock_type == "service":
        return EnhancedEmbeddingServiceMock(model_name, vector_dim, error_injector)
    elif mock_type == "generator":
        return EmbeddingGeneratorMock(vector_dim, error_injector)
    elif mock_type == "fastembed":
        return FastEmbedMock(model_name, error_injector)
    else:
        raise ValueError(f"Unknown mock type: {mock_type}")


# Convenience functions for common scenarios
def create_basic_embedding_service() -> EnhancedEmbeddingServiceMock:
    """Create basic embedding service mock without error injection."""
    return create_embedding_mock("service")


def create_failing_embedding_service(error_rate: float = 0.3) -> EnhancedEmbeddingServiceMock:
    """Create embedding service mock with high failure rate."""
    return create_embedding_mock("service", with_error_injection=True, error_probability=error_rate)


def create_realistic_embedding_service() -> EnhancedEmbeddingServiceMock:
    """Create embedding service mock with realistic error rates."""
    return create_embedding_mock("service", with_error_injection=True, error_probability=0.05)
