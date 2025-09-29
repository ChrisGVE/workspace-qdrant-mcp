"""
Test data generators and factories.

Provides utilities for generating realistic test data including
documents, embeddings, metadata, and search queries.
"""

import random
import string
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class TestDataGenerator:
    """Generate test data for workspace-qdrant-mcp testing."""

    @staticmethod
    def generate_text(
        word_count: int = 100, topic: str = "general"
    ) -> str:
        """
        Generate random text content.

        Args:
            word_count: Number of words to generate
            topic: Topic for context (general, technical, narrative)

        Returns:
            Generated text string
        """
        word_pools = {
            "general": [
                "document",
                "content",
                "information",
                "data",
                "text",
                "section",
                "paragraph",
                "detail",
                "description",
                "explanation",
            ],
            "technical": [
                "API",
                "function",
                "method",
                "class",
                "module",
                "component",
                "service",
                "endpoint",
                "parameter",
                "configuration",
            ],
            "narrative": [
                "story",
                "character",
                "plot",
                "scene",
                "dialogue",
                "action",
                "setting",
                "chapter",
                "narrative",
                "development",
            ],
        }

        words = word_pools.get(topic, word_pools["general"])
        return " ".join(random.choices(words, k=word_count))

    @staticmethod
    def generate_document(
        size: str = "small",
        with_metadata: bool = True,
        topic: str = "general",
    ) -> Dict[str, Any]:
        """
        Generate a test document.

        Args:
            size: Document size (tiny, small, medium, large)
            with_metadata: Include metadata
            topic: Content topic

        Returns:
            Document dict with content and metadata
        """
        size_map = {
            "tiny": 20,
            "small": 100,
            "medium": 500,
            "large": 2000,
        }

        word_count = size_map.get(size, 100)
        content = TestDataGenerator.generate_text(word_count, topic)

        doc = {"content": content}

        if with_metadata:
            doc["metadata"] = TestDataGenerator.generate_metadata(topic)

        return doc

    @staticmethod
    def generate_metadata(topic: str = "general") -> Dict[str, Any]:
        """
        Generate document metadata.

        Args:
            topic: Content topic for contextual metadata

        Returns:
            Metadata dict
        """
        now = datetime.now()
        return {
            "title": f"Test Document - {topic.title()}",
            "author": random.choice(["Alice", "Bob", "Charlie", "Diana"]),
            "created_at": (now - timedelta(days=random.randint(0, 365))).isoformat(),
            "modified_at": now.isoformat(),
            "tags": [topic, "test", random.choice(["important", "draft", "final"])],
            "source": f"test_{topic}",
            "language": "en",
            "version": f"1.{random.randint(0, 9)}",
        }

    @staticmethod
    def generate_embedding(dimensions: int = 384) -> List[float]:
        """
        Generate a random embedding vector.

        Args:
            dimensions: Vector dimensionality

        Returns:
            List of random floats normalized to unit length
        """
        # Generate random vector
        vector = [random.gauss(0, 1) for _ in range(dimensions)]

        # Normalize to unit length
        magnitude = sum(x * x for x in vector) ** 0.5
        return [x / magnitude for x in vector]

    @staticmethod
    def generate_search_query(topic: str = "general") -> str:
        """
        Generate a search query.

        Args:
            topic: Query topic

        Returns:
            Search query string
        """
        query_templates = {
            "general": [
                "find documents about {}",
                "search for {}",
                "show me information on {}",
                "what is {}",
            ],
            "technical": [
                "how to implement {}",
                "{} API documentation",
                "{} configuration guide",
                "troubleshoot {} issues",
            ],
        }

        terms = {
            "general": ["testing", "development", "documentation", "guides"],
            "technical": ["authentication", "database", "caching", "deployment"],
        }

        template = random.choice(query_templates.get(topic, query_templates["general"]))
        term = random.choice(terms.get(topic, terms["general"]))

        return template.format(term)

    @staticmethod
    def generate_batch_documents(
        count: int = 10,
        size: str = "small",
        topic: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of documents.

        Args:
            count: Number of documents to generate
            size: Document size
            topic: Content topic

        Returns:
            List of document dicts
        """
        return [
            TestDataGenerator.generate_document(size, with_metadata=True, topic=topic)
            for _ in range(count)
        ]

    @staticmethod
    def generate_collection_name(prefix: str = "test") -> str:
        """
        Generate a unique collection name.

        Args:
            prefix: Collection name prefix

        Returns:
            Unique collection name
        """
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}_{suffix}"

    @staticmethod
    def generate_mcp_request(
        method: str = "tools/list",
        request_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an MCP protocol request.

        Args:
            method: RPC method name
            request_id: Request ID (auto-generated if None)
            params: Method parameters

        Returns:
            MCP request dict
        """
        if request_id is None:
            request_id = random.randint(1, 1000000)

        request = {"jsonrpc": "2.0", "id": request_id, "method": method}

        if params is not None:
            request["params"] = params
        else:
            request["params"] = {}

        return request

    @staticmethod
    def generate_file_tree(
        base_path: str = "/test", depth: int = 2, files_per_dir: int = 3
    ) -> List[str]:
        """
        Generate a file tree structure.

        Args:
            base_path: Base directory path
            depth: Directory depth
            files_per_dir: Files per directory

        Returns:
            List of file paths
        """

        def _generate_level(path: str, current_depth: int) -> List[str]:
            if current_depth >= depth:
                return []

            files = []

            # Generate files at this level
            extensions = [".md", ".txt", ".py", ".json"]
            for i in range(files_per_dir):
                ext = random.choice(extensions)
                files.append(f"{path}/file{i}{ext}")

            # Generate subdirectories
            for i in range(2):
                subdir = f"{path}/subdir{i}"
                files.extend(_generate_level(subdir, current_depth + 1))

            return files

        return _generate_level(base_path, 0)


class SampleData:
    """Pre-generated sample data for common test scenarios."""

    @staticmethod
    def get_sample_documents() -> List[Dict[str, Any]]:
        """Get a set of sample documents for testing."""
        return [
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {
                    "title": "ML Introduction",
                    "tags": ["ml", "ai"],
                    "source": "test",
                },
            },
            {
                "content": "Vector databases enable efficient similarity search.",
                "metadata": {
                    "title": "Vector DB",
                    "tags": ["database", "vectors"],
                    "source": "test",
                },
            },
            {
                "content": "Hybrid search combines dense and sparse retrieval methods.",
                "metadata": {
                    "title": "Hybrid Search",
                    "tags": ["search", "hybrid"],
                    "source": "test",
                },
            },
        ]

    @staticmethod
    def get_sample_queries() -> List[str]:
        """Get sample search queries."""
        return [
            "artificial intelligence and machine learning",
            "vector database similarity search",
            "hybrid search methods",
            "document retrieval techniques",
            "semantic search implementation",
        ]

    @staticmethod
    def get_edge_case_documents() -> List[Dict[str, Any]]:
        """Get documents with edge case content."""
        return [
            {"content": "", "metadata": {"title": "Empty"}},  # Empty content
            {"content": "x" * 100000, "metadata": {"title": "Large"}},  # Very large
            {
                "content": "Special chars: 你好 مرحبا 😀",
                "metadata": {"title": "Unicode"},
            },  # Unicode
            {"content": "\n\n\n", "metadata": {"title": "Whitespace"}},  # Whitespace only
            {
                "content": 'Quotes "test" and \'test\'',
                "metadata": {"title": "Quotes"},
            },  # Quotes
        ]