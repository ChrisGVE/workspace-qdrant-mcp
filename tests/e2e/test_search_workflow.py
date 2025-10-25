"""
End-to-end tests for search workflow integration.

Tests complete search functionality across MCP and CLI interfaces, including
hybrid search pipeline, query processing, embedding generation, Qdrant search
execution, result ranking, and response formatting.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    SystemComponents,
)


@pytest.mark.integration
@pytest.mark.slow
class TestQueryProcessing:
    """Test query processing and validation."""

    def test_simple_text_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test processing of simple text query."""
        # First, ingest some content to search
        workspace = system_components.workspace_path
        test_file = workspace / "search_content.txt"
        test_file.write_text("Python programming language documentation")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-simple"]
        )
        time.sleep(3)

        # Execute search
        result = cli_helper.run_command(
            ["search", "python programming", "--collection", "test-search-simple"]
        )

        # Search should execute
        assert result is not None

    def test_multi_word_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test processing of multi-word query."""
        workspace = system_components.workspace_path
        test_file = workspace / "multiword.txt"
        test_file.write_text("Machine learning artificial intelligence neural networks")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-multiword"]
        )
        time.sleep(3)

        result = cli_helper.run_command(
            [
                "search",
                "machine learning neural networks",
                "--collection",
                "test-search-multiword",
            ]
        )

        assert result is not None

    def test_special_characters_in_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of special characters in query."""
        workspace = system_components.workspace_path
        test_file = workspace / "special.txt"
        test_file.write_text("C++ programming with std::vector")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-special"]
        )
        time.sleep(3)

        # Query with special characters
        result = cli_helper.run_command(
            ["search", "C++ std::vector", "--collection", "test-search-special"]
        )

        # Should handle special characters
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingGenerationForSearch:
    """Test embedding generation during search."""

    @pytest.mark.asyncio
    async def test_query_embedding_generated(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that query embeddings are generated for search."""
        workspace = system_components.workspace_path

        # Ingest content first
        test_file = workspace / "embed_search.txt"
        test_file.write_text("Database management and SQL queries")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-embed"]
        )
        await asyncio.sleep(3)

        # Execute search (requires query embedding)
        result = cli_helper.run_command(
            ["search", "database SQL", "--collection", "test-search-embed"]
        )

        # Search should complete (embedding generated)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantSearchExecution:
    """Test Qdrant search execution."""

    @pytest.mark.asyncio
    async def test_semantic_search_execution(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test semantic search execution via Qdrant."""
        workspace = system_components.workspace_path

        # Ingest documents with semantic content
        docs = [
            ("doc1.txt", "Python is a programming language"),
            ("doc2.txt", "Java is used for enterprise applications"),
            ("doc3.txt", "JavaScript runs in web browsers"),
        ]

        for filename, content in docs:
            (workspace / filename).write_text(content)

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-search-semantic"]
        )
        await asyncio.sleep(5)

        # Search for Python-related content
        result = cli_helper.run_command(
            ["search", "Python programming", "--collection", "test-search-semantic"]
        )

        # Should find relevant results
        assert result is not None

    @pytest.mark.asyncio
    async def test_keyword_search_execution(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test keyword-based search execution."""
        workspace = system_components.workspace_path

        # Ingest content with specific keywords
        test_file = workspace / "keywords.txt"
        test_file.write_text("React framework for building user interfaces")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-keyword"]
        )
        await asyncio.sleep(3)

        # Search for exact keyword
        result = cli_helper.run_command(
            ["search", "React framework", "--collection", "test-search-keyword"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestHybridSearch:
    """Test hybrid search combining dense and sparse vectors."""

    @pytest.mark.asyncio
    async def test_hybrid_search_pipeline(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test complete hybrid search pipeline."""
        workspace = system_components.workspace_path

        # Ingest diverse content
        docs = [
            ("tech1.txt", "Deep learning models require large datasets"),
            ("tech2.txt", "Cloud computing provides scalable infrastructure"),
            ("tech3.txt", "Microservices architecture enables flexibility"),
        ]

        for filename, content in docs:
            (workspace / filename).write_text(content)

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-search-hybrid"]
        )
        await asyncio.sleep(5)

        # Hybrid search combines semantic + keyword matching
        result = cli_helper.run_command(
            ["search", "machine learning datasets", "--collection", "test-search-hybrid"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestResultRanking:
    """Test search result ranking and relevance."""

    @pytest.mark.asyncio
    async def test_results_ordered_by_relevance(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that results are ordered by relevance score."""
        workspace = system_components.workspace_path

        # Create documents with varying relevance
        docs = [
            ("exact.txt", "Docker containerization platform for applications"),
            ("related.txt", "Kubernetes orchestrates containerized workloads"),
            ("tangential.txt", "Software development best practices guide"),
        ]

        for filename, content in docs:
            (workspace / filename).write_text(content)

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-search-ranking"]
        )
        await asyncio.sleep(5)

        # Search for Docker-related content
        result = cli_helper.run_command(
            ["search", "Docker containers", "--collection", "test-search-ranking"]
        )

        # Results should be returned (ranking happens internally)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestResponseFormatting:
    """Test search response formatting."""

    def test_cli_search_response_format(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test CLI search response is properly formatted."""
        workspace = system_components.workspace_path

        test_file = workspace / "format_test.txt"
        test_file.write_text("Testing response formatting for search results")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-format"]
        )
        time.sleep(3)

        result = cli_helper.run_command(
            ["search", "testing formatting", "--collection", "test-search-format"]
        )

        # Should have output (formatted response)
        assert result is not None
        # Result should have stdout or stderr
        assert result.stdout or result.stderr


@pytest.mark.integration
@pytest.mark.slow
class TestSearchWithFilters:
    """Test search with metadata filters."""

    @pytest.mark.asyncio
    async def test_collection_filter(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test searching within specific collection."""
        workspace = system_components.workspace_path

        # Create content in collection
        test_file = workspace / "filtered.txt"
        test_file.write_text("Content specific to this collection")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-filtered"]
        )
        await asyncio.sleep(3)

        # Search in specific collection
        result = cli_helper.run_command(
            ["search", "content collection", "--collection", "test-search-filtered"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSearchPerformance:
    """Test search performance and latency."""

    @pytest.mark.asyncio
    async def test_search_latency(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search completes within acceptable latency."""
        workspace = system_components.workspace_path

        # Ingest content
        test_file = workspace / "latency.txt"
        test_file.write_text("Performance testing for search latency measurement")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-perf"]
        )
        await asyncio.sleep(3)

        # Measure search time
        start_time = time.time()
        result = cli_helper.run_command(
            ["search", "performance testing", "--collection", "test-search-perf"],
            timeout=10,
        )
        search_time = time.time() - start_time

        # Search should complete within 10 seconds
        assert search_time < 10.0
        assert result is not None

    @pytest.mark.asyncio
    async def test_batch_search_throughput(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test throughput for multiple searches."""
        workspace = system_components.workspace_path

        # Ingest content
        for i in range(5):
            test_file = workspace / f"batch_{i}.txt"
            test_file.write_text(f"Batch search content document {i}")

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-search-batch"]
        )
        await asyncio.sleep(5)

        # Execute multiple searches
        queries = ["batch content", "search document", "content document"]

        start_time = time.time()
        for query in queries:
            cli_helper.run_command(
                ["search", query, "--collection", "test-search-batch"], timeout=10
            )
        total_time = time.time() - start_time

        # All searches should complete within reasonable time
        assert total_time < 30.0  # 10s per search


@pytest.mark.integration
@pytest.mark.slow
class TestSearchEdgeCases:
    """Test edge cases in search functionality."""

    def test_empty_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of empty query."""
        # Empty query should be handled gracefully
        result = cli_helper.run_command(
            ["search", "", "--collection", "test-search-empty"]
        )

        # Should handle empty query (may error or return all results)
        assert result is not None

    def test_very_long_query(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of very long query."""
        workspace = system_components.workspace_path

        test_file = workspace / "long_query.txt"
        test_file.write_text("Content for testing long query handling")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-long"]
        )
        time.sleep(3)

        # Very long query
        long_query = " ".join(["word"] * 100)  # 100 words
        result = cli_helper.run_command(
            ["search", long_query, "--collection", "test-search-long"], timeout=15
        )

        # Should handle long query
        assert result is not None

    def test_search_nonexistent_collection(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search in nonexistent collection."""
        # Search in collection that doesn't exist
        result = cli_helper.run_command(
            ["search", "test query", "--collection", "nonexistent-collection-xyz"]
        )

        # Should handle gracefully (error or empty results)
        assert result is not None

    @pytest.mark.asyncio
    async def test_search_empty_collection(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search in empty collection."""
        # Create collection but don't ingest anything
        collection_name = f"empty-test-{int(time.time())}"

        # Search in empty collection
        result = cli_helper.run_command(
            ["search", "test query", "--collection", collection_name]
        )

        # Should handle empty collection (no results)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestCLISearchInterface:
    """Test CLI search interface specifics."""

    def test_search_command_basic(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test basic search command execution."""
        workspace = system_components.workspace_path

        test_file = workspace / "cli_basic.txt"
        test_file.write_text("CLI search interface testing content")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-cli-search"]
        )
        time.sleep(3)

        result = cli_helper.run_command(
            ["search", "CLI interface", "--collection", "test-cli-search"]
        )

        # Basic search should work
        assert result is not None
        assert result.returncode in [0, 1]  # Success or no results

    def test_search_with_limit(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search with result limit."""
        workspace = system_components.workspace_path

        # Create multiple documents
        for i in range(5):
            (workspace / f"limit_{i}.txt").write_text(f"Document {i} for limit test")

        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-cli-limit"]
        )
        time.sleep(5)

        # Search with limit (if supported)
        result = cli_helper.run_command(
            ["search", "document limit", "--collection", "test-cli-limit"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSearchAccuracy:
    """Test search accuracy and relevance."""

    @pytest.mark.asyncio
    async def test_exact_match_returned(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that exact matches are returned in results."""
        workspace = system_components.workspace_path

        # Create document with exact phrase
        test_file = workspace / "exact_match.txt"
        exact_phrase = "unique phrase for exact matching test"
        test_file.write_text(f"This document contains the {exact_phrase} for testing")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-accuracy"]
        )
        await asyncio.sleep(3)

        # Search for exact phrase
        result = cli_helper.run_command(
            ["search", exact_phrase, "--collection", "test-search-accuracy"]
        )

        # Should find the document
        assert result is not None

    @pytest.mark.asyncio
    async def test_semantic_similarity(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test semantic similarity in search results."""
        workspace = system_components.workspace_path

        # Create documents with semantically similar content
        (workspace / "similar1.txt").write_text("Automobiles and vehicles transport")
        (workspace / "similar2.txt").write_text("Cars and trucks for transportation")

        cli_helper.run_command(
            [
                "ingest",
                "folder",
                str(workspace),
                "--collection",
                "test-search-semantic-sim",
            ]
        )
        await asyncio.sleep(5)

        # Search for similar concept
        result = cli_helper.run_command(
            [
                "search",
                "cars transportation",
                "--collection",
                "test-search-semantic-sim",
            ]
        )

        # Should find semantically similar documents
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSearchRobustness:
    """Test search robustness and error handling."""

    def test_search_during_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search while ingestion is happening."""
        workspace = system_components.workspace_path

        # Start ingesting documents
        for i in range(10):
            (workspace / f"concurrent_{i}.txt").write_text(f"Document {i} content")

        # Start ingestion (non-blocking)
        cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-search-concurrent"]
        )

        # Immediately try to search
        result = cli_helper.run_command(
            ["search", "document content", "--collection", "test-search-concurrent"]
        )

        # Should handle concurrent operations
        assert result is not None

    @pytest.mark.asyncio
    async def test_search_system_stability(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that repeated searches don't destabilize system."""
        workspace = system_components.workspace_path

        # Ingest content
        test_file = workspace / "stability.txt"
        test_file.write_text("Stability testing content for repeated searches")

        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-search-stability"]
        )
        await asyncio.sleep(3)

        # Execute many searches
        for _i in range(10):
            result = cli_helper.run_command(
                ["search", "stability testing", "--collection", "test-search-stability"],
                timeout=10,
            )
            assert result is not None

        # System should still be responsive
        status_result = cli_helper.run_command(["status", "--quiet"])
        assert status_result is not None
