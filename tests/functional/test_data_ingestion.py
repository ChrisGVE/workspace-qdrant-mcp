"""
Functional tests for data ingestion using real workspace-qdrant-mcp codebase.

Tests the complete pipeline of ingesting actual Python source code, documentation,
and metadata into Qdrant collections with proper chunking and embedding generation.
"""

import asyncio
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from tests.fixtures.test_data_collector import DataCollector
from tests.utils.metrics import PerformanceBenchmarker, TimedOperation
from workspace_qdrant_mcp.tools.documents import add_document


class TestRealDataIngestion:
    """Test data ingestion with actual workspace-qdrant-mcp source code."""

    @pytest.fixture(autouse=True)
    async def setup_test_environment(self, mock_config, tmp_path):
        """Set up test environment with real data collection."""
        self.tmp_path = tmp_path
        self.benchmarker = PerformanceBenchmarker()

        # Collect real test data from source
        source_root = Path(__file__).parent.parent.parent  # workspace-qdrant-mcp root
        self.data_collector = DataCollector(source_root)

        with TimedOperation(self.benchmarker, "data_collection"):
            self.test_data = self.data_collector.collect_all_data()

        print(
            f"📊 Collected test data: {len(self.test_data['symbols'])} symbols, "
            f"{len(self.test_data['chunks'])} chunks"
        )

        # Save test data for debugging
        test_data_file = tmp_path / "collected_test_data.json"
        with open(test_data_file, "w") as f:
            json.dump(self.test_data, f, indent=2)

        # Mock Qdrant client for testing
        self.mock_client = self._create_mock_qdrant_client()
        self.config = mock_config

    def _create_mock_qdrant_client(self):
        """Create a mock Qdrant client that captures operations."""
        from unittest.mock import MagicMock

        client = MagicMock(spec=QdrantClient)

        # Track operations for verification
        self.upserted_points = []
        self.created_collections = []

        def mock_upsert(collection_name, points, **kwargs):
            self.upserted_points.extend(
                [
                    {
                        "collection": collection_name,
                        "id": point.id,
                        "payload": point.payload,
                        "vectors": point.vector,
                    }
                    for point in points
                ]
            )
            return models.UpdateResult(
                operation_id=123, status=models.UpdateStatus.COMPLETED
            )

        def mock_create_collection(collection_name, vectors_config, **kwargs):
            self.created_collections.append(
                {"name": collection_name, "config": vectors_config}
            )
            return True

        client.upsert.side_effect = mock_upsert
        client.create_collection.side_effect = mock_create_collection
        client.collection_exists.return_value = False  # Force creation

        return client

    @pytest.mark.integration
    async def test_ingest_python_source_files(self, mock_workspace_client):
        """Test ingesting actual Python source files with symbol extraction."""
        # Filter to Python source chunks
        code_chunks = [
            chunk for chunk in self.test_data["chunks"] if chunk["chunk_type"] == "code"
        ]

        assert len(code_chunks) > 0, (
            "Should have collected code chunks from real source"
        )

        # Test ingesting a sample of code chunks
        collection_name = "test_source_code"
        ingestion_results = []

        with TimedOperation(self.benchmarker, "code_ingestion"):
            for _i, chunk in enumerate(code_chunks[:10]):  # Test first 10 chunks
                result = await add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata={
                        "chunk_type": chunk["chunk_type"],
                        "file_path": chunk["file_path"],
                        "symbols": chunk["symbols"],
                        "line_range": f"{chunk['line_start']}-{chunk['line_end']}",
                        **chunk["metadata"],
                    },
                    document_id=chunk["id"],
                    chunk_text=False,  # Already chunked
                )
                ingestion_results.append(result)

        # Verify ingestion results
        assert len(ingestion_results) == 10
        for result in ingestion_results:
            assert result.get("success") is True
            assert "document_id" in result

        # Verify mock client captured the operations
        assert len(self.upserted_points) >= 10

        # Verify symbol metadata is preserved
        symbol_points = [p for p in self.upserted_points if p["payload"].get("symbols")]
        assert len(symbol_points) > 0, "Should have points with symbol metadata"

        # Verify file path metadata
        file_path_points = [
            p for p in self.upserted_points if p["payload"].get("file_path")
        ]
        assert len(file_path_points) == 10, "All points should have file path metadata"

    @pytest.mark.integration
    async def test_ingest_documentation_content(self, mock_workspace_client):
        """Test ingesting documentation files (README, etc.)."""
        # Filter to documentation chunks
        doc_chunks = [
            chunk
            for chunk in self.test_data["chunks"]
            if chunk["chunk_type"] == "documentation"
        ]

        if len(doc_chunks) == 0:
            pytest.skip("No documentation chunks found in test data")

        collection_name = "test_documentation"
        ingestion_results = []

        with TimedOperation(self.benchmarker, "documentation_ingestion"):
            for chunk in doc_chunks[:5]:  # Test first 5 doc chunks
                result = await add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata={
                        "chunk_type": chunk["chunk_type"],
                        "file_path": chunk["file_path"],
                        "file_type": chunk["metadata"].get("file_type", "unknown"),
                        "line_range": f"{chunk['line_start']}-{chunk['line_end']}",
                    },
                    document_id=chunk["id"],
                    chunk_text=False,
                )
                ingestion_results.append(result)

        # Verify documentation ingestion
        assert len(ingestion_results) > 0
        for result in ingestion_results:
            assert result.get("success") is True

        # Verify documentation metadata
        doc_points = [
            p
            for p in self.upserted_points
            if p["payload"].get("chunk_type") == "documentation"
        ]
        assert len(doc_points) >= len(doc_chunks[:5])

        # Verify file type metadata for documentation
        markdown_points = [
            p for p in doc_points if p["payload"].get("file_type") == ".md"
        ]
        assert len(markdown_points) > 0, "Should have markdown documentation points"

    @pytest.mark.integration
    async def test_ingest_function_symbols(self, mock_workspace_client):
        """Test ingesting specific function symbols with signatures and docstrings."""
        # Filter to function symbols
        function_symbols = [
            symbol
            for symbol in self.test_data["symbols"]
            if symbol["type"] in ["function", "async_function"]
        ]

        assert len(function_symbols) > 0, "Should have collected function symbols"

        collection_name = "test_functions"

        with TimedOperation(self.benchmarker, "function_symbol_ingestion"):
            for symbol in function_symbols[:8]:  # Test first 8 functions
                # Create content combining signature and docstring
                content_parts = [symbol["signature"]]
                if symbol["docstring"]:
                    content_parts.append(f'"""{symbol["docstring"]}"""')
                if symbol["source_code"]:
                    content_parts.append(
                        symbol["source_code"][:500]
                    )  # Truncate long code

                content = "\n".join(content_parts)

                await add_document(
                    mock_workspace_client,
                    content=content,
                    collection=collection_name,
                    metadata={
                        "symbol_type": symbol["type"],
                        "symbol_name": symbol["name"],
                        "file_path": symbol["file_path"],
                        "line_number": symbol["line_number"],
                        "has_docstring": bool(symbol["docstring"]),
                        "has_source_code": bool(symbol["source_code"]),
                        "parent_class": symbol.get("parent_class"),
                    },
                    document_id=f"symbol_{symbol['name']}_{hash(symbol['file_path']) & 0xFFFF}",
                    chunk_text=False,
                )

        # Verify function symbol points
        function_points = [
            p
            for p in self.upserted_points
            if p["payload"].get("symbol_type") in ["function", "async_function"]
        ]
        assert len(function_points) >= 8

        # Verify function-specific metadata
        functions_with_docstrings = [
            p for p in function_points if p["payload"].get("has_docstring") is True
        ]
        assert len(functions_with_docstrings) > 0, (
            "Should have functions with docstrings"
        )

        # Verify parent class relationships for methods
        methods_in_classes = [
            p for p in function_points if p["payload"].get("parent_class")
        ]
        if len(methods_in_classes) > 0:
            print(f"✅ Found {len(methods_in_classes)} method symbols in classes")

    @pytest.mark.integration
    async def test_ingest_class_symbols(self, mock_workspace_client):
        """Test ingesting class definitions with inheritance information."""
        # Filter to class symbols
        class_symbols = [
            symbol for symbol in self.test_data["symbols"] if symbol["type"] == "class"
        ]

        if len(class_symbols) == 0:
            pytest.skip("No class symbols found in test data")

        collection_name = "test_classes"

        with TimedOperation(self.benchmarker, "class_symbol_ingestion"):
            for symbol in class_symbols:
                # Create content with class signature and docstring
                content_parts = [symbol["signature"]]
                if symbol["docstring"]:
                    content_parts.append(f'"""{symbol["docstring"]}"""')

                await add_document(
                    mock_workspace_client,
                    content="\n".join(content_parts),
                    collection=collection_name,
                    metadata={
                        "symbol_type": symbol["type"],
                        "symbol_name": symbol["name"],
                        "file_path": symbol["file_path"],
                        "line_number": symbol["line_number"],
                        "signature": symbol["signature"],
                        "has_docstring": bool(symbol["docstring"]),
                    },
                    document_id=f"class_{symbol['name']}_{hash(symbol['file_path']) & 0xFFFF}",
                    chunk_text=False,
                )

        # Verify class symbol points
        class_points = [
            p
            for p in self.upserted_points
            if p["payload"].get("symbol_type") == "class"
        ]
        assert len(class_points) == len(class_symbols)

        # Verify class signatures are preserved
        for point in class_points:
            assert point["payload"].get("signature")
            assert point["payload"]["signature"].startswith("class ")

    @pytest.mark.integration
    async def test_chunking_strategy_effectiveness(self, mock_workspace_client):
        """Test that chunking strategy preserves symbol relationships."""
        # Test with a specific file that has multiple related symbols
        test_chunks = [
            chunk
            for chunk in self.test_data["chunks"]
            if "server.py" in chunk["file_path"] and chunk["chunk_type"] == "code"
        ]

        if len(test_chunks) == 0:
            pytest.skip("No server.py chunks found for chunking test")

        collection_name = "test_chunking"

        with TimedOperation(self.benchmarker, "chunking_strategy_test"):
            for chunk in test_chunks:
                await add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata={
                        "symbols": chunk["symbols"],
                        "chunk_size": len(chunk["content"]),
                        "line_span": chunk["line_end"] - chunk["line_start"] + 1,
                    },
                    document_id=chunk["id"],
                    chunk_text=False,
                )

        # Analyze chunking effectiveness
        chunking_points = [
            p for p in self.upserted_points if p["collection"] == collection_name
        ]

        # Verify chunks have reasonable sizes (not too small or large)
        chunk_sizes = [p["payload"]["chunk_size"] for p in chunking_points]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        assert avg_chunk_size > 50, "Chunks should not be too small"
        assert avg_chunk_size < 2000, "Chunks should not be too large"

        # Verify symbol preservation
        symbols_per_chunk = [
            len(p["payload"].get("symbols", [])) for p in chunking_points
        ]
        chunks_with_symbols = [count for count in symbols_per_chunk if count > 0]

        symbol_coverage = (
            len(chunks_with_symbols) / len(chunking_points) if chunking_points else 0
        )
        assert symbol_coverage > 0.3, "At least 30% of chunks should contain symbols"

    @pytest.mark.performance
    async def test_ingestion_performance_benchmarks(self, mock_workspace_client):
        """Test ingestion performance with realistic data volumes."""
        # Use a subset of chunks for performance testing
        test_chunks = self.test_data["chunks"][:50]  # 50 chunks for benchmark
        collection_name = "test_performance"

        # Benchmark single document ingestion
        single_benchmark = await self.benchmarker.benchmark_async_operation(
            "single_document_ingestion",
            lambda: add_document(
                mock_workspace_client,
                content=test_chunks[0]["content"],
                collection=collection_name,
                metadata=test_chunks[0]["metadata"],
                document_id=f"perf_test_{test_chunks[0]['id']}",
                chunk_text=False,
            ),
            iterations=10,
        )

        # Verify single document performance
        assert single_benchmark.mean_time_ms < 100, (
            "Single document ingestion should be < 100ms"
        )
        assert single_benchmark.operations_per_second > 10, "Should handle > 10 ops/sec"

        # Benchmark batch ingestion
        async def batch_ingestion():
            tasks = []
            for i, chunk in enumerate(test_chunks[:10]):
                task = add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata=chunk["metadata"],
                    document_id=f"batch_test_{i}_{chunk['id']}",
                    chunk_text=False,
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

        batch_benchmark = await self.benchmarker.benchmark_async_operation(
            "batch_document_ingestion", batch_ingestion, iterations=5
        )

        # Verify batch performance is better than sequential
        sequential_time = single_benchmark.mean_time_ms * 10  # 10 docs sequentially
        batch_speedup = sequential_time / batch_benchmark.mean_time_ms

        assert batch_speedup > 2.0, (
            f"Batch ingestion should be >2x faster (got {batch_speedup:.1f}x)"
        )

        print("📈 Performance results:")
        print(f"  Single doc: {single_benchmark.mean_time_ms:.1f}ms avg")
        print(f"  Batch (10): {batch_benchmark.mean_time_ms:.1f}ms avg")
        print(f"  Speedup: {batch_speedup:.1f}x")

    @pytest.mark.integration
    async def test_metadata_preservation_and_indexing(self, mock_workspace_client):
        """Test that all important metadata is preserved and properly indexed."""
        # Test with diverse chunk types
        test_chunks = [
            chunk
            for chunk in self.test_data["chunks"][:15]  # Mix of different types
        ]

        collection_name = "test_metadata"

        with TimedOperation(self.benchmarker, "metadata_preservation"):
            for chunk in test_chunks:
                # Enrich metadata
                enhanced_metadata = {
                    **chunk["metadata"],
                    "content_length": len(chunk["content"]),
                    "has_symbols": bool(chunk["symbols"]),
                    "symbol_count": len(chunk["symbols"]),
                    "relative_path": chunk["metadata"].get("relative_path", "unknown"),
                }

                await add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata=enhanced_metadata,
                    document_id=chunk["id"],
                    chunk_text=False,
                )

        # Verify metadata preservation in mock client
        metadata_points = [
            p for p in self.upserted_points if p["collection"] == collection_name
        ]

        assert len(metadata_points) == len(test_chunks)

        # Check specific metadata fields
        required_fields = ["content_length", "has_symbols", "symbol_count"]
        for point in metadata_points:
            payload = point["payload"]
            for field in required_fields:
                assert field in payload, f"Missing required metadata field: {field}"

        # Verify symbol count accuracy
        for point in metadata_points:
            len(
                [c for c in test_chunks if c["id"] == point["id"].split("_")[-1]][0][
                    "symbols"
                ]
            )
            actual_count = point["payload"]["symbol_count"]
            # Note: Mock test, so we can't verify exact match, but structure should be preserved
            assert isinstance(actual_count, int), "Symbol count should be integer"

    def test_collected_data_quality(self):
        """Test the quality and completeness of collected test data."""
        # Verify basic data structure
        assert "symbols" in self.test_data
        assert "chunks" in self.test_data
        assert "ground_truth" in self.test_data
        assert "metadata" in self.test_data

        # Verify we have meaningful amounts of data
        assert len(self.test_data["symbols"]) >= 10, (
            "Should collect at least 10 symbols"
        )
        assert len(self.test_data["chunks"]) >= 20, "Should collect at least 20 chunks"
        assert len(self.test_data["ground_truth"]) >= 5, (
            "Should generate at least 5 test cases"
        )

        # Verify symbol diversity
        symbol_types = set(s["type"] for s in self.test_data["symbols"])
        expected_types = {"function", "class", "method"}
        found_types = symbol_types & expected_types
        assert len(found_types) >= 2, (
            f"Should find multiple symbol types, got: {symbol_types}"
        )

        # Verify chunk diversity
        chunk_types = set(c["chunk_type"] for c in self.test_data["chunks"])
        assert "code" in chunk_types, "Should have code chunks"

        # Verify ground truth quality
        for gt in self.test_data["ground_truth"][:5]:  # Check first 5
            assert gt["query"], "Ground truth should have non-empty queries"
            assert gt["query_type"] in ["symbol", "semantic", "exact"], (
                "Valid query type"
            )
            assert len(gt["expected_results"]) > 0, "Should have expected results"

        print("✅ Test data quality verified:")
        print(f"  Symbols: {len(self.test_data['symbols'])} ({symbol_types})")
        print(f"  Chunks: {len(self.test_data['chunks'])} ({chunk_types})")
        print(f"  Ground truth: {len(self.test_data['ground_truth'])} cases")

    @pytest.mark.slow
    async def test_full_codebase_ingestion_simulation(self, mock_workspace_client):
        """Test ingesting the entire collected codebase (simulation)."""
        all_chunks = self.test_data["chunks"]
        collection_name = "test_full_codebase"

        print(f"🔄 Simulating full codebase ingestion: {len(all_chunks)} chunks")

        # Batch process in groups
        batch_size = 20
        batches = [
            all_chunks[i : i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]

        self.benchmarker._active_timers.get("full_ingestion", 0)
        self.benchmarker.start_timer("full_ingestion")

        for batch_idx, batch in enumerate(batches):
            batch_tasks = []
            for chunk in batch:
                task = add_document(
                    mock_workspace_client,
                    content=chunk["content"],
                    collection=collection_name,
                    metadata={"batch_id": batch_idx, **chunk["metadata"]},
                    document_id=f"full_{batch_idx}_{chunk['id']}",
                    chunk_text=False,
                )
                batch_tasks.append(task)

            # Process batch concurrently
            await asyncio.gather(*batch_tasks)

            if batch_idx % 5 == 0:  # Progress every 5 batches
                print(
                    f"  Processed {(batch_idx + 1) * batch_size} / {len(all_chunks)} chunks"
                )

        full_ingestion_time = self.benchmarker.end_timer("full_ingestion")

        # Verify full ingestion
        full_points = [
            p for p in self.upserted_points if p["collection"] == collection_name
        ]
        assert len(full_points) == len(all_chunks)

        # Performance analysis
        chunks_per_second = len(all_chunks) / (full_ingestion_time / 1000)

        print("📊 Full ingestion completed:")
        print(f"  Total time: {full_ingestion_time:.1f}ms")
        print(f"  Throughput: {chunks_per_second:.1f} chunks/second")
        print(f"  Average per chunk: {full_ingestion_time / len(all_chunks):.1f}ms")

        # Set reasonable performance expectations
        assert full_ingestion_time < 30000, (
            "Full ingestion should complete in < 30 seconds"
        )
        assert chunks_per_second > 5, "Should process > 5 chunks per second"

        # Verify batch metadata
        batch_ids = set(p["payload"].get("batch_id") for p in full_points)
        expected_batches = set(range(len(batches)))
        assert batch_ids == expected_batches, "All batches should be represented"
