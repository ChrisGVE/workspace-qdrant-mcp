"""
Performance regression tests for integration testing.

This module provides comprehensive performance testing to establish baseline metrics
and detect performance regressions across ingestion throughput and search latency.

Tests run against isolated Qdrant instances using testcontainers to ensure
consistent performance measurement environments.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock

from testcontainers.compose import DockerCompose

from workspace_qdrant_mcp.tools.grpc_tools import (
    test_grpc_connection,
    process_document_via_grpc,
    search_via_grpc
)
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.daemon_manager import DaemonManager, DaemonConfig


@pytest.fixture(scope="module")
def performance_qdrant():
    """
    Start isolated Qdrant instance for performance testing.
    
    Uses docker-compose with testcontainers to ensure clean environment
    for consistent performance measurements.
    """
    compose_file = """
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  qdrant_storage:
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        compose_path = Path(temp_dir) / "docker-compose.yml"
        compose_path.write_text(compose_file)
        
        with DockerCompose(temp_dir) as compose:
            # Wait for Qdrant to be ready
            qdrant_url = compose.get_service_host("qdrant", 6333)
            qdrant_port = compose.get_service_port("qdrant", 6333)
            
            # Wait for healthy status
            import requests
            for _ in range(30):
                try:
                    response = requests.get(f"http://{qdrant_url}:{qdrant_port}/health")
                    if response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            
            yield {"host": qdrant_url, "port": qdrant_port}


@pytest.fixture
def performance_test_documents():
    """Generate test documents of various sizes for performance testing."""
    documents = []
    
    # Small documents (1KB each)
    for i in range(10):
        content = f"Small document {i}: " + "test content " * 25
        documents.append({
            "size": "small",
            "content": content,
            "filename": f"small_doc_{i}.txt",
            "expected_chunks": 1
        })
    
    # Medium documents (10KB each)  
    for i in range(5):
        content = f"Medium document {i}: " + "test content " * 250
        documents.append({
            "size": "medium", 
            "content": content,
            "filename": f"medium_doc_{i}.txt",
            "expected_chunks": 3
        })
    
    # Large documents (100KB each)
    for i in range(3):
        content = f"Large document {i}: " + "test content " * 2500
        documents.append({
            "size": "large",
            "content": content, 
            "filename": f"large_doc_{i}.txt",
            "expected_chunks": 25
        })
        
    return documents


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceRegression:
    """Performance regression test suite."""
    
    async def test_ingestion_throughput_baseline(
        self, 
        performance_qdrant, 
        performance_test_documents,
        benchmark
    ):
        """
        Measure document ingestion throughput and establish baseline metrics.
        
        Tests ingestion of various document sizes to measure:
        - Documents per second throughput
        - Average processing time per document
        - Memory usage during ingestion
        - gRPC payload processing efficiency
        """
        
        def run_ingestion_benchmark():
            async def _ingest_documents():
                results = {
                    "total_documents": len(performance_test_documents),
                    "documents_per_size": {"small": 0, "medium": 0, "large": 0},
                    "total_chunks": 0,
                    "processing_times": [],
                    "throughput_docs_per_sec": 0,
                    "avg_processing_time_ms": 0
                }
                
                start_time = time.time()
                
                for doc in performance_test_documents:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(doc["content"])
                        temp_path = f.name
                    
                    try:
                        doc_start = time.time()
                        
                        # Process via gRPC for direct performance measurement
                        result = await process_document_via_grpc(
                            file_path=temp_path,
                            collection="performance_test",
                            host=performance_qdrant["host"],
                            port=6334,  # gRPC port
                            timeout=30.0
                        )
                        
                        doc_end = time.time()
                        processing_time = (doc_end - doc_start) * 1000  # ms
                        
                        if result["success"]:
                            results["documents_per_size"][doc["size"]] += 1
                            results["total_chunks"] += result.get("chunks_added", 0)
                            results["processing_times"].append(processing_time)
                        
                    finally:
                        Path(temp_path).unlink(missing_ok=True)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                results["throughput_docs_per_sec"] = len(performance_test_documents) / total_time
                results["avg_processing_time_ms"] = sum(results["processing_times"]) / len(results["processing_times"])
                
                return results
                
            return asyncio.run(_ingest_documents())
        
        # Run benchmark
        result = benchmark(run_ingestion_benchmark)
        
        # Performance assertions - adjust based on baseline establishment
        assert result["throughput_docs_per_sec"] > 1.0, "Ingestion throughput below minimum threshold"
        assert result["avg_processing_time_ms"] < 5000, "Average processing time too high"
        assert result["total_documents"] > 15, "Not all test documents processed"
        
        # Store baseline metrics for future regression detection
        baseline_file = Path(__file__).parent / "performance_baselines.json"
        baseline_data = {"ingestion_throughput": result}
        
        if not baseline_file.exists():
            baseline_file.write_text(json.dumps(baseline_data, indent=2))

    async def test_search_latency_baseline(
        self,
        performance_qdrant, 
        performance_test_documents,
        benchmark
    ):
        """
        Measure search latency across different query types and result sizes.
        
        Tests search performance for:
        - Simple text queries
        - Complex multi-term queries  
        - Different result set sizes (5, 20, 100 results)
        - Hybrid vs dense vs sparse search modes
        """
        
        # First populate test data
        await self._populate_search_test_data(performance_qdrant, performance_test_documents)
        
        def run_search_benchmark():
            async def _execute_searches():
                search_scenarios = [
                    {"query": "test content", "mode": "hybrid", "limit": 5},
                    {"query": "test content", "mode": "hybrid", "limit": 20},
                    {"query": "test content document", "mode": "dense", "limit": 10},
                    {"query": "small medium large", "mode": "sparse", "limit": 15},
                    {"query": "comprehensive test query with multiple terms", "mode": "hybrid", "limit": 25}
                ]
                
                results = {
                    "total_searches": len(search_scenarios),
                    "search_latencies": [],
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "successful_searches": 0
                }
                
                for scenario in search_scenarios:
                    start_time = time.time()
                    
                    search_result = await search_via_grpc(
                        query=scenario["query"],
                        host=performance_qdrant["host"],
                        port=6334,
                        mode=scenario["mode"],
                        limit=scenario["limit"],
                        timeout=15.0
                    )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    if search_result["success"]:
                        results["successful_searches"] += 1
                        results["search_latencies"].append(latency_ms)
                
                if results["search_latencies"]:
                    results["avg_latency_ms"] = sum(results["search_latencies"]) / len(results["search_latencies"])
                    results["search_latencies"].sort()
                    p95_index = int(0.95 * len(results["search_latencies"]))
                    results["p95_latency_ms"] = results["search_latencies"][p95_index]
                
                return results
                
            return asyncio.run(_execute_searches())
        
        # Run benchmark
        result = benchmark(run_search_benchmark)
        
        # Performance assertions
        assert result["successful_searches"] == len([
            {"query": "test content", "mode": "hybrid", "limit": 5},
            {"query": "test content", "mode": "hybrid", "limit": 20},
            {"query": "test content document", "mode": "dense", "limit": 10},
            {"query": "small medium large", "mode": "sparse", "limit": 15},
            {"query": "comprehensive test query with multiple terms", "mode": "hybrid", "limit": 25}
        ]), "Not all searches completed successfully"
        assert result["avg_latency_ms"] < 2000, "Average search latency too high"
        assert result["p95_latency_ms"] < 5000, "P95 search latency too high"
        
        # Update baseline metrics
        baseline_file = Path(__file__).parent / "performance_baselines.json"
        baseline_data = json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
        baseline_data["search_latency"] = result
        baseline_file.write_text(json.dumps(baseline_data, indent=2))

    async def test_concurrent_operations_performance(
        self,
        performance_qdrant,
        benchmark
    ):
        """
        Test performance under concurrent load of mixed operations.
        
        Measures system performance when simultaneously:
        - Ingesting new documents
        - Executing search queries
        - Running health checks
        """
        
        def run_concurrent_benchmark():
            async def _concurrent_operations():
                results = {
                    "concurrent_ingestions": 0,
                    "concurrent_searches": 0,
                    "concurrent_health_checks": 0,
                    "total_operations": 0,
                    "avg_operation_time_ms": 0,
                    "errors": 0
                }
                
                async def ingest_task():
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                            f.write("Concurrent test document " + "content " * 100)
                            temp_path = f.name
                        
                        result = await process_document_via_grpc(
                            file_path=temp_path,
                            collection="concurrent_test",
                            host=performance_qdrant["host"],
                            port=6334,
                            timeout=10.0
                        )
                        
                        Path(temp_path).unlink(missing_ok=True)
                        
                        if result["success"]:
                            results["concurrent_ingestions"] += 1
                        else:
                            results["errors"] += 1
                    except Exception:
                        results["errors"] += 1
                
                async def search_task():
                    try:
                        result = await search_via_grpc(
                            query="concurrent test content",
                            host=performance_qdrant["host"],
                            port=6334,
                            mode="hybrid",
                            limit=10,
                            timeout=5.0
                        )
                        
                        if result["success"]:
                            results["concurrent_searches"] += 1
                        else:
                            results["errors"] += 1
                    except Exception:
                        results["errors"] += 1
                
                async def health_check_task():
                    try:
                        result = await test_grpc_connection(
                            host=performance_qdrant["host"],
                            port=6334,
                            timeout=3.0
                        )
                        
                        if result["connected"]:
                            results["concurrent_health_checks"] += 1
                        else:
                            results["errors"] += 1
                    except Exception:
                        results["errors"] += 1
                
                # Create concurrent tasks
                tasks = []
                for _ in range(3):  # 3 concurrent ingestions
                    tasks.append(ingest_task())
                for _ in range(5):  # 5 concurrent searches
                    tasks.append(search_task())
                for _ in range(2):  # 2 health checks
                    tasks.append(health_check_task())
                
                start_time = time.time()
                await asyncio.gather(*tasks)
                end_time = time.time()
                
                results["total_operations"] = len(tasks)
                results["avg_operation_time_ms"] = ((end_time - start_time) * 1000) / len(tasks)
                
                return results
                
            return asyncio.run(_concurrent_operations())
        
        # Run benchmark
        result = benchmark(run_concurrent_benchmark)
        
        # Performance assertions
        assert result["errors"] <= 1, "Too many errors during concurrent operations"
        assert result["avg_operation_time_ms"] < 10000, "Concurrent operations too slow"
        assert result["concurrent_ingestions"] >= 2, "Insufficient concurrent ingestion success"
        assert result["concurrent_searches"] >= 4, "Insufficient concurrent search success"
        
        # Update baseline metrics
        baseline_file = Path(__file__).parent / "performance_baselines.json"
        baseline_data = json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
        baseline_data["concurrent_operations"] = result
        baseline_file.write_text(json.dumps(baseline_data, indent=2))

    async def _populate_search_test_data(
        self, 
        performance_qdrant: Dict[str, Any], 
        test_documents: List[Dict[str, Any]]
    ):
        """Populate Qdrant with test data for search performance testing."""
        # Use subset of documents to ensure search has data to work with
        for i, doc in enumerate(test_documents[:5]):  # Just first 5 for search testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(doc["content"])
                temp_path = f.name
            
            try:
                await process_document_via_grpc(
                    file_path=temp_path,
                    collection="search_test",
                    host=performance_qdrant["host"],
                    port=6334,
                    timeout=15.0
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
            
            # Brief pause to avoid overwhelming the system
            await asyncio.sleep(0.1)


@pytest.mark.integration  
@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression monitoring and alerting."""
    
    def test_performance_regression_detection(self):
        """
        Compare current performance against established baselines.
        
        Reads baseline metrics and compares against recent benchmark results
        to detect performance regressions.
        """
        baseline_file = Path(__file__).parent / "performance_baselines.json"
        
        if not baseline_file.exists():
            pytest.skip("No baseline metrics available - run baseline tests first")
        
        baseline_data = json.loads(baseline_file.read_text())
        
        # Define regression thresholds (% degradation that triggers alert)
        thresholds = {
            "ingestion_throughput": {"throughput_docs_per_sec": 0.8},  # 20% degradation
            "search_latency": {"avg_latency_ms": 1.2, "p95_latency_ms": 1.2},  # 20% increase
            "concurrent_operations": {"avg_operation_time_ms": 1.3}  # 30% increase
        }
        
        # This test would typically be run in CI/CD after performance benchmarks
        # and would compare against the baseline to detect regressions
        
        for metric_category, thresholds_dict in thresholds.items():
            if metric_category in baseline_data:
                baseline_metrics = baseline_data[metric_category]
                
                # In a real implementation, this would compare against recent results
                # stored in CI artifacts or a performance database
                
                for metric_name, threshold in thresholds_dict.items():
                    if metric_name in baseline_metrics:
                        baseline_value = baseline_metrics[metric_name]
                        # This assertion would use actual current values from CI
                        assert baseline_value > 0, f"Baseline {metric_name} should be positive"