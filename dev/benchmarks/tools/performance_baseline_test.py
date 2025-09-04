#!/usr/bin/env python3
"""
Real-world performance baseline test for workspace-qdrant-mcp.

This test performs actual operations against the running Qdrant instance
to establish performance baselines and identify any regressions from recent fixes.
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from qdrant_client.models import Record, ScoredPoint, PointStruct, NamedVector
import numpy as np


class PerformanceBaselineTest:
    """Real-world performance testing against actual Qdrant instance."""
    
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.test_collection = "performance_baseline"
        self.results = {}
        self.existing_collections = []
        
    async def setup(self):
        """Setup test environment."""
        print("🚀 Setting up Performance Baseline Test")
        
        # List existing collections
        collections = self.client.get_collections()
        self.existing_collections = [c.name for c in collections.collections]
        print(f"📊 Found {len(self.existing_collections)} existing collections: {self.existing_collections}")
        
        # Create test collection if it doesn't exist
        if self.test_collection not in self.existing_collections:
            print(f"📁 Creating test collection: {self.test_collection}")
            self.client.create_collection(
                collection_name=self.test_collection,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE),
                },
                # sparse_vectors_config={
                #     "sparse": SparseVectorParams(
                #         index=SparseIndexParams(on_disk=False)
                #     ),
                # }
            )
        
        # Generate test data
        await self.generate_test_data()
        
    async def generate_test_data(self):
        """Generate test vectors and documents for benchmarking."""
        print("🔢 Generating test data...")
        
        # Check if collection already has data
        collection_info = self.client.get_collection(self.test_collection)
        point_count = collection_info.points_count
        
        if point_count > 0:
            print(f"📊 Collection already has {point_count} points, using existing data")
            return
        
        # Generate sample data
        batch_size = 100
        total_points = 1000
        
        print(f"📝 Inserting {total_points} test points...")
        
        for batch_start in range(0, total_points, batch_size):
            points = []
            
            for i in range(batch_start, min(batch_start + batch_size, total_points)):
                # Generate dense vector (384 dimensions)
                dense_vector = np.random.uniform(-1, 1, 384).tolist()
                
                point = PointStruct(
                    id=i,
                    vector=dense_vector,
                    payload={
                        "text": f"Sample document {i} with test content for benchmarking",
                        "category": f"category_{i % 10}",
                        "score": np.random.uniform(0, 1),
                        "batch": batch_start // batch_size
                    }
                )
                points.append(point)
            
            # Insert batch
            self.client.upsert(
                collection_name=self.test_collection,
                points=points
            )
            
        print(f"✅ Inserted {total_points} test points")
    
    def benchmark_search_operations(self) -> Dict[str, Any]:
        """Benchmark various search operations."""
        print("🔍 Benchmarking search operations...")
        
        search_results = {}
        
        # Test different search types and configurations
        search_configs = [
            {"name": "dense_search", "vector_name": "dense", "limit": 10},
            {"name": "dense_search_large", "vector_name": "dense", "limit": 50},
            {"name": "with_payload", "vector_name": "dense", "limit": 10, "with_payload": True},
            {"name": "with_vectors", "vector_name": "dense", "limit": 10, "with_vectors": True},
        ]
        
        for config in search_configs:
            print(f"  Testing {config['name']}...")
            
            # Generate query vector
            query_vector = np.random.uniform(-1, 1, 384).tolist()
            
            times = []
            results_counts = []
            
            # Run multiple iterations
            for iteration in range(20):
                start_time = time.perf_counter()
                
                search_result = self.client.search(
                    collection_name=self.test_collection,
                    query_vector=NamedVector(name=config["vector_name"], vector=query_vector),
                    limit=config.get("limit", 10),
                    with_payload=config.get("with_payload", False),
                    with_vectors=config.get("with_vectors", False),
                )
                
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
                results_counts.append(len(search_result))
            
            # Calculate statistics
            search_results[config["name"]] = {
                "mean_time_ms": statistics.mean(times),
                "median_time_ms": statistics.median(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "p95_time_ms": self._percentile(times, 95),
                "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_results": statistics.mean(results_counts),
                "iterations": len(times)
            }
            
            print(f"    {config['name']}: {search_results[config['name']]['mean_time_ms']:.2f}ms avg")
        
        return search_results
    
    def benchmark_insert_operations(self) -> Dict[str, Any]:
        """Benchmark document insertion operations."""
        print("📝 Benchmarking insert operations...")
        
        insert_results = {}
        temp_collection = f"{self.test_collection}_insert_test"
        
        # Create temporary collection for insert tests
        try:
            self.client.delete_collection(temp_collection)
        except:
            pass
        
        self.client.create_collection(
            collection_name=temp_collection,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),
            },
        )
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            times = []
            
            # Run multiple batches
            for batch_num in range(10):
                points = []
                
                for i in range(batch_size):
                    point_id = batch_num * batch_size + i
                    vector = np.random.uniform(-1, 1, 384).tolist()
                    
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": f"Insert test document {point_id}",
                            "batch": batch_num
                        }
                    )
                    points.append(point)
                
                # Time the insertion
                start_time = time.perf_counter()
                
                self.client.upsert(
                    collection_name=temp_collection,
                    points=points
                )
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            insert_results[f"batch_size_{batch_size}"] = {
                "mean_time_ms": statistics.mean(times),
                "median_time_ms": statistics.median(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "p95_time_ms": self._percentile(times, 95),
                "docs_per_second": batch_size / (statistics.mean(times) / 1000),
                "iterations": len(times)
            }
            
            print(f"    Batch {batch_size}: {insert_results[f'batch_size_{batch_size}']['mean_time_ms']:.2f}ms avg, {insert_results[f'batch_size_{batch_size}']['docs_per_second']:.1f} docs/sec")
        
        # Cleanup
        self.client.delete_collection(temp_collection)
        
        return insert_results
    
    def benchmark_existing_collections(self) -> Dict[str, Any]:
        """Benchmark operations on existing collections."""
        print("📊 Benchmarking existing collections...")
        
        collection_results = {}
        
        # Test collections that have data
        test_collections = ["workspace-qdrant-mcp-project", "bench_project_1000"]
        
        for collection_name in test_collections:
            if collection_name not in self.existing_collections:
                print(f"  Skipping {collection_name} - not found")
                continue
                
            print(f"  Testing collection: {collection_name}")
            
            try:
                # Get collection info
                collection_info = self.client.get_collection(collection_name)
                point_count = collection_info.points_count
                
                if point_count == 0:
                    print(f"    {collection_name}: No points, skipping")
                    continue
                
                # Get a sample point to understand vector structure
                sample_points = self.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_vectors=True
                )[0]
                
                if not sample_points:
                    print(f"    {collection_name}: No accessible points, skipping")
                    continue
                
                sample_point = sample_points[0]
                vector_names = list(sample_point.vector.keys()) if hasattr(sample_point, 'vector') and sample_point.vector else []
                
                print(f"    {collection_name}: {point_count} points, vector fields: {vector_names}")
                
                # Test search with first available vector
                if vector_names:
                    vector_name = vector_names[0]
                    sample_vector = sample_point.vector[vector_name]
                    
                    # Create similar query vector
                    if isinstance(sample_vector, list):
                        query_vector = [v + np.random.normal(0, 0.1) for v in sample_vector[:384]]  # Slight variation
                    else:
                        # Skip non-list vectors for now
                        print(f"    {collection_name}: Skipping non-list vector type")
                        continue
                    
                    # Benchmark search
                    search_times = []
                    for _ in range(10):
                        start_time = time.perf_counter()
                        
                        results = self.client.search(
                            collection_name=collection_name,
                            query_vector=NamedVector(name=vector_name, vector=query_vector),
                            limit=10
                        )
                        
                        end_time = time.perf_counter()
                        search_times.append((end_time - start_time) * 1000)
                    
                    collection_results[collection_name] = {
                        "point_count": point_count,
                        "vector_fields": vector_names,
                        "mean_search_time_ms": statistics.mean(search_times),
                        "median_search_time_ms": statistics.median(search_times),
                        "p95_search_time_ms": self._percentile(search_times, 95)
                    }
                    
                    print(f"    {collection_name} search: {collection_results[collection_name]['mean_search_time_ms']:.2f}ms avg")
                
            except Exception as e:
                print(f"    Error testing {collection_name}: {e}")
                continue
        
        return collection_results
    
    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent search operations."""
        print("🔀 Benchmarking concurrent operations...")
        
        concurrent_results = {}
        query_vector = np.random.uniform(-1, 1, 384).tolist()
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            print(f"  Testing {concurrency} concurrent searches...")
            
            async def concurrent_search():
                return self.client.search(
                    collection_name=self.test_collection,
                    query_vector=NamedVector(name="dense", vector=query_vector),
                    limit=10
                )
            
            async def run_concurrent_searches():
                tasks = [concurrent_search() for _ in range(concurrency)]
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.perf_counter()
                
                # Check for exceptions
                successful = sum(1 for r in results if not isinstance(r, Exception))
                return {
                    "total_time_ms": (end_time - start_time) * 1000,
                    "successful_requests": successful,
                    "failed_requests": concurrency - successful
                }
            
            # Run concurrent test multiple times
            test_runs = []
            for _ in range(5):
                result = asyncio.run(run_concurrent_searches())
                test_runs.append(result)
            
            # Calculate statistics
            total_times = [run["total_time_ms"] for run in test_runs]
            successful_counts = [run["successful_requests"] for run in test_runs]
            
            concurrent_results[f"concurrency_{concurrency}"] = {
                "mean_total_time_ms": statistics.mean(total_times),
                "median_total_time_ms": statistics.median(total_times),
                "mean_per_request_ms": statistics.mean(total_times) / concurrency,
                "throughput_req_per_sec": concurrency / (statistics.mean(total_times) / 1000),
                "success_rate": statistics.mean(successful_counts) / concurrency,
                "iterations": len(test_runs)
            }
            
            print(f"    {concurrency} concurrent: {concurrent_results[f'concurrency_{concurrency}']['mean_total_time_ms']:.2f}ms total, {concurrent_results[f'concurrency_{concurrency}']['throughput_req_per_sec']:.1f} req/sec")
        
        return concurrent_results
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("🎯 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        await self.setup()
        
        # Run all benchmarks
        self.results = {
            "metadata": {
                "timestamp": time.time(),
                "qdrant_host": "localhost:6333",
                "test_collection": self.test_collection,
                "existing_collections": self.existing_collections
            },
            "search_operations": self.benchmark_search_operations(),
            "insert_operations": self.benchmark_insert_operations(),
            "existing_collections": self.benchmark_existing_collections(),
            "concurrent_operations": self.benchmark_concurrent_operations()
        }
        
        end_time = time.time()
        self.results["metadata"]["total_duration_seconds"] = end_time - start_time
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# Performance Baseline Test Results")
        report.append("=" * 50)
        report.append("")
        
        metadata = self.results["metadata"]
        report.append(f"**Test Run:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        report.append(f"**Duration:** {metadata['total_duration_seconds']:.1f} seconds")
        report.append(f"**Qdrant Host:** {metadata['qdrant_host']}")
        report.append(f"**Test Collection:** {metadata['test_collection']}")
        report.append(f"**Existing Collections:** {len(metadata['existing_collections'])}")
        report.append("")
        
        # Search operations
        if "search_operations" in self.results:
            report.append("## Search Operations Performance")
            report.append("")
            for operation, stats in self.results["search_operations"].items():
                report.append(f"### {operation}")
                report.append(f"- Mean Response Time: {stats['mean_time_ms']:.2f}ms")
                report.append(f"- Median Response Time: {stats['median_time_ms']:.2f}ms")
                report.append(f"- P95 Response Time: {stats['p95_time_ms']:.2f}ms")
                report.append(f"- Standard Deviation: {stats['std_dev_ms']:.2f}ms")
                report.append(f"- Average Results: {stats['avg_results']:.1f}")
                report.append("")
        
        # Insert operations
        if "insert_operations" in self.results:
            report.append("## Insert Operations Performance")
            report.append("")
            for operation, stats in self.results["insert_operations"].items():
                report.append(f"### {operation}")
                report.append(f"- Mean Time: {stats['mean_time_ms']:.2f}ms")
                report.append(f"- Throughput: {stats['docs_per_second']:.1f} docs/sec")
                report.append(f"- P95 Time: {stats['p95_time_ms']:.2f}ms")
                report.append("")
        
        # Existing collections
        if "existing_collections" in self.results:
            report.append("## Existing Collections Performance")
            report.append("")
            for collection, stats in self.results["existing_collections"].items():
                report.append(f"### {collection}")
                report.append(f"- Point Count: {stats['point_count']:,}")
                report.append(f"- Vector Fields: {stats['vector_fields']}")
                report.append(f"- Mean Search Time: {stats['mean_search_time_ms']:.2f}ms")
                report.append(f"- P95 Search Time: {stats['p95_search_time_ms']:.2f}ms")
                report.append("")
        
        # Concurrent operations
        if "concurrent_operations" in self.results:
            report.append("## Concurrent Operations Performance")
            report.append("")
            for operation, stats in self.results["concurrent_operations"].items():
                report.append(f"### {operation}")
                report.append(f"- Throughput: {stats['throughput_req_per_sec']:.1f} req/sec")
                report.append(f"- Mean Per Request: {stats['mean_per_request_ms']:.2f}ms")
                report.append(f"- Success Rate: {stats['success_rate']:.1%}")
                report.append("")
        
        report.append("## Summary")
        report.append("")
        
        # Performance summary
        if "search_operations" in self.results:
            search_ops = self.results["search_operations"]
            avg_search_time = statistics.mean([op["mean_time_ms"] for op in search_ops.values()])
            report.append(f"- **Average Search Response Time:** {avg_search_time:.2f}ms")
        
        if "insert_operations" in self.results:
            insert_ops = self.results["insert_operations"]
            best_throughput = max([op["docs_per_second"] for op in insert_ops.values()])
            report.append(f"- **Best Insert Throughput:** {best_throughput:.1f} docs/sec")
        
        if "concurrent_operations" in self.results:
            concurrent_ops = self.results["concurrent_operations"]
            max_throughput = max([op["throughput_req_per_sec"] for op in concurrent_ops.values()])
            report.append(f"- **Maximum Concurrent Throughput:** {max_throughput:.1f} req/sec")
        
        report.append("")
        report.append("✅ Performance baseline test completed successfully!")
        
        return "\n".join(report)
    
    def cleanup(self):
        """Cleanup test resources."""
        try:
            self.client.delete_collection(self.test_collection)
            print(f"🧹 Cleaned up test collection: {self.test_collection}")
        except:
            pass




async def main():
    """Run the performance baseline test."""
    test = PerformanceBaselineTest()
    
    try:
        # Run benchmark
        results = await test.run_full_benchmark()
        
        # Save results
        results_file = Path("performance_results") / f"baseline_results_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = test.generate_report()
        report_file = results_file.with_suffix(".md")
        
        with open(report_file, "w") as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print(report)
        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Report saved to: {report_file}")
        
        return results
    
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        test.cleanup()


if __name__ == "__main__":
    asyncio.run(main())