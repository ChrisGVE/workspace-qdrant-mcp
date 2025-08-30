#!/usr/bin/env python3
"""
Simple performance benchmark for workspace-qdrant-mcp using existing collections.
Focuses on testing search performance with real data already in Qdrant.
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Any

from qdrant_client import QdrantClient
import numpy as np


class SimplePerformanceBenchmark:
    """Simple performance testing using existing Qdrant collections."""
    
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.results = {}
        
    def benchmark_existing_collections(self) -> Dict[str, Any]:
        """Benchmark performance on existing collections with real data."""
        print("🔍 Benchmarking existing collections...")
        
        # Get all collections
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        print(f"📊 Found {len(collection_names)} collections: {collection_names}")
        
        results = {}
        
        for collection_name in collection_names:
            print(f"\n📈 Testing collection: {collection_name}")
            
            try:
                # Get collection info
                collection_info = self.client.get_collection(collection_name)
                point_count = collection_info.points_count
                
                if point_count == 0:
                    print(f"  ⏭️  Skipping - no points")
                    continue
                
                print(f"  📊 Points: {point_count:,}")
                
                # Get sample points to understand structure
                sample_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=5,
                    with_vectors=True,
                    with_payload=True
                )
                
                points = sample_result[0]
                if not points:
                    print(f"  ⏭️  Skipping - no accessible points")
                    continue
                
                # Analyze point structure
                sample_point = points[0]
                has_vector = hasattr(sample_point, 'vector') and sample_point.vector is not None
                has_payload = hasattr(sample_point, 'payload') and sample_point.payload is not None
                
                vector_info = {}
                if has_vector:
                    if isinstance(sample_point.vector, dict):
                        # Named vectors
                        vector_info = {name: len(vec) for name, vec in sample_point.vector.items() if isinstance(vec, list)}
                    elif isinstance(sample_point.vector, list):
                        # Single vector
                        vector_info = {"default": len(sample_point.vector)}
                
                print(f"  🧠 Vector info: {vector_info}")
                if has_payload:
                    payload_keys = list(sample_point.payload.keys())[:5]  # Show first 5 keys
                    print(f"  📝 Payload keys (sample): {payload_keys}")
                
                # Benchmark search operations if vectors are available
                collection_results = {
                    "point_count": point_count,
                    "has_vectors": has_vector,
                    "has_payload": has_payload,
                    "vector_info": vector_info,
                    "search_tests": {}
                }
                
                if has_vector and vector_info:
                    # Test search with different configurations
                    search_configs = [
                        {"name": "basic_search", "limit": 10, "with_payload": False, "with_vectors": False},
                        {"name": "with_payload", "limit": 10, "with_payload": True, "with_vectors": False},
                        {"name": "with_vectors", "limit": 10, "with_payload": False, "with_vectors": True},
                        {"name": "large_limit", "limit": 50, "with_payload": False, "with_vectors": False},
                    ]
                    
                    # Use the first available vector for queries
                    if isinstance(sample_point.vector, dict):
                        vector_name = list(vector_info.keys())[0]
                        query_vector = list(sample_point.vector[vector_name])
                        vector_size = len(query_vector)
                        use_named_vector = True
                    else:
                        query_vector = list(sample_point.vector)
                        vector_size = len(query_vector)
                        use_named_vector = False
                    
                    # Create a slightly modified version for search
                    query_vector = [v + np.random.normal(0, 0.01) for v in query_vector]
                    
                    for config in search_configs:
                        print(f"    🔍 {config['name']}...")
                        
                        search_times = []
                        result_counts = []
                        
                        try:
                            # Run multiple search iterations
                            for _ in range(10):
                                start_time = time.perf_counter()
                                
                                if use_named_vector:
                                    # Use the first vector field name
                                    from qdrant_client.models import NamedVector
                                    search_results = self.client.search(
                                        collection_name=collection_name,
                                        query_vector=NamedVector(name=vector_name, vector=query_vector),
                                        limit=config["limit"],
                                        with_payload=config.get("with_payload", False),
                                        with_vectors=config.get("with_vectors", False),
                                    )
                                else:
                                    search_results = self.client.search(
                                        collection_name=collection_name,
                                        query_vector=query_vector,
                                        limit=config["limit"],
                                        with_payload=config.get("with_payload", False),
                                        with_vectors=config.get("with_vectors", False),
                                    )
                                
                                end_time = time.perf_counter()
                                search_times.append((end_time - start_time) * 1000)  # ms
                                result_counts.append(len(search_results))
                            
                            # Calculate statistics
                            if search_times:
                                config_results = {
                                    "mean_time_ms": statistics.mean(search_times),
                                    "median_time_ms": statistics.median(search_times),
                                    "min_time_ms": min(search_times),
                                    "max_time_ms": max(search_times),
                                    "p95_time_ms": self._percentile(search_times, 95),
                                    "std_dev_ms": statistics.stdev(search_times) if len(search_times) > 1 else 0,
                                    "avg_results": statistics.mean(result_counts),
                                    "throughput_qps": 1000 / statistics.mean(search_times),  # Queries per second
                                    "iterations": len(search_times)
                                }
                                
                                collection_results["search_tests"][config["name"]] = config_results
                                print(f"      ⏱️  {config_results['mean_time_ms']:.2f}ms avg ({config_results['throughput_qps']:.1f} QPS)")
                        
                        except Exception as e:
                            print(f"      ❌ Search failed: {e}")
                            continue
                
                results[collection_name] = collection_results
                
            except Exception as e:
                print(f"  ❌ Error testing {collection_name}: {e}")
                continue
        
        return results
    
    def benchmark_concurrent_search(self, collection_name: str, sample_vector: List[float]) -> Dict[str, Any]:
        """Benchmark concurrent search performance on a specific collection."""
        print(f"🔀 Testing concurrent search on {collection_name}...")
        
        concurrent_results = {}
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            print(f"  📈 {concurrency} concurrent searches...")
            
            def single_search():
                return self.client.search(
                    collection_name=collection_name,
                    query_vector=sample_vector,
                    limit=10
                )
            
            async def run_concurrent_searches():
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    start_time = time.perf_counter()
                    futures = [executor.submit(single_search) for _ in range(concurrency)]
                    results = [f.result() for f in concurrent.futures.as_completed(futures)]
                    end_time = time.perf_counter()
                    
                    return {
                        "total_time_ms": (end_time - start_time) * 1000,
                        "successful_requests": len(results),
                        "failed_requests": concurrency - len(results)
                    }
            
            # Run concurrent test
            test_results = []
            try:
                for _ in range(3):  # 3 test runs
                    result = asyncio.run(run_concurrent_searches())
                    test_results.append(result)
                
                # Calculate statistics
                total_times = [r["total_time_ms"] for r in test_results]
                success_rates = [r["successful_requests"] / concurrency for r in test_results]
                
                concurrent_results[f"concurrency_{concurrency}"] = {
                    "mean_total_time_ms": statistics.mean(total_times),
                    "median_total_time_ms": statistics.median(total_times),
                    "mean_per_request_ms": statistics.mean(total_times) / concurrency,
                    "throughput_qps": concurrency / (statistics.mean(total_times) / 1000),
                    "success_rate": statistics.mean(success_rates),
                    "iterations": len(test_results)
                }
                
                print(f"    ⏱️  {concurrent_results[f'concurrency_{concurrency}']['throughput_qps']:.1f} QPS")
                
            except Exception as e:
                print(f"    ❌ Concurrent test failed: {e}")
        
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
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🎯 Starting Simple Performance Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test existing collections
        collection_results = self.benchmark_existing_collections()
        
        # Run concurrent tests on collections with good performance
        concurrent_results = {}
        for coll_name, coll_data in collection_results.items():
            if coll_data.get("search_tests") and coll_data["has_vectors"]:
                # Get a sample vector from the collection
                try:
                    sample_result = self.client.scroll(
                        collection_name=coll_name,
                        limit=1,
                        with_vectors=True
                    )
                    
                    if sample_result[0]:
                        sample_point = sample_result[0][0]
                        if hasattr(sample_point, 'vector') and sample_point.vector:
                            if isinstance(sample_point.vector, dict):
                                # Get first vector from named vectors
                                first_vector = list(sample_point.vector.values())[0]
                            else:
                                first_vector = sample_point.vector
                            
                            if isinstance(first_vector, list):
                                concurrent_results[coll_name] = self.benchmark_concurrent_search(
                                    coll_name, first_vector
                                )
                
                except Exception as e:
                    print(f"  ⚠️  Skipping concurrent test for {coll_name}: {e}")
                    continue
        
        end_time = time.time()
        
        self.results = {
            "metadata": {
                "timestamp": time.time(),
                "duration_seconds": end_time - start_time,
                "qdrant_host": "localhost:6333",
                "collections_tested": list(collection_results.keys())
            },
            "collection_results": collection_results,
            "concurrent_results": concurrent_results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# Workspace-Qdrant-MCP Performance Baseline Report")
        report.append("=" * 60)
        report.append("")
        
        metadata = self.results["metadata"]
        report.append(f"**Test Run:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
        report.append(f"**Duration:** {metadata['duration_seconds']:.1f} seconds")
        report.append(f"**Qdrant Host:** {metadata['qdrant_host']}")
        report.append(f"**Collections Tested:** {len(metadata['collections_tested'])}")
        report.append("")
        
        # Collection performance results
        report.append("## Collection Performance Results")
        report.append("")
        
        collection_results = self.results.get("collection_results", {})
        for collection_name, data in collection_results.items():
            report.append(f"### {collection_name}")
            report.append(f"- **Points:** {data.get('point_count', 0):,}")
            report.append(f"- **Has Vectors:** {data.get('has_vectors', False)}")
            report.append(f"- **Vector Info:** {data.get('vector_info', {})}")
            report.append("")
            
            search_tests = data.get("search_tests", {})
            if search_tests:
                report.append("**Search Performance:**")
                for test_name, test_data in search_tests.items():
                    report.append(f"- {test_name}: {test_data['mean_time_ms']:.2f}ms avg ({test_data['throughput_qps']:.1f} QPS)")
                report.append("")
        
        # Concurrent performance results
        concurrent_results = self.results.get("concurrent_results", {})
        if concurrent_results:
            report.append("## Concurrent Search Performance")
            report.append("")
            
            for collection_name, concurrent_data in concurrent_results.items():
                report.append(f"### {collection_name}")
                for test_name, test_data in concurrent_data.items():
                    concurrency = test_name.split("_")[1]
                    report.append(f"- {concurrency} concurrent: {test_data['throughput_qps']:.1f} QPS ({test_data['success_rate']:.1%} success)")
                report.append("")
        
        # Performance summary
        report.append("## Performance Summary")
        report.append("")
        
        # Calculate overall statistics
        all_search_times = []
        all_throughputs = []
        
        for coll_data in collection_results.values():
            for test_data in coll_data.get("search_tests", {}).values():
                all_search_times.append(test_data["mean_time_ms"])
                all_throughputs.append(test_data["throughput_qps"])
        
        if all_search_times:
            report.append(f"- **Average Search Time:** {statistics.mean(all_search_times):.2f}ms")
            report.append(f"- **Fastest Search:** {min(all_search_times):.2f}ms")
            report.append(f"- **Slowest Search:** {max(all_search_times):.2f}ms")
        
        if all_throughputs:
            report.append(f"- **Average Throughput:** {statistics.mean(all_throughputs):.1f} QPS")
            report.append(f"- **Maximum Throughput:** {max(all_throughputs):.1f} QPS")
        
        # Collections by performance
        if collection_results:
            fastest_collection = min(
                collection_results.items(),
                key=lambda x: min(t["mean_time_ms"] for t in x[1].get("search_tests", {}).values()) if x[1].get("search_tests") else float('inf')
            )
            
            if fastest_collection[1].get("search_tests"):
                fastest_time = min(t["mean_time_ms"] for t in fastest_collection[1]["search_tests"].values())
                report.append(f"- **Fastest Collection:** {fastest_collection[0]} ({fastest_time:.2f}ms)")
        
        report.append("")
        report.append("## Status")
        report.append("")
        
        # Determine overall performance health
        if all_search_times:
            avg_time = statistics.mean(all_search_times)
            if avg_time < 50:
                status = "🟢 Excellent (< 50ms average)"
            elif avg_time < 100:
                status = "🟡 Good (50-100ms average)"
            elif avg_time < 200:
                status = "🟠 Acceptable (100-200ms average)"
            else:
                status = "🔴 Needs Attention (> 200ms average)"
            
            report.append(f"**Overall Performance:** {status}")
        
        report.append("")
        report.append("✅ Performance baseline established successfully!")
        
        return "\n".join(report)


def main():
    """Run the simple performance benchmark."""
    benchmark = SimplePerformanceBenchmark()
    
    try:
        # Run benchmark
        results = benchmark.run_benchmark()
        
        # Save results
        results_dir = Path("performance_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"simple_benchmark_results_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = benchmark.generate_report()
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


if __name__ == "__main__":
    main()