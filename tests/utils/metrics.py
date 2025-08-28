"""
Recall, precision, and performance measurement utilities for testing.

Provides comprehensive metrics collection and analysis for search quality evaluation.
"""

import time
import statistics
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import asyncio


@dataclass
class SearchResult:
    """Represents a search result with relevance information."""
    document_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class SearchMetrics:
    """Comprehensive search quality metrics."""
    query: str
    query_type: str
    
    # Basic metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Advanced metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # P@1, P@5, P@10
    recall_at_k: Dict[int, float] = field(default_factory=dict)     # R@1, R@5, R@10
    average_precision: float = 0.0
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Result statistics
    total_results: int = 0
    relevant_found: int = 0
    total_relevant: int = 0
    
    # Performance metrics
    search_time_ms: float = 0.0
    
    # Result details
    results: List[SearchResult] = field(default_factory=list)
    expected_results: Set[str] = field(default_factory=set)
    found_relevant: Set[str] = field(default_factory=set)
    missing_relevant: Set[str] = field(default_factory=set)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    operation: str
    
    # Timing metrics
    mean_time_ms: float = 0.0
    median_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    
    # Throughput metrics
    operations_per_second: float = 0.0
    
    # Resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Sample statistics
    sample_count: int = 0
    raw_times: List[float] = field(default_factory=list)


class RecallPrecisionMeter:
    """
    Measures recall, precision, and other search quality metrics.
    
    Provides comprehensive evaluation of search results against ground truth data.
    """
    
    def __init__(self):
        self.metrics: List[SearchMetrics] = []
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
    
    def evaluate_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        expected_results: Set[str],
        query_type: str = "unknown",
        search_time_ms: float = 0.0,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> SearchMetrics:
        """
        Evaluate a single search query results.
        
        Args:
            query: The search query
            results: List of search results with 'id', 'score', 'content'
            expected_results: Set of expected document IDs
            query_type: Type of query (symbol, semantic, exact, hybrid)
            search_time_ms: Time taken for search in milliseconds
            relevance_scores: Optional relevance scores for results
            
        Returns:
            SearchMetrics with comprehensive evaluation
        """
        # Convert results to SearchResult objects
        search_results = []
        for i, result in enumerate(results):
            search_results.append(SearchResult(
                document_id=result.get('id', f'result_{i}'),
                score=result.get('score', 0.0),
                content=result.get('content', ''),
                metadata=result.get('metadata', {}),
                rank=i + 1
            ))
        
        # Calculate basic metrics
        found_relevant = set(r.document_id for r in search_results) & expected_results
        
        precision = len(found_relevant) / len(search_results) if search_results else 0.0
        recall = len(found_relevant) / len(expected_results) if expected_results else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate precision@k and recall@k
        precision_at_k = {}
        recall_at_k = {}
        
        for k in [1, 3, 5, 10, 20]:
            if k <= len(search_results):
                results_at_k = set(r.document_id for r in search_results[:k])
                relevant_at_k = results_at_k & expected_results
                
                precision_at_k[k] = len(relevant_at_k) / k
                recall_at_k[k] = len(relevant_at_k) / len(expected_results) if expected_results else 0.0
        
        # Calculate Average Precision (AP)
        average_precision = self._calculate_average_precision(search_results, expected_results)
        
        # Calculate NDCG
        ndcg = self._calculate_ndcg(search_results, expected_results, relevance_scores)
        
        # Create metrics object
        metrics = SearchMetrics(
            query=query,
            query_type=query_type,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            average_precision=average_precision,
            ndcg=ndcg,
            total_results=len(search_results),
            relevant_found=len(found_relevant),
            total_relevant=len(expected_results),
            search_time_ms=search_time_ms,
            results=search_results,
            expected_results=expected_results,
            found_relevant=found_relevant,
            missing_relevant=expected_results - found_relevant
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def _calculate_average_precision(
        self,
        results: List[SearchResult],
        relevant_docs: Set[str]
    ) -> float:
        """Calculate Average Precision (AP)."""
        if not relevant_docs:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, result in enumerate(results):
            if result.document_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def _calculate_ndcg(
        self,
        results: List[SearchResult],
        relevant_docs: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
        if not relevant_docs:
            return 0.0
        
        # Use binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
        
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(results):
            relevance = relevance_scores.get(result.document_id, 0.0)
            if relevance > 0:
                # DCG formula: rel_i / log2(i + 2)
                dcg += relevance / (1.0 if i == 0 else (i + 1).bit_length())
        
        # Calculate ideal DCG (IDCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            if relevance > 0:
                idcg += relevance / (1.0 if i == 0 else (i + 1).bit_length())
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_aggregate_metrics(self, query_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all evaluated queries.
        
        Args:
            query_types: Optional list to filter by query types
            
        Returns:
            Dictionary with aggregate statistics
        """
        # Filter metrics if query types specified
        filtered_metrics = self.metrics
        if query_types:
            filtered_metrics = [m for m in self.metrics if m.query_type in query_types]
        
        if not filtered_metrics:
            return {}
        
        # Calculate averages
        avg_precision = statistics.mean(m.precision for m in filtered_metrics)
        avg_recall = statistics.mean(m.recall for m in filtered_metrics)
        avg_f1 = statistics.mean(m.f1_score for m in filtered_metrics)
        avg_ap = statistics.mean(m.average_precision for m in filtered_metrics)
        avg_ndcg = statistics.mean(m.ndcg for m in filtered_metrics)
        avg_search_time = statistics.mean(m.search_time_ms for m in filtered_metrics)
        
        # Calculate precision@k and recall@k averages
        precision_at_k_avg = {}
        recall_at_k_avg = {}
        
        for k in [1, 3, 5, 10, 20]:
            precision_values = [m.precision_at_k.get(k) for m in filtered_metrics 
                              if k in m.precision_at_k]
            recall_values = [m.recall_at_k.get(k) for m in filtered_metrics 
                           if k in m.recall_at_k]
            
            if precision_values:
                precision_at_k_avg[k] = statistics.mean(precision_values)
            if recall_values:
                recall_at_k_avg[k] = statistics.mean(recall_values)
        
        # Query type breakdown
        query_type_stats = defaultdict(list)
        for metric in filtered_metrics:
            query_type_stats[metric.query_type].append(metric)
        
        query_type_breakdown = {}
        for query_type, type_metrics in query_type_stats.items():
            query_type_breakdown[query_type] = {
                "count": len(type_metrics),
                "avg_precision": statistics.mean(m.precision for m in type_metrics),
                "avg_recall": statistics.mean(m.recall for m in type_metrics),
                "avg_f1": statistics.mean(m.f1_score for m in type_metrics),
                "avg_search_time_ms": statistics.mean(m.search_time_ms for m in type_metrics)
            }
        
        return {
            "summary": {
                "total_queries": len(filtered_metrics),
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1_score": avg_f1,
                "avg_average_precision": avg_ap,
                "avg_ndcg": avg_ndcg,
                "avg_search_time_ms": avg_search_time
            },
            "precision_at_k": precision_at_k_avg,
            "recall_at_k": recall_at_k_avg,
            "by_query_type": query_type_breakdown,
            "performance": {
                "min_search_time_ms": min(m.search_time_ms for m in filtered_metrics),
                "max_search_time_ms": max(m.search_time_ms for m in filtered_metrics),
                "median_search_time_ms": statistics.median(m.search_time_ms for m in filtered_metrics),
                "p95_search_time_ms": self._calculate_percentile(
                    [m.search_time_ms for m in filtered_metrics], 95
                )
            }
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value from list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def export_detailed_results(self, file_path: str) -> None:
        """Export detailed results to JSON file."""
        export_data = {
            "summary": self.get_aggregate_metrics(),
            "individual_queries": [
                {
                    "query": m.query,
                    "query_type": m.query_type,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "average_precision": m.average_precision,
                    "ndcg": m.ndcg,
                    "search_time_ms": m.search_time_ms,
                    "total_results": m.total_results,
                    "relevant_found": m.relevant_found,
                    "total_relevant": m.total_relevant,
                    "precision_at_k": m.precision_at_k,
                    "recall_at_k": m.recall_at_k,
                    "missing_relevant": list(m.missing_relevant)
                }
                for m in self.metrics
            ],
            "benchmarks": {
                name: {
                    "operation": bench.operation,
                    "mean_time_ms": bench.mean_time_ms,
                    "median_time_ms": bench.median_time_ms,
                    "p95_time_ms": bench.p95_time_ms,
                    "operations_per_second": bench.operations_per_second,
                    "sample_count": bench.sample_count
                }
                for name, bench in self.benchmarks.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class PerformanceBenchmarker:
    """
    Performance benchmarking utilities for search operations.
    
    Measures timing, throughput, and resource usage.
    """
    
    def __init__(self):
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self._active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._active_timers[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration in milliseconds."""
        if operation in self._active_timers:
            duration_ms = (time.perf_counter() - self._active_timers[operation]) * 1000
            del self._active_timers[operation]
            return duration_ms
        return 0.0
    
    def benchmark_operation(
        self,
        operation_name: str,
        operation_func,
        iterations: int = 10,
        warmup_iterations: int = 2
    ) -> PerformanceBenchmark:
        """
        Benchmark an operation with multiple iterations.
        
        Args:
            operation_name: Name of the operation being benchmarked
            operation_func: Function to benchmark (can be async)
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not counted)
            
        Returns:
            PerformanceBenchmark with timing statistics
        """
        times = []
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            if asyncio.iscoroutinefunction(operation_func):
                asyncio.run(operation_func())
            else:
                operation_func()
        
        # Actual benchmark iterations
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            if asyncio.iscoroutinefunction(operation_func):
                asyncio.run(operation_func())
            else:
                operation_func()
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        benchmark = PerformanceBenchmark(
            operation=operation_name,
            mean_time_ms=statistics.mean(times),
            median_time_ms=statistics.median(times),
            p95_time_ms=self._calculate_percentile(times, 95),
            min_time_ms=min(times),
            max_time_ms=max(times),
            operations_per_second=1000 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
            sample_count=iterations,
            raw_times=times
        )
        
        self.benchmarks[operation_name] = benchmark
        return benchmark
    
    async def benchmark_async_operation(
        self,
        operation_name: str,
        operation_func,
        iterations: int = 10,
        warmup_iterations: int = 2
    ) -> PerformanceBenchmark:
        """
        Benchmark an async operation with multiple iterations.
        """
        times = []
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            await operation_func()
        
        # Actual benchmark iterations
        for _ in range(iterations):
            start_time = time.perf_counter()
            await operation_func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        benchmark = PerformanceBenchmark(
            operation=operation_name,
            mean_time_ms=statistics.mean(times),
            median_time_ms=statistics.median(times),
            p95_time_ms=self._calculate_percentile(times, 95),
            min_time_ms=min(times),
            max_time_ms=max(times),
            operations_per_second=1000 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
            sample_count=iterations,
            raw_times=times
        )
        
        self.benchmarks[operation_name] = benchmark
        return benchmark
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value from list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        return {
            operation: {
                "mean_time_ms": bench.mean_time_ms,
                "median_time_ms": bench.median_time_ms,
                "p95_time_ms": bench.p95_time_ms,
                "operations_per_second": bench.operations_per_second,
                "sample_count": bench.sample_count
            }
            for operation, bench in self.benchmarks.items()
        }


# Context managers for easy timing
class TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, benchmarker: PerformanceBenchmarker, operation_name: str):
        self.benchmarker = benchmarker
        self.operation_name = operation_name
        self.start_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000


class AsyncTimedOperation:
    """Async context manager for timing operations."""
    
    def __init__(self, benchmarker: PerformanceBenchmarker, operation_name: str):
        self.benchmarker = benchmarker
        self.operation_name = operation_name
        self.start_time = None
        self.duration_ms = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000