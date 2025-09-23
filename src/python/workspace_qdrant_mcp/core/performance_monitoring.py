"""
Advanced performance monitoring and benchmarking system for metadata filtering operations.

This module provides comprehensive performance monitoring capabilities specifically designed
for multi-tenant search systems with metadata filtering. It includes baseline comparison,
accuracy tracking, regression testing, and dashboard components.

Performance Baselines:
    - Response Time: 2.18ms average target
    - Search Precision: 94.2% minimum baseline
    - Multi-tenant Isolation: 100% enforcement
    - Cache Hit Rate: 80%+ target

Key Features:
    - Real-time performance monitoring with baseline comparison
    - Search accuracy and precision tracking with historical analysis
    - Automated performance regression testing
    - Multi-tenant search performance dashboards
    - Comprehensive benchmarking tools
    - Performance alerting and notifications

Task 233.6: Implementing performance monitoring and benchmarking for metadata filtering.
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .metadata_optimization import PerformanceTracker, FilterOptimizer, QueryOptimizer

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hybrid_search import HybridSearchEngine


@dataclass
class PerformanceBaseline:
    """Performance baseline configuration and tracking."""

    # Response time baselines (in milliseconds)
    target_response_time: float = 2.18
    acceptable_response_time: float = 3.0
    critical_response_time: float = 5.0

    # Accuracy baselines (percentage)
    target_precision: float = 94.2
    minimum_precision: float = 90.0
    target_recall: float = 92.0
    minimum_recall: float = 85.0

    # Cache performance baselines
    target_cache_hit_rate: float = 80.0
    minimum_cache_hit_rate: float = 70.0

    # Multi-tenant isolation
    tenant_isolation_enforcement: float = 100.0

    def to_dict(self) -> Dict:
        """Convert baseline to dictionary."""
        return {
            "response_time": {
                "target_ms": self.target_response_time,
                "acceptable_ms": self.acceptable_response_time,
                "critical_ms": self.critical_response_time
            },
            "accuracy": {
                "target_precision": self.target_precision,
                "minimum_precision": self.minimum_precision,
                "target_recall": self.target_recall,
                "minimum_recall": self.minimum_recall
            },
            "cache_performance": {
                "target_hit_rate": self.target_cache_hit_rate,
                "minimum_hit_rate": self.minimum_cache_hit_rate
            },
            "tenant_isolation": {
                "enforcement_rate": self.tenant_isolation_enforcement
            }
        }


@dataclass
class SearchAccuracyMeasurement:
    """Search accuracy measurement for precision/recall tracking."""

    timestamp: datetime
    query_id: str
    query_text: str
    collection_name: str

    # Results analysis
    total_results: int
    relevant_results: int
    expected_results: int

    # Calculated metrics
    precision: float
    recall: float
    f1_score: float

    # Metadata filtering context
    filter_complexity: int = 0
    tenant_context: Optional[str] = None
    has_multi_tenant_filters: bool = False

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_results > 0:
            self.precision = (self.relevant_results / self.total_results) * 100
        else:
            self.precision = 0.0

        if self.expected_results > 0:
            self.recall = (self.relevant_results / self.expected_results) * 100
        else:
            self.recall = 0.0

        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0


@dataclass
class PerformanceBenchmarkResult:
    """Result of a performance benchmark test."""

    benchmark_id: str
    timestamp: datetime
    test_name: str

    # Performance metrics
    response_times: List[float]
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float

    # Accuracy metrics
    avg_precision: float
    avg_recall: float
    avg_f1_score: float

    # Baseline comparison
    baseline_comparison: Dict[str, Dict]
    performance_regression: bool
    accuracy_regression: bool

    # Test configuration
    test_config: Dict
    metadata: Dict = field(default_factory=dict)

    def passes_baseline(self, baseline: PerformanceBaseline) -> bool:
        """Check if benchmark passes baseline requirements."""
        response_pass = self.avg_response_time <= baseline.target_response_time
        precision_pass = self.avg_precision >= baseline.target_precision
        recall_pass = self.avg_recall >= baseline.target_recall

        return response_pass and precision_pass and recall_pass


class SearchAccuracyTracker:
    """
    Track and analyze search accuracy for precision and recall monitoring.

    This class provides comprehensive accuracy tracking for search operations,
    particularly focusing on metadata filtering accuracy and multi-tenant isolation.
    """

    def __init__(self, baseline: PerformanceBaseline):
        """Initialize accuracy tracker."""
        self.baseline = baseline
        self._measurements: List[SearchAccuracyMeasurement] = []
        self._accuracy_alerts: List[Dict] = []

        logger.info("Search accuracy tracker initialized",
                   target_precision=baseline.target_precision,
                   target_recall=baseline.target_recall)

    def record_search_accuracy(
        self,
        query_id: str,
        query_text: str,
        collection_name: str,
        search_results: List,
        expected_results: List,
        tenant_context: Optional[str] = None,
        filter_complexity: int = 0
    ) -> SearchAccuracyMeasurement:
        """
        Record search accuracy measurement.

        Args:
            query_id: Unique query identifier
            query_text: Query text for analysis
            collection_name: Collection searched
            search_results: Actual search results
            expected_results: Expected/ground truth results
            tenant_context: Tenant context for multi-tenant analysis
            filter_complexity: Complexity score of applied filters

        Returns:
            SearchAccuracyMeasurement with calculated metrics
        """
        # Calculate result overlaps for accuracy
        actual_ids = {getattr(r, 'id', str(r)) for r in search_results}
        expected_ids = {getattr(r, 'id', str(r)) for r in expected_results}

        relevant_results = len(actual_ids.intersection(expected_ids))

        measurement = SearchAccuracyMeasurement(
            timestamp=datetime.now(),
            query_id=query_id,
            query_text=query_text,
            collection_name=collection_name,
            total_results=len(search_results),
            relevant_results=relevant_results,
            expected_results=len(expected_results),
            precision=0.0,  # Will be calculated in __post_init__
            recall=0.0,     # Will be calculated in __post_init__
            f1_score=0.0,   # Will be calculated in __post_init__
            filter_complexity=filter_complexity,
            tenant_context=tenant_context,
            has_multi_tenant_filters=tenant_context is not None
        )

        # Store measurement
        self._measurements.append(measurement)

        # Keep only recent measurements (last 10,000)
        if len(self._measurements) > 10000:
            self._measurements = self._measurements[-10000:]

        # Check for accuracy regressions
        self._check_accuracy_regression(measurement)

        logger.debug("Search accuracy recorded",
                    query_id=query_id,
                    precision=measurement.precision,
                    recall=measurement.recall,
                    f1_score=measurement.f1_score)

        return measurement

    def _check_accuracy_regression(self, measurement: SearchAccuracyMeasurement) -> None:
        """Check for accuracy regression and create alerts."""
        alerts = []

        if measurement.precision < self.baseline.minimum_precision:
            alerts.append({
                "type": "precision_regression",
                "severity": "critical" if measurement.precision < self.baseline.minimum_precision * 0.9 else "warning",
                "measurement": measurement.precision,
                "baseline": self.baseline.target_precision,
                "minimum": self.baseline.minimum_precision
            })

        if measurement.recall < self.baseline.minimum_recall:
            alerts.append({
                "type": "recall_regression",
                "severity": "critical" if measurement.recall < self.baseline.minimum_recall * 0.9 else "warning",
                "measurement": measurement.recall,
                "baseline": self.baseline.target_recall,
                "minimum": self.baseline.minimum_recall
            })

        for alert in alerts:
            alert_record = {
                "timestamp": datetime.now().isoformat(),
                "query_id": measurement.query_id,
                "collection": measurement.collection_name,
                "tenant_context": measurement.tenant_context,
                **alert
            }

            self._accuracy_alerts.append(alert_record)

            logger.warning("Accuracy regression detected",
                          type=alert["type"],
                          severity=alert["severity"],
                          measurement=alert["measurement"],
                          baseline=alert["baseline"])

    def get_accuracy_summary(self, hours: int = 24) -> Dict:
        """Get accuracy summary for recent measurements."""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_measurements = [
            m for m in self._measurements
            if m.timestamp > cutoff
        ]

        if not recent_measurements:
            return {"error": "No recent measurements available"}

        # Calculate aggregated metrics
        precisions = [m.precision for m in recent_measurements]
        recalls = [m.recall for m in recent_measurements]
        f1_scores = [m.f1_score for m in recent_measurements]

        # Multi-tenant analysis
        tenant_stats = defaultdict(list)
        for m in recent_measurements:
            if m.tenant_context:
                tenant_stats[m.tenant_context].append(m)

        tenant_accuracy = {}
        for tenant, measurements in tenant_stats.items():
            tenant_precisions = [m.precision for m in measurements]
            tenant_accuracy[tenant] = {
                "measurement_count": len(measurements),
                "avg_precision": statistics.mean(tenant_precisions) if tenant_precisions else 0,
                "avg_recall": statistics.mean([m.recall for m in measurements]) if measurements else 0,
                "precision_variance": statistics.variance(tenant_precisions) if len(tenant_precisions) > 1 else 0
            }

        return {
            "measurement_period_hours": hours,
            "total_measurements": len(recent_measurements),
            "overall_accuracy": {
                "avg_precision": statistics.mean(precisions),
                "avg_recall": statistics.mean(recalls),
                "avg_f1_score": statistics.mean(f1_scores),
                "precision_p95": sorted(precisions)[int(0.95 * len(precisions))] if precisions else 0,
                "recall_p95": sorted(recalls)[int(0.95 * len(recalls))] if recalls else 0
            },
            "baseline_comparison": {
                "precision_above_target": sum(1 for p in precisions if p >= self.baseline.target_precision) / len(precisions) * 100,
                "recall_above_target": sum(1 for r in recalls if r >= self.baseline.target_recall) / len(recalls) * 100,
                "precision_above_minimum": sum(1 for p in precisions if p >= self.baseline.minimum_precision) / len(precisions) * 100,
                "recall_above_minimum": sum(1 for r in recalls if r >= self.baseline.minimum_recall) / len(recalls) * 100
            },
            "multi_tenant_accuracy": tenant_accuracy,
            "recent_alerts": len([a for a in self._accuracy_alerts if datetime.now() - datetime.fromisoformat(a["timestamp"]) < timedelta(hours=hours)])
        }

    def get_recent_accuracy_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent accuracy alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self._accuracy_alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmarking suite for metadata filtering performance.

    Provides standardized benchmarks to compare against baseline performance
    and detect regressions in search performance and accuracy.
    """

    def __init__(
        self,
        search_engine: "HybridSearchEngine",
        baseline: PerformanceBaseline,
        test_data_path: Optional[str] = None
    ):
        """Initialize benchmark suite."""
        self.search_engine = search_engine
        self.baseline = baseline
        self.test_data_path = test_data_path
        self._benchmark_history: List[PerformanceBenchmarkResult] = []

        logger.info("Performance benchmark suite initialized",
                   baseline_response_time=baseline.target_response_time,
                   baseline_precision=baseline.target_precision)

    async def run_metadata_filtering_benchmark(
        self,
        collection_name: str,
        test_queries: List[Dict],
        iterations: int = 100
    ) -> PerformanceBenchmarkResult:
        """
        Run comprehensive metadata filtering benchmark.

        Args:
            collection_name: Collection to benchmark
            test_queries: List of test queries with expected results
            iterations: Number of iterations per query

        Returns:
            PerformanceBenchmarkResult with comprehensive metrics
        """
        benchmark_id = f"metadata_filter_bench_{int(time.time())}"
        logger.info("Starting metadata filtering benchmark",
                   benchmark_id=benchmark_id,
                   collection=collection_name,
                   queries=len(test_queries),
                   iterations=iterations)

        all_response_times = []
        accuracy_measurements = []

        for query_data in test_queries:
            query_embeddings = query_data.get("embeddings")
            expected_results = query_data.get("expected_results", [])
            project_context = query_data.get("project_context")
            filter_conditions = query_data.get("filter_conditions")

            query_response_times = []

            # Run multiple iterations of the same query
            for iteration in range(iterations):
                start_time = time.time()

                try:
                    # Execute hybrid search with metadata filtering
                    search_result = await self.search_engine.hybrid_search(
                        collection_name=collection_name,
                        query_embeddings=query_embeddings,
                        project_context=project_context,
                        filter_conditions=filter_conditions,
                        limit=20,
                        fusion_method="rrf"
                    )

                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    query_response_times.append(response_time)
                    all_response_times.append(response_time)

                    # Record accuracy for first iteration only (to avoid skew)
                    if iteration == 0:
                        actual_results = search_result.get("fused_results", [])

                        # Calculate accuracy metrics
                        actual_ids = {getattr(r, 'id', str(r)) for r in actual_results}
                        expected_ids = {str(e) for e in expected_results}

                        relevant_results = len(actual_ids.intersection(expected_ids))
                        precision = (relevant_results / len(actual_results)) * 100 if actual_results else 0
                        recall = (relevant_results / len(expected_results)) * 100 if expected_results else 0

                        accuracy_measurements.append({
                            "precision": precision,
                            "recall": recall,
                            "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        })

                except Exception as e:
                    logger.error("Benchmark query failed",
                               benchmark_id=benchmark_id,
                               iteration=iteration,
                               error=str(e))
                    continue

        # Calculate performance statistics
        if all_response_times:
            sorted_times = sorted(all_response_times)
            avg_response_time = statistics.mean(all_response_times)
            p50_response_time = sorted_times[len(sorted_times) // 2]
            p95_response_time = sorted_times[int(0.95 * len(sorted_times))]
            p99_response_time = sorted_times[int(0.99 * len(sorted_times))]
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0

        # Calculate accuracy statistics
        if accuracy_measurements:
            avg_precision = statistics.mean([a["precision"] for a in accuracy_measurements])
            avg_recall = statistics.mean([a["recall"] for a in accuracy_measurements])
            avg_f1_score = statistics.mean([a["f1_score"] for a in accuracy_measurements])
        else:
            avg_precision = avg_recall = avg_f1_score = 0

        # Baseline comparison
        baseline_comparison = self._compare_with_baseline(
            avg_response_time, avg_precision, avg_recall
        )

        # Create benchmark result
        result = PerformanceBenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=datetime.now(),
            test_name="metadata_filtering_benchmark",
            response_times=all_response_times,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1_score,
            baseline_comparison=baseline_comparison,
            performance_regression=avg_response_time > self.baseline.acceptable_response_time,
            accuracy_regression=avg_precision < self.baseline.minimum_precision or avg_recall < self.baseline.minimum_recall,
            test_config={
                "collection_name": collection_name,
                "test_queries": len(test_queries),
                "iterations": iterations,
                "total_operations": len(test_queries) * iterations
            }
        )

        # Store result
        self._benchmark_history.append(result)

        # Keep only recent results (last 100)
        if len(self._benchmark_history) > 100:
            self._benchmark_history = self._benchmark_history[-100:]

        logger.info("Metadata filtering benchmark completed",
                   benchmark_id=benchmark_id,
                   avg_response_time=avg_response_time,
                   avg_precision=avg_precision,
                   passes_baseline=result.passes_baseline(self.baseline))

        return result

    def _compare_with_baseline(
        self,
        response_time: float,
        precision: float,
        recall: float
    ) -> Dict[str, Dict]:
        """Compare results with baseline performance."""
        return {
            "response_time": {
                "measured": response_time,
                "baseline_target": self.baseline.target_response_time,
                "baseline_acceptable": self.baseline.acceptable_response_time,
                "performance_ratio": response_time / self.baseline.target_response_time,
                "meets_target": response_time <= self.baseline.target_response_time,
                "meets_acceptable": response_time <= self.baseline.acceptable_response_time
            },
            "precision": {
                "measured": precision,
                "baseline_target": self.baseline.target_precision,
                "baseline_minimum": self.baseline.minimum_precision,
                "performance_ratio": precision / self.baseline.target_precision if self.baseline.target_precision > 0 else 0,
                "meets_target": precision >= self.baseline.target_precision,
                "meets_minimum": precision >= self.baseline.minimum_precision
            },
            "recall": {
                "measured": recall,
                "baseline_target": self.baseline.target_recall,
                "baseline_minimum": self.baseline.minimum_recall,
                "performance_ratio": recall / self.baseline.target_recall if self.baseline.target_recall > 0 else 0,
                "meets_target": recall >= self.baseline.target_recall,
                "meets_minimum": recall >= self.baseline.minimum_recall
            }
        }

    async def run_multi_tenant_isolation_benchmark(
        self,
        collection_name: str,
        tenant_test_data: Dict[str, List[Dict]]
    ) -> PerformanceBenchmarkResult:
        """
        Benchmark multi-tenant isolation performance and accuracy.

        Args:
            collection_name: Collection to test
            tenant_test_data: Dict mapping tenant IDs to their test queries

        Returns:
            PerformanceBenchmarkResult focused on multi-tenant performance
        """
        benchmark_id = f"multi_tenant_bench_{int(time.time())}"
        logger.info("Starting multi-tenant isolation benchmark",
                   benchmark_id=benchmark_id,
                   tenants=len(tenant_test_data))

        all_response_times = []
        tenant_isolation_violations = 0
        total_tenant_queries = 0

        for tenant_id, tenant_queries in tenant_test_data.items():
            for query_data in tenant_queries:
                total_tenant_queries += 1

                start_time = time.time()

                # Execute search with tenant isolation
                search_result = await self.search_engine.search_tenant_namespace(
                    collection_name=collection_name,
                    query_embeddings=query_data["embeddings"],
                    tenant_namespace=tenant_id,
                    limit=10
                )

                response_time = (time.time() - start_time) * 1000
                all_response_times.append(response_time)

                # Check for tenant isolation violations
                results = search_result.get("fused_results", [])
                for result in results:
                    result_tenant = getattr(result, 'payload', {}).get('tenant_namespace')
                    if result_tenant and result_tenant != tenant_id:
                        tenant_isolation_violations += 1
                        logger.warning("Tenant isolation violation detected",
                                     expected_tenant=tenant_id,
                                     actual_tenant=result_tenant)

        # Calculate isolation metrics
        isolation_enforcement_rate = ((total_tenant_queries - tenant_isolation_violations) /
                                    total_tenant_queries) * 100 if total_tenant_queries > 0 else 100

        # Performance calculations
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        sorted_times = sorted(all_response_times)
        p95_response_time = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0

        result = PerformanceBenchmarkResult(
            benchmark_id=benchmark_id,
            timestamp=datetime.now(),
            test_name="multi_tenant_isolation_benchmark",
            response_times=all_response_times,
            avg_response_time=avg_response_time,
            p50_response_time=sorted_times[len(sorted_times) // 2] if sorted_times else 0,
            p95_response_time=p95_response_time,
            p99_response_time=sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0,
            avg_precision=100.0,  # Set to 100% as we're testing isolation, not search relevance
            avg_recall=100.0,     # Set to 100% as we're testing isolation, not search relevance
            avg_f1_score=100.0,   # Set to 100% as we're testing isolation, not search relevance
            baseline_comparison=self._compare_with_baseline(avg_response_time, 100.0, 100.0),
            performance_regression=avg_response_time > self.baseline.acceptable_response_time,
            accuracy_regression=isolation_enforcement_rate < self.baseline.tenant_isolation_enforcement,
            test_config={
                "collection_name": collection_name,
                "tenant_count": len(tenant_test_data),
                "total_queries": total_tenant_queries,
                "isolation_violations": tenant_isolation_violations,
                "isolation_enforcement_rate": isolation_enforcement_rate
            },
            metadata={
                "tenant_isolation": {
                    "enforcement_rate": isolation_enforcement_rate,
                    "violations": tenant_isolation_violations,
                    "total_queries": total_tenant_queries,
                    "meets_baseline": isolation_enforcement_rate >= self.baseline.tenant_isolation_enforcement
                }
            }
        )

        self._benchmark_history.append(result)

        logger.info("Multi-tenant isolation benchmark completed",
                   benchmark_id=benchmark_id,
                   isolation_enforcement_rate=isolation_enforcement_rate,
                   avg_response_time=avg_response_time)

        return result

    def get_benchmark_history(self, limit: int = 10) -> List[PerformanceBenchmarkResult]:
        """Get recent benchmark history."""
        return self._benchmark_history[-limit:] if self._benchmark_history else []

    def generate_performance_regression_report(self) -> Dict:
        """Generate comprehensive performance regression analysis."""
        if len(self._benchmark_history) < 2:
            return {"error": "Insufficient benchmark history for regression analysis"}

        recent_results = self._benchmark_history[-10:]  # Last 10 benchmarks

        # Calculate trends
        response_time_trend = [r.avg_response_time for r in recent_results]
        precision_trend = [r.avg_precision for r in recent_results]
        recall_trend = [r.avg_recall for r in recent_results]

        # Identify regressions
        regressions = []

        # Check for performance regressions
        if len(response_time_trend) >= 2:
            recent_avg = statistics.mean(response_time_trend[-3:])  # Last 3 results
            historical_avg = statistics.mean(response_time_trend[:-3])  # Earlier results

            if recent_avg > historical_avg * 1.1:  # 10% degradation threshold
                regressions.append({
                    "type": "performance_regression",
                    "metric": "response_time",
                    "current_avg": recent_avg,
                    "historical_avg": historical_avg,
                    "degradation_percent": ((recent_avg - historical_avg) / historical_avg) * 100
                })

        # Check for accuracy regressions
        if len(precision_trend) >= 2:
            recent_precision = statistics.mean(precision_trend[-3:])
            historical_precision = statistics.mean(precision_trend[:-3])

            if recent_precision < historical_precision * 0.95:  # 5% degradation threshold
                regressions.append({
                    "type": "accuracy_regression",
                    "metric": "precision",
                    "current_avg": recent_precision,
                    "historical_avg": historical_precision,
                    "degradation_percent": ((historical_precision - recent_precision) / historical_precision) * 100
                })

        return {
            "analysis_date": datetime.now().isoformat(),
            "benchmark_count": len(recent_results),
            "trends": {
                "response_time": {
                    "values": response_time_trend,
                    "current_avg": statistics.mean(response_time_trend[-3:]) if len(response_time_trend) >= 3 else 0,
                    "trend": "improving" if response_time_trend[-1] < response_time_trend[0] else "degrading"
                },
                "precision": {
                    "values": precision_trend,
                    "current_avg": statistics.mean(precision_trend[-3:]) if len(precision_trend) >= 3 else 0,
                    "trend": "improving" if precision_trend[-1] > precision_trend[0] else "degrading"
                }
            },
            "regressions": regressions,
            "baseline_compliance": {
                "response_time_compliance": sum(1 for r in recent_results if r.avg_response_time <= self.baseline.target_response_time) / len(recent_results) * 100,
                "precision_compliance": sum(1 for r in recent_results if r.avg_precision >= self.baseline.target_precision) / len(recent_results) * 100
            }
        }


class PerformanceMonitoringDashboard:
    """
    Real-time performance monitoring dashboard for metadata filtering operations.

    Provides comprehensive monitoring interface with real-time metrics,
    baseline comparison, and alerting capabilities.
    """

    def __init__(
        self,
        search_engine: "HybridSearchEngine",
        accuracy_tracker: SearchAccuracyTracker,
        benchmark_suite: PerformanceBenchmarkSuite,
        baseline: PerformanceBaseline
    ):
        """Initialize monitoring dashboard."""
        self.search_engine = search_engine
        self.accuracy_tracker = accuracy_tracker
        self.benchmark_suite = benchmark_suite
        self.baseline = baseline

        # Real-time metrics storage
        self._real_time_metrics = deque(maxlen=1000)  # Last 1000 measurements
        self._dashboard_cache = {}
        self._cache_expiry = None

        logger.info("Performance monitoring dashboard initialized")

    def record_real_time_metric(
        self,
        operation_type: str,
        response_time: float,
        accuracy_metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record real-time performance metric."""
        metric = {
            "timestamp": datetime.now(),
            "operation_type": operation_type,
            "response_time": response_time,
            "accuracy_metrics": accuracy_metrics or {},
            "metadata": metadata or {}
        }

        self._real_time_metrics.append(metric)

        # Clear cache on new data
        self._dashboard_cache.clear()
        self._cache_expiry = None

    def get_real_time_dashboard(self) -> Dict:
        """Get real-time dashboard data with comprehensive metrics."""
        # Check cache
        if (self._cache_expiry and
            datetime.now() < self._cache_expiry and
            self._dashboard_cache):
            return self._dashboard_cache

        logger.debug("Generating real-time dashboard data")

        # Get recent metrics (last hour)
        now = datetime.now()
        recent_metrics = [
            m for m in self._real_time_metrics
            if now - m["timestamp"] < timedelta(hours=1)
        ]

        if not recent_metrics:
            return {"error": "No recent metrics available", "generated_at": now.isoformat()}

        # Calculate performance statistics
        response_times = [m["response_time"] for m in recent_metrics]
        avg_response_time = statistics.mean(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]

        # Baseline comparison
        performance_status = "excellent" if avg_response_time <= self.baseline.target_response_time else \
                           "good" if avg_response_time <= self.baseline.acceptable_response_time else \
                           "degraded"

        # Calculate baseline compliance rates
        target_compliance = sum(1 for t in response_times if t <= self.baseline.target_response_time) / len(response_times) * 100
        acceptable_compliance = sum(1 for t in response_times if t <= self.baseline.acceptable_response_time) / len(response_times) * 100

        # Get accuracy summary
        accuracy_summary = self.accuracy_tracker.get_accuracy_summary(hours=1)

        # Get search engine optimization performance
        optimization_performance = {}
        if hasattr(self.search_engine, 'get_optimization_performance'):
            optimization_performance = self.search_engine.get_optimization_performance()

        # Recent benchmark results
        recent_benchmarks = self.benchmark_suite.get_benchmark_history(limit=5)
        benchmark_summary = {
            "recent_count": len(recent_benchmarks),
            "latest_result": recent_benchmarks[-1].to_dict() if recent_benchmarks else None,
            "avg_performance": {
                "response_time": statistics.mean([b.avg_response_time for b in recent_benchmarks]) if recent_benchmarks else 0,
                "precision": statistics.mean([b.avg_precision for b in recent_benchmarks]) if recent_benchmarks else 0
            }
        }

        dashboard_data = {
            "generated_at": now.isoformat(),
            "monitoring_period_hours": 1,
            "total_operations": len(recent_metrics),

            "performance_overview": {
                "status": performance_status,
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "baseline_target": self.baseline.target_response_time,
                "baseline_acceptable": self.baseline.acceptable_response_time,
                "target_compliance_rate": target_compliance,
                "acceptable_compliance_rate": acceptable_compliance
            },

            "accuracy_overview": accuracy_summary,

            "baseline_comparison": {
                "response_time": {
                    "current": avg_response_time,
                    "target": self.baseline.target_response_time,
                    "status": "pass" if avg_response_time <= self.baseline.target_response_time else "fail",
                    "variance": avg_response_time - self.baseline.target_response_time
                },
                "precision": {
                    "current": accuracy_summary.get("overall_accuracy", {}).get("avg_precision", 0),
                    "target": self.baseline.target_precision,
                    "status": "pass" if accuracy_summary.get("overall_accuracy", {}).get("avg_precision", 0) >= self.baseline.target_precision else "fail"
                }
            },

            "optimization_performance": optimization_performance,
            "benchmark_summary": benchmark_summary,

            "alerts": {
                "performance_alerts": [],  # Would be populated from performance tracker
                "accuracy_alerts": self.accuracy_tracker.get_recent_accuracy_alerts(hours=1),
                "total_critical_alerts": 0  # Would be calculated from all alert sources
            },

            "trends": {
                "response_time_trend": response_times[-20:] if len(response_times) >= 20 else response_times,  # Last 20 measurements
                "operations_per_minute": len(recent_metrics) / 60 if recent_metrics else 0
            }
        }

        # Cache dashboard data for 30 seconds
        self._dashboard_cache = dashboard_data
        self._cache_expiry = now + timedelta(seconds=30)

        return dashboard_data

    def export_performance_report(self, filepath: Optional[str] = None) -> Dict:
        """Export comprehensive performance report."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"performance_report_{timestamp}.json"

        report_data = {
            "report_generated": datetime.now().isoformat(),
            "baseline_configuration": self.baseline.to_dict(),
            "real_time_metrics": self.get_real_time_dashboard(),
            "accuracy_analysis": self.accuracy_tracker.get_accuracy_summary(hours=24),
            "benchmark_history": [b.to_dict() for b in self.benchmark_suite.get_benchmark_history()],
            "regression_analysis": self.benchmark_suite.generate_performance_regression_report(),
            "optimization_status": {}
        }

        # Add search engine optimization status if available
        if hasattr(self.search_engine, 'get_optimization_performance'):
            report_data["optimization_status"] = self.search_engine.get_optimization_performance()

        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info("Performance report exported", filepath=filepath)

            return {
                "success": True,
                "filepath": filepath,
                "report_size": len(json.dumps(report_data, default=str))
            }

        except Exception as e:
            logger.error("Failed to export performance report", error=str(e))
            return {"error": f"Export failed: {e}"}


# Convenience class that integrates all monitoring components
class MetadataFilteringPerformanceMonitor:
    """
    Integrated performance monitoring system for metadata filtering operations.

    This class provides a unified interface to all performance monitoring components
    including baseline tracking, benchmarking, accuracy monitoring, and dashboards.
    """

    def __init__(
        self,
        search_engine: "HybridSearchEngine",
        baseline_config: Optional[Dict] = None
    ):
        """Initialize integrated performance monitor."""
        # Initialize baseline
        self.baseline = PerformanceBaseline()
        if baseline_config:
            for key, value in baseline_config.items():
                if hasattr(self.baseline, key):
                    setattr(self.baseline, key, value)

        # Initialize components
        self.accuracy_tracker = SearchAccuracyTracker(self.baseline)
        self.benchmark_suite = PerformanceBenchmarkSuite(search_engine, self.baseline)
        self.dashboard = PerformanceMonitoringDashboard(
            search_engine,
            self.accuracy_tracker,
            self.benchmark_suite,
            self.baseline
        )

        logger.info("Metadata filtering performance monitor initialized")

    async def run_comprehensive_benchmark(
        self,
        collection_name: str,
        test_queries: List[Dict],
        tenant_test_data: Optional[Dict[str, List[Dict]]] = None
    ) -> Dict:
        """Run comprehensive performance and accuracy benchmarks."""
        results = {}

        # Run metadata filtering benchmark
        results["metadata_filtering"] = await self.benchmark_suite.run_metadata_filtering_benchmark(
            collection_name, test_queries
        )

        # Run multi-tenant isolation benchmark if data provided
        if tenant_test_data:
            results["multi_tenant_isolation"] = await self.benchmark_suite.run_multi_tenant_isolation_benchmark(
                collection_name, tenant_test_data
            )

        # Generate regression analysis
        results["regression_analysis"] = self.benchmark_suite.generate_performance_regression_report()

        return results

    def get_performance_status(self) -> Dict:
        """Get comprehensive performance status."""
        return {
            "baseline_configuration": self.baseline.to_dict(),
            "real_time_dashboard": self.dashboard.get_real_time_dashboard(),
            "accuracy_summary": self.accuracy_tracker.get_accuracy_summary(),
            "benchmark_status": {
                "recent_benchmarks": len(self.benchmark_suite.get_benchmark_history()),
                "latest_benchmark": self.benchmark_suite.get_benchmark_history(limit=1)
            }
        }


# Export main classes
__all__ = [
    "PerformanceBaseline",
    "SearchAccuracyMeasurement",
    "SearchAccuracyTracker",
    "PerformanceBenchmarkSuite",
    "PerformanceBenchmarkResult",
    "PerformanceMonitoringDashboard",
    "MetadataFilteringPerformanceMonitor"
]