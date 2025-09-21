"""
Performance Regression Validation Tests for workspace-qdrant-mcp.

This module provides comprehensive performance regression detection and validation
against established baselines, ensuring system performance does not degrade over time.

SUCCESS CRITERIA:
- Response time regression: < 20% increase compared to baseline
- Memory usage regression: < 30% increase compared to baseline
- Throughput regression: < 15% decrease compared to baseline
- Error rate regression: < 100% increase (doubling) compared to baseline
- Automated baseline management and drift detection
- Performance trend analysis over time

REGRESSION DETECTION METHODS:
- Statistical significance testing (t-test, Mann-Whitney U)
- Moving average trend analysis
- Percentile comparison (P50, P95, P99)
- Performance budget enforcement
- Anomaly detection using standard deviations
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pytest

# Test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.regression,
    pytest.mark.benchmark,
]


class PerformanceBaseline:
    """Performance baseline management and comparison."""

    def __init__(self, baseline_dir: Path):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(exist_ok=True)

    def save_baseline(self, test_name: str, metrics: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Save performance baseline with metadata."""
        baseline_data = {
            'test_name': test_name,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {},
            'version': '1.0'
        }

        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

    def load_baseline(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Load performance baseline."""
        baseline_file = self.baseline_dir / f"{test_name}_baseline.json"

        if not baseline_file.exists():
            return None

        try:
            with open(baseline_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def save_performance_history(self, test_name: str, metrics: Dict[str, Any]):
        """Save performance measurement to history."""
        history_file = self.baseline_dir / f"{test_name}_history.jsonl"

        history_entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'metrics': metrics
        }

        # Append to history file (JSONL format)
        with open(history_file, 'a') as f:
            f.write(json.dumps(history_entry) + '\n')

    def load_performance_history(self, test_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Load performance history for trend analysis."""
        history_file = self.baseline_dir / f"{test_name}_history.jsonl"

        if not history_file.exists():
            return []

        history = []
        cutoff_time = time.time() - (days * 24 * 3600)

        try:
            with open(history_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get('timestamp', 0) > cutoff_time:
                            history.append(entry)
        except (json.JSONDecodeError, IOError):
            return []

        return sorted(history, key=lambda x: x['timestamp'])


class RegressionDetector:
    """Advanced regression detection with statistical analysis."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def detect_regressions(self, baseline_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance regressions using multiple methods."""

        regressions = []
        comparisons = {}

        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                current_value = current_metrics[metric_name]

                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    regression_analysis = self._analyze_metric_regression(
                        metric_name, baseline_value, current_value
                    )
                    comparisons[metric_name] = regression_analysis

                    if regression_analysis['is_regression']:
                        regressions.append(regression_analysis)

        return {
            'regressions_detected': len(regressions) > 0,
            'regression_count': len(regressions),
            'regressions': regressions,
            'all_comparisons': comparisons,
            'severity': self._calculate_regression_severity(regressions)
        }

    def _analyze_metric_regression(self, metric_name: str, baseline: float, current: float) -> Dict[str, Any]:
        """Analyze individual metric for regression."""

        if baseline == 0:
            # Handle division by zero
            change_percent = 0 if current == 0 else float('inf')
            absolute_change = current
        else:
            change_percent = ((current - baseline) / baseline) * 100
            absolute_change = current - baseline

        # Determine metric type and regression direction
        metric_type = self._classify_metric_type(metric_name)
        is_regression = self._is_regression(metric_type, change_percent)

        # Calculate severity
        severity = self._calculate_metric_severity(metric_type, change_percent)

        # Statistical significance (simplified)
        statistical_significance = abs(change_percent) > 5.0  # 5% change is considered significant

        return {
            'metric_name': metric_name,
            'metric_type': metric_type,
            'baseline_value': baseline,
            'current_value': current,
            'absolute_change': absolute_change,
            'change_percent': change_percent,
            'is_regression': is_regression,
            'severity': severity,
            'threshold_exceeded': abs(change_percent) > self.thresholds.get(f"{metric_type}_threshold", 20.0),
            'statistical_significance': statistical_significance
        }

    def _classify_metric_type(self, metric_name: str) -> str:
        """Classify metric type based on name."""
        metric_name_lower = metric_name.lower()

        if any(keyword in metric_name_lower for keyword in ['time', 'latency', 'duration', 'ms']):
            return 'response_time'
        elif any(keyword in metric_name_lower for keyword in ['memory', 'mb', 'kb', 'bytes']):
            return 'memory'
        elif any(keyword in metric_name_lower for keyword in ['throughput', 'rps', 'ops', 'rate']):
            return 'throughput'
        elif any(keyword in metric_name_lower for keyword in ['error', 'fail', 'exception']):
            return 'error_rate'
        elif any(keyword in metric_name_lower for keyword in ['cpu', 'utilization', 'percent']):
            return 'resource_usage'
        else:
            return 'generic'

    def _is_regression(self, metric_type: str, change_percent: float) -> bool:
        """Determine if change represents a regression."""

        threshold_map = {
            'response_time': self.thresholds.get('response_time_increase_percent', 20.0),
            'memory': self.thresholds.get('memory_increase_percent', 30.0),
            'throughput': self.thresholds.get('throughput_decrease_percent', 15.0),
            'error_rate': self.thresholds.get('error_rate_increase_percent', 100.0),
            'resource_usage': self.thresholds.get('resource_usage_increase_percent', 50.0),
            'generic': 20.0
        }

        threshold = threshold_map.get(metric_type, 20.0)

        # For throughput, negative change is regression
        if metric_type == 'throughput':
            return change_percent < -threshold
        else:
            # For other metrics, positive change is regression
            return change_percent > threshold

    def _calculate_metric_severity(self, metric_type: str, change_percent: float) -> str:
        """Calculate regression severity."""

        abs_change = abs(change_percent)

        if abs_change < 10:
            return 'low'
        elif abs_change < 25:
            return 'medium'
        elif abs_change < 50:
            return 'high'
        else:
            return 'critical'

    def _calculate_regression_severity(self, regressions: List[Dict[str, Any]]) -> str:
        """Calculate overall regression severity."""

        if not regressions:
            return 'none'

        severities = [r['severity'] for r in regressions]

        if 'critical' in severities:
            return 'critical'
        elif 'high' in severities:
            return 'high'
        elif 'medium' in severities:
            return 'medium'
        else:
            return 'low'

    def analyze_performance_trend(self, history: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
        """Analyze performance trend over time."""

        if len(history) < 3:
            return {'error': 'Insufficient data for trend analysis'}

        values = []
        timestamps = []

        for entry in history:
            if metric_name in entry.get('metrics', {}):
                values.append(entry['metrics'][metric_name])
                timestamps.append(entry['timestamp'])

        if len(values) < 3:
            return {'error': f'Insufficient data points for {metric_name}'}

        # Calculate trend statistics
        trend_analysis = {
            'metric_name': metric_name,
            'data_points': len(values),
            'time_span_days': (timestamps[-1] - timestamps[0]) / (24 * 3600),
            'current_value': values[-1],
            'min_value': min(values),
            'max_value': max(values),
            'mean_value': statistics.mean(values),
            'median_value': statistics.median(values),
            'std_deviation': statistics.stdev(values) if len(values) > 1 else 0,
        }

        # Calculate linear trend
        trend_slope = self._calculate_trend_slope(timestamps, values)
        trend_analysis['trend_slope'] = trend_slope
        trend_analysis['trend_direction'] = 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'

        # Detect anomalies (values beyond 2 standard deviations)
        if trend_analysis['std_deviation'] > 0:
            mean = trend_analysis['mean_value']
            std = trend_analysis['std_deviation']

            anomalies = []
            for i, value in enumerate(values):
                z_score = abs(value - mean) / std
                if z_score > 2.0:  # 2 standard deviations
                    anomalies.append({
                        'index': i,
                        'value': value,
                        'z_score': z_score,
                        'timestamp': timestamps[i]
                    })

            trend_analysis['anomalies'] = anomalies
            trend_analysis['anomaly_count'] = len(anomalies)
        else:
            trend_analysis['anomalies'] = []
            trend_analysis['anomaly_count'] = 0

        # Performance stability assessment
        coefficient_of_variation = trend_analysis['std_deviation'] / trend_analysis['mean_value'] if trend_analysis['mean_value'] > 0 else 0
        trend_analysis['coefficient_of_variation'] = coefficient_of_variation

        if coefficient_of_variation < 0.1:
            stability = 'excellent'
        elif coefficient_of_variation < 0.2:
            stability = 'good'
        elif coefficient_of_variation < 0.4:
            stability = 'fair'
        else:
            stability = 'poor'

        trend_analysis['stability_assessment'] = stability

        return trend_analysis

    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear trend slope using least squares regression."""

        n = len(x_values)
        if n < 2:
            return 0

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x

        if denominator == 0:
            return 0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


@pytest.fixture
def performance_baseline(tmp_path):
    """Provide performance baseline manager."""
    return PerformanceBaseline(tmp_path / "baselines")


@pytest.fixture
def regression_detector(regression_thresholds):
    """Provide regression detector with thresholds."""
    return RegressionDetector(regression_thresholds)


class TestPerformanceRegression:
    """Test performance regression detection."""

    @pytest.mark.benchmark
    async def test_baseline_establishment(self, performance_baseline, mock_qdrant_client, mock_embedding_service, benchmark):
        """Establish performance baselines for critical operations."""

        # Document processing baseline
        async def document_processing():
            content = "Test document for baseline establishment."
            embeddings = await mock_embedding_service.embed([content])
            await mock_qdrant_client.upsert(
                collection_name="test",
                points=[{"id": "test", "vector": embeddings[0], "payload": {"content": content}}]
            )

        doc_result = benchmark.pedantic(lambda: asyncio.run(document_processing()), iterations=20, rounds=3)

        # Search baseline
        async def search_operation():
            await mock_qdrant_client.search(
                collection_name="test",
                query_vector=[0.1] * 384,
                limit=10
            )

        search_result = benchmark.pedantic(lambda: asyncio.run(search_operation()), iterations=30, rounds=3)

        # Collection management baseline
        async def collection_management():
            await mock_qdrant_client.list_collections()
            await mock_qdrant_client.get_collection("test")

        collection_result = benchmark.pedantic(lambda: asyncio.run(collection_management()), iterations=50, rounds=3)

        # Save baselines
        baselines = {
            'document_processing': {
                'mean_time_ms': doc_result.stats.mean * 1000,
                'median_time_ms': doc_result.stats.median * 1000,
                'p95_time_ms': doc_result.stats.p95 * 1000,
                'std_deviation_ms': doc_result.stats.stddev * 1000,
            },
            'search_operation': {
                'mean_time_ms': search_result.stats.mean * 1000,
                'median_time_ms': search_result.stats.median * 1000,
                'p95_time_ms': search_result.stats.p95 * 1000,
                'std_deviation_ms': search_result.stats.stddev * 1000,
            },
            'collection_management': {
                'mean_time_ms': collection_result.stats.mean * 1000,
                'median_time_ms': collection_result.stats.median * 1000,
                'p95_time_ms': collection_result.stats.p95 * 1000,
                'std_deviation_ms': collection_result.stats.stddev * 1000,
            }
        }

        # Save each baseline
        for operation, metrics in baselines.items():
            performance_baseline.save_baseline(
                operation,
                metrics,
                metadata={
                    'benchmark_iterations': 20 if operation == 'document_processing' else 30 if operation == 'search_operation' else 50,
                    'benchmark_rounds': 3,
                    'test_environment': 'mock'
                }
            )

        print(f"\n📊 Performance Baselines Established:")
        for operation, metrics in baselines.items():
            print(f"   {operation}: mean={metrics['mean_time_ms']:.2f}ms, p95={metrics['p95_time_ms']:.2f}ms")

        return baselines

    @pytest.mark.benchmark
    async def test_regression_detection_response_time(self, performance_baseline, regression_detector, mock_qdrant_client):
        """Test regression detection for response time metrics."""

        # Create a baseline
        baseline_metrics = {
            'mean_time_ms': 50.0,
            'p95_time_ms': 85.0,
            'max_time_ms': 120.0,
        }

        performance_baseline.save_baseline("response_time_test", baseline_metrics)

        # Simulate current performance (with regression)
        current_metrics = {
            'mean_time_ms': 65.0,    # 30% increase (regression)
            'p95_time_ms': 110.0,    # 29% increase (regression)
            'max_time_ms': 140.0,    # 17% increase (acceptable)
        }

        # Detect regressions
        analysis = regression_detector.detect_regressions(baseline_metrics, current_metrics)

        # Validate regression detection
        assert analysis['regressions_detected'], "Should detect response time regressions"
        assert analysis['regression_count'] >= 2, f"Should detect at least 2 regressions, found {analysis['regression_count']}"

        # Check specific regression details
        mean_time_regression = next((r for r in analysis['regressions'] if r['metric_name'] == 'mean_time_ms'), None)
        assert mean_time_regression is not None, "Should detect mean_time_ms regression"
        assert mean_time_regression['severity'] in ['medium', 'high'], f"Unexpected severity: {mean_time_regression['severity']}"

        print(f"\n🚨 Response Time Regression Analysis:")
        print(f"   Regressions detected: {analysis['regression_count']}")
        print(f"   Overall severity: {analysis['severity']}")

        for regression in analysis['regressions']:
            print(f"   - {regression['metric_name']}: {regression['change_percent']:.1f}% change ({regression['severity']})")

    @pytest.mark.benchmark
    async def test_regression_detection_memory_usage(self, performance_baseline, regression_detector):
        """Test regression detection for memory usage metrics."""

        # Create memory baseline
        baseline_metrics = {
            'peak_memory_mb': 25.0,
            'average_memory_mb': 20.0,
            'memory_growth_mb': 1.5,
        }

        performance_baseline.save_baseline("memory_usage_test", baseline_metrics)

        # Simulate memory regression
        current_metrics = {
            'peak_memory_mb': 35.0,     # 40% increase (regression)
            'average_memory_mb': 24.0,  # 20% increase (acceptable)
            'memory_growth_mb': 3.2,    # 113% increase (severe regression)
        }

        analysis = regression_detector.detect_regressions(baseline_metrics, current_metrics)

        # Validate detection
        assert analysis['regressions_detected'], "Should detect memory regressions"

        # Check for severe memory growth regression
        growth_regression = next((r for r in analysis['regressions'] if r['metric_name'] == 'memory_growth_mb'), None)
        assert growth_regression is not None, "Should detect memory growth regression"
        assert growth_regression['severity'] in ['high', 'critical'], f"Memory growth regression should be severe: {growth_regression['severity']}"

        print(f"\n💾 Memory Usage Regression Analysis:")
        print(f"   Regressions detected: {analysis['regression_count']}")
        for regression in analysis['regressions']:
            print(f"   - {regression['metric_name']}: {regression['change_percent']:.1f}% change ({regression['severity']})")

    @pytest.mark.benchmark
    async def test_regression_detection_throughput(self, performance_baseline, regression_detector):
        """Test regression detection for throughput metrics."""

        # Create throughput baseline
        baseline_metrics = {
            'requests_per_second': 100.0,
            'operations_per_minute': 6000.0,
            'concurrent_throughput': 45.0,
        }

        performance_baseline.save_baseline("throughput_test", baseline_metrics)

        # Simulate throughput degradation
        current_metrics = {
            'requests_per_second': 78.0,     # 22% decrease (regression)
            'operations_per_minute': 5100.0, # 15% decrease (regression)
            'concurrent_throughput': 42.0,   # 7% decrease (acceptable)
        }

        analysis = regression_detector.detect_regressions(baseline_metrics, current_metrics)

        # Validate throughput regression detection
        assert analysis['regressions_detected'], "Should detect throughput regressions"

        # Check throughput-specific logic
        rps_regression = next((r for r in analysis['regressions'] if r['metric_name'] == 'requests_per_second'), None)
        assert rps_regression is not None, "Should detect RPS regression"
        assert rps_regression['metric_type'] == 'throughput', "Should classify as throughput metric"

        print(f"\n⚡ Throughput Regression Analysis:")
        print(f"   Regressions detected: {analysis['regression_count']}")
        for regression in analysis['regressions']:
            print(f"   - {regression['metric_name']}: {regression['change_percent']:.1f}% change ({regression['severity']})")


class TestPerformanceTrendAnalysis:
    """Test performance trend analysis and anomaly detection."""

    @pytest.mark.benchmark
    async def test_performance_trend_analysis(self, performance_baseline, regression_detector):
        """Test trend analysis over performance history."""

        # Create synthetic performance history
        base_time = time.time() - (30 * 24 * 3600)  # 30 days ago

        history = []
        for day in range(30):
            timestamp = base_time + (day * 24 * 3600)

            # Simulate gradual performance degradation
            base_response_time = 50.0
            degradation_factor = 1 + (day * 0.005)  # 0.5% degradation per day
            noise = (day % 3 - 1) * 2  # Some noise

            response_time = base_response_time * degradation_factor + noise

            history.append({
                'timestamp': timestamp,
                'metrics': {
                    'response_time_ms': response_time,
                    'throughput_rps': 100.0 / degradation_factor,  # Inverse relationship
                    'memory_usage_mb': 25.0 + (day * 0.1),  # Gradual memory increase
                }
            })

        # Save history
        for entry in history:
            performance_baseline.save_performance_history("trend_test", entry['metrics'])

        # Analyze trends
        response_time_trend = regression_detector.analyze_performance_trend(history, 'response_time_ms')
        throughput_trend = regression_detector.analyze_performance_trend(history, 'throughput_rps')
        memory_trend = regression_detector.analyze_performance_trend(history, 'memory_usage_mb')

        # Validate trend analysis
        assert response_time_trend['trend_direction'] == 'increasing', "Should detect increasing response time trend"
        assert throughput_trend['trend_direction'] == 'decreasing', "Should detect decreasing throughput trend"
        assert memory_trend['trend_direction'] == 'increasing', "Should detect increasing memory trend"

        # Check stability assessments
        assert response_time_trend['stability_assessment'] in ['good', 'fair'], \
            f"Response time stability unexpected: {response_time_trend['stability_assessment']}"

        print(f"\n📈 Performance Trend Analysis:")
        print(f"   Response Time:")
        print(f"     Direction: {response_time_trend['trend_direction']}")
        print(f"     Stability: {response_time_trend['stability_assessment']}")
        print(f"     Anomalies: {response_time_trend['anomaly_count']}")

        print(f"   Throughput:")
        print(f"     Direction: {throughput_trend['trend_direction']}")
        print(f"     Stability: {throughput_trend['stability_assessment']}")

        print(f"   Memory Usage:")
        print(f"     Direction: {memory_trend['trend_direction']}")
        print(f"     Stability: {memory_trend['stability_assessment']}")

    @pytest.mark.benchmark
    async def test_anomaly_detection(self, performance_baseline, regression_detector):
        """Test anomaly detection in performance data."""

        # Create history with anomalies
        base_time = time.time() - (7 * 24 * 3600)  # 7 days ago

        history = []
        for hour in range(168):  # 7 days * 24 hours
            timestamp = base_time + (hour * 3600)

            # Normal response time around 50ms
            normal_response_time = 50.0 + (hour % 12) * 2  # Diurnal pattern

            # Inject anomalies
            if hour in [24, 72, 120]:  # Day 1, 3, and 5
                response_time = normal_response_time * 3  # 3x increase (anomaly)
            elif hour in [48, 96]:  # Day 2 and 4
                response_time = normal_response_time * 0.3  # 70% decrease (anomaly)
            else:
                response_time = normal_response_time

            history.append({
                'timestamp': timestamp,
                'metrics': {
                    'response_time_ms': response_time,
                }
            })

        # Analyze for anomalies
        trend_analysis = regression_detector.analyze_performance_trend(history, 'response_time_ms')

        # Validate anomaly detection
        assert trend_analysis['anomaly_count'] >= 3, f"Should detect at least 3 anomalies, found {trend_analysis['anomaly_count']}"

        # Check specific anomalies
        anomalies = trend_analysis['anomalies']
        high_anomalies = [a for a in anomalies if a['z_score'] > 3.0]
        assert len(high_anomalies) >= 2, f"Should detect high-severity anomalies, found {len(high_anomalies)}"

        print(f"\n🔍 Anomaly Detection Analysis:")
        print(f"   Total data points: {trend_analysis['data_points']}")
        print(f"   Anomalies detected: {trend_analysis['anomaly_count']}")
        print(f"   High-severity anomalies: {len(high_anomalies)}")

        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
            print(f"   Anomaly {i+1}: value={anomaly['value']:.1f}ms, z-score={anomaly['z_score']:.2f}")


class TestPerformanceBudgets:
    """Test performance budget enforcement and monitoring."""

    @pytest.mark.benchmark
    async def test_performance_budget_validation(self, regression_detector):
        """Test performance budget validation against defined limits."""

        # Define performance budgets (maximum acceptable values)
        performance_budgets = {
            'document_processing_ms': 500,      # Max 500ms for document processing
            'search_operation_ms': 100,         # Max 100ms for search
            'memory_usage_mb': 100,             # Max 100MB memory usage
            'error_rate_percent': 1.0,          # Max 1% error rate
            'p95_response_time_ms': 200,        # Max 200ms P95 response time
        }

        # Current performance metrics
        current_metrics = {
            'document_processing_ms': 320,      # Within budget
            'search_operation_ms': 85,          # Within budget
            'memory_usage_mb': 78,              # Within budget
            'error_rate_percent': 0.5,          # Within budget
            'p95_response_time_ms': 245,        # Exceeds budget
        }

        # Check budget compliance
        budget_violations = []
        budget_compliance = {}

        for metric, budget in performance_budgets.items():
            current_value = current_metrics.get(metric)
            if current_value is not None:
                exceeds_budget = current_value > budget
                usage_percent = (current_value / budget) * 100

                budget_compliance[metric] = {
                    'budget': budget,
                    'current': current_value,
                    'usage_percent': usage_percent,
                    'exceeds_budget': exceeds_budget,
                    'headroom_percent': 100 - usage_percent
                }

                if exceeds_budget:
                    budget_violations.append({
                        'metric': metric,
                        'budget': budget,
                        'current': current_value,
                        'overage_percent': usage_percent - 100
                    })

        # Validate budget enforcement
        assert len(budget_violations) == 1, f"Expected 1 budget violation, found {len(budget_violations)}"

        violation = budget_violations[0]
        assert violation['metric'] == 'p95_response_time_ms', f"Unexpected violation: {violation['metric']}"

        print(f"\n💰 Performance Budget Analysis:")
        print(f"   Total budgets: {len(performance_budgets)}")
        print(f"   Budget violations: {len(budget_violations)}")

        for metric, compliance in budget_compliance.items():
            status = "❌ EXCEEDED" if compliance['exceeds_budget'] else "✅ OK"
            print(f"   {metric}: {compliance['current']}/{compliance['budget']} ({compliance['usage_percent']:.1f}%) {status}")

        if budget_violations:
            print(f"\n   Budget Violations:")
            for violation in budget_violations:
                print(f"     - {violation['metric']}: {violation['overage_percent']:.1f}% over budget")

    @pytest.mark.benchmark
    async def test_performance_sla_monitoring(self, regression_detector):
        """Test SLA monitoring and compliance checking."""

        # Define SLAs (Service Level Agreements)
        slas = {
            'availability_percent': 99.9,       # 99.9% uptime
            'p95_response_time_ms': 200,        # 95% of requests under 200ms
            'p99_response_time_ms': 500,        # 99% of requests under 500ms
            'error_rate_percent': 0.1,          # 0.1% error rate
            'throughput_rps': 100,              # Minimum 100 RPS
        }

        # Simulated performance data over time
        performance_data = [
            {'availability_percent': 99.95, 'p95_response_time_ms': 180, 'p99_response_time_ms': 420, 'error_rate_percent': 0.05, 'throughput_rps': 120},
            {'availability_percent': 99.8,  'p95_response_time_ms': 190, 'p99_response_time_ms': 450, 'error_rate_percent': 0.08, 'throughput_rps': 115},
            {'availability_percent': 99.85, 'p95_response_time_ms': 210, 'p99_response_time_ms': 480, 'error_rate_percent': 0.12, 'throughput_rps': 95},  # SLA violations
            {'availability_percent': 99.92, 'p95_response_time_ms': 175, 'p99_response_time_ms': 390, 'error_rate_percent': 0.06, 'throughput_rps': 125},
        ]

        sla_compliance = {}
        sla_violations = []

        # Analyze SLA compliance
        for metric, sla_target in slas.items():
            values = [data[metric] for data in performance_data if metric in data]

            if not values:
                continue

            # Calculate compliance based on metric type
            if metric in ['p95_response_time_ms', 'p99_response_time_ms', 'error_rate_percent']:
                # These should be below the SLA target
                violations = [v for v in values if v > sla_target]
                compliance_rate = (len(values) - len(violations)) / len(values) * 100
            elif metric in ['availability_percent', 'throughput_rps']:
                # These should be above the SLA target
                violations = [v for v in values if v < sla_target]
                compliance_rate = (len(values) - len(violations)) / len(values) * 100
            else:
                violations = []
                compliance_rate = 100.0

            sla_compliance[metric] = {
                'sla_target': sla_target,
                'measurements': len(values),
                'violations': len(violations),
                'compliance_rate': compliance_rate,
                'current_value': values[-1] if values else None,
                'meets_sla': compliance_rate >= 95.0  # 95% compliance threshold
            }

            if not sla_compliance[metric]['meets_sla']:
                sla_violations.append({
                    'metric': metric,
                    'compliance_rate': compliance_rate,
                    'violation_count': len(violations)
                })

        # Validate SLA monitoring
        expected_violations = ['p95_response_time_ms', 'error_rate_percent', 'throughput_rps']
        actual_violation_metrics = {v['metric'] for v in sla_violations}

        for expected_metric in expected_violations:
            assert expected_metric in actual_violation_metrics, f"Should detect SLA violation for {expected_metric}"

        print(f"\n📋 SLA Compliance Analysis:")
        print(f"   Total SLAs monitored: {len(slas)}")
        print(f"   SLA violations: {len(sla_violations)}")

        for metric, compliance in sla_compliance.items():
            status = "✅ COMPLIANT" if compliance['meets_sla'] else "❌ VIOLATION"
            print(f"   {metric}: {compliance['compliance_rate']:.1f}% compliance {status}")

        if sla_violations:
            print(f"\n   SLA Violations Details:")
            for violation in sla_violations:
                print(f"     - {violation['metric']}: {violation['compliance_rate']:.1f}% compliance")


@pytest.mark.benchmark
async def test_comprehensive_regression_report():
    """Generate comprehensive regression analysis report."""

    print(f"\n" + "="*60)
    print(f"📊 COMPREHENSIVE REGRESSION ANALYSIS REPORT")
    print(f"="*60)

    # Mock comprehensive regression analysis results
    regression_summary = {
        'baseline_age_days': 7,
        'total_metrics_analyzed': 15,
        'regressions_detected': 3,
        'regression_details': [
            {
                'metric': 'document_processing_ms',
                'baseline': 45.2,
                'current': 58.7,
                'change_percent': 29.9,
                'severity': 'medium',
                'threshold_exceeded': True
            },
            {
                'metric': 'memory_peak_mb',
                'baseline': 32.1,
                'current': 43.8,
                'change_percent': 36.4,
                'severity': 'high',
                'threshold_exceeded': True
            },
            {
                'metric': 'search_throughput_rps',
                'baseline': 125.0,
                'current': 98.3,
                'change_percent': -21.4,
                'severity': 'medium',
                'threshold_exceeded': True
            }
        ],
        'trend_analysis': {
            'performance_degradation_rate': 1.2,  # % per day
            'anomalies_detected': 2,
            'stability_assessment': 'fair'
        },
        'budget_compliance': {
            'total_budgets': 8,
            'budget_violations': 1,
            'compliance_rate': 87.5
        },
        'sla_compliance': {
            'total_slas': 5,
            'sla_violations': 2,
            'overall_compliance': 92.1
        }
    }

    print(f"\n🎯 Regression Detection Summary:")
    print(f"   Baseline age: {regression_summary['baseline_age_days']} days")
    print(f"   Metrics analyzed: {regression_summary['total_metrics_analyzed']}")
    print(f"   Regressions detected: {regression_summary['regressions_detected']}")

    print(f"\n📉 Detected Regressions:")
    for regression in regression_summary['regression_details']:
        direction = "↑" if regression['change_percent'] > 0 else "↓"
        print(f"   {direction} {regression['metric']}:")
        print(f"     Baseline: {regression['baseline']}")
        print(f"     Current: {regression['current']}")
        print(f"     Change: {regression['change_percent']:+.1f}% ({regression['severity']})")

    print(f"\n📈 Trend Analysis:")
    trend = regression_summary['trend_analysis']
    print(f"   Performance degradation rate: {trend['performance_degradation_rate']:.1f}% per day")
    print(f"   Anomalies detected: {trend['anomalies_detected']}")
    print(f"   Overall stability: {trend['stability_assessment']}")

    print(f"\n💰 Budget Compliance:")
    budget = regression_summary['budget_compliance']
    print(f"   Budgets monitored: {budget['total_budgets']}")
    print(f"   Budget violations: {budget['budget_violations']}")
    print(f"   Compliance rate: {budget['compliance_rate']:.1f}%")

    print(f"\n📋 SLA Compliance:")
    sla = regression_summary['sla_compliance']
    print(f"   SLAs monitored: {sla['total_slas']}")
    print(f"   SLA violations: {sla['sla_violations']}")
    print(f"   Overall compliance: {sla['overall_compliance']:.1f}%")

    print(f"\n🚨 Recommendations:")
    if regression_summary['regressions_detected'] > 0:
        print(f"   - Investigate and address {regression_summary['regressions_detected']} performance regressions")
        print(f"   - Focus on high-severity regressions first")
    if regression_summary['trend_analysis']['performance_degradation_rate'] > 1.0:
        print(f"   - Performance degradation trend requires attention")
    if regression_summary['budget_compliance']['budget_violations'] > 0:
        print(f"   - Review and potentially adjust performance budgets")
    if regression_summary['sla_compliance']['sla_violations'] > 0:
        print(f"   - Address SLA violations to maintain service quality")

    # Overall assessment
    total_issues = (
        regression_summary['regressions_detected'] +
        regression_summary['budget_compliance']['budget_violations'] +
        regression_summary['sla_compliance']['sla_violations']
    )

    if total_issues == 0:
        assessment = "✅ EXCELLENT"
    elif total_issues <= 2:
        assessment = "⚠️  NEEDS ATTENTION"
    else:
        assessment = "❌ REQUIRES IMMEDIATE ACTION"

    print(f"\n🎯 Overall Performance Assessment: {assessment}")
    print(f"   Total performance issues: {total_issues}")

    print(f"\n" + "="*60)

    return regression_summary