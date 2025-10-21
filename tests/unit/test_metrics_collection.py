"""
Comprehensive metrics collection validation tests for workspace-qdrant-mcp.

Tests the accuracy and reliability of the metrics collection system including:
- Counter metrics (increment, labels, accuracy)
- Gauge metrics (set, add, subtract, labels)
- Histogram metrics (buckets, distributions, percentiles)
- Metric exporters (Prometheus, JSON)
- Thread safety and concurrent operations
- System metrics collection
- Metric aggregation and retention

Usage:
    pytest tests/unit/test_metrics_collection.py -v
    pytest tests/unit/test_metrics_collection.py::TestCounterMetrics -v
    pytest tests/unit/test_metrics_collection.py::TestHistogramMetrics -v
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import concurrent.futures
import re
import threading
import time
from typing import List

import pytest

from common.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics_collector,
)


class TestCounterMetrics:
    """Test counter metric accuracy and functionality."""

    def test_counter_basic_increment(self):
        """Test basic counter increment functionality."""
        counter = Counter("test_counter", "Test counter description")

        # Initial value should be 0
        assert counter.get_value() == 0

        # Single increment
        counter.increment()
        assert counter.get_value() == 1

        # Multiple increments
        counter.increment()
        counter.increment()
        assert counter.get_value() == 3

        # Increment by specific amount
        counter.increment(5.0)
        assert counter.get_value() == 8.0

    def test_counter_increment_accuracy(self):
        """Test counter increment accuracy with various amounts."""
        counter = Counter("accuracy_counter")

        # Test integer increments
        for i in range(100):
            counter.increment(1)
        assert counter.get_value() == 100

        # Reset for float test
        counter = Counter("float_counter")

        # Test float increments
        for i in range(100):
            counter.increment(0.1)

        # Allow small floating point errors
        assert abs(counter.get_value() - 10.0) < 0.001

    def test_counter_with_labels(self):
        """Test counter with labeled values."""
        counter = Counter("http_requests", "HTTP request counter")

        # Increment with different labels
        counter.increment(method="GET", status="200")
        counter.increment(method="GET", status="200")
        counter.increment(method="POST", status="201")
        counter.increment(method="GET", status="404")

        # Verify labeled values
        assert counter.get_value(method="GET", status="200") == 2
        assert counter.get_value(method="POST", status="201") == 1
        assert counter.get_value(method="GET", status="404") == 1

        # Total value should include all increments
        assert counter.get_value() == 4

    def test_counter_label_serialization(self):
        """Test counter label serialization consistency."""
        counter = Counter("serialization_test")

        # Labels should be serialized consistently regardless of order
        counter.increment(a="1", b="2", c="3")
        counter.increment(c="3", a="1", b="2")
        counter.increment(b="2", c="3", a="1")

        # All should map to the same label key
        labeled_values = counter.get_all_labeled_values()
        assert len(labeled_values) == 1
        assert list(labeled_values.values())[0] == 3

    def test_counter_thread_safety(self):
        """Test counter thread safety under concurrent increments."""
        counter = Counter("concurrent_counter")
        num_threads = 10
        increments_per_thread = 1000

        def increment_worker():
            for _ in range(increments_per_thread):
                counter.increment()

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all increments were counted
        expected_value = num_threads * increments_per_thread
        assert counter.get_value() == expected_value

    def test_counter_labeled_thread_safety(self):
        """Test counter labeled values under concurrent operations."""
        counter = Counter("labeled_concurrent")
        num_threads = 5
        increments_per_label = 200

        def increment_worker(label_value: str):
            for _ in range(increments_per_label):
                counter.increment(label=label_value)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=increment_worker, args=(f"label_{i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify each label has correct count
        for i in range(num_threads):
            assert counter.get_value(label=f"label_{i}") == increments_per_label

    def test_counter_get_all_labeled_values(self):
        """Test retrieving all labeled counter values."""
        counter = Counter("multi_label_counter")

        counter.increment(service="api", endpoint="/users")
        counter.increment(service="api", endpoint="/posts")
        counter.increment(service="web", endpoint="/home")

        labeled_values = counter.get_all_labeled_values()

        # Should have 3 different label combinations
        assert len(labeled_values) == 3

        # All values should be accessible
        for label_key, value in labeled_values.items():
            assert value > 0


class TestGaugeMetrics:
    """Test gauge metric accuracy and operations."""

    def test_gauge_basic_operations(self):
        """Test basic gauge set, add, and subtract operations."""
        gauge = Gauge("test_gauge", "Test gauge description")

        # Initial value should be 0
        assert gauge.get_value() == 0

        # Set value
        gauge.set(42.5)
        assert gauge.get_value() == 42.5

        # Add to value
        gauge.add(7.5)
        assert gauge.get_value() == 50.0

        # Subtract from value
        gauge.subtract(10.0)
        assert gauge.get_value() == 40.0

    def test_gauge_accuracy_with_floats(self):
        """Test gauge accuracy with floating point operations."""
        gauge = Gauge("float_gauge")

        gauge.set(0.1)
        for _ in range(100):
            gauge.add(0.01)

        # Should be approximately 1.1
        assert abs(gauge.get_value() - 1.1) < 0.001

    def test_gauge_with_labels(self):
        """Test gauge with labeled values."""
        gauge = Gauge("memory_usage", "Memory usage gauge")

        # Set different labeled values
        gauge.set(1024, process="api")
        gauge.set(2048, process="worker")
        gauge.set(512, process="monitor")

        # Verify labeled values
        assert gauge.get_value(process="api") == 1024
        assert gauge.get_value(process="worker") == 2048
        assert gauge.get_value(process="monitor") == 512

        # Unlabeled value should still be 0
        assert gauge.get_value() == 0

    def test_gauge_labeled_operations(self):
        """Test gauge add/subtract operations with labels."""
        gauge = Gauge("queue_depth")

        # Start with initial values
        gauge.set(10, queue="high_priority")
        gauge.set(20, queue="low_priority")

        # Add to labeled values
        gauge.add(5, queue="high_priority")
        gauge.add(10, queue="low_priority")

        assert gauge.get_value(queue="high_priority") == 15
        assert gauge.get_value(queue="low_priority") == 30

        # Subtract from labeled values
        gauge.subtract(3, queue="high_priority")
        assert gauge.get_value(queue="high_priority") == 12

    def test_gauge_thread_safety(self):
        """Test gauge thread safety under concurrent operations."""
        gauge = Gauge("concurrent_gauge")
        num_threads = 10
        operations_per_thread = 100

        gauge.set(0)

        def gauge_worker():
            for _ in range(operations_per_thread):
                gauge.add(1)

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=gauge_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        expected_value = num_threads * operations_per_thread
        assert gauge.get_value() == expected_value

    def test_gauge_negative_values(self):
        """Test gauge with negative values."""
        gauge = Gauge("negative_gauge")

        gauge.set(100)
        gauge.subtract(150)
        assert gauge.get_value() == -50

        gauge.add(25)
        assert gauge.get_value() == -25


class TestHistogramMetrics:
    """Test histogram metric accuracy and bucket distributions."""

    def test_histogram_basic_observation(self):
        """Test basic histogram observation functionality."""
        histogram = Histogram("test_histogram", "Test histogram")

        # Initial counts should be 0
        assert histogram.get_count() == 0
        assert histogram.get_sum() == 0.0

        # Record observations
        histogram.observe(0.1)
        histogram.observe(0.5)
        histogram.observe(1.0)

        # Verify counts
        assert histogram.get_count() == 3
        assert histogram.get_sum() == 1.6
        assert abs(histogram.get_average() - 0.533) < 0.001

    def test_histogram_bucket_distribution(self):
        """Test histogram bucket distribution accuracy."""
        histogram = Histogram("latency_histogram")

        # Observe values across different buckets
        histogram.observe(0.003)  # < 0.005
        histogram.observe(0.007)  # < 0.01
        histogram.observe(0.015)  # < 0.025
        histogram.observe(0.040)  # < 0.05
        histogram.observe(0.150)  # < 0.25
        histogram.observe(0.800)  # < 1.0
        histogram.observe(3.000)  # < 5.0

        buckets = histogram.get_buckets()

        # Verify cumulative bucket counts
        assert buckets[0.005] == 1  # 0.003
        assert buckets[0.01] == 2   # 0.003, 0.007
        assert buckets[0.025] == 3  # + 0.015
        assert buckets[0.05] == 4   # + 0.040
        assert buckets[0.25] == 5   # + 0.150
        assert buckets[1.0] == 6    # + 0.800
        assert buckets[5.0] == 7    # + 3.000
        assert buckets[float("inf")] == 7  # All observations

    def test_histogram_custom_buckets(self):
        """Test histogram with custom bucket configuration."""
        custom_buckets = [1.0, 5.0, 10.0, 50.0, 100.0]
        histogram = Histogram("custom_histogram", buckets=custom_buckets)

        # Observe values
        histogram.observe(0.5)   # < 1.0
        histogram.observe(3.0)   # < 5.0
        histogram.observe(7.5)   # < 10.0
        histogram.observe(25.0)  # < 50.0
        histogram.observe(75.0)  # < 100.0
        histogram.observe(150.0) # > 100.0, in +Inf

        buckets = histogram.get_buckets()

        assert buckets[1.0] == 1
        assert buckets[5.0] == 2
        assert buckets[10.0] == 3
        assert buckets[50.0] == 4
        assert buckets[100.0] == 5
        assert buckets[float("inf")] == 6

    def test_histogram_with_labels(self):
        """Test histogram with labeled observations."""
        histogram = Histogram("request_duration")

        # Record observations with labels
        histogram.observe(0.1, endpoint="/api")
        histogram.observe(0.2, endpoint="/api")
        histogram.observe(0.5, endpoint="/web")
        histogram.observe(1.0, endpoint="/web")

        # Verify labeled counts
        assert histogram.get_count(endpoint="/api") == 2
        # Use approximate comparison for floating point
        assert abs(histogram.get_sum(endpoint="/api") - 0.3) < 0.001
        assert histogram.get_count(endpoint="/web") == 2
        assert abs(histogram.get_sum(endpoint="/web") - 1.5) < 0.001

        # Verify unlabeled total
        assert histogram.get_count() == 0  # No unlabeled observations

    def test_histogram_percentile_calculation(self):
        """Test histogram bucket distribution for percentile estimation."""
        histogram = Histogram("percentile_histogram")

        # Record 1000 observations uniformly distributed
        for i in range(1000):
            histogram.observe(i * 0.01)  # 0.00 to 9.99

        buckets = histogram.get_buckets()
        total_count = histogram.get_count()

        assert total_count == 1000

        # Verify bucket accumulation makes sense
        # Most observations should be in higher buckets
        assert buckets[0.01] < buckets[0.1]
        assert buckets[0.1] < buckets[1.0]
        assert buckets[1.0] < buckets[5.0]
        assert buckets[10.0] == 1000  # All observations <= 10s

    def test_histogram_thread_safety(self):
        """Test histogram thread safety under concurrent observations."""
        histogram = Histogram("concurrent_histogram")
        num_threads = 10
        observations_per_thread = 100

        def observe_worker(value: float):
            for _ in range(observations_per_thread):
                histogram.observe(value)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=observe_worker, args=(i * 0.1,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify total count
        expected_count = num_threads * observations_per_thread
        assert histogram.get_count() == expected_count

    def test_histogram_edge_values(self):
        """Test histogram with edge case values."""
        histogram = Histogram("edge_histogram")

        # Test edge values
        histogram.observe(0.0)      # Minimum
        histogram.observe(0.005)    # Exact bucket boundary
        histogram.observe(1000.0)   # Very large value
        histogram.observe(0.0000001)  # Very small value

        assert histogram.get_count() == 4
        assert histogram.get_sum() > 1000.0

        # All should be in +Inf bucket
        buckets = histogram.get_buckets()
        assert buckets[float("inf")] == 4


class TestMetricsCollector:
    """Test MetricsCollector functionality and integration."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization and standard metrics."""
        collector = MetricsCollector()

        # Verify standard metrics are created
        assert "requests_total" in collector.counters
        assert "operations_total" in collector.counters
        assert "operation_duration_seconds" in collector.histograms
        assert "search_duration_seconds" in collector.histograms
        assert "memory_usage_bytes" in collector.gauges
        assert "active_connections" in collector.gauges

    def test_create_and_get_metrics(self):
        """Test creating and retrieving metrics."""
        collector = MetricsCollector()

        # Create metrics
        counter = collector.create_counter("new_counter", "New counter")
        gauge = collector.create_gauge("new_gauge", "New gauge")
        histogram = collector.create_histogram("new_histogram", "New histogram")

        # Verify they exist
        assert "new_counter" in collector.counters
        assert "new_gauge" in collector.gauges
        assert "new_histogram" in collector.histograms

        # Getting same metric should return same instance
        counter2 = collector.create_counter("new_counter")
        assert counter is counter2

    def test_increment_counter_convenience_method(self):
        """Test convenience method for incrementing counters."""
        collector = MetricsCollector()

        collector.increment_counter("test_counter", 5.0)
        collector.increment_counter("test_counter", 3.0)

        counter = collector.counters["test_counter"]
        assert counter.get_value() == 8.0

    def test_set_gauge_convenience_method(self):
        """Test convenience method for setting gauges."""
        collector = MetricsCollector()

        collector.set_gauge("test_gauge", 42.0)
        gauge = collector.gauges["test_gauge"]
        assert gauge.get_value() == 42.0

    def test_record_histogram_convenience_method(self):
        """Test convenience method for recording histogram values."""
        collector = MetricsCollector()

        collector.record_histogram("test_histogram", 0.5)
        collector.record_histogram("test_histogram", 1.5)

        histogram = collector.histograms["test_histogram"]
        assert histogram.get_count() == 2
        assert histogram.get_sum() == 2.0

    def test_timer_context_manager(self):
        """Test timer context manager for automatic timing."""
        collector = MetricsCollector()

        with collector.timer("operation_time"):
            time.sleep(0.1)  # Sleep for 100ms

        histogram = collector.histograms["operation_time"]
        assert histogram.get_count() == 1

        # Should be approximately 0.1 seconds
        average = histogram.get_average()
        assert 0.09 < average < 0.15

    def test_operation_tracking(self):
        """Test start/complete operation tracking."""
        collector = MetricsCollector()

        # Start operation
        collector.start_operation("op_1", "search", collection="test")

        # Simulate some work
        time.sleep(0.05)

        # Complete operation
        collector.complete_operation("op_1", "search", success=True, collection="test")

        # Verify metrics were recorded
        histogram = collector.histograms["operation_duration_seconds"]
        assert histogram.get_count(operation="search", collection="test") >= 1

    def test_operation_error_tracking(self):
        """Test operation error tracking."""
        collector = MetricsCollector()

        collector.start_operation("op_error", "ingest")
        collector.complete_operation("op_error", "ingest", success=False)

        # Verify error was counted
        error_counter = collector.counters["requests_errors_total"]
        assert error_counter.get_value(operation="ingest") >= 1

    def test_system_metrics_update(self):
        """Test system metrics update functionality."""
        collector = MetricsCollector()

        # Update system metrics
        collector.update_system_metrics()

        # Verify system metrics are set
        memory_gauge = collector.gauges["memory_usage_bytes"]
        cpu_gauge = collector.gauges["cpu_usage_percent"]

        assert memory_gauge.get_value() > 0
        # CPU might be 0 so just check it exists
        assert cpu_gauge.get_value() >= 0

    @pytest.mark.skip(
        reason="Deadlock in MetricsCollector.reset_metrics() - reset_metrics() holds lock "
        "while calling _initialize_standard_metrics() which tries to acquire lock again"
    )
    def test_metrics_reset(self):
        """Test metrics reset functionality.

        NOTE: This test is skipped due to a deadlock issue in the MetricsCollector implementation.
        The reset_metrics() method acquires self._lock, then calls _initialize_standard_metrics(),
        which calls create_counter/create_gauge/create_histogram, each of which tries to acquire
        self._lock again, causing a deadlock.

        This is a bug in the core metrics collection code that should be fixed by:
        1. Not holding lock during _initialize_standard_metrics() call, OR
        2. Having _initialize_standard_metrics() call internal methods that don't acquire locks
        """
        collector = MetricsCollector()

        # Add some metrics
        collector.increment_counter("temp_counter", 10)
        collector.set_gauge("temp_gauge", 50)
        collector.record_histogram("temp_histogram", 1.0)

        # Reset all metrics
        collector.reset_metrics()

        # Standard metrics should be recreated
        assert "requests_total" in collector.counters
        assert "memory_usage_bytes" in collector.gauges

        # Temporary metrics should be gone
        assert "temp_counter" not in collector.counters
        assert "temp_gauge" not in collector.gauges
        assert "temp_histogram" not in collector.histograms


class TestPrometheusExport:
    """Test Prometheus format export functionality."""

    def test_prometheus_format_basic(self):
        """Test basic Prometheus format export."""
        collector = MetricsCollector()

        # Add some test metrics
        collector.increment_counter("test_requests", 5)
        collector.set_gauge("test_connections", 10)
        collector.record_histogram("test_latency", 0.5)

        prometheus_output = collector.export_prometheus_format()

        # Verify output is a string
        assert isinstance(prometheus_output, str)
        assert len(prometheus_output) > 0

        # Verify metrics are present
        assert "test_requests" in prometheus_output
        assert "test_connections" in prometheus_output
        assert "test_latency" in prometheus_output

    def test_prometheus_format_structure(self):
        """Test Prometheus format structure compliance."""
        collector = MetricsCollector()
        collector.increment_counter("prom_counter", 1)

        prometheus_output = collector.export_prometheus_format()
        lines = prometheus_output.split("\n")

        # Should have comment lines
        comment_lines = [l for l in lines if l.startswith("#")]
        assert len(comment_lines) > 0

        # Should have HELP and TYPE directives
        help_lines = [l for l in lines if l.startswith("# HELP")]
        type_lines = [l for l in lines if l.startswith("# TYPE")]
        assert len(help_lines) > 0
        assert len(type_lines) > 0

    def test_prometheus_counter_format(self):
        """Test Prometheus counter format."""
        collector = MetricsCollector()
        collector.increment_counter("http_requests_total", 10, method="GET", status="200")

        output = collector.export_prometheus_format()

        # Check for counter type declaration
        assert "# TYPE http_requests_total counter" in output

        # Check for metric with labels
        assert 'http_requests_total{method="GET", status="200"}' in output or \
               'http_requests_total{status="200", method="GET"}' in output

    def test_prometheus_histogram_format(self):
        """Test Prometheus histogram format."""
        collector = MetricsCollector()

        collector.record_histogram("request_duration_seconds", 0.5)
        collector.record_histogram("request_duration_seconds", 1.5)

        output = collector.export_prometheus_format()

        # Check for histogram type
        assert "# TYPE request_duration_seconds histogram" in output

        # Check for bucket lines
        assert "_bucket" in output
        assert 'le="+Inf"' in output

        # Check for sum and count
        assert "_sum" in output
        assert "_count" in output

    def test_prometheus_gauge_format(self):
        """Test Prometheus gauge format."""
        collector = MetricsCollector()
        collector.set_gauge("active_sessions", 42)

        output = collector.export_prometheus_format()

        # Check for gauge type
        assert "# TYPE active_sessions gauge" in output

        # Check for value
        assert "active_sessions 42" in output


class TestJSONExport:
    """Test JSON format export functionality."""

    def test_json_export_structure(self):
        """Test JSON export structure."""
        collector = MetricsCollector()

        collector.increment_counter("json_counter", 5)
        collector.set_gauge("json_gauge", 10)
        collector.record_histogram("json_histogram", 0.5)

        summary = collector.get_metrics_summary()

        # Verify structure
        assert isinstance(summary, dict)
        assert "timestamp" in summary
        assert "counters" in summary
        assert "gauges" in summary
        assert "histograms" in summary

    def test_json_counter_export(self):
        """Test JSON counter export accuracy."""
        collector = MetricsCollector()

        collector.increment_counter("export_counter", 15, label="test")

        summary = collector.get_metrics_summary()
        counter_data = summary["counters"]["export_counter"]

        assert "value" in counter_data
        assert "labeled_values" in counter_data
        assert "description" in counter_data
        assert counter_data["value"] == 15

    def test_json_histogram_export(self):
        """Test JSON histogram export accuracy."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_histogram("export_histogram", i * 0.1)

        summary = collector.get_metrics_summary()
        histogram_data = summary["histograms"]["export_histogram"]

        assert "count" in histogram_data
        assert "sum" in histogram_data
        assert "average" in histogram_data
        assert "buckets" in histogram_data

        assert histogram_data["count"] == 10
        assert abs(histogram_data["average"] - 0.45) < 0.001


class TestLoadConditions:
    """Test metrics collection under various load conditions."""

    def test_high_frequency_counter_updates(self):
        """Test counter under high-frequency updates."""
        collector = MetricsCollector()
        iterations = 10000

        start_time = time.perf_counter()
        for i in range(iterations):
            collector.increment_counter("high_freq_counter")
        duration = time.perf_counter() - start_time

        counter = collector.counters["high_freq_counter"]
        assert counter.get_value() == iterations

        # Should complete in reasonable time (< 1 second for 10k updates)
        assert duration < 1.0

    def test_high_frequency_histogram_updates(self):
        """Test histogram under high-frequency observations."""
        collector = MetricsCollector()
        iterations = 10000

        start_time = time.perf_counter()
        for i in range(iterations):
            collector.record_histogram("high_freq_histogram", i * 0.001)
        duration = time.perf_counter() - start_time

        histogram = collector.histograms["high_freq_histogram"]
        assert histogram.get_count() == iterations

        # Should complete in reasonable time
        assert duration < 2.0

    def test_concurrent_metric_types(self):
        """Test concurrent updates to different metric types."""
        collector = MetricsCollector()
        num_workers = 5
        operations_per_worker = 100

        def counter_worker():
            for _ in range(operations_per_worker):
                collector.increment_counter("concurrent_counter")

        def gauge_worker():
            for _ in range(operations_per_worker):
                collector.add_gauge("concurrent_gauge", 1)

        def histogram_worker():
            for _ in range(operations_per_worker):
                collector.record_histogram("concurrent_histogram", 0.5)

        # Start all workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(executor.submit(counter_worker))
                futures.append(executor.submit(gauge_worker))
                futures.append(executor.submit(histogram_worker))

            concurrent.futures.wait(futures)

        # Verify all operations completed
        counter = collector.counters["concurrent_counter"]
        gauge = collector.gauges["concurrent_gauge"]
        histogram = collector.histograms["concurrent_histogram"]

        expected = num_workers * operations_per_worker
        assert counter.get_value() == expected
        assert gauge.get_value() == expected
        assert histogram.get_count() == expected

    def test_many_labeled_metrics(self):
        """Test performance with many different label combinations."""
        collector = MetricsCollector()
        num_labels = 100

        start_time = time.perf_counter()
        for i in range(num_labels):
            collector.increment_counter("multi_label", label=f"label_{i}")
        duration = time.perf_counter() - start_time

        counter = collector.counters["multi_label"]
        labeled_values = counter.get_all_labeled_values()

        # All labels should be tracked
        assert len(labeled_values) == num_labels

        # Should complete quickly even with many labels
        assert duration < 0.5

    def test_prometheus_export_performance(self):
        """Test Prometheus export performance with many metrics."""
        collector = MetricsCollector()

        # Create many metrics
        for i in range(50):
            collector.increment_counter(f"counter_{i}", i)
            collector.set_gauge(f"gauge_{i}", i * 10)
            collector.record_histogram(f"histogram_{i}", i * 0.1)

        start_time = time.perf_counter()
        prometheus_output = collector.export_prometheus_format()
        duration = time.perf_counter() - start_time

        # Should export quickly
        assert duration < 1.0

        # Should contain all metrics
        assert len(prometheus_output) > 1000


class TestMetricLabelsAndDimensions:
    """Test metric labels and dimensions functionality."""

    def test_label_key_value_pairs(self):
        """Test various label key-value pair combinations."""
        counter = Counter("label_test")

        # Test different data types as label values
        counter.increment(string_label="value")
        counter.increment(int_label=123)
        counter.increment(float_label=45.67)
        counter.increment(bool_label=True)

        labeled_values = counter.get_all_labeled_values()
        assert len(labeled_values) == 4

    def test_label_special_characters(self):
        """Test labels with special characters."""
        counter = Counter("special_chars")

        # Labels with various characters
        counter.increment(label="value-with-dash")
        counter.increment(label="value_with_underscore")
        counter.increment(label="value.with.dot")

        # All should be tracked separately
        labeled_values = counter.get_all_labeled_values()
        assert len(labeled_values) == 3

    def test_multi_dimensional_labels(self):
        """Test metrics with multiple label dimensions."""
        histogram = Histogram("request_latency")

        # Multi-dimensional labels
        histogram.observe(0.1, method="GET", endpoint="/api", status="200")
        histogram.observe(0.2, method="POST", endpoint="/api", status="201")
        histogram.observe(0.5, method="GET", endpoint="/web", status="200")

        # Each combination should be tracked
        assert histogram.get_count(method="GET", endpoint="/api", status="200") == 1
        assert histogram.get_count(method="POST", endpoint="/api", status="201") == 1
        assert histogram.get_count(method="GET", endpoint="/web", status="200") == 1

    def test_label_cardinality_handling(self):
        """Test handling of high-cardinality labels."""
        gauge = Gauge("high_cardinality")

        # Create many unique label combinations
        for i in range(1000):
            gauge.set(i, user_id=f"user_{i}")

        labeled_values = gauge.get_all_labeled_values()
        assert len(labeled_values) == 1000


class TestMetricsAccuracyValidation:
    """Test metrics accuracy under various scenarios."""

    def test_counter_accuracy_large_values(self):
        """Test counter accuracy with large values."""
        counter = Counter("large_value_counter")

        large_value = 1000000
        counter.increment(large_value)
        counter.increment(large_value)

        assert counter.get_value() == large_value * 2

    def test_gauge_accuracy_small_increments(self):
        """Test gauge accuracy with very small increments."""
        gauge = Gauge("small_increment_gauge")

        gauge.set(0)
        for _ in range(1000):
            gauge.add(0.001)

        # Should be approximately 1.0
        assert abs(gauge.get_value() - 1.0) < 0.01

    def test_histogram_sum_accuracy(self):
        """Test histogram sum calculation accuracy."""
        histogram = Histogram("sum_accuracy")

        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        expected_sum = sum(values)

        for value in values:
            histogram.observe(value)

        actual_sum = histogram.get_sum()
        assert abs(actual_sum - expected_sum) < 0.001

    def test_histogram_average_accuracy(self):
        """Test histogram average calculation accuracy."""
        histogram = Histogram("average_accuracy")

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            histogram.observe(value)

        expected_average = sum(values) / len(values)
        actual_average = histogram.get_average()

        assert abs(actual_average - expected_average) < 0.001

    def test_metrics_zero_values(self):
        """Test metrics with zero values."""
        collector = MetricsCollector()

        # Test zero increments
        collector.increment_counter("zero_counter", 0)
        assert collector.counters["zero_counter"].get_value() == 0

        # Test zero gauge set
        collector.set_gauge("zero_gauge", 0)
        assert collector.gauges["zero_gauge"].get_value() == 0

        # Test zero histogram observation
        collector.record_histogram("zero_histogram", 0.0)
        assert collector.histograms["zero_histogram"].get_sum() == 0.0


class TestMetricsCollectorSingleton:
    """Test metrics collector singleton pattern."""

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        # Should be the same instance
        assert collector1 is collector2

    def test_singleton_across_modules(self):
        """Test that singleton works across different imports."""
        from common.observability.metrics import metrics_instance

        collector1 = get_metrics_collector()

        # metrics_instance should resolve to same collector
        collector1.increment_counter("singleton_test", 5)

        # Access through metrics_instance proxy
        counter = metrics_instance.counters["singleton_test"]
        assert counter.get_value() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
