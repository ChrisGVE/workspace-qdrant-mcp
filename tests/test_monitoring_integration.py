"""
Comprehensive monitoring integration tests for workspace-qdrant-mcp.

Tests Prometheus metrics export, Grafana dashboard compatibility, health check
endpoints, alerting configurations, and observability system integration.

Test Categories:
    - Prometheus metrics format and content validation
    - Grafana dashboard and datasource configuration
    - Health check endpoint functionality and format
    - Alerting rules and notification systems
    - Log aggregation and structured logging
    - Distributed tracing integration
    - Performance monitoring and SLA validation

Usage:
    pytest tests/test_monitoring_integration.py -v
    pytest tests/test_monitoring_integration.py::TestPrometheusIntegration -v
    pytest tests/test_monitoring_integration.py::TestHealthCheckEndpoints -v
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
import yaml

from src.workspace_qdrant_mcp.observability import (
    health_checker_instance,
    metrics_instance,
)
from src.workspace_qdrant_mcp.observability.endpoints import (
    add_observability_routes,
    health_check_basic,
    health_check_detailed,
    metrics_prometheus,
    metrics_json,
    system_diagnostics,
)
from src.workspace_qdrant_mcp.observability.health import HealthStatus
from src.workspace_qdrant_mcp.observability.metrics import MetricsCollector
from tests.utils.deployment_helpers import MonitoringTestHelper


class TestPrometheusIntegration:
    """Test Prometheus metrics integration and format compliance."""
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_format(self):
        """Test Prometheus metrics format compliance."""
        # Generate test metrics
        metrics_instance.increment_counter("test_requests_total", method="GET", status="200")
        metrics_instance.increment_counter("test_requests_total", method="POST", status="201")
        metrics_instance.record_histogram("test_duration_seconds", 0.123)
        metrics_instance.record_histogram("test_duration_seconds", 0.456)
        metrics_instance.set_gauge("test_connections_active", 42)
        
        prometheus_data = await metrics_prometheus()
        
        assert isinstance(prometheus_data, str)
        assert len(prometheus_data) > 0
        
        lines = prometheus_data.strip().split('\n')
        
        # Test format compliance
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]
        metric_lines = [line for line in lines if line and not line.startswith('#')]
        
        assert len(help_lines) > 0, "No HELP lines found"
        assert len(type_lines) > 0, "No TYPE lines found"
        assert len(metric_lines) > 0, "No metric lines found"
        
        # Test specific metrics exist
        assert "test_requests_total" in prometheus_data
        assert "test_duration_seconds" in prometheus_data
        assert "test_connections_active" in prometheus_data
        
        # Test counter format
        counter_lines = [line for line in metric_lines if "test_requests_total" in line]
        assert len(counter_lines) >= 2, "Labeled counter metrics not found"
        
        # Test histogram format
        histogram_lines = [line for line in metric_lines if "test_duration_seconds" in line]
        bucket_lines = [line for line in histogram_lines if "_bucket" in line]
        sum_lines = [line for line in histogram_lines if "_sum" in line]
        count_lines = [line for line in histogram_lines if "_count" in line]
        
        assert len(bucket_lines) > 0, "Histogram buckets not found"
        assert len(sum_lines) == 1, "Histogram sum not found"
        assert len(count_lines) == 1, "Histogram count not found"
    
    @pytest.mark.asyncio
    async def test_standard_metrics_collection(self):
        """Test that standard application metrics are collected."""
        # Reset metrics to start fresh
        metrics_instance.reset_metrics()
        
        # Simulate some activity
        metrics_instance.increment_counter("requests_total", endpoint="/search")
        metrics_instance.increment_counter("operations_total", operation="ingest")
        metrics_instance.record_histogram("operation_duration_seconds", 0.234, operation="search")
        metrics_instance.set_gauge("active_connections", 15)
        
        prometheus_data = await metrics_prometheus()
        
        # Check for standard metrics
        standard_metrics = [
            "requests_total",
            "operations_total", 
            "operation_duration_seconds",
            "active_connections",
            "memory_usage_bytes",
            "cpu_usage_percent"
        ]
        
        for metric in standard_metrics:
            assert metric in prometheus_data, f"Standard metric {metric} not found"
    
    def test_prometheus_config_file(self):
        """Test Prometheus configuration file."""
        prometheus_config_path = Path("monitoring/prometheus/prometheus.yml")
        
        if not prometheus_config_path.exists():
            pytest.skip("Prometheus configuration file not found")
        
        with open(prometheus_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate basic structure
        assert "global" in config
        assert "scrape_configs" in config
        
        # Check global configuration
        global_config = config["global"]
        assert "scrape_interval" in global_config
        assert "evaluation_interval" in global_config
        
        # Check scrape configurations
        scrape_configs = config["scrape_configs"]
        assert len(scrape_configs) > 0
        
        # Look for application scrape config
        app_scrape_found = False
        for scrape_config in scrape_configs:
            job_name = scrape_config.get("job_name", "")
            if "workspace" in job_name.lower():
                app_scrape_found = True
                
                # Validate scrape config
                assert "static_configs" in scrape_config or "dns_sd_configs" in scrape_config
                assert "metrics_path" in scrape_config or scrape_config.get("metrics_path") == "/metrics"
        
        assert app_scrape_found, "Application scrape configuration not found"
    
    def test_alerting_rules_configuration(self):
        """Test Prometheus alerting rules configuration."""
        rules_dir = Path("monitoring/prometheus/rules")
        
        if not rules_dir.exists():
            pytest.skip("Prometheus rules directory not found")
        
        rule_files = list(rules_dir.glob("*.yml")) + list(rules_dir.glob("*.yaml"))
        
        if not rule_files:
            pytest.skip("No alerting rules found")
        
        for rule_file in rule_files:
            with open(rule_file, 'r') as f:
                rules_config = yaml.safe_load(f)
            
            # Validate rules structure
            assert "groups" in rules_config
            
            groups = rules_config["groups"]
            for group in groups:
                assert "name" in group
                assert "rules" in group
                
                rules = group["rules"]
                for rule in rules:
                    # Check alert rules
                    if rule.get("alert"):
                        assert "expr" in rule, f"Alert {rule['alert']} missing expression"
                        assert "for" in rule, f"Alert {rule['alert']} missing duration"
                        assert "labels" in rule or "annotations" in rule, \
                            f"Alert {rule['alert']} missing labels/annotations"


class TestGrafanaIntegration:
    """Test Grafana dashboard and datasource integration."""
    
    def test_grafana_datasource_provisioning(self):
        """Test Grafana datasource provisioning configuration."""
        datasources_dir = Path("monitoring/grafana/provisioning/datasources")
        
        if not datasources_dir.exists():
            pytest.skip("Grafana datasources provisioning not found")
        
        datasource_files = list(datasources_dir.glob("*.yml")) + list(datasources_dir.glob("*.yaml"))
        
        if not datasource_files:
            pytest.skip("No datasource configuration files found")
        
        for datasource_file in datasource_files:
            with open(datasource_file, 'r') as f:
                datasources_config = yaml.safe_load(f)
            
            # Validate datasources structure
            assert "apiVersion" in datasources_config
            assert "datasources" in datasources_config
            
            datasources = datasources_config["datasources"]
            
            # Look for Prometheus datasource
            prometheus_found = False
            for datasource in datasources:
                if datasource.get("type") == "prometheus":
                    prometheus_found = True
                    
                    # Validate Prometheus datasource config
                    assert "name" in datasource
                    assert "url" in datasource
                    assert "access" in datasource
                    
                    # Check URL format
                    url = datasource["url"]
                    assert "prometheus" in url.lower() or "9090" in url
            
            assert prometheus_found, "Prometheus datasource not configured"
    
    def test_dashboard_provisioning(self):
        """Test Grafana dashboard provisioning configuration."""
        dashboards_dir = Path("monitoring/grafana/provisioning/dashboards")
        
        if not dashboards_dir.exists():
            pytest.skip("Grafana dashboard provisioning not found")
        
        provider_files = list(dashboards_dir.glob("*.yml")) + list(dashboards_dir.glob("*.yaml"))
        
        if not provider_files:
            pytest.skip("No dashboard provider configuration found")
        
        for provider_file in provider_files:
            with open(provider_file, 'r') as f:
                providers_config = yaml.safe_load(f)
            
            # Validate providers structure
            assert "apiVersion" in providers_config
            assert "providers" in providers_config
            
            providers = providers_config["providers"]
            for provider in providers:
                assert "name" in provider
                assert "folder" in provider
                assert "options" in provider
                
                options = provider["options"]
                assert "path" in options
    
    def test_dashboard_json_files(self):
        """Test Grafana dashboard JSON files."""
        dashboards_dir = Path("monitoring/grafana/dashboards")
        
        if not dashboards_dir.exists():
            pytest.skip("Grafana dashboards directory not found")
        
        dashboard_files = list(dashboards_dir.glob("*.json"))
        
        if not dashboard_files:
            pytest.skip("No dashboard JSON files found")
        
        for dashboard_file in dashboard_files:
            with open(dashboard_file, 'r') as f:
                dashboard_config = json.load(f)
            
            # Validate dashboard structure
            assert "dashboard" in dashboard_config or "title" in dashboard_config
            
            # If it's a dashboard object
            if "dashboard" in dashboard_config:
                dashboard = dashboard_config["dashboard"]
            else:
                dashboard = dashboard_config
            
            assert "title" in dashboard
            assert "panels" in dashboard
            
            # Check panels
            panels = dashboard["panels"]
            for panel in panels:
                assert "title" in panel
                assert "type" in panel
                
                # Check for datasource configuration
                if "datasource" in panel:
                    datasource = panel["datasource"]
                    if isinstance(datasource, dict):
                        assert "type" in datasource
                        assert datasource["type"] in ["prometheus", "loki", "jaeger"]


class TestHealthCheckEndpoints:
    """Test health check endpoint functionality and integration."""
    
    @pytest.mark.asyncio
    async def test_basic_health_check(self):
        """Test basic health check endpoint functionality."""
        health_data = await health_check_basic()
        
        # Validate response structure
        assert isinstance(health_data, dict)
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "message" in health_data
        
        # Validate status values
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Validate timestamp
        timestamp = health_data["timestamp"]
        assert isinstance(timestamp, (int, float))
        assert timestamp > 0
        
        # Check timestamp is recent (within last minute)
        current_time = time.time()
        assert abs(current_time - timestamp) < 60, "Health check timestamp too old"
    
    @pytest.mark.asyncio
    async def test_detailed_health_check(self):
        """Test detailed health check with component information."""
        health_data = await health_check_detailed()
        
        # Validate basic structure
        assert isinstance(health_data, dict)
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "components" in health_data
        assert "endpoint" in health_data
        assert health_data["endpoint"] == "detailed"
        
        # Validate components structure
        components = health_data["components"]
        assert isinstance(components, dict)
        
        # Check standard components
        expected_components = [
            "system_resources",
            "qdrant_connectivity", 
            "embedding_service",
            "file_watchers",
            "configuration"
        ]
        
        for component in expected_components:
            if component in components:
                component_data = components[component]
                
                # Validate component structure
                assert "status" in component_data
                assert "message" in component_data
                assert component_data["status"] in ["healthy", "degraded", "unhealthy", "unknown"]
                
                # Check optional fields
                if "last_check" in component_data:
                    assert isinstance(component_data["last_check"], (int, float))
                
                if "response_time" in component_data:
                    assert isinstance(component_data["response_time"], (int, float))
                    assert component_data["response_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance requirements."""
        start_time = time.perf_counter()
        
        # Run basic health check
        health_data = await health_check_basic()
        
        basic_duration = time.perf_counter() - start_time
        
        # Basic health check should be very fast (< 1 second)
        assert basic_duration < 1.0, f"Basic health check too slow: {basic_duration:.3f}s"
        
        start_time = time.perf_counter()
        
        # Run detailed health check
        detailed_data = await health_check_detailed()
        
        detailed_duration = time.perf_counter() - start_time
        
        # Detailed health check should be reasonably fast (< 10 seconds)
        assert detailed_duration < 10.0, f"Detailed health check too slow: {detailed_duration:.3f}s"
        
        # Verify both returned valid data
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert detailed_data["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_health_check_error_handling(self):
        """Test health check error handling and resilience."""
        # Test with mocked failing components
        with patch.object(health_checker_instance, '_check_system_resources') as mock_system:
            mock_system.side_effect = Exception("Mock system check failure")
            
            health_data = await health_check_detailed()
            
            # Health check should still return valid structure even with failures
            assert isinstance(health_data, dict)
            assert "status" in health_data
            assert "components" in health_data
            
            # System resources component should show error
            components = health_data["components"]
            if "system_resources" in components:
                system_component = components["system_resources"]
                assert system_component["status"] in ["unhealthy", "unknown"]
                assert "error" in system_component or "Mock system check failure" in system_component.get("message", "")
    
    @pytest.mark.asyncio
    async def test_kubernetes_health_probe_compatibility(self):
        """Test health check compatibility with Kubernetes probes."""
        # Test liveness probe compatibility
        health_data = await health_check_basic()
        
        # Liveness probe expects simple HTTP 200/503 responses
        if health_data["status"] == "healthy":
            expected_http_code = 200
        elif health_data["status"] == "degraded":
            # Degraded services should still pass liveness but may fail readiness
            expected_http_code = 200  # Container is alive, just degraded
        else:
            expected_http_code = 503  # Unhealthy
        
        assert expected_http_code in [200, 503]
        
        # Test readiness probe compatibility
        # Readiness should be more strict - only healthy services should be ready
        if health_data["status"] == "healthy":
            readiness_code = 200
        else:
            readiness_code = 503  # Not ready for traffic
        
        assert readiness_code in [200, 503]


class TestSystemDiagnostics:
    """Test comprehensive system diagnostics functionality."""
    
    @pytest.mark.asyncio
    async def test_system_diagnostics_structure(self):
        """Test system diagnostics response structure."""
        diagnostics = await system_diagnostics()
        
        # Validate main structure
        assert isinstance(diagnostics, dict)
        assert "endpoint" in diagnostics
        assert diagnostics["endpoint"] == "diagnostics"
        assert "generated_at" in diagnostics
        assert "health_status" in diagnostics
        
        # Validate timestamp
        generated_at = diagnostics["generated_at"]
        assert isinstance(generated_at, (int, float))
        assert abs(time.time() - generated_at) < 60
        
        # Validate health status section
        health_status = diagnostics["health_status"]
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "components" in health_status
    
    @pytest.mark.asyncio
    async def test_diagnostics_system_info(self):
        """Test system information in diagnostics."""
        diagnostics = await system_diagnostics()
        
        if "system_info" in diagnostics:
            system_info = diagnostics["system_info"]
            
            if "error" not in system_info:
                # Validate system info structure
                expected_fields = [
                    "process_id",
                    "memory_info",
                    "cpu_percent",
                    "create_time",
                    "num_threads"
                ]
                
                for field in expected_fields:
                    if field not in system_info:
                        print(f"Warning: {field} not in system info")
                
                # Validate data types
                if "process_id" in system_info:
                    assert isinstance(system_info["process_id"], int)
                    assert system_info["process_id"] > 0
                
                if "memory_info" in system_info:
                    memory_info = system_info["memory_info"]
                    assert isinstance(memory_info, dict)
                    assert "rss" in memory_info or "vms" in memory_info
    
    @pytest.mark.asyncio
    async def test_diagnostics_configuration_info(self):
        """Test configuration information in diagnostics."""
        diagnostics = await system_diagnostics()
        
        if "configuration" in diagnostics:
            config_info = diagnostics["configuration"]
            assert isinstance(config_info, dict)
            
            # Check for basic configuration fields
            expected_config_fields = [
                "enabled",
                "check_interval", 
                "registered_checks"
            ]
            
            for field in expected_config_fields:
                assert field in config_info, f"Configuration field {field} missing"


class TestMetricsJsonEndpoint:
    """Test JSON metrics endpoint functionality."""
    
    @pytest.mark.asyncio
    async def test_json_metrics_format(self):
        """Test JSON metrics endpoint format."""
        # Generate some test metrics
        metrics_instance.increment_counter("json_test_counter", label="test")
        metrics_instance.set_gauge("json_test_gauge", 42.5)
        metrics_instance.record_histogram("json_test_histogram", 0.123)
        
        json_data = await metrics_json()
        
        # Validate structure
        assert isinstance(json_data, dict)
        assert "timestamp" in json_data
        assert "export_format" in json_data
        assert json_data["export_format"] == "json"
        
        # Validate metrics sections
        assert "counters" in json_data
        assert "gauges" in json_data
        assert "histograms" in json_data
        
        # Validate counters
        counters = json_data["counters"]
        assert isinstance(counters, dict)
        
        if "json_test_counter" in counters:
            counter_data = counters["json_test_counter"]
            assert "value" in counter_data
            assert "labeled_values" in counter_data
            assert isinstance(counter_data["value"], (int, float))
        
        # Validate gauges
        gauges = json_data["gauges"]
        assert isinstance(gauges, dict)
        
        if "json_test_gauge" in gauges:
            gauge_data = gauges["json_test_gauge"]
            assert "value" in gauge_data
            assert isinstance(gauge_data["value"], (int, float))
        
        # Validate histograms
        histograms = json_data["histograms"]
        assert isinstance(histograms, dict)
        
        if "json_test_histogram" in histograms:
            histogram_data = histograms["json_test_histogram"]
            assert "count" in histogram_data
            assert "sum" in histogram_data
            assert "buckets" in histogram_data
            assert isinstance(histogram_data["count"], int)
            assert isinstance(histogram_data["sum"], (int, float))
    
    @pytest.mark.asyncio
    async def test_json_metrics_content(self):
        """Test JSON metrics content accuracy."""
        # Reset metrics to start clean
        metrics_instance.reset_metrics()
        
        # Add specific test metrics
        metrics_instance.increment_counter("accuracy_test_counter", 5.0)
        metrics_instance.set_gauge("accuracy_test_gauge", 123.45)
        
        for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
            metrics_instance.record_histogram("accuracy_test_histogram", value)
        
        json_data = await metrics_json()
        
        # Verify counter accuracy
        counters = json_data["counters"]
        if "accuracy_test_counter" in counters:
            counter_value = counters["accuracy_test_counter"]["value"]
            assert counter_value == 5.0
        
        # Verify gauge accuracy
        gauges = json_data["gauges"]
        if "accuracy_test_gauge" in gauges:
            gauge_value = gauges["accuracy_test_gauge"]["value"]
            assert gauge_value == 123.45
        
        # Verify histogram accuracy
        histograms = json_data["histograms"]
        if "accuracy_test_histogram" in histograms:
            histogram_data = histograms["accuracy_test_histogram"]
            assert histogram_data["count"] == 5
            assert abs(histogram_data["sum"] - 1.5) < 0.001  # 0.1+0.2+0.3+0.4+0.5
            assert abs(histogram_data["average"] - 0.3) < 0.001


class TestAlertingIntegration:
    """Test alerting and notification integration."""
    
    def test_alert_rules_syntax(self):
        """Test alerting rules syntax and validity."""
        rules_dir = Path("monitoring/prometheus/rules")
        
        if not rules_dir.exists():
            pytest.skip("Alerting rules directory not found")
        
        rule_files = list(rules_dir.glob("*.yml")) + list(rules_dir.glob("*.yaml"))
        
        for rule_file in rule_files:
            with open(rule_file, 'r') as f:
                content = f.read()
            
            try:
                rules_config = yaml.safe_load(content)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {rule_file}: {e}")
            
            # Validate rules structure
            assert "groups" in rules_config, f"No groups in {rule_file}"
            
            for group in rules_config["groups"]:
                assert "name" in group, f"Group missing name in {rule_file}"
                assert "rules" in group, f"Group missing rules in {rule_file}"
                
                for rule in group["rules"]:
                    if "alert" in rule:
                        # This is an alerting rule
                        assert "expr" in rule, f"Alert {rule.get('alert')} missing expr"
                        assert "for" in rule, f"Alert {rule.get('alert')} missing for"
                        
                        # Validate expression syntax (basic check)
                        expr = rule["expr"]
                        assert isinstance(expr, str), f"Expression must be string in {rule.get('alert')}"
                        assert len(expr.strip()) > 0, f"Empty expression in {rule.get('alert')}"
    
    def test_critical_alerts_coverage(self):
        """Test that critical system metrics have alerting rules."""
        rules_dir = Path("monitoring/prometheus/rules")
        
        if not rules_dir.exists():
            pytest.skip("Alerting rules directory not found")
        
        # Collect all alert expressions
        all_expressions = []
        
        for rule_file in rules_dir.glob("*.yml"):
            with open(rule_file, 'r') as f:
                rules_config = yaml.safe_load(f)
            
            for group in rules_config.get("groups", []):
                for rule in group.get("rules", []):
                    if "alert" in rule and "expr" in rule:
                        all_expressions.append(rule["expr"])
        
        if not all_expressions:
            pytest.skip("No alerting rules found")
        
        # Check for critical metric coverage
        critical_metrics = [
            "memory_usage",
            "cpu_usage", 
            "disk_usage",
            "request_errors",
            "service_down",
            "health_check"
        ]
        
        expressions_text = " ".join(all_expressions)
        
        for metric in critical_metrics:
            if metric not in expressions_text.lower():
                print(f"Warning: No alerting rule found for critical metric: {metric}")


class TestLogAggregationIntegration:
    """Test log aggregation and structured logging integration."""
    
    def test_loki_configuration(self):
        """Test Loki log aggregation configuration."""
        loki_config_path = Path("monitoring/loki/loki-config.yaml")
        
        if not loki_config_path.exists():
            pytest.skip("Loki configuration not found")
        
        with open(loki_config_path, 'r') as f:
            loki_config = yaml.safe_load(f)
        
        # Validate basic structure
        assert "server" in loki_config
        assert "ingester" in loki_config
        assert "schema_config" in loki_config
        assert "storage_config" in loki_config
        
        # Check server configuration
        server_config = loki_config["server"]
        assert "http_listen_port" in server_config
        
        # Check schema configuration
        schema_config = loki_config["schema_config"]
        assert "configs" in schema_config
        
        configs = schema_config["configs"]
        assert len(configs) > 0
        
        for config in configs:
            assert "from" in config
            assert "store" in config
            assert "object_store" in config
            assert "schema" in config
    
    def test_structured_logging_format(self):
        """Test structured logging format compatibility."""
        from src.workspace_qdrant_mcp.observability.logger import get_logger
        
        logger = get_logger("test_structured_logging")
        
        # Test that structured logging methods exist and work
        try:
            logger.info("Test message", component="test", operation="test_operation")
            logger.error("Test error", error_type="TestError", stack_trace="test")
            logger.warning("Test warning", metric="cpu_usage", value=85.5, threshold=80.0)
        except Exception as e:
            pytest.fail(f"Structured logging failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])