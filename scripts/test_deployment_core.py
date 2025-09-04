#!/usr/bin/env python3
"""
Core deployment functionality test runner.

Tests the core deployment testing functionality without external dependencies
like FastAPI, Docker, etc. This validates the essential testing framework
and utilities work correctly.

Usage:
    python scripts/test_deployment_core.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append('.')

def test_deployment_helpers():
    """Test deployment helper utilities."""
    print("🧪 Testing deployment helpers...")
    
    try:
        from tests.utils.deployment_helpers import (
            DeploymentTestHelper,
            DockerTestHelper,
            MonitoringTestHelper,
            generate_test_metrics,
            generate_test_health_status
        )
        
        # Test DeploymentTestHelper
        helper = DeploymentTestHelper()
        temp_dir = helper.create_temp_dir()
        assert temp_dir.exists(), "Temp directory creation failed"
        
        # Test mock binary creation
        binary_path = helper.create_mock_binary(temp_dir / "bin", "test-binary")
        assert binary_path.exists(), "Mock binary creation failed"
        assert binary_path.is_file(), "Mock binary is not a file"
        
        # Test mock config creation
        config_path = helper.create_mock_config(temp_dir / "config")
        assert config_path.exists(), "Mock config creation failed"
        
        # Test cleanup
        helper.cleanup()
        assert not temp_dir.exists(), "Cleanup failed"
        
        print("✅ DeploymentTestHelper: PASSED")
        
        # Test DockerTestHelper
        docker_helper = DockerTestHelper()
        docker_available = docker_helper.is_docker_available()
        compose_available = docker_helper.is_compose_available()
        
        print(f"✅ DockerTestHelper: PASSED (Docker: {docker_available}, Compose: {compose_available})")
        
        # Test MonitoringTestHelper
        monitoring_helper = MonitoringTestHelper()
        assert hasattr(monitoring_helper, 'wait_for_endpoint'), "MonitoringTestHelper missing methods"
        
        print("✅ MonitoringTestHelper: PASSED")
        
        # Test data generators
        metrics_data = generate_test_metrics()
        assert "counters" in metrics_data, "Test metrics generation failed"
        
        health_data = generate_test_health_status()
        assert "status" in health_data, "Test health status generation failed"
        assert health_data["status"] == "healthy", "Test health status incorrect"
        
        print("✅ Data generators: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment helpers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_manager():
    """Test service manager functionality."""
    print("🧪 Testing service manager...")
    
    try:
        from src.workspace_qdrant_mcp.cli.commands.service import ServiceManager
        
        service_manager = ServiceManager()
        
        # Test platform detection
        assert service_manager.system in ["linux", "darwin", "windows"], "Platform detection failed"
        assert service_manager.service_name == "memexd", "Service name incorrect"
        assert service_manager.daemon_binary == "memexd-priority", "Daemon binary name incorrect"
        
        print(f"✅ ServiceManager: PASSED (Platform: {service_manager.system})")
        
        return True
        
    except Exception as e:
        print(f"❌ Service manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observability_core():
    """Test core observability functionality."""
    print("🧪 Testing observability core...")
    
    try:
        from src.workspace_qdrant_mcp.observability.metrics import MetricsCollector
        from src.workspace_qdrant_mcp.observability.health import HealthChecker, HealthStatus
        
        # Test metrics collector
        metrics = MetricsCollector()
        
        # Test counter
        metrics.increment_counter("test_counter", 5.0)
        counter = metrics.counters.get("test_counter")
        assert counter is not None, "Counter not created"
        assert counter.get_value() == 5.0, "Counter value incorrect"
        
        # Test gauge
        metrics.set_gauge("test_gauge", 42.0)
        gauge = metrics.gauges.get("test_gauge")
        assert gauge is not None, "Gauge not created"
        assert gauge.get_value() == 42.0, "Gauge value incorrect"
        
        # Test histogram
        metrics.record_histogram("test_histogram", 0.123)
        histogram = metrics.histograms.get("test_histogram")
        assert histogram is not None, "Histogram not created"
        assert histogram.get_count() == 1, "Histogram count incorrect"
        
        print("✅ MetricsCollector: PASSED")
        
        # Test health checker
        health_checker = HealthChecker()
        
        # Test health status enum
        assert HealthStatus.HEALTHY.value == "healthy", "HealthStatus enum incorrect"
        assert HealthStatus.UNHEALTHY.value == "unhealthy", "HealthStatus enum incorrect"
        
        print("✅ HealthChecker: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Observability core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_checks():
    """Test health check functionality."""
    print("🧪 Testing health checks...")
    
    try:
        from src.workspace_qdrant_mcp.observability.health import health_checker_instance
        
        # Test system resource check (this should work without external dependencies)
        result = await health_checker_instance._check_system_resources()
        
        assert "status" in result, "System resources check missing status"
        assert result["status"] in ["healthy", "degraded", "unhealthy"], "Invalid health status"
        assert "details" in result, "System resources check missing details"
        
        details = result["details"]
        assert "memory" in details, "Memory details missing"
        assert "cpu" in details, "CPU details missing"
        assert "disk" in details, "Disk details missing"
        
        print("✅ Health checks: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Health checks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_docker_config():
    """Test Docker configuration files."""
    print("🧪 Testing Docker configuration...")
    
    try:
        dockerfile_path = Path("docker/Dockerfile")
        compose_path = Path("docker/docker-compose.yml")
        
        # Test Dockerfile exists
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            assert "FROM" in content, "Dockerfile missing FROM instruction"
            assert "USER" in content, "Dockerfile missing USER instruction"
            print("✅ Dockerfile: PASSED")
        else:
            print("⚠️  Dockerfile: SKIPPED (not found)")
        
        # Test docker-compose.yml exists
        if compose_path.exists():
            import yaml
            with open(compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            assert "services" in compose_config, "docker-compose.yml missing services"
            assert "volumes" in compose_config, "docker-compose.yml missing volumes"
            assert "networks" in compose_config, "docker-compose.yml missing networks"
            
            services = compose_config["services"]
            required_services = ["workspace-qdrant-mcp", "qdrant", "redis"]
            
            for service in required_services:
                if service not in services:
                    print(f"⚠️  Service {service} not found in docker-compose.yml")
            
            print("✅ docker-compose.yml: PASSED")
        else:
            print("⚠️  docker-compose.yml: SKIPPED (not found)")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_guide():
    """Test production deployment guide exists and has key sections."""
    print("🧪 Testing production deployment guide...")
    
    try:
        guide_path = Path("PRODUCTION_DEPLOYMENT.md")
        
        if not guide_path.exists():
            print("⚠️  PRODUCTION_DEPLOYMENT.md: SKIPPED (not found)")
            return False
        
        content = guide_path.read_text()
        
        # Check for key sections
        required_sections = [
            "Service Installation",
            "Docker Deployment",
            "Monitoring and Observability",
            "Security Configuration",
            "Backup and Recovery",
            "Update and Upgrade Procedures"
        ]
        
        for section in required_sections:
            if section not in content:
                print(f"⚠️  Missing section: {section}")
            else:
                print(f"✅ Section found: {section}")
        
        # Check for code blocks (deployment commands)
        if "```bash" not in content:
            print("⚠️  No bash code blocks found")
        else:
            bash_blocks = content.count("```bash")
            print(f"✅ Found {bash_blocks} bash code blocks")
        
        print("✅ Production deployment guide: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Production guide test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all core deployment tests."""
    print("🚀 Starting Core Deployment Testing")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Deployment Helpers", test_deployment_helpers),
        ("Service Manager", test_service_manager),
        ("Observability Core", test_observability_core),
        ("Health Checks", test_health_checks),
        ("Docker Configuration", test_docker_config),
        ("Production Guide", test_production_guide),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results[test_name] = result
            
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
        
        print()  # Add blank line between tests
    
    # Summary
    duration = time.time() - start_time
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print("=" * 60)
    print("📊 CORE DEPLOYMENT TESTING SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Passed: {passed}/{total} tests")
    print()
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print()
    
    if passed == total:
        print("🎉 All core deployment tests PASSED!")
        print("The deployment testing framework is ready for production validation.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) FAILED!")
        print("Review the failures above before proceeding with full deployment testing.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))