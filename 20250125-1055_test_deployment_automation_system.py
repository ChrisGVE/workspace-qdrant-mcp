#!/usr/bin/env python3
"""
Comprehensive unit tests for deployment automation system with edge cases.

Tests cover:
- All deployment strategies (blue-green, canary, rolling, recreate)
- Health check failures and edge cases
- Rollback mechanisms and error scenarios
- Performance monitoring and thresholds
- Multi-environment deployment coordination
- State persistence and recovery
- Network failures and timeout handling
- Concurrent deployment scenarios
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
import requests_mock

# Import our deployment system
from deployment_automation_system import (
    DeploymentAutomationSystem,
    DeploymentConfig,
    DeploymentState,
    HealthCheckResult,
    DeploymentStrategy,
    DeploymentStatus,
    Environment
)


class TestDeploymentAutomationSystem:
    """Comprehensive test suite for deployment automation system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def deployment_system(self, temp_dir):
        """Create deployment system instance for testing."""
        # Create test configuration
        config_file = temp_dir / "test_config.yaml"
        config_content = """
environments:
  development:
    kubernetes_context: 'dev-cluster'
    namespace: 'workspace-qdrant-dev'
    replicas: 1
    health_check_path: '/health'
  staging:
    kubernetes_context: 'staging-cluster'
    namespace: 'workspace-qdrant-staging'
    replicas: 2
    health_check_path: '/health'
  production:
    kubernetes_context: 'prod-cluster'
    namespace: 'workspace-qdrant-prod'
    replicas: 3
    health_check_path: '/health'
monitoring:
  prometheus_url: 'http://test-prometheus:9090'
  grafana_url: 'http://test-grafana:3000'
"""
        config_file.write_text(config_content)

        # Initialize system with test config
        with patch.object(Path, 'cwd', return_value=temp_dir):
            system = DeploymentAutomationSystem(str(config_file))
            system.state_file = temp_dir / "deployment_state.json"
            yield system

    @pytest.fixture
    def sample_deployment_config(self):
        """Create sample deployment configuration."""
        return DeploymentConfig(
            environment=Environment.STAGING,
            strategy=DeploymentStrategy.BLUE_GREEN,
            version="v1.2.3",
            artifact_url="https://artifacts.test.com/app/v1.2.3",
            health_check_url="https://staging-app.test.com/health"
        )

    @pytest.mark.asyncio
    async def test_blue_green_deployment_success(self, deployment_system, sample_deployment_config):
        """Test successful blue-green deployment."""
        system = deployment_system
        config = sample_deployment_config

        # Mock external calls
        with patch.object(system, '_get_current_version', return_value="v1.2.2"), \
             patch.object(system, '_deploy_to_environment'), \
             patch.object(system, '_switch_traffic'), \
             patch.object(system, '_cleanup_environment'), \
             patch.object(system, '_perform_health_checks', return_value=True):

            deployment_id = await system.deploy(config)

            # Verify deployment state
            state = system.get_deployment_status(deployment_id)
            assert state is not None
            assert state.status == DeploymentStatus.SUCCESS
            assert state.previous_version == "v1.2.2"
            assert state.config.version == "v1.2.3"

    @pytest.mark.asyncio
    async def test_blue_green_deployment_health_check_failure(self, deployment_system, sample_deployment_config):
        """Test blue-green deployment with health check failure."""
        system = deployment_system
        config = sample_deployment_config

        # Mock health check failure on green environment
        health_check_calls = []

        async def mock_health_checks(url, state, duration=None):
            health_check_calls.append(url)
            if "env=green" in url:
                return False  # Green environment fails
            return True  # Blue environment succeeds

        with patch.object(system, '_get_current_version', return_value="v1.2.2"), \
             patch.object(system, '_deploy_to_environment'), \
             patch.object(system, '_switch_traffic'), \
             patch.object(system, '_perform_health_checks', side_effect=mock_health_checks):

            deployment_id = await system.deploy(config)

            # Verify deployment failed
            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.FAILED

    @pytest.mark.asyncio
    async def test_canary_deployment_success(self, deployment_system):
        """Test successful canary deployment."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            version="v1.3.0",
            artifact_url="https://artifacts.test.com/app/v1.3.0",
            health_check_url="https://app.test.com/health",
            canary_traffic_percentage=10
        )

        # Mock successful canary deployment
        with patch.object(system, '_get_current_version', return_value="v1.2.9"), \
             patch.object(system, '_deploy_canary'), \
             patch.object(system, '_configure_canary_traffic'), \
             patch.object(system, '_monitor_canary_performance', return_value=True), \
             patch.object(system, '_promote_canary'):

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.SUCCESS
            assert state.config.strategy == DeploymentStrategy.CANARY

    @pytest.mark.asyncio
    async def test_canary_deployment_performance_failure(self, deployment_system):
        """Test canary deployment with performance failure and rollback."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            version="v1.3.0",
            artifact_url="https://artifacts.test.com/app/v1.3.0",
            health_check_url="https://app.test.com/health",
            canary_traffic_percentage=10,
            auto_rollback=True
        )

        # Mock canary performance failure
        performance_calls = []

        async def mock_canary_performance(state):
            performance_calls.append(state.deployment_id)
            return False  # Performance monitoring fails

        with patch.object(system, '_get_current_version', return_value="v1.2.9"), \
             patch.object(system, '_deploy_canary'), \
             patch.object(system, '_configure_canary_traffic'), \
             patch.object(system, '_monitor_canary_performance', side_effect=mock_canary_performance), \
             patch.object(system, '_rollback_canary') as mock_rollback, \
             patch.object(system, '_execute_rollback') as mock_execute_rollback:

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status in [DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]
            mock_rollback.assert_called()

    @pytest.mark.asyncio
    async def test_rolling_deployment_success(self, deployment_system):
        """Test successful rolling deployment."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.ROLLING,
            version="v1.4.0",
            artifact_url="https://artifacts.test.com/app/v1.4.0",
            health_check_url="https://app.test.com/health"
        )

        # Mock successful rolling deployment
        replica_updates = []

        async def mock_update_replica(replica_id, config):
            replica_updates.append(replica_id)

        async def mock_health_check_replica(replica_id, config):
            return True  # All replicas healthy

        with patch.object(system, '_get_current_version', return_value="v1.3.9"), \
             patch.object(system, '_update_replica', side_effect=mock_update_replica), \
             patch.object(system, '_health_check_replica', side_effect=mock_health_check_replica), \
             patch.object(system, '_perform_health_checks', return_value=True):

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.SUCCESS

            # Verify all replicas were updated
            expected_replicas = system.config['environments']['production']['replicas']
            assert len(replica_updates) == expected_replicas

    @pytest.mark.asyncio
    async def test_rolling_deployment_replica_failure(self, deployment_system):
        """Test rolling deployment with replica failure and rollback."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.ROLLING,
            version="v1.4.0",
            artifact_url="https://artifacts.test.com/app/v1.4.0",
            health_check_url="https://app.test.com/health"
        )

        # Mock replica failure
        async def mock_health_check_replica(replica_id, config):
            return replica_id != 1  # Second replica fails

        rollback_calls = []

        async def mock_rollback_replica(replica_id, config):
            rollback_calls.append(replica_id)

        with patch.object(system, '_get_current_version', return_value="v1.3.9"), \
             patch.object(system, '_update_replica'), \
             patch.object(system, '_health_check_replica', side_effect=mock_health_check_replica), \
             patch.object(system, '_rollback_replica', side_effect=mock_rollback_replica):

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.FAILED

            # Verify rollback was attempted
            assert len(rollback_calls) == 2  # Replica 0 and 1 should be rolled back

    @pytest.mark.asyncio
    async def test_recreate_deployment_success(self, deployment_system):
        """Test successful recreate deployment."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.RECREATE,
            version="v1.5.0",
            artifact_url="https://artifacts.test.com/app/v1.5.0",
            health_check_url="https://dev-app.test.com/health"
        )

        with patch.object(system, '_get_current_version', return_value="v1.4.9"), \
             patch.object(system, '_stop_current_version'), \
             patch.object(system, '_deploy_to_environment'), \
             patch.object(system, '_perform_health_checks', return_value=True):

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_health_checks_comprehensive(self, deployment_system, sample_deployment_config):
        """Test comprehensive health check scenarios."""
        system = deployment_system
        config = sample_deployment_config

        # Create test state
        state = DeploymentState(
            deployment_id="test-health-check",
            config=config,
            status=DeploymentStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        # Test successful health checks
        with requests_mock.Mocker() as m:
            m.get(config.health_check_url, status_code=200)

            healthy = await system._perform_health_checks(config.health_check_url, state, duration=5)

            assert healthy
            assert len(state.health_checks) > 0
            assert all(hc.healthy for hc in state.health_checks)

    @pytest.mark.asyncio
    async def test_health_checks_failure_scenarios(self, deployment_system, sample_deployment_config):
        """Test health check failure scenarios."""
        system = deployment_system
        config = sample_deployment_config

        test_cases = [
            (500, "Internal Server Error"),
            (503, "Service Unavailable"),
            (404, "Not Found"),
            (408, "Request Timeout")
        ]

        for status_code, description in test_cases:
            state = DeploymentState(
                deployment_id=f"test-health-{status_code}",
                config=config,
                status=DeploymentStatus.IN_PROGRESS,
                started_at=datetime.utcnow()
            )

            with requests_mock.Mocker() as m:
                m.get(config.health_check_url, status_code=status_code)

                healthy = await system._perform_health_checks(config.health_check_url, state, duration=5)

                assert not healthy, f"Health check should fail for {status_code} {description}"
                assert any(not hc.healthy for hc in state.health_checks)

    @pytest.mark.asyncio
    async def test_health_checks_network_errors(self, deployment_system, sample_deployment_config):
        """Test health check network error scenarios."""
        system = deployment_system
        config = sample_deployment_config

        # Test connection timeout
        state = DeploymentState(
            deployment_id="test-timeout",
            config=config,
            status=DeploymentStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        with requests_mock.Mocker() as m:
            m.get(config.health_check_url, exc=requests.exceptions.Timeout)

            healthy = await system._perform_health_checks(config.health_check_url, state, duration=5)

            assert not healthy
            assert any(hc.error_message for hc in state.health_checks if not hc.healthy)

        # Test connection error
        state = DeploymentState(
            deployment_id="test-connection-error",
            config=config,
            status=DeploymentStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        with requests_mock.Mocker() as m:
            m.get(config.health_check_url, exc=requests.exceptions.ConnectionError)

            healthy = await system._perform_health_checks(config.health_check_url, state, duration=5)

            assert not healthy

    @pytest.mark.asyncio
    async def test_canary_performance_monitoring(self, deployment_system):
        """Test canary performance monitoring with various thresholds."""
        system = deployment_system

        config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            version="v1.3.0",
            artifact_url="https://artifacts.test.com/app/v1.3.0",
            health_check_url="https://app.test.com/health",
            rollback_threshold=0.05  # 5% error rate threshold
        )

        state = DeploymentState(
            deployment_id="test-canary-perf",
            config=config,
            status=DeploymentStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        # Test with acceptable metrics
        acceptable_metrics = {
            "error_rate": 0.02,  # 2% - below threshold
            "avg_response_time": 0.15,
            "throughput": 100.0,
            "baseline_response_time": 0.12,
            "baseline_throughput": 95.0
        }

        with patch.object(system, '_collect_performance_metrics', return_value=acceptable_metrics):
            result = await system._monitor_canary_performance(state)
            assert result is True

        # Test with high error rate
        high_error_metrics = acceptable_metrics.copy()
        high_error_metrics["error_rate"] = 0.08  # 8% - above 5% threshold

        with patch.object(system, '_collect_performance_metrics', return_value=high_error_metrics):
            result = await system._monitor_canary_performance(state)
            assert result is False

        # Test with degraded response time
        slow_response_metrics = acceptable_metrics.copy()
        slow_response_metrics["avg_response_time"] = 0.20  # 200ms - 67% higher than baseline

        with patch.object(system, '_collect_performance_metrics', return_value=slow_response_metrics):
            result = await system._monitor_canary_performance(state)
            assert result is False

        # Test with degraded throughput
        low_throughput_metrics = acceptable_metrics.copy()
        low_throughput_metrics["throughput"] = 70.0  # 26% lower than baseline

        with patch.object(system, '_collect_performance_metrics', return_value=low_throughput_metrics):
            result = await system._monitor_canary_performance(state)
            assert result is False

    @pytest.mark.asyncio
    async def test_rollback_mechanism(self, deployment_system, sample_deployment_config):
        """Test rollback mechanism with various scenarios."""
        system = deployment_system
        config = sample_deployment_config
        config.rollback_enabled = True
        config.auto_rollback = True

        # Create deployment state with rollback info
        state = DeploymentState(
            deployment_id="test-rollback",
            config=config,
            status=DeploymentStatus.FAILED,
            started_at=datetime.utcnow(),
            previous_version="v1.2.1",
            rollback_version="v1.2.1"
        )

        system.deployments[state.deployment_id] = state

        # Mock rollback deployment
        with patch.object(system, '_execute_recreate_deployment', return_value=True):
            await system._execute_rollback(state)

            assert state.status == DeploymentStatus.ROLLED_BACK

        # Test manual rollback
        success = await system.rollback_deployment(state.deployment_id)
        assert success

        # Test rollback with no previous version
        state.rollback_version = None
        await system._execute_rollback(state)
        # Should handle gracefully without crash

    def test_state_persistence(self, deployment_system, sample_deployment_config):
        """Test deployment state persistence and recovery."""
        system = deployment_system
        config = sample_deployment_config

        # Create test deployment state
        state = DeploymentState(
            deployment_id="test-persistence",
            config=config,
            status=DeploymentStatus.SUCCESS,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            previous_version="v1.2.1"
        )

        # Add health check results
        state.health_checks.append(HealthCheckResult(
            healthy=True,
            response_time=0.15,
            error_rate=0.01,
            status_code=200
        ))

        system.deployments[state.deployment_id] = state

        # Save state
        system._save_deployment_state()

        # Create new system instance and verify state is loaded
        new_system = DeploymentAutomationSystem()
        new_system.state_file = system.state_file

        new_system._load_deployment_state()

        # Verify state was preserved
        loaded_state = new_system.get_deployment_status(state.deployment_id)
        assert loaded_state is not None
        assert loaded_state.deployment_id == state.deployment_id
        assert loaded_state.status == state.status
        assert loaded_state.config.version == config.version
        assert len(loaded_state.health_checks) == 1

    def test_deployment_listing_and_filtering(self, deployment_system, sample_deployment_config):
        """Test deployment listing and filtering functionality."""
        system = deployment_system

        # Create multiple deployments
        environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
        deployment_ids = []

        for i, env in enumerate(environments):
            config = DeploymentConfig(
                environment=env,
                strategy=DeploymentStrategy.BLUE_GREEN,
                version=f"v1.{i}.0",
                artifact_url=f"https://artifacts.test.com/app/v1.{i}.0",
                health_check_url=f"https://{env.value}-app.test.com/health"
            )

            state = DeploymentState(
                deployment_id=f"deploy-{env.value}-{i}",
                config=config,
                status=DeploymentStatus.SUCCESS,
                started_at=datetime.utcnow() - timedelta(minutes=i)
            )

            system.deployments[state.deployment_id] = state
            deployment_ids.append(state.deployment_id)

        # Test listing all deployments
        all_deployments = system.list_deployments()
        assert len(all_deployments) == 3

        # Should be sorted by start time (most recent first)
        assert all_deployments[0].deployment_id == deployment_ids[-1]

        # Test filtering by environment
        staging_deployments = system.list_deployments(Environment.STAGING)
        assert len(staging_deployments) == 1
        assert staging_deployments[0].config.environment == Environment.STAGING

    def test_deployment_report_generation(self, deployment_system, sample_deployment_config):
        """Test comprehensive deployment report generation."""
        system = deployment_system
        config = sample_deployment_config

        # Create detailed deployment state
        state = DeploymentState(
            deployment_id="test-report",
            config=config,
            status=DeploymentStatus.SUCCESS,
            started_at=datetime.utcnow() - timedelta(minutes=10),
            completed_at=datetime.utcnow(),
            previous_version="v1.2.1"
        )

        # Add multiple health check results
        for i in range(5):
            state.health_checks.append(HealthCheckResult(
                healthy=i < 4,  # 4/5 healthy
                response_time=0.1 + (i * 0.02),
                error_rate=0.0 if i < 4 else 0.1,
                status_code=200 if i < 4 else 500
            ))

        # Add performance metrics
        state.performance_metrics = {
            "avg_cpu_usage": 45.2,
            "avg_memory_usage": 512.7,
            "peak_response_time": 0.18
        }

        system.deployments[state.deployment_id] = state

        # Generate report
        report = system.generate_deployment_report(state.deployment_id)

        # Verify report contents
        assert report["deployment_id"] == state.deployment_id
        assert report["status"] == DeploymentStatus.SUCCESS.value
        assert report["version"] == config.version
        assert report["health_checks"]["total"] == 5
        assert report["health_checks"]["success_rate"] == 0.8  # 4/5
        assert "avg_response_time" in report["health_checks"]
        assert "performance_metrics" in report
        assert "deployment_duration_seconds" in report

        # Test report for non-existent deployment
        error_report = system.generate_deployment_report("non-existent")
        assert "error" in error_report

    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, deployment_system):
        """Test handling of concurrent deployments to different environments."""
        system = deployment_system

        # Create configurations for different environments
        configs = [
            DeploymentConfig(
                environment=Environment.DEVELOPMENT,
                strategy=DeploymentStrategy.RECREATE,
                version="v1.5.0",
                artifact_url="https://artifacts.test.com/app/v1.5.0",
                health_check_url="https://dev-app.test.com/health"
            ),
            DeploymentConfig(
                environment=Environment.STAGING,
                strategy=DeploymentStrategy.BLUE_GREEN,
                version="v1.4.0",
                artifact_url="https://artifacts.test.com/app/v1.4.0",
                health_check_url="https://staging-app.test.com/health"
            )
        ]

        # Mock all deployment operations
        with patch.object(system, '_get_current_version', return_value="v1.3.0"), \
             patch.object(system, '_deploy_to_environment'), \
             patch.object(system, '_stop_current_version'), \
             patch.object(system, '_switch_traffic'), \
             patch.object(system, '_cleanup_environment'), \
             patch.object(system, '_perform_health_checks', return_value=True):

            # Start concurrent deployments
            tasks = [system.deploy(config) for config in configs]
            deployment_ids = await asyncio.gather(*tasks)

            # Verify both deployments completed
            assert len(deployment_ids) == 2

            for deployment_id in deployment_ids:
                state = system.get_deployment_status(deployment_id)
                assert state.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deployment_timeout_scenarios(self, deployment_system, sample_deployment_config):
        """Test deployment timeout handling."""
        system = deployment_system
        config = sample_deployment_config
        config.deployment_timeout = 5  # 5 second timeout for testing

        # Mock long-running operation
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return True

        with patch.object(system, '_get_current_version', return_value="v1.2.2"), \
             patch.object(system, '_deploy_to_environment', side_effect=slow_operation):

            start_time = time.time()
            deployment_id = await system.deploy(config)
            elapsed_time = time.time() - start_time

            # Should not take longer than reasonable time due to timeout
            assert elapsed_time < 15  # Allow some overhead

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.FAILED

    @pytest.mark.asyncio
    async def test_deployment_error_handling(self, deployment_system, sample_deployment_config):
        """Test comprehensive error handling in deployments."""
        system = deployment_system
        config = sample_deployment_config

        # Test deployment operation exception
        with patch.object(system, '_get_current_version', return_value="v1.2.2"), \
             patch.object(system, '_deploy_to_environment', side_effect=Exception("Deployment failed")):

            deployment_id = await system.deploy(config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.FAILED
            assert "Deployment failed" in state.error_message

    @pytest.mark.asyncio
    async def test_deployment_validation(self, deployment_system):
        """Test deployment configuration validation."""
        system = deployment_system

        # Test invalid configuration scenarios
        invalid_configs = [
            # Missing required fields would be caught by dataclass validation
            # Test business logic validation
        ]

        # Valid configuration should work
        valid_config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            version="v2.0.0",
            artifact_url="https://artifacts.test.com/app/v2.0.0",
            health_check_url="https://app.test.com/health"
        )

        with patch.object(system, '_get_current_version', return_value="v1.9.9"), \
             patch.object(system, '_deploy_canary'), \
             patch.object(system, '_configure_canary_traffic'), \
             patch.object(system, '_monitor_canary_performance', return_value=True), \
             patch.object(system, '_promote_canary'):

            deployment_id = await system.deploy(valid_config)

            state = system.get_deployment_status(deployment_id)
            assert state.status == DeploymentStatus.SUCCESS

    def test_edge_case_health_check_results(self, deployment_system, sample_deployment_config):
        """Test edge cases in health check result handling."""
        # Test health check result with extreme values
        extreme_result = HealthCheckResult(
            healthy=True,
            response_time=999.999,  # Very slow but still healthy
            error_rate=0.0,
            status_code=200
        )

        assert extreme_result.healthy
        assert extreme_result.response_time > 0

        # Test health check result with edge case timestamps
        past_result = HealthCheckResult(
            healthy=False,
            response_time=0.001,
            error_rate=1.0,
            status_code=500,
            timestamp=datetime(1970, 1, 1)  # Unix epoch
        )

        assert not past_result.healthy
        assert past_result.timestamp.year == 1970

    def test_deployment_state_edge_cases(self, deployment_system, sample_deployment_config):
        """Test edge cases in deployment state management."""
        system = deployment_system
        config = sample_deployment_config

        # Test deployment state with no health checks
        empty_state = DeploymentState(
            deployment_id="empty-health-checks",
            config=config,
            status=DeploymentStatus.SUCCESS,
            started_at=datetime.utcnow()
        )

        system.deployments[empty_state.deployment_id] = empty_state

        # Generate report should handle empty health checks
        report = system.generate_deployment_report(empty_state.deployment_id)
        assert report["health_checks"]["total"] == 0
        assert report["health_checks"]["success_rate"] == 0.0

        # Test deployment state with same start and completion time
        instant_state = DeploymentState(
            deployment_id="instant-deployment",
            config=config,
            status=DeploymentStatus.SUCCESS,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

        system.deployments[instant_state.deployment_id] = instant_state

        report = system.generate_deployment_report(instant_state.deployment_id)
        assert report["deployment_duration_seconds"] >= 0


if __name__ == "__main__":
    """Run comprehensive deployment automation tests."""
    print("🧪 Running comprehensive deployment automation tests...")

    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])