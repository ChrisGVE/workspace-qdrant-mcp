#!/usr/bin/env python3
"""
Comprehensive unit tests for the Deployment Automation system.

Tests cover:
- Deployment pipeline execution and strategies
- Quality gate validation and failure handling
- Multi-environment deployment orchestration
- Blue-green, canary, and rolling deployment strategies
- Health checks and readiness probes
- Rollback mechanisms and error recovery
- Command execution and timeout handling
- Deployment progress tracking and reporting
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import the module under test
try:
    from deployment_automation import (
        DeploymentConfig,
        DeploymentPipeline,
        DeploymentProgress,
        DeploymentStatus,
        DeploymentStrategy,
        DeploymentTarget,
        QualityGate,
        QualityGateType,
        ValidationResult,
        ValidationStatus,
        create_example_web_app_deployment,
    )
except ImportError:
    # For when running as standalone module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from deployment_automation import (
        DeploymentConfig,
        DeploymentPipeline,
        DeploymentProgress,
        DeploymentStatus,
        DeploymentStrategy,
        DeploymentTarget,
        QualityGate,
        QualityGateType,
        ValidationResult,
        ValidationStatus,
        create_example_web_app_deployment,
    )


class TestQualityGate:
    """Test QualityGate class."""

    def test_quality_gate_creation(self):
        """Test basic quality gate creation."""
        gate = QualityGate(
            name="unit-tests",
            gate_type=QualityGateType.UNIT_TESTS,
            command=["python", "-m", "pytest"],
            timeout=300,
            required=True,
            success_criteria={"min_coverage": 80.0}
        )

        assert gate.name == "unit-tests"
        assert gate.gate_type == QualityGateType.UNIT_TESTS
        assert gate.command == ["python", "-m", "pytest"]
        assert gate.timeout == 300
        assert gate.required is True
        assert gate.success_criteria["min_coverage"] == 80.0

    def test_quality_gate_defaults(self):
        """Test quality gate with default values."""
        gate = QualityGate(
            name="simple-test",
            gate_type=QualityGateType.SMOKE_TEST,
            command=["echo", "test"]
        )

        assert gate.timeout == 300
        assert gate.required is True
        assert gate.retry_count == 0
        assert gate.environment_vars == {}
        assert gate.success_criteria == {}

    def test_quality_gate_complex_config(self):
        """Test quality gate with complex configuration."""
        gate = QualityGate(
            name="performance-test",
            gate_type=QualityGateType.PERFORMANCE_TEST,
            command=["artillery", "run", "perf-test.yml"],
            timeout=600,
            required=True,
            retry_count=2,
            environment_vars={"TARGET_URL": "http://staging.example.com"},
            success_criteria={
                "max_response_time": 500.0,
                "min_throughput": 1000.0,
                "max_error_rate": 1.0
            }
        )

        assert gate.success_criteria["max_response_time"] == 500.0
        assert gate.environment_vars["TARGET_URL"] == "http://staging.example.com"
        assert gate.retry_count == 2


class TestDeploymentTarget:
    """Test DeploymentTarget class."""

    def test_deployment_target_creation(self):
        """Test deployment target creation."""
        target = DeploymentTarget(
            name="production",
            environment="prod",
            infrastructure_config="kubernetes",
            health_check_url="http://example.com/health",
            readiness_probe={"type": "http", "url": "http://example.com/ready"},
            liveness_probe={"type": "tcp", "port": 8080}
        )

        assert target.name == "production"
        assert target.environment == "prod"
        assert target.infrastructure_config == "kubernetes"
        assert target.health_check_url == "http://example.com/health"
        assert target.readiness_probe["type"] == "http"
        assert target.liveness_probe["port"] == 8080

    def test_deployment_target_defaults(self):
        """Test deployment target with default values."""
        target = DeploymentTarget(
            name="staging",
            environment="staging"
        )

        assert target.infrastructure_config is None
        assert target.health_check_url is None
        assert target.health_check_timeout == 60
        assert target.readiness_probe is None
        assert target.liveness_probe is None


class TestDeploymentConfig:
    """Test DeploymentConfig class."""

    def test_deployment_config_creation(self):
        """Test deployment configuration creation."""
        targets = [
            DeploymentTarget(name="staging", environment="staging"),
            DeploymentTarget(name="production", environment="production")
        ]

        gates = [
            QualityGate(
                name="tests",
                gate_type=QualityGateType.UNIT_TESTS,
                command=["python", "-m", "pytest"]
            )
        ]

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.BLUE_GREEN,
            targets=targets,
            quality_gates=gates,
            rollback_config={"auto_rollback": True},
            timeout=1800,
            parallel_deployments=False,
            deployment_variables={"ENV": "prod"},
            secrets=["API_KEY"]
        )

        assert config.name == "test-app"
        assert config.version == "1.0.0"
        assert config.strategy == DeploymentStrategy.BLUE_GREEN
        assert len(config.targets) == 2
        assert len(config.quality_gates) == 1
        assert config.rollback_config["auto_rollback"] is True
        assert config.deployment_variables["ENV"] == "prod"
        assert "API_KEY" in config.secrets

    def test_deployment_config_defaults(self):
        """Test deployment configuration with default values."""
        targets = [DeploymentTarget(name="test", environment="test")]

        config = DeploymentConfig(
            name="minimal-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets
        )

        assert config.quality_gates == []
        assert config.rollback_config is None
        assert config.notification_config is None
        assert config.timeout == 1800
        assert config.parallel_deployments is False
        assert config.deployment_variables == {}
        assert config.secrets == []


class TestDeploymentPipeline:
    """Test DeploymentPipeline class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create a test deployment pipeline."""
        return DeploymentPipeline(
            artifacts_dir=temp_dir / "artifacts",
            logs_dir=temp_dir / "logs",
            config_dir=temp_dir / "config",
            enable_monitoring=True
        )

    @pytest.fixture
    def simple_config(self):
        """Create a simple deployment configuration."""
        targets = [
            DeploymentTarget(
                name="test-env",
                environment="test",
                health_check_url="http://test.example.com/health"
            )
        ]

        return DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets
        )

    @pytest.fixture
    def complex_config(self):
        """Create a complex deployment configuration."""
        quality_gates = [
            QualityGate(
                name="unit-tests",
                gate_type=QualityGateType.UNIT_TESTS,
                command=["python", "-m", "pytest", "tests/"],
                timeout=300,
                required=True
            ),
            QualityGate(
                name="security-scan",
                gate_type=QualityGateType.SECURITY_SCAN,
                command=["bandit", "-r", "src/"],
                timeout=180,
                required=True,
                success_criteria={"max_vulnerabilities": 0}
            ),
            QualityGate(
                name="smoke-tests",
                gate_type=QualityGateType.SMOKE_TEST,
                command=["python", "-m", "pytest", "tests/smoke/"],
                timeout=120,
                required=False
            )
        ]

        targets = [
            DeploymentTarget(
                name="staging",
                environment="staging",
                infrastructure_config="kubernetes",
                health_check_url="http://staging.example.com/health",
                readiness_probe={"type": "http", "url": "http://staging.example.com/ready"}
            ),
            DeploymentTarget(
                name="production",
                environment="production",
                infrastructure_config="kubernetes",
                health_check_url="http://example.com/health",
                readiness_probe={"type": "http", "url": "http://example.com/ready"}
            )
        ]

        return DeploymentConfig(
            name="complex-app",
            version="2.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets,
            quality_gates=quality_gates,
            rollback_config={
                "auto_rollback": True,
                "previous_version": "1.9.0"
            },
            deployment_variables={"LOG_LEVEL": "INFO"}
        )

    def test_pipeline_initialization(self, temp_dir):
        """Test pipeline initialization."""
        pipeline = DeploymentPipeline(
            artifacts_dir=temp_dir / "artifacts",
            logs_dir=temp_dir / "logs",
            config_dir=temp_dir / "config"
        )

        assert pipeline.artifacts_dir == temp_dir / "artifacts"
        assert pipeline.logs_dir == temp_dir / "logs"
        assert pipeline.config_dir == temp_dir / "config"

        # Directories should be created
        assert pipeline.artifacts_dir.exists()
        assert pipeline.logs_dir.exists()
        assert pipeline.config_dir.exists()

        assert pipeline.active_deployments == {}
        assert pipeline.deployment_history == []

    @patch('asyncio.create_subprocess_exec')
    async def test_execute_command_success(self, mock_subprocess, pipeline):
        """Test successful command execution."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"success", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        result = await pipeline._execute_command(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout == "success"
        assert result.stderr == ""

    @patch('asyncio.create_subprocess_exec')
    async def test_execute_command_failure(self, mock_subprocess, pipeline):
        """Test failed command execution."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        result = await pipeline._execute_command(["false"])

        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr == "error"

    @patch('asyncio.create_subprocess_exec')
    async def test_execute_command_timeout(self, mock_subprocess, pipeline):
        """Test command execution timeout."""
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        with pytest.raises(asyncio.TimeoutError, match="Command timed out"):
            await pipeline._execute_command(["sleep", "300"], timeout=1)

        mock_process.kill.assert_called_once()

    async def test_pre_deployment_validation_success(self, pipeline, simple_config):
        """Test successful pre-deployment validation."""
        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Should pass without exceptions
        await pipeline._pre_deployment_validation(simple_config, progress)

    async def test_pre_deployment_validation_missing_environment(self, pipeline):
        """Test pre-deployment validation with missing environment."""
        targets = [DeploymentTarget(name="test", environment="")]  # Empty environment
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets
        )

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        with pytest.raises(ValueError, match="missing environment configuration"):
            await pipeline._pre_deployment_validation(config, progress)

    async def test_pre_deployment_validation_conflicting_deployments(self, pipeline, simple_config):
        """Test pre-deployment validation with conflicting deployments."""
        # Add an active deployment using the same target
        existing_progress = DeploymentProgress(
            deployment_id="existing-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        existing_progress.current_target = "test-env"
        pipeline.active_deployments["existing-123"] = existing_progress

        new_progress = DeploymentProgress(
            deployment_id="new-456",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        with pytest.raises(RuntimeError, match="Deployment conflict"):
            await pipeline._pre_deployment_validation(simple_config, new_progress)

    @patch('asyncio.create_subprocess_exec')
    async def test_quality_gate_unit_tests_success(self, mock_subprocess, pipeline, complex_config):
        """Test successful unit tests quality gate."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"tests passed", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        await pipeline._execute_quality_gates(complex_config, progress)

        # Should have validation results for all gates
        assert len(progress.validation_results) == 3

        # Unit tests should have passed
        unit_test_result = next(
            (r for r in progress.validation_results if r.gate_name == "unit-tests"),
            None
        )
        assert unit_test_result is not None
        assert unit_test_result.status == ValidationStatus.PASSED

    @patch('asyncio.create_subprocess_exec')
    async def test_quality_gate_required_failure(self, mock_subprocess, pipeline, complex_config):
        """Test required quality gate failure."""
        # Mock failing commands
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"tests failed")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        with pytest.raises(RuntimeError, match="Required quality gates failed"):
            await pipeline._execute_quality_gates(complex_config, progress)

        # Should have validation results showing failures
        failed_results = [
            r for r in progress.validation_results
            if r.status == ValidationStatus.FAILED
        ]
        assert len(failed_results) > 0

    @patch('asyncio.create_subprocess_exec')
    async def test_quality_gate_optional_failure(self, mock_subprocess, pipeline):
        """Test optional quality gate failure."""
        quality_gates = [
            QualityGate(
                name="optional-test",
                gate_type=QualityGateType.SMOKE_TEST,
                command=["python", "-m", "pytest", "tests/smoke/"],
                required=False  # Optional gate
            )
        ]

        targets = [DeploymentTarget(name="test", environment="test")]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets,
            quality_gates=quality_gates
        )

        # Mock failing command
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"smoke tests failed")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Should not raise exception for optional gate failure
        await pipeline._execute_quality_gates(config, progress)

        # Should have validation result showing failure
        assert len(progress.validation_results) == 1
        assert progress.validation_results[0].status == ValidationStatus.FAILED

    async def test_quality_gate_no_gates_configured(self, pipeline, simple_config):
        """Test quality gates execution with no gates configured."""
        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Should pass without exceptions
        await pipeline._execute_quality_gates(simple_config, progress)
        assert progress.validation_results == []

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    async def test_rolling_deployment_success(
        self,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline,
        simple_config
    ):
        """Test successful rolling deployment."""
        # Mock successful deployment steps
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        await pipeline._rolling_deployment(simple_config, progress, dry_run=True)

        # Should have completed the target
        assert "test-env" in progress.completed_targets
        assert len(progress.failed_targets) == 0
        assert progress.current_target is None

        # Verify all steps were called
        mock_deploy.assert_called_once()
        mock_wait.assert_called_once()
        mock_validate.assert_called_once()

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    async def test_rolling_deployment_failure(
        self,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline,
        simple_config
    ):
        """Test rolling deployment failure."""
        # Mock deployment failure
        mock_deploy.side_effect = Exception("Deployment failed")

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        with pytest.raises(Exception, match="Deployment failed"):
            await pipeline._rolling_deployment(simple_config, progress, dry_run=True)

        # Should have added target to failed list
        assert "test-env" in progress.failed_targets
        assert len(progress.completed_targets) == 0

    async def test_blue_green_deployment_invalid_targets(self, pipeline):
        """Test blue-green deployment with invalid number of targets."""
        # Only one target (should require exactly 2)
        targets = [DeploymentTarget(name="single", environment="test")]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.BLUE_GREEN,
            targets=targets
        )

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        with pytest.raises(ValueError, match="exactly 2 targets"):
            await pipeline._blue_green_deployment(config, progress, dry_run=True)

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    @patch.object(DeploymentPipeline, '_switch_traffic')
    async def test_blue_green_deployment_success(
        self,
        mock_switch,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline
    ):
        """Test successful blue-green deployment."""
        targets = [
            DeploymentTarget(name="blue", environment="blue"),
            DeploymentTarget(name="green", environment="green")
        ]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.BLUE_GREEN,
            targets=targets
        )

        # Mock successful deployment steps
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None
        mock_switch.return_value = None

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        await pipeline._blue_green_deployment(config, progress, dry_run=True)

        # Should have deployed to green and switched traffic
        assert "green" in progress.completed_targets
        assert progress.current_target is None

        mock_deploy.assert_called_once()
        mock_wait.assert_called_once()
        mock_validate.assert_called_once()
        mock_switch.assert_called_once()

    async def test_canary_deployment_insufficient_targets(self, pipeline):
        """Test canary deployment with insufficient targets."""
        targets = [DeploymentTarget(name="single", environment="test")]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=targets
        )

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        with pytest.raises(ValueError, match="at least 2 targets"):
            await pipeline._canary_deployment(config, progress, dry_run=True)

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    @patch.object(DeploymentPipeline, '_monitor_canary')
    async def test_canary_deployment_success(
        self,
        mock_monitor,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline
    ):
        """Test successful canary deployment."""
        targets = [
            DeploymentTarget(name="canary", environment="canary"),
            DeploymentTarget(name="prod1", environment="production"),
            DeploymentTarget(name="prod2", environment="production")
        ]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=targets
        )

        # Mock successful steps
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None
        mock_monitor.return_value = True  # Canary success

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        await pipeline._canary_deployment(config, progress, dry_run=True)

        # Should have deployed to all targets
        assert len(progress.completed_targets) == 3
        assert "canary" in progress.completed_targets
        assert "prod1" in progress.completed_targets
        assert "prod2" in progress.completed_targets

        # Should have called monitor canary
        mock_monitor.assert_called_once()

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    @patch.object(DeploymentPipeline, '_monitor_canary')
    async def test_canary_deployment_failure(
        self,
        mock_monitor,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline
    ):
        """Test canary deployment failure."""
        targets = [
            DeploymentTarget(name="canary", environment="canary"),
            DeploymentTarget(name="prod", environment="production")
        ]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=targets
        )

        # Mock successful deployment but failed monitoring
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None
        mock_monitor.return_value = False  # Canary failure

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        with pytest.raises(RuntimeError, match="Canary deployment failed validation"):
            await pipeline._canary_deployment(config, progress, dry_run=True)

        # Should have deployed canary but not production
        assert "canary" in progress.completed_targets
        assert len(progress.completed_targets) == 1

    @patch.object(DeploymentPipeline, '_stop_target')
    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    async def test_recreate_deployment_success(
        self,
        mock_validate,
        mock_wait,
        mock_deploy,
        mock_stop,
        pipeline,
        simple_config
    ):
        """Test successful recreate deployment."""
        # Mock successful steps
        mock_stop.return_value = None
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        await pipeline._recreate_deployment(simple_config, progress, dry_run=True)

        # Should have completed the target
        assert "test-env" in progress.completed_targets

        # In dry run mode, stop should not be called
        mock_stop.assert_not_called()
        mock_deploy.assert_called_once()

    async def test_ab_testing_deployment_invalid_targets(self, pipeline):
        """Test A/B testing deployment with invalid number of targets."""
        targets = [DeploymentTarget(name="single", environment="test")]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.A_B_TESTING,
            targets=targets
        )

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        with pytest.raises(ValueError, match="exactly 2 targets"):
            await pipeline._ab_testing_deployment(config, progress, dry_run=True)

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    @patch.object(DeploymentPipeline, '_configure_ab_traffic')
    async def test_ab_testing_deployment_success(
        self,
        mock_configure,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline
    ):
        """Test successful A/B testing deployment."""
        targets = [
            DeploymentTarget(name="variant-a", environment="production"),
            DeploymentTarget(name="variant-b", environment="production")
        ]
        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.A_B_TESTING,
            targets=targets
        )

        # Mock successful steps
        mock_deploy.return_value = None
        mock_wait.return_value = None
        mock_validate.return_value = None
        mock_configure.return_value = None

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        await pipeline._ab_testing_deployment(config, progress, dry_run=True)

        # Should have deployed to both variants
        assert len(progress.completed_targets) == 2
        assert "variant-a" in progress.completed_targets
        assert "variant-b" in progress.completed_targets

        # Should have configured A/B traffic
        mock_configure.assert_called_once()

    @patch('aiohttp.ClientSession.get')
    async def test_check_health_url_success(self, mock_get, pipeline):
        """Test successful health URL check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await pipeline._check_health_url("http://example.com/health")
        assert result is True

    @patch('aiohttp.ClientSession.get')
    async def test_check_health_url_failure(self, mock_get, pipeline):
        """Test failed health URL check."""
        mock_get.side_effect = Exception("Connection failed")

        result = await pipeline._check_health_url("http://example.com/health")
        assert result is False

    @patch('aiohttp.ClientSession.get')
    async def test_check_readiness_probe_http_success(self, mock_get, pipeline):
        """Test successful HTTP readiness probe."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "http",
                "url": "http://test.example.com/ready",
                "expected_status": 200
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is True

    @patch('aiohttp.ClientSession.get')
    async def test_check_readiness_probe_http_wrong_status(self, mock_get, pipeline):
        """Test HTTP readiness probe with wrong status code."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_get.return_value.__aenter__.return_value = mock_response

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "http",
                "url": "http://test.example.com/ready",
                "expected_status": 200
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is False

    @patch('asyncio.open_connection')
    async def test_check_readiness_probe_tcp_success(self, mock_open_connection, pipeline):
        """Test successful TCP readiness probe."""
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_writer.wait_closed = AsyncMock()
        mock_open_connection.return_value = (mock_reader, mock_writer)

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "tcp",
                "host": "test.example.com",
                "port": 8080
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is True
        mock_writer.close.assert_called_once()

    @patch('asyncio.open_connection')
    async def test_check_readiness_probe_tcp_failure(self, mock_open_connection, pipeline):
        """Test failed TCP readiness probe."""
        mock_open_connection.side_effect = ConnectionRefusedError("Connection refused")

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "tcp",
                "host": "test.example.com",
                "port": 8080
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is False

    @patch.object(DeploymentPipeline, '_execute_command')
    async def test_check_readiness_probe_command_success(self, mock_execute, pipeline):
        """Test successful command readiness probe."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_execute.return_value = mock_result

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "command",
                "command": ["curl", "-f", "http://test/health"],
                "expected_exit_code": 0
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is True

    @patch.object(DeploymentPipeline, '_execute_command')
    async def test_check_readiness_probe_command_failure(self, mock_execute, pipeline):
        """Test failed command readiness probe."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_execute.return_value = mock_result

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={
                "type": "command",
                "command": ["curl", "-f", "http://test/health"],
                "expected_exit_code": 0
            }
        )

        result = await pipeline._check_readiness_probe(target)
        assert result is False

    async def test_check_readiness_probe_no_probe(self, pipeline):
        """Test readiness probe check with no probe configured."""
        target = DeploymentTarget(name="test", environment="test")
        result = await pipeline._check_readiness_probe(target)
        assert result is True  # Should return True if no probe configured

    @patch.object(DeploymentPipeline, '_check_readiness_probe')
    async def test_wait_for_readiness_success(self, mock_probe, pipeline):
        """Test successful wait for readiness."""
        mock_probe.return_value = True

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={"type": "http", "url": "http://test/ready"}
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        # Should complete quickly
        await pipeline._wait_for_readiness(target, config, max_wait=30)
        mock_probe.assert_called()

    @patch.object(DeploymentPipeline, '_check_readiness_probe')
    async def test_wait_for_readiness_timeout(self, mock_probe, pipeline):
        """Test wait for readiness timeout."""
        mock_probe.return_value = False  # Never becomes ready

        target = DeploymentTarget(
            name="test",
            environment="test",
            readiness_probe={"type": "http", "url": "http://test/ready"}
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        with pytest.raises(RuntimeError, match="failed to become ready"):
            await pipeline._wait_for_readiness(target, config, max_wait=1)

    @patch.object(DeploymentPipeline, '_check_health_url')
    async def test_wait_for_readiness_health_url(self, mock_health, pipeline):
        """Test wait for readiness using health URL."""
        mock_health.return_value = True

        target = DeploymentTarget(
            name="test",
            environment="test",
            health_check_url="http://test.example.com/health"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        await pipeline._wait_for_readiness(target, config, max_wait=30)
        mock_health.assert_called()

    @patch.object(DeploymentPipeline, '_collect_canary_metrics')
    @patch.object(DeploymentPipeline, '_check_health_url')
    async def test_monitor_canary_success(self, mock_health, mock_metrics, pipeline):
        """Test successful canary monitoring."""
        mock_health.return_value = True
        mock_metrics.return_value = {
            "error_rate": 1.0,
            "response_time": 150.0,
            "success_rate": 99.0
        }

        target = DeploymentTarget(
            name="canary",
            environment="canary",
            health_check_url="http://canary.example.com/health"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=[target]
        )

        result = await pipeline._monitor_canary(target, config, monitor_duration=1)
        assert result is True

    @patch.object(DeploymentPipeline, '_collect_canary_metrics')
    @patch.object(DeploymentPipeline, '_check_health_url')
    async def test_monitor_canary_high_error_rate(self, mock_health, mock_metrics, pipeline):
        """Test canary monitoring with high error rate."""
        mock_health.return_value = True
        mock_metrics.return_value = {
            "error_rate": 10.0,  # High error rate
            "response_time": 150.0,
            "success_rate": 90.0
        }

        target = DeploymentTarget(
            name="canary",
            environment="canary",
            health_check_url="http://canary.example.com/health"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=[target]
        )

        result = await pipeline._monitor_canary(target, config, monitor_duration=1)
        assert result is False

    @patch.object(DeploymentPipeline, '_check_health_url')
    async def test_monitor_canary_health_failure(self, mock_health, pipeline):
        """Test canary monitoring with health check failure."""
        mock_health.return_value = False  # Health check fails

        target = DeploymentTarget(
            name="canary",
            environment="canary",
            health_check_url="http://canary.example.com/health"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=[target]
        )

        result = await pipeline._monitor_canary(target, config, monitor_duration=1)
        assert result is False

    async def test_collect_canary_metrics(self, pipeline):
        """Test canary metrics collection."""
        target = DeploymentTarget(name="canary", environment="canary")

        metrics = await pipeline._collect_canary_metrics(target)

        # Should return mock metrics
        assert "error_rate" in metrics
        assert "response_time" in metrics
        assert "success_rate" in metrics
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics

    async def test_generate_kubernetes_commands(self, pipeline, temp_dir):
        """Test Kubernetes deployment command generation."""
        pipeline.config_dir = temp_dir

        # Create kubernetes manifest directory
        k8s_dir = temp_dir / "kubernetes" / "production"
        k8s_dir.mkdir(parents=True)

        # Create a sample manifest
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test-app"}
        }
        with open(k8s_dir / "deployment.yaml", "w") as f:
            json.dump(manifest, f)

        target = DeploymentTarget(
            name="production",
            environment="production",
            infrastructure_config="kubernetes"
        )

        config = DeploymentConfig(
            name="test-app",
            version="2.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        commands = await pipeline._generate_kubernetes_commands(target, config)

        assert len(commands) >= 3  # apply, set image, rollout status
        assert any("kubectl" in cmd for cmd in commands)
        assert any("apply" in cmd for cmd in commands)
        assert any("set" in cmd and "image" in cmd for cmd in commands)

    async def test_generate_docker_commands(self, pipeline, temp_dir):
        """Test Docker deployment command generation."""
        pipeline.config_dir = temp_dir

        # Create docker compose file
        docker_dir = temp_dir / "docker" / "staging"
        docker_dir.mkdir(parents=True)

        compose_config = {
            "version": "3.8",
            "services": {
                "test-app": {
                    "image": "test-app:latest"
                }
            }
        }

        with open(docker_dir / "docker-compose.yml", "w") as f:
            json.dump(compose_config, f)

        target = DeploymentTarget(
            name="staging",
            environment="staging",
            infrastructure_config="docker"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.5.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        commands = await pipeline._generate_docker_commands(target, config)

        assert len(commands) >= 2  # pull and up commands
        assert any("docker-compose" in cmd for cmd in commands)
        assert any("pull" in cmd for cmd in commands)
        assert any("up" in cmd for cmd in commands)

    async def test_generate_terraform_commands(self, pipeline, temp_dir):
        """Test Terraform deployment command generation."""
        pipeline.config_dir = temp_dir

        # Create terraform directory
        tf_dir = temp_dir / "terraform" / "production"
        tf_dir.mkdir(parents=True)

        # Create a sample terraform file
        with open(tf_dir / "main.tf", "w") as f:
            f.write('resource "aws_instance" "web" {}')

        target = DeploymentTarget(
            name="production",
            environment="production",
            infrastructure_config="terraform"
        )

        config = DeploymentConfig(
            name="test-app",
            version="3.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        commands = await pipeline._generate_terraform_commands(target, config)

        assert len(commands) >= 3  # init, plan, apply
        assert any("terraform" in cmd for cmd in commands)
        assert any("init" in cmd for cmd in commands)
        assert any("plan" in cmd for cmd in commands)
        assert any("apply" in cmd for cmd in commands)

    async def test_generate_generic_commands(self, pipeline, temp_dir):
        """Test generic deployment command generation."""
        pipeline.config_dir = temp_dir

        # Create scripts directory and deployment script
        scripts_dir = temp_dir / "scripts"
        scripts_dir.mkdir(parents=True)

        deploy_script = scripts_dir / "deploy-test.sh"
        deploy_script.write_text("#!/bin/bash\necho 'Deploying $1 $2 to $3'")
        deploy_script.chmod(0o755)

        target = DeploymentTarget(
            name="test-target",
            environment="test"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[target]
        )

        commands = await pipeline._generate_generic_commands(target, config)

        assert len(commands) == 1
        assert commands[0][0] == "bash"
        assert str(deploy_script) in commands[0]
        assert "test-app" in commands[0]
        assert "1.0.0" in commands[0]

    @patch.object(DeploymentPipeline, '_execute_deployment_strategy')
    async def test_execute_rollback(self, mock_deploy_strategy, pipeline, simple_config):
        """Test rollback execution."""
        # Add rollback configuration
        simple_config.rollback_config = {
            "auto_rollback": True,
            "previous_version": "0.9.0"
        }

        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.FAILED,
            start_time=time.time()
        )

        await pipeline._execute_rollback(simple_config, progress, "Test failure")

        # Should have updated progress
        assert progress.status == DeploymentStatus.ROLLED_BACK
        assert progress.rollback_reason == "Test failure"

        # Should have called deployment strategy with rollback config
        mock_deploy_strategy.assert_called_once()
        call_args = mock_deploy_strategy.call_args[0]
        rollback_config = call_args[0]
        assert rollback_config.version == "0.9.0"

    async def test_execute_rollback_no_previous_version(self, pipeline, simple_config):
        """Test rollback execution without previous version."""
        # No rollback configuration
        progress = DeploymentProgress(
            deployment_id="test-123",
            status=DeploymentStatus.FAILED,
            start_time=time.time()
        )

        await pipeline._execute_rollback(simple_config, progress, "Test failure")

        # Should still update status but not execute rollback
        assert progress.status == DeploymentStatus.ROLLING_BACK

    @patch.object(DeploymentPipeline, '_pre_deployment_validation')
    @patch.object(DeploymentPipeline, '_execute_quality_gates')
    @patch.object(DeploymentPipeline, '_execute_deployment_strategy')
    @patch.object(DeploymentPipeline, '_post_deployment_validation')
    async def test_deploy_success_dry_run(
        self,
        mock_post_validation,
        mock_deploy_strategy,
        mock_quality_gates,
        mock_pre_validation,
        pipeline,
        simple_config
    ):
        """Test successful deployment in dry run mode."""
        deployment_id = await pipeline.deploy(simple_config, dry_run=True)

        assert deployment_id.startswith(f"{simple_config.name}-{simple_config.version}")
        assert deployment_id not in pipeline.active_deployments  # Moved to history
        assert len(pipeline.deployment_history) == 1

        # Verify all steps were called
        mock_pre_validation.assert_called_once()
        mock_quality_gates.assert_called_once()
        mock_deploy_strategy.assert_called_once()
        mock_post_validation.assert_called_once()

        # Check deployment was marked as completed
        deployment = pipeline.deployment_history[0]
        assert deployment.status == DeploymentStatus.COMPLETED

    @patch.object(DeploymentPipeline, '_pre_deployment_validation')
    @patch.object(DeploymentPipeline, '_execute_quality_gates')
    @patch.object(DeploymentPipeline, '_execute_deployment_strategy')
    async def test_deploy_failure_with_rollback(
        self,
        mock_deploy_strategy,
        mock_quality_gates,
        mock_pre_validation,
        pipeline,
        simple_config
    ):
        """Test deployment failure with automatic rollback."""
        # Configure automatic rollback
        simple_config.rollback_config = {
            "auto_rollback": True,
            "previous_version": "0.9.0"
        }

        # Mock deployment strategy failure
        mock_deploy_strategy.side_effect = Exception("Deployment failed")

        with pytest.raises(Exception, match="Deployment failed"):
            await pipeline.deploy(simple_config)

        # Check deployment was marked as failed
        assert len(pipeline.deployment_history) == 1
        deployment = pipeline.deployment_history[0]
        assert deployment.status == DeploymentStatus.FAILED
        assert deployment.error_message == "Deployment failed"

    @patch.object(DeploymentPipeline, '_execute_quality_gates')
    async def test_deploy_force_skip_quality_gates(
        self,
        mock_quality_gates,
        pipeline,
        simple_config
    ):
        """Test deployment with force flag skips quality gates."""
        # Add quality gates to config
        simple_config.quality_gates = [
            QualityGate(
                name="test",
                gate_type=QualityGateType.UNIT_TESTS,
                command=["echo", "test"]
            )
        ]

        with patch.object(pipeline, '_execute_deployment_strategy'):
            with patch.object(pipeline, '_post_deployment_validation'):
                await pipeline.deploy(simple_config, force=True, dry_run=True)

        # Quality gates should not have been executed
        mock_quality_gates.assert_not_called()

    def test_get_deployment_status_active(self, pipeline):
        """Test getting status of active deployment."""
        progress = DeploymentProgress(
            deployment_id="active-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        pipeline.active_deployments["active-123"] = progress

        result = pipeline.get_deployment_status("active-123")
        assert result == progress

    def test_get_deployment_status_history(self, pipeline):
        """Test getting status from deployment history."""
        progress = DeploymentProgress(
            deployment_id="history-123",
            status=DeploymentStatus.COMPLETED,
            start_time=time.time()
        )
        pipeline.deployment_history.append(progress)

        result = pipeline.get_deployment_status("history-123")
        assert result == progress

    def test_get_deployment_status_not_found(self, pipeline):
        """Test getting status of non-existent deployment."""
        result = pipeline.get_deployment_status("not-found")
        assert result is None

    def test_list_active_deployments(self, pipeline):
        """Test listing active deployments."""
        progress1 = DeploymentProgress(
            deployment_id="active-1",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        progress2 = DeploymentProgress(
            deployment_id="active-2",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        pipeline.active_deployments["active-1"] = progress1
        pipeline.active_deployments["active-2"] = progress2

        result = pipeline.list_active_deployments()
        assert set(result) == {"active-1", "active-2"}

    def test_get_deployment_history(self, pipeline):
        """Test getting deployment history."""
        for i in range(60):  # More than default limit
            progress = DeploymentProgress(
                deployment_id=f"deploy-{i}",
                status=DeploymentStatus.COMPLETED,
                start_time=time.time()
            )
            pipeline.deployment_history.append(progress)

        # Default limit
        result = pipeline.get_deployment_history()
        assert len(result) == 50
        assert result[-1].deployment_id == "deploy-59"  # Most recent

        # Custom limit
        result = pipeline.get_deployment_history(limit=10)
        assert len(result) == 10

    def test_cancel_deployment_success(self, pipeline):
        """Test successful deployment cancellation."""
        progress = DeploymentProgress(
            deployment_id="cancel-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        pipeline.active_deployments["cancel-123"] = progress

        result = pipeline.cancel_deployment("cancel-123", "Test cancellation")

        assert result is True
        assert "cancel-123" not in pipeline.active_deployments
        assert len(pipeline.deployment_history) == 1

        cancelled_deployment = pipeline.deployment_history[0]
        assert cancelled_deployment.status == DeploymentStatus.CANCELLED
        assert cancelled_deployment.error_message == "Test cancellation"

    def test_cancel_deployment_not_found(self, pipeline):
        """Test cancelling non-existent deployment."""
        result = pipeline.cancel_deployment("not-found")
        assert result is False

    def test_generate_deployment_report(self, pipeline):
        """Test deployment report generation."""
        progress = DeploymentProgress(
            deployment_id="report-123",
            status=DeploymentStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 300,
            duration=300.0
        )
        progress.completed_targets = ["target1", "target2"]
        progress.failed_targets = []
        progress.validation_results = [
            ValidationResult(
                gate_name="unit-tests",
                status=ValidationStatus.PASSED,
                start_time=time.time(),
                end_time=time.time() + 30,
                duration=30.0,
                message="Tests passed",
                metrics={"coverage": 85.0}
            )
        ]

        pipeline.deployment_history.append(progress)

        report = pipeline.generate_deployment_report("report-123")

        assert report["deployment_id"] == "report-123"
        assert report["status"] == "completed"
        assert report["duration"] == 300.0
        assert len(report["targets"]["completed"]) == 2
        assert len(report["quality_gates"]) == 1
        assert report["quality_gates"][0]["status"] == "passed"

    def test_generate_deployment_report_not_found(self, pipeline):
        """Test deployment report generation for non-existent deployment."""
        with pytest.raises(ValueError, match="not found"):
            pipeline.generate_deployment_report("not-found")

    def test_print_deployment_status_table(self, pipeline):
        """Test printing deployment status table."""
        progress = DeploymentProgress(
            deployment_id="table-test-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        progress.completed_targets = ["target1"]
        progress.current_target = "target2"

        pipeline.active_deployments["table-test-123"] = progress

        # Should not raise exception (output testing is complex with Rich)
        pipeline.print_deployment_status_table()


class TestComplexDeploymentScenarios:
    """Test complex deployment scenarios and edge cases."""

    @pytest.fixture
    def complex_multi_stage_config(self):
        """Create complex multi-stage deployment configuration."""
        quality_gates = [
            QualityGate(
                name="lint-check",
                gate_type=QualityGateType.CODE_QUALITY,
                command=["ruff", "check", "src/"],
                timeout=120,
                required=True
            ),
            QualityGate(
                name="unit-tests",
                gate_type=QualityGateType.UNIT_TESTS,
                command=["python", "-m", "pytest", "tests/unit/", "--cov"],
                timeout=300,
                required=True,
                success_criteria={"min_coverage": 80.0}
            ),
            QualityGate(
                name="integration-tests",
                gate_type=QualityGateType.INTEGRATION_TESTS,
                command=["python", "-m", "pytest", "tests/integration/"],
                timeout=600,
                required=True
            ),
            QualityGate(
                name="security-scan",
                gate_type=QualityGateType.SECURITY_SCAN,
                command=["bandit", "-r", "src/", "-f", "json"],
                timeout=180,
                required=True,
                success_criteria={"max_critical_vulnerabilities": 0}
            ),
            QualityGate(
                name="performance-test",
                gate_type=QualityGateType.PERFORMANCE_TEST,
                command=["artillery", "run", "perf-test.yml"],
                timeout=900,
                required=False,
                environment_vars={"TARGET_URL": "http://staging.example.com"},
                success_criteria={
                    "max_response_time": 500.0,
                    "min_throughput": 1000.0,
                    "max_error_rate": 1.0
                }
            ),
            QualityGate(
                name="smoke-tests",
                gate_type=QualityGateType.SMOKE_TEST,
                command=["python", "-m", "pytest", "tests/smoke/"],
                timeout=300,
                required=False
            )
        ]

        targets = [
            DeploymentTarget(
                name="development",
                environment="dev",
                infrastructure_config="docker",
                health_check_url="http://dev.example.com/health",
                readiness_probe={
                    "type": "http",
                    "url": "http://dev.example.com/ready",
                    "timeout": 10
                }
            ),
            DeploymentTarget(
                name="staging",
                environment="staging",
                infrastructure_config="kubernetes",
                health_check_url="http://staging.example.com/health",
                readiness_probe={
                    "type": "http",
                    "url": "http://staging.example.com/ready",
                    "timeout": 15
                },
                liveness_probe={
                    "type": "http",
                    "url": "http://staging.example.com/health",
                    "timeout": 5
                }
            ),
            DeploymentTarget(
                name="production-us-east",
                environment="production",
                infrastructure_config="kubernetes",
                health_check_url="http://us-east.example.com/health",
                readiness_probe={
                    "type": "http",
                    "url": "http://us-east.example.com/ready",
                    "timeout": 20
                },
                liveness_probe={
                    "type": "http",
                    "url": "http://us-east.example.com/health",
                    "timeout": 10
                }
            ),
            DeploymentTarget(
                name="production-eu-west",
                environment="production",
                infrastructure_config="kubernetes",
                health_check_url="http://eu-west.example.com/health",
                readiness_probe={
                    "type": "http",
                    "url": "http://eu-west.example.com/ready",
                    "timeout": 20
                },
                liveness_probe={
                    "type": "http",
                    "url": "http://eu-west.example.com/health",
                    "timeout": 10
                }
            )
        ]

        return DeploymentConfig(
            name="enterprise-app",
            version="2.1.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=targets,
            quality_gates=quality_gates,
            rollback_config={
                "auto_rollback": True,
                "previous_version": "2.0.5",
                "rollback_timeout": 900
            },
            timeout=3600,  # 1 hour for complex deployment
            parallel_deployments=False,
            deployment_variables={
                "ENVIRONMENT": "production",
                "LOG_LEVEL": "INFO",
                "FEATURE_FLAGS": "advanced_search,real_time_updates",
                "CACHE_TTL": "3600"
            },
            secrets=["DATABASE_PASSWORD", "API_KEYS", "JWT_SECRET", "ENCRYPTION_KEY"]
        )

    @pytest.fixture
    def pipeline_with_temp_dir(self):
        """Create pipeline with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DeploymentPipeline(
                artifacts_dir=Path(tmpdir) / "artifacts",
                logs_dir=Path(tmpdir) / "logs",
                config_dir=Path(tmpdir) / "config"
            )

    @patch('asyncio.create_subprocess_exec')
    async def test_complex_quality_gates_mixed_results(
        self,
        mock_subprocess,
        pipeline_with_temp_dir,
        complex_multi_stage_config
    ):
        """Test complex quality gates with mixed pass/fail results."""
        pipeline = pipeline_with_temp_dir

        # Mock different results for different gates
        def mock_command_side_effect(*args, **kwargs):
            command = args[0]
            mock_process = AsyncMock()

            if "ruff" in command:
                # Lint check passes
                mock_process.communicate.return_value = (b"No issues found", b"")
                mock_process.returncode = 0
            elif "pytest" in command and "unit" in command:
                # Unit tests pass
                mock_process.communicate.return_value = (b"85% coverage", b"")
                mock_process.returncode = 0
            elif "pytest" in command and "integration" in command:
                # Integration tests pass
                mock_process.communicate.return_value = (b"All tests passed", b"")
                mock_process.returncode = 0
            elif "bandit" in command:
                # Security scan fails (required gate)
                mock_process.communicate.return_value = (b"", b"Critical vulnerabilities found")
                mock_process.returncode = 1
            elif "artillery" in command:
                # Performance test fails (optional gate)
                mock_process.communicate.return_value = (b"", b"Performance targets not met")
                mock_process.returncode = 1
            elif "pytest" in command and "smoke" in command:
                # Smoke tests pass
                mock_process.communicate.return_value = (b"Smoke tests passed", b"")
                mock_process.returncode = 0
            else:
                mock_process.communicate.return_value = (b"", b"Unknown command")
                mock_process.returncode = 1

            return mock_process

        mock_subprocess.side_effect = mock_command_side_effect

        progress = DeploymentProgress(
            deployment_id="complex-123",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Should fail due to required security gate failure
        with pytest.raises(RuntimeError, match="Required quality gates failed"):
            await pipeline._execute_quality_gates(complex_multi_stage_config, progress)

        # Check validation results
        assert len(progress.validation_results) == 6

        passed_gates = [r for r in progress.validation_results if r.status == ValidationStatus.PASSED]
        failed_gates = [r for r in progress.validation_results if r.status == ValidationStatus.FAILED]

        # Should have both passed and failed gates
        assert len(passed_gates) > 0
        assert len(failed_gates) > 0

        # Security scan should be in failed gates (required)
        security_result = next(
            (r for r in progress.validation_results if r.gate_name == "security-scan"),
            None
        )
        assert security_result is not None
        assert security_result.status == ValidationStatus.FAILED

    @patch.object(DeploymentPipeline, '_deploy_to_target')
    @patch.object(DeploymentPipeline, '_wait_for_readiness')
    @patch.object(DeploymentPipeline, '_validate_target_deployment')
    async def test_multi_target_rolling_deployment_partial_failure(
        self,
        mock_validate,
        mock_wait,
        mock_deploy,
        pipeline_with_temp_dir,
        complex_multi_stage_config
    ):
        """Test multi-target rolling deployment with partial failure."""
        pipeline = pipeline_with_temp_dir

        # Mock deployment success for first 2 targets, failure for 3rd
        call_count = [0]

        def mock_deploy_side_effect(target, config, dry_run):
            call_count[0] += 1
            if call_count[0] <= 2:
                return None  # Success
            else:
                raise Exception(f"Deployment failed for {target.name}")

        mock_deploy.side_effect = mock_deploy_side_effect
        mock_wait.return_value = None
        mock_validate.return_value = None

        progress = DeploymentProgress(
            deployment_id="multi-target-123",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )

        # Should fail on the 3rd target
        with pytest.raises(Exception, match="Deployment failed for production-us-east"):
            await pipeline._rolling_deployment(complex_multi_stage_config, progress)

        # Should have completed first 2 targets
        assert len(progress.completed_targets) == 2
        assert "development" in progress.completed_targets
        assert "staging" in progress.completed_targets

        # Should have failed on 3rd target
        assert len(progress.failed_targets) == 1
        assert "production-us-east" in progress.failed_targets

    @patch.object(DeploymentPipeline, '_check_health_url')
    @patch.object(DeploymentPipeline, '_check_readiness_probe')
    async def test_complex_health_checks_timeout_scenarios(
        self,
        mock_readiness,
        mock_health,
        pipeline_with_temp_dir,
        complex_multi_stage_config
    ):
        """Test complex health checks with various timeout scenarios."""
        pipeline = pipeline_with_temp_dir

        # Mock different readiness behaviors
        call_count = [0]

        def mock_readiness_side_effect(target):
            call_count[0] += 1
            # First few calls fail, then succeed (simulating slow startup)
            return call_count[0] > 3

        mock_readiness.side_effect = mock_readiness_side_effect
        mock_health.return_value = True

        target = complex_multi_stage_config.targets[1]  # Staging target
        config = complex_multi_stage_config

        # Should eventually succeed
        await pipeline._wait_for_readiness(target, config, max_wait=30)

        # Should have called readiness probe multiple times
        assert mock_readiness.call_count >= 4

    @patch.object(DeploymentPipeline, '_collect_canary_metrics')
    @patch.object(DeploymentPipeline, '_check_health_url')
    async def test_canary_monitoring_edge_cases(
        self,
        mock_health,
        mock_metrics,
        pipeline_with_temp_dir
    ):
        """Test canary monitoring with various edge case scenarios."""
        pipeline = pipeline_with_temp_dir

        # Mock fluctuating metrics
        metrics_calls = [0]

        def mock_metrics_side_effect(target):
            metrics_calls[0] += 1
            call_number = metrics_calls[0]

            if call_number <= 2:
                # Good metrics initially
                return {
                    "error_rate": 1.0,
                    "response_time": 150.0,
                    "success_rate": 99.0
                }
            elif call_number <= 4:
                # Degrading metrics
                return {
                    "error_rate": 3.0,
                    "response_time": 400.0,
                    "success_rate": 97.0
                }
            else:
                # Critical metrics (should trigger failure)
                return {
                    "error_rate": 8.0,  # Above 5% threshold
                    "response_time": 3000.0,  # Above 2s threshold
                    "success_rate": 92.0
                }

        mock_metrics.side_effect = mock_metrics_side_effect
        mock_health.return_value = True

        target = DeploymentTarget(
            name="canary-edge",
            environment="canary",
            health_check_url="http://canary.example.com/health"
        )

        config = DeploymentConfig(
            name="test-app",
            version="1.0.0",
            strategy=DeploymentStrategy.CANARY,
            targets=[target]
        )

        # Should fail due to high error rate
        result = await pipeline._monitor_canary(target, config, monitor_duration=5)
        assert result is False

        # Should have collected metrics multiple times
        assert metrics_calls[0] >= 3

    async def test_concurrent_deployment_conflict_detection(self, pipeline_with_temp_dir):
        """Test detection of conflicting concurrent deployments."""
        pipeline = pipeline_with_temp_dir

        # Create first deployment
        config1 = DeploymentConfig(
            name="app-1",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[DeploymentTarget(name="shared-target", environment="prod")]
        )

        progress1 = DeploymentProgress(
            deployment_id="deploy-1",
            status=DeploymentStatus.RUNNING,
            start_time=time.time()
        )
        progress1.current_target = "shared-target"
        pipeline.active_deployments["deploy-1"] = progress1

        # Create second deployment targeting same environment
        config2 = DeploymentConfig(
            name="app-2",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[DeploymentTarget(name="shared-target", environment="prod")]
        )

        progress2 = DeploymentProgress(
            deployment_id="deploy-2",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Should detect conflict
        with pytest.raises(RuntimeError, match="Deployment conflict"):
            await pipeline._pre_deployment_validation(config2, progress2)

    async def test_deployment_artifact_validation(self, pipeline_with_temp_dir):
        """Test validation of deployment artifacts."""
        pipeline = pipeline_with_temp_dir

        config = DeploymentConfig(
            name="artifact-app",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[DeploymentTarget(name="test", environment="test")]
        )

        progress = DeploymentProgress(
            deployment_id="artifact-test",
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        # Add non-existent artifact
        progress.deployment_artifacts = ["non-existent-artifact.tar.gz"]

        # Should fail validation
        with pytest.raises(FileNotFoundError, match="Deployment artifact not found"):
            await pipeline._pre_deployment_validation(config, progress)

    async def test_deployment_timeout_handling(self, pipeline_with_temp_dir):
        """Test deployment timeout handling."""
        pipeline = pipeline_with_temp_dir

        config = DeploymentConfig(
            name="timeout-test",
            version="1.0.0",
            strategy=DeploymentStrategy.ROLLING,
            targets=[DeploymentTarget(name="slow-target", environment="test")],
            timeout=1  # Very short timeout
        )

        with patch.object(pipeline, '_execute_deployment_strategy') as mock_deploy:
            # Mock slow deployment
            async def slow_deploy(*args):
                await asyncio.sleep(2)  # Longer than timeout

            mock_deploy.side_effect = slow_deploy

            with patch.object(pipeline, '_execute_quality_gates'):
                # Should handle timeout gracefully
                with pytest.raises(Exception):  # Would be TimeoutError in real implementation
                    await pipeline.deploy(config, dry_run=True)

    def test_example_configuration_creation(self):
        """Test creation of example web app deployment configuration."""
        config = create_example_web_app_deployment()

        assert config.name == "workspace-qdrant-mcp"
        assert config.version == "0.3.0"
        assert config.strategy == DeploymentStrategy.ROLLING
        assert len(config.targets) == 2
        assert len(config.quality_gates) == 4

        # Verify quality gates
        gate_types = {gate.gate_type for gate in config.quality_gates}
        assert QualityGateType.UNIT_TESTS in gate_types
        assert QualityGateType.INTEGRATION_TESTS in gate_types
        assert QualityGateType.SECURITY_SCAN in gate_types
        assert QualityGateType.SMOKE_TEST in gate_types

        # Verify rollback configuration
        assert config.rollback_config["auto_rollback"] is True
        assert config.rollback_config["previous_version"] == "0.2.1"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])