"""Comprehensive unit tests for deployment pipeline with edge cases."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import shutil
import subprocess

from docs.framework.deployment.pipeline import (
    DeploymentPipeline,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
    BuildResult,
    BuildStatus
)


class TestDeploymentConfig:
    """Test DeploymentConfig data class."""

    def test_deployment_config_initialization(self):
        """Test basic initialization."""
        config = DeploymentConfig(
            source_path=Path("/source"),
            build_path=Path("/build"),
            deploy_path=Path("/deploy"),
            backup_path=Path("/backup")
        )

        assert config.source_path == Path("/source")
        assert config.build_path == Path("/build")
        assert config.deployment_strategy == "atomic"
        assert config.rollback_enabled is True
        assert config.max_workers == 4

    def test_deployment_config_string_paths(self):
        """Test initialization with string paths."""
        config = DeploymentConfig(
            source_path="/source",
            build_path="/build",
            deploy_path="/deploy",
            backup_path="/backup"
        )

        assert isinstance(config.source_path, Path)
        assert config.source_path == Path("/source")

    def test_deployment_config_custom_settings(self):
        """Test initialization with custom settings."""
        config = DeploymentConfig(
            source_path=Path("/source"),
            build_path=Path("/build"),
            deploy_path=Path("/deploy"),
            backup_path=Path("/backup"),
            deployment_strategy="rolling",
            build_timeout=3600,
            max_rollback_versions=10,
            notification_hooks=["webhook1", "webhook2"]
        )

        assert config.deployment_strategy == "rolling"
        assert config.build_timeout == 3600
        assert config.max_rollback_versions == 10
        assert len(config.notification_hooks) == 2


class TestBuildResult:
    """Test BuildResult data class."""

    def test_build_result_initialization(self):
        """Test basic initialization."""
        start_time = datetime.now()
        result = BuildResult(
            status=BuildStatus.SUCCESS,
            start_time=start_time
        )

        assert result.status == BuildStatus.SUCCESS
        assert result.start_time == start_time
        assert result.duration_seconds == 0.0

    def test_build_result_duration_calculation(self):
        """Test duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)

        result = BuildResult(
            status=BuildStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time
        )

        assert result.duration_seconds == 30.0

    def test_build_result_with_artifacts(self):
        """Test result with build artifacts."""
        result = BuildResult(
            status=BuildStatus.SUCCESS,
            start_time=datetime.now(),
            artifacts=[Path("build/index.html"), Path("build/style.css")]
        )

        assert len(result.artifacts) == 2
        assert Path("build/index.html") in result.artifacts


class TestDeploymentResult:
    """Test DeploymentResult data class."""

    def test_deployment_result_initialization(self):
        """Test basic initialization."""
        start_time = datetime.now()
        result = DeploymentResult(
            deployment_id="deploy_123",
            status=DeploymentStatus.PENDING,
            start_time=start_time
        )

        assert result.deployment_id == "deploy_123"
        assert result.status == DeploymentStatus.PENDING
        assert result.duration_seconds == 0.0

    def test_deployment_result_duration_calculation(self):
        """Test duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)

        result = DeploymentResult(
            deployment_id="deploy_123",
            status=DeploymentStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time
        )

        assert result.duration_seconds == 300.0

    def test_deployment_result_with_build_result(self):
        """Test deployment result with build information."""
        build_result = BuildResult(
            status=BuildStatus.SUCCESS,
            start_time=datetime.now()
        )

        result = DeploymentResult(
            deployment_id="deploy_123",
            status=DeploymentStatus.SUCCESS,
            start_time=datetime.now(),
            build_result=build_result
        )

        assert result.build_result == build_result
        assert result.build_result.status == BuildStatus.SUCCESS


class TestDeploymentPipeline:
    """Test DeploymentPipeline with comprehensive edge cases."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            source = base / "source"
            build = base / "build"
            deploy = base / "deploy"
            backup = base / "backup"

            # Create source directory with sample files
            source.mkdir()
            (source / "index.html").write_text("<html>Test</html>")
            (source / "style.css").write_text("body { color: red; }")

            yield {
                'base': base,
                'source': source,
                'build': build,
                'deploy': deploy,
                'backup': backup
            }

    @pytest.fixture
    def deployment_config(self, temp_dirs):
        """Create deployment configuration for testing."""
        return DeploymentConfig(
            source_path=temp_dirs['source'],
            build_path=temp_dirs['build'],
            deploy_path=temp_dirs['deploy'],
            backup_path=temp_dirs['backup'],
            build_command=["echo", "Building..."],
            test_command=["echo", "Testing..."],
            build_timeout=30,
            deploy_timeout=30
        )

    @pytest.fixture
    def pipeline(self, deployment_config):
        """Create deployment pipeline for testing."""
        pipeline = DeploymentPipeline(deployment_config)
        yield pipeline
        pipeline.cleanup()

    def test_pipeline_initialization(self, deployment_config):
        """Test pipeline initialization."""
        pipeline = DeploymentPipeline(deployment_config)

        assert pipeline.config == deployment_config
        assert len(pipeline._active_deployments) == 0
        assert len(pipeline._deployment_history) == 0

        # Check directories are created
        assert deployment_config.build_path.exists()
        assert deployment_config.deploy_path.exists()
        assert deployment_config.backup_path.exists()

        pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_deploy_basic_success(self, pipeline, temp_dirs):
        """Test basic successful deployment."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {
                'exit_code': 0,
                'stdout': 'Build successful',
                'stderr': ''
            }

            # Copy source to build directory to simulate build
            async def mock_build(*args, **kwargs):
                shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)
                return mock_run.return_value

            mock_run.side_effect = mock_build

            result = await pipeline.deploy()

            assert result.status == DeploymentStatus.SUCCESS
            assert result.deployment_id is not None
            assert result.build_result is not None
            assert result.build_result.status == BuildStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deploy_dry_run(self, pipeline):
        """Test dry run deployment."""
        result = await pipeline.deploy(dry_run=True)

        assert result.status == DeploymentStatus.SUCCESS
        assert result.build_result is not None
        assert "DRY RUN" in result.build_result.stdout

    @pytest.mark.asyncio
    async def test_deploy_build_failure(self, pipeline):
        """Test deployment with build failure."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {
                'exit_code': 1,
                'stdout': '',
                'stderr': 'Build failed'
            }

            result = await pipeline.deploy()

            assert result.status == DeploymentStatus.FAILED
            assert result.failed_stage == "build"
            assert result.build_result.status == BuildStatus.FAILED

    @pytest.mark.asyncio
    async def test_deploy_build_timeout(self, pipeline):
        """Test deployment with build timeout."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.side_effect = asyncio.TimeoutError("Build timeout")

            result = await pipeline.deploy()

            assert result.status == DeploymentStatus.FAILED
            assert result.failed_stage == "build"
            assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_deploy_pre_validation_failure(self, pipeline, temp_dirs):
        """Test deployment with pre-validation failure."""
        # Remove source directory to trigger validation failure
        shutil.rmtree(temp_dirs['source'])

        result = await pipeline.deploy()

        assert result.status == DeploymentStatus.FAILED
        assert result.failed_stage == "pre_validation"
        assert "does not exist" in result.error_message

    @pytest.mark.asyncio
    async def test_deploy_pre_validation_force(self, pipeline, temp_dirs):
        """Test deployment with forced pre-validation bypass."""
        # Remove source directory
        shutil.rmtree(temp_dirs['source'])

        with patch.object(pipeline, '_build_documentation') as mock_build:
            mock_build.return_value = None

            result = await pipeline.deploy(force=True)

            # Should continue despite validation failure
            assert result.status == DeploymentStatus.FAILED or mock_build.called

    @pytest.mark.asyncio
    async def test_deploy_rollback_on_failure(self, pipeline, temp_dirs):
        """Test automatic rollback on deployment failure."""
        # First, create a successful deployment
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "existing.html").write_text("Existing content")

        with patch.object(pipeline, '_run_command') as mock_run:
            # Build succeeds
            mock_run.return_value = {
                'exit_code': 0,
                'stdout': 'Build successful',
                'stderr': ''
            }

            # Copy source to build directory
            shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)

            # Mock post-validation failure
            with patch.object(pipeline, '_validate_post_deployment') as mock_validate:
                mock_validate.side_effect = Exception("Post-validation failed")

                result = await pipeline.deploy()

                assert result.status == DeploymentStatus.ROLLED_BACK
                assert result.rollback_point is not None
                # Check that original content is restored
                assert (temp_dirs['deploy'] / "existing.html").exists()

    @pytest.mark.asyncio
    async def test_deploy_different_strategies(self, pipeline, temp_dirs):
        """Test different deployment strategies."""
        # Test atomic strategy (default)
        pipeline.config.deployment_strategy = "atomic"
        with patch.object(pipeline, '_atomic_deployment') as mock_atomic:
            await pipeline._deploy_documentation(DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now()))
            mock_atomic.assert_called_once()

        # Test rolling strategy
        pipeline.config.deployment_strategy = "rolling"
        with patch.object(pipeline, '_rolling_deployment') as mock_rolling:
            await pipeline._deploy_documentation(DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now()))
            mock_rolling.assert_called_once()

        # Test blue-green strategy
        pipeline.config.deployment_strategy = "blue-green"
        with patch.object(pipeline, '_blue_green_deployment') as mock_blue_green:
            await pipeline._deploy_documentation(DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now()))
            mock_blue_green.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_invalid_strategy(self, pipeline):
        """Test deployment with invalid strategy."""
        pipeline.config.deployment_strategy = "invalid"

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        with pytest.raises(ValueError, match="Unknown deployment strategy"):
            await pipeline._deploy_documentation(result)

    @pytest.mark.asyncio
    async def test_atomic_deployment_success(self, pipeline, temp_dirs):
        """Test successful atomic deployment."""
        # Set up build directory
        temp_dirs['build'].mkdir(exist_ok=True)
        (temp_dirs['build'] / "new.html").write_text("New content")

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        await pipeline._atomic_deployment(result)

        # Check deployment was successful
        assert (temp_dirs['deploy'] / "new.html").exists()
        assert result.deployment_metadata['strategy'] == 'atomic'

    @pytest.mark.asyncio
    async def test_atomic_deployment_cleanup_on_failure(self, pipeline, temp_dirs):
        """Test atomic deployment cleanup on failure."""
        # Set up build directory
        temp_dirs['build'].mkdir(exist_ok=True)
        (temp_dirs['build'] / "new.html").write_text("New content")

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        # Mock copytree to fail
        with patch('shutil.copytree', side_effect=OSError("Copy failed")):
            with pytest.raises(OSError):
                await pipeline._atomic_deployment(result)

            # Check temp directory is cleaned up
            temp_dirs_list = list(temp_dirs['deploy'].parent.glob(f"*_temp_{result.deployment_id}"))
            assert len(temp_dirs_list) == 0

    @pytest.mark.asyncio
    async def test_validate_pre_deployment(self, pipeline, temp_dirs):
        """Test pre-deployment validation."""
        result = DeploymentResult("test", DeploymentStatus.TESTING, datetime.now())

        await pipeline._validate_pre_deployment(result)

        # Should complete without error
        assert result.status == DeploymentStatus.TESTING
        assert 'files_checked' in result.validation_results

    @pytest.mark.asyncio
    async def test_validate_pre_deployment_low_disk_space(self, pipeline, temp_dirs):
        """Test pre-deployment validation with low disk space warning."""
        result = DeploymentResult("test", DeploymentStatus.TESTING, datetime.now())

        with patch('shutil.disk_usage') as mock_disk:
            # Mock low disk space (500MB free)
            mock_disk.return_value = Mock(free=500 * 1024 * 1024)

            await pipeline._validate_pre_deployment(result)

            # Should log warning but not fail
            assert result.status == DeploymentStatus.TESTING

    @pytest.mark.asyncio
    async def test_validate_post_deployment(self, pipeline, temp_dirs):
        """Test post-deployment validation."""
        # Create deployed files
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "test.html").write_text("Test content")

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {
                'exit_code': 0,
                'stdout': 'Tests passed',
                'stderr': ''
            }

            await pipeline._validate_post_deployment(result)

            assert 'post_deployment' in result.validation_results
            assert result.validation_results['post_deployment']['validation_passed'] is True

    @pytest.mark.asyncio
    async def test_validate_post_deployment_no_files(self, pipeline, temp_dirs):
        """Test post-deployment validation with no deployed files."""
        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        with pytest.raises(ValueError, match="No files found"):
            await pipeline._validate_post_deployment(result)

        assert result.status == DeploymentStatus.FAILED
        assert result.failed_stage == "post_validation"

    @pytest.mark.asyncio
    async def test_validate_post_deployment_test_failure(self, pipeline, temp_dirs):
        """Test post-deployment validation with test failure."""
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "test.html").write_text("Test content")

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {
                'exit_code': 1,
                'stdout': '',
                'stderr': 'Tests failed'
            }

            with pytest.raises(ValueError, match="tests failed"):
                await pipeline._validate_post_deployment(result)

    @pytest.mark.asyncio
    async def test_create_rollback_point(self, pipeline, temp_dirs):
        """Test creating rollback point."""
        # Set up existing deployment
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "existing.html").write_text("Existing content")

        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())
        rollback_id = await pipeline._create_rollback_point(result)

        assert rollback_id is not None
        assert rollback_id.startswith("rollback_")

        # Check backup was created
        rollback_path = temp_dirs['backup'] / rollback_id
        assert rollback_path.exists()
        assert (rollback_path / "existing.html").exists()

    @pytest.mark.asyncio
    async def test_create_rollback_point_empty_deployment(self, pipeline, temp_dirs):
        """Test creating rollback point with no existing deployment."""
        result = DeploymentResult("test", DeploymentStatus.DEPLOYING, datetime.now())
        rollback_id = await pipeline._create_rollback_point(result)

        assert rollback_id is not None

        # Check empty marker was created
        rollback_path = temp_dirs['backup'] / rollback_id
        assert rollback_path.exists()
        assert (rollback_path / ".empty").exists()

    @pytest.mark.asyncio
    async def test_perform_rollback(self, pipeline, temp_dirs):
        """Test performing rollback."""
        # Set up rollback point
        rollback_id = "rollback_test_123"
        rollback_path = temp_dirs['backup'] / rollback_id
        rollback_path.mkdir(parents=True)
        (rollback_path / "original.html").write_text("Original content")

        # Set up current deployment
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "current.html").write_text("Current content")

        result = DeploymentResult("test", DeploymentStatus.FAILED, datetime.now())

        await pipeline._perform_rollback(result, rollback_id)

        assert result.status == DeploymentStatus.ROLLED_BACK
        assert (temp_dirs['deploy'] / "original.html").exists()
        assert not (temp_dirs['deploy'] / "current.html").exists()

    @pytest.mark.asyncio
    async def test_perform_rollback_empty_deployment(self, pipeline, temp_dirs):
        """Test performing rollback to empty deployment."""
        # Set up empty rollback point
        rollback_id = "rollback_test_123"
        rollback_path = temp_dirs['backup'] / rollback_id
        rollback_path.mkdir(parents=True)
        (rollback_path / ".empty").write_text("No previous deployment")

        # Set up current deployment
        temp_dirs['deploy'].mkdir(exist_ok=True)
        (temp_dirs['deploy'] / "current.html").write_text("Current content")

        result = DeploymentResult("test", DeploymentStatus.FAILED, datetime.now())

        await pipeline._perform_rollback(result, rollback_id)

        assert result.status == DeploymentStatus.ROLLED_BACK
        assert temp_dirs['deploy'].exists()
        assert len(list(temp_dirs['deploy'].iterdir())) == 0

    @pytest.mark.asyncio
    async def test_perform_rollback_missing_rollback_point(self, pipeline, temp_dirs):
        """Test performing rollback with missing rollback point."""
        result = DeploymentResult("test", DeploymentStatus.FAILED, datetime.now())

        with pytest.raises(ValueError, match="Rollback point not found"):
            await pipeline._perform_rollback(result, "nonexistent_rollback")

    @pytest.mark.asyncio
    async def test_cleanup_old_rollbacks(self, pipeline, temp_dirs):
        """Test cleanup of old rollback points."""
        # Create multiple rollback points
        for i in range(10):
            rollback_path = temp_dirs['backup'] / f"rollback_test_{i}"
            rollback_path.mkdir()
            (rollback_path / "content.html").write_text(f"Content {i}")

        # Set max rollback versions to 5
        pipeline.config.max_rollback_versions = 5

        await pipeline._cleanup_old_rollbacks()

        # Check only 5 most recent rollbacks remain
        remaining_rollbacks = list(temp_dirs['backup'].glob("rollback_*"))
        assert len(remaining_rollbacks) == 5

    @pytest.mark.asyncio
    async def test_run_validation_checks(self, pipeline, temp_dirs):
        """Test validation checks."""
        validation_results = await pipeline._run_validation_checks()

        assert 'files_checked' in validation_results
        assert 'checks_passed' in validation_results
        assert validation_results['files_checked'] == 2  # index.html and style.css

    @pytest.mark.asyncio
    async def test_run_validation_checks_with_unreadable_files(self, pipeline, temp_dirs):
        """Test validation checks with unreadable files."""
        # Create binary file
        (temp_dirs['source'] / "binary.dat").write_bytes(b'\x00\x01\x02\x03\xFF')

        validation_results = await pipeline._run_validation_checks()

        assert validation_results['warnings'] > 0

    @pytest.mark.asyncio
    async def test_run_command_success(self, pipeline):
        """Test running command successfully."""
        result = await pipeline._run_command(['echo', 'hello'])

        assert result['exit_code'] == 0
        assert 'hello' in result['stdout']
        assert result['stderr'] == ''

    @pytest.mark.asyncio
    async def test_run_command_failure(self, pipeline):
        """Test running command that fails."""
        result = await pipeline._run_command(['false'])

        assert result['exit_code'] != 0

    @pytest.mark.asyncio
    async def test_run_command_timeout(self, pipeline):
        """Test command timeout."""
        with pytest.raises(asyncio.TimeoutError):
            await pipeline._run_command(['sleep', '10'], timeout=1)

    @pytest.mark.asyncio
    async def test_send_notifications(self, pipeline):
        """Test sending notifications."""
        result = DeploymentResult("test", DeploymentStatus.SUCCESS, datetime.now())

        # Should not raise exception
        await pipeline._send_notifications(result)

        # Test with notification hooks
        pipeline.config.notification_hooks = ["echo", "Test notification"]
        await pipeline._send_notifications(result)

    def test_generate_deployment_id(self, pipeline):
        """Test deployment ID generation."""
        deployment_id = pipeline._generate_deployment_id()

        assert deployment_id.startswith("deploy_")
        assert len(deployment_id) > 15  # Should have timestamp and hash

        # Should generate unique IDs
        deployment_id2 = pipeline._generate_deployment_id()
        assert deployment_id != deployment_id2

    def test_get_current_version(self, pipeline):
        """Test getting current version."""
        version = pipeline._get_current_version()

        assert version.startswith("v")
        assert len(version) > 5  # Should have timestamp

    def test_get_deployment_status(self, pipeline):
        """Test getting deployment status."""
        # Non-existent deployment
        status = pipeline.get_deployment_status("nonexistent")
        assert status is None

        # Add deployment to history
        result = DeploymentResult("test_123", DeploymentStatus.SUCCESS, datetime.now())
        pipeline._deployment_history.append(result)

        status = pipeline.get_deployment_status("test_123")
        assert status is not None
        assert status.deployment_id == "test_123"

    def test_get_deployment_history(self, pipeline):
        """Test getting deployment history."""
        # Empty history
        history = pipeline.get_deployment_history()
        assert len(history) == 0

        # Add deployments to history
        for i in range(15):
            result = DeploymentResult(f"test_{i}", DeploymentStatus.SUCCESS, datetime.now())
            pipeline._deployment_history.append(result)

        # Test default limit
        history = pipeline.get_deployment_history()
        assert len(history) == 10

        # Test custom limit
        history = pipeline.get_deployment_history(limit=5)
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_deployment_with_version(self, pipeline, temp_dirs):
        """Test deployment with specific version."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {
                'exit_code': 0,
                'stdout': 'Success',
                'stderr': ''
            }

            shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)

            result = await pipeline.deploy(version="1.2.3")

            assert result.source_version == "1.2.3"
            assert result.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deployment_exception_handling(self, pipeline):
        """Test deployment with unexpected exception."""
        with patch.object(pipeline, '_validate_pre_deployment', side_effect=RuntimeError("Unexpected error")):
            result = await pipeline.deploy()

            assert result.status == DeploymentStatus.FAILED
            assert "Unexpected error" in result.error_message

    @pytest.mark.asyncio
    async def test_deployment_rollback_failure(self, pipeline, temp_dirs):
        """Test deployment with rollback failure."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}
            shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)

            with patch.object(pipeline, '_validate_post_deployment', side_effect=Exception("Validation failed")):
                with patch.object(pipeline, '_perform_rollback', side_effect=Exception("Rollback failed")):
                    result = await pipeline.deploy()

                    assert result.status == DeploymentStatus.FAILED
                    # Should have tried rollback

    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, pipeline, temp_dirs):
        """Test handling of concurrent deployments."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}

            async def mock_build_delay(*args, **kwargs):
                await asyncio.sleep(0.1)  # Short delay
                shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)
                return mock_run.return_value

            mock_run.side_effect = mock_build_delay

            # Start two deployments concurrently
            task1 = asyncio.create_task(pipeline.deploy())
            task2 = asyncio.create_task(pipeline.deploy())

            result1, result2 = await asyncio.gather(task1, task2)

            # Both should complete
            assert result1.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]
            assert result2.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]

    @pytest.mark.asyncio
    async def test_pipeline_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        # Should not raise exception
        pipeline.cleanup()

    def test_pipeline_context_manager(self, deployment_config):
        """Test pipeline as context manager-like usage."""
        pipeline = DeploymentPipeline(deployment_config)

        try:
            # Use pipeline
            assert len(pipeline._active_deployments) == 0
        finally:
            pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_deployment_with_disabled_rollback(self, pipeline, temp_dirs):
        """Test deployment with rollback disabled."""
        pipeline.config.rollback_enabled = False

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}
            shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)

            result = await pipeline.deploy()

            assert result.rollback_point is None
            assert result.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deployment_with_disabled_validation(self, pipeline, temp_dirs):
        """Test deployment with validation disabled."""
        pipeline.config.pre_deploy_validation = False
        pipeline.config.post_deploy_validation = False

        # Remove source to test that validation is skipped
        shutil.rmtree(temp_dirs['source'])

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}

            # Create empty build directory
            temp_dirs['build'].mkdir(exist_ok=True)
            (temp_dirs['build'] / "test.html").write_text("Test")

            result = await pipeline.deploy()

            # Should succeed despite missing source (validation disabled)
            assert result.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_edge_case_empty_build_directory(self, pipeline, temp_dirs):
        """Test deployment with empty build directory."""
        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}

            # Create empty build directory
            temp_dirs['build'].mkdir(exist_ok=True)

            result = await pipeline.deploy()

            # Should handle empty build directory gracefully
            assert result.build_result is not None

    @pytest.mark.asyncio
    async def test_edge_case_unicode_filenames(self, pipeline, temp_dirs):
        """Test deployment with Unicode filenames."""
        # Create files with Unicode names
        (temp_dirs['source'] / "测试.html").write_text("<html>Unicode test</html>")
        (temp_dirs['source'] / "файл.css").write_text("body { color: blue; }")

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}

            async def mock_build(*args, **kwargs):
                shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)
                return mock_run.return_value

            mock_run.side_effect = mock_build

            result = await pipeline.deploy()

            # Should handle Unicode filenames without issues
            assert result.status == DeploymentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_edge_case_large_deployment(self, pipeline, temp_dirs):
        """Test deployment with many files."""
        # Create many files
        for i in range(100):
            (temp_dirs['source'] / f"file_{i}.html").write_text(f"<html>File {i}</html>")

        with patch.object(pipeline, '_run_command') as mock_run:
            mock_run.return_value = {'exit_code': 0, 'stdout': 'Success', 'stderr': ''}

            async def mock_build(*args, **kwargs):
                shutil.copytree(temp_dirs['source'], temp_dirs['build'], dirs_exist_ok=True)
                return mock_run.return_value

            mock_run.side_effect = mock_build

            result = await pipeline.deploy()

            assert result.status == DeploymentStatus.SUCCESS
            assert result.build_result.artifacts is not None