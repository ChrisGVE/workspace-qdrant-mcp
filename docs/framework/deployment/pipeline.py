"""Comprehensive deployment pipeline with versioning and rollback capabilities."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import json
import shutil
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment pipeline status."""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class BuildStatus(Enum):
    """Build process status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """Configuration for deployment pipeline."""

    # Source and target paths
    source_path: Path
    build_path: Path
    deploy_path: Path
    backup_path: Path

    # Build configuration
    build_command: List[str] = field(default_factory=lambda: ["make", "build"])
    test_command: List[str] = field(default_factory=lambda: ["make", "test"])
    clean_command: List[str] = field(default_factory=lambda: ["make", "clean"])

    # Deployment settings
    deployment_strategy: str = "atomic"  # atomic, rolling, blue-green
    rollback_enabled: bool = True
    max_rollback_versions: int = 5

    # Validation settings
    pre_deploy_validation: bool = True
    post_deploy_validation: bool = True
    validation_timeout: int = 300  # seconds

    # Concurrency settings
    max_workers: int = 4
    build_timeout: int = 1800  # 30 minutes
    deploy_timeout: int = 600   # 10 minutes

    # Notification settings
    notify_on_success: bool = True
    notify_on_failure: bool = True
    notification_hooks: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields."""
        for path_attr in ['source_path', 'build_path', 'deploy_path', 'backup_path']:
            path_value = getattr(self, path_attr)
            if isinstance(path_value, str):
                setattr(self, path_attr, Path(path_value))


@dataclass
class BuildResult:
    """Result of build process."""
    status: BuildStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    artifacts: List[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    build_id: Optional[str] = None

    def __post_init__(self):
        """Calculate duration if end time is set."""
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class DeploymentResult:
    """Result of deployment pipeline."""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Stage results
    build_result: Optional[BuildResult] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    deployment_metadata: Dict[str, Any] = field(default_factory=dict)

    # Version information
    source_version: Optional[str] = None
    deployed_version: Optional[str] = None
    rollback_point: Optional[str] = None

    # Error information
    error_message: Optional[str] = None
    failed_stage: Optional[str] = None

    def __post_init__(self):
        """Calculate duration if end time is set."""
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class DeploymentPipeline:
    """Comprehensive deployment pipeline with versioning and rollback."""

    def __init__(self, config: DeploymentConfig):
        """Initialize deployment pipeline.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._active_deployments: Dict[str, DeploymentResult] = {}
        self._deployment_history: List[DeploymentResult] = []

        # Initialize directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.config.build_path, self.config.deploy_path,
                         self.config.backup_path]:
            directory.mkdir(parents=True, exist_ok=True)

    async def deploy(self,
                    version: Optional[str] = None,
                    force: bool = False,
                    dry_run: bool = False) -> DeploymentResult:
        """Deploy documentation with comprehensive pipeline.

        Args:
            version: Specific version to deploy (defaults to current)
            force: Force deployment even if validation fails
            dry_run: Perform dry run without actual deployment

        Returns:
            DeploymentResult with detailed information
        """
        deployment_id = self._generate_deployment_id()
        start_time = datetime.now()

        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=start_time,
            source_version=version or self._get_current_version()
        )

        self._active_deployments[deployment_id] = result

        try:
            logger.info(f"Starting deployment {deployment_id} (dry_run={dry_run})")

            # Stage 1: Pre-deployment validation
            if self.config.pre_deploy_validation:
                result.status = DeploymentStatus.TESTING
                await self._validate_pre_deployment(result)
                if result.status == DeploymentStatus.FAILED and not force:
                    return result

            # Stage 2: Build
            result.status = DeploymentStatus.BUILDING
            await self._build_documentation(result, dry_run)
            if result.status == DeploymentStatus.FAILED:
                return result

            # Stage 3: Deploy (if not dry run)
            if not dry_run:
                result.status = DeploymentStatus.DEPLOYING

                # Create rollback point before deployment
                if self.config.rollback_enabled:
                    rollback_point = await self._create_rollback_point(result)
                    result.rollback_point = rollback_point

                # Perform deployment
                await self._deploy_documentation(result)
                if result.status == DeploymentStatus.FAILED:
                    # Attempt automatic rollback
                    if self.config.rollback_enabled and rollback_point:
                        await self._perform_rollback(result, rollback_point)
                    return result

            # Stage 4: Post-deployment validation
            if self.config.post_deploy_validation and not dry_run:
                await self._validate_post_deployment(result)
                if result.status == DeploymentStatus.FAILED and self.config.rollback_enabled:
                    await self._perform_rollback(result, result.rollback_point)
                    return result

            # Success
            result.status = DeploymentStatus.SUCCESS
            result.deployed_version = result.source_version
            result.end_time = datetime.now()

            logger.info(f"Deployment {deployment_id} completed successfully")

            # Cleanup old rollback points
            await self._cleanup_old_rollbacks()

        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()

            # Attempt rollback on critical failure
            if self.config.rollback_enabled and result.rollback_point:
                try:
                    await self._perform_rollback(result, result.rollback_point)
                except Exception as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")

        finally:
            self._active_deployments.pop(deployment_id, None)
            self._deployment_history.append(result)

            # Send notifications
            await self._send_notifications(result)

        return result

    async def _validate_pre_deployment(self, result: DeploymentResult):
        """Validate system before deployment."""
        try:
            logger.debug("Running pre-deployment validation")

            # Check source directory exists and has content
            if not self.config.source_path.exists():
                raise ValueError(f"Source path does not exist: {self.config.source_path}")

            # Check disk space
            stats = shutil.disk_usage(self.config.deploy_path)
            free_gb = stats.free / (1024**3)
            if free_gb < 1.0:  # Less than 1GB free
                logger.warning(f"Low disk space: {free_gb:.1f}GB available")

            # Validate source files
            validation_results = await self._run_validation_checks()
            result.validation_results.update(validation_results)

            if validation_results.get('critical_errors', 0) > 0:
                raise ValueError(f"Critical validation errors found: {validation_results}")

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.failed_stage = "pre_validation"
            result.error_message = str(e)
            raise

    async def _build_documentation(self, result: DeploymentResult, dry_run: bool):
        """Build documentation with comprehensive error handling."""
        build_start = datetime.now()
        build_id = f"build_{result.deployment_id}_{int(time.time())}"

        build_result = BuildResult(
            status=BuildStatus.IN_PROGRESS,
            start_time=build_start,
            build_id=build_id
        )
        result.build_result = build_result

        try:
            logger.debug(f"Starting build {build_id} (dry_run={dry_run})")

            if dry_run:
                # Simulate build process
                await asyncio.sleep(1)
                build_result.status = BuildStatus.SUCCESS
                build_result.stdout = "DRY RUN: Build would be executed"
                logger.info(f"Dry run build {build_id} completed")
            else:
                # Clean previous build
                if self.config.build_path.exists():
                    shutil.rmtree(self.config.build_path)
                self.config.build_path.mkdir(parents=True, exist_ok=True)

                # Execute build command
                process_result = await self._run_command(
                    self.config.build_command,
                    cwd=self.config.source_path,
                    timeout=self.config.build_timeout
                )

                build_result.exit_code = process_result['exit_code']
                build_result.stdout = process_result['stdout']
                build_result.stderr = process_result['stderr']

                if process_result['exit_code'] == 0:
                    build_result.status = BuildStatus.SUCCESS
                    # Collect build artifacts
                    artifacts = list(self.config.build_path.rglob('*'))
                    build_result.artifacts = [f for f in artifacts if f.is_file()]
                    logger.info(f"Build {build_id} completed with {len(build_result.artifacts)} artifacts")
                else:
                    build_result.status = BuildStatus.FAILED
                    raise subprocess.CalledProcessError(
                        process_result['exit_code'],
                        self.config.build_command,
                        process_result['stderr']
                    )

        except asyncio.TimeoutError:
            build_result.status = BuildStatus.FAILED
            result.status = DeploymentStatus.FAILED
            result.failed_stage = "build"
            result.error_message = f"Build timeout after {self.config.build_timeout} seconds"
            logger.error(f"Build {build_id} timed out")

        except Exception as e:
            build_result.status = BuildStatus.FAILED
            result.status = DeploymentStatus.FAILED
            result.failed_stage = "build"
            result.error_message = str(e)
            logger.error(f"Build {build_id} failed: {e}")

        finally:
            build_result.end_time = datetime.now()

    async def _deploy_documentation(self, result: DeploymentResult):
        """Deploy built documentation with atomic strategy."""
        try:
            logger.debug(f"Deploying documentation for {result.deployment_id}")

            if self.config.deployment_strategy == "atomic":
                await self._atomic_deployment(result)
            elif self.config.deployment_strategy == "rolling":
                await self._rolling_deployment(result)
            elif self.config.deployment_strategy == "blue-green":
                await self._blue_green_deployment(result)
            else:
                raise ValueError(f"Unknown deployment strategy: {self.config.deployment_strategy}")

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.failed_stage = "deployment"
            result.error_message = str(e)
            raise

    async def _atomic_deployment(self, result: DeploymentResult):
        """Perform atomic deployment by replacing entire directory."""
        temp_deploy_path = self.config.deploy_path.parent / f"{self.config.deploy_path.name}_temp_{result.deployment_id}"

        try:
            # Copy built files to temporary location
            shutil.copytree(self.config.build_path, temp_deploy_path, dirs_exist_ok=True)

            # Atomic swap
            if self.config.deploy_path.exists():
                old_deploy_path = self.config.deploy_path.parent / f"{self.config.deploy_path.name}_old_{result.deployment_id}"
                self.config.deploy_path.rename(old_deploy_path)
                temp_deploy_path.rename(self.config.deploy_path)
                shutil.rmtree(old_deploy_path)
            else:
                temp_deploy_path.rename(self.config.deploy_path)

            result.deployment_metadata['strategy'] = 'atomic'
            result.deployment_metadata['deployed_files'] = len(list(self.config.deploy_path.rglob('*')))

        except Exception:
            # Cleanup on failure
            if temp_deploy_path.exists():
                shutil.rmtree(temp_deploy_path)
            raise

    async def _rolling_deployment(self, result: DeploymentResult):
        """Perform rolling deployment by updating files incrementally."""
        # Implementation would update files in batches to minimize downtime
        # For simplicity, falling back to atomic for now
        await self._atomic_deployment(result)
        result.deployment_metadata['strategy'] = 'rolling'

    async def _blue_green_deployment(self, result: DeploymentResult):
        """Perform blue-green deployment with traffic switching."""
        # Implementation would maintain two environments and switch traffic
        # For simplicity, falling back to atomic for now
        await self._atomic_deployment(result)
        result.deployment_metadata['strategy'] = 'blue-green'

    async def _validate_post_deployment(self, result: DeploymentResult):
        """Validate deployment after completion."""
        try:
            logger.debug("Running post-deployment validation")

            # Check deployed files exist
            if not self.config.deploy_path.exists():
                raise ValueError("Deploy path does not exist after deployment")

            deployed_files = list(self.config.deploy_path.rglob('*'))
            if not deployed_files:
                raise ValueError("No files found in deploy path")

            # Run test command if specified
            if self.config.test_command:
                test_result = await self._run_command(
                    self.config.test_command,
                    cwd=self.config.deploy_path,
                    timeout=self.config.validation_timeout
                )

                if test_result['exit_code'] != 0:
                    raise ValueError(f"Post-deployment tests failed: {test_result['stderr']}")

            result.validation_results['post_deployment'] = {
                'files_deployed': len(deployed_files),
                'validation_passed': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.failed_stage = "post_validation"
            result.error_message = str(e)
            result.validation_results['post_deployment'] = {
                'validation_passed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise

    async def _create_rollback_point(self, result: DeploymentResult) -> str:
        """Create rollback point before deployment."""
        rollback_id = f"rollback_{result.deployment_id}_{int(time.time())}"
        rollback_path = self.config.backup_path / rollback_id

        try:
            if self.config.deploy_path.exists():
                shutil.copytree(self.config.deploy_path, rollback_path)
                logger.info(f"Created rollback point: {rollback_id}")
            else:
                # Create empty rollback point marker
                rollback_path.mkdir(parents=True, exist_ok=True)
                (rollback_path / ".empty").write_text("No previous deployment")

            return rollback_id

        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            raise

    async def _perform_rollback(self, result: DeploymentResult, rollback_point: str):
        """Perform rollback to previous state."""
        try:
            logger.info(f"Rolling back deployment {result.deployment_id} to {rollback_point}")
            result.status = DeploymentStatus.ROLLING_BACK

            rollback_path = self.config.backup_path / rollback_point

            if not rollback_path.exists():
                raise ValueError(f"Rollback point not found: {rollback_point}")

            # Remove current deployment
            if self.config.deploy_path.exists():
                shutil.rmtree(self.config.deploy_path)

            # Restore from rollback point
            if (rollback_path / ".empty").exists():
                # Was empty deployment, just create directory
                self.config.deploy_path.mkdir(parents=True, exist_ok=True)
            else:
                shutil.copytree(rollback_path, self.config.deploy_path)

            result.status = DeploymentStatus.ROLLED_BACK
            result.deployment_metadata['rolled_back_to'] = rollback_point
            logger.info(f"Rollback completed for {result.deployment_id}")

        except Exception as e:
            logger.error(f"Rollback failed for {result.deployment_id}: {e}")
            result.error_message = f"Rollback failed: {e}"
            raise

    async def _cleanup_old_rollbacks(self):
        """Clean up old rollback points beyond the maximum limit."""
        try:
            rollback_dirs = [d for d in self.config.backup_path.glob("rollback_*") if d.is_dir()]
            rollback_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep only the most recent rollback points
            to_remove = rollback_dirs[self.config.max_rollback_versions:]

            for rollback_dir in to_remove:
                shutil.rmtree(rollback_dir)
                logger.debug(f"Cleaned up old rollback point: {rollback_dir.name}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old rollbacks: {e}")

    async def _run_validation_checks(self) -> Dict[str, Any]:
        """Run comprehensive validation checks on source files."""
        validation_results = {
            'critical_errors': 0,
            'warnings': 0,
            'files_checked': 0,
            'checks_passed': 0
        }

        try:
            # Check source files exist and are readable
            source_files = list(self.config.source_path.rglob('*'))
            source_files = [f for f in source_files if f.is_file()]
            validation_results['files_checked'] = len(source_files)

            for file in source_files:
                try:
                    file.read_text(encoding='utf-8')
                    validation_results['checks_passed'] += 1
                except UnicodeDecodeError:
                    validation_results['warnings'] += 1
                except Exception:
                    validation_results['critical_errors'] += 1

        except Exception as e:
            validation_results['critical_errors'] += 1
            validation_results['error'] = str(e)

        return validation_results

    async def _run_command(self,
                          command: List[str],
                          cwd: Optional[Path] = None,
                          timeout: int = 300) -> Dict[str, Any]:
        """Run command with timeout and capture output."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

            return {
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace')
            }

        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise

    async def _send_notifications(self, result: DeploymentResult):
        """Send deployment notifications."""
        if (result.status == DeploymentStatus.SUCCESS and self.config.notify_on_success) or \
           (result.status == DeploymentStatus.FAILED and self.config.notify_on_failure):

            # Basic logging notification
            logger.info(f"Deployment {result.deployment_id} notification: {result.status.value}")

            # Additional notification hooks would be implemented here
            for hook in self.config.notification_hooks:
                try:
                    # Execute notification hook
                    logger.debug(f"Executing notification hook: {hook}")
                except Exception as e:
                    logger.error(f"Notification hook failed: {e}")

    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        timestamp = int(time.time())
        source_hash = hashlib.md5(str(self.config.source_path).encode()).hexdigest()[:8]
        return f"deploy_{timestamp}_{source_hash}"

    def _get_current_version(self) -> str:
        """Get current version from source."""
        # This would typically read from version file or git
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of specific deployment."""
        # Check active deployments
        if deployment_id in self._active_deployments:
            return self._active_deployments[deployment_id]

        # Check deployment history
        for deployment in self._deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment

        return None

    def get_deployment_history(self, limit: int = 10) -> List[DeploymentResult]:
        """Get recent deployment history."""
        return sorted(self._deployment_history,
                     key=lambda x: x.start_time,
                     reverse=True)[:limit]

    def cleanup(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=True)