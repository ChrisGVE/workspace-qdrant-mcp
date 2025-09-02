
from ...observability import get_logger
logger = get_logger(__name__)
"""
Watch folder validation and error recovery system.

This module provides comprehensive validation and error recovery mechanisms
for file watching operations, including path validation, permission checks,
and automatic retry capabilities.
"""

import asyncio
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryAttempt:
    """Record of an error recovery attempt."""
    
    timestamp: str
    error_type: str
    recovery_action: str
    success: bool
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "recovery_action": self.recovery_action,
            "success": self.success,
            "details": self.details,
        }


class WatchPathValidator:
    """Comprehensive path validation for watch configurations."""
    
    @staticmethod
    def validate_path_existence(path: Path) -> ValidationResult:
        """Validate that the path exists and is accessible."""
        try:
            if not path.exists():
                return ValidationResult(
                    valid=False,
                    error_code="PATH_NOT_EXISTS",
                    error_message=f"Path does not exist: {path}",
                )
            
            if not path.is_dir():
                return ValidationResult(
                    valid=False,
                    error_code="PATH_NOT_DIRECTORY",
                    error_message=f"Path is not a directory: {path}",
                )
            
            return ValidationResult(
                valid=True,
                metadata={"resolved_path": str(path.resolve())},
            )
            
        except PermissionError:
            return ValidationResult(
                valid=False,
                error_code="PATH_ACCESS_DENIED",
                error_message=f"Permission denied accessing path: {path}",
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_code="PATH_ACCESS_ERROR",
                error_message=f"Error accessing path {path}: {e}",
            )
    
    @staticmethod
    def validate_permissions(path: Path) -> ValidationResult:
        """Validate that the path has necessary permissions for watching."""
        try:
            warnings = []
            metadata = {}
            
            # Check basic read permission
            if not path.is_readable():
                return ValidationResult(
                    valid=False,
                    error_code="PATH_NOT_READABLE",
                    error_message=f"Path is not readable: {path}",
                )
            
            # Get detailed permission information
            stat_info = path.stat()
            permissions = stat.filemode(stat_info.st_mode)
            metadata["permissions"] = permissions
            metadata["owner_uid"] = stat_info.st_uid
            metadata["group_gid"] = stat_info.st_gid
            
            # Check if we can list directory contents
            try:\n                list(path.iterdir())\n                metadata["can_list_contents"] = True\n            except PermissionError:\n                warnings.append("Cannot list directory contents - may affect file discovery")\n                metadata["can_list_contents"] = False\n            \n            # Check write permissions (needed for some operations)\n            if not path.is_writeable():\n                warnings.append("Directory is not writeable - some operations may be limited")\n                metadata["writeable"] = False\n            else:\n                metadata["writeable"] = True\n            \n            # Check execute permission (needed to traverse directories)\n            if not os.access(path, os.X_OK):\n                return ValidationResult(\n                    valid=False,\n                    error_code="PATH_NOT_EXECUTABLE",\n                    error_message=f"Cannot traverse directory (no execute permission): {path}",\n                )\n            \n            return ValidationResult(\n                valid=True,\n                warnings=warnings,\n                metadata=metadata,\n            )\n            \n        except Exception as e:\n            return ValidationResult(\n                valid=False,\n                error_code="PERMISSION_CHECK_ERROR",\n                error_message=f"Error checking permissions for {path}: {e}",\n            )\n    \n    @staticmethod\n    def validate_path_type(path: Path) -> ValidationResult:\n        """Validate path type and handle special cases like symlinks."""\n        try:\n            warnings = []\n            metadata = {}\n            \n            # Check if it's a symlink\n            if path.is_symlink():\n                warnings.append("Path is a symbolic link")\n                metadata["is_symlink"] = True\n                metadata["symlink_target"] = str(path.resolve())\n                \n                # Check if symlink target exists and is valid\n                try:\n                    target = path.resolve()\n                    if not target.exists():\n                        return ValidationResult(\n                            valid=False,\n                            error_code="SYMLINK_BROKEN",\n                            error_message=f"Symbolic link points to non-existent target: {path} -> {target}",\n                        )\n                    \n                    if not target.is_dir():\n                        return ValidationResult(\n                            valid=False,\n                            error_code="SYMLINK_NOT_DIRECTORY",\n                            error_message=f"Symbolic link does not point to a directory: {path} -> {target}",\n                        )\n                        \n                except Exception as e:\n                    return ValidationResult(\n                        valid=False,\n                        error_code="SYMLINK_RESOLVE_ERROR",\n                        error_message=f"Cannot resolve symbolic link {path}: {e}",\n                    )\n            \n            # Check for network paths\n            path_str = str(path)\n            if path_str.startswith(('\\\\\\\\', '//', 'smb://', 'nfs://')) or '://' in path_str:\n                warnings.append("Path appears to be a network location - may have reliability issues")\n                metadata["is_network_path"] = True\n            \n            # Check for mount points\n            if path.is_mount():\n                warnings.append("Path is a mount point")\n                metadata["is_mount_point"] = True\n            \n            return ValidationResult(\n                valid=True,\n                warnings=warnings,\n                metadata=metadata,\n            )\n            \n        except Exception as e:\n            return ValidationResult(\n                valid=False,\n                error_code="PATH_TYPE_CHECK_ERROR",\n                error_message=f"Error checking path type for {path}: {e}",\n            )\n    \n    @staticmethod\n    def validate_filesystem_compatibility(path: Path) -> ValidationResult:\n        """Validate filesystem compatibility for file watching."""\n        try:\n            warnings = []\n            metadata = {}\n            \n            # Get filesystem information\n            try:\n                stat_info = path.stat()\n                metadata["device_id"] = stat_info.st_dev\n                \n                # Try to get filesystem type (platform-specific)\n                if hasattr(os, 'statvfs'):  # Unix-like systems\n                    statvfs = os.statvfs(path)\n                    metadata["filesystem_info"] = {\n                        "block_size": statvfs.f_bsize,\n                        "fragment_size": statvfs.f_frsize,\n                        "total_blocks": statvfs.f_blocks,\n                        "free_blocks": statvfs.f_bavail,\n                    }\n                    \n                    # Check available space\n                    free_space_bytes = statvfs.f_bavail * statvfs.f_frsize\n                    free_space_mb = free_space_bytes / (1024 * 1024)\n                    metadata["free_space_mb"] = free_space_mb\n                    \n                    if free_space_mb < 100:  # Less than 100MB\n                        warnings.append(f"Low disk space: {free_space_mb:.1f} MB available")\n                    \n            except Exception as e:\n                logger.debug(f"Could not get filesystem info for {path}: {e}")\n                warnings.append("Could not determine filesystem information")\n            \n            # Test file watching capabilities\n            try:\n                # Create a temporary file to test watching\n                test_file = path / f".watch_test_{int(time.time())}.tmp"\n                try:\n                    test_file.touch()\n                    test_file.unlink()\n                    metadata["supports_file_creation"] = True\n                except PermissionError:\n                    warnings.append("Cannot create test files - watch reliability may be affected")\n                    metadata["supports_file_creation"] = False\n                except Exception as e:\n                    warnings.append(f"File creation test failed: {e}")\n                    metadata["supports_file_creation"] = False\n            except Exception as e:\n                logger.debug(f"File watching test failed for {path}: {e}")\n                warnings.append("Could not test file watching capabilities")\n            \n            return ValidationResult(\n                valid=True,\n                warnings=warnings,\n                metadata=metadata,\n            )\n            \n        except Exception as e:\n            return ValidationResult(\n                valid=False,\n                error_code="FILESYSTEM_CHECK_ERROR",\n                error_message=f"Error checking filesystem compatibility for {path}: {e}",\n            )\n    \n    @classmethod\n    def validate_watch_path(cls, path: str | Path) -> ValidationResult:\n        """Perform comprehensive validation of a watch path."""\n        path = Path(path) if isinstance(path, str) else path\n        \n        all_warnings = []\n        all_metadata = {}\n        \n        # Run all validation checks\n        validators = [\n            cls.validate_path_existence,\n            cls.validate_permissions,\n            cls.validate_path_type,\n            cls.validate_filesystem_compatibility,\n        ]\n        \n        for validator in validators:\n            result = validator(path)\n            \n            if not result.valid:\n                # Return first critical failure\n                return result\n            \n            all_warnings.extend(result.warnings)\n            all_metadata.update(result.metadata)\n        \n        return ValidationResult(\n            valid=True,\n            warnings=all_warnings,\n            metadata=all_metadata,\n        )


class WatchErrorRecovery:
    """Error recovery mechanisms for watch operations."""\n    \n    def __init__(self):\n        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = {}\n        self.max_recovery_attempts = 5\n        self.recovery_backoff_seconds = [1, 2, 5, 10, 30]  # Progressive backoff\n    \n    async def attempt_recovery(\n        self, \n        watch_id: str, \n        error_type: str, \n        path: Path,\n        error_details: str = ""\n    ) -> Tuple[bool, str]:\n        """Attempt to recover from a watch error."""\n        \n        if watch_id not in self.recovery_attempts:\n            self.recovery_attempts[watch_id] = []\n        \n        attempts = self.recovery_attempts[watch_id]\n        attempt_count = len([a for a in attempts if a.error_type == error_type])\n        \n        if attempt_count >= self.max_recovery_attempts:\n            return False, f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded for {error_type}"\n        \n        # Determine recovery strategy based on error type\n        recovery_strategy = self._get_recovery_strategy(error_type)\n        if not recovery_strategy:\n            return False, f"No recovery strategy available for error type: {error_type}"\n        \n        # Apply backoff delay\n        if attempt_count > 0:\n            backoff_delay = self.recovery_backoff_seconds[min(attempt_count - 1, len(self.recovery_backoff_seconds) - 1)]\n            logger.info(f"Waiting {backoff_delay} seconds before recovery attempt {attempt_count + 1}")\n            await asyncio.sleep(backoff_delay)\n        \n        # Attempt recovery\n        success, details = await self._execute_recovery_strategy(recovery_strategy, path, error_details)\n        \n        # Record attempt\n        attempt = RecoveryAttempt(\n            timestamp=datetime.now(timezone.utc).isoformat(),\n            error_type=error_type,\n            recovery_action=recovery_strategy["name"],\n            success=success,\n            details=details,\n        )\n        \n        attempts.append(attempt)\n        logger.info(f"Recovery attempt for {watch_id} ({error_type}): {'SUCCESS' if success else 'FAILED'} - {details}")\n        \n        return success, details\n    \n    def _get_recovery_strategy(self, error_type: str) -> Optional[Dict[str, Any]]:\n        """Get recovery strategy for a specific error type."""\n        \n        strategies = {\n            "PATH_NOT_EXISTS": {\n                "name": "wait_for_path_creation",\n                "description": "Wait for path to be created",\n                "action": "wait_and_validate"\n            },\n            "PATH_ACCESS_DENIED": {\n                "name": "retry_access",\n                "description": "Retry accessing path after delay",\n                "action": "validate_permissions"\n            },\n            "NETWORK_PATH_UNAVAILABLE": {\n                "name": "reconnect_network_path",\n                "description": "Attempt to reconnect to network path",\n                "action": "wait_and_validate"\n            },\n            "FILESYSTEM_ERROR": {\n                "name": "retry_filesystem_operation",\n                "description": "Retry filesystem operation",\n                "action": "validate_filesystem"\n            },\n            "SYMLINK_BROKEN": {\n                "name": "wait_for_symlink_target",\n                "description": "Wait for symbolic link target to become available",\n                "action": "validate_symlink"\n            },\n        }\n        \n        return strategies.get(error_type)\n    \n    async def _execute_recovery_strategy(\n        self, \n        strategy: Dict[str, Any], \n        path: Path, \n        error_details: str\n    ) -> Tuple[bool, str]:\n        """Execute a specific recovery strategy."""\n        \n        try:\n            action = strategy.get("action", "wait_and_validate")\n            \n            if action == "wait_and_validate":\n                # Simple wait and re-validate\n                result = WatchPathValidator.validate_path_existence(path)\n                if result.valid:\n                    return True, f"Path is now accessible: {path}"\n                else:\n                    return False, f"Path still not accessible: {result.error_message}"\n            \n            elif action == "validate_permissions":\n                # Re-check permissions\n                result = WatchPathValidator.validate_permissions(path)\n                if result.valid:\n                    return True, f"Permissions are now valid: {path}"\n                else:\n                    return False, f"Permission issues persist: {result.error_message}"\n            \n            elif action == "validate_filesystem":\n                # Check filesystem compatibility\n                result = WatchPathValidator.validate_filesystem_compatibility(path)\n                if result.valid:\n                    return True, f"Filesystem is now accessible: {path}"\n                else:\n                    return False, f"Filesystem issues persist: {result.error_message}"\n            \n            elif action == "validate_symlink":\n                # Re-validate symlink\n                result = WatchPathValidator.validate_path_type(path)\n                if result.valid:\n                    return True, f"Symbolic link is now valid: {path}"\n                else:\n                    return False, f"Symbolic link issues persist: {result.error_message}"\n            \n            else:\n                return False, f"Unknown recovery action: {action}"\n                \n        except Exception as e:\n            return False, f"Recovery strategy execution failed: {e}"\n    \n    def get_recovery_history(self, watch_id: str) -> List[Dict[str, Any]]:\n        """Get recovery attempt history for a watch."""\n        attempts = self.recovery_attempts.get(watch_id, [])\n        return [attempt.to_dict() for attempt in attempts]\n    \n    def clear_recovery_history(self, watch_id: str) -> None:\n        """Clear recovery history for a watch."""\n        if watch_id in self.recovery_attempts:\n            del self.recovery_attempts[watch_id]\n    \n    def should_retry_recovery(self, watch_id: str, error_type: str) -> bool:\n        """Check if recovery should be attempted for an error."""\n        attempts = self.recovery_attempts.get(watch_id, [])\n        error_attempts = [a for a in attempts if a.error_type == error_type]\n        \n        return len(error_attempts) < self.max_recovery_attempts


class WatchHealthMonitor:
    """Monitor watch health and trigger recovery when needed."""\n    \n    def __init__(self, error_recovery: WatchErrorRecovery):\n        self.error_recovery = error_recovery\n        self.health_checks: Dict[str, Dict[str, Any]] = {}\n        self.monitoring_interval = 300  # 5 minutes\n        self._monitoring_task: Optional[asyncio.Task] = None\n        self._running = False\n    \n    def register_watch(self, watch_id: str, path: Path) -> None:\n        """Register a watch for health monitoring."""\n        self.health_checks[watch_id] = {\n            "path": path,\n            "last_check": None,\n            "consecutive_failures": 0,\n            "last_success": None,\n            "status": "unknown"\n        }\n    \n    def unregister_watch(self, watch_id: str) -> None:\n        """Unregister a watch from health monitoring."""\n        if watch_id in self.health_checks:\n            del self.health_checks[watch_id]\n    \n    async def start_monitoring(self) -> None:\n        """Start the health monitoring task."""\n        if self._running:\n            return\n        \n        self._running = True\n        self._monitoring_task = asyncio.create_task(self._monitoring_loop())\n        logger.info("Started watch health monitoring")\n    \n    async def stop_monitoring(self) -> None:\n        """Stop the health monitoring task."""\n        self._running = False\n        \n        if self._monitoring_task:\n            self._monitoring_task.cancel()\n            try:\n                await self._monitoring_task\n            except asyncio.CancelledError:\n                pass\n        \n        logger.info("Stopped watch health monitoring")\n    \n    async def _monitoring_loop(self) -> None:\n        """Main monitoring loop."""\n        while self._running:\n            try:\n                await self._perform_health_checks()\n                await asyncio.sleep(self.monitoring_interval)\n            except asyncio.CancelledError:\n                break\n            except Exception as e:\n                logger.error(f"Error in health monitoring loop: {e}")\n                await asyncio.sleep(min(self.monitoring_interval, 60))  # Wait at least 1 minute on error\n    \n    async def _perform_health_checks(self) -> None:\n        """Perform health checks on all registered watches."""\n        current_time = datetime.now(timezone.utc).isoformat()\n        \n        for watch_id, health_info in self.health_checks.items():\n            try:\n                path = health_info["path"]\n                validation_result = WatchPathValidator.validate_watch_path(path)\n                \n                health_info["last_check"] = current_time\n                \n                if validation_result.valid:\n                    # Health check passed\n                    health_info["consecutive_failures"] = 0\n                    health_info["last_success"] = current_time\n                    health_info["status"] = "healthy"\n                else:\n                    # Health check failed\n                    health_info["consecutive_failures"] += 1\n                    health_info["status"] = "unhealthy"\n                    \n                    logger.warning(\n                        f"Health check failed for watch {watch_id}: {validation_result.error_message}"\n                    )\n                    \n                    # Trigger recovery if multiple failures\n                    if health_info["consecutive_failures"] >= 3:\n                        logger.info(f"Triggering recovery for watch {watch_id} after {health_info['consecutive_failures']} failures")\n                        \n                        success, details = await self.error_recovery.attempt_recovery(\n                            watch_id=watch_id,\n                            error_type=validation_result.error_code or "HEALTH_CHECK_FAILED",\n                            path=path,\n                            error_details=validation_result.error_message or "Health check failed"\n                        )\n                        \n                        if success:\n                            health_info["consecutive_failures"] = 0\n                            health_info["last_success"] = current_time\n                            health_info["status"] = "recovered"\n                            logger.info(f"Recovery successful for watch {watch_id}: {details}")\n                        else:\n                            logger.error(f"Recovery failed for watch {watch_id}: {details}")\n                            health_info["status"] = "recovery_failed"\n            \n            except Exception as e:\n                logger.error(f"Error during health check for watch {watch_id}: {e}")\n                health_info["status"] = "check_error"\n    \n    def get_health_status(self, watch_id: Optional[str] = None) -> Dict[str, Any]:\n        """Get health status for watches."""\n        if watch_id:\n            return self.health_checks.get(watch_id, {})\n        else:\n            return self.health_checks.copy()\n    \n    def is_monitoring(self) -> bool:\n        """Check if monitoring is currently running."""\n        return self._running