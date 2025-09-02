
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
            try:
                list(path.iterdir())
                metadata["can_list_contents"] = True
            except PermissionError:
                warnings.append("Cannot list directory contents - may affect file discovery")
                metadata["can_list_contents"] = False
            
            # Check write permissions (needed for some operations)
            if not path.is_writeable():
                warnings.append("Directory is not writeable - some operations may be limited")
                metadata["writeable"] = False
            else:
                metadata["writeable"] = True
            
            # Check execute permission (needed to traverse directories)
            if not os.access(path, os.X_OK):
                return ValidationResult(
                    valid=False,
                    error_code="PATH_NOT_EXECUTABLE",
                    error_message=f"Cannot traverse directory (no execute permission): {path}",
                )
            
            return ValidationResult(
                valid=True,
                warnings=warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_code="PERMISSION_CHECK_ERROR",
                error_message=f"Error checking permissions for {path}: {e}",
            )
    
    @staticmethod
    def validate_path_type(path: Path) -> ValidationResult:
        """Validate path type and handle special cases like symlinks."""
        try:
            warnings = []
            metadata = {}
            
            # Check if it's a symlink
            if path.is_symlink():
                warnings.append("Path is a symbolic link")
                metadata["is_symlink"] = True
                metadata["symlink_target"] = str(path.resolve())
                
                # Check if symlink target exists and is valid
                try:
                    target = path.resolve()
                    if not target.exists():
                        return ValidationResult(
                            valid=False,
                            error_code="SYMLINK_BROKEN",
                            error_message=f"Symbolic link points to non-existent target: {path} -> {target}",
                        )
                    
                    if not target.is_dir():
                        return ValidationResult(
                            valid=False,
                            error_code="SYMLINK_NOT_DIRECTORY",
                            error_message=f"Symbolic link does not point to a directory: {path} -> {target}",
                        )
                        
                except Exception as e:
                    return ValidationResult(
                        valid=False,
                        error_code="SYMLINK_RESOLVE_ERROR",
                        error_message=f"Cannot resolve symbolic link {path}: {e}",
                    )
            
            # Check for network paths
            path_str = str(path)
            if path_str.startswith(('\\\\\\\\', '//', 'smb://', 'nfs://')) or '://' in path_str:
                warnings.append("Path appears to be a network location - may have reliability issues")
                metadata["is_network_path"] = True
            
            # Check for mount points
            if path.is_mount():
                warnings.append("Path is a mount point")
                metadata["is_mount_point"] = True
            
            return ValidationResult(
                valid=True,
                warnings=warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_code="PATH_TYPE_CHECK_ERROR",
                error_message=f"Error checking path type for {path}: {e}",
            )
    
    @staticmethod
    def validate_filesystem_compatibility(path: Path) -> ValidationResult:
        """Validate filesystem compatibility for file watching."""
        try:
            warnings = []
            metadata = {}
            
            # Get filesystem information
            try:
                stat_info = path.stat()
                metadata["device_id"] = stat_info.st_dev
                
                # Try to get filesystem type (platform-specific)
                if hasattr(os, 'statvfs'):  # Unix-like systems
                    statvfs = os.statvfs(path)
                    metadata["filesystem_info"] = {
                        "block_size": statvfs.f_bsize,
                        "fragment_size": statvfs.f_frsize,
                        "total_blocks": statvfs.f_blocks,
                        "free_blocks": statvfs.f_bavail,
                    }
                    
                    # Check available space
                    free_space_bytes = statvfs.f_bavail * statvfs.f_frsize
                    free_space_mb = free_space_bytes / (1024 * 1024)
                    metadata["free_space_mb"] = free_space_mb
                    
                    if free_space_mb < 100:  # Less than 100MB
                        warnings.append(f"Low disk space: {free_space_mb:.1f} MB available")
                    
            except Exception as e:
                logger.debug(f"Could not get filesystem info for {path}: {e}")
                warnings.append("Could not determine filesystem information")
            
            # Test file watching capabilities
            try:
                # Create a temporary file to test watching
                test_file = path / f".watch_test_{int(time.time())}.tmp"
                try:
                    test_file.touch()
                    test_file.unlink()
                    metadata["supports_file_creation"] = True
                except PermissionError:
                    warnings.append("Cannot create test files - watch reliability may be affected")
                    metadata["supports_file_creation"] = False
                except Exception as e:
                    warnings.append(f"File creation test failed: {e}")
                    metadata["supports_file_creation"] = False
            except Exception as e:
                logger.debug(f"File watching test failed for {path}: {e}")
                warnings.append("Could not test file watching capabilities")
            
            return ValidationResult(
                valid=True,
                warnings=warnings,
                metadata=metadata,
            )
            
        except Exception as e:
            return ValidationResult(
                valid=False,
                error_code="FILESYSTEM_CHECK_ERROR",
                error_message=f"Error checking filesystem compatibility for {path}: {e}",
            )
    
    @classmethod
    def validate_watch_path(cls, path: str | Path) -> ValidationResult:
        """Perform comprehensive validation of a watch path."""
        path = Path(path) if isinstance(path, str) else path
        
        all_warnings = []
        all_metadata = {}
        
        # Run all validation checks
        validators = [
            cls.validate_path_existence,
            cls.validate_permissions,
            cls.validate_path_type,
            cls.validate_filesystem_compatibility,
        ]
        
        for validator in validators:
            result = validator(path)
            
            if not result.valid:
                # Return first critical failure
                return result
            
            all_warnings.extend(result.warnings)
            all_metadata.update(result.metadata)
        
        return ValidationResult(
            valid=True,
            warnings=all_warnings,
            metadata=all_metadata,
        )


class WatchErrorRecovery:
    """Error recovery mechanisms for watch operations."""
    
    def __init__(self):
        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = {}
        self.max_recovery_attempts = 5
        self.recovery_backoff_seconds = [1, 2, 5, 10, 30]  # Progressive backoff
    
    async def attempt_recovery(
        self, 
        watch_id: str, 
        error_type: str, 
        path: Path,
        error_details: str = ""
    ) -> Tuple[bool, str]:
        """Attempt to recover from a watch error."""
        
        if watch_id not in self.recovery_attempts:
            self.recovery_attempts[watch_id] = []
        
        attempts = self.recovery_attempts[watch_id]
        attempt_count = len([a for a in attempts if a.error_type == error_type])
        
        if attempt_count >= self.max_recovery_attempts:
            return False, f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded for {error_type}"
        
        # Determine recovery strategy based on error type
        recovery_strategy = self._get_recovery_strategy(error_type)
        if not recovery_strategy:
            return False, f"No recovery strategy available for error type: {error_type}"
        
        # Apply backoff delay
        if attempt_count > 0:
            backoff_delay = self.recovery_backoff_seconds[min(attempt_count - 1, len(self.recovery_backoff_seconds) - 1)]
            logger.info(f"Waiting {backoff_delay} seconds before recovery attempt {attempt_count + 1}")
            await asyncio.sleep(backoff_delay)
        
        # Attempt recovery
        success, details = await self._execute_recovery_strategy(recovery_strategy, path, error_details)
        
        # Record attempt
        attempt = RecoveryAttempt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            recovery_action=recovery_strategy["name"],
            success=success,
            details=details,
        )
        
        attempts.append(attempt)
        logger.info(f"Recovery attempt for {watch_id} ({error_type}): {'SUCCESS' if success else 'FAILED'} - {details}")
        
        return success, details
    
    def _get_recovery_strategy(self, error_type: str) -> Optional[Dict[str, Any]]:
        """Get recovery strategy for a specific error type."""
        
        strategies = {
            "PATH_NOT_EXISTS": {
                "name": "wait_for_path_creation",
                "description": "Wait for path to be created",
                "action": "wait_and_validate"
            },
            "PATH_ACCESS_DENIED": {
                "name": "retry_access",
                "description": "Retry accessing path after delay",
                "action": "validate_permissions"
            },
            "NETWORK_PATH_UNAVAILABLE": {
                "name": "reconnect_network_path",
                "description": "Attempt to reconnect to network path",
                "action": "wait_and_validate"
            },
            "FILESYSTEM_ERROR": {
                "name": "retry_filesystem_operation",
                "description": "Retry filesystem operation",
                "action": "validate_filesystem"
            },
            "SYMLINK_BROKEN": {
                "name": "wait_for_symlink_target",
                "description": "Wait for symbolic link target to become available",
                "action": "validate_symlink"
            },
        }
        
        return strategies.get(error_type)
    
    async def _execute_recovery_strategy(
        self, 
        strategy: Dict[str, Any], 
        path: Path, 
        error_details: str
    ) -> Tuple[bool, str]:
        """Execute a specific recovery strategy."""
        
        try:
            action = strategy.get("action", "wait_and_validate")
            
            if action == "wait_and_validate":
                # Simple wait and re-validate
                result = WatchPathValidator.validate_path_existence(path)
                if result.valid:
                    return True, f"Path is now accessible: {path}"
                else:
                    return False, f"Path still not accessible: {result.error_message}"
            
            elif action == "validate_permissions":
                # Re-check permissions
                result = WatchPathValidator.validate_permissions(path)
                if result.valid:
                    return True, f"Permissions are now valid: {path}"
                else:
                    return False, f"Permission issues persist: {result.error_message}"
            
            elif action == "validate_filesystem":
                # Check filesystem compatibility
                result = WatchPathValidator.validate_filesystem_compatibility(path)
                if result.valid:
                    return True, f"Filesystem is now accessible: {path}"
                else:
                    return False, f"Filesystem issues persist: {result.error_message}"
            
            elif action == "validate_symlink":
                # Re-validate symlink
                result = WatchPathValidator.validate_path_type(path)
                if result.valid:
                    return True, f"Symbolic link is now valid: {path}"
                else:
                    return False, f"Symbolic link issues persist: {result.error_message}"
            
            else:
                return False, f"Unknown recovery action: {action}"
                
        except Exception as e:
            return False, f"Recovery strategy execution failed: {e}"
    
    def get_recovery_history(self, watch_id: str) -> List[Dict[str, Any]]:
        """Get recovery attempt history for a watch."""
        attempts = self.recovery_attempts.get(watch_id, [])
        return [attempt.to_dict() for attempt in attempts]
    
    def clear_recovery_history(self, watch_id: str) -> None:
        """Clear recovery history for a watch."""
        if watch_id in self.recovery_attempts:
            del self.recovery_attempts[watch_id]
    
    def should_retry_recovery(self, watch_id: str, error_type: str) -> bool:
        """Check if recovery should be attempted for an error."""
        attempts = self.recovery_attempts.get(watch_id, [])
        error_attempts = [a for a in attempts if a.error_type == error_type]
        
        return len(error_attempts) < self.max_recovery_attempts


class WatchHealthMonitor:
    """Monitor watch health and trigger recovery when needed."""
    
    def __init__(self, error_recovery: WatchErrorRecovery):
        self.error_recovery = error_recovery
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.monitoring_interval = 300  # 5 minutes
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_watch(self, watch_id: str, path: Path) -> None:
        """Register a watch for health monitoring."""
        self.health_checks[watch_id] = {
            "path": path,
            "last_check": None,
            "consecutive_failures": 0,
            "last_success": None,
            "status": "unknown"
        }
    
    def unregister_watch(self, watch_id: str) -> None:
        """Unregister a watch from health monitoring."""
        if watch_id in self.health_checks:
            del self.health_checks[watch_id]
    
    async def start_monitoring(self) -> None:
        """Start the health monitoring task."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started watch health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop the health monitoring task."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped watch health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(min(self.monitoring_interval, 60))  # Wait at least 1 minute on error
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered watches."""
        current_time = datetime.now(timezone.utc).isoformat()
        
        for watch_id, health_info in self.health_checks.items():
            try:
                path = health_info["path"]
                validation_result = WatchPathValidator.validate_watch_path(path)
                
                health_info["last_check"] = current_time
                
                if validation_result.valid:
                    # Health check passed
                    health_info["consecutive_failures"] = 0
                    health_info["last_success"] = current_time
                    health_info["status"] = "healthy"
                else:
                    # Health check failed
                    health_info["consecutive_failures"] += 1
                    health_info["status"] = "unhealthy"
                    
                    logger.warning(
                        f"Health check failed for watch {watch_id}: {validation_result.error_message}"
                    )
                    
                    # Trigger recovery if multiple failures
                    if health_info["consecutive_failures"] >= 3:
                        logger.info(f"Triggering recovery for watch {watch_id} after {health_info['consecutive_failures']} failures")
                        
                        success, details = await self.error_recovery.attempt_recovery(
                            watch_id=watch_id,
                            error_type=validation_result.error_code or "HEALTH_CHECK_FAILED",
                            path=path,
                            error_details=validation_result.error_message or "Health check failed"
                        )
                        
                        if success:
                            health_info["consecutive_failures"] = 0
                            health_info["last_success"] = current_time
                            health_info["status"] = "recovered"
                            logger.info(f"Recovery successful for watch {watch_id}: {details}")
                        else:
                            logger.error(f"Recovery failed for watch {watch_id}: {details}")
                            health_info["status"] = "recovery_failed"
            
            except Exception as e:
                logger.error(f"Error during health check for watch {watch_id}: {e}")
                health_info["status"] = "check_error"
    
    def get_health_status(self, watch_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for watches."""
        if watch_id:
            return self.health_checks.get(watch_id, {})
        else:
            return self.health_checks.copy()
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently running."""
        return self._running