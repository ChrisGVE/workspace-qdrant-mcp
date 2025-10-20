"""
Project-Specific Configuration Management System for Multi-Instance Daemons.

This module provides isolated configuration management for each daemon instance with
project-specific settings, validation, inheritance, and hot-reloading capabilities.
"""

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager

import yaml
from pydantic import BaseModel, Field, ValidationError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import get_config_manager
from .resource_manager import ResourceLimits
from ..utils.project_detection import ProjectDetector
from loguru import logger

# logger imported from loguru


class ConfigScope(Enum):
    """Configuration scope enumeration."""
    GLOBAL = "global"
    PROJECT = "project" 
    INSTANCE = "instance"


@dataclass
class ConfigSource:
    """Information about a configuration source."""
    
    scope: ConfigScope
    path: Path
    last_modified: datetime
    checksum: str
    priority: int  # Higher number = higher priority
    
    def is_stale(self, other_modified: datetime) -> bool:
        """Check if this config source is stale compared to another."""
        return self.last_modified < other_modified


class DaemonProjectConfig(BaseModel):
    """Project-specific daemon configuration."""
    
    # Project identification
    project_name: str
    project_path: str
    project_id: Optional[str] = None
    
    # Daemon settings
    grpc_host: str = "127.0.0.1"
    grpc_port: Optional[int] = None  # Auto-allocated if None
    log_level: str = "info"
    max_concurrent_jobs: int = 4
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_open_files: int = 1024
    
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # Processing settings
    health_check_interval: float = 90.0  # Optimized for lower idle CPU usage (was 30.0)
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    
    # Ingestion settings
    default_collection: Optional[str] = None
    auto_create_collections: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Watch settings
    enable_file_watching: bool = True
    watch_patterns: List[str] = Field(default_factory=lambda: ["**/*.py", "**/*.md", "**/*.txt"])
    ignore_patterns: List[str] = Field(default_factory=lambda: ["**/.git/**", "**/node_modules/**"])
    
    # Advanced settings
    enable_resource_monitoring: bool = True
    enable_metrics_collection: bool = True
    config_hot_reload: bool = True
    
    # Custom settings (for extensibility)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    def merge_with(self, other: "DaemonProjectConfig") -> "DaemonProjectConfig":
        """Merge this config with another, with the other taking precedence."""
        merged_data = self.model_dump()
        other_data = other.model_dump()
        
        # Deep merge dictionaries
        for key, value in other_data.items():
            if value is not None:
                if isinstance(value, dict) and key in merged_data:
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value
        
        return DaemonProjectConfig(**merged_data)


class ConfigWatcher(FileSystemEventHandler):
    """Watches configuration files for changes."""
    
    def __init__(self, callback: Callable[[Path], None]):
        super().__init__()
        self.callback = callback
        self.watched_files: Set[Path] = set()
    
    def add_file(self, path: Path) -> None:
        """Add a file to watch."""
        self.watched_files.add(path.resolve())
    
    def remove_file(self, path: Path) -> None:
        """Remove a file from watching."""
        self.watched_files.discard(path.resolve())
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_path = Path(event.src_path).resolve()
            if file_path in self.watched_files:
                logger.info(f"Configuration file changed: {file_path}")
                try:
                    self.callback(file_path)
                except Exception as e:
                    logger.error(f"Error handling config change: {e}")


class ProjectConfigManager:
    """
    Manages project-specific daemon configurations with inheritance and hot-reloading.
    
    Configuration hierarchy (highest to lowest priority):
    1. Instance-specific config (.wqm-daemon-{instance_id}.json)
    2. Project-local config (.wqm-daemon.json)
    3. Global daemon config (~/.config/workspace-qdrant/daemon.yaml)
    4. Built-in defaults
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.project_detector = ProjectDetector()
        
        # Generate project identifier
        self.daemon_identifier = self.project_detector.create_daemon_identifier(str(self.project_path))
        self.project_id = self.daemon_identifier.generate_identifier()
        
        # Configuration sources
        self.config_sources: List[ConfigSource] = []
        self.cached_config: Optional[DaemonProjectConfig] = None
        self.config_checksum: Optional[str] = None
        
        # File watching
        self.observer: Optional[Observer] = None
        self.watcher: Optional[ConfigWatcher] = None
        self.hot_reload_enabled = True
        self.change_callbacks: List[Callable[[DaemonProjectConfig], None]] = []
        
        # Backup management
        self.backup_dir = self.project_path / ".wqm" / "config-backups"
        self.max_backups = 10
    
    def add_change_callback(self, callback: Callable[[DaemonProjectConfig], None]) -> None:
        """Add a callback to be called when configuration changes."""
        self.change_callbacks.append(callback)
    
    def get_config_paths(self, instance_id: Optional[str] = None) -> Dict[ConfigScope, Path]:
        """Get all possible configuration file paths."""
        paths = {}
        
        # Instance-specific config
        if instance_id:
            paths[ConfigScope.INSTANCE] = self.project_path / f".wqm-daemon-{instance_id}.json"
        
        # Project-local config
        paths[ConfigScope.PROJECT] = self.project_path / ".wqm-daemon.json"
        
        # Global config
        global_config_dir = Path.home() / ".config" / "workspace-qdrant"
        paths[ConfigScope.GLOBAL] = global_config_dir / "daemon.yaml"
        
        return paths
    
    def load_config(self, instance_id: Optional[str] = None, force_reload: bool = False) -> DaemonProjectConfig:
        """
        Load configuration with inheritance from multiple sources.
        
        Args:
            instance_id: Optional instance identifier for instance-specific config
            force_reload: Force reload even if cached config exists
            
        Returns:
            Merged configuration from all sources
        """
        # Check if we can use cached config
        if not force_reload and self.cached_config and not self._config_changed():
            return self.cached_config
        
        logger.info(f"Loading configuration for project {self.project_id}")
        
        # Start with defaults
        project_name = self.project_path.name
        base_config = DaemonProjectConfig(
            project_name=project_name,
            project_path=str(self.project_path),
            project_id=self.project_id
        )
        
        # Load and merge configurations in priority order
        config_paths = self.get_config_paths(instance_id)
        self.config_sources.clear()
        
        # Global config (lowest priority)
        global_path = config_paths[ConfigScope.GLOBAL]
        if global_path.exists():
            try:
                global_config = self._load_config_file(global_path, ConfigScope.GLOBAL)
                if global_config:
                    base_config = base_config.merge_with(global_config)
                    logger.debug(f"Loaded global config from {global_path}")
            except Exception as e:
                logger.warning(f"Failed to load global config from {global_path}: {e}")
        
        # Project config (medium priority)
        project_path = config_paths[ConfigScope.PROJECT]
        if project_path.exists():
            try:
                project_config = self._load_config_file(project_path, ConfigScope.PROJECT)
                if project_config:
                    base_config = base_config.merge_with(project_config)
                    logger.debug(f"Loaded project config from {project_path}")
            except Exception as e:
                logger.warning(f"Failed to load project config from {project_path}: {e}")
        
        # Instance config (highest priority)
        if ConfigScope.INSTANCE in config_paths:
            instance_path = config_paths[ConfigScope.INSTANCE]
            if instance_path.exists():
                try:
                    instance_config = self._load_config_file(instance_path, ConfigScope.INSTANCE)
                    if instance_config:
                        base_config = base_config.merge_with(instance_config)
                        logger.debug(f"Loaded instance config from {instance_path}")
                except Exception as e:
                    logger.warning(f"Failed to load instance config from {instance_path}: {e}")
        
        # Validate final configuration
        try:
            self.cached_config = base_config
            self.config_checksum = self._calculate_config_checksum()
            logger.info(f"Configuration loaded successfully for project {project_name}")
            return base_config
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def save_config(
        self, 
        config: DaemonProjectConfig, 
        scope: ConfigScope = ConfigScope.PROJECT,
        instance_id: Optional[str] = None,
        create_backup: bool = True
    ) -> Path:
        """
        Save configuration to the specified scope.
        
        Args:
            config: Configuration to save
            scope: Configuration scope (global, project, or instance)
            instance_id: Instance ID for instance-specific configs
            create_backup: Whether to create a backup before saving
            
        Returns:
            Path where configuration was saved
        """
        config_paths = self.get_config_paths(instance_id)
        target_path = config_paths[scope]
        
        # Create backup if requested
        if create_backup and target_path.exists():
            self._create_backup(target_path)
        
        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        if target_path.suffix.lower() in ['.yaml', '.yml']:
            with open(target_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)
        else:
            with open(target_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {target_path}")
        
        # Update cache
        self.cached_config = config
        self.config_checksum = self._calculate_config_checksum()
        
        return target_path
    
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        if not self.hot_reload_enabled or self.observer:
            return
        
        self.watcher = ConfigWatcher(self._on_config_change)
        self.observer = Observer()
        
        # Watch all potential config files
        config_paths = self.get_config_paths()
        for scope, path in config_paths.items():
            if path.exists():
                self.watcher.add_file(path)
                self.observer.schedule(self.watcher, str(path.parent), recursive=False)
        
        self.observer.start()
        logger.info(f"Started configuration watching for project {self.project_id}")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watcher = None
        logger.info(f"Stopped configuration watching for project {self.project_id}")
    
    @asynccontextmanager
    async def config_context(self, instance_id: Optional[str] = None):
        """Context manager for configuration lifecycle."""
        config = self.load_config(instance_id)
        self.start_watching()
        try:
            yield config
        finally:
            self.stop_watching()
    
    def validate_config(self, config: DaemonProjectConfig) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        try:
            # Basic Pydantic validation
            config.model_validate(config.model_dump())
        except ValidationError as e:
            issues.extend([f"Validation error: {error['msg']}" for error in e.errors()])
        
        # Custom validation logic
        if config.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        
        if config.max_cpu_percent <= 0 or config.max_cpu_percent > 100:
            issues.append("max_cpu_percent must be between 0 and 100")
        
        if config.startup_timeout <= 0:
            issues.append("startup_timeout must be positive")
        
        if config.health_check_interval <= 0:
            issues.append("health_check_interval must be positive")
        
        # Check if Qdrant URL is valid
        if not config.qdrant_url.startswith(('http://', 'https://')):
            issues.append("qdrant_url must be a valid HTTP/HTTPS URL")
        
        # Validate collections if specified
        if config.default_collection and not config.default_collection.strip():
            issues.append("default_collection cannot be empty string")
        
        return issues
    
    def get_resource_limits(self, config: Optional[DaemonProjectConfig] = None) -> ResourceLimits:
        """Convert configuration to ResourceLimits object."""
        if not config:
            config = self.load_config()
        
        return ResourceLimits(
            max_memory_mb=config.max_memory_mb,
            max_cpu_percent=config.max_cpu_percent,
            max_open_files=config.max_open_files,
            processing_timeout=config.startup_timeout,
            connection_timeout=config.startup_timeout
        )
    
    def create_template_config(self, scope: ConfigScope = ConfigScope.PROJECT) -> Path:
        """Create a template configuration file."""
        template_config = DaemonProjectConfig(
            project_name=self.project_path.name,
            project_path=str(self.project_path),
            project_id=self.project_id
        )
        
        return self.save_config(template_config, scope, create_backup=False)
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """Restore configuration from a backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            logger.error(f"Backup {backup_name} not found")
            return False
        
        try:
            # Load backup
            with open(backup_path, 'r') as f:
                if backup_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            config = DaemonProjectConfig(**data)
            
            # Save as current config
            self.save_config(config, ConfigScope.PROJECT, create_backup=False)
            logger.info(f"Restored configuration from backup {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_name}: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups."""
        if not self.backup_dir.exists():
            return []
        
        backups = []
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'size': stat.st_size
                })
            except Exception as e:
                logger.warning(f"Error reading backup {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)
    
    def _load_config_file(self, path: Path, scope: ConfigScope) -> Optional[DaemonProjectConfig]:
        """Load configuration from a file."""
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Create config source record
            stat = path.stat()
            source = ConfigSource(
                scope=scope,
                path=path,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                checksum=self._calculate_file_checksum(path),
                priority=scope.value.__hash__()  # Simple priority based on scope
            )
            self.config_sources.append(source)
            
            # Convert to config object
            return DaemonProjectConfig(**data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return None
    
    def _calculate_config_checksum(self) -> str:
        """Calculate checksum of all configuration sources."""
        import hashlib
        
        checksums = []
        for source in self.config_sources:
            checksums.append(f"{source.path}:{source.checksum}")
        
        content = "|".join(sorted(checksums))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_file_checksum(self, path: Path) -> str:
        """Calculate checksum of a file."""
        import hashlib
        
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _config_changed(self) -> bool:
        """Check if any configuration source has changed."""
        if not self.config_checksum:
            return True
        
        current_checksum = self._calculate_config_checksum()
        return current_checksum != self.config_checksum
    
    def _on_config_change(self, changed_path: Path) -> None:
        """Handle configuration file changes."""
        try:
            logger.info(f"Configuration change detected: {changed_path}")
            
            # Reload configuration
            new_config = self.load_config(force_reload=True)
            
            # Validate new configuration
            issues = self.validate_config(new_config)
            if issues:
                logger.error(f"Invalid configuration after reload: {issues}")
                return
            
            # Notify callbacks
            for callback in self.change_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Error handling configuration change: {e}")
    
    def _create_backup(self, config_path: Path) -> None:
        """Create a backup of the configuration file."""
        if not config_path.exists():
            return
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{config_path.stem}_{timestamp}.json"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(config_path, backup_path)
            logger.debug(f"Created configuration backup: {backup_path}")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.warning(f"Failed to create configuration backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files, keeping only the most recent ones."""
        if not self.backup_dir.exists():
            return
        
        backup_files = list(self.backup_dir.glob("*.json"))
        if len(backup_files) <= self.max_backups:
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove excess backups
        for old_backup in backup_files[self.max_backups:]:
            try:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")


# Global config managers for different projects
_config_managers: Dict[str, ProjectConfigManager] = {}


def get_project_config_manager(project_path: str) -> ProjectConfigManager:
    """Get or create a project config manager for the given path."""
    project_path = str(Path(project_path).resolve())
    
    if project_path not in _config_managers:
        _config_managers[project_path] = ProjectConfigManager(project_path)
    
    return _config_managers[project_path]


def cleanup_config_managers() -> None:
    """Cleanup all config managers."""
    for manager in _config_managers.values():
        manager.stop_watching()
    _config_managers.clear()


# Convenience functions
def load_project_config(project_path: str, instance_id: Optional[str] = None) -> DaemonProjectConfig:
    """Load configuration for a project."""
    manager = get_project_config_manager(project_path)
    return manager.load_config(instance_id)


def save_project_config(
    project_path: str,
    config: DaemonProjectConfig,
    scope: ConfigScope = ConfigScope.PROJECT,
    instance_id: Optional[str] = None
) -> Path:
    """Save configuration for a project."""
    manager = get_project_config_manager(project_path)
    return manager.save_config(config, scope, instance_id)