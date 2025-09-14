"""
Unified configuration system for daemon and MCP server.

This module provides a unified configuration interface that supports both TOML (for Rust daemon)
and YAML (for Python MCP server) formats with automatic format detection, conversion utilities,
and shared schema validation.

Features:
- Automatic format detection based on file extension
- Bidirectional format conversion (TOML ↔ YAML)
- Environment variable override support
- Configuration file watching with hot-reload
- Shared schema validation for both formats
- Migration utilities between formats
- Comprehensive error handling and logging

Example:
    ```python
    from workspace_qdrant_mcp.core.unified_config import UnifiedConfigManager
    
    # Auto-detect format and load
    config_manager = UnifiedConfigManager()
    config = config_manager.load_config()
    
    # Convert between formats
    config_manager.convert_config('config.toml', 'config.yaml')
    
    # Watch for changes
    config_manager.watch_config(callback=on_config_change)
    ```
"""

import os
from common.logging import get_logger
import yaml
import toml
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List
from enum import Enum
from dataclasses import dataclass
from threading import Thread, Event
from time import sleep
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pydantic import BaseModel, Field, ValidationError

from .config import (
    Config, 
    QdrantConfig, 
    EmbeddingConfig, 
    WorkspaceConfig, 
    GrpcConfig, 
    AutoIngestionConfig
)

logger = get_logger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats."""
    TOML = "toml"
    YAML = "yaml"
    JSON = "json"  # For future extensibility


@dataclass
class ConfigSource:
    """Configuration source information."""
    file_path: Path
    format: ConfigFormat
    exists: bool
    last_modified: Optional[float] = None


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigFormatError(Exception):
    """Raised when configuration format is invalid or unsupported."""
    pass


class ConfigWatchHandler(FileSystemEventHandler):
    """File system event handler for configuration file watching."""
    
    def __init__(self, config_files: List[Path], callback: Callable[[Path], None]):
        self.config_files = set(str(f.resolve()) for f in config_files)
        self.callback = callback
        
    def on_modified(self, event):
        if not event.is_directory and str(Path(event.src_path).resolve()) in self.config_files:
            try:
                self.callback(Path(event.src_path))
            except Exception as e:
                logger.error(f"Error in config watch callback: {e}")


class UnifiedConfigManager:
    """
    Unified configuration manager supporting both TOML and YAML formats.
    
    This class provides a single interface for managing configuration that can be
    consumed by both the Rust daemon (TOML preference) and Python MCP server 
    (YAML preference). It handles format detection, conversion, validation, and
    hot-reloading.
    """
    
    def __init__(self, 
                 config_dir: Optional[Union[str, Path]] = None,
                 env_prefix: str = "WORKSPACE_QDRANT_"):
        """
        Initialize the unified configuration manager.
        
        Args:
            config_dir: Directory to search for configuration files
            env_prefix: Prefix for environment variable overrides
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.env_prefix = env_prefix
        self.config_sources: List[ConfigSource] = []
        self.current_config: Optional[Config] = None
        self.observer: Optional[Observer] = None
        self.watch_callbacks: List[Callable[[Config], None]] = []
        
        # Standard configuration file names in order of preference
        self.config_file_patterns = [
            # TOML files (preferred by Rust daemon)
            "workspace_qdrant_config.toml",
            ".workspace-qdrant.toml",
            "config.toml",
            # YAML files (preferred by Python MCP server)  
            "workspace_qdrant_config.yaml",
            "workspace_qdrant_config.yml",
            ".workspace-qdrant.yaml",
            ".workspace-qdrant.yml",
            "config.yaml",
            "config.yml"
        ]
        
        self._discover_config_sources()
    
    def _discover_config_sources(self) -> None:
        """Discover available configuration files."""
        self.config_sources.clear()
        
        for pattern in self.config_file_patterns:
            config_file = self.config_dir / pattern
            format_type = self._detect_format(config_file)
            
            source = ConfigSource(
                file_path=config_file,
                format=format_type,
                exists=config_file.exists(),
                last_modified=config_file.stat().st_mtime if config_file.exists() else None
            )
            
            self.config_sources.append(source)
            
        logger.debug(f"Discovered {len([s for s in self.config_sources if s.exists])} configuration files")
    
    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """
        Detect configuration format from file extension.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            ConfigFormat: Detected format
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.toml':
            return ConfigFormat.TOML
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML  
        elif suffix == '.json':
            return ConfigFormat.JSON
        else:
            # Default to YAML for extensionless files
            return ConfigFormat.YAML
    
    def get_preferred_config_source(self, prefer_format: Optional[ConfigFormat] = None) -> Optional[ConfigSource]:
        """
        Get the preferred configuration source.
        
        Args:
            prefer_format: Preferred format (TOML for Rust, YAML for Python)
            
        Returns:
            ConfigSource: The preferred source or None if no config found
        """
        existing_sources = [s for s in self.config_sources if s.exists]
        
        if not existing_sources:
            return None
        
        # If format preference specified, try to find that format first
        if prefer_format:
            format_sources = [s for s in existing_sources if s.format == prefer_format]
            if format_sources:
                return format_sources[0]
        
        # Return first existing source (follows discovery order)
        return existing_sources[0]
    
    def load_config(self, 
                   config_file: Optional[Union[str, Path]] = None,
                   prefer_format: Optional[ConfigFormat] = None) -> Config:
        """
        Load configuration from file with environment variable overrides.
        
        Args:
            config_file: Specific config file to load (optional)
            prefer_format: Preferred format for auto-discovery
            
        Returns:
            Config: Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration is invalid
            ConfigFormatError: If configuration format is unsupported
        """
        config_data = {}
        source_file = None
        
        # Load from specific file or auto-discover
        if config_file:
            source_file = Path(config_file)
            if not source_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {source_file}")
            config_data = self._load_config_file(source_file)
        else:
            # Auto-discover configuration
            source = self.get_preferred_config_source(prefer_format)
            if source:
                source_file = source.file_path
                config_data = self._load_config_file(source_file)
                logger.info(f"Loaded configuration from: {source_file}")
            else:
                logger.info("No configuration file found, using defaults")
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Create Config instance with validation
        try:
            if config_data:
                # Handle nested configuration sections
                processed_data = self._process_config_structure(config_data)
                self.current_config = Config(**processed_data)
            else:
                self.current_config = Config()
                
            # Validate the complete configuration
            validation_issues = self.current_config.validate_config()
            if validation_issues:
                error_msg = f"Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in validation_issues)
                logger.error(error_msg)
                raise ConfigValidationError(error_msg)
            
            logger.info("Configuration loaded and validated successfully")
            return self.current_config
            
        except ValidationError as e:
            error_msg = f"Configuration validation error: {e}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration data from file based on format.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Dict containing configuration data
            
        Raises:
            ConfigFormatError: If file format is invalid
        """
        format_type = self._detect_format(file_path)
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
            
            if format_type == ConfigFormat.TOML:
                return toml.loads(content)
            elif format_type == ConfigFormat.YAML:
                return yaml.safe_load(content) or {}
            elif format_type == ConfigFormat.JSON:
                import json
                return json.loads(content)
            else:
                raise ConfigFormatError(f"Unsupported configuration format: {format_type}")
                
        except (toml.TomlDecodeError, yaml.YAMLError, ValueError) as e:
            raise ConfigFormatError(f"Error parsing {format_type.value} file {file_path}: {e}") from e
        except Exception as e:
            raise ConfigFormatError(f"Error reading configuration file {file_path}: {e}") from e
    
    def _process_config_structure(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration data structure to match Pydantic model expectations.
        
        Args:
            config_data: Raw configuration data
            
        Returns:
            Dict: Processed configuration data
        """
        processed = {}
        
        # Map TOML structure to Config model structure
        for key, value in config_data.items():
            if key == "qdrant" and isinstance(value, dict):
                # Handle Qdrant configuration
                qdrant_config = {}
                for k, v in value.items():
                    if k == "url":
                        qdrant_config["url"] = v
                    elif k == "api_key":
                        qdrant_config["api_key"] = v
                    elif k in ["timeout", "timeout_ms"]:
                        # Handle both timeout and timeout_ms
                        qdrant_config["timeout"] = v // 1000 if k == "timeout_ms" else v
                    elif k == "transport":
                        qdrant_config["prefer_grpc"] = v.lower() == "grpc"
                    elif k in ["dense_vector_size", "http2", "max_retries", "retry_delay_ms", 
                              "pool_size", "tls", "check_compatibility"]:
                        # Store additional config for potential future use
                        continue
                processed["qdrant"] = QdrantConfig(**qdrant_config)
                
            elif key == "embedding" and isinstance(value, dict):
                processed["embedding"] = EmbeddingConfig(**value)
                
            elif key == "workspace" and isinstance(value, dict):
                processed["workspace"] = WorkspaceConfig(**value)
                
            elif key == "grpc" and isinstance(value, dict):
                processed["grpc"] = GrpcConfig(**value)
                
            elif key == "auto_ingestion" and isinstance(value, dict):
                processed["auto_ingestion"] = AutoIngestionConfig(**value)
                
            elif key == "daemon" and isinstance(value, dict):
                # Handle daemon-specific settings that don't directly map
                # These might be Rust-specific and should be preserved
                continue
                
            elif key == "logging" and isinstance(value, dict):
                # Logging configuration might be format-specific
                continue
                
            elif key in ["host", "port", "debug"]:
                # Server-level configuration
                processed[key] = value
                
            elif key in ["log_file", "max_concurrent_tasks", "default_timeout_ms", 
                        "enable_preemption", "chunk_size", "enable_lsp", "log_level",
                        "enable_metrics", "metrics_interval_secs", "project_path"]:
                # Daemon-specific settings that might affect server config
                # Map some of these to appropriate server settings
                if key == "log_level":
                    processed["debug"] = value.lower() in ["debug", "trace"]
                # Other daemon settings are preserved for access via raw config
                continue
            else:
                # Pass through unknown keys
                processed[key] = value
        
        return processed
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration data.
        
        Args:
            config_data: Base configuration data
            
        Returns:
            Dict: Configuration data with environment overrides applied
        """
        # Create a mutable copy
        result = config_data.copy()
        
        # Apply environment variable overrides using the same pattern as Config class
        env_overrides = {}
        
        # Server-level overrides
        if host := os.getenv(f"{self.env_prefix}HOST"):
            env_overrides["host"] = host
        if port := os.getenv(f"{self.env_prefix}PORT"):
            env_overrides["port"] = int(port)
        if debug := os.getenv(f"{self.env_prefix}DEBUG"):
            env_overrides["debug"] = debug.lower() == "true"
            
        # Nested configuration overrides
        nested_mappings = {
            "QDRANT__URL": ("qdrant", "url"),
            "QDRANT__API_KEY": ("qdrant", "api_key"),
            "QDRANT__TIMEOUT": ("qdrant", "timeout"),
            "QDRANT__PREFER_GRPC": ("qdrant", "prefer_grpc"),
            "EMBEDDING__MODEL": ("embedding", "model"),
            "EMBEDDING__ENABLE_SPARSE_VECTORS": ("embedding", "enable_sparse_vectors"),
            "EMBEDDING__CHUNK_SIZE": ("embedding", "chunk_size"),
            "EMBEDDING__CHUNK_OVERLAP": ("embedding", "chunk_overlap"),
            "EMBEDDING__BATCH_SIZE": ("embedding", "batch_size"),
            "WORKSPACE__COLLECTION_SUFFIXES": ("workspace", "collection_suffixes"),
            "WORKSPACE__GLOBAL_COLLECTIONS": ("workspace", "global_collections"),
            "WORKSPACE__GITHUB_USER": ("workspace", "github_user"),
            "WORKSPACE__COLLECTION_PREFIX": ("workspace", "collection_prefix"),
            "WORKSPACE__MAX_COLLECTIONS": ("workspace", "max_collections"),
            "WORKSPACE__AUTO_CREATE_COLLECTIONS": ("workspace", "auto_create_collections"),
            "AUTO_INGESTION__ENABLED": ("auto_ingestion", "enabled"),
            "AUTO_INGESTION__AUTO_CREATE_WATCHES": ("auto_ingestion", "auto_create_watches"),
            "AUTO_INGESTION__TARGET_COLLECTION_SUFFIX": ("auto_ingestion", "target_collection_suffix"),
            "GRPC__ENABLED": ("grpc", "enabled"),
            "GRPC__HOST": ("grpc", "host"),
            "GRPC__PORT": ("grpc", "port"),
        }
        
        for env_key, (section, field) in nested_mappings.items():
            env_value = os.getenv(f"{self.env_prefix}{env_key}")
            if env_value is not None:
                # Initialize section if needed
                if section not in result:
                    result[section] = {}
                
                # Type conversion based on field
                if field in ["timeout", "chunk_size", "chunk_overlap", "batch_size", 
                            "max_collections", "port"]:
                    result[section][field] = int(env_value)
                elif field in ["prefer_grpc", "enable_sparse_vectors", "auto_create_collections",
                              "enabled", "auto_create_watches"]:
                    result[section][field] = env_value.lower() == "true"
                elif field in ["collection_suffixes", "global_collections"]:
                    result[section][field] = [c.strip() for c in env_value.split(",") if c.strip()]
                else:
                    result[section][field] = env_value
        
        # Apply server-level overrides
        result.update(env_overrides)
        
        return result
    
    def save_config(self, 
                   config: Config, 
                   file_path: Union[str, Path],
                   format_type: Optional[ConfigFormat] = None) -> None:
        """
        Save configuration to file in specified format.
        
        Args:
            config: Configuration to save
            file_path: Target file path
            format_type: Format to save in (auto-detect if None)
            
        Raises:
            ConfigFormatError: If format is unsupported
        """
        file_path = Path(file_path)
        
        if format_type is None:
            format_type = self._detect_format(file_path)
        
        # Convert Config to dictionary
        config_dict = self._config_to_dict(config)
        
        # Write in specified format
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with file_path.open('w', encoding='utf-8') as f:
                if format_type == ConfigFormat.TOML:
                    toml.dump(config_dict, f)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                elif format_type == ConfigFormat.JSON:
                    import json
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigFormatError(f"Unsupported format for saving: {format_type}")
                    
            logger.info(f"Configuration saved to {file_path} in {format_type.value} format")
            
        except Exception as e:
            raise ConfigFormatError(f"Error saving configuration to {file_path}: {e}") from e
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """
        Convert Config object to dictionary representation.
        
        Args:
            config: Config object to convert
            
        Returns:
            Dict: Configuration as dictionary
        """
        return {
            "host": config.host,
            "port": config.port,
            "debug": config.debug,
            "qdrant": {
                "url": config.qdrant.url,
                "api_key": config.qdrant.api_key,
                "timeout": config.qdrant.timeout,
                "prefer_grpc": config.qdrant.prefer_grpc,
            },
            "embedding": {
                "model": config.embedding.model,
                "enable_sparse_vectors": config.embedding.enable_sparse_vectors,
                "chunk_size": config.embedding.chunk_size,
                "chunk_overlap": config.embedding.chunk_overlap,
                "batch_size": config.embedding.batch_size,
            },
            "workspace": {
                "collection_suffixes": config.workspace.collection_suffixes,
                "global_collections": config.workspace.global_collections,
                "github_user": config.workspace.github_user,
                "collection_prefix": config.workspace.collection_prefix,
                "max_collections": config.workspace.max_collections,
                "auto_create_collections": config.workspace.auto_create_collections,
            },
            "grpc": {
                "enabled": config.grpc.enabled,
                "host": config.grpc.host,
                "port": config.grpc.port,
                "fallback_to_direct": config.grpc.fallback_to_direct,
                "connection_timeout": config.grpc.connection_timeout,
                "max_retries": config.grpc.max_retries,
            },
            "auto_ingestion": {
                "enabled": config.auto_ingestion.enabled,
                "auto_create_watches": config.auto_ingestion.auto_create_watches,
                "include_common_files": config.auto_ingestion.include_common_files,
                "include_source_files": config.auto_ingestion.include_source_files,
                "target_collection_suffix": config.auto_ingestion.target_collection_suffix,
                "max_files_per_batch": config.auto_ingestion.max_files_per_batch,
                "batch_delay_seconds": config.auto_ingestion.batch_delay_seconds,
                "max_file_size_mb": config.auto_ingestion.max_file_size_mb,
                "recursive_depth": config.auto_ingestion.recursive_depth,
                "debounce_seconds": config.auto_ingestion.debounce_seconds,
            },
        }
    
    def convert_config(self, 
                      source_file: Union[str, Path], 
                      target_file: Union[str, Path],
                      target_format: Optional[ConfigFormat] = None) -> None:
        """
        Convert configuration file between formats.
        
        Args:
            source_file: Source configuration file
            target_file: Target configuration file
            target_format: Target format (auto-detect if None)
            
        Raises:
            ConfigFormatError: If conversion fails
        """
        source_path = Path(source_file)
        target_path = Path(target_file)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source configuration file not found: {source_path}")
        
        if target_format is None:
            target_format = self._detect_format(target_path)
        
        logger.info(f"Converting configuration from {source_path} to {target_path} ({target_format.value})")
        
        # Load source configuration
        config_data = self._load_config_file(source_path)
        processed_data = self._process_config_structure(config_data)
        config = Config(**processed_data)
        
        # Save in target format
        self.save_config(config, target_path, target_format)
        
        logger.info(f"Configuration conversion completed: {source_path} -> {target_path}")
    
    def watch_config(self, callback: Callable[[Config], None]) -> None:
        """
        Start watching configuration files for changes.
        
        Args:
            callback: Function to call when configuration changes
        """
        if self.observer is not None:
            logger.warning("Configuration watching already active")
            return
        
        config_files = [source.file_path for source in self.config_sources if source.exists]
        if not config_files:
            logger.warning("No configuration files to watch")
            return
        
        self.watch_callbacks.append(callback)
        
        def on_config_change(file_path: Path):
            try:
                logger.info(f"Configuration file changed: {file_path}")
                new_config = self.load_config(file_path)
                for cb in self.watch_callbacks:
                    try:
                        cb(new_config)
                    except Exception as e:
                        logger.error(f"Error in config change callback: {e}")
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}")
        
        # Set up file system watcher
        self.observer = Observer()
        handler = ConfigWatchHandler(config_files, on_config_change)
        
        # Watch the directory containing config files
        watch_dirs = set(f.parent for f in config_files)
        for watch_dir in watch_dirs:
            self.observer.schedule(handler, str(watch_dir), recursive=False)
        
        self.observer.start()
        logger.info(f"Started watching {len(config_files)} configuration files")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watch_callbacks.clear()
            logger.info("Stopped configuration file watching")
    
    def validate_config_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Validate a configuration file without loading it.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            List[str]: Validation issues (empty if valid)
        """
        try:
            config_data = self._load_config_file(Path(file_path))
            processed_data = self._process_config_structure(config_data)
            config = Config(**processed_data)
            return config.validate_config()
        except Exception as e:
            return [f"Configuration validation error: {e}"]
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about discovered configuration sources.
        
        Returns:
            Dict: Information about configuration sources
        """
        return {
            "config_dir": str(self.config_dir),
            "env_prefix": self.env_prefix,
            "sources": [
                {
                    "file_path": str(source.file_path),
                    "format": source.format.value,
                    "exists": source.exists,
                    "last_modified": source.last_modified,
                }
                for source in self.config_sources
            ],
            "preferred_source": str(self.get_preferred_config_source().file_path) 
                               if self.get_preferred_config_source() else None,
            "current_config_loaded": self.current_config is not None,
        }
    
    def create_default_configs(self, formats: List[ConfigFormat] = None) -> Dict[ConfigFormat, Path]:
        """
        Create default configuration files in specified formats.
        
        Args:
            formats: List of formats to create (default: TOML and YAML)
            
        Returns:
            Dict mapping formats to created file paths
        """
        if formats is None:
            formats = [ConfigFormat.TOML, ConfigFormat.YAML]
        
        default_config = Config()
        created_files = {}
        
        for format_type in formats:
            if format_type == ConfigFormat.TOML:
                file_path = self.config_dir / "workspace_qdrant_config.toml"
            elif format_type == ConfigFormat.YAML:
                file_path = self.config_dir / "workspace_qdrant_config.yaml"
            elif format_type == ConfigFormat.JSON:
                file_path = self.config_dir / "workspace_qdrant_config.json"
            else:
                continue
            
            if not file_path.exists():
                self.save_config(default_config, file_path, format_type)
                created_files[format_type] = file_path
                logger.info(f"Created default configuration: {file_path}")
            else:
                logger.info(f"Configuration file already exists: {file_path}")
        
        return created_files
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup watchers."""
        self.stop_watching()