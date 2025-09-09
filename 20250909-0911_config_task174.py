"""Canonical configuration directory system with XDG-compliant resolution and TOML loading.

This module provides a comprehensive configuration management system that:
- Resolves configuration directories using XDG Base Directory Specification
- Supports OS-specific fallbacks for macOS, Linux, and Windows
- Loads TOML configuration files with proper precedence handling
- Validates configuration structure and provides error handling
- Supports both MCP server and daemon configuration files

Configuration precedence (highest to lowest):
1. Command-line arguments
2. Environment variables
3. User configuration files
4. Default values
"""

import os
import platform
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import toml
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Configuration file types supported by the system."""
    MCP_SERVER = "workspace-qdrant-mcp-config.toml"
    DAEMON = "workspace-qdrant-daemon.toml"


@dataclass
class ConfigPaths:
    """Container for configuration-related paths."""
    config_dir: Path
    mcp_config_file: Path
    daemon_config_file: Path
    cache_dir: Optional[Path] = None
    data_dir: Optional[Path] = None


@dataclass
class McpConfig:
    """MCP Server configuration structure."""
    # Server configuration
    server_name: str = "workspace-qdrant-mcp"
    server_version: str = "1.0.0"
    
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_ms: int = 30000
    qdrant_max_retries: int = 3
    qdrant_pool_size: int = 10
    
    # Processing
    max_concurrent_tasks: int = 4
    default_timeout_ms: int = 30000
    chunk_size: int = 1000
    
    # Features
    enable_lsp: bool = True
    enable_metrics: bool = True
    enable_web_ui: bool = False  # Disabled by default for security
    
    # Additional configuration from existing files
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DaemonConfig:
    """Daemon configuration structure."""
    # Daemon settings
    daemon_name: str = "workspace-qdrant-daemon"
    log_file: str = ""
    log_level: str = "info"
    
    # Processing engine
    max_concurrent_tasks: int = 4
    default_timeout_ms: int = 30000
    enable_preemption: bool = True
    chunk_size: int = 1000
    
    # Auto-ingestion
    auto_ingestion_enabled: bool = True
    auto_create_watches: bool = True
    include_common_files: bool = True
    include_source_files: bool = True
    
    # Qdrant connection (inherits from MCP config if available)
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_ms: int = 30000
    qdrant_max_retries: int = 3
    
    # Additional configuration from existing files
    additional_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Exception raised when required configuration file is not found."""
    pass


def resolve_config_directory() -> Path:
    """Resolve configuration directory using XDG Base Directory Specification.
    
    Returns the appropriate configuration directory based on the operating system:
    - macOS: ~/Library/Application Support/workspace-qdrant-mcp
    - Linux: $XDG_CONFIG_HOME/workspace-qdrant-mcp or ~/.config/workspace-qdrant-mcp
    - Windows: %APPDATA%/workspace-qdrant-mcp
    - Other: ~/.workspace-qdrant-mcp
    
    The function respects the XDG_CONFIG_HOME environment variable on Unix-like systems.
    
    Returns:
        Path: The resolved configuration directory path.
        
    Raises:
        ConfigurationError: If the directory cannot be determined or created.
    """
    system = platform.system().lower()
    app_name = "workspace-qdrant-mcp"
    
    try:
        if system == "darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / app_name
        elif system == "linux":
            # Check XDG_CONFIG_HOME first, fallback to ~/.config
            xdg_config = os.environ.get("XDG_CONFIG_HOME")
            if xdg_config:
                config_dir = Path(xdg_config) / app_name
            else:
                config_dir = Path.home() / ".config" / app_name
        elif system == "windows":
            # Use APPDATA on Windows
            appdata = os.environ.get("APPDATA")
            if appdata:
                config_dir = Path(appdata) / app_name
            else:
                config_dir = Path.home() / "AppData" / "Roaming" / app_name
        else:
            # Fallback for other systems
            config_dir = Path.home() / f".{app_name}"
            
        # Create directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Resolved configuration directory: {config_dir}")
        return config_dir
        
    except (OSError, PermissionError) as e:
        raise ConfigurationError(
            f"Failed to create configuration directory: {e}"
        ) from e


def get_config_paths() -> ConfigPaths:
    """Get all configuration-related paths.
    
    Returns:
        ConfigPaths: Container with all relevant configuration paths.
        
    Raises:
        ConfigurationError: If paths cannot be resolved.
    """
    config_dir = resolve_config_directory()
    
    return ConfigPaths(
        config_dir=config_dir,
        mcp_config_file=config_dir / ConfigType.MCP_SERVER.value,
        daemon_config_file=config_dir / ConfigType.DAEMON.value,
        cache_dir=config_dir / "cache",
        data_dir=config_dir / "data"
    )


def load_toml_file(config_path: Path, required: bool = False) -> Dict[str, Any]:
    """Load a TOML configuration file.
    
    Args:
        config_path: Path to the TOML file to load.
        required: Whether the file is required to exist.
        
    Returns:
        Dict containing the parsed TOML data, or empty dict if file doesn't exist
        and is not required.
        
    Raises:
        ConfigFileNotFoundError: If required file is not found.
        ConfigurationError: If file cannot be parsed or read.
    """
    if not config_path.exists():
        if required:
            raise ConfigFileNotFoundError(
                f"Required configuration file not found: {config_path}"
            )
        logger.debug(f"Configuration file not found (optional): {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = toml.load(f)
        logger.debug(f"Loaded configuration from: {config_path}")
        return config_data
        
    except toml.TomlDecodeError as e:
        raise ConfigurationError(
            f"Invalid TOML syntax in {config_path}: {e}"
        ) from e
    except (OSError, IOError) as e:
        raise ConfigurationError(
            f"Failed to read configuration file {config_path}: {e}"
        ) from e


def merge_config_sources(
    defaults: Dict[str, Any],
    user_config: Dict[str, Any],
    env_vars: Dict[str, Any],
    cli_args: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge configuration from multiple sources with proper precedence.
    
    Configuration precedence (highest to lowest):
    1. Command-line arguments (cli_args)
    2. Environment variables (env_vars)
    3. User configuration files (user_config)
    4. Default values (defaults)
    
    Args:
        defaults: Default configuration values.
        user_config: Configuration from user config files.
        env_vars: Configuration from environment variables.
        cli_args: Configuration from command-line arguments.
        
    Returns:
        Dict with merged configuration, higher precedence values override lower ones.
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    # Start with defaults and merge in order of precedence
    merged = defaults.copy()
    merged = deep_merge(merged, user_config)
    merged = deep_merge(merged, env_vars)
    merged = deep_merge(merged, cli_args)
    
    logger.debug("Configuration merged from all sources")
    return merged


def load_mcp_config(
    config_file: Optional[Path] = None,
    env_vars: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> McpConfig:
    """Load MCP server configuration with proper precedence handling.
    
    Args:
        config_file: Optional path to specific config file. If None, uses default location.
        env_vars: Optional environment variable overrides.
        cli_args: Optional command-line argument overrides.
        
    Returns:
        McpConfig: Loaded and validated MCP configuration.
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid.
    """
    if env_vars is None:
        env_vars = {}
    if cli_args is None:
        cli_args = {}
        
    # Determine config file path
    if config_file is None:
        config_paths = get_config_paths()
        config_file = config_paths.mcp_config_file
    
    # Load user configuration
    user_config = load_toml_file(config_file, required=False)
    
    # Default MCP configuration
    defaults = {
        "server_name": "workspace-qdrant-mcp",
        "server_version": "1.0.0",
        "qdrant_url": "http://localhost:6333",
        "qdrant_timeout_ms": 30000,
        "qdrant_max_retries": 3,
        "qdrant_pool_size": 10,
        "max_concurrent_tasks": 4,
        "default_timeout_ms": 30000,
        "chunk_size": 1000,
        "enable_lsp": True,
        "enable_metrics": True,
        "enable_web_ui": False,
    }
    
    # Merge all configuration sources
    merged_config = merge_config_sources(defaults, user_config, env_vars, cli_args)
    
    # Create McpConfig instance
    try:
        # Extract known fields
        known_fields = {
            field.name for field in McpConfig.__dataclass_fields__.values()
            if field.name != "additional_config"
        }
        
        config_kwargs = {}
        additional_config = {}
        
        for key, value in merged_config.items():
            if key in known_fields:
                config_kwargs[key] = value
            else:
                additional_config[key] = value
        
        config_kwargs["additional_config"] = additional_config
        
        mcp_config = McpConfig(**config_kwargs)
        logger.info(f"Loaded MCP configuration from {config_file}")
        return mcp_config
        
    except TypeError as e:
        raise ConfigValidationError(f"Invalid MCP configuration structure: {e}") from e


def load_daemon_config(
    config_file: Optional[Path] = None,
    env_vars: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> DaemonConfig:
    """Load daemon configuration with proper precedence handling.
    
    Args:
        config_file: Optional path to specific config file. If None, uses default location.
        env_vars: Optional environment variable overrides.
        cli_args: Optional command-line argument overrides.
        
    Returns:
        DaemonConfig: Loaded and validated daemon configuration.
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid.
    """
    if env_vars is None:
        env_vars = {}
    if cli_args is None:
        cli_args = {}
        
    # Determine config file path
    if config_file is None:
        config_paths = get_config_paths()
        config_file = config_paths.daemon_config_file
    
    # Load user configuration
    user_config = load_toml_file(config_file, required=False)
    
    # Default daemon configuration
    defaults = {
        "daemon_name": "workspace-qdrant-daemon",
        "log_file": str(Path.home() / "Library" / "Logs" / "workspace-qdrant-daemon.log"),
        "log_level": "info",
        "max_concurrent_tasks": 4,
        "default_timeout_ms": 30000,
        "enable_preemption": True,
        "chunk_size": 1000,
        "auto_ingestion_enabled": True,
        "auto_create_watches": True,
        "include_common_files": True,
        "include_source_files": True,
        "qdrant_url": "http://localhost:6333",
        "qdrant_timeout_ms": 30000,
        "qdrant_max_retries": 3,
    }
    
    # Merge all configuration sources
    merged_config = merge_config_sources(defaults, user_config, env_vars, cli_args)
    
    # Create DaemonConfig instance
    try:
        # Extract known fields
        known_fields = {
            field.name for field in DaemonConfig.__dataclass_fields__.values()
            if field.name != "additional_config"
        }
        
        config_kwargs = {}
        additional_config = {}
        
        for key, value in merged_config.items():
            if key in known_fields:
                config_kwargs[key] = value
            else:
                additional_config[key] = value
        
        config_kwargs["additional_config"] = additional_config
        
        daemon_config = DaemonConfig(**config_kwargs)
        logger.info(f"Loaded daemon configuration from {config_file}")
        return daemon_config
        
    except TypeError as e:
        raise ConfigValidationError(f"Invalid daemon configuration structure: {e}") from e


def validate_config(config: Union[McpConfig, DaemonConfig]) -> List[str]:
    """Validate configuration structure and values.
    
    Args:
        config: Configuration object to validate.
        
    Returns:
        List of validation error messages. Empty list if configuration is valid.
    """
    errors = []
    
    if isinstance(config, McpConfig):
        # Validate MCP-specific configuration
        if not config.server_name or not isinstance(config.server_name, str):
            errors.append("server_name must be a non-empty string")
            
        if not config.qdrant_url or not isinstance(config.qdrant_url, str):
            errors.append("qdrant_url must be a non-empty string")
            
        if config.qdrant_timeout_ms <= 0:
            errors.append("qdrant_timeout_ms must be positive")
            
        if config.max_concurrent_tasks <= 0:
            errors.append("max_concurrent_tasks must be positive")
            
        if config.chunk_size <= 0:
            errors.append("chunk_size must be positive")
            
    elif isinstance(config, DaemonConfig):
        # Validate daemon-specific configuration
        if not config.daemon_name or not isinstance(config.daemon_name, str):
            errors.append("daemon_name must be a non-empty string")
            
        if config.log_level not in ["debug", "info", "warn", "error"]:
            errors.append("log_level must be one of: debug, info, warn, error")
            
        if not config.qdrant_url or not isinstance(config.qdrant_url, str):
            errors.append("qdrant_url must be a non-empty string")
            
        if config.qdrant_timeout_ms <= 0:
            errors.append("qdrant_timeout_ms must be positive")
            
        if config.max_concurrent_tasks <= 0:
            errors.append("max_concurrent_tasks must be positive")
    
    # Common validation for both types
    if hasattr(config, 'additional_config') and not isinstance(config.additional_config, dict):
        errors.append("additional_config must be a dictionary")
    
    return errors


def create_default_config_files() -> ConfigPaths:
    """Create default configuration files if they don't exist.
    
    Returns:
        ConfigPaths: Paths to the created configuration files.
        
    Raises:
        ConfigurationError: If files cannot be created.
    """
    config_paths = get_config_paths()
    
    # Create MCP config if it doesn't exist
    if not config_paths.mcp_config_file.exists():
        default_mcp_config = {
            "server_name": "workspace-qdrant-mcp",
            "server_version": "1.0.0",
            "qdrant_url": "http://localhost:6333",
            "qdrant_timeout_ms": 30000,
            "qdrant_max_retries": 3,
            "qdrant_pool_size": 10,
            "max_concurrent_tasks": 4,
            "default_timeout_ms": 30000,
            "chunk_size": 1000,
            "enable_lsp": True,
            "enable_metrics": True,
            "enable_web_ui": False,
        }
        
        try:
            with open(config_paths.mcp_config_file, 'w', encoding='utf-8') as f:
                toml.dump(default_mcp_config, f)
            logger.info(f"Created default MCP config: {config_paths.mcp_config_file}")
        except (OSError, IOError) as e:
            raise ConfigurationError(f"Failed to create MCP config file: {e}") from e
    
    # Create daemon config if it doesn't exist
    if not config_paths.daemon_config_file.exists():
        default_daemon_config = {
            "daemon_name": "workspace-qdrant-daemon",
            "log_file": str(Path.home() / "Library" / "Logs" / "workspace-qdrant-daemon.log"),
            "log_level": "info",
            "max_concurrent_tasks": 4,
            "default_timeout_ms": 30000,
            "enable_preemption": True,
            "chunk_size": 1000,
            "auto_ingestion": {
                "enabled": True,
                "auto_create_watches": True,
                "include_common_files": True,
                "include_source_files": True,
            },
            "qdrant": {
                "url": "http://localhost:6333",
                "timeout_ms": 30000,
                "max_retries": 3,
                "pool_size": 10,
            }
        }
        
        try:
            with open(config_paths.daemon_config_file, 'w', encoding='utf-8') as f:
                toml.dump(default_daemon_config, f)
            logger.info(f"Created default daemon config: {config_paths.daemon_config_file}")
        except (OSError, IOError) as e:
            raise ConfigurationError(f"Failed to create daemon config file: {e}") from e
    
    return config_paths


# Utility functions for environment variable parsing
def parse_env_vars_for_mcp() -> Dict[str, Any]:
    """Parse environment variables relevant to MCP configuration.
    
    Environment variables should be prefixed with 'WQM_MCP_' followed by the config key in uppercase.
    Example: WQM_MCP_QDRANT_URL, WQM_MCP_MAX_CONCURRENT_TASKS
    
    Returns:
        Dict with parsed environment variables.
    """
    env_config = {}
    prefix = "WQM_MCP_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            
            # Handle boolean values
            if value.lower() in ('true', 'false'):
                env_config[config_key] = value.lower() == 'true'
            # Handle integer values
            elif value.isdigit():
                env_config[config_key] = int(value)
            # Handle float values
            elif value.replace('.', '', 1).isdigit():
                env_config[config_key] = float(value)
            # Handle string values
            else:
                env_config[config_key] = value
    
    return env_config


def parse_env_vars_for_daemon() -> Dict[str, Any]:
    """Parse environment variables relevant to daemon configuration.
    
    Environment variables should be prefixed with 'WQM_DAEMON_' followed by the config key in uppercase.
    Example: WQM_DAEMON_LOG_LEVEL, WQM_DAEMON_QDRANT_URL
    
    Returns:
        Dict with parsed environment variables.
    """
    env_config = {}
    prefix = "WQM_DAEMON_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            
            # Handle boolean values
            if value.lower() in ('true', 'false'):
                env_config[config_key] = value.lower() == 'true'
            # Handle integer values
            elif value.isdigit():
                env_config[config_key] = int(value)
            # Handle float values
            elif value.replace('.', '', 1).isdigit():
                env_config[config_key] = float(value)
            # Handle string values
            else:
                env_config[config_key] = value
    
    return env_config