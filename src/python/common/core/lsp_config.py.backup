"""
LSP Configuration Management System

This module provides comprehensive configuration management for LSP detection,
including settings persistence, cache management, and validation.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class LSPConfigSchema(BaseModel):
    """Pydantic schema for LSP configuration validation."""
    
    # Global LSP detection settings
    enabled: bool = Field(default=True, description="Enable LSP detection")
    cache_ttl: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")
    detection_timeout: float = Field(default=5.0, ge=1.0, le=30.0, description="Detection timeout in seconds")
    
    # LSP-specific settings
    enabled_lsps: Set[str] = Field(
        default_factory=set,
        description="Set of LSP names to enable (empty means all)"
    )
    disabled_lsps: Set[str] = Field(
        default_factory=set,
        description="Set of LSP names to disable"
    )
    custom_lsp_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom paths for specific LSP binaries"
    )
    
    # Extension handling
    include_fallbacks: bool = Field(default=True, description="Include language fallbacks")
    include_build_tools: bool = Field(default=True, description="Include build tool extensions")
    include_infrastructure: bool = Field(default=True, description="Include infrastructure extensions")
    
    # Environment overrides
    respect_env_vars: bool = Field(default=True, description="Respect environment variable overrides")
    env_prefix: str = Field(default="WQM_LSP", description="Environment variable prefix")
    
    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Configuration creation time"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last update time"
    )
    version: str = Field(default="1.0", description="Configuration version")

    @field_validator("enabled_lsps", "disabled_lsps")
    @classmethod
    def validate_lsp_names(cls, v: Set[str]) -> Set[str]:
        """Validate LSP names are non-empty strings."""
        for lsp_name in v:
            if not isinstance(lsp_name, str) or not lsp_name.strip():
                raise ValueError("LSP names must be non-empty strings")
        return v

    @field_validator("custom_lsp_paths")
    @classmethod
    def validate_custom_paths(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate custom LSP paths exist."""
        for lsp_name, path in v.items():
            if not isinstance(lsp_name, str) or not lsp_name.strip():
                raise ValueError("LSP names must be non-empty strings")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("LSP paths must be non-empty strings")
            # Note: We don't validate file existence here as it may not be available during validation
        return v


@dataclass
class LSPCacheEntry:
    """Individual cache entry for LSP detection results."""
    
    lsp_name: str
    binary_path: str
    version: Optional[str] = None
    supported_extensions: List[str] = field(default_factory=list)
    priority: int = 0
    capabilities: Set[str] = field(default_factory=set)
    detection_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def is_expired(self, ttl: int) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - self.detection_time) > ttl
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class LSPConfig:
    """LSP configuration management with persistence and validation."""
    
    # Configuration settings
    enabled: bool = True
    cache_ttl: int = 300
    detection_timeout: float = 5.0
    enabled_lsps: Set[str] = field(default_factory=set)
    disabled_lsps: Set[str] = field(default_factory=set)
    custom_lsp_paths: Dict[str, str] = field(default_factory=dict)
    include_fallbacks: bool = True
    include_build_tools: bool = True
    include_infrastructure: bool = True
    respect_env_vars: bool = True
    env_prefix: str = "WQM_LSP"
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0"
    
    # Runtime state
    _config_file: Optional[Path] = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        if self.respect_env_vars:
            self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        prefix = self.env_prefix
        
        # Override boolean settings
        if os.getenv(f"{prefix}_ENABLED"):
            self.enabled = os.getenv(f"{prefix}_ENABLED", "").lower() in ("true", "1", "yes")
        
        if os.getenv(f"{prefix}_INCLUDE_FALLBACKS"):
            self.include_fallbacks = os.getenv(f"{prefix}_INCLUDE_FALLBACKS", "").lower() in ("true", "1", "yes")
        
        if os.getenv(f"{prefix}_INCLUDE_BUILD_TOOLS"):
            self.include_build_tools = os.getenv(f"{prefix}_INCLUDE_BUILD_TOOLS", "").lower() in ("true", "1", "yes")
        
        if os.getenv(f"{prefix}_INCLUDE_INFRASTRUCTURE"):
            self.include_infrastructure = os.getenv(f"{prefix}_INCLUDE_INFRASTRUCTURE", "").lower() in ("true", "1", "yes")
        
        # Override numeric settings
        if os.getenv(f"{prefix}_CACHE_TTL"):
            try:
                self.cache_ttl = int(os.getenv(f"{prefix}_CACHE_TTL"))
            except ValueError:
                logger.warning(f"Invalid {prefix}_CACHE_TTL value, using default")
        
        if os.getenv(f"{prefix}_DETECTION_TIMEOUT"):
            try:
                self.detection_timeout = float(os.getenv(f"{prefix}_DETECTION_TIMEOUT"))
            except ValueError:
                logger.warning(f"Invalid {prefix}_DETECTION_TIMEOUT value, using default")
        
        # Override LSP lists
        if os.getenv(f"{prefix}_ENABLED_LSPS"):
            enabled_str = os.getenv(f"{prefix}_ENABLED_LSPS", "")
            self.enabled_lsps = set(lsp.strip() for lsp in enabled_str.split(",") if lsp.strip())
        
        if os.getenv(f"{prefix}_DISABLED_LSPS"):
            disabled_str = os.getenv(f"{prefix}_DISABLED_LSPS", "")
            self.disabled_lsps = set(lsp.strip() for lsp in disabled_str.split(",") if lsp.strip())

    def is_lsp_enabled(self, lsp_name: str) -> bool:
        """Check if a specific LSP is enabled."""
        # Check if explicitly disabled
        if lsp_name in self.disabled_lsps:
            return False
        
        # If enabled_lsps is empty, all are enabled (except disabled ones)
        if not self.enabled_lsps:
            return True
        
        # Check if explicitly enabled
        return lsp_name in self.enabled_lsps

    def get_lsp_binary_path(self, lsp_name: str) -> Optional[str]:
        """Get custom binary path for LSP, if configured."""
        return self.custom_lsp_paths.get(lsp_name)

    def set_custom_lsp_path(self, lsp_name: str, path: str) -> None:
        """Set custom binary path for LSP."""
        with self._lock:
            self.custom_lsp_paths[lsp_name] = path
            self._mark_updated()

    def enable_lsp(self, lsp_name: str) -> None:
        """Enable a specific LSP."""
        with self._lock:
            self.enabled_lsps.add(lsp_name)
            self.disabled_lsps.discard(lsp_name)
            self._mark_updated()

    def disable_lsp(self, lsp_name: str) -> None:
        """Disable a specific LSP."""
        with self._lock:
            self.disabled_lsps.add(lsp_name)
            self.enabled_lsps.discard(lsp_name)
            self._mark_updated()

    def _mark_updated(self) -> None:
        """Mark configuration as updated."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['enabled_lsps'] = list(self.enabled_lsps)
        data['disabled_lsps'] = list(self.disabled_lsps)
        data['capabilities'] = list(getattr(self, 'capabilities', set()))
        # Remove private fields
        return {k: v for k, v in data.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSPConfig":
        """Create from dictionary with validation."""
        # Convert lists back to sets
        if 'enabled_lsps' in data:
            data['enabled_lsps'] = set(data['enabled_lsps'])
        if 'disabled_lsps' in data:
            data['disabled_lsps'] = set(data['disabled_lsps'])
        
        # Validate using Pydantic schema
        validated_data = LSPConfigSchema(**data)
        return cls(**validated_data.dict())

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate TTL range
        if self.cache_ttl < 60 or self.cache_ttl > 3600:
            issues.append("Cache TTL must be between 60 and 3600 seconds")
        
        # Validate timeout range
        if self.detection_timeout < 1.0 or self.detection_timeout > 30.0:
            issues.append("Detection timeout must be between 1.0 and 30.0 seconds")
        
        # Check for conflicting LSP settings
        conflicts = self.enabled_lsps.intersection(self.disabled_lsps)
        if conflicts:
            issues.append(f"LSPs cannot be both enabled and disabled: {conflicts}")
        
        # Validate custom paths exist
        for lsp_name, path in self.custom_lsp_paths.items():
            path_obj = Path(path)
            if not path_obj.exists():
                issues.append(f"Custom LSP path does not exist: {lsp_name} -> {path}")
            elif not path_obj.is_file():
                issues.append(f"Custom LSP path is not a file: {lsp_name} -> {path}")
            elif not os.access(path_obj, os.X_OK):
                issues.append(f"Custom LSP path is not executable: {lsp_name} -> {path}")
        
        return issues

    def save_to_file(self, config_file: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_file)
        
        with self._lock:
            try:
                # Create parent directories if needed
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write configuration
                with open(config_path, 'w') as f:
                    json.dump(self.to_dict(), f, indent=2, sort_keys=True)
                
                self._config_file = config_path
                logger.debug(f"LSP configuration saved to {config_path}")
                
            except Exception as e:
                logger.error(f"Failed to save LSP configuration: {e}")
                raise

    @classmethod
    def load_from_file(cls, config_file: Union[str, Path]) -> "LSPConfig":
        """Load configuration from file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.debug(f"LSP config file not found: {config_path}, creating default")
            config = cls()
            config._config_file = config_path
            return config
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            config = cls.from_dict(data)
            config._config_file = config_path
            logger.debug(f"LSP configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load LSP configuration: {e}")
            # Return default config on error
            config = cls()
            config._config_file = config_path
            return config


class LSPCache:
    """Thread-safe cache for LSP detection results with TTL and persistence."""
    
    def __init__(self, cache_file: Optional[Union[str, Path]] = None, default_ttl: int = 300):
        """
        Initialize LSP cache.
        
        Args:
            cache_file: Optional file path for cache persistence
            default_ttl: Default TTL for cache entries in seconds
        """
        self.default_ttl = default_ttl
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: Dict[str, LSPCacheEntry] = {}
        self._lock = threading.RLock()
        
        # Load existing cache if file exists
        if self.cache_file and self.cache_file.exists():
            self._load_cache()

    def get(self, lsp_name: str, ttl: Optional[int] = None) -> Optional[LSPCacheEntry]:
        """Get cache entry for LSP, if valid."""
        ttl = ttl or self.default_ttl
        
        with self._lock:
            entry = self._cache.get(lsp_name)
            if entry and not entry.is_expired(ttl):
                entry.touch()
                return entry
            elif entry:
                # Remove expired entry
                del self._cache[lsp_name]
            return None

    def set(self, lsp_name: str, entry: LSPCacheEntry) -> None:
        """Set cache entry for LSP."""
        with self._lock:
            self._cache[lsp_name] = entry
            self._save_cache_async()

    def remove(self, lsp_name: str) -> bool:
        """Remove cache entry for LSP."""
        with self._lock:
            if lsp_name in self._cache:
                del self._cache[lsp_name]
                self._save_cache_async()
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._save_cache_async()

    def cleanup_expired(self, ttl: Optional[int] = None) -> int:
        """Remove expired cache entries and return count removed."""
        ttl = ttl or self.default_ttl
        removed_count = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(ttl)
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            
            if removed_count > 0:
                self._save_cache_async()
        
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values()
                if entry.is_expired(self.default_ttl)
            )
            
            if total_entries > 0:
                avg_access_count = sum(entry.access_count for entry in self._cache.values()) / total_entries
                oldest_entry = min(self._cache.values(), key=lambda e: e.detection_time)
                newest_entry = max(self._cache.values(), key=lambda e: e.detection_time)
            else:
                avg_access_count = 0
                oldest_entry = newest_entry = None
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'valid_entries': total_entries - expired_entries,
                'avg_access_count': avg_access_count,
                'oldest_entry_age': time.time() - oldest_entry.detection_time if oldest_entry else 0,
                'newest_entry_age': time.time() - newest_entry.detection_time if newest_entry else 0,
            }

    def _load_cache(self) -> None:
        """Load cache from file."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            for lsp_name, entry_data in data.items():
                # Convert capabilities list back to set
                if 'capabilities' in entry_data:
                    entry_data['capabilities'] = set(entry_data['capabilities'])
                
                entry = LSPCacheEntry(**entry_data)
                self._cache[lsp_name] = entry
            
            logger.debug(f"Loaded {len(self._cache)} LSP cache entries from {self.cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load LSP cache from {self.cache_file}: {e}")
            self._cache.clear()

    def _save_cache_async(self) -> None:
        """Save cache to file asynchronously."""
        if not self.cache_file:
            return
        
        # Note: In a real implementation, you might want to use a background thread
        # or async task for this to avoid blocking. For now, we'll do it synchronously.
        self._save_cache()

    def _save_cache(self) -> None:
        """Save cache to file."""
        if not self.cache_file:
            return
        
        try:
            # Create parent directories if needed
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert cache to serializable format
            cache_data = {}
            for lsp_name, entry in self._cache.items():
                entry_data = asdict(entry)
                # Convert set to list for JSON serialization
                entry_data['capabilities'] = list(entry_data['capabilities'])
                cache_data[lsp_name] = entry_data
            
            # Write to file
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, sort_keys=True)
            
            logger.debug(f"Saved {len(self._cache)} LSP cache entries to {self.cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save LSP cache to {self.cache_file}: {e}")


# Global configuration and cache instances
_default_config: Optional[LSPConfig] = None
_default_cache: Optional[LSPCache] = None
_config_lock = threading.RLock()


def get_default_config(config_file: Optional[Union[str, Path]] = None) -> LSPConfig:
    """Get the default global LSP configuration instance."""
    global _default_config
    
    with _config_lock:
        if _default_config is None:
            if config_file:
                _default_config = LSPConfig.load_from_file(config_file)
            else:
                # Use default project-local config path
                config_dir = Path.cwd() / ".wqm" / "config"
                config_path = config_dir / "lsp_config.json"
                _default_config = LSPConfig.load_from_file(config_path)
        
        return _default_config


def get_default_cache(cache_file: Optional[Union[str, Path]] = None) -> LSPCache:
    """Get the default global LSP cache instance."""
    global _default_cache
    
    with _config_lock:
        if _default_cache is None:
            if cache_file:
                _default_cache = LSPCache(cache_file)
            else:
                # Use default project-local cache path
                cache_dir = Path.cwd() / ".wqm" / "cache"
                cache_path = cache_dir / "lsp_cache.json"
                _default_cache = LSPCache(cache_path)
        
        return _default_cache


def reset_globals() -> None:
    """Reset global instances (useful for testing)."""
    global _default_config, _default_cache
    with _config_lock:
        _default_config = None
        _default_cache = None