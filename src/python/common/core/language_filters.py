"""
Language-Aware File Filtering System for Workspace Qdrant MCP.

This module provides sophisticated file filtering capabilities with language-specific
ignore patterns, MIME type detection, and performance optimization. It supports
configurable filtering rules loaded from YAML configuration files.

Key Features:
    - Language-specific filtering for 25+ programming languages
    - Glob and regex pattern matching with compilation caching
    - MIME type detection for accurate file classification
    - Directory-based exclusions with precedence rules
    - User-customizable patterns with override capabilities
    - Performance optimization through pattern caching
    - Statistics tracking for filtered vs processed files

Example:
    ```python
    from workspace_qdrant_mcp.core.language_filters import LanguageAwareFilter
    
    # Initialize filter with configuration
    filter_system = LanguageAwareFilter(config_path="~/.config/workspace-qdrant-mcp/")
    await filter_system.load_configuration()
    
    # Check if file should be processed
    should_process = filter_system.should_process_file("/path/to/file.py")
    
    # Get filtering statistics
    stats = filter_system.get_statistics()
    ```
"""

import asyncio
import fnmatch
from common.logging import get_logger
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union

import yaml

logger = get_logger(__name__)

# Try to import python-magic for MIME type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    logger.warning("python-magic not available, MIME type detection disabled")
    MAGIC_AVAILABLE = False


@dataclass
class FilterStatistics:
    """Statistics for file filtering operations."""
    
    total_files_checked: int = 0
    files_processed: int = 0
    files_filtered_out: int = 0
    files_by_extension: Dict[str, int] = field(default_factory=dict)
    files_by_mime_type: Dict[str, int] = field(default_factory=dict)
    filter_reasons: Dict[str, int] = field(default_factory=dict)  # reason -> count
    pattern_cache_hits: int = 0
    pattern_cache_misses: int = 0
    mime_detection_time_ms: float = 0.0
    total_filter_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format."""
        return {
            "total_files_checked": self.total_files_checked,
            "files_processed": self.files_processed,
            "files_filtered_out": self.files_filtered_out,
            "files_by_extension": dict(self.files_by_extension),
            "files_by_mime_type": dict(self.files_by_mime_type),
            "filter_reasons": dict(self.filter_reasons),
            "pattern_cache_performance": {
                "hits": self.pattern_cache_hits,
                "misses": self.pattern_cache_misses,
                "hit_ratio": self.pattern_cache_hits / max(1, self.pattern_cache_hits + self.pattern_cache_misses)
            },
            "performance_metrics": {
                "mime_detection_time_ms": self.mime_detection_time_ms,
                "total_filter_time_ms": self.total_filter_time_ms,
                "average_filter_time_per_file_ms": self.total_filter_time_ms / max(1, self.total_files_checked)
            }
        }


@dataclass
class FilterConfiguration:
    """Configuration for language-aware filtering."""
    
    # Ignore patterns configuration
    dot_files_ignored: bool = True
    ignored_directories: List[str] = field(default_factory=list)
    ignored_file_extensions: List[str] = field(default_factory=list)
    ignored_regex_patterns: List[str] = field(default_factory=list)
    
    # Performance constraints
    max_file_size_mb: float = 10.0
    mime_detection_enabled: bool = True
    cache_patterns: bool = True
    
    # User overrides
    force_include_directories: List[str] = field(default_factory=list)
    force_include_extensions: List[str] = field(default_factory=list)
    additional_ignore_directories: List[str] = field(default_factory=list)
    additional_ignore_extensions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml_config(cls, config_data: Dict[str, Any]) -> "FilterConfiguration":
        """Create configuration from YAML data."""
        ignore_patterns = config_data.get("ignore_patterns", {})
        performance = config_data.get("performance", {})
        user_overrides = config_data.get("user_overrides", {})
        
        return cls(
            dot_files_ignored=ignore_patterns.get("dot_files", True),
            ignored_directories=ignore_patterns.get("directories", []),
            ignored_file_extensions=ignore_patterns.get("file_extensions", []),
            max_file_size_mb=performance.get("max_file_size_mb", 10.0),
            mime_detection_enabled=MAGIC_AVAILABLE,
            force_include_directories=user_overrides.get("force_include", {}).get("directories", []),
            force_include_extensions=user_overrides.get("force_include", {}).get("file_extensions", []),
            additional_ignore_directories=user_overrides.get("additional_ignores", {}).get("directories", []),
            additional_ignore_extensions=user_overrides.get("additional_ignores", {}).get("file_extensions", []),
        )


class CompiledPatterns:
    """Compiled pattern cache for performance optimization."""
    
    def __init__(self):
        self._glob_patterns: Dict[str, bool] = {}
        self._regex_patterns: Dict[str, Pattern[str]] = {}
        self._pattern_hits: Dict[str, int] = {}
        self._cache_size_limit = 1000
        
    def match_glob(self, pattern: str, path: str) -> bool:
        """Match glob pattern with caching."""
        cache_key = f"glob:{pattern}:{path}"
        
        if cache_key in self._glob_patterns:
            self._pattern_hits[cache_key] = self._pattern_hits.get(cache_key, 0) + 1
            return self._glob_patterns[cache_key]
        
        # Limit cache size
        if len(self._glob_patterns) >= self._cache_size_limit:
            # Remove least used entries
            sorted_items = sorted(self._pattern_hits.items(), key=lambda x: x[1])
            for key, _ in sorted_items[:100]:  # Remove 100 least used
                cache_pattern = key.split(":", 2)
                if len(cache_pattern) == 3:
                    self._glob_patterns.pop(f"{cache_pattern[0]}:{cache_pattern[1]}:{cache_pattern[2]}", None)
                self._pattern_hits.pop(key, None)
        
        result = fnmatch.fnmatch(path, pattern)
        self._glob_patterns[cache_key] = result
        return result
    
    def match_regex(self, pattern: str, text: str) -> bool:
        """Match regex pattern with compilation caching."""
        if pattern not in self._regex_patterns:
            if len(self._regex_patterns) >= self._cache_size_limit:
                # Clear oldest patterns
                keys_to_remove = list(self._regex_patterns.keys())[:100]
                for key in keys_to_remove:
                    self._regex_patterns.pop(key, None)
            
            try:
                self._regex_patterns[pattern] = re.compile(pattern)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                return False
        
        compiled_pattern = self._regex_patterns[pattern]
        return bool(compiled_pattern.search(text))
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            "glob_cache_size": len(self._glob_patterns),
            "regex_cache_size": len(self._regex_patterns),
            "total_hits": sum(self._pattern_hits.values()),
            "unique_patterns": len(self._pattern_hits)
        }


class MimeTypeDetector:
    """MIME type detection with caching."""
    
    def __init__(self):
        self._magic = None
        self._mime_cache: Dict[str, Optional[str]] = {}
        self._cache_size_limit = 5000
        
        if MAGIC_AVAILABLE:
            try:
                self._magic = magic.Magic(mime=True)
                logger.info("MIME type detection initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MIME detection: {e}")
                self._magic = None
    
    def get_mime_type(self, file_path: Path) -> Optional[str]:
        """Get MIME type for file with caching."""
        if not self._magic:
            return None
            
        path_str = str(file_path)
        
        if path_str in self._mime_cache:
            return self._mime_cache[path_str]
        
        # Limit cache size
        if len(self._mime_cache) >= self._cache_size_limit:
            # Remove 1/4 of oldest entries
            items_to_remove = list(self._mime_cache.keys())[:self._cache_size_limit // 4]
            for key in items_to_remove:
                self._mime_cache.pop(key, None)
        
        try:
            if file_path.exists() and file_path.is_file():
                mime_type = self._magic.from_file(path_str)
                self._mime_cache[path_str] = mime_type
                return mime_type
        except Exception as e:
            logger.debug(f"MIME detection failed for {file_path}: {e}")
        
        self._mime_cache[path_str] = None
        return None


class LanguageAwareFilter:
    """
    Sophisticated file filtering system with language-specific patterns.
    
    Provides comprehensive filtering capabilities including:
    - Language-specific ignore patterns for 25+ programming languages
    - MIME type detection for accurate file classification  
    - Configurable glob and regex pattern matching
    - Performance optimization through pattern caching
    - Statistics tracking for monitoring and optimization
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the language-aware filter.
        
        Args:
            config_path: Path to directory containing ingestion.yaml config file
        """
        self.config_path = Path(config_path) if config_path else Path.home() / ".config" / "workspace-qdrant-mcp"
        self.config: Optional[FilterConfiguration] = None
        self.compiled_patterns = CompiledPatterns()
        self.mime_detector = MimeTypeDetector()
        self.statistics = FilterStatistics()
        self._initialized = False
        
    async def load_configuration(self, config_file: Optional[str] = None) -> None:
        """
        Load filtering configuration from YAML file.
        
        Args:
            config_file: Specific config file name (default: ingestion.yaml)
        """
        config_file_name = config_file or "ingestion.yaml"
        config_file_path = self.config_path / config_file_name
        
        if not config_file_path.exists():
            # Look for template file and copy it
            template_path = Path(__file__).parent.parent.parent.parent / "config" / "ingestion.yaml.template"
            if template_path.exists():
                logger.info(f"Copying template config from {template_path} to {config_file_path}")
                self.config_path.mkdir(parents=True, exist_ok=True)
                with template_path.open() as src, config_file_path.open('w') as dst:
                    dst.write(src.read())
            else:
                logger.warning(f"No configuration file found at {config_file_path}, using defaults")
                self.config = FilterConfiguration()
                self._initialized = True
                return
        
        try:
            with config_file_path.open() as f:
                config_data = yaml.safe_load(f)
            
            self.config = FilterConfiguration.from_yaml_config(config_data)
            self._initialized = True
            logger.info(f"Loaded language filter configuration from {config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file_path}: {e}")
            self.config = FilterConfiguration()
            self._initialized = True
    
    def should_process_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Determine if a file should be processed based on filtering rules.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (should_process, reason) where reason explains the decision
        """
        start_time = time.perf_counter()
        
        if not self._initialized:
            asyncio.create_task(self.load_configuration())
            return True, "filter_not_initialized"
        
        file_path = Path(file_path)
        self.statistics.total_files_checked += 1
        
        try:
            # Track file extension
            extension = file_path.suffix.lower()
            self.statistics.files_by_extension[extension] = self.statistics.files_by_extension.get(extension, 0) + 1
            
            # Check if file exists and get basic info
            if not file_path.exists():
                self._record_filter_decision(False, "file_not_found", start_time)
                return False, "file_not_found"
            
            if not file_path.is_file():
                self._record_filter_decision(False, "not_a_file", start_time)
                return False, "not_a_file"
            
            # Check file size constraints
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    self._record_filter_decision(False, "file_too_large", start_time)
                    return False, f"file_too_large ({file_size_mb:.2f}MB > {self.config.max_file_size_mb}MB)"
            except OSError:
                self._record_filter_decision(False, "file_stat_error", start_time)
                return False, "file_stat_error"
            
            file_name = file_path.name
            file_parts = file_path.parts
            relative_path = str(file_path)
            
            # Check force include patterns first (these override ignores)
            if self._matches_force_include(file_path, file_name, extension):
                self._record_filter_decision(True, "force_included", start_time)
                return True, "force_included"
            
            # Check dot files if configured to ignore
            if self.config.dot_files_ignored and file_name.startswith('.'):
                self._record_filter_decision(False, "dot_file_ignored", start_time)
                return False, "dot_file_ignored"
            
            # Check directory-based ignores
            if self._matches_ignored_directories(file_parts):
                self._record_filter_decision(False, "directory_ignored", start_time)
                return False, "directory_ignored"
            
            # Check file extension ignores
            if self._matches_ignored_extensions(extension, file_name):
                self._record_filter_decision(False, "extension_ignored", start_time)
                return False, "extension_ignored"
            
            # Check regex patterns
            regex_reason = self._matches_ignored_regex(relative_path)
            if regex_reason:
                self._record_filter_decision(False, "regex_ignored", start_time)
                return False, f"regex_ignored: {regex_reason}"
            
            # MIME type detection and filtering (if enabled)
            if self.config.mime_detection_enabled:
                mime_start = time.perf_counter()
                mime_type = self.mime_detector.get_mime_type(file_path)
                mime_time = (time.perf_counter() - mime_start) * 1000
                self.statistics.mime_detection_time_ms += mime_time
                
                if mime_type:
                    self.statistics.files_by_mime_type[mime_type] = self.statistics.files_by_mime_type.get(mime_type, 0) + 1
                    
                    # Filter out binary files that aren't useful for text search
                    if self._is_filtered_mime_type(mime_type):
                        self._record_filter_decision(False, "mime_type_ignored", start_time)
                        return False, f"mime_type_ignored: {mime_type}"
            
            # File passes all filters
            self._record_filter_decision(True, "accepted", start_time)
            return True, "accepted"
            
        except Exception as e:
            logger.error(f"Error filtering file {file_path}: {e}")
            self._record_filter_decision(False, "filter_error", start_time)
            return False, f"filter_error: {e}"
    
    def _matches_force_include(self, file_path: Path, file_name: str, extension: str) -> bool:
        """Check if file matches force include patterns."""
        # Check force include directories
        for pattern in self.config.force_include_directories:
            if any(self.compiled_patterns.match_glob(pattern, str(part)) for part in file_path.parts):
                return True
        
        # Check force include extensions
        for pattern in self.config.force_include_extensions:
            if self.compiled_patterns.match_glob(pattern, file_name) or pattern == extension:
                return True
        
        return False
    
    def _matches_ignored_directories(self, file_parts: Tuple[str, ...]) -> bool:
        """Check if file path contains ignored directories."""
        all_ignored_dirs = set(self.config.ignored_directories + self.config.additional_ignore_directories)
        
        for part in file_parts[:-1]:  # Exclude filename
            for ignored_dir in all_ignored_dirs:
                if self.compiled_patterns.match_glob(ignored_dir, part):
                    return True
        
        return False
    
    def _matches_ignored_extensions(self, extension: str, file_name: str) -> bool:
        """Check if file matches ignored extension patterns."""
        all_ignored_exts = set(self.config.ignored_file_extensions + self.config.additional_ignore_extensions)
        
        for pattern in all_ignored_exts:
            if pattern.startswith('*.'):
                # Extension pattern like "*.pyc"
                if extension == pattern[1:]:  # Remove the *
                    return True
            elif self.compiled_patterns.match_glob(pattern, file_name):
                return True
        
        return False
    
    def _matches_ignored_regex(self, file_path: str) -> Optional[str]:
        """Check if file path matches ignored regex patterns."""
        for pattern in self.config.ignored_regex_patterns:
            if self.compiled_patterns.match_regex(pattern, file_path):
                return pattern
        return None
    
    def _is_filtered_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type should be filtered out."""
        # Filter out binary file types that aren't useful for text search
        binary_mime_types = {
            'application/octet-stream',
            'application/x-executable',
            'application/x-sharedlib',
            'image/',
            'video/',
            'audio/',
            'font/',
        }
        
        for binary_type in binary_mime_types:
            if mime_type.startswith(binary_type):
                return True
        
        return False
    
    def _record_filter_decision(self, should_process: bool, reason: str, start_time: float) -> None:
        """Record filtering decision and update statistics."""
        end_time = time.perf_counter()
        filter_time_ms = (end_time - start_time) * 1000
        self.statistics.total_filter_time_ms += filter_time_ms
        
        if should_process:
            self.statistics.files_processed += 1
        else:
            self.statistics.files_filtered_out += 1
        
        self.statistics.filter_reasons[reason] = self.statistics.filter_reasons.get(reason, 0) + 1
    
    def get_statistics(self) -> FilterStatistics:
        """Get current filtering statistics."""
        # Update cache statistics
        cache_stats = self.compiled_patterns.get_cache_stats()
        self.statistics.pattern_cache_hits = cache_stats.get("total_hits", 0)
        self.statistics.pattern_cache_misses = cache_stats.get("unique_patterns", 0) - cache_stats.get("total_hits", 0)
        
        return self.statistics
    
    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        self.statistics = FilterStatistics()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current filtering configuration."""
        if not self.config:
            return {"error": "Configuration not loaded"}
        
        return {
            "dot_files_ignored": self.config.dot_files_ignored,
            "ignored_directories_count": len(self.config.ignored_directories),
            "ignored_extensions_count": len(self.config.ignored_file_extensions),
            "max_file_size_mb": self.config.max_file_size_mb,
            "mime_detection_enabled": self.config.mime_detection_enabled,
            "force_include_rules": {
                "directories": len(self.config.force_include_directories),
                "extensions": len(self.config.force_include_extensions)
            },
            "pattern_cache_enabled": self.config.cache_patterns,
            "magic_library_available": MAGIC_AVAILABLE
        }