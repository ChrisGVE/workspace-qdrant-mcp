"""
Advanced watch configuration options and validation.

This module provides enhanced configuration capabilities for file watching,
including advanced pattern matching, collection targeting, and performance tuning options.
"""

import fnmatch
from common.logging import get_logger
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from pydantic import BaseModel, Field, field_validator

# Import LSP detector for dynamic extension detection
try:
    from .lsp_detector import get_default_detector
except ImportError:
    # Fallback if LSP detector is not available
    get_default_detector = None

logger = get_logger(__name__)


class FileFilterConfig(BaseModel):
    """Advanced file filtering configuration."""

    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"],
        description="File patterns to include (glob patterns)",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            ".DS_Store",
            "*.tmp",
        ],
        description="File patterns to exclude (glob patterns)",
    )
    mime_types: list[str] = Field(
        default_factory=list,
        description="MIME types to include (e.g., 'text/plain', 'application/pdf')",
    )
    size_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "min_bytes": 1,
            "max_bytes": 100 * 1024 * 1024,
        },  # 100MB max
        description="File size constraints in bytes",
    )
    regex_patterns: dict[str, str] = Field(
        default_factory=dict,
        description="Advanced regex patterns: {'include': 'pattern', 'exclude': 'pattern'}",
    )

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def validate_patterns(cls, v: list[str]) -> list[str]:
        """Validate glob patterns."""
        if not v:
            raise ValueError("Pattern lists cannot be empty")

        for pattern in v:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("All patterns must be non-empty strings")
            # Test if it's a valid glob pattern
            try:
                fnmatch.fnmatch("test.txt", pattern)
            except Exception:
                raise ValueError(f"Invalid glob pattern: {pattern}")
        return v

    @field_validator("regex_patterns")
    @classmethod
    def validate_regex_patterns(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate regex patterns."""
        for key, pattern in v.items():
            if key not in ["include", "exclude"]:
                raise ValueError("Regex patterns must have keys 'include' or 'exclude'")
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return v


class RecursiveConfig(BaseModel):
    """Recursive directory scanning configuration."""

    enabled: bool = Field(default=True, description="Enable recursive scanning")
    max_depth: int = Field(
        default=-1,
        ge=-1,
        le=20,
        description="Maximum recursion depth (-1 for unlimited, max 20)",
    )
    follow_symlinks: bool = Field(default=False, description="Follow symbolic links")
    skip_hidden: bool = Field(default=True, description="Skip hidden directories")
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [".git", ".svn", ".hg", "node_modules", "__pycache__"],
        description="Directory names to exclude from recursion",
    )

    @field_validator("exclude_dirs")
    @classmethod
    def validate_exclude_dirs(cls, v: list[str]) -> list[str]:
        """Validate excluded directory patterns."""
        for dirname in v:
            if not isinstance(dirname, str) or not dirname.strip():
                raise ValueError("Excluded directory names must be non-empty strings")
        return v


class PerformanceConfig(BaseModel):
    """Performance and resource management configuration."""

    update_frequency_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="File system check frequency in milliseconds (100ms-60s)",
    )
    debounce_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Debounce delay before processing changes (1-300 seconds)",
    )
    batch_processing: bool = Field(
        default=True, description="Process multiple file changes in batches"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of files to process in one batch",
    )
    memory_limit_mb: int = Field(
        default=256,
        ge=64,
        le=2048,
        description="Memory usage limit in MB for file content caching",
    )
    max_concurrent_ingestions: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent file ingestions"
    )


class CollectionTargeting(BaseModel):
    """Collection targeting and routing configuration."""

    default_collection: str = Field(
        ..., min_length=1, description="Default target collection"
    )
    routing_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rules for routing files to different collections",
    )
    collection_prefixes: dict[str, str] = Field(
        default_factory=dict,
        description="Collection prefixes based on file characteristics",
    )

    @field_validator("routing_rules")
    @classmethod
    def validate_routing_rules(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate collection routing rules."""
        required_keys = {"pattern", "collection"}
        for rule in v:
            if not isinstance(rule, dict):
                raise ValueError("Routing rules must be dictionaries")

            rule_keys = set(rule.keys())
            if not required_keys.issubset(rule_keys):
                raise ValueError(f"Routing rules must contain keys: {required_keys}")

            # Validate pattern
            pattern = rule.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("Routing rule patterns must be non-empty strings")

            # Validate collection
            collection = rule.get("collection")
            if not isinstance(collection, str) or not collection.strip():
                raise ValueError("Routing rule collections must be non-empty strings")

        return v


@dataclass
class AdvancedWatchConfig:
    """Complete advanced watch configuration."""

    # Basic configuration
    id: str
    path: str
    enabled: bool = True

    # File filtering
    file_filters: FileFilterConfig = field(default_factory=FileFilterConfig)

    # Recursion settings
    recursive: RecursiveConfig = field(default_factory=RecursiveConfig)

    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Collection targeting
    collection_config: CollectionTargeting = field(
        default_factory=lambda: CollectionTargeting(default_collection="default")
    )

    # LSP-based extension detection
    lsp_based_extensions: bool = True
    lsp_detection_cache_ttl: int = 300
    lsp_fallback_enabled: bool = True
    lsp_include_build_tools: bool = True
    lsp_include_infrastructure: bool = True

    # Processing options
    auto_ingest: bool = True
    preserve_timestamps: bool = True
    create_backup_on_error: bool = False

    # Metadata and tracking
    created_at: str = field(default="")
    last_modified: str = field(default="")
    version: str = field(default="1.0")
    tags: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate the complete configuration."""
        issues = []

        # Validate path
        try:
            path = Path(self.path)
            if not path.exists():
                issues.append(f"Watch path does not exist: {self.path}")
            elif not path.is_dir():
                issues.append(f"Watch path is not a directory: {self.path}")
        except Exception as e:
            issues.append(f"Invalid path format: {e}")

        # Validate component configurations
        try:
            self.file_filters.dict()
        except Exception as e:
            issues.append(f"File filters validation failed: {e}")

        try:
            self.recursive.dict()
        except Exception as e:
            issues.append(f"Recursive configuration validation failed: {e}")

        try:
            self.performance.dict()
        except Exception as e:
            issues.append(f"Performance configuration validation failed: {e}")

        try:
            self.collection_config.dict()
        except Exception as e:
            issues.append(f"Collection configuration validation failed: {e}")

        return issues

    def get_effective_patterns(self) -> tuple[list[str], list[str]]:
        """
        Get effective include and exclude patterns, optionally enhanced with LSP detection.
        
        Returns:
            Tuple of (include_patterns, exclude_patterns)
        """
        # Start with configured patterns
        include_patterns = list(self.file_filters.include_patterns)
        exclude_patterns = list(self.file_filters.exclude_patterns)
        
        # Add LSP-based patterns if enabled and detector is available
        if self.lsp_based_extensions and get_default_detector is not None:
            try:
                detector = get_default_detector()
                detector.cache_ttl = self.lsp_detection_cache_ttl
                
                # Get LSP-detected extensions
                lsp_extensions = detector.get_supported_extensions(
                    include_fallbacks=self.lsp_fallback_enabled
                )
                
                # Convert extensions to glob patterns
                lsp_patterns = [f"*{ext}" for ext in lsp_extensions if ext.startswith('.')]
                
                # Add non-extension patterns (like Dockerfile, Makefile)
                for ext in lsp_extensions:
                    if not ext.startswith('.'):
                        lsp_patterns.append(ext)
                
                # Optionally add build tool patterns
                if self.lsp_include_build_tools:
                    build_patterns = []
                    for patterns in detector.BUILD_TOOL_EXTENSIONS.values():
                        for pattern in patterns:
                            if pattern.startswith('*.'):
                                build_patterns.append(pattern)
                            elif not pattern.startswith('.'):
                                build_patterns.append(pattern)
                    lsp_patterns.extend(build_patterns)
                
                # Optionally add infrastructure patterns  
                if self.lsp_include_infrastructure:
                    infra_patterns = []
                    for patterns in detector.INFRASTRUCTURE_EXTENSIONS.values():
                        for pattern in patterns:
                            if pattern.startswith('*.'):
                                infra_patterns.append(pattern)
                            elif not pattern.startswith('.'):
                                infra_patterns.append(pattern)
                    lsp_patterns.extend(infra_patterns)
                
                # Merge patterns, avoiding duplicates
                all_patterns = set(include_patterns + lsp_patterns)
                include_patterns = sorted(list(all_patterns))
                
                logger.debug(f"Enhanced patterns with LSP detection: {len(lsp_patterns)} LSP patterns added")
                
            except Exception as e:
                logger.warning(f"Failed to get LSP-based patterns: {e}")
                # Fall back to original patterns
        
        return (include_patterns, exclude_patterns)

    def should_process_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Determine if a file should be processed based on configuration.

        Returns:
            Tuple of (should_process, reason)
        """
        # Check file existence
        if not file_path.exists() or not file_path.is_file():
            return False, "File does not exist or is not a file"

        # Check size limits
        try:
            size = file_path.stat().st_size
            min_size = self.file_filters.size_limits.get("min_bytes", 0)
            max_size = self.file_filters.size_limits.get("max_bytes", float("inf"))

            if size < min_size:
                return False, f"File too small ({size} < {min_size} bytes)"
            if size > max_size:
                return False, f"File too large ({size} > {max_size} bytes)"
        except Exception as e:
            return False, f"Cannot get file size: {e}"

        # Check include patterns
        include_match = False
        for pattern in self.file_filters.include_patterns:
            if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(
                str(file_path), pattern
            ):
                include_match = True
                break

        if not include_match:
            return False, "File does not match include patterns"

        # Check exclude patterns
        for pattern in self.file_filters.exclude_patterns:
            if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(
                str(file_path), pattern
            ):
                return False, f"File matches exclude pattern: {pattern}"

        # Check regex patterns if specified
        if self.file_filters.regex_patterns:
            file_str = str(file_path)

            # Check include regex
            if "include" in self.file_filters.regex_patterns:
                pattern = self.file_filters.regex_patterns["include"]
                if not re.search(pattern, file_str):
                    return False, f"File does not match include regex: {pattern}"

            # Check exclude regex
            if "exclude" in self.file_filters.regex_patterns:
                pattern = self.file_filters.regex_patterns["exclude"]
                if re.search(pattern, file_str):
                    return False, f"File matches exclude regex: {pattern}"

        return True, "File passes all filters"

    def get_target_collection(self, file_path: Path) -> str:
        """Determine target collection for a file based on routing rules."""

        # Check routing rules first
        for rule in self.collection_config.routing_rules:
            pattern = rule["pattern"]
            collection = cast(str, rule["collection"])

            # Support different rule types
            rule_type = rule.get("type", "glob")

            if rule_type == "glob":
                if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(
                    str(file_path), pattern
                ):
                    return collection
            elif rule_type == "regex":
                if re.search(pattern, str(file_path)):
                    return collection
            elif rule_type == "extension":
                if file_path.suffix.lower() == pattern.lower():
                    return collection

        # Check collection prefixes
        for (
            prefix_key,
            prefix_value,
        ) in self.collection_config.collection_prefixes.items():
            if prefix_key == "extension" and file_path.suffix:
                return (
                    f"{prefix_value}{file_path.suffix[1:]}"  # Remove dot from extension
                )
            elif prefix_key == "directory" and file_path.parent.name:
                return f"{prefix_value}{file_path.parent.name}"

        # Return default collection
        return self.collection_config.default_collection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "path": self.path,
            "enabled": self.enabled,
            "file_filters": self.file_filters.dict(),
            "recursive": self.recursive.dict(),
            "performance": self.performance.dict(),
            "collection_config": self.collection_config.dict(),
            "auto_ingest": self.auto_ingest,
            "preserve_timestamps": self.preserve_timestamps,
            "create_backup_on_error": self.create_backup_on_error,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedWatchConfig":
        """Create from dictionary representation."""
        # Handle nested configurations
        file_filters_data = data.get("file_filters", {})
        recursive_data = data.get("recursive", {})
        performance_data = data.get("performance", {})
        collection_config_data = data.get("collection_config", {})

        return cls(
            id=data["id"],
            path=data["path"],
            enabled=data.get("enabled", True),
            file_filters=FileFilterConfig(**file_filters_data),
            recursive=RecursiveConfig(**recursive_data),
            performance=PerformanceConfig(**performance_data),
            collection_config=CollectionTargeting(**collection_config_data),
            auto_ingest=data.get("auto_ingest", True),
            preserve_timestamps=data.get("preserve_timestamps", True),
            create_backup_on_error=data.get("create_backup_on_error", False),
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
        )


class AdvancedConfigValidator:
    """Validator for advanced watch configurations."""

    @staticmethod
    def validate_patterns(patterns: List[str]) -> List[str]:
        """Validate glob patterns and return issues."""
        issues = []

        for pattern in patterns:
            try:
                # Test if it's a valid glob pattern
                fnmatch.fnmatch("test.txt", pattern)
            except Exception as e:
                issues.append(f"Invalid glob pattern '{pattern}': {e}")

        return issues

    @staticmethod
    def validate_regex(pattern: str) -> Optional[str]:
        """Validate regex pattern and return error if invalid."""
        try:
            re.compile(pattern)
            return None
        except re.error as e:
            return f"Invalid regex pattern: {e}"

    @staticmethod
    def validate_collection_routing(routing_rules: List[Dict[str, Any]]) -> List[str]:
        """Validate collection routing rules."""
        issues = []

        required_keys = {"pattern", "collection"}
        valid_types = {"glob", "regex", "extension"}

        for i, rule in enumerate(routing_rules):
            rule_prefix = f"Rule {i + 1}: "

            # Check required keys
            rule_keys = set(rule.keys())
            missing_keys = required_keys - rule_keys
            if missing_keys:
                issues.append(f"{rule_prefix}Missing required keys: {missing_keys}")
                continue

            # Validate rule type
            rule_type = rule.get("type", "glob")
            if rule_type not in valid_types:
                issues.append(
                    f"{rule_prefix}Invalid rule type '{rule_type}'. Valid types: {valid_types}"
                )

            # Validate pattern based on type
            pattern = rule.get("pattern", "")
            if rule_type == "regex":
                error = AdvancedConfigValidator.validate_regex(pattern)
                if error:
                    issues.append(f"{rule_prefix}{error}")
            elif rule_type in ["glob", "extension"]:
                if not pattern or not isinstance(pattern, str):
                    issues.append(f"{rule_prefix}Pattern must be a non-empty string")

        return issues

    @staticmethod
    def validate_performance_settings(
        performance_config: PerformanceConfig,
    ) -> List[str]:
        """Validate performance configuration settings."""
        issues = []

        # Check for conflicting settings
        if performance_config.batch_processing and performance_config.batch_size < 1:
            issues.append("Batch processing enabled but batch size is less than 1")

        if (
            performance_config.debounce_seconds > 60
            and performance_config.update_frequency_ms < 5000
        ):
            issues.append(
                "High debounce delay with low update frequency may cause delays"
            )

        if (
            performance_config.max_concurrent_ingestions > 10
            and performance_config.memory_limit_mb < 512
        ):
            issues.append(
                "High concurrency with low memory limit may cause performance issues"
            )

        return issues
