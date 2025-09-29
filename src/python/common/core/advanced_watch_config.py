"""
Advanced watch configuration options with lua-style configuration access.

This module provides lua-style configuration access functions for enhanced file watching,
including advanced pattern matching, collection targeting, and performance tuning options.
All configuration is accessed through get_config() functions matching the Rust pattern.
"""

import fnmatch
from loguru import logger
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from .config import get_config_dict, get_config_list, get_config_int, get_config_string, get_config_bool

# Import LSP detector for dynamic extension detection
try:
    from .lsp_detector import get_default_detector
except ImportError:
    # Fallback if LSP detector is not available
    get_default_detector = None

# Import PatternManager for default patterns
try:
    from .pattern_manager import PatternManager

    def _get_advanced_default_patterns() -> List[str]:
        """Get default include patterns for advanced watch config."""
        try:
            pattern_manager = PatternManager()
            # TODO: Get from PatternManager YAML in future
            return ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback patterns: {e}")
            return ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"]

    def _get_advanced_default_exclude_patterns() -> List[str]:
        """Get default exclude patterns for advanced watch config."""
        try:
            pattern_manager = PatternManager()
            # TODO: Get from PatternManager YAML in future
            return [
                ".git/*", "node_modules/*", "__pycache__/*", ".DS_Store",
                "*.tmp", "*.temp", "*.log", "*.swp", "*.swo"
            ]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback exclude patterns: {e}")
            return [
                ".git/*", "node_modules/*", "__pycache__/*", ".DS_Store",
                "*.tmp", "*.temp", "*.log", "*.swp", "*.swo"
            ]

except ImportError:
    logger.debug("PatternManager not available - using hardcoded advanced patterns")

    def _get_advanced_default_patterns() -> List[str]:
        """Fallback default include patterns for advanced watch config."""
        return ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"]

    def _get_advanced_default_exclude_patterns() -> List[str]:
        """Fallback default exclude patterns for advanced watch config."""
        return [
            ".git/*", "node_modules/*", "__pycache__/*", ".DS_Store",
            "*.tmp", "*.temp", "*.log", "*.swp", "*.swo"
        ]


# =============================================================================
# LUA-STYLE CONFIGURATION ACCESS FUNCTIONS
# =============================================================================

# File filtering configuration functions
def get_file_filter_include_patterns() -> List[str]:
    """Get file filter include patterns using lua-style configuration access."""
    return get_config_list("auto_ingestion.file_filters.include_patterns", _get_advanced_default_patterns())


def get_file_filter_exclude_patterns() -> List[str]:
    """Get file filter exclude patterns using lua-style configuration access."""
    return get_config_list("auto_ingestion.file_filters.exclude_patterns", _get_advanced_default_exclude_patterns())


def get_file_filter_mime_types() -> List[str]:
    """Get file filter MIME types using lua-style configuration access."""
    return get_config_list("auto_ingestion.file_filters.mime_types", [])


def get_file_filter_size_limits() -> Dict[str, int]:
    """Get file filter size limits using lua-style configuration access."""
    return get_config_dict("auto_ingestion.file_filters.size_limits", {
        "min_bytes": 1,
        "max_bytes": 100 * 1024 * 1024,  # 100MB max
    })


def get_file_filter_regex_patterns() -> Dict[str, str]:
    """Get file filter regex patterns using lua-style configuration access."""
    return get_config_dict("auto_ingestion.file_filters.regex_patterns", {})


# Recursive scanning configuration functions
def get_recursive_config_enabled() -> bool:
    """Get recursive scanning enabled setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.recursive.enabled", True)


def get_recursive_config_max_depth() -> int:
    """Get recursive scanning max depth using lua-style configuration access."""
    return get_config_int("auto_ingestion.recursive.max_depth", -1)


def get_recursive_config_follow_symlinks() -> bool:
    """Get recursive scanning follow symlinks setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.recursive.follow_symlinks", False)


def get_recursive_config_skip_hidden() -> bool:
    """Get recursive scanning skip hidden setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.recursive.skip_hidden", True)


def get_recursive_config_exclude_dirs() -> List[str]:
    """Get recursive scanning exclude directories using lua-style configuration access."""
    return get_config_list("auto_ingestion.recursive.exclude_dirs", [".git", ".svn", ".hg", "node_modules", "__pycache__"])


# Performance configuration functions
def get_performance_config_update_frequency() -> int:
    """Get performance update frequency using lua-style configuration access."""
    return get_config_int("auto_ingestion.performance.update_frequency_ms", 1000)


def get_performance_config_debounce_seconds() -> int:
    """Get performance debounce seconds using lua-style configuration access."""
    return get_config_int("auto_ingestion.performance.debounce_seconds", 5)


def get_performance_config_batch_processing() -> bool:
    """Get performance batch processing setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.performance.batch_processing", True)


def get_performance_config_batch_size() -> int:
    """Get performance batch size using lua-style configuration access."""
    return get_config_int("auto_ingestion.performance.batch_size", 10)


def get_performance_config_memory_limit() -> int:
    """Get performance memory limit using lua-style configuration access."""
    return get_config_int("auto_ingestion.performance.memory_limit_mb", 256)


def get_performance_config_max_concurrent() -> int:
    """Get performance max concurrent ingestions using lua-style configuration access."""
    return get_config_int("auto_ingestion.performance.max_concurrent_ingestions", 5)


# Collection targeting configuration functions
def get_collection_targeting_default() -> str:
    """Get default collection for targeting using lua-style configuration access."""
    return get_config_string("auto_ingestion.collection_targeting.default_collection", "default")


def get_collection_targeting_routing_rules() -> List[Dict[str, Any]]:
    """Get collection targeting routing rules using lua-style configuration access."""
    return get_config_list("auto_ingestion.collection_targeting.routing_rules", [])


def get_collection_targeting_prefixes() -> Dict[str, str]:
    """Get collection targeting prefixes using lua-style configuration access."""
    return get_config_dict("auto_ingestion.collection_targeting.collection_prefixes", {})


# LSP configuration functions
def get_lsp_based_extensions_enabled() -> bool:
    """Get LSP-based extension detection setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.lsp.enabled", True)


def get_lsp_detection_cache_ttl() -> int:
    """Get LSP detection cache TTL using lua-style configuration access."""
    return get_config_int("auto_ingestion.lsp.cache_ttl", 300)


def get_lsp_fallback_enabled() -> bool:
    """Get LSP fallback enabled setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.lsp.fallback_enabled", True)


def get_lsp_include_build_tools() -> bool:
    """Get LSP include build tools setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.lsp.include_build_tools", True)


def get_lsp_include_infrastructure() -> bool:
    """Get LSP include infrastructure setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.lsp.include_infrastructure", True)


# Processing options functions
def get_auto_ingest_enabled() -> bool:
    """Get auto-ingestion enabled setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.enabled", True)


def get_preserve_timestamps() -> bool:
    """Get preserve timestamps setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.preserve_timestamps", True)


def get_create_backup_on_error() -> bool:
    """Get create backup on error setting using lua-style configuration access."""
    return get_config_bool("auto_ingestion.create_backup_on_error", False)


# =============================================================================
# VALIDATION FUNCTIONS (NO LONGER CLASS-BASED)
# =============================================================================

def validate_patterns(patterns: List[str]) -> bool:
    """Validate glob patterns."""
    if not patterns:
        return False

    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            return False
        # Test if it's a valid glob pattern
        try:
            fnmatch.fnmatch("test.txt", pattern)
        except Exception:
            return False
    return True


def validate_regex_patterns(regex_patterns: Dict[str, str]) -> bool:
    """Validate regex patterns."""
    for key, pattern in regex_patterns.items():
        if key not in ["include", "exclude"]:
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
    return True


def validate_path(path: str) -> List[str]:
    """Validate watch path and return any issues."""
    issues = []
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            issues.append(f"Watch path does not exist: {path}")
        elif not path_obj.is_dir():
            issues.append(f"Watch path is not a directory: {path}")
    except Exception as e:
        issues.append(f"Invalid path format: {e}")
    return issues


def validate_routing_rules(routing_rules: List[Dict[str, Any]]) -> List[str]:
    """Validate collection routing rules and return any issues."""
    issues = []
    required_keys = {"pattern", "collection"}

    for i, rule in enumerate(routing_rules):
        if not isinstance(rule, dict):
            issues.append(f"Routing rule {i} must be a dictionary")
            continue

        rule_keys = set(rule.keys())
        if not required_keys.issubset(rule_keys):
            issues.append(f"Routing rule {i} must contain keys: {required_keys}")
            continue

        # Validate pattern
        pattern = rule.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            issues.append(f"Routing rule {i} pattern must be a non-empty string")

        # Validate collection
        collection = rule.get("collection")
        if not isinstance(collection, str) or not collection.strip():
            issues.append(f"Routing rule {i} collection must be a non-empty string")

    return issues


def validate_complete_config(path: str) -> List[str]:
    """Validate complete advanced watch configuration and return any issues."""
    issues = []

    # Validate path
    issues.extend(validate_path(path))

    # Validate file filter patterns
    include_patterns = get_file_filter_include_patterns()
    if not validate_patterns(include_patterns):
        issues.append("Invalid include patterns in file filters")

    exclude_patterns = get_file_filter_exclude_patterns()
    if not validate_patterns(exclude_patterns):
        issues.append("Invalid exclude patterns in file filters")

    # Validate regex patterns
    regex_patterns = get_file_filter_regex_patterns()
    if regex_patterns and not validate_regex_patterns(regex_patterns):
        issues.append("Invalid regex patterns in file filters")

    # Validate routing rules
    routing_rules = get_collection_targeting_routing_rules()
    issues.extend(validate_routing_rules(routing_rules))

    # Validate performance settings
    max_depth = get_recursive_config_max_depth()
    if max_depth < -1 or max_depth > 20:
        issues.append("Recursive max depth must be between -1 and 20")

    update_frequency = get_performance_config_update_frequency()
    if update_frequency < 100 or update_frequency > 60000:
        issues.append("Update frequency must be between 100ms and 60s")

    debounce_seconds = get_performance_config_debounce_seconds()
    if debounce_seconds < 1 or debounce_seconds > 300:
        issues.append("Debounce seconds must be between 1 and 300")

    batch_size = get_performance_config_batch_size()
    if batch_size < 1 or batch_size > 100:
        issues.append("Batch size must be between 1 and 100")

    memory_limit = get_performance_config_memory_limit()
    if memory_limit < 64 or memory_limit > 2048:
        issues.append("Memory limit must be between 64MB and 2048MB")

    max_concurrent = get_performance_config_max_concurrent()
    if max_concurrent < 1 or max_concurrent > 20:
        issues.append("Max concurrent ingestions must be between 1 and 20")

    # Performance optimization warnings
    if max_concurrent > 10 and memory_limit < 512:
        issues.append("High concurrency with low memory limit may cause performance issues")

    return issues