"""
PatternManager for workspace-qdrant-mcp.

This module provides a centralized pattern management system that loads hardcoded
patterns from embedded YAML files and merges them with custom user patterns.
It serves as the single source of truth for file inclusion/exclusion decisions,
ecosystem detection, and language information.

Key Features:
    - Loads patterns from embedded YAML files at runtime
    - Merges custom user patterns with higher precedence than hardcoded patterns
    - Provides unified interface for file filtering and ecosystem detection
    - Supports glob pattern matching for file paths
    - Caches pattern compilation for performance

Architecture:
    - Hardcoded patterns loaded from patterns/ directory YAML files
    - Custom patterns injected from WorkspaceConfig
    - Pattern matching uses fnmatch for glob support
    - Language detection based on file extensions and content analysis
    - Ecosystem detection based on project indicators

Example:
    ```python
    from workspace_qdrant_mcp.core.pattern_manager import PatternManager

    # Initialize with custom patterns
    pattern_manager = PatternManager(
        custom_include_patterns=["*.custom"],
        custom_exclude_patterns=["build/**"]
    )

    # Check file inclusion
    should_include = pattern_manager.should_include("/path/to/file.py")

    # Detect project ecosystem
    ecosystem = pattern_manager.detect_ecosystem("/path/to/project")
    ```
"""

import fnmatch
import os
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

# logger imported from loguru


class PatternManager:
    """
    Centralized pattern management system for file filtering and ecosystem detection.

    This class loads hardcoded patterns from YAML files and merges them with
    custom user patterns to provide unified file filtering and project analysis.
    """

    def __init__(
        self,
        custom_include_patterns: Optional[List[str]] = None,
        custom_exclude_patterns: Optional[List[str]] = None,
        custom_project_indicators: Optional[Dict[str, Any]] = None,
        patterns_base_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize PatternManager with optional custom patterns.

        Args:
            custom_include_patterns: Additional patterns to include files
            custom_exclude_patterns: Additional patterns to exclude files
            custom_project_indicators: Custom ecosystem indicators
            patterns_base_dir: Base directory containing pattern YAML files
        """
        self.custom_include_patterns = custom_include_patterns or []
        self.custom_exclude_patterns = custom_exclude_patterns or []
        self.custom_project_indicators = custom_project_indicators or {}

        # Set patterns directory - default to patterns/ in project root
        if patterns_base_dir:
            self.patterns_dir = Path(patterns_base_dir)
        else:
            # Find patterns directory relative to this file
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[4]  # Go up to project root
            self.patterns_dir = project_root / "patterns"

        # Initialize pattern storage
        self._include_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._exclude_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._project_indicators: Dict[str, Dict[str, Any]] = {}
        self._language_extensions: Dict[str, List[str]] = {}

        # Pattern matching cache for performance
        self._pattern_cache: Dict[str, bool] = {}
        self._cache_size_limit = 10000

        # Load patterns on initialization
        self._load_all_patterns()

        logger.info("PatternManager initialized",
                   patterns_dir=str(self.patterns_dir),
                   custom_include_count=len(self.custom_include_patterns),
                   custom_exclude_count=len(self.custom_exclude_patterns))

    def _load_all_patterns(self) -> None:
        """Load all patterns from YAML files."""
        try:
            self._load_include_patterns()
            self._load_exclude_patterns()
            self._load_project_indicators()
            self._load_language_extensions()
            logger.info("All patterns loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            # Set empty patterns as fallback
            self._include_patterns = {}
            self._exclude_patterns = {}
            self._project_indicators = {}
            self._language_extensions = {}

    def _load_include_patterns(self) -> None:
        """Load include patterns from YAML file."""
        include_file = self.patterns_dir / "include_patterns.yaml"
        if not include_file.exists():
            logger.warning(f"Include patterns file not found: {include_file}")
            return

        try:
            with include_file.open() as f:
                data = yaml.safe_load(f)

            # Process each category of patterns
            for category_name, patterns in data.items():
                if isinstance(patterns, list):
                    self._include_patterns[category_name] = patterns
                elif isinstance(patterns, dict):
                    # Skip metadata fields
                    if category_name in ['version', 'last_updated', 'research_coverage']:
                        continue
                    self._include_patterns[category_name] = patterns if isinstance(patterns, list) else []

            logger.debug(f"Loaded {sum(len(p) for p in self._include_patterns.values())} include patterns")

        except Exception as e:
            logger.error(f"Failed to load include patterns: {e}")
            self._include_patterns = {}

    def _load_exclude_patterns(self) -> None:
        """Load exclude patterns from YAML file."""
        exclude_file = self.patterns_dir / "exclude_patterns.yaml"
        if not exclude_file.exists():
            logger.warning(f"Exclude patterns file not found: {exclude_file}")
            return

        try:
            with exclude_file.open() as f:
                data = yaml.safe_load(f)

            # Process each category of patterns
            for category_name, patterns in data.items():
                if isinstance(patterns, list):
                    self._exclude_patterns[category_name] = patterns
                elif isinstance(patterns, dict):
                    # Skip metadata fields and process pattern dictionaries
                    if category_name in ['version', 'last_updated', 'research_coverage']:
                        continue
                    self._exclude_patterns[category_name] = patterns if isinstance(patterns, list) else []

            logger.debug(f"Loaded {sum(len(p) for p in self._exclude_patterns.values())} exclude patterns")

        except Exception as e:
            logger.error(f"Failed to load exclude patterns: {e}")
            self._exclude_patterns = {}

    def _load_project_indicators(self) -> None:
        """Load project indicator patterns from YAML file."""
        indicators_file = self.patterns_dir / "project_indicators.yaml"
        if not indicators_file.exists():
            logger.warning(f"Project indicators file not found: {indicators_file}")
            return

        try:
            with indicators_file.open() as f:
                data = yaml.safe_load(f)

            # Process ecosystem indicators
            ecosystems = data.get('ecosystems', {})
            for ecosystem_name, indicators in ecosystems.items():
                if isinstance(indicators, dict):
                    self._project_indicators[ecosystem_name] = indicators

            logger.debug(f"Loaded indicators for {len(self._project_indicators)} ecosystems")

        except Exception as e:
            logger.error(f"Failed to load project indicators: {e}")
            self._project_indicators = {}

    def _load_language_extensions(self) -> None:
        """Load language extension mappings from YAML file."""
        lang_file = self.patterns_dir / "language_extensions.yaml"
        if not lang_file.exists():
            logger.warning(f"Language extensions file not found: {lang_file}")
            return

        try:
            with lang_file.open() as f:
                data = yaml.safe_load(f)

            # Process all categories in the language extensions file
            for category_name, category_data in data.items():
                if isinstance(category_data, dict):
                    # Skip metadata fields
                    if category_name in ['version', 'last_updated', 'research_coverage']:
                        continue

                    # Process languages in this category
                    for language_name, lang_data in category_data.items():
                        if isinstance(lang_data, dict) and 'extensions' in lang_data:
                            self._language_extensions[language_name] = lang_data['extensions']

            logger.debug(f"Loaded extensions for {len(self._language_extensions)} languages")

        except Exception as e:
            logger.error(f"Failed to load language extensions: {e}")
            self._language_extensions = {}

    def should_include(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Determine if a file should be included based on patterns.

        Args:
            file_path: Path to the file to check

        Returns:
            Tuple of (should_include, reason)
        """
        file_path = Path(file_path)
        path_str = str(file_path)

        # Check cache first
        cache_key = f"include:{path_str}"
        if cache_key in self._pattern_cache:
            result = self._pattern_cache[cache_key]
            return result, "cached_decision"

        # Check custom include patterns first (higher precedence)
        for pattern in self.custom_include_patterns:
            if self._match_pattern(pattern, path_str):
                self._cache_result(cache_key, True)
                return True, f"custom_include_pattern: {pattern}"

        # Check hardcoded include patterns
        for category_name, patterns in self._include_patterns.items():
            for pattern_info in patterns:
                if isinstance(pattern_info, dict):
                    pattern = pattern_info.get('pattern', '')
                    if self._match_pattern(pattern, path_str):
                        self._cache_result(cache_key, True)
                        return True, f"hardcoded_include: {category_name}"
                elif isinstance(pattern_info, str):
                    if self._match_pattern(pattern_info, path_str):
                        self._cache_result(cache_key, True)
                        return True, f"hardcoded_include: {category_name}"

        # If no include pattern matches, file is not included by default
        self._cache_result(cache_key, False)
        return False, "no_include_pattern_match"

    def should_exclude(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Determine if a file should be excluded based on patterns.

        Args:
            file_path: Path to the file to check

        Returns:
            Tuple of (should_exclude, reason)
        """
        file_path = Path(file_path)
        path_str = str(file_path)

        # Check cache first
        cache_key = f"exclude:{path_str}"
        if cache_key in self._pattern_cache:
            result = self._pattern_cache[cache_key]
            return result, "cached_decision"

        # Check custom exclude patterns first (higher precedence)
        for pattern in self.custom_exclude_patterns:
            if self._match_pattern(pattern, path_str):
                self._cache_result(cache_key, True)
                return True, f"custom_exclude_pattern: {pattern}"

        # Check hardcoded exclude patterns
        for category_name, patterns in self._exclude_patterns.items():
            for pattern_info in patterns:
                if isinstance(pattern_info, dict):
                    pattern = pattern_info.get('pattern', '')
                    if self._match_pattern(pattern, path_str):
                        self._cache_result(cache_key, True)
                        return True, f"hardcoded_exclude: {category_name}"
                elif isinstance(pattern_info, str):
                    if self._match_pattern(pattern_info, path_str):
                        self._cache_result(cache_key, True)
                        return True, f"hardcoded_exclude: {category_name}"

        # If no exclude pattern matches, file is not excluded
        self._cache_result(cache_key, False)
        return False, "no_exclude_pattern_match"

    def detect_ecosystem(self, project_path: Union[str, Path]) -> List[str]:
        """
        Detect project ecosystem(s) based on indicator files.

        Args:
            project_path: Path to the project directory

        Returns:
            List of detected ecosystem names
        """
        project_path = Path(project_path)
        detected_ecosystems = []

        if not project_path.exists() or not project_path.is_dir():
            return detected_ecosystems

        # Check custom project indicators first
        for ecosystem, indicators in self.custom_project_indicators.items():
            if self._check_ecosystem_indicators(project_path, indicators):
                detected_ecosystems.append(ecosystem)

        # Check hardcoded project indicators
        for ecosystem, indicators in self._project_indicators.items():
            if self._check_ecosystem_indicators(project_path, indicators):
                if ecosystem not in detected_ecosystems:
                    detected_ecosystems.append(ecosystem)

        logger.debug(f"Detected ecosystems for {project_path}: {detected_ecosystems}")
        return detected_ecosystems

    def get_language_info(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get language information for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with language information or None if not recognized
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # Search through language extensions
        for language, extensions in self._language_extensions.items():
            if extension in extensions:
                return {
                    'language': language,
                    'extension': extension,
                    'file_path': str(file_path)
                }

        return None

    def _match_pattern(self, pattern: str, path: str) -> bool:
        """
        Match a glob pattern against a file path.

        Args:
            pattern: Glob pattern to match
            path: File path to test

        Returns:
            True if pattern matches path
        """
        try:
            # Handle directory patterns
            if pattern.endswith('/'):
                # Directory pattern - match if path is within directory
                dir_pattern = pattern.rstrip('/')
                path_parts = Path(path).parts
                return any(fnmatch.fnmatch(str(Path(*path_parts[:i+1])), dir_pattern)
                          for i in range(len(path_parts)))

            # Handle file patterns with ** (recursive match)
            if '**' in pattern:
                # Convert ** pattern to match any depth
                pattern_parts = pattern.split('**')
                if len(pattern_parts) == 2:
                    prefix, suffix = pattern_parts
                    prefix = prefix.rstrip('/')
                    suffix = suffix.lstrip('/')

                    if not prefix:  # Pattern starts with **
                        return fnmatch.fnmatch(Path(path).name, suffix)
                    elif not suffix:  # Pattern ends with **
                        return path.startswith(prefix)
                    else:  # ** in middle
                        return path.startswith(prefix) and fnmatch.fnmatch(Path(path).name, suffix)

            # Regular glob pattern matching
            return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)

        except Exception as e:
            logger.debug(f"Pattern matching error for '{pattern}' against '{path}': {e}")
            return False

    def _check_ecosystem_indicators(self, project_path: Path, indicators: Dict[str, Any]) -> bool:
        """
        Check if a project matches ecosystem indicators.

        Args:
            project_path: Path to project directory
            indicators: Ecosystem indicator configuration

        Returns:
            True if project matches indicators
        """
        try:
            # Check required files
            required_files = indicators.get('required_files', [])
            for file_pattern in required_files:
                found = False
                for file_path in project_path.rglob('*'):
                    if self._match_pattern(file_pattern, str(file_path.relative_to(project_path))):
                        found = True
                        break
                if not found:
                    return False

            # Check optional files (if any found, increases confidence)
            optional_files = indicators.get('optional_files', [])
            if optional_files:
                found_optional = 0
                for file_pattern in optional_files:
                    for file_path in project_path.rglob('*'):
                        if self._match_pattern(file_pattern, str(file_path.relative_to(project_path))):
                            found_optional += 1
                            break

                # Require at least some optional files to be present
                min_optional = indicators.get('min_optional_files', 1)
                if found_optional < min_optional:
                    return False

            return True

        except Exception as e:
            logger.debug(f"Error checking ecosystem indicators: {e}")
            return False

    def _cache_result(self, key: str, result: bool) -> None:
        """Cache a pattern matching result with size management."""
        if len(self._pattern_cache) >= self._cache_size_limit:
            # Remove oldest entries (simple FIFO eviction)
            keys_to_remove = list(self._pattern_cache.keys())[:self._cache_size_limit // 4]
            for old_key in keys_to_remove:
                self._pattern_cache.pop(old_key, None)

        self._pattern_cache[key] = result

    def clear_cache(self) -> None:
        """Clear the pattern matching cache."""
        self._pattern_cache.clear()
        logger.debug("PatternManager cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns and cache performance."""
        return {
            "include_patterns": {
                "categories": len(self._include_patterns),
                "total_patterns": sum(len(patterns) for patterns in self._include_patterns.values()),
                "custom_patterns": len(self.custom_include_patterns)
            },
            "exclude_patterns": {
                "categories": len(self._exclude_patterns),
                "total_patterns": sum(len(patterns) for patterns in self._exclude_patterns.values()),
                "custom_patterns": len(self.custom_exclude_patterns)
            },
            "project_indicators": {
                "ecosystems": len(self._project_indicators),
                "custom_indicators": len(self.custom_project_indicators)
            },
            "language_extensions": {
                "languages": len(self._language_extensions),
                "total_extensions": sum(len(exts) for exts in self._language_extensions.values())
            },
            "cache": {
                "size": len(self._pattern_cache),
                "limit": self._cache_size_limit
            }
        }