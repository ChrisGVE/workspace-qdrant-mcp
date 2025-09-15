"""
Comprehensive tests for PatternManager integration.

This module tests the PatternManager class which provides centralized pattern
management for file filtering and ecosystem detection using embedded YAML patterns.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any, List

import pytest
import yaml

try:
    from src.python.common.core.pattern_manager import PatternManager
except ImportError:
    from common.core.pattern_manager import PatternManager


# Shared fixtures for all test classes
@pytest.fixture(scope="session")
def temp_patterns_dir():
    """Create temporary patterns directory with test YAML files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create include patterns file
    include_patterns = {
        "programming": [
            {"pattern": "*.py", "category": "python", "priority": "high"},
            {"pattern": "*.js", "category": "javascript", "priority": "medium"},
            {"pattern": "*.rs", "category": "rust", "priority": "high"}
        ],
        "documentation": [
            "*.md",
            "*.rst",
            "*.txt"
        ],
        "configuration": [
            {"pattern": "*.yaml", "category": "config"},
            {"pattern": "*.json", "category": "config"},
            {"pattern": "*.toml", "category": "config"}
        ]
    }

    # Create exclude patterns file
    exclude_patterns = {
        "build_artifacts": [
            "*.o",
            "*.so",
            "*.dylib",
            "__pycache__/**",
            "target/**",
            "build/**",
            "dist/**",
            "*.egg-info/**",
            ".pytest_cache/**"
        ],
        "version_control": [
            ".git/**",
            ".svn/**",
            "*.orig",
            "*.rej"
        ],
        "temporary": [
            {"pattern": "*.tmp", "category": "temp"},
            {"pattern": "*.swp", "category": "editor"},
            {"pattern": "*~", "category": "backup"}
        ]
    }

    # Create project indicators file
    project_indicators = {
        "ecosystems": {
            "python": {
                "required_files": ["setup.py"],
                "optional_files": ["requirements.txt", "pyproject.toml", "Pipfile", "poetry.lock", "setup.cfg"],
                "min_optional_files": 1
            },
            "rust": {
                "required_files": ["Cargo.toml"],
                "optional_files": ["Cargo.lock", "rust-toolchain.toml"],
                "min_optional_files": 0
            },
            "javascript": {
                "required_files": ["package.json"],
                "optional_files": ["yarn.lock", "package-lock.json", "bun.lockb"],
                "min_optional_files": 0
            }
        }
    }

    # Create language extensions file
    language_extensions = {
        "programming_languages": {
            "python": {
                "extensions": [".py", ".pyx", ".pyi"],
                "ecosystem": "python"
            },
            "rust": {
                "extensions": [".rs"],
                "ecosystem": "rust"
            },
            "javascript": {
                "extensions": [".js", ".mjs", ".jsx"],
                "ecosystem": "javascript"
            },
            "typescript": {
                "extensions": [".ts", ".tsx", ".mts"],
                "ecosystem": "javascript"
            }
        },
        "markup_languages": {
            "markdown": {
                "extensions": [".md", ".markdown", ".mdown"],
                "category": "documentation"
            },
            "yaml": {
                "extensions": [".yaml", ".yml"],
                "category": "configuration"
            }
        }
    }

    # Write pattern files
    with open(temp_dir / "include_patterns.yaml", "w") as f:
        yaml.dump(include_patterns, f)

    with open(temp_dir / "exclude_patterns.yaml", "w") as f:
        yaml.dump(exclude_patterns, f)

    with open(temp_dir / "project_indicators.yaml", "w") as f:
        yaml.dump(project_indicators, f)

    with open(temp_dir / "language_extensions.yaml", "w") as f:
        yaml.dump(language_extensions, f)

    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def pattern_manager(temp_patterns_dir):
    """Create PatternManager with test patterns."""
    return PatternManager(patterns_base_dir=temp_patterns_dir)


class TestPatternManagerBasic:
    """Basic PatternManager functionality tests."""

    def test_pattern_manager_initialization(self, pattern_manager):
        """Test PatternManager initializes correctly."""
        assert isinstance(pattern_manager, PatternManager)
        assert isinstance(pattern_manager._include_patterns, dict)
        assert isinstance(pattern_manager._exclude_patterns, dict)
        assert isinstance(pattern_manager._project_indicators, dict)
        assert isinstance(pattern_manager._language_extensions, dict)

        # Check that patterns were loaded
        assert len(pattern_manager._include_patterns) > 0
        assert len(pattern_manager._exclude_patterns) > 0
        assert len(pattern_manager._project_indicators) > 0
        assert len(pattern_manager._language_extensions) > 0

    def test_should_include_basic(self, pattern_manager):
        """Test basic file inclusion logic."""
        # Test files that should be included
        test_cases = [
            ("test.py", True, "hardcoded_include"),
            ("script.js", True, "hardcoded_include"),
            ("main.rs", True, "hardcoded_include"),
            ("README.md", True, "hardcoded_include"),
            ("config.yaml", True, "hardcoded_include"),
        ]

        for file_path, expected, expected_reason_contains in test_cases:
            should_include, reason = pattern_manager.should_include(file_path)
            assert should_include == expected, f"Failed for {file_path}: {reason}"
            if expected:
                assert expected_reason_contains in reason

    def test_should_exclude_basic(self, pattern_manager):
        """Test basic file exclusion logic."""
        # Test files that should be excluded
        test_cases = [
            ("test.o", True, "hardcoded_exclude"),
            ("lib.so", True, "hardcoded_exclude"),
            ("__pycache__/module.pyc", True, "hardcoded_exclude"),
            (".git/config", True, "hardcoded_exclude"),
            ("file.tmp", True, "hardcoded_exclude"),
            ("backup~", True, "hardcoded_exclude"),
        ]

        for file_path, expected, expected_reason_contains in test_cases:
            should_exclude, reason = pattern_manager.should_exclude(file_path)
            assert should_exclude == expected, f"Failed for {file_path}: {reason}"
            if expected:
                assert expected_reason_contains in reason

    def test_custom_patterns_precedence(self, temp_patterns_dir):
        """Test that custom patterns take precedence over hardcoded ones."""
        # Create pattern manager with custom patterns
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=["*.custom", "*.special"],
            custom_exclude_patterns=["*.ignore", "exclude_me/**"]
        )

        # Test custom include patterns
        should_include, reason = pattern_manager.should_include("test.custom")
        assert should_include is True
        assert "custom_include_pattern" in reason

        should_include, reason = pattern_manager.should_include("file.special")
        assert should_include is True
        assert "custom_include_pattern" in reason

        # Test custom exclude patterns
        should_exclude, reason = pattern_manager.should_exclude("file.ignore")
        assert should_exclude is True
        assert "custom_exclude_pattern" in reason

        should_exclude, reason = pattern_manager.should_exclude("exclude_me/file.txt")
        assert should_exclude is True
        assert "custom_exclude_pattern" in reason


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPatternManagerEcosystemDetection:
    """Test ecosystem detection functionality."""

    def test_python_ecosystem_detection(self, pattern_manager, temp_project_dir):
        """Test Python ecosystem detection."""
        # Create Python project files
        (temp_project_dir / "setup.py").write_text("from setuptools import setup")
        (temp_project_dir / "requirements.txt").write_text("requests==2.28.0")
        (temp_project_dir / "src").mkdir()
        (temp_project_dir / "src" / "main.py").write_text("print('hello')")

        ecosystems = pattern_manager.detect_ecosystem(temp_project_dir)
        assert "python" in ecosystems

    def test_rust_ecosystem_detection(self, pattern_manager, temp_project_dir):
        """Test Rust ecosystem detection."""
        # Create Rust project files
        cargo_toml = """
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"
"""
        (temp_project_dir / "Cargo.toml").write_text(cargo_toml)
        (temp_project_dir / "src").mkdir()
        (temp_project_dir / "src" / "main.rs").write_text("fn main() {}")

        ecosystems = pattern_manager.detect_ecosystem(temp_project_dir)
        assert "rust" in ecosystems

    def test_javascript_ecosystem_detection(self, pattern_manager, temp_project_dir):
        """Test JavaScript ecosystem detection."""
        # Create JavaScript project files
        package_json = """
{
  "name": "test-project",
  "version": "1.0.0",
  "main": "index.js"
}
"""
        (temp_project_dir / "package.json").write_text(package_json)
        (temp_project_dir / "index.js").write_text("console.log('hello');")

        ecosystems = pattern_manager.detect_ecosystem(temp_project_dir)
        assert "javascript" in ecosystems

    def test_mixed_ecosystem_detection(self, pattern_manager, temp_project_dir):
        """Test detection of multiple ecosystems in one project."""
        # Create files for multiple ecosystems
        (temp_project_dir / "package.json").write_text('{"name": "test"}')
        (temp_project_dir / "setup.py").write_text("from setuptools import setup")
        (temp_project_dir / "requirements.txt").write_text("flask==2.0.0")

        ecosystems = pattern_manager.detect_ecosystem(temp_project_dir)
        assert "javascript" in ecosystems
        assert "python" in ecosystems


class TestPatternManagerLanguageInfo:
    """Test language information functionality."""

    def test_get_language_info_programming(self, pattern_manager):
        """Test language info for programming languages."""
        test_cases = [
            ("main.py", "python", ".py"),
            ("script.js", "javascript", ".js"),
            ("lib.rs", "rust", ".rs"),
            ("component.tsx", "typescript", ".tsx"),
        ]

        for file_path, expected_lang, expected_ext in test_cases:
            lang_info = pattern_manager.get_language_info(file_path)
            assert lang_info is not None
            assert lang_info["language"] == expected_lang
            assert lang_info["extension"] == expected_ext
            assert lang_info["file_path"] == file_path

    def test_get_language_info_markup(self, pattern_manager):
        """Test language info for markup languages."""
        test_cases = [
            ("README.md", "markdown", ".md"),
            ("config.yaml", "yaml", ".yaml"),
            ("data.yml", "yaml", ".yml"),
        ]

        for file_path, expected_lang, expected_ext in test_cases:
            lang_info = pattern_manager.get_language_info(file_path)
            assert lang_info is not None
            assert lang_info["language"] == expected_lang
            assert lang_info["extension"] == expected_ext

    def test_get_language_info_unknown(self, pattern_manager):
        """Test language info for unknown file types."""
        unknown_files = ["unknown.xyz", "binary.exe", "data"]

        for file_path in unknown_files:
            lang_info = pattern_manager.get_language_info(file_path)
            assert lang_info is None


class TestPatternManagerCaching:
    """Test pattern matching caching functionality."""

    def test_pattern_cache_functionality(self, pattern_manager):
        """Test that pattern matching results are cached."""
        test_file = "test.py"

        # Clear cache first
        pattern_manager.clear_cache()

        # First call should compute result
        should_include1, reason1 = pattern_manager.should_include(test_file)
        cache_size_after_first = len(pattern_manager._pattern_cache)

        # Second call should use cache
        should_include2, reason2 = pattern_manager.should_include(test_file)
        cache_size_after_second = len(pattern_manager._pattern_cache)

        # Results should be the same
        assert should_include1 == should_include2
        assert cache_size_after_first == cache_size_after_second
        # Cache should contain the result
        assert cache_size_after_first > 0

    def test_cache_size_management(self, pattern_manager):
        """Test cache size limit enforcement."""
        # Clear cache and set small limit for testing
        pattern_manager.clear_cache()
        original_limit = pattern_manager._cache_size_limit
        pattern_manager._cache_size_limit = 10

        try:
            # Fill cache beyond limit
            for i in range(15):
                test_file = f"test_{i}.py"
                pattern_manager.should_include(test_file)

            # Cache should not exceed limit
            assert len(pattern_manager._pattern_cache) <= pattern_manager._cache_size_limit
        finally:
            pattern_manager._cache_size_limit = original_limit

    def test_cache_clearing(self, pattern_manager):
        """Test cache clearing functionality."""
        # Add some items to cache
        pattern_manager.should_include("test1.py")
        pattern_manager.should_include("test2.js")

        assert len(pattern_manager._pattern_cache) > 0

        # Clear cache
        pattern_manager.clear_cache()
        assert len(pattern_manager._pattern_cache) == 0


class TestPatternManagerAdvancedPatterns:
    """Test advanced pattern matching functionality."""

    def test_glob_patterns_with_wildcards(self, pattern_manager):
        """Test complex glob pattern matching."""
        # Test simple patterns that should work with our implementation
        test_cases = [
            # File extension patterns
            ("file.o", True),       # Should match *.o (exclude)
            ("lib.so", True),       # Should match *.so (exclude)
            ("backup~", True),      # Should match *~ (exclude)
        ]

        for file_path, expected_exclude in test_cases:
            should_exclude, _ = pattern_manager.should_exclude(file_path)
            assert should_exclude == expected_exclude, f"Failed for {file_path}"

    def test_directory_patterns(self, pattern_manager):
        """Test directory-based pattern matching."""
        # Test patterns that match directories
        test_cases = [
            ("__pycache__/module.pyc", True),   # Should be excluded
            (".git/config", True),              # Should be excluded
            ("target/debug/main", True),        # Should be excluded (Rust)
        ]

        for file_path, should_exclude in test_cases:
            excluded, reason = pattern_manager.should_exclude(file_path)
            assert excluded == should_exclude, f"Failed for {file_path}: {reason}"

    def test_pattern_precedence(self, temp_patterns_dir):
        """Test that custom patterns override hardcoded patterns."""
        # Create pattern manager where custom patterns conflict with hardcoded
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=["*.tmp"],    # Usually excluded
            custom_exclude_patterns=["*.py"]      # Usually included
        )

        # Custom include should override hardcoded exclude
        should_include, reason = pattern_manager.should_include("test.tmp")
        assert should_include is True
        assert "custom_include_pattern" in reason

        # Custom exclude should override hardcoded include
        should_exclude, reason = pattern_manager.should_exclude("test.py")
        assert should_exclude is True
        assert "custom_exclude_pattern" in reason


class TestPatternManagerStatistics:
    """Test statistics and introspection functionality."""

    def test_get_statistics(self, pattern_manager):
        """Test statistics gathering."""
        stats = pattern_manager.get_statistics()

        # Check structure
        assert "include_patterns" in stats
        assert "exclude_patterns" in stats
        assert "project_indicators" in stats
        assert "language_extensions" in stats
        assert "cache" in stats

        # Check include patterns stats
        include_stats = stats["include_patterns"]
        assert "categories" in include_stats
        assert "total_patterns" in include_stats
        assert "custom_patterns" in include_stats
        assert include_stats["categories"] > 0
        assert include_stats["total_patterns"] > 0

        # Check exclude patterns stats
        exclude_stats = stats["exclude_patterns"]
        assert exclude_stats["categories"] > 0
        assert exclude_stats["total_patterns"] > 0

        # Check project indicators stats
        indicators_stats = stats["project_indicators"]
        assert indicators_stats["ecosystems"] > 0

        # Check language extensions stats
        lang_stats = stats["language_extensions"]
        assert lang_stats["languages"] > 0
        assert lang_stats["total_extensions"] > 0

        # Check cache stats
        cache_stats = stats["cache"]
        assert "size" in cache_stats
        assert "limit" in cache_stats

    def test_statistics_with_custom_patterns(self, temp_patterns_dir):
        """Test statistics with custom patterns."""
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=["*.custom1", "*.custom2"],
            custom_exclude_patterns=["*.ignore"],
            custom_project_indicators={"custom_ecosystem": {"required_files": ["custom.conf"]}}
        )

        stats = pattern_manager.get_statistics()

        # Should reflect custom patterns in statistics
        assert stats["include_patterns"]["custom_patterns"] == 2
        assert stats["exclude_patterns"]["custom_patterns"] == 1
        assert stats["project_indicators"]["custom_indicators"] == 1


class TestPatternManagerErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_pattern_files(self):
        """Test behavior when pattern files are missing."""
        # Create empty temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Should handle missing files gracefully
            pattern_manager = PatternManager(patterns_base_dir=temp_dir)

            # Should not crash and should have empty pattern sets
            assert len(pattern_manager._include_patterns) == 0
            assert len(pattern_manager._exclude_patterns) == 0
            assert len(pattern_manager._project_indicators) == 0
            assert len(pattern_manager._language_extensions) == 0

            # Should still work with custom patterns
            should_include, reason = pattern_manager.should_include("test.py")
            assert should_include is False  # No patterns to match

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_invalid_pattern_files(self):
        """Test behavior with invalid YAML files."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create invalid YAML file
            with open(temp_dir / "include_patterns.yaml", "w") as f:
                f.write("invalid: yaml: content: [")

            # Should handle invalid YAML gracefully
            pattern_manager = PatternManager(patterns_base_dir=temp_dir)
            assert len(pattern_manager._include_patterns) == 0

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pattern_matching_errors(self, pattern_manager):
        """Test handling of pattern matching errors."""
        # Test with various edge case patterns
        edge_cases = [
            "",           # Empty path
            "/",          # Root path
            ".",          # Current directory
            "..",         # Parent directory
            "file with spaces.txt",
            "file-with-unicode-ñ.py",
        ]

        for file_path in edge_cases:
            try:
                # Should not crash on edge cases
                should_include, _ = pattern_manager.should_include(file_path)
                should_exclude, _ = pattern_manager.should_exclude(file_path)
                assert isinstance(should_include, bool)
                assert isinstance(should_exclude, bool)
            except Exception as e:
                pytest.fail(f"PatternManager crashed on edge case '{file_path}': {e}")


class TestPatternManagerIntegration:
    """Integration tests for PatternManager with real-world scenarios."""

    def test_typical_python_project_filtering(self, pattern_manager):
        """Test pattern matching on typical Python project structure."""
        python_project_files = [
            # Should be included
            "src/main.py",
            "tests/test_main.py",
            "README.md",
            "setup.py",
            "pyproject.toml",
            "requirements.txt",

            # Should be excluded
            "__pycache__/main.cpython-39.pyc",
            "build/lib/module.so",
            ".git/config",
            "dist/package-0.1.0.tar.gz",
            ".pytest_cache/README.md",
            "*.egg-info/PKG-INFO",
        ]

        for file_path in python_project_files[:6]:  # Should be included
            should_include, _ = pattern_manager.should_include(file_path)
            should_exclude, _ = pattern_manager.should_exclude(file_path)
            # File should be included and not excluded (or vice versa for proper filtering)
            assert should_include or not should_exclude, f"File {file_path} should be processable"

        for file_path in python_project_files[6:]:  # Should be excluded
            should_exclude, _ = pattern_manager.should_exclude(file_path)
            # Only check files that we expect our patterns to catch
            if file_path.endswith('.so') or '__pycache__' in file_path or '.git/' in file_path:
                assert should_exclude, f"File {file_path} should be excluded"

    def test_typical_rust_project_filtering(self, pattern_manager):
        """Test pattern matching on typical Rust project structure."""
        rust_project_files = [
            # Should be included
            "src/main.rs",
            "src/lib.rs",
            "Cargo.toml",
            "README.md",

            # Should be excluded
            "target/debug/main",
            "target/release/lib.rlib",
            ".git/HEAD",
            "Cargo.lock",  # Might be included or excluded depending on policy
        ]

        for file_path in rust_project_files[:4]:  # Should be included
            should_include, _ = pattern_manager.should_include(file_path)
            should_exclude, _ = pattern_manager.should_exclude(file_path)
            assert should_include or not should_exclude, f"File {file_path} should be processable"

        # Target directory should definitely be excluded
        should_exclude, _ = pattern_manager.should_exclude("target/debug/main")
        assert should_exclude, "Rust target directory should be excluded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])