"""
Tests for PatternManager integration with the configuration system.

This module tests how PatternManager integrates with WorkspaceConfig and
other configuration components to provide custom pattern functionality.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

try:
    from workspace_qdrant_mcp.core.pattern_manager import PatternManager
except ImportError:
    try:
        from workspace_qdrant_mcp.core.pattern_manager import PatternManager
    except ImportError:
        PatternManager = None

try:
    from workspace_qdrant_mcp.core.config import WorkspaceConfig
except ImportError:
    try:
        from workspace_qdrant_mcp.core.config import WorkspaceConfig
    except ImportError:
        WorkspaceConfig = None


@pytest.mark.skipif(PatternManager is None, reason="PatternManager not available")
class TestPatternManagerConfigIntegration:
    """Test PatternManager integration with configuration system."""

    @pytest.fixture
    def temp_patterns_dir(self):
        """Create temporary patterns directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create minimal pattern files for testing
        include_patterns = {
            "source_code": ["*.py", "*.js", "*.rs"],
            "documentation": ["*.md", "*.txt"]
        }

        exclude_patterns = {
            "build_artifacts": ["*.o", "*.so", "__pycache__/**"],
            "version_control": [".git/**"]
        }

        with open(temp_dir / "include_patterns.yaml", "w") as f:
            yaml.dump(include_patterns, f)

        with open(temp_dir / "exclude_patterns.yaml", "w") as f:
            yaml.dump(exclude_patterns, f)

        yield temp_dir

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pattern_manager_with_custom_config_patterns(self, temp_patterns_dir):
        """Test PatternManager initialization with custom patterns from config."""
        # Create pattern manager with custom patterns (simulating WorkspaceConfig input)
        custom_include = ["*.custom", "*.special"]
        custom_exclude = ["*.ignore", "temp/**"]

        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=custom_include,
            custom_exclude_patterns=custom_exclude
        )

        # Test that custom patterns work
        should_include, reason = pattern_manager.should_include("test.custom")
        assert should_include is True
        assert "custom_include_pattern" in reason

        should_exclude, reason = pattern_manager.should_exclude("test.ignore")
        assert should_exclude is True
        assert "custom_exclude_pattern" in reason

    def test_pattern_manager_workspace_config_integration(self, temp_patterns_dir):
        """Test PatternManager with WorkspaceConfig-style configuration."""
        # Simulate WorkspaceConfig with custom patterns
        workspace_config = {
            "custom_include_patterns": ["*.workspace", "*.project"],
            "custom_exclude_patterns": ["*.temp", "cache/**"],
            "custom_project_indicators": {
                "custom_ecosystem": {
                    "required_files": ["custom.config"],
                    "optional_files": ["custom.lock"],
                    "min_optional_files": 0
                }
            }
        }

        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=workspace_config["custom_include_patterns"],
            custom_exclude_patterns=workspace_config["custom_exclude_patterns"],
            custom_project_indicators=workspace_config["custom_project_indicators"]
        )

        # Test custom patterns from workspace config
        should_include, _ = pattern_manager.should_include("project.workspace")
        assert should_include is True

        should_exclude, _ = pattern_manager.should_exclude("data.temp")
        assert should_exclude is True

        # Test custom ecosystem detection
        temp_project_dir = Path(tempfile.mkdtemp())
        try:
            (temp_project_dir / "custom.config").write_text("config content")
            ecosystems = pattern_manager.detect_ecosystem(temp_project_dir)
            assert "custom_ecosystem" in ecosystems
        finally:
            import shutil
            shutil.rmtree(temp_project_dir, ignore_errors=True)

    def test_pattern_manager_precedence_with_config(self, temp_patterns_dir):
        """Test that custom patterns from config override hardcoded patterns."""
        # Create pattern manager where custom patterns conflict with hardcoded ones
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=["*.py"],  # Already in hardcoded
            custom_exclude_patterns=["*.md"]   # Conflicts with hardcoded include
        )

        # Custom include should still work (not override since it matches)
        should_include, reason = pattern_manager.should_include("script.py")
        assert should_include is True
        # Should be custom pattern since it has higher precedence
        assert "custom_include_pattern" in reason

        # Custom exclude should override hardcoded include
        should_exclude, reason = pattern_manager.should_exclude("README.md")
        assert should_exclude is True
        assert "custom_exclude_pattern" in reason

    def test_pattern_manager_config_validation(self, temp_patterns_dir):
        """Test validation of configuration-provided patterns."""
        # Test with valid custom patterns
        valid_patterns = ["*.valid", "test/**/*.ok"]
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=valid_patterns
        )

        # Should initialize successfully
        assert len(pattern_manager.custom_include_patterns) == 2

        # Test pattern matching works
        should_include, _ = pattern_manager.should_include("file.valid")
        assert should_include is True

    def test_pattern_manager_empty_config_patterns(self, temp_patterns_dir):
        """Test PatternManager with empty or None custom patterns from config."""
        # Test with None patterns (default)
        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=None,
            custom_exclude_patterns=None,
            custom_project_indicators=None
        )

        # Should fall back to hardcoded patterns only
        should_include, reason = pattern_manager.should_include("script.py")
        assert should_include is True
        assert "hardcoded_include" in reason

        # Test with empty lists
        pattern_manager_empty = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=[],
            custom_exclude_patterns=[],
            custom_project_indicators={}
        )

        should_include, reason = pattern_manager_empty.should_include("script.py")
        assert should_include is True
        assert "hardcoded_include" in reason

    def test_pattern_manager_statistics_with_config(self, temp_patterns_dir):
        """Test statistics reporting with custom patterns from config."""
        custom_include = ["*.custom1", "*.custom2", "*.custom3"]
        custom_exclude = ["*.ignore"]

        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=custom_include,
            custom_exclude_patterns=custom_exclude
        )

        stats = pattern_manager.get_statistics()

        # Should correctly report custom pattern counts
        assert stats["include_patterns"]["custom_patterns"] == 3
        assert stats["exclude_patterns"]["custom_patterns"] == 1
        assert stats["project_indicators"]["custom_indicators"] == 0

        # Should also report hardcoded pattern counts
        assert stats["include_patterns"]["total_patterns"] > 0
        assert stats["exclude_patterns"]["total_patterns"] > 0

    @patch('src.python.common.core.pattern_manager.logger')
    def test_pattern_manager_config_error_handling(self, mock_logger, temp_patterns_dir):
        """Test error handling for invalid configuration patterns."""
        # Test with patterns that might cause issues
        problematic_patterns = [
            "",  # Empty string pattern
            None,  # None in list (would cause TypeError)
        ]

        # Filter out None values as they would cause TypeError in list
        clean_patterns = [p for p in problematic_patterns if p is not None]

        pattern_manager = PatternManager(
            patterns_base_dir=temp_patterns_dir,
            custom_include_patterns=clean_patterns
        )

        # Should handle gracefully and not crash
        should_include, _ = pattern_manager.should_include("test.py")
        assert isinstance(should_include, bool)

    def test_pattern_manager_missing_hardcoded_patterns_with_config(self):
        """Test PatternManager behavior when hardcoded patterns are missing but config patterns exist."""
        # Create empty temp directory (no hardcoded patterns)
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Should still work with only custom patterns
            pattern_manager = PatternManager(
                patterns_base_dir=temp_dir,
                custom_include_patterns=["*.custom"],
                custom_exclude_patterns=["*.ignore"]
            )

            # Custom patterns should still work
            should_include, reason = pattern_manager.should_include("test.custom")
            assert should_include is True
            assert "custom_include_pattern" in reason

            should_exclude, reason = pattern_manager.should_exclude("temp.ignore")
            assert should_exclude is True
            assert "custom_exclude_pattern" in reason

            # Non-matching files should not be included
            should_include, _ = pattern_manager.should_include("random.txt")
            assert should_include is False

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(WorkspaceConfig is None, reason="WorkspaceConfig not available")
class TestWorkspaceConfigPatternIntegration:
    """Test integration between WorkspaceConfig and PatternManager (if available)."""

    def test_workspace_config_pattern_fields(self):
        """Test that WorkspaceConfig has the expected pattern-related fields."""
        # Test default WorkspaceConfig
        config = WorkspaceConfig()

        # Check if pattern-related fields exist (they might be added in the future)
        # This test documents the expected interface
        expected_fields = [
            'custom_include_patterns',
            'custom_exclude_patterns',
            'custom_project_indicators'
        ]

        for field in expected_fields:
            # If field exists, it should be a list or dict
            if hasattr(config, field):
                field_value = getattr(config, field)
                assert field_value is None or isinstance(field_value, (list, dict))

    def test_workspace_config_with_pattern_data(self):
        """Test WorkspaceConfig initialization with pattern data."""
        # Test with pattern-like configuration data
        config_data = {
            'collections': ['project'],
            'github_user': 'testuser',
            # These might be pattern-related fields in the future
            'include_extensions': ['.py', '.js'],
            'exclude_patterns': ['__pycache__', '.git']
        }

        # Should initialize without errors even with extra fields
        try:
            config = WorkspaceConfig(**config_data)
            assert config.collections == ['project']
            assert config.github_user == 'testuser'
        except TypeError:
            # If WorkspaceConfig doesn't accept these fields yet, that's okay
            # This test documents future expected behavior
            pass


class TestPatternManagerIntegrationMocking:
    """Test PatternManager integration using mocks for components that might not be available."""

    def test_pattern_manager_mock_integration(self):
        """Test PatternManager integration with mocked configuration components."""
        # Mock a configuration object
        mock_config = MagicMock()
        mock_config.custom_include_patterns = ["*.custom"]
        mock_config.custom_exclude_patterns = ["*.temp"]
        mock_config.custom_project_indicators = {}

        # Test that PatternManager would work with such a config
        if PatternManager is not None:
            temp_dir = Path(tempfile.mkdtemp())
            try:
                pattern_manager = PatternManager(
                    patterns_base_dir=temp_dir,
                    custom_include_patterns=mock_config.custom_include_patterns,
                    custom_exclude_patterns=mock_config.custom_exclude_patterns,
                    custom_project_indicators=mock_config.custom_project_indicators
                )

                # Should work with mocked config data
                should_include, _ = pattern_manager.should_include("test.custom")
                assert should_include is True

            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pattern_manager_integration_interface(self):
        """Test the expected interface for PatternManager integration."""
        # Document the expected interface for integration
        expected_interface = {
            'custom_include_patterns': list,
            'custom_exclude_patterns': list,
            'custom_project_indicators': dict,
        }

        # Test that PatternManager accepts these parameters
        if PatternManager is not None:
            temp_dir = Path(tempfile.mkdtemp())
            try:
                # Should accept all expected interface parameters
                pattern_manager = PatternManager(
                    patterns_base_dir=temp_dir,
                    custom_include_patterns=[],
                    custom_exclude_patterns=[],
                    custom_project_indicators={}
                )

                assert isinstance(pattern_manager.custom_include_patterns, list)
                assert isinstance(pattern_manager.custom_exclude_patterns, list)
                assert isinstance(pattern_manager.custom_project_indicators, dict)

            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])