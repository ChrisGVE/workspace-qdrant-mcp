"""
Unit tests for Tree-sitter Grammar Configuration Management.

Tests configuration loading, saving, updating, and thread safety.
"""

import pytest
import tempfile
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.python.common.core.grammar_config import (
    GrammarConfig,
    ConfigManager
)


class TestGrammarConfig:
    """Test suite for GrammarConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GrammarConfig()

        assert config.grammar_directories == []
        assert config.installation_directory is None
        assert config.c_compiler is None
        assert config.cpp_compiler is None
        assert config.auto_compile is True
        assert config.parallel_builds == 1
        assert config.optimization_level == "O2"
        assert config.keep_build_artifacts is False
        assert config.default_clone_depth is None
        assert config.custom_compiler_flags == {}

    def test_config_with_values(self):
        """Test configuration with custom values."""
        config = GrammarConfig(
            grammar_directories=["/path/to/grammars"],
            c_compiler="clang",
            cpp_compiler="clang++",
            auto_compile=False,
            parallel_builds=4
        )

        assert config.grammar_directories == ["/path/to/grammars"]
        assert config.c_compiler == "clang"
        assert config.cpp_compiler == "clang++"
        assert config.auto_compile is False
        assert config.parallel_builds == 4

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = GrammarConfig(
            grammar_directories=["/path/to/grammars"],
            c_compiler="gcc"
        )

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["grammar_directories"] == ["/path/to/grammars"]
        assert data["c_compiler"] == "gcc"
        assert "auto_compile" in data
        assert "parallel_builds" in data

    def test_from_dict(self):
        """Test configuration deserialization from dictionary."""
        data = {
            "grammar_directories": ["/path/to/grammars"],
            "c_compiler": "clang",
            "auto_compile": False,
            "parallel_builds": 2
        }

        config = GrammarConfig.from_dict(data)

        assert config.grammar_directories == ["/path/to/grammars"]
        assert config.c_compiler == "clang"
        assert config.auto_compile is False
        assert config.parallel_builds == 2

    def test_from_dict_with_unknown_fields(self):
        """Test from_dict gracefully handles unknown fields."""
        data = {
            "c_compiler": "gcc",
            "unknown_field": "value",  # Should be ignored
            "another_unknown": 123
        }

        config = GrammarConfig.from_dict(data)

        assert config.c_compiler == "gcc"
        assert not hasattr(config, "unknown_field")

    def test_custom_compiler_flags(self):
        """Test custom compiler flags configuration."""
        config = GrammarConfig(
            custom_compiler_flags={
                "gcc": ["-Wall", "-Werror"],
                "clang": ["-Wextra"]
            }
        )

        assert config.custom_compiler_flags["gcc"] == ["-Wall", "-Werror"]
        assert config.custom_compiler_flags["clang"] == ["-Wextra"]


class TestConfigManager:
    """Test suite for ConfigManager class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(config_dir=temp_config_dir)

    def test_manager_initialization(self, temp_config_dir):
        """Test config manager initialization creates directory and default config."""
        manager = ConfigManager(config_dir=temp_config_dir)

        assert temp_config_dir.exists()
        assert (temp_config_dir / "config.json").exists()

    def test_load_default_config(self, manager):
        """Test loading default configuration."""
        config = manager.load()

        assert isinstance(config, GrammarConfig)
        assert config.auto_compile is True
        assert config.parallel_builds == 1

    def test_save_and_load(self, manager):
        """Test saving and loading configuration."""
        config = GrammarConfig(
            c_compiler="clang",
            cpp_compiler="clang++",
            auto_compile=False
        )

        manager.save(config)
        loaded = manager.load()

        assert loaded.c_compiler == "clang"
        assert loaded.cpp_compiler == "clang++"
        assert loaded.auto_compile is False

    def test_save_creates_valid_json(self, manager):
        """Test saved configuration is valid JSON."""
        config = GrammarConfig(c_compiler="gcc")
        manager.save(config)

        with open(manager.config_file, 'r') as f:
            data = json.load(f)

        assert data["c_compiler"] == "gcc"

    def test_load_corrupted_file(self, manager):
        """Test loading corrupted configuration file raises error."""
        # Write invalid JSON
        with open(manager.config_file, 'w') as f:
            f.write("{invalid json")

        with pytest.raises(ValueError, match="corrupted"):
            manager.load()

    def test_update_config(self, manager):
        """Test updating configuration."""
        updated = manager.update(c_compiler="clang", parallel_builds=4)

        assert updated.c_compiler == "clang"
        assert updated.parallel_builds == 4

        # Verify it was saved
        loaded = manager.load()
        assert loaded.c_compiler == "clang"
        assert loaded.parallel_builds == 4

    def test_update_unknown_field_warns(self, manager):
        """Test updating unknown field logs warning."""
        with patch('src.python.common.core.grammar_config.logger') as mock_logger:
            manager.update(unknown_field="value")
            mock_logger.warning.assert_called_once()

    def test_add_grammar_directory(self, manager):
        """Test adding grammar directory."""
        config = manager.add_grammar_directory("/path/to/grammars")

        assert "/path/to/grammars" in config.grammar_directories

        # Verify it was saved
        loaded = manager.load()
        assert "/path/to/grammars" in loaded.grammar_directories

    def test_add_duplicate_directory(self, manager):
        """Test adding duplicate directory doesn't create duplicates."""
        manager.add_grammar_directory("/path/to/grammars")
        config = manager.add_grammar_directory("/path/to/grammars")

        # Should only appear once
        assert config.grammar_directories.count("/path/to/grammars") == 1

    def test_remove_grammar_directory(self, manager):
        """Test removing grammar directory."""
        manager.add_grammar_directory("/path/to/grammars")
        config = manager.remove_grammar_directory("/path/to/grammars")

        assert "/path/to/grammars" not in config.grammar_directories

    def test_remove_nonexistent_directory(self, manager):
        """Test removing non-existent directory logs warning."""
        with patch('src.python.common.core.grammar_config.logger') as mock_logger:
            manager.remove_grammar_directory("/nonexistent")
            mock_logger.warning.assert_called_once()

    def test_set_compiler(self, manager):
        """Test setting compilers."""
        config = manager.set_compiler(c_compiler="gcc", cpp_compiler="g++")

        assert config.c_compiler == "gcc"
        assert config.cpp_compiler == "g++"

    def test_set_only_c_compiler(self, manager):
        """Test setting only C compiler."""
        config = manager.set_compiler(c_compiler="clang")

        assert config.c_compiler == "clang"
        assert config.cpp_compiler is None

    def test_set_only_cpp_compiler(self, manager):
        """Test setting only C++ compiler."""
        config = manager.set_compiler(cpp_compiler="clang++")

        assert config.c_compiler is None
        assert config.cpp_compiler == "clang++"

    def test_reset_to_defaults(self, manager):
        """Test resetting configuration to defaults."""
        # Modify config
        manager.update(c_compiler="clang", parallel_builds=4)

        # Reset
        config = manager.reset_to_defaults()

        assert config.c_compiler is None
        assert config.parallel_builds == 1

    def test_get_installation_directory_default(self, manager):
        """Test getting default installation directory."""
        install_dir = manager.get_installation_directory()

        assert install_dir == manager.config_dir / "grammars"

    def test_get_installation_directory_custom(self, manager):
        """Test getting custom installation directory."""
        manager.update(installation_directory="/custom/path")

        install_dir = manager.get_installation_directory()

        assert install_dir == Path("/custom/path")

    def test_get_grammar_search_paths_empty(self, manager):
        """Test getting search paths when none configured."""
        paths = manager.get_grammar_search_paths()

        # Should return installation directory if it exists
        # For this test, it won't exist yet
        assert isinstance(paths, list)

    def test_get_grammar_search_paths_with_directories(self, manager, temp_config_dir):
        """Test getting search paths with configured directories."""
        # Create some test directories
        dir1 = temp_config_dir / "grammars1"
        dir2 = temp_config_dir / "grammars2"
        dir1.mkdir()
        dir2.mkdir()

        manager.add_grammar_directory(str(dir1))
        manager.add_grammar_directory(str(dir2))

        paths = manager.get_grammar_search_paths()

        assert dir1 in paths
        assert dir2 in paths

    def test_get_grammar_search_paths_filters_nonexistent(self, manager):
        """Test search paths filters non-existent directories."""
        manager.add_grammar_directory("/nonexistent/path")

        with patch('src.python.common.core.grammar_config.logger') as mock_logger:
            paths = manager.get_grammar_search_paths()
            # Should log warning for non-existent directory
            mock_logger.warning.assert_called()

    def test_get_compiler_flags_none(self, manager):
        """Test getting compiler flags when none set."""
        flags = manager.get_compiler_flags("gcc")

        assert flags == []

    def test_get_compiler_flags_with_values(self, manager):
        """Test getting compiler flags with configured values."""
        manager.set_compiler_flags("gcc", ["-Wall", "-Werror"])

        flags = manager.get_compiler_flags("gcc")

        assert flags == ["-Wall", "-Werror"]

    def test_set_compiler_flags(self, manager):
        """Test setting compiler flags."""
        config = manager.set_compiler_flags("clang", ["-Wextra", "-O3"])

        assert config.custom_compiler_flags["clang"] == ["-Wextra", "-O3"]

    def test_export_config(self, manager, temp_config_dir):
        """Test exporting configuration."""
        manager.update(c_compiler="gcc", parallel_builds=2)

        export_file = temp_config_dir / "exported.json"
        manager.export_config(export_file)

        assert export_file.exists()

        with open(export_file, 'r') as f:
            data = json.load(f)

        assert data["c_compiler"] == "gcc"
        assert data["parallel_builds"] == 2

    def test_import_config(self, manager, temp_config_dir):
        """Test importing configuration."""
        # Create import file
        import_file = temp_config_dir / "import.json"
        data = {
            "c_compiler": "clang",
            "cpp_compiler": "clang++",
            "parallel_builds": 4
        }

        with open(import_file, 'w') as f:
            json.dump(data, f)

        config = manager.import_config(import_file)

        assert config.c_compiler == "clang"
        assert config.cpp_compiler == "clang++"
        assert config.parallel_builds == 4

    def test_import_invalid_config(self, manager, temp_config_dir):
        """Test importing invalid configuration raises error."""
        import_file = temp_config_dir / "invalid.json"

        with open(import_file, 'w') as f:
            f.write("{invalid json")

        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.import_config(import_file)

    def test_thread_safety(self, manager):
        """Test configuration operations are thread-safe."""
        results = []

        def update_config(value):
            config = manager.update(parallel_builds=value)
            results.append(config.parallel_builds)

        # Create multiple threads updating config
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_config, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All updates should have succeeded
        assert len(results) == 10

        # Final value should be one of the update values
        final_config = manager.load()
        assert final_config.parallel_builds in range(10)

    def test_atomic_write(self, manager):
        """Test configuration writes are atomic."""
        # Save initial config
        manager.update(c_compiler="gcc")

        # Simulate failure during write by patching rename
        original_rename = Path.replace

        def failing_rename(self, target):
            raise IOError("Simulated failure")

        with patch.object(Path, 'replace', failing_rename):
            with pytest.raises(IOError):
                manager.update(c_compiler="clang")

        # Original config should still be intact
        config = manager.load()
        assert config.c_compiler == "gcc"

    def test_default_config_creation(self, temp_config_dir):
        """Test default config is created on first initialization."""
        config_file = temp_config_dir / "config.json"
        assert not config_file.exists()

        manager = ConfigManager(config_dir=temp_config_dir)

        assert config_file.exists()

        config = manager.load()
        assert isinstance(config, GrammarConfig)

    def test_config_persistence(self, manager):
        """Test configuration persists across manager instances."""
        manager.update(c_compiler="clang", parallel_builds=4)

        # Create new manager instance
        new_manager = ConfigManager(config_dir=manager.config_dir)
        config = new_manager.load()

        assert config.c_compiler == "clang"
        assert config.parallel_builds == 4
