"""
Configuration Management System for Tree-sitter Grammars.

This module provides functionality to manage tree-sitter configuration including
custom grammar locations, compiler preferences, and installation settings.

Key features:
- Configuration file management (~/.config/tree-sitter/config.json)
- Custom grammar installation directories
- Compiler preferences (C and C++)
- Installation settings and defaults
- Thread-safe file operations
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class GrammarConfig:
    """Configuration for tree-sitter grammar management."""

    grammar_directories: List[str] = field(default_factory=lambda: [])
    """Custom directories to search for grammars"""

    installation_directory: Optional[str] = None
    """Directory for installing new grammars (defaults to ~/.config/tree-sitter/grammars)"""

    c_compiler: Optional[str] = None
    """Preferred C compiler (e.g., 'gcc', 'clang')"""

    cpp_compiler: Optional[str] = None
    """Preferred C++ compiler (e.g., 'g++', 'clang++')"""

    auto_compile: bool = True
    """Automatically compile grammars after installation"""

    parallel_builds: int = 1
    """Number of parallel compilation jobs"""

    optimization_level: str = "O2"
    """Compiler optimization level (O0, O1, O2, O3)"""

    keep_build_artifacts: bool = False
    """Keep intermediate build files after compilation"""

    default_clone_depth: Optional[int] = None
    """Default git clone depth (None for full clone)"""

    custom_compiler_flags: Dict[str, List[str]] = field(default_factory=dict)
    """Custom compiler flags by compiler name"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrammarConfig":
        """Create configuration from dictionary."""
        # Filter only known fields to handle future additions gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


class ConfigManager:
    """
    Manages tree-sitter configuration file.

    Handles loading, saving, and updating configuration with
    thread-safe operations and atomic file writes.
    """

    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "tree-sitter"
    DEFAULT_CONFIG_FILE = "config.json"

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Configuration directory (defaults to ~/.config/tree-sitter)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_file = self.config_dir / self.DEFAULT_CONFIG_FILE

        # Thread safety
        self._lock = threading.Lock()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize config if needed
        if not self.config_file.exists():
            self._create_default_config()

    def load(self) -> GrammarConfig:
        """
        Load configuration from file.

        Returns:
            GrammarConfig instance

        Raises:
            ValueError: If configuration file is corrupted
        """
        with self._lock:
            try:
                if not self.config_file.exists():
                    return GrammarConfig()

                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                config = GrammarConfig.from_dict(data)
                logger.debug(f"Loaded configuration from {self.config_file}")
                return config

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse configuration file: {e}")
                raise ValueError(f"Configuration file is corrupted: {e}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

    def save(self, config: GrammarConfig) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save

        Raises:
            IOError: If file cannot be written
        """
        with self._lock:
            try:
                # Write to temporary file first (atomic write)
                temp_file = self.config_file.with_suffix('.tmp')

                with open(temp_file, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)

                # Atomic rename
                temp_file.replace(self.config_file)

                logger.info(f"Saved configuration to {self.config_file}")

            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise IOError(f"Failed to save configuration: {e}")

    def update(self, **kwargs) -> GrammarConfig:
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration fields to update

        Returns:
            Updated GrammarConfig instance

        Example:
            >>> manager = ConfigManager()
            >>> config = manager.update(c_compiler='clang', auto_compile=True)
        """
        config = self.load()

        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration field: {key}")

        self.save(config)
        return config

    def add_grammar_directory(self, directory: str) -> GrammarConfig:
        """
        Add a custom grammar directory to search path.

        Args:
            directory: Path to grammar directory

        Returns:
            Updated configuration
        """
        config = self.load()

        if directory not in config.grammar_directories:
            config.grammar_directories.append(directory)
            self.save(config)
            logger.info(f"Added grammar directory: {directory}")
        else:
            logger.debug(f"Grammar directory already exists: {directory}")

        return config

    def remove_grammar_directory(self, directory: str) -> GrammarConfig:
        """
        Remove a grammar directory from search path.

        Args:
            directory: Path to grammar directory

        Returns:
            Updated configuration
        """
        config = self.load()

        if directory in config.grammar_directories:
            config.grammar_directories.remove(directory)
            self.save(config)
            logger.info(f"Removed grammar directory: {directory}")
        else:
            logger.warning(f"Grammar directory not found: {directory}")

        return config

    def set_compiler(
        self,
        c_compiler: Optional[str] = None,
        cpp_compiler: Optional[str] = None
    ) -> GrammarConfig:
        """
        Set preferred compilers.

        Args:
            c_compiler: C compiler to use
            cpp_compiler: C++ compiler to use

        Returns:
            Updated configuration
        """
        updates = {}
        if c_compiler is not None:
            updates['c_compiler'] = c_compiler
        if cpp_compiler is not None:
            updates['cpp_compiler'] = cpp_compiler

        return self.update(**updates)

    def reset_to_defaults(self) -> GrammarConfig:
        """
        Reset configuration to defaults.

        Returns:
            Default configuration
        """
        config = GrammarConfig()
        self.save(config)
        logger.info("Reset configuration to defaults")
        return config

    def get_installation_directory(self) -> Path:
        """
        Get the grammar installation directory.

        Returns:
            Path to installation directory
        """
        config = self.load()

        if config.installation_directory:
            return Path(config.installation_directory)

        # Default location
        return self.config_dir / "grammars"

    def get_grammar_search_paths(self) -> List[Path]:
        """
        Get all grammar search paths.

        Returns:
            List of paths to search for grammars
        """
        config = self.load()
        paths = []

        # Add configured directories
        for dir_str in config.grammar_directories:
            path = Path(dir_str).expanduser()
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"Grammar directory does not exist: {path}")

        # Add installation directory
        install_dir = self.get_installation_directory()
        if install_dir.exists() and install_dir not in paths:
            paths.append(install_dir)

        return paths

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        config = GrammarConfig()
        self.save(config)
        logger.info(f"Created default configuration at {self.config_file}")

    def get_compiler_flags(self, compiler_name: str) -> List[str]:
        """
        Get custom compiler flags for a specific compiler.

        Args:
            compiler_name: Name of the compiler

        Returns:
            List of custom flags
        """
        config = self.load()
        return config.custom_compiler_flags.get(compiler_name, [])

    def set_compiler_flags(self, compiler_name: str, flags: List[str]) -> GrammarConfig:
        """
        Set custom compiler flags.

        Args:
            compiler_name: Name of the compiler
            flags: List of compiler flags

        Returns:
            Updated configuration
        """
        config = self.load()
        config.custom_compiler_flags[compiler_name] = flags
        self.save(config)
        logger.info(f"Set compiler flags for {compiler_name}: {flags}")
        return config

    def export_config(self, output_file: Path) -> None:
        """
        Export configuration to a file.

        Args:
            output_file: Path to export file
        """
        config = self.load()
        with open(output_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Exported configuration to {output_file}")

    def import_config(self, input_file: Path) -> GrammarConfig:
        """
        Import configuration from a file.

        Args:
            input_file: Path to import file

        Returns:
            Imported configuration

        Raises:
            ValueError: If import file is invalid
        """
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)

            config = GrammarConfig.from_dict(data)
            self.save(config)
            logger.info(f"Imported configuration from {input_file}")
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to import configuration: {e}")


# Export main classes
__all__ = ["GrammarConfig", "ConfigManager"]
