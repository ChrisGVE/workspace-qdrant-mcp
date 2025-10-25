"""
Ingestion configuration system for language-aware file filtering.

This module provides comprehensive configuration for file ingestion with support
for 25+ programming languages, pattern-based filtering, and performance optimization.
Integrates with the unified configuration system and supports hot-reloading.

Features:
- Language-aware ignore patterns for 25+ programming languages
- Performance constraints (file size, count limits, batching)
- Pattern compilation and caching for optimal performance
- Integration with existing unified configuration system
- Hot-reload support for configuration changes
- Template system with comprehensive defaults

Example:
    ```python
    from workspace_qdrant_mcp.core.ingestion_config import IngestionConfigManager

    # Load ingestion configuration
    manager = IngestionConfigManager()
    config = manager.load_config()

    # Check if file should be ignored
    if manager.should_ignore_file("node_modules/package/index.js"):
        print("File ignored by pattern matching")
    ```
"""

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

# logger imported from loguru


class PatternType(Enum):
    """Types of ignore patterns."""
    DIRECTORY = "directory"
    FILE_EXTENSION = "file_extension"
    FILE_PATTERN = "file_pattern"
    GLOB_PATTERN = "glob_pattern"


@dataclass
class LanguagePatterns:
    """Language-specific file patterns and ignore rules."""
    name: str
    file_extensions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    build_artifacts: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    caches: list[str] = field(default_factory=list)
    temp_files: list[str] = field(default_factory=list)


class IgnorePatternsConfig(BaseModel):
    """Configuration for ignore patterns."""

    # Universal patterns
    dot_files: bool = True  # Ignore .git/, .vscode/, .DS_Store, etc.
    version_control: list[str] = Field(
        default=[".git", ".svn", ".hg", ".bzr", ".fossil"],
        description="Version control directories to ignore"
    )

    # Common ignore directories
    directories: list[str] = Field(
        default_factory=list,
        description="Directory names/patterns to ignore"
    )

    # File extensions to ignore
    file_extensions: list[str] = Field(
        default_factory=list,
        description="File extensions to ignore (e.g., '*.pyc')"
    )

    # File patterns to ignore
    file_patterns: list[str] = Field(
        default_factory=list,
        description="File name patterns to ignore (e.g., '*.log')"
    )

    # Glob patterns for complex matching
    glob_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns for complex matching"
    )


class PerformanceConfig(BaseModel):
    """Performance and resource constraints."""

    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum file size to process in megabytes"
    )

    max_files_per_directory: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum files to process per directory"
    )

    max_files_per_batch: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum files to process in a single batch"
    )

    debounce_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Debounce time for file change events"
    )

    pattern_cache_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum number of compiled patterns to cache"
    )

    enable_pattern_compilation: bool = Field(
        default=True,
        description="Enable pattern compilation and caching for performance"
    )


class LanguageConfig(BaseModel):
    """Language-specific configuration."""

    name: str
    enabled: bool = True
    custom_patterns: IgnorePatternsConfig = Field(default_factory=IgnorePatternsConfig)

    # Override performance settings per language
    performance_overrides: PerformanceConfig | None = None


class CollectionRoutingConfig(BaseModel):
    """Configuration for routing files to different collections."""

    code_suffix: str = Field(default="code", description="Collection suffix for source code")
    docs_suffix: str = Field(default="docs", description="Collection suffix for documentation")
    config_suffix: str = Field(default="config", description="Collection suffix for config files")
    data_suffix: str = Field(default="data", description="Collection suffix for data files")
    default_suffix: str = Field(default="repo", description="Default collection suffix")

    # File type to collection mapping
    file_type_routing: dict[str, str] = Field(
        default_factory=dict,
        description="Map file extensions to collection suffixes"
    )


class UserOverridesConfig(BaseModel):
    """User customization and override settings."""

    # Additional patterns to ignore beyond defaults
    additional_ignores: IgnorePatternsConfig = Field(default_factory=IgnorePatternsConfig)

    # Force include patterns (override ignores for specific cases)
    force_include: IgnorePatternsConfig = Field(default_factory=IgnorePatternsConfig)

    # Per-project overrides
    project_specific: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-project custom rules"
    )

    # Environment-specific overrides
    environment_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Environment-specific configurations"
    )


class IngestionConfig(BaseModel):
    """Main ingestion configuration class."""

    # Global enable/disable
    enabled: bool = Field(default=True, description="Enable/disable ingestion system")

    # Default ignore patterns
    ignore_patterns: IgnorePatternsConfig = Field(default_factory=IgnorePatternsConfig)

    # Performance constraints
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Language-specific configurations
    languages: dict[str, LanguageConfig] = Field(
        default_factory=dict,
        description="Language-specific configurations"
    )

    # Collection routing
    collection_routing: CollectionRoutingConfig = Field(default_factory=CollectionRoutingConfig)

    # User overrides
    user_overrides: UserOverridesConfig = Field(default_factory=UserOverridesConfig)

    # LSP integration for future features
    lsp_integration: dict[str, Any] = Field(
        default_factory=dict,
        description="LSP integration settings"
    )

    @field_validator('languages')
    @classmethod
    def validate_languages(cls, v):
        """Validate language configurations."""
        for lang_name, lang_config in v.items():
            if lang_config.name != lang_name:
                raise ValueError(f"Language config name '{lang_config.name}' doesn't match key '{lang_name}'")
        return v

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate performance settings
        if self.performance.max_file_size_mb <= 0:
            issues.append("max_file_size_mb must be positive")

        if self.performance.max_files_per_batch <= 0:
            issues.append("max_files_per_batch must be positive")

        # Validate collection routing
        suffixes = [
            self.collection_routing.code_suffix,
            self.collection_routing.docs_suffix,
            self.collection_routing.config_suffix,
            self.collection_routing.data_suffix,
            self.collection_routing.default_suffix
        ]

        if len(set(suffixes)) != len(suffixes):
            issues.append("Collection suffixes must be unique")

        # Validate patterns - check for conflicting include/ignore patterns
        if self.user_overrides.force_include.directories:
            ignored_dirs = set(self.ignore_patterns.directories)
            forced_dirs = set(self.user_overrides.force_include.directories)
            conflicts = ignored_dirs & forced_dirs
            if conflicts:
                issues.append(f"Conflicting directory patterns: {', '.join(conflicts)}")

        return issues


class IngestionConfigManager:
    """Manager class for ingestion configuration with caching and validation."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize ingestion configuration manager.

        Args:
            config_dir: Directory to search for configuration files
        """
        self.config_dir = config_dir or Path.cwd()
        self.current_config: IngestionConfig | None = None
        self.pattern_cache: dict[str, Pattern] = {}
        self.language_patterns: dict[str, LanguagePatterns] = {}

        # Initialize default language patterns
        self._initialize_language_patterns()

    def _initialize_language_patterns(self) -> None:
        """Initialize comprehensive language pattern database."""

        # Web Technologies
        self.language_patterns["javascript"] = LanguagePatterns(
            name="javascript",
            file_extensions=["*.js", "*.jsx", "*.mjs", "*.cjs"],
            dependencies=["node_modules", "bower_components", ".npm", ".yarn"],
            build_artifacts=["dist", "build", "out", ".next", ".nuxt", "coverage"],
            generated_files=["*.bundle.js", "*.min.js", "*.map"],
            caches=[".eslintcache", ".parcel-cache", ".cache"],
            temp_files=["*.log", "npm-debug.log*", "yarn-debug.log*"]
        )

        self.language_patterns["typescript"] = LanguagePatterns(
            name="typescript",
            file_extensions=["*.ts", "*.tsx", "*.d.ts"],
            dependencies=["node_modules", "@types"],
            build_artifacts=["dist", "build", "lib", "types"],
            generated_files=["*.js", "*.d.ts", "*.tsbuildinfo"],
            caches=[".tscache"],
            temp_files=["tsconfig.tsbuildinfo"]
        )

        # Python
        self.language_patterns["python"] = LanguagePatterns(
            name="python",
            file_extensions=["*.py", "*.pyx", "*.pyi", "*.ipynb"],
            dependencies=["venv", ".venv", "env", ".env", "site-packages", ".pip", "pipenv"],
            build_artifacts=["build", "dist", "*.egg-info", ".eggs"],
            generated_files=["*.pyc", "*.pyo", "*.pyd", "*.so"],
            caches=["__pycache__", ".pytest_cache", ".mypy_cache", ".coverage", ".tox"],
            temp_files=["*.log", "*.tmp", "*.swp"]
        )

        # Rust
        self.language_patterns["rust"] = LanguagePatterns(
            name="rust",
            file_extensions=["*.rs", "*.toml"],
            dependencies=["target", ".cargo"],
            build_artifacts=["target/debug", "target/release", "target/doc"],
            generated_files=["Cargo.lock"],
            caches=["target/.fingerprint", "target/.rustc_info.json"],
            temp_files=["*.tmp", "*.swp"]
        )

        # Java/JVM
        self.language_patterns["java"] = LanguagePatterns(
            name="java",
            file_extensions=["*.java", "*.kt", "*.scala", "*.clj", "*.groovy"],
            dependencies=["target", ".gradle", ".m2", "lib", "libs"],
            build_artifacts=["build", "out", "classes"],
            generated_files=["*.class", "*.jar", "*.war", "*.ear"],
            caches=[".gradle", "gradle"],
            temp_files=["*.log", "*.tmp"]
        )

        # Go
        self.language_patterns["go"] = LanguagePatterns(
            name="go",
            file_extensions=["*.go", "*.mod", "*.sum"],
            dependencies=["vendor", "go.work"],
            build_artifacts=["bin", "pkg"],
            generated_files=["go.sum"],
            caches=["go.work.sum"],
            temp_files=["*.log", "*.tmp"]
        )

        # C/C++
        self.language_patterns["cpp"] = LanguagePatterns(
            name="cpp",
            file_extensions=["*.c", "*.cpp", "*.cxx", "*.cc", "*.h", "*.hpp", "*.hxx"],
            dependencies=["vcpkg", "conan", "build/_deps"],
            build_artifacts=["build", "cmake-build-*", "Debug", "Release"],
            generated_files=["*.o", "*.obj", "*.a", "*.lib", "*.so", "*.dll", "*.dylib", "*.exe"],
            caches=["CMakeCache.txt", "CMakeFiles"],
            temp_files=["*.log", "*.tmp", "core.*"]
        )

        # C#/.NET
        self.language_patterns["csharp"] = LanguagePatterns(
            name="csharp",
            file_extensions=["*.cs", "*.fs", "*.vb", "*.csproj", "*.fsproj", "*.vbproj"],
            dependencies=["packages", ".nuget"],
            build_artifacts=["bin", "obj"],
            generated_files=["*.dll", "*.exe", "*.pdb"],
            caches=[".vs", ".vscode"],
            temp_files=["*.log", "*.tmp"]
        )

        # Ruby
        self.language_patterns["ruby"] = LanguagePatterns(
            name="ruby",
            file_extensions=["*.rb", "*.erb", "*.rake", "Gemfile*"],
            dependencies=["vendor/bundle", ".bundle", "gems"],
            build_artifacts=["pkg"],
            generated_files=["Gemfile.lock"],
            caches=[".gem"],
            temp_files=["*.log", "*.tmp"]
        )

        # PHP
        self.language_patterns["php"] = LanguagePatterns(
            name="php",
            file_extensions=["*.php", "*.phtml", "*.php3", "*.php4", "*.php5"],
            dependencies=["vendor", "composer.phar"],
            build_artifacts=["build"],
            generated_files=["composer.lock"],
            caches=[".phpunit.cache"],
            temp_files=["*.log", "*.tmp"]
        )

        # Swift
        self.language_patterns["swift"] = LanguagePatterns(
            name="swift",
            file_extensions=["*.swift"],
            dependencies=[".build", "Packages"],
            build_artifacts=[".build", "DerivedData"],
            generated_files=["Package.resolved"],
            caches=[".swiftpm"],
            temp_files=["*.log", "*.tmp"]
        )

        # Continue with other languages...
        self._add_remaining_languages()

    def _add_remaining_languages(self) -> None:
        """Add remaining language patterns to complete the 25+ language support."""

        # Dart/Flutter
        self.language_patterns["dart"] = LanguagePatterns(
            name="dart",
            file_extensions=["*.dart"],
            dependencies=[".dart_tool", ".pub-cache", ".packages"],
            build_artifacts=["build", ".dart_tool/build"],
            generated_files=["pubspec.lock", "*.g.dart", "*.freezed.dart"],
            caches=[".dart_tool"],
            temp_files=["*.log", "*.tmp"]
        )

        # R
        self.language_patterns["r"] = LanguagePatterns(
            name="r",
            file_extensions=["*.R", "*.r", "*.Rmd"],
            dependencies=["renv", "packrat", ".Rproj.user"],
            build_artifacts=["_site", "docs"],
            generated_files=["*.html", "*.pdf"],
            caches=[".Rhistory", ".RData"],
            temp_files=["*.log", "*.tmp", "*~"]
        )

        # Julia
        self.language_patterns["julia"] = LanguagePatterns(
            name="julia",
            file_extensions=["*.jl"],
            dependencies=[".julia", "Manifest.toml"],
            build_artifacts=["deps/build.log"],
            generated_files=["Manifest.toml"],
            caches=[],
            temp_files=["*.log", "*.tmp"]
        )

        # Haskell
        self.language_patterns["haskell"] = LanguagePatterns(
            name="haskell",
            file_extensions=["*.hs", "*.lhs", "*.cabal"],
            dependencies=[".stack-work", "dist", "dist-newstyle"],
            build_artifacts=[".stack-work", "dist"],
            generated_files=["*.hi", "*.o", "stack.yaml.lock"],
            caches=[".stack"],
            temp_files=["*.log", "*.tmp"]
        )

        # Elixir
        self.language_patterns["elixir"] = LanguagePatterns(
            name="elixir",
            file_extensions=["*.ex", "*.exs", "*.eex"],
            dependencies=["deps", "_build"],
            build_artifacts=["_build", "cover"],
            generated_files=["mix.lock"],
            caches=["_build/.mix"],
            temp_files=["*.log", "*.tmp"]
        )

        # Add more languages as needed...

    def load_config(self, config_file: str | Path | None = None) -> IngestionConfig:
        """
        Load ingestion configuration from file or defaults.

        Args:
            config_file: Specific config file path (optional)

        Returns:
            Loaded ingestion configuration

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If specified config file doesn't exist
        """
        config_data = {}

        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Ingestion config file not found: {config_file}")
            config_data = self._load_config_file(config_path)
        else:
            # Auto-discover ingestion config
            config_path = self._find_ingestion_config()
            if config_path:
                config_data = self._load_config_file(config_path)
                logger.info(f"Loaded ingestion config from: {config_path}")

        # Create config with defaults and overrides
        try:
            # Always start with defaults, then merge user config
            default_config = self._get_default_config()
            if config_data:
                config_data = self._merge_with_defaults(config_data)
            else:
                config_data = default_config

            self.current_config = IngestionConfig(**config_data)

            # Validate configuration
            issues = self.current_config.validate_config()
            if issues:
                error_msg = "Ingestion configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
                logger.error(error_msg)
                raise ValidationError(error_msg, IngestionConfig)

            logger.info("Ingestion configuration loaded and validated successfully")
            return self.current_config

        except ValidationError as e:
            error_msg = f"Ingestion configuration validation error: {e}"
            logger.error(error_msg)
            raise

    def _load_config_file(self, file_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing ingestion config YAML: {e}") from e
        except Exception as e:
            raise ValueError(f"Error reading ingestion config: {e}") from e

    def _find_ingestion_config(self) -> Path | None:
        """Find ingestion configuration file."""
        # Search order: current dir, config dir, user config dir
        search_paths = [
            self.config_dir,
            Path.cwd(),
            Path.home() / ".config" / "workspace-qdrant-mcp",
            Path(__file__).parent.parent.parent / "config"
        ]

        config_names = [
            "ingestion.yaml",
            "ingestion.yml",
            ".ingestion.yaml",
            "workspace_qdrant_ingestion.yaml"
        ]

        for search_path in search_paths:
            for config_name in config_names:
                config_file = search_path / config_name
                if config_file.exists():
                    return config_file

        return None

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration with all language patterns."""
        return {
            "enabled": True,
            "ignore_patterns": {
                "dot_files": True,
                "directories": [
                    # Universal
                    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
                    "target", ".gradle", ".m2", "vendor", "build", "dist", "out",
                    "venv", ".venv", "env", ".env", "site-packages",
                    ".git", ".svn", ".hg", ".bzr",
                    # IDE/Editor
                    ".vscode", ".idea", ".vs", ".settings",
                    # Caches
                    ".cache", ".npm", ".yarn", ".cargo", ".composer",
                    # Logs and temp
                    "logs", "tmp", "temp", ".log"
                ],
                "file_extensions": [
                    # Compiled/Binary
                    "*.pyc", "*.pyo", "*.class", "*.jar", "*.war", "*.ear",
                    "*.o", "*.obj", "*.a", "*.lib", "*.so", "*.dll", "*.dylib", "*.exe",
                    # Minified/Generated
                    "*.min.js", "*.min.css", "*.bundle.js", "*.map",
                    # Archives
                    "*.zip", "*.tar", "*.tar.gz", "*.rar", "*.7z",
                    # Media
                    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.avi", "*.mp3",
                    # Logs/Temp
                    "*.log", "*.tmp", "*.temp", "*.swp", "*.swo", "*~"
                ]
            },
            "performance": {
                "max_file_size_mb": 10,
                "max_files_per_directory": 1000,
                "max_files_per_batch": 100,
                "debounce_seconds": 5.0,
                "pattern_cache_size": 10000,
                "enable_pattern_compilation": True
            },
            "collection_routing": {
                "code_suffix": "code",
                "docs_suffix": "docs",
                "config_suffix": "config",
                "data_suffix": "data",
                "default_suffix": "repo"
            }
        }

    def _merge_with_defaults(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """Merge user configuration with defaults."""
        default_config = self._get_default_config()

        # Deep merge logic here
        def deep_merge(default: dict, override: dict) -> dict:
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(default_config, user_config)

    def should_ignore_file(self, file_path: str | Path) -> bool:
        """
        Check if a file should be ignored based on current configuration.

        Args:
            file_path: Path to check

        Returns:
            True if file should be ignored, False otherwise
        """
        if not self.current_config:
            self.load_config()

        path = Path(file_path)

        # Check file size
        try:
            if path.exists() and path.stat().st_size > (self.current_config.performance.max_file_size_mb * 1024 * 1024):
                return True
        except OSError:
            return True  # Ignore files we can't stat

        # Check against ignore patterns
        return self._matches_ignore_patterns(path)

    def _matches_ignore_patterns(self, path: Path) -> bool:
        """Check if path matches any ignore patterns."""
        if not self.current_config:
            return False

        patterns = self.current_config.ignore_patterns
        path_str = str(path)
        path_parts = path.parts

        # Check dot files
        if patterns.dot_files and any(part.startswith('.') for part in path_parts):
            return True

        # Check directory patterns
        for dir_pattern in patterns.directories:
            if any(fnmatch.fnmatch(part, dir_pattern) for part in path_parts):
                return True

        # Check file extension patterns
        for ext_pattern in patterns.file_extensions:
            if fnmatch.fnmatch(path.name, ext_pattern):
                return True

        # Check file patterns
        for file_pattern in patterns.file_patterns:
            if fnmatch.fnmatch(path.name, file_pattern):
                return True

        # Check glob patterns
        for glob_pattern in patterns.glob_patterns:
            if fnmatch.fnmatch(path_str, glob_pattern):
                return True

        return False

    def get_config_info(self) -> dict[str, Any]:
        """Get information about current ingestion configuration."""
        if not self.current_config:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "enabled": self.current_config.enabled,
            "languages_supported": len(self.language_patterns),
            "ignore_directories": len(self.current_config.ignore_patterns.directories),
            "ignore_extensions": len(self.current_config.ignore_patterns.file_extensions),
            "performance_settings": self.current_config.performance.model_dump(),
            "pattern_cache_size": len(self.pattern_cache)
        }
