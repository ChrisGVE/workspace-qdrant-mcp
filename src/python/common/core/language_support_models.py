"""Pydantic models for language support configuration.

This module defines the data models for parsing and validating language_support.yaml
configuration files. These models represent the complete schema for language support
including LSP servers, tree-sitter grammars, file extensions, and content signatures.

Models:
    ProjectIndicators: Version control and language ecosystem indicators
    LSPServerConfig: LSP server configuration with features and install notes
    TreeSitterGrammars: Available tree-sitter grammar list
    ContentSignatures: Shebang and keyword patterns for language detection
    BuildSystem: Build system configuration
    MetadataSchema: Metadata schema definitions
    LanguageSupportConfig: Root configuration model for comprehensive asset file

    LSPDefinition: LSP configuration for database schema (v4 format)
    TreeSitterDefinition: Tree-sitter configuration for database schema (v4 format)
    LanguageDefinition: Language definition for database schema (v4 format)
    LanguageSupportDatabaseConfig: Root configuration for database loader (v4 format)

Example:
    >>> from pathlib import Path
    >>> config = LanguageSupportConfig.from_yaml(Path("language_support.yaml"))
    >>> print(config.file_extensions[".py"])
    'python'

    >>> # V4 database schema format
    >>> db_config = LanguageSupportDatabaseConfig.from_yaml(Path("language_support_v4.yaml"))
    >>> print(db_config.languages[0].name)
    'python'
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# COMPREHENSIVE ASSET FILE MODELS (Original/Existing)
# =============================================================================


class ProjectIndicators(BaseModel):
    """Project detection indicators for identifying project roots.

    Attributes:
        version_control: Version control system directories (.git, .hg, etc.)
        language_ecosystems: Language-specific project files (package.json, Cargo.toml, etc.)
    """

    version_control: list[str] = Field(
        default_factory=list,
        description="Version control directories that indicate project roots",
    )
    language_ecosystems: list[str] = Field(
        default_factory=list,
        description="Language ecosystem files that indicate project roots",
    )


class LSPServerConfig(BaseModel):
    """Language Server Protocol server configuration.

    Attributes:
        primary: Primary LSP server name/command
        features: List of supported LSP features (symbols, completion, etc.)
        rationale: Why this LSP server was chosen
        install_notes: Installation instructions
    """

    primary: str = Field(..., description="Primary LSP server name")
    features: list[str] = Field(
        default_factory=list, description="Supported LSP features"
    )
    rationale: str = Field(default="", description="LSP server selection rationale")
    install_notes: str = Field(default="", description="Installation instructions")


class TreeSitterGrammars(BaseModel):
    """Tree-sitter grammar availability.

    Attributes:
        available: List of language identifiers with available tree-sitter grammars
    """

    available: list[str] = Field(
        default_factory=list,
        description="Languages with available tree-sitter grammars",
    )


class ContentSignatures(BaseModel):
    """Content-based language detection signatures.

    Attributes:
        shebangs: Shebang patterns mapping to language identifiers
        keyword_patterns: Language-specific keyword patterns for detection
    """

    shebangs: dict[str, str] = Field(
        default_factory=dict,
        description="Shebang patterns to language identifier mapping",
    )
    keyword_patterns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Language keyword patterns for content detection",
    )

    @field_validator("shebangs")
    @classmethod
    def validate_shebangs(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate shebang patterns start with #!.

        Args:
            v: Shebang dictionary to validate

        Returns:
            Validated shebang dictionary

        Raises:
            ValueError: If shebang pattern doesn't start with #!
        """
        for shebang in v.keys():
            if not shebang.startswith("#!"):
                raise ValueError(f"Shebang pattern must start with #!: {shebang}")
        return v


class BuildSystem(BaseModel):
    """Build system configuration.

    Attributes:
        files: Configuration file patterns for detecting build system
        language: Associated programming language
        commands: List of build commands
    """

    files: list[str] = Field(
        default_factory=list, description="Build system configuration files"
    )
    language: str = Field(default="", description="Associated programming language")
    commands: list[str] = Field(
        default_factory=list, description="Build system commands"
    )


class MetadataSchema(BaseModel):
    """Metadata schema definitions.

    Attributes:
        required: Required metadata fields
        optional: Optional metadata fields
    """

    required: list[str] = Field(
        default_factory=list, description="Required metadata fields"
    )
    optional: list[str] = Field(
        default_factory=list, description="Optional metadata fields"
    )


class LanguageSupportConfig(BaseModel):
    """Root configuration model for language support.

    This model represents the complete language_support.yaml configuration
    including all language definitions, LSP servers, tree-sitter grammars,
    and detection patterns.

    Attributes:
        project_indicators: Project detection indicators
        file_extensions: File extension to language identifier mapping
        lsp_servers: LSP server configurations by language
        exclusion_patterns: File/directory patterns to exclude
        tree_sitter_grammars: Available tree-sitter grammars
        content_signatures: Content-based language detection
        build_systems: Build system configurations
        metadata_schemas: Metadata schema definitions
    """

    project_indicators: ProjectIndicators = Field(
        default_factory=ProjectIndicators,
        description="Project detection indicators",
    )
    file_extensions: dict[str, str] = Field(
        default_factory=dict,
        description="File extension to language identifier mapping",
    )
    lsp_servers: dict[str, LSPServerConfig] = Field(
        default_factory=dict,
        description="LSP server configurations by language",
    )
    exclusion_patterns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Exclusion patterns for files and directories",
    )
    tree_sitter_grammars: TreeSitterGrammars = Field(
        default_factory=TreeSitterGrammars,
        description="Available tree-sitter grammars",
    )
    content_signatures: ContentSignatures = Field(
        default_factory=ContentSignatures,
        description="Content-based language detection signatures",
    )
    build_systems: dict[str, BuildSystem] = Field(
        default_factory=dict,
        description="Build system configurations",
    )
    metadata_schemas: dict[str, MetadataSchema] = Field(
        default_factory=dict,
        description="Metadata schema definitions",
    )

    @field_validator("file_extensions")
    @classmethod
    def validate_file_extensions(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate file extensions start with a dot.

        Args:
            v: File extensions dictionary to validate

        Returns:
            Validated file extensions dictionary

        Raises:
            ValueError: If extension doesn't start with a dot
        """
        for ext in v.keys():
            if not ext.startswith("."):
                raise ValueError(f"File extension must start with a dot: {ext}")
        return v

    @field_validator("lsp_servers")
    @classmethod
    def validate_lsp_servers(
        cls, v: dict[str, LSPServerConfig]
    ) -> dict[str, LSPServerConfig]:
        """Validate LSP server configurations have required fields.

        Args:
            v: LSP servers dictionary to validate

        Returns:
            Validated LSP servers dictionary

        Raises:
            ValueError: If primary field is empty
        """
        for lang, config in v.items():
            if not config.primary:
                raise ValueError(f"LSP server for {lang} must have primary field")
        return v


# =============================================================================
# DATABASE LOADER MODELS (V4 Schema - for database population)
# =============================================================================


class LSPDefinition(BaseModel):
    """LSP server configuration for database loader (v4 schema).

    Maps to language_name, lsp_name, lsp_executable fields in languages table.

    Attributes:
        name: LSP server identifier (stored in lsp_name column)
        executable: LSP executable name (stored in lsp_executable column)
    """

    name: str = Field(
        ...,
        min_length=1,
        description="LSP server identifier (e.g., 'ruff-lsp', 'rust-analyzer')",
    )
    executable: str = Field(
        ...,
        min_length=1,
        description="LSP executable name (e.g., 'ruff-lsp', 'rust-analyzer')",
    )

    @field_validator("name", "executable")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that strings are not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Value cannot be empty or whitespace only")
        return v.strip()


class TreeSitterDefinition(BaseModel):
    """Tree-sitter parser configuration for database loader (v4 schema).

    Maps to ts_grammar field in languages table.

    Attributes:
        grammar: Tree-sitter grammar name (stored in ts_grammar column)
        repo: Optional GitHub repository URL (not stored in database)
    """

    grammar: str = Field(
        ...,
        min_length=1,
        description="Tree-sitter grammar name (e.g., 'python', 'rust')",
    )
    repo: str | None = Field(
        None, description="GitHub repository URL for the grammar (optional)"
    )

    @field_validator("grammar")
    @classmethod
    def validate_grammar_not_empty(cls, v: str) -> str:
        """Validate that grammar name is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Grammar name cannot be empty or whitespace only")
        return v.strip()

    @field_validator("repo")
    @classmethod
    def validate_repo_url(cls, v: str | None) -> str | None:
        """Validate that repo URL is a valid GitHub URL if provided."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        # Basic GitHub URL validation
        if not v.startswith(("http://", "https://")):
            raise ValueError("Repository URL must start with http:// or https://")
        if "github.com" not in v:
            raise ValueError("Repository URL must be a GitHub URL")
        return v


class LanguageDefinition(BaseModel):
    """
    Language definition for database loader (v4 schema).

    Maps directly to a row in the languages table in language_support_schema.sql.
    File extensions are stored as JSON array in the database.

    Database mapping:
        - name -> language_name (TEXT UNIQUE NOT NULL)
        - extensions -> file_extensions (TEXT, JSON array)
        - lsp.name -> lsp_name (TEXT)
        - lsp.executable -> lsp_executable (TEXT)
        - treesitter.grammar -> ts_grammar (TEXT)

    Attributes:
        name: Language name (unique identifier)
        extensions: List of file extensions (must start with '.')
        lsp: Optional LSP server configuration
        treesitter: Optional Tree-sitter parser configuration
    """

    name: str = Field(
        ..., min_length=1, description="Language name (e.g., 'python', 'rust')"
    )
    extensions: list[str] = Field(
        ...,
        min_items=1,
        description="File extensions for this language (e.g., ['.py', '.pyw'])",
    )
    lsp: LSPDefinition | None = Field(
        None, description="LSP server configuration (optional)"
    )
    treesitter: TreeSitterDefinition | None = Field(
        None, description="Tree-sitter parser configuration (optional)"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate language name is not empty and contains only valid characters."""
        if not v or not v.strip():
            raise ValueError("Language name cannot be empty or whitespace only")
        v = v.strip()
        # Language names should be alphanumeric with hyphens/underscores
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Language name must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @field_validator("extensions")
    @classmethod
    def validate_extensions(cls, v: list[str]) -> list[str]:
        """Validate that all extensions start with '.' and are not empty."""
        if not v:
            raise ValueError("At least one file extension is required")

        validated = []
        for ext in v:
            ext = ext.strip()
            if not ext:
                raise ValueError("File extension cannot be empty or whitespace only")
            if not ext.startswith("."):
                raise ValueError(f"File extension must start with '.': {ext}")
            if len(ext) == 1:
                raise ValueError("File extension cannot be just '.'")
            validated.append(ext)

        # Check for duplicates
        if len(validated) != len(set(validated)):
            raise ValueError("Duplicate file extensions are not allowed")

        return validated

    def model_post_init(self, __context) -> None:
        """Post-initialization validation to ensure at least one tool is configured."""
        if self.lsp is None and self.treesitter is None:
            raise ValueError(
                f"Language '{self.name}' must have at least one tool configured (LSP or Tree-sitter)"
            )


class LanguageSupportDatabaseConfig(BaseModel):
    """
    Root configuration for language support database loader (v4 schema).

    This is the NEW simplified format for loading language definitions into the
    database. It replaces the comprehensive asset file structure with a focused
    list-based format that maps directly to database rows.

    Maps to:
        - languages table rows (one per language)
        - language_support_version table (version tracking via hash)

    YAML Example:
        version: "1.0.0"
        languages:
          - name: python
            extensions: [".py", ".pyw"]
            lsp:
              name: ruff-lsp
              executable: ruff-lsp
            treesitter:
              grammar: python
              repo: https://github.com/tree-sitter/tree-sitter-python

    Attributes:
        version: Semantic version of the configuration (for change tracking)
        languages: List of language definitions to load into database
    """

    version: str = Field(
        ...,
        description="Semantic version of the language support configuration (e.g., '1.0.0')",
    )
    languages: list[LanguageDefinition] = Field(
        ..., min_items=1, description="List of language definitions"
    )

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate that version follows semantic versioning format."""
        if not v or not v.strip():
            raise ValueError("Version cannot be empty or whitespace only")

        v = v.strip()
        # Validate semantic versioning format: MAJOR.MINOR.PATCH
        semver_pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"

        if not re.match(semver_pattern, v):
            raise ValueError(
                f"Version must follow semantic versioning format (MAJOR.MINOR.PATCH): {v}"
            )

        return v

    @field_validator("languages")
    @classmethod
    def validate_unique_languages(
        cls, v: list[LanguageDefinition]
    ) -> list[LanguageDefinition]:
        """Validate that language names and extensions are unique."""
        if not v:
            raise ValueError("At least one language definition is required")

        # Check for duplicate language names
        names = [lang.name for lang in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Duplicate language names found: {', '.join(set(duplicates))}"
            )

        # Check for duplicate extensions across languages
        all_extensions = {}
        for lang in v:
            for ext in lang.extensions:
                if ext in all_extensions:
                    raise ValueError(
                        f"Extension '{ext}' is defined for both '{all_extensions[ext]}' and '{lang.name}'"
                    )
                all_extensions[ext] = lang.name

        return v
