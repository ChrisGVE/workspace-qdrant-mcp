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
    LanguageSupportConfig: Root configuration model

Example:
    >>> from pathlib import Path
    >>> config = LanguageSupportConfig.from_yaml(Path("language_support.yaml"))
    >>> print(config.file_extensions[".py"])
    'python'
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


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
