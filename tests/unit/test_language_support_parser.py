"""Unit tests for language support YAML parser."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.python.common.core.language_support_models import (
    BuildSystem,
    ContentSignatures,
    LanguageSupportConfig,
    LSPServerConfig,
    MetadataSchema,
    ProjectIndicators,
    TreeSitterGrammars,
)
from src.python.common.core.language_support_parser import (
    LanguageSupportParser,
    parse_language_support_yaml,
)


class TestLanguageSupportParser:
    """Test suite for LanguageSupportParser."""

    def test_parse_valid_yaml_string(self):
        """Test parsing valid YAML from string."""
        yaml_content = """
file_extensions:
  .py: python
  .rs: rust
  .js: javascript

lsp_servers:
  python:
    primary: pylsp
    features: ["completion", "hover"]
    rationale: "Good Python support"
    install_notes: "pip install python-lsp-server"

tree_sitter_grammars:
  available:
    - python
    - rust
    - javascript
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.file_extensions) == 3
        assert config.file_extensions[".py"] == "python"
        assert config.file_extensions[".rs"] == "rust"
        assert config.file_extensions[".js"] == "javascript"

        assert "python" in config.lsp_servers
        assert config.lsp_servers["python"].primary == "pylsp"
        assert "completion" in config.lsp_servers["python"].features

        assert len(config.tree_sitter_grammars.available) == 3
        assert "python" in config.tree_sitter_grammars.available

    def test_parse_minimal_yaml(self):
        """Test parsing minimal valid YAML."""
        yaml_content = """
file_extensions:
  .py: python
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.file_extensions) == 1
        assert config.file_extensions[".py"] == "python"
        assert len(config.lsp_servers) == 0
        assert len(config.tree_sitter_grammars.available) == 0

    def test_parse_empty_yaml(self):
        """Test parsing empty YAML file."""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string("")

        assert isinstance(config, LanguageSupportConfig)
        assert len(config.file_extensions) == 0

    def test_parse_project_indicators(self):
        """Test parsing project indicators."""
        yaml_content = """
project_indicators:
  version_control:
    - .git
    - .hg
  language_ecosystems:
    - package.json
    - Cargo.toml
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.project_indicators.version_control) == 2
        assert ".git" in config.project_indicators.version_control
        assert len(config.project_indicators.language_ecosystems) == 2
        assert "package.json" in config.project_indicators.language_ecosystems

    def test_parse_content_signatures(self):
        """Test parsing content signatures."""
        yaml_content = """
content_signatures:
  shebangs:
    "#!/usr/bin/env python": python
    "#!/bin/bash": bash
  keyword_patterns:
    python:
      - "def "
      - "class "
      - "import "
    rust:
      - "fn "
      - "struct "
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.content_signatures.shebangs) == 2
        assert config.content_signatures.shebangs["#!/usr/bin/env python"] == "python"

        assert len(config.content_signatures.keyword_patterns) == 2
        assert "def " in config.content_signatures.keyword_patterns["python"]
        assert "fn " in config.content_signatures.keyword_patterns["rust"]

    def test_parse_build_systems(self):
        """Test parsing build systems."""
        yaml_content = """
build_systems:
  cargo:
    files: ["Cargo.toml", "Cargo.lock"]
    language: rust
    commands: ["cargo build", "cargo test"]
  npm:
    files: ["package.json"]
    language: javascript
    commands: ["npm install", "npm test"]
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.build_systems) == 2
        assert "cargo" in config.build_systems
        assert config.build_systems["cargo"].language == "rust"
        assert "Cargo.toml" in config.build_systems["cargo"].files
        assert "cargo build" in config.build_systems["cargo"].commands

    def test_parse_metadata_schemas(self):
        """Test parsing metadata schemas."""
        yaml_content = """
metadata_schemas:
  source_code:
    required:
      - language
      - file_path
    optional:
      - symbols
      - complexity_score
  documentation:
    required:
      - title
      - format
    optional:
      - author
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert len(config.metadata_schemas) == 2
        assert "source_code" in config.metadata_schemas
        assert "language" in config.metadata_schemas["source_code"].required
        assert "symbols" in config.metadata_schemas["source_code"].optional

    def test_invalid_yaml_syntax(self):
        """Test handling of malformed YAML."""
        yaml_content = """
file_extensions:
  .py: python
  invalid yaml syntax here: [unclosed
"""
        parser = LanguageSupportParser()

        with pytest.raises(yaml.YAMLError) as exc_info:
            parser.parse_yaml_string(yaml_content)

        assert "Failed to parse YAML" in str(exc_info.value)

    def test_invalid_file_extension(self):
        """Test validation of file extensions."""
        yaml_content = """
file_extensions:
  py: python
"""
        parser = LanguageSupportParser()

        with pytest.raises(ValidationError) as exc_info:
            parser.parse_yaml_string(yaml_content)

        errors = exc_info.value.errors()
        assert any("must start with a dot" in str(error) for error in errors)

    def test_invalid_shebang(self):
        """Test validation of shebang patterns."""
        yaml_content = """
content_signatures:
  shebangs:
    "/usr/bin/env python": python
"""
        parser = LanguageSupportParser()

        with pytest.raises(ValidationError) as exc_info:
            parser.parse_yaml_string(yaml_content)

        errors = exc_info.value.errors()
        assert any("must start with #!" in str(error) for error in errors)

    def test_invalid_lsp_server(self):
        """Test validation of LSP server configuration."""
        yaml_content = """
lsp_servers:
  python:
    features: ["completion"]
"""
        parser = LanguageSupportParser()

        with pytest.raises(ValidationError) as exc_info:
            parser.parse_yaml_string(yaml_content)

        # Should fail because 'primary' field is required
        errors = exc_info.value.errors()
        assert any("primary" in str(error) for error in errors)

    def test_parse_yaml_file_not_found(self, tmp_path: Path):
        """Test handling of missing file."""
        parser = LanguageSupportParser()
        non_existent = tmp_path / "non_existent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            parser.parse_yaml(non_existent)

        assert "not found" in str(exc_info.value)

    def test_parse_yaml_file(self, tmp_path: Path):
        """Test parsing YAML from file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            """
file_extensions:
  .py: python
  .rs: rust

lsp_servers:
  python:
    primary: pylsp
    features: ["completion"]
""",
            encoding="utf-8",
        )

        parser = LanguageSupportParser()
        config = parser.parse_yaml(yaml_file)

        assert len(config.file_extensions) == 2
        assert config.file_extensions[".py"] == "python"
        assert "python" in config.lsp_servers

    def test_validate_yaml_success(self, tmp_path: Path):
        """Test validate_yaml method with valid YAML."""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(
            """
file_extensions:
  .py: python
""",
            encoding="utf-8",
        )

        parser = LanguageSupportParser()
        is_valid, error = parser.validate_yaml(yaml_file)

        assert is_valid is True
        assert error == ""

    def test_validate_yaml_failure(self, tmp_path: Path):
        """Test validate_yaml method with invalid YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(
            """
file_extensions:
  py: python
""",
            encoding="utf-8",
        )

        parser = LanguageSupportParser()
        is_valid, error = parser.validate_yaml(yaml_file)

        assert is_valid is False
        assert len(error) > 0
        assert "must start with a dot" in error

    def test_convenience_function(self, tmp_path: Path):
        """Test parse_language_support_yaml convenience function."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            """
file_extensions:
  .py: python
""",
            encoding="utf-8",
        )

        config = parse_language_support_yaml(yaml_file)

        assert isinstance(config, LanguageSupportConfig)
        assert config.file_extensions[".py"] == "python"

    def test_parse_actual_language_support_file(self):
        """Test parsing the actual languages_support.yaml file if it exists."""
        yaml_path = Path(__file__).parents[2] / "assets" / "languages_support.yaml"

        if not yaml_path.exists():
            pytest.skip("languages_support.yaml not found")

        parser = LanguageSupportParser()
        config = parser.parse_yaml(yaml_path)

        # Validate structure
        assert isinstance(config, LanguageSupportConfig)
        assert len(config.file_extensions) > 0
        assert len(config.lsp_servers) > 0

        # Validate common languages
        assert config.file_extensions.get(".py") == "python"
        assert config.file_extensions.get(".rs") == "rust"

        # Validate LSP servers exist for common languages
        assert "python" in config.lsp_servers or "ruff" in config.lsp_servers
        assert "rust" in config.lsp_servers

        # Validate tree-sitter grammars
        assert len(config.tree_sitter_grammars.available) > 0

    def test_error_message_includes_line_number(self):
        """Test that YAML errors include line numbers."""
        yaml_content = """line 1
line 2
line 3
invalid: [unclosed bracket
line 5
"""
        parser = LanguageSupportParser()

        with pytest.raises(yaml.YAMLError) as exc_info:
            parser.parse_yaml_string(yaml_content)

        error_msg = str(exc_info.value)
        assert "Line" in error_msg or "line" in error_msg

    def test_exclusion_patterns(self):
        """Test parsing exclusion patterns."""
        yaml_content = """
exclusion_patterns:
  directories:
    - node_modules
    - __pycache__
    - .git
  files:
    - "*.pyc"
    - "*.log"
"""
        parser = LanguageSupportParser()
        config = parser.parse_yaml_string(yaml_content)

        assert "directories" in config.exclusion_patterns
        assert "node_modules" in config.exclusion_patterns["directories"]
        assert "files" in config.exclusion_patterns
        assert "*.pyc" in config.exclusion_patterns["files"]
