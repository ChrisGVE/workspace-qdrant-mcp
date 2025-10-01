"""YAML parser for language support configuration.

This module provides parsing and validation for language_support.yaml configuration
files using Pydantic models for schema validation. It handles YAML parsing errors,
validation errors, and provides clear error messages with line numbers when possible.

Classes:
    LanguageSupportParser: Main parser class for language support YAML files

Example:
    >>> from pathlib import Path
    >>> parser = LanguageSupportParser()
    >>> config = parser.parse_yaml(Path("assets/language_support.yaml"))
    >>> print(len(config.file_extensions))
    500+

    >>> yaml_content = '''
    ... file_extensions:
    ...   .py: python
    ...   .rs: rust
    ... '''
    >>> config = parser.parse_yaml_string(yaml_content)
    >>> print(config.file_extensions[".py"])
    'python'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .language_support_models import LanguageSupportConfig


class LanguageSupportParser:
    """Parser for language support YAML configuration files.

    This class handles parsing and validation of language_support.yaml files,
    converting them into validated LanguageSupportConfig objects. It provides
    clear error messages for parsing failures and validation errors.

    Methods:
        parse_yaml: Parse YAML file from disk
        parse_yaml_string: Parse YAML content from string
    """

    def parse_yaml(self, file_path: Path) -> LanguageSupportConfig:
        """Parse and validate a language support YAML file.

        Loads a YAML file from disk, parses it, and validates it against the
        LanguageSupportConfig schema using Pydantic models.

        Args:
            file_path: Path to the YAML file to parse

        Returns:
            Validated LanguageSupportConfig object

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed
            ValidationError: If the YAML doesn't match the expected schema

        Example:
            >>> parser = LanguageSupportParser()
            >>> config = parser.parse_yaml(Path("language_support.yaml"))
            >>> print(config.file_extensions[".py"])
            'python'
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Language support YAML file not found: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
        except OSError as e:
            raise OSError(f"Failed to read YAML file {file_path}: {e}") from e

        return self.parse_yaml_string(yaml_content, source_path=file_path)

    def parse_yaml_string(
        self, yaml_content: str, source_path: Path | None = None
    ) -> LanguageSupportConfig:
        """Parse and validate language support YAML from a string.

        Parses YAML content from a string and validates it against the
        LanguageSupportConfig schema using Pydantic models.

        Args:
            yaml_content: YAML content as a string
            source_path: Optional path for better error messages

        Returns:
            Validated LanguageSupportConfig object

        Raises:
            yaml.YAMLError: If the YAML is malformed
            ValidationError: If the YAML doesn't match the expected schema

        Example:
            >>> parser = LanguageSupportParser()
            >>> yaml_str = 'file_extensions:\\n  .py: python'
            >>> config = parser.parse_yaml_string(yaml_str)
            >>> print(config.file_extensions[".py"])
            'python'
        """
        # Parse YAML with safe_load to prevent code execution
        try:
            yaml_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            source_info = f" in {source_path}" if source_path else ""
            error_msg = f"Failed to parse YAML{source_info}: {e}"

            # Try to extract line number from YAML error
            if hasattr(e, "problem_mark"):
                mark = e.problem_mark
                error_msg += f"\n  Line {mark.line + 1}, Column {mark.column + 1}"
                # Add snippet of the problematic line if available
                lines = yaml_content.split("\n")
                if 0 <= mark.line < len(lines):
                    error_msg += f"\n  >>> {lines[mark.line]}"
                    error_msg += f"\n  >>> {' ' * mark.column}^"

            raise yaml.YAMLError(error_msg) from e

        # Handle empty YAML files
        if yaml_data is None:
            yaml_data = {}

        # Validate YAML data against Pydantic model
        try:
            config = LanguageSupportConfig(**yaml_data)
        except ValidationError as e:
            source_info = f" in {source_path}" if source_path else ""
            error_msg = f"YAML validation failed{source_info}:\n"

            # Format validation errors with field paths and messages
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_msg += f"  - {field_path}: {error['msg']}\n"

                # Add value that caused the error if available
                if "input" in error:
                    error_msg += f"    Got: {error['input']!r}\n"

            raise ValidationError.from_exception_data(
                title="Language Support Validation Error",
                line_errors=e.errors(),
            ) from e

        return config

    def validate_yaml(self, file_path: Path) -> tuple[bool, str]:
        """Validate a YAML file without raising exceptions.

        Convenience method that validates a YAML file and returns a success
        flag with an error message instead of raising exceptions.

        Args:
            file_path: Path to the YAML file to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation succeeded, False otherwise
            - error_message: Empty string if valid, error description otherwise

        Example:
            >>> parser = LanguageSupportParser()
            >>> is_valid, error = parser.validate_yaml(Path("config.yaml"))
            >>> if not is_valid:
            ...     print(f"Validation failed: {error}")
        """
        try:
            self.parse_yaml(file_path)
            return True, ""
        except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
            return False, str(e)


def parse_language_support_yaml(file_path: Path) -> LanguageSupportConfig:
    """Convenience function to parse a language support YAML file.

    Args:
        file_path: Path to the YAML file to parse

    Returns:
        Validated LanguageSupportConfig object

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is malformed
        ValidationError: If the YAML doesn't match the expected schema

    Example:
        >>> config = parse_language_support_yaml(Path("language_support.yaml"))
        >>> print(config.lsp_servers["python"].primary)
        'pylsp'
    """
    parser = LanguageSupportParser()
    return parser.parse_yaml(file_path)
