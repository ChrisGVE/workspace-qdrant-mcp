"""
Base documentation generator for workspace-qdrant-mcp.

This module provides the abstract base class for all documentation generators,
defining common functionality and interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import jinja2
from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""

    title: str
    content: str
    section_type: str
    metadata: Dict[str, Any]
    examples: List[Dict[str, Any]]
    subsections: List['DocumentationSection']

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "metadata": self.metadata,
            "examples": self.examples,
            "subsections": [sub.to_dict() for sub in self.subsections]
        }


@dataclass
class GenerationResult:
    """Results of documentation generation."""

    sections: List[DocumentationSection]
    metadata: Dict[str, Any]
    statistics: Dict[str, int]
    errors: List[str]
    warnings: List[str]

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
            "statistics": self.statistics,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success
        }


class BaseDocumentationGenerator(ABC):
    """
    Abstract base class for documentation generators.

    This class provides common functionality for all documentation generators
    including template rendering, file operations, and error handling.
    """

    def __init__(self, config):
        """Initialize the generator."""
        self.config = config
        self.console = Console()
        self.template_env = self._setup_template_environment()
        self.errors = []
        self.warnings = []

    def _setup_template_environment(self) -> jinja2.Environment:
        """Set up Jinja2 template environment."""
        template_paths = [
            Path(__file__).parent.parent / "templates",
        ]

        if self.config.custom_templates:
            template_paths.insert(0, self.config.custom_templates)

        loader = jinja2.FileSystemLoader([str(path) for path in template_paths])
        env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        env.filters['format_datetime'] = self._format_datetime
        env.filters['markdown_escape'] = self._markdown_escape
        env.filters['rst_escape'] = self._rst_escape

        return env

    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """Format datetime for documentation."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _markdown_escape(text: str) -> str:
        """Escape special characters for Markdown."""
        special_chars = ['*', '_', '`', '[', ']', '(', ')', '#', '+', '-', '.', '!', '\\', '|']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    @staticmethod
    def _rst_escape(text: str) -> str:
        """Escape special characters for reStructuredText."""
        special_chars = ['*', '`', '_', '\\', '|', '[', ']', '(', ')', '#', '+', '-', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    @abstractmethod
    async def generate_documentation(self) -> GenerationResult:
        """
        Generate documentation.

        Returns:
            GenerationResult with generated documentation sections
        """
        pass

    @abstractmethod
    async def generate_examples(self) -> List[Dict[str, Any]]:
        """
        Generate interactive examples.

        Returns:
            List of example dictionaries
        """
        pass

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a template with the given context.

        Args:
            template_name: Name of the template file
            context: Template context variables

        Returns:
            Rendered template content

        Raises:
            TemplateError: If template rendering fails
        """
        try:
            template = self.template_env.get_template(template_name)
            return template.render(**context)
        except jinja2.TemplateError as e:
            error_msg = f"Template rendering failed for {template_name}: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            raise TemplateError(error_msg) from e

    async def write_output_file(
        self,
        content: str,
        output_path: Path,
        format_type: str = "markdown"
    ) -> bool:
        """
        Write content to output file.

        Args:
            content: Content to write
            output_path: Path to output file
            format_type: Type of content format

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            async def _write_file():
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            await asyncio.get_event_loop().run_in_executor(None, _write_file)
            logger.info(f"Generated {format_type} documentation: {output_path}")
            return True

        except Exception as e:
            error_msg = f"Failed to write {format_type} documentation to {output_path}: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)

    def clear_messages(self) -> None:
        """Clear all error and warning messages."""
        self.errors.clear()
        self.warnings.clear()

    def get_messages(self) -> Dict[str, List[str]]:
        """Get all error and warning messages."""
        return {
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy()
        }


class TemplateError(Exception):
    """Exception raised for template rendering errors."""
    pass


class GenerationError(Exception):
    """Exception raised for documentation generation errors."""
    pass