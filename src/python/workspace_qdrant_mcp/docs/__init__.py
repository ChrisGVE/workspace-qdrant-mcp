"""
Comprehensive documentation framework for workspace-qdrant-mcp.

This module provides automated documentation generation, interactive examples,
validation, and deployment capabilities for the workspace-qdrant-mcp project.

Key Components:
    - generators: Automated documentation generation from source code
    - validators: Documentation validation and consistency checking
    - templates: Jinja2 templates for various output formats
    - examples: Interactive API examples and tutorials
    - deployment: Documentation deployment and versioning pipeline

Features:
    - MCP tool documentation extraction
    - Rust engine API documentation
    - CLI command reference generation
    - Interactive code examples with validation
    - Multi-format output (HTML, Markdown, PDF)
    - Comprehensive validation and testing
"""

from .core import DocumentationFramework
from .generators import (
    MCPToolDocumentationGenerator,
    PythonAPIDocumentationGenerator,
    RustAPIDocumentationGenerator,
    CLIDocumentationGenerator,
)
from .validators import (
    DocumentationValidator,
    ExampleValidator,
    ConsistencyChecker,
)

__all__ = [
    "DocumentationFramework",
    "MCPToolDocumentationGenerator",
    "PythonAPIDocumentationGenerator",
    "RustAPIDocumentationGenerator",
    "CLIDocumentationGenerator",
    "DocumentationValidator",
    "ExampleValidator",
    "ConsistencyChecker",
]