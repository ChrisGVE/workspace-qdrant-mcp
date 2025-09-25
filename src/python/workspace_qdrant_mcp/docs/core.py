"""
Core documentation framework for workspace-qdrant-mcp.

This module provides the central DocumentationFramework class that coordinates
all documentation generation, validation, and deployment activities.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
from .deployment import DocumentationDeployer

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    RST = "rst"


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""

    # Source paths
    source_root: Path = field(default_factory=lambda: Path("src/python"))
    rust_source_root: Path = field(default_factory=lambda: Path("rust-engine"))
    output_root: Path = field(default_factory=lambda: Path("docs"))

    # Output formats
    output_formats: List[OutputFormat] = field(default_factory=lambda: [
        OutputFormat.MARKDOWN, OutputFormat.HTML
    ])

    # Generation settings
    include_private_apis: bool = False
    generate_interactive_examples: bool = True
    validate_examples: bool = True
    include_performance_benchmarks: bool = True

    # MCP-specific settings
    extract_mcp_tools: bool = True
    extract_tool_schemas: bool = True
    generate_tool_examples: bool = True

    # Rust-specific settings
    extract_rust_docs: bool = True
    include_rust_examples: bool = True

    # CLI-specific settings
    extract_cli_commands: bool = True
    generate_cli_examples: bool = True

    # Validation settings
    validate_links: bool = True
    validate_code_examples: bool = True
    check_consistency: bool = True

    # Deployment settings
    enable_versioning: bool = True
    deploy_on_generation: bool = False

    # Template settings
    custom_templates: Optional[Path] = None
    theme: str = "default"


@dataclass
class DocumentationMetrics:
    """Metrics for documentation generation."""

    total_files_processed: int = 0
    total_functions_documented: int = 0
    total_classes_documented: int = 0
    total_mcp_tools_documented: int = 0
    total_examples_generated: int = 0
    total_examples_validated: int = 0
    validation_errors: List[str] = field(default_factory=list)
    generation_time_seconds: float = 0.0
    last_updated: Optional[datetime] = None


class DocumentationFramework:
    """
    Comprehensive documentation framework for workspace-qdrant-mcp.

    This class coordinates all documentation generation activities including:
    - Source code analysis and extraction
    - MCP tool documentation generation
    - Interactive example creation and validation
    - Multi-format output generation
    - Validation and consistency checking
    - Deployment pipeline management
    """

    def __init__(self, config: Optional[DocumentationConfig] = None):
        """Initialize the documentation framework."""
        self.config = config or DocumentationConfig()
        self.console = Console()
        self.metrics = DocumentationMetrics()

        # Initialize generators
        self.mcp_generator = MCPToolDocumentationGenerator(self.config)
        self.python_generator = PythonAPIDocumentationGenerator(self.config)
        self.rust_generator = RustAPIDocumentationGenerator(self.config)
        self.cli_generator = CLIDocumentationGenerator(self.config)

        # Initialize validators
        self.doc_validator = DocumentationValidator(self.config)
        self.example_validator = ExampleValidator(self.config)
        self.consistency_checker = ConsistencyChecker(self.config)

        # Initialize deployer
        self.deployer = DocumentationDeployer(self.config)

        # Track generation state
        self.generation_started = False
        self.generation_completed = False

    async def generate_all_documentation(
        self,
        force_regenerate: bool = False,
        validate_before_deploy: bool = True
    ) -> DocumentationMetrics:
        """
        Generate all documentation components.

        Args:
            force_regenerate: Force regeneration even if files are up-to-date
            validate_before_deploy: Run validation before deployment

        Returns:
            DocumentationMetrics with generation statistics

        Raises:
            DocumentationError: If generation fails
        """
        start_time = datetime.now(timezone.utc)
        self.generation_started = True

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:

                # Phase 1: Generate MCP tool documentation
                task = progress.add_task("Generating MCP tool documentation...", total=1)
                mcp_docs = await self.mcp_generator.generate_documentation()
                self.metrics.total_mcp_tools_documented = len(mcp_docs)
                progress.update(task, advance=1)

                # Phase 2: Generate Python API documentation
                task = progress.add_task("Generating Python API documentation...", total=1)
                python_docs = await self.python_generator.generate_documentation()
                self.metrics.total_functions_documented = python_docs.get("functions", 0)
                self.metrics.total_classes_documented = python_docs.get("classes", 0)
                progress.update(task, advance=1)

                # Phase 3: Generate Rust API documentation
                if self.config.extract_rust_docs:
                    task = progress.add_task("Generating Rust API documentation...", total=1)
                    await self.rust_generator.generate_documentation()
                    progress.update(task, advance=1)

                # Phase 4: Generate CLI documentation
                if self.config.extract_cli_commands:
                    task = progress.add_task("Generating CLI documentation...", total=1)
                    await self.cli_generator.generate_documentation()
                    progress.update(task, advance=1)

                # Phase 5: Generate interactive examples
                if self.config.generate_interactive_examples:
                    task = progress.add_task("Generating interactive examples...", total=1)
                    examples = await self._generate_interactive_examples()
                    self.metrics.total_examples_generated = len(examples)
                    progress.update(task, advance=1)

                # Phase 6: Validate documentation
                if validate_before_deploy:
                    task = progress.add_task("Validating documentation...", total=1)
                    validation_results = await self._validate_all_documentation()
                    self.metrics.validation_errors = validation_results.get("errors", [])
                    progress.update(task, advance=1)

                # Phase 7: Deploy documentation
                if self.config.deploy_on_generation:
                    task = progress.add_task("Deploying documentation...", total=1)
                    await self.deployer.deploy_documentation()
                    progress.update(task, advance=1)

            # Update metrics
            end_time = datetime.now(timezone.utc)
            self.metrics.generation_time_seconds = (end_time - start_time).total_seconds()
            self.metrics.last_updated = end_time
            self.generation_completed = True

            self.console.print(f"✅ Documentation generation completed in {self.metrics.generation_time_seconds:.2f}s")
            return self.metrics

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            self.generation_completed = False
            raise DocumentationError(f"Documentation generation failed: {e}") from e

    async def validate_documentation(self) -> Dict[str, Any]:
        """
        Validate all generated documentation.

        Returns:
            Dict containing validation results
        """
        return await self._validate_all_documentation()

    async def deploy_documentation(self) -> bool:
        """
        Deploy generated documentation.

        Returns:
            True if deployment successful, False otherwise
        """
        try:
            return await self.deployer.deploy_documentation()
        except Exception as e:
            logger.error(f"Documentation deployment failed: {e}")
            return False

    async def _generate_interactive_examples(self) -> List[Dict[str, Any]]:
        """Generate interactive examples for all components."""
        examples = []

        # Generate MCP tool examples
        if self.config.generate_tool_examples:
            mcp_examples = await self.mcp_generator.generate_examples()
            examples.extend(mcp_examples)

        # Generate CLI examples
        if self.config.generate_cli_examples:
            cli_examples = await self.cli_generator.generate_examples()
            examples.extend(cli_examples)

        # Validate examples if requested
        if self.config.validate_examples:
            validated_examples = []
            for example in examples:
                if await self.example_validator.validate_example(example):
                    validated_examples.append(example)
                    self.metrics.total_examples_validated += 1
            examples = validated_examples

        return examples

    async def _validate_all_documentation(self) -> Dict[str, Any]:
        """Validate all documentation components."""
        validation_results = {
            "errors": [],
            "warnings": [],
            "passed_checks": 0,
            "total_checks": 0
        }

        # Validate documentation structure
        doc_validation = await self.doc_validator.validate_structure()
        validation_results["errors"].extend(doc_validation.get("errors", []))
        validation_results["warnings"].extend(doc_validation.get("warnings", []))
        validation_results["passed_checks"] += doc_validation.get("passed_checks", 0)
        validation_results["total_checks"] += doc_validation.get("total_checks", 0)

        # Validate examples
        example_validation = await self.example_validator.validate_all_examples()
        validation_results["errors"].extend(example_validation.get("errors", []))
        validation_results["warnings"].extend(example_validation.get("warnings", []))
        validation_results["passed_checks"] += example_validation.get("passed_checks", 0)
        validation_results["total_checks"] += example_validation.get("total_checks", 0)

        # Check consistency
        if self.config.check_consistency:
            consistency_validation = await self.consistency_checker.check_consistency()
            validation_results["errors"].extend(consistency_validation.get("errors", []))
            validation_results["warnings"].extend(consistency_validation.get("warnings", []))
            validation_results["passed_checks"] += consistency_validation.get("passed_checks", 0)
            validation_results["total_checks"] += consistency_validation.get("total_checks", 0)

        return validation_results

    def get_metrics(self) -> DocumentationMetrics:
        """Get current documentation generation metrics."""
        return self.metrics

    def get_config(self) -> DocumentationConfig:
        """Get current documentation configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update documentation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


class DocumentationError(Exception):
    """Exception raised for documentation generation errors."""
    pass


class ValidationError(Exception):
    """Exception raised for documentation validation errors."""
    pass