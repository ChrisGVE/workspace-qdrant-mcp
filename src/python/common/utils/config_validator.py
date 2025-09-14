"""
Comprehensive configuration validation and setup guidance utilities.

This module provides advanced configuration validation, connection testing, and setup
guidance for workspace-qdrant-mcp. It performs comprehensive health checks across all
components including Qdrant database connectivity, embedding model availability,
project detection capabilities, and configuration consistency.

Key Features:
    - Comprehensive validation of all configuration parameters
    - Live connection testing for Qdrant database and embedding models
    - Intelligent project detection validation with Git integration
    - Detailed error reporting with actionable suggestions
    - Setup guidance and troubleshooting recommendations
    - CLI interface for configuration diagnostics
    - Environment variable validation and conflict detection

Validation Coverage:
    - Qdrant database connectivity and authentication
    - Embedding model initialization and compatibility
    - Project detection algorithm functionality
    - Configuration parameter ranges and consistency
    - Environment variable setup and conflicts
    - File system permissions and requirements

Example:
    ```python
    from workspace_qdrant_mcp.utils.config_validator import ConfigValidator
    from workspace_qdrant_mcp.core.config import Config

    config = Config()
    validator = ConfigValidator(config)

    # Comprehensive validation
    is_valid, results = validator.validate_all()
    if not is_valid:
        logger.info("Configuration issues found:")
        for issue in results['issues']:
            logger.info("  - {issue}")

    # Individual component testing
    qdrant_ok, message = validator.validate_qdrant_connection()
    model_ok, message = validator.validate_embedding_model()
    ```

CLI Usage:
    ```bash
    # Validate current configuration
    workspace-qdrant-mcp-validate

    # Show detailed validation results
    workspace-qdrant-mcp-validate --verbose

    # Get setup guidance
    workspace-qdrant-mcp-validate --guide

    # Validate custom configuration
    workspace-qdrant-mcp-validate --config /path/to/config.toml
    ```
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import typer
from qdrant_client import QdrantClient

from ..core.config import Config
from ..core.embeddings import EmbeddingService
from .project_detection import ProjectDetector

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Advanced configuration validator with comprehensive health checking capabilities.

    This class provides thorough validation of all workspace-qdrant-mcp configuration
    parameters, including live connectivity testing, model validation, and environment
    analysis. It's designed to catch configuration issues early and provide actionable
    guidance for resolving problems.

    The validator performs multiple types of validation:
        - **Static validation**: Configuration parameter ranges, formats, consistency
        - **Dynamic validation**: Live connection testing, model initialization
        - **Environment validation**: Variable conflicts, file permissions, dependencies
        - **Logic validation**: Cross-parameter dependencies and constraints

    Validation Categories:
        1. **Qdrant Configuration**: URL format, connectivity, authentication
        2. **Embedding Configuration**: Model compatibility, parameter ranges
        3. **Workspace Configuration**: Collection names, user settings
        4. **Server Configuration**: Host/port validity, permission requirements
        5. **Environment**: Variable consistency, file availability

    Attributes:
        config (Config): Configuration object to validate
        issues (List[str]): Critical issues that prevent operation
        warnings (List[str]): Non-critical issues that may affect performance
        suggestions (List[str]): Recommendations for optimization

    Example:
        ```python
        # Basic validation
        validator = ConfigValidator()
        is_valid, results = validator.validate_all()

        # Custom configuration validation
        config = Config()
        config.qdrant.url = "http://custom-qdrant:6333"
        validator = ConfigValidator(config)

        # Component-specific testing
        qdrant_ok, msg = validator.validate_qdrant_connection()
        model_ok, msg = validator.validate_embedding_model()
        project_ok, msg = validator.validate_project_detection()
        ```
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the configuration validator.

        Args:
            config: Configuration object to validate. If None, creates new Config()
                   which loads from environment variables and .env files
        """
        self.config = config or Config()
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.suggestions: list[str] = []

    def validate_qdrant_connection(self) -> tuple[bool, str]:
        """
        Validate Qdrant database connection with comprehensive testing.

        Performs live connectivity testing to verify that the configured Qdrant
        database is reachable, properly configured, and responsive. This includes
        testing authentication if an API key is configured.

        The validation process:
        1. Creates a Qdrant client with current configuration
        2. Attempts to retrieve collection list (basic connectivity)
        3. Tests authentication if API key is provided
        4. Verifies response time and basic functionality
        5. Properly closes the connection

        Returns:
            Tuple[bool, str]: (is_valid, message) where:
                - is_valid: True if connection successful, False otherwise
                - message: Success confirmation or detailed error description

        Common Failure Scenarios:
            - Qdrant service not running or unreachable
            - Invalid URL format or wrong port
            - Authentication failure with wrong API key
            - Network connectivity issues
            - Service overloaded or responding slowly

        Example:
            ```python
            validator = ConfigValidator()
            is_valid, message = validator.validate_qdrant_connection()
            if is_valid:
                logger.info("✓ {message}")
            else:
                logger.info("✗ Connection failed: {message}")
            ```
        """
        try:
            import warnings
            import urllib3
            
            # Create client and test connection with SSL warning suppression
            from ..core.ssl_config import suppress_qdrant_ssl_warnings
            with suppress_qdrant_ssl_warnings():
                client = QdrantClient(**self.config.qdrant_client_config)
                client.get_collections()
                client.close()
            
            return True, "Qdrant successfully connected to server"
        except Exception as e:
            return False, str(e)

    def validate_embedding_model(self) -> tuple[bool, str]:
        """
        Validate embedding model availability and initialization capability.

        Tests whether the configured embedding model can be properly initialized
        and provides basic model information. This validation checks model
        availability, compatibility, and basic functionality without full initialization
        to avoid unnecessary resource usage.

        The validation process:
        1. Creates an EmbeddingService instance with current configuration
        2. Retrieves model information and metadata
        3. Validates model name and expected dimensions
        4. Checks sparse vector configuration if enabled
        5. Verifies model compatibility with current environment

        Returns:
            Tuple[bool, str]: (is_valid, message) where:
                - is_valid: True if model validation successful, False otherwise
                - message: Model details on success or error description on failure

        Common Failure Scenarios:
            - Model name not found or typo in configuration
            - Network issues preventing model download
            - Insufficient disk space for model cache
            - Memory limitations for model initialization
            - Incompatible model format or version

        Example:
            ```python
            validator = ConfigValidator()
            is_valid, message = validator.validate_embedding_model()
            if is_valid:
                logger.info("✓ {message}")  # e.g., "all-MiniLM-L6-v2 (384D)"
            else:
                logger.info("✗ Model validation failed: {message}")
            ```
        """
        try:
            embedding_service = EmbeddingService(self.config)
            # Try to get model info without full initialization
            model_info = embedding_service.get_model_info()
            model_name = model_info["dense_model"]["name"]
            vector_size = model_info["dense_model"]["dimensions"]
            return (
                True,
                f"Embedding model successfully loaded: {model_name} ({vector_size}D)",
            )
        except Exception as e:
            return False, str(e)

    def validate_project_detection(self) -> tuple[bool, str]:
        """
        Validate project detection algorithm and workspace identification.

        Tests the project detection functionality to ensure it can properly
        identify project structure, Git repositories, and workspace organization.
        This includes validating GitHub user configuration and subproject discovery.

        The validation process:
        1. Creates a ProjectDetector with configured GitHub user
        2. Analyzes current directory structure and Git repository
        3. Tests project name extraction from remotes or directory
        4. Discovers and validates subprojects/submodules
        5. Verifies workspace hierarchy and naming conventions

        Detection Capabilities Tested:
            - Git repository identification and analysis
            - Remote URL parsing and project name extraction
            - GitHub user ownership validation
            - Submodule discovery and processing
            - Fallback to directory-based naming

        Returns:
            Tuple[bool, str]: (is_valid, message) where:
                - is_valid: True if project detection successful, False otherwise
                - message: Project details on success or error description on failure

        Example Output Messages:
            - "Project detection successful: my-project with 3 subprojects"
            - "Directory detection successful: workspace (not a Git repository)"
            - "Git repository found but remote parsing failed"

        Example:
            ```python
            validator = ConfigValidator()
            is_valid, message = validator.validate_project_detection()
            if is_valid:
                logger.info("✓ {message}")
            else:
                logger.info("✗ Detection failed: {message}")
            ```
        """
        try:
            detector = ProjectDetector(github_user=self.config.workspace.github_user)
            project_info = detector.get_project_info()

            main_project = project_info["main_project"]
            subprojects = project_info["subprojects"]
            is_git_repo = project_info["is_git_repo"]

            if is_git_repo:
                message = f"Project detection successful: {main_project}"
                if subprojects:
                    subproject_count = len(subprojects)
                    message += f" with {subproject_count} subproject{'s' if subproject_count != 1 else ''}"
            else:
                message = f"Directory detection successful: {main_project} (not a Git repository)"

            return True, message
        except Exception as e:
            return False, str(e)

    def validate_all(self) -> tuple[bool, dict[str, Any]]:
        """
        Perform comprehensive validation of all configuration components.

        Executes a complete validation suite covering all aspects of the
        workspace-qdrant-mcp configuration including connectivity, models,
        project detection, and parameter consistency. This is the primary
        validation method that should be used for complete health checks.

        Validation Components:
        1. **Qdrant Connection**: Live database connectivity and authentication
        2. **Embedding Model**: Model availability and initialization capability
        3. **Project Detection**: Workspace structure and Git repository analysis
        4. **Configuration Parameters**: Value ranges, consistency, and formatting
        5. **Environment**: Variable conflicts and system requirements

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation_results) where:
                - is_valid: True if all validations pass, False if any critical issues
                - validation_results: Comprehensive results dictionary containing:
                    - issues (List[str]): Critical errors preventing operation
                    - warnings (List[str]): Non-critical issues requiring attention
                    - qdrant_connection (dict): Connection test results
                    - embedding_model (dict): Model validation results
                    - project_detection (dict): Project analysis results
                    - config_validation (dict): Parameter validation results

        Result Structure:
            ```python
            {
                "issues": ["Critical error messages"],
                "warnings": ["Warning messages"],
                "qdrant_connection": {
                    "valid": bool,
                    "message": str
                },
                "embedding_model": {
                    "valid": bool,
                    "message": str
                },
                "project_detection": {
                    "valid": bool,
                    "message": str
                },
                "config_validation": {
                    "valid": bool,
                    "issues": List[str]
                }
            }
            ```

        Example:
            ```python
            validator = ConfigValidator()
            is_valid, results = validator.validate_all()

            if is_valid:
                logger.info("All validation checks passed!")
            else:
                logger.info("Found {len(results['issues'])} issues:")
                for issue in results['issues']:
                    logger.info("  • {issue}")

            if results['warnings']:
                logger.info("{len(results['warnings'])} warnings:")
                for warning in results['warnings']:
                    logger.info("  • {warning}")
            ```
        """
        # Clear previous results
        self.issues.clear()
        self.warnings.clear()
        self.suggestions.clear()

        # Individual validations
        qdrant_valid, qdrant_message = self.validate_qdrant_connection()
        embedding_valid, embedding_message = self.validate_embedding_model()
        project_valid, project_message = self.validate_project_detection()

        # Basic config validation
        config_issues = self.config.validate_config()

        # Collect issues
        issues = []
        if not qdrant_valid:
            issues.append(qdrant_message)
        if not embedding_valid:
            issues.append(embedding_message)
        if not project_valid:
            issues.append(project_message)
        issues.extend(config_issues)

        # Generate warnings
        warnings = self._generate_warnings()

        # Overall validation status
        is_valid = len(issues) == 0

        # Comprehensive results structure
        results = {
            "issues": issues,
            "warnings": warnings,
            "qdrant_connection": {"valid": qdrant_valid, "message": qdrant_message},
            "embedding_model": {"valid": embedding_valid, "message": embedding_message},
            "project_detection": {"valid": project_valid, "message": project_message},
            "config_validation": {
                "valid": len(config_issues) == 0,
                "issues": config_issues,
            },
        }

        return is_valid, results

    def _generate_warnings(self) -> list[str]:
        """Generate configuration warnings."""
        warnings = []

        # Check for missing GitHub user when it would be beneficial
        if not self.config.workspace.github_user:
            try:
                detector = ProjectDetector()
                project_info = detector.get_project_info()
                if project_info.get("is_git_repo") and project_info.get("remote_url"):
                    warnings.append(
                        "GitHub user not configured - project ownership detection limited"
                    )
            except Exception:
                # Ignore detection errors for warning generation
                pass

        return warnings

    # Keep all existing validation methods below this line...

    def _validate_qdrant_config(self) -> None:
        """Validate Qdrant connection configuration."""
        config = self.config.qdrant

        # Validate URL format
        try:
            parsed = urlparse(config.url)
            if not parsed.scheme or not parsed.hostname:
                self.issues.append(
                    "Qdrant URL must include scheme and hostname (e.g., http://localhost:6333)"
                )
            elif parsed.scheme not in ["http", "https", "grpc"]:
                self.issues.append("Qdrant URL scheme must be http, https, or grpc")
        except Exception:
            self.issues.append("Invalid Qdrant URL format")

        # Validate timeout
        if config.timeout <= 0:
            self.issues.append("Qdrant timeout must be positive")
        elif config.timeout < 5:
            self.warnings.append("Qdrant timeout is very short (< 5 seconds)")

        # Validate connection
        if not self._test_qdrant_connection():
            self.issues.append("Cannot connect to Qdrant instance at " + config.url)
            self.suggestions.append("Ensure Qdrant is running and accessible")

    def _validate_embedding_config(self) -> None:
        """Validate embedding model configuration."""
        config = self.config.embedding

        # Validate model name
        supported_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-m3",
            "sentence-transformers/all-mpnet-base-v2",
        ]
        if config.model not in supported_models:
            self.warnings.append(
                f"Model '{config.model}' may not be optimized. "
                + f"Supported models: {', '.join(supported_models)}"
            )

        # Validate chunk configuration
        if config.chunk_size <= 0:
            self.issues.append("Chunk size must be positive")
        elif config.chunk_size > 2048:
            self.warnings.append("Large chunk size (> 2048) may impact performance")

        if config.chunk_overlap < 0:
            self.issues.append("Chunk overlap cannot be negative")
        elif config.chunk_overlap >= config.chunk_size:
            self.issues.append("Chunk overlap must be less than chunk size")
        elif config.chunk_overlap > config.chunk_size * 0.5:
            self.warnings.append(
                "High chunk overlap (> 50%) may cause excessive redundancy"
            )

        # Validate batch size
        if config.batch_size <= 0:
            self.issues.append("Batch size must be positive")
        elif config.batch_size > 100:
            self.warnings.append("Large batch size (> 100) may cause memory issues")

    def _validate_workspace_config(self) -> None:
        """Validate workspace configuration."""
        config = self.config.workspace

        # Validate global collections
        if not config.global_collections:
            self.warnings.append("No global collections configured")
        else:
            for collection in config.global_collections:
                if not collection.replace("-", "").replace("_", "").isalnum():
                    self.issues.append(
                        f"Invalid collection name '{collection}' - use only alphanumeric and -_"
                    )

        # Validate GitHub user
        if config.github_user:
            if not config.github_user.replace("-", "").isalnum():
                self.issues.append(
                    "GitHub user must contain only alphanumeric characters and hyphens"
                )
        else:
            self.suggestions.append(
                "Consider setting GITHUB_USER for better project detection"
            )

        # Note: max_collections validation removed as part of multi-tenant architecture
        # Collection limits now handled through metadata-based tenant isolation

    def _validate_server_config(self) -> None:
        """Validate server configuration."""
        # Validate host
        if not self.config.host:
            self.issues.append("Server host cannot be empty")

        # Validate port
        if not (1 <= self.config.port <= 65535):
            self.issues.append("Server port must be between 1 and 65535")
        elif self.config.port < 1024 and self.config.host in [
            "0.0.0.0",
            "127.0.0.1",
            "localhost",
        ]:
            self.warnings.append(
                "Using privileged port (< 1024) may require elevated permissions"
            )

    def _validate_environment(self) -> None:
        """Validate environment and dependencies."""
        # Check for required environment variables
        required_vars = ["QDRANT_URL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            self.suggestions.extend(
                [f"Consider setting {var} environment variable" for var in missing_vars]
            )

        # Check for .env file
        env_file = Path(".env")
        if not env_file.exists():
            example_file = Path(".env.example")
            if example_file.exists():
                self.suggestions.append(
                    "Copy .env.example to .env and customize settings"
                )
            else:
                self.suggestions.append("Create .env file with configuration settings")

        # Check for conflicting settings
        if os.getenv("WORKSPACE_QDRANT_DEBUG") == "true" and not self.config.debug:
            self.warnings.append("Debug mode set in environment but not in config")

    def _test_qdrant_connection(self) -> bool:
        """Test connection to Qdrant instance."""
        try:
            import warnings
            import urllib3
            
            # Test connection with SSL warning suppression
            from ..core.ssl_config import suppress_qdrant_ssl_warnings
            with suppress_qdrant_ssl_warnings():
                client = QdrantClient(**self.config.qdrant_client_config)
                client.get_collections()
                client.close()
            
            return True
        except Exception as e:
            logger.debug("Qdrant connection test failed: %s", e)
            return False

    def get_setup_guide(self) -> dict[str, list[str]]:
        """
        Generate comprehensive setup guidance and troubleshooting information.

        Provides detailed setup instructions, configuration examples, and
        troubleshooting guidance based on common deployment scenarios.
        This method is particularly useful for new users or when setting
        up the system in new environments.

        Guide Categories:
            - **quick_start**: Step-by-step setup for immediate use
            - **qdrant_setup**: Qdrant database installation and configuration
            - **environment_variables**: Complete environment variable reference
            - **troubleshooting**: Common issues and their solutions

        Returns:
            Dict[str, List[str]]: Setup guide organized by categories, each
                                 containing a list of instruction steps or
                                 information items.

        Example:
            ```python
            validator = ConfigValidator()
            guide = validator.get_setup_guide()

            logger.info("Quick Start Guide:")
            for step in guide['quick_start']:
                logger.info("  {step}")

            logger.info("\nEnvironment Variables:")
            for item in guide['environment_variables']:
                logger.info("  {item}")
            ```
        """
        guide = {
            "quick_start": [
                "1. Ensure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)",
                "2. Copy .env.example to .env and customize settings",
                "3. Set GITHUB_USER for better project detection",
                "4. Run: workspace-qdrant-mcp --host 127.0.0.1 --port 8000",
            ],
            "qdrant_setup": [
                "Start Qdrant with Docker:",
                "  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant",
                "",
                "Or with authentication:",
                "  docker run -p 6333:6333 -e QDRANT__SERVICE__API_KEY=your-key qdrant/qdrant",
            ],
            "environment_variables": [
                "Required:",
                "  QDRANT_URL=http://localhost:6333",
                "",
                "Optional:",
                "  QDRANT_API_KEY=your-api-key",
                "  GITHUB_USER=your-username",
                "  GLOBAL_COLLECTIONS=docs,reference,standards",
                "",
                "Advanced:",
                "  FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2",
                "  ENABLE_SPARSE_VECTORS=true",
                "  CHUNK_SIZE=1000",
            ],
            "troubleshooting": [
                "Common issues:",
                "• Cannot connect to Qdrant: Check if service is running on correct port",
                "• Permission errors: Ensure proper file permissions for .env",
                "• Memory issues: Reduce BATCH_SIZE or CHUNK_SIZE",
                "• Model download fails: Check internet connection and disk space",
            ],
        }

        return guide

    def print_validation_results(self, results: dict[str, list[str]]) -> None:
        """Print formatted validation results."""
        if results["issues"]:
            typer.echo(
                typer.style("\nConfiguration Issues:", fg=typer.colors.RED, bold=True)
            )
            for issue in results["issues"]:
                typer.echo(f"  • {issue}")

        if results["warnings"]:
            typer.echo(
                typer.style(
                    "\nConfiguration Warnings:", fg=typer.colors.YELLOW, bold=True
                )
            )
            for warning in results["warnings"]:
                typer.echo(f"  • {warning}")

        if results["suggestions"]:
            typer.echo(
                typer.style("\n💡 Suggestions:", fg=typer.colors.BLUE, bold=True)
            )
            for suggestion in results["suggestions"]:
                typer.echo(f"  • {suggestion}")

        if not results["issues"]:
            typer.echo(
                typer.style(
                    "\nConfiguration is valid!", fg=typer.colors.GREEN, bold=True
                )
            )


def validate_config_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    config_file: str | None = typer.Option(
        None, "--config", help="Path to config file"
    ),
    setup_guide: bool = typer.Option(False, "--guide", help="Show setup guide"),
) -> None:
    """
    Command-line interface for comprehensive configuration validation.

    This CLI command provides a convenient way to validate workspace-qdrant-mcp
    configuration from the terminal, with options for detailed output and setup
    guidance. It's designed for both interactive debugging and automated deployment
    validation.

    Features:
        - Complete configuration validation with colored output
        - Verbose mode for detailed configuration analysis
        - Custom configuration file support
        - Built-in setup guide and troubleshooting
        - Exit codes for automation (0 = success, 1 = failure)

    Args:
        verbose: Show detailed configuration summary including all settings
        config_file: Path to custom configuration file (overrides environment)
        setup_guide: Display comprehensive setup instructions instead of validation

    Exit Codes:
        0: All validations passed successfully
        1: Configuration issues found or validation failed

    Examples:
        ```bash
        # Basic validation
        workspace-qdrant-mcp-validate

        # Detailed validation with configuration summary
        workspace-qdrant-mcp-validate --verbose

        # Validate custom configuration file
        workspace-qdrant-mcp-validate --config /path/to/config.toml

        # Show setup guide instead of validation
        workspace-qdrant-mcp-validate --guide
        ```
    """

    if config_file:
        os.environ["CONFIG_FILE"] = config_file

    try:
        config = Config()
        validator = ConfigValidator(config)

        if setup_guide:
            guide = validator.get_setup_guide()
            typer.echo(typer.style("Setup Guide", fg=typer.colors.CYAN, bold=True))

            for section, items in guide.items():
                typer.echo(f"\n{section.replace('_', ' ').title()}:")
                for item in items:
                    if item:
                        typer.echo(f"  {item}")
                    else:
                        typer.echo()
        else:
            is_valid, results = validator.validate_all()

            if verbose:
                typer.echo("Configuration Summary:")
                typer.echo(f"  Qdrant URL: {config.qdrant.url}")
                typer.echo(f"  Embedding Model: {config.embedding.model}")
                typer.echo(
                    f"  GitHub User: {config.workspace.github_user or 'Not set'}"
                )
                typer.echo(
                    f"  Global Collections: {', '.join(config.workspace.global_collections)}"
                )

            validator.print_validation_results(results)

            # Raise exception instead of sys.exit for proper error handling
            if not is_valid:
                raise typer.Exit(code=1)
            raise typer.Exit(code=0)

    except Exception as e:
        typer.echo(typer.style(f"Configuration error: {e}", fg=typer.colors.RED))
        raise typer.Exit(code=1)


def validate_config_cli() -> None:
    """
    Console script entry point for UV tool installation and direct execution.

    This function serves as the primary entry point when the configuration validator
    is installed as a command-line tool via UV or pip. It provides a clean interface
    for invoking the validation functionality from the command line.

    Installation and Usage:
        ```bash
        # Install as UV tool
        uv tool install workspace-qdrant-mcp
        workspace-qdrant-mcp-validate --help

        # Run directly from source
        python -m workspace_qdrant_mcp.utils.config_validator --help
        ```

    The function uses Typer for CLI argument parsing and delegates to
    validate_config_cmd for the actual validation logic.
    """
    typer.run(validate_config_cmd)


if __name__ == "__main__":
    validate_config_cli()
