"""
Configuration validation utilities.

Provides comprehensive validation and setup guidance for workspace-qdrant-mcp.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import typer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.config import Config

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates configuration and provides setup guidance.
    
    Performs comprehensive validation of all configuration settings.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
    
    def validate_all(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Perform comprehensive configuration validation.
        
        Returns:
            Tuple of (is_valid, validation_results)
        """
        self.issues.clear()
        self.warnings.clear()
        self.suggestions.clear()
        
        # Validate each component
        self._validate_qdrant_config()
        self._validate_embedding_config()
        self._validate_workspace_config()
        self._validate_server_config()
        self._validate_environment()
        
        is_valid = len(self.issues) == 0
        
        return is_valid, {
            "issues": self.issues.copy(),
            "warnings": self.warnings.copy(),
            "suggestions": self.suggestions.copy()
        }
    
    def _validate_qdrant_config(self) -> None:
        """Validate Qdrant connection configuration."""
        config = self.config.qdrant
        
        # Validate URL format
        try:
            parsed = urlparse(config.url)
            if not parsed.scheme or not parsed.hostname:
                self.issues.append("Qdrant URL must include scheme and hostname (e.g., http://localhost:6333)")
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
            "sentence-transformers/all-mpnet-base-v2"
        ]
        if config.model not in supported_models:
            self.warnings.append(f"Model '{config.model}' may not be optimized. " +
                               f"Supported models: {', '.join(supported_models)}")
        
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
            self.warnings.append("High chunk overlap (> 50%) may cause excessive redundancy")
        
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
                    self.issues.append(f"Invalid collection name '{collection}' - use only alphanumeric and -_")
        
        # Validate GitHub user
        if config.github_user:
            if not config.github_user.replace("-", "").isalnum():
                self.issues.append("GitHub user must contain only alphanumeric characters and hyphens")
        else:
            self.suggestions.append("Consider setting GITHUB_USER for better project detection")
        
        # Validate limits
        if config.max_collections <= 0:
            self.issues.append("Max collections must be positive")
        elif config.max_collections > 1000:
            self.warnings.append("High max collections limit may impact performance")
    
    def _validate_server_config(self) -> None:
        """Validate server configuration."""
        # Validate host
        if not self.config.host:
            self.issues.append("Server host cannot be empty")
        
        # Validate port
        if not (1 <= self.config.port <= 65535):
            self.issues.append("Server port must be between 1 and 65535")
        elif self.config.port < 1024 and self.config.host in ["0.0.0.0", "127.0.0.1", "localhost"]:
            self.warnings.append("Using privileged port (< 1024) may require elevated permissions")
    
    def _validate_environment(self) -> None:
        """Validate environment and dependencies."""
        # Check for required environment variables
        required_vars = ["QDRANT_URL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            self.suggestions.extend([
                f"Consider setting {var} environment variable" for var in missing_vars
            ])
        
        # Check for .env file
        env_file = Path(".env")
        if not env_file.exists():
            example_file = Path(".env.example")
            if example_file.exists():
                self.suggestions.append("Copy .env.example to .env and customize settings")
            else:
                self.suggestions.append("Create .env file with configuration settings")
        
        # Check for conflicting settings
        if os.getenv("WORKSPACE_QDRANT_DEBUG") == "true" and not self.config.debug:
            self.warnings.append("Debug mode set in environment but not in config")
    
    def _test_qdrant_connection(self) -> bool:
        """Test connection to Qdrant instance."""
        try:
            client = QdrantClient(**self.config.qdrant_client_config)
            client.get_collections()
            client.close()
            return True
        except Exception as e:
            logger.debug("Qdrant connection test failed: %s", e)
            return False
    
    def get_setup_guide(self) -> Dict[str, List[str]]:
        """Generate setup guidance based on current configuration."""
        guide = {
            "quick_start": [
                "1. Ensure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)",
                "2. Copy .env.example to .env and customize settings",
                "3. Set GITHUB_USER for better project detection",
                "4. Run: workspace-qdrant-mcp --host 127.0.0.1 --port 8000"
            ],
            "qdrant_setup": [
                "Start Qdrant with Docker:",
                "  docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant",
                "",
                "Or with authentication:",
                "  docker run -p 6333:6333 -e QDRANT__SERVICE__API_KEY=your-key qdrant/qdrant"
            ],
            "environment_variables": [
                "Required:",
                "  QDRANT_URL=http://localhost:6333",
                "",
                "Optional:",
                "  QDRANT_API_KEY=your-api-key",
                "  GITHUB_USER=your-username",
                "  GLOBAL_COLLECTIONS=docs,references,standards",
                "",
                "Advanced:",
                "  FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2",
                "  ENABLE_SPARSE_VECTORS=true",
                "  CHUNK_SIZE=1000"
            ],
            "troubleshooting": [
                "Common issues:",
                "• Cannot connect to Qdrant: Check if service is running on correct port",
                "• Permission errors: Ensure proper file permissions for .env",
                "• Memory issues: Reduce BATCH_SIZE or CHUNK_SIZE",
                "• Model download fails: Check internet connection and disk space"
            ]
        }
        
        return guide
    
    def print_validation_results(self, results: Dict[str, List[str]]) -> None:
        """Print formatted validation results."""
        if results["issues"]:
            typer.echo(typer.style("\n❌ Configuration Issues:", fg=typer.colors.RED, bold=True))
            for issue in results["issues"]:
                typer.echo(f"  • {issue}")
        
        if results["warnings"]:
            typer.echo(typer.style("\n⚠️  Configuration Warnings:", fg=typer.colors.YELLOW, bold=True))
            for warning in results["warnings"]:
                typer.echo(f"  • {warning}")
        
        if results["suggestions"]:
            typer.echo(typer.style("\n💡 Suggestions:", fg=typer.colors.BLUE, bold=True))
            for suggestion in results["suggestions"]:
                typer.echo(f"  • {suggestion}")
        
        if not results["issues"]:
            typer.echo(typer.style("\n✅ Configuration is valid!", fg=typer.colors.GREEN, bold=True))


def validate_config_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
    setup_guide: bool = typer.Option(False, "--guide", help="Show setup guide"),
) -> None:
    """Validate workspace-qdrant-mcp configuration."""
    
    if config_file:
        os.environ["CONFIG_FILE"] = config_file
    
    try:
        config = Config()
        validator = ConfigValidator(config)
        
        if setup_guide:
            guide = validator.get_setup_guide()
            typer.echo(typer.style("📚 Setup Guide", fg=typer.colors.CYAN, bold=True))
            
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
                typer.echo(f"  GitHub User: {config.workspace.github_user or 'Not set'}")
                typer.echo(f"  Global Collections: {', '.join(config.workspace.global_collections)}")
            
            validator.print_validation_results(results)
            
            sys.exit(0 if is_valid else 1)
            
    except Exception as e:
        typer.echo(typer.style(f"❌ Configuration error: {e}", fg=typer.colors.RED))
        sys.exit(1)


def validate_config_cli() -> None:
    """Console script entry point for uv tool installation."""
    typer.run(validate_config_cmd)


if __name__ == "__main__":
    validate_config_cli()