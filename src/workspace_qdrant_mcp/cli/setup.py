"""
Interactive setup wizard for workspace-qdrant-mcp.

This module provides a comprehensive setup wizard that guides users through
initial configuration, service connectivity testing, and Claude Desktop integration.
It creates .env files with validated settings and helps users get up and running
quickly with their workspace-qdrant-mcp installation.

Key Features:
    - Interactive prompts for all configuration options
    - Automatic service discovery and connectivity testing
    - Configuration validation with helpful error messages
    - Claude Desktop configuration file generation
    - Sample document creation and ingestion
    - Final system verification and health checks

The wizard walks users through:
    1. Qdrant server configuration and testing
    2. Embedding model selection and validation
    3. Workspace and collection setup
    4. Claude Desktop/Code integration
    5. Sample document creation
    6. Final system verification

Example:
    ```bash
    # Run interactive setup wizard
    workspace-qdrant-setup

    # Non-interactive mode with defaults
    workspace-qdrant-setup --non-interactive

    # Advanced mode with all options
    workspace-qdrant-setup --advanced
    ```
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from ..core.client import QdrantWorkspaceClient
from ..core.config import Config, EmbeddingConfig, QdrantConfig, WorkspaceConfig
from ..core.embeddings import EmbeddingService
from ..utils.config_validator import ConfigValidator
from ..utils.project_detection import ProjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()

# Typer app instance
app = typer.Typer(
    name="workspace-qdrant-setup",
    help="Interactive setup wizard for workspace-qdrant-mcp",
    no_args_is_help=False,
)


class SetupResult:
    """Result of setup wizard operation."""

    def __init__(self, success: bool, message: str, config_path: Optional[Path] = None):
        self.success = success
        self.message = message
        self.config_path = config_path


class SetupWizard:
    """
    Interactive setup wizard for workspace-qdrant-mcp.

    Provides a user-friendly interface for configuring all aspects of the
    workspace-qdrant-mcp system, including Qdrant connectivity, embedding
    models, workspace settings, and Claude Desktop integration.

    The wizard follows a logical flow:
        1. Welcome and system requirements check
        2. Qdrant server configuration and testing
        3. Embedding model selection and validation
        4. Workspace configuration (collections, GitHub user)
        5. Claude Desktop integration setup
        6. Sample document creation
        7. Final verification and next steps

    Attributes:
        console: Rich console for output formatting
        advanced_mode: Whether to show advanced configuration options
        non_interactive: Whether to use defaults without prompts
        config: Current configuration being built
        project_detector: Project detection service
    """

    def __init__(self, advanced_mode: bool = False, non_interactive: bool = False):
        self.console = console
        self.advanced_mode = advanced_mode
        self.non_interactive = non_interactive
        self.config = None
        self.project_detector = ProjectDetector()

    async def run_interactive_setup(self) -> SetupResult:
        """Run the complete interactive setup process.

        Returns:
            SetupResult: Success status, message, and config file path
        """
        try:
            # Welcome message
            self._show_welcome()

            if not self.non_interactive:
                if not Confirm.ask("\n🚀 Ready to set up workspace-qdrant-mcp?"):
                    return SetupResult(False, "Setup cancelled by user")

            # System requirements check
            console.print("\n📋 Checking system requirements...", style="blue")
            req_result = await self._check_requirements()
            if not req_result:
                return SetupResult(False, "System requirements check failed")

            # Build configuration step by step
            console.print("\n⚙️  Building configuration...", style="blue")

            # 1. Qdrant configuration
            qdrant_config = await self._configure_qdrant()
            if not qdrant_config:
                return SetupResult(False, "Qdrant configuration failed")

            # 2. Embedding configuration
            embedding_config = await self._configure_embedding()
            if not embedding_config:
                return SetupResult(False, "Embedding configuration failed")

            # 3. Workspace configuration
            workspace_config = await self._configure_workspace()
            if not workspace_config:
                return SetupResult(False, "Workspace configuration failed")

            # Create complete configuration
            self.config = Config(
                qdrant=qdrant_config,
                embedding=embedding_config,
                workspace=workspace_config,
            )

            # 4. Test complete configuration
            console.print("\n🔍 Testing complete configuration...", style="blue")
            test_result = await self._test_configuration()
            if not test_result:
                return SetupResult(False, "Configuration testing failed")

            # 5. Save configuration
            console.print("\n💾 Saving configuration...", style="blue")
            config_path = await self._save_configuration()
            if not config_path:
                return SetupResult(False, "Failed to save configuration")

            # 6. Claude Desktop integration
            console.print("\n🔧 Setting up Claude Desktop integration...", style="blue")
            claude_result = await self._setup_claude_integration()

            # 7. Create sample documents
            if not self.non_interactive:
                if Confirm.ask(
                    "\n📚 Would you like to create sample documents for testing?"
                ):
                    console.print("\n📄 Creating sample documents...", style="blue")
                    sample_result = await self._create_sample_documents()
                    if sample_result:
                        console.print(
                            "✅ Sample documents created successfully", style="green"
                        )

            # 8. Final verification
            console.print("\n✨ Running final system verification...", style="blue")
            await self._verify_installation()

            # Success message
            self._show_completion_message(config_path, claude_result)

            return SetupResult(True, "Setup completed successfully", config_path)

        except KeyboardInterrupt:
            console.print("\n❌ Setup cancelled by user", style="red")
            return SetupResult(False, "Setup cancelled by user")
        except Exception as e:
            console.print(f"\n❌ Setup failed: {e}", style="red")
            logger.error(f"Setup failed: {e}", exc_info=True)
            return SetupResult(False, f"Setup failed: {e}")

    def _show_welcome(self) -> None:
        """Display welcome message and introduction."""
        welcome_text = Text()
        welcome_text.append("Welcome to the ", style="white")
        welcome_text.append("workspace-qdrant-mcp", style="bold cyan")
        welcome_text.append(" setup wizard!\n\n", style="white")
        welcome_text.append("This wizard will guide you through:", style="white")
        welcome_text.append(
            "\n• Qdrant server configuration and testing", style="green"
        )
        welcome_text.append("\n• Embedding model selection", style="green")
        welcome_text.append("\n• Workspace and collection setup", style="green")
        welcome_text.append("\n• Claude Desktop/Code integration", style="green")
        welcome_text.append("\n• Sample document creation", style="green")
        welcome_text.append("\n• System verification", style="green")

        panel = Panel(
            welcome_text, title="🚀 Setup Wizard", border_style="blue", padding=(1, 2)
        )
        console.print(panel)

    async def _check_requirements(self) -> bool:
        """Check system requirements."""
        requirements = [
            ("Python version", self._check_python_version()),
            ("Required packages", self._check_packages()),
            ("Working directory", self._check_working_directory()),
        ]

        table = Table(title="System Requirements")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        all_ok = True
        for name, (status, details) in requirements:
            status_icon = "✅" if status else "❌"
            status_text = "OK" if status else "FAILED"

            table.add_row(
                name,
                f"{status_icon} {status_text}",
                details,
            )

            if not status:
                all_ok = False

        console.print(table)

        if not all_ok:
            console.print(
                "\n❌ Some requirements are not met. Please fix the issues above.",
                style="red",
            )

        return all_ok

    def _check_python_version(self) -> tuple[bool, str]:
        """Check Python version compatibility."""
        import sys

        version = sys.version_info
        if version >= (3, 10):
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        return (
            False,
            f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.10)",
        )

    def _check_packages(self) -> tuple[bool, str]:
        """Check required packages are available."""
        required_packages = ["qdrant_client", "fastembed", "pydantic", "typer", "rich"]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            return False, f"Missing: {', '.join(missing)}"
        return True, "All required packages available"

    def _check_working_directory(self) -> tuple[bool, str]:
        """Check working directory permissions."""
        try:
            # Check if we can write to current directory
            test_file = Path(".workspace_qdrant_test")
            test_file.write_text("test")
            test_file.unlink()
            return True, f"Write access to {Path.cwd()}"
        except Exception as e:
            return False, f"No write access: {e}"

    async def _configure_qdrant(self) -> Optional[QdrantConfig]:
        """Configure Qdrant database connection."""
        console.print("\n🗄️  Qdrant Database Configuration", style="bold blue")
        console.print("Configure your Qdrant vector database connection.\n")

        if self.non_interactive:
            return QdrantConfig()  # Use defaults

        # URL configuration
        default_url = "http://localhost:6333"
        url = Prompt.ask("Qdrant server URL", default=default_url, show_default=True)

        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                console.print("❌ Invalid URL format", style="red")
                return None
        except Exception:
            console.print("❌ Invalid URL format", style="red")
            return None

        # API key (optional)
        api_key = None
        if Confirm.ask("Does your Qdrant server require an API key?", default=False):
            api_key = Prompt.ask("API key", password=True, show_default=False)

        # Advanced options
        timeout = 30
        prefer_grpc = False

        if self.advanced_mode:
            timeout = IntPrompt.ask(
                "Connection timeout (seconds)", default=30, show_default=True
            )
            prefer_grpc = Confirm.ask("Prefer gRPC protocol?", default=False)

        # Create and test configuration
        qdrant_config = QdrantConfig(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc
        )

        # Test connection
        console.print("\n🔍 Testing Qdrant connection...", style="blue")
        connection_ok, message = await self._test_qdrant_connection(qdrant_config)

        if connection_ok:
            console.print(f"✅ {message}", style="green")
            return qdrant_config
        else:
            console.print(f"❌ {message}", style="red")

            if Confirm.ask("\nWould you like to try a different configuration?"):
                return await self._configure_qdrant()
            return None

    async def _test_qdrant_connection(self, config: QdrantConfig) -> tuple[bool, str]:
        """Test Qdrant database connection."""
        try:
            from qdrant_client import QdrantClient

            client_config = {
                "url": config.url,
                "timeout": config.timeout,
                "prefer_grpc": config.prefer_grpc,
            }

            if config.api_key:
                client_config["api_key"] = config.api_key

            client = QdrantClient(**client_config)

            # Test connection with a simple operation
            collections = await asyncio.get_event_loop().run_in_executor(
                None, client.get_collections
            )

            client.close()

            return (
                True,
                f"Connected successfully to {config.url} ({len(collections.collections)} collections found)",
            )

        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    async def _configure_embedding(self) -> Optional[EmbeddingConfig]:
        """Configure embedding service."""
        console.print("\n🧠 Embedding Model Configuration", style="bold blue")
        console.print("Configure text embedding generation settings.\n")

        if self.non_interactive:
            return EmbeddingConfig()  # Use defaults

        # Model selection
        available_models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Default - fast and good quality
            "sentence-transformers/all-mpnet-base-v2",  # Better quality, slower
            "BAAI/bge-small-en-v1.5",  # BGE models are excellent
            "BAAI/bge-base-en-v1.5",
        ]

        console.print("Available embedding models:")
        for i, model in enumerate(available_models, 1):
            style = "bold green" if i == 1 else "white"
            quality = (
                "⚡ Fast, Good Quality (Recommended)"
                if i == 1
                else ("🔥 High Quality, Slower" if i == 2 else "⭐ Excellent Quality")
            )
            console.print(f"  {i}. {model} - {quality}", style=style)

        if self.advanced_mode:
            choice = IntPrompt.ask(
                "\nSelect model (1-4) or press Enter for default",
                default=1,
                show_default=False,
            )
        else:
            choice = IntPrompt.ask("\nSelect model (1-4)", default=1, show_default=True)

        if 1 <= choice <= len(available_models):
            model = available_models[choice - 1]
        else:
            model = available_models[0]  # Default

        # Sparse vectors
        enable_sparse = True
        if self.advanced_mode:
            enable_sparse = Confirm.ask(
                "\nEnable sparse vectors for hybrid search?", default=True
            )
            console.print(
                "💡 Sparse vectors improve search quality but add ~30% processing time",
                style="dim",
            )

        # Text processing settings
        chunk_size = 1000
        chunk_overlap = 200
        batch_size = 50

        if self.advanced_mode:
            console.print("\n📝 Text Processing Settings")
            chunk_size = IntPrompt.ask(
                "Chunk size (characters)", default=1000, show_default=True
            )
            chunk_overlap = IntPrompt.ask(
                "Chunk overlap (characters)", default=200, show_default=True
            )
            batch_size = IntPrompt.ask(
                "Batch size (documents per batch)", default=50, show_default=True
            )

        # Create and test configuration
        embedding_config = EmbeddingConfig(
            model=model,
            enable_sparse_vectors=enable_sparse,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )

        # Test embedding model
        console.print("\n🔍 Testing embedding model...", style="blue")
        embedding_ok, message = await self._test_embedding_model(embedding_config)

        if embedding_ok:
            console.print(f"✅ {message}", style="green")
            return embedding_config
        else:
            console.print(f"❌ {message}", style="red")

            if Confirm.ask("\nWould you like to try a different model?"):
                return await self._configure_embedding()
            return None

    async def _test_embedding_model(self, config: EmbeddingConfig) -> tuple[bool, str]:
        """Test embedding model initialization."""
        try:
            # Create temporary config for testing
            temp_config = Config(embedding=config)
            embedding_service = EmbeddingService(temp_config)

            # Test model initialization
            await embedding_service.initialize()

            # Test embedding generation
            test_text = "This is a test document for embedding generation."
            embeddings = await embedding_service.generate_embeddings([test_text])

            if embeddings and len(embeddings) > 0:
                await embedding_service.close()
                return (
                    True,
                    f"Model '{config.model}' loaded successfully (embedding dimension: {len(embeddings[0])})",
                )
            else:
                await embedding_service.close()
                return False, "Model loaded but failed to generate embeddings"

        except Exception as e:
            return False, f"Model initialization failed: {str(e)}"

    async def _configure_workspace(self) -> Optional[WorkspaceConfig]:
        """Configure workspace settings."""
        console.print("\n🏗️  Workspace Configuration", style="bold blue")
        console.print("Configure workspace collections and project settings.\n")

        if self.non_interactive:
            return WorkspaceConfig()  # Use defaults

        # Detect current project
        project_info = self.project_detector.get_project_info()
        if project_info and project_info.get("main_project"):
            console.print(
                f"📁 Detected project: {project_info['main_project']}", style="green"
            )
            if project_info.get("subprojects"):
                console.print(
                    f"📂 Subprojects: {', '.join(project_info['subprojects'])}",
                    style="cyan",
                )

        # GitHub user for project detection
        github_user = None
        if Confirm.ask(
            "\nDo you have a GitHub username for project detection?", default=True
        ):
            github_user = Prompt.ask("GitHub username", show_default=False)

        # Collection configuration
        collections = ["project"]
        global_collections = ["docs", "references", "standards"]

        if self.advanced_mode:
            console.print("\n📚 Collection Configuration")
            console.print("Project collections are created for each detected project.")
            console.print("Global collections are shared across all projects.\n")

            collections_str = Prompt.ask(
                "Project collection types (comma-separated)",
                default="project",
                show_default=True,
            )
            collections = [c.strip() for c in collections_str.split(",") if c.strip()]

            global_collections_str = Prompt.ask(
                "Global collections (comma-separated)",
                default="docs,references,standards",
                show_default=True,
            )
            global_collections = [
                c.strip() for c in global_collections_str.split(",") if c.strip()
            ]

            collection_prefix = (
                Prompt.ask(
                    "Collection prefix (optional)", default="", show_default=False
                )
                or ""
            )

            max_collections = IntPrompt.ask(
                "Maximum collections limit", default=100, show_default=True
            )
        else:
            collection_prefix = ""
            max_collections = 100

        return WorkspaceConfig(
            collections=collections,
            global_collections=global_collections,
            github_user=github_user,
            collection_prefix=collection_prefix,
            max_collections=max_collections,
        )

    async def _test_configuration(self) -> bool:
        """Test complete configuration."""
        try:
            validator = ConfigValidator(self.config)
            is_valid, results = await asyncio.get_event_loop().run_in_executor(
                None, validator.validate_all
            )

            if is_valid:
                console.print("✅ Configuration is valid", style="green")
                return True
            else:
                console.print("❌ Configuration validation failed:", style="red")
                for issue in results.get("issues", []):
                    console.print(f"  • {issue}", style="red")
                return False

        except Exception as e:
            console.print(f"❌ Configuration testing failed: {e}", style="red")
            return False

    async def _save_configuration(self) -> Optional[Path]:
        """Save configuration to .env file."""
        try:
            env_path = Path(".env")

            # Check if .env already exists
            if env_path.exists():
                if not self.non_interactive:
                    if not Confirm.ask("\n⚠️  .env file already exists. Overwrite?"):
                        backup_path = Path(
                            f".env.backup.{int(asyncio.get_event_loop().time())}"
                        )
                        env_path.rename(backup_path)
                        console.print(
                            f"📋 Existing .env backed up to {backup_path}",
                            style="yellow",
                        )

            # Create configuration content
            config_content = self._generate_env_content()

            # Write to file
            env_path.write_text(config_content)

            console.print(
                f"✅ Configuration saved to {env_path.absolute()}", style="green"
            )
            return env_path

        except Exception as e:
            console.print(f"❌ Failed to save configuration: {e}", style="red")
            return None

    def _generate_env_content(self) -> str:
        """Generate .env file content from configuration."""
        lines = [
            "# Workspace Qdrant MCP Configuration",
            f"# Generated by setup wizard on {asyncio.get_event_loop().time()}",
            "",
            "# Qdrant Database Configuration",
            f"WORKSPACE_QDRANT_QDRANT__URL={self.config.qdrant.url}",
        ]

        if self.config.qdrant.api_key:
            lines.append(
                f"WORKSPACE_QDRANT_QDRANT__API_KEY={self.config.qdrant.api_key}"
            )

        if self.config.qdrant.timeout != 30:
            lines.append(
                f"WORKSPACE_QDRANT_QDRANT__TIMEOUT={self.config.qdrant.timeout}"
            )

        if self.config.qdrant.prefer_grpc:
            lines.append("WORKSPACE_QDRANT_QDRANT__PREFER_GRPC=true")

        lines.extend(
            [
                "",
                "# Embedding Model Configuration",
                f"WORKSPACE_QDRANT_EMBEDDING__MODEL={self.config.embedding.model}",
                f"WORKSPACE_QDRANT_EMBEDDING__ENABLE_SPARSE_VECTORS={str(self.config.embedding.enable_sparse_vectors).lower()}",
            ]
        )

        if self.config.embedding.chunk_size != 1000:
            lines.append(
                f"WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE={self.config.embedding.chunk_size}"
            )

        if self.config.embedding.chunk_overlap != 200:
            lines.append(
                f"WORKSPACE_QDRANT_EMBEDDING__CHUNK_OVERLAP={self.config.embedding.chunk_overlap}"
            )

        if self.config.embedding.batch_size != 50:
            lines.append(
                f"WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE={self.config.embedding.batch_size}"
            )

        lines.extend(
            [
                "",
                "# Workspace Configuration",
                f"WORKSPACE_QDRANT_WORKSPACE__COLLECTIONS={','.join(self.config.workspace.collections)}",
                f"WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS={','.join(self.config.workspace.global_collections)}",
            ]
        )

        if self.config.workspace.github_user:
            lines.append(
                f"WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER={self.config.workspace.github_user}"
            )

        if self.config.workspace.collection_prefix:
            lines.append(
                f"WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX={self.config.workspace.collection_prefix}"
            )

        if self.config.workspace.max_collections != 100:
            lines.append(
                f"WORKSPACE_QDRANT_WORKSPACE__MAX_COLLECTIONS={self.config.workspace.max_collections}"
            )

        return "\n".join(lines) + "\n"

    async def _setup_claude_integration(self) -> bool:
        """Set up Claude Desktop/Code integration."""
        try:
            # Check for Claude Desktop config directory
            claude_config_dirs = [
                Path.home() / ".claude" / "claude_desktop_config.json",
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json",
                Path.home()
                / "AppData"
                / "Roaming"
                / "Claude"
                / "claude_desktop_config.json",
            ]

            claude_config_path = None
            for path in claude_config_dirs:
                if path.parent.exists():
                    claude_config_path = path
                    break

            if not claude_config_path:
                console.print(
                    "⚠️  Claude Desktop configuration directory not found",
                    style="yellow",
                )
                console.print(
                    "You'll need to manually add the MCP server configuration.",
                    style="dim",
                )
                self._show_manual_claude_config()
                return False

            # Load existing configuration
            config_data = {}
            if claude_config_path.exists():
                try:
                    config_data = json.loads(claude_config_path.read_text())
                except json.JSONDecodeError:
                    console.print(
                        "⚠️  Invalid existing Claude configuration", style="yellow"
                    )
                    config_data = {}

            # Add or update MCP server configuration
            if "mcpServers" not in config_data:
                config_data["mcpServers"] = {}

            server_config = {"command": "workspace-qdrant-mcp", "args": [], "env": {}}

            # Add environment variables if not using .env
            if not Path(".env").exists():
                server_config["env"] = {
                    "WORKSPACE_QDRANT_QDRANT__URL": self.config.qdrant.url,
                    "WORKSPACE_QDRANT_EMBEDDING__MODEL": self.config.embedding.model,
                }
                if self.config.qdrant.api_key:
                    server_config["env"]["WORKSPACE_QDRANT_QDRANT__API_KEY"] = (
                        self.config.qdrant.api_key
                    )
                if self.config.workspace.github_user:
                    server_config["env"]["WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER"] = (
                        self.config.workspace.github_user
                    )

            config_data["mcpServers"]["workspace-qdrant-mcp"] = server_config

            # Save configuration
            claude_config_path.write_text(json.dumps(config_data, indent=2))

            console.print(
                f"✅ Claude Desktop configuration updated: {claude_config_path}",
                style="green",
            )
            return True

        except Exception as e:
            console.print(f"❌ Failed to setup Claude integration: {e}", style="red")
            self._show_manual_claude_config()
            return False

    def _show_manual_claude_config(self) -> None:
        """Show manual Claude configuration instructions."""
        config_json = {
            "mcpServers": {
                "workspace-qdrant-mcp": {
                    "command": "workspace-qdrant-mcp",
                    "args": [],
                    "env": {},
                }
            }
        }

        panel_content = f"""Add this to your Claude Desktop configuration:

{json.dumps(config_json, indent=2)}

Configuration file locations:
• macOS: ~/.claude/claude_desktop_config.json
• Windows: %APPDATA%/Claude/claude_desktop_config.json
• Linux: ~/.claude/claude_desktop_config.json"""

        panel = Panel(
            panel_content,
            title="📋 Manual Claude Configuration",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print(panel)

    async def _create_sample_documents(self) -> bool:
        """Create sample documents for testing."""
        try:
            # Create sample directory
            sample_dir = Path("sample_documents")
            sample_dir.mkdir(exist_ok=True)

            # Sample documents
            samples = {
                "README.md": """# Sample Project Documentation

This is a sample README file created by the workspace-qdrant-mcp setup wizard.

## Features

- Document ingestion and search
- Hybrid search with dense and sparse vectors
- Project-aware collections
- Claude Desktop integration

## Getting Started

1. Install dependencies
2. Configure Qdrant connection
3. Run the setup wizard
4. Start using the MCP server with Claude

## Search Examples

You can now search for:
- "project documentation"
- "getting started guide"
- "installation steps"
""",
                "project_notes.txt": """Project Development Notes
========================

These are sample development notes that demonstrate
how text documents are processed and indexed.

Key Topics:
- Vector search implementation
- Embedding model selection
- Performance optimization
- Integration patterns

The workspace-qdrant-mcp system will automatically
chunk this document and generate embeddings for
efficient semantic search.
""",
                "api_reference.md": """# API Reference

## Search Functions

### semantic_search(query, collection)
Perform semantic search across documents.

**Parameters:**
- query (str): Search query text
- collection (str): Target collection name

**Returns:**
- List of matching documents with relevance scores

### hybrid_search(query, collection, alpha=0.5)
Perform hybrid search combining semantic and keyword matching.

**Parameters:**
- query (str): Search query text
- collection (str): Target collection name
- alpha (float): Weight for semantic vs keyword search

**Returns:**
- Ranked list of documents with combined scores

## Document Management

### add_document(content, metadata, collection)
Add a document to the specified collection.

### delete_document(doc_id, collection)
Remove a document from the collection.
""",
            }

            # Write sample files
            for filename, content in samples.items():
                file_path = sample_dir / filename
                file_path.write_text(content)

            console.print(
                f"✅ Created {len(samples)} sample documents in {sample_dir}/",
                style="green",
            )

            # Ingest sample documents
            if not self.non_interactive:
                if Confirm.ask(
                    "\nWould you like to ingest these sample documents now?"
                ):
                    await self._ingest_sample_documents(sample_dir)

            return True

        except Exception as e:
            console.print(f"❌ Failed to create sample documents: {e}", style="red")
            return False

    async def _ingest_sample_documents(self, sample_dir: Path) -> None:
        """Ingest sample documents into the system."""
        try:
            # Initialize client with our configuration
            client = QdrantWorkspaceClient(self.config)
            await client.initialize()

            # Determine collection name
            project_info = client.get_project_info()
            if project_info and project_info.get("main_project"):
                collection = f"{project_info['main_project']}-project"
            else:
                collection = "sample-project"

            console.print(f"📚 Ingesting documents into collection: {collection}")

            # Simple ingestion of sample files
            for file_path in sample_dir.glob("*"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text()

                        # Use the existing add_document tool
                        from ..tools.documents import add_document

                        result = await add_document(
                            content=content,
                            metadata={
                                "filename": file_path.name,
                                "filepath": str(file_path),
                                "created_by": "setup_wizard",
                                "sample_document": True,
                            },
                            collection=collection,
                        )

                        if "successfully" in result.lower():
                            console.print(f"  ✅ {file_path.name}", style="green")
                        else:
                            console.print(
                                f"  ❌ {file_path.name}: {result}", style="red"
                            )

                    except Exception as e:
                        console.print(f"  ❌ {file_path.name}: {e}", style="red")

            await client.close()
            console.print("✅ Sample documents ingested successfully", style="green")

        except Exception as e:
            console.print(f"❌ Failed to ingest sample documents: {e}", style="red")

    async def _verify_installation(self) -> bool:
        """Run final system verification."""
        try:
            # Test complete system
            client = QdrantWorkspaceClient(self.config)
            await client.initialize()

            status = await client.get_status()

            if status.get("connected"):
                console.print("✅ System verification passed", style="green")

                # Show status summary
                table = Table(title="System Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="white")

                table.add_row("Qdrant Connection", "✅ Connected")
                table.add_row("Embedding Model", "✅ Loaded")
                table.add_row(
                    "Project Detection",
                    f"✅ {status.get('current_project', 'Unknown')}",
                )
                table.add_row(
                    "Collections",
                    f"✅ {len(status.get('workspace_collections', []))} available",
                )

                console.print(table)

                await client.close()
                return True
            else:
                console.print("❌ System verification failed", style="red")
                await client.close()
                return False

        except Exception as e:
            console.print(f"❌ System verification failed: {e}", style="red")
            return False

    def _show_completion_message(self, config_path: Path, claude_success: bool) -> None:
        """Show setup completion message with next steps."""
        completion_text = Text()
        completion_text.append(
            "🎉 Setup completed successfully!\n\n", style="bold green"
        )

        completion_text.append("What's been configured:\n", style="bold white")
        completion_text.append(
            f"• Configuration saved to {config_path}\n", style="green"
        )
        completion_text.append(
            f"• Qdrant connection: {self.config.qdrant.url}\n", style="green"
        )
        completion_text.append(
            f"• Embedding model: {self.config.embedding.model}\n", style="green"
        )

        if claude_success:
            completion_text.append("• Claude Desktop integration: ✅\n", style="green")
        else:
            completion_text.append(
                "• Claude Desktop integration: ⚠️  Manual setup required\n",
                style="yellow",
            )

        completion_text.append("\nNext steps:\n", style="bold white")
        completion_text.append(
            "1. Restart Claude Desktop to load the new MCP server\n", style="cyan"
        )
        completion_text.append(
            "2. Test the connection with a simple search\n", style="cyan"
        )
        completion_text.append("3. Ingest your project documents\n", style="cyan")
        completion_text.append(
            "4. Start using semantic search in Claude!\n", style="cyan"
        )

        completion_text.append("\nUseful commands:\n", style="bold white")
        completion_text.append(
            "• workspace-qdrant-test - Test system health\n", style="dim"
        )
        completion_text.append(
            "• workspace-qdrant-ingest - Batch ingest documents\n", style="dim"
        )
        completion_text.append(
            "• workspace-qdrant-health - Monitor system status\n", style="dim"
        )

        panel = Panel(
            completion_text,
            title="🚀 Setup Complete",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)


@app.command()
def main(
    advanced: bool = typer.Option(
        False, "--advanced", help="Show advanced configuration options"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Use defaults without prompts"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run the interactive setup wizard for workspace-qdrant-mcp.

    This wizard will guide you through configuring your Qdrant connection,
    embedding models, workspace settings, and Claude Desktop integration.

    Examples:
        # Basic interactive setup
        workspace-qdrant-setup

        # Advanced mode with all options
        workspace-qdrant-setup --advanced

        # Non-interactive mode with defaults
        workspace-qdrant-setup --non-interactive
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    wizard = SetupWizard(advanced_mode=advanced, non_interactive=non_interactive)
    result = asyncio.run(wizard.run_interactive_setup())

    if result.success:
        sys.exit(0)
    else:
        console.print(f"\n❌ Setup failed: {result.message}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    app()
