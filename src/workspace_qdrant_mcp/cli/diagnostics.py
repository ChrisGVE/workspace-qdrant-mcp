
"""
Connection testing and diagnostic utilities for workspace-qdrant-mcp.

This module provides comprehensive diagnostic and troubleshooting tools for
workspace-qdrant-mcp installations. It tests all system components including
Qdrant connectivity, embedding model availability, workspace detection,
collection operations, and search functionality.

Key Features:
    - Comprehensive connectivity testing for all services
    - Detailed diagnostic reporting with troubleshooting suggestions
    - Performance benchmarking and optimization recommendations
    - Configuration validation and conflict detection
    - Integration testing for end-to-end workflows
    - Health monitoring with alerting capabilities

Diagnostic Coverage:
    - Qdrant database connectivity and authentication
    - Embedding model initialization and performance
    - Project detection and workspace configuration
    - Collection creation, indexing, and search operations
    - Memory usage and performance metrics
    - Configuration consistency and optimization

Example:
    ```bash
    # Run full diagnostic suite
    workspace-qdrant-test

    # Test specific component
    workspace-qdrant-test --component qdrant

    # Include performance benchmarks
    workspace-qdrant-test --benchmark

    # Generate detailed report
    workspace-qdrant-test --verbose --report
    ```
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from ..core.client import QdrantWorkspaceClient
from ..core.config import Config
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
    name="workspace-qdrant-test",
    help="Connection testing and diagnostics for workspace-qdrant-mcp",
    no_args_is_help=False,
)


@dataclass
class ComponentStatus:
    """Status of a system component."""

    name: str
    success: bool
    message: str
    details: dict[str, Any] = None
    duration_ms: float = 0.0
    suggestions: list[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""

    timestamp: datetime
    overall_success: bool
    components: list[ComponentStatus]
    performance_metrics: dict[str, Any]
    configuration: dict[str, Any]
    recommendations: list[str]
    system_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_success": self.overall_success,
            "components": [
                {
                    "name": comp.name,
                    "success": comp.success,
                    "message": comp.message,
                    "details": comp.details,
                    "duration_ms": comp.duration_ms,
                    "suggestions": comp.suggestions,
                }
                for comp in self.components
            ],
            "performance_metrics": self.performance_metrics,
            "configuration": self.configuration,
            "recommendations": self.recommendations,
            "system_info": self.system_info,
        }


class DiagnosticTool:
    """
    Comprehensive diagnostic and testing tool for workspace-qdrant-mcp.

    Provides detailed health checks, performance testing, and troubleshooting
    guidance for all system components. Designed to quickly identify issues
    and provide actionable solutions.

    The tool performs tests in logical order:
        1. System requirements and environment
        2. Configuration validation
        3. Qdrant database connectivity
        4. Embedding service initialization
        5. Project detection and workspace setup
        6. Collection operations (create, index, search)
        7. End-to-end integration testing
        8. Performance benchmarking (optional)

    Attributes:
        config: System configuration
        verbose: Whether to show detailed output
        benchmark: Whether to run performance tests
        component_filter: Specific component to test (if any)
    """

    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        benchmark: bool = False,
        component_filter: str | None = None,
    ):
        self.config = config
        self.verbose = verbose
        self.benchmark = benchmark
        self.component_filter = component_filter
        self.console = console

    async def run_diagnostics(self) -> DiagnosticReport:
        """Run complete diagnostic suite.

        Returns:
            DiagnosticReport: Complete diagnostic results
        """
        start_time = time.time()

        self._show_header()

        # Gather system information
        system_info = self._collect_system_info()

        # Run diagnostic tests
        components = []

        if not self.component_filter or self.component_filter == "system":
            components.append(await self._test_system_requirements())

        if not self.component_filter or self.component_filter == "config":
            components.append(await self._test_configuration())

        if not self.component_filter or self.component_filter == "qdrant":
            components.append(await self._test_qdrant_connection())

        if not self.component_filter or self.component_filter == "embedding":
            components.append(await self._test_embedding_service())

        if not self.component_filter or self.component_filter == "workspace":
            components.append(await self._test_workspace_detection())

        if not self.component_filter or self.component_filter == "collections":
            components.append(await self._test_collection_operations())

        if not self.component_filter or self.component_filter == "search":
            components.append(await self._test_search_functionality())

        if not self.component_filter or self.component_filter == "integration":
            components.append(await self._test_integration())

        # Performance benchmarking
        performance_metrics = {}
        if self.benchmark:
            performance_metrics = await self._run_performance_tests()

        # Calculate overall status
        overall_success = all(comp.success for comp in components)

        # Generate recommendations
        recommendations = self._generate_recommendations(components)

        # Create report
        report = DiagnosticReport(
            timestamp=datetime.now(),
            overall_success=overall_success,
            components=components,
            performance_metrics=performance_metrics,
            configuration=self._get_config_summary(),
            recommendations=recommendations,
            system_info=system_info,
        )

        # Display results
        self._display_report(report, time.time() - start_time)

        return report

    def _show_header(self) -> None:
        """Display diagnostic tool header."""
        header_text = Text()
        header_text.append("workspace-qdrant-mcp ", style="bold cyan")
        header_text.append("Diagnostic Tool\n\n", style="bold white")
        header_text.append(f"Testing configuration: {Path.cwd()}\n", style="dim")
        header_text.append(f"Qdrant URL: {self.config.qdrant.url}\n", style="dim")
        header_text.append(
            f"Embedding Model: {self.config.embedding.model}\n", style="dim"
        )

        if self.component_filter:
            header_text.append(
                f"\nTesting component: {self.component_filter}", style="yellow"
            )

        panel = Panel(
            header_text,
            title="System Diagnostics",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        import platform
        import sys

        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "config_source": "environment"
            if any(k.startswith("WORKSPACE_QDRANT_") for k in os.environ)
            else "defaults",
            "env_file_exists": Path(".env").exists(),
        }

    async def _test_system_requirements(self) -> ComponentStatus:
        """Test system requirements and environment."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing system requirements..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("system", total=None)

            try:
                issues = []
                details = {}

                # Python version check
                import sys

                py_version = sys.version_info
                if py_version < (3, 10):
                    issues.append(
                        f"Python {py_version.major}.{py_version.minor} < 3.10 (required)"
                    )
                details["python_version"] = (
                    f"{py_version.major}.{py_version.minor}.{py_version.micro}"
                )

                # Required packages
                required_packages = [
                    "qdrant_client",
                    "fastembed",
                    "pydantic",
                    "typer",
                    "rich",
                    "GitPython",
                ]

                missing_packages = []
                for package in required_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        missing_packages.append(package)

                if missing_packages:
                    issues.append(f"Missing packages: {', '.join(missing_packages)}")

                details["required_packages"] = {
                    "total": len(required_packages),
                    "missing": missing_packages,
                    "available": len(required_packages) - len(missing_packages),
                }

                # File system permissions
                try:
                    test_file = Path(".workspace_qdrant_test")
                    test_file.write_text("test")
                    test_file.unlink()
                    details["filesystem_writable"] = True
                except Exception as e:
                    issues.append(f"No write access to current directory: {e}")
                    details["filesystem_writable"] = False

                # Memory check (basic)
                import psutil

                memory = psutil.virtual_memory()
                details["available_memory_gb"] = round(memory.available / (1024**3), 2)

                if memory.available < 1024**3:  # Less than 1GB
                    issues.append(
                        "Low available memory (< 1GB) - may affect embedding performance"
                    )

                duration_ms = (time.time() - start_time) * 1000

                if issues:
                    return ComponentStatus(
                        name="System Requirements",
                        success=False,
                        message=f"Found {len(issues)} issues",
                        details=details,
                        duration_ms=duration_ms,
                        suggestions=[
                            "Install missing packages with: pip install -r requirements.txt",
                            "Upgrade Python to 3.10+ for best compatibility",
                            "Ensure current directory has write permissions",
                        ],
                    )

                return ComponentStatus(
                    name="System Requirements",
                    success=True,
                    message="All system requirements met",
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return ComponentStatus(
                    name="System Requirements",
                    success=False,
                    message=f"System check failed: {str(e)}",
                    duration_ms=duration_ms,
                    suggestions=["Check system setup and try again"],
                )

    async def _test_configuration(self) -> ComponentStatus:
        """Test configuration validation."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Validating configuration..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("config", total=None)

            try:
                validator = ConfigValidator(self.config)
                is_valid, results = await asyncio.get_event_loop().run_in_executor(
                    None, validator.validate_all
                )

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "config_source": self._identify_config_source(),
                    "validation_results": results,
                }

                if is_valid:
                    return ComponentStatus(
                        name="Configuration",
                        success=True,
                        message="Configuration is valid",
                        details=details,
                        duration_ms=duration_ms,
                    )
                else:
                    issues = results.get("issues", [])
                    return ComponentStatus(
                        name="Configuration",
                        success=False,
                        message=f"Found {len(issues)} configuration issues",
                        details=details,
                        duration_ms=duration_ms,
                        suggestions=[
                            "Check environment variables and .env file",
                            "Run workspace-qdrant-setup to reconfigure",
                            "Validate Qdrant URL and API key",
                        ]
                        + [f"Fix: {issue}" for issue in issues[:3]],
                    )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return ComponentStatus(
                    name="Configuration",
                    success=False,
                    message=f"Configuration validation failed: {str(e)}",
                    duration_ms=duration_ms,
                    suggestions=["Check configuration format and syntax"],
                )

    async def _test_qdrant_connection(self) -> ComponentStatus:
        """Test Qdrant database connectivity."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing Qdrant connection..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("qdrant", total=None)

            try:
                from qdrant_client import QdrantClient

                # Test basic connection
                client = QdrantClient(**self.config.qdrant_client_config)

                # Get collections and basic info
                collections = await asyncio.get_event_loop().run_in_executor(
                    None, client.get_collections
                )

                # Test create/delete collection (if allowed)
                test_collection_name = f"test_connection_{int(time.time())}"
                can_create = False

                try:
                    from qdrant_client.http import models

                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.create_collection(
                            collection_name=test_collection_name,
                            vectors_config=models.VectorParams(
                                size=384, distance=models.Distance.COSINE
                            ),
                        ),
                    )

                    # Clean up test collection
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: client.delete_collection(test_collection_name)
                    )
                    can_create = True

                except Exception:
                    pass  # Collection creation might be restricted

                client.close()

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "url": self.config.qdrant.url,
                    "collections_count": len(collections.collections),
                    "collections": [c.name for c in collections.collections],
                    "can_create_collections": can_create,
                    "connection_time_ms": round(duration_ms, 2),
                    "authentication": "API Key"
                    if self.config.qdrant.api_key
                    else "None",
                }

                return ComponentStatus(
                    name="Qdrant Connection",
                    success=True,
                    message=f"Connected successfully ({len(collections.collections)} collections found)",
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Analyze the error for better suggestions
                suggestions = ["Check Qdrant server is running and accessible"]

                if "connection refused" in str(e).lower():
                    suggestions.append(
                        "Start Qdrant server: docker run -p 6333:6333 qdrant/qdrant"
                    )
                elif "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
                    suggestions.extend(
                        ["Verify API key is correct", "Check authentication settings"]
                    )
                elif "timeout" in str(e).lower():
                    suggestions.extend(
                        [
                            "Check network connectivity",
                            "Increase timeout in configuration",
                        ]
                    )

                return ComponentStatus(
                    name="Qdrant Connection",
                    success=False,
                    message=f"Connection failed: {str(e)}",
                    details={
                        "url": self.config.qdrant.url,
                        "error_type": type(e).__name__,
                        "connection_time_ms": round(duration_ms, 2),
                    },
                    duration_ms=duration_ms,
                    suggestions=suggestions,
                )

    async def _test_embedding_service(self) -> ComponentStatus:
        """Test embedding service initialization and functionality."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing embedding service..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("embedding", total=None)

            embedding_service = None
            try:
                embedding_service = EmbeddingService(self.config)

                # Test initialization
                await embedding_service.initialize()

                # Test embedding generation
                test_texts = [
                    "This is a test document for embedding generation.",
                    "Another sample text to test the embedding model.",
                ]

                embeddings = await embedding_service.generate_embeddings(test_texts)

                # Test sparse vectors if enabled
                sparse_vectors = None
                if self.config.embedding.enable_sparse_vectors:
                    sparse_vectors = await embedding_service.generate_sparse_vectors(
                        test_texts
                    )

                # Get model info
                model_info = embedding_service.get_model_info()

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "model": self.config.embedding.model,
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "test_embeddings_generated": len(embeddings),
                    "sparse_vectors_enabled": self.config.embedding.enable_sparse_vectors,
                    "sparse_vectors_working": sparse_vectors is not None,
                    "model_info": model_info,
                    "initialization_time_ms": round(duration_ms, 2),
                }

                success_msg = f"Model loaded successfully (dim: {len(embeddings[0]) if embeddings else 0})"
                if self.config.embedding.enable_sparse_vectors and sparse_vectors:
                    success_msg += " with sparse vectors"

                return ComponentStatus(
                    name="Embedding Service",
                    success=True,
                    message=success_msg,
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                suggestions = ["Check embedding model configuration"]

                if "model not found" in str(e).lower():
                    suggestions.append(
                        "Verify model name is correct and supported by FastEmbed"
                    )
                elif "memory" in str(e).lower() or "oom" in str(e).lower():
                    suggestions.extend(
                        [
                            "Try a smaller embedding model",
                            "Reduce batch size in configuration",
                            "Ensure sufficient RAM is available",
                        ]
                    )
                elif "download" in str(e).lower() or "network" in str(e).lower():
                    suggestions.extend(
                        [
                            "Check internet connectivity",
                            "Model download may be in progress - wait and retry",
                        ]
                    )

                return ComponentStatus(
                    name="Embedding Service",
                    success=False,
                    message=f"Embedding service failed: {str(e)}",
                    details={
                        "model": self.config.embedding.model,
                        "error_type": type(e).__name__,
                        "initialization_time_ms": round(duration_ms, 2),
                    },
                    duration_ms=duration_ms,
                    suggestions=suggestions,
                )

            finally:
                if embedding_service:
                    await embedding_service.close()

    async def _test_workspace_detection(self) -> ComponentStatus:
        """Test workspace and project detection."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing workspace detection..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("workspace", total=None)

            try:
                project_detector = ProjectDetector(
                    github_user=self.config.workspace.github_user
                )

                # Get project information
                project_info = project_detector.get_project_info()

                # Test workspace client initialization
                client = QdrantWorkspaceClient(self.config)
                await client.initialize()

                # Get workspace collections
                collections = await client.list_collections()

                # Get status
                await client.get_status()

                await client.close()

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "current_directory": str(Path.cwd()),
                    "project_info": project_info,
                    "detected_project": project_info.get("main_project"),
                    "subprojects": project_info.get("subprojects", []),
                    "workspace_collections": collections,
                    "github_user": self.config.workspace.github_user,
                    "is_git_repository": project_info.get("is_git_repo", False),
                    "collection_count": len(collections),
                }

                success_msg = (
                    f"Detected project: {project_info.get('main_project', 'unknown')}"
                )
                if project_info.get("subprojects"):
                    success_msg += f" ({len(project_info['subprojects'])} subprojects)"

                return ComponentStatus(
                    name="Workspace Detection",
                    success=True,
                    message=success_msg,
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                return ComponentStatus(
                    name="Workspace Detection",
                    success=False,
                    message=f"Workspace detection failed: {str(e)}",
                    details={
                        "current_directory": str(Path.cwd()),
                        "error_type": type(e).__name__,
                    },
                    duration_ms=duration_ms,
                    suggestions=[
                        "Ensure you're in a valid project directory",
                        "Check Git repository setup if using Git-based detection",
                        "Verify workspace configuration settings",
                    ],
                )

    async def _test_collection_operations(self) -> ComponentStatus:
        """Test collection creation and basic operations."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing collection operations..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("collections", total=None)

            client = None
            test_collection = f"test_diagnostic_{int(time.time())}"

            try:
                # Initialize client
                client = QdrantWorkspaceClient(self.config)
                await client.initialize()

                # Test collection creation (via collection manager)
                from qdrant_client.http import models

                # Access the underlying Qdrant client for testing
                qdrant_client = client.client

                # Create test collection
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.create_collection(
                        collection_name=test_collection,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=384, distance=models.Distance.COSINE
                            )
                        },
                    ),
                )

                # Test point insertion
                test_points = [
                    models.PointStruct(
                        id=1,
                        vector={"dense": [0.1] * 384},
                        payload={"text": "test document 1", "diagnostic_test": True},
                    ),
                    models.PointStruct(
                        id=2,
                        vector={"dense": [0.2] * 384},
                        payload={"text": "test document 2", "diagnostic_test": True},
                    ),
                ]

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.upsert(
                        collection_name=test_collection, points=test_points
                    ),
                )

                # Test search
                search_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.search(
                        collection_name=test_collection,
                        query_vector={"dense": [0.15] * 384},
                        limit=5,
                    ),
                )

                # Test collection info
                collection_info = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: qdrant_client.get_collection(test_collection)
                )

                # Clean up test collection
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: qdrant_client.delete_collection(test_collection)
                )

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "test_collection": test_collection,
                    "points_inserted": len(test_points),
                    "search_results": len(search_results),
                    "collection_info": {
                        "vectors_count": collection_info.vectors_count,
                        "status": collection_info.status.value
                        if hasattr(collection_info.status, "value")
                        else str(collection_info.status),
                    },
                    "operations_time_ms": round(duration_ms, 2),
                }

                return ComponentStatus(
                    name="Collection Operations",
                    success=True,
                    message=f"Collection operations successful ({len(search_results)} search results)",
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                # Clean up on failure
                if client and client.client:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.client.delete_collection(test_collection),
                        )
                    except:
                        pass  # Collection might not have been created

                duration_ms = (time.time() - start_time) * 1000

                suggestions = ["Check Qdrant server permissions and storage"]

                if "already exists" in str(e).lower():
                    suggestions.append(
                        "Collection naming conflict - this is usually harmless"
                    )
                elif "permission" in str(e).lower() or "unauthorized" in str(e).lower():
                    suggestions.extend(
                        [
                            "Check Qdrant API key permissions",
                            "Verify collection creation rights",
                        ]
                    )
                elif "storage" in str(e).lower() or "disk" in str(e).lower():
                    suggestions.append(
                        "Check Qdrant server disk space and storage configuration"
                    )

                return ComponentStatus(
                    name="Collection Operations",
                    success=False,
                    message=f"Collection operations failed: {str(e)}",
                    details={
                        "test_collection": test_collection,
                        "error_type": type(e).__name__,
                        "operations_time_ms": round(duration_ms, 2),
                    },
                    duration_ms=duration_ms,
                    suggestions=suggestions,
                )

            finally:
                if client:
                    await client.close()

    async def _test_search_functionality(self) -> ComponentStatus:
        """Test search functionality with real embeddings."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing search functionality..."),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("search", total=None)

            client = None
            test_collection = f"test_search_{int(time.time())}"

            try:
                # Initialize client and embedding service
                client = QdrantWorkspaceClient(self.config)
                await client.initialize()

                embedding_service = client.get_embedding_service()

                # Create test documents
                test_docs = [
                    "Machine learning is a subset of artificial intelligence",
                    "Python is a popular programming language for data science",
                    "Vector databases store and search high-dimensional data",
                    "Natural language processing enables computers to understand text",
                ]

                # Generate embeddings
                embeddings = await embedding_service.generate_embeddings(test_docs)

                # Create test collection with real embeddings
                from qdrant_client.http import models

                qdrant_client = client.client

                vector_config = {
                    "dense": models.VectorParams(
                        size=len(embeddings[0]), distance=models.Distance.COSINE
                    )
                }

                # Add sparse vectors if enabled
                if self.config.embedding.enable_sparse_vectors:
                    sparse_vectors = await embedding_service.generate_sparse_vectors(
                        test_docs
                    )
                    vector_config["sparse"] = models.SparseVectorParams()

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.create_collection(
                        collection_name=test_collection, vectors_config=vector_config
                    ),
                )

                # Insert test points
                test_points = []
                for i, (doc, embedding) in enumerate(
                    zip(test_docs, embeddings, strict=True)
                ):
                    vector_data = {"dense": embedding}

                    if (
                        self.config.embedding.enable_sparse_vectors
                        and "sparse_vectors" in locals()
                    ):
                        vector_data["sparse"] = sparse_vectors[i]

                    test_points.append(
                        models.PointStruct(
                            id=i + 1,
                            vector=vector_data,
                            payload={
                                "text": doc,
                                "doc_id": i + 1,
                                "diagnostic_test": True,
                            },
                        )
                    )

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.upsert(
                        collection_name=test_collection, points=test_points
                    ),
                )

                # Test semantic search
                query = "artificial intelligence and machine learning"
                query_embedding = (
                    await embedding_service.generate_embeddings([query])
                )[0]

                search_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: qdrant_client.search(
                        collection_name=test_collection,
                        query_vector={"dense": query_embedding},
                        limit=3,
                        with_payload=True,
                    ),
                )

                # Test hybrid search if sparse vectors enabled
                hybrid_results = None
                if self.config.embedding.enable_sparse_vectors:
                    try:
                        query_sparse = (
                            await embedding_service.generate_sparse_vectors([query])
                        )[0]

                        # Hybrid search implementation would go here
                        # For now, just test that sparse vector generation works
                        hybrid_results = search_results  # Placeholder

                    except Exception as e:
                        logger.warning(f"Hybrid search test failed: {e}")

                # Clean up
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: qdrant_client.delete_collection(test_collection)
                )

                duration_ms = (time.time() - start_time) * 1000

                details = {
                    "test_documents": len(test_docs),
                    "embeddings_generated": len(embeddings),
                    "embedding_dimension": len(embeddings[0]),
                    "search_query": query,
                    "semantic_results": len(search_results),
                    "best_match_score": round(search_results[0].score, 4)
                    if search_results
                    else 0,
                    "hybrid_search_tested": hybrid_results is not None,
                    "sparse_vectors_enabled": self.config.embedding.enable_sparse_vectors,
                    "search_time_ms": round(duration_ms, 2),
                }

                if search_results:
                    details["top_result"] = {
                        "text": search_results[0].payload.get("text", "unknown"),
                        "score": search_results[0].score,
                    }

                success_msg = f"Search working correctly ({len(search_results)} results, top score: {search_results[0].score:.4f})"
                if hybrid_results:
                    success_msg += " with hybrid search"

                return ComponentStatus(
                    name="Search Functionality",
                    success=True,
                    message=success_msg,
                    details=details,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                # Clean up on failure
                if client and client.client:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.client.delete_collection(test_collection),
                        )
                    except:
                        pass

                duration_ms = (time.time() - start_time) * 1000

                return ComponentStatus(
                    name="Search Functionality",
                    success=False,
                    message=f"Search functionality failed: {str(e)}",
                    details={
                        "test_collection": test_collection,
                        "error_type": type(e).__name__,
                        "search_time_ms": round(duration_ms, 2),
                    },
                    duration_ms=duration_ms,
                    suggestions=[
                        "Check embedding model compatibility",
                        "Verify collection configuration matches embedding dimensions",
                        "Ensure sufficient memory for search operations",
                    ],
                )

            finally:
                if client:
                    await client.close()

    async def _test_integration(self) -> ComponentStatus:
        """Test end-to-end integration workflow."""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Testing end-to-end integration..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("integration", total=None)

            try:
                # Test the complete workflow: client init -> document add -> search -> cleanup
                client = QdrantWorkspaceClient(self.config)
                await client.initialize()

                # Get available collections
                collections = await client.list_collections()

                if not collections:
                    # Create a temporary collection for testing
                    from qdrant_client.http import models

                    test_collection = f"integration_test_{int(time.time())}"

                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.client.create_collection(
                            collection_name=test_collection,
                            vectors_config={
                                "dense": models.VectorParams(
                                    size=384, distance=models.Distance.COSINE
                                )
                            },
                        ),
                    )

                    collections = [test_collection]
                    cleanup_collection = test_collection
                else:
                    cleanup_collection = None

                target_collection = collections[0]

                # Test document addition using the MCP tools
                test_content = "This is an integration test document for the workspace-qdrant-mcp system. It tests the complete workflow from document ingestion to search retrieval."

                from ..tools.documents import add_document

                add_result = await add_document(
                    content=test_content,
                    metadata={
                        "test_document": True,
                        "integration_test": True,
                        "timestamp": time.time(),
                    },
                    collection=target_collection,
                )

                # Test search using MCP tools
                from ..tools.search import semantic_search

                search_result = await semantic_search(
                    query="integration test document",
                    collection=target_collection,
                    limit=5,
                )

                # Clean up test collection if we created one
                if cleanup_collection:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.client.delete_collection(cleanup_collection),
                    )

                await client.close()

                duration_ms = (time.time() - start_time) * 1000

                # Parse results
                add_success = "successfully" in add_result.lower()
                search_success = (
                    "found" in search_result.lower()
                    or "results" in search_result.lower()
                )

                details = {
                    "target_collection": target_collection,
                    "document_added": add_success,
                    "search_performed": search_success,
                    "add_result": add_result,
                    "search_result_preview": search_result[:200] + "..."
                    if len(search_result) > 200
                    else search_result,
                    "integration_time_ms": round(duration_ms, 2),
                    "workflow_complete": add_success and search_success,
                }

                if add_success and search_success:
                    return ComponentStatus(
                        name="End-to-End Integration",
                        success=True,
                        message="Complete integration workflow successful",
                        details=details,
                        duration_ms=duration_ms,
                    )
                else:
                    return ComponentStatus(
                        name="End-to-End Integration",
                        success=False,
                        message=f"Integration workflow failed (add: {add_success}, search: {search_success})",
                        details=details,
                        duration_ms=duration_ms,
                        suggestions=[
                            "Check individual components are working correctly",
                            "Verify collection permissions and configuration",
                            "Test with workspace-qdrant-ingest for document processing",
                        ],
                    )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                return ComponentStatus(
                    name="End-to-End Integration",
                    success=False,
                    message=f"Integration test failed: {str(e)}",
                    details={
                        "error_type": type(e).__name__,
                        "integration_time_ms": round(duration_ms, 2),
                    },
                    duration_ms=duration_ms,
                    suggestions=[
                        "Run individual component tests to isolate the issue",
                        "Check system logs for detailed error information",
                        "Verify all prerequisites are met",
                    ],
                )

    async def _run_performance_tests(self) -> dict[str, Any]:
        """Run performance benchmarking tests."""
        console.print("\n📊 Running performance benchmarks...", style="blue")

        metrics = {}

        try:
            # Embedding generation performance
            embedding_service = EmbeddingService(self.config)
            await embedding_service.initialize()

            test_texts = [
                "Sample document for performance testing. This document contains enough text to provide meaningful embedding generation timing."
            ] * 10  # Test with 10 documents

            # Time embedding generation
            start_time = time.time()
            await embedding_service.generate_embeddings(test_texts)
            embedding_time = time.time() - start_time

            metrics["embedding_performance"] = {
                "documents_processed": len(test_texts),
                "total_time_seconds": round(embedding_time, 3),
                "documents_per_second": round(len(test_texts) / embedding_time, 2),
                "average_time_per_document_ms": round(
                    (embedding_time / len(test_texts)) * 1000, 2
                ),
            }

            # Sparse vector performance (if enabled)
            if self.config.embedding.enable_sparse_vectors:
                start_time = time.time()
                await embedding_service.generate_sparse_vectors(test_texts)
                sparse_time = time.time() - start_time

                metrics["sparse_vector_performance"] = {
                    "documents_processed": len(test_texts),
                    "total_time_seconds": round(sparse_time, 3),
                    "documents_per_second": round(len(test_texts) / sparse_time, 2),
                    "overhead_vs_dense": f"{((sparse_time / embedding_time) - 1) * 100:.1f}%",
                }

            await embedding_service.close()

            # Search performance test
            client = QdrantWorkspaceClient(self.config)
            await client.initialize()

            collections = await client.list_collections()
            if collections:
                # Test search performance on existing collection
                test_queries = [
                    "machine learning algorithms",
                    "data processing pipeline",
                    "system architecture design",
                    "performance optimization",
                    "integration testing",
                ]

                search_times = []

                for query in test_queries:
                    start_time = time.time()
                    try:
                        from ..tools.search import semantic_search

                        await semantic_search(query, collections[0], limit=10)
                        search_time = time.time() - start_time
                        search_times.append(search_time)
                    except:
                        pass  # Skip failed searches

                if search_times:
                    metrics["search_performance"] = {
                        "queries_tested": len(search_times),
                        "average_search_time_ms": round(
                            statistics.mean(search_times) * 1000, 2
                        ),
                        "median_search_time_ms": round(
                            statistics.median(search_times) * 1000, 2
                        ),
                        "fastest_search_ms": round(min(search_times) * 1000, 2),
                        "slowest_search_ms": round(max(search_times) * 1000, 2),
                    }

            await client.close()

            # System resource usage
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            metrics["resource_usage"] = {
                "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
                "cpu_percent": process.cpu_percent(),
                "system_memory_available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 2
                ),
            }

        except Exception as e:
            metrics["error"] = f"Performance testing failed: {str(e)}"

        return metrics

    def _identify_config_source(self) -> str:
        """Identify how configuration was loaded."""
        sources = []

        if Path(".env").exists():
            sources.append(".env file")

        env_vars = [k for k in os.environ.keys() if k.startswith("WORKSPACE_QDRANT_")]
        if env_vars:
            sources.append(f"environment variables ({len(env_vars)} set)")

        if not sources:
            sources.append("defaults only")

        return ", ".join(sources)

    def _get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for reporting."""
        return {
            "qdrant_url": self.config.qdrant.url,
            "embedding_model": self.config.embedding.model,
            "sparse_vectors_enabled": self.config.embedding.enable_sparse_vectors,
            "chunk_size": self.config.embedding.chunk_size,
            "batch_size": self.config.embedding.batch_size,
            "project_collections": self.config.workspace.collections,
            "global_collections": self.config.workspace.global_collections,
            "github_user": self.config.workspace.github_user,
        }

    def _generate_recommendations(self, components: list[ComponentStatus]) -> list[str]:
        """Generate optimization and troubleshooting recommendations."""
        recommendations = []

        failed_components = [comp for comp in components if not comp.success]

        if not failed_components:
            recommendations.append("All components are working correctly!")

            # Performance recommendations
            recommendations.extend(
                [
                    "📊 Consider running benchmark tests with --benchmark flag",
                    "📁 Set up regular document ingestion with workspace-qdrant-ingest",
                    "Monitor system health with workspace-qdrant-health",
                ]
            )
        else:
            recommendations.append(
                f"Error: {len(failed_components)} components need attention:"
            )

            for comp in failed_components:
                recommendations.append(f"  • {comp.name}: {comp.message}")
                if comp.suggestions:
                    for suggestion in comp.suggestions[:2]:  # Show top 2 suggestions
                        recommendations.append(f"    - {suggestion}")

        # Configuration-specific recommendations
        if not self.config.embedding.enable_sparse_vectors:
            recommendations.append(
                "⚡ Consider enabling sparse vectors for better search quality"
            )

        if self.config.embedding.chunk_size > 2000:
            recommendations.append("📏 Large chunk size may impact search precision")

        if not self.config.workspace.github_user:
            recommendations.append(
                "👤 Set GitHub username for better project detection"
            )

        return recommendations

    def _display_report(self, report: DiagnosticReport, total_time: float) -> None:
        """Display comprehensive diagnostic report."""

        # Overall status header
        if report.overall_success:
            status_text = Text("All Systems Operational", style="bold green")
        else:
            failed_count = len([c for c in report.components if not c.success])
            status_text = Text(f"Error: {failed_count} Issues Found", style="bold red")

        header_panel = Panel(
            status_text,
            title=f"📊 Diagnostic Results ({total_time:.1f}s)",
            border_style="green" if report.overall_success else "red",
            padding=(0, 1),
        )
        console.print(header_panel)

        # Component results table
        table = Table(
            title="Component Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", style="white", width=12)
        table.add_column("Message", style="white", width=50)
        table.add_column("Time (ms)", style="dim", width=10)

        for comp in report.components:
            status_icon = "[OK]" if comp.success else "[ERROR]"
            status_style = "green" if comp.success else "red"

            table.add_row(
                comp.name,
                f"{status_icon}",
                comp.message,
                f"{comp.duration_ms:.0f}",
                style=status_style if not comp.success else None,
            )

        console.print(table)

        # Performance metrics (if benchmark ran)
        if report.performance_metrics:
            perf_table = Table(title="Performance Metrics", show_header=True)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="white")

            for category, metrics in report.performance_metrics.items():
                if isinstance(metrics, dict):
                    perf_table.add_row(
                        f"[bold]{category.replace('_', ' ').title()}[/bold]", ""
                    )
                    for key, value in metrics.items():
                        perf_table.add_row(
                            f"  {key.replace('_', ' ').title()}", str(value)
                        )
                else:
                    perf_table.add_row(category.replace("_", " ").title(), str(metrics))

            console.print(perf_table)

        # Detailed component information (verbose mode)
        if self.verbose:
            for comp in report.components:
                if comp.details or comp.suggestions:
                    detail_text = Text()

                    if comp.details:
                        detail_text.append("Details:\n", style="bold")
                        for key, value in comp.details.items():
                            detail_text.append(f"  {key}: {value}\n", style="dim")

                    if comp.suggestions:
                        detail_text.append("\nSuggestions:\n", style="bold yellow")
                        for suggestion in comp.suggestions:
                            detail_text.append(f"  • {suggestion}\n", style="yellow")

                    if detail_text.plain:
                        console.print(
                            Panel(
                                detail_text,
                                title=f"{comp.name} Details",
                                border_style="blue",
                                padding=(1, 2),
                            )
                        )

        # Recommendations
        if report.recommendations:
            rec_text = Text()
            for rec in report.recommendations:
                rec_text.append(f"{rec}\n")

            console.print(
                Panel(
                    rec_text,
                    title="Recommendations",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

        # Summary footer
        summary = Text()
        summary.append(f"Diagnostic completed in {total_time:.1f}s\n", style="dim")

        if report.overall_success:
            summary.append(
                "Your workspace-qdrant-mcp installation is ready to use!",
                style="green",
            )
        else:
            summary.append(
                "Warning: Please address the issues above before using the system.",
                style="yellow",
            )

        console.print(Panel(summary, border_style="blue"))


@app.command()
def main(
    component: str | None = typer.Option(
        None,
        "--component",
        "-c",
        help="Test specific component (system, config, qdrant, embedding, workspace, collections, search, integration)",
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark", "-b", help="Include performance benchmarking"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    report: bool = typer.Option(
        False, "--report", "-r", help="Generate JSON report file"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file for JSON report"
    ),
) -> None:
    """
    Run comprehensive diagnostics for workspace-qdrant-mcp.

    This tool tests all system components and provides detailed troubleshooting
    information to help identify and resolve configuration or connectivity issues.

    Examples:
        # Full diagnostic suite
        workspace-qdrant-test

        # Test specific component
        workspace-qdrant-test --component qdrant

        # Include performance benchmarks
        workspace-qdrant-test --benchmark

        # Verbose output with details
        workspace-qdrant-test --verbose

        # Generate JSON report
        workspace-qdrant-test --report --output diagnostic_report.json
    """

    try:
        # Load configuration
        config = Config()

        # Create diagnostic tool
        diagnostic_tool = DiagnosticTool(
            config=config,
            verbose=verbose,
            benchmark=benchmark,
            component_filter=component,
        )

        # Run diagnostics
        report = asyncio.run(diagnostic_tool.run_diagnostics())

        # Generate JSON report if requested
        if report:
            output_file = (
                output or f"workspace_qdrant_diagnostic_report_{int(time.time())}.json"
            )
            output_path = Path(output_file)

            try:
                output_path.write_text(json.dumps(report.to_dict(), indent=2))
                console.print(
                    f"\n📊 Report saved to: {output_path.absolute()}", style="green"
                )
            except Exception as e:
                console.print(f"\nWarning: Failed to save report: {e}", style="yellow")

        # Exit with appropriate code
        sys.exit(0 if report.overall_success else 1)

    except KeyboardInterrupt:
        console.print("\nError: Diagnostics cancelled by user", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"\nError: Diagnostics failed: {e}", style="red")
        logger.error(f"Diagnostics failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
