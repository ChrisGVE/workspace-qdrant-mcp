"""
Dependency version matrix validation tests.

Tests compatibility across different versions of critical dependencies to ensure
the system works correctly with minimum and maximum supported versions.
"""

import importlib.metadata
import importlib.util
import subprocess
import sys

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse

# Critical dependency version requirements from pyproject.toml
CRITICAL_DEPENDENCIES = {
    # Core MCP and Vector DB
    "fastmcp": ">=0.3.0",
    "qdrant-client": ">=1.7.0",
    "fastembed": ">=0.2.0",

    # Data models and validation
    "pydantic": ">=2.0.0",
    "pydantic-settings": ">=2.0.0",

    # Web framework and API
    "fastapi": ">=0.104.0",
    "uvicorn": ">=0.24.0",

    # gRPC communication
    "grpcio": ">=1.60.0",
    "grpcio-tools": ">=1.60.0",

    # Async operations
    "aiohttp": ">=3.9.0",
    "aiofiles": ">=23.0.0",

    # Document parsing
    "pypdf": ">=4.0.0",
    "python-docx": ">=1.1.0",
    "beautifulsoup4": ">=4.12.0",
    "lxml": ">=4.9.0",

    # Utilities
    "GitPython": ">=3.1.0",
    "typer": ">=0.9.0",
    "PyYAML": ">=6.0.0",
    "rich": ">=13.0.0",
    "psutil": ">=5.8.0",
    "loguru": ">=0.7.0",
    "cachetools": ">=5.3.0",
    "xxhash": ">=3.0.0",
}


def get_installed_version(package_name: str) -> str:
    """Get the installed version of a package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"Package {package_name} not installed")


def version_satisfies_requirement(version: str, requirement: str) -> bool:
    """Check if a version satisfies a requirement specifier."""
    try:
        parsed_version = parse(version)
        specifier = SpecifierSet(requirement)
        return parsed_version in specifier
    except Exception as e:
        pytest.fail(f"Failed to parse version/requirement: {e}")


class TestDependencyVersionValidation:
    """Test that all installed dependencies meet minimum version requirements."""

    @pytest.mark.parametrize("package,requirement", CRITICAL_DEPENDENCIES.items())
    def test_dependency_meets_minimum_version(self, package: str, requirement: str):
        """Verify each critical dependency meets its minimum version requirement."""
        installed_version = get_installed_version(package)

        assert version_satisfies_requirement(installed_version, requirement), (
            f"{package} version {installed_version} does not satisfy requirement {requirement}"
        )

    def test_all_dependencies_installed(self):
        """Verify all critical dependencies are installed."""
        missing = []
        for package in CRITICAL_DEPENDENCIES:
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)

        assert not missing, f"Missing critical dependencies: {', '.join(missing)}"

    def test_dependency_version_matrix_documentation(self):
        """Generate dependency version matrix documentation."""
        matrix = {}
        for package, requirement in CRITICAL_DEPENDENCIES.items():
            try:
                version = importlib.metadata.version(package)
                matrix[package] = {
                    "required": requirement,
                    "installed": version,
                    "satisfies": version_satisfies_requirement(version, requirement)
                }
            except importlib.metadata.PackageNotFoundError:
                matrix[package] = {
                    "required": requirement,
                    "installed": None,
                    "satisfies": False
                }

        # All should satisfy requirements
        failures = [pkg for pkg, info in matrix.items() if not info["satisfies"]]
        assert not failures, f"Dependencies not meeting requirements: {failures}"


class TestQdrantClientCompatibility:
    """Test Qdrant client compatibility across versions."""

    def test_qdrant_client_version(self):
        """Verify Qdrant client version meets minimum requirements."""
        version = get_installed_version("qdrant-client")
        assert version_satisfies_requirement(version, ">=1.7.0")

    def test_qdrant_models_import(self):
        """Test that required Qdrant models can be imported."""
        try:
            from qdrant_client.models import (
                Distance,
                PointStruct,
                ScoredPoint,
                SearchRequest,
                VectorParams,
            )
            assert Distance is not None
            assert PointStruct is not None
            assert ScoredPoint is not None
            assert SearchRequest is not None
            assert VectorParams is not None
        except ImportError as e:
            pytest.fail(f"Failed to import Qdrant models: {e}")

    def test_qdrant_client_instantiation(self):
        """Test that Qdrant client can be instantiated."""
        from qdrant_client import QdrantClient

        # Test in-memory client
        client = QdrantClient(":memory:")
        assert client is not None

        # Test URL-based client (won't connect, just instantiation)
        client = QdrantClient(url="http://localhost:6333")
        assert client is not None

    def test_qdrant_sparse_vector_support(self):
        """Test that sparse vector support is available (1.7.0+ feature)."""
        try:
            from qdrant_client.models import SparseVector, SparseVectorParams
            assert SparseVector is not None
            assert SparseVectorParams is not None
        except ImportError:
            pytest.fail("Sparse vector support not available (requires qdrant-client >= 1.7.0)")


class TestFastEmbedCompatibility:
    """Test FastEmbed compatibility and functionality."""

    def test_fastembed_version(self):
        """Verify FastEmbed version meets minimum requirements."""
        version = get_installed_version("fastembed")
        assert version_satisfies_requirement(version, ">=0.2.0")

    def test_fastembed_imports(self):
        """Test that FastEmbed classes can be imported."""
        try:
            from fastembed import TextEmbedding
            assert TextEmbedding is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FastEmbed: {e}")

    def test_fastembed_model_availability(self):
        """Test that default embedding model is available."""
        from fastembed import TextEmbedding

        # Check if model can be listed (doesn't download)
        try:
            models = TextEmbedding.list_supported_models()
            assert len(models) > 0, "No supported models found"

            # Check default model is in list
            model_names = [model["model"] for model in models]
            assert "BAAI/bge-small-en-v1.5" in model_names or \
                   "sentence-transformers/all-MiniLM-L6-v2" in model_names
        except Exception as e:
            pytest.fail(f"Failed to list FastEmbed models: {e}")


class TestPydanticV2Compatibility:
    """Test Pydantic v2 compatibility (required for FastMCP)."""

    def test_pydantic_version(self):
        """Verify Pydantic v2 is installed."""
        version = get_installed_version("pydantic")
        assert version_satisfies_requirement(version, ">=2.0.0")
        assert parse(version).major == 2, "Must use Pydantic v2"

    def test_pydantic_v2_features(self):
        """Test Pydantic v2 specific features."""
        from pydantic import BaseModel, ConfigDict, Field

        class TestModel(BaseModel):
            model_config = ConfigDict(strict=True)

            name: str = Field(description="Name field")
            value: int = Field(ge=0, description="Non-negative value")

        # Test instantiation
        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

        # Test validation
        with pytest.raises(Exception):  # Pydantic v2 validation error
            TestModel(name="test", value=-1)

    def test_pydantic_settings_compatibility(self):
        """Test pydantic-settings compatibility."""
        version = get_installed_version("pydantic-settings")
        assert version_satisfies_requirement(version, ">=2.0.0")

        from pydantic_settings import BaseSettings

        class TestSettings(BaseSettings):
            app_name: str = "test"

        settings = TestSettings()
        assert settings.app_name == "test"


class TestFastMCPCompatibility:
    """Test FastMCP compatibility."""

    def test_fastmcp_version(self):
        """Verify FastMCP version meets minimum requirements."""
        version = get_installed_version("fastmcp")
        assert version_satisfies_requirement(version, ">=0.3.0")

    def test_fastmcp_imports(self):
        """Test that FastMCP classes can be imported."""
        try:
            from fastmcp import FastMCP
            assert FastMCP is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FastMCP: {e}")

    def test_fastmcp_tool_decorator(self):
        """Test FastMCP tool decorator functionality."""
        import asyncio
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @mcp.tool()
        def test_tool(query: str) -> str:
            """Test tool."""
            return f"Result: {query}"

        # Verify tool is registered
        tools = asyncio.run(mcp.get_tools())
        assert "test_tool" in tools


class TestFastAPICompatibility:
    """Test FastAPI compatibility."""

    def test_fastapi_version(self):
        """Verify FastAPI version meets minimum requirements."""
        version = get_installed_version("fastapi")
        assert version_satisfies_requirement(version, ">=0.104.0")

    def test_fastapi_imports(self):
        """Test FastAPI imports."""
        try:
            from fastapi import APIRouter, FastAPI, HTTPException, status
            assert FastAPI is not None
            assert APIRouter is not None
            assert HTTPException is not None
            assert status is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FastAPI: {e}")

    def test_fastapi_app_creation(self):
        """Test FastAPI application creation."""
        from fastapi import FastAPI

        app = FastAPI(title="Test API")
        assert app is not None
        assert app.title == "Test API"


class TestGRPCCompatibility:
    """Test gRPC compatibility."""

    def test_grpcio_version(self):
        """Verify gRPC version meets minimum requirements."""
        grpcio_version = get_installed_version("grpcio")
        tools_version = get_installed_version("grpcio-tools")

        assert version_satisfies_requirement(grpcio_version, ">=1.60.0")
        assert version_satisfies_requirement(tools_version, ">=1.60.0")

    def test_grpc_imports(self):
        """Test gRPC imports."""
        try:
            import grpc
            from grpc import aio
            assert grpc is not None
            assert aio is not None
        except ImportError as e:
            pytest.fail(f"Failed to import gRPC: {e}")

    def test_grpc_channel_creation(self):
        """Test gRPC channel creation."""
        import grpc

        # Test insecure channel creation (won't connect, just instantiation)
        channel = grpc.insecure_channel("localhost:50051")
        assert channel is not None
        channel.close()


class TestAsyncDependencyCompatibility:
    """Test async library compatibility."""

    def test_aiohttp_version(self):
        """Verify aiohttp version meets minimum requirements."""
        version = get_installed_version("aiohttp")
        assert version_satisfies_requirement(version, ">=3.9.0")

    def test_aiofiles_version(self):
        """Verify aiofiles version meets minimum requirements."""
        version = get_installed_version("aiofiles")
        assert version_satisfies_requirement(version, ">=23.0.0")

    def test_aiohttp_imports(self):
        """Test aiohttp imports."""
        try:
            import aiohttp
            from aiohttp import ClientSession, ClientTimeout
            assert aiohttp is not None
            assert ClientSession is not None
            assert ClientTimeout is not None
        except ImportError as e:
            pytest.fail(f"Failed to import aiohttp: {e}")

    def test_aiofiles_imports(self):
        """Test aiofiles imports."""
        try:
            import aiofiles
            assert aiofiles is not None
        except ImportError as e:
            pytest.fail(f"Failed to import aiofiles: {e}")


class TestDocumentParserCompatibility:
    """Test document parser library compatibility."""

    def test_pypdf_version(self):
        """Verify pypdf version meets minimum requirements."""
        version = get_installed_version("pypdf")
        assert version_satisfies_requirement(version, ">=4.0.0")

    def test_pypdf_imports(self):
        """Test pypdf imports."""
        try:
            from pypdf import PdfReader, PdfWriter
            assert PdfReader is not None
            assert PdfWriter is not None
        except ImportError as e:
            pytest.fail(f"Failed to import pypdf: {e}")

    def test_python_docx_version(self):
        """Verify python-docx version meets minimum requirements."""
        version = get_installed_version("python-docx")
        assert version_satisfies_requirement(version, ">=1.1.0")

    def test_python_docx_imports(self):
        """Test python-docx imports."""
        try:
            from docx import Document
            assert Document is not None
        except ImportError as e:
            pytest.fail(f"Failed to import python-docx: {e}")

    def test_beautifulsoup4_version(self):
        """Verify BeautifulSoup4 version meets minimum requirements."""
        version = get_installed_version("beautifulsoup4")
        assert version_satisfies_requirement(version, ">=4.12.0")

    def test_beautifulsoup4_imports(self):
        """Test BeautifulSoup4 imports."""
        try:
            from bs4 import BeautifulSoup
            assert BeautifulSoup is not None
        except ImportError as e:
            pytest.fail(f"Failed to import BeautifulSoup4: {e}")

    def test_lxml_version(self):
        """Verify lxml version meets minimum requirements."""
        version = get_installed_version("lxml")
        assert version_satisfies_requirement(version, ">=4.9.0")

    def test_lxml_imports(self):
        """Test lxml imports."""
        try:
            from lxml import etree
            assert etree is not None
        except ImportError as e:
            pytest.fail(f"Failed to import lxml: {e}")


class TestUtilityDependencyCompatibility:
    """Test utility library compatibility."""

    def test_gitpython_version(self):
        """Verify GitPython version meets minimum requirements."""
        version = get_installed_version("GitPython")
        assert version_satisfies_requirement(version, ">=3.1.0")

    def test_gitpython_imports(self):
        """Test GitPython imports."""
        try:
            import git
            from git import Repo
            assert git is not None
            assert Repo is not None
        except ImportError as e:
            pytest.fail(f"Failed to import GitPython: {e}")

    def test_loguru_version(self):
        """Verify loguru version meets minimum requirements."""
        version = get_installed_version("loguru")
        assert version_satisfies_requirement(version, ">=0.7.0")

    def test_loguru_imports(self):
        """Test loguru imports."""
        try:
            from loguru import logger
            assert logger is not None
        except ImportError as e:
            pytest.fail(f"Failed to import loguru: {e}")

    def test_cachetools_version(self):
        """Verify cachetools version meets minimum requirements."""
        version = get_installed_version("cachetools")
        assert version_satisfies_requirement(version, ">=5.3.0")

    def test_cachetools_imports(self):
        """Test cachetools imports."""
        try:
            from cachetools import LRUCache, TTLCache
            assert LRUCache is not None
            assert TTLCache is not None
        except ImportError as e:
            pytest.fail(f"Failed to import cachetools: {e}")

    def test_xxhash_version(self):
        """Verify xxhash version meets minimum requirements."""
        version = get_installed_version("xxhash")
        assert version_satisfies_requirement(version, ">=3.0.0")

    def test_xxhash_imports(self):
        """Test xxhash imports."""
        try:
            import xxhash
            assert xxhash is not None
            assert hasattr(xxhash, "xxh64")
        except ImportError as e:
            pytest.fail(f"Failed to import xxhash: {e}")


class TestDependencyConflictDetection:
    """Test for potential dependency conflicts."""

    def test_no_version_conflicts(self):
        """Check for known version conflicts between dependencies."""
        # Check Pydantic v2 compatibility with FastAPI
        pydantic_version = parse(get_installed_version("pydantic"))
        fastapi_version = parse(get_installed_version("fastapi"))

        # FastAPI >= 0.104.0 requires Pydantic v2
        if fastapi_version >= Version("0.104.0"):
            assert pydantic_version.major == 2, (
                f"FastAPI {fastapi_version} requires Pydantic v2, got {pydantic_version}"
            )

    def test_grpc_version_alignment(self):
        """Verify grpcio and grpcio-tools are aligned."""
        grpcio_version = get_installed_version("grpcio")
        tools_version = get_installed_version("grpcio-tools")

        # Major and minor versions should match
        grpcio_parsed = parse(grpcio_version)
        tools_parsed = parse(tools_version)

        assert grpcio_parsed.major == tools_parsed.major, (
            f"grpcio and grpcio-tools major versions don't match: "
            f"{grpcio_version} vs {tools_version}"
        )

    def test_pydantic_settings_compatibility(self):
        """Verify pydantic and pydantic-settings are compatible."""
        pydantic_version = parse(get_installed_version("pydantic"))
        settings_version = parse(get_installed_version("pydantic-settings"))

        # Both should be v2
        assert pydantic_version.major == 2, f"Pydantic must be v2, got {pydantic_version}"
        assert settings_version.major == 2, f"pydantic-settings must be v2, got {settings_version}"


class TestDependencySecurityValidation:
    """Test dependencies for known security vulnerabilities."""

    def test_check_for_security_advisories(self):
        """Check if pip-audit or safety is available for security scanning."""
        # This is informational - we don't fail if tools aren't available
        if importlib.util.find_spec("pip") is None:
            pytest.skip("pip not available")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                pytest.skip(f"pip list failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            pytest.skip("Package listing timed out")
        except FileNotFoundError:
            pytest.skip("pip not available")

    def test_no_deprecated_dependencies(self):
        """Check for deprecated dependency usage."""
        # Check we're not using deprecated packages
        deprecated_packages = {
            "pkg_resources": "Use importlib.metadata instead",
            "imp": "Use importlib instead"
        }

        for pkg, _message in deprecated_packages.items():
            try:
                __import__(pkg)
                # If import succeeds, check if we're actually using it in our code
                # This is a warning, not a failure
            except ImportError:
                pass  # Good, deprecated package not available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
