"""
Test dependency version compatibility for workspace-qdrant-mcp.

Validates that all critical dependencies can be imported and basic
functionality works across different Python versions and dependency versions.
"""

import importlib.metadata
import sys

import pytest


class TestDependencyImports:
    """Test that all dependencies can be imported."""

    def test_fastmcp_import(self):
        """Test fastmcp can be imported."""
        import fastmcp
        assert fastmcp is not None

    def test_qdrant_client_import(self):
        """Test qdrant_client can be imported."""
        import qdrant_client
        assert qdrant_client is not None
        if hasattr(qdrant_client, "__version__"):
            assert qdrant_client.__version__
        else:
            assert importlib.metadata.version("qdrant-client")

    def test_fastembed_import(self):
        """Test fastembed can be imported."""
        import fastembed
        assert fastembed is not None
        assert hasattr(fastembed, '__version__')

    def test_pydantic_import(self):
        """Test pydantic v2 can be imported."""
        import pydantic
        assert pydantic is not None
        assert hasattr(pydantic, 'VERSION')
        # Ensure we're using Pydantic v2
        major_version = int(pydantic.VERSION.split('.')[0])
        assert major_version >= 2, "Must use Pydantic v2"

    def test_fastapi_import(self):
        """Test fastapi can be imported."""
        import fastapi
        assert fastapi is not None

    def test_gitpython_import(self):
        """Test GitPython can be imported."""
        import git
        assert git is not None

    def test_typer_import(self):
        """Test typer can be imported."""
        import typer
        assert typer is not None

    def test_rich_import(self):
        """Test rich can be imported."""
        import rich
        assert rich is not None

    def test_loguru_import(self):
        """Test loguru can be imported."""
        from loguru import logger
        assert logger is not None

    def test_psutil_import(self):
        """Test psutil can be imported."""
        import psutil
        assert psutil is not None


class TestQdrantClientCompatibility:
    """Test Qdrant client dependency compatibility."""

    def test_qdrant_client_version(self):
        """Test Qdrant client version is 1.7+."""
        import qdrant_client
        from packaging import version

        if hasattr(qdrant_client, "__version__"):
            ver = version.parse(qdrant_client.__version__)
            ver_str = qdrant_client.__version__
        else:
            ver_str = importlib.metadata.version("qdrant-client")
            ver = version.parse(ver_str)
        assert ver >= version.parse("1.7.0"), \
            f"Qdrant client {ver_str} is below minimum 1.7.0"

    def test_qdrant_models_import(self):
        """Test Qdrant models can be imported."""
        from qdrant_client.models import (
            Distance,
            FieldCondition,
            Filter,
            MatchValue,
            PointStruct,
            ScoredPoint,
            VectorParams,
        )

        assert Distance is not None
        assert VectorParams is not None
        assert PointStruct is not None
        assert ScoredPoint is not None
        assert Filter is not None
        assert FieldCondition is not None
        assert MatchValue is not None

    def test_qdrant_client_instantiation(self):
        """Test Qdrant client can be instantiated."""
        from qdrant_client import QdrantClient

        # Don't actually connect, just verify instantiation works
        client = QdrantClient(url="http://localhost:6333", timeout=1)
        assert client is not None


class TestFastEmbedCompatibility:
    """Test FastEmbed dependency compatibility."""

    def test_fastembed_version(self):
        """Test FastEmbed version is 0.2+."""
        import fastembed
        from packaging import version

        ver = version.parse(fastembed.__version__)
        assert ver >= version.parse("0.2.0"), \
            f"FastEmbed {fastembed.__version__} is below minimum 0.2.0"

    def test_fastembed_text_embedding_import(self):
        """Test TextEmbedding class can be imported."""
        from fastembed import TextEmbedding

        assert TextEmbedding is not None

    def test_fastembed_supported_models(self):
        """Test FastEmbed supported models can be listed."""
        from fastembed import TextEmbedding

        # This should not raise an error
        models = TextEmbedding.list_supported_models()
        assert models is not None
        assert len(models) > 0


class TestPydanticCompatibility:
    """Test Pydantic v2 compatibility."""

    def test_pydantic_version(self):
        """Test Pydantic version is 2.0+."""
        import pydantic
        from packaging import version

        ver = version.parse(pydantic.VERSION)
        assert ver >= version.parse("2.0.0"), \
            f"Pydantic {pydantic.VERSION} is below minimum 2.0.0"

    def test_pydantic_basemodel(self):
        """Test Pydantic BaseModel works."""
        from pydantic import BaseModel, Field

        class TestModel(BaseModel):
            name: str
            value: int = Field(default=0, ge=0, le=100)

        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

        # Test model_dump (v2 method)
        data = model.model_dump()
        assert data == {"name": "test", "value": 42}

    def test_pydantic_settings(self):
        """Test Pydantic Settings works."""
        from pydantic_settings import BaseSettings

        class TestSettings(BaseSettings):
            app_name: str = "test-app"
            debug: bool = False

        settings = TestSettings()
        assert settings.app_name == "test-app"
        assert settings.debug is False


class TestFastAPICompatibility:
    """Test FastAPI compatibility."""

    def test_fastapi_version(self):
        """Test FastAPI version is 0.104+."""
        import fastapi
        from packaging import version

        ver = version.parse(fastapi.__version__)
        assert ver >= version.parse("0.104.0"), \
            f"FastAPI {fastapi.__version__} is below minimum 0.104.0"

    def test_fastapi_basic_app(self):
        """Test FastAPI app can be created."""
        from fastapi import APIRouter, FastAPI

        app = FastAPI(title="Test App")
        assert app is not None

        router = APIRouter(prefix="/api")
        assert router is not None

    def test_fastmcp_integration(self):
        """Test fastmcp integration with FastAPI."""
        try:
            import fastmcp
            from fastmcp import FastMCP

            # Test basic FastMCP instantiation
            mcp = FastMCP("test-server")
            assert mcp is not None
        except ImportError:
            pytest.skip("fastmcp not available")


class TestDocumentParsingDependencies:
    """Test document parsing dependency compatibility."""

    def test_pypdf_import(self):
        """Test pypdf can be imported."""
        import pypdf
        assert pypdf is not None

    def test_docx_import(self):
        """Test python-docx can be imported."""
        import docx
        assert docx is not None

    def test_pptx_import(self):
        """Test python-pptx can be imported."""
        import pptx
        assert pptx is not None

    def test_beautifulsoup_import(self):
        """Test beautifulsoup4 can be imported."""
        from bs4 import BeautifulSoup
        assert BeautifulSoup is not None

    def test_markdown_import(self):
        """Test markdown can be imported."""
        import markdown
        assert markdown is not None


class TestAsyncDependencies:
    """Test async-related dependencies."""

    def test_aiohttp_import(self):
        """Test aiohttp can be imported."""
        import aiohttp
        assert aiohttp is not None

    def test_aiofiles_import(self):
        """Test aiofiles can be imported."""
        import aiofiles
        assert aiofiles is not None

    @pytest.mark.asyncio
    async def test_aiohttp_session(self):
        """Test aiohttp session creation."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            assert session is not None


class TestTestingDependencies:
    """Test testing-related dependencies."""

    def test_pytest_import(self):
        """Test pytest can be imported."""
        import pytest as pt
        assert pt is not None

    def test_pytest_asyncio_import(self):
        """Test pytest-asyncio can be imported."""
        import pytest_asyncio
        assert pytest_asyncio is not None

    def test_pytest_mock_import(self):
        """Test pytest-mock can be imported."""
        try:
            import pytest_mock
            assert pytest_mock is not None
        except ImportError:
            pytest.skip("pytest-mock not available")

    def test_hypothesis_import(self):
        """Test hypothesis can be imported."""
        try:
            import hypothesis
            assert hypothesis is not None
        except ImportError:
            pytest.skip("hypothesis not available")

    def test_testcontainers_import(self):
        """Test testcontainers can be imported."""
        try:
            import testcontainers
            assert testcontainers is not None
        except ImportError:
            pytest.skip("testcontainers not available")


def test_print_dependency_versions():
    """Print versions of all critical dependencies."""
    print(f"\n{'='*60}")
    print(f"Python: {sys.version}")
    print(f"{'='*60}")

    dependencies = [
        ("qdrant-client", "qdrant_client"),
        ("fastembed", "fastembed"),
        ("pydantic", "pydantic"),
        ("fastapi", "fastapi"),
        ("GitPython", "git"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("pytest", "pytest"),
    ]

    for name, module_name in dependencies:
        try:
            module = __import__(module_name)
            version_attr = getattr(module, '__version__', None) or \
                          getattr(module, 'VERSION', None) or \
                          "unknown"
            print(f"{name:20s}: {version_attr}")
        except ImportError:
            print(f"{name:20s}: NOT INSTALLED")

    print(f"{'='*60}\n")
