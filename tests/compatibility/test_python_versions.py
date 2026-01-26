"""
Test Python version compatibility for workspace-qdrant-mcp.

Tests core functionality across Python 3.10, 3.11, 3.12, and 3.13 to ensure
all features work consistently across supported Python versions.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from src.python.common.core.client import create_qdrant_client
from src.python.common.core.config import get_config
from src.python.common.core.hybrid_search import HybridSearchEngine
from src.python.common.core.memory import MemoryManager


class TestPythonVersionInfo:
    """Tests to verify Python version information."""

    def test_python_version_is_supported(self):
        """Verify running on a supported Python version."""
        version_info = sys.version_info
        assert version_info.major == 3, "Must be running Python 3"
        assert version_info.minor >= 10, "Must be Python 3.10 or higher"
        assert version_info.minor <= 13, "Tested up to Python 3.13"

    def test_python_version_tuple(self):
        """Verify version tuple structure."""
        version = sys.version_info
        assert isinstance(version.major, int)
        assert isinstance(version.minor, int)
        assert isinstance(version.micro, int)

    def test_sys_executable_exists(self):
        """Verify sys.executable points to valid Python."""
        executable = Path(sys.executable)
        assert executable.exists()
        assert executable.is_file()


class TestStdlibCompatibility:
    """Test standard library compatibility across versions."""

    def test_pathlib_operations(self):
        """Test pathlib Path operations."""
        path = Path("/tmp/test")
        assert path.name == "test"
        assert path.parent == Path("/tmp")
        assert str(path) == "/tmp/test"

    def test_type_hints_available(self):
        """Test that type hints work correctly."""
        from typing import Optional, Union

        def test_func(x: int | None = None) -> str | None:
            return str(x) if x is not None else None

        assert test_func(42) == "42"
        assert test_func(None) is None

    def test_dataclass_support(self):
        """Test dataclass functionality."""
        from dataclasses import dataclass, field

        @dataclass
        class TestData:
            name: str
            value: int = 0
            tags: list = field(default_factory=list)

        data = TestData(name="test", value=42)
        assert data.name == "test"
        assert data.value == 42
        assert data.tags == []

    def test_match_statement_py310_plus(self):
        """Test match statement (Python 3.10+)."""
        def classify(value: Any) -> str:
            match value:
                case int():
                    return "integer"
                case str():
                    return "string"
                case list():
                    return "list"
                case _:
                    return "other"

        assert classify(42) == "integer"
        assert classify("hello") == "string"
        assert classify([1, 2, 3]) == "list"
        assert classify(3.14) == "other"

    def test_union_type_syntax_py310_plus(self):
        """Test union type syntax with | operator (Python 3.10+)."""
        def process_value(val: int | str) -> str:
            return str(val)

        assert process_value(42) == "42"
        assert process_value("hello") == "hello"


class TestAsyncioCompatibility:
    """Test asyncio compatibility across versions."""

    @pytest.mark.asyncio
    async def test_async_await_basic(self):
        """Test basic async/await functionality."""
        async def async_func():
            return "result"

        result = await async_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context managers."""
        class AsyncResource:
            async def __aenter__(self):
                return "resource"

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        async with AsyncResource() as resource:
            assert resource == "resource"

    @pytest.mark.asyncio
    async def test_async_generator(self):
        """Test async generators."""
        async def async_gen():
            for i in range(3):
                yield i

        results = []
        async for val in async_gen():
            results.append(val)

        assert results == [0, 1, 2]


class TestCoreModuleCompatibility:
    """Test core module compatibility across Python versions."""

    def test_config_initialization(self):
        """Test config module can be initialized."""
        config = get_config()
        assert config is not None
        if isinstance(config, dict):
            assert "qdrant" in config
            assert "workspace" in config
        else:
            assert hasattr(config, "qdrant_url")
            assert hasattr(config, "project_root")

    def test_qdrant_client_initialization(self):
        """Test Qdrant client can be created."""
        # This should not fail to import/instantiate
        client = create_qdrant_client()
        assert client is not None

    def test_hybrid_search_initialization(self):
        """Test hybrid search engine can be instantiated."""
        # Skip actual Qdrant connection in compatibility tests
        # Just verify the module imports and classes exist
        from src.python.common.core.hybrid_search import (
            HybridSearchEngine,
            RRFFusionRanker,
            WeightedSumFusionRanker,
        )

        assert HybridSearchEngine is not None
        assert RRFFusionRanker is not None
        assert WeightedSumFusionRanker is not None

    def test_memory_manager_imports(self):
        """Test memory manager module imports."""
        from src.python.common.core.memory import (
            AuthorityLevel,
            MemoryCategory,
            MemoryManager,
            MemoryRule,
        )

        assert MemoryCategory is not None
        assert MemoryRule is not None
        assert MemoryManager is not None
        assert AuthorityLevel is not None


class TestDependencyVersions:
    """Test that critical dependencies work with current Python version."""

    def test_pydantic_v2_compatibility(self):
        """Test Pydantic v2 compatibility."""
        from pydantic import BaseModel, Field

        class TestModel(BaseModel):
            name: str
            value: int = Field(default=0, ge=0)

        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

        # Test v2 model_dump
        assert model.model_dump() == {"name": "test", "value": 42}

    def test_fastapi_imports(self):
        """Test FastAPI imports work."""
        from fastapi import APIRouter, FastAPI
        from pydantic import BaseModel

        app = FastAPI()
        assert app is not None

        router = APIRouter()
        assert router is not None

    def test_qdrant_client_imports(self):
        """Test Qdrant client imports."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        assert QdrantClient is not None
        assert Distance is not None
        assert VectorParams is not None
        assert PointStruct is not None

    def test_fastembed_imports(self):
        """Test FastEmbed imports."""
        from fastembed import TextEmbedding

        assert TextEmbedding is not None

    def test_gitpython_imports(self):
        """Test GitPython compatibility."""
        import git

        assert git is not None
        assert hasattr(git, "Repo")


class TestVersionSpecificFeatures:
    """Test version-specific Python features."""

    def test_py311_exception_groups(self):
        """Test ExceptionGroup (Python 3.11+)."""
        if sys.version_info >= (3, 11):
            try:
                raise ExceptionGroup(
                    "multiple errors",
                    [ValueError("error 1"), TypeError("error 2")]
                )
            except ExceptionGroup as eg:
                assert len(eg.exceptions) == 2
                assert isinstance(eg.exceptions[0], ValueError)
                assert isinstance(eg.exceptions[1], TypeError)

    def test_py311_tomllib(self):
        """Test tomllib (Python 3.11+)."""
        if sys.version_info >= (3, 11):
            import tomllib

            toml_str = """
            [project]
            name = "test"
            version = "0.1.0"
            """

            data = tomllib.loads(toml_str)
            assert data["project"]["name"] == "test"
            assert data["project"]["version"] == "0.1.0"
        else:
            # Use toml package for Python 3.10
            import toml

            toml_str = """
            [project]
            name = "test"
            version = "0.1.0"
            """

            data = toml.loads(toml_str)
            assert data["project"]["name"] == "test"

    def test_py312_type_parameter_syntax(self):
        """Test type parameter syntax (Python 3.12+)."""
        if sys.version_info >= (3, 12):
            # Python 3.12+ supports type parameter syntax
            # def func[T](x: T) -> T:
            #     return x
            # For now, just verify typing module works
            from typing import TypeVar
            T = TypeVar('T')

            def func(x: T) -> T:
                return x

            assert func(42) == 42
            assert func("hello") == "hello"


class TestErrorHandling:
    """Test error handling compatibility."""

    def test_exception_handling_compatibility(self):
        """Test exception handling works consistently."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            assert str(e) == "test error"
            assert isinstance(e, ValueError)
            assert isinstance(e, Exception)

    def test_exception_chaining(self):
        """Test exception chaining works."""
        try:
            try:
                raise ValueError("original")
            except ValueError as e:
                raise TypeError("chained") from e
        except TypeError as e:
            assert str(e) == "chained"
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "original"


def test_compatibility_matrix_info():
    """Print compatibility matrix information."""
    version = sys.version_info
    print(f"\n{'='*60}")
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {sys.platform}")
    print(f"Implementation: {sys.implementation.name}")
    print(f"{'='*60}\n")
