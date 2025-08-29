"""
Main pytest configuration and shared fixtures.

Provides common test fixtures, mock configurations, and test utilities.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import (
    Config,
    EmbeddingConfig,
    QdrantConfig,
    WorkspaceConfig,
)
from workspace_qdrant_mcp.utils.project_detection import ProjectDetector

# Test markers
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration for testing."""
    return Config(
        host="127.0.0.1",
        port=8000,
        debug=True,
        qdrant=QdrantConfig(
            url="http://localhost:6333", api_key=None, timeout=30, prefer_grpc=False
        ),
        embedding=EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            enable_sparse_vectors=True,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=50,
        ),
        workspace=WorkspaceConfig(
            global_collections=["docs", "references", "standards"],
            github_user="testuser",
            collection_prefix="test_",
            max_collections=100,
        ),
    )


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = MagicMock(spec=QdrantClient)

    # Mock common methods
    client.get_collections.return_value = models.CollectionsResponse(
        collections=[
            models.CollectionDescription(
                name="test_collection"
            )
        ]
    )

    client.search.return_value = [
        models.ScoredPoint(
            id="test_id_1",
            score=0.95,
            version=0,
            payload={"content": "Test document 1", "source": "test"},
        ),
        models.ScoredPoint(
            id="test_id_2",
            score=0.85,
            version=0,
            payload={"content": "Test document 2", "source": "test"},
        ),
    ]

    client.create_collection.return_value = True
    client.upsert.return_value = models.UpdateResult(
        operation_id=123, status=models.UpdateStatus.COMPLETED
    )

    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.initialize = AsyncMock()
    service.generate_embeddings.return_value = {
        "dense": [0.1] * 384,
        "sparse": {"indices": [1, 5, 10, 20], "values": [0.8, 0.6, 0.9, 0.7]},
    }
    service.get_model_info.return_value = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_size": 384,
        "sparse_enabled": True,
    }
    return service


@pytest.fixture
def temp_git_repo():
    """Create a temporary Git repository for testing."""
    import git

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize Git repo
        repo = git.Repo.init(temp_dir)

        # Configure user
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Add remote origin
        repo.create_remote("origin", "https://github.com/testuser/test-project.git")

        # Create initial commit
        test_file = Path(temp_dir) / "README.md"
        test_file.write_text("# Test Project")
        repo.index.add(["README.md"])
        repo.index.commit("Initial commit")

        yield temp_dir


@pytest.fixture
def temp_git_repo_with_submodules():
    """Create a temporary Git repository with submodules for testing."""
    import git

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize main repo
        main_repo = git.Repo.init(temp_dir)

        # Configure user
        main_repo.config_writer().set_value("user", "name", "Test User").release()
        main_repo.config_writer().set_value(
            "user", "email", "test@example.com"
        ).release()

        # Add remote origin
        main_repo.create_remote(
            "origin", "https://github.com/testuser/main-project.git"
        )

        # Create .gitmodules file
        gitmodules_content = """[submodule "subproject1"]
    path = subproject1
    url = https://github.com/testuser/subproject1.git
[submodule "subproject2"]
    path = libs/subproject2
    url = https://github.com/otheruser/subproject2.git
"""

        gitmodules_path = Path(temp_dir) / ".gitmodules"
        gitmodules_path.write_text(gitmodules_content)

        # Create submodule directories (simulate initialized submodules)
        (Path(temp_dir) / "subproject1").mkdir()
        (Path(temp_dir) / "libs").mkdir()
        (Path(temp_dir) / "libs" / "subproject2").mkdir()

        # Create some files in submodules
        (Path(temp_dir) / "subproject1" / "file.txt").write_text("subproject1 content")
        (Path(temp_dir) / "libs" / "subproject2" / "file.txt").write_text(
            "subproject2 content"
        )

        # Add and commit
        main_repo.index.add([".gitmodules"])
        main_repo.index.commit("Add submodules")

        yield temp_dir


@pytest.fixture
def mock_project_detector():
    """Create a mock project detector."""
    detector = MagicMock(spec=ProjectDetector)
    detector.github_user = "testuser"
    detector.get_project_name.return_value = "test-project"
    detector.get_subprojects.return_value = ["subproject1"]
    detector.get_project_and_subprojects.return_value = (
        "test-project",
        ["subproject1"],
    )
    detector.get_project_info.return_value = {
        "main_project": "test-project",
        "subprojects": ["subproject1"],
        "git_root": "/tmp/test-project",
        "remote_url": "https://github.com/testuser/test-project.git",
        "github_user": "testuser",
        "is_git_repo": True,
        "belongs_to_user": True,
        "submodule_count": 1,
    }
    return detector


@pytest.fixture
async def mock_workspace_client(
    mock_config, mock_qdrant_client, mock_embedding_service
):
    """Create a mock workspace client."""
    client = MagicMock(spec=QdrantWorkspaceClient)
    client.config = mock_config
    client.client = mock_qdrant_client
    client.embedding_service = mock_embedding_service
    client.initialized = True

    # Mock async methods
    client.initialize = AsyncMock()
    client.get_status = AsyncMock(
        return_value={
            "connected": True,
            "collections_count": 1,
            "current_project": "test-project",
        }
    )
    client.list_collections = AsyncMock(return_value=["test_collection"])
    client.get_embedding_service.return_value = mock_embedding_service

    return client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is a sample document about Python programming.",
            "metadata": {
                "source": "docs",
                "category": "programming",
                "language": "python",
            },
        },
        {
            "id": "doc2",
            "content": "Machine learning algorithms and neural networks explained.",
            "metadata": {"source": "research", "category": "ml", "language": "english"},
        },
        {
            "id": "doc3",
            "content": "Web development best practices using FastAPI framework.",
            "metadata": {
                "source": "tutorials",
                "category": "web",
                "language": "python",
            },
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return {
        "dense": [0.1, 0.2, 0.3] + [0.0] * 381,  # 384-dimensional vector
        "sparse": {"indices": [1, 5, 10, 20, 50], "values": [0.8, 0.6, 0.9, 0.7, 0.5]},
    }


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return [
        {
            "id": "doc1",
            "score": 0.95,
            "payload": {"content": "Python programming guide", "source": "docs"},
        },
        {
            "id": "doc2",
            "score": 0.85,
            "payload": {"content": "Machine learning tutorial", "source": "tutorials"},
        },
        {
            "id": "doc3",
            "score": 0.75,
            "payload": {"content": "Web development with FastAPI", "source": "guides"},
        },
    ]


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    test_env = {
        "WORKSPACE_QDRANT_HOST": "127.0.0.1",
        "WORKSPACE_QDRANT_PORT": "8000",
        "WORKSPACE_QDRANT_DEBUG": "true",
        "WORKSPACE_QDRANT_QDRANT__URL": "http://localhost:6333",
        "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER": "testuser",
    }

    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_env

    # Cleanup
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_fastmcp_app():
    """Create a mock FastMCP application."""
    from unittest.mock import MagicMock

    app = MagicMock()
    app.tool = MagicMock()
    app.run = MagicMock()

    return app


class MockQdrantCollection:
    """Mock Qdrant collection for testing."""

    def __init__(self, name: str):
        self.name = name
        self.points = {}
        self.config = {
            "params": {
                "vectors": {
                    "dense": {"size": 384, "distance": "Cosine"},
                    "sparse": {"modifier": "idf"},
                }
            }
        }

    def upsert(self, points):
        """Mock upsert operation."""
        for point in points:
            self.points[point.id] = point
        return {"status": "completed", "operation_id": 123}

    def search(self, query_vector, limit=10, **kwargs):
        """Mock search operation."""
        # Return mock results
        return [
            {
                "id": f"result_{i}",
                "score": 0.9 - (i * 0.1),
                "payload": {"content": f"Mock result {i}"},
            }
            for i in range(min(limit, len(self.points)))
        ]


@pytest.fixture
def mock_collections():
    """Create mock collections for testing."""
    return {
        "test_docs": MockQdrantCollection("test_docs"),
        "test_scratchbook": MockQdrantCollection("test_scratchbook"),
        "test_references": MockQdrantCollection("test_references"),
    }


# Async test utilities
class AsyncContextManager:
    """Mock async context manager."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


# Test data helpers
def create_test_point(point_id: str, content: str, metadata: dict = None) -> dict:
    """Create a test point for Qdrant operations."""
    return {
        "id": point_id,
        "vector": {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]},
        },
        "payload": {"content": content, **(metadata or {})},
    }


def create_test_collection_config(vector_size: int = 384) -> dict:
    """Create test collection configuration."""
    return {
        "vectors": {
            "dense": {"size": vector_size, "distance": "Cosine"},
            "sparse": {"modifier": "idf"},
        },
        "optimizers_config": {"default_segment_number": 2},
        "replication_factor": 1,
        "write_consistency_factor": 1,
    }
