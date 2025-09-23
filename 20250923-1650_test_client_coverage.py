"""
Focused test file to achieve 100% coverage of client.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.core.client import QdrantWorkspaceClient, create_qdrant_client
from common.core.config import Config


class SimpleConfig:
    """Minimal config for testing."""
    def __init__(self):
        self.environment = "development"
        self.qdrant = MagicMock()
        self.qdrant.url = "http://localhost:6333"
        self.qdrant_client_config = {"host": "localhost", "port": 6333}
        self.embedding = MagicMock()
        self.embedding.model = "test-model"
        self.embedding.enable_sparse_vectors = True
        self.workspace = MagicMock()
        self.workspace.github_user = "testuser"
        self.workspace.global_collections = ["scratchbook"]
        self.security = MagicMock()
        self.security.qdrant_auth_token = None
        self.security.qdrant_api_key = None


def test_client_init():
    """Test basic client initialization."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    assert client.config is config
    assert client.client is None
    assert client.initialized is False


def test_get_project_info_none():
    """Test get_project_info when none set."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = client.get_project_info()
    assert result is None


def test_get_project_context_none():
    """Test get_project_context when no project info."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)
    client.project_info = None

    result = client.get_project_context()
    assert result is None


def test_get_project_context_empty_project():
    """Test get_project_context with empty project name."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)
    client.project_info = {"main_project": ""}

    result = client.get_project_context()
    assert result is None


def test_get_project_context_valid():
    """Test get_project_context with valid project."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)
    client.project_info = {"main_project": "test-project"}

    result = client.get_project_context("docs")

    assert result["project_name"] == "test-project"
    assert result["collection_type"] == "docs"
    assert result["workspace_scope"] == "project"
    assert "project_id" in result
    assert result["tenant_namespace"] == "test-project.docs"


def test_generate_project_id():
    """Test project ID generation."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    project_id = client._generate_project_id("test-project")

    assert isinstance(project_id, str)
    assert len(project_id) == 12

    # Same input should generate same ID
    project_id2 = client._generate_project_id("test-project")
    assert project_id == project_id2


def test_get_embedding_service():
    """Test getting embedding service."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    service = client.get_embedding_service()
    assert service is client.embedding_service


def test_list_collections_not_initialized():
    """Test list_collections when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = client.list_collections()
    assert result == []


def test_create_qdrant_client_factory():
    """Test the factory function."""
    config_data = {"host": "localhost", "port": 6333}

    client = create_qdrant_client(config_data)

    assert isinstance(client, QdrantWorkspaceClient)
    assert isinstance(client.config, Config)


@pytest.mark.asyncio
async def test_get_status_not_initialized():
    """Test get_status when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    status = await client.get_status()

    assert "error" in status
    assert status["error"] == "Client not initialized"


@pytest.mark.asyncio
async def test_search_with_project_context_not_initialized():
    """Test search when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = await client.search_with_project_context(
        "test-collection",
        {"dense": [0.1] * 384}
    )

    assert "error" in result
    assert result["error"] == "Workspace client not initialized"


@pytest.mark.asyncio
async def test_ensure_collection_exists_not_initialized():
    """Test ensure_collection_exists when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    with pytest.raises(RuntimeError, match="Client not initialized"):
        await client.ensure_collection_exists("test-collection")


@pytest.mark.asyncio
async def test_ensure_collection_exists_empty_name():
    """Test ensure_collection_exists with empty name."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)
    client.initialized = True

    with pytest.raises(ValueError, match="Collection name cannot be empty"):
        await client.ensure_collection_exists("")


@pytest.mark.asyncio
async def test_create_collection_not_initialized():
    """Test create_collection when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = await client.create_collection("test-collection")

    assert "error" in result
    assert result["error"] == "Client not initialized"


def test_get_enhanced_collection_selector_not_initialized():
    """Test enhanced collection selector when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    with pytest.raises(RuntimeError, match="Client must be initialized"):
        client.get_enhanced_collection_selector()


def test_select_collections_by_type_not_initialized():
    """Test select_collections_by_type when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = client.select_collections_by_type("memory_collection")

    expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                    'project_collections', 'fallback_collections']
    for key in expected_keys:
        assert key in result
        assert result[key] == []


def test_get_searchable_collections_not_initialized():
    """Test get_searchable_collections when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    result = client.get_searchable_collections()

    assert result == []


def test_validate_collection_access_not_initialized():
    """Test collection access validation when not initialized."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)

    is_allowed, reason = client.validate_collection_access("test", "read")

    assert is_allowed is False
    assert reason == "Client not initialized"


@pytest.mark.asyncio
async def test_close_with_none_services():
    """Test close with None services."""
    config = SimpleConfig()
    client = QdrantWorkspaceClient(config)
    client.embedding_service = None
    client.client = None

    # Should not raise exception
    await client.close()

    assert client.initialized is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])