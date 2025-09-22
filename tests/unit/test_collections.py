"""
Unit tests for collection management.

Tests workspace collection creation, configuration, and management.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from workspace_qdrant_mcp.core.collections import (
    CollectionConfig,
    WorkspaceCollectionManager,
)
from workspace_qdrant_mcp.core.config import Config, EmbeddingConfig, WorkspaceConfig


class TestCollectionConfig:
    """Test CollectionConfig dataclass."""

    def test_collection_config_defaults(self):
        """Test CollectionConfig with default values."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="docs",
        )

        assert config.name == "test-collection"
        assert config.description == "Test collection"
        assert config.collection_type == "docs"
        assert config.project_name is None
        assert config.vector_size == 384
        assert config.distance_metric == "Cosine"
        assert config.enable_sparse_vectors is True

    def test_collection_config_custom_values(self):
        """Test CollectionConfig with custom values."""
        config = CollectionConfig(
            name="custom-collection",
            description="Custom collection",
            collection_type="scratchbook",
            project_name="test-project",
            vector_size=768,
            distance_metric="Dot",
            enable_sparse_vectors=False,
        )

        assert config.name == "custom-collection"
        assert config.description == "Custom collection"
        assert config.collection_type == "scratchbook"
        assert config.project_name == "test-project"
        assert config.vector_size == 768
        assert config.distance_metric == "Dot"
        assert config.enable_sparse_vectors is False


class TestWorkspaceCollectionManager:
    """Test WorkspaceCollectionManager class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        return MagicMock(spec=QdrantClient)

    @pytest.fixture
    def workspace_config(self):
        """Create workspace configuration."""
        return Config(
            embedding=EmbeddingConfig(
                model="sentence-transformers/all-MiniLM-L6-v2",
                enable_sparse_vectors=True,
                chunk_size=1000,
                chunk_overlap=200,
            ),
            workspace=WorkspaceConfig(
                global_collections=["references", "standards"],
                github_user="testuser",
                collection_prefix="test_",
                max_collections=10,
                auto_create_collections=True,
            ),
        )

    @pytest.fixture
    def collection_manager(self, mock_qdrant_client, workspace_config):
        """Create WorkspaceCollectionManager instance."""
        return WorkspaceCollectionManager(mock_qdrant_client, workspace_config)

    def test_init(self, collection_manager, mock_qdrant_client, workspace_config):
        """Test WorkspaceCollectionManager initialization."""
        assert collection_manager.client == mock_qdrant_client
        assert collection_manager.config == workspace_config
        assert collection_manager._collections_cache is None

    @pytest.mark.asyncio
    async def test_initialize_workspace_collections_basic(self, collection_manager):
        """Test basic workspace collection initialization."""
        # Mock existing collections check
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )

        # Mock collection creation
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections(
            "test-project", subprojects=[]
        )

        # Verify collections were created
        assert collection_manager.client.create_collection.call_count >= 1

        # Should create at least the main project collections
        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]
        assert any("test-project" in name for name in created_names)

    @pytest.mark.asyncio
    async def test_initialize_workspace_collections_with_subprojects(
        self, collection_manager
    ):
        """Test workspace initialization with subprojects."""
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections(
            "main-project", subprojects=["sub1", "sub2"]
        )

        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]

        # Should create collections for main project and subprojects
        assert any("main-project" in name for name in created_names)
        assert any("sub1" in name for name in created_names)
        assert any("sub2" in name for name in created_names)

    @pytest.mark.asyncio
    async def test_initialize_workspace_collections_existing_collections(
        self, collection_manager
    ):
        """Test initialization when some collections already exist."""
        # Mock existing collections
        existing_collections = [
            models.CollectionDescription(
                name="test-project-docs",
                status=models.CollectionStatus.GREEN,
                vectors_count=100,
                indexed_vectors_count=100,
                points_count=50,
                segments_count=1,
                config=models.CollectionConfig(
                    params=models.CollectionParams(
                        vectors=models.VectorParams(
                            size=384, distance=models.Distance.COSINE
                        )
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16, ef_construct=100, full_scan_threshold=10000
                    ),
                    optimizer_config=models.OptimizersConfig(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=0,
                        flush_interval_sec=5,
                    ),
                ),
            )
        ]

        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=existing_collections)
        )
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections("test-project")

        # Should not recreate existing collections
        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]
        assert "test-project-docs" not in created_names

    @pytest.mark.asyncio
    async def test_initialize_workspace_collections_global_collections(
        self, collection_manager
    ):
        """Test creation of global collections."""
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections("test-project")

        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]

        # Should create global collections
        assert any("references" in name for name in created_names)
        assert any("standards" in name for name in created_names)

    @pytest.mark.skip(
        reason="create_collection method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_create_collection_success(self, collection_manager):
        """Test successful collection creation."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="docs",
            vector_size=384,
            enable_sparse_vectors=True,
        )

        collection_manager.client.create_collection = MagicMock()

        result = await collection_manager.create_collection(config)

        assert result is True
        collection_manager.client.create_collection.assert_called_once()

        # Verify collection configuration
        call_args = collection_manager.client.create_collection.call_args
        assert call_args[1]["collection_name"] == "test-collection"

        vectors_config = call_args[1]["vectors_config"]
        assert "dense" in vectors_config
        assert "sparse" in vectors_config  # Sparse vectors enabled

    @pytest.mark.skip(
        reason="create_collection method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_create_collection_without_sparse_vectors(self, collection_manager):
        """Test collection creation without sparse vectors."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="docs",
            enable_sparse_vectors=False,
        )

        collection_manager.client.create_collection = MagicMock()

        result = await collection_manager.create_collection(config)

        assert result is True

        call_args = collection_manager.client.create_collection.call_args
        vectors_config = call_args[1]["vectors_config"]
        assert "dense" in vectors_config
        assert "sparse" not in vectors_config

    @pytest.mark.skip(
        reason="create_collection method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_create_collection_failure(self, collection_manager):
        """Test collection creation failure."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="docs",
        )

        collection_manager.client.create_collection.side_effect = (
            ResponseHandlingException("Collection creation failed")
        )

        result = await collection_manager.create_collection(config)

        assert result is False

    @pytest.mark.skip(
        reason="delete_collection method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_delete_collection_success(self, collection_manager):
        """Test successful collection deletion."""
        collection_manager.client.delete_collection = MagicMock()

        result = await collection_manager.delete_collection("test-collection")

        assert result is True
        collection_manager.client.delete_collection.assert_called_once_with(
            collection_name="test-collection"
        )

    @pytest.mark.skip(
        reason="delete_collection method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_delete_collection_failure(self, collection_manager):
        """Test collection deletion failure."""
        collection_manager.client.delete_collection.side_effect = (
            ResponseHandlingException("Collection not found")
        )

        result = await collection_manager.delete_collection("nonexistent-collection")

        assert result is False

    @pytest.mark.skip(
        reason="collection_exists method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_collection_exists_true(self, collection_manager):
        """Test collection existence check - exists."""
        existing_collections = [
            models.CollectionDescription(
                name="test-collection",
                status=models.CollectionStatus.GREEN,
                vectors_count=100,
                indexed_vectors_count=100,
                points_count=50,
                segments_count=1,
            )
        ]

        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=existing_collections)
        )

        result = await collection_manager.collection_exists("test-collection")

        assert result is True

    @pytest.mark.skip(
        reason="collection_exists method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_collection_exists_false(self, collection_manager):
        """Test collection existence check - does not exist."""
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )

        result = await collection_manager.collection_exists("nonexistent-collection")

        assert result is False

    @pytest.mark.skip(
        reason="collection_exists method does not exist in actual implementation"
    )
    @pytest.mark.asyncio
    async def test_collection_exists_exception(self, collection_manager):
        """Test collection existence check with exception."""
        collection_manager.client.get_collections.side_effect = (
            ResponseHandlingException("Connection error")
        )

        result = await collection_manager.collection_exists("test-collection")

        assert result is False

    @pytest.mark.skip(
        reason="list_collections method does not exist - use list_workspace_collections instead"
    )
    @pytest.mark.asyncio
    async def test_list_collections(self, collection_manager):
        """Test listing collections."""
        existing_collections = [
            models.CollectionDescription(name="collection1"),
            models.CollectionDescription(name="collection2"),
            models.CollectionDescription(name="collection3"),
        ]

        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=existing_collections)
        )

        result = await collection_manager.list_collections()

        assert result == ["collection1", "collection2", "collection3"]

    @pytest.mark.skip(
        reason="list_collections method does not exist - use list_workspace_collections instead"
    )
    @pytest.mark.asyncio
    async def test_list_collections_empty(self, collection_manager):
        """Test listing collections when none exist."""
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )

        result = await collection_manager.list_collections()

        assert result == []

    @pytest.mark.skip(
        reason="list_collections method does not exist - use list_workspace_collections instead"
    )
    @pytest.mark.asyncio
    async def test_list_collections_filtered(self, collection_manager):
        """Test listing collections with filter."""
        existing_collections = [
            models.CollectionDescription(name="project1-docs"),
            models.CollectionDescription(name="project1-scratchbook"),
            models.CollectionDescription(name="project2-docs"),
            models.CollectionDescription(name="global-references"),
        ]

        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=existing_collections)
        )

        result = await collection_manager.list_collections(name_filter="project1")

        assert "project1-docs" in result
        assert "project1-scratchbook" in result
        assert "project2-docs" not in result
        assert "global-references" not in result

    def test_get_collection_info_success(self, collection_manager):
        """Test getting collection information."""
        collection_desc = models.CollectionInfo(
            status=models.CollectionStatus.GREEN,
            vectors_count=150,
            indexed_vectors_count=150,
            points_count=75,
            segments_count=2,
            optimizer_status=models.OptimizersStatusOneOf.OK,
            payload_schema={},    # Required field
            config=models.CollectionConfig(
                params=models.CollectionParams(
                    vectors=models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    )
                ),
                hnsw_config=models.HnswConfig(
                    m=16, ef_construct=100, full_scan_threshold=10000
                ),
                optimizer_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=0,
                    flush_interval_sec=5,
                ),
            ),
        )

        # Mock list_workspace_collections to return our test collection
        collection_manager.list_workspace_collections = MagicMock(
            return_value=["test-collection"]
        )
        collection_manager.client.get_collection.return_value = collection_desc

        result = collection_manager.get_collection_info()

        assert result is not None
        assert "collections" in result
        assert "test-collection" in result["collections"]
        assert result["collections"]["test-collection"]["vectors_count"] == 150
        assert result["collections"]["test-collection"]["points_count"] == 75
        assert result["collections"]["test-collection"]["config"]["vector_size"] == 384

    def test_get_collection_info_not_found(self, collection_manager):
        """Test getting information when collection access fails."""
        # Mock list_workspace_collections to return a collection that will fail
        collection_manager.list_workspace_collections = MagicMock(
            return_value=["failing-collection"]
        )
        collection_manager.client.get_collection.side_effect = (
            ResponseHandlingException("Collection not found")
        )

        result = collection_manager.get_collection_info()

        # Should return a dict with error information
        assert result is not None
        assert "collections" in result
        assert "failing-collection" in result["collections"]
        assert "error" in result["collections"]["failing-collection"]

    @pytest.mark.skip(
        reason="_generate_collection_name method does not exist in actual implementation"
    )
    def test_generate_collection_name_project_scoped(self, collection_manager):
        """Test project-scoped collection name generation."""
        name = collection_manager._generate_collection_name(
            "docs", project_name="test-project"
        )

        assert name == "test-project-docs"

    @pytest.mark.skip(
        reason="_generate_collection_name method does not exist in actual implementation"
    )
    def test_generate_collection_name_global(self, collection_manager):
        """Test global collection name generation."""
        name = collection_manager._generate_collection_name(
            "references", project_name=None
        )

        assert name == "references"

    @pytest.mark.skip(
        reason="_generate_collection_name method does not exist in actual implementation"
    )
    def test_generate_collection_name_with_prefix(self, collection_manager):
        """Test collection name generation with prefix."""
        # Enable prefix in config
        collection_manager.config.workspace.collection_prefix = "dev_"

        name = collection_manager._generate_collection_name(
            "docs", project_name="test-project"
        )

        assert name == "dev_test-project-docs"

    @pytest.mark.skip(
        reason="_build_vectors_config method does not exist in actual implementation"
    )
    def test_build_vectors_config_with_sparse(self, collection_manager):
        """Test vectors configuration building with sparse vectors."""
        config = CollectionConfig(
            name="test",
            description="test",
            collection_type="docs",
            vector_size=384,
            distance_metric="Cosine",
            enable_sparse_vectors=True,
        )

        vectors_config = collection_manager._build_vectors_config(config)

        assert "dense" in vectors_config
        assert "sparse" in vectors_config

        # Check dense vector config
        dense_config = vectors_config["dense"]
        assert dense_config.size == 384
        assert dense_config.distance == models.Distance.COSINE

        # Check sparse vector config
        sparse_config = vectors_config["sparse"]
        assert sparse_config.modifier == models.Modifier.IDF

    @pytest.mark.skip(
        reason="_build_vectors_config method does not exist in actual implementation"
    )
    def test_build_vectors_config_without_sparse(self, collection_manager):
        """Test vectors configuration building without sparse vectors."""
        config = CollectionConfig(
            name="test",
            description="test",
            collection_type="docs",
            vector_size=768,
            distance_metric="Dot",
            enable_sparse_vectors=False,
        )

        vectors_config = collection_manager._build_vectors_config(config)

        assert "dense" in vectors_config
        assert "sparse" not in vectors_config

        dense_config = vectors_config["dense"]
        assert dense_config.size == 768
        assert dense_config.distance == models.Distance.DOT

    @pytest.mark.skip(
        reason="_build_vectors_config method does not exist in actual implementation"
    )
    def test_build_vectors_config_invalid_distance(self, collection_manager):
        """Test vectors configuration with invalid distance metric."""
        config = CollectionConfig(
            name="test",
            description="test",
            collection_type="docs",
            distance_metric="InvalidMetric",
        )

        vectors_config = collection_manager._build_vectors_config(config)

        # Should default to COSINE
        dense_config = vectors_config["dense"]
        assert dense_config.distance == models.Distance.COSINE

    @pytest.mark.skip(
        reason="ensure_collection_exists public method does not exist - only _ensure_collection_exists private method"
    )
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_create_new(self, collection_manager):
        """Test ensuring collection exists - creates new collection."""
        # Collection doesn't exist
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        config = CollectionConfig(
            name="new-collection", description="New collection", collection_type="docs"
        )

        result = await collection_manager.ensure_collection_exists(config)

        assert result is True
        collection_manager.client.create_collection.assert_called_once()

    @pytest.mark.skip(
        reason="ensure_collection_exists public method does not exist - only _ensure_collection_exists private method"
    )
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_already_exists(self, collection_manager):
        """Test ensuring collection exists - collection already exists."""
        # Collection exists
        existing_collections = [
            models.CollectionDescription(name="existing-collection")
        ]
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=existing_collections)
        )
        collection_manager.client.create_collection = MagicMock()

        config = CollectionConfig(
            name="existing-collection",
            description="Existing collection",
            collection_type="docs",
        )

        result = await collection_manager.ensure_collection_exists(config)

        assert result is True
        collection_manager.client.create_collection.assert_not_called()

    @pytest.mark.skip(
        reason="_validate_collection_limits method does not exist in actual implementation"
    )
    def test_validate_collection_limits_under_limit(self, collection_manager):
        """Test collection limit validation - under limit."""
        existing_collections = [f"collection{i}" for i in range(5)]  # 5 collections

        # Max is 10, so should pass
        result = collection_manager._validate_collection_limits(existing_collections)

        assert result is True

    @pytest.mark.skip(
        reason="_validate_collection_limits method does not exist in actual implementation"
    )
    def test_validate_collection_limits_at_limit(self, collection_manager):
        """Test collection limit validation - at limit."""
        existing_collections = [f"collection{i}" for i in range(10)]  # 10 collections

        # At max limit, should pass
        result = collection_manager._validate_collection_limits(existing_collections)

        assert result is True

    @pytest.mark.skip(
        reason="_validate_collection_limits method does not exist in actual implementation"
    )
    def test_validate_collection_limits_over_limit(self, collection_manager):
        """Test collection limit validation - over limit."""
        existing_collections = [f"collection{i}" for i in range(15)]  # 15 collections

        # Over max limit (10), should fail
        result = collection_manager._validate_collection_limits(existing_collections)

        assert result is False

    @pytest.mark.skip(
        reason="_validate_collection_limits method does not exist in actual implementation"
    )
    def test_validate_collection_limits_unlimited(self, collection_manager):
        """Test collection limit validation - unlimited."""
        # Set unlimited
        collection_manager.config.workspace.max_collections = 0

        existing_collections = [
            f"collection{i}" for i in range(100)
        ]  # Many collections

        result = collection_manager._validate_collection_limits(existing_collections)

        assert result is True  # Should always pass when unlimited

    @pytest.mark.asyncio
    async def test_initialize_workspace_collections_auto_create_disabled(self):
        """Test that when auto_create_collections=False, only scratchbook is created."""
        # Create config with auto_create_collections=False
        workspace_config = WorkspaceConfig(
            collections=["project", "docs"],
            global_collections=["scratchbook", "references", "standards"], 
            auto_create_collections=False
        )
        embedding_config = EmbeddingConfig()
        config = Config(workspace=workspace_config, embedding=embedding_config)
        
        # Create collection manager
        mock_client = MagicMock(spec=QdrantClient)
        collection_manager = WorkspaceCollectionManager(mock_client, config)
        
        # Mock existing collections check
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections(
            "test-project", subprojects=["sub1", "sub2"]
        )

        # Verify only scratchbook collection was created
        assert collection_manager.client.create_collection.call_count == 1
        
        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]
        
        # Should only create scratchbook
        assert created_names == ["scratchbook"]

    @pytest.mark.asyncio 
    async def test_initialize_workspace_collections_auto_create_enabled(self):
        """Test that when auto_create_collections=True, all collections are created."""
        # Create config with auto_create_collections=True
        workspace_config = WorkspaceConfig(
            collections=["project", "docs"],
            global_collections=["scratchbook", "references"],
            auto_create_collections=True
        )
        embedding_config = EmbeddingConfig()
        config = Config(workspace=workspace_config, embedding=embedding_config)
        
        # Create collection manager
        mock_client = MagicMock(spec=QdrantClient)
        collection_manager = WorkspaceCollectionManager(mock_client, config)
        
        # Mock existing collections check
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        await collection_manager.initialize_workspace_collections(
            "test-project", subprojects=["sub1"]
        )

        created_names = [
            call[1]["collection_name"]
            for call in collection_manager.client.create_collection.call_args_list
        ]
        
        # Should create project collections, subproject collections, and global collections
        expected_collections = [
            "test-project-project",  # main project with "project" suffix
            "test-project-docs",     # main project with "docs" suffix
            "sub1-project",          # subproject with "project" suffix
            "sub1-docs",            # subproject with "docs" suffix
            "scratchbook",          # global collection
            "references"            # global collection
        ]
        
        assert len(created_names) == 6
        for expected in expected_collections:
            assert expected in created_names

    @pytest.mark.asyncio
    async def test_codebase_collection_creation(self, collection_manager):
        """Test that _codebase library collection is automatically created with proper configuration."""
        # Mock empty collections list
        collection_manager.client.get_collections.return_value = (
            models.CollectionsResponse(collections=[])
        )
        collection_manager.client.create_collection = MagicMock()

        # Initialize workspace collections
        await collection_manager.initialize_workspace_collections("test-project")

        # Extract all created collection configurations
        created_calls = collection_manager.client.create_collection.call_args_list
        created_names = [call[1]["collection_name"] for call in created_calls]

        # Verify _codebase collection is created
        assert "_codebase" in created_names

        # Find the _codebase collection call
        codebase_call = None
        for call in created_calls:
            if call[1]["collection_name"] == "_codebase":
                codebase_call = call
                break

        assert codebase_call is not None

        # Verify _codebase collection configuration
        call_args = codebase_call[1]
        assert call_args["collection_name"] == "_codebase"

        # Verify sparse vectors are enabled for optimal code search
        vectors_config = call_args["vectors_config"]
        sparse_vectors_config = call_args["sparse_vectors_config"]

        # Should have both dense and sparse vectors for hybrid code search
        assert "dense" in vectors_config
        assert "sparse" in sparse_vectors_config

        # Verify vector size is set correctly
        assert vectors_config["dense"].size == 384  # all-MiniLM-L6-v2 size
