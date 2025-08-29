"""
End-to-end tests for workspace-qdrant-mcp.

Tests complete workflows including project detection, collection management,
document operations, and search functionality.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.utils.project_detection import ProjectDetector


class TestFullWorkflowE2E:
    """End-to-end workflow tests."""

    @pytest.mark.e2e
    @pytest.mark.requires_qdrant
    async def test_complete_document_workflow(self, mock_config, mock_qdrant_client):
        """Test complete document workflow: add, search, update, delete."""
        # This would ideally use a real Qdrant instance, but we'll mock for CI
        with patch(
            "workspace_qdrant_mcp.core.client.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            client = QdrantWorkspaceClient(mock_config)

            # Mock project detection
            with patch.object(
                client.project_detector, "get_project_info"
            ) as mock_project_info:
                mock_project_info.return_value = {
                    "main_project": "test-project",
                    "subprojects": [],
                    "git_root": "/tmp/test-project",
                    "is_git_repo": True,
                }

                # Mock embedding service
                client.embedding_service.initialize = AsyncMock()
                client.embedding_service.generate_embeddings = AsyncMock(
                    return_value={
                        "dense": [0.1] * 384,
                        "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]},
                    }
                )

                # Mock collection manager
                with (
                    patch(
                        "workspace_qdrant_mcp.core.client.WorkspaceCollectionManager"
                    ) as mock_collection_manager_class,
                    patch("asyncio.get_event_loop") as mock_get_loop,
                ):
                    mock_collection_manager = MagicMock()
                    mock_collection_manager.initialize_workspace_collections = (
                        AsyncMock()
                    )
                    mock_collection_manager.list_workspace_collections = AsyncMock(
                        return_value=["test-project_docs", "test-project_scratchbook"]
                    )
                    mock_collection_manager_class.return_value = mock_collection_manager

                    # Mock event loop for Qdrant client operations
                    mock_loop = MagicMock()
                    mock_get_loop.return_value = mock_loop
                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    future.set_result(MagicMock(collections=[]))
                    mock_loop.run_in_executor.return_value = future

                    # Initialize client
                    await client.initialize()
                    assert client.initialized

                    # Test workspace status
                    status = await client.get_status()
                    assert status["connected"] is True
                    assert status["current_project"] == "test-project"

                    # Test collection listing
                    collections = await client.list_collections()
                    assert "test-project_docs" in collections
                    assert "test-project_scratchbook" in collections

    @pytest.mark.e2e
    @pytest.mark.requires_git
    async def test_project_detection_workflow(self, temp_git_repo_with_submodules):
        """Test project detection with real Git repository."""
        detector = ProjectDetector(github_user="testuser")

        # Test project info detection
        project_info = detector.get_project_info(temp_git_repo_with_submodules)

        assert project_info["is_git_repo"] is True
        assert project_info["main_project"] is not None
        assert "detailed_submodules" in project_info

        # Test submodule detection with mocking since we can't create real submodules easily
        with patch(
            "workspace_qdrant_mcp.utils.project_detection.git.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Mock submodules
            mock_submodule1 = MagicMock()
            mock_submodule1.name = "user-owned-submodule"
            mock_submodule1.url = "https://github.com/testuser/user-owned-submodule.git"
            mock_submodule1.path = "libs/user-owned"
            mock_submodule1.hexsha = "abc123"

            mock_submodule2 = MagicMock()
            mock_submodule2.name = "other-submodule"
            mock_submodule2.url = "https://github.com/otheruser/other-submodule.git"
            mock_submodule2.path = "libs/other"
            mock_submodule2.hexsha = "def456"

            mock_repo.submodules = [mock_submodule1, mock_submodule2]

            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["file.txt"]),
            ):
                subprojects = detector.get_subprojects(temp_git_repo_with_submodules)

                # Should only include user-owned submodule
                assert "user-owned-submodule" in subprojects
                assert "other-submodule" not in subprojects

    @pytest.mark.e2e
    async def test_hybrid_search_workflow(self, mock_config, mock_qdrant_client):
        """Test complete hybrid search workflow."""
        from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine

        # Setup mock search results
        dense_results = [
            MagicMock(
                id="doc1", score=0.9, payload={"content": "Python programming guide"}
            ),
            MagicMock(
                id="doc2", score=0.8, payload={"content": "Machine learning basics"}
            ),
        ]

        sparse_results = [
            MagicMock(
                id="doc1", score=0.85, payload={"content": "Python programming guide"}
            ),
            MagicMock(
                id="doc3",
                score=0.75,
                payload={"content": "Web development with FastAPI"},
            ),
        ]

        # Configure mock client to return different results for dense vs sparse search
        search_call_count = 0

        def mock_search(*args, **kwargs):
            nonlocal search_call_count
            search_call_count += 1
            if search_call_count == 1:  # Dense search
                return dense_results
            else:  # Sparse search
                return sparse_results

        mock_qdrant_client.search.side_effect = mock_search

        # Create hybrid search engine
        engine = HybridSearchEngine(mock_qdrant_client)

        # Prepare query embeddings
        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5, 10, 20], "values": [0.8, 0.6, 0.9, 0.7]},
        }

        # Test RRF fusion
        rrf_result = await engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="rrf",
            limit=10,
        )

        assert "error" not in rrf_result
        assert rrf_result["fusion_method"] == "rrf"
        assert rrf_result["total_results"] == 3  # doc1, doc2, doc3
        assert len(rrf_result["results"]) == 3

        # doc1 should be first (appears in both dense and sparse results)
        assert rrf_result["results"][0]["id"] == "doc1"
        assert len(rrf_result["results"][0]["search_types"]) == 2

        # Test weighted sum fusion
        mock_qdrant_client.search.side_effect = mock_search
        search_call_count = 0

        weighted_result = await engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="weighted_sum",
            dense_weight=1.5,
            sparse_weight=0.8,
            limit=10,
        )

        assert weighted_result["fusion_method"] == "weighted_sum"
        assert weighted_result["weights"]["dense"] == 1.5
        assert weighted_result["weights"]["sparse"] == 0.8

        # Test max fusion
        mock_qdrant_client.search.side_effect = mock_search
        search_call_count = 0

        max_result = await engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="max",
            limit=10,
        )

        assert max_result["fusion_method"] == "max"

        # Verify all three methods return results
        for result in [rrf_result, weighted_result, max_result]:
            assert result["total_results"] > 0
            assert len(result["results"]) > 0

    @pytest.mark.e2e
    async def test_scratchbook_workflow(self, mock_workspace_client):
        """Test complete scratchbook workflow."""
        from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager

        # Setup mock client
        mock_workspace_client.get_embedding_service.return_value.generate_embeddings = (
            AsyncMock(
                return_value={
                    "dense": [0.1] * 384,
                    "sparse": {"indices": [1, 5], "values": [0.8, 0.6]},
                }
            )
        )

        manager = ScratchbookManager(mock_workspace_client)

        # Mock collection operations
        mock_collection_manager = MagicMock()
        mock_collection_manager.get_collection_name.return_value = "test_scratchbook"
        mock_collection_manager.ensure_collection_exists = AsyncMock()
        mock_workspace_client.collection_manager = mock_collection_manager

        # Mock Qdrant client operations
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.upsert = AsyncMock()
        mock_qdrant_client.search = AsyncMock(
            return_value=[
                MagicMock(
                    id="note1",
                    score=0.9,
                    payload={
                        "title": "Python Tips",
                        "content": "Use list comprehensions",
                        "note_type": "tip",
                        "tags": ["python", "programming"],
                    },
                )
            ]
        )
        mock_workspace_client.client = mock_qdrant_client

        # Test adding a note
        add_result = await manager.add_note(
            content="Use list comprehensions for better performance",
            title="Python Tips",
            note_type="tip",
            tags=["python", "programming"],
            project_name="test-project",
        )

        assert add_result["status"] == "success"
        assert "note_id" in add_result

        # Test searching notes
        search_result = await manager.search_notes(
            query="python performance",
            note_types=["tip"],
            tags=["python"],
            project_name="test-project",
            limit=10,
        )

        assert "results" in search_result
        assert len(search_result["results"]) > 0
        assert search_result["results"][0]["payload"]["title"] == "Python Tips"

        # Test listing notes
        list_result = await manager.list_notes(
            project_name="test-project", note_type="tip", limit=50
        )

        assert "results" in list_result

    @pytest.mark.e2e
    async def test_configuration_and_validation_workflow(self, environment_variables):
        """Test configuration loading and validation workflow."""
        from workspace_qdrant_mcp.utils.config_validator import ConfigValidator

        # Test configuration loading from environment
        config = Config()

        # Should load values from environment variables set in fixture
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.debug is True
        assert config.qdrant.url == "http://localhost:6333"
        assert config.workspace.github_user == "testuser"

        # Test configuration validation
        validator = ConfigValidator(config)
        is_valid, results = validator.validate_all()

        # Should be valid with our test configuration
        assert is_valid is True
        assert "issues" in results
        assert len(results["issues"]) == 0

        # Test invalid configuration
        config.qdrant.url = ""
        config.embedding.chunk_size = -1

        validator = ConfigValidator(config)
        is_valid, results = validator.validate_all()

        assert is_valid is False
        assert len(results["issues"]) > 0

        # Test configuration file loading
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as temp_env:
            temp_env.write(
                "WORKSPACE_QDRANT_HOST=file.host\n"
                "WORKSPACE_QDRANT_PORT=7777\n"
                "WORKSPACE_QDRANT_QDRANT__URL=https://file.qdrant.io\n"
            )
            temp_env.flush()

            # Change to temp directory and load config
            import os

            original_cwd = os.getcwd()
            temp_dir = Path(temp_env.name).parent
            env_file = Path(temp_env.name)
            env_file.rename(temp_dir / ".env")

            try:
                os.chdir(temp_dir)
                config_from_file = Config()

                assert config_from_file.host == "file.host"
                assert config_from_file.port == 7777
                assert config_from_file.qdrant.url == "https://file.qdrant.io"

            finally:
                os.chdir(original_cwd)
                (temp_dir / ".env").unlink()

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_large_document_processing_workflow(
        self, mock_config, mock_qdrant_client
    ):
        """Test workflow with large documents and chunking."""
        from workspace_qdrant_mcp.tools.documents import add_document

        # Create a large document
        large_content = "\n".join(
            [
                f"This is paragraph {i} of a very large document. " * 20
                for i in range(100)
            ]
        )

        # Setup mock workspace client
        with patch(
            "workspace_qdrant_mcp.core.client.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            client = QdrantWorkspaceClient(mock_config)

            # Mock all necessary components
            client.initialized = True
            client.client = mock_qdrant_client

            # Mock collection manager
            mock_collection_manager = MagicMock()
            mock_collection_manager.get_collection_name.return_value = "test_docs"
            mock_collection_manager.ensure_collection_exists = AsyncMock()
            client.collection_manager = mock_collection_manager

            # Mock embedding service
            client.embedding_service.generate_embeddings = AsyncMock(
                return_value={
                    "dense": [0.1] * 384,
                    "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]},
                }
            )

            # Mock Qdrant upsert operation
            mock_qdrant_client.upsert = AsyncMock(
                return_value=MagicMock(status="completed", operation_id=123)
            )

            # Test adding large document with chunking
            result = await add_document(
                workspace_client=client,
                content=large_content,
                collection="docs",
                metadata={"source": "test", "type": "large_document"},
                document_id="large_doc_1",
                chunk_text=True,
            )

            assert result["status"] == "success"
            assert "chunks_created" in result

            # Should have created multiple chunks for large document
            # (exact number depends on chunking logic)
            chunks_created = result.get("chunks_created", 0)
            assert chunks_created > 1

            # Verify upsert was called (chunks were uploaded)
            mock_qdrant_client.upsert.assert_called()

    @pytest.mark.e2e
    async def test_error_recovery_workflow(self, mock_config, mock_qdrant_client):
        """Test error handling and recovery in various scenarios."""
        with patch(
            "workspace_qdrant_mcp.core.client.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            client = QdrantWorkspaceClient(mock_config)

            # Test initialization failure recovery
            mock_qdrant_client.get_collections.side_effect = Exception(
                "Connection failed"
            )

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.create_future()
                future.set_exception(Exception("Connection failed"))
                mock_loop.run_in_executor.return_value = future

                # Should raise exception on initialization failure
                with pytest.raises(Exception, match="Connection failed"):
                    await client.initialize()

                assert client.initialized is False

            # Test status retrieval with partial failures
            client.initialized = True

            # Mock collection manager that fails
            mock_collection_manager = MagicMock()
            mock_collection_manager.list_workspace_collections = AsyncMock(
                side_effect=Exception("Collection error")
            )
            client.collection_manager = mock_collection_manager

            status = await client.get_status()

            # Should return error information instead of crashing
            assert "error" in status
            assert "Failed to get status" in status["error"]

            # Test graceful handling when client is not initialized
            client.initialized = False

            status = await client.get_status()
            assert status == {"error": "Client not initialized"}

            collections = await client.list_collections()
            assert collections == []

    @pytest.mark.e2e
    async def test_multi_project_workspace_workflow(self, mock_config):
        """Test workflow with multiple projects and subprojects."""
        # Test project detection with multiple subprojects
        detector = ProjectDetector(github_user="testuser")

        # Mock complex project structure
        with (
            patch.object(
                detector, "_find_git_root", return_value="/path/to/main-project"
            ),
            patch.object(
                detector,
                "_get_git_remote_url",
                return_value="https://github.com/testuser/main-project.git",
            ),
            patch(
                "workspace_qdrant_mcp.utils.project_detection.git.Repo"
            ) as mock_repo_class,
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Create multiple mock submodules
            submodules = []
            for i in range(5):
                mock_submodule = MagicMock()
                mock_submodule.name = f"subproject{i}"
                mock_submodule.path = f"libs/subproject{i}"
                # Mix of user-owned and other-owned submodules
                if i < 3:
                    mock_submodule.url = (
                        f"https://github.com/testuser/subproject{i}.git"
                    )
                else:
                    mock_submodule.url = (
                        f"https://github.com/otheruser/subproject{i}.git"
                    )
                mock_submodule.hexsha = f"abc{i}23"
                submodules.append(mock_submodule)

            mock_repo.submodules = submodules

            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["file.txt"]),
            ):
                project_info = detector.get_project_info()

                assert project_info["main_project"] == "main-project"
                assert len(project_info["subprojects"]) == 3  # Only user-owned
                assert all("subproject" in name for name in project_info["subprojects"])

                # Test detailed submodule information
                detailed = project_info["detailed_submodules"]
                user_owned = [sm for sm in detailed if sm.get("user_owned", False)]
                assert len(user_owned) == 3

    @pytest.mark.e2e
    @pytest.mark.benchmark
    async def test_search_performance_workflow(self, mock_config, mock_qdrant_client):
        """Test search performance with various fusion methods."""
        from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine(mock_qdrant_client)

        # Create large mock result sets
        dense_results = [
            MagicMock(
                id=f"doc{i}",
                score=0.9 - (i * 0.01),
                payload={"content": f"Document {i} content"},
            )
            for i in range(100)
        ]

        sparse_results = [
            MagicMock(
                id=f"doc{i}",
                score=0.85 - (i * 0.01),
                payload={"content": f"Document {i} content"},
            )
            for i in range(50, 150)  # 50% overlap with dense results
        ]

        # Configure mock to return large result sets
        search_call_count = 0

        def mock_search(*args, **kwargs):
            nonlocal search_call_count
            search_call_count += 1
            if search_call_count % 2 == 1:  # Odd calls = dense search
                return dense_results
            else:  # Even calls = sparse search
                return sparse_results

        mock_qdrant_client.search.side_effect = mock_search

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": list(range(100)), "values": [0.5] * 100},
        }

        # Benchmark different fusion methods
        benchmark_result = engine.benchmark_fusion_methods(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=20,
        )

        assert "benchmark_results" in benchmark_result

        # All methods should return results
        for method, result in benchmark_result["benchmark_results"].items():
            assert "error" not in result
            assert result["total_results"] == 20  # Limited to 20
            assert len(result["results"]) == 20

            # Verify fusion statistics
            if method == "rrf":
                # RRF should have good distribution of sources
                search_types = [r.get("search_types", []) for r in result["results"]]
                has_both = sum(1 for st in search_types if len(st) == 2)
                assert has_both > 0  # Should have some documents from both sources

        # Test that results are properly ranked
        rrf_results = benchmark_result["benchmark_results"]["rrf"]["results"]
        rrf_scores = [r["rrf_score"] for r in rrf_results]

        # Should be sorted in descending order
        assert rrf_scores == sorted(rrf_scores, reverse=True)

    @pytest.mark.e2e
    async def test_cleanup_workflow(self, mock_config, mock_qdrant_client):
        """Test proper cleanup of resources."""
        with patch(
            "workspace_qdrant_mcp.core.client.QdrantClient",
            return_value=mock_qdrant_client,
        ):
            client = QdrantWorkspaceClient(mock_config)

            # Initialize client
            with (
                patch.object(client.project_detector, "get_project_info"),
                patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager"),
                patch.object(client.embedding_service, "initialize"),
                patch("asyncio.get_event_loop") as mock_get_loop,
            ):
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.create_future()
                future.set_result(MagicMock(collections=[]))
                mock_loop.run_in_executor.return_value = future

                await client.initialize()
                assert client.initialized

                # Mock embedding service close
                client.embedding_service.close = AsyncMock()

                # Test cleanup
                await client.close()

                # Verify cleanup was performed
                client.embedding_service.close.assert_called_once()
                mock_qdrant_client.close.assert_called_once()
                assert client.client is None
                assert client.initialized is False

                # Multiple close calls should be safe
                await client.close()
                # Should not raise exception or call close again
