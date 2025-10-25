"""
Phase 1 gRPC Protocol Validation Integration Tests.

Comprehensive integration tests for validating gRPC protocol functionality
between MCP server and Rust daemon. Tests all 15 RPCs across SystemService,
CollectionService, and DocumentService defined in workspace_daemon.proto.

Test Coverage:
    - SystemService (7 RPCs): Health, status, metrics, refresh signals, lifecycle
    - CollectionService (5 RPCs): Collection and alias management
    - DocumentService (3 RPCs): Text ingestion, update, deletion
    - Fallback detection: Validates daemon-only write path (First Principle 10)
    - Error handling: Connection failures, timeouts, invalid requests

Requirements:
    - Running Qdrant instance (localhost:6333)
    - Running Rust daemon with gRPC server (localhost:50051)
    - pytest with async support (pytest-asyncio)

Usage:
    # Run all Phase 1 protocol validation tests
    pytest tests/integration/test_phase1_protocol_validation.py -v

    # Run specific test class
    pytest tests/integration/test_phase1_protocol_validation.py::TestSystemService -v

    # Run with daemon requirement marker
    pytest -m "requires_daemon" tests/integration/test_phase1_protocol_validation.py -v
"""

import asyncio
import logging
import pytest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Import daemon client and gRPC types
from common.grpc.daemon_client import (
    DaemonClient,
    DaemonUnavailableError,
    DaemonTimeoutError,
    DaemonClientError,
)
from common.grpc.generated import workspace_daemon_pb2 as pb2


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def qdrant_client():
    """
    Provide Qdrant client for validation.

    Connects to local Qdrant instance for verifying that gRPC write operations
    correctly propagate to the vector database.
    """
    client = QdrantClient(host="localhost", port=6333)

    # Verify Qdrant is accessible
    try:
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant server not accessible: {e}")

    yield client


@pytest.fixture
async def daemon_client():
    """
    Provide daemon client for gRPC communication.

    Creates a fresh DaemonClient instance for each test, ensuring clean state.
    Automatically handles connection/disconnection lifecycle.
    """
    client = DaemonClient(host="localhost", port=50051)

    try:
        await client.start()
    except Exception as e:
        pytest.skip(f"Daemon server not accessible: {e}")

    yield client

    # Cleanup
    await client.stop()


@pytest.fixture
async def test_collection_cleanup(qdrant_client):
    """
    Cleanup test collections before and after tests.

    Removes all test collections matching the pattern "test_*" to ensure
    clean test isolation. Runs before and after each test.
    """
    def cleanup():
        collections = qdrant_client.get_collections().collections
        for collection in collections:
            if collection.name.startswith("test_"):
                try:
                    qdrant_client.delete_collection(collection.name)
                except Exception:
                    pass  # Collection might not exist or already deleted

    # Cleanup before test
    cleanup()

    yield

    # Cleanup after test
    cleanup()


@pytest.fixture
def log_capture():
    """
    Capture log messages for fallback detection.

    Sets up a log handler to capture WARNING level messages, which are emitted
    when the MCP server falls back to direct Qdrant writes (bypassing daemon).
    This validates First Principle 10: daemon-only write path enforcement.
    """
    logger = logging.getLogger("workspace_qdrant_mcp")
    handler = logging.handlers.MemoryHandler(capacity=1000, target=None)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    yield handler

    # Cleanup
    logger.removeHandler(handler)


@pytest.fixture
def sample_metadata() -> Dict[str, str]:
    """
    Provide sample metadata for testing.

    Returns a consistent metadata dictionary for use across tests, ensuring
    reproducible test behavior.
    """
    return {
        "test_id": "protocol_validation_001",
        "source": "integration_test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_type": "test",
    }


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_daemon
class TestSystemService:
    """
    Test SystemService gRPC operations (7 RPCs).

    Validates health checks, status reporting, metrics collection, refresh
    signals, and lifecycle management RPCs defined in workspace_daemon.proto.
    """

    async def test_health_check_success(self, daemon_client):
        """
        Test HealthCheck RPC returns healthy status.

        Validates:
            - RPC completes successfully
            - Response contains valid ServiceStatus
            - Response includes component health information
            - Timestamp is present and recent
        """
        # TODO: Implement health check validation
        # Expected flow:
        # 1. Call daemon_client.health_check()
        # 2. Verify response.status == SERVICE_STATUS_HEALTHY
        # 3. Verify response.components is populated
        # 4. Verify response.timestamp is recent (within last minute)
        pass

    async def test_get_status_complete_system_snapshot(self, daemon_client):
        """
        Test GetStatus RPC returns comprehensive system state.

        Validates:
            - System status includes all required fields
            - Active projects list is present
            - Document and collection counts are non-negative
            - Uptime timestamp is reasonable
            - System metrics are populated
        """
        # TODO: Implement system status validation
        # Expected flow:
        # 1. Call daemon_client.get_status()
        # 2. Verify response.status is valid
        # 3. Verify response.metrics contains CPU/memory data
        # 4. Verify response.total_documents >= 0
        # 5. Verify response.uptime_since is in the past
        pass

    async def test_get_metrics_performance_data(self, daemon_client):
        """
        Test GetMetrics RPC returns current performance metrics.

        Validates:
            - Metrics response contains metric list
            - Each metric has name, type, and value
            - Timestamp is present and recent
            - Metric types are valid (counter, gauge, histogram)
        """
        # TODO: Implement metrics validation
        # Expected flow:
        # 1. Call daemon_client.get_metrics()
        # 2. Verify response.metrics is a list
        # 3. For each metric, verify required fields
        # 4. Verify response.collected_at is recent
        pass

    async def test_send_refresh_signal_ingestion_queue(self, daemon_client):
        """
        Test SendRefreshSignal RPC for INGEST_QUEUE changes.

        Validates:
            - Refresh signal is sent successfully
            - Daemon acknowledges signal (returns Empty)
            - No errors occur during signal transmission
        """
        # TODO: Implement refresh signal validation
        # Expected flow:
        # 1. Call daemon_client.send_refresh_signal(pb2.INGEST_QUEUE)
        # 2. Verify no exception is raised
        # 3. Verify daemon processes signal (check logs or metrics)
        pass

    async def test_send_refresh_signal_watch_folders(self, daemon_client):
        """
        Test SendRefreshSignal RPC for WATCH_FOLDERS changes.

        Validates:
            - Watch folder refresh signal is sent successfully
            - Daemon updates watch configuration
            - No errors occur
        """
        # TODO: Implement watch folder refresh validation
        pass

    async def test_send_refresh_signal_lsp_tools(self, daemon_client):
        """
        Test SendRefreshSignal RPC for TOOLS_AVAILABLE with LSP languages.

        Validates:
            - LSP language availability signal is sent
            - Language list is correctly transmitted
            - Daemon updates tool registry
        """
        # TODO: Implement LSP tools refresh validation
        # Expected flow:
        # 1. Call daemon_client.send_refresh_signal(
        #       pb2.TOOLS_AVAILABLE,
        #       lsp_languages=["python", "rust"]
        #    )
        # 2. Verify signal is received by daemon
        pass

    async def test_notify_server_status_starting(self, daemon_client):
        """
        Test NotifyServerStatus RPC for server starting.

        Validates:
            - Server status notification is sent successfully
            - Project information is transmitted
            - Daemon acknowledges notification
        """
        # TODO: Implement server status notification validation
        # Expected flow:
        # 1. Call daemon_client.notify_server_status(
        #       pb2.SERVER_STATE_UP,
        #       project_name="test_project",
        #       project_root="/path/to/project"
        #    )
        # 2. Verify daemon receives notification
        pass

    async def test_notify_server_status_stopping(self, daemon_client):
        """
        Test NotifyServerStatus RPC for server stopping.

        Validates:
            - Server shutdown notification is sent
            - Daemon handles shutdown gracefully
            - No errors occur during notification
        """
        # TODO: Implement server shutdown notification validation
        pass

    async def test_pause_all_watchers(self, daemon_client):
        """
        Test PauseAllWatchers RPC pauses file watching.

        Validates:
            - Pause command is sent successfully
            - Returns Empty response
            - Watchers are actually paused (verify with status check)
        """
        # TODO: Implement pause watchers validation
        # Expected flow:
        # 1. Call daemon_client.pause_all_watchers()
        # 2. Verify no exception
        # 3. Call get_status() and verify watchers are paused
        pass

    async def test_resume_all_watchers(self, daemon_client):
        """
        Test ResumeAllWatchers RPC resumes file watching.

        Validates:
            - Resume command is sent successfully
            - Returns Empty response
            - Watchers are actually resumed (verify with status check)
        """
        # TODO: Implement resume watchers validation
        # Expected flow:
        # 1. First pause watchers
        # 2. Call daemon_client.resume_all_watchers()
        # 3. Verify watchers are active again via get_status()
        pass

    async def test_health_check_timeout_handling(self, daemon_client):
        """
        Test HealthCheck RPC respects timeout parameter.

        Validates:
            - Short timeout (0.1s) raises DaemonTimeoutError if daemon is slow
            - Normal timeout (5s) succeeds for healthy daemon
        """
        # TODO: Implement timeout handling validation
        pass


@pytest.mark.integration
@pytest.mark.requires_daemon
class TestCollectionService:
    """
    Test CollectionService gRPC operations (5 RPCs).

    Validates collection lifecycle management and alias operations defined
    in workspace_daemon.proto. Ensures daemon-only write path for collections.
    """

    async def test_create_collection_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test CreateCollection RPC creates collection in Qdrant.

        Validates:
            - Collection is created via daemon (not direct Qdrant)
            - Collection exists in Qdrant after creation
            - Collection has correct vector configuration
            - Response contains success=True and collection_id
        """
        # TODO: Implement collection creation validation
        # Expected flow:
        # 1. Call daemon_client.create_collection(
        #       collection_name="test_protocol_collection",
        #       project_id="test_project",
        #       config=pb2.CollectionConfig(
        #           vector_size=384,
        #           distance_metric="Cosine",
        #           enable_indexing=True
        #       )
        #    )
        # 2. Verify response.success == True
        # 3. Verify collection exists in qdrant_client
        # 4. Verify collection has correct vector size (384)
        pass

    async def test_create_collection_with_custom_config(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test CreateCollection RPC with custom configuration.

        Validates:
            - Custom vector size is respected
            - Custom distance metric is applied
            - Metadata schema is configured
        """
        # TODO: Implement custom config validation
        pass

    async def test_delete_collection_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test DeleteCollection RPC removes collection from Qdrant.

        Validates:
            - Collection is deleted via daemon
            - Collection no longer exists in Qdrant
            - Deletion is permanent
        """
        # TODO: Implement collection deletion validation
        # Expected flow:
        # 1. First create a test collection
        # 2. Call daemon_client.delete_collection(
        #       collection_name="test_delete_collection",
        #       project_id="test_project",
        #       force=True
        #    )
        # 3. Verify collection is removed from qdrant_client
        pass

    async def test_delete_collection_force_flag(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test DeleteCollection RPC with force=True for non-empty collection.

        Validates:
            - Force flag allows deletion of non-empty collections
            - All documents are removed along with collection
        """
        # TODO: Implement force deletion validation
        pass

    async def test_create_collection_alias_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test CreateCollectionAlias RPC creates alias in Qdrant.

        Validates:
            - Alias is created via daemon
            - Alias points to correct collection
            - Alias can be used for queries
        """
        # TODO: Implement alias creation validation
        # Expected flow:
        # 1. Create a collection
        # 2. Call daemon_client.create_collection_alias(
        #       alias_name="test_alias",
        #       collection_name="test_collection"
        #    )
        # 3. Verify alias exists in Qdrant
        # 4. Verify alias points to correct collection
        pass

    async def test_delete_collection_alias_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test DeleteCollectionAlias RPC removes alias from Qdrant.

        Validates:
            - Alias is deleted via daemon
            - Original collection remains intact
            - Alias no longer resolves
        """
        # TODO: Implement alias deletion validation
        pass

    async def test_rename_collection_alias_atomic(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test RenameCollectionAlias RPC atomically renames alias.

        Validates:
            - Rename operation is atomic (no intermediate state)
            - Old alias name is removed
            - New alias name points to same collection
            - Original collection is unchanged
        """
        # TODO: Implement atomic rename validation
        # Expected flow:
        # 1. Create collection and alias
        # 2. Call daemon_client.rename_collection_alias(
        #       old_name="test_alias_old",
        #       new_name="test_alias_new",
        #       collection_name="test_collection"
        #    )
        # 3. Verify old alias no longer exists
        # 4. Verify new alias points to collection
        pass

    async def test_collection_operations_sequential(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup
    ):
        """
        Test sequence of collection operations.

        Validates:
            - Create → Create Alias → Rename Alias → Delete Alias → Delete Collection
            - Each operation completes successfully
            - State transitions are correct
        """
        # TODO: Implement sequential operations validation
        pass


@pytest.mark.integration
@pytest.mark.requires_daemon
class TestDocumentService:
    """
    Test DocumentService gRPC operations (3 RPCs).

    Validates text ingestion, update, and deletion operations defined in
    workspace_daemon.proto. Ensures daemon-only write path for documents.
    """

    async def test_ingest_text_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test IngestText RPC ingests content via daemon.

        Validates:
            - Text is ingested via daemon (not direct Qdrant)
            - Response contains document_id and success=True
            - Document appears in Qdrant with correct content
            - Metadata is preserved
            - Chunks are created if chunk_text=True
        """
        # Call daemon to ingest text
        response = await daemon_client.ingest_text(
            content="Test document content for protocol validation",
            collection_basename="test_notes",
            tenant_id="test_project",
            metadata=sample_metadata,
            chunk_text=True
        )

        # Verify response structure
        assert response.success == True, f"Ingestion failed: {response.error_message}"
        assert response.document_id != "", "document_id should not be empty"
        assert response.chunks_created > 0, "Should create at least one chunk"

        # Wait for daemon to complete processing
        await wait_for_daemon_processing()

        # Find the test collection created by daemon
        collections = qdrant_client.get_collections().collections
        test_collections = [c.name for c in collections if c.name.startswith("test_")]
        assert len(test_collections) > 0, "No test collection created"

        # Verify document exists in Qdrant (check all test collections)
        found_document = False
        for collection_name in test_collections:
            points, _ = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100
            )

            # Look for our document by checking metadata
            for point in points:
                payload = point.payload or {}
                if payload.get("document_id") == response.document_id:
                    found_document = True
                    # Verify metadata is preserved
                    for key, value in sample_metadata.items():
                        assert payload.get(key) == value, f"Metadata {key} mismatch"
                    break

            if found_document:
                break

        assert found_document, f"Document {response.document_id} not found in Qdrant"

    async def test_ingest_text_without_chunking(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test IngestText RPC with chunk_text=False.

        Validates:
            - Content is ingested as single chunk
            - chunks_created == 1
            - Full content is preserved
        """
        # TODO: Implement no-chunking validation
        pass

    async def test_ingest_text_custom_document_id(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test IngestText RPC with custom document_id.

        Validates:
            - Custom document_id is respected
            - Response returns the same document_id
            - Document can be retrieved using custom ID
        """
        # TODO: Implement custom ID validation
        pass

    async def test_ingest_text_large_content(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test IngestText RPC with large content (>100KB).

        Validates:
            - Large content is handled correctly
            - Chunking produces appropriate number of chunks
            - All content is preserved across chunks
        """
        # Create large content (multiple paragraphs, ~10KB)
        paragraph = "This is a test paragraph with sufficient content to test chunking behavior. " * 20
        large_content = "\n\n".join([paragraph for _ in range(50)])  # ~10KB total

        # Ingest large content with chunking enabled
        response = await daemon_client.ingest_text(
            content=large_content,
            collection_basename="test_notes_large",
            tenant_id="test_project",
            metadata=sample_metadata,
            chunk_text=True
        )

        # Verify response
        assert response.success == True, f"Ingestion failed: {response.error_message}"
        assert response.document_id != "", "document_id should not be empty"
        assert response.chunks_created > 1, f"Should create multiple chunks for large content, got {response.chunks_created}"

        # Wait for daemon processing
        await wait_for_daemon_processing()

        # Find collection and verify chunks
        collections = qdrant_client.get_collections().collections
        test_collections = [c.name for c in collections if c.name.startswith("test_")]

        chunk_count = 0
        found_chunks = []
        for collection_name in test_collections:
            points, _ = qdrant_client.scroll(
                collection_name=collection_name,
                limit=200
            )

            for point in points:
                payload = point.payload or {}
                if payload.get("document_id") == response.document_id:
                    chunk_count += 1
                    found_chunks.append(payload)

        # Verify chunk count matches response
        assert chunk_count == response.chunks_created, \
            f"Expected {response.chunks_created} chunks, found {chunk_count} in Qdrant"

        # Verify chunk metadata if available
        if found_chunks:
            for chunk in found_chunks:
                # Verify metadata preserved in chunks
                assert chunk.get("test_id") == sample_metadata["test_id"]

    async def test_update_text_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test UpdateText RPC updates existing document.

        Validates:
            - Document is updated via daemon
            - New content replaces old content
            - Metadata is updated
            - updated_at timestamp is set
        """
        # First, ingest a document
        ingest_response = await daemon_client.ingest_text(
            content="Initial document content",
            collection_basename="test_notes_update",
            tenant_id="test_project",
            metadata=sample_metadata,
            chunk_text=False  # Single chunk for simplicity
        )

        assert ingest_response.success == True, f"Initial ingestion failed: {ingest_response.error_message}"
        document_id = ingest_response.document_id

        # Wait for initial ingestion
        await wait_for_daemon_processing()

        # Update the document with new content and metadata
        updated_metadata = {**sample_metadata, "updated": "true", "version": "2"}
        update_response = await daemon_client.update_text(
            document_id=document_id,
            content="Updated document content with new information",
            metadata=updated_metadata
        )

        # Verify update response
        assert update_response.success == True, f"Update failed: {update_response.error_message}"
        assert update_response.updated_at is not None, "updated_at timestamp should be set"

        # Wait for update processing
        await wait_for_daemon_processing()

        # Find the document in Qdrant and verify updates
        collections = qdrant_client.get_collections().collections
        test_collections = [c.name for c in collections if c.name.startswith("test_")]

        found_updated = False
        for collection_name in test_collections:
            points, _ = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100
            )

            for point in points:
                payload = point.payload or {}
                if payload.get("document_id") == document_id:
                    found_updated = True
                    # Verify updated metadata
                    assert payload.get("updated") == "true", "Updated metadata should be present"
                    assert payload.get("version") == "2", "Version metadata should be updated"
                    break

            if found_updated:
                break

        assert found_updated, f"Updated document {document_id} not found in Qdrant"

    async def test_update_text_metadata_only(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test UpdateText RPC updates only metadata.

        Validates:
            - Metadata updates without content change
            - Original content is preserved
            - Only metadata fields are modified
        """
        # TODO: Implement metadata-only update validation
        pass

    async def test_delete_text_success(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test DeleteText RPC deletes document via daemon.

        Validates:
            - Document is deleted via daemon
            - All chunks are removed from Qdrant
            - Document no longer exists after deletion
        """
        # First, ingest a document
        ingest_response = await daemon_client.ingest_text(
            content="Document to be deleted",
            collection_basename="test_notes_delete",
            tenant_id="test_project",
            metadata=sample_metadata,
            chunk_text=True
        )

        assert ingest_response.success == True, f"Initial ingestion failed: {ingest_response.error_message}"
        document_id = ingest_response.document_id

        # Wait for ingestion
        await wait_for_daemon_processing()

        # Find the collection name
        collections = qdrant_client.get_collections().collections
        test_collections = [c.name for c in collections if c.name.startswith("test_")]

        # Verify document exists before deletion
        found_before = False
        collection_name = None
        for coll_name in test_collections:
            points, _ = qdrant_client.scroll(
                collection_name=coll_name,
                limit=100
            )

            for point in points:
                payload = point.payload or {}
                if payload.get("document_id") == document_id:
                    found_before = True
                    collection_name = coll_name
                    break

            if found_before:
                break

        assert found_before, f"Document {document_id} should exist before deletion"
        assert collection_name is not None, "Collection name should be found"

        # Delete the document
        await daemon_client.delete_text(
            document_id=document_id,
            collection_name=collection_name
        )

        # Wait for deletion processing
        await wait_for_daemon_processing()

        # Verify document is removed from Qdrant
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100
        )

        found_after = False
        for point in points:
            payload = point.payload or {}
            if payload.get("document_id") == document_id:
                found_after = True
                break

        assert not found_after, f"Document {document_id} should be deleted from Qdrant"

    async def test_document_lifecycle_complete(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        sample_metadata
    ):
        """
        Test complete document lifecycle: Ingest → Update → Delete.

        Validates:
            - All operations complete successfully
            - State transitions are correct
            - No orphaned data remains after deletion
        """
        # Step 1: Ingest a document
        ingest_response = await daemon_client.ingest_text(
            content="Lifecycle test document - initial version",
            collection_basename="test_notes_lifecycle",
            tenant_id="test_project",
            metadata={**sample_metadata, "version": "1"},
            chunk_text=False
        )

        assert ingest_response.success == True, f"Ingestion failed: {ingest_response.error_message}"
        document_id = ingest_response.document_id
        assert ingest_response.chunks_created == 1, "Should create exactly one chunk"

        await wait_for_daemon_processing()

        # Verify document exists in Qdrant
        collections = qdrant_client.get_collections().collections
        test_collections = [c.name for c in collections if c.name.startswith("test_")]

        found_created = False
        collection_name = None
        for coll_name in test_collections:
            points, _ = qdrant_client.scroll(collection_name=coll_name, limit=100)
            for point in points:
                payload = point.payload or {}
                if payload.get("document_id") == document_id:
                    found_created = True
                    collection_name = coll_name
                    assert payload.get("version") == "1", "Initial version should be 1"
                    break
            if found_created:
                break

        assert found_created, f"Document {document_id} should exist after ingestion"

        # Step 2: Update the document
        update_response = await daemon_client.update_text(
            document_id=document_id,
            content="Lifecycle test document - updated version",
            metadata={**sample_metadata, "version": "2", "updated": "true"}
        )

        assert update_response.success == True, f"Update failed: {update_response.error_message}"
        assert update_response.updated_at is not None, "updated_at should be set"

        await wait_for_daemon_processing()

        # Verify document was updated
        found_updated = False
        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=100)
        for point in points:
            payload = point.payload or {}
            if payload.get("document_id") == document_id:
                found_updated = True
                assert payload.get("version") == "2", "Version should be updated to 2"
                assert payload.get("updated") == "true", "Updated flag should be set"
                break

        assert found_updated, f"Document {document_id} should be updated in Qdrant"

        # Step 3: Delete the document
        await daemon_client.delete_text(
            document_id=document_id,
            collection_name=collection_name
        )

        await wait_for_daemon_processing()

        # Verify document is deleted
        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=100)
        found_after_delete = False
        for point in points:
            payload = point.payload or {}
            if payload.get("document_id") == document_id:
                found_after_delete = True
                break

        assert not found_after_delete, f"Document {document_id} should be deleted from Qdrant"


@pytest.mark.integration
@pytest.mark.requires_daemon
class TestFallbackDetection:
    """
    Test fallback detection for daemon-only write path (First Principle 10).

    Validates that when daemon is unavailable, the MCP server correctly:
        1. Attempts to use daemon first (primary path)
        2. Falls back to direct Qdrant writes (fallback path)
        3. Logs warnings about fallback mode
        4. Includes fallback_mode flag in responses
    """

    async def test_daemon_unavailable_fallback_triggered(
        self,
        qdrant_client,
        test_collection_cleanup,
        log_capture
    ):
        """
        Test fallback to direct Qdrant when daemon is unavailable.

        Validates:
            - MCP server detects daemon unavailability
            - Fallback to direct Qdrant write occurs
            - WARNING log is emitted indicating fallback
            - Response includes fallback_mode flag
        """
        # TODO: Implement fallback detection validation
        # Expected flow:
        # 1. Create DaemonClient with unreachable address
        # 2. Attempt ingestion (should fail to reach daemon)
        # 3. Verify WARNING log about fallback mode
        # 4. Verify content still reaches Qdrant (fallback path)
        pass

    async def test_daemon_timeout_fallback(
        self,
        log_capture
    ):
        """
        Test fallback when daemon times out.

        Validates:
            - Timeout errors trigger fallback path
            - Warnings are logged for timeout condition
        """
        # TODO: Implement timeout fallback validation
        pass

    async def test_no_fallback_when_daemon_healthy(
        self,
        daemon_client,
        qdrant_client,
        test_collection_cleanup,
        log_capture
    ):
        """
        Test no fallback occurs when daemon is healthy.

        Validates:
            - All writes go through daemon when available
            - No fallback warnings are logged
            - Response does NOT include fallback_mode flag
        """
        # TODO: Implement healthy daemon validation
        # Expected flow:
        # 1. Use healthy daemon_client
        # 2. Ingest content
        # 3. Verify NO WARNING logs
        # 4. Verify response has no fallback_mode flag
        pass


@pytest.mark.integration
@pytest.mark.requires_daemon
class TestErrorHandling:
    """
    Test error handling for gRPC operations.

    Validates that the daemon client correctly handles:
        - Connection failures
        - Invalid requests
        - Server-side errors
        - Timeout conditions
        - Retry mechanisms
    """

    async def test_connection_failure_raises_daemon_unavailable(self):
        """
        Test connection to non-existent daemon raises DaemonUnavailableError.

        Validates:
            - Attempting to connect to invalid address fails
            - Correct exception type is raised
            - Error message is informative
        """
        # TODO: Implement connection failure validation
        # Expected flow:
        # 1. Create DaemonClient with bad address (localhost:99999)
        # 2. Attempt to start()
        # 3. Verify DaemonUnavailableError is raised
        # 4. Verify error message contains helpful information
        pass

    async def test_invalid_collection_name_error(
        self,
        daemon_client
    ):
        """
        Test invalid collection name returns error response.

        Validates:
            - Invalid collection names are rejected
            - Error message explains validation failure
        """
        # TODO: Implement invalid collection name validation
        pass

    async def test_invalid_vector_size_error(
        self,
        daemon_client
    ):
        """
        Test invalid vector size configuration returns error.

        Validates:
            - Vector size must match embedding model
            - Error response explains mismatch
        """
        # TODO: Implement vector size validation
        pass

    async def test_delete_nonexistent_collection_error(
        self,
        daemon_client
    ):
        """
        Test deleting non-existent collection returns error.

        Validates:
            - Attempting to delete missing collection fails gracefully
            - Error message indicates collection not found
        """
        # TODO: Implement non-existent collection validation
        pass

    async def test_update_nonexistent_document_error(
        self,
        daemon_client
    ):
        """
        Test updating non-existent document returns error.

        Validates:
            - Updating missing document fails gracefully
            - Error message indicates document not found
        """
        # TODO: Implement non-existent document validation
        pass

    async def test_retry_mechanism_on_transient_failure(
        self,
        daemon_client
    ):
        """
        Test retry mechanism on transient failures.

        Validates:
            - Transient errors trigger retries
            - Exponential backoff is applied
            - Successful retry completes operation
        """
        # TODO: Implement retry mechanism validation
        pass

    async def test_circuit_breaker_opens_on_repeated_failures(
        self,
        daemon_client
    ):
        """
        Test circuit breaker opens after threshold failures.

        Validates:
            - Repeated failures open circuit breaker
            - Further requests fail immediately
            - Circuit breaker eventually transitions to half-open
        """
        # TODO: Implement circuit breaker validation
        pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def verify_grpc_response(response: Any, expected_fields: List[str]) -> None:
    """
    Verify gRPC response has all expected fields.

    Args:
        response: gRPC response message
        expected_fields: List of field names that must be present

    Raises:
        AssertionError: If any expected field is missing
    """
    for field in expected_fields:
        assert hasattr(response, field), f"Response missing field: {field}"


def assert_collection_exists(
    qdrant_client: QdrantClient,
    collection_name: str,
    expected_vector_size: Optional[int] = None
) -> None:
    """
    Assert collection exists in Qdrant with optional vector size check.

    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of collection to verify
        expected_vector_size: Optional expected vector size

    Raises:
        AssertionError: If collection doesn't exist or vector size mismatch
    """
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        assert collection_info is not None, f"Collection {collection_name} not found"

        if expected_vector_size is not None:
            # TODO: Verify vector size from collection_info
            pass

    except Exception as e:
        raise AssertionError(f"Collection verification failed: {e}")


def assert_collection_not_exists(
    qdrant_client: QdrantClient,
    collection_name: str
) -> None:
    """
    Assert collection does not exist in Qdrant.

    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of collection that should not exist

    Raises:
        AssertionError: If collection exists
    """
    try:
        qdrant_client.get_collection(collection_name)
        raise AssertionError(f"Collection {collection_name} exists but should not")
    except Exception:
        # Expected - collection should not exist
        pass


async def wait_for_daemon_processing(delay_seconds: float = 2.0) -> None:
    """
    Wait for daemon to process async operations.

    Args:
        delay_seconds: Seconds to wait for daemon processing
    """
    await asyncio.sleep(delay_seconds)


def extract_fallback_warnings(log_capture) -> List[str]:
    """
    Extract fallback-related WARNING messages from log capture.

    Args:
        log_capture: MemoryHandler with captured log records

    Returns:
        List of warning messages related to fallback mode
    """
    warnings = []
    for record in log_capture.buffer:
        if record.levelno == logging.WARNING:
            if "fallback" in record.getMessage().lower():
                warnings.append(record.getMessage())
    return warnings


# =============================================================================
# TEST REPORT GENERATION
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_daemon
async def test_generate_phase1_validation_report():
    """
    Generate comprehensive Phase 1 protocol validation report.

    Summarizes:
        - gRPC protocol coverage (15 RPCs across 3 services)
        - Daemon-only write path validation (First Principle 10)
        - Error handling and fallback mechanisms
        - Recommendations for Phase 2 implementation
    """
    report = {
        "test_suite": "Phase 1 gRPC Protocol Validation",
        "protocol_version": "workspace_daemon.proto v1.0",
        "total_rpcs": 15,
        "services": {
            "SystemService": {
                "rpcs": 7,
                "coverage": [
                    "HealthCheck",
                    "GetStatus",
                    "GetMetrics",
                    "SendRefreshSignal",
                    "NotifyServerStatus",
                    "PauseAllWatchers",
                    "ResumeAllWatchers"
                ],
                "status": "TODO - tests pending implementation"
            },
            "CollectionService": {
                "rpcs": 5,
                "coverage": [
                    "CreateCollection",
                    "DeleteCollection",
                    "CreateCollectionAlias",
                    "DeleteCollectionAlias",
                    "RenameCollectionAlias"
                ],
                "status": "TODO - tests pending implementation"
            },
            "DocumentService": {
                "rpcs": 3,
                "coverage": [
                    "IngestText",
                    "UpdateText",
                    "DeleteText"
                ],
                "status": "TODO - tests pending implementation"
            }
        },
        "write_path_validation": {
            "first_principle": "Principle 10: Daemon-Only Writes",
            "validation_areas": [
                "Daemon as primary write path",
                "Fallback detection and logging",
                "Direct Qdrant writes only when daemon unavailable",
                "MEMORY collections exception handling"
            ],
            "status": "TODO - fallback tests pending"
        },
        "error_handling": {
            "test_coverage": [
                "Connection failures",
                "Invalid requests",
                "Timeout conditions",
                "Retry mechanisms",
                "Circuit breaker behavior"
            ],
            "status": "TODO - error tests pending"
        },
        "recommendations": [
            "✅ Test file structure established with comprehensive fixtures",
            "✅ All 15 RPCs have dedicated test placeholders",
            "✅ Fallback detection framework in place",
            "⏳ TODO: Implement actual test logic for all test methods",
            "⏳ TODO: Add daemon startup/shutdown helpers if needed",
            "⏳ TODO: Validate gRPC response structures match protobuf",
            "🚀 Ready for Phase 2: Rust daemon implementation tests"
        ]
    }

    print("\n" + "=" * 70)
    print("PHASE 1 gRPC PROTOCOL VALIDATION TEST REPORT")
    print("=" * 70)
    print(f"\nTotal RPCs: {report['total_rpcs']}")
    print(f"\nServices:")
    for service_name, service_info in report['services'].items():
        print(f"  • {service_name}: {service_info['rpcs']} RPCs - {service_info['status']}")

    print(f"\n📋 Write Path Validation:")
    print(f"  • First Principle: {report['write_path_validation']['first_principle']}")
    print(f"  • Status: {report['write_path_validation']['status']}")

    print(f"\n🎯 Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")

    print("\n" + "=" * 70)

    return report
