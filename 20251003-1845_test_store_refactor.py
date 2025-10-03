"""
Test script to verify store() refactor works correctly.
"""
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Import after path setup
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from workspace_qdrant_mcp import server
from common.core.daemon_client import IngestTextResponse

async def test_store_with_daemon():
    """Test store() uses DaemonClient when available."""
    # Mock daemon client
    mock_daemon = AsyncMock()
    mock_response = IngestTextResponse(
        document_id="test-doc-123",
        success=True,
        chunks_created=1
    )
    mock_daemon.ingest_text.return_value = mock_response

    # Mock other components
    mock_qdrant = MagicMock()

    # Get the actual function from the tool
    store_func = server.store.fn

    # Patch globals and skip initialize_components
    with patch.object(server, 'daemon_client', mock_daemon), \
         patch.object(server, 'qdrant_client', mock_qdrant), \
         patch('workspace_qdrant_mcp.server.get_project_name', return_value='test-project'), \
         patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='abc123456789'), \
         patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

        # Call store()
        result = await store_func(
            content="Test content",
            title="Test Document",
            source="user_input"
        )

        # Verify daemon was called
        assert mock_daemon.ingest_text.called
        call_args = mock_daemon.ingest_text.call_args

        # Verify parameters
        assert call_args[1]['content'] == "Test content"
        assert call_args[1]['collection_basename'] == ""  # Project collection
        assert call_args[1]['tenant_id'] == "abc123456789"
        assert call_args[1]['chunk_text'] == True

        # Verify response
        assert result['success'] == True
        assert result['document_id'] == "test-doc-123"
        assert result['chunks_created'] == 1

        # Verify NO direct Qdrant writes
        assert not mock_qdrant.upsert.called

        print("✅ Test passed: store() uses DaemonClient correctly")
        print(f"   - daemon_client.ingest_text() was called")
        print(f"   - qdrant_client.upsert() was NOT called")
        print(f"   - Response: {result}")

async def test_store_fallback_without_daemon():
    """Test store() falls back to direct Qdrant write when daemon unavailable."""
    # Mock components
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = Exception("Collection not found")
    mock_qdrant.create_collection.return_value = True
    mock_qdrant.upsert.return_value = MagicMock(operation_id=123)

    # Mock embedding model - return generator that yields numpy array
    mock_embedding = MagicMock()
    mock_embedding.embed.return_value = (np.array([0.1] * 384) for _ in range(1))

    # Get the actual function from the tool
    store_func = server.store.fn

    # Patch globals (daemon_client is None) and skip initialize_components
    with patch.object(server, 'daemon_client', None), \
         patch.object(server, 'qdrant_client', mock_qdrant), \
         patch.object(server, 'embedding_model', mock_embedding), \
         patch('workspace_qdrant_mcp.server.get_project_name', return_value='test-project'), \
         patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='abc123456789'), \
         patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

        # Call store()
        result = await store_func(
            content="Test content",
            title="Test Document",
            source="user_input"
        )

        # Verify fallback to direct Qdrant write
        assert mock_qdrant.upsert.called
        assert result['success'] == True
        assert 'fallback_mode' in result

        print("✅ Test passed: store() falls back to direct Qdrant write")
        print(f"   - qdrant_client.upsert() was called (fallback)")
        print(f"   - Response: {result}")

async def main():
    print("Testing store() refactor (Task 375.3)\n")

    try:
        await test_store_with_daemon()
        print()
        await test_store_fallback_without_daemon()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
