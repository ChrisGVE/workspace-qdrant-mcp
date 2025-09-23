#!/usr/bin/env python3
"""
Final push to achieve 100% coverage for client.py by testing all code paths directly
"""
import asyncio
import sys
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

def run_comprehensive_coverage():
    """Run comprehensive coverage by exercising all code paths."""

    from common.core.client import QdrantWorkspaceClient, create_qdrant_client
    from common.core.config import Config

    print("="*80)
    print("COMPREHENSIVE COVERAGE TESTING FOR CLIENT.PY")
    print("="*80)

    results = []

    # Test 1: Basic initialization
    try:
        config = Config()
        client = QdrantWorkspaceClient(config)
        results.append("✓ Basic initialization")
    except Exception as e:
        results.append(f"✗ Basic initialization: {e}")

    # Test 2: Project context with different scenarios
    try:
        client = QdrantWorkspaceClient(Config())

        # None project info
        client.project_info = None
        assert client.get_project_context() is None

        # Empty project name
        client.project_info = {"main_project": ""}
        assert client.get_project_context() is None

        # Valid project
        client.project_info = {"main_project": "test-project"}
        context = client.get_project_context("docs")
        assert context["project_name"] == "test-project"
        assert context["collection_type"] == "docs"

        results.append("✓ Project context scenarios")
    except Exception as e:
        results.append(f"✗ Project context scenarios: {e}")

    # Test 3: Project ID generation consistency
    try:
        client = QdrantWorkspaceClient(Config())
        id1 = client._generate_project_id("test-project")
        id2 = client._generate_project_id("test-project")
        assert id1 == id2
        assert len(id1) == 12
        results.append("✓ Project ID generation")
    except Exception as e:
        results.append(f"✗ Project ID generation: {e}")

    # Test 4: Not initialized states
    try:
        client = QdrantWorkspaceClient(Config())

        # List collections
        assert client.list_collections() == []

        # Get searchable collections
        assert client.get_searchable_collections() == []

        # Validate collection access
        allowed, reason = client.validate_collection_access("test", "read")
        assert not allowed
        assert reason == "Client not initialized"

        # Select collections by type
        result = client.select_collections_by_type("memory_collection")
        expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                        'project_collections', 'fallback_collections']
        assert all(key in result and result[key] == [] for key in expected_keys)

        results.append("✓ Not initialized states")
    except Exception as e:
        results.append(f"✗ Not initialized states: {e}")

    # Test 5: Enhanced collection selector error handling
    try:
        client = QdrantWorkspaceClient(Config())

        # Should raise RuntimeError when not initialized
        try:
            client.get_enhanced_collection_selector()
            results.append("✗ Enhanced selector should raise RuntimeError")
        except RuntimeError as e:
            if "Client must be initialized" in str(e):
                results.append("✓ Enhanced selector RuntimeError")
            else:
                results.append(f"✗ Enhanced selector wrong error: {e}")

    except Exception as e:
        results.append(f"✗ Enhanced collection selector: {e}")

    # Test 6: Async methods - not initialized
    async def test_async_not_initialized():
        try:
            client = QdrantWorkspaceClient(Config())

            # Get status
            status = await client.get_status()
            assert status["error"] == "Client not initialized"

            # Search with project context
            result = await client.search_with_project_context("test", {"dense": [0.1] * 384})
            assert result["error"] == "Workspace client not initialized"

            # Ensure collection exists - should raise RuntimeError
            try:
                await client.ensure_collection_exists("test")
                return "✗ ensure_collection_exists should raise RuntimeError"
            except RuntimeError as e:
                if "Client not initialized" in str(e):
                    pass
                else:
                    return f"✗ ensure_collection_exists wrong error: {e}"

            # Empty collection name - with initialized flag
            client.initialized = True
            try:
                await client.ensure_collection_exists("")
                return "✗ Empty collection name should raise ValueError"
            except ValueError as e:
                if "Collection name cannot be empty" in str(e):
                    pass
                else:
                    return f"✗ Empty collection name wrong error: {e}"
            client.initialized = False

            # Create collection
            result = await client.create_collection("test")
            assert result["error"] == "Client not initialized"

            # Close with None services
            client.embedding_service = None
            client.client = None
            await client.close()
            assert not client.initialized

            return "✓ Async not initialized states"

        except Exception as e:
            return f"✗ Async not initialized: {e}"

    # Run async tests
    async_result = asyncio.run(test_async_not_initialized())
    results.append(async_result)

    # Test 7: Refresh project detection path coverage
    try:
        client = QdrantWorkspaceClient(Config())

        # With None project detector - should initialize new one
        client.project_detector = None

        with patch('common.utils.project_detection.ProjectDetector') as mock_detector:
            mock_detector_instance = MagicMock()
            mock_detector_instance.get_project_info.return_value = {
                "main_project": "new-project",
                "subprojects": [],
                "git_info": {},
                "directory_structure": {}
            }
            mock_detector.return_value = mock_detector_instance

            result = client.refresh_project_detection()
            assert result["main_project"] == "new-project"
            assert client.project_detector is mock_detector_instance

        results.append("✓ Refresh project detection")
    except Exception as e:
        results.append(f"✗ Refresh project detection: {e}")

    # Test 8: Factory function
    try:
        client = create_qdrant_client({"host": "localhost", "port": 6333})
        assert isinstance(client, QdrantWorkspaceClient)
        results.append("✓ Factory function")
    except Exception as e:
        results.append(f"✗ Factory function: {e}")

    # Test 9: Complex initialize scenarios with mocking
    async def test_initialize_scenarios():
        try:
            config = Config()
            config.environment = "development"
            config.qdrant = MagicMock()
            config.qdrant.url = "http://localhost:6333"
            config.qdrant_client_config = {"host": "localhost", "port": 6333}
            config.embedding = MagicMock()
            config.embedding.enable_sparse_vectors = True
            config.workspace = MagicMock()
            config.workspace.github_user = "testuser"

            client = QdrantWorkspaceClient(config)

            # Mock all the complex dependencies
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

            mock_collection_manager = MagicMock()
            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock(return_value={"created": [], "existing": []})

            mock_project_detector = MagicMock()
            mock_project_detector.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": ["frontend"],
                "git_info": {},
                "directory_structure": {}
            }

            mock_ssl_manager = MagicMock()
            mock_ssl_manager.is_localhost_url.return_value = True
            mock_ssl_context = MagicMock()
            mock_ssl_context.__enter__ = Mock(return_value=None)
            mock_ssl_context.__exit__ = Mock(return_value=None)
            mock_ssl_manager.for_localhost.return_value = mock_ssl_context

            with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client), \
                 patch('common.core.collections.WorkspaceCollectionManager', return_value=mock_collection_manager), \
                 patch('common.core.collections.MemoryCollectionManager', return_value=mock_memory_manager), \
                 patch.object(client.embedding_service, 'initialize', new_callable=AsyncMock), \
                 patch('common.utils.project_detection.ProjectDetector', return_value=mock_project_detector), \
                 patch('common.core.ssl_config.get_ssl_manager', return_value=mock_ssl_manager), \
                 patch('common.core.ssl_config.create_secure_qdrant_config', return_value={"host": "localhost", "port": 6333}), \
                 patch('common.core.ssl_config.suppress_qdrant_ssl_warnings') as mock_ssl_suppress:

                mock_ssl_suppress.return_value.__enter__ = Mock()
                mock_ssl_suppress.return_value.__exit__ = Mock(return_value=None)

                await client.initialize()

                assert client.initialized is True
                assert client.client is mock_qdrant_client
                assert client.collection_manager is mock_collection_manager
                assert client.project_info["main_project"] == "test-project"

            # Test idempotent initialize
            await client.initialize()  # Should return early

            return "✓ Initialize scenarios"

        except Exception as e:
            return f"✗ Initialize scenarios: {e}"

    # Run complex initialize test
    init_result = asyncio.run(test_initialize_scenarios())
    results.append(init_result)

    # Test 10: Exception handling in various methods
    async def test_exception_handling():
        try:
            client = QdrantWorkspaceClient(Config())
            client.initialized = True

            # Test get_status with client exception
            mock_client = MagicMock()
            mock_client.get_collections.side_effect = Exception("Connection lost")
            client.client = mock_client

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = AsyncMock()
                mock_executor.run_in_executor = AsyncMock(side_effect=Exception("Connection lost"))
                mock_loop.return_value = mock_executor

                status = await client.get_status()
                assert "error" in status
                assert "Connection lost" in status["error"]

            # Test list_collections with exception
            mock_collection_manager = MagicMock()
            mock_collection_manager.list_workspace_collections.side_effect = Exception("Manager error")
            client.collection_manager = mock_collection_manager

            collections = client.list_collections()
            assert collections == []

            # Test select_collections_by_type with exception
            with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")):
                result = client.select_collections_by_type("memory_collection")
                expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                                'project_collections', 'fallback_collections']
                assert all(key in result and result[key] == [] for key in expected_keys)

            # Test get_searchable_collections with exception fallback
            client.collection_manager = MagicMock()
            client.collection_manager.list_workspace_collections.return_value = ["fallback-collection"]

            with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")):
                result = client.get_searchable_collections()
                assert "fallback-collection" in result

            # Test validate_collection_access with exception
            with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Validation error")):
                is_allowed, reason = client.validate_collection_access("test", "read")
                assert not is_allowed
                assert "Validation error" in reason

            return "✓ Exception handling"

        except Exception as e:
            return f"✗ Exception handling: {e}"

    # Run exception handling tests
    exception_result = asyncio.run(test_exception_handling())
    results.append(exception_result)

    # Print results
    print("\nTEST RESULTS:")
    print("-" * 50)

    passed = 0
    failed = 0

    for result in results:
        print(result)
        if result.startswith("✓"):
            passed += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print("COVERAGE TESTING SUMMARY")
    print("="*80)
    print(f"TOTAL TESTS: {passed + failed}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"SUCCESS RATE: {passed / (passed + failed) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✓ Basic functionality covered")
        print("✓ Edge cases covered")
        print("✓ Exception handling covered")
        print("✓ Async methods covered")
        print("✓ Complex initialization covered")
        print("\nREADY FOR COVERAGE MEASUREMENT!")
    else:
        print(f"\n❗ {failed} tests failed. Review and fix before proceeding.")

    return passed == len(results)

if __name__ == "__main__":
    success = run_comprehensive_coverage()
    sys.exit(0 if success else 1)