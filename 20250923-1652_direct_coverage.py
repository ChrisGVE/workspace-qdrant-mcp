#!/usr/bin/env python3
"""
Direct coverage analysis for client.py without pytest interference
"""
import sys
import os
import tempfile
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

def analyze_coverage():
    """Analyze current coverage of client.py by importing and testing methods."""

    # Import the client module directly
    try:
        from common.core.client import QdrantWorkspaceClient, create_qdrant_client
        from common.core.config import Config
        print("✓ Successfully imported client module")
    except Exception as e:
        print(f"✗ Failed to import client module: {e}")
        return

    # Test coverage by calling methods
    coverage_results = []

    # Test 1: Client initialization
    try:
        config = Config()
        client = QdrantWorkspaceClient(config)
        coverage_results.append(("Client.__init__", True, "Constructor works"))
    except Exception as e:
        coverage_results.append(("Client.__init__", False, str(e)))

    # Test 2: get_project_info with None
    try:
        client = QdrantWorkspaceClient(Config())
        client.project_info = None
        result = client.get_project_info()
        coverage_results.append(("get_project_info(None)", True, f"Returns: {result}"))
    except Exception as e:
        coverage_results.append(("get_project_info(None)", False, str(e)))

    # Test 3: get_project_context with None project
    try:
        client = QdrantWorkspaceClient(Config())
        client.project_info = None
        result = client.get_project_context()
        coverage_results.append(("get_project_context(None)", True, f"Returns: {result}"))
    except Exception as e:
        coverage_results.append(("get_project_context(None)", False, str(e)))

    # Test 4: get_project_context with empty project name
    try:
        client = QdrantWorkspaceClient(Config())
        client.project_info = {"main_project": ""}
        result = client.get_project_context()
        coverage_results.append(("get_project_context(empty)", True, f"Returns: {result}"))
    except Exception as e:
        coverage_results.append(("get_project_context(empty)", False, str(e)))

    # Test 5: get_project_context with valid project
    try:
        client = QdrantWorkspaceClient(Config())
        client.project_info = {"main_project": "test-project"}
        result = client.get_project_context("docs")
        coverage_results.append(("get_project_context(valid)", True, f"Returns valid context"))
    except Exception as e:
        coverage_results.append(("get_project_context(valid)", False, str(e)))

    # Test 6: _generate_project_id
    try:
        client = QdrantWorkspaceClient(Config())
        project_id = client._generate_project_id("test-project")
        coverage_results.append(("_generate_project_id", True, f"Generated: {project_id}"))
    except Exception as e:
        coverage_results.append(("_generate_project_id", False, str(e)))

    # Test 7: get_embedding_service
    try:
        client = QdrantWorkspaceClient(Config())
        service = client.get_embedding_service()
        coverage_results.append(("get_embedding_service", True, f"Service type: {type(service)}"))
    except Exception as e:
        coverage_results.append(("get_embedding_service", False, str(e)))

    # Test 8: list_collections (not initialized)
    try:
        client = QdrantWorkspaceClient(Config())
        result = client.list_collections()
        coverage_results.append(("list_collections(not_init)", True, f"Returns: {result}"))
    except Exception as e:
        coverage_results.append(("list_collections(not_init)", False, str(e)))

    # Test 9: select_collections_by_type (not initialized)
    try:
        client = QdrantWorkspaceClient(Config())
        result = client.select_collections_by_type("memory_collection")
        expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                        'project_collections', 'fallback_collections']
        has_all_keys = all(key in result for key in expected_keys)
        coverage_results.append(("select_collections_by_type(not_init)", True, f"Has all keys: {has_all_keys}"))
    except Exception as e:
        coverage_results.append(("select_collections_by_type(not_init)", False, str(e)))

    # Test 10: get_searchable_collections (not initialized)
    try:
        client = QdrantWorkspaceClient(Config())
        result = client.get_searchable_collections()
        coverage_results.append(("get_searchable_collections(not_init)", True, f"Returns: {result}"))
    except Exception as e:
        coverage_results.append(("get_searchable_collections(not_init)", False, str(e)))

    # Test 11: validate_collection_access (not initialized)
    try:
        client = QdrantWorkspaceClient(Config())
        is_allowed, reason = client.validate_collection_access("test", "read")
        coverage_results.append(("validate_collection_access(not_init)", True, f"Allowed: {is_allowed}, Reason: {reason}"))
    except Exception as e:
        coverage_results.append(("validate_collection_access(not_init)", False, str(e)))

    # Test 12: create_qdrant_client factory function
    try:
        config_data = {"host": "localhost", "port": 6333}
        client = create_qdrant_client(config_data)
        coverage_results.append(("create_qdrant_client", True, f"Client type: {type(client)}"))
    except Exception as e:
        coverage_results.append(("create_qdrant_client", False, str(e)))

    # Print results
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS RESULTS")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, success, details in coverage_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name:<35} {details}")
        if success:
            passed += 1
        else:
            failed += 1

    print("="*80)
    print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"SUCCESS RATE: {passed / (passed + failed) * 100:.1f}%")

    # Now run async tests
    print("\n" + "="*80)
    print("ASYNC COVERAGE TESTS")
    print("="*80)

    import asyncio

    async def run_async_tests():
        async_results = []
        config = Config()

        # Async Test 1: get_status (not initialized)
        try:
            client = QdrantWorkspaceClient(config)
            status = await client.get_status()
            async_results.append(("get_status(not_init)", True, f"Status: {status.get('error', 'OK')}"))
        except Exception as e:
            async_results.append(("get_status(not_init)", False, str(e)))

        # Async Test 2: search_with_project_context (not initialized)
        try:
            client = QdrantWorkspaceClient(config)
            result = await client.search_with_project_context(
                "test-collection",
                {"dense": [0.1] * 384}
            )
            async_results.append(("search_project_context(not_init)", True, f"Error: {result.get('error', 'OK')}"))
        except Exception as e:
            async_results.append(("search_project_context(not_init)", False, str(e)))

        # Async Test 3: ensure_collection_exists (not initialized)
        try:
            client = QdrantWorkspaceClient(config)
            await client.ensure_collection_exists("test-collection")
            async_results.append(("ensure_collection_exists(not_init)", False, "Should have raised RuntimeError"))
        except RuntimeError as e:
            async_results.append(("ensure_collection_exists(not_init)", True, f"Correctly raised: {str(e)[:50]}"))
        except Exception as e:
            async_results.append(("ensure_collection_exists(not_init)", False, str(e)))

        # Async Test 4: ensure_collection_exists (empty name)
        try:
            client = QdrantWorkspaceClient(config)
            client.initialized = True
            await client.ensure_collection_exists("")
            async_results.append(("ensure_collection_exists(empty)", False, "Should have raised ValueError"))
        except ValueError as e:
            async_results.append(("ensure_collection_exists(empty)", True, f"Correctly raised: {str(e)[:50]}"))
        except Exception as e:
            async_results.append(("ensure_collection_exists(empty)", False, str(e)))

        # Async Test 5: create_collection (not initialized)
        try:
            client = QdrantWorkspaceClient(config)
            result = await client.create_collection("test-collection")
            async_results.append(("create_collection(not_init)", True, f"Error: {result.get('error', 'OK')}"))
        except Exception as e:
            async_results.append(("create_collection(not_init)", False, str(e)))

        # Async Test 6: close with None services
        try:
            client = QdrantWorkspaceClient(config)
            client.embedding_service = None
            client.client = None
            await client.close()
            async_results.append(("close(none_services)", True, "Closed without error"))
        except Exception as e:
            async_results.append(("close(none_services)", False, str(e)))

        return async_results

    # Run async tests
    async_results = asyncio.run(run_async_tests())

    async_passed = 0
    async_failed = 0

    for test_name, success, details in async_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name:<35} {details}")
        if success:
            async_passed += 1
        else:
            async_failed += 1

    print("="*80)
    print(f"ASYNC SUMMARY: {async_passed} passed, {async_failed} failed, {async_passed + async_failed} total")
    print(f"ASYNC SUCCESS RATE: {async_passed / (async_passed + async_failed) * 100:.1f}%")

    # Overall summary
    total_passed = passed + async_passed
    total_tests = passed + failed + async_passed + async_failed

    print("\n" + "="*80)
    print("OVERALL COVERAGE SUMMARY")
    print("="*80)
    print(f"TOTAL TESTS: {total_tests}")
    print(f"TOTAL PASSED: {total_passed}")
    print(f"TOTAL FAILED: {failed + async_failed}")
    print(f"SUCCESS RATE: {total_passed / total_tests * 100:.1f}%")

if __name__ == "__main__":
    analyze_coverage()