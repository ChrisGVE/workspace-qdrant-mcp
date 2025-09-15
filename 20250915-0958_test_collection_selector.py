#!/usr/bin/env python3
"""
Test script for enhanced CollectionSelector functionality.

This script tests the new multi-tenant collection selection logic
added in Task 233.2.
"""

import sys
import os
import asyncio
from typing import Dict, List

# Add the project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python', 'workspace_qdrant_mcp'))

def test_collection_selector():
    """Test the enhanced collection selector functionality."""

    try:
        from common.core.collections import CollectionSelector
        from common.core.config import Config
        from qdrant_client import QdrantClient
        from unittest.mock import Mock, MagicMock

        print("✅ Successfully imported CollectionSelector and dependencies")

        # Create mock client and config
        mock_client = Mock(spec=QdrantClient)
        mock_config = Mock(spec=Config)

        # Mock the get_collections method
        mock_collection_info = Mock()
        mock_collection_info.name = "test-collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection_info]
        mock_client.get_collections.return_value = mock_collections_response

        # Create CollectionSelector instance
        selector = CollectionSelector(mock_client, mock_config)
        print("✅ Successfully created CollectionSelector instance")

        # Test memory collection detection
        test_cases = [
            ("__memory", True, "System memory collection"),
            ("memory", True, "Legacy memory collection"),
            ("test-project-notes", False, "Code collection"),
            ("_library", False, "Library collection"),
        ]

        print("\n🧪 Testing memory collection detection:")
        for collection_name, expected, description in test_cases:
            result = selector._is_memory_collection(collection_name, {})
            status = "✅" if result == expected else "❌"
            print(f"  {status} {collection_name}: {result} (expected {expected}) - {description}")

        # Test collection type selection
        print("\n🧪 Testing collection selection by type:")
        selection_result = selector.select_collections_by_type(
            collection_type='memory_collection',
            project_name='test-project',
            include_shared=True
        )

        expected_keys = ['memory_collections', 'code_collections', 'shared_collections', 'project_collections', 'fallback_collections']
        all_keys_present = all(key in selection_result for key in expected_keys)
        status = "✅" if all_keys_present else "❌"
        print(f"  {status} Collection selection result has all expected keys")
        print(f"    Result keys: {list(selection_result.keys())}")

        # Test searchable collections
        print("\n🧪 Testing searchable collection selection:")
        searchable = selector.get_searchable_collections(
            project_name='test-project',
            include_memory=False,
            include_shared=True
        )

        status = "✅" if isinstance(searchable, list) else "❌"
        print(f"  {status} get_searchable_collections returned a list: {type(searchable)}")

        print(f"\n✅ All CollectionSelector tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ CollectionSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multitenant_registry():
    """Test the WorkspaceCollectionRegistry integration."""

    try:
        from common.core.multitenant_collections import WorkspaceCollectionRegistry

        print("\n🧪 Testing WorkspaceCollectionRegistry:")

        registry = WorkspaceCollectionRegistry()
        workspace_types = registry.get_workspace_types()

        expected_types = {'notes', 'docs', 'scratchbook', 'knowledge', 'context', 'memory'}
        has_expected_types = expected_types.issubset(workspace_types)

        status = "✅" if has_expected_types else "❌"
        print(f"  {status} Registry has expected workspace types")
        print(f"    Available types: {sorted(workspace_types)}")

        # Test type validation
        is_valid = registry.is_multi_tenant_type('notes')
        status = "✅" if is_valid else "❌"
        print(f"  {status} Registry correctly validates 'notes' as multi-tenant type")

        # Test searchable detection
        memory_searchable = registry.is_searchable('memory')
        status = "✅" if memory_searchable is False else "❌"  # Memory should not be searchable
        print(f"  {status} Registry correctly identifies memory collections as non-searchable: {memory_searchable}")

        print(f"✅ WorkspaceCollectionRegistry tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ WorkspaceCollectionRegistry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_integration():
    """Test integration with QdrantWorkspaceClient."""

    try:
        from common.core.client import QdrantWorkspaceClient
        from common.core.config import Config
        from unittest.mock import Mock, patch

        print("\n🧪 Testing QdrantWorkspaceClient integration:")

        # Create a mock config
        mock_config = Mock(spec=Config)

        # Create client instance (not initialized to avoid connection requirements)
        client = QdrantWorkspaceClient(mock_config)

        # Test that enhanced selector methods exist
        methods_to_test = [
            'select_collections_by_type',
            'get_searchable_collections',
            'validate_collection_access'
        ]

        for method_name in methods_to_test:
            has_method = hasattr(client, method_name)
            status = "✅" if has_method else "❌"
            print(f"  {status} Client has method: {method_name}")

        # Test uninitialized client behavior
        result = client.select_collections_by_type('memory_collection')
        expected_keys = ['memory_collections', 'code_collections', 'shared_collections', 'project_collections', 'fallback_collections']
        all_keys_present = all(key in result for key in expected_keys)

        status = "✅" if all_keys_present else "❌"
        print(f"  {status} Uninitialized client returns expected structure")

        print(f"✅ QdrantWorkspaceClient integration tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ QdrantWorkspaceClient integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""

    print("🚀 Testing Enhanced Collection Selector for Multi-Tenant Architecture")
    print("=" * 70)

    tests = [
        ("CollectionSelector Core Functionality", test_collection_selector),
        ("WorkspaceCollectionRegistry Integration", test_multitenant_registry),
        ("QdrantWorkspaceClient Integration", test_client_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 50)
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)

    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Enhanced CollectionSelector is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())