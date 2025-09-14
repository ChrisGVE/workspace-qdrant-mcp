#!/usr/bin/env python3
"""
Test migration changes to ensure deprecated fields are properly removed.
"""

import sys
import os
sys.path.append('src/python')

from common.core.config import Config, WorkspaceConfig

def test_deprecated_fields_removed():
    """Test that deprecated fields are removed from WorkspaceConfig."""
    print("Testing deprecated field removal...")

    # Test that deprecated fields don't exist in WorkspaceConfig
    workspace_config = WorkspaceConfig()

    # These should not be attributes anymore
    try:
        _ = workspace_config.collection_prefix
        print("❌ collection_prefix still exists - migration incomplete")
        return False
    except AttributeError:
        print("✅ collection_prefix properly removed")

    try:
        _ = workspace_config.max_collections
        print("❌ max_collections still exists - migration incomplete")
        return False
    except AttributeError:
        print("✅ max_collections properly removed")

    # Test that valid fields still exist
    assert hasattr(workspace_config, 'collection_types'), "collection_types should exist"
    assert hasattr(workspace_config, 'global_collections'), "global_collections should exist"
    assert hasattr(workspace_config, 'github_user'), "github_user should exist"
    assert hasattr(workspace_config, 'auto_create_collections'), "auto_create_collections should exist"
    print("✅ Valid fields still present")

    return True

def test_config_loading():
    """Test that Config can still be loaded without deprecated fields."""
    print("\nTesting Config loading...")

    try:
        config = Config()
        print("✅ Config loads successfully")

        # Test validation works
        issues = config.validate_config()
        print(f"✅ Config validation works, found {len(issues)} issues")

        # Test that workspace config is accessible
        workspace = config.workspace
        assert hasattr(workspace, 'collection_types'), "collection_types should be accessible"
        print("✅ Workspace config accessible")

        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_yaml_export():
    """Test that YAML export works without deprecated fields."""
    print("\nTesting YAML export...")

    try:
        config = Config()
        yaml_output = config.to_yaml()

        # Check that deprecated fields are not in YAML output
        if 'collection_prefix' in yaml_output:
            print("❌ collection_prefix found in YAML output")
            return False

        if 'max_collections' in yaml_output:
            print("❌ max_collections found in YAML output")
            return False

        print("✅ YAML export clean of deprecated fields")
        return True
    except Exception as e:
        print(f"❌ YAML export failed: {e}")
        return False

def test_migration_script():
    """Test that migration script works."""
    print("\nTesting migration script analysis...")

    try:
        sys.path.append('.')
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_migration", "20250114-0906_config_migration.py")
        config_migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_migration)
        ConfigMigration = config_migration.ConfigMigration

        migration = ConfigMigration(dry_run=True)
        analysis = migration.analyze_current_usage()

        print(f"✅ Migration analysis completed")
        print(f"  - Deprecated fields found: {analysis['deprecated_fields_found']}")
        print(f"  - Migration complexity: {analysis['migration_complexity']}")

        return True
    except Exception as e:
        print(f"❌ Migration script test failed: {e}")
        return False

def main():
    """Run all migration tests."""
    print("=== Testing Migration to Multi-Tenant Architecture ===\n")

    tests = [
        test_deprecated_fields_removed,
        test_config_loading,
        test_yaml_export,
        test_migration_script,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print(f"\n=== Migration Test Results ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All migration tests passed! Ready for multi-tenant architecture.")
        return True
    else:
        print("⚠️  Some migration tests failed. Review above output.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)