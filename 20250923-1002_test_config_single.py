#!/usr/bin/env python3
"""
Single file test for src/python/common/core/config.py
Testing file-by-file approach with strict 5-minute timeout.

Goal: Prove that file-by-file testing works by testing ONE module successfully.
Target: src/python/common/core/config.py
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path

# Add the src directory to Python path so we can import the module
project_root = Path(__file__).parent
src_path = project_root / "src" / "python"
sys.path.insert(0, str(src_path))

def test_config_module_import():
    """Test that we can import the config module successfully."""
    try:
        from common.core.config import Config, EmbeddingConfig, QdrantConfig, WorkspaceConfig
        assert True, "Config module imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")

def test_config_basic_instantiation():
    """Test basic instantiation of config classes."""
    from common.core.config import Config, EmbeddingConfig, QdrantConfig, WorkspaceConfig

    # Test individual config classes
    embedding_config = EmbeddingConfig()
    assert embedding_config.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_config.chunk_size == 800
    assert embedding_config.enable_sparse_vectors is True

    qdrant_config = QdrantConfig()
    assert qdrant_config.url == "http://localhost:6333"
    assert qdrant_config.timeout == 30
    assert qdrant_config.prefer_grpc is True

    workspace_config = WorkspaceConfig()
    assert workspace_config.collection_types == []
    assert workspace_config.auto_create_collections is False

def test_config_main_class():
    """Test the main Config class instantiation and basic methods."""
    from common.core.config import Config

    # Test basic instantiation
    config = Config()
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.debug is False

    # Test nested config objects
    assert config.qdrant.url == "http://localhost:6333"
    assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.workspace.memory_collection_name == "__memory"

def test_config_qdrant_client_config():
    """Test the qdrant_client_config property."""
    from common.core.config import Config

    config = Config()
    client_config = config.qdrant_client_config

    assert "url" in client_config
    assert "timeout" in client_config
    assert "prefer_grpc" in client_config
    assert client_config["url"] == "http://localhost:6333"
    assert client_config["timeout"] == 30
    assert client_config["prefer_grpc"] is True

def test_config_validation():
    """Test config validation method."""
    from common.core.config import Config

    config = Config()
    issues = config.validate_config()

    # Should return a list (empty or with issues)
    assert isinstance(issues, list)

    # For default config, should have no major issues
    print(f"Validation issues found: {issues}")

def test_config_with_custom_values():
    """Test config with custom initialization values."""
    from common.core.config import Config

    config = Config(
        host="0.0.0.0",
        port=9000,
        debug=True
    )

    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.debug is True

def test_config_to_yaml():
    """Test YAML export functionality."""
    from common.core.config import Config

    config = Config()
    yaml_output = config.to_yaml()

    # Should return a string
    assert isinstance(yaml_output, str)
    assert len(yaml_output) > 0
    assert "host:" in yaml_output
    assert "qdrant:" in yaml_output

if __name__ == "__main__":
    print("Running single file test for config.py...")
    print("=" * 60)

    # Run the tests directly
    test_functions = [
        test_config_module_import,
        test_config_basic_instantiation,
        test_config_main_class,
        test_config_qdrant_client_config,
        test_config_validation,
        test_config_with_custom_values,
        test_config_to_yaml,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("SUCCESS: All tests passed for config.py!")
        exit(0)
    else:
        print("FAILURE: Some tests failed")
        exit(1)