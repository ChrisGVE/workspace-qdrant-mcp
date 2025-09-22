"""
Basic tests that exercise project code for coverage measurement.
"""
import pytest
from unittest.mock import Mock, patch
import tempfile
import os


def test_config_import_and_basic_usage():
    """Test basic config functionality."""
    from workspace_qdrant_mcp.core.config import Config

    # Test basic instantiation
    config = Config()
    assert config is not None

    # Test some basic attributes exist (based on actual structure)
    assert hasattr(config, 'qdrant')
    assert hasattr(config, 'embedding')
    assert hasattr(config, 'workspace')

    # Test nested attributes
    assert hasattr(config.qdrant, 'url')
    assert hasattr(config.qdrant, 'api_key')


def test_project_detector_basic():
    """Test basic project detector functionality."""
    # Import from the actual location
    import sys
    sys.path.insert(0, '/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/src/python')
    from common.utils.project_detection import ProjectDetector

    # Test instantiation
    detector = ProjectDetector()
    assert detector is not None

    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        result = detector.detect_project(temp_dir)
        # Should return some result even for non-git directory
        assert result is not None


def test_client_class_exists():
    """Test that client class can be imported and instantiated."""
    from workspace_qdrant_mcp.core.client import WorkspaceQdrantClient

    # Mock the config to avoid connection issues
    with patch('workspace_qdrant_mcp.core.client.Config') as mock_config:
        mock_config.return_value.qdrant_url = "http://localhost:6333"
        mock_config.return_value.qdrant_api_key = None

        # Test basic instantiation (should not hang)
        client = WorkspaceQdrantClient()
        assert client is not None
        assert hasattr(client, 'client')


def test_embeddings_basic():
    """Test basic embeddings functionality."""
    from workspace_qdrant_mcp.core.embeddings import EmbeddingManager

    # Test instantiation
    embeddings = EmbeddingManager()
    assert embeddings is not None
    assert hasattr(embeddings, 'model')