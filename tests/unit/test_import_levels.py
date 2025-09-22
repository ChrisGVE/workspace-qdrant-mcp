"""
Progressive import testing to identify where hanging occurs.
Each test adds more complex imports to isolate the issue.
"""
import sys


def test_level_1_basic_python():
    """Test 1: Basic Python imports."""
    import os
    import json
    import tempfile
    assert os.path.exists("/tmp") or os.path.exists("/var/folders")


def test_level_2_external_libs():
    """Test 2: External libraries that should be available."""
    import fastapi
    import qdrant_client
    assert fastapi.__version__
    assert hasattr(qdrant_client, 'QdrantClient')


def test_level_3_workspace_config():
    """Test 3: Import workspace config (simplest project module)."""
    try:
        from workspace_qdrant_mcp.core.config import Config
        assert Config is not None
    except ImportError as e:
        # Allow this to fail, we're diagnosing
        assert "config" in str(e).lower()


def test_level_4_workspace_utils():
    """Test 4: Import workspace utilities."""
    try:
        from workspace_qdrant_mcp.utils.project_detection import ProjectDetector
        assert ProjectDetector is not None
    except ImportError as e:
        # Allow this to fail, we're diagnosing
        assert "project_detection" in str(e).lower()


def test_level_5_workspace_client():
    """Test 5: Import workspace client (more complex)."""
    try:
        from workspace_qdrant_mcp.core.client import WorkspaceQdrantClient
        assert WorkspaceQdrantClient is not None
    except ImportError as e:
        # Allow this to fail, we're diagnosing
        assert "client" in str(e).lower()


def test_level_6_server_module():
    """Test 6: Import server module (most complex)."""
    try:
        from workspace_qdrant_mcp import server
        assert server is not None
    except ImportError as e:
        # Allow this to fail, we're diagnosing
        assert "server" in str(e).lower()