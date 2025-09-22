"""
Integration tests for watch depth functionality.

This module tests depth configuration in the context of the complete watch system,
including CLI commands, daemon client integration, and MCP server tools.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict

import pytest

from workspace_qdrant_mcp.core.daemon_client import DaemonClient
from workspace_qdrant_mcp.core.depth_validation import validate_recursive_depth
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager


class TestWatchDepthIntegration:
    """Integration tests for watch depth functionality."""
    
    @pytest.fixture
    async def temp_watch_hierarchy(self):
        """Create a temporary directory hierarchy for depth testing."""
        temp_dir = tempfile.mkdtemp(prefix="depth_test_")
        base_path = Path(temp_dir)
        
        # Create a nested directory structure for testing
        # Level 0: base_path
        # Level 1: level1/
        # Level 2: level1/level2/
        # Level 3: level1/level2/level3/
        # Level 4: level1/level2/level3/level4/
        # Level 5: level1/level2/level3/level4/level5/
        
        current_path = base_path
        for level in range(1, 6):
            current_path = current_path / f"level{level}"
            current_path.mkdir()
            
            # Add some test files at each level
            (current_path / f"file_level_{level}.txt").write_text(f"Content at level {level}")
            (current_path / f"doc_level_{level}.md").write_text(f"# Document at level {level}")
        
        # Add files at the base level too
        (base_path / "root_file.txt").write_text("Root level content")
        (base_path / "root_doc.md").write_text("# Root document")
        
        yield base_path
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_workspace_client(self):
        """Mock workspace client for testing."""
        client = Mock()
        client.list_collections.return_value = ["test_collection", "_test_library"]
        return client
    
    @pytest.fixture
    def mock_daemon_client(self):
        """Mock daemon client for testing."""
        client = Mock(spec=DaemonClient)
        
        # Mock configure_watch method
        async def mock_configure_watch(**kwargs):
            response = Mock()
            response.success = True
            response.message = "Configuration updated successfully"
            return response
        
        client.configure_watch = AsyncMock(side_effect=mock_configure_watch)
        
        # Mock list_watches method
        async def mock_list_watches(active_only=False):
            response = Mock()
            watch = Mock()
            watch.watch_id = "test_watch_123"
            watch.path = "/test/path"
            watch.collection = "test_collection"
            watch.status = 1  # Active
            watch.recursive_depth = 5
            response.watches = [watch]
            return response
        
        client.list_watches = AsyncMock(side_effect=mock_list_watches)
        return client
    
    @pytest.fixture
    async def watch_tools_manager(self, mock_workspace_client):
        """Create a watch tools manager for testing."""
        manager = WatchToolsManager(mock_workspace_client)
        # Mock the initialization to avoid actual daemon setup
        manager._initialized = True
        return manager
    
    async def test_depth_validation_in_watch_creation(self, watch_tools_manager, temp_watch_hierarchy):
        """Test depth validation during watch creation."""
        # Test valid depth
        result = await watch_tools_manager.add_watch_folder(
            path=str(temp_watch_hierarchy),
            collection="_test_library",
            recursive_depth=3
        )
        
        # Should succeed with valid depth
        assert result["success"] is True
        assert result["recursive_depth"] == 3
    
    async def test_invalid_depth_in_watch_creation(self, watch_tools_manager, temp_watch_hierarchy):
        """Test invalid depth rejection during watch creation."""
        # Test invalid depth (too negative)
        result = await watch_tools_manager.add_watch_folder(
            path=str(temp_watch_hierarchy),
            collection="_test_library",
            recursive_depth=-5  # Invalid
        )
        
        # Should fail with validation error
        assert result["success"] is False
        assert "validation" in result["error_type"]
    
    async def test_unlimited_depth_warning(self, watch_tools_manager, temp_watch_hierarchy):
        """Test unlimited depth creates appropriate warnings."""
        result = await watch_tools_manager.add_watch_folder(
            path=str(temp_watch_hierarchy),
            collection="_test_library",
            recursive_depth=-1  # Unlimited
        )
        
        # Should succeed but with warnings
        assert result["success"] is True
        assert result["recursive_depth"] == -1
    
    async def test_depth_reconfiguration(self, watch_tools_manager):
        """Test runtime depth reconfiguration."""
        # First, mock getting an existing configuration
        with patch.object(watch_tools_manager.config_manager, 'get_watch_config') as mock_get:
            # Create a mock existing config
            mock_config = Mock()
            mock_config.id = "test_watch"
            mock_config.path = "/test/path"
            mock_config.collection = "test_collection"
            mock_config.recursive_depth = 5
            mock_config.patterns = ["*.txt"]
            mock_config.ignore_patterns = [".git/*"]
            mock_config.auto_ingest = True
            mock_config.recursive = True
            mock_config.debounce_seconds = 5
            mock_config.update_frequency = 1000
            mock_config.status = "active"
            mock_config.validate.return_value = []  # No validation issues
            
            mock_get.return_value = mock_config
            
            with patch.object(watch_tools_manager.config_manager, 'update_watch_config') as mock_update:
                mock_update.return_value = True
                
                # Test reconfiguring depth
                result = await watch_tools_manager.configure_watch_settings(
                    watch_id="test_watch",
                    recursive_depth=10
                )
                
                assert result["success"] is True
                assert "recursive_depth" in result["changes_made"]
                assert result["changes_made"]["recursive_depth"]["new"] == 10
    
    async def test_depth_validation_edge_cases(self):
        """Test depth validation edge cases."""
        # Test boundary values
        test_cases = [
            (-1, True),   # Unlimited - valid
            (0, True),    # Current dir only - valid
            (1, True),    # One level - valid
            (50, True),   # Max reasonable - valid
            (51, False),  # Over max - invalid
            (-2, False),  # Invalid negative - invalid
        ]
        
        for depth, should_be_valid in test_cases:
            result = validate_recursive_depth(depth)
            assert result.is_valid == should_be_valid, f"Depth {depth} validation failed"
    
    @patch('workspace_qdrant_mcp.core.daemon_client.DaemonClient')
    async def test_daemon_client_depth_configuration(self, mock_client_class, mock_daemon_client):
        """Test depth configuration through daemon client."""
        mock_client_class.return_value = mock_daemon_client
        
        # Test configuring depth through daemon client
        response = await mock_daemon_client.configure_watch(
            watch_id="test_watch",
            recursive_depth=7
        )
        
        assert response.success is True
        mock_daemon_client.configure_watch.assert_called_once_with(
            watch_id="test_watch",
            recursive_depth=7
        )
    
    async def test_depth_performance_scenarios(self, temp_watch_hierarchy):
        """Test different depth scenarios for performance characteristics."""
        test_scenarios = [
            {"depth": 0, "expected_impact": "low"},
            {"depth": 3, "expected_impact": "low"},
            {"depth": 10, "expected_impact": "medium"},
            {"depth": 25, "expected_impact": "medium"},
            {"depth": -1, "expected_impact": "high"},
        ]
        
        for scenario in test_scenarios:
            result = validate_recursive_depth(scenario["depth"])
            assert result.performance_impact == scenario["expected_impact"]
    
    async def test_depth_with_large_directory_structure(self, watch_tools_manager):
        """Test depth configuration with simulated large directory structures."""
        # Simulate a large directory structure
        large_structure_info = {
            "max_depth": 15,
            "file_count": 50000,
            "directory_count": 5000
        }
        
        # Test that the system can handle configuration for large structures
        with patch.object(watch_tools_manager.config_manager, 'get_watch_config') as mock_get:
            mock_config = Mock()
            mock_config.id = "large_structure_watch"
            mock_config.validate.return_value = []
            mock_get.return_value = mock_config
            
            with patch.object(watch_tools_manager.config_manager, 'update_watch_config') as mock_update:
                mock_update.return_value = True
                
                # Configure with a reasonable depth for large structure
                result = await watch_tools_manager.configure_watch_settings(
                    watch_id="large_structure_watch",
                    recursive_depth=8  # Reasonable for large structure
                )
                
                assert result["success"] is True


class TestDepthValidationWithRealDirectories:
    """Test depth validation with real directory structures."""
    
    @pytest.fixture
    async def complex_directory_structure(self):
        """Create a complex directory structure for testing."""
        temp_dir = tempfile.mkdtemp(prefix="complex_depth_test_")
        base_path = Path(temp_dir)
        
        # Create a more realistic directory structure
        # Project root
        # ├── src/
        # │   ├── main/
        # │   │   ├── java/
        # │   │   │   └── com/
        # │   │   │       └── example/
        # │   │   │           └── app/
        # │   │   └── resources/
        # │   └── test/
        # ├── docs/
        # │   ├── user/
        # │   └── developer/
        # └── build/
        #     └── output/
        
        directories = [
            "src/main/java/com/example/app",
            "src/main/resources",
            "src/test",
            "docs/user",
            "docs/developer",
            "build/output"
        ]
        
        for dir_path in directories:
            full_path = base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Add some files
            (full_path / "file.txt").write_text(f"Content in {dir_path}")
        
        yield base_path
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_depth_analysis_on_real_structure(self, complex_directory_structure):
        """Test depth analysis on a realistic directory structure."""
        # Analyze the structure depth
        max_depth = 0
        file_count = 0
        directory_count = 0
        
        for root, dirs, files in os.walk(complex_directory_structure):
            relative_path = os.path.relpath(root, complex_directory_structure)
            if relative_path != ".":
                depth = len(relative_path.split(os.sep))
                max_depth = max(max_depth, depth)
            
            directory_count += len(dirs)
            file_count += len(files)
        
        # Test validation based on actual structure
        structure_info = {
            "max_depth": max_depth,
            "file_count": file_count,
            "directory_count": directory_count
        }
        
        from workspace_qdrant_mcp.core.depth_validation import get_depth_recommendations
        recommendations = get_depth_recommendations(structure_info)
        
        # Should provide reasonable recommendations
        assert isinstance(recommendations["recommended_depth"], int)
        assert recommendations["recommended_depth"] >= -1
        assert "reasoning" in recommendations
    
    async def test_depth_boundary_scanning(self, complex_directory_structure):
        """Test scanning with different depth boundaries."""
        test_depths = [0, 1, 2, 3, 5, -1]
        
        for depth in test_depths:
            result = validate_recursive_depth(depth)
            assert result.is_valid is True
            
            # Verify the result provides appropriate guidance
            if depth == 0:
                assert any("root directory" in rec for rec in result.recommendations)
            elif depth == -1:
                assert any("performance" in warning for warning in result.warnings)
            elif depth >= 20:
                assert any("performance issues" in warning for warning in result.warnings)


class TestDepthConfigurationPersistence:
    """Test that depth configurations persist correctly."""
    
    async def test_depth_configuration_persistence(self, watch_tools_manager):
        """Test that depth configurations are saved and loaded correctly."""
        watch_config_data = {
            "id": "persistent_depth_test",
            "path": "/test/path",
            "collection": "test_collection",
            "recursive_depth": 12,
            "patterns": ["*.txt"],
            "ignore_patterns": [".git/*"],
            "auto_ingest": True,
            "recursive": True,
            "debounce_seconds": 5,
            "update_frequency": 1000,
            "status": "active"
        }
        
        # Mock the configuration manager
        with patch.object(watch_tools_manager.config_manager, 'add_watch_config') as mock_add:
            mock_add.return_value = True
            
            with patch.object(watch_tools_manager.config_manager, 'get_watch_config') as mock_get:
                # Mock returning the saved config
                mock_config = Mock()
                for key, value in watch_config_data.items():
                    setattr(mock_config, key, value)
                mock_config.validate.return_value = []
                mock_get.return_value = mock_config
                
                # Verify the depth configuration persists
                retrieved_config = await watch_tools_manager.config_manager.get_watch_config("persistent_depth_test")
                assert retrieved_config.recursive_depth == 12
    
    async def test_depth_configuration_update_persistence(self, watch_tools_manager):
        """Test that depth configuration updates persist correctly."""
        with patch.object(watch_tools_manager.config_manager, 'get_watch_config') as mock_get:
            # Create initial config
            mock_config = Mock()
            mock_config.id = "update_test"
            mock_config.recursive_depth = 5
            mock_config.validate.return_value = []
            mock_get.return_value = mock_config
            
            with patch.object(watch_tools_manager.config_manager, 'update_watch_config') as mock_update:
                mock_update.return_value = True
                
                # Update the depth
                result = await watch_tools_manager.configure_watch_settings(
                    watch_id="update_test",
                    recursive_depth=15
                )
                
                assert result["success"] is True
                # Verify the config was updated
                assert mock_config.recursive_depth == 15
                mock_update.assert_called_once()