"""
Comprehensive tests for LSP Schema Integration in SQLite State Manager.

This module tests the LSP-specific database schema extensions including:
- Projects table functionality
- LSP servers table functionality  
- File metadata LSP extensions
- Database migration system
- Transaction safety for LSP operations
- Concurrent access patterns
- Data integrity validation
"""

import asyncio
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any
import pytest
import pytest_asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from src.workspace_qdrant_mcp.core.sqlite_state_manager import (
    SQLiteStateManager, FileProcessingStatus, ProcessingPriority, LSPServerStatus,
    FileProcessingRecord, WatchFolderConfig, ProcessingQueueItem, ProjectRecord, LSPServerRecord,
    DatabaseTransaction
)


class TestLSPSchemaIntegration:
    """Test LSP schema integration functionality."""
    
    @pytest_asyncio.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass
    
    @pytest_asyncio.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager with LSP schema."""
        manager = SQLiteStateManager(temp_db_path)
        
        success = await manager.initialize()
        assert success, "State manager initialization failed"
        
        yield manager
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_schema_version_upgrade(self, temp_db_path):
        """Test database schema migration from version 1 to current version."""
        # Create an old version 1 database
        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Create minimal version 1 schema
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.commit()
        conn.close()
        
        # Initialize with current version - should trigger migration
        manager = SQLiteStateManager(temp_db_path)
        success = await manager.initialize()
        assert success
        
        # Verify migration occurred
        async with manager.transaction() as conn:
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0]
            assert current_version == manager.SCHEMA_VERSION
            
            # Verify new tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
            assert cursor.fetchone() is not None
            
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lsp_servers'")
            assert cursor.fetchone() is not None
            
        await manager.close()

    @pytest.mark.asyncio 
    async def test_project_crud_operations(self, state_manager):
        """Test complete CRUD operations for projects."""
        # Create test project
        project = ProjectRecord(
            id=None,
            name="test-project",
            root_path="/path/to/project",
            collection_name="test-collection",
            lsp_enabled=True,
            metadata={"description": "Test project"}
        )
        
        # Create project
        project_id = await state_manager.create_project(project)
        assert project_id is not None
        assert isinstance(project_id, int)
        
        # Read project
        retrieved = await state_manager.get_project(project_id)
        assert retrieved is not None
        assert retrieved.name == "test-project"
        assert retrieved.root_path == "/path/to/project"
        assert retrieved.collection_name == "test-collection"
        assert retrieved.lsp_enabled == True
        assert retrieved.metadata["description"] == "Test project"
        
        # Read by path
        retrieved_by_path = await state_manager.get_project_by_path("/path/to/project")
        assert retrieved_by_path is not None
        assert retrieved_by_path.id == project_id
        
        # Update project
        retrieved.lsp_enabled = False
        retrieved.metadata = {"description": "Updated project", "version": "1.1"}
        success = await state_manager.update_project(retrieved)
        assert success
        
        # Verify update
        updated = await state_manager.get_project(project_id)
        assert updated.lsp_enabled == False
        assert updated.metadata["version"] == "1.1"
        
        # List projects
        all_projects = await state_manager.list_projects()
        assert len(all_projects) == 1
        assert all_projects[0].id == project_id
        
        # List LSP-enabled projects only
        lsp_projects = await state_manager.list_projects(lsp_enabled_only=True)
        assert len(lsp_projects) == 0  # We disabled LSP
        
        # Update scan time
        success = await state_manager.update_project_scan_time(project_id)
        assert success
        
        scan_updated = await state_manager.get_project(project_id)
        assert scan_updated.last_scan is not None
        
        # Delete project
        success = await state_manager.delete_project(project_id)
        assert success
        
        # Verify deletion
        deleted = await state_manager.get_project(project_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_lsp_server_crud_operations(self, state_manager):
        """Test complete CRUD operations for LSP servers."""
        # Create test LSP server
        server = LSPServerRecord(
            id=None,
            language="python",
            server_path="/usr/bin/pylsp",
            version="1.0.0",
            capabilities={"completion": True, "hover": True},
            status=LSPServerStatus.ACTIVE,
            metadata={"install_method": "pip"}
        )
        
        # Create server
        server_id = await state_manager.create_lsp_server(server)
        assert server_id is not None
        assert isinstance(server_id, int)
        
        # Read server
        retrieved = await state_manager.get_lsp_server(server_id)
        assert retrieved is not None
        assert retrieved.language == "python"
        assert retrieved.server_path == "/usr/bin/pylsp"
        assert retrieved.version == "1.0.0"
        assert retrieved.capabilities["completion"] == True
        assert retrieved.status == LSPServerStatus.ACTIVE
        
        # Read by language
        retrieved_by_lang = await state_manager.get_lsp_server_by_language("python")
        assert retrieved_by_lang is not None
        assert retrieved_by_lang.id == server_id
        
        # Update server
        retrieved.status = LSPServerStatus.ERROR
        retrieved.metadata = {"install_method": "pip", "error": "Connection failed"}
        success = await state_manager.update_lsp_server(retrieved)
        assert success
        
        # Update health status
        success = await state_manager.update_lsp_server_health(server_id, LSPServerStatus.ACTIVE)
        assert success
        
        updated = await state_manager.get_lsp_server(server_id)
        assert updated.status == LSPServerStatus.ACTIVE
        assert updated.last_health_check is not None
        
        # List servers
        all_servers = await state_manager.list_lsp_servers()
        assert len(all_servers) == 1
        
        # List by language
        python_servers = await state_manager.list_lsp_servers(language="python")
        assert len(python_servers) == 1
        
        # List by status
        active_servers = await state_manager.list_lsp_servers(status=LSPServerStatus.ACTIVE)
        assert len(active_servers) == 1
        
        # Get active servers
        active = await state_manager.get_active_lsp_servers()
        assert len(active) == 1
        
        # Delete server
        success = await state_manager.delete_lsp_server(server_id)
        assert success
        
        # Verify deletion
        deleted = await state_manager.get_lsp_server(server_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_file_lsp_metadata_operations(self, state_manager):
        """Test LSP metadata operations on file processing records."""
        # First create a file processing record
        success = await state_manager.start_file_processing(
            "/test/file.py",
            "test-collection", 
            priority=ProcessingPriority.HIGH,
            file_size=1024,
            file_hash="abc123"
        )
        assert success
        
        # Create LSP server for reference
        server = LSPServerRecord(
            id=None,
            language="python",
            server_path="/usr/bin/pylsp",
            status=LSPServerStatus.ACTIVE
        )
        server_id = await state_manager.create_lsp_server(server)
        assert server_id is not None
        
        # Update LSP metadata
        success = await state_manager.update_file_lsp_metadata(
            "/test/file.py",
            "python",
            server_id,
            symbols_count=42,
            lsp_metadata={"classes": 3, "functions": 15}
        )
        assert success
        
        # Get LSP metadata
        metadata = await state_manager.get_file_lsp_metadata("/test/file.py")
        assert metadata is not None
        assert metadata["language_id"] == "python"
        assert metadata["lsp_extracted"] == True
        assert metadata["symbols_count"] == 42
        assert metadata["lsp_server_id"] == server_id
        assert metadata["lsp_metadata"]["classes"] == 3
        
        # Get files by language
        python_files = await state_manager.get_files_by_language("python")
        assert "/test/file.py" in python_files
        
        # Mark as needing analysis
        await state_manager.complete_file_processing("/test/file.py", success=True)
        
        # Reset LSP extraction flag
        async with state_manager.transaction() as conn:
            conn.execute("UPDATE file_processing SET lsp_extracted = 0 WHERE file_path = ?", ("/test/file.py",))
        
        # Get files needing analysis
        files_needing = await state_manager.get_files_needing_lsp_analysis()
        assert "/test/file.py" in files_needing
        
        files_needing_python = await state_manager.get_files_needing_lsp_analysis("python")
        assert "/test/file.py" in files_needing_python
        
        # Mark LSP analysis as failed
        success = await state_manager.mark_file_lsp_failed("/test/file.py", "Server unavailable")
        assert success
        
        # Get analysis statistics
        stats = await state_manager.get_lsp_analysis_stats()
        assert "total_files" in stats
        assert "analyzed_files" in stats
        assert "language_breakdown" in stats

    @pytest.mark.asyncio
    async def test_transaction_safety_lsp_operations(self, state_manager):
        """Test transaction safety for LSP operations."""
        # Test rollback on project creation failure
        project = ProjectRecord(
            id=None,
            name="test-project",
            root_path="/path/to/project",
            collection_name="test-collection"
        )
        
        project_id = await state_manager.create_project(project)
        assert project_id is not None
        
        # Try to create duplicate project (should fail due to unique constraint)
        duplicate_project = ProjectRecord(
            id=None,
            name="test-project",  # Same name should fail
            root_path="/different/path",
            collection_name="different-collection"
        )
        
        duplicate_id = await state_manager.create_project(duplicate_project)
        assert duplicate_id is None  # Should fail
        
        # Verify original project still exists
        original = await state_manager.get_project(project_id)
        assert original is not None

    @pytest.mark.asyncio
    async def test_concurrent_access_patterns(self, state_manager):
        """Test concurrent access to LSP tables."""
        # Create multiple projects concurrently
        async def create_project(index):
            project = ProjectRecord(
                id=None,
                name=f"project-{index}",
                root_path=f"/path/to/project{index}",
                collection_name=f"collection-{index}"
            )
            return await state_manager.create_project(project)
        
        # Run concurrent operations
        tasks = [create_project(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        success_count = sum(1 for r in results if r is not None)
        assert success_count == 10
        
        # Verify all projects exist
        all_projects = await state_manager.list_projects()
        assert len(all_projects) == 10
        
        # Test concurrent LSP server operations
        async def create_lsp_server(lang):
            server = LSPServerRecord(
                id=None,
                language=lang,
                server_path=f"/usr/bin/{lang}lsp",
                status=LSPServerStatus.ACTIVE
            )
            return await state_manager.create_lsp_server(server)
        
        languages = ["python", "javascript", "rust", "go", "typescript"]
        server_tasks = [create_lsp_server(lang) for lang in languages]
        server_results = await asyncio.gather(*server_tasks)
        
        # Verify all servers created
        server_success_count = sum(1 for r in server_results if r is not None)
        assert server_success_count == 5
        
        # Verify servers exist
        all_servers = await state_manager.list_lsp_servers()
        assert len(all_servers) == 5

    @pytest.mark.asyncio
    async def test_database_integrity_constraints(self, state_manager):
        """Test database integrity constraints and foreign key relationships."""
        # Create project and server
        project = ProjectRecord(
            id=None,
            name="test-project",
            root_path="/path/to/project",
            collection_name="test-collection"
        )
        project_id = await state_manager.create_project(project)
        
        server = LSPServerRecord(
            id=None,
            language="python",
            server_path="/usr/bin/pylsp",
            status=LSPServerStatus.ACTIVE
        )
        server_id = await state_manager.create_lsp_server(server)
        
        # Create file processing record with LSP server reference
        await state_manager.start_file_processing("/test/file.py", "test-collection")
        await state_manager.update_file_lsp_metadata("/test/file.py", "python", server_id, 10)
        
        # Verify foreign key relationship
        metadata = await state_manager.get_file_lsp_metadata("/test/file.py")
        assert metadata["lsp_server_id"] == server_id
        
        # Delete server - should set file's lsp_server_id to NULL due to foreign key constraint
        await state_manager.delete_lsp_server(server_id)
        
        # Check that file's server reference is now NULL
        async with state_manager.transaction() as conn:
            cursor = conn.execute("SELECT lsp_server_id FROM file_processing WHERE file_path = ?", ("/test/file.py",))
            row = cursor.fetchone()
            assert row[0] is None  # Should be NULL due to ON DELETE SET NULL

    @pytest.mark.asyncio
    async def test_performance_with_large_datasets(self, state_manager):
        """Test performance with larger datasets."""
        import time
        
        # Create many projects
        start_time = time.time()
        project_ids = []
        
        for i in range(100):
            project = ProjectRecord(
                id=None,
                name=f"project-{i:03d}",
                root_path=f"/path/to/project{i:03d}",
                collection_name=f"collection-{i:03d}",
                lsp_enabled=(i % 2 == 0)  # Half enabled
            )
            project_id = await state_manager.create_project(project)
            project_ids.append(project_id)
        
        create_time = time.time() - start_time
        print(f"Created 100 projects in {create_time:.3f}s")
        
        # Test bulk queries
        start_time = time.time()
        all_projects = await state_manager.list_projects()
        query_time = time.time() - start_time
        
        assert len(all_projects) == 100
        print(f"Listed 100 projects in {query_time:.3f}s")
        
        # Test filtered queries
        start_time = time.time()
        lsp_projects = await state_manager.list_projects(lsp_enabled_only=True)
        filter_time = time.time() - start_time
        
        assert len(lsp_projects) == 50  # Half were enabled
        print(f"Filtered projects in {filter_time:.3f}s")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, state_manager):
        """Test error handling and recovery scenarios."""
        # Test invalid project operations
        invalid_project = ProjectRecord(
            id=None,
            name="",  # Invalid empty name
            root_path="/path",
            collection_name="test"
        )
        
        # Should handle gracefully
        project_id = await state_manager.create_project(invalid_project)
        # Note: SQLite allows empty strings, so this might succeed
        
        # Test update of non-existent project
        fake_project = ProjectRecord(
            id=99999,  # Non-existent ID
            name="fake",
            root_path="/fake",
            collection_name="fake"
        )
        
        success = await state_manager.update_project(fake_project)
        assert not success
        
        # Test LSP operations on non-existent files
        success = await state_manager.update_file_lsp_metadata(
            "/non/existent/file.py", 
            "python", 
            1, 
            10
        )
        assert not success
        
        metadata = await state_manager.get_file_lsp_metadata("/non/existent/file.py")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_database_stats_with_lsp_data(self, state_manager):
        """Test database statistics include LSP data."""
        # Create test data
        project = ProjectRecord(
            id=None,
            name="test-project",
            root_path="/path",
            collection_name="test"
        )
        await state_manager.create_project(project)
        
        server = LSPServerRecord(
            id=None,
            language="python",
            server_path="/usr/bin/pylsp",
            status=LSPServerStatus.ACTIVE
        )
        await state_manager.create_lsp_server(server)
        
        # Get database stats
        stats = await state_manager.get_database_stats()
        
        assert "table_counts" in stats
        assert "projects" in stats["table_counts"]
        assert "lsp_servers" in stats["table_counts"]
        assert stats["table_counts"]["projects"] == 1
        assert stats["table_counts"]["lsp_servers"] == 1
        
        # Verify schema version is current
        assert stats["schema_version"] == state_manager.SCHEMA_VERSION


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])