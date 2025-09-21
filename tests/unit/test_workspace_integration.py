"""
Unit tests for workspace integration and project detection in context injection.

Tests the integration between document processing workflows and workspace management,
including project detection, multi-tenant architecture support, and workspace context
extraction for LLM rule injection.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

from common.core.config import Config
from common.utils.project_detection import ProjectDetector
from wqm_cli.cli.parsers.base import ParsedDocument


class MockWorkspaceManager:
    """Mock workspace manager for testing integration."""

    def __init__(self, project_detector: ProjectDetector):
        self.project_detector = project_detector
        self.workspaces = {}  # workspace_id -> workspace_info
        self.active_workspace = None
        self.processed_files = {}  # file_path -> processing_info
        self.workspace_contexts = {}  # workspace_id -> context

    async def create_workspace(
        self,
        workspace_id: str,
        project_root: str,
        **workspace_options
    ) -> Dict[str, Any]:
        """Create a new workspace with project detection."""
        project_info = await self.project_detector.detect_project(project_root)

        workspace_info = {
            "id": workspace_id,
            "project_root": project_root,
            "project_info": project_info,
            "created_at": asyncio.get_event_loop().time(),
            "file_count": 0,
            "processed_files": [],
            **workspace_options
        }

        self.workspaces[workspace_id] = workspace_info
        return workspace_info

    async def activate_workspace(self, workspace_id: str) -> bool:
        """Activate a workspace for processing."""
        if workspace_id not in self.workspaces:
            return False

        self.active_workspace = workspace_id
        return True

    async def add_document_to_workspace(
        self,
        workspace_id: str,
        document: ParsedDocument,
        collection_name: str = None
    ) -> bool:
        """Add a processed document to workspace."""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]

        processing_info = {
            "file_path": document.file_path,
            "file_type": document.file_type,
            "content_hash": document.content_hash,
            "processed_at": document.parsed_at,
            "collection": collection_name or f"{workspace_id}-documents",
            "metadata": document.metadata
        }

        workspace["processed_files"].append(processing_info)
        workspace["file_count"] += 1

        self.processed_files[document.file_path] = processing_info
        return True

    async def get_workspace_context(self, workspace_id: str) -> Dict[str, Any]:
        """Get comprehensive workspace context for LLM injection."""
        if workspace_id not in self.workspaces:
            return {}

        workspace = self.workspaces[workspace_id]

        # Extract file statistics
        file_types = {}
        languages = {}
        for file_info in workspace["processed_files"]:
            file_type = file_info["file_type"]
            file_types[file_type] = file_types.get(file_type, 0) + 1

            language = file_info["metadata"].get("estimated_language", "unknown")
            languages[language] = languages.get(language, 0) + 1

        context = {
            "workspace_id": workspace_id,
            "project": workspace["project_info"],
            "statistics": {
                "total_files": workspace["file_count"],
                "file_types": file_types,
                "languages": languages
            },
            "recent_files": workspace["processed_files"][-10:],  # Last 10 files
            "workspace_root": workspace["project_root"]
        }

        self.workspace_contexts[workspace_id] = context
        return context

    async def detect_workspace_changes(
        self,
        workspace_id: str,
        check_git_status: bool = True
    ) -> Dict[str, Any]:
        """Detect changes in workspace since last processing."""
        if workspace_id not in self.workspaces:
            return {"error": "Workspace not found"}

        workspace = self.workspaces[workspace_id]
        project_root = Path(workspace["project_root"])

        changes = {
            "new_files": [],
            "modified_files": [],
            "deleted_files": [],
            "git_changes": {}
        }

        # Simulate file system scanning
        if project_root.exists():
            current_files = list(project_root.rglob("*.py"))  # Example: only Python files
            processed_paths = {info["file_path"] for info in workspace["processed_files"]}

            for file_path in current_files:
                if str(file_path) not in processed_paths:
                    changes["new_files"].append(str(file_path))

        # Git status simulation
        if check_git_status:
            changes["git_changes"] = {
                "modified": [],
                "untracked": changes["new_files"],
                "staged": [],
                "branch": "main"
            }

        return changes

    async def get_multi_workspace_context(self) -> Dict[str, Any]:
        """Get context across all workspaces for multi-tenant scenarios."""
        total_files = sum(ws["file_count"] for ws in self.workspaces.values())

        workspace_summaries = []
        for ws_id, workspace in self.workspaces.items():
            summary = {
                "id": ws_id,
                "project_name": workspace["project_info"].get("name", "unknown"),
                "project_type": workspace["project_info"].get("type", "unknown"),
                "file_count": workspace["file_count"],
                "project_root": workspace["project_root"]
            }
            workspace_summaries.append(summary)

        return {
            "total_workspaces": len(self.workspaces),
            "total_files": total_files,
            "active_workspace": self.active_workspace,
            "workspaces": workspace_summaries
        }

    def get_workspace_list(self) -> List[str]:
        """Get list of all workspace IDs."""
        return list(self.workspaces.keys())

    def get_workspace_info(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific workspace."""
        return self.workspaces.get(workspace_id)

    async def cleanup_workspace(self, workspace_id: str) -> bool:
        """Clean up workspace resources."""
        if workspace_id not in self.workspaces:
            return False

        # Remove processed files associated with workspace
        workspace = self.workspaces[workspace_id]
        for file_info in workspace["processed_files"]:
            self.processed_files.pop(file_info["file_path"], None)

        # Remove workspace
        del self.workspaces[workspace_id]

        # Clear active workspace if it was this one
        if self.active_workspace == workspace_id:
            self.active_workspace = None

        return True


class TestWorkspaceIntegration:
    """Test workspace integration with document processing."""

    @pytest.fixture
    def mock_project_detector(self):
        """Create mock project detector with various project types."""
        detector = MagicMock()

        # Default project info
        detector.detect_project = AsyncMock(return_value={
            "type": "python",
            "root": "/test/project",
            "name": "test-project",
            "submodules": [],
            "git_root": "/test/project/.git",
            "language": "python",
            "framework": "none"
        })

        return detector

    @pytest.fixture
    async def workspace_manager(self, mock_project_detector):
        """Create workspace manager for testing."""
        return MockWorkspaceManager(mock_project_detector)

    @pytest.fixture
    def sample_documents(self):
        """Create sample parsed documents for testing."""
        documents = []

        for i in range(3):
            doc = ParsedDocument.create(
                content=f"Sample content for document {i}",
                file_path=f"/test/project/file_{i}.py",
                file_type="text",
                additional_metadata={
                    "estimated_language": "python",
                    "word_count": 10 + i,
                    "project_context": True
                }
            )
            documents.append(doc)

        return documents

    @pytest.fixture
    def temp_project_structure(self):
        """Create temporary project structure for testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "test_project"
        project_dir.mkdir()

        # Create various project files
        (project_dir / "main.py").write_text("def main():\n    print('Hello')")
        (project_dir / "utils.py").write_text("def helper():\n    return 42")
        (project_dir / "README.md").write_text("# Test Project\n\nA test project.")

        # Create subdirectories
        (project_dir / "tests").mkdir()
        (project_dir / "tests" / "test_main.py").write_text("import unittest")

        (project_dir / "docs").mkdir()
        (project_dir / "docs" / "guide.md").write_text("# User Guide")

        # Create git directory (simulated)
        git_dir = project_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]\n    repositoryformatversion = 0")

        yield {
            "project_root": str(project_dir),
            "temp_dir": temp_dir,
            "files": {
                "python": ["main.py", "utils.py", "tests/test_main.py"],
                "markdown": ["README.md", "docs/guide.md"],
                "total": 5
            }
        }

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_workspace_creation_with_project_detection(
        self, workspace_manager, mock_project_detector, temp_project_structure
    ):
        """Test workspace creation with automatic project detection."""
        project_root = temp_project_structure["project_root"]

        workspace_info = await workspace_manager.create_workspace(
            "test-workspace",
            project_root,
            description="Test workspace for integration testing"
        )

        # Verify workspace was created
        assert workspace_info["id"] == "test-workspace"
        assert workspace_info["project_root"] == project_root
        assert workspace_info["description"] == "Test workspace for integration testing"

        # Verify project detection was called
        mock_project_detector.detect_project.assert_called_once_with(project_root)

        # Verify project info was stored
        assert workspace_info["project_info"]["type"] == "python"
        assert workspace_info["project_info"]["name"] == "test-project"
        assert workspace_info["file_count"] == 0

    @pytest.mark.asyncio
    async def test_workspace_activation(self, workspace_manager, temp_project_structure):
        """Test workspace activation and switching."""
        project_root = temp_project_structure["project_root"]

        # Create multiple workspaces
        await workspace_manager.create_workspace("workspace-1", project_root)
        await workspace_manager.create_workspace("workspace-2", project_root)

        # Test activation
        assert await workspace_manager.activate_workspace("workspace-1") is True
        assert workspace_manager.active_workspace == "workspace-1"

        # Test switching
        assert await workspace_manager.activate_workspace("workspace-2") is True
        assert workspace_manager.active_workspace == "workspace-2"

        # Test invalid workspace
        assert await workspace_manager.activate_workspace("nonexistent") is False

    @pytest.mark.asyncio
    async def test_document_addition_to_workspace(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test adding processed documents to workspace."""
        project_root = temp_project_structure["project_root"]

        # Create workspace
        await workspace_manager.create_workspace("test-workspace", project_root)

        # Add documents
        for i, document in enumerate(sample_documents):
            success = await workspace_manager.add_document_to_workspace(
                "test-workspace",
                document,
                f"collection-{i}"
            )
            assert success is True

        # Verify workspace state
        workspace_info = workspace_manager.get_workspace_info("test-workspace")
        assert workspace_info["file_count"] == 3
        assert len(workspace_info["processed_files"]) == 3

        # Verify processed files tracking
        for doc in sample_documents:
            assert doc.file_path in workspace_manager.processed_files

    @pytest.mark.asyncio
    async def test_workspace_context_extraction(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test extraction of comprehensive workspace context."""
        project_root = temp_project_structure["project_root"]

        # Setup workspace with documents
        await workspace_manager.create_workspace("test-workspace", project_root)
        for document in sample_documents:
            await workspace_manager.add_document_to_workspace("test-workspace", document)

        # Get workspace context
        context = await workspace_manager.get_workspace_context("test-workspace")

        # Verify context structure
        assert context["workspace_id"] == "test-workspace"
        assert context["project"]["type"] == "python"
        assert context["project"]["name"] == "test-project"

        # Verify statistics
        stats = context["statistics"]
        assert stats["total_files"] == 3
        assert stats["file_types"]["text"] == 3
        assert stats["languages"]["python"] == 3

        # Verify recent files
        assert len(context["recent_files"]) == 3
        assert context["workspace_root"] == project_root

    @pytest.mark.asyncio
    async def test_workspace_change_detection(
        self, workspace_manager, temp_project_structure
    ):
        """Test detection of workspace changes."""
        project_root = temp_project_structure["project_root"]

        # Create workspace
        await workspace_manager.create_workspace("test-workspace", project_root)

        # Detect changes (should find new Python files)
        changes = await workspace_manager.detect_workspace_changes("test-workspace")

        # Verify change detection
        assert "new_files" in changes
        assert "modified_files" in changes
        assert "deleted_files" in changes
        assert "git_changes" in changes

        # Should detect Python files in the project
        assert len(changes["new_files"]) >= 2  # main.py, utils.py, test_main.py

        # Verify git changes structure
        git_changes = changes["git_changes"]
        assert "modified" in git_changes
        assert "untracked" in git_changes
        assert "branch" in git_changes

    @pytest.mark.asyncio
    async def test_multi_workspace_context(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test multi-workspace context for multi-tenant scenarios."""
        project_root = temp_project_structure["project_root"]

        # Create multiple workspaces
        await workspace_manager.create_workspace("workspace-1", project_root)
        await workspace_manager.create_workspace("workspace-2", project_root)

        # Add documents to different workspaces
        await workspace_manager.add_document_to_workspace("workspace-1", sample_documents[0])
        await workspace_manager.add_document_to_workspace("workspace-1", sample_documents[1])
        await workspace_manager.add_document_to_workspace("workspace-2", sample_documents[2])

        # Get multi-workspace context
        context = await workspace_manager.get_multi_workspace_context()

        # Verify multi-workspace context
        assert context["total_workspaces"] == 2
        assert context["total_files"] == 3
        assert len(context["workspaces"]) == 2

        # Verify workspace summaries
        workspace_summaries = context["workspaces"]
        ws1_summary = next(ws for ws in workspace_summaries if ws["id"] == "workspace-1")
        ws2_summary = next(ws for ws in workspace_summaries if ws["id"] == "workspace-2")

        assert ws1_summary["file_count"] == 2
        assert ws2_summary["file_count"] == 1
        assert ws1_summary["project_name"] == "test-project"

    @pytest.mark.asyncio
    async def test_project_type_specific_detection(self, workspace_manager):
        """Test project type specific detection and context."""
        # Mock different project types
        project_types = [
            {
                "type": "react",
                "name": "frontend-app",
                "framework": "nextjs",
                "language": "typescript"
            },
            {
                "type": "django",
                "name": "backend-api",
                "framework": "django",
                "language": "python"
            },
            {
                "type": "rust",
                "name": "cli-tool",
                "framework": "clap",
                "language": "rust"
            }
        ]

        for i, project_info in enumerate(project_types):
            workspace_manager.project_detector.detect_project = AsyncMock(return_value=project_info)

            workspace_id = f"workspace-{project_info['type']}"
            workspace_info = await workspace_manager.create_workspace(
                workspace_id,
                f"/projects/{project_info['name']}"
            )

            # Verify project-specific information
            assert workspace_info["project_info"]["type"] == project_info["type"]
            assert workspace_info["project_info"]["framework"] == project_info["framework"]
            assert workspace_info["project_info"]["language"] == project_info["language"]

    @pytest.mark.asyncio
    async def test_workspace_cleanup(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test workspace cleanup and resource management."""
        project_root = temp_project_structure["project_root"]

        # Create workspace with documents
        await workspace_manager.create_workspace("test-workspace", project_root)
        await workspace_manager.activate_workspace("test-workspace")

        for document in sample_documents:
            await workspace_manager.add_document_to_workspace("test-workspace", document)

        # Verify initial state
        assert len(workspace_manager.workspaces) == 1
        assert len(workspace_manager.processed_files) == 3
        assert workspace_manager.active_workspace == "test-workspace"

        # Cleanup workspace
        success = await workspace_manager.cleanup_workspace("test-workspace")
        assert success is True

        # Verify cleanup
        assert len(workspace_manager.workspaces) == 0
        assert len(workspace_manager.processed_files) == 0
        assert workspace_manager.active_workspace is None

    @pytest.mark.asyncio
    async def test_workspace_error_handling(self, workspace_manager):
        """Test error handling in workspace operations."""
        # Test operations on non-existent workspace
        success = await workspace_manager.add_document_to_workspace(
            "nonexistent",
            sample_documents[0] if 'sample_documents' in locals() else None
        )
        assert success is False

        context = await workspace_manager.get_workspace_context("nonexistent")
        assert context == {}

        changes = await workspace_manager.detect_workspace_changes("nonexistent")
        assert "error" in changes

        cleanup_success = await workspace_manager.cleanup_workspace("nonexistent")
        assert cleanup_success is False

    @pytest.mark.asyncio
    async def test_workspace_project_detection_failure(self, workspace_manager):
        """Test handling of project detection failures."""
        # Mock project detection failure
        workspace_manager.project_detector.detect_project = AsyncMock(
            side_effect=Exception("Project detection failed")
        )

        # Should handle gracefully
        with pytest.raises(Exception, match="Project detection failed"):
            await workspace_manager.create_workspace("test-workspace", "/invalid/path")

    @pytest.mark.asyncio
    async def test_workspace_context_for_llm_injection(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test workspace context specifically for LLM rule injection."""
        project_root = temp_project_structure["project_root"]

        # Setup complex workspace scenario
        await workspace_manager.create_workspace("llm-workspace", project_root)

        # Add documents with various metadata
        documents = []
        for i, base_doc in enumerate(sample_documents):
            # Enrich documents with LLM-relevant metadata
            base_doc.metadata.update({
                "importance": i + 1,
                "category": ["core", "utility", "test"][i],
                "last_modified": f"2024-01-{i+1:02d}",
                "author": f"developer{i+1}",
                "complexity": ["low", "medium", "high"][i]
            })
            documents.append(base_doc)

        for doc in documents:
            await workspace_manager.add_document_to_workspace("llm-workspace", doc)

        # Get context for LLM injection
        context = await workspace_manager.get_workspace_context("llm-workspace")

        # Verify LLM-relevant context structure
        assert "workspace_id" in context
        assert "project" in context
        assert "statistics" in context
        assert "recent_files" in context

        # Verify detailed file metadata is available
        recent_files = context["recent_files"]
        assert len(recent_files) == 3

        for file_info in recent_files:
            assert "metadata" in file_info
            assert "importance" in file_info["metadata"]
            assert "category" in file_info["metadata"]
            assert "complexity" in file_info["metadata"]

    @pytest.mark.asyncio
    async def test_workspace_scalability(self, workspace_manager, temp_project_structure):
        """Test workspace manager scalability with many workspaces."""
        project_root = temp_project_structure["project_root"]

        # Create many workspaces
        workspace_count = 20
        for i in range(workspace_count):
            await workspace_manager.create_workspace(f"workspace-{i}", project_root)

        # Verify all workspaces were created
        workspace_list = workspace_manager.get_workspace_list()
        assert len(workspace_list) == workspace_count

        # Test multi-workspace context with many workspaces
        context = await workspace_manager.get_multi_workspace_context()
        assert context["total_workspaces"] == workspace_count
        assert len(context["workspaces"]) == workspace_count

    @pytest.mark.asyncio
    async def test_workspace_concurrent_operations(
        self, workspace_manager, sample_documents, temp_project_structure
    ):
        """Test concurrent workspace operations."""
        project_root = temp_project_structure["project_root"]

        # Create workspace
        await workspace_manager.create_workspace("concurrent-workspace", project_root)

        # Concurrent document additions
        tasks = [
            workspace_manager.add_document_to_workspace("concurrent-workspace", doc)
            for doc in sample_documents
        ]

        results = await asyncio.gather(*tasks)
        assert all(results)

        # Verify all documents were added
        workspace_info = workspace_manager.get_workspace_info("concurrent-workspace")
        assert workspace_info["file_count"] == len(sample_documents)

    @pytest.mark.asyncio
    async def test_workspace_metadata_preservation(
        self, workspace_manager, temp_project_structure
    ):
        """Test preservation of document metadata through workspace operations."""
        project_root = temp_project_structure["project_root"]

        # Create document with rich metadata
        document = ParsedDocument.create(
            content="Complex document with rich metadata",
            file_path="/test/complex.py",
            file_type="python",
            additional_metadata={
                "author": "test_author",
                "created_date": "2024-01-01",
                "tags": ["important", "core", "api"],
                "dependencies": ["numpy", "pandas"],
                "test_coverage": 85.5,
                "complexity_score": 7.2
            }
        )

        # Add to workspace
        await workspace_manager.create_workspace("metadata-workspace", project_root)
        await workspace_manager.add_document_to_workspace("metadata-workspace", document)

        # Retrieve and verify metadata preservation
        context = await workspace_manager.get_workspace_context("metadata-workspace")
        recent_files = context["recent_files"]

        assert len(recent_files) == 1
        file_metadata = recent_files[0]["metadata"]

        # Verify all metadata was preserved
        assert file_metadata["author"] == "test_author"
        assert file_metadata["tags"] == ["important", "core", "api"]
        assert file_metadata["dependencies"] == ["numpy", "pandas"]
        assert file_metadata["test_coverage"] == 85.5
        assert file_metadata["complexity_score"] == 7.2

    def test_workspace_manager_initialization(self, mock_project_detector):
        """Test workspace manager initialization."""
        manager = MockWorkspaceManager(mock_project_detector)

        assert manager.project_detector == mock_project_detector
        assert manager.workspaces == {}
        assert manager.active_workspace is None
        assert manager.processed_files == {}
        assert manager.workspace_contexts == {}

    def test_workspace_list_operations(self, workspace_manager, temp_project_structure):
        """Test workspace list and info operations."""
        project_root = temp_project_structure["project_root"]

        # Initially empty
        assert workspace_manager.get_workspace_list() == []
        assert workspace_manager.get_workspace_info("nonexistent") is None

        # After creating workspaces
        asyncio.run(workspace_manager.create_workspace("ws1", project_root))
        asyncio.run(workspace_manager.create_workspace("ws2", project_root))

        workspace_list = workspace_manager.get_workspace_list()
        assert len(workspace_list) == 2
        assert "ws1" in workspace_list
        assert "ws2" in workspace_list

        # Get specific workspace info
        ws1_info = workspace_manager.get_workspace_info("ws1")
        assert ws1_info is not None
        assert ws1_info["id"] == "ws1"
        assert ws1_info["project_root"] == project_root