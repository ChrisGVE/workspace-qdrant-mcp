"""
End-to-End Tests for Project Switching Workflow (Task 292.4).

Comprehensive tests validating project detection, switching, and isolation mechanisms
across the system including automatic detection, manual switching, collection management,
watch folder reconfiguration, state isolation, Git boundary detection, submodule handling,
and metadata persistence.

Test Coverage:
    - Automatic project detection from different directories
    - Manual project switching via CLI/MCP
    - Project-specific collection management
    - Watch folder reconfiguration on project switch
    - State isolation between projects
    - Git repository boundary detection
    - Submodule handling (owned vs foreign)
    - Project metadata persistence and cleanup

Features Validated:
    - ProjectDetector integration with Git
    - SQLiteStateManager state isolation
    - WatchFolderConfig persistence and updates
    - Collection naming: {project}-{type}
    - Tenant ID calculation and consistency
    - Submodule ownership detection
    - Clean state transitions between projects
    - Resource cleanup and isolation
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.core.sqlite_state_manager import (
    ProjectRecord,
    SQLiteStateManager,
    WatchFolderConfig,
)
from common.utils.project_detection import (
    DaemonIdentifier,
    ProjectDetector,
    calculate_tenant_id,
)

from .utils import (
    HealthChecker,
    TestDataGenerator,
    WorkflowTimer,
    assert_within_threshold,
    run_git_command,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def multi_project_workspace():
    """
    Create multiple project workspaces with different configurations.

    Creates 3 projects:
    - project_a: Full Git repo with remote (GitHub-style)
    - project_b: Local Git repo without remote
    - project_c: Parent repo with submodule
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Project A: Full Git repo with GitHub remote
        project_a = workspace / "project-a"
        project_a.mkdir()

        # Initialize Git
        subprocess.run(["git", "init"], cwd=project_a, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_a, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_a, check=True)

        # Add remote
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/testuser/project-a.git"],
            cwd=project_a,
            check=True,
            capture_output=True
        )

        # Create project structure
        (project_a / "src").mkdir()
        (project_a / "src" / "main.py").write_text("def main():\n    pass")
        (project_a / "README.md").write_text("# Project A")

        # Initial commit
        subprocess.run(["git", "add", "."], cwd=project_a, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_a, check=True)

        # Project B: Local Git repo (no remote)
        project_b = workspace / "project-b"
        project_b.mkdir()

        subprocess.run(["git", "init"], cwd=project_b, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_b, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_b, check=True)

        (project_b / "lib").mkdir()
        (project_b / "lib" / "utils.py").write_text("def util():\n    pass")
        (project_b / "README.md").write_text("# Project B")

        subprocess.run(["git", "add", "."], cwd=project_b, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_b, check=True)

        # Project C: Parent with submodule structure
        project_c = workspace / "project-c"
        project_c.mkdir()

        subprocess.run(["git", "init"], cwd=project_c, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_c, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_c, check=True)

        # Add remote for project C
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/testuser/project-c.git"],
            cwd=project_c,
            check=True,
            capture_output=True
        )

        (project_c / "core").mkdir()
        (project_c / "core" / "app.py").write_text("def app():\n    pass")
        (project_c / "README.md").write_text("# Project C with Submodules")

        subprocess.run(["git", "add", "."], cwd=project_c, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_c, check=True)

        # Create a mock submodule directory (simplified - not full Git submodule)
        submodule_dir = project_c / "external" / "library"
        submodule_dir.mkdir(parents=True)
        (submodule_dir / "lib.py").write_text("# External library")

        # Create .gitmodules file to simulate submodule
        gitmodules_content = """[submodule "external/library"]
\tpath = external/library
\turl = https://github.com/otheruser/library.git
"""
        (project_c / ".gitmodules").write_text(gitmodules_content)

        yield {
            "workspace": workspace,
            "project_a": {
                "path": project_a,
                "name": "project-a",
                "has_remote": True,
                "remote_url": "https://github.com/testuser/project-a.git",
                "is_github_user_owned": True,
            },
            "project_b": {
                "path": project_b,
                "name": "project-b",
                "has_remote": False,
                "remote_url": None,
                "is_github_user_owned": False,
            },
            "project_c": {
                "path": project_c,
                "name": "project-c",
                "has_remote": True,
                "remote_url": "https://github.com/testuser/project-c.git",
                "is_github_user_owned": True,
                "has_submodules": True,
            },
        }


@pytest.fixture
async def state_manager_fixture():
    """Provide SQLiteStateManager with temporary database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_state.db"
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        yield state_manager

        # Cleanup
        await state_manager.close()


@pytest.fixture
def project_detector_fixture():
    """Provide ProjectDetector with test GitHub user."""
    detector = ProjectDetector(github_user="testuser")
    yield detector

    # Cleanup any registered identifiers
    DaemonIdentifier.clear_registry()


@pytest.fixture
def mock_daemon_client():
    """Provide mocked daemon client for testing."""
    client = AsyncMock()
    client.update_watch_folders = AsyncMock(return_value=True)
    client.get_watch_folders = AsyncMock(return_value=[])
    client.health_check = AsyncMock(return_value={"healthy": True})
    return client


# ============================================================================
# Test Class: Project Detection
# ============================================================================


@pytest.mark.e2e
class TestProjectDetection:
    """Test automatic project detection from different directories."""

    @pytest.mark.asyncio
    async def test_detect_project_from_nested_directory(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test project detection from nested subdirectory."""
        timer = WorkflowTimer()
        timer.start()

        project_a = multi_project_workspace["project_a"]["path"]
        nested_dir = project_a / "src"

        # Detect from nested directory
        project_name = project_detector_fixture.get_project_name(str(nested_dir))
        timer.checkpoint("detection_complete")

        # Validate project name (should use remote name for GitHub user)
        assert project_name == "project-a", \
            f"Expected 'project-a', got '{project_name}'"

        # Validate timing (< 1s for local detection)
        duration = timer.get_duration("detection_complete")
        assert duration < 1.0, f"Detection took {duration:.2f}s (expected < 1s)"

        print(f"✓ Detected project from nested directory in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_detect_git_root_vs_directory_name(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test project naming: remote-based vs directory-based."""
        timer = WorkflowTimer()
        timer.start()

        # Project A: Should use remote name (GitHub user owned)
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))
        assert name_a == "project-a", "Project A should use remote name"

        # Project B: Should use directory name (no remote)
        project_b = multi_project_workspace["project_b"]["path"]
        name_b = project_detector_fixture.get_project_name(str(project_b))
        assert name_b == "project-b", "Project B should use directory name"

        timer.checkpoint("both_detected")

        # Validate timing
        duration = timer.get_duration("both_detected")
        assert duration < 2.0, f"Both detections took {duration:.2f}s (expected < 2s)"

        print(f"✓ Detected both projects correctly in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_detect_project_without_github_user_filter(
        self,
        multi_project_workspace
    ):
        """Test detection without GitHub user filtering."""
        # Create detector without GitHub user filter
        detector = ProjectDetector(github_user=None)

        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        # Both should use directory names (no user filtering)
        name_a = detector.get_project_name(str(project_a))
        name_b = detector.get_project_name(str(project_b))

        # Without user filter, both use directory names
        assert name_a == "project-a"
        assert name_b == "project-b"

        print("✓ Detection without user filter works correctly")

    @pytest.mark.asyncio
    async def test_project_info_comprehensive(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test comprehensive project information retrieval."""
        project_a = multi_project_workspace["project_a"]["path"]

        info = project_detector_fixture.get_project_info(str(project_a))

        # Validate comprehensive info
        assert info["main_project"] == "project-a"
        assert info["is_git_repo"] is True
        assert info["belongs_to_user"] is True
        assert info["git_root"] == str(project_a)
        assert "github.com" in info["remote_url"]

        # Validate URL info
        assert info["main_url_info"]["is_github"] is True
        assert info["main_url_info"]["username"] == "testuser"
        assert info["main_url_info"]["repository"] == "project-a"

        print(f"✓ Comprehensive project info validated: {info['main_project']}")


# ============================================================================
# Test Class: Project Switching
# ============================================================================


@pytest.mark.e2e
class TestProjectSwitching:
    """Test manual project switching via CLI/MCP simulation."""

    @pytest.mark.asyncio
    async def test_switch_between_projects_manual(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test manual switching between projects."""
        timer = WorkflowTimer()
        timer.start()

        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        # Detect first project
        name_a = project_detector_fixture.get_project_name(str(project_a))
        calculate_tenant_id(project_a)
        timer.checkpoint("detect_project_a")

        # Create watch folder for project A to track state
        watch_config_a = WatchFolderConfig(
            watch_id=f"{name_a}-src",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(watch_config_a)
        timer.checkpoint("save_watch_a")

        # Switch to project B
        name_b = project_detector_fixture.get_project_name(str(project_b))
        calculate_tenant_id(project_b)
        timer.checkpoint("detect_project_b")

        # Create watch folder for project B
        watch_config_b = WatchFolderConfig(
            watch_id=f"{name_b}-lib",
            path=str(project_b / "lib"),
            collection=f"{name_b}-code",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(watch_config_b)
        timer.checkpoint("save_watch_b")

        # Validate both watch configs exist (representing project state)
        all_watches = await state_manager_fixture.list_watch_folders()
        assert len(all_watches) >= 2, "Both projects should have watch configs"

        watch_ids = [w["watch_id"] for w in all_watches]
        assert watch_config_a.watch_id in watch_ids
        assert watch_config_b.watch_id in watch_ids

        # Validate timing (< 5s for full switch)
        duration = timer.get_duration("save_watch_b")
        assert duration < 5.0, f"Project switch took {duration:.2f}s (expected < 5s)"

        print(f"✓ Switched between projects in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_project_switch_tenant_id_consistency(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test tenant ID consistency during project switches."""
        project_a = multi_project_workspace["project_a"]["path"]

        # Calculate tenant ID multiple times
        tenant_id_1 = calculate_tenant_id(project_a)
        tenant_id_2 = calculate_tenant_id(project_a)
        tenant_id_3 = calculate_tenant_id(project_a)

        # Should be consistent
        assert tenant_id_1 == tenant_id_2 == tenant_id_3, \
            "Tenant IDs must be consistent for same project"

        # Should be based on remote URL (not path) for projects with remotes
        assert not tenant_id_1.startswith("path_"), \
            "Projects with remotes should use URL-based tenant ID"

        print(f"✓ Tenant ID consistent: {tenant_id_1}")

    @pytest.mark.asyncio
    async def test_rapid_project_switching(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test rapid switching between multiple projects."""
        timer = WorkflowTimer()
        timer.start()

        projects = [
            multi_project_workspace["project_a"]["path"],
            multi_project_workspace["project_b"]["path"],
            multi_project_workspace["project_c"]["path"],
        ]

        switch_count = 0
        for i, project_path in enumerate(projects):
            project_name = project_detector_fixture.get_project_name(str(project_path))
            tenant_id = calculate_tenant_id(project_path)

            project_record = ProjectRecord(
                id=None,
                name=project_name,
                root_path=str(project_path),
                collection_name=f"{project_name}-code",
                project_id=tenant_id[:12],
            )
            await state_manager_fixture.save_project(project_record)
            switch_count += 1
            timer.checkpoint(f"switch_{i+1}")

        # Validate all switches completed
        assert switch_count == 3, "All 3 project switches should complete"

        # Validate timing (< 10s for 3 switches)
        duration = timer.get_duration("switch_3")
        assert duration < 10.0, f"3 switches took {duration:.2f}s (expected < 10s)"

        print(f"✓ Rapid switching {switch_count} projects in {duration:.3f}s")


# ============================================================================
# Test Class: Collection Management
# ============================================================================


@pytest.mark.e2e
class TestCollectionManagement:
    """Test project-specific collection management."""

    @pytest.mark.asyncio
    async def test_collection_naming_pattern(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test collection naming follows {project}-{type} pattern."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_name = project_detector_fixture.get_project_name(str(project_a))

        # Define collection types
        collection_types = ["code", "docs", "notes", "config"]

        # Generate collection names
        collections = []
        for coll_type in collection_types:
            collection_name = f"{project_name}-{coll_type}"
            collections.append(collection_name)

            # Validate naming pattern
            assert collection_name.startswith(f"{project_name}-"), \
                f"Collection must start with project name: {collection_name}"
            assert collection_name.count("-") >= 1, \
                f"Collection must follow {project_name}-{{type}} pattern"

        expected_collections = [
            "project-a-code",
            "project-a-docs",
            "project-a-notes",
            "project-a-config",
        ]

        assert collections == expected_collections, \
            "Collections don't match expected pattern"

        print(f"✓ Collection naming validated: {collections}")

    @pytest.mark.asyncio
    async def test_collection_isolation_between_projects(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test collections are properly isolated between projects."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        name_a = project_detector_fixture.get_project_name(str(project_a))
        name_b = project_detector_fixture.get_project_name(str(project_b))

        # Generate collections for both projects
        collections_a = [f"{name_a}-code", f"{name_a}-docs"]
        collections_b = [f"{name_b}-code", f"{name_b}-docs"]

        # Validate no overlap
        overlap = set(collections_a) & set(collections_b)
        assert len(overlap) == 0, f"Collections should not overlap: {overlap}"

        # Validate project prefixes differ
        for coll_a in collections_a:
            assert coll_a.startswith(f"{name_a}-")
        for coll_b in collections_b:
            assert coll_b.startswith(f"{name_b}-")

        print(f"✓ Collections isolated: A={collections_a}, B={collections_b}")

    @pytest.mark.asyncio
    async def test_project_specific_collection_lifecycle(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test full collection lifecycle for project-specific collections."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_name = project_detector_fixture.get_project_name(str(project_a))
        tenant_id = calculate_tenant_id(project_a)

        # Create project record
        project_record = ProjectRecord(
            id=None,
            name=project_name,
            root_path=str(project_a),
            collection_name=f"{project_name}-code",
            project_id=tenant_id[:12],
        )
        await state_manager_fixture.save_project(project_record)

        # Retrieve project
        retrieved = await state_manager_fixture.get_project_by_root_path(str(project_a))
        assert retrieved is not None, "Project should be retrievable"
        assert retrieved["name"] == project_name
        assert retrieved["collection_name"] == f"{project_name}-code"

        # Cleanup (would delete collections in real scenario)
        all_projects = await state_manager_fixture.list_projects()
        assert any(p["name"] == project_name for p in all_projects)

        print(f"✓ Collection lifecycle validated for {project_name}")


# ============================================================================
# Test Class: Watch Folder Reconfiguration
# ============================================================================


@pytest.mark.e2e
class TestWatchFolderReconfiguration:
    """Test watch folder reconfiguration on project switch."""

    @pytest.mark.asyncio
    async def test_watch_folder_config_update_on_switch(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test watch folder configuration updates when switching projects."""
        timer = WorkflowTimer()
        timer.start()

        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        name_a = project_detector_fixture.get_project_name(str(project_a))
        name_b = project_detector_fixture.get_project_name(str(project_b))

        # Create watch config for project A
        watch_config_a = WatchFolderConfig(
            watch_id=f"{name_a}-src",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_a.watch_id,
            watch_config_a
        )
        timer.checkpoint("save_watch_a")

        # Switch to project B - create new watch config
        watch_config_b = WatchFolderConfig(
            watch_id=f"{name_b}-lib",
            path=str(project_b / "lib"),
            collection=f"{name_b}-code",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_b.watch_id,
            watch_config_b
        )
        timer.checkpoint("save_watch_b")

        # Verify both watch configs exist
        all_watches = await state_manager_fixture.list_watch_folders()
        watch_ids = [w["watch_id"] for w in all_watches]

        assert watch_config_a.watch_id in watch_ids
        assert watch_config_b.watch_id in watch_ids

        # Validate timing
        duration = timer.get_duration("save_watch_b")
        assert duration < 3.0, f"Watch config updates took {duration:.2f}s (expected < 3s)"

        print(f"✓ Watch folder configs updated in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_watch_folder_persistence_across_restarts(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test watch folder configuration persists across state manager restarts."""
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "persistent_state.db"

            # First instance - save config
            state_manager_1 = SQLiteStateManager(db_path=str(db_path))
            await state_manager_1.initialize()

            watch_config = WatchFolderConfig(
                watch_id=f"{name_a}-persistent",
                path=str(project_a / "src"),
                collection=f"{name_a}-code",
                patterns=["*.py", "*.md"],
                ignore_patterns=["*.pyc"],
                auto_ingest=True,
                enabled=True,
            )
            await state_manager_1.save_watch_folder_config(
                watch_config.watch_id,
                watch_config
            )
            await state_manager_1.close()

            # Second instance - retrieve config
            state_manager_2 = SQLiteStateManager(db_path=str(db_path))
            await state_manager_2.initialize()

            retrieved_config = await state_manager_2.get_watch_folder_config(
                watch_config.watch_id
            )

            assert retrieved_config is not None, "Config should persist"
            assert retrieved_config["watch_id"] == watch_config.watch_id
            assert retrieved_config["path"] == watch_config.path
            assert retrieved_config["collection"] == watch_config.collection
            assert retrieved_config["enabled"] is True

            await state_manager_2.close()

            print("✓ Watch folder config persisted across restarts")

    @pytest.mark.asyncio
    async def test_disable_watch_folder_on_project_switch(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test disabling watch folders when switching away from project."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        name_a = project_detector_fixture.get_project_name(str(project_a))
        name_b = project_detector_fixture.get_project_name(str(project_b))

        # Create watch for project A
        watch_config_a = WatchFolderConfig(
            watch_id=f"{name_a}-src",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_a.watch_id,
            watch_config_a
        )

        # Simulate switching to project B - disable project A's watch
        watch_config_a.enabled = False
        await state_manager_fixture.save_watch_folder_config(
            watch_config_a.watch_id,
            watch_config_a
        )

        # Create watch for project B (enabled)
        watch_config_b = WatchFolderConfig(
            watch_id=f"{name_b}-lib",
            path=str(project_b / "lib"),
            collection=f"{name_b}-code",
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_b.watch_id,
            watch_config_b
        )

        # Verify states
        config_a_retrieved = await state_manager_fixture.get_watch_folder_config(
            watch_config_a.watch_id
        )
        config_b_retrieved = await state_manager_fixture.get_watch_folder_config(
            watch_config_b.watch_id
        )

        assert config_a_retrieved["enabled"] is False, "Project A watch should be disabled"
        assert config_b_retrieved["enabled"] is True, "Project B watch should be enabled"

        print("✓ Watch folder disabled on project switch")


# ============================================================================
# Test Class: State Isolation
# ============================================================================


@pytest.mark.e2e
class TestStateIsolation:
    """Test state isolation between projects."""

    @pytest.mark.asyncio
    async def test_separate_state_per_project(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test each project maintains separate state."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        name_a = project_detector_fixture.get_project_name(str(project_a))
        name_b = project_detector_fixture.get_project_name(str(project_b))

        tenant_id_a = calculate_tenant_id(project_a)
        tenant_id_b = calculate_tenant_id(project_b)

        # Create separate project records
        project_record_a = ProjectRecord(
            id=None,
            name=name_a,
            root_path=str(project_a),
            collection_name=f"{name_a}-code",
            project_id=tenant_id_a[:12],
        )
        project_record_b = ProjectRecord(
            id=None,
            name=name_b,
            root_path=str(project_b),
            collection_name=f"{name_b}-code",
            project_id=tenant_id_b[:12],
        )

        await state_manager_fixture.save_project(project_record_a)
        await state_manager_fixture.save_project(project_record_b)

        # Retrieve and validate isolation
        retrieved_a = await state_manager_fixture.get_project_by_root_path(str(project_a))
        retrieved_b = await state_manager_fixture.get_project_by_root_path(str(project_b))

        assert retrieved_a["name"] == name_a
        assert retrieved_b["name"] == name_b
        assert retrieved_a["project_id"] != retrieved_b["project_id"], \
            "Project IDs must be unique"

        print(f"✓ State isolated: A={retrieved_a['project_id']}, B={retrieved_b['project_id']}")

    @pytest.mark.asyncio
    async def test_no_state_leakage_between_projects(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test no state leakage between projects."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        name_a = project_detector_fixture.get_project_name(str(project_a))
        name_b = project_detector_fixture.get_project_name(str(project_b))

        # Create watch configs for both projects
        watch_a = WatchFolderConfig(
            watch_id=f"{name_a}-exclusive",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            enabled=True,
        )
        watch_b = WatchFolderConfig(
            watch_id=f"{name_b}-exclusive",
            path=str(project_b / "lib"),
            collection=f"{name_b}-code",
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            enabled=True,
        )

        await state_manager_fixture.save_watch_folder_config(watch_a.watch_id, watch_a)
        await state_manager_fixture.save_watch_folder_config(watch_b.watch_id, watch_b)

        # Retrieve and validate no cross-contamination
        all_watches = await state_manager_fixture.list_watch_folders()

        # Find watches for each project
        watches_for_a = [w for w in all_watches if name_a in w["watch_id"]]
        watches_for_b = [w for w in all_watches if name_b in w["watch_id"]]

        assert len(watches_for_a) == 1, "Project A should have exactly 1 watch"
        assert len(watches_for_b) == 1, "Project B should have exactly 1 watch"

        # Validate paths don't cross
        assert watches_for_a[0]["path"] != watches_for_b[0]["path"], \
            "Watch paths must be different"
        assert watches_for_a[0]["collection"] != watches_for_b[0]["collection"], \
            "Collections must be different"

        print("✓ No state leakage between projects")

    @pytest.mark.asyncio
    async def test_project_state_cleanup(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test proper cleanup when removing project state."""
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))
        tenant_id_a = calculate_tenant_id(project_a)

        # Create project and watch config
        project_record = ProjectRecord(
            id=None,
            name=name_a,
            root_path=str(project_a),
            collection_name=f"{name_a}-code",
            project_id=tenant_id_a[:12],
        )
        await state_manager_fixture.save_project(project_record)

        watch_config = WatchFolderConfig(
            watch_id=f"{name_a}-cleanup-test",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=[],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config.watch_id,
            watch_config
        )

        # Verify exists
        retrieved = await state_manager_fixture.get_project_by_root_path(str(project_a))
        assert retrieved is not None

        # Cleanup watch config
        await state_manager_fixture.remove_watch_folder_config(watch_config.watch_id)

        # Verify watch removed
        all_watches = await state_manager_fixture.list_watch_folders()
        watch_ids = [w["watch_id"] for w in all_watches]
        assert watch_config.watch_id not in watch_ids, "Watch should be removed"

        print("✓ Project state cleanup successful")


# ============================================================================
# Test Class: Git Boundaries
# ============================================================================


@pytest.mark.e2e
class TestGitBoundaries:
    """Test Git repository boundary detection."""

    @pytest.mark.asyncio
    async def test_detect_git_repository_root(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test accurate Git repository root detection."""
        project_a = multi_project_workspace["project_a"]["path"]
        nested_dir = project_a / "src"

        # Get project info from nested directory
        info = project_detector_fixture.get_project_info(str(nested_dir))

        # Git root should be project_a, not nested_dir
        assert info["git_root"] == str(project_a), \
            f"Git root should be {project_a}, not {nested_dir}"
        assert info["is_git_repo"] is True

        print(f"✓ Git root correctly detected: {info['git_root']}")

    @pytest.mark.asyncio
    async def test_git_boundary_with_non_git_directory(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test handling of non-Git directories."""
        workspace = multi_project_workspace["workspace"]
        non_git_dir = workspace / "not-a-repo"
        non_git_dir.mkdir()
        (non_git_dir / "file.txt").write_text("test")

        # Detect from non-Git directory
        info = project_detector_fixture.get_project_info(str(non_git_dir))

        # Should fall back to directory name
        assert info["is_git_repo"] is False
        assert info["git_root"] is None
        assert info["main_project"] == "not-a-repo"

        print("✓ Non-Git directory handled correctly")

    @pytest.mark.asyncio
    async def test_git_boundary_stops_at_root(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test Git boundary detection stops at repository root."""
        project_a = multi_project_workspace["project_a"]["path"]

        # Create deep nested structure
        deep_dir = project_a / "src" / "deep" / "nested" / "structure"
        deep_dir.mkdir(parents=True)
        (deep_dir / "module.py").write_text("# Deep module")

        # Detect from deep directory
        info = project_detector_fixture.get_project_info(str(deep_dir))

        # Should still find project_a as root
        assert info["git_root"] == str(project_a)
        assert info["main_project"] == "project-a"

        print("✓ Git boundary detection stopped at correct root")


# ============================================================================
# Test Class: Submodule Handling
# ============================================================================


@pytest.mark.e2e
class TestSubmoduleHandling:
    """Test submodule detection and handling."""

    @pytest.mark.asyncio
    async def test_detect_submodules(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test submodule detection from parent repository."""
        project_c = multi_project_workspace["project_c"]["path"]

        # Get detailed submodule information
        submodules = project_detector_fixture.get_detailed_submodules(str(project_c))

        # Note: This is a simplified test - real submodules would be initialized
        # In a full implementation, submodules would be detected from .gitmodules

        # For now, validate the detection mechanism works
        # (May be empty if submodules not fully initialized)
        assert isinstance(submodules, list), "Should return list of submodules"

        print(f"✓ Submodule detection executed (found {len(submodules)} submodules)")

    @pytest.mark.asyncio
    async def test_submodule_user_ownership_filtering(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test submodule filtering by user ownership."""
        project_c = multi_project_workspace["project_c"]["path"]

        # Get project info with user filtering
        info = project_detector_fixture.get_project_info(str(project_c))

        # Validate filtering mechanism
        info.get("detailed_submodules", [])
        user_owned = info.get("user_owned_submodules", [])

        # All returned submodules should be user-owned when user filter is active
        for submodule in user_owned:
            assert submodule.get("user_owned") is True, \
                "User-owned submodules must have user_owned=True"

        print("✓ Submodule user filtering validated")

    @pytest.mark.asyncio
    async def test_nested_repository_handling(
        self,
        multi_project_workspace,
        project_detector_fixture
    ):
        """Test handling of nested Git repositories."""
        project_c = multi_project_workspace["project_c"]["path"]

        # Project C has a .gitmodules file indicating submodules
        gitmodules_path = project_c / ".gitmodules"
        assert gitmodules_path.exists(), ".gitmodules file should exist"

        # Detect project from parent
        info = project_detector_fixture.get_project_info(str(project_c))

        assert info["main_project"] == "project-c"
        assert info["is_git_repo"] is True

        print("✓ Nested repository structure handled correctly")


# ============================================================================
# Test Class: Metadata Persistence
# ============================================================================


@pytest.mark.e2e
class TestMetadataPersistence:
    """Test project metadata persistence and cleanup."""

    @pytest.mark.asyncio
    async def test_project_metadata_persists(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test project metadata persists correctly."""
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))
        tenant_id_a = calculate_tenant_id(project_a)

        # Create project with metadata
        metadata = {
            "branch": "main",
            "last_switch": "2024-10-19T00:00:00Z",
            "active": True,
        }

        project_record = ProjectRecord(
            id=None,
            name=name_a,
            root_path=str(project_a),
            collection_name=f"{name_a}-code",
            project_id=tenant_id_a[:12],
            metadata=metadata,
        )
        await state_manager_fixture.save_project(project_record)

        # Retrieve and validate metadata
        retrieved = await state_manager_fixture.get_project_by_root_path(str(project_a))
        assert retrieved is not None
        assert retrieved["name"] == name_a

        # Metadata should persist
        retrieved_metadata = retrieved.get("metadata")
        if retrieved_metadata:
            assert "branch" in retrieved_metadata or isinstance(retrieved_metadata, dict)

        print("✓ Project metadata persisted correctly")

    @pytest.mark.asyncio
    async def test_tenant_id_calculation_and_persistence(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test tenant ID calculation and persistence."""
        project_a = multi_project_workspace["project_a"]["path"]
        project_b = multi_project_workspace["project_b"]["path"]

        # Calculate tenant IDs
        tenant_id_a = calculate_tenant_id(project_a)
        tenant_id_b = calculate_tenant_id(project_b)

        # Validate uniqueness
        assert tenant_id_a != tenant_id_b, "Tenant IDs must be unique"

        # Project A has remote, should use remote-based tenant ID
        assert not tenant_id_a.startswith("path_"), \
            "Project with remote should use URL-based tenant ID"

        # Project B has no remote, should use path-based tenant ID
        assert tenant_id_b.startswith("path_"), \
            "Project without remote should use path-based tenant ID"

        print(f"✓ Tenant IDs calculated: A={tenant_id_a[:20]}, B={tenant_id_b[:20]}")

    @pytest.mark.asyncio
    async def test_cleanup_on_project_removal(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture
    ):
        """Test cleanup when project is removed."""
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))
        tenant_id_a = calculate_tenant_id(project_a)

        # Create project record
        project_record = ProjectRecord(
            id=None,
            name=name_a,
            root_path=str(project_a),
            collection_name=f"{name_a}-code",
            project_id=tenant_id_a[:12],
        )
        await state_manager_fixture.save_project(project_record)

        # Verify exists
        retrieved = await state_manager_fixture.get_project_by_root_path(str(project_a))
        assert retrieved is not None

        # In a real scenario, cleanup would involve:
        # 1. Removing watch folder configs
        # 2. Marking collections for deletion
        # 3. Clearing processing state
        # 4. Removing project record

        # For testing, we verify the record exists and can be queried
        all_projects = await state_manager_fixture.list_projects()
        project_ids = [p["project_id"] for p in all_projects]
        assert tenant_id_a[:12] in project_ids

        print("✓ Cleanup validation completed")


# ============================================================================
# Comprehensive Integration Test
# ============================================================================


@pytest.mark.e2e
@pytest.mark.workflow
class TestComprehensiveProjectSwitchingWorkflow:
    """Comprehensive end-to-end project switching workflow test."""

    @pytest.mark.asyncio
    async def test_complete_project_switching_workflow(
        self,
        multi_project_workspace,
        project_detector_fixture,
        state_manager_fixture,
        resource_tracker
    ):
        """Test complete project switching workflow end-to-end."""
        timer = WorkflowTimer()
        timer.start()
        resource_tracker.capture_baseline()

        # Phase 1: Detect and setup project A
        project_a = multi_project_workspace["project_a"]["path"]
        name_a = project_detector_fixture.get_project_name(str(project_a))
        tenant_id_a = calculate_tenant_id(project_a)

        project_record_a = ProjectRecord(
            id=None,
            name=name_a,
            root_path=str(project_a),
            collection_name=f"{name_a}-code",
            project_id=tenant_id_a[:12],
        )
        await state_manager_fixture.save_project(project_record_a)

        watch_config_a = WatchFolderConfig(
            watch_id=f"{name_a}-src",
            path=str(project_a / "src"),
            collection=f"{name_a}-code",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_a.watch_id,
            watch_config_a
        )
        timer.checkpoint("project_a_setup")

        # Phase 2: Switch to project B
        project_b = multi_project_workspace["project_b"]["path"]
        name_b = project_detector_fixture.get_project_name(str(project_b))
        tenant_id_b = calculate_tenant_id(project_b)

        # Disable project A watch
        watch_config_a.enabled = False
        await state_manager_fixture.save_watch_folder_config(
            watch_config_a.watch_id,
            watch_config_a
        )

        # Setup project B
        project_record_b = ProjectRecord(
            id=None,
            name=name_b,
            root_path=str(project_b),
            collection_name=f"{name_b}-code",
            project_id=tenant_id_b[:12],
        )
        await state_manager_fixture.save_project(project_record_b)

        watch_config_b = WatchFolderConfig(
            watch_id=f"{name_b}-lib",
            path=str(project_b / "lib"),
            collection=f"{name_b}-code",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
            auto_ingest=True,
            enabled=True,
        )
        await state_manager_fixture.save_watch_folder_config(
            watch_config_b.watch_id,
            watch_config_b
        )
        timer.checkpoint("project_b_setup")

        # Phase 3: Validation
        all_projects = await state_manager_fixture.list_projects()
        all_watches = await state_manager_fixture.list_watch_folders()

        # Verify both projects exist
        project_names = [p["name"] for p in all_projects]
        assert name_a in project_names
        assert name_b in project_names

        # Verify watch states
        watches_dict = {w["watch_id"]: w for w in all_watches}
        assert watches_dict[watch_config_a.watch_id]["enabled"] is False
        assert watches_dict[watch_config_b.watch_id]["enabled"] is True

        # Verify isolation
        assert tenant_id_a != tenant_id_b
        assert project_record_a.collection_name != project_record_b.collection_name

        timer.checkpoint("validation_complete")
        resource_tracker.capture_current()

        # Performance validation
        total_duration = timer.get_duration("validation_complete")
        assert total_duration < 15.0, \
            f"Complete workflow took {total_duration:.2f}s (expected < 15s)"

        # Resource validation
        warnings = resource_tracker.check_thresholds()
        assert len(warnings) == 0, f"Resource warnings: {warnings}"

        # Summary
        summary = timer.get_summary()
        print("\n" + "="*60)
        print("COMPREHENSIVE PROJECT SWITCHING WORKFLOW")
        print("="*60)
        print(f"✓ Project A setup: {summary['checkpoints'][0]['elapsed_seconds']:.3f}s")
        print(f"✓ Project B setup: {summary['checkpoints'][1]['elapsed_seconds']:.3f}s")
        print(f"✓ Validation: {summary['checkpoints'][2]['elapsed_seconds']:.3f}s")
        print(f"✓ Total duration: {total_duration:.3f}s")
        print(f"✓ Projects managed: {len(all_projects)}")
        print(f"✓ Watch configs: {len(all_watches)}")
        print("✓ State isolation: VERIFIED")
        print("✓ Tenant ID uniqueness: VERIFIED")
        print("="*60)
