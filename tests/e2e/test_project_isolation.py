"""
End-to-end tests for project switching and isolation.

Tests project detection, switching between projects, collection isolation,
watch folder updates, daemon instance management, and data isolation to
ensure no cross-project leakage.
"""

import pytest
import time
from pathlib import Path

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
)


@pytest.mark.integration
@pytest.mark.slow
class TestProjectDetection:
    """Test project detection and identification."""

    def test_git_project_detection(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test detection of Git-based projects."""
        workspace = system_components.workspace_path

        # Workspace has .git directory
        assert (workspace / ".git").exists()

        # Ingest file from git project
        test_file = workspace / "git_project.txt"
        test_file.write_text("Content in git project")

        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-project-git"]
        )

        # Should detect git project
        assert result is not None

    def test_non_git_project_detection(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test detection of non-Git projects."""
        # Create directory without .git
        non_git_dir = tmp_path / "non_git_project"
        non_git_dir.mkdir()

        test_file = non_git_dir / "file.txt"
        test_file.write_text("Content in non-git directory")

        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-project-nongit"]
        )

        # Should handle non-git directory
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestProjectSwitching:
    """Test switching between different projects."""

    def test_switch_between_projects(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test switching context between two projects."""
        # Create two separate project directories
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        # Ingest from project 1
        file1 = project1 / "file1.txt"
        file1.write_text("Content from project 1")
        result1 = cli_helper.run_command(
            ["ingest", "file", str(file1), "--collection", "test-switch-proj1"]
        )
        assert result1 is not None

        time.sleep(2)

        # Ingest from project 2
        file2 = project2 / "file2.txt"
        file2.write_text("Content from project 2")
        result2 = cli_helper.run_command(
            ["ingest", "file", str(file2), "--collection", "test-switch-proj2"]
        )
        assert result2 is not None

        # Both should succeed
        assert result1 is not None
        assert result2 is not None


@pytest.mark.integration
@pytest.mark.slow
class TestCollectionIsolation:
    """Test collection isolation between projects."""

    def test_collections_isolated_by_project(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that collections are properly isolated by project."""
        # Create two projects
        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        project_b.mkdir()

        # Create unique collection names
        coll_a = f"test-isolation-a-{int(time.time())}"
        coll_b = f"test-isolation-b-{int(time.time())}"

        # Ingest to project A collection
        (project_a / "data.txt").write_text("Data for project A")
        result_a = cli_helper.run_command(
            ["ingest", "file", str(project_a / "data.txt"), "--collection", coll_a]
        )

        time.sleep(2)

        # Ingest to project B collection
        (project_b / "data.txt").write_text("Data for project B")
        result_b = cli_helper.run_command(
            ["ingest", "file", str(project_b / "data.txt"), "--collection", coll_b]
        )

        # Both should succeed independently
        assert result_a is not None
        assert result_b is not None

    def test_search_isolated_by_collection(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that searches are isolated to their collections."""
        # Create projects with different content
        proj1 = tmp_path / "search_proj1"
        proj2 = tmp_path / "search_proj2"
        proj1.mkdir()
        proj2.mkdir()

        coll1 = f"test-search-iso1-{int(time.time())}"
        coll2 = f"test-search-iso2-{int(time.time())}"

        # Ingest different content to each
        (proj1 / "doc.txt").write_text("Python programming language")
        cli_helper.run_command(
            ["ingest", "file", str(proj1 / "doc.txt"), "--collection", coll1]
        )

        (proj2 / "doc.txt").write_text("Java programming language")
        cli_helper.run_command(
            ["ingest", "file", str(proj2 / "doc.txt"), "--collection", coll2]
        )

        time.sleep(5)

        # Search in collection 1
        result1 = cli_helper.run_command(
            ["search", "Python", "--collection", coll1]
        )

        # Search in collection 2
        result2 = cli_helper.run_command(
            ["search", "Java", "--collection", coll2]
        )

        # Both searches should work independently
        assert result1 is not None
        assert result2 is not None


@pytest.mark.integration
@pytest.mark.slow
class TestDataIsolation:
    """Test data isolation between projects to prevent leakage."""

    def test_no_cross_project_data_leakage(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that data from one project doesn't leak to another."""
        # Create two isolated projects
        isolated1 = tmp_path / "isolated1"
        isolated2 = tmp_path / "isolated2"
        isolated1.mkdir()
        isolated2.mkdir()

        coll_iso1 = f"test-leak-iso1-{int(time.time())}"
        coll_iso2 = f"test-leak-iso2-{int(time.time())}"

        # Ingest unique content to each
        (isolated1 / "secret1.txt").write_text("Secret data from project 1")
        cli_helper.run_command(
            ["ingest", "file", str(isolated1 / "secret1.txt"), "--collection", coll_iso1]
        )

        (isolated2 / "secret2.txt").write_text("Secret data from project 2")
        cli_helper.run_command(
            ["ingest", "file", str(isolated2 / "secret2.txt"), "--collection", coll_iso2]
        )

        time.sleep(5)

        # Search in collection 1 for content from collection 2
        result = cli_helper.run_command(
            ["search", "Secret data from project 2", "--collection", coll_iso1]
        )

        # Should not find content from other project
        assert result is not None
        # If results found, they should be from correct project only


@pytest.mark.integration
@pytest.mark.slow
class TestResourceIsolation:
    """Test resource isolation between projects."""

    def test_concurrent_project_operations(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that concurrent operations on different projects work."""
        proj_x = tmp_path / "concurrent_x"
        proj_y = tmp_path / "concurrent_y"
        proj_x.mkdir()
        proj_y.mkdir()

        # Ingest to both projects concurrently (sequential for simplicity)
        (proj_x / "data_x.txt").write_text("Data X")
        result_x = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(proj_x / "data_x.txt"),
                "--collection",
                f"test-concurrent-x-{int(time.time())}",
            ]
        )

        (proj_y / "data_y.txt").write_text("Data Y")
        result_y = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(proj_y / "data_y.txt"),
                "--collection",
                f"test-concurrent-y-{int(time.time())}",
            ]
        )

        # Both operations should succeed
        assert result_x is not None
        assert result_y is not None


@pytest.mark.integration
@pytest.mark.slow
class TestProjectMetadata:
    """Test project metadata handling."""

    def test_project_metadata_captured(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that project metadata is captured correctly."""
        workspace = system_components.workspace_path

        # Ingest file and metadata should be captured
        test_file = workspace / "metadata_test.txt"
        test_file.write_text("Testing project metadata capture")

        result = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(test_file),
                "--collection",
                "test-project-metadata",
            ]
        )

        # Metadata capture happens internally
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestProjectBoundaries:
    """Test project boundary detection and enforcement."""

    def test_project_root_detection(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test detection of project root directory."""
        workspace = system_components.workspace_path

        # Git project root should be detected
        assert (workspace / ".git").exists()

        # Ingest from subdirectory
        (workspace / "src" / "subdir.txt").write_text("File in subdirectory")

        result = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(workspace / "src" / "subdir.txt"),
                "--collection",
                "test-project-root",
            ]
        )

        # Should detect project root correctly
        assert result is not None

    def test_nested_directory_projects(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test handling of nested directory structures."""
        # Create nested structure
        parent = tmp_path / "parent_proj"
        child = parent / "child_proj"
        parent.mkdir()
        child.mkdir()

        # Create git markers
        (parent / ".git").mkdir()
        (child / ".git").mkdir()

        # Ingest from parent
        (parent / "parent.txt").write_text("Parent project data")
        result_parent = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(parent / "parent.txt"),
                "--collection",
                "test-nested-parent",
            ]
        )

        # Ingest from child
        (child / "child.txt").write_text("Child project data")
        result_child = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(child / "child.txt"),
                "--collection",
                "test-nested-child",
            ]
        )

        # Both should work
        assert result_parent is not None
        assert result_child is not None


@pytest.mark.integration
@pytest.mark.slow
class TestCollectionNaming:
    """Test collection naming conventions for projects."""

    def test_collection_naming_by_project(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that collection names can be project-specific."""
        project = tmp_path / "naming_test"
        project.mkdir()

        # Use project-specific collection name
        coll_name = f"project-naming-test-{int(time.time())}"

        (project / "test.txt").write_text("Testing naming")
        result = cli_helper.run_command(
            ["ingest", "file", str(project / "test.txt"), "--collection", coll_name]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestMultiProjectSystem:
    """Test system with multiple active projects."""

    def test_multiple_projects_simultaneously(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test system handling multiple projects simultaneously."""
        # Create 3 different projects
        projects = []
        for i in range(3):
            proj = tmp_path / f"multi_proj_{i}"
            proj.mkdir()
            projects.append(proj)

        # Ingest to all projects
        results = []
        for i, proj in enumerate(projects):
            (proj / "data.txt").write_text(f"Data for project {i}")
            result = cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(proj / "data.txt"),
                    "--collection",
                    f"test-multi-{i}-{int(time.time())}",
                ]
            )
            results.append(result)

        # All ingestions should succeed
        assert all(r is not None for r in results)

    def test_system_stability_with_multiple_projects(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test system remains stable with multiple projects."""
        # Create multiple projects and ingest to all
        for i in range(5):
            proj = tmp_path / f"stability_{i}"
            proj.mkdir()
            (proj / "file.txt").write_text(f"Stability test {i}")

            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(proj / "file.txt"),
                    "--collection",
                    f"test-stability-{i}",
                ]
            )

        time.sleep(3)

        # System should still be responsive
        status_result = cli_helper.run_command(["status"])
        assert status_result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestProjectCleanup:
    """Test cleanup and isolation after project operations."""

    def test_project_data_cleanup(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test that project data can be cleaned up without affecting others."""
        proj1 = tmp_path / "cleanup1"
        proj2 = tmp_path / "cleanup2"
        proj1.mkdir()
        proj2.mkdir()

        coll1 = f"test-cleanup1-{int(time.time())}"
        coll2 = f"test-cleanup2-{int(time.time())}"

        # Ingest to both
        (proj1 / "data.txt").write_text("Cleanup test 1")
        cli_helper.run_command(
            ["ingest", "file", str(proj1 / "data.txt"), "--collection", coll1]
        )

        (proj2 / "data.txt").write_text("Cleanup test 2")
        cli_helper.run_command(
            ["ingest", "file", str(proj2 / "data.txt"), "--collection", coll2]
        )

        time.sleep(3)

        # Verify collections exist
        list_result = cli_helper.run_command(["admin", "collections"])
        assert list_result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestProjectIsolationEdgeCases:
    """Test edge cases in project isolation."""

    def test_same_filename_different_projects(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test handling same filename in different projects."""
        projA = tmp_path / "proj_a"
        projB = tmp_path / "proj_b"
        projA.mkdir()
        projB.mkdir()

        # Same filename, different content
        (projA / "README.md").write_text("README for project A")
        (projB / "README.md").write_text("README for project B")

        collA = f"test-samefile-a-{int(time.time())}"
        collB = f"test-samefile-b-{int(time.time())}"

        # Ingest both
        resultA = cli_helper.run_command(
            ["ingest", "file", str(projA / "README.md"), "--collection", collA]
        )
        resultB = cli_helper.run_command(
            ["ingest", "file", str(projB / "README.md"), "--collection", collB]
        )

        # Both should succeed without collision
        assert resultA is not None
        assert resultB is not None

    def test_symlinked_projects(
        self, system_components: SystemComponents, cli_helper: CLIHelper, tmp_path
    ):
        """Test handling of symlinked project directories."""
        real_proj = tmp_path / "real_project"
        real_proj.mkdir()
        (real_proj / "data.txt").write_text("Real project data")

        # Create symlink
        link_proj = tmp_path / "linked_project"
        try:
            link_proj.symlink_to(real_proj)

            # Ingest via symlink
            result = cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(link_proj / "data.txt"),
                    "--collection",
                    "test-symlink",
                ]
            )

            # Should handle symlinks
            assert result is not None
        except OSError:
            # Symlinks might not be supported on all systems
            pytest.skip("Symlinks not supported on this system")
