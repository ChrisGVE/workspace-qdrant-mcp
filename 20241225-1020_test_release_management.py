#!/usr/bin/env python3
"""
Comprehensive unit tests for the Release Management system.

Tests cover:
- Semantic version parsing, comparison, and bumping
- Conventional commit parsing and analysis
- Changelog generation and formatting
- Release orchestration and quality gates
- Hotfix creation and emergency releases
- Version history tracking and rollback mechanisms
- Git integration and branch management
- Release notes generation and artifact management
"""

import asyncio
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import toml

# Import the module under test
try:
    from release_management import (
        ChangelogEntry,
        CommitType,
        ConventionalCommit,
        ReleaseConfig,
        ReleaseManager,
        ReleaseNotes,
        ReleaseProgress,
        ReleaseStatus,
        ReleaseType,
        SemanticVersion,
        create_example_release_config,
    )
except ImportError:
    # For when running as standalone module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from release_management import (
        ChangelogEntry,
        CommitType,
        ConventionalCommit,
        ReleaseConfig,
        ReleaseManager,
        ReleaseNotes,
        ReleaseProgress,
        ReleaseStatus,
        ReleaseType,
        SemanticVersion,
        create_example_release_config,
    )


class TestSemanticVersion:
    """Test SemanticVersion class."""

    def test_semantic_version_creation(self):
        """Test semantic version creation."""
        version = SemanticVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_semantic_version_with_prerelease(self):
        """Test semantic version with prerelease."""
        version = SemanticVersion(1, 0, 0, prerelease="alpha.1")
        assert version.prerelease == "alpha.1"
        assert str(version) == "1.0.0-alpha.1"

    def test_semantic_version_with_build(self):
        """Test semantic version with build metadata."""
        version = SemanticVersion(1, 0, 0, build="20231225.1")
        assert version.build == "20231225.1"
        assert str(version) == "1.0.0+20231225.1"

    def test_semantic_version_full(self):
        """Test semantic version with prerelease and build."""
        version = SemanticVersion(1, 0, 0, prerelease="beta.2", build="exp.sha.5114f85")
        assert str(version) == "1.0.0-beta.2+exp.sha.5114f85"

    def test_semantic_version_parsing(self):
        """Test parsing semantic versions from strings."""
        # Basic version
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

        # With prerelease
        version = SemanticVersion.parse("2.0.0-alpha.1")
        assert version.major == 2
        assert version.prerelease == "alpha.1"

        # With build
        version = SemanticVersion.parse("1.0.0+20231225")
        assert version.build == "20231225"

        # Full version
        version = SemanticVersion.parse("3.1.4-beta.2+exp.sha.abc123")
        assert version.major == 3
        assert version.minor == 1
        assert version.patch == 4
        assert version.prerelease == "beta.2"
        assert version.build == "exp.sha.abc123"

        # With 'v' prefix
        version = SemanticVersion.parse("v1.0.0")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0

    def test_semantic_version_parsing_invalid(self):
        """Test parsing invalid semantic versions."""
        invalid_versions = [
            "1.2",
            "1.2.3.4",
            "v1.2.3.4",
            "1.2.three",
            "not-a-version",
            "",
            "1.2.-3",
            "1.2.3-",
            "1.2.3+"
        ]

        for invalid_version in invalid_versions:
            with pytest.raises(ValueError, match="Invalid semantic version"):
                SemanticVersion.parse(invalid_version)

    def test_semantic_version_comparison(self):
        """Test semantic version comparison."""
        v1_0_0 = SemanticVersion(1, 0, 0)
        v1_0_1 = SemanticVersion(1, 0, 1)
        v1_1_0 = SemanticVersion(1, 1, 0)
        v2_0_0 = SemanticVersion(2, 0, 0)

        # Basic comparisons
        assert v1_0_0 < v1_0_1
        assert v1_0_1 < v1_1_0
        assert v1_1_0 < v2_0_0

        # Prerelease comparisons
        v1_0_0_alpha = SemanticVersion(1, 0, 0, prerelease="alpha.1")
        v1_0_0_beta = SemanticVersion(1, 0, 0, prerelease="beta.1")

        assert v1_0_0_alpha < v1_0_0_beta
        assert v1_0_0_alpha < v1_0_0
        assert v1_0_0_beta < v1_0_0

    def test_semantic_version_bumping(self):
        """Test version bumping."""
        version = SemanticVersion(1, 2, 3)

        # Major bump
        major = version.bump(ReleaseType.MAJOR)
        assert str(major) == "2.0.0"

        # Minor bump
        minor = version.bump(ReleaseType.MINOR)
        assert str(minor) == "1.3.0"

        # Patch bump
        patch = version.bump(ReleaseType.PATCH)
        assert str(patch) == "1.2.4"

        # Prerelease bump (no existing prerelease)
        prerelease = version.bump(ReleaseType.PRERELEASE)
        assert str(prerelease) == "1.2.3-alpha.1"

    def test_semantic_version_prerelease_bumping(self):
        """Test prerelease version bumping."""
        version = SemanticVersion(1, 0, 0, prerelease="alpha.1")

        # Prerelease bump
        bumped = version.bump(ReleaseType.PRERELEASE)
        assert str(bumped) == "1.0.0-alpha.2"

        # Complex prerelease
        version = SemanticVersion(1, 0, 0, prerelease="beta")
        bumped = version.bump(ReleaseType.PRERELEASE)
        assert str(bumped) == "1.0.0-beta.1"

    def test_semantic_version_unsupported_bump(self):
        """Test unsupported version bump types."""
        version = SemanticVersion(1, 0, 0)

        with pytest.raises(ValueError, match="Unsupported release type"):
            version.bump(ReleaseType.HOTFIX)


class TestConventionalCommit:
    """Test ConventionalCommit class."""

    def test_conventional_commit_parsing_basic(self):
        """Test basic conventional commit parsing."""
        commit_line = "feat: add user authentication"
        commit = ConventionalCommit.parse(
            commit_line,
            "abc123",
            "John Doe",
            datetime.now(timezone.utc)
        )

        assert commit is not None
        assert commit.commit_type == CommitType.FEAT
        assert commit.scope is None
        assert commit.description == "add user authentication"
        assert not commit.breaking_change

    def test_conventional_commit_parsing_with_scope(self):
        """Test conventional commit parsing with scope."""
        commit_line = "fix(auth): resolve login timeout issue"
        commit = ConventionalCommit.parse(
            commit_line,
            "def456",
            "Jane Smith",
            datetime.now(timezone.utc)
        )

        assert commit is not None
        assert commit.commit_type == CommitType.FIX
        assert commit.scope == "auth"
        assert commit.description == "resolve login timeout issue"
        assert not commit.breaking_change

    def test_conventional_commit_parsing_breaking_change_marker(self):
        """Test conventional commit parsing with breaking change marker."""
        commit_line = "feat!: remove deprecated API endpoint"
        commit = ConventionalCommit.parse(
            commit_line,
            "ghi789",
            "Bob Johnson",
            datetime.now(timezone.utc)
        )

        assert commit is not None
        assert commit.commit_type == CommitType.FEAT
        assert commit.breaking_change

    def test_conventional_commit_parsing_breaking_change_footer(self):
        """Test conventional commit parsing with breaking change in footer."""
        commit_line = """feat(api): add new endpoint for user management

Add comprehensive user management API with CRUD operations.

BREAKING CHANGE: The old user API endpoints have been deprecated
and will be removed in the next major version."""

        commit = ConventionalCommit.parse(
            commit_line,
            "jkl012",
            "Alice Brown",
            datetime.now(timezone.utc)
        )

        assert commit is not None
        assert commit.commit_type == CommitType.FEAT
        assert commit.scope == "api"
        assert commit.breaking_change
        assert "BREAKING CHANGE" in commit.footer

    def test_conventional_commit_parsing_with_body(self):
        """Test conventional commit parsing with body."""
        commit_line = """refactor: restructure authentication module

Move authentication logic to separate service classes
for better maintainability and testability."""

        commit = ConventionalCommit.parse(
            commit_line,
            "mno345",
            "Charlie Wilson",
            datetime.now(timezone.utc)
        )

        assert commit is not None
        assert commit.commit_type == CommitType.REFACTOR
        assert commit.body is not None
        assert "Move authentication logic" in commit.body

    def test_conventional_commit_parsing_invalid(self):
        """Test parsing invalid conventional commits."""
        invalid_commits = [
            "random commit message",
            "FEAT: should be lowercase",
            "fix:",  # Empty description
            ": no type",
            "fix(unclosed-scope: description",
        ]

        for invalid_commit in invalid_commits:
            result = ConventionalCommit.parse(
                invalid_commit,
                "invalid",
                "Test User",
                datetime.now(timezone.utc)
            )
            assert result is None

    def test_conventional_commit_types(self):
        """Test all conventional commit types."""
        commit_types = [
            ("feat: add feature", CommitType.FEAT),
            ("fix: bug fix", CommitType.FIX),
            ("docs: update documentation", CommitType.DOCS),
            ("style: code formatting", CommitType.STYLE),
            ("refactor: code refactoring", CommitType.REFACTOR),
            ("perf: performance improvement", CommitType.PERF),
            ("test: add tests", CommitType.TEST),
            ("build: build system changes", CommitType.BUILD),
            ("ci: CI configuration", CommitType.CI),
            ("chore: maintenance tasks", CommitType.CHORE),
            ("revert: revert previous change", CommitType.REVERT),
        ]

        for commit_line, expected_type in commit_types:
            commit = ConventionalCommit.parse(
                commit_line,
                "test",
                "Test User",
                datetime.now(timezone.utc)
            )
            assert commit is not None
            assert commit.commit_type == expected_type


class TestReleaseConfig:
    """Test ReleaseConfig class."""

    def test_release_config_creation(self):
        """Test release configuration creation."""
        config = ReleaseConfig(
            project_name="test-project",
            repository_url="https://github.com/test/repo",
            default_branch="main",
            version_files=["pyproject.toml", "package.json"],
            quality_gates=["tests", "security"]
        )

        assert config.project_name == "test-project"
        assert config.repository_url == "https://github.com/test/repo"
        assert config.default_branch == "main"
        assert "pyproject.toml" in config.version_files
        assert "tests" in config.quality_gates

    def test_release_config_defaults(self):
        """Test release configuration defaults."""
        config = ReleaseConfig(
            project_name="minimal-project",
            repository_url="https://github.com/test/minimal"
        )

        assert config.default_branch == "main"
        assert config.release_branches == ["main", "release/*"]
        assert config.version_files == ["pyproject.toml", "package.json"]
        assert config.changelog_file == "CHANGELOG.md"
        assert config.auto_tag is True
        assert config.auto_publish is False


class TestReleaseManager:
    """Test ReleaseManager class."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
            subprocess.run(["git", "config", "init.defaultBranch", "main"], cwd=repo_path, check=True)

            # Create initial commit
            (repo_path / "README.md").write_text("# Test Project\n")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)

            # Create pyproject.toml with version
            pyproject_content = """
[project]
name = "test-project"
version = "1.0.0"
description = "Test project"
"""
            (repo_path / "pyproject.toml").write_text(pyproject_content)
            subprocess.run(["git", "add", "pyproject.toml"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", "Add pyproject.toml"], cwd=repo_path, check=True)

            yield repo_path

    @pytest.fixture
    def release_config(self):
        """Create test release configuration."""
        return ReleaseConfig(
            project_name="test-project",
            repository_url="https://github.com/test/project",
            version_files=["pyproject.toml"],
            quality_gates=["tests"]
        )

    @pytest.fixture
    def release_manager(self, temp_git_repo, release_config):
        """Create test release manager."""
        return ReleaseManager(
            config=release_config,
            repo_path=temp_git_repo,
            artifacts_dir=temp_git_repo / "artifacts",
            logs_dir=temp_git_repo / "logs"
        )

    def test_release_manager_initialization(self, temp_git_repo, release_config):
        """Test release manager initialization."""
        manager = ReleaseManager(
            config=release_config,
            repo_path=temp_git_repo
        )

        assert manager.config == release_config
        assert manager.repo_path == temp_git_repo
        assert manager.artifacts_dir.exists()
        assert manager.logs_dir.exists()

    def test_release_manager_invalid_repo(self, release_config):
        """Test release manager with invalid repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_repo = Path(tmpdir)  # Not a Git repository

            with pytest.raises(RuntimeError, match="Not a Git repository"):
                ReleaseManager(config=release_config, repo_path=invalid_repo)

    def test_get_current_branch(self, release_manager):
        """Test getting current Git branch."""
        branch = release_manager._get_current_branch()
        assert branch == "main"

    def test_get_current_version(self, release_manager):
        """Test getting current version from files."""
        version = release_manager._get_current_version()
        assert str(version) == "1.0.0"

    def test_get_current_version_missing_files(self, temp_git_repo):
        """Test getting current version with missing version files."""
        config = ReleaseConfig(
            project_name="test",
            repository_url="https://example.com",
            version_files=["nonexistent.json"]
        )

        manager = ReleaseManager(config=config, repo_path=temp_git_repo)

        with pytest.raises(RuntimeError, match="No valid version found"):
            manager._get_current_version()

    def test_update_version_files(self, release_manager, temp_git_repo):
        """Test updating version files."""
        new_version = SemanticVersion(2, 1, 0)
        release_manager._update_version_files(new_version)

        # Verify version was updated
        data = toml.load(temp_git_repo / "pyproject.toml")
        assert data["project"]["version"] == "2.1.0"

    def test_get_commits_since_last_release(self, release_manager, temp_git_repo):
        """Test getting commits since last release."""
        # Add some conventional commits
        (temp_git_repo / "test.py").write_text("print('hello')")
        subprocess.run(["git", "add", "test.py"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "feat: add hello world script"], cwd=temp_git_repo, check=True)

        (temp_git_repo / "test.py").write_text("print('hello world')")
        subprocess.run(["git", "add", "test.py"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "fix: improve greeting message"], cwd=temp_git_repo, check=True)

        # Get commits
        commits = release_manager._get_commits_since_last_release()

        # Should have at least the two commits we added
        assert len(commits) >= 2

        # Check that conventional commits were parsed
        feat_commits = [c for c in commits if c.commit_type == CommitType.FEAT]
        fix_commits = [c for c in commits if c.commit_type == CommitType.FIX]

        assert len(feat_commits) >= 1
        assert len(fix_commits) >= 1

    def test_determine_next_version(self, release_manager):
        """Test determining next version from commits."""
        current_version = SemanticVersion(1, 0, 0)

        # Test with feature commits
        feat_commit = ConventionalCommit(
            commit_type=CommitType.FEAT,
            scope=None,
            description="add new feature",
            body=None,
            footer=None,
            breaking_change=False,
            commit_hash="abc123",
            author="Test User",
            timestamp=datetime.now(timezone.utc)
        )

        next_version, release_type = release_manager.determine_next_version([feat_commit], current_version)
        assert str(next_version) == "1.1.0"
        assert release_type == ReleaseType.MINOR

        # Test with fix commits only
        fix_commit = ConventionalCommit(
            commit_type=CommitType.FIX,
            scope=None,
            description="fix bug",
            body=None,
            footer=None,
            breaking_change=False,
            commit_hash="def456",
            author="Test User",
            timestamp=datetime.now(timezone.utc)
        )

        next_version, release_type = release_manager.determine_next_version([fix_commit], current_version)
        assert str(next_version) == "1.0.1"
        assert release_type == ReleaseType.PATCH

        # Test with breaking changes
        breaking_commit = ConventionalCommit(
            commit_type=CommitType.FEAT,
            scope=None,
            description="add breaking feature",
            body=None,
            footer=None,
            breaking_change=True,
            commit_hash="ghi789",
            author="Test User",
            timestamp=datetime.now(timezone.utc)
        )

        next_version, release_type = release_manager.determine_next_version([breaking_commit], current_version)
        assert str(next_version) == "2.0.0"
        assert release_type == ReleaseType.MAJOR

    def test_generate_changelog(self, release_manager):
        """Test changelog generation."""
        version = SemanticVersion(1, 1, 0)

        commits = [
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope="auth",
                description="add user authentication",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="abc123",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FIX,
                scope="ui",
                description="fix button alignment",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="def456",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope=None,
                description="remove old API",
                body=None,
                footer="BREAKING CHANGE: Old API endpoints removed",
                breaking_change=True,
                commit_hash="ghi789",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            )
        ]

        changelog = release_manager.generate_changelog(version, commits)

        assert changelog.version == version
        assert CommitType.FEAT in changelog.changes
        assert CommitType.FIX in changelog.changes
        assert len(changelog.breaking_changes) == 1
        assert changelog.breaking_changes[0].description == "remove old API"

    def test_format_changelog_entry(self, release_manager):
        """Test changelog entry formatting."""
        version = SemanticVersion(1, 2, 0)
        commits = [
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope="api",
                description="add user endpoints",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="abc123",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FIX,
                scope=None,
                description="resolve memory leak",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="def456",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            )
        ]

        changelog = ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc),
            changes={
                CommitType.FEAT: [commits[0]],
                CommitType.FIX: [commits[1]]
            },
            breaking_changes=[]
        )

        formatted = release_manager._format_changelog_entry(changelog)

        assert "## [1.2.0]" in formatted
        assert "### Added" in formatted
        assert "### Fixed" in formatted
        assert "**api**: add user endpoints" in formatted
        assert "resolve memory leak" in formatted

    def test_update_changelog_file(self, release_manager, temp_git_repo):
        """Test updating changelog file."""
        version = SemanticVersion(1, 1, 0)
        commit = ConventionalCommit(
            commit_type=CommitType.FEAT,
            scope=None,
            description="add new feature",
            body=None,
            footer=None,
            breaking_change=False,
            commit_hash="abc123",
            author="Test User",
            timestamp=datetime.now(timezone.utc)
        )

        changelog = ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc),
            changes={CommitType.FEAT: [commit]},
            breaking_changes=[]
        )

        release_manager.update_changelog_file(changelog)

        # Verify changelog file was created
        changelog_path = temp_git_repo / "CHANGELOG.md"
        assert changelog_path.exists()

        content = changelog_path.read_text()
        assert "## [1.1.0]" in content
        assert "### Added" in content
        assert "add new feature" in content

    def test_generate_release_notes(self, release_manager):
        """Test release notes generation."""
        version = SemanticVersion(2, 0, 0)
        commits = [
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope="api",
                description="add GraphQL support",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="abc123",
                author="Alice",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FIX,
                scope="db",
                description="fix connection pooling",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="def456",
                author="Bob",
                timestamp=datetime.now(timezone.utc)
            )
        ]

        changelog = ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc),
            changes={
                CommitType.FEAT: [commits[0]],
                CommitType.FIX: [commits[1]]
            },
            breaking_changes=[]
        )

        release_notes = release_manager.generate_release_notes(version, changelog)

        assert release_notes.version == version
        assert "test-project v2.0.0" in release_notes.title
        assert "2 commits" in release_notes.summary
        assert "1 new features" in release_notes.highlights
        assert "1 bug fixes" in release_notes.highlights
        assert "Alice" in release_notes.contributors
        assert "Bob" in release_notes.contributors

    def test_format_release_notes(self, release_manager):
        """Test release notes formatting."""
        version = SemanticVersion(1, 5, 2)
        release_notes = ReleaseNotes(
            version=version,
            title="Test Project v1.5.2",
            summary="Bug fix release with 3 commits",
            highlights=["2 bug fixes", "1 performance improvement"],
            breaking_changes=[],
            migration_guide=None,
            dependencies={"requests": ">=2.28.0", "pydantic": ">=2.0.0"},
            compatibility={"python": ">=3.10"},
            known_issues=[],
            contributors=["Alice", "Bob"]
        )

        formatted = release_manager._format_release_notes(release_notes)

        assert "# Test Project v1.5.2" in formatted
        assert "## Highlights" in formatted
        assert "2 bug fixes" in formatted
        assert "## Dependencies" in formatted
        assert "requests: >=2.28.0" in formatted
        assert "## Contributors" in formatted
        assert "@Alice" in formatted

    @patch('subprocess.run')
    async def test_validate_release_conditions_clean_repo(self, mock_run, release_manager):
        """Test release condition validation with clean repository."""
        # Mock git status (clean repo)
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        progress = ReleaseProgress(
            release_id="test-123",
            version="1.1.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )

        # Should not raise exception
        await release_manager._validate_release_conditions(progress)

    @patch('subprocess.run')
    async def test_validate_release_conditions_invalid_branch(self, mock_run, release_manager):
        """Test release condition validation with invalid branch."""
        # Mock git status (clean)
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        # Mock current branch as invalid
        release_manager._get_current_branch = MagicMock(return_value="feature/test")

        progress = ReleaseProgress(
            release_id="test-123",
            version="1.1.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="feature/test"
        )

        with pytest.raises(RuntimeError, match="Invalid release branch"):
            await release_manager._validate_release_conditions(progress)

    @patch('subprocess.run')
    async def test_commit_changes(self, mock_run, release_manager):
        """Test committing changes to Git."""
        await release_manager._commit_changes("test commit message")

        # Should have called git add and git commit
        assert mock_run.call_count >= 2

        # Check that add was called for version files and changelog
        add_calls = [call for call in mock_run.call_args_list if "add" in call[0][0]]
        assert len(add_calls) > 0

        # Check that commit was called
        commit_calls = [call for call in mock_run.call_args_list if "commit" in call[0][0]]
        assert len(commit_calls) == 1
        assert "test commit message" in str(commit_calls[0])

    @patch('subprocess.run')
    async def test_create_tag(self, mock_run, release_manager):
        """Test creating Git tag."""
        await release_manager._create_tag("v1.1.0", "Release v1.1.0")

        mock_run.assert_called_with(
            ["git", "tag", "-a", "v1.1.0", "-m", "Release v1.1.0"],
            cwd=release_manager.repo_path,
            check=True
        )

    @patch('subprocess.run')
    async def test_get_latest_commit_hash(self, mock_run, release_manager):
        """Test getting latest commit hash."""
        mock_run.return_value = MagicMock(stdout="abc1234567890\n", returncode=0)

        commit_hash = await release_manager._get_latest_commit_hash()

        assert commit_hash == "abc1234567890"
        mock_run.assert_called_with(
            ["git", "rev-parse", "HEAD"],
            cwd=release_manager.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

    async def test_save_release_notes(self, release_manager):
        """Test saving release notes to file."""
        version = SemanticVersion(1, 2, 3)
        release_notes = ReleaseNotes(
            version=version,
            title="Test Release v1.2.3",
            summary="Test release",
            highlights=[],
            breaking_changes=[],
            migration_guide=None,
            dependencies={},
            compatibility={},
            known_issues=[],
            contributors=[]
        )

        await release_manager._save_release_notes(release_notes)

        # Check that file was created
        notes_path = release_manager.artifacts_dir / "release-notes-1.2.3.md"
        assert notes_path.exists()

        content = notes_path.read_text()
        assert "# Test Release v1.2.3" in content

    @patch.object(ReleaseManager, '_run_test_suite')
    @patch.object(ReleaseManager, '_run_security_scan')
    async def test_run_release_quality_gates_success(
        self,
        mock_security,
        mock_tests,
        release_manager
    ):
        """Test successful quality gate execution."""
        mock_tests.return_value = True
        mock_security.return_value = True

        # Update config to include quality gates
        release_manager.config.quality_gates = ["tests", "security"]

        progress = ReleaseProgress(
            release_id="test-123",
            version="1.1.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )

        await release_manager._run_release_quality_gates(progress)

        assert progress.validation_results["tests"] is True
        assert progress.validation_results["security"] is True

    @patch.object(ReleaseManager, '_run_test_suite')
    async def test_run_release_quality_gates_failure(self, mock_tests, release_manager):
        """Test quality gate failure."""
        mock_tests.return_value = False

        release_manager.config.quality_gates = ["tests"]

        progress = ReleaseProgress(
            release_id="test-123",
            version="1.1.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )

        with pytest.raises(RuntimeError, match="Quality gate 'tests' failed"):
            await release_manager._run_release_quality_gates(progress)

        assert progress.validation_results["tests"] is False

    @patch('subprocess.run')
    async def test_run_test_suite_success(self, mock_run, release_manager):
        """Test successful test suite execution."""
        mock_run.return_value = MagicMock(returncode=0)

        result = await release_manager._run_test_suite()

        assert result is True
        mock_run.assert_called()

    @patch('subprocess.run')
    async def test_run_test_suite_failure(self, mock_run, release_manager):
        """Test failed test suite execution."""
        mock_run.return_value = MagicMock(returncode=1)

        result = await release_manager._run_test_suite()

        assert result is False

    @patch('subprocess.run')
    async def test_run_security_scan_success(self, mock_run, release_manager):
        """Test successful security scan."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"results": []}')

        result = await release_manager._run_security_scan()

        assert result is True

    @patch('subprocess.run')
    async def test_run_security_scan_with_high_severity(self, mock_run, release_manager):
        """Test security scan with high severity issues."""
        scan_results = {
            "results": [
                {"issue_severity": "HIGH", "issue_text": "Critical vulnerability"},
                {"issue_severity": "LOW", "issue_text": "Minor issue"}
            ]
        }
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=json.dumps(scan_results)
        )

        result = await release_manager._run_security_scan()

        assert result is False

    @patch.object(ReleaseManager, '_get_current_version')
    @patch.object(ReleaseManager, '_get_commits_since_last_release')
    @patch.object(ReleaseManager, '_validate_release_conditions')
    @patch.object(ReleaseManager, '_update_version_files')
    @patch.object(ReleaseManager, '_commit_changes')
    @patch.object(ReleaseManager, '_get_latest_commit_hash')
    @patch.object(ReleaseManager, '_create_tag')
    @patch.object(ReleaseManager, '_save_release_notes')
    @patch.object(ReleaseManager, '_run_release_quality_gates')
    async def test_create_release_success(
        self,
        mock_quality_gates,
        mock_save_notes,
        mock_create_tag,
        mock_get_hash,
        mock_commit,
        mock_update_version,
        mock_validate,
        mock_get_commits,
        mock_get_version,
        release_manager
    ):
        """Test successful release creation."""
        # Mock dependencies
        mock_get_version.return_value = SemanticVersion(1, 0, 0)
        mock_get_commits.return_value = [
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope=None,
                description="add feature",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="abc123",
                author="Test User",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        mock_get_hash.return_value = "def456"

        release_id = await release_manager.create_release(dry_run=True)

        assert release_id.startswith("release-")
        assert len(release_manager.release_history) == 1

        release = release_manager.release_history[0]
        assert release.status == ReleaseStatus.RELEASED
        assert release.version == "1.1.0"
        assert release.release_type == ReleaseType.MINOR

    @patch.object(ReleaseManager, '_get_current_version')
    @patch.object(ReleaseManager, '_get_commits_since_last_release')
    async def test_create_release_no_commits(
        self,
        mock_get_commits,
        mock_get_version,
        release_manager
    ):
        """Test release creation with no commits."""
        mock_get_version.return_value = SemanticVersion(1, 0, 0)
        mock_get_commits.return_value = []

        with pytest.raises(RuntimeError, match="No commits since last release"):
            await release_manager.create_release()

    @patch.object(ReleaseManager, '_get_current_version')
    @patch.object(ReleaseManager, '_get_commits_since_last_release')
    async def test_create_release_force_no_commits(
        self,
        mock_get_commits,
        mock_get_version,
        release_manager
    ):
        """Test forced release creation with no commits."""
        mock_get_version.return_value = SemanticVersion(1, 0, 0)
        mock_get_commits.return_value = []

        with patch.object(release_manager, '_validate_release_conditions'):
            with patch.object(release_manager, '_update_version_files'):
                with patch.object(release_manager, '_commit_changes'):
                    with patch.object(release_manager, '_get_latest_commit_hash', return_value="abc123"):
                        with patch.object(release_manager, '_create_tag'):
                            with patch.object(release_manager, '_save_release_notes'):
                                with patch.object(release_manager, '_run_release_quality_gates'):
                                    release_id = await release_manager.create_release(
                                        release_type=ReleaseType.PATCH,
                                        force=True,
                                        dry_run=True
                                    )

        assert release_id.startswith("release-")
        release = release_manager.release_history[0]
        assert release.version == "1.0.1"

    @patch.object(ReleaseManager, '_get_current_version')
    @patch.object(ReleaseManager, '_create_hotfix_branch')
    @patch.object(ReleaseManager, '_update_version_files')
    @patch.object(ReleaseManager, '_commit_changes')
    @patch.object(ReleaseManager, '_get_latest_commit_hash')
    @patch.object(ReleaseManager, '_create_tag')
    @patch.object(ReleaseManager, '_save_release_notes')
    @patch.object(ReleaseManager, '_get_current_user')
    async def test_create_hotfix_success(
        self,
        mock_get_user,
        mock_save_notes,
        mock_create_tag,
        mock_get_hash,
        mock_commit,
        mock_update_version,
        mock_create_branch,
        mock_get_version,
        release_manager
    ):
        """Test successful hotfix creation."""
        mock_get_version.return_value = SemanticVersion(1, 2, 3)
        mock_get_hash.return_value = "hotfix123"
        mock_get_user.return_value = "Hotfix User"

        hotfix_id = await release_manager.create_hotfix(
            fix_description="fix critical security vulnerability",
            dry_run=True
        )

        assert hotfix_id.startswith("hotfix-")
        assert len(release_manager.release_history) == 1

        hotfix = release_manager.release_history[0]
        assert hotfix.status == ReleaseStatus.RELEASED
        assert hotfix.version == "1.2.4-hotfix"
        assert hotfix.release_type == ReleaseType.HOTFIX

    @patch('subprocess.run')
    async def test_create_hotfix_branch(self, mock_run, release_manager):
        """Test creating hotfix branch."""
        await release_manager._create_hotfix_branch("hotfix/1.2.4", "1.2.3")

        mock_run.assert_called_with(
            ["git", "checkout", "-b", "hotfix/1.2.4", "v1.2.3"],
            cwd=release_manager.repo_path,
            check=True
        )

    @patch('subprocess.run')
    async def test_get_current_user(self, mock_run, release_manager):
        """Test getting current Git user."""
        mock_run.return_value = MagicMock(stdout="Test User\n", returncode=0)

        user = await release_manager._get_current_user()

        assert user == "Test User"

    def test_matches_pattern(self, release_manager):
        """Test branch pattern matching."""
        assert release_manager._matches_pattern("main", "main")
        assert release_manager._matches_pattern("release/1.0.0", "release/*")
        assert not release_manager._matches_pattern("feature/test", "main")
        assert not release_manager._matches_pattern("feature/test", "release/*")

    def test_get_release_status(self, release_manager):
        """Test getting release status."""
        # Add active release
        active_progress = ReleaseProgress(
            release_id="active-123",
            version="1.1.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )
        release_manager.active_releases["active-123"] = active_progress

        # Add historical release
        historical_progress = ReleaseProgress(
            release_id="history-456",
            version="1.0.0",
            release_type=ReleaseType.PATCH,
            status=ReleaseStatus.RELEASED,
            start_time=time.time(),
            branch="main"
        )
        release_manager.release_history.append(historical_progress)

        # Test getting active release
        status = release_manager.get_release_status("active-123")
        assert status == active_progress

        # Test getting historical release
        status = release_manager.get_release_status("history-456")
        assert status == historical_progress

        # Test getting non-existent release
        status = release_manager.get_release_status("non-existent")
        assert status is None

    def test_list_releases(self, release_manager):
        """Test listing releases."""
        # Add multiple releases to history
        for i in range(5):
            progress = ReleaseProgress(
                release_id=f"release-{i}",
                version=f"1.0.{i}",
                release_type=ReleaseType.PATCH,
                status=ReleaseStatus.RELEASED,
                start_time=time.time() - (5 - i) * 100,  # Different times for sorting
                branch="main"
            )
            release_manager.release_history.append(progress)

        # Add active release
        active_progress = ReleaseProgress(
            release_id="active",
            version="1.0.5",
            release_type=ReleaseType.PATCH,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )
        release_manager.active_releases["active"] = active_progress

        releases = release_manager.list_releases(limit=3)

        # Should return most recent releases first
        assert len(releases) == 3
        assert releases[0].release_id == "active"  # Most recent
        assert releases[1].release_id == "release-4"
        assert releases[2].release_id == "release-3"

    @patch('subprocess.run')
    def test_get_version_history(self, mock_run, release_manager):
        """Test getting version history from Git tags."""
        mock_run.return_value = MagicMock(
            stdout="v1.0.0\nv1.0.1\nv1.1.0\nv2.0.0\n",
            returncode=0
        )

        versions = release_manager.get_version_history()

        assert len(versions) == 4
        assert str(versions[0]) == "1.0.0"
        assert str(versions[1]) == "1.0.1"
        assert str(versions[2]) == "1.1.0"
        assert str(versions[3]) == "2.0.0"

    @patch('subprocess.run')
    def test_get_version_history_git_error(self, mock_run, release_manager):
        """Test getting version history with Git error."""
        mock_run.return_value = MagicMock(returncode=1)

        versions = release_manager.get_version_history()

        assert versions == []

    def test_generate_release_report(self, release_manager):
        """Test generating release report."""
        progress = ReleaseProgress(
            release_id="report-test",
            version="1.2.0",
            release_type=ReleaseType.MINOR,
            status=ReleaseStatus.RELEASED,
            start_time=time.time(),
            end_time=time.time() + 300,
            duration=300.0,
            branch="main",
            commit_hash="abc123",
            tag="v1.2.0",
            environments_deployed=["staging", "production"],
            validation_results={"tests": True, "security": True}
        )
        release_manager.release_history.append(progress)

        report = release_manager.generate_release_report("report-test")

        assert report["release_id"] == "report-test"
        assert report["version"] == "1.2.0"
        assert report["release_type"] == "minor"
        assert report["status"] == "released"
        assert report["duration"] == 300.0
        assert report["commit_hash"] == "abc123"
        assert report["tag"] == "v1.2.0"
        assert report["environments_deployed"] == ["staging", "production"]
        assert report["validation_results"]["tests"] is True

    def test_generate_release_report_not_found(self, release_manager):
        """Test generating report for non-existent release."""
        with pytest.raises(ValueError, match="not found"):
            release_manager.generate_release_report("non-existent")

    def test_print_release_status_table(self, release_manager):
        """Test printing release status table."""
        # Add some releases
        for i in range(3):
            progress = ReleaseProgress(
                release_id=f"table-test-{i}",
                version=f"1.0.{i}",
                release_type=ReleaseType.PATCH,
                status=ReleaseStatus.RELEASED,
                start_time=time.time(),
                end_time=time.time() + 100,
                duration=100.0,
                branch="main"
            )
            release_manager.release_history.append(progress)

        # Should not raise exception (output testing is complex with Rich)
        release_manager.print_release_status_table()


class TestComplexReleaseScenarios:
    """Test complex release scenarios and edge cases."""

    @pytest.fixture
    def complex_release_config(self):
        """Create complex release configuration."""
        return ReleaseConfig(
            project_name="enterprise-app",
            repository_url="https://github.com/company/enterprise-app",
            default_branch="main",
            release_branches=["main", "release/*", "hotfix/*", "support/*"],
            version_files=["pyproject.toml", "package.json", "VERSION"],
            changelog_file="CHANGELOG.md",
            quality_gates=["lint", "tests", "security", "performance", "integration", "smoke"],
            auto_tag=True,
            auto_publish=True,
            environments=["dev", "staging", "production", "canary"],
            notification_webhooks=["https://hooks.slack.com/services/..."]
        )

    @pytest.fixture
    def complex_release_manager(self, temp_git_repo, complex_release_config):
        """Create complex release manager."""
        # Add multiple version files
        package_json = {
            "name": "enterprise-app",
            "version": "2.1.0",
            "description": "Enterprise application"
        }
        (temp_git_repo / "package.json").write_text(json.dumps(package_json, indent=2))

        (temp_git_repo / "VERSION").write_text("2.1.0\n")

        # Commit version files
        subprocess.run(["git", "add", "package.json", "VERSION"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Add version files"], cwd=temp_git_repo, check=True)

        return ReleaseManager(
            config=complex_release_config,
            repo_path=temp_git_repo,
            artifacts_dir=temp_git_repo / "release-artifacts",
            logs_dir=temp_git_repo / "release-logs"
        )

    def test_complex_version_file_handling(self, complex_release_manager, temp_git_repo):
        """Test handling multiple version files."""
        # Get current version
        version = complex_release_manager._get_current_version()
        assert str(version) == "2.1.0"

        # Update version files
        new_version = SemanticVersion(3, 0, 0)
        complex_release_manager._update_version_files(new_version)

        # Verify all files were updated
        pyproject_data = toml.load(temp_git_repo / "pyproject.toml")
        assert pyproject_data["project"]["version"] == "3.0.0"

        with open(temp_git_repo / "package.json") as f:
            package_data = json.load(f)
        assert package_data["version"] == "3.0.0"

        version_content = (temp_git_repo / "VERSION").read_text()
        assert version_content.strip() == "3.0.0"

    def test_complex_conventional_commit_scenarios(self, complex_release_manager, temp_git_repo):
        """Test complex conventional commit scenarios."""
        # Add various types of commits
        commit_messages = [
            "feat(api): add GraphQL endpoint for user management",
            "fix(auth): resolve JWT token expiration issue",
            "perf(db): optimize query performance for large datasets",
            "docs: update API documentation with new endpoints",
            "test: add comprehensive integration tests for payment flow",
            "ci: update GitHub Actions workflow for multi-environment deployment",
            "feat!: remove deprecated REST API endpoints",
            """feat(payments): implement Stripe payment integration

Add comprehensive Stripe payment processing with webhook support
for handling payment events and subscription management.

BREAKING CHANGE: The old payment API has been completely replaced.
Users must migrate to the new Stripe-based payment system.""",
        ]

        # Add commits to repository
        for i, message in enumerate(commit_messages):
            test_file = temp_git_repo / f"feature_{i}.py"
            test_file.write_text(f"# Feature {i}\nprint('feature {i}')\n")
            subprocess.run(["git", "add", str(test_file)], cwd=temp_git_repo, check=True)
            subprocess.run(["git", "commit", "-m", message], cwd=temp_git_repo, check=True)

        # Get commits and analyze
        commits = complex_release_manager._get_commits_since_last_release()

        # Should have parsed multiple commit types
        commit_types = {commit.commit_type for commit in commits}
        assert CommitType.FEAT in commit_types
        assert CommitType.FIX in commit_types
        assert CommitType.PERF in commit_types
        assert CommitType.DOCS in commit_types
        assert CommitType.TEST in commit_types
        assert CommitType.CI in commit_types

        # Should detect breaking changes
        breaking_commits = [commit for commit in commits if commit.breaking_change]
        assert len(breaking_commits) >= 2  # feat! and BREAKING CHANGE commits

        # Test version determination
        current_version = SemanticVersion(2, 1, 0)
        next_version, release_type = complex_release_manager.determine_next_version(commits, current_version)

        # Should be major release due to breaking changes
        assert release_type == ReleaseType.MAJOR
        assert next_version.major == 3

    def test_complex_changelog_generation(self, complex_release_manager):
        """Test complex changelog generation with all commit types."""
        version = SemanticVersion(3, 0, 0)

        commits = [
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope="api",
                description="add GraphQL support",
                body="Comprehensive GraphQL API implementation",
                footer=None,
                breaking_change=False,
                commit_hash="abc123",
                author="Alice",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FEAT,
                scope="payments",
                description="remove old payment API",
                body="Replace legacy payment system",
                footer="BREAKING CHANGE: Old payment endpoints removed",
                breaking_change=True,
                commit_hash="def456",
                author="Bob",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FIX,
                scope="auth",
                description="fix session timeout",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="ghi789",
                author="Charlie",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.PERF,
                scope="db",
                description="optimize database queries",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="jkl012",
                author="Diana",
                timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.DOCS,
                scope=None,
                description="update README and API docs",
                body=None,
                footer=None,
                breaking_change=False,
                commit_hash="mno345",
                author="Eve",
                timestamp=datetime.now(timezone.utc)
            )
        ]

        changelog = complex_release_manager.generate_changelog(version, commits)

        # Verify changelog structure
        assert changelog.version == version
        assert len(changelog.changes) == 5  # All commit types represented
        assert len(changelog.breaking_changes) == 1

        # Generate formatted changelog
        formatted = complex_release_manager._format_changelog_entry(changelog)

        # Verify sections are present
        assert "## [3.0.0]" in formatted
        assert "### Added" in formatted  # Features
        assert "### Fixed" in formatted  # Fixes
        assert "### Performance" in formatted  # Performance improvements
        assert "### Documentation" in formatted  # Documentation
        assert "### BREAKING CHANGES" in formatted  # Breaking changes

        # Verify specific entries
        assert "**api**: add GraphQL support" in formatted
        assert "**auth**: fix session timeout" in formatted
        assert "**db**: optimize database queries" in formatted
        assert "Old payment endpoints removed" in formatted

    def test_complex_release_notes_generation(self, complex_release_manager):
        """Test complex release notes generation."""
        version = SemanticVersion(4, 0, 0)

        # Create complex changelog with many changes
        commits = [
            # Multiple features
            ConventionalCommit(
                commit_type=CommitType.FEAT, scope="api", description="add REST API v2",
                body=None, footer=None, breaking_change=False,
                commit_hash="1", author="Dev1", timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FEAT, scope="ui", description="redesign dashboard",
                body=None, footer=None, breaking_change=False,
                commit_hash="2", author="Dev2", timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FEAT, scope="auth", description="add SSO support",
                body=None, footer=None, breaking_change=False,
                commit_hash="3", author="Dev3", timestamp=datetime.now(timezone.utc)
            ),
            # Multiple fixes
            ConventionalCommit(
                commit_type=CommitType.FIX, scope="db", description="fix connection leaks",
                body=None, footer=None, breaking_change=False,
                commit_hash="4", author="Dev1", timestamp=datetime.now(timezone.utc)
            ),
            ConventionalCommit(
                commit_type=CommitType.FIX, scope="api", description="fix rate limiting",
                body=None, footer=None, breaking_change=False,
                commit_hash="5", author="Dev4", timestamp=datetime.now(timezone.utc)
            ),
            # Performance improvements
            ConventionalCommit(
                commit_type=CommitType.PERF, scope="cache", description="implement Redis caching",
                body=None, footer=None, breaking_change=False,
                commit_hash="6", author="Dev2", timestamp=datetime.now(timezone.utc)
            ),
            # Breaking change
            ConventionalCommit(
                commit_type=CommitType.FEAT, scope="api", description="replace JSON API with GraphQL",
                body=None, footer="BREAKING CHANGE: JSON API deprecated",
                breaking_change=True, commit_hash="7", author="Dev1", timestamp=datetime.now(timezone.utc)
            )
        ]

        changelog = ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc),
            changes={
                CommitType.FEAT: [commits[0], commits[1], commits[2], commits[6]],
                CommitType.FIX: [commits[3], commits[4]],
                CommitType.PERF: [commits[5]]
            },
            breaking_changes=[commits[6]]
        )

        release_notes = complex_release_manager.generate_release_notes(version, changelog)

        # Verify comprehensive release notes
        assert release_notes.version == version
        assert "enterprise-app v4.0.0" in release_notes.title
        assert "7 commits" in release_notes.summary

        # Verify highlights
        assert "4 new features" in release_notes.highlights
        assert "2 bug fixes" in release_notes.highlights
        assert "1 breaking changes" in release_notes.highlights

        # Verify breaking changes
        assert len(release_notes.breaking_changes) == 1
        assert "api: replace JSON API with GraphQL" in release_notes.breaking_changes[0]

        # Verify contributors
        expected_contributors = {"Dev1", "Dev2", "Dev3", "Dev4"}
        assert set(release_notes.contributors) == expected_contributors

        # Verify migration guide generated for breaking changes
        assert release_notes.migration_guide is not None
        assert "Migration Guide" in release_notes.migration_guide

    @patch.object(ReleaseManager, '_run_test_suite')
    @patch.object(ReleaseManager, '_run_security_scan')
    @patch.object(ReleaseManager, '_run_performance_test')
    async def test_comprehensive_quality_gates(
        self,
        mock_performance,
        mock_security,
        mock_tests,
        complex_release_manager
    ):
        """Test comprehensive quality gates execution."""
        # Mock all quality gates to pass
        mock_tests.return_value = True
        mock_security.return_value = True
        mock_performance.return_value = True

        progress = ReleaseProgress(
            release_id="quality-test",
            version="3.0.0",
            release_type=ReleaseType.MAJOR,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch="main"
        )

        await complex_release_manager._run_release_quality_gates(progress)

        # Verify all quality gates were executed
        expected_gates = ["lint", "tests", "security", "performance", "integration", "smoke"]
        for gate in expected_gates:
            if gate in ["tests", "security", "performance"]:
                # These have specific implementations
                continue
            assert gate in progress.validation_results

    async def test_rollback_scenario(self, complex_release_manager):
        """Test complex rollback scenario."""
        # Create a problematic release
        problematic_release = ReleaseProgress(
            release_id="problematic-release",
            version="3.0.0",
            release_type=ReleaseType.MAJOR,
            status=ReleaseStatus.RELEASED,
            start_time=time.time() - 3600,
            end_time=time.time() - 3600 + 1800,
            duration=1800.0,
            branch="main",
            tag="v3.0.0"
        )
        complex_release_manager.release_history.append(problematic_release)

        # Mock subprocess calls for rollback
        with patch('subprocess.run') as mock_run:
            # Mock git tag list
            mock_run.return_value = MagicMock(
                stdout="v3.0.0\nv2.1.0\nv2.0.0\nv1.9.0\n",
                returncode=0
            )

            with patch.object(complex_release_manager, 'create_hotfix') as mock_create_hotfix:
                mock_create_hotfix.return_value = "rollback-hotfix-123"

                await complex_release_manager.rollback_release(
                    "problematic-release",
                    "Critical security vulnerability found",
                    target_version="2.1.0"
                )

        # Verify rollback was processed
        assert problematic_release.status == ReleaseStatus.ROLLED_BACK
        assert problematic_release.rollback_reason == "Critical security vulnerability found"

        # Verify hotfix creation was called
        mock_create_hotfix.assert_called_once()

    def test_version_history_with_prereleases_and_builds(self, complex_release_manager):
        """Test version history with complex version patterns."""
        complex_tags = [
            "v1.0.0",
            "v1.0.1-alpha.1",
            "v1.0.1-alpha.2",
            "v1.0.1-beta.1",
            "v1.0.1",
            "v1.1.0-rc.1",
            "v1.1.0-rc.2",
            "v1.1.0",
            "v2.0.0-alpha.1+build.123",
            "v2.0.0-beta.1+build.456",
            "v2.0.0",
            "invalid-tag",  # Should be ignored
            "not-a-version"  # Should be ignored
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout='\n'.join(complex_tags) + '\n',
                returncode=0
            )

            versions = complex_release_manager.get_version_history()

        # Should parse valid semantic versions and ignore invalid ones
        assert len(versions) == 11  # All valid versions

        # Verify complex versions were parsed correctly
        version_strings = [str(v) for v in versions]
        assert "1.0.1-alpha.1" in version_strings
        assert "2.0.0-alpha.1+build.123" in version_strings
        assert "invalid-tag" not in version_strings
        assert "not-a-version" not in version_strings

        # Verify ordering
        assert versions[0] < versions[-1]  # Should be sorted

    def test_concurrent_release_management(self, complex_release_manager):
        """Test handling concurrent release operations."""
        # Simulate multiple active releases
        releases = []
        for i in range(3):
            progress = ReleaseProgress(
                release_id=f"concurrent-{i}",
                version=f"3.{i}.0",
                release_type=ReleaseType.MINOR,
                status=ReleaseStatus.IN_PROGRESS,
                start_time=time.time() - i * 100,
                branch=f"release/3.{i}.0"
            )
            releases.append(progress)
            complex_release_manager.active_releases[progress.release_id] = progress

        # Test listing concurrent releases
        active_releases = complex_release_manager.list_releases()
        assert len(active_releases) >= 3

        # Test getting status of each release
        for release in releases:
            status = complex_release_manager.get_release_status(release.release_id)
            assert status is not None
            assert status.status == ReleaseStatus.IN_PROGRESS

        # Test generating reports for concurrent releases
        for release in releases:
            report = complex_release_manager.generate_release_report(release.release_id)
            assert report["release_id"] == release.release_id
            assert report["status"] == "in_progress"


def test_create_example_release_config():
    """Test example release configuration creation."""
    config = create_example_release_config()

    assert config.project_name == "workspace-qdrant-mcp"
    assert config.repository_url == "https://github.com/ChrisGVE/workspace-qdrant-mcp"
    assert config.default_branch == "main"
    assert "main" in config.release_branches
    assert "release/*" in config.release_branches
    assert "hotfix/*" in config.release_branches
    assert "pyproject.toml" in config.version_files
    assert config.changelog_file == "CHANGELOG.md"
    assert "tests" in config.quality_gates
    assert "security" in config.quality_gates
    assert config.auto_tag is True
    assert config.auto_publish is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])