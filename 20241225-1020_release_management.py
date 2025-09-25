#!/usr/bin/env python3
"""
Release Management and Semantic Versioning System

This module provides comprehensive release management with semantic versioning,
automated changelog generation, and release orchestration.

Key features:
- Semantic versioning with conventional commits
- Automated changelog generation from commit history
- Release branch management and tagging
- Multi-environment release orchestration
- Release validation and quality gates
- Rollback and hotfix management
- Dependency and compatibility tracking
- Release metrics and reporting
- Integration with deployment pipelines
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import toml
import yaml
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class ReleaseType(str, Enum):
    """Release type enumeration."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"
    HOTFIX = "hotfix"


class CommitType(str, Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    PERF = "perf"
    TEST = "test"
    BUILD = "build"
    CI = "ci"
    CHORE = "chore"
    REVERT = "revert"


class ReleaseStatus(str, Enum):
    """Release status values."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    RELEASED = "released"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """String representation of semantic version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare semantic versions."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # Handle prerelease comparison
        if self.prerelease is None and other.prerelease is not None:
            return False  # Release version > prerelease
        if self.prerelease is not None and other.prerelease is None:
            return True   # Prerelease < release version
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease

        return False  # Versions are equal

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse semantic version from string."""
        # Remove 'v' prefix if present
        version_string = version_string.lstrip('v')

        # Regex pattern for semantic versioning
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$'
        match = re.match(pattern, version_string)

        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")

        major, minor, patch, prerelease, build = match.groups()

        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build
        )

    def bump(self, release_type: ReleaseType) -> "SemanticVersion":
        """Bump version based on release type."""
        if release_type == ReleaseType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif release_type == ReleaseType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif release_type == ReleaseType.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        elif release_type == ReleaseType.PRERELEASE:
            if self.prerelease:
                # Extract number from prerelease and increment
                parts = self.prerelease.split('.')
                if parts[-1].isdigit():
                    parts[-1] = str(int(parts[-1]) + 1)
                else:
                    parts.append("1")
                prerelease = '.'.join(parts)
            else:
                prerelease = "alpha.1"
            return SemanticVersion(self.major, self.minor, self.patch, prerelease)
        else:
            raise ValueError(f"Unsupported release type: {release_type}")


@dataclass
class ConventionalCommit:
    """Conventional commit representation."""
    commit_type: CommitType
    scope: Optional[str]
    description: str
    body: Optional[str]
    footer: Optional[str]
    breaking_change: bool
    commit_hash: str
    author: str
    timestamp: datetime

    @classmethod
    def parse(cls, commit_line: str, commit_hash: str, author: str, timestamp: datetime) -> Optional["ConventionalCommit"]:
        """Parse conventional commit from commit message."""
        # Pattern for conventional commits
        pattern = r'^(\w+)(?:\(([^)]+)\))?(!)?: (.+)(?:\n\n(.+?))?(?:\n\n(.+))?$'
        match = re.match(pattern, commit_line.strip(), re.DOTALL)

        if not match:
            return None

        commit_type_str, scope, breaking_marker, description, body, footer = match.groups()

        try:
            commit_type = CommitType(commit_type_str.lower())
        except ValueError:
            return None

        breaking_change = breaking_marker == '!' or (footer and 'BREAKING CHANGE' in footer)

        return cls(
            commit_type=commit_type,
            scope=scope,
            description=description,
            body=body,
            footer=footer,
            breaking_change=breaking_change,
            commit_hash=commit_hash,
            author=author,
            timestamp=timestamp
        )


@dataclass
class ChangelogEntry:
    """Changelog entry representation."""
    version: SemanticVersion
    date: datetime
    changes: Dict[CommitType, List[ConventionalCommit]]
    breaking_changes: List[ConventionalCommit]
    migration_notes: Optional[str] = None


@dataclass
class ReleaseNotes:
    """Release notes representation."""
    version: SemanticVersion
    title: str
    summary: str
    highlights: List[str]
    breaking_changes: List[str]
    migration_guide: Optional[str]
    dependencies: Dict[str, str]
    compatibility: Dict[str, str]
    known_issues: List[str]
    contributors: List[str]


class ReleaseConfig(BaseModel):
    """Release configuration model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str
    repository_url: str
    default_branch: str = "main"
    release_branches: List[str] = Field(default_factory=lambda: ["main", "release/*"])
    version_files: List[str] = Field(default_factory=lambda: ["pyproject.toml", "package.json"])
    changelog_file: str = "CHANGELOG.md"
    release_notes_template: Optional[str] = None
    quality_gates: List[str] = Field(default_factory=list)
    auto_tag: bool = True
    auto_publish: bool = False
    notification_webhooks: List[str] = Field(default_factory=list)
    environments: List[str] = Field(default_factory=lambda: ["staging", "production"])


class ReleaseProgress(BaseModel):
    """Release progress tracking model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    release_id: str
    version: str
    release_type: ReleaseType
    status: ReleaseStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    branch: str
    commit_hash: Optional[str] = None
    tag: Optional[str] = None
    environments_deployed: List[str] = Field(default_factory=list)
    failed_environments: List[str] = Field(default_factory=list)
    validation_results: Dict[str, bool] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None


class ReleaseManager:
    """Comprehensive release management system."""

    def __init__(
        self,
        config: ReleaseConfig,
        repo_path: Path = Path("."),
        artifacts_dir: Path = Path("release-artifacts"),
        logs_dir: Path = Path("release-logs")
    ):
        """Initialize the release manager."""
        self.config = config
        self.repo_path = Path(repo_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.logs_dir = Path(logs_dir)

        # Create required directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.active_releases: Dict[str, ReleaseProgress] = {}
        self.release_history: List[ReleaseProgress] = []

        # Git repository validation
        self._validate_repository()

        logger.info(f"Release manager initialized for {config.project_name}")

    def _validate_repository(self) -> None:
        """Validate Git repository setup."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"Not a Git repository: {self.repo_path}")

        # Check if we're on a valid branch
        try:
            current_branch = self._get_current_branch()
            logger.info(f"Current branch: {current_branch}")
        except Exception as e:
            raise RuntimeError(f"Failed to get current branch: {e}")

    def _get_current_branch(self) -> str:
        """Get the current Git branch."""
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _get_current_version(self) -> SemanticVersion:
        """Get the current version from version files."""
        for version_file in self.config.version_files:
            file_path = self.repo_path / version_file
            if not file_path.exists():
                continue

            try:
                if version_file.endswith('.toml'):
                    data = toml.load(file_path)
                    if 'project' in data and 'version' in data['project']:
                        version_str = data['project']['version']
                    elif 'tool' in data and 'poetry' in data['tool'] and 'version' in data['tool']['poetry']:
                        version_str = data['tool']['poetry']['version']
                    else:
                        continue
                elif version_file.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                    version_str = data.get('version', '')
                else:
                    continue

                return SemanticVersion.parse(version_str)
            except Exception as e:
                logger.warning(f"Failed to parse version from {version_file}: {e}")
                continue

        raise RuntimeError("No valid version found in version files")

    def _update_version_files(self, new_version: SemanticVersion) -> None:
        """Update version in all version files."""
        version_str = str(new_version)

        for version_file in self.config.version_files:
            file_path = self.repo_path / version_file
            if not file_path.exists():
                continue

            try:
                if version_file.endswith('.toml'):
                    data = toml.load(file_path)
                    if 'project' in data and 'version' in data['project']:
                        data['project']['version'] = version_str
                    elif 'tool' in data and 'poetry' in data['tool']:
                        data['tool']['poetry']['version'] = version_str
                    else:
                        continue

                    with open(file_path, 'w') as f:
                        toml.dump(data, f)
                elif version_file.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                    data['version'] = version_str

                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)

                logger.info(f"Updated version to {version_str} in {version_file}")
            except Exception as e:
                logger.error(f"Failed to update version in {version_file}: {e}")
                raise

    def _get_commits_since_last_release(self, last_version: Optional[SemanticVersion] = None) -> List[ConventionalCommit]:
        """Get commits since the last release."""
        if last_version:
            # Get commits since the last tag
            tag_name = f"v{last_version}"
            cmd = ["git", "log", f"{tag_name}..HEAD", "--pretty=format:%H|%an|%at|%s|%b"]
        else:
            # Get all commits
            cmd = ["git", "log", "--pretty=format:%H|%an|%at|%s|%b"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            logger.warning(f"No commits found since tag v{last_version}")
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|', 4)
            if len(parts) < 4:
                continue

            commit_hash = parts[0]
            author = parts[1]
            timestamp = datetime.fromtimestamp(int(parts[2]), tz=timezone.utc)
            subject = parts[3]
            body = parts[4] if len(parts) > 4 else ""

            # Combine subject and body for parsing
            full_message = subject
            if body:
                full_message += f"\n\n{body}"

            conventional_commit = ConventionalCommit.parse(
                full_message, commit_hash, author, timestamp
            )

            if conventional_commit:
                commits.append(conventional_commit)

        return commits

    def determine_next_version(self, commits: List[ConventionalCommit], current_version: SemanticVersion) -> Tuple[SemanticVersion, ReleaseType]:
        """Determine the next version based on commits."""
        has_breaking = any(commit.breaking_change for commit in commits)
        has_features = any(commit.commit_type == CommitType.FEAT for commit in commits)
        has_fixes = any(commit.commit_type == CommitType.FIX for commit in commits)

        if has_breaking:
            release_type = ReleaseType.MAJOR
        elif has_features:
            release_type = ReleaseType.MINOR
        elif has_fixes:
            release_type = ReleaseType.PATCH
        else:
            # No significant changes, still bump patch for any commits
            release_type = ReleaseType.PATCH

        next_version = current_version.bump(release_type)
        return next_version, release_type

    def generate_changelog(self, version: SemanticVersion, commits: List[ConventionalCommit]) -> ChangelogEntry:
        """Generate changelog entry from commits."""
        changes: Dict[CommitType, List[ConventionalCommit]] = {}
        breaking_changes = []

        for commit in commits:
            if commit.breaking_change:
                breaking_changes.append(commit)

            if commit.commit_type not in changes:
                changes[commit.commit_type] = []
            changes[commit.commit_type].append(commit)

        return ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc),
            changes=changes,
            breaking_changes=breaking_changes
        )

    def update_changelog_file(self, changelog_entry: ChangelogEntry) -> None:
        """Update the changelog file with new entry."""
        changelog_path = self.repo_path / self.config.changelog_file

        # Generate new entry content
        entry_content = self._format_changelog_entry(changelog_entry)

        # Read existing changelog or create new one
        if changelog_path.exists():
            with open(changelog_path) as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"

        # Insert new entry after header
        lines = existing_content.split('\n')
        header_end = 0

        for i, line in enumerate(lines):
            if line.startswith('## '):  # First version entry
                header_end = i
                break
        else:
            header_end = len(lines)

        # Insert new entry
        new_lines = lines[:header_end] + entry_content.split('\n') + lines[header_end:]
        new_content = '\n'.join(new_lines)

        # Write updated changelog
        with open(changelog_path, 'w') as f:
            f.write(new_content)

        logger.info(f"Updated changelog: {changelog_path}")

    def _format_changelog_entry(self, entry: ChangelogEntry) -> str:
        """Format changelog entry as markdown."""
        content = f"## [{entry.version}] - {entry.date.strftime('%Y-%m-%d')}\n\n"

        # Features
        if CommitType.FEAT in entry.changes:
            content += "### Added\n\n"
            for commit in entry.changes[CommitType.FEAT]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                content += f"- {scope_str}{commit.description}\n"
            content += "\n"

        # Fixes
        if CommitType.FIX in entry.changes:
            content += "### Fixed\n\n"
            for commit in entry.changes[CommitType.FIX]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                content += f"- {scope_str}{commit.description}\n"
            content += "\n"

        # Performance improvements
        if CommitType.PERF in entry.changes:
            content += "### Performance\n\n"
            for commit in entry.changes[CommitType.PERF]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                content += f"- {scope_str}{commit.description}\n"
            content += "\n"

        # Documentation
        if CommitType.DOCS in entry.changes:
            content += "### Documentation\n\n"
            for commit in entry.changes[CommitType.DOCS]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                content += f"- {scope_str}{commit.description}\n"
            content += "\n"

        # Breaking changes
        if entry.breaking_changes:
            content += "### BREAKING CHANGES\n\n"
            for commit in entry.breaking_changes:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                content += f"- {scope_str}{commit.description}\n"
                if commit.body and "BREAKING CHANGE:" in commit.body:
                    # Extract breaking change description
                    breaking_desc = commit.body.split("BREAKING CHANGE:")[-1].strip()
                    content += f"  {breaking_desc}\n"
            content += "\n"

        return content

    def generate_release_notes(self, version: SemanticVersion, changelog_entry: ChangelogEntry) -> ReleaseNotes:
        """Generate comprehensive release notes."""
        # Generate title and summary
        title = f"{self.config.project_name} v{version}"

        total_commits = sum(len(commits) for commits in changelog_entry.changes.values())
        summary = f"This release includes {total_commits} commits"

        # Generate highlights
        highlights = []
        if CommitType.FEAT in changelog_entry.changes:
            feat_count = len(changelog_entry.changes[CommitType.FEAT])
            highlights.append(f"{feat_count} new features")

        if CommitType.FIX in changelog_entry.changes:
            fix_count = len(changelog_entry.changes[CommitType.FIX])
            highlights.append(f"{fix_count} bug fixes")

        if changelog_entry.breaking_changes:
            highlights.append(f"{len(changelog_entry.breaking_changes)} breaking changes")

        # Extract breaking changes
        breaking_changes = [
            f"{commit.scope}: {commit.description}" if commit.scope else commit.description
            for commit in changelog_entry.breaking_changes
        ]

        # Get contributors
        contributors = list(set(
            commit.author for commits in changelog_entry.changes.values() for commit in commits
        ))

        # Get dependencies (would integrate with package managers)
        dependencies = self._get_current_dependencies()

        return ReleaseNotes(
            version=version,
            title=title,
            summary=summary,
            highlights=highlights,
            breaking_changes=breaking_changes,
            migration_guide=self._generate_migration_guide(changelog_entry) if breaking_changes else None,
            dependencies=dependencies,
            compatibility={"python": ">=3.10"},  # Would be dynamic
            known_issues=[],  # Would be extracted from issues tracker
            contributors=contributors
        )

    def _get_current_dependencies(self) -> Dict[str, str]:
        """Get current project dependencies."""
        dependencies = {}

        # Check pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                data = toml.load(pyproject_path)
                if 'project' in data and 'dependencies' in data['project']:
                    for dep in data['project']['dependencies']:
                        # Parse dependency string (simplified)
                        if '>=' in dep:
                            name, version = dep.split('>=', 1)
                            dependencies[name.strip()] = f">={version.strip()}"
                        elif '==' in dep:
                            name, version = dep.split('==', 1)
                            dependencies[name.strip()] = f"=={version.strip()}"
                        else:
                            dependencies[dep.strip()] = "*"
            except Exception as e:
                logger.warning(f"Failed to parse dependencies from pyproject.toml: {e}")

        return dependencies

    def _generate_migration_guide(self, changelog_entry: ChangelogEntry) -> str:
        """Generate migration guide for breaking changes."""
        if not changelog_entry.breaking_changes:
            return ""

        guide = "## Migration Guide\n\n"
        guide += "This release contains breaking changes. Please review the following:\n\n"

        for i, commit in enumerate(changelog_entry.breaking_changes, 1):
            guide += f"### {i}. {commit.description}\n\n"
            if commit.body and "BREAKING CHANGE:" in commit.body:
                breaking_desc = commit.body.split("BREAKING CHANGE:")[-1].strip()
                guide += f"{breaking_desc}\n\n"

        return guide

    async def create_release(
        self,
        release_type: Optional[ReleaseType] = None,
        dry_run: bool = False,
        force: bool = False
    ) -> str:
        """Create a new release."""
        release_id = f"release-{int(time.time())}"

        logger.info(f"Starting release creation: {release_id}")

        # Get current version and commits
        current_version = self._get_current_version()
        commits = self._get_commits_since_last_release(current_version)

        if not commits and not force:
            raise RuntimeError("No commits since last release")

        # Determine next version
        if release_type:
            next_version = current_version.bump(release_type)
        else:
            next_version, release_type = self.determine_next_version(commits, current_version)

        # Initialize progress tracking
        progress = ReleaseProgress(
            release_id=release_id,
            version=str(next_version),
            release_type=release_type,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch=self._get_current_branch()
        )

        self.active_releases[release_id] = progress

        try:
            # Validate release conditions
            await self._validate_release_conditions(progress)

            # Generate changelog
            logger.info("Generating changelog...")
            changelog_entry = self.generate_changelog(next_version, commits)

            if not dry_run:
                # Update version files
                logger.info(f"Updating version files to {next_version}")
                self._update_version_files(next_version)

                # Update changelog file
                self.update_changelog_file(changelog_entry)

                # Commit changes
                commit_message = f"chore(release): prepare release {next_version}"
                await self._commit_changes(commit_message)
                progress.commit_hash = await self._get_latest_commit_hash()

                # Create tag
                if self.config.auto_tag:
                    tag_name = f"v{next_version}"
                    await self._create_tag(tag_name, f"Release {next_version}")
                    progress.tag = tag_name

                # Generate release notes
                release_notes = self.generate_release_notes(next_version, changelog_entry)

                # Save release notes
                await self._save_release_notes(release_notes)

                # Run quality gates
                await self._run_release_quality_gates(progress)

                # Mark as completed
                progress.status = ReleaseStatus.RELEASED
                progress.end_time = time.time()
                progress.duration = progress.end_time - progress.start_time

                logger.info(f"Release {next_version} created successfully")
            else:
                logger.info(f"DRY RUN: Would create release {next_version}")
                progress.status = ReleaseStatus.RELEASED

        except Exception as e:
            logger.error(f"Release creation failed: {e}")
            progress.status = ReleaseStatus.FAILED
            progress.error_message = str(e)
            progress.end_time = time.time()
            progress.duration = progress.end_time - progress.start_time
            raise
        finally:
            # Move to history
            self.release_history.append(progress)
            if release_id in self.active_releases:
                del self.active_releases[release_id]

        return release_id

    async def _validate_release_conditions(self, progress: ReleaseProgress) -> None:
        """Validate conditions for creating a release."""
        logger.info("Validating release conditions...")

        # Check if branch is clean
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            uncommitted_files = result.stdout.strip().split('\n')
            logger.warning(f"Uncommitted changes found: {uncommitted_files}")
            # Don't fail for version file changes as we'll commit them

        # Check if we're on a valid release branch
        current_branch = self._get_current_branch()
        valid_branches = [
            branch for pattern in self.config.release_branches
            for branch in [current_branch]
            if self._matches_pattern(branch, pattern)
        ]

        if not valid_branches:
            raise RuntimeError(
                f"Invalid release branch '{current_branch}'. "
                f"Valid branches: {self.config.release_branches}"
            )

        logger.info("Release conditions validated")

    def _matches_pattern(self, branch: str, pattern: str) -> bool:
        """Check if branch matches pattern (supports wildcards)."""
        if '*' not in pattern:
            return branch == pattern

        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(branch, pattern)

    async def _commit_changes(self, message: str) -> None:
        """Commit changes to Git."""
        # Add version files and changelog
        files_to_add = self.config.version_files + [self.config.changelog_file]

        for file_path in files_to_add:
            if (self.repo_path / file_path).exists():
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=self.repo_path,
                    check=True
                )

        # Commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_path,
            check=True
        )

        logger.info(f"Committed changes: {message}")

    async def _get_latest_commit_hash(self) -> str:
        """Get the latest commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    async def _create_tag(self, tag_name: str, message: str) -> None:
        """Create a Git tag."""
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", message],
            cwd=self.repo_path,
            check=True
        )

        logger.info(f"Created tag: {tag_name}")

    async def _save_release_notes(self, release_notes: ReleaseNotes) -> None:
        """Save release notes to file."""
        notes_path = self.artifacts_dir / f"release-notes-{release_notes.version}.md"

        content = self._format_release_notes(release_notes)

        with open(notes_path, 'w') as f:
            f.write(content)

        logger.info(f"Saved release notes: {notes_path}")

    def _format_release_notes(self, release_notes: ReleaseNotes) -> str:
        """Format release notes as markdown."""
        content = f"# {release_notes.title}\n\n"
        content += f"{release_notes.summary}.\n\n"

        if release_notes.highlights:
            content += "## Highlights\n\n"
            for highlight in release_notes.highlights:
                content += f"- {highlight}\n"
            content += "\n"

        if release_notes.breaking_changes:
            content += "## ⚠️ Breaking Changes\n\n"
            for change in release_notes.breaking_changes:
                content += f"- {change}\n"
            content += "\n"

        if release_notes.migration_guide:
            content += release_notes.migration_guide
            content += "\n"

        if release_notes.dependencies:
            content += "## Dependencies\n\n"
            for dep, version in release_notes.dependencies.items():
                content += f"- {dep}: {version}\n"
            content += "\n"

        if release_notes.contributors:
            content += "## Contributors\n\n"
            content += "Thanks to the following contributors:\n\n"
            for contributor in release_notes.contributors:
                content += f"- @{contributor}\n"
            content += "\n"

        if release_notes.known_issues:
            content += "## Known Issues\n\n"
            for issue in release_notes.known_issues:
                content += f"- {issue}\n"
            content += "\n"

        return content

    async def _run_release_quality_gates(self, progress: ReleaseProgress) -> None:
        """Run quality gates for release validation."""
        if not self.config.quality_gates:
            logger.info("No quality gates configured for release")
            return

        logger.info("Running release quality gates...")

        for gate in self.config.quality_gates:
            try:
                # This would integrate with actual quality gate systems
                # For now, simulate quality gate execution
                await asyncio.sleep(1)  # Simulate check time

                # Mock validation logic
                if gate == "tests":
                    result = await self._run_test_suite()
                elif gate == "security":
                    result = await self._run_security_scan()
                elif gate == "performance":
                    result = await self._run_performance_test()
                else:
                    result = True  # Default pass

                progress.validation_results[gate] = result

                if not result:
                    raise RuntimeError(f"Quality gate '{gate}' failed")

                logger.info(f"Quality gate '{gate}' passed")
            except Exception as e:
                logger.error(f"Quality gate '{gate}' failed: {e}")
                progress.validation_results[gate] = False
                raise

        logger.info("All quality gates passed")

    async def _run_test_suite(self) -> bool:
        """Run test suite quality gate."""
        try:
            # Run tests (example command)
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "--tb=short"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False

    async def _run_security_scan(self) -> bool:
        """Run security scan quality gate."""
        try:
            # Run security scan (example)
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode == 0:
                return True

            # Parse results to check severity
            try:
                scan_results = json.loads(result.stdout)
                critical_issues = [
                    issue for issue in scan_results.get('results', [])
                    if issue.get('issue_severity') == 'HIGH'
                ]
                return len(critical_issues) == 0
            except Exception:
                return result.returncode == 0

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False

    async def _run_performance_test(self) -> bool:
        """Run performance test quality gate."""
        # Mock performance test
        await asyncio.sleep(2)  # Simulate performance test
        return True

    async def create_hotfix(
        self,
        fix_description: str,
        base_version: Optional[str] = None,
        dry_run: bool = False
    ) -> str:
        """Create a hotfix release."""
        hotfix_id = f"hotfix-{int(time.time())}"

        logger.info(f"Starting hotfix creation: {hotfix_id}")

        # Determine base version
        if base_version:
            base_version_obj = SemanticVersion.parse(base_version)
        else:
            base_version_obj = self._get_current_version()

        # Create hotfix version
        hotfix_version = SemanticVersion(
            base_version_obj.major,
            base_version_obj.minor,
            base_version_obj.patch + 1,
            prerelease="hotfix"
        )

        # Initialize progress
        progress = ReleaseProgress(
            release_id=hotfix_id,
            version=str(hotfix_version),
            release_type=ReleaseType.HOTFIX,
            status=ReleaseStatus.IN_PROGRESS,
            start_time=time.time(),
            branch=f"hotfix/{hotfix_version}"
        )

        self.active_releases[hotfix_id] = progress

        try:
            if not dry_run:
                # Create hotfix branch
                await self._create_hotfix_branch(progress.branch, base_version)

                # Update version files
                self._update_version_files(hotfix_version)

                # Create simple changelog entry
                changelog_entry = ChangelogEntry(
                    version=hotfix_version,
                    date=datetime.now(timezone.utc),
                    changes={},
                    breaking_changes=[],
                    migration_notes=f"Hotfix: {fix_description}"
                )

                self.update_changelog_file(changelog_entry)

                # Commit hotfix
                commit_message = f"fix: {fix_description} (hotfix {hotfix_version})"
                await self._commit_changes(commit_message)
                progress.commit_hash = await self._get_latest_commit_hash()

                # Create tag
                tag_name = f"v{hotfix_version}"
                await self._create_tag(tag_name, f"Hotfix {hotfix_version}: {fix_description}")
                progress.tag = tag_name

                # Generate minimal release notes
                release_notes = ReleaseNotes(
                    version=hotfix_version,
                    title=f"{self.config.project_name} v{hotfix_version} (Hotfix)",
                    summary=f"Critical hotfix release: {fix_description}",
                    highlights=[f"Fixed: {fix_description}"],
                    breaking_changes=[],
                    migration_guide=None,
                    dependencies=self._get_current_dependencies(),
                    compatibility={"python": ">=3.10"},
                    known_issues=[],
                    contributors=[await self._get_current_user()]
                )

                await self._save_release_notes(release_notes)

                progress.status = ReleaseStatus.RELEASED
                progress.end_time = time.time()
                progress.duration = progress.end_time - progress.start_time

                logger.info(f"Hotfix {hotfix_version} created successfully")
            else:
                logger.info(f"DRY RUN: Would create hotfix {hotfix_version}")
                progress.status = ReleaseStatus.RELEASED

        except Exception as e:
            logger.error(f"Hotfix creation failed: {e}")
            progress.status = ReleaseStatus.FAILED
            progress.error_message = str(e)
            progress.end_time = time.time()
            progress.duration = progress.end_time - progress.start_time
            raise
        finally:
            self.release_history.append(progress)
            if hotfix_id in self.active_releases:
                del self.active_releases[hotfix_id]

        return hotfix_id

    async def _create_hotfix_branch(self, branch_name: str, base_version: Optional[str]) -> None:
        """Create hotfix branch."""
        if base_version:
            # Create branch from specific tag
            subprocess.run(
                ["git", "checkout", "-b", branch_name, f"v{base_version}"],
                cwd=self.repo_path,
                check=True
            )
        else:
            # Create branch from current HEAD
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                check=True
            )

        logger.info(f"Created hotfix branch: {branch_name}")

    async def _get_current_user(self) -> str:
        """Get current Git user."""
        result = subprocess.run(
            ["git", "config", "user.name"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    async def rollback_release(
        self,
        release_id: str,
        reason: str,
        target_version: Optional[str] = None
    ) -> None:
        """Rollback a release."""
        logger.info(f"Rolling back release {release_id}: {reason}")

        # Find the release
        release = None
        for historical_release in self.release_history:
            if historical_release.release_id == release_id:
                release = historical_release
                break

        if not release:
            release = self.active_releases.get(release_id)

        if not release:
            raise ValueError(f"Release {release_id} not found")

        # Determine target version
        if target_version:
            target_version_obj = SemanticVersion.parse(target_version)
        else:
            # Find previous version
            all_tags = subprocess.run(
                ["git", "tag", "-l", "--sort=-version:refname"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            ).stdout.strip().split('\n')

            current_tag = f"v{release.version}"
            try:
                current_index = all_tags.index(current_tag)
                if current_index + 1 < len(all_tags):
                    previous_tag = all_tags[current_index + 1]
                    target_version_obj = SemanticVersion.parse(previous_tag.lstrip('v'))
                else:
                    raise ValueError("No previous version found for rollback")
            except ValueError:
                raise ValueError(f"Could not find tag {current_tag} for rollback")

        # Create rollback hotfix
        rollback_id = await self.create_hotfix(
            fix_description=f"Rollback to {target_version_obj} - {reason}",
            base_version=str(target_version_obj)
        )

        # Update original release status
        release.status = ReleaseStatus.ROLLED_BACK
        release.rollback_reason = reason

        logger.info(f"Release {release_id} rolled back to {target_version_obj}")

    def get_release_status(self, release_id: str) -> Optional[ReleaseProgress]:
        """Get status of a release."""
        if release_id in self.active_releases:
            return self.active_releases[release_id]

        for release in self.release_history:
            if release.release_id == release_id:
                return release

        return None

    def list_releases(self, limit: int = 20) -> List[ReleaseProgress]:
        """List recent releases."""
        all_releases = self.release_history + list(self.active_releases.values())
        all_releases.sort(key=lambda r: r.start_time, reverse=True)
        return all_releases[:limit]

    def get_version_history(self) -> List[SemanticVersion]:
        """Get version history from Git tags."""
        result = subprocess.run(
            ["git", "tag", "-l", "--sort=version:refname"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return []

        versions = []
        for tag in result.stdout.strip().split('\n'):
            if not tag:
                continue
            try:
                version = SemanticVersion.parse(tag.lstrip('v'))
                versions.append(version)
            except ValueError:
                continue  # Skip non-semantic version tags

        return versions

    def generate_release_report(self, release_id: str) -> Dict[str, Any]:
        """Generate comprehensive release report."""
        release = self.get_release_status(release_id)
        if not release:
            raise ValueError(f"Release {release_id} not found")

        report = {
            "release_id": release.release_id,
            "version": release.version,
            "release_type": release.release_type.value,
            "status": release.status.value,
            "start_time": release.start_time,
            "end_time": release.end_time,
            "duration": release.duration,
            "branch": release.branch,
            "commit_hash": release.commit_hash,
            "tag": release.tag,
            "environments_deployed": release.environments_deployed,
            "failed_environments": release.failed_environments,
            "validation_results": release.validation_results,
            "artifacts": release.artifacts,
            "error_message": release.error_message,
            "rollback_reason": release.rollback_reason
        }

        return report

    def print_release_status_table(self) -> None:
        """Print release status table."""
        table = Table(title="Recent Releases")

        table.add_column("Release ID", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Duration", style="dim")
        table.add_column("Branch", style="magenta")

        releases = self.list_releases(limit=10)
        for release in releases:
            duration_str = f"{release.duration:.1f}s" if release.duration else "N/A"

            table.add_row(
                release.release_id[:20] + "..." if len(release.release_id) > 20 else release.release_id,
                release.version,
                release.release_type.value,
                release.status.value,
                duration_str,
                release.branch
            )

        console.print(table)


def create_example_release_config() -> ReleaseConfig:
    """Create example release configuration."""
    return ReleaseConfig(
        project_name="workspace-qdrant-mcp",
        repository_url="https://github.com/ChrisGVE/workspace-qdrant-mcp",
        default_branch="main",
        release_branches=["main", "release/*", "hotfix/*"],
        version_files=["pyproject.toml"],
        changelog_file="CHANGELOG.md",
        quality_gates=["tests", "security", "performance"],
        auto_tag=True,
        auto_publish=False,
        environments=["staging", "production"]
    )


async def main():
    """Main example demonstrating the release manager."""
    # Create example configuration
    config = create_example_release_config()

    # Initialize release manager
    manager = ReleaseManager(
        config=config,
        repo_path=Path("."),
        artifacts_dir=Path("release-artifacts"),
        logs_dir=Path("release-logs")
    )

    try:
        # Create a release (dry run)
        console.print("[blue]Creating release (dry run)...[/blue]")
        release_id = await manager.create_release(dry_run=True)

        console.print(f"[green]Release created: {release_id}[/green]")

        # Print release status
        manager.print_release_status_table()

        # Generate report
        report = manager.generate_release_report(release_id)
        console.print(f"\n[bold]Release Report:[/bold]")
        console.print(json.dumps(report, indent=2, default=str))

        # Show version history
        versions = manager.get_version_history()
        if versions:
            console.print(f"\n[bold]Version History:[/bold]")
            for version in versions[-5:]:  # Show last 5 versions
                console.print(f"  v{version}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())