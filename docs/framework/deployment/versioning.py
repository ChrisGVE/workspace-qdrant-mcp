"""Comprehensive version management system for documentation deployments."""

import logging
import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import subprocess
import shutil

logger = logging.getLogger(__name__)


class VersioningStrategy(Enum):
    """Version numbering strategies."""
    SEMANTIC = "semantic"  # x.y.z format
    TIMESTAMP = "timestamp"  # YYYYMMDD-HHMMSS format
    SEQUENTIAL = "sequential"  # v1, v2, v3, etc.
    GIT_HASH = "git_hash"  # Git commit hash
    CUSTOM = "custom"  # User-defined format


class VersionStatus(Enum):
    """Version deployment status."""
    DRAFT = "draft"
    STAGED = "staged"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class Version:
    """Represents a documentation version."""

    version_string: str
    strategy: VersioningStrategy
    status: VersionStatus = VersionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None

    # Version metadata
    title: Optional[str] = None
    description: Optional[str] = None
    changelog: Optional[str] = None
    author: Optional[str] = None

    # Technical metadata
    source_hash: Optional[str] = None
    build_hash: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    # Deployment information
    deployment_id: Optional[str] = None
    deployment_path: Optional[Path] = None
    size_bytes: int = 0
    file_count: int = 0

    # Dependencies and compatibility
    dependencies: Dict[str, str] = field(default_factory=dict)
    compatibility: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.deployment_path, str):
            self.deployment_path = Path(self.deployment_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)

        # Convert datetime objects to ISO strings
        for field_name in ['created_at', 'deployed_at', 'deprecated_at']:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()

        # Convert Path objects to strings
        if data['deployment_path']:
            data['deployment_path'] = str(data['deployment_path'])

        # Convert enums to values
        data['strategy'] = data['strategy'].value
        data['status'] = data['status'].value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Version':
        """Create from dictionary."""
        data = data.copy()

        # Convert ISO strings to datetime objects
        for field_name in ['created_at', 'deployed_at', 'deprecated_at']:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])

        # Convert strings to Path objects
        if data.get('deployment_path'):
            data['deployment_path'] = Path(data['deployment_path'])

        # Convert strings to enums
        if 'strategy' in data:
            data['strategy'] = VersioningStrategy(data['strategy'])
        if 'status' in data:
            data['status'] = VersionStatus(data['status'])

        return cls(**data)

    @property
    def is_active(self) -> bool:
        """Check if version is currently active."""
        return self.status == VersionStatus.ACTIVE

    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.status == VersionStatus.DEPRECATED

    @property
    def age_days(self) -> int:
        """Get age of version in days."""
        return (datetime.now() - self.created_at).days

    def compare_version(self, other: 'Version') -> int:
        """Compare versions (-1: less, 0: equal, 1: greater)."""
        if self.strategy != other.strategy:
            # Fall back to timestamp comparison
            if self.created_at < other.created_at:
                return -1
            elif self.created_at > other.created_at:
                return 1
            return 0

        if self.strategy == VersioningStrategy.SEMANTIC:
            return self._compare_semantic(self.version_string, other.version_string)
        elif self.strategy == VersioningStrategy.SEQUENTIAL:
            return self._compare_sequential(self.version_string, other.version_string)
        elif self.strategy == VersioningStrategy.TIMESTAMP:
            return self._compare_timestamp(self.version_string, other.version_string)
        else:
            # For git hash and custom, use creation time
            return self._compare_datetime(self.created_at, other.created_at)

    def _compare_semantic(self, v1: str, v2: str) -> int:
        """Compare semantic versions."""
        try:
            # Remove 'v' prefix if present
            v1 = v1.lstrip('v')
            v2 = v2.lstrip('v')

            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]

            # Pad to same length
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))

            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            return 0

        except ValueError:
            # Invalid semantic version, fall back to string comparison
            return self._compare_string(v1, v2)

    def _compare_sequential(self, v1: str, v2: str) -> int:
        """Compare sequential versions."""
        try:
            # Extract numbers from version strings
            num1 = int(re.search(r'\d+', v1).group())
            num2 = int(re.search(r'\d+', v2).group())

            if num1 < num2:
                return -1
            elif num1 > num2:
                return 1
            return 0

        except (ValueError, AttributeError):
            return self._compare_string(v1, v2)

    def _compare_timestamp(self, v1: str, v2: str) -> int:
        """Compare timestamp versions."""
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        return 0

    def _compare_datetime(self, dt1: datetime, dt2: datetime) -> int:
        """Compare datetime objects."""
        if dt1 < dt2:
            return -1
        elif dt1 > dt2:
            return 1
        return 0

    def _compare_string(self, s1: str, s2: str) -> int:
        """Compare strings lexicographically."""
        if s1 < s2:
            return -1
        elif s1 > s2:
            return 1
        return 0


@dataclass
class VersioningConfig:
    """Configuration for version management."""

    strategy: VersioningStrategy = VersioningStrategy.SEMANTIC
    version_file: Optional[Path] = None
    auto_increment: bool = True

    # Version lifecycle
    max_versions: int = 50
    auto_archive_days: int = 365
    auto_deprecate_days: int = 180

    # Git integration
    use_git_info: bool = True
    git_tag_pattern: Optional[str] = r"v(\d+\.\d+\.\d+)"

    # Custom format (for CUSTOM strategy)
    custom_format: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields."""
        if isinstance(self.version_file, str):
            self.version_file = Path(self.version_file)


class VersionManager:
    """Comprehensive version management for documentation."""

    def __init__(self,
                 storage_path: Union[str, Path],
                 config: Optional[VersioningConfig] = None):
        """Initialize version manager.

        Args:
            storage_path: Path to store version metadata
            config: Version management configuration
        """
        self.storage_path = Path(storage_path)
        self.config = config or VersioningConfig()
        self._versions_file = self.storage_path / "versions.json"
        self._versions: Dict[str, Version] = {}

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing versions
        self._load_versions()

    def create_version(self,
                      source_path: Path,
                      version_string: Optional[str] = None,
                      title: Optional[str] = None,
                      description: Optional[str] = None,
                      changelog: Optional[str] = None) -> Version:
        """Create new version from source.

        Args:
            source_path: Path to source files
            version_string: Explicit version string (auto-generated if None)
            title: Version title
            description: Version description
            changelog: Changelog for this version

        Returns:
            Created Version object
        """
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        # Generate version string if not provided
        if not version_string:
            version_string = self._generate_next_version()

        # Calculate source hash
        source_hash = self._calculate_directory_hash(source_path)

        # Get Git information if available
        git_info = self._get_git_info(source_path) if self.config.use_git_info else {}

        # Create version object
        version = Version(
            version_string=version_string,
            strategy=self.config.strategy,
            title=title,
            description=description,
            changelog=changelog,
            source_hash=source_hash,
            git_commit=git_info.get('commit'),
            git_branch=git_info.get('branch'),
            author=git_info.get('author')
        )

        # Calculate size and file count
        files = list(source_path.rglob('*'))
        version.file_count = len([f for f in files if f.is_file()])
        version.size_bytes = sum(f.stat().st_size for f in files if f.is_file())

        # Store version
        self._versions[version_string] = version
        self._save_versions()

        logger.info(f"Created version {version_string} with {version.file_count} files ({version.size_bytes} bytes)")

        return version

    def deploy_version(self,
                      version_string: str,
                      deployment_path: Path,
                      deployment_id: Optional[str] = None) -> bool:
        """Mark version as deployed.

        Args:
            version_string: Version to mark as deployed
            deployment_path: Path where version is deployed
            deployment_id: Associated deployment ID

        Returns:
            True if successful
        """
        if version_string not in self._versions:
            raise ValueError(f"Version not found: {version_string}")

        version = self._versions[version_string]

        # Deactivate current active version
        self._deactivate_current_version()

        # Activate new version
        version.status = VersionStatus.ACTIVE
        version.deployed_at = datetime.now()
        version.deployment_path = deployment_path
        version.deployment_id = deployment_id

        self._save_versions()
        logger.info(f"Deployed version {version_string}")

        return True

    def rollback_to_version(self, version_string: str) -> bool:
        """Rollback to a specific version.

        Args:
            version_string: Version to rollback to

        Returns:
            True if successful
        """
        if version_string not in self._versions:
            raise ValueError(f"Version not found: {version_string}")

        target_version = self._versions[version_string]

        if target_version.status == VersionStatus.DRAFT:
            raise ValueError(f"Cannot rollback to draft version: {version_string}")

        # Deactivate current version
        self._deactivate_current_version()

        # Activate target version
        target_version.status = VersionStatus.ACTIVE
        target_version.deployed_at = datetime.now()

        self._save_versions()
        logger.info(f"Rolled back to version {version_string}")

        return True

    def deprecate_version(self,
                         version_string: str,
                         reason: Optional[str] = None) -> bool:
        """Deprecate a version.

        Args:
            version_string: Version to deprecate
            reason: Reason for deprecation

        Returns:
            True if successful
        """
        if version_string not in self._versions:
            raise ValueError(f"Version not found: {version_string}")

        version = self._versions[version_string]

        if version.status == VersionStatus.ACTIVE:
            raise ValueError(f"Cannot deprecate active version: {version_string}")

        version.status = VersionStatus.DEPRECATED
        version.deprecated_at = datetime.now()

        if reason:
            if not version.description:
                version.description = f"Deprecated: {reason}"
            else:
                version.description += f"\nDeprecated: {reason}"

        self._save_versions()
        logger.info(f"Deprecated version {version_string}")

        return True

    def archive_version(self, version_string: str) -> bool:
        """Archive a version.

        Args:
            version_string: Version to archive

        Returns:
            True if successful
        """
        if version_string not in self._versions:
            raise ValueError(f"Version not found: {version_string}")

        version = self._versions[version_string]

        if version.status == VersionStatus.ACTIVE:
            raise ValueError(f"Cannot archive active version: {version_string}")

        version.status = VersionStatus.ARCHIVED
        self._save_versions()

        logger.info(f"Archived version {version_string}")
        return True

    def delete_version(self, version_string: str, force: bool = False) -> bool:
        """Delete a version.

        Args:
            version_string: Version to delete
            force: Force deletion even if active

        Returns:
            True if successful
        """
        if version_string not in self._versions:
            raise ValueError(f"Version not found: {version_string}")

        version = self._versions[version_string]

        if version.status == VersionStatus.ACTIVE and not force:
            raise ValueError(f"Cannot delete active version without force: {version_string}")

        # Delete deployment files if they exist
        if version.deployment_path and version.deployment_path.exists():
            try:
                shutil.rmtree(version.deployment_path)
                logger.debug(f"Deleted deployment files for version {version_string}")
            except Exception as e:
                logger.warning(f"Failed to delete deployment files: {e}")

        # Remove from versions
        del self._versions[version_string]
        self._save_versions()

        logger.info(f"Deleted version {version_string}")
        return True

    def get_version(self, version_string: str) -> Optional[Version]:
        """Get specific version."""
        return self._versions.get(version_string)

    def get_active_version(self) -> Optional[Version]:
        """Get currently active version."""
        for version in self._versions.values():
            if version.status == VersionStatus.ACTIVE:
                return version
        return None

    def list_versions(self,
                     status: Optional[VersionStatus] = None,
                     limit: Optional[int] = None) -> List[Version]:
        """List versions with optional filtering.

        Args:
            status: Filter by status
            limit: Limit number of results

        Returns:
            List of versions sorted by creation time (newest first)
        """
        versions = list(self._versions.values())

        # Filter by status
        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        # Apply limit
        if limit:
            versions = versions[:limit]

        return versions

    def cleanup_old_versions(self) -> int:
        """Clean up old versions based on configuration.

        Returns:
            Number of versions cleaned up
        """
        cleaned = 0

        # Auto-archive old versions
        if self.config.auto_archive_days > 0:
            archive_cutoff = datetime.now() - timedelta(days=self.config.auto_archive_days)

            for version in self._versions.values():
                if (version.status == VersionStatus.STAGED and
                    version.created_at < archive_cutoff):
                    version.status = VersionStatus.ARCHIVED
                    cleaned += 1

        # Auto-deprecate old versions
        if self.config.auto_deprecate_days > 0:
            deprecate_cutoff = datetime.now() - timedelta(days=self.config.auto_deprecate_days)

            for version in self._versions.values():
                if (version.status in [VersionStatus.STAGED, VersionStatus.ARCHIVED] and
                    version.created_at < deprecate_cutoff):
                    version.status = VersionStatus.DEPRECATED
                    version.deprecated_at = datetime.now()
                    cleaned += 1

        # Remove excess versions
        if self.config.max_versions > 0:
            all_versions = sorted(self._versions.values(), key=lambda v: v.created_at, reverse=True)

            # Keep active version regardless of limit
            active_version = self.get_active_version()
            non_active_versions = [v for v in all_versions if v.status != VersionStatus.ACTIVE]

            if len(non_active_versions) > self.config.max_versions - (1 if active_version else 0):
                to_remove = non_active_versions[self.config.max_versions:]

                for version in to_remove:
                    if version.status != VersionStatus.ACTIVE:  # Double check
                        del self._versions[version.version_string]
                        cleaned += 1

        if cleaned > 0:
            self._save_versions()
            logger.info(f"Cleaned up {cleaned} old versions")

        return cleaned

    def export_version_history(self, output_path: Path) -> bool:
        """Export version history to file.

        Args:
            output_path: Path to export file

        Returns:
            True if successful
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'config': {
                    'strategy': self.config.strategy.value,
                    'max_versions': self.config.max_versions,
                    'auto_archive_days': self.config.auto_archive_days
                },
                'versions': {
                    version_string: version.to_dict()
                    for version_string, version in self._versions.items()
                },
                'statistics': self._get_statistics()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported version history to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export version history: {e}")
            return False

    def _generate_next_version(self) -> str:
        """Generate next version string based on strategy."""
        if self.config.strategy == VersioningStrategy.SEMANTIC:
            return self._generate_semantic_version()
        elif self.config.strategy == VersioningStrategy.TIMESTAMP:
            return self._generate_timestamp_version()
        elif self.config.strategy == VersioningStrategy.SEQUENTIAL:
            return self._generate_sequential_version()
        elif self.config.strategy == VersioningStrategy.GIT_HASH:
            return self._generate_git_version()
        elif self.config.strategy == VersioningStrategy.CUSTOM:
            return self._generate_custom_version()
        else:
            raise ValueError(f"Unknown versioning strategy: {self.config.strategy}")

    def _generate_semantic_version(self) -> str:
        """Generate next semantic version."""
        versions = [v for v in self._versions.values()
                   if v.strategy == VersioningStrategy.SEMANTIC]

        if not versions:
            return "1.0.0"

        # Find latest version
        latest = max(versions, key=lambda v: v.created_at)

        try:
            # Parse version (remove 'v' prefix if present)
            version_str = latest.version_string.lstrip('v')
            major, minor, patch = map(int, version_str.split('.'))

            # Increment patch version by default
            patch += 1
            return f"{major}.{minor}.{patch}"

        except ValueError:
            # If parsing fails, start fresh
            return "1.0.0"

    def _generate_timestamp_version(self) -> str:
        """Generate timestamp-based version."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _generate_sequential_version(self) -> str:
        """Generate sequential version."""
        versions = [v for v in self._versions.values()
                   if v.strategy == VersioningStrategy.SEQUENTIAL]

        if not versions:
            return "v1"

        # Extract numbers and find max
        max_num = 0
        for version in versions:
            try:
                num = int(re.search(r'\d+', version.version_string).group())
                max_num = max(max_num, num)
            except (ValueError, AttributeError):
                continue

        return f"v{max_num + 1}"

    def _generate_git_version(self) -> str:
        """Generate version from Git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return f"git-{result.stdout.strip()}"
        except subprocess.CalledProcessError:
            # Fall back to timestamp if Git not available
            return self._generate_timestamp_version()

    def _generate_custom_version(self) -> str:
        """Generate custom format version."""
        if not self.config.custom_format:
            raise ValueError("Custom format not specified")

        # Simple template replacement
        format_str = self.config.custom_format
        format_str = format_str.replace("{timestamp}", datetime.now().strftime("%Y%m%d-%H%M%S"))
        format_str = format_str.replace("{date}", datetime.now().strftime("%Y%m%d"))
        format_str = format_str.replace("{time}", datetime.now().strftime("%H%M%S"))

        return format_str

    def _deactivate_current_version(self):
        """Deactivate currently active version."""
        active_version = self.get_active_version()
        if active_version:
            active_version.status = VersionStatus.STAGED

    def _calculate_directory_hash(self, path: Path) -> str:
        """Calculate hash of directory contents."""
        hash_md5 = hashlib.md5()

        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                # Include file path and content in hash
                hash_md5.update(str(file_path.relative_to(path)).encode())

                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                except Exception as e:
                    logger.warning(f"Could not read file for hashing: {file_path}: {e}")

        return hash_md5.hexdigest()

    def _get_git_info(self, path: Path) -> Dict[str, str]:
        """Get Git information for path."""
        git_info = {}

        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info['commit'] = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info['branch'] = result.stdout.strip()

            # Get author of last commit
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%an'],
                cwd=path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info['author'] = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logger.debug(f"Failed to get Git info: {e}")

        return git_info

    def _load_versions(self):
        """Load versions from storage."""
        if not self._versions_file.exists():
            return

        try:
            with open(self._versions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for version_string, version_data in data.get('versions', {}).items():
                try:
                    version = Version.from_dict(version_data)
                    self._versions[version_string] = version
                except Exception as e:
                    logger.error(f"Failed to load version {version_string}: {e}")

            logger.debug(f"Loaded {len(self._versions)} versions from storage")

        except Exception as e:
            logger.error(f"Failed to load versions: {e}")

    def _save_versions(self):
        """Save versions to storage."""
        try:
            data = {
                'updated_at': datetime.now().isoformat(),
                'versions': {
                    version_string: version.to_dict()
                    for version_string, version in self._versions.items()
                }
            }

            with open(self._versions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
            raise

    def _get_statistics(self) -> Dict[str, Any]:
        """Get version statistics."""
        total_versions = len(self._versions)
        status_counts = {}

        for status in VersionStatus:
            count = len([v for v in self._versions.values() if v.status == status])
            status_counts[status.value] = count

        total_size = sum(v.size_bytes for v in self._versions.values())
        total_files = sum(v.file_count for v in self._versions.values())

        return {
            'total_versions': total_versions,
            'status_counts': status_counts,
            'total_size_bytes': total_size,
            'total_files': total_files,
            'active_version': self.get_active_version().version_string if self.get_active_version() else None
        }