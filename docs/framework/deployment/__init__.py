"""Deployment and CI/CD integration for documentation framework."""

from .builder import DocumentationBuilder
from .deployer import DocumentationDeployer
from .pipeline import DeploymentPipeline, DeploymentConfig, DeploymentResult, DeploymentStatus, BuildResult, BuildStatus
from .versioning import VersionManager, Version, VersioningStrategy, VersionStatus, VersioningConfig

__all__ = [
    # Existing components
    "DocumentationBuilder",
    "DocumentationDeployer",

    # Pipeline management
    "DeploymentPipeline",
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentStatus",
    "BuildResult",
    "BuildStatus",

    # Version management
    "VersionManager",
    "Version",
    "VersioningStrategy",
    "VersionStatus",
    "VersioningConfig",
]