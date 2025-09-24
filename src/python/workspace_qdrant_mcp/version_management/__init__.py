"""
Version Management System for workspace-qdrant-mcp.

This package implements comprehensive version management with document type-based versioning,
conflict resolution, archive management, and user workflow integration.

Task 262 implementation:
- Document Type-Based Versioning
- Conflict Resolution with Format Precedence
- Archive Collections Management
- User Workflow Integration
"""

from .version_manager import (
    VersionManager,
    DocumentType,
    FileFormat,
    ConflictType,
    ResolutionStrategy,
    VersionInfo,
    VersionConflict
)

from .archive_manager import (
    ArchiveManager,
    ArchivePolicy,
    ArchiveStatus,
    ArchiveEntry
)

from .workflow_integration import (
    WorkflowIntegrator,
    WorkflowStatus,
    UserDecision,
    WorkflowStep,
    UserPrompt,
    BatchOperation
)

__all__ = [
    # Version Manager
    "VersionManager",
    "DocumentType",
    "FileFormat",
    "ConflictType",
    "ResolutionStrategy",
    "VersionInfo",
    "VersionConflict",

    # Archive Manager
    "ArchiveManager",
    "ArchivePolicy",
    "ArchiveStatus",
    "ArchiveEntry",

    # Workflow Integration
    "WorkflowIntegrator",
    "WorkflowStatus",
    "UserDecision",
    "WorkflowStep",
    "UserPrompt",
    "BatchOperation"
]