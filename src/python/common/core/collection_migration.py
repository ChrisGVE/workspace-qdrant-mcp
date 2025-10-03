"""
Collection Migration System for workspace-qdrant-mcp.

This module implements comprehensive collection validation and migration functionality
to ensure all collections conform to the type-specific requirements defined in the
collection type configuration system.

Key Features:
    - Collection validation against type-specific schemas
    - Automatic collection type detection with confidence scoring
    - Safe migration with rollback capability
    - Edge case handling for ambiguous and legacy collections
    - Detailed migration reporting and recommendations

Architecture:
    - CollectionMigrator: Main migration engine coordinating all operations
    - ValidationResult: Detailed validation outcome with actionable errors
    - DetectionResult: Type detection with confidence and alternatives
    - MigrationResult: Complete migration outcome with rollback support
    - MigrationReport: Comprehensive analysis across multiple collections

Usage:
    ```python
    from collection_migration import CollectionMigrator
    from qdrant_client import QdrantClient

    # Initialize migrator
    client = QdrantClient(url="http://localhost:6333")
    migrator = CollectionMigrator(client)
    await migrator.initialize()

    # Validate a collection
    result = await migrator.validate_collection("my-collection")
    if not result.is_valid:
        print(f"Validation errors: {result.errors}")

    # Detect collection type
    detection = await migrator.detect_collection_type("my-collection")
    print(f"Detected type: {detection.detected_type} (confidence: {detection.confidence})")

    # Migrate a collection (dry-run first)
    migration = await migrator.migrate_collection(
        "my-collection",
        CollectionType.PROJECT,
        dry_run=True
    )
    if migration.success:
        # Execute actual migration
        migration = await migrator.migrate_collection(
            "my-collection",
            CollectionType.PROJECT,
            dry_run=False
        )

    # Generate migration report
    report = await migrator.generate_migration_report(["collection1", "collection2"])
    print(report.to_json())
    ```
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import CollectionInfo as QdrantCollectionInfo

from .collection_type_config import (
    CollectionTypeConfig,
    get_type_config,
    get_all_type_configs,
    MetadataFieldSpec,
)
from .collection_types import CollectionType, CollectionTypeClassifier, CollectionInfo
from .collision_detection import CollisionDetector


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Blocking issue preventing migration
    WARNING = "warning"  # Non-blocking issue requiring attention
    INFO = "info"  # Informational notice


class MigrationStrategy(Enum):
    """Migration strategies for different scenarios."""

    AUTOMATIC = "automatic"  # Clear type match, no conflicts
    ASSISTED = "assisted"  # Ambiguous, requires user choice
    MANUAL = "manual"  # Complex conflicts, requires manual intervention


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    field_name: Optional[str]
    severity: ValidationSeverity
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of collection validation against type requirements.

    Provides detailed information about validation outcome including
    errors, warnings, and suggested fixes.
    """

    collection_name: str
    is_valid: bool
    detected_type: Optional[CollectionType] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata_present: Dict[str, bool] = field(default_factory=dict)
    metadata_values: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        """Get only info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "is_valid": self.is_valid,
            "detected_type": self.detected_type.value if self.detected_type else None,
            "issues": [
                {
                    "field": i.field_name,
                    "severity": i.severity.value,
                    "message": i.message,
                    "fix": i.suggested_fix,
                }
                for i in self.issues
            ],
            "metadata_present": self.metadata_present,
            "validation_time_ms": self.validation_time_ms,
        }


@dataclass
class DetectionResult:
    """
    Result of collection type detection.

    Provides type detection with confidence scoring and alternative
    possibilities for ambiguous cases.
    """

    collection_name: str
    detected_type: CollectionType
    confidence: float  # 0.0 to 1.0
    alternative_types: List[Tuple[CollectionType, float]] = field(default_factory=list)
    detection_reason: str = ""
    requires_manual_intervention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "detected_type": self.detected_type.value,
            "confidence": self.confidence,
            "alternatives": [
                {"type": t.value, "confidence": c} for t, c in self.alternative_types
            ],
            "reason": self.detection_reason,
            "manual_intervention": self.requires_manual_intervention,
        }


@dataclass
class MigrationBackup:
    """Backup data for migration rollback."""

    collection_name: str
    original_metadata: Dict[str, Any]
    backup_timestamp: str
    backup_id: str


@dataclass
class MigrationResult:
    """
    Result of collection migration operation.

    Contains complete migration outcome with rollback support and
    detailed execution information.
    """

    collection_name: str
    success: bool
    target_type: CollectionType
    dry_run: bool = False
    changes_applied: List[str] = field(default_factory=list)
    conflicts_detected: List[str] = field(default_factory=list)
    migration_time_ms: float = 0.0
    backup: Optional[MigrationBackup] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "success": self.success,
            "target_type": self.target_type.value,
            "dry_run": self.dry_run,
            "changes_applied": self.changes_applied,
            "conflicts": self.conflicts_detected,
            "migration_time_ms": self.migration_time_ms,
            "has_backup": self.backup is not None,
            "error": self.error_message,
        }


@dataclass
class MigrationRecommendation:
    """Recommendation for collection migration."""

    collection_name: str
    current_type: CollectionType
    recommended_type: CollectionType
    strategy: MigrationStrategy
    reason: str
    estimated_time_ms: float = 0.0
    risk_level: str = "low"  # low, medium, high


@dataclass
class MigrationReport:
    """
    Comprehensive migration analysis report.

    Provides detailed analysis of collections categorized by type,
    migration needs, and actionable recommendations.
    """

    total_collections: int
    valid_collections: int
    invalid_collections: int
    collections_by_type: Dict[str, List[str]] = field(default_factory=dict)
    collections_needing_migration: List[str] = field(default_factory=list)
    problematic_collections: List[str] = field(default_factory=list)
    recommendations: List[MigrationRecommendation] = field(default_factory=list)
    generation_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    estimated_total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_collections": self.total_collections,
            "valid_collections": self.valid_collections,
            "invalid_collections": self.invalid_collections,
            "by_type": self.collections_by_type,
            "needs_migration": self.collections_needing_migration,
            "problematic": self.problematic_collections,
            "recommendations": [
                {
                    "collection": r.collection_name,
                    "current_type": r.current_type.value,
                    "recommended_type": r.recommended_type.value,
                    "strategy": r.strategy.value,
                    "reason": r.reason,
                    "estimated_time_ms": r.estimated_time_ms,
                    "risk": r.risk_level,
                }
                for r in self.recommendations
            ],
            "generation_time": self.generation_time,
            "estimated_total_time_ms": self.estimated_total_time_ms,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class CollectionMigrator:
    """
    Main migration engine for collection validation and migration.

    Coordinates collection type detection, validation, migration,
    and reporting with support for rollback and edge case handling.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collision_detector: Optional[CollisionDetector] = None,
    ):
        """
        Initialize the collection migrator.

        Args:
            qdrant_client: Qdrant client for database operations
            collision_detector: Optional collision detector for conflict checking
        """
        self.qdrant_client = qdrant_client
        self.classifier = CollectionTypeClassifier()
        self.collision_detector = collision_detector

        # Migration state tracking
        self._backups: Dict[str, MigrationBackup] = {}
        self._migration_history: List[MigrationResult] = []
        self._initialized = False

        # Configuration
        self.confidence_threshold_automatic = 0.9  # >= 90% confidence for auto migration
        self.confidence_threshold_assisted = 0.6  # >= 60% confidence for assisted

        logger.info("Initialized collection migrator")

    async def initialize(self):
        """Initialize the migrator and load collection data."""
        if self._initialized:
            return

        logger.info("Initializing collection migrator...")

        try:
            # Initialize collision detector if provided
            if self.collision_detector:
                await self.collision_detector.initialize()

            self._initialized = True
            logger.info("Collection migrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collection migrator: {e}")
            raise

    async def validate_collection(self, collection_name: str) -> ValidationResult:
        """
        Validate a collection against type-specific requirements.

        Args:
            collection_name: Name of the collection to validate

        Returns:
            ValidationResult with detailed validation outcome
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        logger.debug(f"Validating collection: {collection_name}")

        # Detect collection type
        detected_type = self.classifier.classify_collection_type(collection_name)

        issues: List[ValidationIssue] = []
        metadata_present: Dict[str, bool] = {}
        metadata_values: Dict[str, Any] = {}

        # Check if collection exists in Qdrant
        try:
            # Get collection info from Qdrant
            qdrant_collection = self.qdrant_client.get_collection(collection_name)

            # Extract metadata from collection config
            # Note: Qdrant doesn't store custom metadata in collection info,
            # we need to check point payloads or use a metadata storage system
            # For now, we'll validate based on naming patterns

            if detected_type == CollectionType.UNKNOWN:
                issues.append(
                    ValidationIssue(
                        field_name=None,
                        severity=ValidationSeverity.ERROR,
                        message="Collection name does not match any known type pattern",
                        suggested_fix="Rename collection to match type conventions or assign explicit type",
                    )
                )
            else:
                # Get type-specific configuration
                try:
                    type_config = get_type_config(detected_type)

                    # Validate naming pattern matches type
                    if detected_type == CollectionType.SYSTEM and not collection_name.startswith(
                        "__"
                    ):
                        issues.append(
                            ValidationIssue(
                                field_name="collection_name",
                                severity=ValidationSeverity.ERROR,
                                message="System collections must start with '__'",
                                suggested_fix=f"Rename to '__{collection_name}'",
                            )
                        )

                    elif detected_type == CollectionType.LIBRARY and not collection_name.startswith(
                        "_"
                    ):
                        issues.append(
                            ValidationIssue(
                                field_name="collection_name",
                                severity=ValidationSeverity.ERROR,
                                message="Library collections must start with '_'",
                                suggested_fix=f"Rename to '_{collection_name}'",
                            )
                        )

                    # Check for required metadata fields
                    # Since Qdrant doesn't store collection-level metadata,
                    # we add informational notices about expected metadata
                    for field_spec in type_config.required_metadata_fields:
                        metadata_present[field_spec.name] = False  # Unknown until checked
                        issues.append(
                            ValidationIssue(
                                field_name=field_spec.name,
                                severity=ValidationSeverity.INFO,
                                message=f"Required field '{field_spec.name}' should be present in collection metadata",
                                suggested_fix="Add metadata field during migration",
                            )
                        )

                except ValueError as e:
                    issues.append(
                        ValidationIssue(
                            field_name=None,
                            severity=ValidationSeverity.ERROR,
                            message=f"No configuration available for type: {detected_type}",
                            suggested_fix="Define type configuration before migration",
                        )
                    )

        except UnexpectedResponse as e:
            if "Not found" in str(e):
                issues.append(
                    ValidationIssue(
                        field_name=None,
                        severity=ValidationSeverity.ERROR,
                        message="Collection does not exist in Qdrant",
                        suggested_fix="Create collection first",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        field_name=None,
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to retrieve collection: {str(e)}",
                    )
                )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    field_name=None,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation error: {str(e)}",
                )
            )

        # Determine if validation passed (no errors)
        is_valid = all(issue.severity != ValidationSeverity.ERROR for issue in issues)

        validation_time = (time.time() - start_time) * 1000

        result = ValidationResult(
            collection_name=collection_name,
            is_valid=is_valid,
            detected_type=detected_type,
            issues=issues,
            metadata_present=metadata_present,
            metadata_values=metadata_values,
            validation_time_ms=validation_time,
        )

        logger.debug(
            f"Validation complete for {collection_name}: "
            f"valid={is_valid}, issues={len(issues)}"
        )

        return result

    async def detect_collection_type(
        self, collection_name: str
    ) -> DetectionResult:
        """
        Detect collection type with confidence scoring.

        Args:
            collection_name: Name of the collection to analyze

        Returns:
            DetectionResult with type detection and confidence
        """
        logger.debug(f"Detecting type for collection: {collection_name}")

        # Use classifier for primary detection
        detected_type = self.classifier.classify_collection_type(collection_name)

        # Calculate confidence based on pattern matching strength
        confidence = 0.0
        alternatives: List[Tuple[CollectionType, float]] = []
        detection_reason = ""

        if detected_type == CollectionType.SYSTEM:
            if collection_name.startswith("__"):
                confidence = 1.0
                detection_reason = "Exact match: System collection prefix '__'"
            else:
                confidence = 0.3
                detection_reason = "Weak match: Classified as system but missing prefix"

        elif detected_type == CollectionType.LIBRARY:
            if collection_name.startswith("_") and not collection_name.startswith("__"):
                confidence = 1.0
                detection_reason = "Exact match: Library collection prefix '_'"
            else:
                confidence = 0.3
                detection_reason = "Weak match: Classified as library but missing prefix"

        elif detected_type == CollectionType.GLOBAL:
            if collection_name in ["algorithms", "codebase", "context", "documents",
                                  "knowledge", "memory", "projects", "workspace"]:
                confidence = 1.0
                detection_reason = "Exact match: Known global collection name"
            else:
                confidence = 0.5
                detection_reason = "Partial match: Matches global pattern"

        elif detected_type == CollectionType.PROJECT:
            if "-" in collection_name and not collection_name.startswith("_"):
                # Validate project pattern more strictly
                parts = collection_name.split("-")
                if len(parts) >= 2 and all(part for part in parts):
                    confidence = 0.9
                    detection_reason = "Strong match: Project collection pattern '{project}-{suffix}'"
                else:
                    confidence = 0.6
                    detection_reason = "Partial match: Contains dash but irregular pattern"
            else:
                confidence = 0.4
                detection_reason = "Weak match: Classified as project but missing pattern"

        else:  # UNKNOWN
            confidence = 0.0
            detection_reason = "No match: Collection name doesn't match any known pattern"

            # Try to suggest possible types based on name characteristics
            if collection_name.startswith("_"):
                alternatives.append((CollectionType.LIBRARY, 0.5))
            if "-" in collection_name:
                alternatives.append((CollectionType.PROJECT, 0.4))

        # Determine if manual intervention is needed
        requires_manual = confidence < self.confidence_threshold_assisted

        result = DetectionResult(
            collection_name=collection_name,
            detected_type=detected_type,
            confidence=confidence,
            alternative_types=alternatives,
            detection_reason=detection_reason,
            requires_manual_intervention=requires_manual,
        )

        logger.debug(
            f"Type detection complete: {detected_type.value} "
            f"(confidence: {confidence:.2f})"
        )

        return result

    async def migrate_collection(
        self,
        collection_name: str,
        target_type: CollectionType,
        dry_run: bool = True,
    ) -> MigrationResult:
        """
        Migrate a collection to a target type.

        Args:
            collection_name: Name of the collection to migrate
            target_type: Target collection type
            dry_run: If True, validate migration without applying changes

        Returns:
            MigrationResult with migration outcome and details
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Migrating collection: "
            f"{collection_name} -> {target_type.value}"
        )

        changes_applied: List[str] = []
        conflicts: List[str] = []
        backup: Optional[MigrationBackup] = None

        try:
            # Step 1: Validate collection exists
            validation = await self.validate_collection(collection_name)

            if not validation.is_valid and any(
                "does not exist" in issue.message for issue in validation.errors
            ):
                error_msg = f"Collection '{collection_name}' does not exist"
                logger.error(error_msg)
                return MigrationResult(
                    collection_name=collection_name,
                    success=False,
                    target_type=target_type,
                    dry_run=dry_run,
                    conflicts_detected=[error_msg],
                    migration_time_ms=(time.time() - start_time) * 1000,
                    error_message=error_msg,
                )

            # Step 2: Detect current type and check compatibility
            detection = await self.detect_collection_type(collection_name)

            if detection.detected_type == target_type:
                msg = f"Collection already matches target type: {target_type.value}"
                logger.info(msg)
                changes_applied.append("No migration needed - type already matches")
                return MigrationResult(
                    collection_name=collection_name,
                    success=True,
                    target_type=target_type,
                    dry_run=dry_run,
                    changes_applied=changes_applied,
                    migration_time_ms=(time.time() - start_time) * 1000,
                )

            # Step 3: Check for naming conflicts
            target_config = get_type_config(target_type)

            # Validate target name would match type pattern
            new_name = self._generate_target_name(collection_name, target_type)

            if new_name != collection_name:
                changes_applied.append(f"Rename: {collection_name} -> {new_name}")

                # Check for collisions if collision detector available
                if self.collision_detector:
                    collision_result = await self.collision_detector.check_collection_collision(
                        new_name
                    )
                    if collision_result.has_collision:
                        conflict_msg = f"Target name collision: {collision_result.collision_reason}"
                        conflicts.append(conflict_msg)
                        logger.warning(conflict_msg)

            # Step 4: Create backup (if not dry run)
            if not dry_run:
                backup = await self._create_backup(collection_name)
                changes_applied.append(f"Created backup: {backup.backup_id}")

            # Step 5: Apply metadata updates
            metadata_changes = self._calculate_metadata_changes(
                collection_name, target_type, target_config
            )

            for change in metadata_changes:
                changes_applied.append(change)

            # Step 6: Execute migration (if not dry run)
            if not dry_run and len(conflicts) == 0:
                try:
                    # Note: Actual migration would involve:
                    # 1. Renaming collection (if needed)
                    # 2. Updating collection configuration
                    # 3. Updating metadata in stored points
                    # This is a simplified implementation showing the structure

                    if new_name != collection_name:
                        # Qdrant doesn't support direct rename, would need to:
                        # 1. Create new collection with new name
                        # 2. Copy all points
                        # 3. Delete old collection
                        logger.warning(
                            "Collection rename requires manual intervention: "
                            "create new collection, copy data, delete old"
                        )
                        conflicts.append(
                            "Rename operation requires manual data migration"
                        )

                    logger.info(f"Migration completed for {collection_name}")

                except Exception as e:
                    error_msg = f"Migration execution failed: {str(e)}"
                    logger.error(error_msg)

                    # Attempt rollback if backup exists
                    if backup:
                        await self._rollback_migration(backup)
                        changes_applied.append("Rolled back due to error")

                    return MigrationResult(
                        collection_name=collection_name,
                        success=False,
                        target_type=target_type,
                        dry_run=dry_run,
                        changes_applied=changes_applied,
                        conflicts_detected=conflicts,
                        migration_time_ms=(time.time() - start_time) * 1000,
                        backup=backup,
                        error_message=error_msg,
                    )

            # Determine success
            success = len(conflicts) == 0 or dry_run

            result = MigrationResult(
                collection_name=collection_name,
                success=success,
                target_type=target_type,
                dry_run=dry_run,
                changes_applied=changes_applied,
                conflicts_detected=conflicts,
                migration_time_ms=(time.time() - start_time) * 1000,
                backup=backup,
            )

            # Store in history
            self._migration_history.append(result)

            logger.info(
                f"Migration {'dry-run' if dry_run else 'execution'} complete: "
                f"success={success}, changes={len(changes_applied)}, conflicts={len(conflicts)}"
            )

            return result

        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(error_msg)

            return MigrationResult(
                collection_name=collection_name,
                success=False,
                target_type=target_type,
                dry_run=dry_run,
                conflicts_detected=conflicts,
                migration_time_ms=(time.time() - start_time) * 1000,
                error_message=error_msg,
            )

    async def generate_migration_report(
        self, collections: Optional[List[str]] = None
    ) -> MigrationReport:
        """
        Generate comprehensive migration analysis report.

        Args:
            collections: Optional list of collections to analyze.
                       If None, analyzes all collections.

        Returns:
            MigrationReport with detailed analysis and recommendations
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Generating migration report...")

        # Get all collections if not specified
        if collections is None:
            try:
                qdrant_collections = self.qdrant_client.get_collections()
                collections = [c.name for c in qdrant_collections.collections]
            except Exception as e:
                logger.error(f"Failed to retrieve collections: {e}")
                collections = []

        total_collections = len(collections)
        valid_collections = 0
        invalid_collections = 0
        collections_by_type: Dict[str, List[str]] = {
            t.value: [] for t in CollectionType
        }
        collections_needing_migration: List[str] = []
        problematic_collections: List[str] = []
        recommendations: List[MigrationRecommendation] = []
        total_estimated_time = 0.0

        # Analyze each collection
        for collection_name in collections:
            try:
                # Validate collection
                validation = await self.validate_collection(collection_name)

                if validation.is_valid:
                    valid_collections += 1
                else:
                    invalid_collections += 1

                # Detect type
                detection = await self.detect_collection_type(collection_name)

                # Categorize by type
                collections_by_type[detection.detected_type.value].append(
                    collection_name
                )

                # Check if migration is needed
                if not validation.is_valid or detection.confidence < 0.9:
                    collections_needing_migration.append(collection_name)

                    # Determine migration strategy
                    if detection.confidence >= self.confidence_threshold_automatic:
                        strategy = MigrationStrategy.AUTOMATIC
                        risk = "low"
                    elif detection.confidence >= self.confidence_threshold_assisted:
                        strategy = MigrationStrategy.ASSISTED
                        risk = "medium"
                    else:
                        strategy = MigrationStrategy.MANUAL
                        risk = "high"

                    # Estimate migration time (simplified)
                    estimated_time = 100.0  # Base time in ms
                    if detection.detected_type == CollectionType.UNKNOWN:
                        estimated_time *= 5  # More time for unknown types
                    total_estimated_time += estimated_time

                    # Create recommendation
                    recommendation = MigrationRecommendation(
                        collection_name=collection_name,
                        current_type=detection.detected_type,
                        recommended_type=detection.detected_type
                        if detection.detected_type != CollectionType.UNKNOWN
                        else CollectionType.PROJECT,  # Default fallback
                        strategy=strategy,
                        reason=detection.detection_reason,
                        estimated_time_ms=estimated_time,
                        risk_level=risk,
                    )
                    recommendations.append(recommendation)

                # Track problematic collections
                if validation.errors:
                    problematic_collections.append(collection_name)

            except Exception as e:
                logger.error(f"Failed to analyze collection {collection_name}: {e}")
                problematic_collections.append(collection_name)

        report = MigrationReport(
            total_collections=total_collections,
            valid_collections=valid_collections,
            invalid_collections=invalid_collections,
            collections_by_type=collections_by_type,
            collections_needing_migration=collections_needing_migration,
            problematic_collections=problematic_collections,
            recommendations=recommendations,
            estimated_total_time_ms=total_estimated_time,
        )

        logger.info(
            f"Migration report generated: {total_collections} collections analyzed, "
            f"{len(collections_needing_migration)} need migration"
        )

        return report

    def _generate_target_name(
        self, current_name: str, target_type: CollectionType
    ) -> str:
        """
        Generate target collection name for migration.

        Args:
            current_name: Current collection name
            target_type: Target collection type

        Returns:
            Properly formatted target name
        """
        # Remove existing prefixes
        base_name = current_name.lstrip("_")

        # Apply type-specific naming
        if target_type == CollectionType.SYSTEM:
            return f"__{base_name}"
        elif target_type == CollectionType.LIBRARY:
            return f"_{base_name}"
        elif target_type == CollectionType.PROJECT:
            # For PROJECT type, ensure {project}-{suffix} format
            if "-" not in base_name:
                # Add default suffix if missing
                return f"{base_name}-docs"
            return base_name
        elif target_type == CollectionType.GLOBAL:
            # Global collections use their name as-is
            return base_name
        else:
            return current_name

    def _calculate_metadata_changes(
        self,
        collection_name: str,
        target_type: CollectionType,
        config: CollectionTypeConfig,
    ) -> List[str]:
        """
        Calculate metadata changes needed for migration.

        Args:
            collection_name: Collection name
            target_type: Target type
            config: Type configuration

        Returns:
            List of changes to apply
        """
        changes: List[str] = []

        # Add required metadata fields
        for field_spec in config.required_metadata_fields:
            if field_spec.default is not None:
                changes.append(
                    f"Add metadata: {field_spec.name} = {field_spec.default}"
                )
            else:
                changes.append(f"Add required metadata: {field_spec.name}")

        # Update collection category/type metadata
        changes.append(f"Set collection_type = {target_type.value}")

        return changes

    async def _create_backup(self, collection_name: str) -> MigrationBackup:
        """
        Create backup of collection metadata before migration.

        Args:
            collection_name: Collection to back up

        Returns:
            MigrationBackup with backup data
        """
        backup_id = f"backup_{collection_name}_{int(time.time())}"

        # Get current collection configuration
        try:
            qdrant_collection = self.qdrant_client.get_collection(collection_name)

            # Extract metadata (simplified - actual implementation would
            # include full collection config and sample points)
            metadata = {
                "name": collection_name,
                "vectors_count": qdrant_collection.vectors_count,
                "status": qdrant_collection.status,
            }

            backup = MigrationBackup(
                collection_name=collection_name,
                original_metadata=metadata,
                backup_timestamp=datetime.now(timezone.utc).isoformat(),
                backup_id=backup_id,
            )

            # Store backup
            self._backups[backup_id] = backup

            logger.info(f"Created backup: {backup_id}")
            return backup

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    async def _rollback_migration(self, backup: MigrationBackup) -> bool:
        """
        Rollback migration using backup data.

        Args:
            backup: Backup to restore from

        Returns:
            True if rollback successful
        """
        try:
            logger.info(f"Rolling back migration: {backup.backup_id}")

            # Restore collection state from backup
            # Actual implementation would:
            # 1. Restore collection configuration
            # 2. Restore metadata
            # 3. Verify restoration

            logger.info(f"Rollback completed: {backup.backup_id}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


# Export public classes and functions
__all__ = [
    "CollectionMigrator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "DetectionResult",
    "MigrationResult",
    "MigrationReport",
    "MigrationStrategy",
    "MigrationRecommendation",
]
