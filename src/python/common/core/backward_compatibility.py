"""
Backward compatibility and migration support for multi-tenant metadata schema.

This module provides comprehensive support for migrating existing collections
to the new metadata-based multi-tenant system while maintaining full backward
compatibility. It handles detection, analysis, and migration of existing
collections without breaking existing workflows.

Key Features:
    - Automatic detection and classification of existing collections
    - Non-destructive migration that preserves existing collection names
    - Metadata injection without changing collection structure
    - Fallback mechanisms for collections without metadata
    - Migration validation and rollback support
    - Performance monitoring during migration

Migration Strategies:
    - **Additive Migration**: Add metadata to existing collections
    - **Gradual Migration**: Support both old and new filtering methods
    - **Rollback Support**: Ability to remove metadata if needed
    - **Validation**: Comprehensive testing of migrated collections

Example:
    ```python
    from backward_compatibility import BackwardCompatibilityManager

    # Initialize migration manager
    manager = BackwardCompatibilityManager(qdrant_client, config)

    # Analyze existing collections
    analysis = await manager.analyze_existing_collections()

    # Migrate collections to metadata schema
    results = await manager.migrate_collections(analysis.collections)

    # Validate migration
    validation = await manager.validate_migration(results)
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

try:
    from .metadata_schema import (
        MultiTenantMetadataSchema,
        CollectionCategory,
        WorkspaceScope,
        AccessLevel
    )
    from .metadata_validator import MetadataValidator, ValidationResult
    from .collection_types import CollectionTypeClassifier, CollectionType
    from .collection_naming import CollectionNamingManager
    from .config import Config
except ImportError:
    logger.error("Cannot import required modules")
    raise


class MigrationStatus(Enum):
    """Status of collection migration."""

    PENDING = "pending"           # Not yet migrated
    IN_PROGRESS = "in_progress"   # Currently being migrated
    COMPLETED = "completed"       # Successfully migrated
    FAILED = "failed"            # Migration failed
    SKIPPED = "skipped"          # Skipped (already has metadata)
    ROLLBACK = "rollback"        # Rolled back


class CollectionAnalysisResult(Enum):
    """Result of collection analysis."""

    SYSTEM_COLLECTION = "system_collection"        # __ prefix
    LIBRARY_COLLECTION = "library_collection"      # _ prefix
    PROJECT_COLLECTION = "project_collection"      # project-suffix pattern
    GLOBAL_COLLECTION = "global_collection"        # predefined global
    LEGACY_COLLECTION = "legacy_collection"        # other patterns
    UNKNOWN_COLLECTION = "unknown_collection"      # unrecognized
    HAS_METADATA = "has_metadata"                  # already migrated


@dataclass
class CollectionAnalysis:
    """Analysis of an existing collection for migration planning."""

    name: str
    analysis_result: CollectionAnalysisResult
    collection_type: CollectionType
    estimated_metadata: Optional[MultiTenantMetadataSchema] = None
    migration_strategy: str = "additive"
    migration_priority: int = 3
    has_existing_metadata: bool = False
    existing_metadata: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_documents: int = 0
    last_modified: Optional[str] = None


@dataclass
class MigrationResult:
    """Result of migrating a single collection."""

    collection_name: str
    status: MigrationStatus
    metadata_added: Optional[MultiTenantMetadataSchema] = None
    documents_updated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    migration_time_seconds: float = 0.0
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class MigrationBatch:
    """Batch migration results and statistics."""

    total_collections: int
    successful_migrations: int
    failed_migrations: int
    skipped_collections: int
    total_documents_updated: int
    total_time_seconds: float
    results: List[MigrationResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class BackwardCompatibilityManager:
    """
    Manages backward compatibility and migration for multi-tenant metadata.

    This class provides comprehensive support for migrating existing collections
    to use the new metadata schema while maintaining full backward compatibility.
    It supports additive migration that preserves existing collection names and
    structures.
    """

    def __init__(self, client: QdrantClient, config: Config):
        """
        Initialize the backward compatibility manager.

        Args:
            client: Qdrant client for database operations
            config: Configuration object with workspace settings
        """
        self.client = client
        self.config = config

        # Initialize supporting systems
        self.validator = MetadataValidator(strict_mode=False)  # Lenient for migration
        self.type_classifier = CollectionTypeClassifier()
        self.naming_manager = CollectionNamingManager(
            global_collections=config.workspace.global_collections,
            valid_project_suffixes=config.workspace.effective_collection_types
        )

        # Migration tracking
        self._migration_history: List[MigrationResult] = []
        self._rollback_data: Dict[str, Any] = {}

    async def analyze_existing_collections(self) -> Dict[str, CollectionAnalysis]:
        """
        Analyze all existing collections for migration planning.

        Returns:
            Dictionary mapping collection names to analysis results
        """
        logger.info("Starting analysis of existing collections for migration")

        try:
            # Get all collections from Qdrant
            collections_response = self.client.get_collections()
            all_collections = [col.name for col in collections_response.collections]

            logger.info(f"Found {len(all_collections)} collections to analyze")

            # Analyze each collection
            analyses = {}
            for collection_name in all_collections:
                analysis = await self._analyze_single_collection(collection_name)
                analyses[collection_name] = analysis

            # Generate summary statistics
            self._log_analysis_summary(analyses)

            return analyses

        except Exception as e:
            logger.error(f"Failed to analyze existing collections: {e}")
            raise

    async def _analyze_single_collection(self, collection_name: str) -> CollectionAnalysis:
        """Analyze a single collection for migration."""
        logger.debug(f"Analyzing collection: {collection_name}")

        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            document_count = collection_info.points_count

            # Check for existing metadata by sampling a few documents
            has_metadata, existing_metadata = await self._check_existing_metadata(collection_name)

            if has_metadata:
                return CollectionAnalysis(
                    name=collection_name,
                    analysis_result=CollectionAnalysisResult.HAS_METADATA,
                    collection_type=CollectionType.UNKNOWN,
                    has_existing_metadata=True,
                    existing_metadata=existing_metadata,
                    estimated_documents=document_count,
                    recommendations=["Collection already has metadata, migration not needed"]
                )

            # Classify collection using existing type system
            collection_type_info = self.type_classifier.get_collection_info(collection_name)
            analysis_result = self._map_collection_type_to_analysis(collection_type_info.type)

            # Generate estimated metadata for migration
            estimated_metadata = self._generate_estimated_metadata(
                collection_name, collection_type_info, analysis_result
            )

            # Determine migration strategy and priority
            migration_strategy, priority = self._determine_migration_strategy(
                collection_name, analysis_result, document_count
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                collection_name, analysis_result, document_count
            )

            return CollectionAnalysis(
                name=collection_name,
                analysis_result=analysis_result,
                collection_type=collection_type_info.type,
                estimated_metadata=estimated_metadata,
                migration_strategy=migration_strategy,
                migration_priority=priority,
                has_existing_metadata=False,
                estimated_documents=document_count,
                recommendations=recommendations
            )

        except Exception as e:
            logger.warning(f"Failed to analyze collection {collection_name}: {e}")
            return CollectionAnalysis(
                name=collection_name,
                analysis_result=CollectionAnalysisResult.UNKNOWN_COLLECTION,
                collection_type=CollectionType.UNKNOWN,
                issues=[f"Analysis failed: {e}"],
                recommendations=["Manual investigation required"]
            )

    async def _check_existing_metadata(self, collection_name: str) -> Tuple[bool, Optional[Dict]]:
        """Check if collection already has metadata by sampling documents."""
        try:
            # Sample a few documents to check for metadata
            search_result = self.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True
            )

            if not search_result[0]:  # No documents
                return False, None

            # Check if any document has our metadata fields
            metadata_fields = ["project_id", "tenant_namespace", "collection_type"]

            for point in search_result[0]:
                if point.payload:
                    has_metadata = any(field in point.payload for field in metadata_fields)
                    if has_metadata:
                        # Extract metadata from first document that has it
                        metadata = {k: v for k, v in point.payload.items()
                                  if k in MultiTenantMetadataSchema.__annotations__}
                        return True, metadata

            return False, None

        except Exception as e:
            logger.debug(f"Error checking metadata for {collection_name}: {e}")
            return False, None

    def _map_collection_type_to_analysis(self, collection_type: CollectionType) -> CollectionAnalysisResult:
        """Map CollectionType to CollectionAnalysisResult."""
        mapping = {
            CollectionType.SYSTEM: CollectionAnalysisResult.SYSTEM_COLLECTION,
            CollectionType.LIBRARY: CollectionAnalysisResult.LIBRARY_COLLECTION,
            CollectionType.PROJECT: CollectionAnalysisResult.PROJECT_COLLECTION,
            CollectionType.GLOBAL: CollectionAnalysisResult.GLOBAL_COLLECTION,
            CollectionType.UNKNOWN: CollectionAnalysisResult.UNKNOWN_COLLECTION
        }
        return mapping.get(collection_type, CollectionAnalysisResult.UNKNOWN_COLLECTION)

    def _generate_estimated_metadata(
        self,
        collection_name: str,
        collection_info,
        analysis_result: CollectionAnalysisResult
    ) -> Optional[MultiTenantMetadataSchema]:
        """Generate estimated metadata for a collection based on its name and type."""

        try:
            if analysis_result == CollectionAnalysisResult.SYSTEM_COLLECTION:
                return MultiTenantMetadataSchema.create_for_system(
                    collection_name=collection_name,
                    collection_type="memory_collection",
                    created_by="migration"
                )

            elif analysis_result == CollectionAnalysisResult.LIBRARY_COLLECTION:
                return MultiTenantMetadataSchema.create_for_library(
                    collection_name=collection_name,
                    collection_type="code_collection",
                    created_by="migration"
                )

            elif analysis_result == CollectionAnalysisResult.PROJECT_COLLECTION:
                # Extract project name and collection type from name
                if '-' in collection_name:
                    parts = collection_name.split('-', 1)
                    project_name = parts[0]
                    collection_type = parts[1]
                else:
                    project_name = collection_name
                    collection_type = "general"

                return MultiTenantMetadataSchema.create_for_project(
                    project_name=project_name,
                    collection_type=collection_type,
                    created_by="migration"
                )

            elif analysis_result == CollectionAnalysisResult.GLOBAL_COLLECTION:
                return MultiTenantMetadataSchema.create_for_global(
                    collection_name=collection_name,
                    collection_type="global",
                    created_by="migration"
                )

            else:
                # For unknown collections, create as project collection
                return MultiTenantMetadataSchema.create_for_project(
                    project_name="unknown",
                    collection_type=collection_name,
                    created_by="migration"
                )

        except Exception as e:
            logger.warning(f"Failed to generate metadata for {collection_name}: {e}")
            return None

    def _determine_migration_strategy(
        self,
        collection_name: str,
        analysis_result: CollectionAnalysisResult,
        document_count: int
    ) -> Tuple[str, int]:
        """Determine migration strategy and priority for a collection."""

        # All migrations use additive strategy (don't change collection names)
        strategy = "additive"

        # Determine priority based on collection type and size
        if analysis_result == CollectionAnalysisResult.SYSTEM_COLLECTION:
            priority = 5  # High priority for system collections
        elif analysis_result == CollectionAnalysisResult.GLOBAL_COLLECTION:
            priority = 4  # High priority for global collections
        elif analysis_result == CollectionAnalysisResult.PROJECT_COLLECTION:
            priority = 3  # Medium priority for project collections
        elif analysis_result == CollectionAnalysisResult.LIBRARY_COLLECTION:
            priority = 2  # Lower priority for library collections
        else:
            priority = 1  # Lowest priority for unknown collections

        # Adjust priority based on collection size
        if document_count > 10000:
            priority = max(1, priority - 1)  # Lower priority for large collections
        elif document_count == 0:
            priority = max(1, priority - 2)  # Much lower priority for empty collections

        return strategy, priority

    def _generate_recommendations(
        self,
        collection_name: str,
        analysis_result: CollectionAnalysisResult,
        document_count: int
    ) -> List[str]:
        """Generate migration recommendations for a collection."""
        recommendations = []

        if analysis_result == CollectionAnalysisResult.SYSTEM_COLLECTION:
            recommendations.append("System collection: Verify access control settings after migration")
            recommendations.append("Consider if MCP read-only access is appropriate")

        elif analysis_result == CollectionAnalysisResult.LIBRARY_COLLECTION:
            recommendations.append("Library collection: Will be set to MCP read-only")
            recommendations.append("Ensure CLI workflows can handle write operations")

        elif analysis_result == CollectionAnalysisResult.PROJECT_COLLECTION:
            recommendations.append("Project collection: Will maintain current access patterns")
            recommendations.append("Verify project isolation works correctly")

        elif analysis_result == CollectionAnalysisResult.GLOBAL_COLLECTION:
            recommendations.append("Global collection: Will be publicly accessible")
            recommendations.append("Review content for appropriate global visibility")

        else:
            recommendations.append("Unknown collection type: Manual review recommended")
            recommendations.append("Consider reclassifying or documenting purpose")

        # Size-based recommendations
        if document_count == 0:
            recommendations.append("Empty collection: Consider if migration is needed")
        elif document_count > 50000:
            recommendations.append("Large collection: Plan migration during low-usage period")
            recommendations.append("Consider batch processing for metadata updates")

        return recommendations

    async def migrate_collections(
        self,
        analyses: Dict[str, CollectionAnalysis],
        batch_size: int = 10,
        max_concurrent: int = 3
    ) -> MigrationBatch:
        """
        Migrate collections to use metadata schema.

        Args:
            analyses: Collection analyses from analyze_existing_collections
            batch_size: Number of collections to process in each batch
            max_concurrent: Maximum concurrent migration operations

        Returns:
            MigrationBatch with results and statistics
        """
        logger.info(f"Starting migration of {len(analyses)} collections")

        start_time = datetime.now()
        results = []

        # Filter collections that need migration
        collections_to_migrate = [
            analysis for analysis in analyses.values()
            if analysis.analysis_result != CollectionAnalysisResult.HAS_METADATA
        ]

        # Sort by priority (highest first)
        collections_to_migrate.sort(key=lambda x: x.migration_priority, reverse=True)

        logger.info(f"Migrating {len(collections_to_migrate)} collections (skipping {len(analyses) - len(collections_to_migrate)} with existing metadata)")

        # Process collections in batches
        for i in range(0, len(collections_to_migrate), batch_size):
            batch = collections_to_migrate[i:i + batch_size]
            batch_results = await self._migrate_batch(batch, max_concurrent)
            results.extend(batch_results)

            # Log progress
            completed = i + len(batch)
            logger.info(f"Migration progress: {completed}/{len(collections_to_migrate)} collections processed")

        # Calculate statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        successful = len([r for r in results if r.status == MigrationStatus.COMPLETED])
        failed = len([r for r in results if r.status == MigrationStatus.FAILED])
        skipped = len([r for r in results if r.status == MigrationStatus.SKIPPED])
        total_docs = sum(r.documents_updated for r in results)

        migration_batch = MigrationBatch(
            total_collections=len(analyses),
            successful_migrations=successful,
            failed_migrations=failed,
            skipped_collections=skipped + (len(analyses) - len(collections_to_migrate)),
            total_documents_updated=total_docs,
            total_time_seconds=total_time,
            results=results
        )

        logger.info(f"Migration completed: {successful} successful, {failed} failed, {skipped} skipped")
        return migration_batch

    async def _migrate_batch(
        self,
        analyses: List[CollectionAnalysis],
        max_concurrent: int
    ) -> List[MigrationResult]:
        """Migrate a batch of collections with concurrency control."""

        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []

        for analysis in analyses:
            task = self._migrate_single_collection_with_semaphore(semaphore, analysis)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        migration_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Migration task failed for {analyses[i].name}: {result}")
                migration_results.append(MigrationResult(
                    collection_name=analyses[i].name,
                    status=MigrationStatus.FAILED,
                    errors=[f"Task failed: {result}"]
                ))
            else:
                migration_results.append(result)

        return migration_results

    async def _migrate_single_collection_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        analysis: CollectionAnalysis
    ) -> MigrationResult:
        """Migrate a single collection with semaphore control."""
        async with semaphore:
            return await self._migrate_single_collection(analysis)

    async def _migrate_single_collection(self, analysis: CollectionAnalysis) -> MigrationResult:
        """Migrate a single collection to use metadata schema."""
        collection_name = analysis.name
        logger.debug(f"Migrating collection: {collection_name}")

        start_time = datetime.now()

        try:
            # Skip if already has metadata
            if analysis.has_existing_metadata:
                return MigrationResult(
                    collection_name=collection_name,
                    status=MigrationStatus.SKIPPED,
                    warnings=["Collection already has metadata"]
                )

            # Skip if no estimated metadata (analysis failed)
            if not analysis.estimated_metadata:
                return MigrationResult(
                    collection_name=collection_name,
                    status=MigrationStatus.FAILED,
                    errors=["No estimated metadata available"]
                )

            # Validate estimated metadata
            validation_result = self.validator.validate_metadata(analysis.estimated_metadata)
            if not validation_result.is_valid:
                return MigrationResult(
                    collection_name=collection_name,
                    status=MigrationStatus.FAILED,
                    errors=[f"Metadata validation failed: {error.message}" for error in validation_result.errors]
                )

            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            document_count = collection_info.points_count

            # If collection is empty, just create metadata (no documents to update)
            if document_count == 0:
                logger.debug(f"Collection {collection_name} is empty, metadata created but no documents to update")
                end_time = datetime.now()
                return MigrationResult(
                    collection_name=collection_name,
                    status=MigrationStatus.COMPLETED,
                    metadata_added=analysis.estimated_metadata,
                    documents_updated=0,
                    migration_time_seconds=(end_time - start_time).total_seconds()
                )

            # Update existing documents with metadata
            documents_updated = await self._add_metadata_to_documents(
                collection_name, analysis.estimated_metadata
            )

            end_time = datetime.now()
            migration_time = (end_time - start_time).total_seconds()

            # Store rollback data
            self._rollback_data[collection_name] = {
                'metadata_added': analysis.estimated_metadata.to_qdrant_payload(),
                'migration_time': migration_time
            }

            logger.info(f"Successfully migrated {collection_name}: {documents_updated} documents updated")

            return MigrationResult(
                collection_name=collection_name,
                status=MigrationStatus.COMPLETED,
                metadata_added=analysis.estimated_metadata,
                documents_updated=documents_updated,
                migration_time_seconds=migration_time
            )

        except Exception as e:
            end_time = datetime.now()
            migration_time = (end_time - start_time).total_seconds()

            logger.error(f"Failed to migrate {collection_name}: {e}")
            return MigrationResult(
                collection_name=collection_name,
                status=MigrationStatus.FAILED,
                errors=[str(e)],
                migration_time_seconds=migration_time
            )

    async def _add_metadata_to_documents(
        self,
        collection_name: str,
        metadata: MultiTenantMetadataSchema
    ) -> int:
        """Add metadata to all documents in a collection."""
        metadata_payload = metadata.to_qdrant_payload()
        documents_updated = 0

        try:
            # Scroll through all documents in batches
            offset = None
            batch_size = 100

            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for metadata update
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Update points with metadata
                updates = []
                for point in points:
                    # Merge existing payload with new metadata
                    updated_payload = {**(point.payload or {}), **metadata_payload}

                    updates.append(models.PointStruct(
                        id=point.id,
                        payload=updated_payload
                    ))

                # Apply updates
                self.client.upsert(
                    collection_name=collection_name,
                    points=updates
                )

                documents_updated += len(updates)
                logger.debug(f"Updated {len(updates)} documents in {collection_name} (total: {documents_updated})")

                # Continue with next batch
                offset = next_offset
                if not next_offset:
                    break

            logger.info(f"Added metadata to {documents_updated} documents in {collection_name}")
            return documents_updated

        except Exception as e:
            logger.error(f"Failed to add metadata to documents in {collection_name}: {e}")
            raise

    async def validate_migration(self, migration_batch: MigrationBatch) -> Dict[str, Any]:
        """
        Validate that migration was successful and collections work correctly.

        Args:
            migration_batch: Results from migrate_collections

        Returns:
            Validation results with detailed checks
        """
        logger.info("Validating migration results")

        validation_results = {
            'overall_status': 'success',
            'collections_validated': 0,
            'validation_errors': [],
            'validation_warnings': [],
            'performance_metrics': {},
            'collection_details': {}
        }

        try:
            for result in migration_batch.results:
                if result.status == MigrationStatus.COMPLETED:
                    collection_validation = await self._validate_single_collection_migration(
                        result.collection_name, result.metadata_added
                    )
                    validation_results['collection_details'][result.collection_name] = collection_validation
                    validation_results['collections_validated'] += 1

                    # Aggregate errors and warnings
                    if collection_validation.get('errors'):
                        validation_results['validation_errors'].extend(collection_validation['errors'])
                        validation_results['overall_status'] = 'partial_failure'

                    if collection_validation.get('warnings'):
                        validation_results['validation_warnings'].extend(collection_validation['warnings'])

            # Overall status assessment
            if validation_results['validation_errors']:
                validation_results['overall_status'] = 'failure' if len(validation_results['validation_errors']) > len(migration_batch.results) // 2 else 'partial_failure'

            logger.info(f"Migration validation completed: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            validation_results['overall_status'] = 'validation_failed'
            validation_results['validation_errors'].append(f"Validation process failed: {e}")
            return validation_results

    async def _validate_single_collection_migration(
        self,
        collection_name: str,
        expected_metadata: MultiTenantMetadataSchema
    ) -> Dict[str, Any]:
        """Validate migration of a single collection."""
        validation = {
            'collection_name': collection_name,
            'metadata_present': False,
            'metadata_consistent': False,
            'document_count': 0,
            'sample_checks': [],
            'errors': [],
            'warnings': []
        }

        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            validation['document_count'] = collection_info.points_count

            if collection_info.points_count == 0:
                validation['metadata_present'] = True
                validation['metadata_consistent'] = True
                validation['warnings'].append("Empty collection - no documents to validate")
                return validation

            # Sample a few documents to check metadata
            sample_result = self.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True
            )

            sampled_points = sample_result[0]
            if not sampled_points:
                validation['errors'].append("No documents found for validation")
                return validation

            # Check metadata in sampled documents
            expected_payload = expected_metadata.to_qdrant_payload()
            metadata_checks = []

            for point in sampled_points:
                if not point.payload:
                    validation['errors'].append(f"Document {point.id} has no payload")
                    continue

                # Check for required metadata fields
                missing_fields = []
                inconsistent_fields = []

                for field, expected_value in expected_payload.items():
                    if field not in point.payload:
                        missing_fields.append(field)
                    elif point.payload[field] != expected_value:
                        inconsistent_fields.append(f"{field}: got {point.payload[field]}, expected {expected_value}")

                check_result = {
                    'document_id': str(point.id),
                    'missing_fields': missing_fields,
                    'inconsistent_fields': inconsistent_fields,
                    'valid': len(missing_fields) == 0 and len(inconsistent_fields) == 0
                }

                metadata_checks.append(check_result)

            validation['sample_checks'] = metadata_checks

            # Determine overall validation status
            valid_samples = [check for check in metadata_checks if check['valid']]
            validation['metadata_present'] = len(valid_samples) > 0
            validation['metadata_consistent'] = len(valid_samples) == len(metadata_checks)

            # Generate errors/warnings based on results
            if not validation['metadata_present']:
                validation['errors'].append("No documents found with expected metadata")
            elif not validation['metadata_consistent']:
                validation['warnings'].append(f"Only {len(valid_samples)}/{len(metadata_checks)} sampled documents have consistent metadata")

            return validation

        except Exception as e:
            validation['errors'].append(f"Validation failed: {e}")
            return validation

    def _log_analysis_summary(self, analyses: Dict[str, CollectionAnalysis]):
        """Log summary of collection analysis."""
        by_type = {}
        for analysis in analyses.values():
            result_type = analysis.analysis_result.value
            if result_type not in by_type:
                by_type[result_type] = 0
            by_type[result_type] += 1

        logger.info("Collection analysis summary:")
        for result_type, count in by_type.items():
            logger.info(f"  {result_type}: {count} collections")

        # Log migration priorities
        priority_counts = {}
        for analysis in analyses.values():
            priority = analysis.migration_priority
            if priority not in priority_counts:
                priority_counts[priority] = 0
            priority_counts[priority] += 1

        logger.info("Migration priority distribution:")
        for priority in sorted(priority_counts.keys(), reverse=True):
            logger.info(f"  Priority {priority}: {priority_counts[priority]} collections")

    async def rollback_migration(self, collection_names: List[str]) -> Dict[str, bool]:
        """
        Rollback migration for specified collections.

        Args:
            collection_names: List of collection names to rollback

        Returns:
            Dictionary mapping collection names to rollback success status
        """
        logger.info(f"Rolling back migration for {len(collection_names)} collections")

        rollback_results = {}

        for collection_name in collection_names:
            try:
                if collection_name not in self._rollback_data:
                    logger.warning(f"No rollback data available for {collection_name}")
                    rollback_results[collection_name] = False
                    continue

                # Remove metadata from all documents
                await self._remove_metadata_from_documents(collection_name)
                rollback_results[collection_name] = True

                logger.info(f"Successfully rolled back migration for {collection_name}")

            except Exception as e:
                logger.error(f"Failed to rollback migration for {collection_name}: {e}")
                rollback_results[collection_name] = False

        return rollback_results

    async def _remove_metadata_from_documents(self, collection_name: str) -> int:
        """Remove migration metadata from all documents in a collection."""
        documents_updated = 0
        metadata_fields = list(MultiTenantMetadataSchema.__annotations__.keys())

        try:
            # Scroll through all documents
            offset = None
            batch_size = 100

            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Remove metadata fields from points
                updates = []
                for point in points:
                    if point.payload:
                        # Remove metadata fields
                        cleaned_payload = {
                            k: v for k, v in point.payload.items()
                            if k not in metadata_fields
                        }

                        updates.append(models.PointStruct(
                            id=point.id,
                            payload=cleaned_payload
                        ))

                # Apply updates
                if updates:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=updates
                    )

                documents_updated += len(updates)

                offset = next_offset
                if not next_offset:
                    break

            logger.info(f"Removed metadata from {documents_updated} documents in {collection_name}")
            return documents_updated

        except Exception as e:
            logger.error(f"Failed to remove metadata from {collection_name}: {e}")
            raise

    def get_migration_history(self) -> List[MigrationResult]:
        """Get history of all migration operations."""
        return self._migration_history.copy()

    def clear_migration_history(self):
        """Clear migration history and rollback data."""
        self._migration_history.clear()
        self._rollback_data.clear()
        logger.info("Migration history and rollback data cleared")


# Export all public classes and functions
__all__ = [
    'BackwardCompatibilityManager',
    'CollectionAnalysis',
    'MigrationResult',
    'MigrationBatch',
    'MigrationStatus',
    'CollectionAnalysisResult'
]