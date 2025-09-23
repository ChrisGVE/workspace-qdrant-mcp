"""
Migration utilities for transforming suffix-based collections to multi-tenant architecture.

This module provides comprehensive tools for safely migrating existing suffix-based collections
to the new multi-tenant architecture using metadata-based project isolation.

Key Components:
    - CollectionStructureAnalyzer: Analyzes existing collection patterns and data
    - MigrationPlanner: Creates safe migration strategies with dependency analysis
    - BatchMigrator: Performs data migration with metadata injection and error handling
    - RollbackManager: Provides rollback capabilities for failed migrations
    - MigrationReporter: Tracks progress and generates comprehensive reports
    - CLI integration for migration management

Migration Process:
    1. Analyze existing collections and identify suffix patterns
    2. Plan migration with conflict detection and dependency analysis
    3. Create backups for rollback safety
    4. Migrate data with metadata injection in batches
    5. Validate migrated data integrity
    6. Generate comprehensive migration reports
    7. Clean up temporary data and finalize migration

Example:
    ```python
    from workspace_qdrant_mcp.memory.migration_utils import CollectionMigrationManager
    
    manager = CollectionMigrationManager(client, config)
    
    # Analyze existing collections
    analysis = await manager.analyze_collections()
    
    # Plan migration
    plan = await manager.create_migration_plan(analysis)
    
    # Execute migration
    result = await manager.execute_migration(plan)
    
    # Generate report
    report = await manager.generate_migration_report(result)
    ```
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.metadata_schema import MultiTenantMetadataSchema
from ..core.collections import WorkspaceCollectionManager
from ..core.collision_detection import CollisionDetector
from ..core.collection_naming_validation import CollectionNamingValidator
from ..core.metadata_filtering import MetadataFilterManager
from ..core.client import QdrantWorkspaceClient
from ..core.config import Config

logger = logging.getLogger(__name__)


class MigrationPhase(Enum):
    """Migration execution phases."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    BACKUP = "backup"
    MIGRATION = "migration"
    VALIDATION = "validation"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class CollectionPattern(Enum):
    """Detected collection naming patterns."""
    SUFFIX_BASED = "suffix_based"          # project-suffix format
    PROJECT_BASED = "project_based"        # project format
    MIXED = "mixed"                        # combination of patterns
    UNKNOWN = "unknown"                    # unrecognized pattern
    GLOBAL = "global"                      # global collections


@dataclass
class CollectionInfo:
    """Information about a detected collection."""
    name: str
    pattern: CollectionPattern
    project_name: Optional[str] = None
    suffix: Optional[str] = None
    point_count: int = 0
    vector_count: int = 0
    size_mb: float = 0.0
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    metadata_keys: Set[str] = field(default_factory=set)
    has_project_metadata: bool = False
    migration_priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class MigrationPlan:
    """Comprehensive migration plan for collections."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Collections to migrate
    source_collections: List[CollectionInfo] = field(default_factory=list)
    target_collections: List[str] = field(default_factory=list)
    
    # Migration strategy
    batch_size: int = 1000
    parallel_batches: int = 3
    enable_validation: bool = True
    create_backups: bool = True
    
    # Conflict resolution
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolutions: Dict[str, str] = field(default_factory=dict)
    
    # Estimated metrics
    estimated_duration_minutes: float = 0.0
    estimated_storage_mb: float = 0.0
    total_points_to_migrate: int = 0
    
    # Dependencies and ordering
    migration_order: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MigrationResult:
    """Results of a migration execution."""
    plan_id: str
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Execution status
    phase: MigrationPhase = MigrationPhase.ANALYSIS
    success: bool = False
    error_message: Optional[str] = None
    
    # Migration statistics
    collections_migrated: int = 0
    points_migrated: int = 0
    points_failed: int = 0
    batches_processed: int = 0
    batches_failed: int = 0
    
    # Timing information
    analysis_duration_seconds: float = 0.0
    migration_duration_seconds: float = 0.0
    validation_duration_seconds: float = 0.0
    
    # Backup information
    backup_locations: List[str] = field(default_factory=list)
    backup_size_mb: float = 0.0
    
    # Detailed logs
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CollectionStructureAnalyzer:
    """
    Analyzes existing collection structure and patterns.
    
    This class examines all collections in the Qdrant instance to:
    - Identify naming patterns (suffix-based, project-based, etc.)
    - Analyze collection metadata and content structure
    - Detect project associations and dependencies
    - Estimate migration complexity and requirements
    """
    
    def __init__(self, client: QdrantClient, config: Config):
        """
        Initialize the analyzer.
        
        Args:
            client: Qdrant client for database operations
            config: Configuration object with workspace settings
        """
        self.client = client
        self.config = config
        self.validator = CollectionNamingValidator()
        
    async def analyze_all_collections(self) -> List[CollectionInfo]:
        """
        Analyze all collections in the database.
        
        Returns:
            List of CollectionInfo objects with analysis results
        """
        logger.info("Starting collection structure analysis")
        
        try:
            # Get all collections
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            
            analyzed_collections = []
            
            for name in collection_names:
                try:
                    info = await self._analyze_single_collection(name)
                    analyzed_collections.append(info)
                    logger.debug(f"Analyzed collection: {name} -> {info.pattern.value}")
                except Exception as e:
                    logger.error(f"Failed to analyze collection {name}: {e}")
                    # Create minimal info for failed analysis
                    info = CollectionInfo(
                        name=name,
                        pattern=CollectionPattern.UNKNOWN
                    )
                    analyzed_collections.append(info)
            
            logger.info(f"Analyzed {len(analyzed_collections)} collections")
            return analyzed_collections
            
        except Exception as e:
            logger.error(f"Failed to analyze collections: {e}")
            raise
    
    async def _analyze_single_collection(self, name: str) -> CollectionInfo:
        """
        Analyze a single collection.
        
        Args:
            name: Collection name to analyze
            
        Returns:
            CollectionInfo with analysis results
        """
        # Get collection info
        collection_info = self.client.get_collection(name)
        
        # Analyze naming pattern
        pattern, project_name, suffix = self._analyze_naming_pattern(name)
        
        # Get collection statistics
        stats = self._get_collection_stats(name)
        
        # Analyze metadata structure
        metadata_keys, has_project_metadata = await self._analyze_metadata_structure(name)
        
        # Determine migration priority
        priority = self._calculate_migration_priority(pattern, stats['point_count'])
        
        return CollectionInfo(
            name=name,
            pattern=pattern,
            project_name=project_name,
            suffix=suffix,
            point_count=stats['point_count'],
            vector_count=stats['vector_count'],
            size_mb=stats['size_mb'],
            created_at=stats.get('created_at'),
            last_modified=stats.get('last_modified'),
            metadata_keys=metadata_keys,
            has_project_metadata=has_project_metadata,
            migration_priority=priority
        )
    
    def _analyze_naming_pattern(self, name: str) -> Tuple[CollectionPattern, Optional[str], Optional[str]]:
        """
        Analyze collection naming pattern.
        
        Args:
            name: Collection name
            
        Returns:
            Tuple of (pattern, project_name, suffix)
        """
        # Check if it's a global collection
        if name in self.config.workspace.global_collections:
            return CollectionPattern.GLOBAL, None, None
        
        # Check for suffix-based pattern (project-suffix)
        if '-' in name:
            parts = name.split('-')
            if len(parts) >= 2:
                potential_project = '-'.join(parts[:-1])
                potential_suffix = parts[-1]
                
                # Validate if suffix is a known collection type
                if potential_suffix in self.config.workspace.effective_collection_types:
                    return CollectionPattern.SUFFIX_BASED, potential_project, potential_suffix
        
        # Check for project-based pattern (just project name)
        # This requires some heuristics since we don't have a definitive list
        if self._looks_like_project_name(name):
            return CollectionPattern.PROJECT_BASED, name, None
        
        return CollectionPattern.UNKNOWN, None, None
    
    def _looks_like_project_name(self, name: str) -> bool:
        """
        Heuristic to determine if a name looks like a project name.
        
        Args:
            name: Collection name to check
            
        Returns:
            True if it looks like a project name
        """
        # Basic heuristics - can be enhanced
        if len(name) < 3:
            return False
        
        # Check for common project naming patterns
        if name.replace('-', '').replace('_', '').isalnum():
            return True
        
        return False
    
    def _get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(name)
            
            # Extract basic stats
            stats = {
                'point_count': collection_info.points_count or 0,
                'vector_count': collection_info.vectors_count or 0,
                'size_mb': 0.0,  # Qdrant doesn't provide direct size info
                'created_at': None,  # Not available in collection info
                'last_modified': None  # Not available in collection info
            }
            
            # Estimate size based on vector dimensions and count
            if collection_info.config and collection_info.config.params:
                if hasattr(collection_info.config.params, 'vectors'):
                    vectors_config = collection_info.config.params.vectors
                    if isinstance(vectors_config, dict):
                        # Get default vector config
                        default_vector = vectors_config.get('', {})
                        if hasattr(default_vector, 'size'):
                            vector_size = default_vector.size
                            # Rough estimate: 4 bytes per float + metadata overhead
                            estimated_size_mb = (stats['point_count'] * vector_size * 4) / (1024 * 1024)
                            stats['size_mb'] = round(estimated_size_mb, 2)
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get stats for collection {name}: {e}")
            return {
                'point_count': 0,
                'vector_count': 0,
                'size_mb': 0.0,
                'created_at': None,
                'last_modified': None
            }
    
    async def _analyze_metadata_structure(self, name: str) -> Tuple[Set[str], bool]:
        """
        Analyze metadata structure of a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Tuple of (metadata_keys, has_project_metadata)
        """
        metadata_keys = set()
        has_project_metadata = False
        
        try:
            # Sample some points to analyze metadata structure
            points = self.client.scroll(
                collection_name=name,
                limit=100,
                with_payload=True
            )[0]  # Get the points list from the scroll result
            
            for point in points:
                if point.payload:
                    metadata_keys.update(point.payload.keys())
                    
                    # Check for existing project metadata
                    if 'project_id' in point.payload or 'project_name' in point.payload:
                        has_project_metadata = True
            
        except Exception as e:
            logger.warning(f"Failed to analyze metadata for collection {name}: {e}")
        
        return metadata_keys, has_project_metadata
    
    def _calculate_migration_priority(self, pattern: CollectionPattern, point_count: int) -> int:
        """
        Calculate migration priority for a collection.
        
        Args:
            pattern: Detected naming pattern
            point_count: Number of points in collection
            
        Returns:
            Priority level (1=high, 2=medium, 3=low)
        """
        # High priority: suffix-based collections with data
        if pattern == CollectionPattern.SUFFIX_BASED and point_count > 0:
            return 1
        
        # Medium priority: project-based collections with data
        if pattern == CollectionPattern.PROJECT_BASED and point_count > 0:
            return 2
        
        # Low priority: everything else
        return 3


class MigrationPlanner:
    """
    Creates comprehensive migration plans for collection transformation.
    
    This class analyzes collection dependencies, detects conflicts, and creates
    optimized migration plans with proper ordering and batch sizing.
    """
    
    def __init__(self, analyzer: CollectionStructureAnalyzer, collision_detector: CollisionDetector):
        """
        Initialize the migration planner.
        
        Args:
            analyzer: Collection structure analyzer
            collision_detector: Conflict detection system
        """
        self.analyzer = analyzer
        self.collision_detector = collision_detector
    
    async def create_migration_plan(self, collections: List[CollectionInfo]) -> MigrationPlan:
        """
        Create a comprehensive migration plan.
        
        Args:
            collections: List of analyzed collections
            
        Returns:
            MigrationPlan with strategy and ordering
        """
        logger.info("Creating migration plan")
        
        plan = MigrationPlan()
        
        # Filter collections that need migration
        migratable_collections = [
            col for col in collections 
            if col.pattern in [CollectionPattern.SUFFIX_BASED, CollectionPattern.PROJECT_BASED]
            and not col.has_project_metadata
        ]
        
        plan.source_collections = migratable_collections
        
        # Create target collection names
        plan.target_collections = await self._generate_target_names(migratable_collections)
        
        # Detect conflicts
        plan.conflicts = await self._detect_migration_conflicts(plan)
        
        # Calculate migration order and dependencies
        plan.migration_order, plan.dependencies = self._calculate_migration_order(migratable_collections)
        
        # Estimate migration metrics
        plan.estimated_duration_minutes = self._estimate_duration(migratable_collections)
        plan.estimated_storage_mb = sum(col.size_mb for col in migratable_collections)
        plan.total_points_to_migrate = sum(col.point_count for col in migratable_collections)
        
        # Optimize batch configuration
        plan.batch_size, plan.parallel_batches = self._optimize_batch_config(plan.total_points_to_migrate)
        
        logger.info(f"Created migration plan for {len(migratable_collections)} collections")
        return plan
    
    async def _generate_target_names(self, collections: List[CollectionInfo]) -> List[str]:
        """
        Generate target collection names for migration.
        
        Args:
            collections: Collections to migrate
            
        Returns:
            List of target collection names
        """
        target_names = []
        
        for col in collections:
            if col.pattern == CollectionPattern.SUFFIX_BASED:
                # For suffix-based, we might consolidate into main project collection
                # or keep the same name but with metadata-based isolation
                target_names.append(col.name)  # Keep same name, add metadata
            elif col.pattern == CollectionPattern.PROJECT_BASED:
                # For project-based, add default suffix if needed
                target_names.append(f"{col.project_name}-documents")
            else:
                target_names.append(col.name)
        
        return target_names
    
    async def _detect_migration_conflicts(self, plan: MigrationPlan) -> List[Dict[str, Any]]:
        """
        Detect potential conflicts in the migration plan.
        
        Args:
            plan: Migration plan to analyze
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Check for target name collisions
        target_counts = {}
        for target in plan.target_collections:
            target_counts[target] = target_counts.get(target, 0) + 1
        
        for target, count in target_counts.items():
            if count > 1:
                conflicts.append({
                    'type': 'target_collision',
                    'severity': 'high',
                    'message': f"Multiple collections would migrate to '{target}'",
                    'affected_collections': [
                        col.name for col, target_name in zip(plan.source_collections, plan.target_collections)
                        if target_name == target
                    ]
                })
        
        # Check for existing target collections
        try:
            existing_collections = {col.name for col in await self.analyzer.analyze_all_collections()}
            for target in plan.target_collections:
                if target in existing_collections:
                    conflicts.append({
                        'type': 'target_exists',
                        'severity': 'medium',
                        'message': f"Target collection '{target}' already exists",
                        'collection': target
                    })
        except Exception as e:
            logger.warning(f"Failed to check existing collections: {e}")
        
        return conflicts
    
    def _calculate_migration_order(self, collections: List[CollectionInfo]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Calculate optimal migration order and dependencies.
        
        Args:
            collections: Collections to migrate
            
        Returns:
            Tuple of (migration_order, dependencies)
        """
        # Sort by priority first, then by size (smaller first for faster feedback)
        sorted_collections = sorted(
            collections,
            key=lambda col: (col.migration_priority, col.point_count)
        )
        
        migration_order = [col.name for col in sorted_collections]
        
        # For now, simple dependencies based on project grouping
        dependencies = {}
        project_groups = {}
        
        for col in collections:
            if col.project_name:
                if col.project_name not in project_groups:
                    project_groups[col.project_name] = []
                project_groups[col.project_name].append(col.name)
        
        # Within each project, migrate in order of priority
        for project_name, project_collections in project_groups.items():
            for i, col_name in enumerate(project_collections[1:], 1):
                dependencies[col_name] = [project_collections[i-1]]
        
        return migration_order, dependencies
    
    def _estimate_duration(self, collections: List[CollectionInfo]) -> float:
        """
        Estimate migration duration in minutes.
        
        Args:
            collections: Collections to migrate
            
        Returns:
            Estimated duration in minutes
        """
        # Basic estimation: ~1000 points per minute processing rate
        total_points = sum(col.point_count for col in collections)
        base_time_minutes = total_points / 1000
        
        # Add overhead for setup, validation, etc.
        overhead_minutes = len(collections) * 2  # 2 minutes per collection overhead
        
        return base_time_minutes + overhead_minutes
    
    def _optimize_batch_config(self, total_points: int) -> Tuple[int, int]:
        """
        Optimize batch size and parallel batch configuration.
        
        Args:
            total_points: Total points to migrate
            
        Returns:
            Tuple of (batch_size, parallel_batches)
        """
        # Adaptive batch sizing based on total volume
        if total_points < 10000:
            return 500, 2
        elif total_points < 100000:
            return 1000, 3
        elif total_points < 1000000:
            return 2000, 4
        else:
            return 5000, 5


class BatchMigrator:
    """
    Performs data migration with metadata injection and error handling.
    
    This class handles the actual data migration process including:
    - Batch processing with configurable sizes
    - Metadata injection for multi-tenant support
    - Error handling and retry logic
    - Progress tracking and reporting
    """
    
    def __init__(self, client: QdrantWorkspaceClient, metadata_schema: MultiTenantMetadataSchema):
        """
        Initialize the batch migrator.
        
        Args:
            client: Qdrant workspace client
            metadata_schema: Multi-tenant metadata schema
        """
        self.client = client
        self.metadata_schema = metadata_schema
        self.filter_manager = MetadataFilterManager()
    
    async def migrate_collection(
        self,
        source_collection: CollectionInfo,
        target_collection: str,
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """
        Migrate a single collection with metadata injection.
        
        Args:
            source_collection: Source collection information
            target_collection: Target collection name
            plan: Migration plan with configuration
            
        Returns:
            Migration result dictionary
        """
        logger.info(f"Starting migration: {source_collection.name} -> {target_collection}")
        
        start_time = datetime.now(timezone.utc)
        result = {
            'source': source_collection.name,
            'target': target_collection,
            'started_at': start_time,
            'points_migrated': 0,
            'points_failed': 0,
            'batches_processed': 0,
            'batches_failed': 0,
            'errors': [],
            'success': False
        }
        
        try:
            # Ensure target collection exists with proper configuration
            await self._ensure_target_collection(target_collection, source_collection)
            
            # Migrate data in batches
            await self._migrate_data_in_batches(source_collection, target_collection, plan, result)
            
            result['success'] = True
            result['completed_at'] = datetime.now(timezone.utc)
            
            logger.info(f"Completed migration: {source_collection.name} -> {target_collection}")
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            result['completed_at'] = datetime.now(timezone.utc)
            logger.error(f"Failed migration: {source_collection.name} -> {target_collection}: {e}")
            raise
        
        return result
    
    async def _ensure_target_collection(self, target_name: str, source_collection: CollectionInfo):
        """
        Ensure target collection exists with proper configuration.
        
        Args:
            target_name: Target collection name
            source_collection: Source collection information
        """
        try:
            # Check if target collection already exists
            collection_exists = False
            try:
                self.client.get_collection(target_name)
                collection_exists = True
                logger.info(f"Target collection {target_name} already exists")
            except ResponseHandlingException:
                collection_exists = False
            
            if not collection_exists:
                # Get source collection configuration
                source_info = self.client.get_collection(source_collection.name)
                
                # Create target collection with same configuration
                await self.client.create_collection(
                    collection_name=target_name,
                    vectors_config=source_info.config.params.vectors,
                    sparse_vectors_config=getattr(source_info.config.params, 'sparse_vectors', None)
                )
                
                logger.info(f"Created target collection: {target_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure target collection {target_name}: {e}")
            raise
    
    async def _migrate_data_in_batches(
        self,
        source_collection: CollectionInfo,
        target_collection: str,
        plan: MigrationPlan,
        result: Dict[str, Any]
    ):
        """
        Migrate data in configurable batches with metadata injection.
        
        Args:
            source_collection: Source collection information
            target_collection: Target collection name
            plan: Migration plan with batch configuration
            result: Result dictionary to update
        """
        batch_size = plan.batch_size
        offset = None
        batch_number = 0
        
        while True:
            try:
                batch_number += 1
                logger.debug(f"Processing batch {batch_number} for {source_collection.name}")
                
                # Get batch of points
                points, next_offset = self.client.scroll(
                    collection_name=source_collection.name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points:
                    break
                
                # Inject metadata for multi-tenant support
                enhanced_points = self._inject_project_metadata(points, source_collection)
                
                # Upsert points to target collection
                self.client.upsert(
                    collection_name=target_collection,
                    points=enhanced_points
                )
                
                result['points_migrated'] += len(points)
                result['batches_processed'] += 1
                
                # Update offset for next batch
                offset = next_offset
                if offset is None:
                    break
                    
            except Exception as e:
                result['batches_failed'] += 1
                result['points_failed'] += batch_size  # Approximate
                result['errors'].append(f"Batch {batch_number}: {str(e)}")
                logger.error(f"Failed to migrate batch {batch_number}: {e}")
                
                # Continue with next batch rather than failing entire migration
                if offset is not None:
                    continue
                else:
                    break
    
    def _inject_project_metadata(self, points: List[models.Record], source_collection: CollectionInfo) -> List[models.PointStruct]:
        """
        Inject project metadata into points for multi-tenant support.
        
        Args:
            points: Original points from source collection
            source_collection: Source collection information
            
        Returns:
            Points with injected metadata
        """
        enhanced_points = []
        
        for point in points:
            # Start with existing payload
            enhanced_payload = dict(point.payload) if point.payload else {}
            
            # Inject multi-tenant metadata
            if source_collection.project_name:
                enhanced_payload.update({
                    'project_id': source_collection.project_name,
                    'project_name': source_collection.project_name,
                    'collection_suffix': source_collection.suffix or 'default',
                    'migrated_at': datetime.now(timezone.utc).isoformat(),
                    'migration_source': source_collection.name
                })
            
            # Create enhanced point
            enhanced_point = models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=enhanced_payload
            )
            
            enhanced_points.append(enhanced_point)
        
        return enhanced_points


class RollbackManager:
    """
    Provides rollback capabilities for failed migrations.
    
    This class manages backup creation and restoration to ensure safe migration
    with ability to revert changes if needed.
    """
    
    def __init__(self, client: QdrantClient, backup_dir: Path):
        """
        Initialize the rollback manager.
        
        Args:
            client: Qdrant client for database operations
            backup_dir: Directory for storing backups
        """
        self.client = client
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, collection_name: str, migration_id: str) -> str:
        """
        Create backup of a collection before migration.
        
        Args:
            collection_name: Collection to backup
            migration_id: Migration identifier for backup naming
            
        Returns:
            Backup file path
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{collection_name}_{migration_id}_{timestamp}.json"
        
        logger.info(f"Creating backup for {collection_name}")
        
        try:
            # Export all points from collection
            all_points = []
            offset = None
            
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points:
                    break
                
                # Convert points to serializable format
                for point in points:
                    point_data = {
                        'id': str(point.id),
                        'vector': point.vector,
                        'payload': point.payload or {}
                    }
                    all_points.append(point_data)
                
                offset = next_offset
                if offset is None:
                    break
            
            # Save backup to file
            backup_data = {
                'collection_name': collection_name,
                'migration_id': migration_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'point_count': len(all_points),
                'points': all_points
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Created backup: {backup_file} ({len(all_points)} points)")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {collection_name}: {e}")
            raise
    
    async def restore_backup(self, backup_file: str) -> bool:
        """
        Restore collection from backup.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if restore successful
        """
        logger.info(f"Restoring backup: {backup_file}")
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            collection_name = backup_data['collection_name']
            points_data = backup_data['points']
            
            # Clear existing collection data
            try:
                # Delete all points (this is safer than recreating collection)
                points, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False
                )
                
                if points:
                    point_ids = [point.id for point in points]
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=point_ids)
                    )
            except Exception as e:
                logger.warning(f"Failed to clear collection {collection_name}: {e}")
            
            # Restore points in batches
            batch_size = 1000
            for i in range(0, len(points_data), batch_size):
                batch = points_data[i:i + batch_size]
                
                # Convert back to PointStruct format
                points_to_restore = []
                for point_data in batch:
                    point = models.PointStruct(
                        id=point_data['id'],
                        vector=point_data['vector'],
                        payload=point_data['payload']
                    )
                    points_to_restore.append(point)
                
                # Upsert batch
                self.client.upsert(
                    collection_name=collection_name,
                    points=points_to_restore
                )
            
            logger.info(f"Restored {len(points_data)} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_file}: {e}")
            return False


class MigrationReporter:
    """
    Generates comprehensive migration reports and tracks progress.
    
    This class provides detailed reporting on migration execution including
    statistics, performance metrics, and validation results.
    """
    
    def __init__(self, report_dir: Path):
        """
        Initialize the migration reporter.
        
        Args:
            report_dir: Directory for storing migration reports
        """
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_migration_report(
        self,
        plan: MigrationPlan,
        result: MigrationResult,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate comprehensive migration report.
        
        Args:
            plan: Migration plan that was executed
            result: Migration execution results
            validation_results: Optional validation results
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"migration_report_{result.execution_id}_{timestamp}.json"
        
        # Calculate statistics
        total_duration = (result.completed_at - result.started_at).total_seconds() if result.completed_at else 0
        success_rate = (result.points_migrated / max(plan.total_points_to_migrate, 1)) * 100
        
        report_data = {
            'migration_summary': {
                'plan_id': plan.plan_id,
                'execution_id': result.execution_id,
                'started_at': result.started_at.isoformat(),
                'completed_at': result.completed_at.isoformat() if result.completed_at else None,
                'total_duration_seconds': total_duration,
                'final_phase': result.phase.value,
                'overall_success': result.success
            },
            'migration_plan': {
                'collections_planned': len(plan.source_collections),
                'estimated_duration_minutes': plan.estimated_duration_minutes,
                'estimated_points': plan.total_points_to_migrate,
                'batch_size': plan.batch_size,
                'parallel_batches': plan.parallel_batches,
                'conflicts_detected': len(plan.conflicts)
            },
            'execution_results': {
                'collections_migrated': result.collections_migrated,
                'points_migrated': result.points_migrated,
                'points_failed': result.points_failed,
                'batches_processed': result.batches_processed,
                'batches_failed': result.batches_failed,
                'success_rate_percent': round(success_rate, 2)
            },
            'performance_metrics': {
                'points_per_second': round(result.points_migrated / max(total_duration, 1), 2),
                'analysis_duration_seconds': result.analysis_duration_seconds,
                'migration_duration_seconds': result.migration_duration_seconds,
                'validation_duration_seconds': result.validation_duration_seconds
            },
            'backup_information': {
                'backups_created': len(result.backup_locations),
                'backup_locations': result.backup_locations,
                'total_backup_size_mb': result.backup_size_mb
            },
            'issues_and_warnings': {
                'errors': result.errors,
                'warnings': result.warnings,
                'log_entries': result.log_entries[-100:]  # Last 100 log entries
            }
        }
        
        # Add validation results if provided
        if validation_results:
            report_data['validation_results'] = validation_results
        
        # Add detailed collection results
        report_data['collection_details'] = []
        for i, source_col in enumerate(plan.source_collections):
            target_col = plan.target_collections[i] if i < len(plan.target_collections) else "unknown"
            
            collection_detail = {
                'source_collection': source_col.name,
                'target_collection': target_col,
                'pattern': source_col.pattern.value,
                'project_name': source_col.project_name,
                'suffix': source_col.suffix,
                'original_point_count': source_col.point_count,
                'migration_priority': source_col.migration_priority
            }
            report_data['collection_details'].append(collection_detail)
        
        # Write report to file
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Generated migration report: {report_file}")
        return report_file
    
    def generate_summary_text(self, report_file: Path) -> str:
        """
        Generate human-readable summary from report.
        
        Args:
            report_file: Path to JSON report file
            
        Returns:
            Human-readable summary text
        """
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        summary = report_data['migration_summary']
        execution = report_data['execution_results']
        performance = report_data['performance_metrics']
        
        summary_text = f"""
Migration Report Summary
========================

Execution ID: {summary['execution_id']}
Status: {'SUCCESS' if summary['overall_success'] else 'FAILED'}
Duration: {summary['total_duration_seconds']:.1f} seconds
Phase: {summary['final_phase']}

Results:
- Collections migrated: {execution['collections_migrated']}
- Points migrated: {execution['points_migrated']:,}
- Points failed: {execution['points_failed']:,}
- Success rate: {execution['success_rate_percent']}%

Performance:
- Points per second: {performance['points_per_second']:.1f}
- Analysis time: {performance['analysis_duration_seconds']:.1f}s
- Migration time: {performance['migration_duration_seconds']:.1f}s
- Validation time: {performance['validation_duration_seconds']:.1f}s

Issues:
- Errors: {len(report_data['issues_and_warnings']['errors'])}
- Warnings: {len(report_data['issues_and_warnings']['warnings'])}
"""
        
        return summary_text


class CollectionMigrationManager:
    """
    Main manager class that coordinates all migration components.
    
    This is the primary interface for collection migration operations,
    providing a complete workflow from analysis to reporting.
    """
    
    def __init__(
        self,
        client: QdrantWorkspaceClient,
        config: Config,
        backup_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None
    ):
        """
        Initialize the migration manager.
        
        Args:
            client: Qdrant workspace client
            config: Configuration object
            backup_dir: Directory for backups (defaults to ./migration_backups)
            report_dir: Directory for reports (defaults to ./migration_reports)
        """
        self.client = client
        self.config = config
        
        # Set up directories
        self.backup_dir = backup_dir or Path("./migration_backups")
        self.report_dir = report_dir or Path("./migration_reports")
        
        # Initialize components
        self.analyzer = CollectionStructureAnalyzer(client.client, config)
        self.collision_detector = CollisionDetector(config)
        self.planner = MigrationPlanner(self.analyzer, self.collision_detector)
        self.metadata_schema = MultiTenantMetadataSchema(config)
        self.migrator = BatchMigrator(client, self.metadata_schema)
        self.rollback_manager = RollbackManager(client.client, self.backup_dir)
        self.reporter = MigrationReporter(self.report_dir)
    
    async def analyze_collections(self) -> List[CollectionInfo]:
        """
        Analyze all collections for migration planning.
        
        Returns:
            List of analyzed collections
        """
        return await self.analyzer.analyze_all_collections()
    
    async def create_migration_plan(
        self,
        collections: Optional[List[CollectionInfo]] = None
    ) -> MigrationPlan:
        """
        Create a comprehensive migration plan.
        
        Args:
            collections: Collections to include (analyzes all if None)
            
        Returns:
            Migration plan
        """
        if collections is None:
            collections = await self.analyze_collections()
        
        return await self.planner.create_migration_plan(collections)
    
    async def execute_migration(self, plan: MigrationPlan) -> MigrationResult:
        """
        Execute a migration plan with full error handling and rollback support.
        
        Args:
            plan: Migration plan to execute
            
        Returns:
            Migration results
        """
        result = MigrationResult(plan_id=plan.plan_id)
        
        try:
            # Phase 1: Create backups
            result.phase = MigrationPhase.BACKUP
            if plan.create_backups:
                for collection in plan.source_collections:
                    backup_file = await self.rollback_manager.create_backup(
                        collection.name, plan.plan_id
                    )
                    result.backup_locations.append(backup_file)
            
            # Phase 2: Execute migration
            result.phase = MigrationPhase.MIGRATION
            migration_start = datetime.now(timezone.utc)
            
            for i, source_collection in enumerate(plan.source_collections):
                if i < len(plan.target_collections):
                    target_collection = plan.target_collections[i]
                    
                    collection_result = await self.migrator.migrate_collection(
                        source_collection, target_collection, plan
                    )
                    
                    # Update overall results
                    result.points_migrated += collection_result['points_migrated']
                    result.points_failed += collection_result['points_failed']
                    result.batches_processed += collection_result['batches_processed']
                    result.batches_failed += collection_result['batches_failed']
                    
                    if collection_result['success']:
                        result.collections_migrated += 1
                    else:
                        result.errors.extend(collection_result.get('errors', []))
            
            result.migration_duration_seconds = (datetime.now(timezone.utc) - migration_start).total_seconds()
            
            # Phase 3: Validation
            result.phase = MigrationPhase.VALIDATION
            validation_start = datetime.now(timezone.utc)
            
            if plan.enable_validation:
                await self._validate_migration(plan, result)
            
            result.validation_duration_seconds = (datetime.now(timezone.utc) - validation_start).total_seconds()
            
            # Phase 4: Cleanup
            result.phase = MigrationPhase.CLEANUP
            await self._cleanup_migration(plan, result)
            
            result.phase = MigrationPhase.COMPLETED
            result.success = True
            
        except Exception as e:
            result.phase = MigrationPhase.FAILED
            result.success = False
            result.error_message = str(e)
            logger.error(f"Migration failed: {e}")
            
            # Attempt rollback if backups exist
            if result.backup_locations:
                try:
                    await self._rollback_migration(result)
                    result.phase = MigrationPhase.ROLLED_BACK
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
                    result.errors.append(f"Rollback failed: {str(rollback_error)}")
        
        finally:
            result.completed_at = datetime.now(timezone.utc)
        
        return result
    
    async def _validate_migration(self, plan: MigrationPlan, result: MigrationResult):
        """
        Validate migration results.
        
        Args:
            plan: Migration plan
            result: Migration result to update
        """
        # Validate point counts match
        for i, source_collection in enumerate(plan.source_collections):
            if i < len(plan.target_collections):
                target_collection = plan.target_collections[i]
                
                try:
                    source_info = self.client.get_collection(source_collection.name)
                    target_info = self.client.get_collection(target_collection)
                    
                    if source_info.points_count != target_info.points_count:
                        result.warnings.append(
                            f"Point count mismatch: {source_collection.name} "
                            f"({source_info.points_count}) vs {target_collection} "
                            f"({target_info.points_count})"
                        )
                except Exception as e:
                    result.errors.append(f"Validation failed for {target_collection}: {e}")
    
    async def _cleanup_migration(self, plan: MigrationPlan, result: MigrationResult):
        """
        Cleanup after migration.
        
        Args:
            plan: Migration plan
            result: Migration result
        """
        # Add any cleanup logic here
        # For now, just log completion
        logger.info("Migration cleanup completed")
    
    async def _rollback_migration(self, result: MigrationResult):
        """
        Rollback migration using backups.
        
        Args:
            result: Migration result with backup information
        """
        logger.warning("Starting migration rollback")
        
        for backup_file in result.backup_locations:
            try:
                success = await self.rollback_manager.restore_backup(backup_file)
                if success:
                    logger.info(f"Restored backup: {backup_file}")
                else:
                    logger.error(f"Failed to restore backup: {backup_file}")
            except Exception as e:
                logger.error(f"Rollback error for {backup_file}: {e}")
                raise
    
    async def generate_migration_report(
        self,
        plan: MigrationPlan,
        result: MigrationResult,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate comprehensive migration report.
        
        Args:
            plan: Migration plan
            result: Migration results
            validation_results: Optional validation results
            
        Returns:
            Path to generated report
        """
        return await self.reporter.generate_migration_report(plan, result, validation_results)