#!/usr/bin/env python3
"""
Migration Example Script

This script demonstrates how to use the collection migration utilities
to transform suffix-based collections to multi-tenant architecture.

This example shows:
- Collection analysis and pattern detection
- Migration plan creation with customization
- Safe migration execution with monitoring
- Validation and reporting
- Error handling and rollback procedures

Usage:
    python migration_example.py --mode analyze
    python migration_example.py --mode migrate --confirm
    python migration_example.py --mode rollback --backup-file backup.json
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import migration utilities
try:
    from src.python.common.core.client import QdrantWorkspaceClient
    from src.python.common.core.config import Config
    from src.python.common.memory.migration_utils import (
        CollectionMigrationManager,
        CollectionInfo,
        CollectionPattern,
        MigrationPlan,
        MigrationResult,
        MigrationPhase
    )
except ImportError as e:
    logger.error(f"Failed to import migration utilities: {e}")
    logger.error("Please ensure you're running from the project root directory")
    sys.exit(1)


class MigrationExample:
    """
    Example class demonstrating migration utilities usage.
    
    This class provides a complete example of how to:
    - Analyze existing collections
    - Create and customize migration plans
    - Execute migrations safely
    - Handle errors and perform rollbacks
    - Generate reports and validate results
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the migration example.
        
        Args:
            config_file: Optional path to configuration file
        """
        # Load configuration
        self.config = Config()
        if config_file:
            self.config.load_from_file(config_file)
        
        # Initialize directories
        self.backup_dir = Path("./migration_backups")
        self.report_dir = Path("./migration_reports")
        self.output_dir = Path("./migration_output")
        
        # Create directories
        for directory in [self.backup_dir, self.report_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize client and manager
        self.client = None
        self.manager = None
        
    async def initialize(self):
        """Initialize the Qdrant client and migration manager."""
        try:
            logger.info("Initializing Qdrant client...")
            self.client = QdrantWorkspaceClient(self.config)
            await self.client.initialize()
            
            logger.info("Initializing migration manager...")
            self.manager = CollectionMigrationManager(
                self.client,
                self.config,
                backup_dir=self.backup_dir,
                report_dir=self.report_dir
            )
            
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def analyze_collections(self, save_results: bool = True) -> List[CollectionInfo]:
        """
        Analyze existing collections to identify migration candidates.
        
        Args:
            save_results: Whether to save analysis results to file
            
        Returns:
            List of analyzed collections
        """
        logger.info("Starting collection analysis...")
        
        try:
            # Analyze all collections
            collections = await self.manager.analyze_collections()
            
            # Categorize collections by pattern
            patterns = {}
            for collection in collections:
                pattern = collection.pattern.value
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(collection)
            
            # Log summary
            logger.info(f"Analysis complete: {len(collections)} collections found")
            for pattern, pattern_collections in patterns.items():
                logger.info(f"  {pattern}: {len(pattern_collections)} collections")
            
            # Display detailed results
            self._display_collections(collections)
            
            # Save results if requested
            if save_results:
                output_file = self.output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self._save_analysis_results(collections, output_file)
                logger.info(f"Analysis results saved to: {output_file}")
            
            return collections
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def create_migration_plan(
        self,
        collections: Optional[List[CollectionInfo]] = None,
        customize_plan: bool = True,
        save_plan: bool = True
    ) -> MigrationPlan:
        """
        Create a migration plan for the collections.
        
        Args:
            collections: Collections to migrate (analyzes all if None)
            customize_plan: Whether to apply custom optimizations
            save_plan: Whether to save plan to file
            
        Returns:
            Migration plan
        """
        logger.info("Creating migration plan...")
        
        try:
            # Use provided collections or analyze fresh
            if collections is None:
                collections = await self.analyze_collections(save_results=False)
            
            # Filter collections that need migration
            migratable = [
                col for col in collections
                if col.pattern in [CollectionPattern.SUFFIX_BASED, CollectionPattern.PROJECT_BASED]
                and not col.has_project_metadata
                and col.point_count > 0  # Only migrate collections with data
            ]
            
            if not migratable:
                logger.warning("No collections found that need migration")
                return None
            
            logger.info(f"Found {len(migratable)} collections requiring migration")
            
            # Create migration plan
            plan = await self.manager.create_migration_plan(migratable)
            
            # Apply customizations if requested
            if customize_plan:
                plan = self._customize_migration_plan(plan)
            
            # Display plan summary
            self._display_migration_plan(plan)
            
            # Save plan if requested
            if save_plan:
                output_file = self.output_dir / f"migration_plan_{plan.plan_id}.json"
                self._save_migration_plan(plan, output_file)
                logger.info(f"Migration plan saved to: {output_file}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            raise
    
    async def execute_migration(
        self,
        plan: Optional[MigrationPlan] = None,
        dry_run: bool = False,
        monitor_progress: bool = True
    ) -> MigrationResult:
        """
        Execute the migration plan.
        
        Args:
            plan: Migration plan to execute (creates one if None)
            dry_run: Whether to perform a dry run without actual migration
            monitor_progress: Whether to monitor and log progress
            
        Returns:
            Migration result
        """
        logger.info(f"{'Starting dry run' if dry_run else 'Starting migration execution'}...")
        
        try:
            # Create plan if not provided
            if plan is None:
                plan = await self.create_migration_plan()
                if plan is None:
                    logger.error("No migration plan available")
                    return None
            
            if dry_run:
                logger.info("DRY RUN MODE - No actual changes will be made")
                self._display_migration_plan(plan, dry_run=True)
                return None
            
            # Execute migration with progress monitoring
            if monitor_progress:
                result = await self._execute_with_monitoring(plan)
            else:
                result = await self.manager.execute_migration(plan)
            
            # Display results
            self._display_migration_result(result)
            
            # Generate and save report
            await self._generate_migration_report(plan, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            raise
    
    async def validate_migration(
        self,
        result: MigrationResult,
        plan: MigrationPlan
    ) -> bool:
        """
        Validate migration results.
        
        Args:
            result: Migration execution result
            plan: Original migration plan
            
        Returns:
            True if validation passes
        """
        logger.info("Validating migration results...")
        
        try:
            validation_passed = True
            
            # Check overall success
            if not result.success:
                logger.error("Migration marked as failed")
                validation_passed = False
            
            # Validate point counts
            for i, source_collection in enumerate(plan.source_collections):
                if i < len(plan.target_collections):
                    target_collection = plan.target_collections[i]
                    
                    try:
                        source_info = self.client.get_collection(source_collection.name)
                        target_info = self.client.get_collection(target_collection)
                        
                        if source_info.points_count != target_info.points_count:
                            logger.error(
                                f"Point count mismatch: {source_collection.name} "
                                f"({source_info.points_count}) vs {target_collection} "
                                f"({target_info.points_count})"
                            )
                            validation_passed = False
                        else:
                            logger.info(
                                f"✓ Point count validated: {source_collection.name} -> {target_collection}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to validate {target_collection}: {e}")
                        validation_passed = False
            
            # Validate metadata injection
            for target_collection in plan.target_collections:
                try:
                    # Sample some points to check metadata
                    points, _ = self.client.scroll(
                        collection_name=target_collection,
                        limit=10,
                        with_payload=True
                    )
                    
                    metadata_found = False
                    for point in points:
                        if point.payload and 'project_id' in point.payload:
                            metadata_found = True
                            break
                    
                    if not metadata_found:
                        logger.warning(f"No project metadata found in {target_collection}")
                    else:
                        logger.info(f"✓ Metadata validated: {target_collection}")
                        
                except Exception as e:
                    logger.error(f"Failed to validate metadata for {target_collection}: {e}")
                    validation_passed = False
            
            if validation_passed:
                logger.info("✓ Migration validation passed")
            else:
                logger.error("✗ Migration validation failed")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    async def perform_rollback(self, backup_file: str) -> bool:
        """
        Perform rollback using a backup file.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if rollback successful
        """
        logger.info(f"Starting rollback from backup: {backup_file}")
        
        try:
            # Load backup info
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            collection_name = backup_data['collection_name']
            point_count = backup_data['point_count']
            
            logger.info(f"Rolling back collection: {collection_name} ({point_count} points)")
            
            # Perform rollback
            success = await self.manager.rollback_manager.restore_backup(backup_file)
            
            if success:
                logger.info(f"✓ Rollback successful for {collection_name}")
            else:
                logger.error(f"✗ Rollback failed for {collection_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def cleanup(self, days: int = 30):
        """
        Clean up old migration artifacts.
        
        Args:
            days: Remove files older than this many days
        """
        logger.info(f"Cleaning up migration artifacts older than {days} days...")
        
        try:
            import time
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            total_cleaned = 0
            total_size = 0
            
            for directory in [self.backup_dir, self.report_dir, self.output_dir]:
                if directory.exists():
                    for file_path in directory.iterdir():
                        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            total_cleaned += 1
                            total_size += size
                            logger.debug(f"Removed: {file_path}")
            
            size_mb = total_size / (1024 * 1024)
            logger.info(f"Cleanup complete: {total_cleaned} files removed ({size_mb:.2f} MB freed)")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _customize_migration_plan(self, plan: MigrationPlan) -> MigrationPlan:
        """
        Apply custom optimizations to the migration plan.
        
        Args:
            plan: Original migration plan
            
        Returns:
            Customized migration plan
        """
        logger.info("Applying custom plan optimizations...")
        
        # Calculate optimal batch size based on total data volume
        total_points = plan.total_points_to_migrate
        
        if total_points < 10000:
            plan.batch_size = 500
            plan.parallel_batches = 2
        elif total_points < 100000:
            plan.batch_size = 1000
            plan.parallel_batches = 3
        elif total_points < 1000000:
            plan.batch_size = 2000
            plan.parallel_batches = 4
        else:
            plan.batch_size = 5000
            plan.parallel_batches = 6
        
        # Enable comprehensive validation for production
        plan.enable_validation = True
        
        # Always create backups for safety
        plan.create_backups = True
        
        logger.info(f"Optimized batch size: {plan.batch_size}")
        logger.info(f"Parallel batches: {plan.parallel_batches}")
        
        return plan
    
    async def _execute_with_monitoring(self, plan: MigrationPlan) -> MigrationResult:
        """
        Execute migration with detailed progress monitoring.
        
        Args:
            plan: Migration plan to execute
            
        Returns:
            Migration result
        """
        logger.info("Starting monitored migration execution...")
        
        # Start migration in background
        migration_task = asyncio.create_task(
            self.manager.execute_migration(plan)
        )
        
        # Monitor progress
        start_time = datetime.now()
        last_check = start_time
        
        while not migration_task.done():
            await asyncio.sleep(10)  # Check every 10 seconds
            
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            since_last = (current_time - last_check).total_seconds()
            
            logger.info(f"Migration running... ({elapsed:.0f}s elapsed)")
            last_check = current_time
        
        # Get result
        result = await migration_task
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Migration completed in {total_time:.1f} seconds")
        
        return result
    
    async def _generate_migration_report(
        self,
        plan: MigrationPlan,
        result: MigrationResult
    ):
        """
        Generate and save comprehensive migration report.
        
        Args:
            plan: Migration plan that was executed
            result: Migration execution result
        """
        logger.info("Generating migration report...")
        
        try:
            # Generate report
            report_file = await self.manager.generate_migration_report(plan, result)
            
            # Generate human-readable summary
            summary = self.manager.reporter.generate_summary_text(report_file)
            
            # Save summary to separate file
            summary_file = self.report_dir / f"migration_summary_{result.execution_id}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            logger.info(f"Migration report saved to: {report_file}")
            logger.info(f"Migration summary saved to: {summary_file}")
            
            # Log key metrics
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            execution = report_data['execution_results']
            performance = report_data['performance_metrics']
            
            logger.info("Migration Summary:")
            logger.info(f"  Collections migrated: {execution['collections_migrated']}")
            logger.info(f"  Points migrated: {execution['points_migrated']:,}")
            logger.info(f"  Success rate: {execution['success_rate_percent']}%")
            logger.info(f"  Throughput: {performance['points_per_second']:.1f} points/sec")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def _display_collections(self, collections: List[CollectionInfo]):
        """Display collection analysis results."""
        logger.info("\nCollection Analysis Results:")
        logger.info("-" * 80)
        
        for collection in sorted(collections, key=lambda c: (c.pattern.value, c.name)):
            logger.info(f"Collection: {collection.name}")
            logger.info(f"  Pattern: {collection.pattern.value}")
            logger.info(f"  Project: {collection.project_name or 'N/A'}")
            logger.info(f"  Suffix: {collection.suffix or 'N/A'}")
            logger.info(f"  Points: {collection.point_count:,}")
            logger.info(f"  Size: {collection.size_mb:.2f} MB")
            logger.info(f"  Priority: {collection.migration_priority}")
            logger.info(f"  Has Metadata: {'Yes' if collection.has_project_metadata else 'No'}")
            logger.info("")
    
    def _display_migration_plan(self, plan: MigrationPlan, dry_run: bool = False):
        """Display migration plan details."""
        title = "Migration Plan (DRY RUN)" if dry_run else "Migration Plan"
        logger.info(f"\n{title}:")
        logger.info("-" * 80)
        logger.info(f"Plan ID: {plan.plan_id}")
        logger.info(f"Collections to migrate: {len(plan.source_collections)}")
        logger.info(f"Total points: {plan.total_points_to_migrate:,}")
        logger.info(f"Estimated duration: {plan.estimated_duration_minutes:.1f} minutes")
        logger.info(f"Estimated storage: {plan.estimated_storage_mb:.1f} MB")
        logger.info(f"Batch size: {plan.batch_size}")
        logger.info(f"Parallel batches: {plan.parallel_batches}")
        logger.info(f"Create backups: {'Yes' if plan.create_backups else 'No'}")
        
        if plan.conflicts:
            logger.warning(f"\nConflicts detected ({len(plan.conflicts)}):")
            for conflict in plan.conflicts:
                logger.warning(f"  - {conflict['severity'].upper()}: {conflict['message']}")
        
        logger.info("\nCollections to migrate:")
        for i, source_col in enumerate(plan.source_collections):
            target_col = plan.target_collections[i] if i < len(plan.target_collections) else "N/A"
            logger.info(f"  {source_col.name} -> {target_col} ({source_col.point_count:,} points)")
    
    def _display_migration_result(self, result: MigrationResult):
        """Display migration execution results."""
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(f"\nMigration Result: {status}")
        logger.info("-" * 80)
        logger.info(f"Execution ID: {result.execution_id}")
        logger.info(f"Phase: {result.phase.value}")
        logger.info(f"Collections migrated: {result.collections_migrated}")
        logger.info(f"Points migrated: {result.points_migrated:,}")
        logger.info(f"Points failed: {result.points_failed:,}")
        logger.info(f"Batches processed: {result.batches_processed}")
        logger.info(f"Batches failed: {result.batches_failed}")
        logger.info(f"Backups created: {len(result.backup_locations)}")
        
        if result.completed_at and result.started_at:
            duration = (result.completed_at - result.started_at).total_seconds()
            logger.info(f"Duration: {duration:.1f} seconds")
        
        if result.errors:
            logger.error(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        if result.warnings:
            logger.warning(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
    
    def _save_analysis_results(self, collections: List[CollectionInfo], output_file: Path):
        """Save analysis results to JSON file."""
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_collections': len(collections),
            'collections': [
                {
                    'name': col.name,
                    'pattern': col.pattern.value,
                    'project_name': col.project_name,
                    'suffix': col.suffix,
                    'point_count': col.point_count,
                    'vector_count': col.vector_count,
                    'size_mb': col.size_mb,
                    'created_at': col.created_at.isoformat() if col.created_at else None,
                    'last_modified': col.last_modified.isoformat() if col.last_modified else None,
                    'metadata_keys': list(col.metadata_keys),
                    'has_project_metadata': col.has_project_metadata,
                    'migration_priority': col.migration_priority
                }
                for col in collections
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_migration_plan(self, plan: MigrationPlan, output_file: Path):
        """Save migration plan to JSON file."""
        plan_data = {
            'plan_id': plan.plan_id,
            'created_at': plan.created_at.isoformat(),
            'source_collections': [
                {
                    'name': col.name,
                    'pattern': col.pattern.value,
                    'project_name': col.project_name,
                    'suffix': col.suffix,
                    'point_count': col.point_count,
                    'migration_priority': col.migration_priority
                }
                for col in plan.source_collections
            ],
            'target_collections': plan.target_collections,
            'batch_size': plan.batch_size,
            'parallel_batches': plan.parallel_batches,
            'enable_validation': plan.enable_validation,
            'create_backups': plan.create_backups,
            'conflicts': plan.conflicts,
            'estimated_duration_minutes': plan.estimated_duration_minutes,
            'estimated_storage_mb': plan.estimated_storage_mb,
            'total_points_to_migrate': plan.total_points_to_migrate,
            'migration_order': plan.migration_order,
            'dependencies': plan.dependencies
        }
        
        with open(output_file, 'w') as f:
            json.dump(plan_data, f, indent=2)


async def main():
    """Main entry point for the migration example."""
    parser = argparse.ArgumentParser(description='Collection Migration Example')
    parser.add_argument(
        '--mode',
        choices=['analyze', 'plan', 'migrate', 'validate', 'rollback', 'cleanup'],
        required=True,
        help='Migration operation to perform'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations without prompting'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without making changes'
    )
    parser.add_argument(
        '--backup-file',
        type=str,
        help='Backup file for rollback operation'
    )
    parser.add_argument(
        '--plan-file',
        type=str,
        help='Migration plan file to use'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days for cleanup operation'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize migration example
        example = MigrationExample(args.config)
        await example.initialize()
        
        # Execute requested operation
        if args.mode == 'analyze':
            collections = await example.analyze_collections()
            logger.info(f"Analysis complete: {len(collections)} collections analyzed")
            
        elif args.mode == 'plan':
            plan = await example.create_migration_plan()
            if plan:
                logger.info(f"Migration plan created: {plan.plan_id}")
            else:
                logger.info("No migration plan needed")
                
        elif args.mode == 'migrate':
            if not args.confirm and not args.dry_run:
                response = input("Are you sure you want to proceed with migration? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Migration cancelled")
                    return
            
            # Load plan if provided
            plan = None
            if args.plan_file:
                with open(args.plan_file, 'r') as f:
                    plan_data = json.load(f)
                    # Convert back to MigrationPlan object (simplified)
                    logger.info(f"Loaded migration plan from {args.plan_file}")
            
            result = await example.execute_migration(plan, dry_run=args.dry_run)
            if result:
                logger.info(f"Migration {'simulation' if args.dry_run else 'execution'} complete")
                
        elif args.mode == 'validate':
            # This would require loading previous migration results
            logger.info("Validation mode requires previous migration results")
            
        elif args.mode == 'rollback':
            if not args.backup_file:
                logger.error("Backup file required for rollback operation")
                return
                
            if not args.confirm:
                response = input(f"Are you sure you want to rollback using {args.backup_file}? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Rollback cancelled")
                    return
            
            success = await example.perform_rollback(args.backup_file)
            logger.info(f"Rollback {'successful' if success else 'failed'}")
            
        elif args.mode == 'cleanup':
            if not args.confirm:
                response = input(f"Are you sure you want to clean up files older than {args.days} days? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Cleanup cancelled")
                    return
            
            await example.cleanup(args.days)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())