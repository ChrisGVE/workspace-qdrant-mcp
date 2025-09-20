"""
CLI commands for collection migration management.

This module provides command-line interface tools for managing collection migrations
from suffix-based to multi-tenant architecture. It includes commands for analysis,
planning, execution, validation, and rollback operations.

Commands:
    - analyze: Analyze existing collections and identify migration candidates
    - plan: Create migration plans with conflict detection and optimization
    - execute: Execute migration plans with progress tracking and error handling
    - validate: Validate migration results and data integrity
    - rollback: Rollback failed migrations using backups
    - status: Check migration status and view reports
    - list-backups: List available migration backups
    - cleanup: Clean up old migration artifacts

Example usage:
    # Analyze collections
    wqm migration analyze --output analysis.json

    # Create migration plan
    wqm migration plan --analysis analysis.json --output plan.json
    
    # Execute migration
    wqm migration execute --plan plan.json --confirm
    
    # Check status
    wqm migration status --execution-id abc123
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

from ...common.core.client import QdrantWorkspaceClient
from ...common.core.config import Config
from ...common.memory.migration_utils import (
    CollectionMigrationManager,
    CollectionInfo,
    MigrationPlan,
    MigrationResult,
    MigrationPhase,
    CollectionPattern
)

console = Console()


@click.group(name='migration')
@click.pass_context
def migration_cli(ctx):
    """Collection migration management commands."""
    pass


@migration_cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.option('--pattern-filter', type=click.Choice(['suffix_based', 'project_based', 'mixed', 'global']), 
              help='Filter by collection pattern')
@click.option('--min-points', type=int, default=0, help='Minimum point count to include')
@click.pass_context
def analyze(ctx, output, format, pattern_filter, min_points):
    """Analyze existing collections for migration planning."""
    async def _analyze():
        try:
            config = Config()
            client = QdrantWorkspaceClient(config)
            await client.initialize()
            
            manager = CollectionMigrationManager(client, config)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing collections...", total=None)
                
                collections = await manager.analyze_collections()
                progress.update(task, description="Analysis complete")
            
            # Apply filters
            if pattern_filter:
                pattern_enum = CollectionPattern(pattern_filter)
                collections = [col for col in collections if col.pattern == pattern_enum]
            
            if min_points > 0:
                collections = [col for col in collections if col.point_count >= min_points]
            
            # Output results
            if format == 'json':
                results = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_collections': len(collections),
                    'collections': [_collection_to_dict(col) for col in collections]
                }
                
                if output:
                    with open(output, 'w') as f:
                        json.dump(results, f, indent=2)
                    console.print(f"[green]Analysis saved to {output}[/green]")
                else:
                    console.print_json(data=results)
            else:
                _display_collections_table(collections)
                
                if output:
                    # Save as JSON even if displaying as table
                    results = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'total_collections': len(collections),
                        'collections': [_collection_to_dict(col) for col in collections]
                    }
                    with open(output, 'w') as f:
                        json.dump(results, f, indent=2)
                    console.print(f"\n[green]Analysis also saved to {output}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error during analysis: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_analyze())


@migration_cli.command()
@click.option('--analysis', '-a', type=click.Path(exists=True), help='Analysis file from analyze command')
@click.option('--output', '-o', type=click.Path(), help='Output file for migration plan')
@click.option('--batch-size', type=int, default=1000, help='Batch size for migration')
@click.option('--parallel-batches', type=int, default=3, help='Number of parallel batches')
@click.option('--enable-backups/--no-backups', default=True, help='Create backups before migration')
@click.option('--format', type=click.Choice(['json', 'summary']), default='summary', help='Output format')
@click.pass_context
def plan(ctx, analysis, output, batch_size, parallel_batches, enable_backups, format):
    """Create migration plan from analysis results."""
    async def _plan():
        try:
            config = Config()
            client = QdrantWorkspaceClient(config)
            await client.initialize()
            
            manager = CollectionMigrationManager(client, config)
            
            # Load collections from analysis or analyze fresh
            if analysis:
                with open(analysis, 'r') as f:
                    analysis_data = json.load(f)
                collections = [_dict_to_collection(col_data) for col_data in analysis_data['collections']]
            else:
                console.print("[yellow]No analysis file provided, analyzing collections...[/yellow]")
                collections = await manager.analyze_collections()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating migration plan...", total=None)
                
                plan = await manager.create_migration_plan(collections)
                
                # Apply CLI options
                plan.batch_size = batch_size
                plan.parallel_batches = parallel_batches
                plan.create_backups = enable_backups
                
                progress.update(task, description="Plan created")
            
            # Output plan
            if format == 'json':
                plan_data = _migration_plan_to_dict(plan)
                
                if output:
                    with open(output, 'w') as f:
                        json.dump(plan_data, f, indent=2)
                    console.print(f"[green]Migration plan saved to {output}[/green]")
                else:
                    console.print_json(data=plan_data)
            else:
                _display_migration_plan(plan)
                
                if output:
                    plan_data = _migration_plan_to_dict(plan)
                    with open(output, 'w') as f:
                        json.dump(plan_data, f, indent=2)
                    console.print(f"\n[green]Migration plan also saved to {output}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error creating migration plan: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_plan())


@migration_cli.command()
@click.option('--plan', '-p', type=click.Path(exists=True), required=True, 
              help='Migration plan file from plan command')
@click.option('--confirm', is_flag=True, help='Confirm execution without prompting')
@click.option('--dry-run', is_flag=True, help='Show what would be migrated without executing')
@click.option('--backup-dir', type=click.Path(), help='Custom backup directory')
@click.option('--report-dir', type=click.Path(), help='Custom report directory')
@click.pass_context
def execute(ctx, plan, confirm, dry_run, backup_dir, report_dir):
    """Execute migration plan."""
    async def _execute():
        try:
            # Load plan
            with open(plan, 'r') as f:
                plan_data = json.load(f)
            migration_plan = _dict_to_migration_plan(plan_data)
            
            config = Config()
            client = QdrantWorkspaceClient(config)
            await client.initialize()
            
            # Set up custom directories if provided
            manager_kwargs = {}
            if backup_dir:
                manager_kwargs['backup_dir'] = Path(backup_dir)
            if report_dir:
                manager_kwargs['report_dir'] = Path(report_dir)
            
            manager = CollectionMigrationManager(client, config, **manager_kwargs)
            
            if dry_run:
                console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
                _display_migration_plan(migration_plan, dry_run=True)
                return
            
            # Display plan and get confirmation
            _display_migration_plan(migration_plan)
            
            if not confirm:
                if not click.confirm("\nProceed with migration?"):
                    console.print("[yellow]Migration cancelled[/yellow]")
                    return
            
            # Execute migration with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                main_task = progress.add_task("Migration Progress", total=len(migration_plan.source_collections))
                
                result = await manager.execute_migration(migration_plan)
                
                progress.update(main_task, advance=len(migration_plan.source_collections))
            
            # Display results
            _display_migration_result(result)
            
            # Generate and save report
            report_file = await manager.generate_migration_report(migration_plan, result)
            console.print(f"\n[green]Migration report saved to: {report_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error during migration execution: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_execute())


@migration_cli.command()
@click.option('--execution-id', type=str, help='Migration execution ID to check')
@click.option('--plan-id', type=str, help='Migration plan ID to check')
@click.option('--report-dir', type=click.Path(), help='Custom report directory')
@click.pass_context
def status(ctx, execution_id, plan_id, report_dir):
    """Check migration status and view reports."""
    report_directory = Path(report_dir) if report_dir else Path("./migration_reports")
    
    if not report_directory.exists():
        console.print(f"[red]Report directory not found: {report_directory}[/red]")
        return
    
    # Find reports
    reports = list(report_directory.glob("migration_report_*.json"))
    
    if not reports:
        console.print("[yellow]No migration reports found[/yellow]")
        return
    
    # Filter by execution or plan ID if provided
    if execution_id or plan_id:
        filtered_reports = []
        for report_file in reports:
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                if execution_id and report_data['migration_summary']['execution_id'] == execution_id:
                    filtered_reports.append(report_file)
                elif plan_id and report_data['migration_summary']['plan_id'] == plan_id:
                    filtered_reports.append(report_file)
            except Exception:
                continue
        reports = filtered_reports
    
    if not reports:
        console.print("[yellow]No matching migration reports found[/yellow]")
        return
    
    # Display reports
    for report_file in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            _display_migration_status(report_data)
        except Exception as e:
            console.print(f"[red]Error reading report {report_file}: {e}[/red]")


@migration_cli.command()
@click.option('--backup-file', type=click.Path(exists=True), required=True,
              help='Backup file to restore')
@click.option('--confirm', is_flag=True, help='Confirm rollback without prompting')
@click.pass_context
def rollback(ctx, backup_file, confirm):
    """Rollback migration using backup file."""
    async def _rollback():
        try:
            config = Config()
            client = QdrantWorkspaceClient(config)
            await client.initialize()
            
            manager = CollectionMigrationManager(client, config)
            
            # Display backup info
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            console.print(Panel(
                f"Collection: {backup_data['collection_name']}\n"
                f"Points: {backup_data['point_count']:,}\n"
                f"Created: {backup_data['created_at']}\n"
                f"Migration ID: {backup_data['migration_id']}",
                title="Backup Information"
            ))
            
            if not confirm:
                if not click.confirm("\nProceed with rollback? This will overwrite current collection data."):
                    console.print("[yellow]Rollback cancelled[/yellow]")
                    return
            
            # Perform rollback
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Rolling back migration...", total=None)
                
                success = await manager.rollback_manager.restore_backup(backup_file)
                
                progress.update(task, description="Rollback complete" if success else "Rollback failed")
            
            if success:
                console.print(f"[green]Successfully rolled back {backup_data['collection_name']}[/green]")
            else:
                console.print(f"[red]Failed to rollback {backup_data['collection_name']}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error during rollback: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_rollback())


@migration_cli.command()
@click.option('--backup-dir', type=click.Path(), help='Custom backup directory')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def list_backups(ctx, backup_dir, format):
    """List available migration backups."""
    backup_directory = Path(backup_dir) if backup_dir else Path("./migration_backups")
    
    if not backup_directory.exists():
        console.print(f"[yellow]Backup directory not found: {backup_directory}[/yellow]")
        return
    
    backups = list(backup_directory.glob("*.json"))
    
    if not backups:
        console.print("[yellow]No backups found[/yellow]")
        return
    
    backup_info = []
    for backup_file in backups:
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            file_size_mb = backup_file.stat().st_size / (1024 * 1024)
            
            backup_info.append({
                'file': str(backup_file),
                'collection': backup_data.get('collection_name', 'unknown'),
                'migration_id': backup_data.get('migration_id', 'unknown'),
                'points': backup_data.get('point_count', 0),
                'created': backup_data.get('created_at', 'unknown'),
                'size_mb': round(file_size_mb, 2)
            })
        except Exception:
            continue
    
    if format == 'json':
        console.print_json(data=backup_info)
    else:
        table = Table(title="Migration Backups")
        table.add_column("Collection")
        table.add_column("Migration ID")
        table.add_column("Points", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Created")
        table.add_column("File")
        
        for backup in sorted(backup_info, key=lambda x: x['created'], reverse=True):
            table.add_row(
                backup['collection'],
                backup['migration_id'][:8] + "..." if len(backup['migration_id']) > 8 else backup['migration_id'],
                f"{backup['points']:,}",
                str(backup['size_mb']),
                backup['created'][:19] if backup['created'] != 'unknown' else backup['created'],
                Path(backup['file']).name
            )
        
        console.print(table)


@migration_cli.command()
@click.option('--backup-dir', type=click.Path(), help='Custom backup directory')
@click.option('--report-dir', type=click.Path(), help='Custom report directory')
@click.option('--days', type=int, default=30, help='Remove files older than N days')
@click.option('--confirm', is_flag=True, help='Confirm cleanup without prompting')
@click.pass_context
def cleanup(ctx, backup_dir, report_dir, days, confirm):
    """Clean up old migration artifacts."""
    backup_directory = Path(backup_dir) if backup_dir else Path("./migration_backups")
    report_directory = Path(report_dir) if report_dir else Path("./migration_reports")
    
    # Find old files
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    old_files = []
    
    for directory in [backup_directory, report_directory]:
        if directory.exists():
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    old_files.append(file_path)
    
    if not old_files:
        console.print(f"[green]No files older than {days} days found[/green]")
        return
    
    # Display files to be deleted
    table = Table(title=f"Files to Delete (older than {days} days)")
    table.add_column("File")
    table.add_column("Size")
    table.add_column("Age (days)")
    
    total_size = 0
    for file_path in old_files:
        age_days = (datetime.now().timestamp() - file_path.stat().st_mtime) / (24 * 60 * 60)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        
        table.add_row(
            str(file_path),
            f"{size_mb:.2f} MB",
            f"{age_days:.1f}"
        )
    
    console.print(table)
    console.print(f"\n[yellow]Total size to be freed: {total_size:.2f} MB[/yellow]")
    
    if not confirm:
        if not click.confirm("\nProceed with cleanup?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
    
    # Delete files
    deleted_count = 0
    for file_path in old_files:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]Failed to delete {file_path}: {e}[/red]")
    
    console.print(f"[green]Deleted {deleted_count} files ({total_size:.2f} MB freed)[/green]")


# Helper functions for data conversion and display

def _collection_to_dict(collection: CollectionInfo) -> dict:
    """Convert CollectionInfo to dictionary."""
    return {
        'name': collection.name,
        'pattern': collection.pattern.value,
        'project_name': collection.project_name,
        'suffix': collection.suffix,
        'point_count': collection.point_count,
        'vector_count': collection.vector_count,
        'size_mb': collection.size_mb,
        'created_at': collection.created_at.isoformat() if collection.created_at else None,
        'last_modified': collection.last_modified.isoformat() if collection.last_modified else None,
        'metadata_keys': list(collection.metadata_keys),
        'has_project_metadata': collection.has_project_metadata,
        'migration_priority': collection.migration_priority
    }


def _dict_to_collection(data: dict) -> CollectionInfo:
    """Convert dictionary to CollectionInfo."""
    return CollectionInfo(
        name=data['name'],
        pattern=CollectionPattern(data['pattern']),
        project_name=data.get('project_name'),
        suffix=data.get('suffix'),
        point_count=data.get('point_count', 0),
        vector_count=data.get('vector_count', 0),
        size_mb=data.get('size_mb', 0.0),
        created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
        last_modified=datetime.fromisoformat(data['last_modified']) if data.get('last_modified') else None,
        metadata_keys=set(data.get('metadata_keys', [])),
        has_project_metadata=data.get('has_project_metadata', False),
        migration_priority=data.get('migration_priority', 1)
    )


def _migration_plan_to_dict(plan: MigrationPlan) -> dict:
    """Convert MigrationPlan to dictionary."""
    return {
        'plan_id': plan.plan_id,
        'created_at': plan.created_at.isoformat(),
        'source_collections': [_collection_to_dict(col) for col in plan.source_collections],
        'target_collections': plan.target_collections,
        'batch_size': plan.batch_size,
        'parallel_batches': plan.parallel_batches,
        'enable_validation': plan.enable_validation,
        'create_backups': plan.create_backups,
        'conflicts': plan.conflicts,
        'resolutions': plan.resolutions,
        'estimated_duration_minutes': plan.estimated_duration_minutes,
        'estimated_storage_mb': plan.estimated_storage_mb,
        'total_points_to_migrate': plan.total_points_to_migrate,
        'migration_order': plan.migration_order,
        'dependencies': plan.dependencies
    }


def _dict_to_migration_plan(data: dict) -> MigrationPlan:
    """Convert dictionary to MigrationPlan."""
    return MigrationPlan(
        plan_id=data['plan_id'],
        created_at=datetime.fromisoformat(data['created_at']),
        source_collections=[_dict_to_collection(col) for col in data['source_collections']],
        target_collections=data['target_collections'],
        batch_size=data['batch_size'],
        parallel_batches=data['parallel_batches'],
        enable_validation=data['enable_validation'],
        create_backups=data['create_backups'],
        conflicts=data['conflicts'],
        resolutions=data['resolutions'],
        estimated_duration_minutes=data['estimated_duration_minutes'],
        estimated_storage_mb=data['estimated_storage_mb'],
        total_points_to_migrate=data['total_points_to_migrate'],
        migration_order=data['migration_order'],
        dependencies=data['dependencies']
    )


def _display_collections_table(collections: List[CollectionInfo]):
    """Display collections in a rich table."""
    table = Table(title="Collection Analysis Results")
    table.add_column("Collection Name")
    table.add_column("Pattern")
    table.add_column("Project")
    table.add_column("Suffix")
    table.add_column("Points", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Priority")
    table.add_column("Has Metadata")
    
    for collection in collections:
        table.add_row(
            collection.name,
            collection.pattern.value,
            collection.project_name or "N/A",
            collection.suffix or "N/A",
            f"{collection.point_count:,}",
            f"{collection.size_mb:.2f}",
            str(collection.migration_priority),
            "✓" if collection.has_project_metadata else "✗"
        )
    
    console.print(table)


def _display_migration_plan(plan: MigrationPlan, dry_run: bool = False):
    """Display migration plan in a readable format."""
    title = "Migration Plan (DRY RUN)" if dry_run else "Migration Plan"
    
    # Plan summary
    console.print(Panel(
        f"Plan ID: {plan.plan_id}\n"
        f"Collections to migrate: {len(plan.source_collections)}\n"
        f"Total points: {plan.total_points_to_migrate:,}\n"
        f"Estimated duration: {plan.estimated_duration_minutes:.1f} minutes\n"
        f"Estimated storage: {plan.estimated_storage_mb:.1f} MB\n"
        f"Batch size: {plan.batch_size}\n"
        f"Parallel batches: {plan.parallel_batches}\n"
        f"Create backups: {'Yes' if plan.create_backups else 'No'}",
        title=title
    ))
    
    # Conflicts
    if plan.conflicts:
        console.print(f"\n[red]Conflicts detected ({len(plan.conflicts)}):[/red]")
        for conflict in plan.conflicts:
            console.print(f"  - {conflict['severity'].upper()}: {conflict['message']}")
    
    # Collections table
    if plan.source_collections:
        table = Table(title="Collections to Migrate")
        table.add_column("Source")
        table.add_column("Target")
        table.add_column("Pattern")
        table.add_column("Points", justify="right")
        table.add_column("Priority")
        
        for i, source_col in enumerate(plan.source_collections):
            target_col = plan.target_collections[i] if i < len(plan.target_collections) else "N/A"
            table.add_row(
                source_col.name,
                target_col,
                source_col.pattern.value,
                f"{source_col.point_count:,}",
                str(source_col.migration_priority)
            )
        
        console.print(table)


def _display_migration_result(result: MigrationResult):
    """Display migration execution results."""
    status_color = "green" if result.success else "red"
    status_text = "SUCCESS" if result.success else "FAILED"
    
    duration = (result.completed_at - result.started_at).total_seconds() if result.completed_at else 0
    
    console.print(Panel(
        f"Status: [{status_color}]{status_text}[/{status_color}]\n"
        f"Phase: {result.phase.value}\n"
        f"Duration: {duration:.1f} seconds\n"
        f"Collections migrated: {result.collections_migrated}\n"
        f"Points migrated: {result.points_migrated:,}\n"
        f"Points failed: {result.points_failed:,}\n"
        f"Batches processed: {result.batches_processed}\n"
        f"Batches failed: {result.batches_failed}\n"
        f"Backups created: {len(result.backup_locations)}",
        title="Migration Results"
    ))
    
    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
    
    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")


def _display_migration_status(report_data: dict):
    """Display migration status from report data."""
    summary = report_data['migration_summary']
    execution = report_data['execution_results']
    
    status_color = "green" if summary['overall_success'] else "red"
    status_text = "SUCCESS" if summary['overall_success'] else "FAILED"
    
    console.print(Panel(
        f"Execution ID: {summary['execution_id']}\n"
        f"Status: [{status_color}]{status_text}[/{status_color}]\n"
        f"Started: {summary['started_at']}\n"
        f"Completed: {summary.get('completed_at', 'In Progress')}\n"
        f"Phase: {summary['final_phase']}\n"
        f"Collections migrated: {execution['collections_migrated']}\n"
        f"Points migrated: {execution['points_migrated']:,}\n"
        f"Success rate: {execution['success_rate_percent']}%",
        title="Migration Status"
    ))


if __name__ == "__main__":
    migration_cli()