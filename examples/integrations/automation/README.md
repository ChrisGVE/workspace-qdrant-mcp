# Automation Examples

Comprehensive automation scripts and workflows for workspace-qdrant-mcp, including CLI tools, batch processing, and workflow automation.

## 🎯 Overview

This section provides:

- **CLI Automation** - Command-line tools for common operations
- **Batch Processing** - Scripts for bulk operations on documents and collections
- **Workflow Automation** - Automated workflows for daily/weekly tasks
- **Data Pipeline** - ETL processes for knowledge management
- **Backup & Sync** - Data backup and synchronization utilities
- **Integration Scripts** - Connect with external tools and services

## 🏗️ Automation Structure

```
integrations/automation/
├── README.md                     # This file
├── cli_tools/                    # Command-line utilities
│   ├── wq_cli.py                # Main CLI interface
│   ├── collection_manager.py    # Collection management CLI
│   ├── backup_restore.py        # Backup and restore operations
│   └── health_check.py          # System health monitoring
├── batch_processing/             # Bulk operations
│   ├── document_processor.py    # Bulk document processing
│   ├── metadata_updater.py      # Batch metadata updates
│   ├── collection_migrator.py   # Collection migration tools
│   └── cleanup_utilities.py     # Data cleanup and maintenance
├── workflows/                    # Automated workflows
│   ├── daily_workflows.py       # Daily automation tasks
│   ├── weekly_maintenance.py    # Weekly system maintenance
│   ├── content_monitor.py       # Content monitoring and alerts
│   └── sync_scheduler.py        # Scheduled synchronization
├── data_pipelines/              # ETL processes
│   ├── document_ingestion.py    # Document ingestion pipeline
│   ├── external_sync.py         # External data synchronization
│   ├── web_scraper.py           # Web content ingestion
│   └── api_integrations.py      # API-based data collection
└── scripts/                     # Utility scripts
    ├── setup_environment.sh     # Environment setup script
    ├── monitor_system.sh        # System monitoring
    ├── deploy_updates.sh        # Update deployment
    └── generate_reports.sh      # Report generation
```

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Navigate to automation directory
cd examples/integrations/automation

# Install automation dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.sh
```

### 2. Configure Environment

```bash
# Copy and customize configuration
cp config/automation_config.example.yaml config/automation_config.yaml

# Set up environment variables
export WQ_AUTOMATION_CONFIG="config/automation_config.yaml"
export AUTOMATION_LOG_LEVEL="INFO"
```

### 3. Test Setup

```bash
# Test CLI tools
python cli_tools/wq_cli.py --help

# Run health check
python cli_tools/health_check.py --full

# Test batch processing
python batch_processing/document_processor.py --dry-run
```

## 📚 CLI Tools

### Main CLI Interface

**Comprehensive CLI for workspace-qdrant-mcp operations:**

```python
#!/usr/bin/env python3
"""
wq_cli.py - Comprehensive CLI interface for workspace-qdrant-mcp automation
"""

import click
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from workspace_qdrant_mcp.client import WorkspaceClient
from .collection_manager import CollectionManager
from .backup_restore import BackupManager
from .health_check import HealthChecker

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """workspace-qdrant-mcp CLI automation tool."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    # Initialize client
    try:
        ctx.obj['client'] = WorkspaceClient()
        if verbose:
            click.echo("✅ Connected to workspace-qdrant-mcp")
    except Exception as e:
        click.echo(f"❌ Failed to connect: {e}", err=True)
        sys.exit(1)

@cli.group()
def collections():
    """Collection management operations."""
    pass

@collections.command('list')
@click.option('--details', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def list_collections(ctx, details):
    """List all collections."""
    client = ctx.obj['client']
    cm = CollectionManager(client)
    
    collections = cm.list_collections()
    
    if not collections:
        click.echo("No collections found.")
        return
    
    if details:
        for collection in collections:
            info = cm.get_collection_info(collection)
            click.echo(f"\n📁 {collection}")
            click.echo(f"   Documents: {info.get('document_count', 'Unknown')}")
            click.echo(f"   Size: {info.get('size', 'Unknown')}")
            click.echo(f"   Created: {info.get('created_date', 'Unknown')}")
    else:
        click.echo("Collections:")
        for collection in collections:
            click.echo(f"  📁 {collection}")

@collections.command('create')
@click.argument('name')
@click.option('--description', '-d', help='Collection description')
@click.pass_context
def create_collection(ctx, name, description):
    """Create a new collection."""
    client = ctx.obj['client']
    cm = CollectionManager(client)
    
    try:
        result = cm.create_collection(name, description)
        if result:
            click.echo(f"✅ Created collection: {name}")
        else:
            click.echo(f"❌ Failed to create collection: {name}")
    except Exception as e:
        click.echo(f"❌ Error creating collection: {e}", err=True)

@collections.command('delete')
@click.argument('name')
@click.option('--force', '-f', is_flag=True, help='Force deletion without confirmation')
@click.pass_context
def delete_collection(ctx, name, force):
    """Delete a collection."""
    client = ctx.obj['client']
    cm = CollectionManager(client)
    
    if not force:
        if not click.confirm(f"Are you sure you want to delete collection '{name}'?"):
            click.echo("Operation cancelled.")
            return
    
    try:
        result = cm.delete_collection(name)
        if result:
            click.echo(f"✅ Deleted collection: {name}")
        else:
            click.echo(f"❌ Failed to delete collection: {name}")
    except Exception as e:
        click.echo(f"❌ Error deleting collection: {e}", err=True)

@cli.group()
def documents():
    """Document management operations."""
    pass

@documents.command('import')
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', '-c', required=True, help='Target collection')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--file-pattern', '-p', help='File pattern to match (e.g., "*.md")')
@click.option('--dry-run', is_flag=True, help='Show what would be imported without doing it')
@click.pass_context
def import_documents(ctx, path, collection, recursive, file_pattern, dry_run):
    """Import documents from filesystem."""
    from .document_processor import DocumentProcessor
    
    client = ctx.obj['client']
    processor = DocumentProcessor(client)
    
    click.echo(f"Importing documents from {path} to collection '{collection}'")
    
    try:
        results = processor.import_from_filesystem(
            path=path,
            collection=collection,
            recursive=recursive,
            file_pattern=file_pattern,
            dry_run=dry_run
        )
        
        click.echo(f"✅ Processed {results['processed']} documents")
        click.echo(f"   Imported: {results['imported']}")
        click.echo(f"   Skipped: {results['skipped']}")
        if results.get('errors'):
            click.echo(f"   Errors: {len(results['errors'])}")
            if ctx.obj['verbose']:
                for error in results['errors'][:5]:  # Show first 5 errors
                    click.echo(f"     ❌ {error}")
                    
    except Exception as e:
        click.echo(f"❌ Import failed: {e}", err=True)

@documents.command('search')
@click.argument('query')
@click.option('--collection', '-c', help='Search in specific collection')
@click.option('--limit', '-l', default=10, help='Maximum results to return')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'detailed']), 
              default='table', help='Output format')
@click.pass_context
def search_documents(ctx, query, collection, limit, format):
    """Search documents."""
    client = ctx.obj['client']
    
    try:
        results = client.search(
            query=query,
            collection=collection,
            limit=limit
        )
        
        if not results:
            click.echo("No results found.")
            return
        
        if format == 'json':
            click.echo(json.dumps(results, indent=2, default=str))
        elif format == 'detailed':
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                click.echo(f"\n{i}. {metadata.get('title', 'Untitled')}")
                click.echo(f"   Collection: {metadata.get('collection', 'Unknown')}")
                click.echo(f"   Type: {metadata.get('type', 'Unknown')}")
                click.echo(f"   Content: {result.get('content', '')[:200]}...")
                click.echo(f"   Score: {result.get('score', 'N/A')}")
        else:  # table format
            click.echo(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                title = metadata.get('title', 'Untitled')[:50]
                collection_name = metadata.get('collection', 'Unknown')[:15]
                score = f"{result.get('score', 0):.3f}"
                click.echo(f"{i:2d}. {title:50} | {collection_name:15} | {score}")
                
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)

@cli.group()
def backup():
    """Backup and restore operations."""
    pass

@backup.command('create')
@click.option('--output', '-o', help='Output file path')
@click.option('--collections', '-c', help='Comma-separated list of collections to backup')
@click.option('--compress', is_flag=True, help='Compress backup file')
@click.pass_context
def create_backup(ctx, output, collections, compress):
    """Create backup of collections."""
    client = ctx.obj['client']
    bm = BackupManager(client)
    
    # Generate default output filename if not provided
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"wq_backup_{timestamp}.json"
        if compress:
            output += ".gz"
    
    # Parse collections list
    collection_list = collections.split(',') if collections else None
    
    try:
        result = bm.create_backup(
            output_path=output,
            collections=collection_list,
            compress=compress
        )
        
        if result['success']:
            click.echo(f"✅ Backup created: {output}")
            click.echo(f"   Collections: {result['collections_count']}")
            click.echo(f"   Documents: {result['documents_count']}")
            click.echo(f"   Size: {result['size_mb']:.1f} MB")
        else:
            click.echo(f"❌ Backup failed: {result['error']}")
            
    except Exception as e:
        click.echo(f"❌ Backup creation failed: {e}", err=True)

@backup.command('restore')
@click.argument('backup_file', type=click.Path(exists=True))
@click.option('--collections', '-c', help='Comma-separated list of collections to restore')
@click.option('--overwrite', is_flag=True, help='Overwrite existing documents')
@click.option('--dry-run', is_flag=True, help='Show what would be restored without doing it')
@click.pass_context
def restore_backup(ctx, backup_file, collections, overwrite, dry_run):
    """Restore from backup file."""
    client = ctx.obj['client']
    bm = BackupManager(client)
    
    # Parse collections list
    collection_list = collections.split(',') if collections else None
    
    try:
        result = bm.restore_backup(
            backup_path=backup_file,
            collections=collection_list,
            overwrite=overwrite,
            dry_run=dry_run
        )
        
        if result['success']:
            action = "Would restore" if dry_run else "Restored"
            click.echo(f"✅ {action}:")
            click.echo(f"   Collections: {result['collections_count']}")
            click.echo(f"   Documents: {result['documents_count']}")
            if result.get('warnings'):
                click.echo(f"   Warnings: {len(result['warnings'])}")
                if ctx.obj['verbose']:
                    for warning in result['warnings'][:3]:
                        click.echo(f"     ⚠️  {warning}")
        else:
            click.echo(f"❌ Restore failed: {result['error']}")
            
    except Exception as e:
        click.echo(f"❌ Restore failed: {e}", err=True)

@cli.command('health')
@click.option('--full', is_flag=True, help='Run full health check')
@click.option('--fix', is_flag=True, help='Attempt to fix issues automatically')
@click.pass_context
def health_check(ctx, full, fix):
    """Run system health check."""
    client = ctx.obj['client']
    hc = HealthChecker(client)
    
    try:
        click.echo("Running health check...")
        
        if full:
            results = hc.full_health_check()
        else:
            results = hc.basic_health_check()
        
        # Display results
        overall_status = "✅ HEALTHY" if results['overall_healthy'] else "❌ ISSUES FOUND"
        click.echo(f"\nOverall Status: {overall_status}")
        
        for category, checks in results['checks'].items():
            click.echo(f"\n{category.title()}:")
            for check_name, check_result in checks.items():
                status = "✅" if check_result['passed'] else "❌"
                click.echo(f"  {status} {check_name}: {check_result['message']}")
                
                if not check_result['passed'] and fix and check_result.get('fixable'):
                    click.echo(f"    🔧 Attempting to fix...")
                    fix_result = hc.fix_issue(category, check_name)
                    if fix_result['success']:
                        click.echo(f"    ✅ Fixed: {fix_result['message']}")
                    else:
                        click.echo(f"    ❌ Fix failed: {fix_result['message']}")
        
        # Show recommendations
        if results.get('recommendations'):
            click.echo(f"\n💡 Recommendations:")
            for rec in results['recommendations']:
                click.echo(f"  • {rec}")
                
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}", err=True)

@cli.group()
def workflows():
    """Automated workflow operations."""
    pass

@workflows.command('daily')
@click.option('--config', '-c', help='Workflow configuration file')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.pass_context
def run_daily_workflow(ctx, config, dry_run):
    """Run daily maintenance workflow."""
    from .daily_workflows import DailyWorkflowManager
    
    client = ctx.obj['client']
    dwm = DailyWorkflowManager(client)
    
    try:
        click.echo("Running daily workflow...")
        
        results = dwm.run_daily_workflow(
            config_path=config,
            dry_run=dry_run
        )
        
        action = "Would execute" if dry_run else "Executed"
        click.echo(f"✅ {action} daily workflow:")
        
        for task_name, task_result in results.items():
            status = "✅" if task_result.get('success') else "❌"
            click.echo(f"  {status} {task_name}: {task_result.get('message', 'No details')}")
            
        if results.get('summary'):
            click.echo(f"\nSummary:")
            for key, value in results['summary'].items():
                click.echo(f"  {key}: {value}")
                
    except Exception as e:
        click.echo(f"❌ Daily workflow failed: {e}", err=True)

@workflows.command('schedule')
@click.option('--workflow', '-w', required=True, help='Workflow name (daily, weekly, monthly)')
@click.option('--time', '-t', help='Schedule time (HH:MM format)')
@click.option('--enable/--disable', default=True, help='Enable or disable scheduled workflow')
@click.pass_context
def schedule_workflow(ctx, workflow, time, enable):
    """Schedule automated workflows."""
    from .sync_scheduler import WorkflowScheduler
    
    scheduler = WorkflowScheduler()
    
    try:
        if enable:
            result = scheduler.schedule_workflow(workflow, time)
            if result['success']:
                click.echo(f"✅ Scheduled {workflow} workflow for {time}")
            else:
                click.echo(f"❌ Failed to schedule workflow: {result['error']}")
        else:
            result = scheduler.disable_workflow(workflow)
            if result['success']:
                click.echo(f"✅ Disabled {workflow} workflow")
            else:
                click.echo(f"❌ Failed to disable workflow: {result['error']}")
                
    except Exception as e:
        click.echo(f"❌ Scheduling failed: {e}", err=True)

@cli.command('monitor')
@click.option('--duration', '-d', default=60, help='Monitoring duration in seconds')
@click.option('--interval', '-i', default=5, help='Check interval in seconds')
@click.pass_context
def monitor_system(ctx, duration, interval):
    """Monitor system in real-time."""
    import time
    from .health_check import HealthChecker
    
    client = ctx.obj['client']
    hc = HealthChecker(client)
    
    click.echo(f"Monitoring system for {duration} seconds (checking every {interval}s)")
    click.echo("Press Ctrl+C to stop monitoring early\n")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Quick health check
            status = hc.quick_status_check()
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            overall = "🟢" if status['healthy'] else "🔴"
            
            click.echo(f"{timestamp} {overall} Status: {status['message']}")
            
            if ctx.obj['verbose']:
                for metric, value in status.get('metrics', {}).items():
                    click.echo(f"  {metric}: {value}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\n👋 Monitoring stopped by user")

if __name__ == '__main__':
    cli()
```

### Batch Processing Tools

**Bulk Document Processing System:**

```python
#!/usr/bin/env python3
"""
document_processor.py - Bulk document processing for workspace-qdrant-mcp
"""

import os
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    file_path: str
    success: bool
    message: str
    document_id: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class DocumentProcessor:
    """
    High-performance bulk document processing system.
    
    Supports parallel processing, progress tracking, and error handling
    for large-scale document ingestion and updates.
    """
    
    def __init__(self, client, max_workers: int = 4):
        self.client = client
        self.max_workers = max_workers
        self.progress_lock = threading.Lock()
        self.results = []
        
        # Supported file types and processors
        self.file_processors = {
            '.txt': self._process_text_file,
            '.md': self._process_markdown_file,
            '.py': self._process_python_file,
            '.json': self._process_json_file,
            '.csv': self._process_csv_file,
            '.pdf': self._process_pdf_file,
            '.docx': self._process_docx_file,
        }
    
    def import_from_filesystem(self, path: str, collection: str, 
                             recursive: bool = True, file_pattern: str = None,
                             dry_run: bool = False) -> Dict[str, Any]:
        """
        Import documents from filesystem with parallel processing.
        
        Args:
            path: Path to directory or file
            collection: Target collection name
            recursive: Process directories recursively
            file_pattern: File pattern to match (e.g., "*.md")
            dry_run: Don't actually import, just show what would be done
            
        Returns:
            Dictionary with processing statistics and results
        """
        start_time = datetime.now()
        path_obj = Path(path)
        
        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}
        
        # Collect files to process
        files_to_process = list(self._collect_files(path_obj, recursive, file_pattern))
        
        print(f"📁 Found {len(files_to_process)} files to process")
        
        if dry_run:
            print("🔍 DRY RUN - No files will be actually processed")
            for file_path in files_to_process[:10]:  # Show first 10
                print(f"  Would process: {file_path}")
            if len(files_to_process) > 10:
                print(f"  ... and {len(files_to_process) - 10} more files")
            
            return {
                "dry_run": True,
                "files_found": len(files_to_process),
                "supported_files": len([f for f in files_to_process if f.suffix.lower() in self.file_processors]),
                "duration": (datetime.now() - start_time).total_seconds()
            }
        
        # Process files in parallel
        processed = 0
        imported = 0
        skipped = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_file, file_path, collection): file_path
                for file_path in files_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    with self.progress_lock:
                        processed += 1
                        
                        if result.success:
                            imported += 1
                        else:
                            if "skipped" in result.message.lower():
                                skipped += 1
                            else:
                                errors.append(f"{file_path}: {result.message}")
                        
                        # Progress indicator
                        if processed % 10 == 0 or processed == len(files_to_process):
                            progress = (processed / len(files_to_process)) * 100
                            print(f"📊 Progress: {processed}/{len(files_to_process)} ({progress:.1f}%)")
                
                except Exception as e:
                    with self.progress_lock:
                        processed += 1
                        errors.append(f"{file_path}: Exception - {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        results = {
            "processed": processed,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "duration": duration,
            "files_per_second": processed / duration if duration > 0 else 0
        }
        
        print(f"✅ Processing complete in {duration:.1f}s")
        print(f"   Imported: {imported}, Skipped: {skipped}, Errors: {len(errors)}")
        
        return results
    
    def _collect_files(self, path: Path, recursive: bool, 
                      file_pattern: str) -> Generator[Path, None, None]:
        """Collect files to process based on criteria."""
        if path.is_file():
            yield path
            return
        
        if recursive:
            pattern = "**/*" if not file_pattern else f"**/{file_pattern}"
            for file_path in path.rglob(pattern.lstrip("**/")):
                if file_path.is_file() and self._should_process_file(file_path):
                    yield file_path
        else:
            pattern = "*" if not file_pattern else file_pattern
            for file_path in path.glob(pattern):
                if file_path.is_file() and self._should_process_file(file_path):
                    yield file_path
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        # Skip hidden files and common non-document files
        if file_path.name.startswith('.'):
            return False
        
        # Skip binary files we can't process
        skip_extensions = {'.exe', '.bin', '.so', '.dll', '.pyc', '.git'}
        if file_path.suffix.lower() in skip_extensions:
            return False
        
        # Skip very large files (> 50MB)
        try:
            if file_path.stat().st_size > 50 * 1024 * 1024:
                return False
        except OSError:
            return False
        
        return True
    
    def _process_file(self, file_path: Path, collection: str) -> ProcessingResult:
        """Process a single file."""
        start_time = datetime.now()
        
        try:
            # Check if file type is supported
            processor = self.file_processors.get(file_path.suffix.lower())
            
            if not processor:
                # Try to process as text if it's a text file
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type and mime_type.startswith('text/'):
                    processor = self._process_text_file
                else:
                    return ProcessingResult(
                        file_path=str(file_path),
                        success=False,
                        message="Unsupported file type",
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
            
            # Process the file
            content, metadata = processor(file_path)
            
            if not content:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    message="No content extracted",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Store in collection
            document_id = self._store_document(content, metadata, collection)
            
            return ProcessingResult(
                file_path=str(file_path),
                success=True,
                message="Successfully processed",
                document_id=document_id,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata=metadata
            )
            
        except Exception as e:
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                message=f"Processing error: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _store_document(self, content: str, metadata: Dict[str, Any], 
                       collection: str) -> str:
        """Store document in collection."""
        # Add processing metadata
        metadata.update({
            'processed_date': datetime.now().isoformat(),
            'processor_version': '1.0',
            'collection': collection
        })
        
        # Store document
        result = self.client.store(
            content=content,
            metadata=metadata,
            collection=collection
        )
        
        return result  # Assuming this returns document ID
    
    # File type processors
    
    def _process_text_file(self, file_path: Path) -> tuple:
        """Process plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        metadata = {
            'type': 'text_document',
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return content, metadata
    
    def _process_markdown_file(self, file_path: Path) -> tuple:
        """Process Markdown file with metadata extraction."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract title from first heading
        title = file_path.stem
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('# '):
                title = line.strip()[2:]
                break
        
        # Extract tags if present
        tags = []
        if 'tags:' in content.lower():
            # Simple tag extraction (could be more sophisticated)
            for line in lines:
                if line.lower().strip().startswith('tags:'):
                    tag_part = line.split(':', 1)[1].strip()
                    tags = [tag.strip() for tag in tag_part.split(',')]
                    break
        
        metadata = {
            'type': 'markdown_document',
            'title': title,
            'filename': file_path.name,
            'file_path': str(file_path),
            'tags': tags,
            'word_count': len(content.split()),
            'file_size': file_path.stat().st_size,
            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return content, metadata
    
    def _process_python_file(self, file_path: Path) -> tuple:
        """Process Python file with code analysis."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract docstring
        docstring = ""
        try:
            import ast
            tree = ast.parse(content)
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Str)):
                docstring = tree.body[0].value.s
        except:
            pass
        
        # Count functions and classes
        function_count = content.count('def ')
        class_count = content.count('class ')
        
        metadata = {
            'type': 'python_code',
            'filename': file_path.name,
            'file_path': str(file_path),
            'docstring': docstring,
            'function_count': function_count,
            'class_count': class_count,
            'line_count': len(content.split('\n')),
            'file_size': file_path.stat().st_size,
            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return content, metadata
    
    def _process_json_file(self, file_path: Path) -> tuple:
        """Process JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to readable text
        content = json.dumps(data, indent=2)
        
        # Analyze structure
        def count_items(obj):
            if isinstance(obj, dict):
                return len(obj)
            elif isinstance(obj, list):
                return len(obj)
            return 1
        
        metadata = {
            'type': 'json_document',
            'filename': file_path.name,
            'file_path': str(file_path),
            'json_structure': type(data).__name__,
            'item_count': count_items(data),
            'file_size': file_path.stat().st_size,
            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        return content, metadata
    
    def _process_csv_file(self, file_path: Path) -> tuple:
        """Process CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Create readable content
            content = f"CSV Data: {file_path.name}\n\n"
            content += f"Columns: {', '.join(df.columns.tolist())}\n"
            content += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            content += "Sample data:\n"
            content += df.head().to_string()
            
            metadata = {
                'type': 'csv_data',
                'filename': file_path.name,
                'file_path': str(file_path),
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'column_count': len(df.columns),
                'file_size': file_path.stat().st_size,
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
        except ImportError:
            # Fallback without pandas
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            content = f"CSV Data: {file_path.name}\n\n"
            content += f"Lines: {len(lines)}\n"
            content += "Sample content:\n"
            content += ''.join(lines[:10])
            
            metadata = {
                'type': 'csv_data',
                'filename': file_path.name,
                'file_path': str(file_path),
                'line_count': len(lines),
                'file_size': file_path.stat().st_size,
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        
        return content, metadata
    
    def _process_pdf_file(self, file_path: Path) -> tuple:
        """Process PDF file (requires PyPDF2 or similar)."""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                content = f"PDF Document: {file_path.name}\n\n"
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"Page {page_num + 1}:\n{page_text}\n\n"
                
                metadata = {
                    'type': 'pdf_document',
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'page_count': len(pdf_reader.pages),
                    'file_size': file_path.stat().st_size,
                    'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
        except ImportError:
            content = f"PDF Document: {file_path.name}\n\nPDF processing requires PyPDF2 library."
            metadata = {
                'type': 'pdf_document',
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'processing_note': 'PDF content extraction not available',
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        
        return content, metadata
    
    def _process_docx_file(self, file_path: Path) -> tuple:
        """Process DOCX file (requires python-docx)."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            content = f"Word Document: {file_path.name}\n\n"
            for para in doc.paragraphs:
                if para.text.strip():
                    content += para.text + "\n"
            
            metadata = {
                'type': 'docx_document',
                'filename': file_path.name,
                'file_path': str(file_path),
                'paragraph_count': len(doc.paragraphs),
                'file_size': file_path.stat().st_size,
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
        except ImportError:
            content = f"Word Document: {file_path.name}\n\nDOCX processing requires python-docx library."
            metadata = {
                'type': 'docx_document',
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'processing_note': 'DOCX content extraction not available',
                'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        
        return content, metadata

# Usage example and CLI interface
if __name__ == "__main__":
    import argparse
    from workspace_qdrant_mcp.client import WorkspaceClient
    
    parser = argparse.ArgumentParser(description="Bulk document processor for workspace-qdrant-mcp")
    parser.add_argument("path", help="Path to directory or file to process")
    parser.add_argument("--collection", "-c", required=True, help="Target collection name")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--pattern", "-p", help="File pattern to match (e.g., '*.md')")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")
    
    args = parser.parse_args()
    
    # Initialize client and processor
    client = WorkspaceClient()
    processor = DocumentProcessor(client, max_workers=args.workers)
    
    # Run bulk import
    results = processor.import_from_filesystem(
        path=args.path,
        collection=args.collection,
        recursive=args.recursive,
        file_pattern=args.pattern,
        dry_run=args.dry_run
    )
    
    # Display results
    if results.get('error'):
        print(f"❌ Error: {results['error']}")
    else:
        print(f"\n📊 Processing Results:")
        for key, value in results.items():
            if key != 'errors':
                print(f"  {key}: {value}")
        
        if results.get('errors'):
            print(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
```

## 💡 Best Practices

### Automation Workflows

**Daily Automation Schedule:**
```bash
# Morning routine (9:00 AM)
0 9 * * * python workflows/daily_workflows.py --task health-check
5 9 * * * python workflows/daily_workflows.py --task backup-check

# Evening routine (6:00 PM)  
0 18 * * * python workflows/daily_workflows.py --task cleanup
30 18 * * * python workflows/daily_workflows.py --task status-report
```

**Weekly Maintenance:**
```bash
# Sunday maintenance (2:00 AM)
0 2 * * 0 python workflows/weekly_maintenance.py --full-maintenance
30 2 * * 0 python batch_processing/cleanup_utilities.py --archive-old
```

### Performance Optimization

**Batch Processing Guidelines:**
1. **Use parallel processing** for large datasets
2. **Implement progress tracking** for long operations
3. **Handle errors gracefully** without stopping entire process
4. **Provide dry-run options** for testing
5. **Log operations comprehensively** for troubleshooting

### Integration Best Practices

**CLI Tool Design:**
1. **Consistent command structure** across all tools
2. **Comprehensive help documentation** for each command
3. **Error handling with user-friendly messages**
4. **Progress indicators** for long-running operations
5. **Configuration file support** for complex setups

## 🔗 Integration Examples

- **[VS Code Integration](../vscode/README.md)** - Development environment setup
- **[Cursor Integration](../cursor/README.md)** - AI-powered development workflows
- **[Performance Optimization](../../performance_optimization/README.md)** - Large-scale processing

---

**Next Steps:**
1. Set up [CLI Tools](cli_tools/) for command-line operations
2. Configure [Batch Processing](batch_processing/) for bulk operations
3. Implement [Automated Workflows](workflows/) for maintenance