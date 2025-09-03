
"""Document ingestion CLI commands.

This module provides manual document processing capabilities for the
unified wqm CLI, handling various file formats and ingestion workflows.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from ...cli.ingestion_engine import DocumentIngestionEngine, IngestionResult
from ...core.daemon_client import get_daemon_client, with_daemon_client
from ...core.yaml_config import load_config
from ...core.yaml_metadata import YamlMetadataWorkflow
from ...observability import get_logger

logger = get_logger(__name__)

# Create the ingest app
ingest_app = typer.Typer(
    help=" Manual document processing",
    no_args_is_help=True,
    rich_markup_mode=None  # Disable Rich formatting completely
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)

@ingest_app.command("file")
def ingest_file(
    path: str = typer.Argument(..., help="Path to file to ingest"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection name"),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Maximum characters per text chunk"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Character overlap between chunks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze file without ingesting"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing documents"),
):
    """Ingest a single file into collection."""
    handle_async(_ingest_file(path, collection, chunk_size, chunk_overlap, dry_run, force))

@ingest_app.command("folder")
def ingest_folder(
    path: str = typer.Argument(..., help="Path to folder to ingest"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection name"),
    formats: Optional[List[str]] = typer.Option(None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Maximum characters per text chunk"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Character overlap between chunks"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Process subdirectories recursively"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Glob patterns to exclude"),
    concurrency: int = typer.Option(5, "--concurrency", help="Number of concurrent processing tasks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze files without ingesting"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing documents"),
):
    """Ingest all files in a folder."""
    handle_async(_ingest_folder(
        path, collection, formats, chunk_size, chunk_overlap,
        recursive, exclude, concurrency, dry_run, force
    ))

@ingest_app.command("yaml")
def ingest_yaml_metadata(
    path: str = typer.Argument(..., help="Path to YAML metadata file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze metadata without processing"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing documents"),
):
    """ Process completed YAML metadata file."""
    handle_async(_ingest_yaml_metadata(path, dry_run, force))

@ingest_app.command("generate-yaml")
def generate_yaml_metadata(
    library_path: str = typer.Argument(..., help="Path to library folder"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target library collection name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output YAML file path"),
    formats: Optional[List[str]] = typer.Option(None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing YAML file"),
):
    """ Generate YAML metadata file for library documents."""
    handle_async(_generate_yaml_metadata(library_path, collection, output, formats, force))

@ingest_app.command("web")
def ingest_web_pages(
    url: str = typer.Argument(..., help="Root URL to crawl"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection name"),
    max_depth: int = typer.Option(2, "--depth", help="Maximum crawl depth"),
    max_pages: int = typer.Option(50, "--max-pages", help="Maximum number of pages to crawl"),
    include_patterns: Optional[List[str]] = typer.Option(None, "--include", help="URL patterns to include"),
    exclude_patterns: Optional[List[str]] = typer.Option(None, "--exclude", help="URL patterns to exclude"),
    delay: float = typer.Option(1.0, "--delay", help="Delay between requests (seconds)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze URLs without crawling"),
):
    """ Crawl and ingest web pages."""
    handle_async(_ingest_web_pages(
        url, collection, max_depth, max_pages,
        include_patterns, exclude_patterns, delay, dry_run
    ))

@ingest_app.command("status")
def ingestion_status(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    recent: bool = typer.Option(False, "--recent", help="Show only recent ingestions"),
):
    """ Show ingestion status and statistics."""
    handle_async(_ingestion_status(collection, recent))

# Async implementation functions
async def _ingest_file(
    path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    force: bool
):
    """Ingest a single file."""
    # Validate file existence first, before connecting to daemon
    file_path = Path(path)
    
    if not file_path.exists():
        print(f"❌ File not found: {path}")
        print("\nPlease check:")
        print("• File path is correct and accessible")
        print("• File has not been moved or deleted")
        print("• You have read permissions for this file")
        raise typer.Exit(1)

    if not file_path.is_file():
        print(f"❌ Path is not a regular file: {path}")
        print("\nThe path appears to be a directory or special file.")
        print("To process a folder, use: wqm ingest folder")
        raise typer.Exit(1)
    
    async def ingest_operation(daemon_client):
        # File validation already done above

        if dry_run:
            print(f"Analyzing File: {file_path.name}")

            # Analyze file without processing
            file_info = {
                "path": str(file_path),
                "size_mb": round(file_path.stat().st_size / (1024*1024), 2),
                "extension": file_path.suffix.lower(),
                "supported": file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']
            }

            # Display file analysis in plain text
            print("File Analysis:")
            print(f"Path: {file_info['path']}")
            print(f"Size: {file_info['size_mb']} MB")
            print(f"Extension: {file_info['extension']}")
            print(f"Supported: {'Yes' if file_info['supported'] else 'No'}")

            if file_info['supported']:
                print("File can be processed with current settings")
                estimated_chunks = max(1, int(file_info['size_mb'] * 1024 * 1024 / chunk_size))
                print(f"Estimated chunks: ~{estimated_chunks}")
            else:
                print(f"Error: Unsupported file format: {file_info['extension']}")

            return

        print(f"Ingesting File: {file_path.name}")

        # Process the file via daemon
        try:
            metadata = {
                "source": "cli",
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap)
            }
            
            response = await daemon_client.process_document(
                file_path=str(file_path),
                collection=collection,
                metadata=metadata,
                chunk_text=True
            )

            if response.success:
                print(f"Document processed successfully")
                print(f"Document ID: {response.document_id}")
                print(f"Chunks added: {response.chunks_added}")
                if response.applied_metadata:
                    print("Applied metadata:")
                    for key, value in response.applied_metadata.items():
                        print(f"  {key}: {value}")
            else:
                print(f"Error: Processing failed: {response.message}")
                raise typer.Exit(1)

        except Exception as e:
            print(f"Error: Processing failed: {e}")
            raise typer.Exit(1)

    try:
        await with_daemon_client(ingest_operation)
    except Exception as e:
        print(f"Error: Ingestion failed: {e}")
        raise typer.Exit(1)

async def _ingest_folder(
    path: str,
    collection: str,
    formats: Optional[List[str]],
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    exclude: Optional[List[str]],
    concurrency: int,
    dry_run: bool,
    force: bool
):
    """Ingest all files in a folder."""
    try:
        folder_path = Path(path)

        if not folder_path.exists():
            print(f"❌ Folder not found: {path}")
            print("\nPlease check:")
            print("• Folder path is correct and accessible")
            print("• Folder has not been moved or deleted")
            print("• You have read permissions for this folder")
            raise typer.Exit(1)

        if not folder_path.is_dir():
            print(f"❌ Path is not a directory: {path}")
            print("\nThe path appears to be a file, not a folder.")
            print("To process a single file, use: wqm ingest file")
            raise typer.Exit(1)

        # Default formats if not specified
        if not formats:
            formats = ['pdf', 'txt', 'md', 'docx']
        else:
            # Clean format specifications
            formats = [fmt.strip().lower().lstrip('.') for fmt in formats]

        # Find files to process
        files = []
        for fmt in formats:
            pattern = f"**/*.{fmt}" if recursive else f"*.{fmt}"
            files.extend(folder_path.glob(pattern))

        # Apply exclusion patterns
        if exclude:
            import fnmatch
            filtered_files = []
            for file_path in files:
                exclude_file = False
                for pattern in exclude:
                    if fnmatch.fnmatch(str(file_path), pattern):
                        exclude_file = True
                        break
                if not exclude_file:
                    filtered_files.append(file_path)
            files = filtered_files

        if not files:
            print(f"No files found matching criteria in {path}")
            return

        print(f" Found {len(files)} files to process")

        if dry_run:
            # Show analysis summary
            from rich.table import Table
            summary_table = Table(title="Folder Analysis Summary")
            summary_table.add_column("Format", style="cyan")
            summary_table.add_column("Count", justify="right")
            summary_table.add_column("Total Size (MB)", justify="right")

            format_stats = {}
            total_size = 0

            for file_path in files:
                ext = file_path.suffix.lower().lstrip('.')
                size_mb = file_path.stat().st_size / (1024*1024)

                if ext not in format_stats:
                    format_stats[ext] = {"count": 0, "size_mb": 0}
                format_stats[ext]["count"] += 1
                format_stats[ext]["size_mb"] += size_mb
                total_size += size_mb

            for ext, stats in format_stats.items():
                summary_table.add_row(
                    f".{ext}",
                    str(stats['count']),
                    f"{stats['size_mb']:.2f}"
                )

            summary_table.add_row(
                "Total",
                f"{len(files)}",
                f"{total_size:.2f}"
            )

            from rich.console import Console
            console = Console()
            console.print(summary_table)
            print(f" {len(files)} files ready for processing")
            return

        from ...core.config import Config
        from ...core.client import create_qdrant_client
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Initialize ingestion engine
        engine = DocumentIngestionEngine(
            client=client,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_concurrency=concurrency
        )

        # Process files with progress
        print(f" Processing {len(files)} files...")

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            main_task = progress.add_task("Overall progress", total=len(files))

            results = []
            processed = 0

            # Process files in batches based on concurrency
            for i in range(0, len(files), concurrency):
                batch = files[i:i+concurrency]

                # Process batch concurrently
                batch_tasks = []
                for file_path in batch:
                    task = engine.ingest_file(file_path)
                    batch_tasks.append(task)

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for file_path, result in zip(batch, batch_results, strict=False):
                    processed += 1
                    if isinstance(result, Exception):
                        print(f"Error: Failed to process {file_path.name}: {result}")
                        results.append(None)
                    else:
                        results.append(result)

                    progress.update(main_task, completed=processed)

        # Display summary
        successful_results = [r for r in results if r is not None]

        print("\n Folder ingestion completed!")
        print(f"  Successfully processed: {len(successful_results)}/{len(files)} files")

        if successful_results:
            total_chunks = sum(r.chunks_created for r in successful_results)
            total_chars = sum(r.total_characters for r in successful_results)
            print(f"  Total chunks created: {total_chunks}")
            print(f"  Total characters processed: {total_chars:,}")

    except Exception as e:
        print(f"Error: Folder ingestion failed: {e}")
        raise typer.Exit(1)

async def _generate_yaml_metadata(
    library_path: str,
    collection: str,
    output: Optional[str],
    formats: Optional[List[str]],
    force: bool
):
    """Generate YAML metadata file for library documents."""
    try:
        lib_path = Path(library_path)

        if not lib_path.exists():
            print(f"Error: Library path not found: {library_path}")
            raise typer.Exit(1)

        if not lib_path.is_dir():
            print(f"Error: Path is not a directory: {library_path}")
            raise typer.Exit(1)

        # Validate collection name (should start with _)
        if not collection.startswith('_'):
            print(f"Error: Library collection name must start with '_': {collection}")
            raise typer.Exit(1)

        # Check output path
        output_path = Path(output) if output else lib_path / 'metadata_completion.yaml'

        if output_path.exists() and not force:
            print(f"Error: Output file exists (use --force to overwrite): {output_path}")
            raise typer.Exit(1)

        print(" Generating YAML Metadata for Library")
        print(f"  Library Path: {lib_path}")
        print(f"  Collection: {collection}")
        print(f"  Output: {output_path}")

        # Create workflow
        from ...core.config import Config
        from ...core.client import create_qdrant_client
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Generate YAML file
        from rich.console import Console
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing documents and extracting metadata...", total=100)

            result_path = await workflow.generate_yaml_file(
                library_path=lib_path,
                library_collection=collection,
                output_path=output_path,
                formats=formats
            )

            progress.update(task, completed=100)

        if result_path:
            from rich.panel import Panel
            result_panel = Panel(
                f""" YAML metadata file generated successfully!

 Location: {result_path}
 Next steps:
  1. Review and complete the metadata in the YAML file
  2. Fill in fields marked with '?'
  3. Run: wqm ingest yaml {result_path}

💡 The file contains:
  • Detected metadata from document analysis
  • Required fields for each document type
  • Processing instructions and examples""",
                title="🎉 YAML Generation Complete",
                border_style="green"
            )
            console.print(result_panel)
        else:
            print(" No documents found to process")

    except Exception as e:
        print(f"Error: YAML generation failed: {e}")
        raise typer.Exit(1)

async def _ingest_yaml_metadata(path: str, dry_run: bool, force: bool):
    """Process completed YAML metadata file."""
    try:
        yaml_path = Path(path)

        if not yaml_path.exists():
            print(f"Error: YAML file not found: {path}")
            raise typer.Exit(1)

        print(f" Processing YAML Metadata: {yaml_path.name}")

        # Create workflow
        from ...core.config import Config
        from ...core.client import create_qdrant_client
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Process YAML file
        from rich.console import Console
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents with metadata...", total=100)

            results = await workflow.process_yaml_file(
                yaml_path=yaml_path,
                dry_run=dry_run
            )

            progress.update(task, completed=100)

        # Display results
        processed = results.get('processed', 0)
        skipped = results.get('skipped', 0)
        errors = results.get('errors', [])
        remaining = results.get('remaining', 0)

        if dry_run:
            from rich.panel import Panel
            result_panel = Panel(
                f""" YAML Metadata Analysis (Dry Run)

 Processing Summary:
  • Ready to process: {processed} documents
  • Missing metadata: {skipped} documents
  • Errors found: {len(errors)} documents
  • Remaining in YAML: {remaining} documents

{" All documents ready for processing!" if remaining == 0 else " Complete remaining metadata and run again"}

{chr(10).join(f" Error: {error}" for error in errors) if errors else ""}
{"  ... and more errors" if len(errors) > 5 else ""}""",
                title=" Dry Run Results",
                border_style="blue"
            )
        else:
            result_panel = Panel(
                f""" YAML Metadata Processing Complete!

 Processing Summary:
  • Successfully processed: {processed} documents
  • Skipped (incomplete metadata): {skipped} documents
  • Errors encountered: {len(errors)} documents
  • Remaining in YAML file: {remaining} documents

{"🎉 All documents processed successfully!" if remaining == 0 else f"📝 {remaining} documents still need metadata completion"}

{chr(10).join(f" Error: {error}" for error in errors) if errors else ""}
{"  ... and more errors" if len(errors) > 3 else ""}""",
                title="🎉 Processing Complete",
                border_style="green"
            )

        console.print(result_panel)

        # Show guidance for next steps
        if remaining > 0 and not dry_run:
            print(f"\nNote: The YAML file has been updated with {remaining} remaining documents.")
            print(f"   Complete their metadata and run 'wqm ingest yaml {yaml_path}' again.")

    except Exception as e:
        print(f"Error: YAML processing failed: {e}")
        logger.exception("YAML metadata processing error")
        raise typer.Exit(1)

async def _ingest_web_pages(
    url: str,
    collection: str,
    max_depth: int,
    max_pages: int,
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    delay: float,
    dry_run: bool
):
    """Crawl and ingest web pages."""
    try:
        print(f" Web Crawling: {url}")

        # TODO: Implement web crawling and ingestion
        # This will be part of future enhancement

        if dry_run:
            print(" Web crawling analysis (dry run)")
            print(f"Would crawl {url} with max depth {max_depth} and max {max_pages} pages")
            print("Web crawling feature will be implemented in a future task")
        else:
            print(" Web crawling")
            print("Web crawling feature will be implemented in a future task")

    except Exception as e:
        print(f"Error: Web crawling failed: {e}")
        raise typer.Exit(1)

async def _ingestion_status(collection: Optional[str], recent: bool):
    """Show ingestion status and statistics."""
    try:
        print(" Ingestion Status")

        from ...core.config import Config
        from ...core.client import create_qdrant_client
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Get all collections or filter by specific collection
        if collection:
            collections = [{'name': collection}]  # Simple format for single collection
        else:
            collections = await client.list_collections()

        from rich.table import Table
        status_table = Table(title="Collection Status")
        status_table.add_column("Collection", style="cyan")
        status_table.add_column("Points", justify="right")
        status_table.add_column("Type", style="white")
        status_table.add_column("Status", justify="center")

        for col in collections:
            name = col.get("name", "unknown")

            try:
                info = await client.get_collection_info(name)
                points = info.get("points_count", 0)
                col_type = "Library" if name.startswith("_") else "Project"

                # Determine status
                if points == 0:
                    status = "Empty"
                elif points < 100:
                    status = "Small"
                else:
                    status = "Active"

                status_table.add_row(name, str(points), col_type, status)

            except Exception:
                status_table.add_row(name, "?", "Unknown", "Error")

        from rich.console import Console
        console = Console()
        console.print(status_table)

        # Show recent activity if requested
        if recent:
            print("\nRecent activity tracking will be implemented in future updates")

    except Exception as e:
        print(f"Error: Status check failed: {e}")
        raise typer.Exit(1)

def _display_ingestion_result(result: IngestionResult, filename: str):
    """Display ingestion result summary."""
    from rich.panel import Panel
    from rich.console import Console

    if result.success:
        result_panel = Panel(
            f""" Successfully ingested: {filename}

 Processing Summary:
  • Chunks created: {result.chunks_created}
  • Total characters: {result.total_characters:,}
  • Processing time: {result.processing_time_seconds:.2f}s
  • Average chunk size: {result.total_characters // max(1, result.chunks_created)} chars

 Collection: {result.collection_name}""",
            title="🎉 Ingestion Complete",
            border_style="green"
        )
    else:
        result_panel = Panel(
            f"""Error: Failed to ingest: {filename} Error: {result.error_message}

 Partial Results:
  • Chunks created: {result.chunks_created}
  • Processing time: {result.processing_time_seconds:.2f}s""",
            title="💥 Ingestion Failed",
            border_style="red"
        )

    console = Console()
    console.print(result_panel)
