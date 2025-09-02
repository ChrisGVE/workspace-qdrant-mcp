
from .....observability import get_logger
logger = get_logger(__name__)
"""Document ingestion CLI commands.

This module provides manual document processing capabilities for the
unified wqm CLI, handling various file formats and ingestion workflows.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...cli.ingestion_engine import DocumentIngestionEngine, IngestionResult
from ...core.client import create_qdrant_client
from ...core.config import Config
from ...core.yaml_metadata import YamlMetadataWorkflow

console = Console()
logger = logging.getLogger(__name__)

# Create the ingest app
ingest_app = typer.Typer(help="📁 Manual document processing")

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.logger.info("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.logger.info("[red]Error: {e}[/red]")
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
    """📄 Ingest a single file into collection."""
    handle_async(_ingest_file(path, collection, chunk_size, chunk_overlap, dry_run, force))

@ingest_app.command("folder")
def ingest_folder(
    path: str = typer.Argument(..., help="Path to folder to ingest"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection name"),
    formats: list[str] | None = typer.Option(None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Maximum characters per text chunk"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Character overlap between chunks"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Process subdirectories recursively"),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Glob patterns to exclude"),
    concurrency: int = typer.Option(5, "--concurrency", help="Number of concurrent processing tasks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze files without ingesting"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing documents"),
):
    """📁 Ingest all files in a folder."""
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
    """📋 Process completed YAML metadata file."""
    handle_async(_ingest_yaml_metadata(path, dry_run, force))

@ingest_app.command("generate-yaml")
def generate_yaml_metadata(
    library_path: str = typer.Argument(..., help="Path to library folder"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target library collection name"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output YAML file path"),
    formats: list[str] | None = typer.Option(None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing YAML file"),
):
    """📋 Generate YAML metadata file for library documents."""
    handle_async(_generate_yaml_metadata(library_path, collection, output, formats, force))

@ingest_app.command("web")
def ingest_web_pages(
    url: str = typer.Argument(..., help="Root URL to crawl"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection name"),
    max_depth: int = typer.Option(2, "--depth", help="Maximum crawl depth"),
    max_pages: int = typer.Option(50, "--max-pages", help="Maximum number of pages to crawl"),
    include_patterns: list[str] | None = typer.Option(None, "--include", help="URL patterns to include"),
    exclude_patterns: list[str] | None = typer.Option(None, "--exclude", help="URL patterns to exclude"),
    delay: float = typer.Option(1.0, "--delay", help="Delay between requests (seconds)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze URLs without crawling"),
):
    """🌐 Crawl and ingest web pages."""
    handle_async(_ingest_web_pages(
        url, collection, max_depth, max_pages,
        include_patterns, exclude_patterns, delay, dry_run
    ))

@ingest_app.command("status")
def ingestion_status(
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    recent: bool = typer.Option(False, "--recent", help="Show only recent ingestions"),
):
    """📊 Show ingestion status and statistics."""
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
    try:
        file_path = Path(path)

        if not file_path.exists():
            console.logger.info("[red]❌ File not found: {path}[/red]")
            raise typer.Exit(1)

        if not file_path.is_file():
            console.logger.info("[red]❌ Path is not a file: {path}[/red]")
            raise typer.Exit(1)

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Initialize ingestion engine
        engine = DocumentIngestionEngine(
            client=client,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if dry_run:
            console.logger.info("[bold blue]📋 Analyzing File: {file_path.name}[/bold blue]")

            # Analyze file without processing
            file_info = {
                "path": str(file_path),
                "size_mb": round(file_path.stat().st_size / (1024*1024), 2),
                "extension": file_path.suffix.lower(),
                "supported": file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.epub', '.docx', '.pptx', '.html', '.htm', '.mobi', '.azw', '.py', '.js', '.java', '.cpp', '.go', '.rs', '.rb', '.php']
            }

            info_table = Table(title="File Analysis")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")

            info_table.add_row("Path", file_info["path"])
            info_table.add_row("Size", f"{file_info['size_mb']} MB")
            info_table.add_row("Extension", file_info["extension"])
            info_table.add_row("Supported", "✅ Yes" if file_info["supported"] else "❌ No")

            console.logger.info("Output", data=info_table)

            if file_info["supported"]:
                console.logger.info("[green]✅ File can be processed with current settings[/green]")
                estimated_chunks = max(1, int(file_info["size_mb"] * 1024 * 1024 / chunk_size))
                console.logger.info("[dim]Estimated chunks: ~{estimated_chunks}[/dim]")
            else:
                console.logger.info("[red]❌ Unsupported file format: {file_info['extension']}[/red]")

            return

        console.logger.info("[bold blue]📄 Ingesting File: {file_path.name}[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing file...", total=100)

            # Process the file
            result = await engine.ingest_file(
                file_path=file_path,
                progress_callback=lambda p: progress.update(task, completed=p)
            )

            _display_ingestion_result(result, file_path.name)

    except Exception as e:
        console.logger.info("[red]❌ Ingestion failed: {e}[/red]")
        raise typer.Exit(1)

async def _ingest_folder(
    path: str,
    collection: str,
    formats: list[str] | None,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    exclude: list[str] | None,
    concurrency: int,
    dry_run: bool,
    force: bool
):
    """Ingest all files in a folder."""
    try:
        folder_path = Path(path)

        if not folder_path.exists():
            console.logger.info("[red]❌ Folder not found: {path}[/red]")
            raise typer.Exit(1)

        if not folder_path.is_dir():
            console.logger.info("[red]❌ Path is not a directory: {path}[/red]")
            raise typer.Exit(1)

        # Default formats if not specified
        if not formats:
            formats = ["pdf", "txt", "md", "epub", "docx", "pptx", "html", "htm", "mobi", "py", "js", "java", "cpp", "go", "rs"]
        else:
            # Clean format specifications
            formats = [f.lower().lstrip('.') for f in formats]

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
            console.logger.info("[yellow]No files found matching criteria in {path}[/yellow]")
            return

        console.logger.info("[bold blue]📁 Found {len(files)} files to process[/bold blue]")

        if dry_run:
            # Show analysis summary
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
                    str(stats["count"]),
                    f"{stats['size_mb']:.2f}"
                )

            summary_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{len(files)}[/bold]",
                f"[bold]{total_size:.2f}[/bold]"
            )

            console.logger.info("Output", data=summary_table)
            console.logger.info("[green]✅ {len(files)} files ready for processing[/green]")
            return

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
        console.logger.info("[bold blue]📁 Processing {len(files)} files...[/bold blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
                        console.logger.info("[red]❌ Failed to process {file_path.name}: {result}[/red]")
                        results.append(None)
                    else:
                        results.append(result)

                    progress.update(main_task, completed=processed)

        # Display summary
        successful_results = [r for r in results if r is not None]

        console.logger.info("\n[green]✅ Folder ingestion completed![/green]")
        console.logger.info("  Successfully processed: {len(successful_results)}/{len(files)} files")

        if successful_results:
            total_chunks = sum(r.chunks_created for r in successful_results)
            total_chars = sum(r.total_characters for r in successful_results)
            console.logger.info("  Total chunks created: {total_chunks}")
            console.logger.info("  Total characters processed: {total_chars:,}")

    except Exception as e:
        console.logger.info("[red]❌ Folder ingestion failed: {e}[/red]")
        raise typer.Exit(1)

async def _generate_yaml_metadata(
    library_path: str,
    collection: str,
    output: str | None,
    formats: list[str] | None,
    force: bool
):
    """Generate YAML metadata file for library documents."""
    try:
        lib_path = Path(library_path)

        if not lib_path.exists():
            console.logger.info("[red]❌ Library path not found: {library_path}[/red]")
            raise typer.Exit(1)

        if not lib_path.is_dir():
            console.logger.info("[red]❌ Path is not a directory: {library_path}[/red]")
            raise typer.Exit(1)

        # Validate collection name (should start with _)
        if not collection.startswith('_'):
            console.logger.info("[red]❌ Library collection name must start with '_': {collection}[/red]")
            raise typer.Exit(1)

        # Check output path
        output_path = Path(output) if output else lib_path / 'metadata_completion.yaml'

        if output_path.exists() and not force:
            console.logger.info("[red]❌ Output file exists (use --force to overwrite): {output_path}[/red]")
            raise typer.Exit(1)

        console.logger.info("[bold blue]📋 Generating YAML Metadata for Library[/bold blue]")
        console.logger.info("  Library Path: {lib_path}")
        console.logger.info("  Collection: {collection}")
        console.logger.info("  Output: {output_path}")

        # Create workflow
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Generate YAML file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
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
            result_panel = Panel(
                f"""[green]✅ YAML metadata file generated successfully![/green]

📁 Location: {result_path}
📋 Next steps:
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
            console.logger.info("Output", data=result_panel)
        else:
            console.logger.info("[yellow]⚠️ No documents found to process[/yellow]")

    except Exception as e:
        console.logger.info("[red]❌ YAML generation failed: {e}[/red]")
        raise typer.Exit(1)

async def _ingest_yaml_metadata(path: str, dry_run: bool, force: bool):
    """Process completed YAML metadata file."""
    try:
        yaml_path = Path(path)

        if not yaml_path.exists():
            console.logger.info("[red]❌ YAML file not found: {path}[/red]")
            raise typer.Exit(1)

        console.logger.info("[bold blue]📋 Processing YAML Metadata: {yaml_path.name}[/bold blue]")

        # Create workflow
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Process YAML file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents with metadata...", total=100)

            results = await workflow.process_yaml_file(
                yaml_path=yaml_path,
                dry_run=dry_run
            )

            progress.update(task, completed=100)

        # Display results
        processed = results['processed']
        skipped = results['skipped']
        errors = results['errors']
        remaining = results['remaining']

        if dry_run:
            result_panel = Panel(
                f"""[blue]📋 YAML Metadata Analysis (Dry Run)[/blue]

📊 Processing Summary:
  • Ready to process: {processed} documents
  • Missing metadata: {skipped} documents
  • Errors found: {len(errors)} documents
  • Remaining in YAML: {remaining} documents

{"✅ All documents ready for processing!" if remaining == 0 else "⚠️ Complete remaining metadata and run again"}

{chr(10).join(f"  ❌ {error}" for error in errors[:5]) if errors else ""}
{"  ... and more errors" if len(errors) > 5 else ""}""",
                title="🔍 Dry Run Results",
                border_style="blue"
            )
        else:
            result_panel = Panel(
                f"""[green]✅ YAML Metadata Processing Complete![/green]

📊 Processing Summary:
  • Successfully processed: {processed} documents
  • Skipped (incomplete metadata): {skipped} documents
  • Errors encountered: {len(errors)} documents
  • Remaining in YAML file: {remaining} documents

{"🎉 All documents processed successfully!" if remaining == 0 else f"📝 {remaining} documents still need metadata completion"}

{chr(10).join(f"  ❌ {error}" for error in errors[:3]) if errors else ""}
{"  ... and more errors" if len(errors) > 3 else ""}""",
                title="🎉 Processing Complete",
                border_style="green"
            )

        console.logger.info("Output", data=result_panel)

        # Show guidance for next steps
        if remaining > 0 and not dry_run:
            console.logger.info("\n[dim]💡 The YAML file has been updated with {remaining} remaining documents.[/dim]")
            console.logger.info("[dim]   Complete their metadata and run 'wqm ingest yaml {yaml_path}' again.[/dim]")

    except Exception as e:
        console.logger.info("[red]❌ YAML processing failed: {e}[/red]")
        logger.exception("YAML metadata processing error")
        raise typer.Exit(1)

async def _ingest_web_pages(
    url: str,
    collection: str,
    max_depth: int,
    max_pages: int,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    delay: float,
    dry_run: bool
):
    """Crawl and ingest web pages."""
    try:
        console.logger.info("[bold blue]🌐 Web Crawling: {url}[/bold blue]")

        # TODO: Implement web crawling and ingestion
        # This will be part of future enhancement

        if dry_run:
            console.logger.info("[yellow]🌐 Web crawling analysis (dry run)[/yellow]")
            console.logger.info("Would crawl {url} with max depth {max_depth} and max {max_pages} pages")
            console.logger.info("Web crawling feature will be implemented in a future task")
        else:
            console.logger.info("[yellow]🌐 Web crawling[/yellow]")
            console.logger.info("Web crawling feature will be implemented in a future task")

    except Exception as e:
        console.logger.info("[red]❌ Web crawling failed: {e}[/red]")
        raise typer.Exit(1)

async def _ingestion_status(collection: str | None, recent: bool):
    """Show ingestion status and statistics."""
    try:
        console.logger.info("[bold blue]📊 Ingestion Status[/bold blue]")

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Get all collections or filter by specific collection
        if collection:
            collections = [{"name": collection}]
        else:
            collections = await client.list_collections()

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
                    status = "[red]Empty[/red]"
                elif points < 100:
                    status = "[yellow]Small[/yellow]"
                else:
                    status = "[green]Active[/green]"

                status_table.add_row(name, str(points), col_type, status)

            except Exception:
                status_table.add_row(name, "?", "Unknown", "[red]Error[/red]")

        console.logger.info("Output", data=status_table)

        # Show recent activity if requested
        if recent:
            console.logger.info("\n[dim]Recent activity tracking will be implemented in future updates[/dim]")

    except Exception as e:
        console.logger.info("[red]❌ Status check failed: {e}[/red]")
        raise typer.Exit(1)

def _display_ingestion_result(result: IngestionResult, filename: str):
    """Display ingestion result summary."""

    if result.success:
        result_panel = Panel(
            f"""[green]✅ Successfully ingested: {filename}[/green]

📊 Processing Summary:
  • Chunks created: {result.chunks_created}
  • Total characters: {result.total_characters:,}
  • Processing time: {result.processing_time_seconds:.2f}s
  • Average chunk size: {result.total_characters // max(1, result.chunks_created)} chars

📁 Collection: {result.collection_name}""",
            title="🎉 Ingestion Complete",
            border_style="green"
        )
    else:
        result_panel = Panel(
            f"""[red]❌ Failed to ingest: {filename}[/red]

Error: {result.error_message}

📊 Partial Results:
  • Chunks created: {result.chunks_created}
  • Processing time: {result.processing_time_seconds:.2f}s""",
            title="💥 Ingestion Failed",
            border_style="red"
        )

    console.logger.info("Output", data=result_panel)
