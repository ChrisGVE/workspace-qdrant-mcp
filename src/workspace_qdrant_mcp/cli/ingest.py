"""
Command-line interface for batch document ingestion.

This module provides the main CLI interface for the workspace-qdrant-mcp
batch document ingestion system. It handles argument parsing, progress
reporting, and user interaction for the ingestion process.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..core.client import QdrantWorkspaceClient
from ..core.config import Config
from .ingestion_engine import DocumentIngestionEngine, IngestionResult, IngestionStats

# Configure logging
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()

# Typer app instance
app = typer.Typer(
    name="workspace-qdrant-ingest",
    help="Batch document ingestion for workspace-qdrant-mcp",
    no_args_is_help=True,
)


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging for CLI operations."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("workspace_qdrant_ingest.log"),
        ],
    )


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to directory or file to ingest"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection name"
    ),
    formats: list[str] | None = typer.Option(
        None, "--formats", "-f", help="File formats to process (e.g., pdf,md,txt)"
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", help="Number of concurrent processing tasks"
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Maximum characters per text chunk"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Character overlap between chunks"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Analyze files without ingesting"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Process subdirectories recursively"
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude", help="Glob patterns to exclude"
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress bar"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """
    Ingest documents from a directory into a Qdrant collection.

    This command processes documents in various formats and adds them to the
    specified workspace collection with automatic chunking, embedding generation,
    and metadata extraction.

    Examples:
        # Basic ingestion
        workspace-qdrant-ingest /path/to/docs --collection my-project

        # PDF and Markdown only with high concurrency
        workspace-qdrant-ingest /path/to/docs -c my-project -f pdf,md --concurrency 10

        # Dry run to preview operation
        workspace-qdrant-ingest /path/to/docs -c my-project --dry-run

        # Exclude certain patterns
        workspace-qdrant-ingest /path/to/docs -c my-project --exclude "*.tmp,**/cache/**"
    """
    setup_logging(verbose, debug)

    # Run the async ingestion process
    asyncio.run(
        _run_ingestion(
            path=path,
            collection=collection,
            formats=formats,
            concurrency=concurrency,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dry_run=dry_run,
            recursive=recursive,
            exclude_patterns=exclude,
            show_progress=progress,
            auto_confirm=yes,
        )
    )


@app.command()
def formats() -> None:
    """List supported file formats and their options."""
    asyncio.run(_show_formats())


@app.command()
def estimate(
    path: str = typer.Argument(..., help="Path to directory to analyze"),
    formats: list[str] | None = typer.Option(
        None, "--formats", "-f", help="File formats to analyze"
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", help="Concurrent processing tasks for estimation"
    ),
) -> None:
    """Estimate processing time and resource requirements."""
    asyncio.run(_estimate_processing(path, formats, concurrency))


async def _run_ingestion(
    path: str,
    collection: str,
    formats: list[str] | None,
    concurrency: int,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    recursive: bool,
    exclude_patterns: list[str] | None,
    show_progress: bool,
    auto_confirm: bool,
) -> None:
    """Run the main ingestion process."""

    try:
        # Initialize client
        console.print("🚀 Initializing workspace client...", style="blue")
        config = Config()
        client = QdrantWorkspaceClient(config)
        await client.initialize()

        console.print(f"✅ Connected to Qdrant at {config.qdrant.url}", style="green")

        # Show project info
        project_info = client.get_project_info()
        if project_info:
            console.print(f"📁 Project: {project_info['main_project']}", style="cyan")

        # Initialize ingestion engine
        engine = DocumentIngestionEngine(
            client=client,
            concurrency=concurrency,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Get estimation
        console.print("📊 Analyzing directory...", style="blue")
        estimation = await engine.estimate_processing_time(path, formats)

        # Display estimation
        _display_estimation(estimation, dry_run)

        # Confirmation (unless dry run or auto-confirmed)
        if not dry_run and not auto_confirm:
            if not typer.confirm("\n🤔 Proceed with ingestion?"):
                console.print("❌ Operation cancelled", style="red")
                return

        # Run ingestion with progress tracking
        progress_task = None
        if show_progress:
            progress_task = _create_progress_tracker()

        try:
            result = await engine.process_directory(
                directory_path=path,
                collection=collection,
                formats=formats,
                dry_run=dry_run,
                recursive=recursive,
                exclude_patterns=exclude_patterns,
                progress_callback=_create_progress_callback(progress_task)
                if progress_task
                else None,
            )

        finally:
            if progress_task:
                progress_task.stop()

        # Display results
        _display_results(result)

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if "client" in locals():
            await client.close()


def _display_estimation(estimation: dict[str, Any], dry_run: bool) -> None:
    """Display processing time estimation."""

    table = Table(title="📈 Processing Estimation", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Files found", f"{estimation['files_found']:,}")
    table.add_row("Total size", f"{estimation['total_size_mb']:.1f} MB")
    table.add_row("Estimated time", estimation["estimated_time_human"])

    # File type breakdown
    if estimation["file_types"]:
        table.add_row("", "")  # Separator
        table.add_row("File types:", "")
        for file_type, count in estimation["file_types"].items():
            table.add_row(f"  {file_type}", f"{count:,} files")

    console.print(table)

    if dry_run:
        console.print(
            "\n🔍 Running in DRY RUN mode - no documents will be ingested",
            style="yellow bold",
        )


def _create_progress_tracker() -> Progress:
    """Create a rich progress tracker."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )
    progress.start()
    return progress


def _create_progress_callback(progress: Progress) -> callable:
    """Create progress callback for file processing updates."""
    task_id = progress.add_task("Processing files...", total=None)

    def update_progress(completed: int, total: int, stats: IngestionStats) -> None:
        if progress.tasks[task_id].total != total:
            progress.update(task_id, total=total)

        description = f"Processing files... ({stats.files_processed} processed, {stats.files_failed} failed)"
        progress.update(task_id, completed=completed, description=description)

    return update_progress


def _display_results(result: IngestionResult) -> None:
    """Display ingestion results with rich formatting."""

    stats = result.stats

    # Create status panel
    if result.success:
        status = "✅ SUCCESS" if not result.dry_run else "🔍 DRY RUN COMPLETE"
    else:
        status = "❌ FAILED"

    # Main results table
    table = Table(title=f"📊 Ingestion Results - {status}", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    # Basic stats
    table.add_row("Collection", result.collection)
    table.add_row("Operation", "Analysis" if result.dry_run else "Ingestion")
    table.add_row("", "")  # Separator

    # File processing stats
    table.add_row("Files found", f"{stats.files_found:,}")
    table.add_row("Files processed", f"{stats.files_processed:,}")
    if stats.files_skipped > 0:
        table.add_row("Files skipped", f"{stats.files_skipped:,}")
    if stats.files_failed > 0:
        table.add_row("Files failed", f"{stats.files_failed:,}")
    if stats.duplicates_found > 0:
        table.add_row("Duplicates skipped", f"{stats.duplicates_found:,}")

    table.add_row("", "")  # Separator

    # Content stats
    table.add_row("Documents created", f"{stats.total_documents:,}")
    table.add_row("Text chunks", f"{stats.total_chunks:,}")
    table.add_row("Total characters", f"{stats.total_characters:,}")
    table.add_row("Total words", f"{stats.total_words:,}")

    table.add_row("", "")  # Separator

    # Performance stats
    table.add_row("Processing time", f"{stats.processing_time:.2f}s")
    table.add_row("Processing rate", f"{stats.files_per_second:.1f} files/sec")
    table.add_row("Success rate", f"{stats.success_rate:.1f}%")

    console.print(table)

    # Show detailed message
    if result.message:
        console.print(f"\n{result.message}")

    # Show errors if any
    if stats.errors:
        console.print("\n❌ Errors encountered:", style="red bold")
        error_table = Table(show_header=True)
        error_table.add_column("File", style="red")
        error_table.add_column("Error", style="yellow")
        error_table.add_column("Parser", style="cyan")

        for error in stats.errors[:10]:  # Show first 10 errors
            error_table.add_row(
                error.get("file", "unknown"),
                error.get("error", "unknown error")[:80] + "..."
                if len(error.get("error", "")) > 80
                else error.get("error", ""),
                error.get("parser", "unknown"),
            )

        console.print(error_table)

        if len(stats.errors) > 10:
            console.print(f"... and {len(stats.errors) - 10} more errors")

    # Show skipped files if any
    if stats.skipped_files:
        console.print(f"\n⏭️  {len(stats.skipped_files)} files skipped")
        if len(stats.skipped_files) <= 5:
            for skip_info in stats.skipped_files:
                console.print(
                    f"  • {skip_info['file']}: {skip_info['reason']}", style="yellow"
                )
        else:
            console.print("  Run with --verbose to see details", style="dim")


async def _show_formats() -> None:
    """Display supported file formats."""

    try:
        # Get format info without initializing full client
        from .parsers import MarkdownParser, PDFParser, TextParser

        parsers = [
            TextParser(),
            MarkdownParser(),
            PDFParser(),
        ]

        console.print("📄 Supported File Formats", style="bold blue")
        console.print()

        for parser in parsers:
            # Format header
            console.print(f"🔹 {parser.format_name}", style="bold cyan")

            # Extensions
            ext_text = ", ".join(parser.supported_extensions)
            console.print(f"   Extensions: {ext_text}")

            # Parsing options
            options = parser.get_parsing_options()
            if options:
                console.print("   Options:")
                for option_name, option_info in options.items():
                    default_val = option_info.get("default", "None")
                    desc = option_info.get("description", "No description")
                    console.print(
                        f"     • {option_name}: {desc} (default: {default_val})"
                    )

            console.print()

    except Exception as e:
        console.print(f"❌ Error getting format info: {e}", style="red")
        sys.exit(1)


async def _estimate_processing(
    path: str, formats: list[str] | None, concurrency: int
) -> None:
    """Display processing time estimation."""

    try:
        # Simple file discovery without full client
        from .parsers import MarkdownParser, PDFParser, TextParser

        # Create a minimal engine for file analysis
        parsers = [TextParser(), MarkdownParser(), PDFParser()]

        # Find files manually
        directory_path = Path(path)
        if not directory_path.exists():
            console.print(f"❌ Directory not found: {path}", style="red")
            return

        # Get supported extensions
        supported_extensions = set()
        format_filter = set(formats) if formats else None

        for parser in parsers:
            # Check if parser should be included based on format filter
            include_parser = False
            if not format_filter:
                include_parser = True
            else:
                # Check format name matches
                parser_name_lower = parser.format_name.lower()
                if any(fmt.lower() in parser_name_lower for fmt in format_filter):
                    include_parser = True
                # Also check if any requested formats match parser extensions
                elif any(f".{fmt.lower()}" in parser.supported_extensions for fmt in format_filter):
                    include_parser = True

            if include_parser:
                supported_extensions.update(parser.supported_extensions)

        # Find files
        files = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)

        # Calculate stats
        total_size = 0
        file_types = {}

        for file_path in files:
            try:
                size = file_path.stat().st_size
                total_size += size

                # Find parser for type classification
                for parser in parsers:
                    if parser.can_parse(file_path):
                        file_types[parser.format_name] = (
                            file_types.get(parser.format_name, 0) + 1
                        )
                        break
            except OSError:
                continue

        # Simple estimation (adjust based on empirical data)
        estimated_seconds = (total_size / (1024 * 1024)) * 2  # ~2 seconds per MB
        estimated_seconds /= concurrency  # Account for concurrency

        estimation = {
            "files_found": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_types,
            "estimated_time_seconds": estimated_seconds,
            "estimated_time_human": f"{estimated_seconds // 60:.0f}m {estimated_seconds % 60:.0f}s",
        }

        console.print("⏱️  Processing Time Estimation", style="bold blue")
        console.print()

        _display_estimation(estimation, False)

    except Exception as e:
        console.print(f"❌ Error during estimation: {e}", style="red")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
