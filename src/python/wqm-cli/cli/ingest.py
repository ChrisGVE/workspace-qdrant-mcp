"""
Command-line interface for batch document ingestion.

This module provides the main CLI interface for the workspace-qdrant-mcp
batch document ingestion system. It handles argument parsing, progress
reporting, and user interaction for the ingestion process.
"""

import asyncio
import logging
import sys
import tempfile
from datetime import datetime, timezone
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

from common.core.daemon_client import get_daemon_client, with_daemon_client
from common.core.yaml_config import load_config
from .ingestion_engine import DocumentIngestionEngine, IngestionResult, IngestionStats
from .parsers import SecurityConfig, WebIngestionInterface, create_secure_web_parser

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


@app.command()
def ingest_web(
    url: str = typer.Argument(..., help="URL to crawl and ingest"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection name"
    ),
    max_pages: int = typer.Option(
        1, "--max-pages", help="Maximum pages to crawl (1 = single page)"
    ),
    max_depth: int = typer.Option(
        0, "--max-depth", help="Maximum crawl depth (0 = no following links)"
    ),
    allowed_domains: list[str] | None = typer.Option(
        None, "--allowed-domains", help="Comma-separated list of allowed domains"
    ),
    request_delay: float = typer.Option(
        1.0, "--request-delay", help="Delay between requests in seconds"
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Maximum characters per text chunk"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Character overlap between chunks"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Analyze content without ingesting"
    ),
    disable_security: bool = typer.Option(
        False, "--disable-security", help="Disable security scanning (NOT RECOMMENDED)"
    ),
    allow_all_domains: bool = typer.Option(
        False, "--allow-all-domains", help="Allow crawling any domain (security risk)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """
    Ingest web content from URLs with security hardening.

    This command safely crawls web content and ingests it into a Qdrant collection.
    Includes malware protection, domain restrictions, and respectful crawling.

    Examples:
        # Ingest single page
        workspace-qdrant-ingest ingest-web https://example.com/docs --collection docs

        # Crawl multiple pages with restrictions
        workspace-qdrant-ingest ingest-web https://example.com/docs -c docs \
            --max-pages 10 --max-depth 2 --allowed-domains example.com

        # Dry run to preview
        workspace-qdrant-ingest ingest-web https://example.com/docs -c docs --dry-run
    """
    setup_logging(verbose, debug)

    asyncio.run(
        _run_web_ingestion(
            url=url,
            collection=collection,
            max_pages=max_pages,
            max_depth=max_depth,
            allowed_domains=allowed_domains,
            request_delay=request_delay,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dry_run=dry_run,
            disable_security=disable_security,
            allow_all_domains=allow_all_domains,
            auto_confirm=yes,
        )
    )


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
        # Initialize daemon client
        console.print("🚀 Connecting to daemon...", style="blue")
        config = load_config()
        daemon_client = get_daemon_client(config)
        await daemon_client.connect()

        console.print(f"✅ Connected to daemon at {config.daemon.grpc.host}:{config.daemon.grpc.port}", style="green")

        # Show system status
        system_status = await daemon_client.get_system_status()
        console.print(f"📁 Daemon status: {system_status.status}", style="cyan")

        # Process folder using daemon with progress tracking
        include_patterns = [f"*.{fmt}" for fmt in formats] if formats else None
        ignore_patterns = exclude_patterns or []
        
        # Show basic estimation first
        directory_path = Path(path)
        if directory_path.exists():
            file_count = len(list(directory_path.rglob("*")))
            console.print(f"📊 Found approximately {file_count} files to analyze", style="blue")
        
        # Confirmation (unless dry run or auto-confirmed)
        if not dry_run and not auto_confirm:
            if not typer.confirm("\n🤔 Proceed with ingestion?"):
                console.print("❌ Operation cancelled", style="red")
                return

        # Run ingestion with progress tracking via daemon
        progress_task = None
        if show_progress:
            progress_task = _create_progress_tracker()

        stats = {
            'files_found': 0,
            'files_processed': 0, 
            'files_failed': 0,
            'files_skipped': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'total_characters': 0,
            'total_words': 0,
            'errors': [],
            'start_time': datetime.now(timezone.utc)
        }
        
        try:
            # Process folder via daemon
            async for progress in daemon_client.process_folder(
                folder_path=path,
                collection=collection,
                include_patterns=include_patterns,
                ignore_patterns=ignore_patterns,
                recursive=recursive,
                dry_run=dry_run
            ):
                # Update stats from progress
                stats['files_found'] = progress.total_files
                stats['files_processed'] = progress.processed_files
                stats['files_failed'] = progress.failed_files
                
                if progress_task:
                    task_id = progress_task.task_ids[0] if progress_task.task_ids else progress_task.add_task("Processing files...", total=progress.total_files)
                    progress_task.update(
                        task_id,
                        completed=progress.processed_files,
                        total=progress.total_files,
                        description=f"Processing files... ({progress.processed_files} processed, {progress.failed_files} failed)"
                    )

        finally:
            if progress_task:
                progress_task.stop()
            
        # Build result from daemon processing
        stats['end_time'] = datetime.now(timezone.utc)
        stats['processing_time'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        # Create result structure compatible with existing display code
        from .ingestion_engine import IngestionStats, IngestionResult
        
        ingestion_stats = IngestionStats(
            files_found=stats['files_found'],
            files_processed=stats['files_processed'],
            files_failed=stats['files_failed'],
            files_skipped=stats['files_skipped'],
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            total_characters=stats['total_characters'],
            total_words=stats['total_words'],
            start_time=stats['start_time'],
            end_time=stats['end_time'],
            processing_time=stats['processing_time'],
            errors=stats['errors']
        )
        
        result = IngestionResult(
            success=stats['files_failed'] == 0,
            stats=ingestion_stats,
            collection=collection,
            message=f"Processed {stats['files_processed']} files via daemon",
            dry_run=dry_run
        )

        # Display results
        _display_results(result)

        # Exit with appropriate code
        raise typer.Exit(0 if result.success else 1)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise typer.Exit(1)

    finally:
        if "daemon_client" in locals():
            await daemon_client.disconnect()


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
        raise typer.Exit(1)


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
                elif any(
                    f".{fmt.lower()}" in parser.supported_extensions
                    for fmt in format_filter
                ):
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
        raise typer.Exit(1)


async def _run_web_ingestion(
    url: str,
    collection: str,
    max_pages: int,
    max_depth: int,
    allowed_domains: list[str] | None,
    request_delay: float,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    disable_security: bool,
    allow_all_domains: bool,
    auto_confirm: bool,
) -> None:
    """Run web content ingestion."""

    try:
        # Security warnings
        if disable_security:
            console.print(
                "⚠️  Security scanning is DISABLED. This is not recommended!",
                style="red bold",
            )
            if not auto_confirm and not typer.confirm(
                "Continue with disabled security?"
            ):
                console.print("❌ Operation cancelled", style="red")
                return

        if allow_all_domains and not allowed_domains:
            console.print(
                "⚠️  All domains are allowed. This increases security risk!",
                style="yellow bold",
            )
            if not auto_confirm and not typer.confirm("Continue allowing all domains?"):
                console.print("❌ Operation cancelled", style="red")
                return

        # Initialize daemon client
        console.print("🚀 Connecting to daemon...", style="blue")
        config = load_config()
        daemon_client = get_daemon_client(config)
        await daemon_client.connect()

        console.print(f"✅ Connected to daemon at {config.daemon.grpc.host}:{config.daemon.grpc.port}", style="green")

        # Show system status
        system_status = await daemon_client.get_system_status()
        console.print(f"📁 Daemon status: {system_status.status}", style="cyan")

        # Configure web parser
        console.print("🔧 Configuring secure web parser...", style="blue")

        # Create security config
        security_config = SecurityConfig()

        if allowed_domains:
            security_config.domain_allowlist = set(allowed_domains)
            console.print(
                f"🔐 Domain allowlist: {', '.join(allowed_domains)}", style="yellow"
            )
        elif not allow_all_domains:
            # Default to same domain as start URL
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            if parsed_url.netloc:
                security_config.domain_allowlist = {parsed_url.netloc}
                console.print(
                    f"🔐 Restricting to same domain: {parsed_url.netloc}",
                    style="yellow",
                )

        security_config.request_delay = request_delay
        security_config.enable_content_scanning = not disable_security
        security_config.quarantine_suspicious = not disable_security
        security_config.max_total_pages = max_pages
        security_config.max_depth = max_depth

        # Initialize web interface
        web_interface = WebIngestionInterface(security_config)

        console.print("🌐 Starting web content ingestion...", style="blue")

        # Parse web content
        if max_pages > 1 or max_depth > 0:
            console.print(
                f"📄 Crawling up to {max_pages} pages (depth: {max_depth})...",
                style="cyan",
            )
            parsed_doc = await web_interface.ingest_site(
                url, max_pages=max_pages, max_depth=max_depth
            )
        else:
            console.print(f"📄 Fetching single page: {url}...", style="cyan")
            parsed_doc = await web_interface.ingest_url(url)

        # Show content stats
        content_length = len(parsed_doc.content)
        console.print(
            f"✅ Content retrieved: {content_length:,} characters", style="green"
        )

        # Security warnings
        if "security_warnings" in parsed_doc.additional_metadata:
            warnings = parsed_doc.additional_metadata["security_warnings"]
            if warnings:
                console.print(
                    f"⚠️  Security warnings found: {len(warnings)}", style="yellow"
                )
                for warning in warnings[:3]:  # Show first 3 warnings
                    console.print(f"  • {warning}", style="yellow")
                if len(warnings) > 3:
                    console.print(
                        f"  ... and {len(warnings) - 3} more warnings", style="yellow"
                    )

        # Pages crawled info
        if "pages_crawled" in parsed_doc.additional_metadata:
            pages_crawled = parsed_doc.additional_metadata["pages_crawled"]
            console.print(
                f"📊 Pages successfully crawled: {pages_crawled}", style="cyan"
            )

        if dry_run:
            console.print(
                "\n🔍 DRY RUN - Content preview (first 500 chars):", style="yellow bold"
            )
            preview = parsed_doc.content[:500]
            if len(parsed_doc.content) > 500:
                preview += "..."
            console.print(f"'{preview}'", style="dim")
            console.print("\n✅ Dry run completed successfully", style="green")
            return

        # Confirmation
        if not auto_confirm:
            console.print(
                f"\n🤔 Ready to ingest {content_length:,} characters into '{collection}' collection"
            )
            if not typer.confirm("Proceed with ingestion?"):
                console.print("❌ Operation cancelled", style="red")
                return

        # Add to collection
        console.print(
            f"💾 Adding content to collection '{collection}'...", style="blue"
        )

        # Process document via daemon
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(parsed_doc.content)
            temp_file_path = temp_file.name
        
        try:
            response = await daemon_client.process_document(
                file_path=temp_file_path,
                collection=collection,
                metadata=parsed_doc.additional_metadata or {},
                chunk_text=True
            )
            
            result = {
                'success': response.success,
                'document_id': response.document_id,
                'chunks_created': response.chunks_created,
                'error': response.error_message if not response.success else None
            }
        finally:
            Path(temp_file_path).unlink(missing_ok=True)

        # Display results
        if result and result.get("success", False):
            console.print(f"✅ Successfully ingested web content", style="green bold")
            console.print(
                f"📄 Document ID: {result.get('document_id', 'unknown')}", style="cyan"
            )
            if "chunks_created" in result:
                console.print(
                    f"🔗 Text chunks created: {result['chunks_created']}", style="cyan"
                )
        else:
            console.print(
                f"❌ Ingestion failed: {result.get('error', 'Unknown error')}",
                style="red",
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Web ingestion failed: {e}", style="red")
        logger.error(f"Web ingestion error: {e}", exc_info=True)
        raise typer.Exit(1)

    finally:
        if "daemon_client" in locals():
            await daemon_client.disconnect()


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
