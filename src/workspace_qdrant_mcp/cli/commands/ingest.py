"""Document ingestion CLI commands.

This module provides manual document processing capabilities for the
unified wqm CLI, handling various file formats and ingestion workflows.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer

from ...cli.enhanced_ingestion import EnhancedIngestionEngine
from ...cli.ingestion_engine import DocumentIngestionEngine, IngestionResult
from ...core.client import QdrantWorkspaceClient
from ...core.daemon_client import get_daemon_client, with_daemon_client
from ...core.yaml_config import load_config
from ...core.yaml_metadata import YamlMetadataWorkflow
from ...observability import get_logger
from ..utils import (
    create_command_app,
    dry_run_option,
    error_message,
    force_option,
    handle_async,
    success_message,
    verbose_option,
    warning_message,
)

logger = get_logger(__name__)

# Create the ingest app using shared utilities
ingest_app = create_command_app(
    name="ingest",
    help_text="Manual document processing and ingestion",
    no_args_is_help=True,
)


@ingest_app.command("file")
def ingest_file(
    path: str = typer.Argument(..., help="Path to file to ingest"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection name"
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Maximum characters per text chunk"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Character overlap between chunks"
    ),
    dry_run: bool = dry_run_option(),
    force: bool = force_option(),
):
    """Ingest a single file into collection."""
    handle_async(
        _ingest_file(path, collection, chunk_size, chunk_overlap, dry_run, force)
    )


@ingest_app.command("folder")
def ingest_folder(
    path: str = typer.Argument(..., help="Path to folder to ingest"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection name"
    ),
    formats: Optional[List[str]] = typer.Option(
        None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Maximum characters per text chunk"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Character overlap between chunks"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Process subdirectories recursively"
    ),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", help="Glob patterns to exclude"
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", help="Number of concurrent processing tasks"
    ),
    dry_run: bool = dry_run_option(),
    force: bool = force_option(),
):
    """Ingest all files in a folder."""
    handle_async(
        _ingest_folder(
            path,
            collection,
            formats,
            chunk_size,
            chunk_overlap,
            recursive,
            exclude,
            concurrency,
            dry_run,
            force,
        )
    )


@ingest_app.command("yaml")
def ingest_yaml_metadata(
    path: str = typer.Argument(..., help="Path to YAML metadata file"),
    dry_run: bool = dry_run_option(),
    force: bool = force_option(),
):
    """Process completed YAML metadata file."""
    handle_async(_ingest_yaml_metadata(path, dry_run, force))


@ingest_app.command("generate-yaml")
def generate_yaml_metadata(
    library_path: str = typer.Argument(..., help="Path to library folder"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target library collection name"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output YAML file path"
    ),
    formats: Optional[List[str]] = typer.Option(
        None, "--format", "-f", help="File formats to process (e.g. pdf,md,txt)"
    ),
    force: bool = force_option(),
):
    """Generate YAML metadata file for library documents."""
    handle_async(
        _generate_yaml_metadata(library_path, collection, output, formats, force)
    )


@ingest_app.command("web")
def ingest_web_pages(
    url: str = typer.Argument(..., help="Root URL to crawl"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection name"
    ),
    max_depth: int = typer.Option(2, "--depth", help="Maximum crawl depth"),
    max_pages: int = typer.Option(
        50, "--max-pages", help="Maximum number of pages to crawl"
    ),
    include_patterns: Optional[List[str]] = typer.Option(
        None, "--include", help="URL patterns to include"
    ),
    exclude_patterns: Optional[List[str]] = typer.Option(
        None, "--exclude", help="URL patterns to exclude"
    ),
    delay: float = typer.Option(
        1.0, "--delay", help="Delay between requests (seconds)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Analyze URLs without crawling"
    ),
):
    """Crawl and ingest web pages."""
    handle_async(
        _ingest_web_pages(
            url,
            collection,
            max_depth,
            max_pages,
            include_patterns,
            exclude_patterns,
            delay,
            dry_run,
        )
    )


@ingest_app.command("status")
def ingestion_status(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Filter by collection"
    ),
    recent: bool = typer.Option(False, "--recent", help="Show only recent ingestions"),
):
    """Show ingestion status and statistics."""
    handle_async(_ingestion_status(collection, recent))


# Enhanced ingestion helper
async def _get_enhanced_engine() -> EnhancedIngestionEngine:
    """Get enhanced ingestion engine with workspace client."""
    try:
        from ...core.config import Config
        from ...core.client import create_qdrant_client
        
        config = Config()
        qdrant_client = create_qdrant_client(config.qdrant_client_config)
        workspace_client = QdrantWorkspaceClient(qdrant_client, config)
        
        return EnhancedIngestionEngine(workspace_client)
    except Exception as e:
        print(f"Error: Failed to initialize enhanced ingestion engine: {e}")
        raise typer.Exit(1)

# Async implementation functions
async def _ingest_file(
    path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    force: bool,
):
    """Ingest a single file using enhanced ingestion engine."""
    file_path = Path(path)
    
    try:
        # Get enhanced engine
        engine = await _get_enhanced_engine()
        
        # Use enhanced ingestion with integrated validation and progress tracking
        result = await engine.ingest_single_file(
            file_path=file_path,
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dry_run=dry_run
        )
        
        if result["success"]:
            if dry_run:
                analysis = result["analysis"]
                print("\nFile Analysis:")
                print(f"  Path: {analysis['path']}")
                print(f"  Size: {analysis['size_mb']} MB")
                print(f"  Extension: {analysis['extension']}")
                print(f"  Estimated chunks: {analysis['estimated_chunks']}")
                print(f"  Target collection: {analysis['target_collection']}")
                print(f"  Processing estimate: {analysis['processing_estimate']}")
            else:
                print(f"\nSuccess! Document ingested:")
                print(f"  Document ID: {result['document_id']}")
                print(f"  Chunks created: {result['chunks_created']}")
                print(f"  Collection: {result['collection']}")
        else:
            print(f"\nError: {result['error']}")
            if "suggestions" in result:
                print("\nSuggestions:")
                for suggestion in result["suggestions"]:
                    print(f"  • {suggestion}")
            raise typer.Exit(1)
            
    except Exception as e:
        print(f"Error: File ingestion failed: {e}")
        raise typer.Exit(1)

    async def ingest_operation(daemon_client):
        # File validation already done above

        if dry_run:
            print(f"Analyzing File: {file_path.name}")

            # Analyze file without processing
            file_info = {
                "path": str(file_path),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "extension": file_path.suffix.lower(),
                "supported": file_path.suffix.lower()
                in [".pdf", ".txt", ".md", ".docx"],
            }

            # Display file analysis in plain text
            print("File Analysis:")
            print(f"Path: {file_info['path']}")
            print(f"Size: {file_info['size_mb']} MB")
            print(f"Extension: {file_info['extension']}")
            print(f"Supported: {'Yes' if file_info['supported'] else 'No'}")

            if file_info["supported"]:
                print("File can be processed with current settings")
                estimated_chunks = max(
                    1, int(file_info["size_mb"] * 1024 * 1024 / chunk_size)
                )
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
                "chunk_overlap": str(chunk_overlap),
            }

            response = await daemon_client.process_document(
                file_path=str(file_path),
                collection=collection,
                metadata=metadata,
                chunk_text=True,
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
    force: bool,
):
    """Ingest all files in a folder using enhanced ingestion engine."""
    folder_path = Path(path)
    
    try:
        # Get enhanced engine
        engine = await _get_enhanced_engine()
        
        # Use enhanced folder ingestion with progress tracking and validation
        result = await engine.ingest_folder(
            folder_path=folder_path,
            collection=collection,
            formats=formats,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=recursive,
            exclude_patterns=exclude,
            concurrency=concurrency,
            dry_run=dry_run
        )
        
        if result["success"]:
            if dry_run:
                analysis = result["analysis"]
                print("\nFolder Analysis Summary:")
                print(f"  Total files: {analysis['total_files']}")
                print(f"  Total size: {analysis['total_size_mb']} MB")
                print(f"  Estimated chunks: {analysis['estimated_total_chunks']}")
                print(f"  Target collection: {analysis['target_collection']}")
                print(f"  Processing estimate: {analysis['processing_estimate']}")
                
                print("\nFormat breakdown:")
                for ext, stats in analysis['format_breakdown'].items():
                    print(f"  .{ext}: {stats['count']} files, {stats['size_mb']:.2f} MB, ~{stats['chunks']} chunks")
            else:
                print(f"\nFolder ingestion completed successfully!")
                print(f"Files processed: {result['files_processed']}")
                print(f"Total chunks created: {result['total_chunks']}")
                summary = result['summary']
                print(f"Success rate: {summary['success_rate']:.1f}%")
                print(f"Processing time: {summary['elapsed_seconds']:.2f}s")
        else:
            print(f"\nError: {result['error']}")
            if "suggestions" in result:
                print("\nSuggestions:")
                for suggestion in result["suggestions"]:
                    print(f"  • {suggestion}")
            raise typer.Exit(1)
            
    except Exception as e:
        print(f"Error: Folder ingestion failed: {e}")
        raise typer.Exit(1)



async def _generate_yaml_metadata(
    library_path: str,
    collection: str,
    output: Optional[str],
    formats: Optional[List[str]],
    force: bool,
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
        if not collection.startswith("_"):
            print(f"Error: Library collection name must start with '_': {collection}")
            raise typer.Exit(1)

        # Check output path
        output_path = Path(output) if output else lib_path / "metadata_completion.yaml"

        if output_path.exists() and not force:
            print(
                f"Error: Output file exists (use --force to overwrite): {output_path}"
            )
            raise typer.Exit(1)

        print("Generating YAML Metadata for Library")
        print(f"Library Path: {lib_path}")
        print(f"Collection: {collection}")
        print(f"Output: {output_path}")

        # Create workflow - this still needs direct client as YAML metadata workflow
        # is not yet integrated with daemon gRPC API
        from ...core.client import create_qdrant_client
        from ...core.config import Config

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Generate YAML file
        print("Analyzing documents and extracting metadata...")

        result_path = await workflow.generate_yaml_file(
            library_path=lib_path,
            library_collection=collection,
            output_path=output_path,
            formats=formats,
        )

        if result_path:
            print("YAML metadata file generated successfully!")
            print(f"Location: {result_path}")
            print("Next steps:")
            print("  1. Review and complete the metadata in the YAML file")
            print("  2. Fill in fields marked with '?'")
            print(f"  3. Run: wqm ingest yaml {result_path}")
            print("The file contains:")
            print("  - Detected metadata from document analysis")
            print("  - Required fields for each document type")
            print("  - Processing instructions and examples")
        else:
            print("No documents found to process")

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

        print(f"Processing YAML Metadata: {yaml_path.name}")

        # Create workflow - this still needs direct client as YAML metadata workflow
        # is not yet integrated with daemon gRPC API
        from ...core.client import create_qdrant_client
        from ...core.config import Config

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        workflow = YamlMetadataWorkflow(client)

        # Process YAML file
        print("Processing documents with metadata...")

        results = await workflow.process_yaml_file(yaml_path=yaml_path, dry_run=dry_run)

        # Display results
        processed = results.get("processed", 0)
        skipped = results.get("skipped", 0)
        errors = results.get("errors", [])
        remaining = results.get("remaining", 0)

        if dry_run:
            print("YAML Metadata Analysis (Dry Run)")
            print("Processing Summary:")
            print(f"  Ready to process: {processed} documents")
            print(f"  Missing metadata: {skipped} documents")
            print(f"  Errors found: {len(errors)} documents")
            print(f"  Remaining in YAML: {remaining} documents")
            if remaining == 0:
                print("All documents ready for processing!")
            else:
                print("Complete remaining metadata and run again")
            for error in errors[:5]:
                print(f"  Error: {error}")
            if len(errors) > 5:
                print("  ... and more errors")
        else:
            print("YAML Metadata Processing Complete!")
            print("Processing Summary:")
            print(f"  Successfully processed: {processed} documents")
            print(f"  Skipped (incomplete metadata): {skipped} documents")
            print(f"  Errors encountered: {len(errors)} documents")
            print(f"  Remaining in YAML file: {remaining} documents")
            if remaining == 0:
                print("All documents processed successfully!")
            else:
                print(f"{remaining} documents still need metadata completion")
            for error in errors[:3]:
                print(f"  Error: {error}")
            if len(errors) > 3:
                print("  ... and more errors")

        # Show guidance for next steps
        if remaining > 0 and not dry_run:
            print(
                f"\nNote: The YAML file has been updated with {remaining} remaining documents."
            )
            print(
                f"   Complete their metadata and run 'wqm ingest yaml {yaml_path}' again."
            )

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
    dry_run: bool,
):
    """Crawl and ingest web pages."""
    try:
        print(f"Web Crawling: {url}")

        # TODO: Implement web crawling and ingestion
        # This will be part of future enhancement

        if dry_run:
            print("Web crawling analysis (dry run)")
            print(
                f"Would crawl {url} with max depth {max_depth} and max {max_pages} pages"
            )
            print("Web crawling feature will be implemented in a future task")
        else:
            print("Web crawling")
            print("Web crawling feature will be implemented in a future task")

    except Exception as e:
        print(f"Error: Web crawling failed: {e}")
        raise typer.Exit(1)


async def _ingestion_status(collection: Optional[str], recent: bool):
    """Show ingestion status and statistics using enhanced engine."""
    try:
        # Get enhanced engine
        engine = await _get_enhanced_engine()
        
        # Use enhanced status functionality
        result = await engine.get_ingestion_status(collection)
        
        if result["success"]:
            print("\nIngestion Status")
            status = result["status"]
            
            if collection:
                # Single collection status
                print(f"Collection: {collection}")
                if "error" not in status:
                    print("Collection information retrieved successfully")
                else:
                    print(f"Error retrieving collection info: {status.get('error')}")
            else:
                # All collections status
                if "collections" in status:
                    collections = status["collections"]
                    print(f"\nFound {len(collections)} collections:")
                    print(f"{'Collection':<30} {'Type':<10} {'Status':<10}")
                    print("-" * 55)
                    
                    for col_name in collections:
                        col_type = "Library" if col_name.startswith("_") else "Project"
                        # For now, we'll show basic status - enhanced collection info could be added
                        print(f"{col_name:<30} {col_type:<10} {'Available':<10}")
                else:
                    print("No collections found or error retrieving collections")
            
            # Show recent activity if requested
            if recent:
                print("\nRecent activity tracking will be available in future updates")
                print("Current status shows available collections for ingestion")
                
        else:
            print(f"\nError retrieving ingestion status: {result['error']}")
            raise typer.Exit(1)
            
    except Exception as e:
        print(f"Error: Status check failed: {e}")
        raise typer.Exit(1)


def _display_ingestion_result(result: IngestionResult, filename: str):
    """Display ingestion result summary."""

    if result.success:
        print(f"Successfully ingested: {filename}")
        print("Processing Summary:")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Total characters: {result.total_characters:,}")
        print(f"  Processing time: {result.processing_time_seconds:.2f}s")
        print(
            f"  Average chunk size: {result.total_characters // max(1, result.chunks_created)} chars"
        )
        print(f"Collection: {result.collection_name}")
    else:
        print(f"Error: Failed to ingest: {filename}")
        print(f"Error: {result.error_message}")
        print("Partial Results:")
        print(f"  Chunks created: {result.chunks_created}")
        print(f"  Processing time: {result.processing_time_seconds:.2f}s")
