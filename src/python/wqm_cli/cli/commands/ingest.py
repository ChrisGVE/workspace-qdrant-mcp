"""Document ingestion CLI commands.

This module provides manual document processing capabilities for the
unified wqm CLI, handling various file formats and ingestion workflows.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer

from ..ingestion_engine import IngestionResult
from common.grpc.daemon_client import get_daemon_client, with_daemon_client
from common.core.yaml_metadata import YamlMetadataWorkflow
from loguru import logger
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

# logger imported from loguru

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


@ingest_app.command("validate")
def validate_files(
    path: str = typer.Argument(..., help="Path to file or folder to validate"),
    formats: Optional[List[str]] = typer.Option(
        None, "--format", "-f", help="File formats to validate (e.g. pdf,md,txt)"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Check subdirectories recursively"
    ),
    verbose: bool = verbose_option(),
):
    """Validate files for ingestion compatibility without processing them."""
    handle_async(_validate_files(path, formats, recursive, verbose))


@ingest_app.command("smart")
def smart_ingest(
    path: str = typer.Argument(..., help="Path to file or folder for smart ingestion"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Target collection (auto-detected if not specified)"
    ),
    auto_chunk: bool = typer.Option(
        True, "--auto-chunk/--no-auto-chunk", help="Automatically determine optimal chunk size"
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", help="Number of concurrent processing tasks for folders"
    ),
    dry_run: bool = dry_run_option(),
):
    """Smart ingestion with auto-detection and optimization."""
    handle_async(_smart_ingest(path, collection, auto_chunk, concurrency, dry_run))



# Async implementation functions
async def _ingest_file(
    path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    force: bool,
):
    """Ingest a single file using daemon client."""
    file_path = Path(path)
    
    if not file_path.exists():
        error_message(f"File not found: {path}")
        raise typer.Exit(1)
        
    if not file_path.is_file():
        error_message(f"Path is not a file: {path}")
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
            print("\nFile Analysis:")
            print(f"  Path: {file_info['path']}")
            print(f"  Size: {file_info['size_mb']} MB")
            print(f"  Extension: {file_info['extension']}")
            print(f"  Supported: {'Yes' if file_info['supported'] else 'No'}")

            if file_info["supported"]:
                print("  File can be processed with current settings")
                estimated_chunks = max(
                    1, int(file_info["size_mb"] * 1024 * 1024 / chunk_size)
                )
                print(f"  Estimated chunks: ~{estimated_chunks}")
                print(f"  Target collection: {collection}")
            else:
                error_message(f"Unsupported file format: {file_info['extension']}")
                raise typer.Exit(1)

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
                success_message(f"Document processed successfully")
                print(f"Document ID: {response.document_id}")
                print(f"Chunks added: {response.chunks_added}")
                print(f"Collection: {collection}")
                if response.applied_metadata:
                    print("Applied metadata:")
                    for key, value in response.applied_metadata.items():
                        print(f"  {key}: {value}")
            else:
                error_message(f"Processing failed: {response.message}")
                raise typer.Exit(1)

        except Exception as e:
            error_message(f"Processing failed: {e}")
            raise typer.Exit(1)

    try:
        await with_daemon_client(ingest_operation)
    except Exception as e:
        error_message(f"Ingestion failed: {e}")
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
    """Ingest all files in a folder using daemon client."""
    folder_path = Path(path)
    
    if not folder_path.exists():
        error_message(f"Folder not found: {path}")
        raise typer.Exit(1)
        
    if not folder_path.is_dir():
        error_message(f"Path is not a directory: {path}")
        raise typer.Exit(1)
    
    async def folder_operation(daemon_client):
        print(f"Processing Folder: {folder_path.name}")
        
        try:
            metadata = {
                "source": "cli",
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap),
                "concurrency": str(concurrency),
            }
            
            # Convert formats to include patterns if provided
            include_patterns = []
            if formats:
                for fmt in formats:
                    if not fmt.startswith('.'):
                        fmt = f'.{fmt}'
                    include_patterns.append(f"*{fmt}")
            
            files_processed = 0
            total_chunks = 0
            
            async for progress in daemon_client.process_folder(
                folder_path=str(folder_path),
                collection=collection,
                include_patterns=include_patterns,
                ignore_patterns=exclude or [],
                recursive=recursive,
                max_depth=10,  # Reasonable default
                dry_run=dry_run,
                metadata=metadata,
            ):
                if dry_run:
                    if hasattr(progress, 'file_path'):
                        print(f"  Would process: {progress.file_path}")
                else:
                    if hasattr(progress, 'file_path') and hasattr(progress, 'chunks_added'):
                        files_processed += 1
                        total_chunks += progress.chunks_added
                        print(f"  Processed: {progress.file_path} ({progress.chunks_added} chunks)")
                    elif hasattr(progress, 'error'):
                        warning_message(f"  Failed: {progress.file_path} - {progress.error}")
            
            if not dry_run:
                success_message(f"Folder processing completed!")
                print(f"Files processed: {files_processed}")
                print(f"Total chunks created: {total_chunks}")
                print(f"Collection: {collection}")
            else:
                print(f"\nDry run analysis completed for folder: {folder_path}")
                
        except Exception as e:
            error_message(f"Folder processing failed: {e}")
            raise typer.Exit(1)
    
    try:
        await with_daemon_client(folder_operation)
    except Exception as e:
        error_message(f"Folder ingestion failed: {e}")
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
        from common.core.client import create_qdrant_client
        from common.core.config import get_config_manager

        config = get_config_manager()
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
        from common.core.client import create_qdrant_client
        from common.core.config import get_config_manager

        config = get_config_manager()
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
    """Show ingestion status and statistics using daemon client."""
    async def status_operation(daemon_client):
        try:
            print("\nIngestion Status")
            
            if collection:
                # Single collection info
                try:
                    info = await daemon_client.get_collection_info(
                        collection_name=collection,
                        include_sample_documents=False
                    )
                    print(f"Collection: {collection}")
                    print(f"  Documents: {info.document_count}")
                    print(f"  Vectors: {info.vector_count}")
                    print(f"  Index status: {info.status}")
                    if info.description:
                        print(f"  Description: {info.description}")
                except Exception as e:
                    error_message(f"Error retrieving collection info: {e}")
                    raise typer.Exit(1)
            else:
                # All collections status
                try:
                    response = await daemon_client.list_collections(include_stats=True)
                    
                    if response.collections:
                        print(f"\nFound {len(response.collections)} collections:")
                        print(f"{'Collection':<30} {'Documents':<12} {'Vectors':<12} {'Type':<10}")
                        print("-" * 70)
                        
                        for collection_info in response.collections:
                            col_type = "Library" if collection_info.name.startswith("_") else "Project"
                            print(f"{collection_info.name:<30} {collection_info.document_count:<12} {collection_info.vector_count:<12} {col_type:<10}")
                    else:
                        print("No collections found")
                        
                    # Show system stats
                    stats = await daemon_client.get_stats(
                        include_collection_stats=True,
                        include_watch_stats=recent
                    )
                    
                    print(f"\nSystem Statistics:")
                    print(f"  Total collections: {stats.total_collections}")
                    print(f"  Total documents: {stats.total_documents}")
                    print(f"  Total vectors: {stats.total_vectors}")
                    
                except Exception as e:
                    error_message(f"Error retrieving status: {e}")
                    raise typer.Exit(1)
            
            # Show recent activity if requested
            if recent:
                try:
                    processing_status = await daemon_client.get_processing_status(
                        include_history=True,
                        history_limit=10
                    )
                    
                    if processing_status.recent_operations:
                        print(f"\nRecent Processing Activity:")
                        for op in processing_status.recent_operations[:5]:
                            print(f"  {op.timestamp}: {op.operation} - {op.status}")
                    else:
                        print(f"\nNo recent processing activity")
                        
                except Exception as e:
                    print(f"  Warning: Could not retrieve recent activity: {e}")
                
        except Exception as e:
            error_message(f"Status check failed: {e}")
            raise typer.Exit(1)
    
    try:
        await with_daemon_client(status_operation)
    except Exception as e:
        error_message(f"Status operation failed: {e}")
        raise typer.Exit(1)


async def _validate_files(
    path: str, 
    formats: Optional[List[str]], 
    recursive: bool, 
    verbose: bool
):
    """Validate files for ingestion compatibility."""
    target_path = Path(path)
    
    if not target_path.exists():
        error_message(f"Path not found: {path}")
        raise typer.Exit(1)
    
    print(f"Validating: {target_path}")
    
    if target_path.is_file():
        # Single file validation
        file_info = {
            "path": str(target_path),
            "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
            "extension": target_path.suffix.lower(),
            "supported": target_path.suffix.lower() in [".pdf", ".txt", ".md", ".docx"],
        }
        
        print(f"\nFile Validation Result:")
        print(f"  Path: {target_path}")
        print(f"  Size: {file_info['size_mb']} MB")
        print(f"  Extension: {file_info['extension']}")
        print(f"  Valid: {'Yes' if file_info['supported'] else 'No'}")
        
        if file_info['supported']:
            success_message("File is ready for ingestion")
        else:
            error_message(f"Unsupported file format: {file_info['extension']}")
            print("  Supported formats: .pdf, .txt, .md, .docx")
    else:
        # Folder validation
        supported_extensions = [".pdf", ".txt", ".md", ".docx"]
        
        # Find files matching criteria
        files = []
        for file_path in target_path.rglob("*" if recursive else "*") if recursive else target_path.iterdir():
            if file_path.is_file():
                if formats:
                    # Check if file matches format filter
                    for fmt in formats:
                        if not fmt.startswith('.'):
                            fmt = f'.{fmt}'
                        if file_path.suffix.lower() == fmt:
                            files.append(file_path)
                            break
                else:
                    # Check against supported extensions
                    if file_path.suffix.lower() in supported_extensions:
                        files.append(file_path)
        
        print(f"\nFolder Validation Result:")
        print(f"  Path: {target_path}")
        print(f"  Files found: {len(files)}")
        
        if not files:
            warning_message("No compatible files found")
            print("  Suggestions:")
            print("    • Check file formats are supported (.pdf, .txt, .md, .docx)")
            print("    • Verify folder contains documents")
            print("    • Try different format filters")
            return
        
        valid_files = 0
        invalid_files = 0
        total_size = 0
        
        print("\nFile-by-file validation:")
        for file_path in files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            supported = file_path.suffix.lower() in supported_extensions
            
            if supported:
                valid_files += 1
                status = "✓ Valid"
            else:
                invalid_files += 1
                status = "✗ Invalid"
            
            if verbose:
                print(f"    {file_path.name:<40} {status} ({size_mb:.2f} MB)")
        
        print(f"\nValidation Summary:")
        print(f"  Valid files: {valid_files}")
        print(f"  Invalid files: {invalid_files}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  Success rate: {(valid_files / len(files)) * 100:.1f}%")
        
        if valid_files > 0:
            success_message(f"{valid_files} files ready for ingestion")
        else:
            warning_message("No files can be processed")


async def _smart_ingest(
    path: str,
    collection: Optional[str],
    auto_chunk: bool,
    concurrency: int, 
    dry_run: bool
):
    """Smart ingestion with auto-detection and optimization using daemon client."""
    target_path = Path(path)
    
    if not target_path.exists():
        error_message(f"Path not found: {path}")
        raise typer.Exit(1)
    
    # Auto-detect collection if not provided
    if not collection:
        if target_path.is_file():
            # Use parent directory name for single files
            collection = target_path.parent.name
        else:
            # Use directory name for folders
            collection = target_path.name
        print(f"Auto-detected collection: {collection}")
    
    # Auto-determine chunk parameters if enabled
    if auto_chunk:
        chunk_size = 1200  # Slightly larger for better semantic coherence
        chunk_overlap = 150  # Reduced overlap for efficiency
        print(f"Auto-optimized chunking: {chunk_size} chars with {chunk_overlap} overlap")
    else:
        chunk_size = 1000
        chunk_overlap = 200
    
    print(f"Smart ingestion: {target_path}")
    
    if target_path.is_file():
        # Smart single file ingestion - reuse existing file ingestion logic
        await _ingest_file(
            path=str(target_path),
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dry_run=dry_run,
            force=False  # Smart ingestion doesn't force overwrite
        )
    else:
        # Smart folder ingestion - reuse existing folder ingestion logic
        await _ingest_folder(
            path=str(target_path),
            collection=collection,
            formats=None,  # Auto-detect all supported formats
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=True,
            exclude=[".*", "__*", "*.tmp", "*.log"],  # Smart exclusions
            concurrency=concurrency,
            dry_run=dry_run,
            force=False  # Smart ingestion doesn't force overwrite
        )


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
