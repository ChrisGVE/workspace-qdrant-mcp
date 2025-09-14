"""Enhanced CLI ingestion integration layer.

This module provides an enhanced document ingestion interface that integrates
directly with the simplified 4-tool interface (qdrant_store, qdrant_find, 
qdrant_manage, qdrant_watch) for improved usability, progress tracking,
and error handling.

The enhanced layer provides:
- Direct integration with SimplifiedToolsRouter
- Real-time progress tracking and user feedback
- Enhanced error handling with recovery suggestions
- Smart file format detection and validation
- Optimized batch processing with concurrency management
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import fnmatch

from common.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.tools.simplified_interface import SimplifiedToolsRouter, get_simplified_router
from common.logging.loguru_config import get_logger

logger = get_logger(__name__)


class IngestionProgress:
    """Track and display ingestion progress with real-time updates."""
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, completed: int = None, failed: int = None, skipped: int = None):
        """Update progress counters and display if needed."""
        if completed is not None:
            self.completed += completed
        if failed is not None:
            self.failed += failed
        if skipped is not None:
            self.skipped += skipped
            
        # Update display every 0.5 seconds or on completion
        current_time = time.time()
        if (current_time - self.last_update > 0.5) or self._is_complete():
            self._display_progress()
            self.last_update = current_time
    
    def _is_complete(self) -> bool:
        """Check if all items have been processed."""
        return (self.completed + self.failed + self.skipped) >= self.total_items
    
    def _display_progress(self):
        """Display current progress with ETA."""
        processed = self.completed + self.failed + self.skipped
        if processed == 0:
            return
            
        percentage = (processed / self.total_items) * 100
        elapsed = time.time() - self.start_time
        
        if processed < self.total_items and elapsed > 1:
            eta_seconds = (elapsed / processed) * (self.total_items - processed)
            eta_str = f", ETA: {int(eta_seconds)}s"
        else:
            eta_str = ""
        
        progress_bar = self._create_progress_bar(percentage)
        status_line = (
            f"\r{self.operation_name}: {progress_bar} "
            f"{processed}/{self.total_items} ({percentage:.1f}%)"
            f"{eta_str}"
        )
        
        print(status_line, end="", flush=True)
        
        if self._is_complete():
            print()  # New line when complete
    
    def _create_progress_bar(self, percentage: float) -> str:
        """Create a visual progress bar."""
        filled = int(percentage / 5)  # 20 character width
        bar = "█" * filled + "░" * (20 - filled)
        return f"[{bar}]"
    
    def summary(self) -> Dict[str, Any]:
        """Get final progress summary."""
        elapsed = time.time() - self.start_time
        return {
            "total": self.total_items,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_seconds": elapsed,
            "success_rate": (self.completed / max(1, self.total_items)) * 100
        }


class EnhancedIngestionEngine:
    """Enhanced ingestion engine with simplified tool integration."""
    
    def __init__(self, workspace_client: QdrantWorkspaceClient):
        self.workspace_client = workspace_client
        self.router = get_simplified_router()
        if not self.router:
            raise RuntimeError("Simplified tools router not initialized")
        
    async def ingest_single_file(
        self,
        file_path: Path,
        collection: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        dry_run: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a single file with enhanced progress tracking and error handling."""
        
        # Validate file before processing
        validation_result = await self._validate_file(file_path)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "suggestions": validation_result.get("suggestions", [])
            }
        
        if dry_run:
            return await self._analyze_file(file_path, collection, chunk_size)
        
        print(f"Ingesting: {file_path.name}")
        
        try:
            # Read file content
            content = await self._read_file_content(file_path)
            
            # Prepare metadata
            enhanced_metadata = {
                "source": "cli_enhanced",
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                **(metadata or {})
            }
            
            # Use qdrant_store tool for ingestion
            result = await self.router.qdrant_store(
                information=content,
                collection=collection,
                metadata=enhanced_metadata,
                chunk_text=True,
                note_type="document"
            )
            
            if result.get("success", True) and not result.get("error"):
                return {
                    "success": True,
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_created", 1),
                    "collection": collection,
                    "file_path": str(file_path)
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown ingestion error"),
                    "suggestions": self._get_error_suggestions(result.get("error", ""))
                }
                
        except Exception as e:
            logger.error("File ingestion failed", file_path=str(file_path), error=str(e))
            return {
                "success": False,
                "error": f"Failed to process file: {str(e)}",
                "suggestions": self._get_error_suggestions(str(e))
            }
    
    async def ingest_folder(
        self,
        folder_path: Path,
        collection: str,
        formats: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        concurrency: int = 3,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Ingest all compatible files in a folder with batch processing."""
        
        # Find and validate files
        files = await self._find_files(
            folder_path, formats, recursive, exclude_patterns
        )
        
        if not files:
            return {
                "success": False,
                "error": "No compatible files found",
                "suggestions": [
                    "Check file formats are supported",
                    "Verify folder path and permissions",
                    "Try different format filters"
                ]
            }
        
        print(f"Found {len(files)} files to process")
        
        if dry_run:
            return await self._analyze_folder(files, collection, chunk_size)
        
        # Process files with progress tracking
        progress = IngestionProgress(len(files), "Ingesting files")
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_file_with_semaphore(file_path: Path) -> Dict[str, Any]:
            async with semaphore:
                result = await self.ingest_single_file(
                    file_path, collection, chunk_size, chunk_overlap
                )
                
                if result["success"]:
                    progress.update(completed=1)
                else:
                    progress.update(failed=1)
                    print(f"\nError processing {file_path.name}: {result['error']}")
                
                return result
        
        # Process files concurrently
        tasks = [process_file_with_semaphore(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                progress.update(failed=1)
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "file_path": str(files[i])
                })
                print(f"\nException processing {files[i].name}: {result}")
            else:
                processed_results.append(result)
        
        # Generate summary
        summary = progress.summary()
        successful_results = [r for r in processed_results if r.get("success")]
        
        print(f"\nFolder ingestion completed!")
        print(f"Successfully processed: {len(successful_results)}/{len(files)} files")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Processing time: {summary['elapsed_seconds']:.2f}s")
        
        if successful_results:
            total_chunks = sum(r.get("chunks_created", 0) for r in successful_results)
            print(f"Total chunks created: {total_chunks}")
        
        return {
            "success": True,
            "summary": summary,
            "results": processed_results,
            "files_processed": len(successful_results),
            "total_chunks": sum(r.get("chunks_created", 0) for r in successful_results)
        }
    
    async def get_ingestion_status(
        self, collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get ingestion status using qdrant_manage tool."""
        
        try:
            if collection:
                # Get specific collection status
                result = await self.router.qdrant_manage(
                    action="get",
                    collection=collection
                )
            else:
                # Get all collections status
                result = await self.router.qdrant_manage(action="collections")
            
            return {
                "success": True,
                "status": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get status: {str(e)}"
            }
    
    async def _validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate file accessibility and format compatibility."""
        
        if not file_path.exists():
            return {
                "valid": False,
                "error": f"File not found: {file_path}",
                "suggestions": [
                    "Check the file path is correct",
                    "Ensure the file hasn't been moved or deleted",
                    "Verify you have read permissions"
                ]
            }
        
        if not file_path.is_file():
            return {
                "valid": False,
                "error": f"Path is not a regular file: {file_path}",
                "suggestions": [
                    "Use 'wqm ingest folder' for directories",
                    "Check if path points to a symbolic link or special file"
                ]
            }
        
        # Check file size (warn if very large)
        file_size = file_path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB warning threshold
        
        if file_size > max_size:
            return {
                "valid": True,
                "warning": f"Large file detected: {file_size / (1024*1024):.1f}MB",
                "suggestions": [
                    "Large files may take significant time to process",
                    "Consider using smaller chunk sizes for better memory usage",
                    "Monitor system resources during processing"
                ]
            }
        
        # Check format compatibility
        supported_formats = [".pdf", ".txt", ".md", ".docx", ".rtf", ".epub"]
        if file_path.suffix.lower() not in supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported file format: {file_path.suffix}",
                "suggestions": [
                    f"Supported formats: {', '.join(supported_formats)}",
                    "Convert file to a supported format",
                    "Check if format support has been added"
                ]
            }
        
        return {"valid": True}
    
    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content with format-specific handling."""
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".txt" or suffix == ".md":
            return file_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            # TODO: Integrate with PDF processing
            # For now, return placeholder
            return f"PDF content processing for: {file_path.name}"
        elif suffix == ".docx":
            # TODO: Integrate with DOCX processing
            return f"DOCX content processing for: {file_path.name}"
        else:
            # Default text reading
            return file_path.read_text(encoding="utf-8", errors="ignore")
    
    async def _find_files(
        self,
        folder_path: Path,
        formats: Optional[List[str]],
        recursive: bool,
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Find files to process with format and exclusion filtering."""
        
        if not formats:
            formats = ["pdf", "txt", "md", "docx", "rtf", "epub"]
        else:
            formats = [fmt.strip().lower().lstrip(".") for fmt in formats]
        
        files = []
        for fmt in formats:
            pattern = f"**/*.{fmt}" if recursive else f"*.{fmt}"
            files.extend(folder_path.glob(pattern))
        
        # Apply exclusion patterns
        if exclude_patterns:
            filtered_files = []
            for file_path in files:
                exclude_file = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(str(file_path), pattern):
                        exclude_file = True
                        break
                if not exclude_file:
                    filtered_files.append(file_path)
            files = filtered_files
        
        return sorted(files)
    
    async def _analyze_file(
        self, file_path: Path, collection: str, chunk_size: int
    ) -> Dict[str, Any]:
        """Analyze file for dry run mode."""
        
        file_stats = file_path.stat()
        file_size_mb = file_stats.st_size / (1024 * 1024)
        estimated_chunks = max(1, int(file_stats.st_size / chunk_size))
        
        return {
            "success": True,
            "analysis": {
                "path": str(file_path),
                "size_mb": round(file_size_mb, 2),
                "extension": file_path.suffix.lower(),
                "estimated_chunks": estimated_chunks,
                "target_collection": collection,
                "processing_estimate": f"~{estimated_chunks} chunks, {file_size_mb/5:.1f}s processing time"
            }
        }
    
    async def _analyze_folder(
        self, files: List[Path], collection: str, chunk_size: int
    ) -> Dict[str, Any]:
        """Analyze folder for dry run mode."""
        
        format_stats = {}
        total_size = 0
        total_chunks = 0
        
        for file_path in files:
            ext = file_path.suffix.lower().lstrip(".")
            size_mb = file_path.stat().st_size / (1024 * 1024)
            chunks = max(1, int(file_path.stat().st_size / chunk_size))
            
            if ext not in format_stats:
                format_stats[ext] = {"count": 0, "size_mb": 0, "chunks": 0}
            
            format_stats[ext]["count"] += 1
            format_stats[ext]["size_mb"] += size_mb
            format_stats[ext]["chunks"] += chunks
            
            total_size += size_mb
            total_chunks += chunks
        
        return {
            "success": True,
            "analysis": {
                "total_files": len(files),
                "total_size_mb": round(total_size, 2),
                "estimated_total_chunks": total_chunks,
                "target_collection": collection,
                "format_breakdown": format_stats,
                "processing_estimate": f"~{total_chunks} chunks, {total_size/3:.1f}s processing time"
            }
        }
    
    def _get_error_suggestions(self, error_message: str) -> List[str]:
        """Generate helpful suggestions based on error message."""
        
        suggestions = []
        error_lower = error_message.lower()
        
        if "connection" in error_lower or "network" in error_lower:
            suggestions.extend([
                "Check if Qdrant server is running",
                "Verify network connectivity",
                "Check firewall settings"
            ])
        
        if "permission" in error_lower or "access" in error_lower:
            suggestions.extend([
                "Check file permissions",
                "Run with appropriate user privileges",
                "Verify folder access rights"
            ])
        
        if "memory" in error_lower or "out of" in error_lower:
            suggestions.extend([
                "Try smaller chunk sizes",
                "Process files individually instead of batch",
                "Check available system memory"
            ])
        
        if "format" in error_lower or "parse" in error_lower:
            suggestions.extend([
                "Verify file is not corrupted",
                "Check file format is supported",
                "Try converting to a different format"
            ])
        
        if not suggestions:
            suggestions = [
                "Check log files for detailed error information",
                "Try processing with debug mode enabled",
                "Verify system requirements are met"
            ]
        
        return suggestions