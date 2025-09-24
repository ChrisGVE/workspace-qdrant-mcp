"""
Workflow manager for complete metadata processing pipeline.

This module provides the main orchestration for the metadata workflow system,
integrating all components (aggregation, YAML generation, batch processing,
and incremental tracking) into a unified, easy-to-use interface.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

from loguru import logger

from .aggregator import MetadataAggregator, DocumentMetadata
from .batch_processor import BatchProcessor, BatchConfig, BatchResult
from .incremental_tracker import IncrementalTracker, DocumentChangeInfo
from .yaml_generator import YAMLGenerator, YAMLConfig
from .exceptions import WorkflowConfigurationError, MetadataError


class WorkflowConfig:
    """Configuration for the complete metadata workflow."""

    def __init__(
        self,
        # Output configuration
        output_directory: Optional[Union[str, Path]] = None,
        generate_individual_yamls: bool = True,
        generate_collection_yaml: bool = True,
        collection_name: str = "document_collection",

        # Processing configuration
        project_name: Optional[str] = None,
        collection_type: str = "documents",
        incremental_updates: bool = True,

        # Component configurations
        batch_config: Optional[BatchConfig] = None,
        yaml_config: Optional[YAMLConfig] = None,

        # Tracking configuration
        tracking_storage_path: Optional[Union[str, Path]] = None,
        cleanup_old_tracking: bool = True,

        # Progress and logging
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verbose_logging: bool = False,
    ):
        """
        Initialize workflow configuration.

        Args:
            output_directory: Directory for output files
            generate_individual_yamls: Generate individual YAML files per document
            generate_collection_yaml: Generate collection YAML file
            collection_name: Name for collection YAML
            project_name: Project name for metadata
            collection_type: Collection type for metadata
            incremental_updates: Enable incremental update tracking
            batch_config: Batch processing configuration
            yaml_config: YAML generation configuration
            tracking_storage_path: Path for incremental tracking database
            cleanup_old_tracking: Clean up old tracking data
            progress_callback: Progress update callback
            verbose_logging: Enable verbose logging
        """
        self.output_directory = Path(output_directory) if output_directory else Path.cwd() / "metadata_output"
        self.generate_individual_yamls = generate_individual_yamls
        self.generate_collection_yaml = generate_collection_yaml
        self.collection_name = collection_name
        self.project_name = project_name
        self.collection_type = collection_type
        self.incremental_updates = incremental_updates
        self.batch_config = batch_config or BatchConfig()
        self.yaml_config = yaml_config or YAMLConfig()
        self.tracking_storage_path = tracking_storage_path
        self.cleanup_old_tracking = cleanup_old_tracking
        self.progress_callback = progress_callback
        self.verbose_logging = verbose_logging

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate workflow configuration."""
        if not self.generate_individual_yamls and not self.generate_collection_yaml:
            raise WorkflowConfigurationError(
                "At least one of generate_individual_yamls or generate_collection_yaml must be True"
            )

        if not self.collection_name:
            raise WorkflowConfigurationError("Collection name cannot be empty")


class WorkflowResult:
    """Results from complete workflow execution."""

    def __init__(
        self,
        batch_result: BatchResult,
        change_info: Optional[List[DocumentChangeInfo]] = None,
        yaml_files: Optional[List[str]] = None,
        collection_yaml_path: Optional[str] = None,
        workflow_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize workflow result.

        Args:
            batch_result: Results from batch processing
            change_info: Document change information (if incremental)
            yaml_files: List of generated individual YAML file paths
            collection_yaml_path: Path to collection YAML file
            workflow_stats: Workflow execution statistics
        """
        self.batch_result = batch_result
        self.change_info = change_info or []
        self.yaml_files = yaml_files or []
        self.collection_yaml_path = collection_yaml_path
        self.workflow_stats = workflow_stats or {}

    @property
    def success_count(self) -> int:
        """Number of successfully processed documents."""
        return self.batch_result.success_count

    @property
    def failure_count(self) -> int:
        """Number of failed documents."""
        return self.batch_result.failure_count

    @property
    def total_count(self) -> int:
        """Total number of documents processed."""
        return self.batch_result.total_count

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return self.batch_result.success_rate

    @property
    def changed_documents_count(self) -> int:
        """Number of documents that changed (if incremental tracking enabled)."""
        return len([c for c in self.change_info if c.has_changed])


class WorkflowManager:
    """
    Complete metadata workflow orchestration system.

    This class provides a unified interface for the entire metadata workflow,
    coordinating document parsing, metadata aggregation, incremental tracking,
    and YAML generation with comprehensive error handling and progress reporting.
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize workflow manager.

        Args:
            config: Optional workflow configuration
        """
        self.config = config or WorkflowConfig()

        # Initialize components
        self.metadata_aggregator = MetadataAggregator()
        self.batch_processor = BatchProcessor(
            metadata_aggregator=self.metadata_aggregator,
            config=self.config.batch_config,
        )
        self.yaml_generator = YAMLGenerator(config=self.config.yaml_config)

        # Initialize incremental tracker if enabled
        self.incremental_tracker: Optional[IncrementalTracker] = None
        if self.config.incremental_updates:
            self.incremental_tracker = IncrementalTracker(
                storage_path=self.config.tracking_storage_path,
                project_name=self.config.project_name,
            )

        # Configure logging
        if self.config.verbose_logging:
            logger.add(
                self.config.output_directory / "workflow.log",
                rotation="10 MB",
                retention="30 days",
                level="DEBUG",
            )

    async def process_documents(
        self,
        file_paths: List[Union[str, Path]],
    ) -> WorkflowResult:
        """
        Process documents through complete metadata workflow.

        Args:
            file_paths: List of document file paths to process

        Returns:
            WorkflowResult with complete processing results

        Raises:
            MetadataError: If workflow processing fails
        """
        try:
            start_time = asyncio.get_event_loop().time()
            logger.info(f"Starting metadata workflow for {len(file_paths)} documents")

            # Ensure output directory exists
            self.config.output_directory.mkdir(parents=True, exist_ok=True)

            # Step 1: Process documents through batch processor
            logger.info("Step 1: Processing documents with batch processor")
            batch_result = await self.batch_processor.process_documents(
                file_paths=file_paths,
                project_name=self.config.project_name,
                collection_type=self.config.collection_type,
                progress_callback=self.config.progress_callback,
            )

            logger.info(f"Batch processing completed: {batch_result.success_count} successful, "
                       f"{batch_result.failure_count} failed")

            # Step 2: Incremental change detection (if enabled)
            change_info = []
            documents_to_generate = batch_result.successful_documents

            if self.incremental_tracker:
                logger.info("Step 2: Detecting incremental changes")
                change_info = self.incremental_tracker.detect_changes(batch_result.successful_documents)

                # Filter to only changed documents if incremental updates enabled
                if self.config.incremental_updates:
                    documents_to_generate = self.incremental_tracker.get_changed_documents(
                        batch_result.successful_documents
                    )
                    logger.info(f"Incremental filtering: {len(documents_to_generate)} changed documents")

                # Update tracking data
                self.incremental_tracker.update_tracking_data(batch_result.successful_documents)

                # Cleanup old tracking data if configured
                if self.config.cleanup_old_tracking:
                    cleanup_count = self.incremental_tracker.cleanup_deleted_documents()
                    if cleanup_count > 0:
                        logger.info(f"Cleaned up {cleanup_count} old tracking records")

            # Step 3: Generate YAML files
            logger.info("Step 3: Generating YAML files")
            yaml_files = []
            collection_yaml_path = None

            # Generate individual YAML files
            if self.config.generate_individual_yamls and documents_to_generate:
                individual_output_dir = self.config.output_directory / "individual"
                yaml_files = self.yaml_generator.generate_batch_yaml_files(
                    document_metadata_list=documents_to_generate,
                    output_directory=individual_output_dir,
                )
                logger.info(f"Generated {len(yaml_files)} individual YAML files")

            # Generate collection YAML file
            if self.config.generate_collection_yaml and batch_result.successful_documents:
                collection_yaml_path = str(
                    self.config.output_directory / f"{self.config.collection_name}.yaml"
                )
                self.yaml_generator.generate_collection_yaml(
                    document_metadata_list=batch_result.successful_documents,
                    output_path=collection_yaml_path,
                    collection_name=self.config.collection_name,
                )
                logger.info(f"Generated collection YAML: {collection_yaml_path}")

            # Step 4: Generate workflow statistics
            end_time = asyncio.get_event_loop().time()
            workflow_stats = self._generate_workflow_stats(
                batch_result=batch_result,
                change_info=change_info,
                yaml_files=yaml_files,
                processing_time=end_time - start_time,
            )

            # Generate summary report
            await self._generate_summary_report(
                batch_result=batch_result,
                change_info=change_info,
                workflow_stats=workflow_stats,
            )

            logger.info(f"Metadata workflow completed successfully in {workflow_stats['processing_time']:.2f}s")

            return WorkflowResult(
                batch_result=batch_result,
                change_info=change_info,
                yaml_files=yaml_files,
                collection_yaml_path=collection_yaml_path,
                workflow_stats=workflow_stats,
            )

        except Exception as e:
            logger.error(f"Metadata workflow failed: {e}")
            raise MetadataError(
                f"Metadata workflow processing failed: {str(e)}",
                details={"original_error": str(e)},
            ) from e

    async def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
    ) -> WorkflowResult:
        """
        Process all documents in a directory.

        Args:
            directory_path: Directory containing documents to process
            recursive: Search subdirectories recursively
            file_patterns: Optional list of file patterns to match

        Returns:
            WorkflowResult with complete processing results

        Raises:
            MetadataError: If directory processing fails
        """
        try:
            directory = Path(directory_path)

            if not directory.exists():
                raise MetadataError(f"Directory does not exist: {directory}")

            if not directory.is_dir():
                raise MetadataError(f"Path is not a directory: {directory}")

            # Find all files in directory
            file_paths = self._find_files_in_directory(
                directory=directory,
                recursive=recursive,
                file_patterns=file_patterns,
            )

            logger.info(f"Found {len(file_paths)} files in directory: {directory}")

            # Process found files
            return await self.process_documents(file_paths)

        except MetadataError:
            raise
        except Exception as e:
            raise MetadataError(
                f"Directory processing failed: {directory_path}",
                details={"original_error": str(e)},
            ) from e

    def _find_files_in_directory(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: Optional[List[str]],
    ) -> List[Path]:
        """Find files in directory matching patterns."""
        file_paths = []

        # Default patterns for document types
        if not file_patterns:
            file_patterns = [
                "*.pdf", "*.epub", "*.mobi", "*.txt", "*.md", "*.html", "*.htm",
                "*.docx", "*.pptx", "*.py", "*.js", "*.ts", "*.java", "*.cpp",
                "*.c", "*.h", "*.hpp", "*.rs", "*.go", "*.rb", "*.php",
            ]

        # Search for files
        for pattern in file_patterns:
            if recursive:
                matching_files = directory.rglob(pattern)
            else:
                matching_files = directory.glob(pattern)

            for file_path in matching_files:
                if file_path.is_file():
                    file_paths.append(file_path)

        # Remove duplicates and sort
        file_paths = sorted(set(file_paths))

        return file_paths

    def _generate_workflow_stats(
        self,
        batch_result: BatchResult,
        change_info: List[DocumentChangeInfo],
        yaml_files: List[str],
        processing_time: float,
    ) -> Dict[str, Any]:
        """Generate workflow execution statistics."""
        # Count change types
        change_counts = {
            "added": len([c for c in change_info if c.is_new]),
            "modified": len([c for c in change_info if c.is_modified]),
            "deleted": len([c for c in change_info if c.is_deleted]),
            "unchanged": len([c for c in change_info if not c.has_changed]),
        }

        # Calculate parser type distribution
        parser_types = {}
        for doc_metadata in batch_result.successful_documents:
            parser_type = doc_metadata.parsed_document.file_type
            parser_types[parser_type] = parser_types.get(parser_type, 0) + 1

        return {
            # Processing statistics
            "processing_time": processing_time,
            "documents_per_second": batch_result.total_count / processing_time if processing_time > 0 else 0,

            # Document statistics
            "total_documents": batch_result.total_count,
            "successful_documents": batch_result.success_count,
            "failed_documents": batch_result.failure_count,
            "success_rate": batch_result.success_rate,

            # Change tracking statistics
            "incremental_enabled": bool(self.incremental_tracker),
            "change_counts": change_counts,

            # Output statistics
            "individual_yaml_files": len(yaml_files),
            "collection_yaml_generated": bool(self.config.generate_collection_yaml),

            # Parser distribution
            "parser_type_distribution": parser_types,

            # Configuration
            "batch_size": self.config.batch_config.batch_size,
            "max_workers": self.config.batch_config.max_workers,
            "parallel_processing": self.config.batch_config.parallel_processing,
        }

    async def _generate_summary_report(
        self,
        batch_result: BatchResult,
        change_info: List[DocumentChangeInfo],
        workflow_stats: Dict[str, Any],
    ) -> None:
        """Generate and save workflow summary report."""
        try:
            report_path = self.config.output_directory / "workflow_summary.yaml"

            summary_data = {
                "workflow_summary": {
                    "generated_at": asyncio.get_event_loop().time(),
                    "project_name": self.config.project_name,
                    "collection_type": self.config.collection_type,
                    "statistics": workflow_stats,
                    "processing_results": {
                        "total_documents": batch_result.total_count,
                        "successful": batch_result.success_count,
                        "failed": batch_result.failure_count,
                        "success_rate_percent": batch_result.success_rate,
                    },
                }
            }

            # Add change information if incremental tracking enabled
            if change_info:
                summary_data["workflow_summary"]["change_tracking"] = {
                    "total_changes": len([c for c in change_info if c.has_changed]),
                    "change_breakdown": workflow_stats["change_counts"],
                }

            # Add failed documents summary (limited)
            if batch_result.failed_documents:
                summary_data["workflow_summary"]["failed_documents"] = [
                    {"file_path": path, "error": error[:200] + "..." if len(error) > 200 else error}
                    for path, error in batch_result.failed_documents[:10]  # Limit to first 10
                ]

            # Generate summary YAML
            summary_yaml = self.yaml_generator.generate_yaml(
                # Create a dummy DocumentMetadata for the summary
                from .aggregator import DocumentMetadata, ParsedDocument

                summary_doc = ParsedDocument.create(
                    content="",  # No content for summary
                    file_path=str(report_path),
                    file_type="workflow_summary",
                    additional_metadata=summary_data,
                )

                summary_metadata = DocumentMetadata(
                    file_path=str(report_path),
                    content_hash="summary",
                    parsed_document=summary_doc,
                )

                await summary_metadata,
                output_path=report_path,
            )

            logger.debug(f"Generated workflow summary report: {report_path}")

        except Exception as e:
            logger.warning(f"Failed to generate summary report: {e}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow manager status.

        Returns:
            Dictionary with workflow status information
        """
        status = {
            "configuration": {
                "project_name": self.config.project_name,
                "collection_type": self.config.collection_type,
                "incremental_updates": self.config.incremental_updates,
                "output_directory": str(self.config.output_directory),
            },
            "components": {
                "metadata_aggregator": {
                    "supported_parsers": self.metadata_aggregator.get_supported_parsers(),
                },
                "batch_processor": self.batch_processor.get_processing_statistics(),
                "yaml_generator": {
                    "include_content": self.yaml_generator.config.include_content,
                    "pretty_format": self.yaml_generator.config.pretty_format,
                },
            },
        }

        # Add incremental tracker status if enabled
        if self.incremental_tracker:
            try:
                status["components"]["incremental_tracker"] = self.incremental_tracker.get_change_summary()
            except Exception as e:
                status["components"]["incremental_tracker"] = {"error": str(e)}

        return status


# Export main classes
__all__ = ["WorkflowManager", "WorkflowConfig", "WorkflowResult"]