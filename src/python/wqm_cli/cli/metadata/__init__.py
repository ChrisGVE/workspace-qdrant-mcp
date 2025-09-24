"""
Metadata workflow system for workspace-qdrant-mcp.

This module provides comprehensive metadata extraction, YAML generation,
batch processing, and incremental update capabilities for document collections.

The metadata workflow system aggregates metadata from all document processors,
generates structured YAML output, and supports efficient batch processing
with incremental updates for changed documents.

Key Components:
    - MetadataAggregator: Aggregates metadata from all parser types
    - YAMLGenerator: Generates structured YAML files from document metadata
    - BatchProcessor: Processes multiple documents efficiently
    - IncrementalTracker: Tracks document changes for incremental updates
    - WorkflowManager: Orchestrates the complete metadata workflow

Example:
    ```python
    from metadata import WorkflowManager, MetadataConfig

    # Initialize workflow manager
    workflow = WorkflowManager()

    # Process documents and generate YAML
    config = MetadataConfig(
        output_format="yaml",
        include_content=True,
        batch_size=50
    )

    results = await workflow.process_documents(
        document_paths=["path/to/docs"],
        config=config
    )
    ```
"""

from .aggregator import MetadataAggregator
from .batch_processor import BatchProcessor
from .exceptions import MetadataError, YAMLGenerationError, AggregationError
from .incremental_tracker import IncrementalTracker
from .workflow_manager import WorkflowManager
from .yaml_generator import YAMLGenerator

__all__ = [
    "MetadataAggregator",
    "YAMLGenerator",
    "BatchProcessor",
    "IncrementalTracker",
    "WorkflowManager",
    "MetadataError",
    "YAMLGenerationError",
    "AggregationError",
]