"""Test fixtures and utilities for workspace-qdrant-mcp testing."""

from .test_data_collector import (
    CodeChunk,
    CodeSymbol,
    DataCollector,
    SearchGroundTruth,
)

from .data_generator import (
    # Core generator classes
    BaseGenerator,
    SyntheticDocumentGenerator,
    CodeFileGenerator,
    BinaryDocumentGenerator,
    ProjectStructureGenerator,
    MetadataGenerator,
    EdgeCaseGenerator,
    # Data containers
    GeneratedDocument,
    GeneratedCodeFile,
    # Convenience functions
    quick_generate,
    # Constants
    SIZE_SCALES,
)

__all__ = [
    # Original exports
    "DataCollector",
    "CodeSymbol",
    "CodeChunk",
    "SearchGroundTruth",
    # Generator classes
    "BaseGenerator",
    "SyntheticDocumentGenerator",
    "CodeFileGenerator",
    "BinaryDocumentGenerator",
    "ProjectStructureGenerator",
    "MetadataGenerator",
    "EdgeCaseGenerator",
    # Data containers
    "GeneratedDocument",
    "GeneratedCodeFile",
    # Utilities
    "quick_generate",
    "SIZE_SCALES",
]
