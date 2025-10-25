"""Core functionality for workspace-qdrant-mcp."""

from .client import QdrantWorkspaceClient
from .collections import WorkspaceCollectionManager
from .config import ConfigManager, get_config
from .embeddings import EmbeddingService
from .hybrid_search import HybridSearchEngine, RRFFusionRanker
from .incremental_processor import (
    ChangeDetector,
    ChangeType,
    ConflictResolution,
    ConflictResolver,
    DifferentialUpdater,
    FileChangeInfo,
    IncrementalProcessor,
    ProcessingMetrics,
    TransactionManager,
    create_incremental_processor,
)
from .priority_queue_manager import (
    MCPActivityLevel,
    PriorityQueueManager,
    ProcessingContextManager,
    ProcessingMode,
    QueueHealthStatus,
    ResourceConfiguration,
)
from .sparse_vectors import (
    BM25SparseEncoder,
    create_named_sparse_vector,
    create_qdrant_sparse_vector,
)
from .sqlite_state_manager import (
    FileProcessingRecord,
    FileProcessingStatus,
    LSPServerStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)

__all__ = [
    "QdrantWorkspaceClient",
    "get_config",
    "ConfigManager",
    "WorkspaceCollectionManager",
    "EmbeddingService",
    "BM25SparseEncoder",
    "create_qdrant_sparse_vector",
    "create_named_sparse_vector",
    "HybridSearchEngine",
    "RRFFusionRanker",
    # Incremental processing components
    "IncrementalProcessor",
    "ChangeDetector",
    "DifferentialUpdater",
    "ConflictResolver",
    "TransactionManager",
    "FileChangeInfo",
    "ChangeType",
    "ConflictResolution",
    "ProcessingMetrics",
    "create_incremental_processor",
    # State management components
    "SQLiteStateManager",
    "FileProcessingRecord",
    "FileProcessingStatus",
    "ProcessingPriority",
    "LSPServerStatus",
    "WatchFolderConfig",
    # Priority queue components
    "PriorityQueueManager",
    "MCPActivityLevel",
    "ProcessingMode",
    "QueueHealthStatus",
    "ProcessingContextManager",
    "ResourceConfiguration",
]
