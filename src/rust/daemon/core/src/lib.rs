//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use thiserror::Error;

pub mod config;
pub mod deletion_strategy;
pub mod embedding;
pub mod error;
pub mod file_classification;
pub mod git_integration;
pub mod metadata_enrichment;
pub mod ipc;
pub mod logging;
pub mod daemon_state;
pub mod patterns;
pub mod priority_manager;
pub mod processing;
pub mod queue_config;
pub mod queue_operations;
pub mod queue_types;
// TODO: queue_processor requires DocumentProcessor implementation
// pub mod queue_processor;
pub mod queue_error_handler;
pub mod service_discovery;
pub mod storage;
pub mod tool_monitor;
pub mod type_aware_processor;
pub mod unified_config;
pub mod watching;
pub mod watching_queue;

use crate::storage::StorageClient;
use crate::config::Config;
pub use crate::deletion_strategy::{
    DeletionMode, DeletionCollectionType, DeletionStrategy, DeletionStrategyFactory,
    DynamicDeletionStrategy, CumulativeDeletionStrategy, BatchCleanupManager,
    DeletionError, DeletionResult, CleanupStats
};
pub use crate::embedding::{
    EmbeddingGenerator, EmbeddingConfig, EmbeddingResult,
    DenseEmbedding, SparseEmbedding, EmbeddingError
};
pub use crate::processing::{
    Pipeline, TaskPriority, TaskPayload, TaskSource, TaskResult, TaskResultData,
    TaskSubmitter, TaskResultHandle
};
pub use crate::error::{
    WorkspaceError, ErrorSeverity, ErrorRecoveryStrategy,
    CircuitBreaker, ErrorMonitor, ErrorRecovery, Result
};
pub use crate::git_integration::{
    GitBranchDetector, GitError, GitResult, CacheStats
};
pub use crate::logging::{
    LoggingConfig, PerformanceMetrics, initialize_logging, initialize_daemon_silence,
    track_async_operation, log_error_with_context, LoggingErrorMonitor
};
pub use crate::daemon_state::{
    DaemonStateManager
};
pub use crate::service_discovery::{
    DiscoveryManager, ServiceRegistry, ServiceInfo, ServiceStatus,
    NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType,
    HealthChecker, HealthStatus, HealthConfig,
    DiscoveryConfig, DiscoveryError, DiscoveryResult
};
pub use crate::patterns::{
    PatternManager, PatternError, PatternResult, AllPatterns,
    ProjectIndicators, ExcludePatterns, IncludePatterns, LanguageExtensions,
    Ecosystem, ProjectIndicator, ConfidenceLevel, LanguageGroup, CommentSyntax
};
pub use crate::priority_manager::{
    PriorityManager, PriorityError, PriorityResult, PriorityTransition
};
pub use crate::queue_operations::{
    QueueManager, QueueOperation, QueueItem, QueueError
};
pub use crate::queue_types::{
    MissingTool, ProcessorConfig
};
pub use crate::type_aware_processor::{
    CollectionTypeSettings, ConcurrentOperationTracker, get_settings_for_type
};
pub use crate::watching_queue::{
    FileWatcherQueue, WatchManager, WatchConfig, WatchingQueueStats, WatchingQueueError,
    calculate_tenant_id, sanitize_remote_url, generate_path_hash_tenant_id, get_current_branch
};
pub use crate::tool_monitor::{
    ToolMonitor, MonitoringError, MonitoringResult, RequeueStats
};
pub use crate::file_classification::{
    classify_file_type, is_test_directory, FileType
};
pub use crate::metadata_enrichment::{
    enrich_metadata, CollectionType
};

/// Core processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

/// Document processing result
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub document_id: String,
    pub collection: String,
    pub chunks_created: Option<usize>,
    pub processing_time_ms: u64,
}

/// Processing statistics for monitoring
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_documents_processed: u64,
    pub total_chunks_created: u64,
    pub average_processing_time_ms: u64,
    pub chunking_config: ChunkingConfig,
}

/// Document type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentType {
    Pdf,
    Epub,
    Docx,
    Text,
    Markdown,
    Code(String), // Language name
    Unknown,
}

/// Text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub content: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub metadata: HashMap<String, String>,
}

/// Document content representation
#[derive(Debug, Clone)]
pub struct DocumentContent {
    pub raw_text: String,
    pub metadata: HashMap<String, String>,
    pub document_type: DocumentType,
    pub chunks: Vec<TextChunk>,
}

/// Chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub preserve_paragraphs: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap_size: 50,
            preserve_paragraphs: true,
        }
    }
}

/// Main ingestion engine
pub struct IngestionEngine {
    _config: Config,
    _storage_client: Arc<StorageClient>,
    _embedding_generator: Arc<EmbeddingGenerator>,
    _pipeline: Arc<crate::processing::Pipeline>,
}

impl IngestionEngine {
    /// Create a new ingestion engine with the provided configuration
    pub fn new(config: Config) -> std::result::Result<Self, ProcessingError> {
        let storage_client = Arc::new(StorageClient::new());
        let embedding_config = EmbeddingConfig::default();
        let embedding_generator = Arc::new(
            EmbeddingGenerator::new(embedding_config)
                .map_err(|e| ProcessingError::Processing(format!("Failed to initialize embedding generator: {}", e)))?
        );

        // Pipeline uses internal task management with max concurrent tasks
        // Default to 8 concurrent tasks for balanced throughput
        let pipeline = Arc::new(crate::processing::Pipeline::new(8));

        Ok(Self {
            _config: config,
            _storage_client: storage_client,
            _embedding_generator: embedding_generator,
            _pipeline: pipeline,
        })
    }

    /// Process a single document
    pub async fn process_document(
        &self,
        file_path: &Path,
        collection: &str,
    ) -> std::result::Result<DocumentResult, ProcessingError> {
        let start = Instant::now();

        // Extract text content
        let content = self.extract_content(file_path).await?;

        // Generate chunks
        let chunks = self.chunk_content(&content)?;

        // TODO: Implement actual pipeline integration
        // For now, generate a placeholder document_id
        let document_id = format!("doc_{}", uuid::Uuid::new_v4());

        let processing_time = start.elapsed();

        Ok(DocumentResult {
            document_id,
            collection: collection.to_string(),
            chunks_created: Some(chunks.len()),
            processing_time_ms: processing_time.as_millis() as u64,
        })
    }

    /// Extract text content from a file
    async fn extract_content(&self, file_path: &Path) -> std::result::Result<DocumentContent, ProcessingError> {
        // Placeholder implementation
        // TODO: Implement actual document parsing based on file type
        let raw_text = tokio::fs::read_to_string(file_path).await?;

        Ok(DocumentContent {
            raw_text,
            metadata: HashMap::new(),
            document_type: DocumentType::Text,
            chunks: Vec::new(),
        })
    }

    /// Chunk document content
    fn chunk_content(&self, content: &DocumentContent) -> std::result::Result<Vec<TextChunk>, ProcessingError> {
        let config = ChunkingConfig::default();
        let mut chunks = Vec::new();

        let text = &content.raw_text;
        let total_chars = text.len();

        let mut start = 0;
        let mut chunk_index = 0;

        while start < total_chars {
            let end = (start + config.chunk_size).min(total_chars);
            let chunk_text = &text[start..end];

            chunks.push(TextChunk {
                content: chunk_text.to_string(),
                chunk_index,
                start_char: start,
                end_char: end,
                metadata: HashMap::new(),
            });

            chunk_index += 1;
            start = end - config.overlap_size;
        }

        Ok(chunks)
    }
}
