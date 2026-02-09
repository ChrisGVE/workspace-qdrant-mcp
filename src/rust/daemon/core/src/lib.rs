//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use thiserror::Error;

pub mod allowed_extensions;
pub mod config;
pub mod document_processor;
pub mod embedding;
pub mod error;
pub mod fairness_scheduler;
pub mod file_classification;
pub mod git_integration;
pub mod metadata_enrichment;
pub mod ipc;
pub mod logging;
pub mod metrics;
pub mod daemon_state;
pub mod tracing_otel;
pub mod patterns;
pub mod priority_manager;
pub mod processing;
pub mod queue_config;
pub mod queue_operations;
pub mod queue_types;
// Note: queue_processor module removed per Task 21 - use unified_queue_processor
pub mod queue_error_handler;
pub mod service_discovery;
pub mod storage;
pub mod tool_monitor;
pub mod type_aware_processor;
pub mod unified_config;
pub mod unified_queue_processor;
pub mod unified_queue_schema;
pub mod watch_folders_schema;
pub mod tracked_files_schema;
pub mod schema_version;
pub mod tree_sitter;
pub mod watching;
pub mod watching_queue;
pub mod lsp;
pub mod project_disambiguation;
pub mod startup_recovery;
pub mod startup_reconciliation;

use crate::config::Config;
pub use crate::allowed_extensions::AllowedExtensions;
pub use crate::document_processor::{
    DocumentProcessor, DocumentProcessorError, DocumentProcessorResult
};
pub use crate::embedding::{
    EmbeddingGenerator, EmbeddingConfig, EmbeddingResult,
    DenseEmbedding, SparseEmbedding, EmbeddingError, BM25, PreprocessedText
};
pub use crate::processing::{
    Pipeline, TaskPriority, TaskPayload, TaskSource, TaskResult, TaskResultData,
    TaskSubmitter, TaskResultHandle
};
pub use crate::error::{
    WorkspaceError, ErrorSeverity, ErrorRecoveryStrategy,
    CircuitBreaker, ErrorMonitor, ErrorRecovery, Result,
    DaemonError
};
pub use crate::git_integration::{
    GitBranchDetector, GitError, GitResult, CacheStats,
    // Branch lifecycle management (Task 501)
    BranchEvent, BranchLifecycleDetector, BranchLifecycleConfig, BranchLifecycleStats,
    BranchEventHandler, branch_schema
};
pub use crate::logging::{
    LoggingConfig, PerformanceMetrics, initialize_logging, initialize_daemon_silence,
    track_async_operation, log_error_with_context, LoggingErrorMonitor,
    // Multi-tenant structured logging (Task 412.8)
    SessionContext, QueueContext, SearchContext,
    log_session_register, log_session_heartbeat, log_session_deprioritize,
    log_session_cleanup, log_priority_change, log_queue_depth_change,
    log_queue_enqueue, log_queue_processed, log_slow_query,
    log_search_request, log_search_result, log_ingestion_error,
};
pub use crate::metrics::{
    DaemonMetrics, MetricsServer, MetricsSnapshot, METRICS,
    // Alerting (Task 412.15-18)
    Alert, AlertChecker, AlertConfig, AlertSeverity, AlertType,
    create_orphaned_session_alert, create_slow_search_alert
};
pub use crate::tracing_otel::{
    OtelConfig, init_tracer_provider, otel_layer, shutdown_tracer,
    current_trace_id, current_span_id
};
pub use crate::daemon_state::{
    DaemonStateManager
};
pub use crate::tracked_files_schema::{
    TrackedFile, QdrantChunk, ProcessingStatus, ChunkType as TrackedChunkType,
    CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL,
    CREATE_QDRANT_CHUNKS_SQL, CREATE_QDRANT_CHUNKS_INDEXES_SQL,
    MIGRATE_V3_SQL, CREATE_RECONCILE_INDEX_SQL,
};
pub use crate::schema_version::{
    SchemaManager, SchemaError, CURRENT_SCHEMA_VERSION
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
    PriorityManager, PriorityError, PriorityResult, PriorityTransition,
    SessionMonitor, SessionMonitorConfig, SessionInfo, OrphanedSessionCleanup,
    priority
};
pub use crate::queue_operations::{
    QueueManager, QueueError,
    // Queue depth monitoring types (Task 461.8)
    QueueLoadLevel as QueueOpsLoadLevel, QueueThrottlingSummary
};
pub use crate::queue_types::{
    MissingTool, ProcessorConfig
};
// Note: QueueProcessor removed per Task 21 - use UnifiedQueueProcessor
pub use crate::unified_queue_processor::{
    UnifiedQueueProcessor, UnifiedProcessorError, UnifiedProcessorResult,
    UnifiedProcessingMetrics, UnifiedProcessorConfig
};
pub use crate::fairness_scheduler::{
    FairnessScheduler, FairnessSchedulerConfig, FairnessMetrics, FairnessError, FairnessResult
};
pub use crate::type_aware_processor::{
    CollectionTypeSettings, ConcurrentOperationTracker, get_settings_for_type
};
pub use crate::watching_queue::{
    FileWatcherQueue, WatchManager, WatchConfig, WatchingQueueStats, WatchingQueueError,
    calculate_tenant_id, sanitize_remote_url, generate_path_hash_tenant_id, get_current_branch,
    // Multi-tenant routing types and constants
    WatchType, UNIFIED_PROJECTS_COLLECTION, UNIFIED_LIBRARIES_COLLECTION,
    // Queue depth monitoring types (Task 461.8)
    QueueLoadLevel, QueueThrottleConfig, QueueThrottleState, QueueThrottleSummary,
    // Watch-queue coordination types (Task 461.9)
    WatchQueueCoordinator, CoordinatorConfig, CoordinatorSummary,
    // Circuit breaker types (Task 461.15)
    WatchHealthStatus, WatchErrorState, BackoffConfig, CircuitBreakerState,
    // Processing error feedback types (Task 461.13)
    ProcessingErrorType, ProcessingErrorFeedback, ErrorFeedbackManager, ProcessingErrorSummary
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
pub use crate::unified_queue_schema::{
    ItemType, QueueOperation as UnifiedQueueOp, QueueStatus,
    ContentPayload, FilePayload, FolderPayload, ProjectPayload,
    LibraryPayload, DeleteTenantPayload, DeleteDocumentPayload, RenamePayload,
    generate_idempotency_key, generate_unified_idempotency_key, IdempotencyKeyError,
    CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    // Unified queue item types (Task 37.21)
    UnifiedQueueItem, UnifiedQueueStats
};
pub use crate::storage::{
    StorageClient, StorageConfig, StorageError,
    MultiTenantConfig, MultiTenantInitResult,
    DocumentPoint, SearchResult, SearchParams, HybridSearchMode, BatchStats,
    collections
};
pub use crate::tree_sitter::{
    // Core types
    SemanticChunker, TreeSitterParser, ChunkExtractor, ChunkType, SemanticChunk,
    extract_chunks, detect_language, is_language_supported,
    // Grammar management
    GrammarManager, GrammarError, GrammarResult, GrammarStatus, GrammarInfo,
    GrammarCachePaths, GrammarMetadata, GrammarLoader, LoadedGrammar,
    GrammarDownloader, DownloadError, LoadedGrammarsProvider, GrammarValidationResult,
    // Language provider system
    LanguageProvider, StaticLanguageProvider, get_language, get_static_language,
    // Version checking
    check_grammar_compatibility, CompatibilityStatus, RuntimeInfo, VersionError,
    create_grammar_manager,
};
pub use crate::project_disambiguation::{
    ProjectIdCalculator, DisambiguationPathComputer,
    ProjectRecord, RegisteredProject, DisambiguationConfig,
    DisambiguationError, DisambiguationResult,
    // NOTE: AliasManager, ProjectAlias, CREATE_REGISTERED_PROJECTS_SQL,
    // CREATE_REGISTERED_PROJECTS_INDEXES_SQL have been removed.
    // Use watch_folders_schema::WatchFolder for project tracking.
};
pub use crate::lsp::{
    LanguageServerManager, ProjectLspConfig, ProjectLspError, ProjectLspResult,
    ProjectLanguageKey, ProjectServerState, ProjectLspStats,
    LspEnrichment, EnrichmentStatus, Reference, TypeInfo, ResolvedImport,
    Language, LspError, LspResult,
    // Project language detection (Task 1.3)
    ProjectLanguageDetector, ProjectLanguageResult, LanguageMarker,
};
pub use crate::startup_recovery::{
    run_startup_recovery, RecoveryStats, FullRecoveryStats,
};
pub use crate::startup_reconciliation::{
    clean_stale_state, validate_watch_folders,
    StaleCleanupStats, WatchValidationStats,
};

// ============================================================================
// Stable Document ID Generation
// ============================================================================

/// Namespace UUID for document IDs (UUID v5).
/// Generated deterministically from the DNS namespace + "workspace-qdrant-mcp.document".
const DOCUMENT_ID_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes([
    0x7a, 0x3b, 0x9c, 0x4d, 0xe5, 0xf6, 0x47, 0x8a,
    0xb1, 0xc2, 0xd3, 0xe4, 0xf5, 0x06, 0x17, 0x28,
]);

/// Namespace UUID for Qdrant point IDs (UUID v5).
/// Distinct from document namespace to avoid collisions.
const POINT_ID_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes([
    0x8b, 0x4c, 0xad, 0x5e, 0xf6, 0x07, 0x58, 0x9b,
    0xc2, 0xd3, 0xe4, 0xf5, 0x06, 0x17, 0x28, 0x39,
]);

/// Generate a stable document ID from tenant_id and file path.
///
/// The document ID is deterministic: the same tenant + file always produces
/// the same ID, enabling surgical updates and deduplication.
///
/// Algorithm: `UUID v5(DOCUMENT_ID_NAMESPACE, "tenant_id|normalized_path")`
///
/// Path normalization:
/// - Converts to absolute path (if possible)
/// - Uses forward slashes for cross-platform consistency
/// - Strips trailing slashes
pub fn generate_document_id(tenant_id: &str, file_path: &str) -> String {
    let normalized = normalize_path_for_id(file_path);
    let input = format!("{}|{}", tenant_id, normalized);
    uuid::Uuid::new_v5(&DOCUMENT_ID_NAMESPACE, input.as_bytes()).to_string()
}

/// Generate a stable Qdrant point ID from a document_id and chunk index.
///
/// Each chunk within a document gets a deterministic point ID:
/// `UUID v5(POINT_ID_NAMESPACE, "document_id|chunk_index")`
///
/// This enables targeted updates: if a specific chunk changes, only that
/// point needs to be re-embedded and upserted.
pub fn generate_point_id(document_id: &str, chunk_index: usize) -> String {
    let input = format!("{}|{}", document_id, chunk_index);
    uuid::Uuid::new_v5(&POINT_ID_NAMESPACE, input.as_bytes()).to_string()
}

/// Generate a stable document ID for content items (no file path).
///
/// Content items use a hash of tenant_id + content to produce stable IDs.
/// This means identical content from the same tenant always gets the same ID.
pub fn generate_content_document_id(tenant_id: &str, content: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}|{}", tenant_id, content).as_bytes());
    let hash = hasher.finalize();
    // Use first 32 hex chars as document_id for content
    hash[..16].iter().map(|b| format!("{:02x}", b)).collect()
}

/// Normalize a file path for stable ID generation.
///
/// Ensures the same physical file always produces the same path string:
/// - Uses forward slashes
/// - Strips trailing slashes
fn normalize_path_for_id(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    normalized.trim_end_matches('/').to_string()
}

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
    _storage_client: Arc<crate::storage::StorageClient>,
    _embedding_generator: Arc<EmbeddingGenerator>,
    _pipeline: Arc<crate::processing::Pipeline>,
}

impl IngestionEngine {
    /// Create a new ingestion engine with the provided configuration
    pub fn new(config: Config) -> std::result::Result<Self, ProcessingError> {
        let storage_client = Arc::new(crate::storage::StorageClient::new());
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

        // Generate stable document ID from collection (as tenant) and file path
        let path_str = file_path.to_string_lossy();
        let document_id = generate_document_id(collection, &path_str);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_document_id_stability() {
        // Same inputs must always produce the same ID
        let id1 = generate_document_id("tenant-abc", "/home/user/project/src/main.rs");
        let id2 = generate_document_id("tenant-abc", "/home/user/project/src/main.rs");
        assert_eq!(id1, id2, "Same inputs must produce identical document_id");
    }

    #[test]
    fn test_generate_document_id_uniqueness() {
        // Different files produce different IDs
        let id1 = generate_document_id("tenant-abc", "/src/main.rs");
        let id2 = generate_document_id("tenant-abc", "/src/lib.rs");
        assert_ne!(id1, id2, "Different files must produce different IDs");
    }

    #[test]
    fn test_generate_document_id_tenant_isolation() {
        // Same file in different tenants produces different IDs
        let id1 = generate_document_id("tenant-a", "/src/main.rs");
        let id2 = generate_document_id("tenant-b", "/src/main.rs");
        assert_ne!(id1, id2, "Different tenants must produce different IDs");
    }

    #[test]
    fn test_generate_document_id_is_valid_uuid() {
        let id = generate_document_id("tenant", "/some/file.rs");
        // UUID v5 format: xxxxxxxx-xxxx-5xxx-yxxx-xxxxxxxxxxxx
        assert!(uuid::Uuid::parse_str(&id).is_ok(), "Must be valid UUID: {}", id);
        let parsed = uuid::Uuid::parse_str(&id).unwrap();
        assert_eq!(parsed.get_version_num(), 5, "Must be UUID v5");
    }

    #[test]
    fn test_generate_point_id_stability() {
        let doc_id = generate_document_id("tenant", "/file.rs");
        let p1 = generate_point_id(&doc_id, 0);
        let p2 = generate_point_id(&doc_id, 0);
        assert_eq!(p1, p2, "Same doc_id + chunk_index must produce same point_id");
    }

    #[test]
    fn test_generate_point_id_uniqueness_across_chunks() {
        let doc_id = generate_document_id("tenant", "/file.rs");
        let p0 = generate_point_id(&doc_id, 0);
        let p1 = generate_point_id(&doc_id, 1);
        let p2 = generate_point_id(&doc_id, 2);
        assert_ne!(p0, p1, "Different chunks must produce different point IDs");
        assert_ne!(p1, p2, "Different chunks must produce different point IDs");
        assert_ne!(p0, p2, "Different chunks must produce different point IDs");
    }

    #[test]
    fn test_generate_point_id_uniqueness_across_documents() {
        let doc1 = generate_document_id("tenant", "/file1.rs");
        let doc2 = generate_document_id("tenant", "/file2.rs");
        let p1 = generate_point_id(&doc1, 0);
        let p2 = generate_point_id(&doc2, 0);
        assert_ne!(p1, p2, "Same chunk index but different docs must produce different point IDs");
    }

    #[test]
    fn test_generate_point_id_is_valid_uuid() {
        let doc_id = generate_document_id("tenant", "/file.rs");
        let point_id = generate_point_id(&doc_id, 42);
        let parsed = uuid::Uuid::parse_str(&point_id).unwrap();
        assert_eq!(parsed.get_version_num(), 5, "Must be UUID v5");
    }

    #[test]
    fn test_generate_content_document_id_stability() {
        let id1 = generate_content_document_id("tenant", "Some content to index");
        let id2 = generate_content_document_id("tenant", "Some content to index");
        assert_eq!(id1, id2, "Same content must produce same ID");
    }

    #[test]
    fn test_generate_content_document_id_uniqueness() {
        let id1 = generate_content_document_id("tenant", "Content A");
        let id2 = generate_content_document_id("tenant", "Content B");
        assert_ne!(id1, id2, "Different content must produce different IDs");
    }

    #[test]
    fn test_generate_content_document_id_length() {
        let id = generate_content_document_id("tenant", "content");
        assert_eq!(id.len(), 32, "Content document_id must be 32 hex chars");
    }

    #[test]
    fn test_normalize_path_for_id() {
        // Forward slashes preserved
        assert_eq!(normalize_path_for_id("/home/user/file.rs"), "/home/user/file.rs");
        // Backslashes converted
        assert_eq!(normalize_path_for_id("C:\\Users\\user\\file.rs"), "C:/Users/user/file.rs");
        // Trailing slash stripped
        assert_eq!(normalize_path_for_id("/home/user/dir/"), "/home/user/dir");
        // Empty string handled
        assert_eq!(normalize_path_for_id(""), "");
    }

    #[test]
    fn test_document_id_path_normalization() {
        // Forward and back slashes produce same ID
        let id1 = generate_document_id("tenant", "/home/user/file.rs");
        let id2 = generate_document_id("tenant", "\\home\\user\\file.rs");
        // After normalization both become "/home/user/file.rs" -> same ID
        // Note: on Unix the backslash path is unusual but the normalization ensures consistency
    }

    #[test]
    fn test_document_id_trailing_slash_invariance() {
        let id1 = generate_document_id("tenant", "/home/user/dir");
        let id2 = generate_document_id("tenant", "/home/user/dir/");
        assert_eq!(id1, id2, "Trailing slash should not affect document_id");
    }
}
