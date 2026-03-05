//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

// ── Module declarations ─────────────────────────────────────────────────

pub mod adaptive_resources;
pub mod allowed_extensions;
pub mod branch_switch;
pub mod clip;
pub mod component_detection;
pub mod config;
pub mod core_types;
pub mod cross_project_search;
pub mod daemon_state;
pub mod document_id;
pub mod document_processor;
pub mod embedding;
pub mod error;
pub mod fairness_scheduler;
pub mod file_classification;
pub mod format_routing;
pub mod fts_batch_processor;
pub mod git;
pub mod graph;
pub mod grouping;
pub mod idle_history;
pub mod image_extraction;
pub mod image_ingestion;
pub mod image_search;
pub mod ingestion;
pub mod ipc;
pub mod library_hierarchy;
pub mod log_pruner;
pub mod metadata_enrichment;
pub mod monitoring;
pub mod ocr;
pub mod patterns;
pub mod priority_manager;
pub mod processing;
pub mod processing_timings;
pub mod queue_config;
pub mod queue_health;
pub mod queue_operations;
pub mod queue_types;
pub mod telemetry;
pub mod tracing_otel;
// Note: queue_processor module removed per Task 21 - use unified_queue_processor
pub mod code_lines_schema;
pub mod cooccurrence_schema;
pub mod grep_search;
pub mod indexed_content_schema;
pub mod keyword_extraction;
pub mod keywords_schema;
pub mod lexicon;
pub mod library_document;
pub mod line_diff;
pub mod lsp;
pub mod metadata_uplift;
pub mod metrics_history_schema;
pub mod parent_unit;
pub mod project_disambiguation;
pub mod queue_error_handler;
pub mod resolution_events_schema;
pub mod schema_version;
pub mod search_db;
pub mod search_events_schema;
pub mod service_discovery;
pub mod source_diversity;
pub mod startup;
pub mod storage;
pub mod tagging;
pub mod text_search;
pub mod thumbnail;
pub mod title_extraction;
pub mod tokenizer;
pub mod tracked_files_schema;
pub mod tree_sitter;
pub mod type_aware_processor;
pub mod unified_config;
pub mod unified_queue_processor;
pub mod unified_queue_schema;
pub mod watch_folders_schema;
pub mod watching;
pub mod watching_queue;

// ── Architectural refactoring modules (Phase 0+) ────────────────────────
pub mod context;
pub mod lifecycle;
pub mod pipeline;
pub mod shared;
pub mod specs;
pub mod strategies;

// ── Backward-compatible module aliases ──────────────────────────────────
// Grouping aliases (used by cross_project_search, schema versions)
pub use grouping::affinity as affinity_grouper;
pub use grouping::dependency as dependency_grouper;
pub use grouping::git_org as git_org_grouper;
pub use grouping::schema as project_groups_schema;
pub use grouping::workspace as workspace_grouper;

// Tagging aliases (unused but retained for safety during refactoring)
pub use tagging as tier1_tagging;
pub use tagging as tier2_tagging;
pub use tagging as tier3_tagging;

// Monitoring aliases (used by priority_manager, queue_operations)
pub use monitoring as logging;
pub use monitoring as metrics;
pub use monitoring as metrics_history;
pub use monitoring as remote_monitor;
pub use monitoring as tool_monitor;

// ── Re-exports: core types ──────────────────────────────────────────────
pub use crate::core_types::{
    ChunkingConfig, DocumentContent, DocumentResult, DocumentType, ProcessingError,
    ProcessingStats, TextChunk,
};
pub use crate::document_id::{
    generate_content_document_id, generate_document_id, generate_point_id,
};
pub use crate::ingestion::IngestionEngine;

// ── Re-exports: subsystem types ─────────────────────────────────────────
pub use crate::allowed_extensions::{AllowedExtensions, FileRoute};
pub use crate::daemon_state::{poll_pause_state, DaemonStateManager};
pub use crate::document_processor::{
    DocumentProcessor, DocumentProcessorError, DocumentProcessorResult,
};
pub use crate::embedding::{
    DenseEmbedding, EmbeddingConfig, EmbeddingError, EmbeddingGenerator, EmbeddingResult,
    PreprocessedText, SparseEmbedding, BM25,
};
pub use crate::error::{
    CircuitBreaker, DaemonError, ErrorMonitor, ErrorRecovery, ErrorRecoveryStrategy, ErrorSeverity,
    Result, WorkspaceError,
};
pub use crate::git::{
    branch_schema, BranchEvent, BranchEventHandler, BranchLifecycleConfig, BranchLifecycleDetector,
    BranchLifecycleStats, CacheStats, GitBranchDetector, GitError, GitResult,
};
pub use crate::monitoring::{
    create_orphaned_session_alert,
    create_slow_search_alert,
    initialize_daemon_silence,
    initialize_logging,
    log_error_with_context,
    log_ingestion_error,
    log_priority_change,
    log_queue_depth_change,
    log_queue_enqueue,
    log_queue_processed,
    log_search_request,
    log_search_result,
    log_session_cleanup,
    log_session_deprioritize,
    log_session_heartbeat,
    log_session_register,
    log_slow_query,
    track_async_operation,
    // Alerting
    Alert,
    AlertChecker,
    AlertConfig,
    AlertSeverity,
    AlertType,
    // Metrics
    DaemonMetrics,
    // Logging
    LoggingConfig,
    LoggingErrorMonitor,
    MetricsServer,
    MetricsSnapshot,
    MonitoringError,
    MonitoringResult,
    PerformanceMetrics,
    QueueContext,
    RequeueStats,
    SearchContext,
    // Multi-tenant structured logging
    SessionContext,
    // Tool monitor
    ToolMonitor,
    METRICS,
};
pub use crate::patterns::{
    AllPatterns, CommentSyntax, ConfidenceLevel, Ecosystem, ExcludePatterns, IncludePatterns,
    LanguageExtensions, LanguageGroup, PatternError, PatternManager, PatternResult,
    ProjectIndicator, ProjectIndicators,
};
pub use crate::priority_manager::{
    priority, OrphanedSessionCleanup, PriorityError, PriorityManager, PriorityResult, SessionInfo,
    SessionMonitor, SessionMonitorConfig,
};
pub use crate::processing::{
    Pipeline, TaskPayload, TaskPriority, TaskResult, TaskResultData, TaskResultHandle, TaskSource,
    TaskSubmitter,
};
pub use crate::queue_health::QueueProcessorHealth;
pub use crate::queue_operations::{
    QueueError, QueueLoadLevel as QueueOpsLoadLevel, QueueManager, QueueThrottlingSummary,
};
pub use crate::queue_types::{MissingTool, ProcessorConfig};
pub use crate::schema_version::{SchemaError, SchemaManager, CURRENT_SCHEMA_VERSION};
pub use crate::search_db::{
    search_db_path_from_state, SearchDbError, SearchDbManager, SearchDbResult, SEARCH_DB_FILENAME,
    SEARCH_SCHEMA_VERSION,
};
pub use crate::service_discovery::{
    DiscoveryConfig, DiscoveryError, DiscoveryManager, DiscoveryMessage, DiscoveryMessageType,
    DiscoveryResult, HealthChecker, HealthConfig, HealthStatus, NetworkDiscovery, ServiceInfo,
    ServiceRegistry, ServiceStatus,
};
pub use crate::tracing_otel::{
    current_span_id, current_trace_id, init_tracer_provider, otel_layer, shutdown_tracer,
    OtelConfig,
};
pub use crate::tracked_files_schema::{
    ChunkType as TrackedChunkType, ProcessingStatus, QdrantChunk, TrackedFile,
    CREATE_QDRANT_CHUNKS_INDEXES_SQL, CREATE_QDRANT_CHUNKS_SQL, CREATE_RECONCILE_INDEX_SQL,
    CREATE_TRACKED_FILES_INDEXES_SQL, CREATE_TRACKED_FILES_SQL, MIGRATE_V3_SQL,
};
// Note: QueueProcessor removed per Task 21 - use UnifiedQueueProcessor
pub use crate::fairness_scheduler::{
    FairnessError, FairnessMetrics, FairnessResult, FairnessScheduler, FairnessSchedulerConfig,
};
pub use crate::file_classification::{
    classify_file_type, get_extension_for_storage, is_test_directory, is_test_file, FileType,
};
pub use crate::keyword_extraction::hierarchy_builder::{
    HierarchyBuilder, HierarchyError, HierarchyRebuildConfig,
};
pub use crate::lexicon::LexiconManager;
pub use crate::lifecycle::{WatchFolderLifecycle, WatchFolderLifecycleError};
pub use crate::lsp::{
    EnrichmentStatus, Language, LanguageMarker, LanguageServerManager, LspEnrichment, LspError,
    LspResult, ProjectLanguageDetector, ProjectLanguageKey, ProjectLanguageResult,
    ProjectLspConfig, ProjectLspError, ProjectLspResult, ProjectLspStats, ProjectServerState,
    Reference, ResolvedImport, TypeInfo,
};
pub use crate::metadata_enrichment::{enrich_metadata, CollectionType};
pub use crate::monitoring::{check_git_state_changes, check_remote_url_changes};
pub use crate::project_disambiguation::{
    DisambiguationConfig, DisambiguationError, DisambiguationPathComputer, DisambiguationResult,
    ProjectIdCalculator, ProjectRecord, RegisteredProject,
};
pub use crate::startup::{
    backfill_rules_mirror, clean_stale_state, run_startup_recovery, validate_watch_folders,
    FullRecoveryStats, RecoveryStats, RulesBackfillStats, StaleCleanupStats, WatchValidationStats,
};
pub use crate::storage::{
    BatchStats, DocumentPoint, HybridSearchMode, MultiTenantConfig, MultiTenantInitResult,
    SearchParams, SearchResult, StorageClient, StorageConfig, StorageError,
};
pub use crate::tree_sitter::{
    check_grammar_compatibility, create_grammar_manager, detect_language, extract_chunks,
    extract_chunks_with_provider, get_language, get_static_language, is_language_available,
    is_language_supported, known_grammar_languages, ChunkExtractor, ChunkType, CompatibilityStatus,
    DownloadError, GrammarCachePaths, GrammarDownloader, GrammarError, GrammarInfo, GrammarLoader,
    GrammarManager, GrammarMetadata, GrammarResult, GrammarStatus, GrammarValidationResult,
    LanguageProvider, LoadedGrammar, LoadedGrammarsProvider, RuntimeInfo, SemanticChunk,
    SemanticChunker, StaticLanguageProvider, TreeSitterParser, VersionError,
};
pub use crate::type_aware_processor::{
    get_settings_for_type, CollectionTypeSettings, ConcurrentOperationTracker,
};
pub use crate::unified_queue_processor::{
    UnifiedProcessingMetrics, UnifiedProcessorConfig, UnifiedProcessorError,
    UnifiedProcessorResult, UnifiedQueueProcessor,
};
pub use crate::unified_queue_schema::{
    generate_idempotency_key, generate_unified_idempotency_key, CollectionPayload, ContentPayload,
    DeleteDocumentPayload, DeleteTenantPayload, FilePayload, FolderPayload, IdempotencyKeyError,
    ItemType, LibraryPayload, ProjectPayload, QueueOperation as UnifiedQueueOp, QueueStatus,
    UnifiedQueueItem, UnifiedQueueStats, WebsitePayload, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    CREATE_UNIFIED_QUEUE_SQL,
};
pub use crate::watching_queue::{
    get_current_branch, BackoffConfig, CircuitBreakerState, CoordinatorConfig, CoordinatorSummary,
    ErrorFeedbackManager, FileWatcherQueue, ProcessingErrorFeedback, ProcessingErrorSummary,
    ProcessingErrorType, QueueLoadLevel, QueueThrottleConfig, QueueThrottleState,
    QueueThrottleSummary, WatchConfig, WatchErrorState, WatchHealthStatus, WatchManager,
    WatchQueueCoordinator, WatchType, WatchingQueueError, WatchingQueueStats,
};
pub use wqm_common::project_id::calculate_tenant_id;
