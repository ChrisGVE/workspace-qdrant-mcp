//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

// ── Module declarations ─────────────────────────────────────────────────

pub mod adaptive_resources;
pub mod grouping;
pub mod component_detection;
pub mod allowed_extensions;
pub mod idle_history;
pub mod config;
pub mod core_types;
pub mod document_id;
pub mod document_processor;
pub mod embedding;
pub mod error;
pub mod fairness_scheduler;
pub mod file_classification;
pub mod format_routing;
pub mod fts_batch_processor;
pub mod branch_switch;
pub mod git;
pub mod clip;
pub mod cross_project_search;
pub mod graph;
pub mod image_extraction;
pub mod library_hierarchy;
pub mod image_ingestion;
pub mod image_search;
pub mod ingestion;
pub mod metadata_enrichment;
pub mod ocr;
pub mod ipc;
pub mod log_pruner;
pub mod monitoring;
pub mod daemon_state;
pub mod telemetry;
pub mod tracing_otel;
pub mod patterns;
pub mod priority_manager;
pub mod processing;
pub mod processing_timings;
pub mod queue_config;
pub mod queue_health;
pub mod queue_operations;
pub mod queue_types;
// Note: queue_processor module removed per Task 21 - use unified_queue_processor
pub mod queue_error_handler;
pub mod service_discovery;
pub mod source_diversity;
pub mod storage;
pub mod type_aware_processor;
pub mod unified_config;
pub mod unified_queue_processor;
pub mod unified_queue_schema;
pub mod watch_folders_schema;
pub mod tracked_files_schema;
pub mod metrics_history_schema;
pub mod search_events_schema;
pub mod code_lines_schema;
pub mod schema_version;
pub mod search_db;
pub mod text_search;
pub mod tagging;
pub mod grep_search;
pub mod tokenizer;
pub mod parent_unit;
pub mod title_extraction;
pub mod resolution_events_schema;
pub mod cooccurrence_schema;
pub mod keyword_extraction;
pub mod indexed_content_schema;
pub mod keywords_schema;
pub mod lexicon;
pub mod line_diff;
pub mod library_document;
pub mod metadata_uplift;
pub mod thumbnail;
pub mod tree_sitter;
pub mod watching;
pub mod watching_queue;
pub mod lsp;
pub mod project_disambiguation;
pub mod startup;

// ── Architectural refactoring modules (Phase 0+) ────────────────────────
pub mod context;
pub mod shared;
pub mod specs;
pub mod strategies;
pub mod pipeline;
pub mod lifecycle;

// ── Backward-compatible module aliases ──────────────────────────────────
// Grouping aliases (used by cross_project_search, schema versions)
pub use grouping::schema as project_groups_schema;
pub use grouping::affinity as affinity_grouper;
pub use grouping::dependency as dependency_grouper;
pub use grouping::workspace as workspace_grouper;
pub use grouping::git_org as git_org_grouper;

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
    ProcessingError, DocumentResult, ProcessingStats,
    DocumentType, TextChunk, DocumentContent, ChunkingConfig,
};
pub use crate::document_id::{
    generate_document_id, generate_point_id, generate_content_document_id,
};
pub use crate::ingestion::IngestionEngine;

// ── Re-exports: subsystem types ─────────────────────────────────────────
pub use crate::allowed_extensions::{AllowedExtensions, FileRoute};
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
pub use crate::git::{
    GitBranchDetector, GitError, GitResult, CacheStats,
    BranchEvent, BranchLifecycleDetector, BranchLifecycleConfig, BranchLifecycleStats,
    BranchEventHandler, branch_schema
};
pub use crate::monitoring::{
    // Logging
    LoggingConfig, PerformanceMetrics, initialize_logging, initialize_daemon_silence,
    track_async_operation, log_error_with_context, LoggingErrorMonitor,
    // Multi-tenant structured logging
    SessionContext, QueueContext, SearchContext,
    log_session_register, log_session_heartbeat, log_session_deprioritize,
    log_session_cleanup, log_priority_change, log_queue_depth_change,
    log_queue_enqueue, log_queue_processed, log_slow_query,
    log_search_request, log_search_result, log_ingestion_error,
    // Metrics
    DaemonMetrics, MetricsServer, MetricsSnapshot, METRICS,
    // Alerting
    Alert, AlertChecker, AlertConfig, AlertSeverity, AlertType,
    create_orphaned_session_alert, create_slow_search_alert,
    // Tool monitor
    ToolMonitor, MonitoringError, MonitoringResult, RequeueStats,
};
pub use crate::tracing_otel::{
    OtelConfig, init_tracer_provider, otel_layer, shutdown_tracer,
    current_trace_id, current_span_id
};
pub use crate::daemon_state::{
    DaemonStateManager, poll_pause_state,
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
pub use crate::search_db::{
    SearchDbManager, SearchDbError, SearchDbResult,
    search_db_path_from_state, SEARCH_SCHEMA_VERSION, SEARCH_DB_FILENAME,
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
    PriorityManager, PriorityError, PriorityResult,
    SessionMonitor, SessionMonitorConfig, SessionInfo, OrphanedSessionCleanup,
    priority
};
pub use crate::queue_health::QueueProcessorHealth;
pub use crate::queue_operations::{
    QueueManager, QueueError,
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
pub use wqm_common::project_id::calculate_tenant_id;
pub use crate::watching_queue::{
    FileWatcherQueue, WatchManager, WatchConfig, WatchingQueueStats, WatchingQueueError,
    get_current_branch,
    WatchType,
    QueueLoadLevel, QueueThrottleConfig, QueueThrottleState, QueueThrottleSummary,
    WatchQueueCoordinator, CoordinatorConfig, CoordinatorSummary,
    WatchHealthStatus, WatchErrorState, BackoffConfig, CircuitBreakerState,
    ProcessingErrorType, ProcessingErrorFeedback, ErrorFeedbackManager, ProcessingErrorSummary
};
pub use crate::file_classification::{
    classify_file_type, is_test_file, is_test_directory, get_extension_for_storage, FileType
};
pub use crate::metadata_enrichment::{
    enrich_metadata, CollectionType
};
pub use crate::unified_queue_schema::{
    ItemType, QueueOperation as UnifiedQueueOp, QueueStatus,
    ContentPayload, FilePayload, FolderPayload, ProjectPayload,
    LibraryPayload, DeleteTenantPayload, DeleteDocumentPayload,
    WebsitePayload, CollectionPayload,
    generate_idempotency_key, generate_unified_idempotency_key, IdempotencyKeyError,
    CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    UnifiedQueueItem, UnifiedQueueStats
};
pub use crate::storage::{
    StorageClient, StorageConfig, StorageError,
    MultiTenantConfig, MultiTenantInitResult,
    DocumentPoint, SearchResult, SearchParams, HybridSearchMode, BatchStats,
};
pub use crate::tree_sitter::{
    SemanticChunker, TreeSitterParser, ChunkExtractor, ChunkType, SemanticChunk,
    extract_chunks, extract_chunks_with_provider, detect_language, is_language_supported, is_language_available,
    GrammarManager, GrammarError, GrammarResult, GrammarStatus, GrammarInfo,
    GrammarCachePaths, GrammarMetadata, GrammarLoader, LoadedGrammar,
    GrammarDownloader, DownloadError, LoadedGrammarsProvider, GrammarValidationResult,
    LanguageProvider, StaticLanguageProvider, get_language, get_static_language,
    check_grammar_compatibility, CompatibilityStatus, RuntimeInfo, VersionError,
    create_grammar_manager,
};
pub use crate::project_disambiguation::{
    ProjectIdCalculator, DisambiguationPathComputer,
    ProjectRecord, RegisteredProject, DisambiguationConfig,
    DisambiguationError, DisambiguationResult,
};
pub use crate::lsp::{
    LanguageServerManager, ProjectLspConfig, ProjectLspError, ProjectLspResult,
    ProjectLanguageKey, ProjectServerState, ProjectLspStats,
    LspEnrichment, EnrichmentStatus, Reference, TypeInfo, ResolvedImport,
    Language, LspError, LspResult,
    ProjectLanguageDetector, ProjectLanguageResult, LanguageMarker,
};
pub use crate::startup::{
    run_startup_recovery, RecoveryStats, FullRecoveryStats,
    backfill_rules_mirror, RulesBackfillStats,
    clean_stale_state, validate_watch_folders,
    StaleCleanupStats, WatchValidationStats,
};
pub use crate::monitoring::{check_remote_url_changes, check_git_state_changes};
pub use crate::keyword_extraction::hierarchy_builder::{
    HierarchyBuilder, HierarchyRebuildConfig, HierarchyError,
};
pub use crate::lexicon::LexiconManager;
pub use crate::lifecycle::{
    WatchFolderLifecycle, WatchFolderLifecycleError,
};
