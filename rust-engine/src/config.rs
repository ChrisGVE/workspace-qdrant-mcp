//! Configuration management for the Workspace Qdrant Daemon

use crate::error::DaemonResult;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::collections::HashMap;

// =============================================================================
// UNIT PARSING UTILITIES
// =============================================================================

/// Size unit that supports parsing from strings with units (B, KB, MB, GB, TB)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SizeUnit(pub u64);

impl FromStr for SizeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("Empty size string".to_string());
        }

        // Extract number and unit
        let (num_str, unit) = if s.ends_with("TB") {
            (&s[..s.len() - 2], 1_099_511_627_776)
        } else if s.ends_with("GB") {
            (&s[..s.len() - 2], 1_073_741_824)
        } else if s.ends_with("MB") {
            (&s[..s.len() - 2], 1_048_576)
        } else if s.ends_with("KB") {
            (&s[..s.len() - 2], 1_024)
        } else if s.ends_with('B') {
            (&s[..s.len() - 1], 1)
        } else {
            // No unit, assume bytes
            (s, 1)
        };

        let num: u64 = num_str.trim().parse()
            .map_err(|_| format!("Invalid number: {}", num_str))?;

        Ok(SizeUnit(num * unit))
    }
}

impl fmt::Display for SizeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0;
        if bytes >= 1_099_511_627_776 {
            write!(f, "{}TB", bytes / 1_099_511_627_776)
        } else if bytes >= 1_073_741_824 {
            write!(f, "{}GB", bytes / 1_073_741_824)
        } else if bytes >= 1_048_576 {
            write!(f, "{}MB", bytes / 1_048_576)
        } else if bytes >= 1_024 {
            write!(f, "{}KB", bytes / 1_024)
        } else {
            write!(f, "{}B", bytes)
        }
    }
}

impl TryFrom<String> for SizeUnit {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<SizeUnit> for String {
    fn from(size: SizeUnit) -> String {
        size.to_string()
    }
}

/// Time duration unit that supports parsing from strings with units (ms, s, m, h)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TimeUnit(pub u64); // Store as milliseconds

impl FromStr for TimeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("Empty time string".to_string());
        }

        // Extract number and unit
        let (num_str, unit_ms) = if s.ends_with("ms") {
            (&s[..s.len() - 2], 1)
        } else if s.ends_with('s') {
            (&s[..s.len() - 1], 1_000)
        } else if s.ends_with('m') {
            (&s[..s.len() - 1], 60_000)
        } else if s.ends_with('h') {
            (&s[..s.len() - 1], 3_600_000)
        } else {
            // No unit, assume seconds
            (s, 1_000)
        };

        let num: u64 = num_str.trim().parse()
            .map_err(|_| format!("Invalid number: {}", num_str))?;

        Ok(TimeUnit(num * unit_ms))
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0;
        if ms >= 3_600_000 {
            write!(f, "{}h", ms / 3_600_000)
        } else if ms >= 60_000 {
            write!(f, "{}m", ms / 60_000)
        } else if ms >= 1_000 {
            write!(f, "{}s", ms / 1_000)
        } else {
            write!(f, "{}ms", ms)
        }
    }
}

impl TryFrom<String> for TimeUnit {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<TimeUnit> for String {
    fn from(time: TimeUnit) -> String {
        time.to_string()
    }
}

// =============================================================================
// MAIN CONFIGURATION STRUCTURE - PRDv3 COMPLIANT
// =============================================================================

/// Main daemon configuration structure matching PRDv3 YAML format
/// Contains 13 major sections as specified in templates/default_config.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// 1. System Architecture & Core Settings
    pub system: SystemConfig,

    /// 2. Memory Collection Configuration
    pub memory: MemoryConfig,

    /// 3. Collection Management & Multi-tenancy
    pub collections: CollectionsConfig,

    /// 4. Project Detection & Management
    pub project_detection: ProjectDetectionConfig,

    /// 5. LSP Integration & Code Intelligence
    pub lsp_integration: LspIntegrationConfig,

    /// 6. Document Processing & Ingestion
    pub document_processing: DocumentProcessingConfig,

    /// 7. Search & Indexing Configuration
    pub search: SearchConfig,

    /// 8. Performance & Resource Management
    pub performance: PerformanceConfig,

    /// 9. Platform & Directory Configuration
    pub platform: PlatformConfig,

    /// 10. CLI Behavior Configuration
    pub cli: CliConfig,

    /// 11. gRPC & Communication Settings
    pub grpc: GrpcConfig,

    /// 12. External Service Configuration
    pub external_services: ExternalServicesConfig,

    /// 13. Monitoring & Logging Configuration
    pub monitoring: MonitoringConfig,
}

// =============================================================================
// 1. SYSTEM ARCHITECTURE & CORE SETTINGS
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System identification and versioning
    pub project_name: String,
    pub version: String,

    /// Four-component architecture enablement
    pub components: ComponentsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentsConfig {
    pub rust_daemon: ComponentConfig,
    pub python_mcp_server: ComponentConfig,
    pub cli_utility: ComponentConfig,
    pub context_injector: ComponentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfig {
    pub enabled: bool,
}

// =============================================================================
// 2. MEMORY COLLECTION CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Core memory collection settings
    pub collection_name: String,

    /// Authority levels for memory rules
    pub authority_levels: AuthorityLevelsConfig,

    /// Conversational memory update settings
    pub conversational_updates: ConversationalUpdatesConfig,

    /// Rule conflict resolution strategies
    pub conflict_resolution: ConflictResolutionConfig,

    /// Token management and optimization
    pub token_management: TokenManagementConfig,

    /// Memory rule scoping
    pub rule_scope: RuleScopeConfig,

    /// Session initialization settings
    pub session_initialization: SessionInitializationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityLevelsConfig {
    pub absolute: AuthorityLevelConfig,
    pub default: AuthorityLevelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityLevelConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalUpdatesConfig {
    pub enabled: bool,
    pub auto_conflict_detection: bool,
    pub immediate_activation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionConfig {
    pub strategy: String,
    pub user_prompt_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenManagementConfig {
    pub max_tokens: u32,
    pub optimization_enabled: bool,
    pub trim_interactive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleScopeConfig {
    pub all_sessions: bool,
    pub project_specific: bool,
    pub temporary_rules: bool,
    pub temporary_rule_duration_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInitializationConfig {
    pub rule_injection_enabled: bool,
    pub conflict_detection_on_startup: bool,
    pub startup_timeout_seconds: u64,
}

// =============================================================================
// 3. COLLECTION MANAGEMENT & MULTI-TENANCY
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionsConfig {
    /// Project collection naming and creation
    pub root_name: String,
    pub types: Vec<String>,

    /// Project content collection (for file artifacts)
    pub project_content: ProjectContentConfig,

    /// Reserved naming patterns and validation
    pub naming: CollectionNamingConfig,

    /// Multi-tenancy isolation settings
    pub multi_tenancy: MultiTenancyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContentConfig {
    pub enabled: bool,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionNamingConfig {
    pub collision_detection: bool,
    pub validation_strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTenancyConfig {
    pub isolation_strategy: String,
    pub cross_project_search: bool,
    pub tenant_metadata_fields: Vec<String>,
}

// =============================================================================
// 4. PROJECT DETECTION & MANAGEMENT
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDetectionConfig {
    /// Git-based project detection (primary method)
    pub git_detection: GitDetectionConfig,

    /// GitHub user configuration for submodule handling
    pub github_integration: GitHubIntegrationConfig,

    /// Custom project indicators (fallback method)
    pub custom_indicators: CustomIndicatorsConfig,

    /// Project naming and collection creation
    pub naming: ProjectNamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitDetectionConfig {
    pub enabled: bool,
    pub priority: u8,
    pub require_initialized: bool,
    pub scan_parent_directories: bool,
    pub max_parent_scan_depth: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubIntegrationConfig {
    pub user: String,
    pub submodule_handling: SubmoduleHandlingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmoduleHandlingConfig {
    pub treat_as_independent_projects: bool,
    pub ignore_external_submodules: bool,
    pub track_ownership_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomIndicatorsConfig {
    pub enabled: bool,
    pub priority: u8,
    pub additional_patterns: AdditionalPatternsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdditionalPatternsConfig {
    pub custom_indicators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectNamingConfig {
    pub root_name_strategy: String,
    pub collection_auto_creation: bool,
    pub prevent_project_explosion: bool,
}

// =============================================================================
// 5. LSP INTEGRATION & CODE INTELLIGENCE
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspIntegrationConfig {
    /// Core LSP settings
    pub enabled: bool,
    pub auto_detection: bool,
    pub auto_installation: bool,

    /// LSP server selection and configuration
    pub server_override: ServerOverrideConfig,

    /// Health monitoring and graceful degradation
    pub health_monitoring: LspHealthMonitoringConfig,

    /// Graceful degradation levels
    pub degradation: LspDegradationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerOverrideConfig {
    pub enabled: bool,
    pub overrides: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspHealthMonitoringConfig {
    pub enabled: bool,
    pub check_interval_seconds: u64,
    pub automatic_recovery: bool,
    pub max_restart_attempts: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspDegradationConfig {
    pub enabled: bool,
    pub level_1_lsp_crash: DegradationActionConfig,
    pub level_2_multiple_failures: DegradationActionConfig,
    pub level_3_complete_failure: DegradationActionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationActionConfig {
    pub action: String,
}

// =============================================================================
// 6. DOCUMENT PROCESSING & INGESTION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentProcessingConfig {
    /// Supported document types and processing
    pub supported_types: SupportedTypesConfig,

    /// File watching and auto-ingestion
    pub file_watching: FileWatchingConfig,

    /// Processing performance and chunking
    pub chunking: ChunkingConfig,

    /// Performance settings
    pub performance: ProcessingPerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedTypesConfig {
    pub text: TextProcessingConfig,
    pub pdf: PdfProcessingConfig,
    pub epub_mobi: EpubMobiProcessingConfig,
    pub code: CodeProcessingConfig,
    pub web: WebProcessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfProcessingConfig {
    pub enabled: bool,
    pub text_extraction: bool,
    pub ocr_required_detection: bool,
    pub store_ocr_required_in_state: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpubMobiProcessingConfig {
    pub enabled: bool,
    pub metadata_preservation: bool,
    pub extract_toc: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProcessingConfig {
    pub enabled: bool,
    pub lsp_enhanced: bool,
    pub fallback_to_treesitter: bool,
    pub fallback_to_storage_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebProcessingConfig {
    pub single_page: WebSinglePageConfig,
    pub recursive_crawling: WebRecursiveCrawlingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSinglePageConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRecursiveCrawlingConfig {
    pub enabled: bool,
    pub max_depth: u8,
    pub rate_limiting: WebRateLimitingConfig,
    pub respect_robots_txt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRateLimitingConfig {
    pub requests_per_second: f32,
    pub concurrent_connections: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWatchingConfig {
    pub enabled: bool,
    pub project_folders: ProjectFoldersConfig,
    pub library_folders: LibraryFoldersConfig,
    pub incremental_updates: IncrementalUpdatesConfig,
    pub debouncing: DebouncingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFoldersConfig {
    pub auto_monitor: bool,
    pub zero_configuration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryFoldersConfig {
    pub user_configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdatesConfig {
    pub content_hash_tracking: bool,
    pub modification_time_tracking: bool,
    pub process_only_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebouncingConfig {
    pub delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub default_chunk_size: u32,
    pub default_chunk_overlap: u32,
    pub max_file_size_bytes: SizeUnit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPerformanceConfig {
    pub max_concurrent_tasks: u8,
    pub batch_size: u32,
}

// =============================================================================
// 7. SEARCH & INDEXING CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Search modes and contexts
    pub modes: SearchModesConfig,

    /// Hybrid search configuration
    pub hybrid: HybridSearchConfig,

    /// Result formatting and limits
    pub results: SearchResultsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchModesConfig {
    pub project: SearchModeConfig,
    pub collection: SearchModeConfig,
    pub global: SearchModeConfig,
    pub all: SearchModeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchModeConfig {
    pub enabled: bool,
    #[serde(default)]
    pub default_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    pub enabled: bool,
    pub fusion_algorithm: String,
    pub dense_weight: f32,
    pub sparse_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultsConfig {
    pub default_limit: u32,
    pub max_limit: u32,
    pub include_metadata: bool,
    pub include_snippets: bool,
}

// =============================================================================
// 8. PERFORMANCE & RESOURCE MANAGEMENT
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Resource constraints and limits
    pub memory: MemoryPerformanceConfig,

    /// Startup and initialization
    pub startup: StartupConfig,

    /// CPU and priority management
    pub cpu: CpuConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceConfig {
    pub max_rss_mb: u32,
    pub warning_threshold_mb: u32,
    pub gc_threshold_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupConfig {
    pub daemon_init_timeout_seconds: u64,
    pub mcp_server_init_timeout_seconds: u64,
    pub health_check_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    pub background_priority: String,
    pub interactive_priority: String,
    pub priority_boost_on_mcp_active: bool,
    pub revert_priority_on_mcp_quit: bool,
}

// =============================================================================
// 9. PLATFORM & DIRECTORY CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// XDG Base Directory Specification compliance
    pub xdg_compliance: XdgComplianceConfig,

    /// Platform-specific directory mapping
    pub directories: PlatformDirectoriesConfig,

    /// File pattern customization
    pub file_patterns: FilePatternsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XdgComplianceConfig {
    pub enabled: bool,
    pub fallback_on_windows_mac: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDirectoriesConfig {
    pub linux: PlatformDirectorySet,
    pub macos: PlatformDirectorySet,
    pub windows: PlatformDirectorySet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformDirectorySet {
    pub cache: String,
    pub logs: String,
    pub config: String,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePatternsConfig {
    pub include_patterns: IncludePatternsConfig,
    pub exclude_patterns: ExcludePatternsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludePatternsConfig {
    pub custom: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludePatternsConfig {
    pub custom: Vec<String>,
}

// =============================================================================
// 10. CLI BEHAVIOR CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// CLI behavior and user experience settings
    pub behavior: CliBehaviorConfig,

    /// Command-specific settings
    pub commands: CliCommandsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliBehaviorConfig {
    pub interactive_mode: bool,
    pub default_output_format: String,
    pub color_output: bool,
    pub progress_indicators: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliCommandsConfig {
    pub memory: CliMemoryConfig,
    pub admin: CliAdminConfig,
    pub ingest: CliIngestConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliMemoryConfig {
    pub auto_backup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliAdminConfig {
    pub confirm_destructive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliIngestConfig {
    pub show_progress: bool,
}

// =============================================================================
// 11. gRPC & COMMUNICATION SETTINGS
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Core gRPC server settings
    pub server: GrpcServerConfig,

    /// Client settings (for Python MCP server)
    pub client: GrpcClientConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcServerConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub max_concurrent_streams: u32,
    pub max_message_size_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcClientConfig {
    pub connection_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub keepalive_interval_seconds: u64,
    pub keepalive_timeout_seconds: u64,
    pub max_retry_attempts: u8,
    pub retry_backoff_ms: u64,
}

// =============================================================================
// 12. EXTERNAL SERVICE CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalServicesConfig {
    /// Qdrant vector database settings
    pub qdrant: QdrantServiceConfig,

    /// FastEmbed embeddings configuration
    pub embeddings: EmbeddingsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantServiceConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    pub model: String,
    pub cache_dir: Option<String>,
}

// =============================================================================
// 13. MONITORING & LOGGING CONFIGURATION
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Logging configuration
    pub logging: LoggingConfig,

    /// Metrics collection
    pub metrics: MetricsConfig,

    /// Health checks
    pub health_checks: HealthChecksConfig,
}

// Note: LoggingConfig is defined in legacy compatibility section

// Note: MetricsConfig is defined in legacy compatibility section

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthChecksConfig {
    pub enabled: bool,
    pub check_interval_seconds: u64,
    pub failure_threshold: u8,
}

// =============================================================================
// IMPLEMENTATION
// =============================================================================

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            memory: MemoryConfig::default(),
            collections: CollectionsConfig::default(),
            project_detection: ProjectDetectionConfig::default(),
            lsp_integration: LspIntegrationConfig::default(),
            document_processing: DocumentProcessingConfig::default(),
            search: SearchConfig::default(),
            performance: PerformanceConfig::default(),
            platform: PlatformConfig::default(),
            cli: CliConfig::default(),
            grpc: GrpcConfig::default(),
            external_services: ExternalServicesConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            project_name: "workspace-qdrant-mcp".to_string(),
            version: "v2.0".to_string(),
            components: ComponentsConfig::default(),
        }
    }
}

impl Default for ComponentsConfig {
    fn default() -> Self {
        Self {
            rust_daemon: ComponentConfig { enabled: true },
            python_mcp_server: ComponentConfig { enabled: true },
            cli_utility: ComponentConfig { enabled: true },
            context_injector: ComponentConfig { enabled: true },
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            collection_name: "llm_rules".to_string(),
            authority_levels: AuthorityLevelsConfig {
                absolute: AuthorityLevelConfig { enabled: true },
                default: AuthorityLevelConfig { enabled: true },
            },
            conversational_updates: ConversationalUpdatesConfig {
                enabled: true,
                auto_conflict_detection: true,
                immediate_activation: true,
            },
            conflict_resolution: ConflictResolutionConfig {
                strategy: "merge_conditional".to_string(),
                user_prompt_timeout_seconds: 30,
            },
            token_management: TokenManagementConfig {
                max_tokens: 2000,
                optimization_enabled: true,
                trim_interactive: true,
            },
            rule_scope: RuleScopeConfig {
                all_sessions: true,
                project_specific: true,
                temporary_rules: true,
                temporary_rule_duration_hours: 24,
            },
            session_initialization: SessionInitializationConfig {
                rule_injection_enabled: true,
                conflict_detection_on_startup: true,
                startup_timeout_seconds: 10,
            },
        }
    }
}

impl Default for CollectionsConfig {
    fn default() -> Self {
        Self {
            root_name: "project".to_string(),
            types: vec![],
            project_content: ProjectContentConfig {
                enabled: true,
                name: "project_content".to_string(),
            },
            naming: CollectionNamingConfig {
                collision_detection: true,
                validation_strict: true,
            },
            multi_tenancy: MultiTenancyConfig {
                isolation_strategy: "metadata_filtering".to_string(),
                cross_project_search: true,
                tenant_metadata_fields: vec![
                    "project_id".to_string(),
                    "project_path".to_string(),
                    "git_repository".to_string(),
                ],
            },
        }
    }
}

impl Default for ProjectDetectionConfig {
    fn default() -> Self {
        Self {
            git_detection: GitDetectionConfig {
                enabled: true,
                priority: 1,
                require_initialized: true,
                scan_parent_directories: true,
                max_parent_scan_depth: 10,
            },
            github_integration: GitHubIntegrationConfig {
                user: "".to_string(),
                submodule_handling: SubmoduleHandlingConfig {
                    treat_as_independent_projects: true,
                    ignore_external_submodules: true,
                    track_ownership_changes: true,
                },
            },
            custom_indicators: CustomIndicatorsConfig {
                enabled: true,
                priority: 2,
                additional_patterns: AdditionalPatternsConfig {
                    custom_indicators: vec![],
                },
            },
            naming: ProjectNamingConfig {
                root_name_strategy: "directory_name".to_string(),
                collection_auto_creation: true,
                prevent_project_explosion: true,
            },
        }
    }
}

impl Default for LspIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_detection: true,
            auto_installation: false,
            server_override: ServerOverrideConfig {
                enabled: false,
                overrides: HashMap::new(),
            },
            health_monitoring: LspHealthMonitoringConfig {
                enabled: true,
                check_interval_seconds: 30,
                automatic_recovery: true,
                max_restart_attempts: 3,
            },
            degradation: LspDegradationConfig {
                enabled: true,
                level_1_lsp_crash: DegradationActionConfig {
                    action: "continue_text_only".to_string(),
                },
                level_2_multiple_failures: DegradationActionConfig {
                    action: "fallback_treesitter".to_string(),
                },
                level_3_complete_failure: DegradationActionConfig {
                    action: "text_only_mode".to_string(),
                },
            },
        }
    }
}

impl Default for DocumentProcessingConfig {
    fn default() -> Self {
        Self {
            supported_types: SupportedTypesConfig {
                text: TextProcessingConfig { enabled: true },
                pdf: PdfProcessingConfig {
                    enabled: true,
                    text_extraction: true,
                    ocr_required_detection: true,
                    store_ocr_required_in_state: true,
                },
                epub_mobi: EpubMobiProcessingConfig {
                    enabled: true,
                    metadata_preservation: true,
                    extract_toc: true,
                },
                code: CodeProcessingConfig {
                    enabled: true,
                    lsp_enhanced: true,
                    fallback_to_treesitter: true,
                    fallback_to_storage_only: true,
                },
                web: WebProcessingConfig {
                    single_page: WebSinglePageConfig { enabled: true },
                    recursive_crawling: WebRecursiveCrawlingConfig {
                        enabled: true,
                        max_depth: 3,
                        rate_limiting: WebRateLimitingConfig {
                            requests_per_second: 2.0,
                            concurrent_connections: 3,
                        },
                        respect_robots_txt: true,
                    },
                },
            },
            file_watching: FileWatchingConfig {
                enabled: true,
                project_folders: ProjectFoldersConfig {
                    auto_monitor: true,
                    zero_configuration: true,
                },
                library_folders: LibraryFoldersConfig {
                    user_configured: true,
                },
                incremental_updates: IncrementalUpdatesConfig {
                    content_hash_tracking: true,
                    modification_time_tracking: true,
                    process_only_changes: true,
                },
                debouncing: DebouncingConfig { delay_ms: 500 },
            },
            chunking: ChunkingConfig {
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: SizeUnit(100 * 1024 * 1024), // 100MB
            },
            performance: ProcessingPerformanceConfig {
                max_concurrent_tasks: 4,
                batch_size: 100,
            },
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            modes: SearchModesConfig {
                project: SearchModeConfig {
                    enabled: true,
                    default_mode: true,
                },
                collection: SearchModeConfig {
                    enabled: true,
                    default_mode: false,
                },
                global: SearchModeConfig {
                    enabled: true,
                    default_mode: false,
                },
                all: SearchModeConfig {
                    enabled: true,
                    default_mode: false,
                },
            },
            hybrid: HybridSearchConfig {
                enabled: true,
                fusion_algorithm: "rrf".to_string(),
                dense_weight: 0.7,
                sparse_weight: 0.3,
            },
            results: SearchResultsConfig {
                default_limit: 20,
                max_limit: 100,
                include_metadata: true,
                include_snippets: true,
            },
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            memory: MemoryPerformanceConfig {
                max_rss_mb: 500,
                warning_threshold_mb: 400,
                gc_threshold_mb: 350,
            },
            startup: StartupConfig {
                daemon_init_timeout_seconds: 2,
                mcp_server_init_timeout_seconds: 5,
                health_check_timeout_seconds: 1,
            },
            cpu: CpuConfig {
                background_priority: "low".to_string(),
                interactive_priority: "high".to_string(),
                priority_boost_on_mcp_active: true,
                revert_priority_on_mcp_quit: true,
            },
        }
    }
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            xdg_compliance: XdgComplianceConfig {
                enabled: true,
                fallback_on_windows_mac: true,
            },
            directories: PlatformDirectoriesConfig {
                linux: PlatformDirectorySet {
                    cache: "$XDG_CACHE_HOME/workspace-qdrant-mcp".to_string(),
                    state: "$XDG_STATE_HOME/workspace-qdrant-mcp".to_string(),
                    config: "$XDG_CONFIG_HOME/workspace-qdrant-mcp".to_string(),
                    logs: "$XDG_STATE_HOME/workspace-qdrant-mcp/logs".to_string(),
                },
                macos: PlatformDirectorySet {
                    cache: "~/Library/Caches/workspace-qdrant-mcp".to_string(),
                    logs: "~/Library/Logs/workspace-qdrant-mcp".to_string(),
                    config: "~/Library/Application Support/workspace-qdrant-mcp".to_string(),
                    state: "~/Library/Application Support/workspace-qdrant-mcp/state".to_string(),
                },
                windows: PlatformDirectorySet {
                    cache: "%LOCALAPPDATA%\\workspace-qdrant-mcp\\cache".to_string(),
                    logs: "%LOCALAPPDATA%\\workspace-qdrant-mcp\\logs".to_string(),
                    config: "%APPDATA%\\workspace-qdrant-mcp".to_string(),
                    state: "%LOCALAPPDATA%\\workspace-qdrant-mcp\\state".to_string(),
                },
            },
            file_patterns: FilePatternsConfig {
                include_patterns: IncludePatternsConfig { custom: vec![] },
                exclude_patterns: ExcludePatternsConfig { custom: vec![] },
            },
        }
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            behavior: CliBehaviorConfig {
                interactive_mode: true,
                default_output_format: "table".to_string(),
                color_output: true,
                progress_indicators: true,
            },
            commands: CliCommandsConfig {
                memory: CliMemoryConfig { auto_backup: true },
                admin: CliAdminConfig {
                    confirm_destructive: true,
                },
                ingest: CliIngestConfig { show_progress: true },
            },
        }
    }
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            server: GrpcServerConfig {
                enabled: true,
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_concurrent_streams: 100,
                max_message_size_mb: 16,
            },
            client: GrpcClientConfig {
                connection_timeout_seconds: 5,
                request_timeout_seconds: 30,
                keepalive_interval_seconds: 30,
                keepalive_timeout_seconds: 5,
                max_retry_attempts: 3,
                retry_backoff_ms: 1000,
            },
        }
    }
}

impl Default for ExternalServicesConfig {
    fn default() -> Self {
        Self {
            qdrant: QdrantServiceConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_seconds: 30,
                max_retries: 3,
            },
            embeddings: EmbeddingsConfig {
                model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                cache_dir: None,
            },
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
            metrics: MetricsConfig {
                enabled: true,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: true,
                prometheus_port: 9090,
            },
            health_checks: HealthChecksConfig {
                enabled: true,
                check_interval_seconds: 30,
                failure_threshold: 3,
            },
        }
    }
}

impl DaemonConfig {
    /// Load configuration from file or use defaults
    pub fn load(config_path: Option<&Path>) -> DaemonResult<Self> {
        match config_path {
            Some(path) => {
                let content = std::fs::read_to_string(path)?;
                let config: DaemonConfig = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                    // Parse as TOML
                    toml::from_str(&content)
                        .map_err(|e| crate::error::DaemonError::Configuration {
                            message: format!("Invalid TOML: {}", e)
                        })?
                } else {
                    // Parse as YAML (default)
                    serde_yaml::from_str(&content)
                        .map_err(|e| crate::error::DaemonError::Configuration {
                            message: format!("Invalid YAML: {}", e)
                        })?
                };
                Ok(config)
            },
            None => {
                // Try to load from environment variables or use defaults
                Self::from_env()
            }
        }
    }

    /// Load configuration from environment variables
    fn from_env() -> DaemonResult<Self> {
        let mut config = Self::default();

        // Override with environment variables if present
        if let Ok(url) = std::env::var("QDRANT_URL") {
            config.external_services.qdrant.url = url;
        }

        if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
            config.external_services.qdrant.api_key = Some(api_key);
        }

        if let Ok(host) = std::env::var("DAEMON_HOST") {
            config.grpc.server.host = host;
        }

        if let Ok(port) = std::env::var("DAEMON_PORT") {
            config.grpc.server.port = port.parse()
                .map_err(|e| crate::error::DaemonError::Configuration {
                    message: format!("Invalid port: {}", e)
                })?;
        }

        Ok(config)
    }

    /// Save configuration to file
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> DaemonResult<()> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| crate::error::DaemonError::Configuration {
                message: format!("Serialization error: {}", e)
            })?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> DaemonResult<()> {
        // Validate gRPC server configuration
        if self.grpc.server.port == 0 {
            return Err(crate::error::DaemonError::Configuration {
                message: "gRPC server port cannot be 0".to_string()
            });
        }

        // Validate Qdrant URL
        if self.external_services.qdrant.url.is_empty() {
            return Err(crate::error::DaemonError::Configuration {
                message: "Qdrant URL cannot be empty".to_string()
            });
        }

        // Validate chunking configuration
        if self.document_processing.chunking.default_chunk_size == 0 {
            return Err(crate::error::DaemonError::Configuration {
                message: "Chunk size must be greater than 0".to_string()
            });
        }

        Ok(())
    }

    /// Compatibility property getters
    pub fn server(&self) -> ServerConfig {
        self.get_legacy_server_config()
    }

    pub fn database(&self) -> DatabaseConfig {
        self.get_legacy_database_config()
    }

    pub fn qdrant(&self) -> QdrantConfig {
        self.get_legacy_qdrant_config()
    }

    pub fn processing(&self) -> ProcessingConfig {
        self.get_legacy_processing_config()
    }

    pub fn file_watcher(&self) -> FileWatcherConfig {
        self.get_legacy_file_watcher_config()
    }

    pub fn metrics(&self) -> MetricsConfig {
        self.get_legacy_metrics_config()
    }

    pub fn logging(&self) -> LoggingConfig {
        self.get_legacy_logging_config()
    }

    pub fn auto_ingestion(&self) -> AutoIngestionConfig {
        self.get_legacy_auto_ingestion_config()
    }

    /// Get legacy server configuration for compatibility
    pub fn get_legacy_server_config(&self) -> ServerConfig {
        ServerConfig {
            host: self.grpc.server.host.clone(),
            port: self.grpc.server.port,
            max_connections: 1000,
            connection_timeout_secs: self.grpc.client.connection_timeout_seconds,
            request_timeout_secs: self.grpc.client.request_timeout_seconds,
            enable_tls: false,
            security: SecurityConfig::default(),
            transport: TransportConfig::default(),
            message: MessageConfig::default(),
            compression: CompressionConfig::default(),
            streaming: StreamingConfig::default(),
        }
    }

    /// Get legacy database configuration for compatibility
    pub fn get_legacy_database_config(&self) -> DatabaseConfig {
        DatabaseConfig {
            sqlite_path: format!("{}/workspace_daemon.db",
                self.platform.directories.linux.state.replace("$XDG_STATE_HOME", "~/.local/state")),
            max_connections: 10,
            connection_timeout_secs: 30,
            enable_wal: true,
        }
    }

    /// Get legacy Qdrant configuration for compatibility
    pub fn get_legacy_qdrant_config(&self) -> QdrantConfig {
        QdrantConfig {
            url: self.external_services.qdrant.url.clone(),
            api_key: self.external_services.qdrant.api_key.clone(),
            timeout_secs: self.external_services.qdrant.timeout_seconds,
            max_retries: self.external_services.qdrant.max_retries as u32,
            default_collection: CollectionConfig {
                vector_size: 384, // Default for all-MiniLM-L6-v2
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        }
    }

    /// Get legacy processing configuration for compatibility
    pub fn get_legacy_processing_config(&self) -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: self.document_processing.performance.max_concurrent_tasks as usize,
            default_chunk_size: self.document_processing.chunking.default_chunk_size as usize,
            default_chunk_overlap: self.document_processing.chunking.default_chunk_overlap as usize,
            max_file_size_bytes: self.document_processing.chunking.max_file_size_bytes.0,
            supported_extensions: vec![
                "rs".to_string(), "py".to_string(), "js".to_string(), "ts".to_string(),
                "md".to_string(), "txt".to_string(), "pdf".to_string(), "html".to_string(),
                "json".to_string(), "xml".to_string(),
            ],
            enable_lsp: self.lsp_integration.enabled,
            lsp_timeout_secs: 10,
        }
    }

    /// Get legacy file watcher configuration for compatibility
    pub fn get_legacy_file_watcher_config(&self) -> FileWatcherConfig {
        FileWatcherConfig {
            enabled: self.document_processing.file_watching.enabled,
            debounce_ms: self.document_processing.file_watching.debouncing.delay_ms,
            max_watched_dirs: 100,
            ignore_patterns: vec![
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                "*.tmp".to_string(),
                "*.log".to_string(),
            ],
            recursive: true,
        }
    }

    /// Get legacy metrics configuration for compatibility
    pub fn get_legacy_metrics_config(&self) -> MetricsConfig {
        MetricsConfig {
            enabled: self.monitoring.metrics.enabled,
            collection_interval_secs: self.monitoring.metrics.collection_interval_secs,
            retention_days: self.monitoring.metrics.retention_days as u32,
            enable_prometheus: true,
            prometheus_port: 9090,
        }
    }

    /// Get legacy logging configuration for compatibility
    pub fn get_legacy_logging_config(&self) -> LoggingConfig {
        LoggingConfig {
            level: self.monitoring.logging.level.clone(),
            file_path: Some(format!("{}/workspace_daemon.log",
                self.platform.directories.linux.logs.replace("$XDG_STATE_HOME", "~/.local/state"))),
            json_format: self.monitoring.logging.json_format,
            max_file_size_mb: self.monitoring.logging.max_file_size_mb as u64,
            max_files: self.monitoring.logging.max_files as u32,
        }
    }

    /// Get legacy auto ingestion configuration for compatibility
    pub fn get_legacy_auto_ingestion_config(&self) -> AutoIngestionConfig {
        AutoIngestionConfig {
            enabled: self.document_processing.file_watching.enabled,
            auto_create_watches: true,
            project_path: None,
            target_collection_suffix: self.collections.root_name.clone(),
            include_source_files: self.document_processing.supported_types.code.enabled,
            include_common_files: self.document_processing.supported_types.text.enabled,
            include_patterns: vec![
                "*.rs".to_string(),
                "*.py".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.md".to_string(),
                "*.txt".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            exclude_patterns: vec![
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                "build/**".to_string(),
                "dist/**".to_string(),
                "*.log".to_string(),
                "*.tmp".to_string(),
                "*.lock".to_string(),
            ],
            recursive: true,
            max_depth: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_unit_parsing() {
        assert_eq!("100B".parse::<SizeUnit>().unwrap().0, 100);
        assert_eq!("1KB".parse::<SizeUnit>().unwrap().0, 1024);
        assert_eq!("1MB".parse::<SizeUnit>().unwrap().0, 1024 * 1024);
        assert_eq!("1GB".parse::<SizeUnit>().unwrap().0, 1024 * 1024 * 1024);
        assert_eq!("1TB".parse::<SizeUnit>().unwrap().0, 1024_u64.pow(4));
        assert_eq!("100".parse::<SizeUnit>().unwrap().0, 100);
    }

    #[test]
    fn test_time_unit_parsing() {
        assert_eq!("100ms".parse::<TimeUnit>().unwrap().0, 100);
        assert_eq!("1s".parse::<TimeUnit>().unwrap().0, 1000);
        assert_eq!("1m".parse::<TimeUnit>().unwrap().0, 60000);
        assert_eq!("1h".parse::<TimeUnit>().unwrap().0, 3600000);
        assert_eq!("10".parse::<TimeUnit>().unwrap().0, 10000); // Default to seconds
    }

    #[test]
    fn test_daemon_config_default() {
        let config = DaemonConfig::default();

        assert_eq!(config.system.project_name, "workspace-qdrant-mcp");
        assert_eq!(config.system.version, "v2.0");
        assert!(config.system.components.rust_daemon.enabled);
        assert!(config.system.components.python_mcp_server.enabled);
        assert!(config.system.components.cli_utility.enabled);
        assert!(config.system.components.context_injector.enabled);

        assert_eq!(config.memory.collection_name, "llm_rules");
        assert!(config.memory.authority_levels.absolute.enabled);
        assert!(config.memory.authority_levels.default.enabled);

        assert_eq!(config.collections.root_name, "project");
        assert!(config.collections.types.is_empty());
        assert!(config.collections.project_content.enabled);
        assert_eq!(config.collections.project_content.name, "project_content");

        assert!(config.project_detection.git_detection.enabled);
        assert_eq!(config.project_detection.git_detection.priority, 1);
        assert!(config.project_detection.git_detection.require_initialized);

        assert!(config.lsp_integration.enabled);
        assert!(config.lsp_integration.auto_detection);
        assert!(!config.lsp_integration.auto_installation);

        assert!(config.document_processing.supported_types.text.enabled);
        assert!(config.document_processing.supported_types.pdf.enabled);
        assert!(config.document_processing.supported_types.code.enabled);
        assert!(config.document_processing.file_watching.enabled);

        assert!(config.search.modes.project.enabled);
        assert!(config.search.modes.project.default_mode);
        assert!(config.search.hybrid.enabled);
        assert_eq!(config.search.hybrid.fusion_algorithm, "rrf");

        assert_eq!(config.performance.memory.max_rss_mb, 500);
        assert_eq!(config.performance.startup.daemon_init_timeout_seconds, 2);

        assert!(config.platform.xdg_compliance.enabled);
        assert!(config.platform.xdg_compliance.fallback_on_windows_mac);

        assert!(config.cli.behavior.interactive_mode);
        assert_eq!(config.cli.behavior.default_output_format, "table");

        assert!(config.grpc.server.enabled);
        assert_eq!(config.grpc.server.host, "127.0.0.1");
        assert_eq!(config.grpc.server.port, 50051);

        assert_eq!(config.external_services.qdrant.url, "http://localhost:6333");
        assert!(config.external_services.qdrant.api_key.is_none());
        assert_eq!(config.external_services.embeddings.model, "sentence-transformers/all-MiniLM-L6-v2");

        assert_eq!(config.monitoring.logging.level, "info");
        assert!(!config.monitoring.logging.json_format);
        assert!(config.monitoring.metrics.enabled);
        assert!(config.monitoring.health_checks.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = DaemonConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid port
        config.grpc.server.port = 0;
        assert!(config.validate().is_err());
        config.grpc.server.port = 50051; // Reset

        // Test empty Qdrant URL
        config.external_services.qdrant.url = String::new();
        assert!(config.validate().is_err());
        config.external_services.qdrant.url = "http://localhost:6333".to_string(); // Reset

        // Test zero chunk size
        config.document_processing.chunking.default_chunk_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_load_config_from_env() {
        // This would test environment variable loading
        // In a real test, we'd set up environment variables first
        let config = DaemonConfig::from_env().unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_size_unit_display() {
        assert_eq!(SizeUnit(100).to_string(), "100B");
        assert_eq!(SizeUnit(1024).to_string(), "1KB");
        assert_eq!(SizeUnit(1024 * 1024).to_string(), "1MB");
        assert_eq!(SizeUnit(1024 * 1024 * 1024).to_string(), "1GB");
        assert_eq!(SizeUnit(1024_u64.pow(4)).to_string(), "1TB");
    }

    #[test]
    fn test_time_unit_display() {
        assert_eq!(TimeUnit(100).to_string(), "100ms");
        assert_eq!(TimeUnit(1000).to_string(), "1s");
        assert_eq!(TimeUnit(60000).to_string(), "1m");
        assert_eq!(TimeUnit(3600000).to_string(), "1h");
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = DaemonConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: DaemonConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.system.project_name, deserialized.system.project_name);
        assert_eq!(config.grpc.server.host, deserialized.grpc.server.host);
    }
}

// =============================================================================
// LEGACY COMPATIBILITY STRUCTURES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub connection_timeout_secs: u64,
    pub request_timeout_secs: u64,
    pub enable_tls: bool,
    pub security: SecurityConfig,
    pub transport: TransportConfig,
    pub message: MessageConfig,
    pub compression: CompressionConfig,
    pub streaming: StreamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub sqlite_path: String,
    pub max_connections: u32,
    pub connection_timeout_secs: u64,
    pub enable_wal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub default_collection: CollectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub vector_size: usize,
    pub distance_metric: String,
    pub enable_indexing: bool,
    pub replication_factor: u32,
    pub shard_number: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub max_concurrent_tasks: usize,
    pub default_chunk_size: usize,
    pub default_chunk_overlap: usize,
    pub max_file_size_bytes: u64,
    pub supported_extensions: Vec<String>,
    pub enable_lsp: bool,
    pub lsp_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWatcherConfig {
    pub enabled: bool,
    pub debounce_ms: u64,
    pub max_watched_dirs: usize,
    pub ignore_patterns: Vec<String>,
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub collection_interval_secs: u64,
    pub retention_days: u32,
    pub enable_prometheus: bool,
    pub prometheus_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file_path: Option<String>,
    pub json_format: bool,
    pub max_file_size_mb: u64,
    pub max_files: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIngestionConfig {
    pub enabled: bool,
    pub auto_create_watches: bool,
    pub project_path: Option<String>,
    pub target_collection_suffix: String,
    pub include_source_files: bool,
    pub include_common_files: bool,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub recursive: bool,
    pub max_depth: u32,
}

/// Message size and validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageConfig {
    /// Maximum message size for incoming requests (bytes)
    pub max_incoming_message_size: usize,

    /// Maximum message size for outgoing responses (bytes)
    pub max_outgoing_message_size: usize,

    /// Enable message size validation
    pub enable_size_validation: bool,

    /// Maximum frame size for HTTP/2
    pub max_frame_size: u32,

    /// Initial window size for HTTP/2
    pub initial_window_size: u32,

    /// Service-specific message size limits
    pub service_limits: ServiceMessageLimits,

    /// Size monitoring and alerting configuration
    pub monitoring: MessageMonitoringConfig,
}

/// Service-specific message size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMessageLimits {
    /// Document processor service limits
    pub document_processor: ServiceLimit,

    /// Search service limits
    pub search_service: ServiceLimit,

    /// Memory service limits
    pub memory_service: ServiceLimit,

    /// System service limits
    pub system_service: ServiceLimit,
}

/// Individual service message limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLimit {
    /// Maximum incoming message size (bytes)
    pub max_incoming: usize,

    /// Maximum outgoing message size (bytes)
    pub max_outgoing: usize,

    /// Enable size validation for this service
    pub enable_validation: bool,
}

/// Message monitoring and alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMonitoringConfig {
    /// Enable detailed message size monitoring
    pub enable_detailed_monitoring: bool,

    /// Alert threshold for oversized messages (percentage of limit)
    pub oversized_alert_threshold: f64,

    /// Enable real-time metrics collection
    pub enable_realtime_metrics: bool,

    /// Metrics collection interval (seconds)
    pub metrics_interval_secs: u64,
}

/// Compression configuration for gRPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable gzip compression
    pub enable_gzip: bool,

    /// Minimum message size to trigger compression (bytes)
    pub compression_threshold: usize,

    /// Compression level (1-9, where 9 is maximum compression)
    pub compression_level: u32,

    /// Enable compression for streaming responses
    pub enable_streaming_compression: bool,

    /// Monitor compression efficiency
    pub enable_compression_monitoring: bool,

    /// Adaptive compression configuration
    pub adaptive: AdaptiveCompressionConfig,

    /// Compression performance monitoring
    pub performance: CompressionPerformanceConfig,
}

/// Adaptive compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompressionConfig {
    /// Enable adaptive compression based on content type
    pub enable_adaptive: bool,

    /// Text content compression level (1-9)
    pub text_compression_level: u32,

    /// Binary content compression level (1-9)
    pub binary_compression_level: u32,

    /// JSON/structured data compression level (1-9)
    pub structured_compression_level: u32,

    /// Maximum time to spend on compression (milliseconds)
    pub max_compression_time_ms: u64,
}

/// Compression performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPerformanceConfig {
    /// Enable compression ratio tracking
    pub enable_ratio_tracking: bool,

    /// Alert threshold for poor compression ratio
    pub poor_ratio_threshold: f64,

    /// Enable compression time monitoring
    pub enable_time_monitoring: bool,

    /// Alert threshold for slow compression (milliseconds)
    pub slow_compression_threshold_ms: u64,

    /// Enable compression failure alerting
    pub enable_failure_alerting: bool,
}

/// Streaming configuration for large operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable server-side streaming
    pub enable_server_streaming: bool,

    /// Enable client-side streaming
    pub enable_client_streaming: bool,

    /// Maximum concurrent streams per connection
    pub max_concurrent_streams: u32,

    /// Stream buffer size (number of items)
    pub stream_buffer_size: usize,

    /// Stream timeout in seconds
    pub stream_timeout_secs: u64,

    /// Enable stream flow control
    pub enable_flow_control: bool,

    /// Progress tracking configuration
    pub progress: StreamProgressConfig,

    /// Stream health and recovery configuration
    pub health: StreamHealthConfig,

    /// Large operation streaming configuration
    pub large_operations: LargeOperationStreamConfig,
}

/// Stream progress tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgressConfig {
    /// Enable progress tracking for streams
    pub enable_progress_tracking: bool,

    /// Progress update interval (milliseconds)
    pub progress_update_interval_ms: u64,

    /// Enable progress callbacks
    pub enable_progress_callbacks: bool,

    /// Minimum operation size to enable progress tracking (bytes)
    pub progress_threshold: usize,
}

/// Stream health monitoring and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamHealthConfig {
    /// Enable stream health monitoring
    pub enable_health_monitoring: bool,

    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,

    /// Enable automatic stream recovery
    pub enable_auto_recovery: bool,

    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,

    /// Recovery backoff multiplier
    pub recovery_backoff_multiplier: f64,

    /// Initial recovery delay (milliseconds)
    pub initial_recovery_delay_ms: u64,
}

/// Large operation streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeOperationStreamConfig {
    /// Enable streaming for large document uploads
    pub enable_large_document_streaming: bool,

    /// Chunk size for large operations (bytes)
    pub large_operation_chunk_size: usize,

    /// Enable streaming for bulk operations
    pub enable_bulk_streaming: bool,

    /// Maximum memory usage for streaming operations (bytes)
    pub max_streaming_memory: usize,

    /// Enable bidirectional streaming optimization
    pub enable_bidirectional_optimization: bool,
}

/// Security configuration for gRPC server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// TLS configuration
    pub tls: TlsConfig,

    /// Authentication configuration
    pub auth: AuthConfig,

    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,

    /// Security audit configuration
    pub audit: SecurityAuditConfig,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Server certificate file path
    pub cert_file: Option<String>,

    /// Server private key file path
    pub key_file: Option<String>,

    /// CA certificate file path for client verification
    pub ca_cert_file: Option<String>,

    /// Enable mutual TLS (mTLS)
    pub enable_mtls: bool,

    /// Client certificate verification mode
    pub client_cert_verification: ClientCertVerification,

    /// TLS protocol versions to support
    pub supported_protocols: Vec<String>,

    /// Cipher suites to use
    pub cipher_suites: Vec<String>,
}

/// Client certificate verification modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCertVerification {
    /// No client certificate required
    None,
    /// Client certificate optional
    Optional,
    /// Client certificate required
    Required,
}

/// Authentication and authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable service-to-service authentication
    pub enable_service_auth: bool,

    /// JWT token configuration
    pub jwt: JwtConfig,

    /// API key configuration
    pub api_key: ApiKeyConfig,

    /// Service authorization rules
    pub authorization: AuthorizationConfig,
}

/// JWT token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// JWT signing secret or public key file path
    pub secret_or_key_file: String,

    /// Token issuer
    pub issuer: String,

    /// Token audience
    pub audience: String,

    /// Token expiration time in seconds
    pub expiration_secs: u64,

    /// Algorithm to use (HS256, RS256, etc.)
    pub algorithm: String,
}

/// API key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Enable API key authentication
    pub enabled: bool,

    /// API key header name
    pub header_name: String,

    /// Valid API keys (in production, load from secure storage)
    pub valid_keys: Vec<String>,

    /// API key permissions mapping
    pub key_permissions: std::collections::HashMap<String, Vec<String>>,
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    /// Enable authorization checks
    pub enabled: bool,

    /// Default permissions for authenticated users
    pub default_permissions: Vec<String>,

    /// Service-specific permissions
    pub service_permissions: ServicePermissions,
}

/// Service-specific permission configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePermissions {
    /// Document processor permissions
    pub document_processor: Vec<String>,

    /// Search service permissions
    pub search_service: Vec<String>,

    /// Memory service permissions
    pub memory_service: Vec<String>,

    /// System service permissions
    pub system_service: Vec<String>,
}

/// Enhanced rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,

    /// Requests per second per client
    pub requests_per_second: u32,

    /// Burst capacity
    pub burst_capacity: u32,

    /// Connection pool limits per service
    pub connection_pool_limits: ConnectionPoolLimits,

    /// Request queue depth limits
    pub queue_depth_limit: u32,

    /// Memory usage protection
    pub memory_protection: MemoryProtectionConfig,

    /// Resource exhaustion protection
    pub resource_protection: ResourceProtectionConfig,
}

/// Connection pool limits per service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolLimits {
    /// Document processor connection limit
    pub document_processor: u32,

    /// Search service connection limit
    pub search_service: u32,

    /// Memory service connection limit
    pub memory_service: u32,

    /// System service connection limit
    pub system_service: u32,
}

/// Memory usage protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProtectionConfig {
    /// Enable memory usage monitoring
    pub enabled: bool,

    /// Maximum memory usage per connection (bytes)
    pub max_memory_per_connection: u64,

    /// Total memory usage limit (bytes)
    pub total_memory_limit: u64,

    /// Memory cleanup interval (seconds)
    pub cleanup_interval_secs: u64,
}

/// Resource exhaustion protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProtectionConfig {
    /// Enable resource protection
    pub enabled: bool,

    /// CPU usage threshold for throttling (percentage)
    pub cpu_threshold_percent: f64,

    /// Disk space threshold for throttling (percentage)
    pub disk_threshold_percent: f64,

    /// Circuit breaker failure threshold
    pub circuit_breaker_failure_threshold: u32,

    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout_secs: u64,
}

/// Security audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditConfig {
    /// Enable security audit logging
    pub enabled: bool,

    /// Audit log file path
    pub log_file_path: String,

    /// Log authentication events
    pub log_auth_events: bool,

    /// Log authorization failures
    pub log_auth_failures: bool,

    /// Log rate limiting events
    pub log_rate_limit_events: bool,

    /// Log suspicious patterns
    pub log_suspicious_patterns: bool,

    /// Audit log rotation configuration
    pub rotation: AuditLogRotation,
}

/// Audit log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogRotation {
    /// Maximum file size before rotation (MB)
    pub max_file_size_mb: u64,

    /// Maximum number of files to keep
    pub max_files: u32,

    /// Compress rotated files
    pub compress: bool,
}

/// Transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// Unix domain socket configuration
    pub unix_socket: UnixSocketConfig,

    /// Local communication optimizations
    pub local_optimization: LocalOptimizationConfig,

    /// Transport selection strategy
    pub transport_strategy: TransportStrategy,
}

/// Unix domain socket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnixSocketConfig {
    /// Enable Unix domain socket support
    pub enabled: bool,

    /// Unix socket file path
    pub socket_path: String,

    /// Socket file permissions (octal)
    pub permissions: u32,

    /// Enable Unix socket for local development
    pub prefer_for_local: bool,
}

/// Local communication optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalOptimizationConfig {
    /// Enable local transport optimizations
    pub enabled: bool,

    /// Use larger buffers for local communication
    pub use_large_buffers: bool,

    /// Buffer size for local communication (bytes)
    pub local_buffer_size: usize,

    /// Enable memory-efficient serialization
    pub memory_efficient_serialization: bool,

    /// Reduced latency settings for local calls
    pub reduce_latency: LocalLatencyConfig,
}

/// Local latency reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLatencyConfig {
    /// Disable Nagle's algorithm for local connections
    pub disable_nagle: bool,

    /// Use custom local connection pooling
    pub custom_connection_pooling: bool,

    /// Local connection pool size
    pub connection_pool_size: u32,

    /// Local connection keep-alive interval (seconds)
    pub keepalive_interval_secs: u64,
}

/// Transport selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportStrategy {
    /// Automatically detect and use best transport
    Auto,
    /// Force TCP transport
    ForceTcp,
    /// Force Unix socket (local only)
    ForceUnixSocket,
    /// Use Unix socket with TCP fallback
    UnixSocketWithTcpFallback,
}

// Default implementations for legacy structures
impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tls: TlsConfig::default(),
            auth: AuthConfig::default(),
            rate_limiting: RateLimitConfig::default(),
            audit: SecurityAuditConfig::default(),
        }
    }
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_file: None,
            key_file: None,
            ca_cert_file: None,
            enable_mtls: false,
            client_cert_verification: ClientCertVerification::None,
            supported_protocols: vec!["TLSv1.2".to_string(), "TLSv1.3".to_string()],
            cipher_suites: vec![], // Use system defaults
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enable_service_auth: false,
            jwt: JwtConfig::default(),
            api_key: ApiKeyConfig::default(),
            authorization: AuthorizationConfig::default(),
        }
    }
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret_or_key_file: "changeme_jwt_secret".to_string(),
            issuer: "workspace-qdrant-mcp".to_string(),
            audience: "workspace-qdrant-clients".to_string(),
            expiration_secs: 3600, // 1 hour
            algorithm: "HS256".to_string(),
        }
    }
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            header_name: "X-API-Key".to_string(),
            valid_keys: vec![],
            key_permissions: std::collections::HashMap::new(),
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_permissions: vec![
                "read".to_string(),
                "write".to_string(),
            ],
            service_permissions: ServicePermissions::default(),
        }
    }
}

impl Default for ServicePermissions {
    fn default() -> Self {
        Self {
            document_processor: vec!["process".to_string(), "read".to_string()],
            search_service: vec!["search".to_string(), "read".to_string()],
            memory_service: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
            system_service: vec!["admin".to_string(), "read".to_string()],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_capacity: 200,
            connection_pool_limits: ConnectionPoolLimits::default(),
            queue_depth_limit: 1000,
            memory_protection: MemoryProtectionConfig::default(),
            resource_protection: ResourceProtectionConfig::default(),
        }
    }
}

impl Default for ConnectionPoolLimits {
    fn default() -> Self {
        Self {
            document_processor: 50,
            search_service: 100,
            memory_service: 75,
            system_service: 25,
        }
    }
}

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_per_connection: 100 * 1024 * 1024, // 100MB
            total_memory_limit: 1024 * 1024 * 1024, // 1GB
            cleanup_interval_secs: 300, // 5 minutes
        }
    }
}

impl Default for ResourceProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cpu_threshold_percent: 80.0,
            disk_threshold_percent: 90.0,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout_secs: 60,
        }
    }
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_file_path: "./security_audit.log".to_string(),
            log_auth_events: true,
            log_auth_failures: true,
            log_rate_limit_events: true,
            log_suspicious_patterns: true,
            rotation: AuditLogRotation::default(),
        }
    }
}

impl Default for AuditLogRotation {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            max_files: 10,
            compress: true,
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            unix_socket: UnixSocketConfig::default(),
            local_optimization: LocalOptimizationConfig::default(),
            transport_strategy: TransportStrategy::Auto,
        }
    }
}

impl Default for UnixSocketConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            socket_path: "/tmp/workspace-qdrant-mcp.sock".to_string(),
            permissions: 0o600, // Owner read/write only
            prefer_for_local: true,
        }
    }
}

impl Default for LocalOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_large_buffers: true,
            local_buffer_size: 64 * 1024, // 64KB
            memory_efficient_serialization: true,
            reduce_latency: LocalLatencyConfig::default(),
        }
    }
}

impl Default for LocalLatencyConfig {
    fn default() -> Self {
        Self {
            disable_nagle: true,
            custom_connection_pooling: true,
            connection_pool_size: 10,
            keepalive_interval_secs: 30,
        }
    }
}

impl Default for MessageConfig {
    fn default() -> Self {
        Self {
            // 16MB default limit (existing baseline mentioned in requirements)
            max_incoming_message_size: 16 * 1024 * 1024,
            max_outgoing_message_size: 16 * 1024 * 1024,
            enable_size_validation: true,
            // 16KB frame size for HTTP/2
            max_frame_size: 16 * 1024,
            // 64KB initial window for HTTP/2
            initial_window_size: 64 * 1024,
            service_limits: ServiceMessageLimits::default(),
            monitoring: MessageMonitoringConfig::default(),
        }
    }
}

impl Default for ServiceMessageLimits {
    fn default() -> Self {
        Self {
            document_processor: ServiceLimit::default_document_processor(),
            search_service: ServiceLimit::default_search(),
            memory_service: ServiceLimit::default_memory(),
            system_service: ServiceLimit::default_system(),
        }
    }
}

impl ServiceLimit {
    fn default_document_processor() -> Self {
        Self {
            max_incoming: 64 * 1024 * 1024, // 64MB for large documents
            max_outgoing: 32 * 1024 * 1024, // 32MB for processed responses
            enable_validation: true,
        }
    }

    fn default_search() -> Self {
        Self {
            max_incoming: 4 * 1024 * 1024,  // 4MB for search queries
            max_outgoing: 16 * 1024 * 1024, // 16MB for search results
            enable_validation: true,
        }
    }

    fn default_memory() -> Self {
        Self {
            max_incoming: 8 * 1024 * 1024,  // 8MB for memory operations
            max_outgoing: 8 * 1024 * 1024,  // 8MB for memory responses
            enable_validation: true,
        }
    }

    fn default_system() -> Self {
        Self {
            max_incoming: 1024 * 1024,      // 1MB for system commands
            max_outgoing: 4 * 1024 * 1024,  // 4MB for system responses
            enable_validation: true,
        }
    }
}

impl Default for MessageMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_detailed_monitoring: true,
            oversized_alert_threshold: 0.8, // Alert at 80% of limit
            enable_realtime_metrics: true,
            metrics_interval_secs: 60,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_gzip: true,
            // Compress messages larger than 1KB
            compression_threshold: 1024,
            // Medium compression level (6) for balance of speed/size
            compression_level: 6,
            enable_streaming_compression: true,
            enable_compression_monitoring: true,
            adaptive: AdaptiveCompressionConfig::default(),
            performance: CompressionPerformanceConfig::default(),
        }
    }
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            enable_adaptive: true,
            text_compression_level: 9,   // High compression for text
            binary_compression_level: 3, // Low compression for binary
            structured_compression_level: 6, // Medium for JSON/structured
            max_compression_time_ms: 100,
        }
    }
}

impl Default for CompressionPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_ratio_tracking: true,
            poor_ratio_threshold: 0.9, // Alert if compression ratio > 90%
            enable_time_monitoring: true,
            slow_compression_threshold_ms: 200, // Alert if compression > 200ms
            enable_failure_alerting: true,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_server_streaming: true,
            enable_client_streaming: true,
            // 128 concurrent streams per connection
            max_concurrent_streams: 128,
            // Buffer 1000 items for streaming
            stream_buffer_size: 1000,
            // 5 minute stream timeout
            stream_timeout_secs: 300,
            enable_flow_control: true,
            progress: StreamProgressConfig::default(),
            health: StreamHealthConfig::default(),
            large_operations: LargeOperationStreamConfig::default(),
        }
    }
}

impl Default for StreamProgressConfig {
    fn default() -> Self {
        Self {
            enable_progress_tracking: true,
            progress_update_interval_ms: 1000, // 1 second updates
            enable_progress_callbacks: true,
            progress_threshold: 1024 * 1024, // 1MB minimum for progress tracking
        }
    }
}

impl Default for StreamHealthConfig {
    fn default() -> Self {
        Self {
            enable_health_monitoring: true,
            health_check_interval_secs: 30,
            enable_auto_recovery: true,
            max_recovery_attempts: 3,
            recovery_backoff_multiplier: 2.0,
            initial_recovery_delay_ms: 500,
        }
    }
}

impl Default for LargeOperationStreamConfig {
    fn default() -> Self {
        Self {
            enable_large_document_streaming: true,
            large_operation_chunk_size: 1024 * 1024, // 1MB chunks
            enable_bulk_streaming: true,
            max_streaming_memory: 128 * 1024 * 1024, // 128MB memory limit
            enable_bidirectional_optimization: true,
        }
    }
}