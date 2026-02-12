//! File Watching with SQLite Queue Integration
//!
//! This module provides Rust-based file watching that writes directly to the
//! unified_queue SQLite table per WORKSPACE_QDRANT_MCP.md specification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use git2::Repository;
use notify::{Event, EventKind, RecursiveMode, Watcher as NotifyWatcher};
use sqlx::{Row, SqlitePool};
use tokio::sync::{mpsc, Notify, RwLock, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use thiserror::Error;
use glob::Pattern;

use crate::queue_operations::{QueueManager, QueueError};
use crate::unified_queue_schema::{ItemType, QueueOperation as UnifiedOp, FilePayload};
use crate::file_classification::classify_file_type;
use crate::allowed_extensions::{AllowedExtensions, FileRoute};
use crate::patterns::exclusion::should_exclude_file;
use crate::project_disambiguation::ProjectIdCalculator;
use serde::{Deserialize, Serialize};

//
// ========== MULTI-TENANT TYPES ==========
//

/// Unified collection names for multi-tenant architecture (canonical names)
///
/// Re-exported from `wqm_common::constants` for backward compatibility.
pub use wqm_common::constants::COLLECTION_PROJECTS as UNIFIED_PROJECTS_COLLECTION;
pub use wqm_common::constants::COLLECTION_LIBRARIES as UNIFIED_LIBRARIES_COLLECTION;

/// Watch type distinguishing project vs library watches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchType {
    /// Project watch - files are routed to `projects` collection with project_id metadata
    Project,
    /// Library watch - files are routed to `libraries` collection with library_name metadata
    Library,
}

impl Default for WatchType {
    fn default() -> Self {
        WatchType::Project
    }
}

impl WatchType {
    pub fn as_str(&self) -> &'static str {
        match self {
            WatchType::Project => "project",
            WatchType::Library => "library",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "project" => Some(WatchType::Project),
            "library" => Some(WatchType::Library),
            _ => None,
        }
    }
}

//
// ========== PUBLIC UTILITY FUNCTIONS ==========
//

/// Calculate a unique tenant ID for a project root directory
///
/// This function implements the tenant ID calculation algorithm:
/// 1. Try to get git remote URL (prefer origin, fallback to upstream)
/// 2. If remote exists: Sanitize URL to create tenant ID
///    - Remove protocol (https://, git@, ssh://)
///    - Replace separators (/, ., :, @) with underscores
///    - Example: github.com/user/repo → github_com_user_repo
/// 3. If no remote: Use SHA256 hash of absolute path
///    - Hash first 16 chars: abc123def456789a
///    - Add prefix: path_abc123def456789a
///
/// # Arguments
/// * `project_root` - Path to the project root directory
///
/// # Returns
/// Unique tenant ID (project ID) string
///
/// Uses `ProjectIdCalculator` to generate consistent, unique project IDs:
/// - For git repositories: SHA256 hash of normalized remote URL (12 characters)
/// - For local projects: "local_" prefix + SHA256 hash of path (18 characters total)
///
/// # Disambiguation
///
/// When multiple clones of the same repository exist, they will get the same
/// project ID unless disambiguation is explicitly provided. For full disambiguation
/// support, use `ProjectIdCalculator` directly with disambiguation paths obtained
/// from the project registry.
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_core::calculate_tenant_id;
///
/// let tenant_id = calculate_tenant_id(Path::new("/path/to/repo"));
/// // Returns: "abc123def456" (12-char hash if git remote exists)
/// // Or: "local_abc123def456" (if no git remote)
/// ```
pub fn calculate_tenant_id(project_root: &Path) -> String {
    let calculator = ProjectIdCalculator::new();

    // Try to get git remote URL using git2
    let remote_url = if let Ok(repo) = Repository::open(project_root) {
        // Try origin first, then upstream, then any remote
        repo.find_remote("origin")
            .or_else(|_| repo.find_remote("upstream"))
            .ok()
            .and_then(|remote| remote.url().map(|url| url.to_string()))
    } else {
        None
    };

    // Calculate project ID using ProjectIdCalculator
    // Note: Disambiguation path is None here - full disambiguation requires
    // database lookup to find existing clones with same remote
    let project_id = calculator.calculate(
        project_root,
        remote_url.as_deref(),
        None, // No disambiguation path in basic calculation
    );

    debug!(
        "Generated project ID for {}: {} (remote: {:?})",
        project_root.display(),
        project_id,
        remote_url.as_ref().map(|u| ProjectIdCalculator::normalize_git_url(u))
    );

    project_id
}

/// Get the current Git branch name for a repository
///
/// This function detects the current Git branch for a directory within a Git repository.
/// It handles various edge cases gracefully by returning "main" as the default branch name.
///
/// # Arguments
/// * `project_root` - Path to a directory within a Git repository
///
/// # Returns
/// Branch name (e.g., "main", "feature/auth") or "main" for edge cases
///
/// # Edge Cases
/// - Non-Git directory: Returns "main"
/// - Git repo with no commits: Returns "main"
/// - Detached HEAD state: Returns "main"
/// - Permission errors: Returns "main"
/// - Any other Git error: Returns "main"
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_core::get_current_branch;
///
/// let branch = get_current_branch(Path::new("/path/to/repo"));
/// // Returns: "feature/new-api" or "main"
/// ```
pub fn get_current_branch(repo_path: &Path) -> String {
    const DEFAULT_BRANCH: &str = "main";

    // Try to open the Git repository
    let repo = match Repository::open(repo_path) {
        Ok(r) => r,
        Err(_) => {
            warn!(
                "Not a Git repository, defaulting to '{}': {}",
                DEFAULT_BRANCH,
                repo_path.display()
            );
            return DEFAULT_BRANCH.to_string();
        }
    };

    // Check if repository has any commits
    if repo.head().is_err() {
        warn!(
            "Git repository has no commits yet, defaulting to '{}': {}",
            DEFAULT_BRANCH,
            repo_path.display()
        );
        return DEFAULT_BRANCH.to_string();
    }

    // Get HEAD reference
    let head = match repo.head() {
        Ok(h) => h,
        Err(_) => {
            warn!(
                "Failed to read HEAD, defaulting to '{}': {}",
                DEFAULT_BRANCH,
                repo_path.display()
            );
            return DEFAULT_BRANCH.to_string();
        }
    };

    // Check if HEAD is detached
    if !head.is_branch() {
        warn!(
            "Git repository in detached HEAD state, defaulting to '{}': {}",
            DEFAULT_BRANCH,
            repo_path.display()
        );
        return DEFAULT_BRANCH.to_string();
    }

    // Get branch name
    match head.shorthand() {
        Some(branch_name) => {
            debug!(
                "Detected Git branch '{}' for repository at {}",
                branch_name,
                repo_path.display()
            );
            branch_name.to_string()
        }
        None => {
            warn!(
                "Failed to get branch name from HEAD, defaulting to '{}': {}",
                DEFAULT_BRANCH,
                repo_path.display()
            );
            DEFAULT_BRANCH.to_string()
        }
    }
}

//
// ========== ORIGINAL MODULE CONTENT ==========
//

/// File watching errors
#[derive(Error, Debug)]
pub enum WatchingQueueError {
    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Git error: {0}")]
    Git(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
}

pub type WatchingQueueResult<T> = Result<T, WatchingQueueError>;

/// Watch configuration from database
#[derive(Debug, Clone)]
pub struct WatchConfig {
    pub id: String,
    pub path: PathBuf,
    /// Legacy collection field - used as fallback if watch_type not set
    pub collection: String,
    pub patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
    pub recursive: bool,
    pub debounce_ms: u64,
    pub enabled: bool,
    /// Watch type: Project or Library (determines target collection)
    pub watch_type: WatchType,
    /// Library name (required for Library watch type)
    pub library_name: Option<String>,
}

/// Compiled patterns for efficient matching
#[derive(Debug)]
struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl CompiledPatterns {
    fn new(config: &WatchConfig) -> Result<Self, WatchingQueueError> {
        let include = config.patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WatchingQueueError::Config {
                message: format!("Invalid include pattern: {}", e)
            })?;

        let exclude = config.ignore_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WatchingQueueError::Config {
                message: format!("Invalid exclude pattern: {}", e)
            })?;

        Ok(Self { include, exclude })
    }

    fn should_process(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check exclude patterns first
        for pattern in &self.exclude {
            if pattern.matches(&path_str) {
                return false;
            }
        }

        // If no include patterns, allow all
        if self.include.is_empty() {
            return true;
        }

        // Check include patterns
        for pattern in &self.include {
            if pattern.matches(&path_str) {
                return true;
            }
        }

        false
    }
}

/// File event with metadata
#[derive(Debug, Clone)]
struct FileEvent {
    path: PathBuf,
    event_kind: EventKind,
    timestamp: SystemTime,
}

/// Event debouncer to prevent duplicate processing
#[derive(Debug)]
struct EventDebouncer {
    events: HashMap<PathBuf, FileEvent>,
    debounce_duration: Duration,
}

impl EventDebouncer {
    fn new(debounce_ms: u64) -> Self {
        Self {
            events: HashMap::new(),
            debounce_duration: Duration::from_millis(debounce_ms),
        }
    }

    /// Add event, returns true if should process immediately
    fn add_event(&mut self, event: FileEvent) -> bool {
        let now = SystemTime::now();

        if let Some(existing) = self.events.get(&event.path) {
            if let Ok(elapsed) = now.duration_since(existing.timestamp) {
                if elapsed < self.debounce_duration {
                    self.events.insert(event.path.clone(), event);
                    return false;
                }
            }
        }

        self.events.insert(event.path.clone(), event);
        true
    }

    /// Get events ready to process
    fn get_ready_events(&mut self) -> Vec<FileEvent> {
        let now = SystemTime::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();

        for (path, event) in &self.events {
            if let Ok(elapsed) = now.duration_since(event.timestamp) {
                if elapsed >= self.debounce_duration {
                    ready.push(event.clone());
                    to_remove.push(path.clone());
                }
            }
        }

        for path in to_remove {
            self.events.remove(&path);
        }

        ready
    }
}

/// Main file watcher with queue integration
pub struct FileWatcherQueue {
    config: Arc<RwLock<WatchConfig>>,
    patterns: Arc<RwLock<CompiledPatterns>>,
    queue_manager: Arc<QueueManager>,
    allowed_extensions: Arc<AllowedExtensions>,
    debouncer: Arc<Mutex<EventDebouncer>>,
    watcher: Arc<Mutex<Option<Box<dyn NotifyWatcher + Send + Sync>>>>,
    event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
    processor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    // Error tracking (Task 461.5)
    error_tracker: Arc<WatchErrorTracker>,

    // Queue depth throttling (Task 461.8)
    throttle_state: Arc<QueueThrottleState>,

    // Statistics
    events_received: Arc<Mutex<u64>>,
    events_processed: Arc<Mutex<u64>>,
    events_filtered: Arc<Mutex<u64>>,
    queue_errors: Arc<Mutex<u64>>,
    events_throttled: Arc<Mutex<u64>>,
}

impl FileWatcherQueue {
    /// Create a new file watcher with queue integration
    pub fn new(
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
        allowed_extensions: Arc<AllowedExtensions>,
    ) -> WatchingQueueResult<Self> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms);
        let error_tracker = WatchErrorTracker::new();
        let throttle_state = QueueThrottleState::new();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            patterns: Arc::new(RwLock::new(patterns)),
            queue_manager,
            allowed_extensions,
            debouncer: Arc::new(Mutex::new(debouncer)),
            watcher: Arc::new(Mutex::new(None)),
            event_receiver: Arc::new(Mutex::new(None)),
            processor_handle: Arc::new(Mutex::new(None)),
            error_tracker: Arc::new(error_tracker),
            throttle_state: Arc::new(throttle_state),
            events_received: Arc::new(Mutex::new(0)),
            events_processed: Arc::new(Mutex::new(0)),
            events_filtered: Arc::new(Mutex::new(0)),
            queue_errors: Arc::new(Mutex::new(0)),
            events_throttled: Arc::new(Mutex::new(0)),
        })
    }

    /// Start watching the configured path
    pub async fn start(&self) -> WatchingQueueResult<()> {
        let config = self.config.read().await;

        // Validate path exists
        if !config.path.exists() {
            return Err(WatchingQueueError::Config {
                message: format!("Path does not exist: {}", config.path.display()),
            });
        }

        // Create file system watcher
        let (tx, rx) = mpsc::unbounded_channel();
        let tx_clone = tx.clone();

        let watcher: Box<dyn NotifyWatcher + Send + Sync> = Box::new(
            notify::RecommendedWatcher::new(
                move |result| {
                    if let Ok(event) = result {
                        Self::handle_notify_event(event, &tx_clone);
                    }
                },
                notify::Config::default(),
            )?
        );

        // Add path to watcher
        let recursive_mode = if config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = Some(watcher);
            if let Some(ref mut w) = *watcher_lock {
                w.watch(&config.path, recursive_mode)?;
            }
        }

        // Store receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = Some(rx);
        }

        // Start event processor
        self.start_event_processor().await?;

        info!("Started file watcher: {} (recursive: {})",
            config.path.display(), config.recursive);

        Ok(())
    }

    /// Stop watching
    pub async fn stop(&self) -> WatchingQueueResult<()> {
        // Stop processor
        {
            let mut handle_lock = self.processor_handle.lock().await;
            if let Some(handle) = handle_lock.take() {
                handle.abort();
            }
        }

        // Clear watcher
        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = None;
        }

        // Clear receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = None;
        }

        info!("Stopped file watcher");
        Ok(())
    }

    /// Handle notify event
    fn handle_notify_event(event: Event, tx: &mpsc::UnboundedSender<FileEvent>) {
        let timestamp = SystemTime::now();

        for path in event.paths {
            let file_event = FileEvent {
                path,
                event_kind: event.kind,
                timestamp,
            };

            if let Err(e) = tx.send(file_event) {
                error!("Failed to send file event: {}", e);
            }
        }
    }

    /// Start event processing task
    async fn start_event_processor(&self) -> WatchingQueueResult<()> {
        {
            let handle_lock = self.processor_handle.lock().await;
            if handle_lock.is_some() {
                return Ok(());
            }
        }

        let event_receiver = self.event_receiver.clone();
        let debouncer = self.debouncer.clone();
        let patterns = self.patterns.clone();
        let config = self.config.clone();
        let queue_manager = self.queue_manager.clone();
        let allowed_extensions = self.allowed_extensions.clone();
        let error_tracker = self.error_tracker.clone();
        let throttle_state = self.throttle_state.clone();
        let events_received = self.events_received.clone();
        let events_processed = self.events_processed.clone();
        let events_filtered = self.events_filtered.clone();
        let queue_errors = self.queue_errors.clone();
        let events_throttled = self.events_throttled.clone();

        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer,
                patterns,
                config,
                queue_manager,
                allowed_extensions,
                error_tracker,
                throttle_state,
                events_received,
                events_processed,
                events_filtered,
                queue_errors,
                events_throttled,
            ).await;
        });

        {
            let mut handle_lock = self.processor_handle.lock().await;
            *handle_lock = Some(handle);
        }

        Ok(())
    }

    /// Main event processing loop
    #[allow(clippy::too_many_arguments)]
    async fn event_processing_loop(
        event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
        debouncer: Arc<Mutex<EventDebouncer>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatchConfig>>,
        queue_manager: Arc<QueueManager>,
        allowed_extensions: Arc<AllowedExtensions>,
        error_tracker: Arc<WatchErrorTracker>,
        throttle_state: Arc<QueueThrottleState>,
        events_received: Arc<Mutex<u64>>,
        events_processed: Arc<Mutex<u64>>,
        events_filtered: Arc<Mutex<u64>>,
        queue_errors: Arc<Mutex<u64>>,
        events_throttled: Arc<Mutex<u64>>,
    ) {
        let mut debounce_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                // Handle incoming events
                event = async {
                    let mut receiver_lock = event_receiver.lock().await;
                    if let Some(ref mut receiver) = *receiver_lock {
                        receiver.recv().await
                    } else {
                        None
                    }
                } => {
                    if let Some(event) = event {
                        Self::process_file_event(
                            event,
                            &debouncer,
                            &patterns,
                            &config,
                            &queue_manager,
                            &allowed_extensions,
                            &error_tracker,
                            &throttle_state,
                            &events_received,
                            &events_processed,
                            &events_filtered,
                            &queue_errors,
                            &events_throttled,
                        ).await;
                    } else {
                        break;
                    }
                },

                // Process debounced events
                _ = debounce_interval.tick() => {
                    Self::process_debounced_events(
                        &debouncer,
                        &patterns,
                        &config,
                        &queue_manager,
                        &allowed_extensions,
                        &error_tracker,
                        &throttle_state,
                        &events_processed,
                        &queue_errors,
                        &events_throttled,
                    ).await;
                },
            }
        }

        info!("Event processing loop stopped");
    }

    /// Process a single file event
    #[allow(clippy::too_many_arguments)]
    async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_received: &Arc<Mutex<u64>>,
        events_processed: &Arc<Mutex<u64>>,
        events_filtered: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        // Update stats
        {
            let mut count = events_received.lock().await;
            *count += 1;
        }

        // NOTE: Legacy active_projects activity update removed per Task 21
        // Activity tracking now handled via watch_folders.last_activity_at in priority_manager
        // See PriorityManager::heartbeat() for spec-compliant activity tracking

        // Check exclusion patterns FIRST (Task 518) — catches .git/, node_modules/,
        // .fastembed_cache/, target/, etc. before any other processing.
        // Skip for delete events (must be able to clean up any file).
        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let file_path_str = event.path.to_string_lossy();
            if should_exclude_file(&file_path_str) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Check allowlist via route_file() before pattern matching (Task 511/567)
        // Skip allowlist for delete events (must be able to clean up any file)
        if !matches!(event.event_kind, EventKind::Remove(_)) {
            let collection_for_check = {
                let config_lock = config.read().await;
                match config_lock.watch_type {
                    WatchType::Library => "libraries",
                    WatchType::Project => "projects",
                }.to_string()
            };
            let file_path_str = event.path.to_string_lossy();
            // tenant_id only affects source_project_id metadata, not the Excluded decision
            if matches!(allowed_extensions.route_file(&file_path_str, &collection_for_check, ""), FileRoute::Excluded) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Check patterns
        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Add to debouncer
        let should_process = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };

        if should_process {
            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                allowed_extensions,
                error_tracker,
                throttle_state,
                events_processed,
                queue_errors,
                events_throttled,
            ).await;
        }
    }

    /// Process debounced events
    #[allow(clippy::too_many_arguments)]
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };

        for event in ready_events {
            // Check exclusion patterns (Task 518) - skip for delete events
            if !matches!(event.event_kind, EventKind::Remove(_)) {
                let file_path_str = event.path.to_string_lossy();
                if should_exclude_file(&file_path_str) {
                    continue;
                }
            }

            // Check allowlist via route_file() (Task 511/567) - skip for delete events
            if !matches!(event.event_kind, EventKind::Remove(_)) {
                let collection_for_check = {
                    let config_lock = config.read().await;
                    match config_lock.watch_type {
                        WatchType::Library => "libraries",
                        WatchType::Project => "projects",
                    }.to_string()
                };
                let file_path_str = event.path.to_string_lossy();
                if matches!(allowed_extensions.route_file(&file_path_str, &collection_for_check, ""), FileRoute::Excluded) {
                    continue;
                }
            }

            // Double-check patterns
            {
                let patterns_lock = patterns.read().await;
                if !patterns_lock.should_process(&event.path) {
                    continue;
                }
            }

            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                allowed_extensions,
                error_tracker,
                throttle_state,
                events_processed,
                queue_errors,
                events_throttled,
            ).await;
        }
    }

    /// Determine operation type based on event and file state
    fn determine_operation_type(event_kind: EventKind, file_path: &Path) -> UnifiedOp {
        match event_kind {
            EventKind::Create(_) => UnifiedOp::Ingest,
            EventKind::Remove(_) => UnifiedOp::Delete,
            EventKind::Modify(_) => {
                // Check if file still exists
                if file_path.exists() {
                    UnifiedOp::Update
                } else {
                    // Race condition: file deleted during debounce
                    UnifiedOp::Delete
                }
            },
            _ => UnifiedOp::Update,  // Default to update for other events
        }
    }

    /// Find project root by looking for .git directory
    fn find_project_root(file_path: &Path) -> PathBuf {
        let mut current = file_path.parent().unwrap_or(file_path);

        while current != current.parent().unwrap_or(Path::new("/")) {
            if current.join(".git").exists() {
                return current.to_path_buf();
            }
            current = current.parent().unwrap_or(Path::new("/"));
        }

        // Fallback to file's parent directory
        file_path.parent().unwrap_or(file_path).to_path_buf()
    }

    /// Determine collection and tenant_id based on watch type
    ///
    /// Multi-tenant routing logic:
    /// - Project watches: route to _projects collection with project_id as tenant
    /// - Library watches: route to _libraries collection with library_name as tenant
    fn determine_collection_and_tenant(
        watch_type: WatchType,
        project_root: &Path,
        library_name: Option<&str>,
        legacy_collection: &str,
    ) -> (String, String) {
        match watch_type {
            WatchType::Project => {
                let project_id = calculate_tenant_id(project_root);
                (UNIFIED_PROJECTS_COLLECTION.to_string(), project_id)
            }
            WatchType::Library => {
                let tenant = library_name
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        // Fallback: extract library name from legacy collection or path
                        if legacy_collection.starts_with('_') {
                            legacy_collection[1..].to_string()
                        } else {
                            project_root.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown")
                                .to_string()
                        }
                    });
                (UNIFIED_LIBRARIES_COLLECTION.to_string(), tenant)
            }
        }
    }

    /// Enqueue file operation with retry logic, multi-tenant routing, error tracking (Task 461.5),
    /// and queue depth throttling (Task 461.8)
    async fn enqueue_file_operation(
        event: FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        error_tracker: &Arc<WatchErrorTracker>,
        throttle_state: &Arc<QueueThrottleState>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        events_throttled: &Arc<Mutex<u64>>,
    ) {
        // Skip if not a file
        if !event.path.is_file() && !matches!(event.event_kind, EventKind::Remove(_)) {
            return;
        }

        // Check comprehensive exclusion patterns (e.g. .fastembed_cache, .mypy_cache, node_modules)
        let file_path_str = event.path.to_string_lossy();
        if should_exclude_file(&file_path_str) {
            debug!(
                "File excluded by exclusion engine, skipping: {}",
                event.path.display()
            );
            return;
        }

        // Get watch_id for error tracking
        let watch_id = {
            let config_lock = config.read().await;
            config_lock.id.clone()
        };

        // Check if this watch is in backoff or disabled (Task 461.5)
        if !error_tracker.can_process(&watch_id).await {
            debug!(
                "Watch {} is in backoff or disabled, skipping file: {}",
                watch_id,
                event.path.display()
            );
            return;
        }

        // Refresh queue depth if needed (Task 461.8)
        if throttle_state.needs_refresh().await {
            throttle_state.update_from_queue(queue_manager).await;
        }

        // Check if we should throttle based on queue depth (Task 461.8)
        if throttle_state.should_throttle().await {
            let load_level = throttle_state.get_load_level().await;
            debug!(
                "Throttling event due to {} queue load (depth: {}): {}",
                load_level.as_str(),
                throttle_state.get_depth().await,
                event.path.display()
            );
            let mut count = events_throttled.lock().await;
            *count += 1;
            return;
        }

        // Determine operation type
        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        // Find project root
        let project_root = Self::find_project_root(&event.path);

        // Get branch
        let branch = get_current_branch(&project_root);

        // Classify file type for metadata
        let file_type = classify_file_type(&event.path);

        // Determine collection and tenant_id based on watch type (multi-tenant routing)
        let (collection, tenant_id) = {
            let config_lock = config.read().await;
            Self::determine_collection_and_tenant(
                config_lock.watch_type,
                &project_root,
                config_lock.library_name.as_deref(),
                &config_lock.collection,
            )
        };

        // Get absolute path
        let file_absolute_path = event.path.to_string_lossy().to_string();

        // Apply format-based routing (Task 567): override collection for library-routed files
        let (final_collection, metadata) = if matches!(operation, UnifiedOp::Delete) {
            // For deletes, use the original collection (tracked_files has the correct value)
            (collection.clone(), None)
        } else {
            match allowed_extensions.route_file(&file_absolute_path, &collection, &tenant_id) {
                FileRoute::LibraryCollection { source_project_id } if collection != UNIFIED_LIBRARIES_COLLECTION => {
                    let meta = source_project_id.as_ref().map(|pid| {
                        serde_json::json!({"source_project_id": pid}).to_string()
                    });
                    debug!(
                        "Format-based routing override: {} -> libraries (source_project={})",
                        file_absolute_path, tenant_id
                    );
                    (UNIFIED_LIBRARIES_COLLECTION.to_string(), meta)
                }
                _ => (collection.clone(), None),
            }
        };

        // Log multi-tenant routing decision
        debug!(
            "Multi-tenant routing: file={}, collection={}, tenant={}, file_type={}, branch={}",
            file_absolute_path, final_collection, tenant_id, file_type.as_str(), branch
        );

        // Create FilePayload for unified queue per spec
        let file_payload = FilePayload {
            file_path: file_absolute_path.clone(),
            file_type: Some(file_type.as_str().to_string()),
            file_hash: None,  // Hash computed during processing if needed
            size_bytes: event.path.metadata().ok().map(|m| m.len()),
        };
        let payload_json = serde_json::to_string(&file_payload).unwrap_or_else(|_| "{}".to_string());

        // Retry logic with exponential backoff
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAYS_MS: [u64; 3] = [500, 1000, 2000];

        for attempt in 0..MAX_RETRIES {
            match queue_manager.enqueue_unified(
                ItemType::File,
                operation,
                &tenant_id,
                &final_collection,
                &payload_json,
                0,  // Priority is computed at dequeue time via CASE/JOIN, not stored
                Some(&branch),
                metadata.as_deref(),
            ).await {
                Ok(_) => {
                    let mut count = events_processed.lock().await;
                    *count += 1;

                    // Record success (Task 461.5)
                    error_tracker.record_success(&watch_id).await;

                    debug!(
                        "Enqueued file to unified_queue: {} (operation={:?}, collection={}, tenant={}, branch={}, file_type={})",
                        file_absolute_path, operation, final_collection, tenant_id, branch, file_type.as_str()
                    );
                    return;
                },
                Err(QueueError::Database(ref e)) if attempt < MAX_RETRIES - 1 => {
                    // Retry on database errors with backoff (transient error)
                    let delay = RETRY_DELAYS_MS[attempt as usize];
                    warn!(
                        "Database error enqueueing {}: {}. Retrying in {}ms (attempt {}/{})",
                        file_absolute_path, e, delay, attempt + 1, MAX_RETRIES
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                },
                Err(ref e) => {
                    let mut count = queue_errors.lock().await;
                    *count += 1;

                    // Categorize error and record (Task 461.5)
                    let error_msg = e.to_string();
                    let error_category = ErrorCategory::categorize_str(&error_msg);

                    // Only record error for final attempt or permanent errors
                    if attempt == MAX_RETRIES - 1 || error_category == ErrorCategory::Permanent {
                        let backoff_delay = error_tracker.record_error(&watch_id, &error_msg).await;
                        let health_status = error_tracker.get_health_status(&watch_id).await;

                        error!(
                            "Failed to enqueue file {}: {} (attempt {}/{}, category={}, health={}, backoff_ms={})",
                            file_absolute_path, e, attempt + 1, MAX_RETRIES,
                            error_category.as_str(),
                            health_status.as_str(),
                            backoff_delay
                        );
                    } else {
                        error!(
                            "Failed to enqueue file {}: {} (attempt {}/{})",
                            file_absolute_path, e, attempt + 1, MAX_RETRIES
                        );
                    }
                    return;
                }
            }
        }

        // All retries failed - record as final error (Task 461.5)
        let error_msg = format!("Failed after {} retries", MAX_RETRIES);
        let _backoff_delay = error_tracker.record_error(&watch_id, &error_msg).await;
        let health_status = error_tracker.get_health_status(&watch_id).await;

        let mut count = queue_errors.lock().await;
        *count += 1;
        error!(
            "Failed to enqueue file {} after {} retries (watch health: {})",
            file_absolute_path,
            MAX_RETRIES,
            health_status.as_str()
        );
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> WatchingQueueStats {
        WatchingQueueStats {
            events_received: *self.events_received.lock().await,
            events_processed: *self.events_processed.lock().await,
            events_filtered: *self.events_filtered.lock().await,
            queue_errors: *self.queue_errors.lock().await,
            events_throttled: *self.events_throttled.lock().await,
        }
    }

    /// Get the throttle state reference (Task 461.8)
    pub fn throttle_state(&self) -> &Arc<QueueThrottleState> {
        &self.throttle_state
    }

    /// Get throttle summary for this watcher (Task 461.8)
    pub async fn get_throttle_summary(&self) -> QueueThrottleSummary {
        self.throttle_state.get_summary().await
    }

    /// Get error state for this watcher (Task 461.5)
    ///
    /// This is an async operation to avoid blocking.
    pub async fn get_error_state(&self) -> Option<WatchErrorState> {
        let config = self.config.read().await;
        self.error_tracker.get_state(&config.id)
    }

    /// Get the error tracker reference (Task 461.5)
    pub fn error_tracker(&self) -> &Arc<WatchErrorTracker> {
        &self.error_tracker
    }

    /// Get the watch_id for this watcher
    pub async fn watch_id(&self) -> String {
        let config = self.config.read().await;
        config.id.clone()
    }

    /// Get health status for this watcher (Task 461.5)
    pub async fn get_health_status(&self) -> WatchHealthStatus {
        let config = self.config.read().await;
        self.error_tracker.get_health_status(&config.id).await
    }
}

/// Watching statistics
#[derive(Debug, Clone)]
pub struct WatchingQueueStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_filtered: u64,
    pub queue_errors: u64,
    pub events_throttled: u64,  // Task 461.8: Events skipped due to queue depth
}

//
// ========== WATCH ERROR TRACKING (Task 461) ==========
//

/// Health status of a watch folder
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchHealthStatus {
    /// Watch is operating normally
    Healthy,
    /// Watch has experienced errors but is still operational
    Degraded,
    /// Watch is in backoff due to repeated failures
    Backoff,
    /// Watch has been disabled due to too many failures (circuit breaker open)
    Disabled,
    /// Circuit breaker half-open - allowing periodic retry attempts (Task 461.15)
    HalfOpen,
}

impl Default for WatchHealthStatus {
    fn default() -> Self {
        WatchHealthStatus::Healthy
    }
}

impl WatchHealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            WatchHealthStatus::Healthy => "healthy",
            WatchHealthStatus::Degraded => "degraded",
            WatchHealthStatus::Backoff => "backoff",
            WatchHealthStatus::Disabled => "disabled",
            WatchHealthStatus::HalfOpen => "half_open",
        }
    }
}

/// Error category for classifying errors as transient or permanent (Task 461.5)
///
/// Transient errors are temporary and may succeed on retry (e.g., database busy).
/// Permanent errors won't succeed on retry (e.g., file not found, permission denied).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    /// Temporary error that may succeed on retry (e.g., database busy, network timeout)
    Transient,
    /// Permanent error that won't succeed on retry (e.g., file not found, permission denied)
    Permanent,
    /// Unknown error category (default for uncategorized errors)
    Unknown,
}

impl Default for ErrorCategory {
    fn default() -> Self {
        ErrorCategory::Unknown
    }
}

impl ErrorCategory {
    /// Categorize an error based on its message and type
    ///
    /// # Arguments
    /// * `error` - The error to categorize
    ///
    /// # Returns
    /// The appropriate error category
    pub fn categorize<E: std::error::Error>(error: &E) -> Self {
        let error_msg = error.to_string().to_lowercase();

        // Permanent errors - won't succeed on retry
        if error_msg.contains("not found")
            || error_msg.contains("no such file")
            || error_msg.contains("permission denied")
            || error_msg.contains("access denied")
            || error_msg.contains("invalid path")
            || error_msg.contains("is a directory")
            || error_msg.contains("not a file")
            || error_msg.contains("invalid format")
            || error_msg.contains("unsupported")
            || error_msg.contains("corrupt")
        {
            return ErrorCategory::Permanent;
        }

        // Transient errors - may succeed on retry
        if error_msg.contains("busy")
            || error_msg.contains("locked")
            || error_msg.contains("timeout")
            || error_msg.contains("connection")
            || error_msg.contains("network")
            || error_msg.contains("temporary")
            || error_msg.contains("unavailable")
            || error_msg.contains("retry")
            || error_msg.contains("again")
            || error_msg.contains("resource temporarily")
        {
            return ErrorCategory::Transient;
        }

        ErrorCategory::Unknown
    }

    /// Categorize based on error string (for cases where error type is not available)
    pub fn categorize_str(error_msg: &str) -> Self {
        let error_lower = error_msg.to_lowercase();

        // Permanent errors
        if error_lower.contains("not found")
            || error_lower.contains("no such file")
            || error_lower.contains("permission denied")
            || error_lower.contains("access denied")
            || error_lower.contains("invalid path")
            || error_lower.contains("is a directory")
            || error_lower.contains("not a file")
            || error_lower.contains("invalid format")
            || error_lower.contains("unsupported")
            || error_lower.contains("corrupt")
        {
            return ErrorCategory::Permanent;
        }

        // Transient errors
        if error_lower.contains("busy")
            || error_lower.contains("locked")
            || error_lower.contains("timeout")
            || error_lower.contains("connection")
            || error_lower.contains("network")
            || error_lower.contains("temporary")
            || error_lower.contains("unavailable")
            || error_lower.contains("retry")
            || error_lower.contains("again")
            || error_lower.contains("resource temporarily")
        {
            return ErrorCategory::Transient;
        }

        ErrorCategory::Unknown
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCategory::Transient => "transient",
            ErrorCategory::Permanent => "permanent",
            ErrorCategory::Unknown => "unknown",
        }
    }

    /// Whether retrying this error category is likely to succeed
    pub fn should_retry(&self) -> bool {
        match self {
            ErrorCategory::Transient => true,
            ErrorCategory::Permanent => false,
            ErrorCategory::Unknown => true, // Default to retry for unknown errors
        }
    }
}

/// Configuration for backoff strategy
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Base delay in milliseconds for backoff calculation
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (cap for exponential backoff)
    pub max_delay_ms: u64,
    /// Number of consecutive errors before entering degraded state
    pub degraded_threshold: u32,
    /// Number of consecutive errors before entering backoff state
    pub backoff_threshold: u32,
    /// Number of consecutive errors before disabling (circuit breaker open)
    pub disable_threshold: u32,
    /// Number of successful operations to reset error state
    pub success_reset_count: u32,
    // Circuit breaker settings (Task 461.15)
    /// Number of errors within the time window that triggers circuit breaker
    pub window_error_threshold: u32,
    /// Time window duration in seconds for counting errors (default: 1 hour)
    pub window_duration_secs: u64,
    /// Cooldown period in seconds before auto-retry in half-open state (default: 1 hour)
    pub cooldown_secs: u64,
    /// Number of successful operations in half-open state to close circuit
    pub half_open_success_threshold: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_delay_ms: 1000,             // 1 second base delay
            max_delay_ms: 300_000,           // 5 minutes max delay
            degraded_threshold: 3,           // 3 errors -> degraded
            backoff_threshold: 5,            // 5 errors -> backoff
            disable_threshold: 20,           // 20 consecutive errors -> circuit open (Task 461.15)
            success_reset_count: 3,          // 3 successes to fully reset
            // Circuit breaker settings (Task 461.15)
            window_error_threshold: 50,      // 50 errors in window -> circuit open
            window_duration_secs: 3600,      // 1 hour time window
            cooldown_secs: 3600,             // 1 hour cooldown before half-open
            half_open_success_threshold: 3,  // 3 successes in half-open to close
        }
    }
}

/// Circuit breaker state summary for telemetry (Task 461.15)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    /// Whether the circuit is currently open (disabled)
    pub is_open: bool,
    /// Whether the circuit is in half-open state (allowing retry)
    pub is_half_open: bool,
    /// When the circuit was opened (if currently open or half-open)
    pub opened_at: Option<SystemTime>,
    /// Number of retry attempts while in half-open state
    pub half_open_attempts: u32,
    /// Number of consecutive successes in half-open state
    pub half_open_successes: u32,
    /// Number of errors in the current time window
    pub errors_in_window: u32,
}

//
// ========== PROCESSING ERROR FEEDBACK (Task 461.13) ==========
//

/// Type of processing error for categorization (Task 461.13)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingErrorType {
    /// File was not found at the expected path
    FileNotFound,
    /// Error parsing the file content
    ParsingError,
    /// Error communicating with or storing in Qdrant
    QdrantError,
    /// Error generating embeddings
    EmbeddingError,
    /// General/unknown error
    Unknown,
}

impl ProcessingErrorType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessingErrorType::FileNotFound => "file_not_found",
            ProcessingErrorType::ParsingError => "parsing_error",
            ProcessingErrorType::QdrantError => "qdrant_error",
            ProcessingErrorType::EmbeddingError => "embedding_error",
            ProcessingErrorType::Unknown => "unknown",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "file_not_found" => ProcessingErrorType::FileNotFound,
            "parsing_error" => ProcessingErrorType::ParsingError,
            "qdrant_error" => ProcessingErrorType::QdrantError,
            "embedding_error" => ProcessingErrorType::EmbeddingError,
            _ => ProcessingErrorType::Unknown,
        }
    }

    /// Determine if this error type should cause permanent file skip
    pub fn should_skip_permanently(&self) -> bool {
        match self {
            ProcessingErrorType::FileNotFound => true,  // File doesn't exist
            ProcessingErrorType::ParsingError => false, // May be fixed with code changes
            ProcessingErrorType::QdrantError => false,  // Transient issue
            ProcessingErrorType::EmbeddingError => false, // Transient issue
            ProcessingErrorType::Unknown => false,
        }
    }
}

/// Processing error feedback from queue processor to watch system (Task 461.13)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingErrorFeedback {
    /// Watch ID that originated the file
    pub watch_id: String,
    /// File path that failed processing
    pub file_path: String,
    /// Type of error that occurred
    pub error_type: ProcessingErrorType,
    /// Detailed error message
    pub error_message: String,
    /// Queue item ID (if available)
    pub queue_item_id: Option<String>,
    /// Timestamp of the error
    pub timestamp: SystemTime,
    /// Additional context (e.g., file hash, chunk index)
    pub context: HashMap<String, String>,
}

impl ProcessingErrorFeedback {
    /// Create new error feedback
    pub fn new(
        watch_id: impl Into<String>,
        file_path: impl Into<String>,
        error_type: ProcessingErrorType,
        error_message: impl Into<String>,
    ) -> Self {
        Self {
            watch_id: watch_id.into(),
            file_path: file_path.into(),
            error_type,
            error_message: error_message.into(),
            queue_item_id: None,
            timestamp: SystemTime::now(),
            context: HashMap::new(),
        }
    }

    /// Add queue item ID
    pub fn with_queue_item_id(mut self, id: impl Into<String>) -> Self {
        self.queue_item_id = Some(id.into());
        self
    }

    /// Add context key-value pair
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Error feedback callback signature (Task 461.13)
pub type ErrorFeedbackCallback = Box<dyn Fn(&ProcessingErrorFeedback) + Send + Sync>;

/// Manager for processing error feedback (Task 461.13)
///
/// Collects error feedback from queue processors and routes it to the appropriate
/// watch error trackers for behavior adjustment.
#[derive(Default)]
pub struct ErrorFeedbackManager {
    /// Recent errors by watch_id for querying
    recent_errors: Arc<RwLock<HashMap<String, Vec<ProcessingErrorFeedback>>>>,
    /// Files to permanently skip (by watch_id -> file_path set)
    permanent_skips: Arc<RwLock<HashMap<String, std::collections::HashSet<String>>>>,
    /// Maximum recent errors to keep per watch
    max_recent_per_watch: usize,
    /// Error callback (optional)
    callback: Option<ErrorFeedbackCallback>,
}

impl std::fmt::Debug for ErrorFeedbackManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorFeedbackManager")
            .field("max_recent_per_watch", &self.max_recent_per_watch)
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl ErrorFeedbackManager {
    /// Create a new error feedback manager
    pub fn new() -> Self {
        Self {
            recent_errors: Arc::new(RwLock::new(HashMap::new())),
            permanent_skips: Arc::new(RwLock::new(HashMap::new())),
            max_recent_per_watch: 100,
            callback: None,
        }
    }

    /// Create with custom max recent errors
    pub fn with_max_recent(mut self, max: usize) -> Self {
        self.max_recent_per_watch = max;
        self
    }

    /// Set error callback
    pub fn with_callback(mut self, callback: ErrorFeedbackCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Record a processing error (called by queue processor)
    ///
    /// This is the main error_callback(watch_id, error_type, context) entry point.
    pub async fn record_error(&self, feedback: ProcessingErrorFeedback) {
        let watch_id = feedback.watch_id.clone();
        let file_path = feedback.file_path.clone();
        let should_skip = feedback.error_type.should_skip_permanently();

        // Add to recent errors
        {
            let mut recent = self.recent_errors.write().await;
            let errors = recent.entry(watch_id.clone()).or_insert_with(Vec::new);
            errors.push(feedback.clone());

            // Trim to max
            if errors.len() > self.max_recent_per_watch {
                errors.remove(0);
            }
        }

        // Add to permanent skip list if appropriate
        if should_skip {
            let mut skips = self.permanent_skips.write().await;
            skips.entry(watch_id.clone())
                .or_insert_with(std::collections::HashSet::new)
                .insert(file_path.clone());

            info!(
                "Added {} to permanent skip list for watch {} (error: {:?})",
                file_path, watch_id, feedback.error_type
            );
        }

        // Invoke callback if set
        if let Some(ref callback) = self.callback {
            callback(&feedback);
        }
    }

    /// Check if a file should be skipped permanently
    pub async fn should_skip_file(&self, watch_id: &str, file_path: &str) -> bool {
        let skips = self.permanent_skips.read().await;
        skips.get(watch_id)
            .map(|set| set.contains(file_path))
            .unwrap_or(false)
    }

    /// Get recent errors for a watch
    pub async fn get_recent_errors(&self, watch_id: &str) -> Vec<ProcessingErrorFeedback> {
        let recent = self.recent_errors.read().await;
        recent.get(watch_id).cloned().unwrap_or_default()
    }

    /// Get error counts by type for a watch
    pub async fn get_error_counts(&self, watch_id: &str) -> HashMap<ProcessingErrorType, usize> {
        let recent = self.recent_errors.read().await;
        let mut counts = HashMap::new();

        if let Some(errors) = recent.get(watch_id) {
            for error in errors {
                *counts.entry(error.error_type).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Get all permanently skipped files for a watch
    pub async fn get_skipped_files(&self, watch_id: &str) -> Vec<String> {
        let skips = self.permanent_skips.read().await;
        skips.get(watch_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Remove a file from the permanent skip list (manual override)
    pub async fn remove_skip(&self, watch_id: &str, file_path: &str) -> bool {
        let mut skips = self.permanent_skips.write().await;
        if let Some(set) = skips.get_mut(watch_id) {
            set.remove(file_path)
        } else {
            false
        }
    }

    /// Clear all skipped files for a watch
    pub async fn clear_skips(&self, watch_id: &str) {
        let mut skips = self.permanent_skips.write().await;
        skips.remove(watch_id);
    }

    /// Clear all recent errors for a watch
    pub async fn clear_recent_errors(&self, watch_id: &str) {
        let mut recent = self.recent_errors.write().await;
        recent.remove(watch_id);
    }

    /// Get summary of all watches with processing errors
    pub async fn get_processing_error_summary(&self) -> Vec<ProcessingErrorSummary> {
        let recent = self.recent_errors.read().await;
        let skips = self.permanent_skips.read().await;

        recent.keys().map(|watch_id| {
            let errors = recent.get(watch_id).map(|e| e.len()).unwrap_or(0);
            let skipped = skips.get(watch_id).map(|s| s.len()).unwrap_or(0);
            let last_error = recent.get(watch_id)
                .and_then(|e| e.last())
                .map(|e| e.timestamp);

            ProcessingErrorSummary {
                watch_id: watch_id.clone(),
                recent_error_count: errors,
                skipped_file_count: skipped,
                last_error_time: last_error,
            }
        }).collect()
    }
}

/// Summary of processing errors for a watch (Task 461.13)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingErrorSummary {
    pub watch_id: String,
    pub recent_error_count: usize,
    pub skipped_file_count: usize,
    pub last_error_time: Option<SystemTime>,
}

/// Error state tracking for a single watch folder (Task 461)
///
/// Tracks consecutive errors, backoff state, and health status for coordinated
/// error handling between file watchers and queue processors.
#[derive(Debug, Clone)]
pub struct WatchErrorState {
    /// Number of consecutive errors for this watch
    pub consecutive_errors: u32,
    /// Total errors since watch started
    pub total_errors: u64,
    /// Timestamp of the last error
    pub last_error_time: Option<SystemTime>,
    /// Description of the last error
    pub last_error_message: Option<String>,
    /// Current backoff level (0 = no backoff, increases with each failure)
    pub backoff_level: u8,
    /// Timestamp of last successful processing
    pub last_successful_processing: Option<SystemTime>,
    /// Current health status
    pub health_status: WatchHealthStatus,
    /// Count of consecutive successes (for recovery tracking)
    pub consecutive_successes: u32,
    /// Time when backoff period ends (if in backoff)
    pub backoff_until: Option<SystemTime>,
    // Circuit breaker fields (Task 461.15)
    /// Timestamps of errors within the time window (for window-based threshold)
    pub errors_in_window: Vec<SystemTime>,
    /// When the circuit breaker was opened (if Disabled or HalfOpen)
    pub circuit_opened_at: Option<SystemTime>,
    /// Number of retry attempts in half-open state
    pub half_open_attempts: u32,
    /// Consecutive successes in half-open state
    pub half_open_successes: u32,
}

impl Default for WatchErrorState {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchErrorState {
    /// Create a new error state with all fields initialized to healthy defaults
    pub fn new() -> Self {
        Self {
            consecutive_errors: 0,
            total_errors: 0,
            last_error_time: None,
            last_error_message: None,
            backoff_level: 0,
            last_successful_processing: None,
            health_status: WatchHealthStatus::Healthy,
            consecutive_successes: 0,
            backoff_until: None,
            // Circuit breaker fields (Task 461.15)
            errors_in_window: Vec::new(),
            circuit_opened_at: None,
            half_open_attempts: 0,
            half_open_successes: 0,
        }
    }

    /// Record an error occurrence
    ///
    /// Increments error counters and updates health status based on thresholds.
    /// Implements circuit breaker pattern with both consecutive error and time-window thresholds.
    /// Returns the calculated backoff delay in milliseconds (0 if no backoff needed).
    pub fn record_error(&mut self, error_message: &str, config: &BackoffConfig) -> u64 {
        let now = SystemTime::now();

        self.consecutive_errors += 1;
        self.total_errors += 1;
        self.last_error_time = Some(now);
        self.last_error_message = Some(error_message.to_string());
        self.consecutive_successes = 0;

        // Track errors in time window (Task 461.15)
        self.errors_in_window.push(now);

        // Remove errors outside the time window
        let window_start = now - Duration::from_secs(config.window_duration_secs);
        self.errors_in_window.retain(|t| *t >= window_start);

        // Check window-based threshold for circuit breaker
        let errors_in_window = self.errors_in_window.len() as u32;

        // If in half-open state, any error immediately reopens the circuit
        if self.health_status == WatchHealthStatus::HalfOpen {
            self.health_status = WatchHealthStatus::Disabled;
            self.circuit_opened_at = Some(now);
            self.half_open_attempts += 1;
            self.half_open_successes = 0;
            return self.calculate_backoff_delay(config);
        }

        // Update health status based on thresholds
        // Circuit opens on: consecutive errors >= disable_threshold OR window errors >= window_threshold
        let should_open_circuit = self.consecutive_errors >= config.disable_threshold
            || errors_in_window >= config.window_error_threshold;

        self.health_status = if should_open_circuit {
            if self.circuit_opened_at.is_none() {
                self.circuit_opened_at = Some(now);
            }
            WatchHealthStatus::Disabled
        } else if self.consecutive_errors >= config.backoff_threshold {
            WatchHealthStatus::Backoff
        } else if self.consecutive_errors >= config.degraded_threshold {
            WatchHealthStatus::Degraded
        } else {
            WatchHealthStatus::Healthy
        };

        // Calculate backoff delay if needed
        let backoff_delay = if self.health_status == WatchHealthStatus::Backoff
            || self.health_status == WatchHealthStatus::Disabled
        {
            self.backoff_level = self.backoff_level.saturating_add(1);
            self.calculate_backoff_delay(config)
        } else {
            0
        };

        // Set backoff_until if there's a delay
        if backoff_delay > 0 {
            self.backoff_until = Some(now + Duration::from_millis(backoff_delay));
        }

        backoff_delay
    }

    /// Record a successful operation
    ///
    /// Resets error state on success, allowing recovery from degraded states.
    /// Handles half-open state for circuit breaker pattern.
    /// Returns true if health status changed (recovered to healthy).
    pub fn record_success(&mut self, config: &BackoffConfig) -> bool {
        let previous_status = self.health_status;

        self.last_successful_processing = Some(SystemTime::now());
        self.consecutive_successes += 1;

        // Handle half-open state (Task 461.15)
        if self.health_status == WatchHealthStatus::HalfOpen {
            self.half_open_successes += 1;

            // If we've had enough successes in half-open, close the circuit
            if self.half_open_successes >= config.half_open_success_threshold {
                self.reset();
                return true;
            }

            // Stay in half-open until threshold is met
            return false;
        }

        // Check if we've had enough consecutive successes to reset
        if self.consecutive_successes >= config.success_reset_count {
            self.reset();
            return previous_status != WatchHealthStatus::Healthy;
        }

        // Gradual recovery: decrease backoff level on each success
        if self.backoff_level > 0 {
            self.backoff_level = self.backoff_level.saturating_sub(1);
        }

        // Clear backoff_until if backoff level is 0
        if self.backoff_level == 0 {
            self.backoff_until = None;
        }

        // Update health status based on recovery
        if self.consecutive_errors > 0 && self.consecutive_successes > 0 {
            // Still recovering
            self.health_status = if self.backoff_level > 0 {
                WatchHealthStatus::Backoff
            } else if self.consecutive_errors >= config.degraded_threshold {
                WatchHealthStatus::Degraded
            } else {
                WatchHealthStatus::Healthy
            };
        }

        previous_status != self.health_status
    }

    /// Reset error state to healthy defaults (close circuit)
    pub fn reset(&mut self) {
        self.consecutive_errors = 0;
        self.backoff_level = 0;
        self.health_status = WatchHealthStatus::Healthy;
        self.consecutive_successes = 0;
        self.backoff_until = None;
        // Circuit breaker reset (Task 461.15)
        self.circuit_opened_at = None;
        self.half_open_attempts = 0;
        self.half_open_successes = 0;
        self.errors_in_window.clear();
        // Note: We keep total_errors, last_error_time, and last_error_message
        // for historical tracking purposes
    }

    /// Check if circuit should transition to half-open (cooldown elapsed) (Task 461.15)
    ///
    /// Returns true if the circuit is currently Disabled and the cooldown period has elapsed.
    /// When true, the caller should attempt a retry and transition to HalfOpen state.
    pub fn should_attempt_half_open(&self, config: &BackoffConfig) -> bool {
        if self.health_status != WatchHealthStatus::Disabled {
            return false;
        }

        if let Some(opened_at) = self.circuit_opened_at {
            let elapsed = SystemTime::now()
                .duration_since(opened_at)
                .unwrap_or(Duration::ZERO);
            elapsed >= Duration::from_secs(config.cooldown_secs)
        } else {
            false
        }
    }

    /// Transition to half-open state for retry attempt (Task 461.15)
    ///
    /// Call this when `should_attempt_half_open` returns true.
    pub fn transition_to_half_open(&mut self) {
        if self.health_status == WatchHealthStatus::Disabled {
            self.health_status = WatchHealthStatus::HalfOpen;
            self.half_open_successes = 0;
            self.consecutive_successes = 0;
        }
    }

    /// Manually reset the circuit breaker (for CLI use) (Task 461.15)
    ///
    /// This is equivalent to `reset()` but provides a semantic name for manual intervention.
    pub fn manual_circuit_reset(&mut self) {
        self.reset();
    }

    /// Get circuit breaker state information (Task 461.15)
    pub fn get_circuit_state(&self) -> CircuitBreakerState {
        CircuitBreakerState {
            is_open: self.health_status == WatchHealthStatus::Disabled,
            is_half_open: self.health_status == WatchHealthStatus::HalfOpen,
            opened_at: self.circuit_opened_at,
            half_open_attempts: self.half_open_attempts,
            half_open_successes: self.half_open_successes,
            errors_in_window: self.errors_in_window.len() as u32,
        }
    }

    /// Calculate backoff delay using exponential backoff with jitter
    ///
    /// Formula: min(max_delay, base_delay * 2^level + random_jitter)
    pub fn calculate_backoff_delay(&self, config: &BackoffConfig) -> u64 {
        if self.backoff_level == 0 {
            return 0;
        }

        // Exponential backoff: base_delay * 2^(level-1)
        let exponential_delay = config.base_delay_ms
            .saturating_mul(1u64 << (self.backoff_level.saturating_sub(1) as u64).min(10));

        // Cap at max delay
        let capped_delay = exponential_delay.min(config.max_delay_ms);

        // Add jitter (±10% of delay) to prevent thundering herd
        let jitter_range = capped_delay / 10;
        let jitter = if jitter_range > 0 {
            // Simple deterministic jitter based on current time
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64;
            (now % (jitter_range * 2)).saturating_sub(jitter_range)
        } else {
            0
        };

        capped_delay.saturating_add(jitter)
    }

    /// Check if currently in backoff period
    pub fn is_in_backoff(&self) -> bool {
        if let Some(backoff_until) = self.backoff_until {
            SystemTime::now() < backoff_until
        } else {
            false
        }
    }

    /// Get remaining backoff time in milliseconds (0 if not in backoff)
    pub fn remaining_backoff_ms(&self) -> u64 {
        if let Some(backoff_until) = self.backoff_until {
            backoff_until
                .duration_since(SystemTime::now())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Check if watch should be disabled (circuit breaker open)
    pub fn should_disable(&self) -> bool {
        self.health_status == WatchHealthStatus::Disabled
    }

    /// Check if watch can process (not in backoff and not disabled)
    pub fn can_process(&self) -> bool {
        !self.should_disable() && !self.is_in_backoff()
    }
}

/// Manager for tracking error states across all watches (Task 461)
///
/// Thread-safe container for WatchErrorState instances keyed by watch_id.
#[derive(Debug)]
pub struct WatchErrorTracker {
    /// Error states keyed by watch_id
    states: Arc<RwLock<HashMap<String, WatchErrorState>>>,
    /// Shared backoff configuration
    config: BackoffConfig,
}

impl WatchErrorTracker {
    /// Create a new error tracker with default configuration
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            config: BackoffConfig::default(),
        }
    }

    /// Create a new error tracker with custom configuration
    pub fn with_config(config: BackoffConfig) -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Get or create error state for a watch_id
    pub async fn get_or_create(&self, watch_id: &str) -> WatchErrorState {
        let states = self.states.read().await;
        states.get(watch_id).cloned().unwrap_or_default()
    }

    /// Record an error for a watch
    ///
    /// Returns the backoff delay in milliseconds.
    pub async fn record_error(&self, watch_id: &str, error_message: &str) -> u64 {
        let mut states = self.states.write().await;
        let state = states.entry(watch_id.to_string()).or_insert_with(WatchErrorState::new);
        let delay = state.record_error(error_message, &self.config);

        debug!(
            "Watch '{}' error recorded: consecutive={}, status={:?}, backoff_ms={}",
            watch_id, state.consecutive_errors, state.health_status, delay
        );

        delay
    }

    /// Record a successful operation for a watch
    ///
    /// Returns true if health status improved.
    pub async fn record_success(&self, watch_id: &str) -> bool {
        let mut states = self.states.write().await;
        let state = states.entry(watch_id.to_string()).or_insert_with(WatchErrorState::new);
        let improved = state.record_success(&self.config);

        if improved {
            debug!(
                "Watch '{}' health improved: status={:?}, consecutive_successes={}",
                watch_id, state.health_status, state.consecutive_successes
            );
        }

        improved
    }

    /// Check if a watch can process (not in backoff and not disabled)
    pub async fn can_process(&self, watch_id: &str) -> bool {
        let states = self.states.read().await;
        states.get(watch_id).map(|s| s.can_process()).unwrap_or(true)
    }

    /// Get health status for a watch
    pub async fn get_health_status(&self, watch_id: &str) -> WatchHealthStatus {
        let states = self.states.read().await;
        states.get(watch_id).map(|s| s.health_status).unwrap_or(WatchHealthStatus::Healthy)
    }

    /// Get all watch health statuses
    pub async fn get_all_health_statuses(&self) -> HashMap<String, WatchHealthStatus> {
        let states = self.states.read().await;
        states.iter().map(|(k, v)| (k.clone(), v.health_status)).collect()
    }

    /// Get error summary for all watches
    pub async fn get_error_summary(&self) -> Vec<WatchErrorSummary> {
        let states = self.states.read().await;
        states.iter().map(|(id, state)| WatchErrorSummary {
            watch_id: id.clone(),
            health_status: state.health_status,
            consecutive_errors: state.consecutive_errors,
            total_errors: state.total_errors,
            backoff_level: state.backoff_level,
            remaining_backoff_ms: state.remaining_backoff_ms(),
            last_error_message: state.last_error_message.clone(),
        }).collect()
    }

    /// Reset error state for a watch (manual recovery)
    pub async fn reset_watch(&self, watch_id: &str) {
        let mut states = self.states.write().await;
        if let Some(state) = states.get_mut(watch_id) {
            state.reset();
            info!("Watch '{}' error state manually reset", watch_id);
        }
    }

    /// Remove error tracking for a watch (when watch is removed)
    pub async fn remove_watch(&self, watch_id: &str) {
        let mut states = self.states.write().await;
        states.remove(watch_id);
    }

    /// Get error state for a specific watch (Task 461.5)
    ///
    /// Returns None if the watch has no error state (never had errors).
    pub fn get_state(&self, watch_id: &str) -> Option<WatchErrorState> {
        // Use try_read to avoid blocking - if lock is held, return None
        self.states.try_read().ok().and_then(|states| states.get(watch_id).cloned())
    }

    /// Set error state for a specific watch (Task 461.5)
    ///
    /// Used to restore state from database on startup.
    pub fn set_state(&self, watch_id: &str, state: WatchErrorState) {
        // Use try_write to avoid blocking - if lock is held, log warning
        match self.states.try_write() {
            Ok(mut states) => {
                states.insert(watch_id.to_string(), state);
            }
            Err(_) => {
                warn!("Could not set error state for watch {} - lock contention", watch_id);
            }
        }
    }

    /// Get error summary for a specific watch (Task 461.5)
    pub fn get_summary(&self, watch_id: &str) -> Option<WatchErrorSummary> {
        self.states.try_read().ok().and_then(|states| {
            states.get(watch_id).map(|state| WatchErrorSummary {
                watch_id: watch_id.to_string(),
                health_status: state.health_status,
                consecutive_errors: state.consecutive_errors,
                total_errors: state.total_errors,
                backoff_level: state.backoff_level,
                remaining_backoff_ms: state.remaining_backoff_ms(),
                last_error_message: state.last_error_message.clone(),
            })
        })
    }
}

impl Default for WatchErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of error state for reporting (Task 461)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchErrorSummary {
    pub watch_id: String,
    pub health_status: WatchHealthStatus,
    pub consecutive_errors: u32,
    pub total_errors: u64,
    pub backoff_level: u8,
    pub remaining_backoff_ms: u64,
    pub last_error_message: Option<String>,
}

//
// ========== QUEUE DEPTH MONITORING (Task 461.8) ==========
//

/// Queue load level for adaptive throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueLoadLevel {
    /// Normal load - no throttling needed
    Normal,
    /// High load - moderate throttling recommended
    High,
    /// Critical load - aggressive throttling required
    Critical,
}

impl QueueLoadLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueLoadLevel::Normal => "normal",
            QueueLoadLevel::High => "high",
            QueueLoadLevel::Critical => "critical",
        }
    }
}

/// Configuration for queue depth throttling
#[derive(Debug, Clone)]
pub struct QueueThrottleConfig {
    /// Queue depth threshold for high load (default: 1000)
    pub high_threshold: i64,
    /// Queue depth threshold for critical load (default: 5000)
    pub critical_threshold: i64,
    /// How often to check queue depth in milliseconds (default: 5000)
    pub check_interval_ms: u64,
    /// Skip ratio when in high load (skip 1 in N events, default: 2)
    pub high_skip_ratio: u64,
    /// Skip ratio when in critical load (skip 1 in N events, default: 4)
    pub critical_skip_ratio: u64,
}

impl Default for QueueThrottleConfig {
    fn default() -> Self {
        Self {
            high_threshold: 1000,
            critical_threshold: 5000,
            check_interval_ms: 5000,   // Check every 5 seconds
            high_skip_ratio: 2,         // Skip every 2nd event
            critical_skip_ratio: 4,     // Skip every 4th event
        }
    }
}

/// State for queue depth throttling
#[derive(Debug)]
pub struct QueueThrottleState {
    /// Current queue depth (periodically updated)
    current_depth: Arc<tokio::sync::RwLock<i64>>,
    /// Current load level
    load_level: Arc<tokio::sync::RwLock<QueueLoadLevel>>,
    /// Per-collection depths
    collection_depths: Arc<tokio::sync::RwLock<HashMap<String, i64>>>,
    /// Event counter for skip ratio calculation
    event_counter: Arc<std::sync::atomic::AtomicU64>,
    /// Configuration
    config: QueueThrottleConfig,
    /// Last check timestamp
    last_check: Arc<tokio::sync::RwLock<SystemTime>>,
}

impl QueueThrottleState {
    /// Create a new throttle state with default configuration
    pub fn new() -> Self {
        Self::with_config(QueueThrottleConfig::default())
    }

    /// Create a new throttle state with custom configuration
    pub fn with_config(config: QueueThrottleConfig) -> Self {
        Self {
            current_depth: Arc::new(tokio::sync::RwLock::new(0)),
            load_level: Arc::new(tokio::sync::RwLock::new(QueueLoadLevel::Normal)),
            collection_depths: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            event_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            config,
            last_check: Arc::new(tokio::sync::RwLock::new(SystemTime::UNIX_EPOCH)),
        }
    }

    /// Update queue depth from queue manager (using unified_queue)
    pub async fn update_from_queue(&self, queue_manager: &QueueManager) {
        match queue_manager.get_unified_queue_depth(None, None).await {
            Ok(depth) => {
                let mut current = self.current_depth.write().await;
                *current = depth;

                // Update load level
                let new_level = if depth >= self.config.critical_threshold {
                    QueueLoadLevel::Critical
                } else if depth >= self.config.high_threshold {
                    QueueLoadLevel::High
                } else {
                    QueueLoadLevel::Normal
                };

                let mut level = self.load_level.write().await;
                if *level != new_level {
                    info!(
                        "Queue load level changed: {:?} -> {:?} (depth: {})",
                        *level, new_level, depth
                    );
                }
                *level = new_level;

                // Update last check time
                let mut last = self.last_check.write().await;
                *last = SystemTime::now();
            }
            Err(e) => {
                warn!("Failed to get queue depth: {}", e);
            }
        }

        // Also update per-collection depths (using unified_queue)
        match queue_manager.get_unified_queue_depth_all_collections().await {
            Ok(depths) => {
                let mut collection_depths = self.collection_depths.write().await;
                *collection_depths = depths;
            }
            Err(e) => {
                warn!("Failed to get per-collection queue depths: {}", e);
            }
        }
    }

    /// Check if we should throttle (skip this event)
    pub async fn should_throttle(&self) -> bool {
        let level = *self.load_level.read().await;
        match level {
            QueueLoadLevel::Normal => false,
            QueueLoadLevel::High => {
                // Skip every Nth event based on high_skip_ratio
                let count = self.event_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                count % self.config.high_skip_ratio != 0
            }
            QueueLoadLevel::Critical => {
                // Skip every Nth event based on critical_skip_ratio
                let count = self.event_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                count % self.config.critical_skip_ratio != 0
            }
        }
    }

    /// Check if we need to refresh queue depth (time-based)
    pub async fn needs_refresh(&self) -> bool {
        let last = *self.last_check.read().await;
        let elapsed = SystemTime::now()
            .duration_since(last)
            .unwrap_or(Duration::ZERO);
        elapsed >= Duration::from_millis(self.config.check_interval_ms)
    }

    /// Get current queue depth
    pub async fn get_depth(&self) -> i64 {
        *self.current_depth.read().await
    }

    /// Get current load level
    pub async fn get_load_level(&self) -> QueueLoadLevel {
        *self.load_level.read().await
    }

    /// Get queue depth for a specific collection
    pub async fn get_collection_depth(&self, collection: &str) -> i64 {
        let depths = self.collection_depths.read().await;
        depths.get(collection).copied().unwrap_or(0)
    }

    /// Get throttle summary for telemetry
    pub async fn get_summary(&self) -> QueueThrottleSummary {
        QueueThrottleSummary {
            total_depth: *self.current_depth.read().await,
            load_level: *self.load_level.read().await,
            events_processed: self.event_counter.load(std::sync::atomic::Ordering::SeqCst),
            high_threshold: self.config.high_threshold,
            critical_threshold: self.config.critical_threshold,
        }
    }
}

impl Default for QueueThrottleState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of throttle state for telemetry (Task 461.8)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueThrottleSummary {
    pub total_depth: i64,
    pub load_level: QueueLoadLevel,
    pub events_processed: u64,
    pub high_threshold: i64,
    pub critical_threshold: i64,
}

//
// ========== WATCH-QUEUE COORDINATION PROTOCOL (Task 461.9) ==========
//

/// Configuration for the watch-queue coordinator
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Total capacity for all watchers combined (default: 10000)
    pub total_capacity: usize,
    /// Capacity reserved per watch for fair distribution
    pub min_per_watch: usize,
    /// Maximum capacity any single watch can hold
    pub max_per_watch: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            total_capacity: 10000,
            min_per_watch: 100,
            max_per_watch: 2000,
        }
    }
}

/// Capacity allocation for a watch
#[derive(Debug, Clone)]
struct WatchAllocation {
    /// Currently held capacity
    held: usize,
    /// Total requested (including held)
    requested: usize,
    /// Last activity timestamp
    last_activity: SystemTime,
}

impl Default for WatchAllocation {
    fn default() -> Self {
        Self {
            held: 0,
            requested: 0,
            last_activity: SystemTime::now(),
        }
    }
}

/// Coordinator for watch-queue flow control (Task 461.9)
///
/// This coordinator manages capacity allocation between multiple file watchers
/// and the queue processor. It uses a semaphore-based approach where watchers
/// request capacity before enqueuing and the processor releases capacity
/// when items are dequeued.
///
/// The coordinator ensures:
/// - No single watcher can overwhelm the queue
/// - Fair distribution of capacity across watchers
/// - Backpressure when the queue is full
pub struct WatchQueueCoordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Total capacity currently allocated
    allocated: Arc<std::sync::atomic::AtomicUsize>,
    /// Per-watch allocations
    allocations: Arc<tokio::sync::RwLock<HashMap<String, WatchAllocation>>>,
    /// SQLite pool for optional persistence
    pool: Option<SqlitePool>,
    /// Metrics callback
    metrics_callback: Option<Box<dyn Fn(&str, usize, bool) + Send + Sync>>,
}

impl std::fmt::Debug for WatchQueueCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchQueueCoordinator")
            .field("config", &self.config)
            .field("allocated", &self.allocated.load(std::sync::atomic::Ordering::SeqCst))
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

impl WatchQueueCoordinator {
    /// Create a new coordinator with default configuration
    pub fn new() -> Self {
        Self::with_config(CoordinatorConfig::default())
    }

    /// Create a coordinator with custom configuration
    pub fn with_config(config: CoordinatorConfig) -> Self {
        Self {
            config,
            allocated: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            allocations: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            pool: None,
            metrics_callback: None,
        }
    }

    /// Create a coordinator with SQLite persistence
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self {
            config: CoordinatorConfig::default(),
            allocated: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            allocations: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            pool: Some(pool),
            metrics_callback: None,
        }
    }

    /// Set a metrics callback for capacity changes
    pub fn with_metrics_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, usize, bool) + Send + Sync + 'static,
    {
        self.metrics_callback = Some(Box::new(callback));
        self
    }

    /// Request capacity for a watch
    ///
    /// Returns true if the requested capacity was granted, false if it was
    /// denied due to insufficient available capacity.
    ///
    /// # Arguments
    /// * `watch_id` - Unique identifier for the watch
    /// * `num_items` - Number of items to reserve capacity for
    pub async fn request_capacity(&self, watch_id: &str, num_items: usize) -> bool {
        // Check available capacity
        let current_allocated = self.allocated.load(std::sync::atomic::Ordering::SeqCst);
        if current_allocated + num_items > self.config.total_capacity {
            debug!(
                "Capacity request denied for {}: current={}, requested={}, total={}",
                watch_id, current_allocated, num_items, self.config.total_capacity
            );
            return false;
        }

        // Check per-watch limits
        let mut allocations = self.allocations.write().await;
        let allocation = allocations
            .entry(watch_id.to_string())
            .or_insert_with(Default::default);

        if allocation.held + num_items > self.config.max_per_watch {
            debug!(
                "Capacity request denied for {}: per-watch limit (held={}, requested={}, max={})",
                watch_id, allocation.held, num_items, self.config.max_per_watch
            );
            return false;
        }

        // Grant capacity
        allocation.held += num_items;
        allocation.requested += num_items;
        allocation.last_activity = SystemTime::now();
        self.allocated.fetch_add(num_items, std::sync::atomic::Ordering::SeqCst);

        // Invoke metrics callback
        if let Some(ref callback) = self.metrics_callback {
            callback(watch_id, num_items, true);
        }

        debug!(
            "Capacity granted for {}: {} items (total held: {})",
            watch_id, num_items, allocation.held
        );

        true
    }

    /// Release capacity after items are processed
    ///
    /// # Arguments
    /// * `watch_id` - Unique identifier for the watch
    /// * `num_items` - Number of items to release
    pub async fn release_capacity(&self, watch_id: &str, num_items: usize) {
        let mut allocations = self.allocations.write().await;

        if let Some(allocation) = allocations.get_mut(watch_id) {
            let to_release = num_items.min(allocation.held);
            allocation.held = allocation.held.saturating_sub(to_release);
            allocation.last_activity = SystemTime::now();

            self.allocated.fetch_sub(to_release, std::sync::atomic::Ordering::SeqCst);

            // Invoke metrics callback
            if let Some(ref callback) = self.metrics_callback {
                callback(watch_id, to_release, false);
            }

            debug!(
                "Capacity released for {}: {} items (remaining: {})",
                watch_id, to_release, allocation.held
            );
        } else {
            warn!("Release capacity called for unknown watch: {}", watch_id);
        }
    }

    /// Get available capacity
    pub fn get_available_capacity(&self) -> usize {
        let allocated = self.allocated.load(std::sync::atomic::Ordering::SeqCst);
        self.config.total_capacity.saturating_sub(allocated)
    }

    /// Get total capacity
    pub fn get_total_capacity(&self) -> usize {
        self.config.total_capacity
    }

    /// Get currently allocated capacity
    pub fn get_allocated_capacity(&self) -> usize {
        self.allocated.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get allocation for a specific watch
    pub async fn get_watch_allocation(&self, watch_id: &str) -> Option<usize> {
        let allocations = self.allocations.read().await;
        allocations.get(watch_id).map(|a| a.held)
    }

    /// Get allocation summary for all watches
    pub async fn get_allocation_summary(&self) -> CoordinatorSummary {
        let allocations = self.allocations.read().await;
        let per_watch: HashMap<String, usize> = allocations
            .iter()
            .map(|(k, v)| (k.clone(), v.held))
            .collect();

        CoordinatorSummary {
            total_capacity: self.config.total_capacity,
            allocated_capacity: self.allocated.load(std::sync::atomic::Ordering::SeqCst),
            available_capacity: self.get_available_capacity(),
            num_watches: allocations.len(),
            per_watch_allocation: per_watch,
        }
    }

    /// Reset allocation for a watch (e.g., when watch is stopped)
    pub async fn reset_watch(&self, watch_id: &str) {
        let mut allocations = self.allocations.write().await;
        if let Some(allocation) = allocations.remove(watch_id) {
            self.allocated.fetch_sub(allocation.held, std::sync::atomic::Ordering::SeqCst);
            info!("Reset allocation for watch {}: released {} items", watch_id, allocation.held);
        }
    }

    /// Clean up stale allocations (watches that haven't been active recently)
    pub async fn cleanup_stale(&self, max_age: Duration) {
        let now = SystemTime::now();
        let mut allocations = self.allocations.write().await;

        let stale: Vec<String> = allocations
            .iter()
            .filter(|(_, alloc)| {
                now.duration_since(alloc.last_activity)
                    .map(|d| d > max_age)
                    .unwrap_or(false)
            })
            .map(|(k, _)| k.clone())
            .collect();

        for watch_id in stale {
            if let Some(allocation) = allocations.remove(&watch_id) {
                self.allocated.fetch_sub(allocation.held, std::sync::atomic::Ordering::SeqCst);
                info!("Cleaned up stale allocation for {}: released {} items", watch_id, allocation.held);
            }
        }
    }
}

impl Default for WatchQueueCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of coordinator state for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorSummary {
    pub total_capacity: usize,
    pub allocated_capacity: usize,
    pub available_capacity: usize,
    pub num_watches: usize,
    pub per_watch_allocation: HashMap<String, usize>,
}

/// Watch manager for multiple watchers
pub struct WatchManager {
    pool: SqlitePool,
    watchers: Arc<RwLock<HashMap<String, Arc<FileWatcherQueue>>>>,
    allowed_extensions: Arc<AllowedExtensions>,
    refresh_signal: Option<Arc<Notify>>,
}

impl WatchManager {
    /// Create a new watch manager
    pub fn new(pool: SqlitePool, allowed_extensions: Arc<AllowedExtensions>) -> Self {
        Self {
            pool,
            watchers: Arc::new(RwLock::new(HashMap::new())),
            allowed_extensions,
            refresh_signal: None,
        }
    }

    /// Set the refresh signal for event-driven watch folder refresh
    pub fn with_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.refresh_signal = Some(signal);
        self
    }

    /// Load watch configurations from database and start watchers
    pub async fn start_all_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Load and start project watches from watch_folders table
        self.load_watch_folders(&queue_manager).await?;

        // Load and start library watches from watch_folders table
        self.load_library_watches(&queue_manager).await?;

        Ok(())
    }

    /// Load watch configurations from watch_folders table (project watches)
    async fn load_watch_folders(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        // Query watch configurations for project watches
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'projects'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} project watches from watch_folders table", rows.len());

        for row in rows {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let _tenant_id: String = row.get("tenant_id");

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Project,
                library_name: None,
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Load library watch configurations from watch_folders table
    async fn load_library_watches(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        // Query watch_folders table for library watches
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE enabled = 1 AND is_archived = 0 AND collection = 'libraries'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} library watches from watch_folders table", rows.len());

        for row in rows {
            let _watch_id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            // Use library prefix for watch ID to avoid conflicts
            let id = format!("lib_{}", tenant_id);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Library,
                library_name: Some(tenant_id.clone()),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start a single watcher with the given configuration
    async fn start_watcher(&self, id: String, config: WatchConfig, queue_manager: Arc<QueueManager>) {
        // NOTE: Legacy register_active_project removed per Task 21
        // Project registration now handled via watch_folders table in daemon_state
        // See DaemonStateManager::upsert_watch_folder() for spec-compliant registration

        match FileWatcherQueue::new(config, queue_manager, self.allowed_extensions.clone()) {
            Ok(watcher) => {
                let watcher = Arc::new(watcher);
                match watcher.start().await {
                    Ok(_) => {
                        info!("Started watcher: {} (type: {:?})", id,
                            if id.starts_with("lib_") { "library" } else { "project" });
                        let mut watchers = self.watchers.write().await;
                        watchers.insert(id, watcher);
                    },
                    Err(e) => {
                        error!("Failed to start watcher {}: {}", id, e);
                    }
                }
            },
            Err(e) => {
                error!("Failed to create watcher {}: {}", id, e);
            }
        }
    }

    /// Refresh watches by checking for config changes (hot-reload support)
    ///
    /// This method:
    /// 1. Gets current enabled watch IDs from both tables
    /// 2. Stops watchers for watches that were disabled or removed
    /// 3. Starts watchers for newly added/enabled watches
    pub async fn refresh_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Get current enabled watch IDs from database
        let db_watch_ids = self.get_enabled_watch_ids().await?;

        // Get currently running watcher IDs
        let running_ids: Vec<String> = {
            let watchers = self.watchers.read().await;
            watchers.keys().cloned().collect()
        };

        // Stop watchers that are no longer enabled
        for id in &running_ids {
            if !db_watch_ids.contains(id) {
                info!("Stopping disabled/removed watcher: {}", id);
                let watcher = {
                    let mut watchers = self.watchers.write().await;
                    watchers.remove(id)
                };
                if let Some(w) = watcher {
                    if let Err(e) = w.stop().await {
                        error!("Failed to stop watcher {}: {}", id, e);
                    }
                }
            }
        }

        // Start watchers for newly enabled watches
        for id in &db_watch_ids {
            let already_running = {
                let watchers = self.watchers.read().await;
                watchers.contains_key(id)
            };

            if !already_running {
                info!("Starting newly enabled watcher: {}", id);
                if id.starts_with("lib_") {
                    // Library watch - reload from watch_folders
                    self.start_single_library_watch(id, &queue_manager).await?;
                } else {
                    // Project watch - reload from watch_folders
                    self.start_single_watch_folder(id, &queue_manager).await?;
                }
            }
        }

        Ok(())
    }

    /// Get all enabled watch IDs from watch_folders table
    async fn get_enabled_watch_ids(&self) -> WatchingQueueResult<Vec<String>> {
        let mut ids = Vec::new();

        let rows = sqlx::query("SELECT watch_id, collection, tenant_id FROM watch_folders WHERE enabled = 1")
            .fetch_all(&self.pool)
            .await?;
        for row in rows {
            let watch_id: String = row.get("watch_id");
            let collection: String = row.get("collection");
            let tenant_id: String = row.get("tenant_id");

            if collection == "libraries" {
                ids.push(format!("lib_{}", tenant_id));
            } else {
                ids.push(watch_id);
            }
        }

        Ok(ids)
    }

    /// Start a single watch folder by ID
    async fn start_single_watch_folder(&self, id: &str, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE watch_id = ? AND enabled = 1 AND collection = 'projects'
            "#
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let _tenant_id: String = row.get("tenant_id");

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Project,
                library_name: None,
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start a single library watch by ID
    async fn start_single_library_watch(&self, id: &str, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        // Extract library_name from id (remove "lib_" prefix)
        let library_name = id.strip_prefix("lib_").unwrap_or(id);

        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, tenant_id
            FROM watch_folders
            WHERE tenant_id = ? AND enabled = 1 AND collection = 'libraries'
            "#
        )
        .bind(library_name)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let tenant_id: String = row.get("tenant_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");

            let id = format!("lib_{}", tenant_id);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns: vec!["*".to_string()],
                ignore_patterns: vec![],
                recursive: true,
                debounce_ms: 1000,
                enabled: true,
                watch_type: WatchType::Library,
                library_name: Some(tenant_id),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start periodic polling for watch configuration changes
    ///
    /// Polls SQLite every `poll_interval_secs` seconds for changes and hot-reloads.
    /// If a refresh signal is set, the poll will also trigger on signal notification.
    pub fn start_polling(self: Arc<Self>, poll_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        let refresh_signal = self.refresh_signal.clone();
        info!("Starting watch configuration polling (interval: {}s, signal-driven: {})",
            poll_interval_secs, refresh_signal.is_some());

        tokio::spawn(async move {
            let mut poll_interval = interval(Duration::from_secs(poll_interval_secs));

            loop {
                if let Some(ref signal) = refresh_signal {
                    tokio::select! {
                        _ = poll_interval.tick() => {
                            debug!("Periodic watch configuration refresh...");
                        }
                        _ = signal.notified() => {
                            info!("Watch configuration refresh triggered by signal");
                        }
                    }
                } else {
                    poll_interval.tick().await;
                    debug!("Polling for watch configuration changes...");
                }

                if let Err(e) = self.refresh_watches().await {
                    error!("Failed to refresh watches: {}", e);
                }
            }
        })
    }

    /// Stop all watchers
    pub async fn stop_all_watches(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        for (id, watcher) in watchers.iter() {
            if let Err(e) = watcher.stop().await {
                error!("Failed to stop watcher {}: {}", id, e);
            }
        }

        Ok(())
    }

    /// Get statistics for all watchers
    pub async fn get_all_stats(&self) -> HashMap<String, WatchingQueueStats> {
        let watchers = self.watchers.read().await;
        let mut stats = HashMap::new();

        for (id, watcher) in watchers.iter() {
            stats.insert(id.clone(), watcher.get_stats().await);
        }

        stats
    }

    /// Get count of active watchers
    pub async fn active_watcher_count(&self) -> usize {
        let watchers = self.watchers.read().await;
        watchers.len()
    }

    /// Check if a specific watch is active
    pub async fn is_watch_active(&self, id: &str) -> bool {
        let watchers = self.watchers.read().await;
        watchers.contains_key(id)
    }

    /// Get error state for a specific watcher (Task 461.5)
    pub async fn get_error_state(&self, watch_id: &str) -> Option<WatchErrorState> {
        let watchers = self.watchers.read().await;
        watchers.get(watch_id)
            .and_then(|w| w.error_tracker().get_state(watch_id))
    }

    /// Get error summaries for all watchers (Task 461.5)
    pub async fn get_all_error_summaries(&self) -> HashMap<String, WatchErrorSummary> {
        let watchers = self.watchers.read().await;
        let mut summaries = HashMap::new();

        for (id, watcher) in watchers.iter() {
            if let Some(summary) = watcher.error_tracker().get_summary(id) {
                summaries.insert(id.clone(), summary);
            }
        }

        summaries
    }

    /// Get watches with health status worse than healthy (Task 461.5)
    pub async fn get_unhealthy_watches(&self) -> Vec<(String, WatchErrorSummary)> {
        let summaries = self.get_all_error_summaries().await;
        summaries.into_iter()
            .filter(|(_, summary)| summary.health_status != WatchHealthStatus::Healthy)
            .collect()
    }

    /// Persist error states to SQLite watch_folders table (Task 461.5)
    ///
    /// Updates the error tracking columns in watch_folders for all project watches.
    /// Library watches are stored in a separate table and not updated here.
    pub async fn persist_error_states(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        for (id, watcher) in watchers.iter() {
            // Skip library watches (they use library_watches table)
            if id.starts_with("lib_") {
                continue;
            }

            // Get error state from tracker
            if let Some(state) = watcher.error_tracker().get_state(id) {
                // Format timestamps as ISO strings
                let last_error_at = state.last_error_time
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                let last_success_at = state.last_successful_processing
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                let backoff_until = state.backoff_until
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .as_ref().map(wqm_common::timestamps::format_utc)
                            .unwrap_or_default()
                    });

                // Update watch_folders table
                let result = sqlx::query(
                    r#"
                    UPDATE watch_folders
                    SET consecutive_errors = ?,
                        total_errors = ?,
                        last_error_at = ?,
                        last_error_message = ?,
                        backoff_until = ?,
                        last_success_at = ?,
                        health_status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE watch_id = ?
                    "#
                )
                .bind(state.consecutive_errors as i64)
                .bind(state.total_errors as i64)
                .bind(last_error_at)
                .bind(&state.last_error_message)
                .bind(backoff_until)
                .bind(last_success_at)
                .bind(state.health_status.as_str())
                .bind(id)
                .execute(&self.pool)
                .await;

                if let Err(e) = result {
                    warn!("Failed to persist error state for watch {}: {}", id, e);
                } else {
                    debug!("Persisted error state for watch {}: health={}", id, state.health_status.as_str());
                }
            }
        }

        Ok(())
    }

    /// Load error states from SQLite into watchers (Task 461.5)
    ///
    /// Called during startup to restore error state from previous daemon session.
    pub async fn load_error_states(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        // Query error states for all project watches
        let rows = sqlx::query(
            r#"
            SELECT watch_id, consecutive_errors, total_errors, last_error_at,
                   last_error_message, backoff_until, last_success_at, health_status
            FROM watch_folders
            WHERE consecutive_errors > 0 OR health_status != 'healthy'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        for row in rows {
            let watch_id: String = row.get("watch_id");

            // Get the watcher's error tracker
            if let Some(watcher) = watchers.get(&watch_id) {
                let consecutive_errors: i64 = row.get("consecutive_errors");
                let total_errors: i64 = row.get("total_errors");
                let last_error_message: Option<String> = row.get("last_error_message");
                let health_status_str: String = row.get("health_status");

                // Parse health status
                let health_status = match health_status_str.as_str() {
                    "healthy" => WatchHealthStatus::Healthy,
                    "degraded" => WatchHealthStatus::Degraded,
                    "backoff" => WatchHealthStatus::Backoff,
                    "disabled" => WatchHealthStatus::Disabled,
                    "half_open" => WatchHealthStatus::HalfOpen,
                    _ => WatchHealthStatus::Healthy,
                };

                // Create a partial state to restore
                let restored_state = WatchErrorState {
                    consecutive_errors: consecutive_errors as u32,
                    total_errors: total_errors as u64,
                    last_error_time: None, // Would need timestamp parsing
                    last_error_message,
                    backoff_level: 0, // Will be recalculated
                    last_successful_processing: None,
                    health_status,
                    consecutive_successes: 0,
                    backoff_until: None, // Will be recalculated if needed
                    // Circuit breaker fields (Task 461.15)
                    errors_in_window: Vec::new(), // Cannot restore from SQLite
                    circuit_opened_at: None,      // Cannot restore from SQLite
                    half_open_attempts: 0,
                    half_open_successes: 0,
                };

                // Set the state in the tracker
                watcher.error_tracker().set_state(&watch_id, restored_state);

                debug!("Restored error state for watch {}: errors={}, health={}",
                    watch_id, consecutive_errors, health_status_str);
            }
        }

        Ok(())
    }

    /// Start periodic error state persistence (Task 461.5)
    ///
    /// Persists error states to SQLite every `persist_interval_secs` seconds.
    pub fn start_error_state_persistence(self: Arc<Self>, persist_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        info!("Starting error state persistence (interval: {}s)", persist_interval_secs);

        tokio::spawn(async move {
            let mut persist_interval = interval(Duration::from_secs(persist_interval_secs));

            loop {
                persist_interval.tick().await;

                debug!("Persisting error states to SQLite...");
                if let Err(e) = self.persist_error_states().await {
                    error!("Failed to persist error states: {}", e);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_get_current_branch_non_git() {
        let temp_dir = tempdir().unwrap();
        let branch = get_current_branch(temp_dir.path());
        assert_eq!(branch, "main");
    }

    // Multi-tenant routing tests
    #[test]
    fn test_watch_type_default() {
        assert_eq!(WatchType::default(), WatchType::Project);
    }

    #[test]
    fn test_watch_type_from_str() {
        assert_eq!(WatchType::from_str("project"), Some(WatchType::Project));
        assert_eq!(WatchType::from_str("library"), Some(WatchType::Library));
        assert_eq!(WatchType::from_str("PROJECT"), Some(WatchType::Project));
        assert_eq!(WatchType::from_str("LIBRARY"), Some(WatchType::Library));
        assert_eq!(WatchType::from_str("invalid"), None);
    }

    #[test]
    fn test_watch_type_as_str() {
        assert_eq!(WatchType::Project.as_str(), "project");
        assert_eq!(WatchType::Library.as_str(), "library");
    }

    #[test]
    fn test_unified_collection_constants() {
        // Canonical collection names (without underscore prefix)
        assert_eq!(UNIFIED_PROJECTS_COLLECTION, "projects");
        assert_eq!(UNIFIED_LIBRARIES_COLLECTION, "libraries");
    }

    #[test]
    fn test_determine_collection_and_tenant_project() {
        let temp_dir = tempdir().unwrap();
        let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Project,
            temp_dir.path(),
            None,
            "_old_collection",
        );

        assert_eq!(collection, UNIFIED_PROJECTS_COLLECTION);
        // Tenant ID should be local_ prefixed since temp_dir is not a git repo
        assert!(tenant_id.starts_with("local_"));
    }

    #[test]
    fn test_determine_collection_and_tenant_library_with_name() {
        let temp_dir = tempdir().unwrap();
        let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Library,
            temp_dir.path(),
            Some("my_library"),
            "_old_collection",
        );

        assert_eq!(collection, UNIFIED_LIBRARIES_COLLECTION);
        assert_eq!(tenant_id, "my_library");
    }

    #[test]
    fn test_determine_collection_and_tenant_library_fallback_from_collection() {
        let temp_dir = tempdir().unwrap();
        let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Library,
            temp_dir.path(),
            None, // No library_name provided
            "_langchain", // Legacy collection
        );

        assert_eq!(collection, UNIFIED_LIBRARIES_COLLECTION);
        // Should extract "langchain" from "_langchain"
        assert_eq!(tenant_id, "langchain");
    }

    #[test]
    fn test_determine_collection_and_tenant_library_fallback_from_path() {
        let temp_dir = tempdir().unwrap();
        let (collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Library,
            temp_dir.path(),
            None, // No library_name
            "some_collection", // No underscore prefix
        );

        assert_eq!(collection, UNIFIED_LIBRARIES_COLLECTION);
        // Should use directory name from path
        assert!(!tenant_id.is_empty());
    }

    // Library watch ID format tests
    #[test]
    fn test_library_watch_id_format() {
        let library_name = "langchain";
        let id = format!("lib_{}", library_name);

        assert!(id.starts_with("lib_"));
        assert_eq!(id, "lib_langchain");

        // Test stripping prefix
        let extracted = id.strip_prefix("lib_").unwrap_or(&id);
        assert_eq!(extracted, "langchain");
    }

    #[test]
    fn test_library_watch_config_creation() {
        let library_name = "my_docs";
        let id = format!("lib_{}", library_name);

        let config = WatchConfig {
            id: id.clone(),
            path: PathBuf::from("/path/to/docs"),
            collection: format!("_{}", library_name),
            patterns: vec!["*.pdf".to_string(), "*.md".to_string()],
            ignore_patterns: vec![".git/*".to_string()],
            recursive: true,
            debounce_ms: 2000,
            enabled: true,
            watch_type: WatchType::Library,
            library_name: Some(library_name.to_string()),
        };

        assert_eq!(config.watch_type, WatchType::Library);
        assert_eq!(config.library_name, Some("my_docs".to_string()));
        assert_eq!(config.collection, "_my_docs");
    }

    #[test]
    fn test_watch_type_routing_for_library() {
        let temp_dir = tempdir().unwrap();

        // Test library routing
        let (collection, tenant) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Library,
            temp_dir.path(),
            Some("langchain"),
            "_legacy",
        );

        // Now routes to canonical `libraries` collection
        assert_eq!(collection, "libraries");
        assert_eq!(tenant, "langchain");
    }

    #[test]
    fn test_watch_type_routing_for_project() {
        let temp_dir = tempdir().unwrap();

        // Test project routing (should use tenant ID calculation)
        let (collection, tenant) = FileWatcherQueue::determine_collection_and_tenant(
            WatchType::Project,
            temp_dir.path(),
            None,
            "_legacy",
        );

        // Now routes to canonical `projects` collection
        assert_eq!(collection, "projects");
        // Tenant should be local_ prefixed hash since temp_dir is not a git repo
        assert!(tenant.starts_with("local_"));
    }

    // ========== Task 461: Watch Error State Tests ==========

    #[test]
    fn test_watch_error_state_new() {
        let state = WatchErrorState::new();
        assert_eq!(state.consecutive_errors, 0);
        assert_eq!(state.total_errors, 0);
        assert_eq!(state.backoff_level, 0);
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        assert!(state.last_error_time.is_none());
        assert!(state.can_process());
    }

    #[test]
    fn test_watch_error_state_record_error() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // First error - should remain healthy
        let delay = state.record_error("test error 1", &config);
        assert_eq!(state.consecutive_errors, 1);
        assert_eq!(state.total_errors, 1);
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        assert_eq!(delay, 0); // No backoff yet

        // Third error - should become degraded
        state.record_error("test error 2", &config);
        let delay = state.record_error("test error 3", &config);
        assert_eq!(state.consecutive_errors, 3);
        assert_eq!(state.health_status, WatchHealthStatus::Degraded);
        assert_eq!(delay, 0); // Degraded but no backoff yet
    }

    #[test]
    fn test_watch_error_state_backoff_threshold() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Record errors up to backoff threshold
        for i in 1..=5 {
            let delay = state.record_error(&format!("error {}", i), &config);
            if i >= 5 {
                // Should be in backoff now
                assert_eq!(state.health_status, WatchHealthStatus::Backoff);
                assert!(delay > 0, "Should have backoff delay");
            }
        }
    }

    #[test]
    fn test_watch_error_state_disable_threshold() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Record errors up to disable threshold (Task 461.15: threshold is now 20)
        for _ in 0..20 {
            state.record_error("repeated error", &config);
        }

        assert_eq!(state.health_status, WatchHealthStatus::Disabled);
        assert!(state.should_disable());
        assert!(!state.can_process());
    }

    #[test]
    fn test_watch_error_state_record_success() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Get into degraded state
        for _ in 0..3 {
            state.record_error("error", &config);
        }
        assert_eq!(state.health_status, WatchHealthStatus::Degraded);

        // Record successes to recover
        for _ in 0..3 {
            state.record_success(&config);
        }

        // Should be fully reset after success_reset_count successes
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        assert_eq!(state.consecutive_errors, 0);
        assert_eq!(state.backoff_level, 0);
    }

    #[test]
    fn test_watch_error_state_reset() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Record some errors
        for _ in 0..5 {
            state.record_error("error", &config);
        }
        assert_eq!(state.health_status, WatchHealthStatus::Backoff);

        // Reset
        state.reset();

        assert_eq!(state.consecutive_errors, 0);
        assert_eq!(state.backoff_level, 0);
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        // Total errors should still be tracked
        assert_eq!(state.total_errors, 5);
    }

    #[test]
    fn test_backoff_delay_calculation() {
        let config = BackoffConfig {
            base_delay_ms: 1000,
            max_delay_ms: 60_000,
            ..BackoffConfig::default()
        };
        let mut state = WatchErrorState::new();

        // Level 0 - no delay
        assert_eq!(state.calculate_backoff_delay(&config), 0);

        // Level 1 - base delay (~1000ms with jitter)
        state.backoff_level = 1;
        let delay1 = state.calculate_backoff_delay(&config);
        assert!(delay1 >= 900 && delay1 <= 1100, "Level 1 delay should be ~1000ms, got {}", delay1);

        // Level 2 - 2x base delay (~2000ms with jitter)
        state.backoff_level = 2;
        let delay2 = state.calculate_backoff_delay(&config);
        assert!(delay2 >= 1800 && delay2 <= 2200, "Level 2 delay should be ~2000ms, got {}", delay2);

        // Level 3 - 4x base delay (~4000ms with jitter)
        state.backoff_level = 3;
        let delay3 = state.calculate_backoff_delay(&config);
        assert!(delay3 >= 3600 && delay3 <= 4400, "Level 3 delay should be ~4000ms, got {}", delay3);
    }

    #[test]
    fn test_backoff_delay_max_cap() {
        let config = BackoffConfig {
            base_delay_ms: 1000,
            max_delay_ms: 5000,
            ..BackoffConfig::default()
        };
        let mut state = WatchErrorState::new();

        // Very high level should be capped
        state.backoff_level = 20;
        let delay = state.calculate_backoff_delay(&config);
        assert!(delay <= 5500, "Delay should be capped at max_delay + jitter, got {}", delay);
    }

    #[test]
    fn test_watch_health_status_as_str() {
        assert_eq!(WatchHealthStatus::Healthy.as_str(), "healthy");
        assert_eq!(WatchHealthStatus::Degraded.as_str(), "degraded");
        assert_eq!(WatchHealthStatus::Backoff.as_str(), "backoff");
        assert_eq!(WatchHealthStatus::Disabled.as_str(), "disabled");
    }

    #[test]
    fn test_backoff_config_default() {
        let config = BackoffConfig::default();
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 300_000);
        assert_eq!(config.degraded_threshold, 3);
        assert_eq!(config.backoff_threshold, 5);
        assert_eq!(config.disable_threshold, 20);  // Task 461.15: updated threshold
        assert_eq!(config.success_reset_count, 3);
        // Circuit breaker settings (Task 461.15)
        assert_eq!(config.window_error_threshold, 50);
        assert_eq!(config.window_duration_secs, 3600);
        assert_eq!(config.cooldown_secs, 3600);
        assert_eq!(config.half_open_success_threshold, 3);
    }

    #[tokio::test]
    async fn test_watch_error_tracker_basic() {
        let tracker = WatchErrorTracker::new();

        // Record error
        let delay = tracker.record_error("watch-1", "test error").await;
        assert_eq!(delay, 0); // First error, no backoff

        // Check status
        let status = tracker.get_health_status("watch-1").await;
        assert_eq!(status, WatchHealthStatus::Healthy);

        // Record success
        tracker.record_success("watch-1").await;

        // Should still be able to process
        assert!(tracker.can_process("watch-1").await);
    }

    #[tokio::test]
    async fn test_watch_error_tracker_multiple_watches() {
        let tracker = WatchErrorTracker::new();

        // Record errors for multiple watches
        for _ in 0..5 {
            tracker.record_error("watch-bad", "error").await;
        }
        tracker.record_error("watch-good", "single error").await;

        // Check different states
        let bad_status = tracker.get_health_status("watch-bad").await;
        let good_status = tracker.get_health_status("watch-good").await;

        assert_eq!(bad_status, WatchHealthStatus::Backoff);
        assert_eq!(good_status, WatchHealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_watch_error_tracker_get_error_summary() {
        let tracker = WatchErrorTracker::new();

        tracker.record_error("watch-1", "error 1").await;
        tracker.record_error("watch-2", "error 2").await;
        tracker.record_error("watch-2", "error 3").await;

        let summary = tracker.get_error_summary().await;
        assert_eq!(summary.len(), 2);

        let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1").unwrap();
        assert_eq!(watch1_summary.consecutive_errors, 1);

        let watch2_summary = summary.iter().find(|s| s.watch_id == "watch-2").unwrap();
        assert_eq!(watch2_summary.consecutive_errors, 2);
    }

    #[tokio::test]
    async fn test_watch_error_tracker_reset_watch() {
        let tracker = WatchErrorTracker::new();

        // Get into bad state (Task 461.15: threshold is now 20)
        for _ in 0..20 {
            tracker.record_error("watch-1", "error").await;
        }
        assert_eq!(tracker.get_health_status("watch-1").await, WatchHealthStatus::Disabled);

        // Reset
        tracker.reset_watch("watch-1").await;

        // Should be healthy again
        assert_eq!(tracker.get_health_status("watch-1").await, WatchHealthStatus::Healthy);
        assert!(tracker.can_process("watch-1").await);
    }

    #[tokio::test]
    async fn test_watch_error_tracker_remove_watch() {
        let tracker = WatchErrorTracker::new();

        tracker.record_error("watch-1", "error").await;
        assert_eq!(tracker.get_error_summary().await.len(), 1);

        tracker.remove_watch("watch-1").await;
        assert_eq!(tracker.get_error_summary().await.len(), 0);
    }

    // ========== Watch-Queue Coordinator Tests (Task 461.9) ==========

    #[test]
    fn test_coordinator_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.total_capacity, 10000);
        assert_eq!(config.min_per_watch, 100);
        assert_eq!(config.max_per_watch, 2000);
    }

    #[test]
    fn test_coordinator_new() {
        let coordinator = WatchQueueCoordinator::new();
        assert_eq!(coordinator.get_total_capacity(), 10000);
        assert_eq!(coordinator.get_available_capacity(), 10000);
        assert_eq!(coordinator.get_allocated_capacity(), 0);
    }

    #[tokio::test]
    async fn test_coordinator_request_and_release_capacity() {
        let coordinator = WatchQueueCoordinator::new();

        // Request capacity
        let granted = coordinator.request_capacity("watch-1", 100).await;
        assert!(granted);
        assert_eq!(coordinator.get_allocated_capacity(), 100);
        assert_eq!(coordinator.get_available_capacity(), 9900);

        // Verify watch allocation
        let alloc = coordinator.get_watch_allocation("watch-1").await;
        assert_eq!(alloc, Some(100));

        // Release capacity
        coordinator.release_capacity("watch-1", 50).await;
        assert_eq!(coordinator.get_allocated_capacity(), 50);
        assert_eq!(coordinator.get_available_capacity(), 9950);

        // Verify watch allocation after release
        let alloc = coordinator.get_watch_allocation("watch-1").await;
        assert_eq!(alloc, Some(50));
    }

    #[tokio::test]
    async fn test_coordinator_capacity_limit() {
        let config = CoordinatorConfig {
            total_capacity: 1000,
            min_per_watch: 10,
            max_per_watch: 200,
        };
        let coordinator = WatchQueueCoordinator::with_config(config);

        // Request up to max_per_watch
        let granted = coordinator.request_capacity("watch-1", 200).await;
        assert!(granted);

        // Request more than max_per_watch should fail
        let granted = coordinator.request_capacity("watch-1", 100).await;
        assert!(!granted);
        assert_eq!(coordinator.get_allocated_capacity(), 200);
    }

    #[tokio::test]
    async fn test_coordinator_total_capacity_limit() {
        let config = CoordinatorConfig {
            total_capacity: 500,
            min_per_watch: 10,
            max_per_watch: 1000,
        };
        let coordinator = WatchQueueCoordinator::with_config(config);

        // Request up to total_capacity
        let granted = coordinator.request_capacity("watch-1", 300).await;
        assert!(granted);
        let granted = coordinator.request_capacity("watch-2", 200).await;
        assert!(granted);

        // Request more should fail (total would exceed)
        let granted = coordinator.request_capacity("watch-3", 100).await;
        assert!(!granted);
        assert_eq!(coordinator.get_allocated_capacity(), 500);
    }

    #[tokio::test]
    async fn test_coordinator_multiple_watches() {
        let coordinator = WatchQueueCoordinator::new();

        // Multiple watches can request capacity
        assert!(coordinator.request_capacity("watch-1", 100).await);
        assert!(coordinator.request_capacity("watch-2", 200).await);
        assert!(coordinator.request_capacity("watch-3", 300).await);

        assert_eq!(coordinator.get_allocated_capacity(), 600);

        // Check summary
        let summary = coordinator.get_allocation_summary().await;
        assert_eq!(summary.num_watches, 3);
        assert_eq!(summary.allocated_capacity, 600);
        assert_eq!(summary.per_watch_allocation.len(), 3);
        assert_eq!(summary.per_watch_allocation.get("watch-1"), Some(&100));
        assert_eq!(summary.per_watch_allocation.get("watch-2"), Some(&200));
        assert_eq!(summary.per_watch_allocation.get("watch-3"), Some(&300));
    }

    #[tokio::test]
    async fn test_coordinator_reset_watch() {
        let coordinator = WatchQueueCoordinator::new();

        coordinator.request_capacity("watch-1", 500).await;
        assert_eq!(coordinator.get_allocated_capacity(), 500);

        coordinator.reset_watch("watch-1").await;
        assert_eq!(coordinator.get_allocated_capacity(), 0);
        assert_eq!(coordinator.get_watch_allocation("watch-1").await, None);
    }

    #[tokio::test]
    async fn test_coordinator_release_more_than_held() {
        let coordinator = WatchQueueCoordinator::new();

        coordinator.request_capacity("watch-1", 100).await;

        // Releasing more than held should only release what's held
        coordinator.release_capacity("watch-1", 500).await;
        assert_eq!(coordinator.get_allocated_capacity(), 0);
        assert_eq!(coordinator.get_watch_allocation("watch-1").await, Some(0));
    }

    #[tokio::test]
    async fn test_coordinator_release_unknown_watch() {
        let coordinator = WatchQueueCoordinator::new();

        // Should not panic, just log a warning
        coordinator.release_capacity("unknown-watch", 100).await;
        assert_eq!(coordinator.get_allocated_capacity(), 0);
    }

    // ========== Circuit Breaker Tests (Task 461.15) ==========

    #[test]
    fn test_circuit_breaker_config_defaults() {
        let config = BackoffConfig::default();
        assert_eq!(config.disable_threshold, 20);
        assert_eq!(config.window_error_threshold, 50);
        assert_eq!(config.window_duration_secs, 3600);
        assert_eq!(config.cooldown_secs, 3600);
        assert_eq!(config.half_open_success_threshold, 3);
    }

    #[test]
    fn test_circuit_breaker_opens_on_consecutive_errors() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Record 19 errors - should not open circuit yet
        for _ in 0..19 {
            state.record_error("test error", &config);
        }
        assert_ne!(state.health_status, WatchHealthStatus::Disabled);

        // 20th error should open circuit
        state.record_error("test error", &config);
        assert_eq!(state.health_status, WatchHealthStatus::Disabled);
        assert!(state.circuit_opened_at.is_some());
    }

    #[test]
    fn test_circuit_breaker_state_info() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Initially closed
        let circuit_state = state.get_circuit_state();
        assert!(!circuit_state.is_open);
        assert!(!circuit_state.is_half_open);

        // Open the circuit
        for _ in 0..20 {
            state.record_error("test error", &config);
        }

        let circuit_state = state.get_circuit_state();
        assert!(circuit_state.is_open);
        assert!(!circuit_state.is_half_open);
        assert!(circuit_state.opened_at.is_some());
        assert_eq!(circuit_state.errors_in_window, 20);
    }

    #[test]
    fn test_half_open_state_transition() {
        let mut config = BackoffConfig::default();
        config.cooldown_secs = 0; // Immediate cooldown for testing
        let mut state = WatchErrorState::new();

        // Open the circuit
        for _ in 0..20 {
            state.record_error("test error", &config);
        }
        assert_eq!(state.health_status, WatchHealthStatus::Disabled);

        // Should transition to half-open after cooldown
        assert!(state.should_attempt_half_open(&config));
        state.transition_to_half_open();
        assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

        let circuit_state = state.get_circuit_state();
        assert!(!circuit_state.is_open);
        assert!(circuit_state.is_half_open);
    }

    #[test]
    fn test_half_open_error_reopens_circuit() {
        let mut config = BackoffConfig::default();
        config.cooldown_secs = 0;
        let mut state = WatchErrorState::new();

        // Open and transition to half-open
        for _ in 0..20 {
            state.record_error("test error", &config);
        }
        state.transition_to_half_open();
        assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);

        // Error in half-open should reopen circuit
        state.record_error("retry failed", &config);
        assert_eq!(state.health_status, WatchHealthStatus::Disabled);
        assert_eq!(state.half_open_attempts, 1);
    }

    #[test]
    fn test_half_open_success_closes_circuit() {
        let mut config = BackoffConfig::default();
        config.cooldown_secs = 0;
        config.half_open_success_threshold = 2;
        let mut state = WatchErrorState::new();

        // Open and transition to half-open
        for _ in 0..20 {
            state.record_error("test error", &config);
        }
        state.transition_to_half_open();

        // First success - still half-open
        let changed = state.record_success(&config);
        assert!(!changed);
        assert_eq!(state.health_status, WatchHealthStatus::HalfOpen);
        assert_eq!(state.half_open_successes, 1);

        // Second success - closes circuit
        let changed = state.record_success(&config);
        assert!(changed);
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        assert!(state.circuit_opened_at.is_none());
    }

    #[test]
    fn test_manual_circuit_reset() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Open the circuit
        for _ in 0..20 {
            state.record_error("test error", &config);
        }
        assert_eq!(state.health_status, WatchHealthStatus::Disabled);

        // Manual reset
        state.manual_circuit_reset();
        assert_eq!(state.health_status, WatchHealthStatus::Healthy);
        assert!(state.circuit_opened_at.is_none());
        assert_eq!(state.consecutive_errors, 0);
        assert!(state.errors_in_window.is_empty());
    }

    #[test]
    fn test_errors_in_window_tracking() {
        let config = BackoffConfig::default();
        let mut state = WatchErrorState::new();

        // Record some errors
        for _ in 0..10 {
            state.record_error("test error", &config);
        }

        let circuit_state = state.get_circuit_state();
        assert_eq!(circuit_state.errors_in_window, 10);
        assert_eq!(state.errors_in_window.len(), 10);
    }

    #[test]
    fn test_watch_health_status_half_open_as_str() {
        assert_eq!(WatchHealthStatus::HalfOpen.as_str(), "half_open");
    }

    // ========== Processing Error Feedback Tests (Task 461.13) ==========

    #[test]
    fn test_processing_error_type_as_str() {
        assert_eq!(ProcessingErrorType::FileNotFound.as_str(), "file_not_found");
        assert_eq!(ProcessingErrorType::ParsingError.as_str(), "parsing_error");
        assert_eq!(ProcessingErrorType::QdrantError.as_str(), "qdrant_error");
        assert_eq!(ProcessingErrorType::EmbeddingError.as_str(), "embedding_error");
        assert_eq!(ProcessingErrorType::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_processing_error_type_from_str() {
        assert_eq!(ProcessingErrorType::from_str("file_not_found"), ProcessingErrorType::FileNotFound);
        assert_eq!(ProcessingErrorType::from_str("parsing_error"), ProcessingErrorType::ParsingError);
        assert_eq!(ProcessingErrorType::from_str("qdrant_error"), ProcessingErrorType::QdrantError);
        assert_eq!(ProcessingErrorType::from_str("embedding_error"), ProcessingErrorType::EmbeddingError);
        assert_eq!(ProcessingErrorType::from_str("other"), ProcessingErrorType::Unknown);
    }

    #[test]
    fn test_processing_error_type_should_skip_permanently() {
        assert!(ProcessingErrorType::FileNotFound.should_skip_permanently());
        assert!(!ProcessingErrorType::ParsingError.should_skip_permanently());
        assert!(!ProcessingErrorType::QdrantError.should_skip_permanently());
        assert!(!ProcessingErrorType::EmbeddingError.should_skip_permanently());
        assert!(!ProcessingErrorType::Unknown.should_skip_permanently());
    }

    #[test]
    fn test_processing_error_feedback_new() {
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            "/path/to/file.txt",
            ProcessingErrorType::ParsingError,
            "Failed to parse file"
        );

        assert_eq!(feedback.watch_id, "watch-1");
        assert_eq!(feedback.file_path, "/path/to/file.txt");
        assert_eq!(feedback.error_type, ProcessingErrorType::ParsingError);
        assert_eq!(feedback.error_message, "Failed to parse file");
        assert!(feedback.queue_item_id.is_none());
        assert!(feedback.context.is_empty());
    }

    #[test]
    fn test_processing_error_feedback_with_context() {
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            "/path/to/file.txt",
            ProcessingErrorType::EmbeddingError,
            "Embedding failed"
        )
        .with_queue_item_id("queue-123")
        .with_context("chunk_index", "5")
        .with_context("model", "all-MiniLM-L6-v2");

        assert_eq!(feedback.queue_item_id, Some("queue-123".to_string()));
        assert_eq!(feedback.context.get("chunk_index"), Some(&"5".to_string()));
        assert_eq!(feedback.context.get("model"), Some(&"all-MiniLM-L6-v2".to_string()));
    }

    #[tokio::test]
    async fn test_error_feedback_manager_record_and_query() {
        let manager = ErrorFeedbackManager::new();

        // Record an error
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            "/path/to/file.txt",
            ProcessingErrorType::ParsingError,
            "Parse error"
        );
        manager.record_error(feedback).await;

        // Query recent errors
        let errors = manager.get_recent_errors("watch-1").await;
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].file_path, "/path/to/file.txt");
    }

    #[tokio::test]
    async fn test_error_feedback_manager_permanent_skip() {
        let manager = ErrorFeedbackManager::new();

        // Record FileNotFound - should add to permanent skip
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            "/missing/file.txt",
            ProcessingErrorType::FileNotFound,
            "File not found"
        );
        manager.record_error(feedback).await;

        // Check if file is skipped
        assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);
        assert!(!manager.should_skip_file("watch-1", "/other/file.txt").await);
        assert!(!manager.should_skip_file("watch-2", "/missing/file.txt").await);
    }

    #[tokio::test]
    async fn test_error_feedback_manager_error_counts() {
        let manager = ErrorFeedbackManager::new();

        // Record multiple errors of different types
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1", "file1.txt", ProcessingErrorType::ParsingError, "error"
        )).await;
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1", "file2.txt", ProcessingErrorType::ParsingError, "error"
        )).await;
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1", "file3.txt", ProcessingErrorType::QdrantError, "error"
        )).await;

        let counts = manager.get_error_counts("watch-1").await;
        assert_eq!(counts.get(&ProcessingErrorType::ParsingError), Some(&2));
        assert_eq!(counts.get(&ProcessingErrorType::QdrantError), Some(&1));
        assert_eq!(counts.get(&ProcessingErrorType::FileNotFound), None);
    }

    #[tokio::test]
    async fn test_error_feedback_manager_remove_skip() {
        let manager = ErrorFeedbackManager::new();

        // Add to skip list
        let feedback = ProcessingErrorFeedback::new(
            "watch-1",
            "/missing/file.txt",
            ProcessingErrorType::FileNotFound,
            "Not found"
        );
        manager.record_error(feedback).await;
        assert!(manager.should_skip_file("watch-1", "/missing/file.txt").await);

        // Remove from skip list
        let removed = manager.remove_skip("watch-1", "/missing/file.txt").await;
        assert!(removed);
        assert!(!manager.should_skip_file("watch-1", "/missing/file.txt").await);
    }

    #[tokio::test]
    async fn test_error_feedback_manager_clear_skips() {
        let manager = ErrorFeedbackManager::new();

        // Add multiple files to skip list
        for i in 0..5 {
            let feedback = ProcessingErrorFeedback::new(
                "watch-1",
                format!("/missing/file{}.txt", i),
                ProcessingErrorType::FileNotFound,
                "Not found"
            );
            manager.record_error(feedback).await;
        }

        let skipped = manager.get_skipped_files("watch-1").await;
        assert_eq!(skipped.len(), 5);

        // Clear all skips
        manager.clear_skips("watch-1").await;
        let skipped = manager.get_skipped_files("watch-1").await;
        assert!(skipped.is_empty());
    }

    #[tokio::test]
    async fn test_error_feedback_manager_summary() {
        let manager = ErrorFeedbackManager::new();

        // Add errors for multiple watches
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1", "file1.txt", ProcessingErrorType::ParsingError, "error"
        )).await;
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-1", "file2.txt", ProcessingErrorType::FileNotFound, "error"
        )).await;
        manager.record_error(ProcessingErrorFeedback::new(
            "watch-2", "file3.txt", ProcessingErrorType::QdrantError, "error"
        )).await;

        let summary = manager.get_processing_error_summary().await;
        assert_eq!(summary.len(), 2);

        let watch1_summary = summary.iter().find(|s| s.watch_id == "watch-1");
        assert!(watch1_summary.is_some());
        let watch1 = watch1_summary.unwrap();
        assert_eq!(watch1.recent_error_count, 2);
        assert_eq!(watch1.skipped_file_count, 1); // FileNotFound adds to skip
    }

    #[tokio::test]
    async fn test_error_feedback_manager_max_recent() {
        let manager = ErrorFeedbackManager::new().with_max_recent(3);

        // Add more errors than max
        for i in 0..5 {
            manager.record_error(ProcessingErrorFeedback::new(
                "watch-1",
                format!("file{}.txt", i),
                ProcessingErrorType::ParsingError,
                format!("error {}", i)
            )).await;
        }

        let errors = manager.get_recent_errors("watch-1").await;
        assert_eq!(errors.len(), 3); // Should be capped at max
        // Should have the most recent 3 (indices 2, 3, 4)
        assert!(errors.iter().any(|e| e.file_path == "file2.txt"));
        assert!(errors.iter().any(|e| e.file_path == "file3.txt"));
        assert!(errors.iter().any(|e| e.file_path == "file4.txt"));
    }
}
