//! File Watching with SQLite Queue Integration
//!
//! This module provides Rust-based file watching that writes directly to the
//! ingestion_queue SQLite table, replacing Python file watching while maintaining
//! compatibility with Python queue processors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use git2::Repository;
use notify::{Event, EventKind, RecursiveMode, Watcher as NotifyWatcher};
use sha2::{Sha256, Digest};
use sqlx::{Row, SqlitePool};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use thiserror::Error;
use glob::Pattern;

use crate::queue_operations::{QueueManager, QueueOperation, QueueError};
use crate::file_classification::classify_file_type;
use serde::{Deserialize, Serialize};

//
// ========== MULTI-TENANT TYPES ==========
//

/// Unified collection names for multi-tenant architecture
pub const UNIFIED_PROJECTS_COLLECTION: &str = "_projects";
pub const UNIFIED_LIBRARIES_COLLECTION: &str = "_libraries";

/// Watch type distinguishing project vs library watches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatchType {
    /// Project watch - files are routed to _projects collection with project_id
    Project,
    /// Library watch - files are routed to _libraries collection with library_name
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
/// Unique tenant ID string
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_daemon_core::calculate_tenant_id;
///
/// let tenant_id = calculate_tenant_id(Path::new("/path/to/repo"));
/// // Returns: "github_com_user_repo" (if git remote exists)
/// // Or: "path_abc123def456789a" (if no git remote)
/// ```
pub fn calculate_tenant_id(project_root: &Path) -> String {
    // Try to get git remote URL using git2
    if let Ok(repo) = Repository::open(project_root) {
        // Try origin first, then upstream, then any remote
        let remote_url = repo
            .find_remote("origin")
            .or_else(|_| repo.find_remote("upstream"))
            .ok()
            .and_then(|remote| remote.url().map(|url| url.to_string()));

        if let Some(url) = remote_url {
            let sanitized = sanitize_remote_url(&url);
            debug!(
                "Generated tenant ID from git remote: {} -> {}",
                url, sanitized
            );
            return sanitized;
        }
    }

    // Fallback: SHA256 hash of absolute path
    let tenant_id = generate_path_hash_tenant_id(project_root);
    debug!(
        "Generated tenant ID from path hash: {} -> {}",
        project_root.display(),
        tenant_id
    );
    tenant_id
}

/// Sanitize a git remote URL to create a tenant ID
///
/// Removes protocols and replaces separators with underscores.
///
/// # Arguments
/// * `remote_url` - Git remote URL (HTTPS or SSH format)
///
/// # Returns
/// Sanitized tenant ID string
///
/// # Examples
/// ```
/// use workspace_qdrant_daemon_core::sanitize_remote_url;
///
/// assert_eq!(
///     sanitize_remote_url("https://github.com/user/repo.git"),
///     "github_com_user_repo"
/// );
/// assert_eq!(
///     sanitize_remote_url("git@github.com:user/repo.git"),
///     "github_com_user_repo"
/// );
/// ```
pub fn sanitize_remote_url(remote_url: &str) -> String {
    let mut url = remote_url.to_string();

    // Remove common protocols
    for protocol in &["https://", "http://", "ssh://", "git://"] {
        if url.starts_with(protocol) {
            url = url[protocol.len()..].to_string();
            break;
        }
    }

    // Remove git@ prefix (SSH format)
    if url.starts_with("git@") {
        url = url[4..].to_string();
    }

    // Remove .git suffix if present
    if url.ends_with(".git") {
        url = url[..url.len() - 4].to_string();
    }

    // Replace all separators with underscores
    url = url.replace([':', '/', '.', '@'], "_");

    // Remove any duplicate underscores
    while url.contains("__") {
        url = url.replace("__", "_");
    }

    // Remove leading/trailing underscores
    url.trim_matches('_').to_string()
}

/// Generate a tenant ID from an absolute path using SHA256 hash
///
/// Creates a hash-based tenant ID with the format: path_{16_char_hash}
///
/// # Arguments
/// * `project_root` - Path to the project directory
///
/// # Returns
/// Tenant ID with path_ prefix and 16-character hash
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_daemon_core::generate_path_hash_tenant_id;
///
/// let tenant_id = generate_path_hash_tenant_id(Path::new("/home/user/project"));
/// assert!(tenant_id.starts_with("path_"));
/// assert_eq!(tenant_id.len(), 21); // "path_" + 16 chars
/// ```
pub fn generate_path_hash_tenant_id(project_root: &Path) -> String {
    // Normalize and canonicalize path
    let abs_path = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());

    // Generate SHA256 hash
    let mut hasher = Sha256::new();
    hasher.update(abs_path.to_string_lossy().as_bytes());
    let hash = hasher.finalize();

    // Take first 16 characters of hex hash
    let hash_hex = format!("{:x}", hash);
    let hash_prefix = &hash_hex[..16];

    // Return with path_ prefix
    format!("path_{}", hash_prefix)
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
/// use workspace_qdrant_daemon_core::get_current_branch;
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
    debouncer: Arc<Mutex<EventDebouncer>>,
    watcher: Arc<Mutex<Option<Box<dyn NotifyWatcher + Send + Sync>>>>,
    event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
    processor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    // Error tracking (Task 461.5)
    error_tracker: Arc<WatchErrorTracker>,

    // Statistics
    events_received: Arc<Mutex<u64>>,
    events_processed: Arc<Mutex<u64>>,
    events_filtered: Arc<Mutex<u64>>,
    queue_errors: Arc<Mutex<u64>>,
}

impl FileWatcherQueue {
    /// Create a new file watcher with queue integration
    pub fn new(
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
    ) -> WatchingQueueResult<Self> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms);
        let error_tracker = WatchErrorTracker::new();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            patterns: Arc::new(RwLock::new(patterns)),
            queue_manager,
            debouncer: Arc::new(Mutex::new(debouncer)),
            watcher: Arc::new(Mutex::new(None)),
            event_receiver: Arc::new(Mutex::new(None)),
            processor_handle: Arc::new(Mutex::new(None)),
            error_tracker: Arc::new(error_tracker),
            events_received: Arc::new(Mutex::new(0)),
            events_processed: Arc::new(Mutex::new(0)),
            events_filtered: Arc::new(Mutex::new(0)),
            queue_errors: Arc::new(Mutex::new(0)),
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
        let error_tracker = self.error_tracker.clone();
        let events_received = self.events_received.clone();
        let events_processed = self.events_processed.clone();
        let events_filtered = self.events_filtered.clone();
        let queue_errors = self.queue_errors.clone();

        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer,
                patterns,
                config,
                queue_manager,
                error_tracker,
                events_received,
                events_processed,
                events_filtered,
                queue_errors,
            ).await;
        });

        {
            let mut handle_lock = self.processor_handle.lock().await;
            *handle_lock = Some(handle);
        }

        Ok(())
    }

    /// Main event processing loop
    async fn event_processing_loop(
        event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
        debouncer: Arc<Mutex<EventDebouncer>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatchConfig>>,
        queue_manager: Arc<QueueManager>,
        error_tracker: Arc<WatchErrorTracker>,
        events_received: Arc<Mutex<u64>>,
        events_processed: Arc<Mutex<u64>>,
        events_filtered: Arc<Mutex<u64>>,
        queue_errors: Arc<Mutex<u64>>,
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
                            &error_tracker,
                            &events_received,
                            &events_processed,
                            &events_filtered,
                            &queue_errors,
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
                        &error_tracker,
                        &events_processed,
                        &queue_errors,
                    ).await;
                },
            }
        }

        info!("Event processing loop stopped");
    }

    /// Process a single file event
    async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        error_tracker: &Arc<WatchErrorTracker>,
        events_received: &Arc<Mutex<u64>>,
        events_processed: &Arc<Mutex<u64>>,
        events_filtered: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        // Update stats
        {
            let mut count = events_received.lock().await;
            *count += 1;
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
                error_tracker,
                events_processed,
                queue_errors,
            ).await;
        }
    }

    /// Process debounced events
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        error_tracker: &Arc<WatchErrorTracker>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };

        for event in ready_events {
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
                error_tracker,
                events_processed,
                queue_errors,
            ).await;
        }
    }

    /// Determine operation type based on event and file state
    fn determine_operation_type(event_kind: EventKind, file_path: &Path) -> QueueOperation {
        match event_kind {
            EventKind::Create(_) => QueueOperation::Ingest,
            EventKind::Remove(_) => QueueOperation::Delete,
            EventKind::Modify(_) => {
                // Check if file still exists
                if file_path.exists() {
                    QueueOperation::Update
                } else {
                    // Race condition: file deleted during debounce
                    QueueOperation::Delete
                }
            },
            _ => QueueOperation::Update,  // Default to update for other events
        }
    }

    /// Calculate priority based on operation type
    fn calculate_priority(operation: QueueOperation) -> i32 {
        match operation {
            QueueOperation::Delete => 8,  // High priority for deletions
            QueueOperation::Update => 5,  // Normal priority for updates
            QueueOperation::Ingest => 5,  // Normal priority for ingestion
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

    /// Enqueue file operation with retry logic, multi-tenant routing, and error tracking (Task 461.5)
    async fn enqueue_file_operation(
        event: FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        error_tracker: &Arc<WatchErrorTracker>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        // Skip if not a file
        if !event.path.is_file() && !matches!(event.event_kind, EventKind::Remove(_)) {
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

        // Determine operation type
        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        // Calculate priority
        let priority = Self::calculate_priority(operation);

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

        // Log multi-tenant routing decision
        debug!(
            "Multi-tenant routing: file={}, collection={}, tenant={}, file_type={}, branch={}",
            file_absolute_path, collection, tenant_id, file_type.as_str(), branch
        );

        // Retry logic with exponential backoff
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAYS_MS: [u64; 3] = [500, 1000, 2000];

        for attempt in 0..MAX_RETRIES {
            match queue_manager.enqueue_file(
                &file_absolute_path,
                &collection,
                &tenant_id,
                &branch,
                operation,
                priority,
                None,
            ).await {
                Ok(_) => {
                    let mut count = events_processed.lock().await;
                    *count += 1;

                    // Record success (Task 461.5)
                    error_tracker.record_success(&watch_id).await;

                    debug!(
                        "Enqueued file: {} (operation={:?}, priority={}, collection={}, tenant={}, branch={}, file_type={})",
                        file_absolute_path, operation, priority, collection, tenant_id, branch, file_type.as_str()
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
        }
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
    /// Number of consecutive errors before disabling (circuit breaker)
    pub disable_threshold: u32,
    /// Number of successful operations to reset error state
    pub success_reset_count: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_delay_ms: 1000,        // 1 second base delay
            max_delay_ms: 300_000,      // 5 minutes max delay
            degraded_threshold: 3,       // 3 errors -> degraded
            backoff_threshold: 5,        // 5 errors -> backoff
            disable_threshold: 10,       // 10 errors -> disabled
            success_reset_count: 3,      // 3 successes to fully reset
        }
    }
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
        }
    }

    /// Record an error occurrence
    ///
    /// Increments error counters and updates health status based on thresholds.
    /// Returns the calculated backoff delay in milliseconds (0 if no backoff needed).
    pub fn record_error(&mut self, error_message: &str, config: &BackoffConfig) -> u64 {
        self.consecutive_errors += 1;
        self.total_errors += 1;
        self.last_error_time = Some(SystemTime::now());
        self.last_error_message = Some(error_message.to_string());
        self.consecutive_successes = 0;

        // Update health status based on thresholds
        self.health_status = if self.consecutive_errors >= config.disable_threshold {
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
            self.backoff_until = Some(
                SystemTime::now() + Duration::from_millis(backoff_delay)
            );
        }

        backoff_delay
    }

    /// Record a successful operation
    ///
    /// Resets error state on success, allowing recovery from degraded states.
    /// Returns true if health status changed (recovered to healthy).
    pub fn record_success(&mut self, config: &BackoffConfig) -> bool {
        let previous_status = self.health_status;

        self.last_successful_processing = Some(SystemTime::now());
        self.consecutive_successes += 1;

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

    /// Reset error state to healthy defaults
    pub fn reset(&mut self) {
        self.consecutive_errors = 0;
        self.backoff_level = 0;
        self.health_status = WatchHealthStatus::Healthy;
        self.consecutive_successes = 0;
        self.backoff_until = None;
        // Note: We keep total_errors, last_error_time, and last_error_message
        // for historical tracking purposes
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

/// Watch manager for multiple watchers
pub struct WatchManager {
    pool: SqlitePool,
    watchers: Arc<RwLock<HashMap<String, Arc<FileWatcherQueue>>>>,
}

impl WatchManager {
    /// Create a new watch manager
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            watchers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load watch configurations from database and start watchers
    pub async fn start_all_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Load and start project watches from watch_folders table
        self.load_watch_folders(&queue_manager).await?;

        // Load and start library watches from library_watches table
        self.load_library_watches(&queue_manager).await?;

        Ok(())
    }

    /// Load watch configurations from watch_folders table (project watches)
    async fn load_watch_folders(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        // Query watch configurations (using correct column names: watch_id, debounce_seconds)
        let rows = sqlx::query(
            r#"
            SELECT watch_id, path, collection, patterns, ignore_patterns,
                   recursive, debounce_seconds, enabled
            FROM watch_folders
            WHERE enabled = TRUE
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} project watches from watch_folders table", rows.len());

        for row in rows {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let patterns_json: String = row.get("patterns");
            let ignore_patterns_json: String = row.get("ignore_patterns");
            let recursive: bool = row.get("recursive");
            let debounce_seconds: f64 = row.get("debounce_seconds");

            // Multi-tenant fields (with defaults for backwards compatibility)
            let watch_type_str: Option<String> = row.try_get("watch_type").ok();
            let library_name: Option<String> = row.try_get("library_name").ok();

            let patterns: Vec<String> = serde_json::from_str(&patterns_json)
                .unwrap_or_else(|_| vec!["*".to_string()]);
            let ignore_patterns: Vec<String> = serde_json::from_str(&ignore_patterns_json)
                .unwrap_or_else(|_| Vec::new());

            // Parse watch_type, default to Project for backwards compatibility
            let watch_type = watch_type_str
                .and_then(|s| WatchType::from_str(&s))
                .unwrap_or(WatchType::Project);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns,
                ignore_patterns,
                recursive,
                debounce_ms: (debounce_seconds * 1000.0) as u64,
                enabled: true,
                watch_type,
                library_name,
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Load library watch configurations from library_watches table
    async fn load_library_watches(&self, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        // Query library_watches table
        let rows = sqlx::query(
            r#"
            SELECT library_name, path, patterns, ignore_patterns,
                   recursive, recursive_depth, debounce_seconds, enabled
            FROM library_watches
            WHERE enabled = 1
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        info!("Loading {} library watches from library_watches table", rows.len());

        for row in rows {
            let library_name: String = row.get("library_name");
            let path: String = row.get("path");
            let patterns_json: String = row.get("patterns");
            let ignore_patterns_json: String = row.get("ignore_patterns");
            let recursive: bool = row.get("recursive");
            let _recursive_depth: i32 = row.get("recursive_depth");
            let debounce_seconds: f64 = row.get("debounce_seconds");

            let patterns: Vec<String> = serde_json::from_str(&patterns_json)
                .unwrap_or_else(|_| vec!["*.pdf".to_string(), "*.epub".to_string(), "*.md".to_string(), "*.txt".to_string()]);
            let ignore_patterns: Vec<String> = serde_json::from_str(&ignore_patterns_json)
                .unwrap_or_else(|_| vec![".git/*".to_string(), "__pycache__/*".to_string()]);

            // Use library prefix for watch ID to avoid conflicts
            let id = format!("lib_{}", library_name);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                // Legacy collection field - used as fallback
                collection: format!("_{}", library_name),
                patterns,
                ignore_patterns,
                recursive,
                debounce_ms: (debounce_seconds * 1000.0) as u64,
                enabled: true,
                // Library watches always use Library type
                watch_type: WatchType::Library,
                library_name: Some(library_name.clone()),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start a single watcher with the given configuration
    async fn start_watcher(&self, id: String, config: WatchConfig, queue_manager: Arc<QueueManager>) {
        match FileWatcherQueue::new(config, queue_manager) {
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
                    // Library watch - reload from library_watches
                    self.start_single_library_watch(id, &queue_manager).await?;
                } else {
                    // Project watch - reload from watch_folders
                    self.start_single_watch_folder(id, &queue_manager).await?;
                }
            }
        }

        Ok(())
    }

    /// Get all enabled watch IDs from both tables
    async fn get_enabled_watch_ids(&self) -> WatchingQueueResult<Vec<String>> {
        let mut ids = Vec::new();

        // Get project watch IDs from watch_folders
        let project_rows = sqlx::query("SELECT watch_id FROM watch_folders WHERE enabled = TRUE")
            .fetch_all(&self.pool)
            .await?;
        for row in project_rows {
            let id: String = row.get("watch_id");
            ids.push(id);
        }

        // Get library watch IDs from library_watches
        let library_rows = sqlx::query("SELECT library_name FROM library_watches WHERE enabled = 1")
            .fetch_all(&self.pool)
            .await?;
        for row in library_rows {
            let library_name: String = row.get("library_name");
            ids.push(format!("lib_{}", library_name));
        }

        Ok(ids)
    }

    /// Start a single watch folder by ID
    async fn start_single_watch_folder(&self, id: &str, queue_manager: &Arc<QueueManager>) -> WatchingQueueResult<()> {
        let row = sqlx::query(
            r#"
            SELECT watch_id, path, collection, patterns, ignore_patterns,
                   recursive, debounce_seconds
            FROM watch_folders
            WHERE watch_id = ? AND enabled = TRUE
            "#
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let id: String = row.get("watch_id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let patterns_json: String = row.get("patterns");
            let ignore_patterns_json: String = row.get("ignore_patterns");
            let recursive: bool = row.get("recursive");
            let debounce_seconds: f64 = row.get("debounce_seconds");

            let watch_type_str: Option<String> = row.try_get("watch_type").ok();
            let library_name: Option<String> = row.try_get("library_name").ok();

            let patterns: Vec<String> = serde_json::from_str(&patterns_json)
                .unwrap_or_else(|_| vec!["*".to_string()]);
            let ignore_patterns: Vec<String> = serde_json::from_str(&ignore_patterns_json)
                .unwrap_or_else(|_| Vec::new());

            let watch_type = watch_type_str
                .and_then(|s| WatchType::from_str(&s))
                .unwrap_or(WatchType::Project);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns,
                ignore_patterns,
                recursive,
                debounce_ms: (debounce_seconds * 1000.0) as u64,
                enabled: true,
                watch_type,
                library_name,
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
            SELECT library_name, path, patterns, ignore_patterns,
                   recursive, recursive_depth, debounce_seconds
            FROM library_watches
            WHERE library_name = ? AND enabled = 1
            "#
        )
        .bind(library_name)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let library_name: String = row.get("library_name");
            let path: String = row.get("path");
            let patterns_json: String = row.get("patterns");
            let ignore_patterns_json: String = row.get("ignore_patterns");
            let recursive: bool = row.get("recursive");
            let _recursive_depth: i32 = row.get("recursive_depth");
            let debounce_seconds: f64 = row.get("debounce_seconds");

            let patterns: Vec<String> = serde_json::from_str(&patterns_json)
                .unwrap_or_else(|_| vec!["*.pdf".to_string(), "*.epub".to_string(), "*.md".to_string(), "*.txt".to_string()]);
            let ignore_patterns: Vec<String> = serde_json::from_str(&ignore_patterns_json)
                .unwrap_or_else(|_| vec![".git/*".to_string(), "__pycache__/*".to_string()]);

            let id = format!("lib_{}", library_name);

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection: format!("_{}", library_name),
                patterns,
                ignore_patterns,
                recursive,
                debounce_ms: (debounce_seconds * 1000.0) as u64,
                enabled: true,
                watch_type: WatchType::Library,
                library_name: Some(library_name),
            };

            self.start_watcher(id, config, queue_manager.clone()).await;
        }

        Ok(())
    }

    /// Start periodic polling for watch configuration changes
    ///
    /// Polls SQLite every `poll_interval_secs` seconds for changes and hot-reloads.
    pub fn start_polling(self: Arc<Self>, poll_interval_secs: u64) -> tokio::task::JoinHandle<()> {
        info!("Starting watch configuration polling (interval: {}s)", poll_interval_secs);

        tokio::spawn(async move {
            let mut poll_interval = interval(Duration::from_secs(poll_interval_secs));

            loop {
                poll_interval.tick().await;

                debug!("Polling for watch configuration changes...");
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
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_default()
                    });

                let last_success_at = state.last_successful_processing
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_default()
                    });

                let backoff_until = state.backoff_until
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| {
                        let secs = d.as_secs() as i64;
                        chrono::DateTime::from_timestamp(secs, 0)
                            .map(|dt| dt.to_rfc3339())
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
    fn test_sanitize_remote_url() {
        assert_eq!(
            sanitize_remote_url("https://github.com/user/repo.git"),
            "github_com_user_repo"
        );

        assert_eq!(
            sanitize_remote_url("git@github.com:user/repo.git"),
            "github_com_user_repo"
        );

        assert_eq!(
            sanitize_remote_url("ssh://git@gitlab.com:2222/user/project.git"),
            "gitlab_com_2222_user_project"
        );
    }

    #[test]
    fn test_generate_path_hash_tenant_id() {
        let path = Path::new("/home/user/project");
        let tenant_id = generate_path_hash_tenant_id(path);

        assert!(tenant_id.starts_with("path_"));
        assert_eq!(tenant_id.len(), 21); // "path_" + 16 chars
    }

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
        assert_eq!(UNIFIED_PROJECTS_COLLECTION, "_projects");
        assert_eq!(UNIFIED_LIBRARIES_COLLECTION, "_libraries");
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
        // Tenant ID should be path-based since temp_dir is not a git repo
        assert!(tenant_id.starts_with("path_"));
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

        assert_eq!(collection, "_libraries");
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

        assert_eq!(collection, "_projects");
        // Tenant should be path-based hash since temp_dir is not a git repo
        assert!(tenant.starts_with("path_"));
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

        // Record errors up to disable threshold
        for _ in 0..10 {
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
        assert_eq!(config.disable_threshold, 10);
        assert_eq!(config.success_reset_count, 3);
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

        // Get into bad state
        for _ in 0..10 {
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
}
