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

/// Maximum consecutive errors before disabling a watch (Task 438)
const MAX_CONSECUTIVE_ERRORS: u32 = 3;

/// Main file watcher with queue integration
pub struct FileWatcherQueue {
    config: Arc<RwLock<WatchConfig>>,
    patterns: Arc<RwLock<CompiledPatterns>>,
    queue_manager: Arc<QueueManager>,
    debouncer: Arc<Mutex<EventDebouncer>>,
    watcher: Arc<Mutex<Option<Box<dyn NotifyWatcher + Send + Sync>>>>,
    event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
    processor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    pool: Option<SqlitePool>,

    // Statistics
    events_received: Arc<Mutex<u64>>,
    events_processed: Arc<Mutex<u64>>,
    events_filtered: Arc<Mutex<u64>>,
    queue_errors: Arc<Mutex<u64>>,

    // Consecutive error tracking (Task 438)
    consecutive_errors: Arc<Mutex<u32>>,
}

impl FileWatcherQueue {
    /// Create a new file watcher with queue integration
    pub fn new(
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
    ) -> WatchingQueueResult<Self> {
        Self::with_pool(config, queue_manager, None)
    }

    /// Create a new file watcher with queue integration and SQLite pool for error tracking
    pub fn with_pool(
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
        pool: Option<SqlitePool>,
    ) -> WatchingQueueResult<Self> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms);

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            patterns: Arc::new(RwLock::new(patterns)),
            queue_manager,
            debouncer: Arc::new(Mutex::new(debouncer)),
            watcher: Arc::new(Mutex::new(None)),
            event_receiver: Arc::new(Mutex::new(None)),
            processor_handle: Arc::new(Mutex::new(None)),
            pool,
            events_received: Arc::new(Mutex::new(0)),
            events_processed: Arc::new(Mutex::new(0)),
            events_filtered: Arc::new(Mutex::new(0)),
            queue_errors: Arc::new(Mutex::new(0)),
            consecutive_errors: Arc::new(Mutex::new(0)),
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
        let events_received = self.events_received.clone();
        let events_processed = self.events_processed.clone();
        let events_filtered = self.events_filtered.clone();
        let queue_errors = self.queue_errors.clone();
        let consecutive_errors = self.consecutive_errors.clone();
        let pool = self.pool.clone();

        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer,
                patterns,
                config,
                queue_manager,
                events_received,
                events_processed,
                events_filtered,
                queue_errors,
                consecutive_errors,
                pool,
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
        events_received: Arc<Mutex<u64>>,
        events_processed: Arc<Mutex<u64>>,
        events_filtered: Arc<Mutex<u64>>,
        queue_errors: Arc<Mutex<u64>>,
        consecutive_errors: Arc<Mutex<u32>>,
        pool: Option<SqlitePool>,
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
                            &events_received,
                            &events_processed,
                            &events_filtered,
                            &queue_errors,
                            &consecutive_errors,
                            &pool,
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
                        &events_processed,
                        &queue_errors,
                        &consecutive_errors,
                        &pool,
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
        events_received: &Arc<Mutex<u64>>,
        events_processed: &Arc<Mutex<u64>>,
        events_filtered: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        consecutive_errors: &Arc<Mutex<u32>>,
        pool: &Option<SqlitePool>,
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
                events_processed,
                queue_errors,
                consecutive_errors,
                pool,
            ).await;
        }
    }

    /// Process debounced events
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        consecutive_errors: &Arc<Mutex<u32>>,
        pool: &Option<SqlitePool>,
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
                events_processed,
                queue_errors,
                consecutive_errors,
                pool,
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

    /// Calculate base priority based on operation type
    ///
    /// Returns base priority value. Task 436 adds active project priority boosting
    /// which is applied separately in `calculate_final_priority`.
    fn calculate_base_priority(operation: QueueOperation) -> i32 {
        use crate::queue_priority;
        match operation {
            QueueOperation::Delete => queue_priority::HIGH,       // 8 - High priority for deletions
            QueueOperation::Update => queue_priority::NORMAL,     // 5 - Normal priority for updates
            QueueOperation::Ingest => queue_priority::NORMAL,     // 5 - Normal priority for ingestion
            QueueOperation::ScanFolder => queue_priority::HIGH_SCAN, // 7 - High priority for initial scans (Task 433)
        }
    }

    /// Calculate final priority considering active project status (Task 436)
    ///
    /// Boosts priority to HIGH if the project has active MCP sessions.
    /// Takes the maximum of operation priority and project priority.
    async fn calculate_final_priority(
        operation: QueueOperation,
        project_root: &Path,
        pool: &Option<SqlitePool>,
    ) -> i32 {
        use crate::queue_priority;

        let base_priority = Self::calculate_base_priority(operation);

        // Check if project has active sessions (Task 436)
        if let Some(db_pool) = pool {
            let project_root_str = project_root.to_string_lossy();
            let query = r#"
                SELECT active_sessions
                FROM projects
                WHERE project_root = ?1
            "#;

            match sqlx::query(query)
                .bind(project_root_str.as_ref())
                .fetch_optional(db_pool)
                .await
            {
                Ok(Some(row)) => {
                    use sqlx::Row;
                    if let Ok(active_sessions) = row.try_get::<i32, _>("active_sessions") {
                        if active_sessions > 0 {
                            let boosted = base_priority.max(queue_priority::HIGH);
                            if boosted > base_priority {
                                debug!(
                                    "Task 436: Boosting priority {} -> {} for active project {}",
                                    base_priority, boosted, project_root_str
                                );
                            }
                            return boosted;
                        }
                    }
                }
                Ok(None) => {
                    // Project not registered - use base priority
                    debug!("Project {} not registered, using base priority", project_root_str);
                }
                Err(e) => {
                    warn!("Failed to check project priority: {}, using base priority", e);
                }
            }
        }

        base_priority
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

    /// Enqueue file operation with retry logic, multi-tenant routing, and consecutive error tracking
    ///
    /// Task 438: Implements consecutive error tracking. After MAX_CONSECUTIVE_ERRORS (3)
    /// consecutive failures, the watch will be automatically disabled in the database.
    async fn enqueue_file_operation(
        event: FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
        consecutive_errors: &Arc<Mutex<u32>>,
        pool: &Option<SqlitePool>,
    ) {
        // Skip if not a file
        if !event.path.is_file() && !matches!(event.event_kind, EventKind::Remove(_)) {
            return;
        }

        // Determine operation type
        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        // Find project root first (needed for priority calculation)
        let project_root = Self::find_project_root(&event.path);

        // Calculate priority with active project boost (Task 436)
        let priority = Self::calculate_final_priority(operation, &project_root, pool).await;

        // Get branch
        let branch = get_current_branch(&project_root);

        // Classify file type for metadata
        let file_type = classify_file_type(&event.path);

        // Get watch_id for potential disabling
        let watch_id = {
            let config_lock = config.read().await;
            config_lock.id.clone()
        };

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

                    // Reset consecutive errors on success (Task 438)
                    {
                        let mut errors = consecutive_errors.lock().await;
                        if *errors > 0 {
                            debug!("Resetting consecutive errors to 0 after successful enqueue");
                        }
                        *errors = 0;
                    }

                    debug!(
                        "Enqueued file: {} (operation={:?}, priority={}, collection={}, tenant={}, branch={}, file_type={})",
                        file_absolute_path, operation, priority, collection, tenant_id, branch, file_type.as_str()
                    );
                    return;
                },
                Err(QueueError::Database(ref e)) if attempt < MAX_RETRIES - 1 => {
                    // Retry on database errors with backoff
                    let delay = RETRY_DELAYS_MS[attempt as usize];
                    warn!(
                        "Database error enqueueing {}: {}. Retrying in {}ms (attempt {}/{})",
                        file_absolute_path, e, delay, attempt + 1, MAX_RETRIES
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                },
                Err(e) => {
                    let mut count = queue_errors.lock().await;
                    *count += 1;

                    error!(
                        "Failed to enqueue file {}: {} (attempt {}/{})",
                        file_absolute_path, e, attempt + 1, MAX_RETRIES
                    );

                    // Track consecutive errors (Task 438)
                    Self::track_consecutive_error(
                        consecutive_errors,
                        pool,
                        &watch_id,
                    ).await;
                    return;
                }
            }
        }

        // All retries failed
        let mut count = queue_errors.lock().await;
        *count += 1;
        error!(
            "Failed to enqueue file {} after {} retries",
            file_absolute_path, MAX_RETRIES
        );

        // Track consecutive errors after all retries failed (Task 438)
        Self::track_consecutive_error(
            consecutive_errors,
            pool,
            &watch_id,
        ).await;
    }

    /// Track consecutive errors and disable watch after MAX_CONSECUTIVE_ERRORS (Task 438)
    async fn track_consecutive_error(
        consecutive_errors: &Arc<Mutex<u32>>,
        pool: &Option<SqlitePool>,
        watch_id: &str,
    ) {
        let should_disable = {
            let mut errors = consecutive_errors.lock().await;
            *errors += 1;
            warn!(
                "Consecutive error count: {} (max: {})",
                *errors, MAX_CONSECUTIVE_ERRORS
            );
            *errors >= MAX_CONSECUTIVE_ERRORS
        };

        if should_disable {
            error!(
                "Watch '{}' has reached {} consecutive errors, disabling watch",
                watch_id, MAX_CONSECUTIVE_ERRORS
            );
            if let Some(pool) = pool {
                if let Err(e) = Self::disable_watch_in_database(pool, watch_id).await {
                    error!("Failed to disable watch '{}' in database: {}", watch_id, e);
                } else {
                    info!("Successfully disabled watch '{}' due to consecutive errors", watch_id);
                }
            } else {
                warn!(
                    "No database pool available to disable watch '{}', watch will continue but may fail",
                    watch_id
                );
            }
        }
    }

    /// Disable a watch in the database (Task 438)
    async fn disable_watch_in_database(
        pool: &SqlitePool,
        watch_id: &str,
    ) -> WatchingQueueResult<()> {
        sqlx::query(
            "UPDATE watch_folders SET enabled = 0, updated_at = datetime('now') WHERE watch_id = ?"
        )
        .bind(watch_id)
        .execute(pool)
        .await?;

        debug!("Disabled watch '{}' in database", watch_id);
        Ok(())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> WatchingQueueStats {
        WatchingQueueStats {
            events_received: *self.events_received.lock().await,
            events_processed: *self.events_processed.lock().await,
            events_filtered: *self.events_filtered.lock().await,
            queue_errors: *self.queue_errors.lock().await,
            consecutive_errors: *self.consecutive_errors.lock().await,
        }
    }

    /// Reset consecutive error counter (Task 438)
    ///
    /// Call this when a watch is re-enabled to reset the error tracking.
    pub async fn reset_consecutive_errors(&self) {
        let mut errors = self.consecutive_errors.lock().await;
        *errors = 0;
        debug!("Reset consecutive errors counter");
    }

    /// Get the current consecutive error count (Task 438)
    pub async fn get_consecutive_errors(&self) -> u32 {
        *self.consecutive_errors.lock().await
    }
}

/// Watching statistics
#[derive(Debug, Clone)]
pub struct WatchingQueueStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_filtered: u64,
    pub queue_errors: u64,
    /// Current consecutive error count (Task 438)
    pub consecutive_errors: u32,
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
    ///
    /// Task 438: Uses with_pool constructor to enable consecutive error tracking
    /// and automatic watch disabling after MAX_CONSECUTIVE_ERRORS failures.
    async fn start_watcher(&self, id: String, config: WatchConfig, queue_manager: Arc<QueueManager>) {
        // Pass pool for consecutive error tracking (Task 438)
        match FileWatcherQueue::with_pool(config, queue_manager, Some(self.pool.clone())) {
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

    /// Process scan_folder items from content_ingestion_queue (Task 433)
    ///
    /// This method:
    /// 1. Polls content_ingestion_queue for pending items with source_type='scan_folder'
    /// 2. Walks the directory with pattern matching
    /// 3. Enqueues discovered files to ingestion_queue
    /// 4. Marks scan_folder items as done
    async fn process_scan_folder_queue(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Query pending scan_folder items from content_ingestion_queue
        let rows = sqlx::query(
            r#"
            SELECT queue_id, content, target_collection, metadata, priority
            FROM content_ingestion_queue
            WHERE source_type = 'scan_folder' AND status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 10
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(());
        }

        info!("Processing {} scan_folder items from content queue", rows.len());

        for row in rows {
            let queue_id: String = row.get("queue_id");
            let content: String = row.get("content");
            let collection: String = row.get("target_collection");
            let metadata_json: Option<String> = row.get("metadata");
            let priority: i32 = row.get("priority");

            // Mark as in_progress
            sqlx::query(
                "UPDATE content_ingestion_queue SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP WHERE queue_id = ?"
            )
            .bind(&queue_id)
            .execute(&self.pool)
            .await?;

            // Parse folder path from content (format: "scan:{path}")
            let folder_path = if content.starts_with("scan:") {
                &content[5..]
            } else {
                warn!("Invalid scan_folder content format: {}", content);
                Self::mark_scan_folder_failed(&self.pool, &queue_id, "Invalid content format").await?;
                continue;
            };

            let folder = Path::new(folder_path);
            if !folder.exists() || !folder.is_dir() {
                warn!("Scan folder path does not exist or is not a directory: {}", folder_path);
                Self::mark_scan_folder_failed(&self.pool, &queue_id, "Path does not exist").await?;
                continue;
            }

            // Parse metadata for patterns and configuration
            let (patterns, ignore_patterns, recursive, recursive_depth, watch_type, library_name) =
                Self::parse_scan_folder_metadata(&metadata_json);

            // Discover and enqueue files
            match Self::scan_and_enqueue_files(
                &queue_manager,
                folder,
                &collection,
                &patterns,
                &ignore_patterns,
                recursive,
                recursive_depth,
                watch_type,
                library_name.as_deref(),
                priority,
            ).await {
                Ok(file_count) => {
                    info!(
                        "Scan folder completed: {} files enqueued from {} to collection {}",
                        file_count, folder_path, collection
                    );
                    Self::mark_scan_folder_done(&self.pool, &queue_id).await?;
                }
                Err(e) => {
                    error!("Failed to scan folder {}: {}", folder_path, e);
                    Self::mark_scan_folder_failed(&self.pool, &queue_id, &e.to_string()).await?;
                }
            }
        }

        Ok(())
    }

    /// Parse metadata JSON for scan_folder configuration
    fn parse_scan_folder_metadata(metadata_json: &Option<String>) -> (Vec<String>, Vec<String>, bool, i32, WatchType, Option<String>) {
        let default_patterns = vec!["*".to_string()];
        let default_ignore = vec![".git/*".to_string(), "__pycache__/*".to_string(), "node_modules/*".to_string()];

        if let Some(json_str) = metadata_json {
            if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(json_str) {
                let patterns = metadata.get("patterns")
                    .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
                    .unwrap_or_else(|| default_patterns.clone());

                let ignore_patterns = metadata.get("ignore_patterns")
                    .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
                    .unwrap_or_else(|| default_ignore.clone());

                let recursive = metadata.get("recursive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                let recursive_depth = metadata.get("recursive_depth")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(10) as i32;

                let watch_type = metadata.get("watch_type")
                    .and_then(|v| v.as_str())
                    .and_then(WatchType::from_str)
                    .unwrap_or(WatchType::Project);

                let library_name = metadata.get("library_name")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                return (patterns, ignore_patterns, recursive, recursive_depth, watch_type, library_name);
            }
        }

        (default_patterns, default_ignore, true, 10, WatchType::Project, None)
    }

    /// Scan a folder and enqueue matching files
    async fn scan_and_enqueue_files(
        queue_manager: &Arc<QueueManager>,
        folder: &Path,
        collection: &str,
        patterns: &[String],
        ignore_patterns: &[String],
        recursive: bool,
        recursive_depth: i32,
        watch_type: WatchType,
        library_name: Option<&str>,
        base_priority: i32,
    ) -> WatchingQueueResult<usize> {
        use walkdir::WalkDir;

        // Compile glob patterns
        let include_patterns: Vec<Pattern> = patterns.iter()
            .filter_map(|p| Pattern::new(p).ok())
            .collect();
        let exclude_patterns: Vec<Pattern> = ignore_patterns.iter()
            .filter_map(|p| Pattern::new(p).ok())
            .collect();

        let max_depth = if recursive {
            if recursive_depth < 0 { usize::MAX } else { recursive_depth as usize }
        } else {
            1
        };

        let walker = WalkDir::new(folder).max_depth(max_depth);
        let mut file_count = 0;

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    warn!("Error walking directory: {}", e);
                    continue;
                }
            };

            let path = entry.path();

            // Skip directories
            if !path.is_file() {
                continue;
            }

            // Get relative path for pattern matching
            let rel_path = path.strip_prefix(folder)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());

            // Check ignore patterns first
            let is_ignored = exclude_patterns.iter().any(|p| p.matches(&rel_path));
            if is_ignored {
                continue;
            }

            // Check include patterns (if any, otherwise include all)
            let is_included = include_patterns.is_empty() ||
                include_patterns.iter().any(|p| p.matches(&rel_path) || p.matches(path.file_name().unwrap_or_default().to_str().unwrap_or("")));
            if !is_included {
                continue;
            }

            // Get project root and branch
            let project_root = FileWatcherQueue::find_project_root(path);
            let branch = get_current_branch(&project_root);

            // Determine collection and tenant_id based on watch type
            let (target_collection, tenant_id) = FileWatcherQueue::determine_collection_and_tenant(
                watch_type,
                &project_root,
                library_name,
                collection,
            );

            // Enqueue the file
            let file_path = path.to_string_lossy().to_string();
            match queue_manager.enqueue_file(
                &file_path,
                &target_collection,
                &tenant_id,
                &branch,
                QueueOperation::Ingest,
                base_priority,
                None,
            ).await {
                Ok(_) => {
                    file_count += 1;
                    debug!("Enqueued file from scan: {} -> {}", file_path, target_collection);
                }
                Err(e) => {
                    warn!("Failed to enqueue file {}: {}", file_path, e);
                }
            }
        }

        Ok(file_count)
    }

    /// Mark scan_folder item as done
    async fn mark_scan_folder_done(pool: &SqlitePool, queue_id: &str) -> WatchingQueueResult<()> {
        sqlx::query(
            "UPDATE content_ingestion_queue SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE queue_id = ?"
        )
        .bind(queue_id)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// Mark scan_folder item as failed
    async fn mark_scan_folder_failed(pool: &SqlitePool, queue_id: &str, error_msg: &str) -> WatchingQueueResult<()> {
        sqlx::query(
            "UPDATE content_ingestion_queue SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE queue_id = ?"
        )
        .bind(error_msg)
        .bind(queue_id)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// Start periodic polling for watch configuration changes
    ///
    /// Polls SQLite every `poll_interval_secs` seconds for changes and hot-reloads.
    /// Also processes scan_folder items from content_ingestion_queue (Task 433).
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

                // Process scan_folder queue items (Task 433: Queue/Watch Handshake)
                if let Err(e) = self.process_scan_folder_queue().await {
                    error!("Failed to process scan_folder queue: {}", e);
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

    /// Re-enable a disabled watch (Task 438)
    ///
    /// This method:
    /// 1. Re-enables the watch in the database
    /// 2. Resets the consecutive error counter for the watcher (if running)
    /// 3. Triggers a refresh to restart the watch
    ///
    /// Call this when a watch was disabled due to consecutive errors and
    /// the underlying issue has been resolved.
    pub async fn reenable_watch(&self, watch_id: &str) -> WatchingQueueResult<()> {
        info!("Re-enabling watch: {}", watch_id);

        // Check if this is a project watch or library watch
        let is_library = watch_id.starts_with("lib_");

        if is_library {
            let library_name = watch_id.strip_prefix("lib_").unwrap_or(watch_id);
            sqlx::query(
                "UPDATE library_watches SET enabled = 1, updated_at = datetime('now') WHERE library_name = ?"
            )
            .bind(library_name)
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query(
                "UPDATE watch_folders SET enabled = 1, updated_at = datetime('now') WHERE watch_id = ?"
            )
            .bind(watch_id)
            .execute(&self.pool)
            .await?;
        }

        // If the watcher is currently running, reset its consecutive error counter
        {
            let watchers = self.watchers.read().await;
            if let Some(watcher) = watchers.get(watch_id) {
                watcher.reset_consecutive_errors().await;
            }
        }

        info!("Watch '{}' re-enabled, will be picked up on next refresh", watch_id);
        Ok(())
    }

    /// Get the consecutive error count for a specific watch (Task 438)
    pub async fn get_watch_consecutive_errors(&self, watch_id: &str) -> Option<u32> {
        let watchers = self.watchers.read().await;
        if let Some(watcher) = watchers.get(watch_id) {
            Some(watcher.get_consecutive_errors().await)
        } else {
            None
        }
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

    #[test]
    fn test_calculate_base_priority() {
        // Task 436: Test base priority calculation
        use crate::queue_priority;

        // Delete operations should have highest priority
        let delete_priority = FileWatcherQueue::calculate_base_priority(QueueOperation::Delete);
        assert_eq!(delete_priority, queue_priority::HIGH);

        // Scan folder operations should have high (but not highest) priority
        let scan_priority = FileWatcherQueue::calculate_base_priority(QueueOperation::ScanFolder);
        assert_eq!(scan_priority, queue_priority::HIGH_SCAN);

        // Ingest and Update should have normal priority
        let ingest_priority = FileWatcherQueue::calculate_base_priority(QueueOperation::Ingest);
        assert_eq!(ingest_priority, queue_priority::NORMAL);

        let update_priority = FileWatcherQueue::calculate_base_priority(QueueOperation::Update);
        assert_eq!(update_priority, queue_priority::NORMAL);

        // Verify ordering: DELETE > SCAN > INGEST/UPDATE
        assert!(delete_priority > scan_priority);
        assert!(scan_priority > ingest_priority);
        assert_eq!(ingest_priority, update_priority);
    }

    #[tokio::test]
    async fn test_calculate_final_priority_without_pool() {
        // Task 436: Test priority calculation without database pool
        // Should return base priority when pool is None
        use crate::queue_priority;

        let temp_dir = tempdir().unwrap();
        let project_root = temp_dir.path();

        // Without a pool, should just return base priority
        let priority = FileWatcherQueue::calculate_final_priority(
            QueueOperation::Ingest,
            project_root,
            &None,
        ).await;

        assert_eq!(priority, queue_priority::NORMAL);
    }

    #[tokio::test]
    async fn test_calculate_final_priority_with_active_project() {
        // Task 436: Test priority boost for active projects
        use crate::queue_priority;

        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority.db");

        // Create test database with projects table
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = sqlx::SqlitePool::connect(&db_url).await.unwrap();

        // Create projects table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                project_root TEXT NOT NULL UNIQUE,
                priority TEXT DEFAULT 'normal',
                active_sessions INTEGER DEFAULT 0,
                last_active TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(&pool)
        .await
        .unwrap();

        // Insert project with active session
        let project_path = "/test/active_project";
        sqlx::query("INSERT INTO projects (project_id, project_root, active_sessions, priority) VALUES (?, ?, ?, ?)")
            .bind("abcd12345678")
            .bind(project_path)
            .bind(1)
            .bind("high")
            .execute(&pool)
            .await
            .unwrap();

        // Test priority calculation for active project
        let priority = FileWatcherQueue::calculate_final_priority(
            QueueOperation::Ingest,
            std::path::Path::new(project_path),
            &Some(pool.clone()),
        ).await;

        // Should be boosted to HIGH
        assert_eq!(priority, queue_priority::HIGH);
    }

    #[tokio::test]
    async fn test_calculate_final_priority_with_inactive_project() {
        // Task 436: Test normal priority for inactive projects
        use crate::queue_priority;

        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority_inactive.db");

        // Create test database with projects table
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = sqlx::SqlitePool::connect(&db_url).await.unwrap();

        // Create projects table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                project_root TEXT NOT NULL UNIQUE,
                priority TEXT DEFAULT 'normal',
                active_sessions INTEGER DEFAULT 0,
                last_active TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(&pool)
        .await
        .unwrap();

        // Insert project with no active sessions
        let project_path = "/test/inactive_project";
        sqlx::query("INSERT INTO projects (project_id, project_root, active_sessions, priority) VALUES (?, ?, ?, ?)")
            .bind("efgh12345678")
            .bind(project_path)
            .bind(0)
            .bind("normal")
            .execute(&pool)
            .await
            .unwrap();

        // Test priority calculation for inactive project
        let priority = FileWatcherQueue::calculate_final_priority(
            QueueOperation::Ingest,
            std::path::Path::new(project_path),
            &Some(pool.clone()),
        ).await;

        // Should remain at NORMAL
        assert_eq!(priority, queue_priority::NORMAL);
    }
}
