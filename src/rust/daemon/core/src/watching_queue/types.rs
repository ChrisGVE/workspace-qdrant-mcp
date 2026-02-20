//! Types, enums, and helper functions for the watching queue module.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use std::collections::HashMap;

use git2::Repository;
use notify::EventKind;
use glob::Pattern;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use crate::queue_operations::QueueError;
use crate::project_disambiguation::ProjectIdCalculator;

use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES};

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
pub(super) struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl CompiledPatterns {
    pub(super) fn new(config: &WatchConfig) -> Result<Self, WatchingQueueError> {
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

    pub(super) fn should_process(&self, path: &Path) -> bool {
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
pub(super) struct FileEvent {
    pub(super) path: PathBuf,
    pub(super) event_kind: EventKind,
    pub(super) timestamp: SystemTime,
}

/// Event debouncer to prevent duplicate processing
#[derive(Debug)]
pub(super) struct EventDebouncer {
    events: HashMap<PathBuf, FileEvent>,
    debounce_duration: Duration,
}

impl EventDebouncer {
    pub(super) fn new(debounce_ms: u64) -> Self {
        Self {
            events: HashMap::new(),
            debounce_duration: Duration::from_millis(debounce_ms),
        }
    }

    /// Add event, returns true if should process immediately
    pub(super) fn add_event(&mut self, event: FileEvent) -> bool {
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
    pub(super) fn get_ready_events(&mut self) -> Vec<FileEvent> {
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

/// Watching statistics
#[derive(Debug, Clone)]
pub struct WatchingQueueStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_filtered: u64,
    pub queue_errors: u64,
    pub events_throttled: u64,  // Task 461.8: Events skipped due to queue depth
}

/// Calculate a unique tenant ID for a project root directory
///
/// This function implements the tenant ID calculation algorithm:
/// 1. Try to get git remote URL (prefer origin, fallback to upstream)
/// 2. If remote exists: Sanitize URL to create tenant ID
///    - Remove protocol (https://, git@, ssh://)
///    - Replace separators (/, ., :, @) with underscores
///    - Example: github.com/user/repo -> github_com_user_repo
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

/// Determine collection and tenant_id based on watch type
///
/// Multi-tenant routing logic:
/// - Project watches: route to _projects collection with project_id as tenant
/// - Library watches: route to _libraries collection with library_name as tenant
pub(super) fn determine_collection_and_tenant(
    watch_type: WatchType,
    project_root: &Path,
    library_name: Option<&str>,
    legacy_collection: &str,
) -> (String, String) {
    match watch_type {
        WatchType::Project => {
            let project_id = calculate_tenant_id(project_root);
            (COLLECTION_PROJECTS.to_string(), project_id)
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
            (COLLECTION_LIBRARIES.to_string(), tenant)
        }
    }
}
