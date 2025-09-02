//! File watching system
//!
//! Cross-platform file watching with event debouncing, pattern matching, and priority-based processing.
//! Integrates with the task pipeline system for responsive file processing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::interval;
use notify::{Watcher as NotifyWatcher, RecursiveMode, Result as NotifyResult, Event, EventKind};
use walkdir::WalkDir;
use glob::{Pattern, PatternError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::processing::{TaskSubmitter, TaskPriority, TaskSource, TaskPayload};

/// Errors that can occur during file watching
#[derive(Error, Debug)]
pub enum WatchingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),
    
    #[error("Pattern compilation error: {0}")]
    Pattern(#[from] PatternError),
    
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    #[error("Task submission error: {0}")]
    TaskSubmission(String),
}

/// File watching configuration with comprehensive options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherConfig {
    /// Patterns to include (glob patterns)
    pub include_patterns: Vec<String>,
    
    /// Patterns to exclude (glob patterns) 
    pub exclude_patterns: Vec<String>,
    
    /// Whether to watch directories recursively
    pub recursive: bool,
    
    /// Maximum recursion depth (-1 for unlimited)
    pub max_depth: i32,
    
    /// Debounce time in milliseconds (minimum time between events for the same file)
    pub debounce_ms: u64,
    
    /// Polling interval in milliseconds (for polling-based watching)
    pub polling_interval_ms: u64,
    
    /// Maximum number of events to queue before dropping
    pub max_queue_size: usize,
    
    /// Priority for tasks generated from file watching
    pub task_priority: TaskPriority,
    
    /// Collection name for processed documents
    pub default_collection: String,
    
    /// Whether to process existing files on startup
    pub process_existing: bool,
    
    /// File size limit in bytes (files larger than this are ignored)
    pub max_file_size: Option<u64>,
    
    /// Whether to use polling mode (useful for network drives)
    pub use_polling: bool,
    
    /// Batch processing settings
    pub batch_processing: BatchConfig,
}

/// Configuration for batch processing of file events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batch processing
    pub enabled: bool,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Maximum time to wait for batch to fill (in milliseconds)
    pub max_batch_wait_ms: u64,
    
    /// Whether to group batches by file type
    pub group_by_type: bool,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            include_patterns: vec![
                "*.txt".to_string(),
                "*.md".to_string(), 
                "*.pdf".to_string(),
                "*.epub".to_string(),
                "*.docx".to_string(),
                "*.py".to_string(),
                "*.rs".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.swp".to_string(),
                "*.bak".to_string(),
                "*~".to_string(),
                ".git/**".to_string(),
                ".svn/**".to_string(),
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "__pycache__/**".to_string(),
                ".pytest_cache/**".to_string(),
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
            ],
            recursive: true,
            max_depth: -1,
            debounce_ms: 1000, // 1 second debounce
            polling_interval_ms: 1000,
            max_queue_size: 10000,
            task_priority: TaskPriority::BackgroundWatching,
            default_collection: "documents".to_string(),
            process_existing: false,
            max_file_size: Some(100 * 1024 * 1024), // 100MB limit
            use_polling: false,
            batch_processing: BatchConfig {
                enabled: true,
                max_batch_size: 10,
                max_batch_wait_ms: 5000, // 5 seconds
                group_by_type: true,
            },
        }
    }
}

/// File event with metadata and debouncing information
#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_kind: EventKind,
    pub timestamp: Instant,
    pub system_time: SystemTime,
    pub size: Option<u64>,
    pub metadata: HashMap<String, String>,
}

/// Compiled patterns for efficient matching
#[derive(Debug)]
struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl CompiledPatterns {
    fn new(config: &WatcherConfig) -> Result<Self, WatchingError> {
        let include = config.include_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;
            
        let exclude = config.exclude_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;
            
        Ok(Self { include, exclude })
    }
    
    fn should_process(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        // Check exclude patterns first (more specific)
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
    
    /// Add event to debouncer, returns true if event should be processed immediately
    fn add_event(&mut self, event: FileEvent) -> bool {
        let now = Instant::now();
        
        if let Some(existing) = self.events.get(&event.path) {
            // If the existing event is within debounce period, update and don't process
            if now.duration_since(existing.timestamp) < self.debounce_duration {
                self.events.insert(event.path.clone(), event);
                return false;
            }
        }
        
        self.events.insert(event.path.clone(), event);
        true
    }
    
    /// Get events that are ready to be processed (past debounce period)
    fn get_ready_events(&mut self) -> Vec<FileEvent> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();
        
        for (path, event) in &self.events {
            if now.duration_since(event.timestamp) >= self.debounce_duration {
                ready.push(event.clone());
                to_remove.push(path.clone());
            }
        }
        
        for path in to_remove {
            self.events.remove(&path);
        }
        
        ready
    }
    
    /// Clear old events (cleanup)
    fn cleanup(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.events.retain(|_, event| {
            now.duration_since(event.timestamp) < max_age
        });
    }
}
