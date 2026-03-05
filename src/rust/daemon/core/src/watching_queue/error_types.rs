//! Error classification, backoff configuration, circuit breaker state,
//! health status, processing error feedback types and management.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

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
    pub fn categorize<E: std::error::Error>(error: &E) -> Self {
        Self::categorize_str(&error.to_string())
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
            ErrorCategory::Unknown => true,
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
            base_delay_ms: 1000,
            max_delay_ms: 300_000,
            degraded_threshold: 3,
            backoff_threshold: 5,
            disable_threshold: 20,
            success_reset_count: 3,
            window_error_threshold: 50,
            window_duration_secs: 3600,
            cooldown_secs: 3600,
            half_open_success_threshold: 3,
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
            ProcessingErrorType::FileNotFound => true,
            ProcessingErrorType::ParsingError => false,
            ProcessingErrorType::QdrantError => false,
            ProcessingErrorType::EmbeddingError => false,
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
    pub async fn record_error(&self, feedback: ProcessingErrorFeedback) {
        let watch_id = feedback.watch_id.clone();
        let file_path = feedback.file_path.clone();
        let should_skip = feedback.error_type.should_skip_permanently();

        // Add to recent errors
        {
            let mut recent = self.recent_errors.write().await;
            let errors = recent.entry(watch_id.clone()).or_insert_with(Vec::new);
            errors.push(feedback.clone());

            if errors.len() > self.max_recent_per_watch {
                errors.remove(0);
            }
        }

        // Add to permanent skip list if appropriate
        if should_skip {
            let mut skips = self.permanent_skips.write().await;
            skips
                .entry(watch_id.clone())
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
        skips
            .get(watch_id)
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
        skips
            .get(watch_id)
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

        recent
            .keys()
            .map(|watch_id| {
                let errors = recent.get(watch_id).map(|e| e.len()).unwrap_or(0);
                let skipped = skips.get(watch_id).map(|s| s.len()).unwrap_or(0);
                let last_error = recent
                    .get(watch_id)
                    .and_then(|e| e.last())
                    .map(|e| e.timestamp);

                ProcessingErrorSummary {
                    watch_id: watch_id.clone(),
                    recent_error_count: errors,
                    skipped_file_count: skipped,
                    last_error_time: last_error,
                }
            })
            .collect()
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
