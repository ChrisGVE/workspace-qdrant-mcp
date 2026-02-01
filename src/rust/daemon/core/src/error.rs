//! Comprehensive error handling for workspace-qdrant-mcp
//!
//! This module provides structured error types with detailed context,
//! recovery strategies, and circuit breaker patterns for reliable operations.

use std::fmt;
use std::time::Duration;
use thiserror::Error;
use tracing::{error, warn, info};

/// Main error type for the workspace-qdrant-mcp system
#[derive(Error, Debug)]
pub enum WorkspaceError {
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Network error: {message} (attempt {attempt}/{max_attempts})")]
    Network {
        message: String,
        attempt: u32,
        max_attempts: u32,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Qdrant connection error: {message}")]
    QdrantConnection {
        message: String,
        url: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("File system error: {message} (path: {path})")]
    FileSystem {
        message: String,
        path: String,
        operation: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Document processing error: {message} (file: {file_path})")]
    DocumentProcessing {
        message: String,
        file_path: String,
        document_type: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Embedding generation error: {message} (model: {model})")]
    Embedding {
        message: String,
        model: String,
        retry_count: u32,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("IPC communication error: {message} (endpoint: {endpoint})")]
    IpcCommunication {
        message: String,
        endpoint: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Task processing error: {message} (task_id: {task_id})")]
    TaskProcessing {
        message: String,
        task_id: String,
        priority: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Validation error: {message}")]
    Validation {
        message: String,
        field: Option<String>,
    },

    #[error("Timeout error: {message} (duration: {duration_ms}ms)")]
    Timeout {
        message: String,
        duration_ms: u64,
        operation: String,
    },

    #[error("Resource exhaustion: {message}")]
    ResourceExhaustion {
        message: String,
        resource_type: String,
        current_usage: Option<u64>,
        limit: Option<u64>,
    },

    #[error("Circuit breaker open: {service} is unavailable")]
    CircuitBreakerOpen {
        service: String,
        failure_count: u32,
        last_failure: String,
    },

    #[error("Authentication error: {message}")]
    Authentication {
        message: String,
        service: String,
    },

    #[error("Internal error: {message}")]
    Internal {
        message: String,
        component: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

/// Daemon-specific errors for internal operations.
#[derive(Error, Debug)]
pub enum DaemonError {
    /// Parse error during source code analysis.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error message.
    #[error("{0}")]
    Other(String),
}

impl WorkspaceError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error with source
    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>, attempt: u32, max_attempts: u32) -> Self {
        Self::Network {
            message: message.into(),
            attempt,
            max_attempts,
            source: None,
        }
    }

    /// Create a Qdrant connection error
    pub fn qdrant_connection(message: impl Into<String>, url: impl Into<String>) -> Self {
        Self::QdrantConnection {
            message: message.into(),
            url: url.into(),
            source: None,
        }
    }

    /// Create a file system error
    pub fn file_system(
        message: impl Into<String>,
        path: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self::FileSystem {
            message: message.into(),
            path: path.into(),
            operation: operation.into(),
            source: None,
        }
    }

    /// Create a document processing error
    pub fn document_processing(
        message: impl Into<String>,
        file_path: impl Into<String>,
        document_type: impl Into<String>,
    ) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
            file_path: file_path.into(),
            document_type: document_type.into(),
            source: None,
        }
    }

    /// Create a timeout error
    pub fn timeout(
        message: impl Into<String>,
        duration: Duration,
        operation: impl Into<String>,
    ) -> Self {
        Self::Timeout {
            message: message.into(),
            duration_ms: duration.as_millis() as u64,
            operation: operation.into(),
        }
    }

    /// Create a circuit breaker error
    pub fn circuit_breaker_open(
        service: impl Into<String>,
        failure_count: u32,
        last_failure: impl Into<String>,
    ) -> Self {
        Self::CircuitBreakerOpen {
            service: service.into(),
            failure_count,
            last_failure: last_failure.into(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::QdrantConnection { .. } => true,
            Self::Timeout { .. } => true,
            Self::ResourceExhaustion { .. } => true,
            Self::IpcCommunication { .. } => true,
            Self::Embedding { .. } => true,
            Self::Configuration { .. } => false,
            Self::Validation { .. } => false,
            Self::Authentication { .. } => false,
            Self::CircuitBreakerOpen { .. } => false,
            Self::FileSystem { operation, .. } => operation != "delete",
            Self::DocumentProcessing { .. } => false,
            Self::TaskProcessing { .. } => false,
            Self::Internal { .. } => false,
        }
    }

    /// Get suggested retry delay
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            Self::Network { attempt, .. } => {
                Some(Duration::from_millis(1000 * (2_u64.pow(*attempt - 1))))
            }
            Self::QdrantConnection { .. } => Some(Duration::from_secs(2)),
            Self::Timeout { .. } => Some(Duration::from_millis(500)),
            Self::ResourceExhaustion { .. } => Some(Duration::from_secs(5)),
            Self::IpcCommunication { .. } => Some(Duration::from_millis(100)),
            Self::Embedding { retry_count, .. } => {
                Some(Duration::from_millis(500 * (2_u64.pow(*retry_count))))
            }
            Self::FileSystem { .. } => Some(Duration::from_millis(100)),
            _ => None,
        }
    }

    /// Get error category for monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::Configuration { .. } => "configuration",
            Self::Network { .. } => "network",
            Self::QdrantConnection { .. } => "database",
            Self::FileSystem { .. } => "filesystem",
            Self::DocumentProcessing { .. } => "processing",
            Self::Embedding { .. } => "embedding",
            Self::IpcCommunication { .. } => "ipc",
            Self::TaskProcessing { .. } => "task",
            Self::Validation { .. } => "validation",
            Self::Timeout { .. } => "timeout",
            Self::ResourceExhaustion { .. } => "resource",
            Self::CircuitBreakerOpen { .. } => "circuit_breaker",
            Self::Authentication { .. } => "auth",
            Self::Internal { .. } => "internal",
        }
    }

    /// Get severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Configuration { .. } => ErrorSeverity::High,
            Self::Authentication { .. } => ErrorSeverity::High,
            Self::CircuitBreakerOpen { .. } => ErrorSeverity::High,
            Self::Internal { .. } => ErrorSeverity::High,
            Self::Network { .. } => ErrorSeverity::Medium,
            Self::QdrantConnection { .. } => ErrorSeverity::Medium,
            Self::ResourceExhaustion { .. } => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::FileSystem { .. } => ErrorSeverity::Low,
            Self::DocumentProcessing { .. } => ErrorSeverity::Low,
            Self::Embedding { .. } => ErrorSeverity::Low,
            Self::IpcCommunication { .. } => ErrorSeverity::Low,
            Self::TaskProcessing { .. } => ErrorSeverity::Low,
            Self::Validation { .. } => ErrorSeverity::Low,
        }
    }

    /// Convert error to dictionary for structured logging
    pub fn to_dict(&self) -> std::collections::HashMap<String, String> {
        let mut dict = std::collections::HashMap::new();
        
        dict.insert("error_type".to_string(), self.category().to_string());
        dict.insert("severity".to_string(), self.severity().to_string());
        dict.insert("retryable".to_string(), self.is_retryable().to_string());
        dict.insert("message".to_string(), self.to_string());
        
        match self {
            Self::Configuration { message, .. } => {
                dict.insert("config_message".to_string(), message.clone());
            }
            Self::Network { message, attempt, max_attempts, .. } => {
                dict.insert("network_message".to_string(), message.clone());
                dict.insert("attempt".to_string(), attempt.to_string());
                dict.insert("max_attempts".to_string(), max_attempts.to_string());
            }
            Self::QdrantConnection { message, url, .. } => {
                dict.insert("qdrant_message".to_string(), message.clone());
                dict.insert("url".to_string(), url.clone());
            }
            Self::FileSystem { message, path, operation, .. } => {
                dict.insert("fs_message".to_string(), message.clone());
                dict.insert("path".to_string(), path.clone());
                dict.insert("operation".to_string(), operation.clone());
            }
            Self::DocumentProcessing { message, file_path, document_type, .. } => {
                dict.insert("doc_message".to_string(), message.clone());
                dict.insert("file_path".to_string(), file_path.clone());
                dict.insert("document_type".to_string(), document_type.clone());
            }
            Self::Embedding { message, model, retry_count, .. } => {
                dict.insert("emb_message".to_string(), message.clone());
                dict.insert("model".to_string(), model.clone());
                dict.insert("retry_count".to_string(), retry_count.to_string());
            }
            Self::IpcCommunication { message, endpoint, .. } => {
                dict.insert("ipc_message".to_string(), message.clone());
                dict.insert("endpoint".to_string(), endpoint.clone());
            }
            Self::TaskProcessing { message, task_id, priority, .. } => {
                dict.insert("task_message".to_string(), message.clone());
                dict.insert("task_id".to_string(), task_id.clone());
                dict.insert("priority".to_string(), priority.clone());
            }
            Self::Validation { message, field } => {
                dict.insert("val_message".to_string(), message.clone());
                if let Some(field) = field {
                    dict.insert("field".to_string(), field.clone());
                }
            }
            Self::Timeout { message, duration_ms, operation } => {
                dict.insert("timeout_message".to_string(), message.clone());
                dict.insert("duration_ms".to_string(), duration_ms.to_string());
                dict.insert("operation".to_string(), operation.clone());
            }
            Self::ResourceExhaustion { message, resource_type, current_usage, limit } => {
                dict.insert("resource_message".to_string(), message.clone());
                dict.insert("resource_type".to_string(), resource_type.clone());
                if let Some(usage) = current_usage {
                    dict.insert("current_usage".to_string(), usage.to_string());
                }
                if let Some(limit) = limit {
                    dict.insert("limit".to_string(), limit.to_string());
                }
            }
            Self::CircuitBreakerOpen { service, failure_count, last_failure } => {
                dict.insert("service".to_string(), service.clone());
                dict.insert("failure_count".to_string(), failure_count.to_string());
                dict.insert("last_failure".to_string(), last_failure.clone());
            }
            Self::Authentication { message, service } => {
                dict.insert("auth_message".to_string(), message.clone());
                dict.insert("service".to_string(), service.clone());
            }
            Self::Internal { message, component, .. } => {
                dict.insert("internal_message".to_string(), message.clone());
                dict.insert("component".to_string(), component.clone());
            }
        }
        
        dict
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub struct ErrorRecoveryStrategy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub exponential_backoff: bool,
    pub circuit_breaker_threshold: Option<u32>,
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(5),
        }
    }
}

impl ErrorRecoveryStrategy {
    /// Create a recovery strategy for network operations
    pub fn network() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(10),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(3),
        }
    }

    /// Create a recovery strategy for database operations
    pub fn database() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(5),
        }
    }

    /// Create a recovery strategy for file operations
    pub fn file_operations() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(1),
            exponential_backoff: false,
            circuit_breaker_threshold: None,
        }
    }

    /// Create a recovery strategy for embedding generation
    pub fn embedding() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(60),
            exponential_backoff: true,
            circuit_breaker_threshold: Some(10),
        }
    }

    /// Calculate delay for given attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if !self.exponential_backoff {
            return self.base_delay;
        }

        let delay = self.base_delay.as_millis() as u64 * (2_u64.pow(attempt.saturating_sub(1)));
        Duration::from_millis(delay.min(self.max_delay.as_millis() as u64))
    }
}

/// Circuit breaker implementation for external service calls
#[derive(Debug)]
pub struct CircuitBreaker {
    name: String,
    failure_threshold: u32,
    timeout: Duration,
    reset_timeout: Duration,
    state: CircuitBreakerState,
    failure_count: u32,
    last_failure_time: Option<std::time::Instant>,
    last_success_time: Option<std::time::Instant>,
}

#[derive(Debug, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: impl Into<String>, failure_threshold: u32) -> Self {
        Self {
            name: name.into(),
            failure_threshold,
            timeout: Duration::from_secs(10),
            reset_timeout: Duration::from_secs(60),
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            last_failure_time: None,
            last_success_time: None,
        }
    }

    /// Execute an operation with circuit breaker protection
    pub async fn execute<F, T, E>(&mut self, operation: F) -> std::result::Result<T, WorkspaceError>
    where
        F: std::future::Future<Output = std::result::Result<T, E>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.reset_timeout {
                        info!("Circuit breaker transitioning to half-open: {}", self.name);
                        self.state = CircuitBreakerState::HalfOpen;
                    } else {
                        return Err(WorkspaceError::circuit_breaker_open(
                            &self.name,
                            self.failure_count,
                            "Circuit breaker is open",
                        ));
                    }
                } else {
                    return Err(WorkspaceError::circuit_breaker_open(
                        &self.name,
                        self.failure_count,
                        "Circuit breaker is open",
                    ));
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow one test request
            }
            CircuitBreakerState::Closed => {
                // Normal operation
            }
        }

        // Execute the operation with timeout
        let result = tokio::time::timeout(self.timeout, operation).await;

        match result {
            Ok(Ok(value)) => {
                self.on_success();
                Ok(value)
            }
            Ok(Err(error)) => {
                self.on_failure();
                Err(WorkspaceError::Internal {
                    message: format!("Operation failed: {}", error),
                    component: self.name.clone(),
                    source: Some(Box::new(error)),
                })
            }
            Err(_) => {
                self.on_failure();
                Err(WorkspaceError::timeout(
                    format!("Operation timed out in circuit breaker: {}", self.name),
                    self.timeout,
                    &self.name,
                ))
            }
        }
    }

    fn on_success(&mut self) {
        self.failure_count = 0;
        self.last_success_time = Some(std::time::Instant::now());
        self.state = CircuitBreakerState::Closed;
        
        if self.state == CircuitBreakerState::HalfOpen {
            info!("Circuit breaker recovered: {}", self.name);
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(std::time::Instant::now());

        if self.failure_count >= self.failure_threshold {
            warn!(
                "Circuit breaker opened due to {} failures: {}",
                self.failure_count, self.name
            );
            self.state = CircuitBreakerState::Open;
        }
    }

    /// Get circuit breaker status
    pub fn status(&self) -> CircuitBreakerStatus {
        CircuitBreakerStatus {
            name: self.name.clone(),
            state: match self.state {
                CircuitBreakerState::Closed => "closed".to_string(),
                CircuitBreakerState::Open => "open".to_string(),
                CircuitBreakerState::HalfOpen => "half-open".to_string(),
            },
            failure_count: self.failure_count,
            failure_threshold: self.failure_threshold,
            last_failure_time: self.last_failure_time,
            last_success_time: self.last_success_time,
        }
    }
}

/// Circuit breaker status for monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub name: String,
    pub state: String,
    pub failure_count: u32,
    pub failure_threshold: u32,
    pub last_failure_time: Option<std::time::Instant>,
    pub last_success_time: Option<std::time::Instant>,
}

/// Error monitoring and alerting hooks
pub trait ErrorMonitor: Send + Sync {
    /// Report an error for monitoring
    fn report_error(&self, error: &WorkspaceError, context: Option<&str>);

    /// Report error recovery
    fn report_recovery(&self, error_category: &str, attempt: u32);

    /// Report circuit breaker state change
    fn report_circuit_breaker_state(&self, name: &str, state: &str);

    /// Get error statistics
    fn get_error_stats(&self) -> ErrorStats;
}

/// Error statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct ErrorStats {
    pub total_errors: u64,
    pub errors_by_category: std::collections::HashMap<String, u64>,
    pub retryable_errors: u64,
    pub non_retryable_errors: u64,
    pub recovery_successes: u64,
    pub circuit_breaker_opens: u64,
}

/// Default error monitor implementation
pub struct DefaultErrorMonitor {
    stats: std::sync::Arc<tokio::sync::Mutex<ErrorStats>>,
}

impl DefaultErrorMonitor {
    pub fn new() -> Self {
        Self {
            stats: std::sync::Arc::new(tokio::sync::Mutex::new(ErrorStats::default())),
        }
    }
}

impl ErrorMonitor for DefaultErrorMonitor {
    fn report_error(&self, error: &WorkspaceError, context: Option<&str>) {
        let category = error.category();
        let severity = error.severity();
        let is_retryable = error.is_retryable();

        match severity {
            ErrorSeverity::Low => info!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
            ErrorSeverity::Medium => warn!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
            ErrorSeverity::High | ErrorSeverity::Critical => error!(
                error_category = category,
                retryable = is_retryable,
                context = context.unwrap_or("none"),
                "Error reported: {}", error
            ),
        }

        // Update statistics (in production, this would be async)
        if let Ok(mut stats) = self.stats.try_lock() {
            stats.total_errors += 1;
            *stats.errors_by_category.entry(category.to_string()).or_insert(0) += 1;
            if is_retryable {
                stats.retryable_errors += 1;
            } else {
                stats.non_retryable_errors += 1;
            }
        }
    }

    fn report_recovery(&self, error_category: &str, attempt: u32) {
        info!(
            error_category = error_category,
            attempt = attempt,
            "Error recovery succeeded"
        );

        if let Ok(mut stats) = self.stats.try_lock() {
            stats.recovery_successes += 1;
        }
    }

    fn report_circuit_breaker_state(&self, name: &str, state: &str) {
        match state {
            "open" => {
                warn!(circuit_breaker = name, "Circuit breaker opened");
                if let Ok(mut stats) = self.stats.try_lock() {
                    stats.circuit_breaker_opens += 1;
                }
            }
            "closed" => info!(circuit_breaker = name, "Circuit breaker closed"),
            "half-open" => info!(circuit_breaker = name, "Circuit breaker half-open"),
            _ => warn!(circuit_breaker = name, state = state, "Unknown circuit breaker state"),
        }
    }

    fn get_error_stats(&self) -> ErrorStats {
        if let Ok(stats) = self.stats.try_lock() {
            stats.clone()
        } else {
            ErrorStats::default()
        }
    }
}

impl Default for DefaultErrorMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Error recovery utility with built-in retry logic
pub struct ErrorRecovery {
    monitor: Box<dyn ErrorMonitor>,
    circuit_breakers: std::collections::HashMap<String, CircuitBreaker>,
}

impl ErrorRecovery {
    /// Create a new error recovery system
    pub fn new(monitor: Box<dyn ErrorMonitor>) -> Self {
        Self {
            monitor,
            circuit_breakers: std::collections::HashMap::new(),
        }
    }

    /// Execute an operation with automatic retry and circuit breaker protection
    pub async fn execute_with_retry<F, T, E>(
        &mut self,
        operation_name: &str,
        operation: F,
        strategy: ErrorRecoveryStrategy,
    ) -> std::result::Result<T, WorkspaceError>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send + 'static>>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut attempt = 1;

        loop {
            // Get or create circuit breaker
            if !self.circuit_breakers.contains_key(operation_name) {
                if let Some(threshold) = strategy.circuit_breaker_threshold {
                    self.circuit_breakers.insert(
                        operation_name.to_string(),
                        CircuitBreaker::new(operation_name, threshold),
                    );
                }
            }

            let result = if let Some(circuit_breaker) = self.circuit_breakers.get_mut(operation_name)
            {
                circuit_breaker.execute(operation()).await
            } else {
                match operation().await {
                    Ok(value) => Ok(value),
                    Err(error) => Err(WorkspaceError::Internal {
                        message: format!("Operation failed: {}", error),
                        component: operation_name.to_string(),
                        source: Some(Box::new(error)),
                    }),
                }
            };

            match result {
                Ok(value) => {
                    if attempt > 1 {
                        self.monitor.report_recovery(operation_name, attempt);
                    }
                    return Ok(value);
                }
                Err(error) => {
                    self.monitor.report_error(&error, Some(operation_name));

                    if !error.is_retryable() || attempt >= strategy.max_retries {
                        return Err(error);
                    }

                    let delay = strategy.calculate_delay(attempt);
                    warn!(
                        operation = operation_name,
                        attempt = attempt,
                        max_attempts = strategy.max_retries,
                        delay_ms = delay.as_millis(),
                        "Retrying operation after error: {}", error
                    );

                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Get circuit breaker status for monitoring
    pub fn get_circuit_breaker_status(&self, name: &str) -> Option<CircuitBreakerStatus> {
        self.circuit_breakers.get(name).map(|cb| cb.status())
    }

    /// Get all circuit breaker statuses
    pub fn get_all_circuit_breaker_statuses(&self) -> Vec<CircuitBreakerStatus> {
        self.circuit_breakers
            .values()
            .map(|cb| cb.status())
            .collect()
    }

    /// Get error statistics
    pub fn get_error_stats(&self) -> ErrorStats {
        self.monitor.get_error_stats()
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new(Box::new(DefaultErrorMonitor::new()))
    }
}

// Note: ProcessingError will be defined in processing module

// Conversion from UnifiedConfigError
impl From<crate::unified_config::UnifiedConfigError> for WorkspaceError {
    fn from(error: crate::unified_config::UnifiedConfigError) -> Self {
        WorkspaceError::Configuration {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

// Type alias for common result type
pub type Result<T> = std::result::Result<T, WorkspaceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = WorkspaceError::configuration("Test configuration error");
        assert_eq!(error.category(), "configuration");
        assert!(!error.is_retryable());
        assert_eq!(error.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_recovery_strategy() {
        let strategy = ErrorRecoveryStrategy::network();
        assert_eq!(strategy.max_retries, 5);
        assert!(strategy.exponential_backoff);

        let delay = strategy.calculate_delay(2);
        assert_eq!(delay, Duration::from_millis(1000));
    }

    #[derive(Debug, thiserror::Error)]
    enum TestError {
        #[error("Test error: {0}")]
        Test(String),
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut circuit_breaker = CircuitBreaker::new("test", 2);
        
        // Simulate failures
        let result1 = circuit_breaker
            .execute(async { Err::<(), TestError>(TestError::Test("test error".to_string())) })
            .await;
        assert!(result1.is_err());

        let result2 = circuit_breaker
            .execute(async { Err::<(), TestError>(TestError::Test("test error".to_string())) })
            .await;
        assert!(result2.is_err());

        // Circuit should be open now
        let status = circuit_breaker.status();
        assert_eq!(status.state, "open");
        assert_eq!(status.failure_count, 2);
    }

    #[test]
    fn test_error_monitor() {
        let monitor = DefaultErrorMonitor::new();
        let error = WorkspaceError::network("Test network error", 1, 3);
        
        monitor.report_error(&error, Some("test context"));
        monitor.report_recovery("network", 2);

        let stats = monitor.get_error_stats();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.recovery_successes, 1);
        assert!(stats.errors_by_category.contains_key("network"));
    }
}