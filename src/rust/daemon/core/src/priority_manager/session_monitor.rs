//! Session Monitor - Background task for orphaned session cleanup.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use super::manager::PriorityManager;
use super::{PriorityError, PriorityResult, SessionMonitorConfig};

/// Session Monitor - Background task for orphaned session cleanup
///
/// Runs periodically to detect and cleanup orphaned sessions where
/// MCP servers died without sending shutdown notifications.
pub struct SessionMonitor {
    priority_manager: PriorityManager,
    config: SessionMonitorConfig,
    cancellation_token: CancellationToken,
    task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
}

impl SessionMonitor {
    /// Create a new session monitor
    pub fn new(priority_manager: PriorityManager, config: SessionMonitorConfig) -> Self {
        Self {
            priority_manager,
            config,
            cancellation_token: CancellationToken::new(),
            task_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(priority_manager: PriorityManager) -> Self {
        Self::new(priority_manager, SessionMonitorConfig::default())
    }

    /// Start the session monitor background task
    pub async fn start(&self) -> PriorityResult<()> {
        let mut handle = self.task_handle.write().await;
        if handle.is_some() {
            return Err(PriorityError::MonitorAlreadyRunning);
        }

        info!(
            "Starting session monitor (heartbeat_timeout={}s, check_interval={}s)",
            self.config.heartbeat_timeout_secs, self.config.check_interval_secs
        );

        let priority_manager = self.priority_manager.clone();
        let timeout_secs = self.config.heartbeat_timeout_secs;
        let check_interval = Duration::from_secs(self.config.check_interval_secs);
        let cancellation_token = self.cancellation_token.clone();

        let task = tokio::spawn(async move {
            loop {
                // Check for cancellation
                if cancellation_token.is_cancelled() {
                    info!("Session monitor shutting down");
                    break;
                }

                // Wait for check interval or cancellation
                tokio::select! {
                    _ = tokio::time::sleep(check_interval) => {
                        // Perform cleanup
                        match priority_manager.cleanup_orphaned_sessions(timeout_secs).await {
                            Ok(cleanup) => {
                                if cleanup.projects_affected > 0 {
                                    info!(
                                        "Session monitor cleanup: {} orphaned sessions",
                                        cleanup.sessions_cleaned
                                    );
                                }
                            }
                            Err(e) => {
                                error!("Session monitor cleanup failed: {}", e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        info!("Session monitor received cancellation signal");
                        break;
                    }
                }
            }
        });

        *handle = Some(task);
        Ok(())
    }

    /// Stop the session monitor
    pub async fn stop(&self) -> PriorityResult<()> {
        let mut handle = self.task_handle.write().await;

        if handle.is_none() {
            return Err(PriorityError::MonitorNotRunning);
        }

        info!("Stopping session monitor...");
        self.cancellation_token.cancel();

        if let Some(task) = handle.take() {
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => {
                    info!("Session monitor stopped cleanly");
                }
                Ok(Err(e)) => {
                    error!("Session monitor task panicked: {}", e);
                }
                Err(_) => {
                    warn!("Session monitor did not stop within timeout, aborting");
                }
            }
        }

        Ok(())
    }

    /// Check if the monitor is running
    pub async fn is_running(&self) -> bool {
        self.task_handle.read().await.is_some()
    }
}
