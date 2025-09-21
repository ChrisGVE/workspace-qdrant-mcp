//! Core daemon implementation for workspace document processing

pub mod core;
pub mod state;
pub mod processing;
pub mod watcher;

use crate::config::DaemonConfig;
use crate::error::{DaemonError, DaemonResult};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Main daemon coordinator
pub struct WorkspaceDaemon {
    config: DaemonConfig,
    state: Arc<RwLock<state::DaemonState>>,
    processing: Arc<processing::DocumentProcessor>,
    watcher: Option<Arc<watcher::FileWatcher>>,
}

impl WorkspaceDaemon {
    /// Create a new daemon instance
    pub async fn new(config: DaemonConfig) -> DaemonResult<Self> {
        // Validate configuration
        config.validate()?;

        info!("Initializing Workspace Daemon with config: {:?}", config);

        // Initialize state management
        let state = Arc::new(RwLock::new(
            state::DaemonState::new(&config.database).await?
        ));

        // Initialize document processor
        let processing = Arc::new(
            processing::DocumentProcessor::new(&config.processing, &config.qdrant).await?
        );

        // Initialize file watcher if enabled
        let watcher = if config.file_watcher.enabled {
            Some(Arc::new(
                watcher::FileWatcher::new(&config.file_watcher, Arc::clone(&processing)).await?
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            state,
            processing,
            watcher,
        })
    }

    /// Start all daemon services
    pub async fn start(&mut self) -> DaemonResult<()> {
        info!("Starting daemon services");

        // Start file watcher if enabled
        if let Some(ref watcher) = self.watcher {
            watcher.start().await?;
            info!("File watcher started");
        }

        info!("All daemon services started successfully");
        Ok(())
    }

    /// Stop all daemon services
    pub async fn stop(&mut self) -> DaemonResult<()> {
        info!("Stopping daemon services");

        // Stop file watcher
        if let Some(ref watcher) = self.watcher {
            watcher.stop().await?;
            info!("File watcher stopped");
        }

        info!("All daemon services stopped");
        Ok(())
    }

    /// Get daemon configuration
    pub fn config(&self) -> &DaemonConfig {
        &self.config
    }

    /// Get daemon state (read-only)
    pub async fn state(&self) -> tokio::sync::RwLockReadGuard<state::DaemonState> {
        self.state.read().await
    }

    /// Get daemon state (read-write)
    pub async fn state_mut(&self) -> tokio::sync::RwLockWriteGuard<state::DaemonState> {
        self.state.write().await
    }

    /// Get document processor
    pub fn processor(&self) -> &Arc<processing::DocumentProcessor> {
        &self.processing
    }

    /// Get file watcher
    pub fn watcher(&self) -> Option<&Arc<watcher::FileWatcher>> {
        self.watcher.as_ref()
    }
}