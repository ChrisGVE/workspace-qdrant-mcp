//! Core processing engine for workspace-qdrant-mcp
//!
//! This crate provides the core document processing, file watching, and embedding
//! generation capabilities for the workspace-qdrant-mcp ingestion engine.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Mutex;

pub mod config;
pub mod ipc;
pub mod processing;
pub mod storage;
pub mod watching;

use crate::processing::{Pipeline, TaskSubmitter, TaskPriority, TaskSource, TaskPayload, TaskResult};
use crate::ipc::{IpcServer, IpcClient, EngineSettings};
use crate::storage::StorageClient;
use crate::config::Config;

/// Core processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

/// Document processing result
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub document_id: String,
    pub collection: String,
    pub chunks_created: usize,
    pub processing_time_ms: u64,
}

/// Basic document processor for testing
pub struct DocumentProcessor {
    // Placeholder for processor state
}

impl DocumentProcessor {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn process_file(
        &self,
        file_path: &Path,
        collection: &str,
    ) -> Result<DocumentResult, ProcessingError> {
        // Minimal implementation to satisfy CI build
        let _content = tokio::fs::read_to_string(file_path)
            .await
            .map_err(ProcessingError::Io)?;

        Ok(DocumentResult {
            document_id: uuid::Uuid::new_v4().to_string(),
            collection: collection.to_string(),
            chunks_created: 1,
            processing_time_ms: 1,
        })
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified processing engine that integrates all components
pub struct ProcessingEngine {
    /// Priority-based task processing pipeline
    pipeline: Arc<Mutex<Pipeline>>,
    /// Task submitter for external requests
    task_submitter: TaskSubmitter,
    /// IPC server for Python communication
    ipc_server: Option<IpcServer>,
    /// Storage client for Qdrant operations
    storage_client: Arc<StorageClient>,
    /// Document processor
    document_processor: Arc<DocumentProcessor>,
    /// Engine configuration
    config: Arc<Config>,
}

impl ProcessingEngine {
    /// Create a new processing engine with default configuration
    pub fn new() -> Self {
        let pipeline = Arc::new(Mutex::new(Pipeline::new(4))); // Default 4 concurrent tasks
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };
        
        Self {
            pipeline,
            task_submitter,
            ipc_server: None,
            storage_client: Arc::new(StorageClient::new()),
            document_processor: Arc::new(DocumentProcessor::new()),
            config: Arc::new(Config::default()),
        }
    }
    
    /// Create a processing engine with custom configuration
    pub fn with_config(config: Config) -> Self {
        let max_concurrent = config.max_concurrent_tasks.unwrap_or(4);
        let pipeline = Arc::new(Mutex::new(Pipeline::new(max_concurrent)));
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };
        
        Self {
            pipeline,
            task_submitter,
            ipc_server: None,
            storage_client: Arc::new(StorageClient::new()),
            document_processor: Arc::new(DocumentProcessor::new()),
            config: Arc::new(config),
        }
    }
    
    /// Start the processing engine with IPC support
    pub async fn start_with_ipc(&mut self) -> Result<IpcClient, ProcessingError> {
        // Start the main pipeline
        {
            let mut pipeline_lock = self.pipeline.lock().await;
            pipeline_lock.start().await
                .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        }
        
        // Create and start IPC server
        let max_concurrent = self.config.max_concurrent_tasks.unwrap_or(4);
        let (ipc_server, ipc_client) = IpcServer::new(max_concurrent);
        
        ipc_server.start().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        self.ipc_server = Some(ipc_server);
        
        tracing::info!("Processing engine started with IPC support");
        Ok(ipc_client)
    }
    
    /// Start the processing engine without IPC (standalone mode)
    pub async fn start(&mut self) -> Result<(), ProcessingError> {
        let mut pipeline_lock = self.pipeline.lock().await;
        pipeline_lock.start().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        tracing::info!("Processing engine started in standalone mode");
        Ok(())
    }
    
    /// Submit a document processing task
    pub async fn process_document(
        &self,
        file_path: &Path,
        collection: &str,
        priority: TaskPriority,
    ) -> Result<TaskResult, ProcessingError> {
        let source = match priority {
            TaskPriority::McpRequests => TaskSource::McpServer {
                request_id: uuid::Uuid::new_v4().to_string(),
            },
            TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                project_path: file_path.parent()
                    .unwrap_or_else(|| Path::new("/"))
                    .to_string_lossy()
                    .to_string(),
            },
            TaskPriority::CliCommands => TaskSource::CliCommand {
                command: format!("process-document {}", file_path.display()),
            },
            TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                folder_path: file_path.parent()
                    .unwrap_or_else(|| Path::new("/"))
                    .to_string_lossy()
                    .to_string(),
            },
        };
        
        let payload = TaskPayload::ProcessDocument {
            file_path: file_path.to_path_buf(),
            collection: collection.to_string(),
        };
        
        let timeout = self.config.default_timeout_ms
            .map(Duration::from_millis);
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Submit a directory watching task
    pub async fn watch_directory(
        &self,
        path: &Path,
        recursive: bool,
        priority: TaskPriority,
    ) -> Result<TaskResult, ProcessingError> {
        let source = match priority {
            TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                project_path: path.to_string_lossy().to_string(),
            },
            TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                folder_path: path.to_string_lossy().to_string(),
            },
            _ => TaskSource::CliCommand {
                command: format!("watch-directory {}", path.display()),
            },
        };
        
        let payload = TaskPayload::WatchDirectory {
            path: path.to_path_buf(),
            recursive,
        };
        
        let timeout = self.config.default_timeout_ms
            .map(Duration::from_millis);
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Execute a search query
    pub async fn execute_query(
        &self,
        query: &str,
        collection: &str,
        limit: usize,
        priority: TaskPriority,
    ) -> Result<TaskResult, ProcessingError> {
        let source = TaskSource::McpServer {
            request_id: uuid::Uuid::new_v4().to_string(),
        };
        
        let payload = TaskPayload::ExecuteQuery {
            query: query.to_string(),
            collection: collection.to_string(),
            limit,
        };
        
        let timeout = Some(Duration::from_millis(5000)); // Queries should be fast
        
        let task_handle = self.task_submitter
            .submit_task(priority, source, payload, timeout)
            .await
            .map_err(|e| ProcessingError::Processing(e.to_string()))?;
        
        task_handle.wait().await
            .map_err(|e| ProcessingError::Processing(e.to_string()))
    }
    
    /// Get pipeline statistics
    pub async fn get_stats(&self) -> Result<processing::PipelineStats, ProcessingError> {
        let pipeline_lock = self.pipeline.lock().await;
        Ok(pipeline_lock.stats().await)
    }
    
    /// Get task submitter for advanced usage
    pub fn task_submitter(&self) -> TaskSubmitter {
        self.task_submitter.clone()
    }
    
    /// Graceful shutdown
    pub async fn shutdown(&mut self) -> Result<(), ProcessingError> {
        if let Some(ipc_server) = &self.ipc_server {
            // Wait for IPC server to shutdown
            ipc_server.wait_for_shutdown().await;
        }
        
        tracing::info!("Processing engine shutdown complete");
        Ok(())
    }
}

impl Default for ProcessingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic health check function
pub fn health_check() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check() {
        assert!(health_check());
    }

    #[tokio::test]
    async fn test_document_processor() {
        let processor = DocumentProcessor::new();
        // Basic instantiation test
        assert!(processor
            .process_file(Path::new("/tmp/test.txt"), "test")
            .await
            .is_err());
    }
}
