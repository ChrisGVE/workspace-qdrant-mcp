//! Shared types for queue processing
//!
//! Types used across multiple queue-related modules.

use chrono::Duration as ChronoDuration;
use serde::{Deserialize, Serialize};

/// Enumeration of tools that might be missing during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingTool {
    /// LSP server not available for the specified language
    LspServer { language: String },
    /// Tree-sitter parser not available for the specified language
    TreeSitterParser { language: String },
    /// Embedding model not loaded or unavailable
    EmbeddingModel { reason: String },
    /// Qdrant connection unavailable
    QdrantConnection { reason: String },
}

impl std::fmt::Display for MissingTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MissingTool::LspServer { language } => {
                write!(f, "LSP server unavailable for language: {}", language)
            }
            MissingTool::TreeSitterParser { language } => {
                write!(f, "Tree-sitter parser unavailable for language: {}", language)
            }
            MissingTool::EmbeddingModel { reason } => {
                write!(f, "Embedding model unavailable: {}", reason)
            }
            MissingTool::QdrantConnection { reason } => {
                write!(f, "Qdrant connection unavailable: {}", reason)
            }
        }
    }
}

/// Configuration for the queue processor
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Number of items to dequeue in each batch
    pub batch_size: i32,

    /// Poll interval between batches (milliseconds)
    pub poll_interval_ms: u64,

    /// Maximum number of retry attempts
    pub max_retries: i32,

    /// Retry delay intervals (exponential backoff)
    pub retry_delays: Vec<ChronoDuration>,

    /// Target processing throughput (docs per minute)
    pub target_throughput: u64,

    /// Enable performance monitoring
    pub enable_metrics: bool,

    /// Number of parallel workers for batch processing (Task 21)
    /// Higher values increase throughput but use more resources
    pub worker_count: usize,

    /// Maximum queue depth before enabling backpressure (Task 21)
    /// When exceeded, enqueue operations may be slowed
    pub backpressure_threshold: i64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval_ms: 500,
            max_retries: 5,
            retry_delays: vec![
                ChronoDuration::minutes(1),
                ChronoDuration::minutes(5),
                ChronoDuration::minutes(15),
                ChronoDuration::hours(1),
            ],
            target_throughput: 1000, // 1000+ docs/min
            enable_metrics: true,
            worker_count: 4,  // Default to 4 parallel workers
            backpressure_threshold: 1000,  // Start backpressure at 1000 items
        }
    }
}
