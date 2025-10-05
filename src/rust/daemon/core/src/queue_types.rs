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
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval_ms: 500,
            max_retries: 5,
            retry_delays: vec![
                ChronoDuration::milliseconds(1000),
                ChronoDuration::milliseconds(2000),
                ChronoDuration::milliseconds(5000),
                ChronoDuration::milliseconds(10000),
                ChronoDuration::milliseconds(30000),
            ],
            target_throughput: 60,
            enable_metrics: true,
        }
    }
}
