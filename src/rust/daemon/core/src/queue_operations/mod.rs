//! Queue Operations Module
//!
//! Provides Rust interface to the ingestion queue system with full compatibility
//! with Python queue client operations.

mod advanced;
mod dequeue;
mod destination;
mod enqueue;
mod legacy;
mod query;
mod update;
mod validation;

#[cfg(test)]
mod tests;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::collections::HashMap;
use thiserror::Error;

use crate::queue_types::MissingTool;
use crate::unified_queue_schema::QueueOperation as UnifiedOp;

/// Queue operation errors
#[derive(Error, Debug)]
pub enum QueueError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid priority: {0}")]
    InvalidPriority(i32),

    #[error("Invalid operation type: {0}")]
    InvalidOperation(String),

    #[error("Queue item not found: {0}")]
    NotFound(String),

    // Task 46: Strict validation errors
    #[error("tenant_id is required and cannot be empty or whitespace")]
    EmptyTenantId,

    #[error("collection is required and cannot be empty or whitespace")]
    EmptyCollection,

    #[error("Invalid payload JSON: {0}")]
    InvalidPayloadJson(String),

    #[error("Missing required field '{field}' in payload for item_type '{item_type}'")]
    MissingPayloadField { item_type: String, field: String },

    #[error("Internal queue error: {0}")]
    InternalError(String),
}

/// Result type for queue operations
pub type QueueResult<T> = Result<T, QueueError>;

/// Missing metadata queue item representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingMetadataItem {
    pub queue_id: String,
    pub file_absolute_path: String,
    pub collection_name: String,
    pub tenant_id: String,
    pub branch: String,
    pub operation: UnifiedOp,
    pub priority: i32,
    pub missing_tools: Vec<MissingTool>,
    pub queued_timestamp: DateTime<Utc>,
    pub retry_count: i32,
    pub last_check_timestamp: Option<DateTime<Utc>>,
    pub metadata: Option<String>,
}

/// Collection type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CollectionType {
    NonWatched,
    WatchedDynamic,
    WatchedCumulative,
    Project,
}

impl CollectionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            CollectionType::NonWatched => "non-watched",
            CollectionType::WatchedDynamic => "watched-dynamic",
            CollectionType::WatchedCumulative => "watched-cumulative",
            CollectionType::Project => "project",
        }
    }
}

/// Collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub collection_name: String,
    pub collection_type: CollectionType,
    pub created_timestamp: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub configuration: HashMap<String, serde_json::Value>,
    pub tenant_id: String,
    pub branch: String,
}

/// Error message record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub id: Option<i64>,
    pub error_type: String,
    pub error_message: String,
    pub error_details: Option<HashMap<String, serde_json::Value>>,
    pub occurred_timestamp: DateTime<Utc>,
    pub file_path: Option<String>,
    pub collection_name: Option<String>,
    pub retry_count: i32,
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_items: i64,
    pub urgent_items: i64,
    pub high_items: i64,
    pub normal_items: i64,
    pub low_items: i64,
    pub retry_items: i64,
    pub error_items: i64,
    pub unique_collections: i64,
    pub oldest_item: Option<DateTime<Utc>>,
    pub newest_item: Option<DateTime<Utc>>,
}

/// Queue load level for adaptive throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueLoadLevel {
    /// Normal load - no throttling needed
    Normal,
    /// High load - moderate throttling recommended
    High,
    /// Critical load - aggressive throttling required
    Critical,
}

impl QueueLoadLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueLoadLevel::Normal => "normal",
            QueueLoadLevel::High => "high",
            QueueLoadLevel::Critical => "critical",
        }
    }
}

/// Queue throttling summary for adaptive rate control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueThrottlingSummary {
    /// Total items across all collections
    pub total_depth: i64,
    /// Per-collection queue depths
    pub by_collection: HashMap<String, i64>,
    /// Current load level
    pub load_level: QueueLoadLevel,
    /// Suggested polling interval multiplier (1.0-4.0)
    pub throttle_factor: f64,
    /// Threshold for high load
    pub high_threshold: i64,
    /// Threshold for critical load
    pub critical_threshold: i64,
}

#[derive(Clone)]
/// Queue manager for Rust daemon operations
pub struct QueueManager {
    pool: SqlitePool,
}

impl QueueManager {
    /// Create a new queue manager with existing connection pool
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get reference to the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
