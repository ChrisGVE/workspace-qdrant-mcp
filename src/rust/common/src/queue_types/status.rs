//! Queue and destination status enums.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Queue item status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueStatus {
    /// Ready to be picked up by processor
    Pending,
    /// Currently being processed (lease acquired)
    InProgress,
    /// Successfully completed
    Done,
    /// Max retries exceeded
    Failed,
}

impl fmt::Display for QueueStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueStatus::Pending => write!(f, "pending"),
            QueueStatus::InProgress => write!(f, "in_progress"),
            QueueStatus::Done => write!(f, "done"),
            QueueStatus::Failed => write!(f, "failed"),
        }
    }
}

impl QueueStatus {
    /// Parse status from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(QueueStatus::Pending),
            "in_progress" => Some(QueueStatus::InProgress),
            "done" => Some(QueueStatus::Done),
            "failed" => Some(QueueStatus::Failed),
            _ => None,
        }
    }
}

/// Per-destination processing status for dual-write queue items.
///
/// Queue items write to both Qdrant (vectors) and the search DB (FTS5).
/// Each destination has its own status, and the item's overall status
/// is derived from both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DestinationStatus {
    /// Not yet processed for this destination
    Pending,
    /// Currently being written to this destination
    InProgress,
    /// Successfully written to this destination
    Done,
    /// Write to this destination failed
    Failed,
}

impl fmt::Display for DestinationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DestinationStatus::Pending => write!(f, "pending"),
            DestinationStatus::InProgress => write!(f, "in_progress"),
            DestinationStatus::Done => write!(f, "done"),
            DestinationStatus::Failed => write!(f, "failed"),
        }
    }
}

impl DestinationStatus {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(DestinationStatus::Pending),
            "in_progress" => Some(DestinationStatus::InProgress),
            "done" => Some(DestinationStatus::Done),
            "failed" => Some(DestinationStatus::Failed),
            _ => None,
        }
    }
}
