//! Internal request types and public statistics

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::processing::{PriorityTask, TaskPriority};

/// Queued request with timeout and metadata (internal use only)
#[derive(Debug)]
pub(super) struct QueuedRequest {
    pub task: PriorityTask,
    pub queued_at: Instant,
    pub timeout: Option<Instant>,
    pub content_hash: Option<u64>,
    pub priority_boosted: bool,
    pub original_priority: TaskPriority,
    #[allow(dead_code)]
    pub retry_count: usize,
}

/// Queue statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_queued: usize,
    pub queued_by_priority: HashMap<TaskPriority, usize>,
    pub oldest_request_age_ms: Option<u64>,
    pub timeout_manager_size: usize,
    pub deduplication_map_size: usize,
}
