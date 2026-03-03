//! `QueueDecision` — pre-computed processing decision stored alongside queue items.

use serde::{Deserialize, Serialize};

/// Pre-computed decision for queue item processing.
///
/// Stored as JSON in `decision_json` column so that retries can skip
/// the decision-making step and proceed directly to execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueDecision {
    /// Whether old Qdrant points should be deleted before upserting new ones
    pub delete_old: bool,
    /// Base point of the previous file version (for reference counting)
    pub old_base_point: Option<String>,
    /// Base point of the new file version
    pub new_base_point: String,
    /// SHA256 hash of the previous file content
    pub old_file_hash: Option<String>,
    /// SHA256 hash of the new file content
    pub new_file_hash: String,
}
