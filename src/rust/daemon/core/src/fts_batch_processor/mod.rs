//! Adaptive Batch Processor for FTS5 Code Lines Updates (Task 51)
//!
//! Monitors the unified queue depth and switches between two processing modes:
//!
//! - **Single-file mode** (queue depth <= `burst_threshold`): Applies each file's
//!   line diff immediately, then rebuilds FTS5. Target latency: ~20ms per file.
//!
//! - **Batch mode** (queue depth > `burst_threshold`): Accumulates file changes
//!   in memory, then applies all diffs in a single transaction followed by one
//!   FTS5 rebuild. Throughput: ~95,000 lines/sec.
//!
//! Uses [`line_diff::compute_line_diff`] for change detection and
//! [`SearchDbManager`] for code_lines / FTS5 operations.

mod diff_apply;
mod processor;

pub use processor::FtsBatchProcessor;

/// Default burst threshold: if the queue has more than this many pending file
/// items, switch to batch mode.
pub const DEFAULT_BURST_THRESHOLD: usize = 10;

/// Configuration for the FTS5 batch processor.
#[derive(Debug, Clone)]
pub struct FtsBatchConfig {
    /// Queue depth above which batch mode is used instead of single-file mode.
    pub burst_threshold: usize,
}

impl Default for FtsBatchConfig {
    fn default() -> Self {
        Self {
            burst_threshold: DEFAULT_BURST_THRESHOLD,
        }
    }
}

/// A pending file change to be applied to code_lines.
#[derive(Debug, Clone)]
pub struct FileChange {
    /// The file_id in tracked_files / code_lines.
    pub file_id: i64,
    /// The old content (empty string if new file).
    pub old_content: String,
    /// The new content (empty string if file deleted).
    pub new_content: String,
    /// Tenant ID for file_metadata scoping.
    pub tenant_id: String,
    /// Branch name (optional).
    pub branch: Option<String>,
    /// File path for file_metadata scoping.
    pub file_path: String,
    /// Base point hash for identity model (optional, added in search.db v5).
    pub base_point: Option<String>,
    /// Relative path within project (optional, added in search.db v5).
    pub relative_path: Option<String>,
    /// File content hash (optional, added in search.db v5).
    pub file_hash: Option<String>,
}

/// Statistics from a batch processing run.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Number of files processed.
    pub files_processed: usize,
    /// Total lines inserted across all files.
    pub lines_inserted: usize,
    /// Total lines updated (content changed) across all files.
    pub lines_updated: usize,
    /// Total lines deleted across all files.
    pub lines_deleted: usize,
    /// Total lines unchanged across all files.
    pub lines_unchanged: usize,
    /// Number of files where rebalancing was triggered.
    pub rebalances_triggered: usize,
    /// Whether batch mode was used (vs single-file mode).
    pub batch_mode: bool,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

impl BatchStats {
    /// Total number of lines affected (inserted + updated + deleted).
    pub fn total_affected(&self) -> usize {
        self.lines_inserted + self.lines_updated + self.lines_deleted
    }
}

#[cfg(test)]
mod tests;
