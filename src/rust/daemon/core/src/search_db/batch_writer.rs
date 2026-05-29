//! FTS5 batch writer actor.
//!
//! Decouples per-file FTS5 work from the per-item queue handler. Workers
//! prepare a `Fts5WorkItem` (file content read + hash + diff base already
//! loaded) and `send` it through an mpsc channel. A single long-lived
//! background task drains the channel, accumulates work into batches
//! sized by `BATCH_SIZE` or aged by `BATCH_TIMEOUT`, and commits the
//! whole batch in **one** search.db transaction.
//!
//! Why: under concurrent load the per-item code path was opening one
//! transaction per file against search.db, which collided on the SQLite
//! write lock and surfaced as `database is locked (code 5)` — driving
//! `search_status='failed'` on hundreds of items per minute. With one
//! actor serializing the commit, `SQLITE_BUSY` disappears entirely and
//! the throughput is bounded by FTS5 work itself, not lock contention.
//!
//! The actor takes responsibility for the full post-batch handshake:
//! upserting `indexed_content` cache rows, flipping `search_status` to
//! `done` / `failed`, and calling `check_and_finalize` so completed
//! items leave `unified_queue` without waiting for the next dequeue.

use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::OnceCell;
use sqlx::SqlitePool;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use crate::indexed_content_schema;
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{DestinationStatus, QueueStatus};

use super::SearchDbManager;

/// Error message recorded when an FTS5 batch reports a per-item failure.
///
/// The `[transient_fts5]` prefix is load-bearing: it makes the item eligible
/// for the idle resurrection pass (`QueueManager::resurrect_failed_transient`,
/// which selects `WHERE error_message LIKE '[transient_%'`). FTS5 batch
/// failures are typically transient — historically `SQLITE_BUSY` write-lock
/// contention (see the module doc), or a poisoned sibling in the same batch.
/// Resurrection bounds retries via `max_resurrections`, then promotes the row
/// to `[permanent_exhausted]`. WITHOUT the prefix the item is neither
/// resurrected nor triaged and sits in `failed` forever.
fn fts5_failure_message(queue_id: &str) -> String {
    format!("[transient_fts5] FTS5 batch reported search_status=failed for queue_id={queue_id}")
}

/// Global sender installed by the daemon's `UnifiedQueueProcessor::with_search_db`.
///
/// Per-item file handlers ([`crate::strategies::processing::file::ingest`])
/// consult `global_sender()` to decide whether to enqueue FTS5 work to the
/// batch actor or fall back to the inline write path. Lookup is cheap
/// (single atomic load) so it's safe to call on every file.
///
/// Why a global instead of plumbing through `ProcessingContext`: the
/// dispatch chain (`UnifiedQueueProcessor` → `dispatch_nonempty_batch` →
/// `process_batch` → `process_item` → ingest) already threads ~12
/// parameters, and the sender is daemon-singleton state, not per-request
/// state. A `OnceCell` matches how `monitoring::METRICS` is structured
/// elsewhere in the daemon.
static FTS5_SENDER: OnceCell<Fts5Sender> = OnceCell::new();

/// Install the FTS5 batch-writer sender as the daemon-wide default.
///
/// Returns `Err(sender)` if a sender was already installed — the caller
/// can decide whether that's a real bug or a benign re-init (e.g.,
/// test setup running `with_search_db` twice). Production daemons call
/// this exactly once.
pub fn install_global_sender(sender: Fts5Sender) -> Result<(), Fts5Sender> {
    FTS5_SENDER.set(sender)
}

/// Look up the daemon-wide FTS5 sender. Returns `None` when no batch
/// writer has been installed (test daemons, library-only mode, or the
/// search_db feature disabled).
pub fn global_sender() -> Option<&'static Fts5Sender> {
    FTS5_SENDER.get()
}

/// Channel capacity. ~20× the batch size — large enough to absorb the
/// burst of an entire batch's worth of items being prepared concurrently
/// by multiple workers, but small enough that backpressure kicks in if
/// the actor genuinely can't keep up.
pub const FTS5_CHANNEL_CAPACITY: usize = 1024;

/// Maximum files per transaction. Each file's diff is bounded so 50
/// files keeps a single batch under ~200 KB of SQL traffic on average.
pub const FTS5_BATCH_SIZE: usize = 50;

/// Flush a partial batch after this long, even if it hasn't reached
/// `FTS5_BATCH_SIZE`. Keeps latency bounded during low-volume periods.
pub const FTS5_BATCH_TIMEOUT: Duration = Duration::from_millis(500);

/// A single file's FTS5 work, prepared by a queue worker and sent to
/// the actor for batched commit.
///
/// The worker performs the disk read + content hash + old-content lookup
/// up-front, so the actor only does database work. This keeps file IO
/// parallel across workers while serializing writes through one actor.
#[derive(Debug)]
pub struct Fts5WorkItem {
    /// Prepared `FileChange` (file_id, old/new content, tenant, branch, path, hash, etc.)
    pub change: FileChange,
    /// Raw bytes of new content — used to update `indexed_content` cache after commit.
    pub new_content_bytes: Vec<u8>,
    /// Hash of new content — paired with `new_content_bytes` for `indexed_content`.
    pub new_hash: String,
    /// Queue item this work belongs to. The actor uses this to flip
    /// `search_status` and finalize the row after the batch commits.
    pub queue_id: String,
}

/// Sender side of the FTS5 work channel. Cloneable handle stored in
/// `ProcessingContext::fts5_sender` when batched mode is enabled.
pub type Fts5Sender = mpsc::Sender<Fts5WorkItem>;

/// Spawn the batch writer actor and return its sender.
///
/// The actor runs until the channel is dropped (i.e., until every
/// `Fts5Sender` clone is dropped, which only happens at daemon shutdown
/// because the sender lives in `ProcessingContext`).
pub fn spawn(
    search_db: Arc<SearchDbManager>,
    state_pool: SqlitePool,
    queue_manager: Arc<QueueManager>,
) -> Fts5Sender {
    let (tx, rx) = mpsc::channel::<Fts5WorkItem>(FTS5_CHANNEL_CAPACITY);
    let writer = Fts5BatchWriter {
        rx,
        search_db,
        state_pool,
        queue_manager,
    };
    tokio::spawn(writer.run());
    tx
}

struct Fts5BatchWriter {
    rx: mpsc::Receiver<Fts5WorkItem>,
    search_db: Arc<SearchDbManager>,
    state_pool: SqlitePool,
    queue_manager: Arc<QueueManager>,
}

impl Fts5BatchWriter {
    async fn run(mut self) {
        info!(
            "FTS5 batch writer started (batch_size={}, batch_timeout={:?}, channel_capacity={})",
            FTS5_BATCH_SIZE, FTS5_BATCH_TIMEOUT, FTS5_CHANNEL_CAPACITY
        );

        let mut buf: Vec<Fts5WorkItem> = Vec::with_capacity(FTS5_BATCH_SIZE);
        let mut ticker = tokio::time::interval(FTS5_BATCH_TIMEOUT);
        // Skip the immediate first tick — we want timeouts to elapse from
        // the moment buf first becomes non-empty, not from actor start.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let _ = ticker.tick().await;

        loop {
            tokio::select! {
                maybe_item = self.rx.recv() => {
                    match maybe_item {
                        Some(item) => {
                            buf.push(item);
                            if buf.len() >= FTS5_BATCH_SIZE {
                                self.flush(&mut buf).await;
                            }
                        }
                        None => {
                            // All senders dropped — drain remaining buffer and exit.
                            if !buf.is_empty() {
                                self.flush(&mut buf).await;
                            }
                            info!("FTS5 batch writer stopping (channel closed)");
                            return;
                        }
                    }
                }
                _ = ticker.tick() => {
                    if !buf.is_empty() {
                        self.flush(&mut buf).await;
                    }
                }
            }
        }
    }

    /// Commit one batch in a single search.db transaction, then update
    /// `indexed_content` cache and finalize each queue item.
    ///
    /// On batch error every item in the batch is marked `search_status=failed`
    /// and `mark_unified_failed` is called for it. This matches the previous
    /// per-item failure semantics — the unified queue's retry path (backoff
    /// via `lease_until`, retry_count) takes over from there.
    async fn flush(&self, buf: &mut Vec<Fts5WorkItem>) {
        let items = std::mem::take(buf);
        let n = items.len();
        let start = std::time::Instant::now();

        // Force batch mode: pass queue_depth=usize::MAX so the processor
        // always picks the single-transaction path regardless of the
        // configured burst_threshold.
        let mut processor = FtsBatchProcessor::new(&self.search_db, FtsBatchConfig::default());
        for item in &items {
            processor.add_change(item.change.clone());
        }

        match processor.flush(usize::MAX).await {
            Ok(stats) => {
                debug!(
                    "FTS5 batch committed: {} files, {} inserted/{} updated/{} deleted lines in {}ms",
                    stats.files_processed,
                    stats.lines_inserted,
                    stats.lines_updated,
                    stats.lines_deleted,
                    stats.processing_time_ms
                );
                self.finalize_success(&items).await;
            }
            Err(e) => {
                warn!(
                    "FTS5 batch failed ({} items): {} — marking search_status=failed and letting unified_queue retry",
                    n, e
                );
                self.finalize_failure(&items, &e.to_string()).await;
            }
        }

        debug!(
            "FTS5 batch flush of {} items in {}ms",
            n,
            start.elapsed().as_millis()
        );
    }

    /// Post-commit work: update `indexed_content` cache, mark search=done,
    /// finalize each queue item.
    async fn finalize_success(&self, items: &[Fts5WorkItem]) {
        for item in items {
            // Best-effort indexed_content cache update. Failures here are
            // logged but don't change the destination status — the FTS5
            // commit already succeeded.
            if let Err(e) = indexed_content_schema::upsert_indexed_content(
                &self.state_pool,
                item.change.file_id,
                &item.new_content_bytes,
                &item.new_hash,
            )
            .await
            {
                warn!(
                    "indexed_content upsert failed for file_id={} ({}): {}",
                    item.change.file_id, item.change.file_path, e
                );
            }

            if let Err(e) = self
                .queue_manager
                .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
                .await
            {
                error!(
                    "update_destination_status(search=Done) failed for queue_id={}: {}",
                    item.queue_id, e
                );
                continue;
            }

            self.finalize_one(&item.queue_id).await;
        }
    }

    async fn finalize_failure(&self, items: &[Fts5WorkItem], batch_err: &str) {
        for item in items {
            if let Err(e) = self
                .queue_manager
                .update_destination_status(&item.queue_id, "search", DestinationStatus::Failed)
                .await
            {
                error!(
                    "update_destination_status(search=Failed) failed for queue_id={}: {}",
                    item.queue_id, e
                );
                continue;
            }

            self.finalize_one(&item.queue_id).await;
            // No need to call mark_unified_failed explicitly — `finalize_one`
            // routes through `check_and_finalize`, which returns Failed when
            // either destination reports failed, and the caller (this loop)
            // already logged the batch error context.
            let _ = batch_err; // currently unused; reserved for richer error_message
        }
    }

    /// Resolve a single queue item's overall status now that we've updated
    /// `search_status`. Mirrors `handle_item_success` in batch_processing.rs:
    /// delete on Done, mark_unified_failed on Failed, leave alone otherwise.
    async fn finalize_one(&self, queue_id: &str) {
        let overall = match self.queue_manager.check_and_finalize(queue_id).await {
            Ok(s) => s,
            Err(e) => {
                error!(
                    "check_and_finalize failed for queue_id={}: {} — item left in current state",
                    queue_id, e
                );
                return;
            }
        };
        match overall {
            QueueStatus::Done => {
                if let Err(e) = self.queue_manager.delete_unified_item(queue_id).await {
                    error!(
                        "delete_unified_item failed for queue_id={}: {}",
                        queue_id, e
                    );
                }
            }
            QueueStatus::Failed => {
                // mark_unified_failed handles retry vs permanent. max_retries is
                // hardcoded to 3 here because the actor doesn't carry the
                // processor config; the wider queue cfg uses the same default
                // (see UnifiedProcessorConfig).
                let err_msg = fts5_failure_message(queue_id);
                if let Err(e) = self
                    .queue_manager
                    .mark_unified_failed(queue_id, &err_msg, false, 3)
                    .await
                {
                    error!(
                        "mark_unified_failed for {} after FTS5 batch error: {}",
                        queue_id, e
                    );
                }
            }
            QueueStatus::InProgress | QueueStatus::Pending => {
                // qdrant_status isn't done yet — leave the item in queue,
                // the qdrant worker will finalize when it completes.
                debug!(
                    "queue_id={} still in_progress after FTS5 batch (qdrant pending)",
                    queue_id
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fts5_failure_message;

    /// The FTS5 failure message MUST carry a `[transient_` prefix so the
    /// idle resurrection pass (`resurrect_failed_transient`, which matches
    /// `error_message LIKE '[transient_%'`) re-queues it. Regression guard:
    /// dropping the prefix would silently strand failed items in `failed`
    /// forever (the bug that left 346 SQLITE_BUSY items dead).
    #[test]
    fn fts5_failure_message_is_classified_transient() {
        let msg = fts5_failure_message("abc123");
        assert!(
            msg.starts_with("[transient_"),
            "must match resurrection's LIKE '[transient_%' pattern, got: {msg}"
        );
        assert!(msg.contains("abc123"), "must embed the queue_id, got: {msg}");
    }
}
