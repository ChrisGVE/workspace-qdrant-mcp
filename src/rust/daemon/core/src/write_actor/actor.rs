//! WriteActor: serializes all daemon state.db mutations through a single channel.
//!
//! The actor owns a `SqlitePool` and processes `WriteCommand` values from a
//! bounded `tokio::mpsc` channel. Results are returned via oneshot senders.
//! This eliminates internal write contention between daemon actors.

use sqlx::SqlitePool;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use super::commands::*;

// TODO: Replace string-based error classification (e.g. matching on "no such column")
// with typed error variants (e.g. WriteError enum) for more robust error handling.
// See: https://github.com/ChrisGVE/workspace-qdrant-mcp/pull/52

/// Channel buffer size for write commands.
const CHANNEL_BUFFER: usize = 1024;

macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* ), $tx:expr) => {
        if $tx.send($self.$method( $($arg),* ).await).is_err() {
            warn!(concat!(stringify!($method), " response dropped: client disconnected"));
        }
    };
}

/// The actor task that processes write commands sequentially.
pub struct WriteActor {
    rx: mpsc::Receiver<WriteCommand>,
    pub(super) pool: SqlitePool,
}

/// Cloneable handle for sending commands to the WriteActor.
#[derive(Clone)]
pub struct WriteActorHandle {
    tx: mpsc::Sender<WriteCommand>,
}

impl WriteActor {
    /// Spawn the actor task and return its handle.
    pub fn spawn(pool: SqlitePool) -> WriteActorHandle {
        let (tx, rx) = mpsc::channel(CHANNEL_BUFFER);
        let actor = WriteActor { rx, pool };
        tokio::spawn(actor.run());
        WriteActorHandle { tx }
    }

    async fn run(mut self) {
        debug!("WriteActor started");
        while let Some(cmd) = self.rx.recv().await {
            let started = std::time::Instant::now();
            self.handle_command(cmd).await;
            crate::monitoring::metrics_core::METRICS.record_sqlite("write", started.elapsed());
        }
        debug!("WriteActor stopped (channel closed)");
    }

    async fn handle_command(&self, cmd: WriteCommand) {
        match cmd {
            WriteCommand::EnqueueItem { data, tx } => dispatch!(self, exec_enqueue_item(data), tx),
            WriteCommand::RetryAll { tx } => dispatch!(self, exec_retry_all(), tx),
            WriteCommand::RetryItem { data, tx } => dispatch!(self, exec_retry_item(data), tx),
            WriteCommand::CleanQueue { data, tx } => dispatch!(self, exec_clean_queue(data), tx),
            WriteCommand::CancelItems { data, tx } => dispatch!(self, exec_cancel_items(data), tx),
            WriteCommand::RemoveItem { data, tx } => dispatch!(self, exec_remove_item(data), tx),
            WriteCommand::CleanQueueByCollection { data, tx } => {
                dispatch!(self, exec_clean_queue_by_collection(data), tx)
            }
            WriteCommand::PauseWatchers { tx } => dispatch!(self, exec_pause_watchers(), tx),
            WriteCommand::ResumeWatchers { tx } => dispatch!(self, exec_resume_watchers(), tx),
            WriteCommand::EnableWatch { data, tx } => dispatch!(self, exec_enable_watch(data), tx),
            WriteCommand::DisableWatch { data, tx } => {
                dispatch!(self, exec_disable_watch(data), tx)
            }
            WriteCommand::ArchiveWatch { data, tx } => {
                dispatch!(self, exec_archive_watch(data), tx)
            }
            WriteCommand::UnarchiveWatch { data, tx } => {
                dispatch!(self, exec_unarchive_watch(data), tx)
            }
            WriteCommand::AddLibrary { data, tx } => dispatch!(self, exec_add_library(data), tx),
            WriteCommand::RemoveLibrary { data, tx } => {
                dispatch!(self, exec_remove_library(data), tx)
            }
            WriteCommand::WatchLibrary { data, tx } => {
                dispatch!(self, exec_watch_library(data), tx)
            }
            WriteCommand::UnwatchLibrary { data, tx } => {
                dispatch!(self, exec_unwatch_library(data), tx)
            }
            WriteCommand::ConfigureLibrary { data, tx } => {
                dispatch!(self, exec_configure_library(data), tx)
            }
            WriteCommand::SetIncremental { data, tx } => {
                dispatch!(self, exec_set_incremental(data), tx)
            }
            WriteCommand::LogSearchEvent { data, tx } => {
                dispatch!(self, exec_log_search_event(data), tx)
            }
            WriteCommand::UpdateSearchEvent { data, tx } => {
                dispatch!(self, exec_update_search_event(data), tx)
            }
            WriteCommand::UpsertRuleMirror { data, tx } => {
                dispatch!(self, exec_upsert_rule_mirror(data), tx)
            }
            WriteCommand::DeleteRuleMirror { data, tx } => {
                dispatch!(self, exec_delete_rule_mirror(data), tx)
            }
            WriteCommand::UpsertScratchpadMirror { data, tx } => {
                dispatch!(self, exec_upsert_scratchpad_mirror(data), tx)
            }
            WriteCommand::DeleteScratchpadMirror { data, tx } => {
                dispatch!(self, exec_delete_scratchpad_mirror(data), tx)
            }
            WriteCommand::RenameTenantAdmin { data, tx } => {
                dispatch!(self, exec_rename_tenant_admin(data), tx)
            }
            WriteCommand::RebalanceIdf { data, tx } => {
                dispatch!(self, exec_rebalance_idf(data), tx)
            }
        }
    }
}

// ── WriteActorHandle typed helpers ─────────────────────────────────────

impl WriteActorHandle {
    /// Send a command and await the result. Returns `Err` if the actor is dead.
    async fn send<T>(
        &self,
        build: impl FnOnce(oneshot::Sender<WriteResult<T>>) -> WriteCommand,
    ) -> WriteResult<T> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(build(tx))
            .await
            .map_err(|_| "WriteActor channel closed".to_string())?;
        rx.await
            .map_err(|_| "WriteActor dropped response".to_string())?
    }

    // ── QueueWriteService ──────────────────────────────────────────

    pub async fn enqueue_item(&self, data: EnqueueItemData) -> WriteResult<EnqueueItemResult> {
        self.send(|tx| WriteCommand::EnqueueItem { data, tx }).await
    }

    pub async fn retry_all(&self) -> WriteResult<RetryAllResult> {
        self.send(|tx| WriteCommand::RetryAll { tx }).await
    }

    pub async fn retry_item(&self, data: RetryItemData) -> WriteResult<RetryItemResult> {
        self.send(|tx| WriteCommand::RetryItem { data, tx }).await
    }

    pub async fn clean_queue(&self, data: CleanQueueData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::CleanQueue { data, tx }).await
    }

    pub async fn cancel_items(&self, data: CancelItemsData) -> WriteResult<CancelItemsResult> {
        self.send(|tx| WriteCommand::CancelItems { data, tx }).await
    }

    pub async fn remove_item(&self, data: RemoveItemData) -> WriteResult<RemoveItemResult> {
        self.send(|tx| WriteCommand::RemoveItem { data, tx }).await
    }

    pub async fn clean_queue_by_collection(
        &self,
        data: CleanQueueByCollectionData,
    ) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::CleanQueueByCollection { data, tx })
            .await
    }

    // ── WatchWriteService ──────────────────────────────────────────

    pub async fn pause_watchers(&self) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::PauseWatchers { tx }).await
    }

    pub async fn resume_watchers(&self) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::ResumeWatchers { tx }).await
    }

    pub async fn enable_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::EnableWatch { data, tx }).await
    }

    pub async fn disable_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::DisableWatch { data, tx })
            .await
    }

    pub async fn archive_watch(&self, data: ArchiveWatchData) -> WriteResult<ArchiveWatchResult> {
        self.send(|tx| WriteCommand::ArchiveWatch { data, tx })
            .await
    }

    pub async fn unarchive_watch(&self, data: WatchIdData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::UnarchiveWatch { data, tx })
            .await
    }

    // ── LibraryWriteService ────────────────────────────────────────

    pub async fn add_library(&self, data: AddLibraryData) -> WriteResult<AddLibraryResult> {
        self.send(|tx| WriteCommand::AddLibrary { data, tx }).await
    }

    pub async fn remove_library(
        &self,
        data: RemoveLibraryData,
    ) -> WriteResult<RemoveLibraryResult> {
        self.send(|tx| WriteCommand::RemoveLibrary { data, tx })
            .await
    }

    pub async fn watch_library(&self, data: WatchLibraryData) -> WriteResult<WatchLibraryResult> {
        self.send(|tx| WriteCommand::WatchLibrary { data, tx })
            .await
    }

    pub async fn unwatch_library(&self, data: UnwatchLibraryData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::UnwatchLibrary { data, tx })
            .await
    }

    pub async fn configure_library(&self, data: ConfigureLibraryData) -> WriteResult<u32> {
        self.send(|tx| WriteCommand::ConfigureLibrary { data, tx })
            .await
    }

    pub async fn set_incremental(
        &self,
        data: SetIncrementalData,
    ) -> WriteResult<SetIncrementalResult> {
        self.send(|tx| WriteCommand::SetIncremental { data, tx })
            .await
    }

    // ── TrackingWriteService ───────────────────────────────────────

    pub async fn log_search_event(&self, data: LogSearchEventData) -> WriteResult<()> {
        self.send(|tx| WriteCommand::LogSearchEvent { data, tx })
            .await
    }

    pub async fn update_search_event(&self, data: UpdateSearchEventData) -> WriteResult<()> {
        self.send(|tx| WriteCommand::UpdateSearchEvent { data, tx })
            .await
    }

    pub async fn upsert_rule_mirror(&self, data: UpsertRuleMirrorData) -> WriteResult<()> {
        self.send(|tx| WriteCommand::UpsertRuleMirror { data, tx })
            .await
    }

    pub async fn delete_rule_mirror(&self, data: DeleteRuleMirrorData) -> WriteResult<()> {
        self.send(|tx| WriteCommand::DeleteRuleMirror { data, tx })
            .await
    }

    pub async fn upsert_scratchpad_mirror(
        &self,
        data: UpsertScratchpadMirrorData,
    ) -> WriteResult<()> {
        self.send(|tx| WriteCommand::UpsertScratchpadMirror { data, tx })
            .await
    }

    pub async fn delete_scratchpad_mirror(
        &self,
        data: DeleteScratchpadMirrorData,
    ) -> WriteResult<()> {
        self.send(|tx| WriteCommand::DeleteScratchpadMirror { data, tx })
            .await
    }

    // ── AdminWriteService ──────────────────────────────────────────

    pub async fn rename_tenant_admin(
        &self,
        data: RenameTenantAdminData,
    ) -> WriteResult<RenameTenantAdminResult> {
        self.send(|tx| WriteCommand::RenameTenantAdmin { data, tx })
            .await
    }

    pub async fn rebalance_idf(&self, data: RebalanceIdfData) -> WriteResult<RebalanceIdfResult> {
        self.send(|tx| WriteCommand::RebalanceIdf { data, tx })
            .await
    }
}
