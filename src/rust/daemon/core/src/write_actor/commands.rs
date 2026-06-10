//! WriteCommand enum for serialized state.db mutations.
//!
//! Each variant corresponds to a gRPC write service RPC. The command carries
//! the request data and a oneshot sender for the result. Internal daemon
//! mutations (queue processor, file tracker) stay on the pool for now.

use tokio::sync::oneshot;

// ── QueueWriteService commands ─────────────────────────────────────────

/// Data for EnqueueItem (maps to QueueWriteService.EnqueueItem)
#[derive(Debug)]
pub struct EnqueueItemData {
    pub item_type: String,
    pub op: String,
    pub tenant_id: String,
    pub collection: String,
    pub payload_json: String,
    pub branch: String,
    pub metadata_json: Option<String>,
}

/// Result of EnqueueItem
#[derive(Debug)]
pub struct EnqueueItemResult {
    pub queue_id: String,
    pub idempotency_key: String,
    pub is_new: bool,
}

/// Result of RetryAll
#[derive(Debug)]
pub struct RetryAllResult {
    pub reset_count: u32,
}

/// Data for RetryItem
#[derive(Debug)]
pub struct RetryItemData {
    pub queue_id: String,
}

/// Result of RetryItem
#[derive(Debug)]
pub struct RetryItemResult {
    pub found: bool,
    pub resolved_id: String,
    pub previous_status: String,
    pub previous_retry_count: i32,
    pub reset: bool,
}

/// Data for CleanQueue
#[derive(Debug)]
pub struct CleanQueueData {
    pub older_than_days: i32,
    pub statuses: Vec<String>,
}

/// Data for CancelItems
#[derive(Debug)]
pub struct CancelItemsData {
    pub tenant_id: String,
    pub statuses: Vec<String>,
    pub dry_run: bool,
}

/// Result of CancelItems
#[derive(Debug)]
pub struct CancelItemsResult {
    pub count: u32,
    pub tenant_id: String,
    pub project_path: String,
    pub is_dry_run: bool,
}

/// Data for RemoveItem
#[derive(Debug)]
pub struct RemoveItemData {
    pub queue_id: String,
}

/// Result of RemoveItem
#[derive(Debug)]
pub struct RemoveItemResult {
    pub found: bool,
    pub resolved_id: String,
    pub item_type: String,
    pub op: String,
    pub collection: String,
    pub status: String,
}

/// Data for CleanQueueByCollection
#[derive(Debug)]
pub struct CleanQueueByCollectionData {
    pub collections: Vec<String>,
    pub statuses: Vec<String>,
}

// ── WatchWriteService commands ─────────────────────────────────────────

/// Data for EnableWatch / DisableWatch / UnarchiveWatch
#[derive(Debug)]
pub struct WatchIdData {
    pub watch_id: String,
}

/// Data for ArchiveWatch
#[derive(Debug)]
pub struct ArchiveWatchData {
    pub watch_id: String,
    pub cascade_submodules: bool,
}

/// Result of ArchiveWatch
#[derive(Debug)]
pub struct ArchiveWatchResult {
    pub affected_count: u32,
    pub submodules_archived: u32,
    pub submodules_skipped: u32,
}

// ── LibraryWriteService commands ───────────────────────────────────────

/// Data for AddLibrary
#[derive(Debug)]
pub struct AddLibraryData {
    pub tag: String,
    pub path: String,
    pub mode: String,
}

/// Result of AddLibrary
#[derive(Debug)]
pub struct AddLibraryResult {
    pub success: bool,
    pub watch_id: String,
    pub message: String,
}

/// Data for RemoveLibrary
#[derive(Debug)]
pub struct RemoveLibraryData {
    pub tag: String,
}

/// Result of RemoveLibrary
#[derive(Debug)]
pub struct RemoveLibraryResult {
    pub success: bool,
    pub queue_items_cancelled: u32,
    pub tracked_files_deleted: u32,
    pub components_deleted: u32,
    pub message: String,
}

/// Data for WatchLibrary
#[derive(Debug)]
pub struct WatchLibraryData {
    pub tag: String,
    pub path: String,
    pub mode: String,
}

/// Result of WatchLibrary
#[derive(Debug)]
pub struct WatchLibraryResult {
    pub success: bool,
    pub is_new: bool,
    pub watch_id: String,
    pub message: String,
}

/// Data for UnwatchLibrary
#[derive(Debug)]
pub struct UnwatchLibraryData {
    pub tag: String,
}

/// Data for ConfigureLibrary
#[derive(Debug)]
pub struct ConfigureLibraryData {
    pub tag: String,
    pub mode: Option<String>,
    pub enable: Option<bool>,
    pub disable: Option<bool>,
}

/// Data for SetIncremental
#[derive(Debug)]
pub struct SetIncrementalData {
    pub file_paths: Vec<String>,
    pub clear: bool,
}

/// Result of SetIncremental
#[derive(Debug)]
pub struct SetIncrementalResult {
    pub updated: u32,
    pub not_found: u32,
}

// ── TrackingWriteService commands ──────────────────────────────────────

/// Data for LogSearchEvent
#[derive(Debug)]
pub struct LogSearchEventData {
    pub id: String,
    pub session_id: Option<String>,
    pub project_id: Option<String>,
    pub actor: String,
    pub tool: String,
    pub op: String,
    pub query_text: Option<String>,
    pub filters: Option<String>,
    pub top_k: Option<i32>,
    pub result_count: Option<i32>,
    pub latency_ms: Option<i64>,
    pub top_result_refs: Option<String>,
    pub outcome: Option<String>,
    pub parent_event_id: Option<String>,
}

/// Data for UpdateSearchEvent
#[derive(Debug)]
pub struct UpdateSearchEventData {
    pub event_id: String,
    pub result_count: Option<i32>,
    pub latency_ms: Option<i64>,
    pub top_result_refs: Option<String>,
    pub outcome: Option<String>,
}

/// Data for UpdateSearchEventEconomy — token-economy metrics from the
/// MCP search shaping pass (spec docs/specs/20-token-economy-instrumentation.md).
#[derive(Debug)]
pub struct UpdateSearchEventEconomyData {
    pub event_id: String,
    pub bytes_in: i64,
    pub bytes_out: i64,
    pub hits_truncated: i32,
    pub shape_mode: String,
    pub tool_version: Option<String>,
}

/// Data for UpsertRuleMirror
#[derive(Debug)]
pub struct UpsertRuleMirrorData {
    pub rule_id: String,
    pub rule_text: String,
    pub scope: String,
    pub tenant_id: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Data for DeleteRuleMirror
#[derive(Debug)]
pub struct DeleteRuleMirrorData {
    pub rule_id: String,
}

/// Data for UpsertScratchpadMirror
#[derive(Debug)]
pub struct UpsertScratchpadMirrorData {
    pub scratchpad_id: String,
    pub content: String,
    pub title: String,
    pub tags: String,
    pub tenant_id: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Data for DeleteScratchpadMirror
#[derive(Debug)]
pub struct DeleteScratchpadMirrorData {
    pub scratchpad_id: String,
}

// ── AdminWriteService commands ─────────────────────────────────────────

/// Data for RenameTenantAdmin
#[derive(Debug)]
pub struct RenameTenantAdminData {
    pub old_tenant_id: String,
    pub new_tenant_id: String,
}

/// Result of RenameTenantAdmin
#[derive(Debug)]
pub struct RenameTenantAdminResult {
    pub success: bool,
    pub total_rows_updated: u32,
    pub message: String,
}

/// Data for RebalanceIdf
#[derive(Debug)]
pub struct RebalanceIdfData {
    pub collection: String,
    pub last_corrected_n: i64,
}

/// Result of RebalanceIdf
#[derive(Debug)]
pub struct RebalanceIdfResult {
    pub success: bool,
    pub message: String,
}

/// Result of ReapplyIgnoreRules
#[derive(Debug)]
pub struct ReapplyIgnoreRulesResult {
    pub projects_processed: u32,
    pub stale_deleted: u32,
    pub missing_added: u32,
}

/// Data for ReembedTenant
#[derive(Debug)]
pub struct ReembedTenantData {
    pub tenant_id: String,
    /// Force full re-processing: the folder scans are enqueued with
    /// `uplift: true`, so every discovered file becomes `File/Uplift` and
    /// bypasses the unchanged-hash + chunker-fingerprint skip. Without it
    /// the re-embed is a repair pass — files whose content hash AND
    /// chunking configuration are unchanged are skipped.
    pub force: bool,
}

/// Result of ReembedTenant
#[derive(Debug)]
pub struct ReembedTenantResult {
    pub files_enqueued: u32,
    pub message: String,
}

// ── WriteCommand enum ──────────────────────────────────────────────────

/// Type alias for write results channeled back via oneshot.
pub type WriteResult<T> = Result<T, String>;

/// All daemon-internal write commands sent through the WriteActor channel.
///
/// Each variant holds the request data and a oneshot sender for the typed result.
/// The `String` error is used instead of `tonic::Status` to avoid coupling the
/// core crate to tonic.
#[derive(Debug)]
pub enum WriteCommand {
    // QueueWriteService
    EnqueueItem {
        data: EnqueueItemData,
        tx: oneshot::Sender<WriteResult<EnqueueItemResult>>,
    },
    RetryAll {
        tx: oneshot::Sender<WriteResult<RetryAllResult>>,
    },
    RetryItem {
        data: RetryItemData,
        tx: oneshot::Sender<WriteResult<RetryItemResult>>,
    },
    CleanQueue {
        data: CleanQueueData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    CancelItems {
        data: CancelItemsData,
        tx: oneshot::Sender<WriteResult<CancelItemsResult>>,
    },
    RemoveItem {
        data: RemoveItemData,
        tx: oneshot::Sender<WriteResult<RemoveItemResult>>,
    },
    CleanQueueByCollection {
        data: CleanQueueByCollectionData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },

    // WatchWriteService
    PauseWatchers {
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    ResumeWatchers {
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    PauseWatch {
        data: WatchIdData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    ResumeWatch {
        data: WatchIdData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    EnableWatch {
        data: WatchIdData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    DisableWatch {
        data: WatchIdData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    ArchiveWatch {
        data: ArchiveWatchData,
        tx: oneshot::Sender<WriteResult<ArchiveWatchResult>>,
    },
    UnarchiveWatch {
        data: WatchIdData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },

    // LibraryWriteService
    AddLibrary {
        data: AddLibraryData,
        tx: oneshot::Sender<WriteResult<AddLibraryResult>>,
    },
    RemoveLibrary {
        data: RemoveLibraryData,
        tx: oneshot::Sender<WriteResult<RemoveLibraryResult>>,
    },
    WatchLibrary {
        data: WatchLibraryData,
        tx: oneshot::Sender<WriteResult<WatchLibraryResult>>,
    },
    UnwatchLibrary {
        data: UnwatchLibraryData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    ConfigureLibrary {
        data: ConfigureLibraryData,
        tx: oneshot::Sender<WriteResult<u32>>,
    },
    SetIncremental {
        data: SetIncrementalData,
        tx: oneshot::Sender<WriteResult<SetIncrementalResult>>,
    },

    // TrackingWriteService
    LogSearchEvent {
        data: LogSearchEventData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    UpdateSearchEvent {
        data: UpdateSearchEventData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    UpdateSearchEventEconomy {
        data: UpdateSearchEventEconomyData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    UpsertRuleMirror {
        data: UpsertRuleMirrorData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    DeleteRuleMirror {
        data: DeleteRuleMirrorData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    UpsertScratchpadMirror {
        data: UpsertScratchpadMirrorData,
        tx: oneshot::Sender<WriteResult<()>>,
    },
    DeleteScratchpadMirror {
        data: DeleteScratchpadMirrorData,
        tx: oneshot::Sender<WriteResult<()>>,
    },

    // AdminWriteService
    RenameTenantAdmin {
        data: RenameTenantAdminData,
        tx: oneshot::Sender<WriteResult<RenameTenantAdminResult>>,
    },
    RebalanceIdf {
        data: RebalanceIdfData,
        tx: oneshot::Sender<WriteResult<RebalanceIdfResult>>,
    },
    ReapplyIgnoreRules {
        tx: oneshot::Sender<WriteResult<ReapplyIgnoreRulesResult>>,
    },
    ReembedTenant {
        data: ReembedTenantData,
        tx: oneshot::Sender<WriteResult<ReembedTenantResult>>,
    },
}
