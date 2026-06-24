//! Storage error type shared across the storage crates and daemon-core.
//!
//! Location: `wqm-common/src/error.rs`. Context: canonical home (F0) of
//! `StorageError`, relocated verbatim from `daemon-core/storage/types.rs` so the
//! read crate (`wqm-storage`) and write crate (`wqm-storage-write`) surface ONE
//! error type (FP-2 / DR GP-9).
//!
//! Hosting it here requires a `qdrant-client` dependency for the typed `Qdrant`
//! variant. This is the deliberate Option-(a) decision: qdrant-client is already
//! a direct dependency of client/daemon-core/grpc/mcp-server and the read crate
//! uses it via `QdrantReadClient`; the leaf principle forbids the WRITE crate /
//! git2 / mutating *methods* (enforced by Guards 1-3), NOT the qdrant-client
//! crate itself. Merely naming `QdrantError` in a variant pulls no mutating
//! symbols, so the Guard-3 symbol-reachability scan stays green. Keeping the
//! variant typed (vs stringifying) makes this a pure move (AC-F0.2, no behavior
//! change) and keeps the `From<QdrantError>` impl legal here (orphan rule:
//! `StorageError` is local to this crate) — every existing daemon-core `?` /
//! `.into()` call site keeps compiling unchanged. Daemon-core re-exports this
//! from `crate::storage` so all call sites are unchanged.
//!
//! F17 adds `ScopeTooBroad` carrying `ScopeTooBroadPayload` — a structured,
//! machine-actionable payload so MCP/JSON clients can read `suggested_scope` and
//! `cliff` as discrete fields rather than parsing prose (AC-F17.5).
//!
//! Neighbors: `crate::search::types::SearchResult`, `crate::hashing`.

use qdrant_client::QdrantError;
use serde::Serialize;
use thiserror::Error;

// ---------------------------------------------------------------------------
// ScopeTooBroad payload (AC-F17.5)
// ---------------------------------------------------------------------------

/// Machine-actionable payload carried by `StorageError::ScopeTooBroad`.
///
/// Every field is a discrete, typed value so an MCP/JSON client reads
/// `suggested_scope` and `cliff` directly without parsing prose (AC-F17.5).
///
/// Per-surface rendering (AC-F17.5):
/// - **MCP/JSON**: serialise this struct — all fields appear as JSON keys.
/// - **CLI**: format as `"Scope too broad: {project_count} projects > cliff
///   {cliff} -- retry with --scope {suggested_scope}"` and exit non-zero.
/// - **Prose/agent**: state count + cliff + suggested scope in a sentence.
#[derive(Debug, Clone, Serialize)]
pub struct ScopeTooBroadPayload {
    /// The scope the caller requested (e.g. `"all"`).
    pub requested_scope: String,
    /// How many projects were found for that scope.
    pub project_count: usize,
    /// The configured project count ceiling for `scope=all`.
    pub cliff: usize,
    /// The scope the caller should use instead (always `"group"` for F17).
    pub suggested_scope: String,
    /// Human-readable hint, names the concrete narrower scope.
    pub hint: String,
}

impl ScopeTooBroadPayload {
    /// Render a one-line CLI banner suitable for stderr + non-zero exit.
    ///
    /// Example: `"Scope too broad: 75 projects > cliff 50 -- retry with
    /// --scope group"`
    pub fn cli_banner(&self) -> String {
        format!(
            "Scope too broad: {} projects > cliff {} -- retry with --scope {}",
            self.project_count, self.cliff, self.suggested_scope
        )
    }
}

// ---------------------------------------------------------------------------
// StorageError
// ---------------------------------------------------------------------------

/// Storage-related errors
#[derive(Error, Debug)]
pub enum StorageError {
    /// Qdrant connection or transport failure.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Collection operation failed (create, delete, get info ...).
    #[error("Collection error: {0}")]
    Collection(String),

    /// Point operation failed (upsert, delete, retrieve ...).
    #[error("Point operation error: {0}")]
    Point(String),

    /// Search or scroll operation failed.
    #[error("Search error: {0}")]
    Search(String),

    /// Batch operation (multi-point upsert / delete) failed.
    #[error("Batch operation error: {0}")]
    Batch(String),

    /// Operation exceeded its deadline.
    #[error("Timeout error: {0}")]
    Timeout(String),

    /// JSON serialization / deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Low-level Qdrant client error (boxed to keep the enum small).
    #[error("Qdrant client error: {0}")]
    Qdrant(Box<QdrantError>),

    /// Input validation failure (e.g. invalid branch_name, AC-F4.6 / SEC-N04).
    ///
    /// The message describes what was rejected and why. Never contains raw
    /// user input rendered for display — callers should sanitize before showing.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Per-project `store.db` (SQLite) operation failed — a query, transaction, or
    /// connection-protocol step in the write/read storage path (arch §5.2). The
    /// message names the failing step so a log reader can locate it; it never carries
    /// raw indexed content.
    #[error("SQLite error: {0}")]
    Sqlite(String),

    /// A second writer attempted to acquire the singleton OS advisory daemon lock
    /// (`<data_dir>/daemon.lock`) while it was already held by another process
    /// (AC-F14.1). The message identifies the lock path and instructs the operator.
    /// Fail-closed: never auto-reclaimed.
    #[error("Lock conflict: {0}")]
    LockConflict(String),

    /// The `scope=all` fan-out would exceed the configured project cliff
    /// (AC-F17.2). The payload carries machine-actionable fields so callers can
    /// programmatically re-narrow without parsing prose (AC-F17.5).
    ///
    /// Never returned for `scope=project` or `scope=group`. The cliff is
    /// configurable; the default is 50 projects (PRD §F17, §14-Q3).
    #[error("scope too broad: {0} projects exceed cliff {1} -- narrow to scope=group")]
    ScopeTooBroad(usize, usize, Box<ScopeTooBroadPayload>),
}

impl From<QdrantError> for StorageError {
    fn from(err: QdrantError) -> Self {
        StorageError::Qdrant(Box::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AC-F17.5: ScopeTooBroadPayload machine-readable fields + CLI banner
    // -----------------------------------------------------------------------

    #[test]
    fn t_f17_05_scope_too_broad_payload_json_has_discrete_fields() {
        let payload = ScopeTooBroadPayload {
            requested_scope: "all".into(),
            project_count: 75,
            cliff: 50,
            suggested_scope: "group".into(),
            hint: "Use --scope group to search only related projects.".into(),
        };

        let json_str = serde_json::to_string(&payload).expect("serialize");
        let val: serde_json::Value = serde_json::from_str(&json_str).expect("parse");

        // suggested_scope and cliff MUST be discrete JSON fields, not embedded
        // in a prose string (AC-F17.5 MCP/JSON surface assertion).
        assert_eq!(
            val["suggested_scope"], "group",
            "suggested_scope must be a top-level JSON field"
        );
        assert_eq!(val["cliff"], 50, "cliff must be a top-level JSON field");
        assert_eq!(val["project_count"], 75);
        assert_eq!(val["requested_scope"], "all");
    }

    #[test]
    fn t_f17_05_cli_banner_carries_count_and_cliff() {
        let payload = ScopeTooBroadPayload {
            requested_scope: "all".into(),
            project_count: 75,
            cliff: 50,
            suggested_scope: "group".into(),
            hint: "Use --scope group.".into(),
        };

        let banner = payload.cli_banner();
        // The banner must name the count, cliff, and the concrete narrower scope.
        assert!(banner.contains("75"), "banner must include project_count");
        assert!(banner.contains("50"), "banner must include cliff");
        assert!(banner.contains("group"), "banner must name suggested_scope");
    }

    #[test]
    fn t_f17_05_scope_too_broad_variant_display() {
        let payload = ScopeTooBroadPayload {
            requested_scope: "all".into(),
            project_count: 75,
            cliff: 50,
            suggested_scope: "group".into(),
            hint: "Use --scope group.".into(),
        };
        let err = StorageError::ScopeTooBroad(75, 50, Box::new(payload));
        let display = err.to_string();
        assert!(display.contains("75"), "display includes count");
        assert!(display.contains("50"), "display includes cliff");
    }

    #[test]
    fn t_f0_storage_error_display_string_variants() {
        assert_eq!(
            StorageError::Connection("host unreachable".into()).to_string(),
            "Connection error: host unreachable"
        );
        assert_eq!(
            StorageError::Collection("not found".into()).to_string(),
            "Collection error: not found"
        );
        assert_eq!(
            StorageError::Point("upsert failed".into()).to_string(),
            "Point operation error: upsert failed"
        );
        assert_eq!(
            StorageError::Search("timeout".into()).to_string(),
            "Search error: timeout"
        );
        assert_eq!(
            StorageError::Batch("partial failure".into()).to_string(),
            "Batch operation error: partial failure"
        );
        assert_eq!(
            StorageError::Timeout("10s exceeded".into()).to_string(),
            "Timeout error: 10s exceeded"
        );
    }

    #[test]
    fn t_f0_storage_error_from_serde_json() {
        // serde_json::Error converts via #[from].
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let se: StorageError = json_err.into();
        assert!(matches!(se, StorageError::Serialization(_)));
        assert!(se.to_string().starts_with("Serialization error:"));
    }

    #[test]
    fn t_f0_storage_error_is_std_error_send_sync() {
        // Satisfies std::error::Error + Send + Sync (required by async contexts)
        // and proves the typed Qdrant(Box<QdrantError>) variant keeps those bounds.
        fn assert_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_send_sync::<StorageError>();
    }
}
