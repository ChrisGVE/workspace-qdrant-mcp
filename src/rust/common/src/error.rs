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
//! Neighbors: `crate::search::types::SearchResult`, `crate::hashing`.

use qdrant_client::QdrantError;
use thiserror::Error;

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
}

impl From<QdrantError> for StorageError {
    fn from(err: QdrantError) -> Self {
        StorageError::Qdrant(Box::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
