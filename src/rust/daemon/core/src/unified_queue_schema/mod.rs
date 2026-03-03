//! Unified Queue Schema Definitions (Task 22)
//!
//! This module defines the types and schema for the unified ingestion queue.
//! All queue operations (content, file, folder, etc.) are processed through
//! a single unified queue table with type discriminators.
//!
//! Core types (ItemType, QueueOperation, QueueStatus, payloads) are defined in
//! `wqm_common` and re-exported here for backward compatibility.

mod hashing;
mod sql;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items so callers see the same paths as before.

pub use hashing::{generate_idempotency_key, generate_unified_idempotency_key, IdempotencyKeyError};

pub use sql::{
    CREATE_QDRANT_STATUS_INDEX_SQL, CREATE_SEARCH_STATUS_INDEX_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    CREATE_UNIFIED_QUEUE_SQL, MIGRATE_V20_ADD_COLUMNS_SQL,
};

pub use types::{
    // Payload types (from wqm_common via types.rs)
    CollectionPayload,
    ContentPayload,
    DeleteDocumentPayload,
    DeleteTenantPayload,
    // Queue status/operation enums (from wqm_common via types.rs)
    DestinationStatus,
    FilePayload,
    FolderPayload,
    ItemType,
    LibraryPayload,
    MemoryPayload,
    ProjectPayload,
    QueueDecision,
    QueueOperation,
    QueueStatus,
    ScratchpadPayload,
    // Local structs
    UnifiedQueueItem,
    UnifiedQueueStats,
    UrlPayload,
    WebsitePayload,
};
