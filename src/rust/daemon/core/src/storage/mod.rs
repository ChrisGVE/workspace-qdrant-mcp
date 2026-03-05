//! Storage abstraction layer
//!
//! This module provides the Qdrant storage interface implementation with
//! comprehensive vector database operations.
//!
//! # Module structure
//!
//! - [`client`]: Connection management, retry logic, and daemon-mode helpers
//! - [`collections`]: Collection CRUD, aliases, and multi-tenant initialization
//! - [`config`]: Configuration types for transport, HTTP/2, and multi-tenant
//! - [`convert`]: Value conversion between JSON and Qdrant protobuf types
//! - [`points`]: Point insert, delete, count, and payload updates
//! - [`scroll`]: Paginated scroll operations for point retrieval
//! - [`search`]: Dense, sparse, and hybrid (RRF) search implementations
//! - [`types`]: Error types, data structures, and result types

mod client;
mod collections;
pub mod config;
pub(crate) mod convert;
mod points;
mod scroll;
mod search;
pub mod types;

// Re-export all public types from the root of the storage module
// so that existing `use crate::storage::StorageClient` paths continue to work.

pub use client::StorageClient;

pub use config::{Http2Config, MultiTenantConfig, StorageConfig, TransportMode};

pub use types::{
    BatchStats, CollectionInfoResult, DocumentPoint, HybridSearchMode, HybridSearchParams,
    MultiTenantInitResult, SearchParams, SearchResult, StorageError,
};
