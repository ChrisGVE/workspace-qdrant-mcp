//! Read-only Qdrant access for the storage read crate.
//!
//! Location: `wqm-storage/src/qdrant/`. Logical context: the ONLY door through
//! which `wqm-storage` touches Qdrant. It exposes the `QdrantReadClient` newtype
//! (read methods only); every mutating call lives in `wqm-storage-write/qdrant/`.
//! Neighbors: `wqm-storage-write::qdrant` (the write sibling), `wqm_common::StorageError`.

pub mod read_client;

pub use read_client::QdrantReadClient;
