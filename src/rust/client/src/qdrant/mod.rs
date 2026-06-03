//! Read-only Qdrant client (WI-d3, #82).
//!
//! [`QdrantReadClient`] is a thin async wrapper over `qdrant_client::Qdrant`
//! exposing **only** read operations (search / search_sparse / scroll /
//! retrieve / collection_exists). It deliberately does NOT expose the raw
//! mutable `Qdrant` handle, and carries no write methods at the type level —
//! mutations are the daemon's exclusive responsibility (ADR-003 / single-writer
//! principle). The `write_guard` tests enforce that no write API leaks into this
//! module's source.
//!
//! REST `:6333` → gRPC `:6334` endpoint translation is the canonical
//! [`wqm_common::qdrant_endpoint::grpc_endpoint`].

pub mod client;

pub use client::{QdrantPoint, QdrantReadClient, QdrantRetrievedPoint};

/// Re-export of the Qdrant point-id type used in scroll pagination, so consumers
/// can thread scroll offsets without a direct `qdrant-client` dependency.
pub use qdrant_client::qdrant::PointId;

// The "no Qdrant write API" guard (AC-d3.2, WI-d3) is enforced crate-wide by
// [`crate::write_service_guard`] (#82 task 21), which scans every wqm-client
// source — including this `client.rs` — for direct Qdrant write methods and
// includes a negative test. A single guard avoids two colliding token lists.
