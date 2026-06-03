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

#[cfg(test)]
mod write_guard {
    //! CI guard: the read-only client source must contain no Qdrant write APIs
    //! (AC-d3.2). Enforced by scanning the included source at test time, with a
    //! negative test proving the guard actually fires on a write token.

    /// The qdrant client source, included at compile time for scanning.
    const CLIENT_SRC: &str = include_str!("client.rs");

    /// Qdrant mutation APIs that must never appear in the read-only client.
    const FORBIDDEN_WRITE_APIS: &[&str] = &[
        "upsert",
        "set_payload",
        "delete_points",
        "update_named",
        "insert_point",
        "delete_payload",
        "create_collection",
        "delete_collection",
        "update_collection",
    ];

    #[test]
    fn no_write_apis_in_read_only_client() {
        for token in FORBIDDEN_WRITE_APIS {
            assert!(
                !CLIENT_SRC.contains(token),
                "forbidden Qdrant write API `{token}` found in the read-only client source — \
                 QdrantReadClient must expose read operations only (WI-d3)"
            );
        }
    }

    #[test]
    fn guard_detects_injected_write_api() {
        // Negative test: prove the scan would catch a write token if one were
        // ever introduced, so a clean pass above is meaningful.
        let injected = "pub async fn upsert(&self) { /* would violate single-writer */ }";
        assert!(
            FORBIDDEN_WRITE_APIS.iter().any(|t| injected.contains(t)),
            "guard must detect an injected write API"
        );
    }
}
