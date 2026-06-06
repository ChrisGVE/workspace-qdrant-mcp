//! gRPC endpoint resolution for the Qdrant read client.
//!
//! Deployments set `QDRANT_URL` (and the config default) to a REST-style
//! `:6333` URL. This server uses `qdrant_client::Qdrant`, which speaks
//! **gRPC** — served by Qdrant on port `6334`. Pointing the gRPC client at the
//! REST port yields an opaque `h2 protocol error` on the first call.
//!
//! The translation itself is the canonical
//! [`wqm_common::qdrant_endpoint::grpc_endpoint`], shared with the daemon's
//! `build_connection_url` so the whole workspace uses one `:6333` → `:6334`
//! convention with no duplicated rule (WI-b1, #82). This thin re-export keeps
//! the local `qdrant::endpoint::grpc_endpoint` call site stable.

pub use wqm_common::qdrant_endpoint::grpc_endpoint;
