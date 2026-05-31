//! gRPC endpoint resolution for the Qdrant read client.
//!
//! The TypeScript MCP server talks to Qdrant over its **REST** API (default
//! port `6333`), so deployments set `QDRANT_URL` (and the config default) to a
//! `:6333` URL. The Rust port uses `qdrant_client::Qdrant`, which speaks
//! **gRPC** — served by Qdrant on port `6334`. Pointing the gRPC client at the
//! REST port yields an opaque `h2 protocol error` on the first call.
//!
//! The translation itself is the canonical
//! [`wqm_common::qdrant_endpoint::grpc_endpoint`], shared with the daemon's
//! `build_connection_url` so the whole workspace uses one `:6333` → `:6334`
//! convention with no duplicated rule. This thin re-export keeps the local
//! `qdrant::endpoint::grpc_endpoint` call site stable.
//!
//! DEFERRED (GitHub #82): consolidate config into one shared module and drop
//! this shim along with the per-component config duplication.

pub use wqm_common::qdrant_endpoint::grpc_endpoint;
