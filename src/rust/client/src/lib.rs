//! `wqm-client` — shared daemon gRPC + Qdrant read client for the wqm CLI and
//! MCP server (WI-d1/d2/d3, #82).
//!
//! This crate is the single home for the client-side machinery both wqm clients
//! need: the [`DaemonClient`] gRPC wrapper (dial / retry / timeout / typed RPC
//! wrappers) and, later, a read-only Qdrant client. It depends only on
//! `wqm-proto` (generated stubs) and `wqm-common` (config, constants,
//! project-id, …) — and **never** on `workspace-qdrant-core`, preserving the
//! "daemon is the engine, clients talk to it" topology.

pub mod grpc;
pub mod qdrant;

pub use grpc::{ClientError, DaemonClient};
pub use qdrant::{QdrantPoint, QdrantReadClient, QdrantRetrievedPoint};

/// Re-export of the generated proto module so consumers can reach request /
/// response message types without depending on `wqm-proto` directly.
pub use wqm_proto::workspace_daemon;
