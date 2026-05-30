//! Transport layer: wires the [`crate::tools::ToolsHandler`] to a concrete
//! MCP transport (stdio now; streamable-HTTP in task 32).
//!
//! ## stdio transport (task 31)
//!
//! Reads JSON-RPC frames from `stdin`, writes responses to `stdout`.
//! All log output goes to `stderr` (see [`crate::observability::logging`]).
//! The server block until the client closes the connection, then runs
//! graceful cleanup via [`crate::session::lifecycle::cleanup_session`].
//!
//! ## HTTP transport (task 32)
//!
//! Not yet implemented.  The `serve_http` placeholder is intentionally left
//! as a `todo!()` stub — it does NOT compile unless the `serve_http` call
//! site is reached at runtime (cfg-guarded in `main.rs`).

pub mod stdio;
