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
//! [`http::serve_http`] assembles the axum router, applies CORS/rate-limit/auth
//! middleware, and drives the rmcp `StreamableHttpService` on the configured
//! bind address.  `/healthz` is unauthenticated.

pub mod auth;
pub mod cors;
pub mod health;
pub mod http;
pub mod rate_limit;
pub mod stdio;
pub mod tls;
