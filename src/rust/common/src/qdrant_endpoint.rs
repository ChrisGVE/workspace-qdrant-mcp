//! Canonical Qdrant gRPC endpoint resolution.
//!
//! Qdrant serves its REST API on port `6333` and its gRPC API on port `6334`.
//! Operators and the TypeScript MCP server set `QDRANT_URL` (and the config
//! default [`crate::constants::DEFAULT_QDRANT_URL`]) to a REST-style `:6333`
//! URL. Every Rust component that uses `qdrant_client` (the daemon, and the
//! Rust MCP server) speaks **gRPC**, so it must dial `:6334` instead —
//! otherwise the first call fails with an opaque `h2 protocol error`.
//!
//! This is the single source of truth for that `:6333` → `:6334` translation.
//! Both `workspace-qdrant-core` (the daemon's `build_connection_url`) and the
//! MCP server's `qdrant::endpoint` delegate here so the whole workspace shares
//! one convention and the rule can never drift between copies.
//!
//! DEFERRED (GitHub #82): this helper is a stop-gap for the deeper problem —
//! three parallel config modules (daemon/CLI/MCP) with no single canonical
//! home. When config is consolidated into one shared module, fold this
//! translation into that module's resolution and drop the delegation shims.

/// The Qdrant REST API port.
pub const QDRANT_REST_PORT: u16 = 6333;

/// The Qdrant gRPC API port.
pub const QDRANT_GRPC_PORT: u16 = 6334;

/// Translate a (possibly REST-style) Qdrant URL to its gRPC endpoint.
///
/// Rules:
/// - An explicit REST port (`:6333`) is rewritten to the gRPC port (`:6334`).
/// - A bare host with no port and no scheme colon gets `:6334` appended.
/// - Any URL that already targets `:6334`, or names some other explicit port,
///   is returned unchanged (an explicit non-default port is an operator choice
///   and is never second-guessed).
///
/// # Examples
/// ```
/// use wqm_common::qdrant_endpoint::grpc_endpoint;
/// assert_eq!(grpc_endpoint("http://localhost:6333"), "http://localhost:6334");
/// assert_eq!(grpc_endpoint("http://localhost:6334"), "http://localhost:6334");
/// assert_eq!(grpc_endpoint("qdrant-host"), "qdrant-host:6334");
/// assert_eq!(grpc_endpoint("http://qdrant:7000"), "http://qdrant:7000");
/// ```
pub fn grpc_endpoint(url: &str) -> String {
    let rest = format!(":{QDRANT_REST_PORT}");
    let grpc = format!(":{QDRANT_GRPC_PORT}");
    if url.contains(&rest) {
        url.replace(&rest, &grpc)
    } else if !url.contains(&grpc) && !url.contains(':') {
        format!("{}{}", url.trim_end_matches('/'), grpc)
    } else {
        url.to_string()
    }
}

#[cfg(test)]
#[path = "qdrant_endpoint_tests.rs"]
mod tests;
