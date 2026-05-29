//! Core types for the MCP server: server identity, session state, and operating mode.
//!
//! This module is intentionally minimal in the scaffold phase.
//! Full type definitions are added in subsequent tasks.

/// MCP server name, matches the TypeScript server's `SERVER_NAME`.
pub const SERVER_NAME: &str = "workspace-qdrant";

/// Server version, populated from Cargo package version at compile time.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build number injected by build.rs from git commit count (4-digit hex).
pub const BUILD_NUMBER: &str = env!("BUILD_NUMBER");

/// Operating mode selected at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMode {
    /// JSON-RPC over stdin/stdout (default).
    Stdio,
    /// Streamable HTTP transport (requires `--http` flag or env var).
    Http,
}

impl Default for ServerMode {
    fn default() -> Self {
        Self::Stdio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_name_is_non_empty() {
        assert!(!SERVER_NAME.is_empty());
    }

    #[test]
    fn server_version_is_non_empty() {
        assert!(!SERVER_VERSION.is_empty());
    }

    #[test]
    fn build_number_is_non_empty() {
        assert!(!BUILD_NUMBER.is_empty());
    }

    #[test]
    fn default_mode_is_stdio() {
        assert_eq!(ServerMode::default(), ServerMode::Stdio);
    }
}
