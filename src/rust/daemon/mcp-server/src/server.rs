//! McpServer: top-level struct that owns transport setup, session lifecycle,
//! and gRPC client handles.
//!
//! This module is intentionally minimal in the scaffold phase.
//! Full implementation is added in subsequent tasks.

use crate::server_types::ServerMode;

/// Central server struct.
///
/// Owns the chosen transport, the gRPC channel to memexd, and the optional
/// SQLite read handle.  Instantiated once per process and driven by `main`.
pub struct McpServer {
    mode: ServerMode,
}

impl McpServer {
    /// Create a new server configured for the given operating mode.
    pub fn new(mode: ServerMode) -> Self {
        Self { mode }
    }

    /// Return the operating mode this server was configured with.
    pub fn mode(&self) -> ServerMode {
        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_stores_mode() {
        let s = McpServer::new(ServerMode::Stdio);
        assert_eq!(s.mode(), ServerMode::Stdio);

        let s = McpServer::new(ServerMode::Http);
        assert_eq!(s.mode(), ServerMode::Http);
    }
}
