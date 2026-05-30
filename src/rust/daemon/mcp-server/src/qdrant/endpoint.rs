//! gRPC endpoint resolution for the Qdrant read client.
//!
//! The TypeScript MCP server talks to Qdrant over its **REST** API (default
//! port `6333`), so deployments set `QDRANT_URL` (and the config default) to a
//! `:6333` URL.  The Rust port uses `qdrant_client::Qdrant`, which speaks
//! **gRPC** — served by Qdrant on port `6334`.  Pointing the gRPC client at the
//! REST port yields an opaque `h2 protocol error` on the first call.
//!
//! [`grpc_endpoint`] translates a REST-style URL to the gRPC port, mirroring
//! the daemon's `build_connection_url` (`core/src/storage/client.rs`) so the
//! whole workspace shares one convention: `:6333` → `:6334`.

/// Translate a Qdrant URL to its gRPC endpoint.
///
/// Rules (identical to the daemon's `build_connection_url`):
/// - A `:6333` port is rewritten to `:6334`.
/// - A bare host with no port (and no scheme colon) gets `:6334` appended.
/// - Any URL that already targets `:6334` (or some other explicit port) is
///   returned unchanged.
pub fn grpc_endpoint(url: &str) -> String {
    if url.contains(":6333") {
        url.replace(":6333", ":6334")
    } else if !url.contains(":6334") && !url.contains(':') {
        format!("{}:6334", url.trim_end_matches('/'))
    } else {
        url.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::grpc_endpoint;

    #[test]
    fn rewrites_rest_port_to_grpc() {
        assert_eq!(
            grpc_endpoint("http://localhost:6333"),
            "http://localhost:6334"
        );
    }

    #[test]
    fn rewrites_rest_port_with_trailing_path() {
        assert_eq!(
            grpc_endpoint("https://remote-qdrant:6333/"),
            "https://remote-qdrant:6334/"
        );
    }

    #[test]
    fn leaves_grpc_port_untouched() {
        assert_eq!(
            grpc_endpoint("http://localhost:6334"),
            "http://localhost:6334"
        );
    }

    #[test]
    fn leaves_other_explicit_port_untouched() {
        // A non-default explicit port is an operator choice; do not rewrite it.
        assert_eq!(grpc_endpoint("http://qdrant:7000"), "http://qdrant:7000");
    }

    #[test]
    fn appends_grpc_port_to_bare_host() {
        // No scheme, no port → append the gRPC port.
        assert_eq!(grpc_endpoint("qdrant-host"), "qdrant-host:6334");
    }
}
