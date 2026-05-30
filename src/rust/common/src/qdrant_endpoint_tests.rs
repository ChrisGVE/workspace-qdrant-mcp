//! Tests for the canonical Qdrant gRPC endpoint translation.

use super::{grpc_endpoint, QDRANT_GRPC_PORT, QDRANT_REST_PORT};

#[test]
fn ports_are_the_qdrant_defaults() {
    assert_eq!(QDRANT_REST_PORT, 6333);
    assert_eq!(QDRANT_GRPC_PORT, 6334);
}

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
fn rewrites_rest_port_for_ipv4_literal() {
    assert_eq!(
        grpc_endpoint("http://127.0.0.1:6333"),
        "http://127.0.0.1:6334"
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
    assert_eq!(grpc_endpoint("http://qdrant:7000"), "http://qdrant:7000");
}

#[test]
fn appends_grpc_port_to_bare_host() {
    assert_eq!(grpc_endpoint("qdrant-host"), "qdrant-host:6334");
}

#[test]
fn appends_grpc_port_to_bare_host_with_trailing_slash() {
    assert_eq!(grpc_endpoint("qdrant-host/"), "qdrant-host:6334");
}

#[test]
fn idempotent_on_already_translated_url() {
    let once = grpc_endpoint("http://localhost:6333");
    let twice = grpc_endpoint(&once);
    assert_eq!(once, twice);
}
