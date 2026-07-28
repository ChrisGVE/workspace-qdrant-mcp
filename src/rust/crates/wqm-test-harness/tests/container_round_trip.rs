//! The harness proving itself against a real container.
//!
//! The v0.1 harness's defining failure was that nothing ever called it, so nobody
//! discovered it could not work (its readiness poll requested an endpoint current
//! Qdrant answers 404). A harness with no caller is a harness with no evidence, and
//! this file is this harness's caller.
//!
//! It runs under [`with_store`], which means it also demonstrates the discipline it
//! is testing: on a machine with no container runtime it does not quietly pass — it
//! announces `WQM-TEST-SKIP` with a reason, and CI gates on the set of those
//! (`P04-GT001-WO014`).

use std::time::Duration;

use wqm_test_harness::with_store;

/// Start a container, reach it, and let `Drop` remove it.
///
/// The assertion deliberately uses `/healthz` rather than `/health`: the latter is
/// the endpoint the v0.1 harness waited on, and pinning the working one here means a
/// regression to it fails a test rather than hanging a build.
#[tokio::test]
async fn an_ephemeral_store_is_reachable_and_answers_readiness() {
    with_store(
        "an_ephemeral_store_is_reachable_and_answers_readiness",
        |ep| async move {
            assert!(
                ep.http_url().starts_with("http://"),
                "endpoint should be a URL, got {}",
                ep.http_url()
            );
            assert_ne!(
                ep.http_url(),
                ep.grpc_url(),
                "HTTP and gRPC must be distinct ports"
            );

            // The container is already ready -- `with_store` waited on Qdrant's own
            // "gRPC listening" line before handing the endpoint over. This is a
            // confirmation, not a poll, so a single attempt is the correct shape: if it
            // needs retries, the readiness condition is wrong and should fail loudly.
            let body = fetch(&format!("{}/healthz", ep.http_url())).await;
            assert!(
                body.is_some(),
                "a store handed over by the harness must already be answering /healthz"
            );

            // And the endpoint is NOT the developer's production instance -- the whole
            // point of defect (a). A container maps to an ephemeral high port.
            assert!(
                !ep.http_url().ends_with(":6333"),
                "an ephemeral store must not be on the default production port: {}",
                ep.http_url()
            );
        },
    )
    .await;
}

/// Minimal one-shot GET over a raw socket, so the harness's own test does not drag in
/// an HTTP client the harness itself does not need.
async fn fetch(url: &str) -> Option<String> {
    let rest = url.strip_prefix("http://")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::net::TcpStream::connect(authority),
    )
    .await
    .ok()?
    .ok()?;

    let request = format!("GET {path} HTTP/1.0\r\nHost: {authority}\r\n\r\n");
    stream.write_all(request.as_bytes()).await.ok()?;

    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut buf))
        .await
        .ok()?
        .ok()?;

    let text = String::from_utf8_lossy(&buf).into_owned();
    text.starts_with("HTTP/1.0 200").then_some(text)
}
