// Integration tests: embedding tool + daemon retry/timeout policy.
//
// it_embedding_provider_status   — real get_embedding_provider_status() RPC
// it_daemon_retry_dead_port      — retry+timeout against an unreachable port
// it_retry_exhausts_max_retries  — injected UNAVAILABLE exhausts MAX_RETRIES
// it_tiny_timeout_deadline       — 1 ms override → DeadlineExceeded

use super::helpers;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Live daemon: embedding provider status
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_embedding_provider_status() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let resp = client
        .get_embedding_provider_status()
        .await
        .expect("get_embedding_provider_status must succeed against live daemon");

    assert!(
        !resp.provider.is_empty(),
        "provider field must be non-empty; got: {:?}",
        resp.provider
    );
    assert!(
        !resp.model.is_empty(),
        "model field must be non-empty; got: {:?}",
        resp.model
    );
    assert!(
        resp.output_dim > 0,
        "output_dim must be positive; got: {}",
        resp.output_dim
    );
    assert!(
        !resp.probe_status.is_empty(),
        "probe_status must be non-empty"
    );
}

// ---------------------------------------------------------------------------
// Failure-path: retry+timeout against a dead port (no live daemon needed).
// Port 1 is always ECONNREFUSED/unreachable without root on macOS/Linux.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_daemon_retry_dead_port_surfaces_error_within_budget() {
    let mut client = mcp_server::grpc::client::DaemonClient::new("http://127.0.0.1:1")
        .expect("constructing client to dead port must not fail");

    let start = Instant::now();
    // 2 s override covers all 3 retry attempts (100+200 ms backoff) with room.
    let result: Result<mcp_server::proto::HealthResponse, tonic::Status> = client
        .call("health", Some(Duration::from_secs(2)), || async {
            // Re-construct each attempt so we can call the dead-port health RPC.
            let mut c = mcp_server::grpc::client::DaemonClient::new("http://127.0.0.1:1").unwrap();
            c.health().await
        })
        .await;
    let elapsed = start.elapsed();

    assert!(result.is_err(), "call to dead port must return Err; got Ok");
    let code = result.unwrap_err().code();
    assert!(
        matches!(
            code,
            tonic::Code::Unavailable
                | tonic::Code::DeadlineExceeded
                | tonic::Code::Unknown
                | tonic::Code::Internal
        ),
        "expected connectivity/timeout error code; got: {code:?}"
    );
    // Must complete well within the 2 s budget.
    assert!(
        elapsed < Duration::from_secs(4),
        "retry loop must finish within budget; elapsed: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Retry exhaustion: injected UNAVAILABLE exhausts MAX_RETRIES
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_retry_exhausts_max_retries() {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    let count = Arc::new(AtomicU32::new(0));
    let c = count.clone();

    let mut client = mcp_server::grpc::client::DaemonClient::new("http://127.0.0.1:50051")
        .expect("client construction must succeed");

    let result: Result<(), tonic::Status> = client
        .call("health", Some(Duration::from_secs(5)), move || {
            c.fetch_add(1, Ordering::SeqCst);
            async { Err(tonic::Status::unavailable("injected")) }
        })
        .await;

    assert!(result.is_err(), "exhausted retries must return Err");
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::Unavailable,
        "last error code must be preserved"
    );
    assert_eq!(
        count.load(Ordering::SeqCst),
        mcp_server::grpc::retry::MAX_RETRIES,
        "must attempt exactly MAX_RETRIES times"
    );
}

// ---------------------------------------------------------------------------
// Timeout override: 1 ms budget → DeadlineExceeded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_tiny_timeout_deadline_exceeded() {
    let mut client = mcp_server::grpc::client::DaemonClient::new("http://127.0.0.1:50051")
        .expect("client construction must succeed");

    let result: Result<(), tonic::Status> = client
        .call("health", Some(Duration::from_millis(1)), || async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(())
        })
        .await;

    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::DeadlineExceeded,
        "1 ms override must surface DeadlineExceeded"
    );
}
