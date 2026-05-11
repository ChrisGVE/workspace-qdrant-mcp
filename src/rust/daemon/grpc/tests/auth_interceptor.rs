//! Integration tests for the gRPC auth interceptor wiring (F-023 / T4).
//!
//! Verifies that when `ServerConfig` is built with `new_secure(...)` (or any
//! config that enables `AuthConfig.enabled = true` with an `api_key`), every
//! gRPC service registered with the server is wrapped by the
//! `AuthInterceptor` and:
//!
//! 1. Rejects requests without an `authorization` metadata header
//!    (`Status::unauthenticated`).
//! 2. Rejects requests carrying a wrong API key
//!    (`Status::unauthenticated`).
//! 3. Accepts requests carrying the correct `Bearer <key>` header.
//!
//! These tests do not exercise the full `GrpcServer::start()` path (which
//! requires a dense embedding provider). Instead they directly compose the
//! same wiring helper (`make_auth_fn`) the factory uses, wrap a real
//! `SystemServiceServer` with `InterceptedService::new(...)`, bind an
//! ephemeral port, and call back through a generated tonic client.

use std::net::SocketAddr;
use std::time::Duration;

use tonic::metadata::MetadataValue;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Code, Request};
use workspace_qdrant_grpc::factory::make_auth_fn;
use workspace_qdrant_grpc::proto::system_service_client::SystemServiceClient;
use workspace_qdrant_grpc::proto::system_service_server::SystemServiceServer;
use workspace_qdrant_grpc::services::SystemServiceImpl;
use workspace_qdrant_grpc::{AuthConfig, ServerConfig};

const TEST_API_KEY: &str = "t4-integration-test-key";

/// Start a tonic server on an ephemeral port with a `SystemService` wrapped
/// by the auth interceptor built from `config`. Returns the bound socket
/// address and a oneshot sender that, when fired, gracefully shuts the
/// server down.
async fn spawn_authed_server(
    config: ServerConfig,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let auth_fn = make_auth_fn(&config);
    let service = SystemServiceServer::new(SystemServiceImpl::default());
    let wrapped = InterceptedService::new(service, auth_fn);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        Server::builder()
            .add_service(wrapped)
            .serve_with_incoming_shutdown(incoming, async {
                rx.await.ok();
            })
            .await
            .expect("server crashed");
    });

    // Give the listener a moment to start accepting.
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, tx)
}

/// Build a tonic client channel to the bound address.
async fn connect(addr: SocketAddr) -> Channel {
    Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .expect("failed to connect to test server")
}

fn secure_config() -> ServerConfig {
    // Use plain ServerConfig with auth enabled (no TLS) so the test does not
    // need to manage certificates. The auth wiring path is independent of
    // TLS configuration.
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    ServerConfig::new(addr).with_auth(AuthConfig {
        enabled: true,
        api_key: Some(TEST_API_KEY.to_string()),
        jwt_secret: None,
        allowed_origins: vec!["*".to_string()],
    })
}

#[tokio::test]
async fn rejects_request_without_authorization_header() {
    let (addr, shutdown) = spawn_authed_server(secure_config()).await;
    let channel = connect(addr).await;
    let mut client = SystemServiceClient::new(channel);

    // Plain request — no `authorization` metadata.
    let err = client
        .health(Request::new(()))
        .await
        .expect_err("auth interceptor must reject requests without an authorization header");
    assert_eq!(err.code(), Code::Unauthenticated, "got: {err:?}");

    shutdown.send(()).ok();
}

#[tokio::test]
async fn rejects_request_with_wrong_api_key() {
    let (addr, shutdown) = spawn_authed_server(secure_config()).await;
    let channel = connect(addr).await;
    let mut client = SystemServiceClient::new(channel);

    let mut req = Request::new(());
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from("Bearer wrong-key").unwrap(),
    );

    let err = client
        .health(req)
        .await
        .expect_err("auth interceptor must reject wrong API key");
    assert_eq!(err.code(), Code::Unauthenticated, "got: {err:?}");

    shutdown.send(()).ok();
}

#[tokio::test]
async fn accepts_request_with_correct_api_key() {
    let (addr, shutdown) = spawn_authed_server(secure_config()).await;
    let channel = connect(addr).await;
    let mut client = SystemServiceClient::new(channel);

    let mut req = Request::new(());
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {TEST_API_KEY}")).unwrap(),
    );

    let resp = client
        .health(req)
        .await
        .expect("correct API key must reach the inner service");
    // We only need to confirm the call reached the handler; the handler's
    // own response shape is exercised by system_service_tests.rs.
    let _ = resp.into_inner();

    shutdown.send(()).ok();
}

#[tokio::test]
async fn insecure_config_passes_through_without_credentials() {
    // Default ServerConfig::new(...) leaves auth disabled — wiring must not
    // reject calls that arrive without any credentials.
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let insecure = ServerConfig::new(addr);
    let (addr, shutdown) = spawn_authed_server(insecure).await;
    let channel = connect(addr).await;
    let mut client = SystemServiceClient::new(channel);

    let resp = client
        .health(Request::new(()))
        .await
        .expect("insecure config must allow unauthenticated calls");
    let _ = resp.into_inner();

    shutdown.send(()).ok();
}
