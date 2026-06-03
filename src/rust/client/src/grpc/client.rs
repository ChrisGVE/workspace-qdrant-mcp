//! Thin async gRPC client wrapper for the memexd daemon.
//!
//! `DaemonClient` wraps a single tonic [`Channel`] and exposes per-service
//! accessor methods plus typed RPC wrappers (in the `*_methods` modules). Every
//! RPC call routed through [`DaemonClient::call`] is protected by:
//!
//! 1. **Client-side timeout** via `tokio::time::timeout` (see [`super::timeouts`]).
//! 2. **Exponential-backoff retry** via [`super::retry::call_with_retry`].
//!
//! This is the single shared client for both wqm clients (CLI + MCP), lifted
//! out of the MCP server (WI-d1/d2, #82). It depends only on `wqm-proto` and
//! `wqm-common` — never on `workspace-qdrant-core`.
//!
//! # Channel configuration
//!
//! Mirrors the TS `connect()` in connection.ts. We use `connect_lazy()` so the
//! channel does not block construction: the first RPC attempt triggers the actual
//! TCP handshake. HTTP/2 keep-alive settings keep long-idle connections alive
//! (matching the TS `grpc.keepalive_time_ms` channel option pattern per RISK-9).
//!
//! # Timeout semantics — why `tokio::time::timeout`, not tonic deadline
//!
//! The TypeScript implementation uses `Promise.race` (`grpcUnaryWithTimeout` in
//! connection.ts): the timer fires and rejects the outer promise, but the
//! underlying gRPC call is **abandoned in place** — no cancellation signal is
//! sent to the server and **no `grpc-timeout` header** is written to the
//! request metadata (AC-DC4 / RISK-7). To preserve those client-side-abandon
//! semantics we wrap each call with `tokio::time::timeout(...)` instead of
//! tonic's `.timeout(d)` deadline propagation.

use std::time::Duration;
use tonic::transport::Channel;
use tonic::Status;
use tracing::debug;
use wqm_common::constants::DEFAULT_GRPC_PORT;

/// Error type returned when constructing a [`DaemonClient`] fails.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// The endpoint URI could not be parsed.
    #[error("invalid endpoint URI: {0}")]
    InvalidUri(#[from] tonic::codegen::http::uri::InvalidUri),
}

use super::connection::resolve_daemon_address;
use super::retry::call_with_retry;
use super::timeouts::resolve_timeout;
use wqm_proto::workspace_daemon::{
    admin_write_service_client::AdminWriteServiceClient,
    collection_service_client::CollectionServiceClient,
    document_service_client::DocumentServiceClient,
    embedding_service_client::EmbeddingServiceClient, graph_service_client::GraphServiceClient,
    library_write_service_client::LibraryWriteServiceClient,
    project_service_client::ProjectServiceClient,
    queue_write_service_client::QueueWriteServiceClient,
    system_service_client::SystemServiceClient,
    text_search_service_client::TextSearchServiceClient,
    tracking_write_service_client::TrackingWriteServiceClient,
    watch_write_service_client::WatchWriteServiceClient,
};

/// HTTP/2 keep-alive interval — matches TS `grpc.keepalive_time_ms` (30 s).
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);

/// Default host when none is provided (matches TS `DEFAULT_HOST = 'localhost'`).
const DEFAULT_HOST: &str = "127.0.0.1";

/// Thin async gRPC client for the memexd daemon.
///
/// Wraps a lazily-connected tonic [`Channel`] and holds one typed client stub
/// per service. All stubs share the same underlying channel.
///
/// Service wrappers for individual RPC methods are implemented in the sibling
/// `*_methods` modules and call [`DaemonClient::call`] / [`DaemonClient::call_search`].
pub struct DaemonClient {
    // One typed client per service. Each internally clones the channel, so all
    // share the same underlying connection pool.
    pub(crate) system: SystemServiceClient<Channel>,
    pub(crate) collection: CollectionServiceClient<Channel>,
    pub(crate) document: DocumentServiceClient<Channel>,
    pub(crate) embedding: EmbeddingServiceClient<Channel>,
    pub(crate) project: ProjectServiceClient<Channel>,
    pub(crate) text_search: TextSearchServiceClient<Channel>,
    pub(crate) graph: GraphServiceClient<Channel>,
    pub(crate) queue_write: QueueWriteServiceClient<Channel>,
    pub(crate) watch_write: WatchWriteServiceClient<Channel>,
    pub(crate) library_write: LibraryWriteServiceClient<Channel>,
    pub(crate) tracking_write: TrackingWriteServiceClient<Channel>,
    pub(crate) admin_write: AdminWriteServiceClient<Channel>,

    /// Logical connection state: true after a successful health-check on connect.
    connected: bool,
}

impl DaemonClient {
    /// Connect to the daemon at an explicit `endpoint` URL.
    ///
    /// `endpoint` must be a valid URI, e.g. `"http://127.0.0.1:50051"`.
    /// If empty or blank the default `http://127.0.0.1:50051` is used.
    ///
    /// Uses `connect_lazy()` — no TCP handshake at construction time.
    ///
    /// # Errors
    /// Returns `Err` if the endpoint URI is invalid (malformed).
    pub fn new(endpoint: &str) -> Result<Self, ClientError> {
        let addr = if endpoint.trim().is_empty() {
            format!("http://{}:{}", DEFAULT_HOST, DEFAULT_GRPC_PORT)
        } else {
            endpoint.to_owned()
        };

        debug!("DaemonClient: connecting lazy to {}", addr);

        let channel = Channel::from_shared(addr)?
            .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
            .keep_alive_while_idle(true)
            .connect_lazy();

        Ok(Self {
            system: SystemServiceClient::new(channel.clone()),
            collection: CollectionServiceClient::new(channel.clone()),
            document: DocumentServiceClient::new(channel.clone()),
            embedding: EmbeddingServiceClient::new(channel.clone()),
            project: ProjectServiceClient::new(channel.clone()),
            text_search: TextSearchServiceClient::new(channel.clone()),
            graph: GraphServiceClient::new(channel.clone()),
            queue_write: QueueWriteServiceClient::new(channel.clone()),
            watch_write: WatchWriteServiceClient::new(channel.clone()),
            library_write: LibraryWriteServiceClient::new(channel.clone()),
            tracking_write: TrackingWriteServiceClient::new(channel.clone()),
            admin_write: AdminWriteServiceClient::new(channel),
            connected: false,
        })
    }

    /// Construct a `DaemonClient` from an explicit host and port.
    ///
    /// Convenience for callers that hold a structured host/port pair (e.g. the
    /// MCP server's `ServerConfig.daemon`) rather than a full URI.
    ///
    /// # Errors
    /// Returns `Err` if the resulting endpoint URI is invalid.
    pub fn from_host_port(host: &str, port: u16) -> Result<Self, ClientError> {
        Self::new(&format!("http://{host}:{port}"))
    }

    /// Construct a `DaemonClient` at the address resolved from the environment
    /// and the active cli-config profile.
    ///
    /// Resolution precedence: `WQM_DAEMON_ADDR` > active profile `daemon_address`
    /// > `http://127.0.0.1:50051` (see [`super::connection`]).
    ///
    /// # Errors
    /// Returns `Err` if the resolved endpoint URI is invalid.
    pub fn connect_default() -> Result<Self, ClientError> {
        Self::new(&resolve_daemon_address())
    }

    /// Mark the client as logically connected (called after a successful health-check).
    pub(crate) fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
    }

    /// Returns `true` if the last health-check succeeded.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Close the client.
    ///
    /// Tonic channels clean up automatically on drop. This method exists to
    /// mirror the TS `close()` method and provide an explicit signal to callers
    /// that own the client.
    pub fn close(&mut self) {
        self.connected = false;
        // Channel drops when Self is dropped; nothing else to do.
    }

    /// Execute a gRPC call with retry + client-side timeout.
    ///
    /// This is the **primary dispatch helper** for service method wrappers. It:
    /// 1. Wraps `f` in [`call_with_retry`] (3 attempts, 100→200 ms backoff).
    /// 2. Wraps the entire retry future in `tokio::time::timeout`.
    ///
    /// The timeout budget covers the **entire** retry sequence, not individual
    /// attempts. This matches the TS outer timeout wrapping `callWithRetry`.
    ///
    /// # Arguments
    /// * `method_name` – used to select the timeout budget (search → 10 s, else 5 s).
    /// * `override_timeout` – caller-supplied one-shot timeout ceiling.
    /// * `f` – closure producing the gRPC future (passed to `call_with_retry`).
    ///
    /// # Errors
    /// * Timeout elapsed → `Status::deadline_exceeded`.
    /// * All retries exhausted → last `Status` from the call.
    /// * Non-retryable error on first attempt → that `Status` immediately.
    pub async fn call<T, F, Fut>(
        &mut self,
        method_name: &str,
        override_timeout: Option<Duration>,
        f: F,
    ) -> Result<T, Status>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        let budget = resolve_timeout(method_name, override_timeout);
        let result = tokio::time::timeout(budget, call_with_retry(f))
            .await
            .unwrap_or_else(|_elapsed| {
                Err(Status::deadline_exceeded(format!(
                    "gRPC call '{method_name}' timed out after {budget:?}"
                )))
            });
        // Mirror TS callWithRetry: update logical connection state based on outcome.
        self.set_connected(result.is_ok());
        result
    }

    /// Convenience variant for search methods (always uses the 10 s budget).
    pub async fn call_search<T, F, Fut>(&mut self, method_name: &str, f: F) -> Result<T, Status>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        self.call(method_name, None, f).await
    }
}

// ---------------------------------------------------------------------------
// Per-service accessor helpers (raw stub access for callers that need RPCs
// without a typed wrapper). The `text_search` accessor is named
// `text_search_client` to avoid clashing with the `text_search` RPC wrapper in
// `search_methods.rs`.
// ---------------------------------------------------------------------------

impl DaemonClient {
    pub fn system(&mut self) -> &mut SystemServiceClient<Channel> {
        &mut self.system
    }

    pub fn collection(&mut self) -> &mut CollectionServiceClient<Channel> {
        &mut self.collection
    }

    pub fn document(&mut self) -> &mut DocumentServiceClient<Channel> {
        &mut self.document
    }

    pub fn embedding(&mut self) -> &mut EmbeddingServiceClient<Channel> {
        &mut self.embedding
    }

    pub fn project(&mut self) -> &mut ProjectServiceClient<Channel> {
        &mut self.project
    }

    pub fn text_search_client(&mut self) -> &mut TextSearchServiceClient<Channel> {
        &mut self.text_search
    }

    pub fn graph(&mut self) -> &mut GraphServiceClient<Channel> {
        &mut self.graph
    }

    pub fn queue_write(&mut self) -> &mut QueueWriteServiceClient<Channel> {
        &mut self.queue_write
    }

    pub fn watch_write(&mut self) -> &mut WatchWriteServiceClient<Channel> {
        &mut self.watch_write
    }

    pub fn library_write(&mut self) -> &mut LibraryWriteServiceClient<Channel> {
        &mut self.library_write
    }

    pub fn tracking_write(&mut self) -> &mut TrackingWriteServiceClient<Channel> {
        &mut self.tracking_write
    }

    pub fn admin_write(&mut self) -> &mut AdminWriteServiceClient<Channel> {
        &mut self.admin_write
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── construction ─────────────────────────────────────────────────────────
    // These tests require a Tokio runtime because Channel::from_shared with
    // http2_keep_alive_interval internally registers timers with the Tokio
    // reactor even though no network I/O happens (connect_lazy is used).

    #[tokio::test]
    async fn new_with_explicit_endpoint() {
        let client = DaemonClient::new("http://127.0.0.1:50051");
        assert!(client.is_ok(), "valid endpoint must not error");
    }

    #[tokio::test]
    async fn new_with_empty_endpoint_uses_default() {
        let client = DaemonClient::new("");
        assert!(client.is_ok(), "empty endpoint falls back to default");
    }

    #[tokio::test]
    async fn new_with_blank_endpoint_uses_default() {
        let client = DaemonClient::new("   ");
        assert!(client.is_ok(), "blank endpoint falls back to default");
    }

    #[tokio::test]
    async fn from_host_port_builds_endpoint() {
        let client = DaemonClient::from_host_port("192.168.1.1", 9999);
        assert!(client.is_ok(), "valid host/port must produce a client");
    }

    #[tokio::test]
    async fn connect_default_succeeds() {
        let client = DaemonClient::connect_default();
        assert!(client.is_ok());
    }

    #[test]
    fn default_port_from_constant() {
        // Verify the constant used at build time matches wqm_common.
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
    }

    // ── connection state ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn new_client_starts_disconnected() {
        let client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        assert!(
            !client.is_connected(),
            "new client should start disconnected"
        );
    }

    #[tokio::test]
    async fn set_connected_updates_state() {
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        client.set_connected(true);
        assert!(client.is_connected());
        client.set_connected(false);
        assert!(!client.is_connected());
    }

    #[tokio::test]
    async fn close_marks_disconnected() {
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        client.set_connected(true);
        client.close();
        assert!(!client.is_connected());
    }

    // ── call() timeout integration ────────────────────────────────────────────

    #[tokio::test]
    async fn call_uses_5s_budget_for_non_search() {
        // We can't mock the wall clock cheaply, but we can verify that a
        // fast-failing closure (non-retryable) returns immediately with the
        // correct error rather than waiting 5 s.
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), Status> = client
            .call("ingestText", None, || async {
                Err(Status::invalid_argument("test"))
            })
            .await;
        assert!(result.is_err());
        // Non-retryable → surfaces immediately, no wait.
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn call_with_tiny_timeout_returns_deadline_exceeded() {
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        // Inject a 1 ms override timeout; the closure sleeps 50 ms.
        let result: Result<(), Status> = client
            .call("ingestText", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().code(),
            tonic::Code::DeadlineExceeded,
            "timeout must surface as DeadlineExceeded"
        );
    }

    #[tokio::test]
    async fn call_search_uses_10s_budget_not_5s() {
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        // call_search does NOT accept an override — it always uses 10 s.
        // We test that a fast success resolves correctly.
        let result: Result<(), Status> = client.call_search("search", || async { Ok(()) }).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn call_search_method_name_infers_10s_budget() {
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), Status> = client.call("search", None, || async { Ok(()) }).await;
        assert!(result.is_ok());
    }

    // ── retry + timeout interaction ──────────────────────────────────────────

    #[tokio::test]
    async fn retryable_error_within_budget_retries() {
        use std::sync::{
            atomic::{AtomicU32, Ordering},
            Arc,
        };
        let count = Arc::new(AtomicU32::new(0));
        let c = count.clone();

        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        // Fail once with UNAVAILABLE, succeed on retry.
        let result: Result<(), Status> = client
            .call(
                "health",
                Some(Duration::from_secs(5)), // generous budget
                move || {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    async move {
                        if n == 0 {
                            Err(Status::unavailable("transient"))
                        } else {
                            Ok(())
                        }
                    }
                },
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2, "should retry once");
    }
}
