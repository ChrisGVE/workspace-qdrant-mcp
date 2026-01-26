#![cfg(feature = "legacy_grpc_tests")]
//! Enhanced connection lifecycle and management tests for gRPC daemon-MCP communication
//!
//! This module provides comprehensive testing of gRPC connection establishment, teardown,
//! lifecycle management, health checking, and graceful shutdown procedures.

use shared_test_utils::{async_test, TestResult};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;
use tonic::transport::{Channel, Server};
use tonic::{Request, Code, Status};
use workspace_qdrant_grpc::{
    ServerConfig, AuthConfig, TimeoutConfig, PerformanceConfig,
    service::IngestionService,
    proto::{
        ingest_service_client::IngestServiceClient,
        ingest_service_server::IngestServiceServer,
        *,
    },
};

/// Connection state tracking for lifecycle tests
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Draining,
    ShuttingDown,
    Terminated,
}

/// Connection lifecycle tracker for testing
pub struct ConnectionTracker {
    state: Arc<RwLock<ConnectionState>>,
    connection_count: Arc<Mutex<usize>>,
    disconnection_count: Arc<Mutex<usize>>,
    last_activity: Arc<Mutex<Instant>>,
}

impl ConnectionTracker {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            connection_count: Arc::new(Mutex::new(0)),
            disconnection_count: Arc::new(Mutex::new(0)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub async fn get_state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }

    pub async fn set_state(&self, state: ConnectionState) {
        *self.state.write().await = state;
        *self.last_activity.lock().await = Instant::now();
    }

    pub async fn increment_connections(&self) {
        *self.connection_count.lock().await += 1;
        *self.last_activity.lock().await = Instant::now();
    }

    pub async fn increment_disconnections(&self) {
        *self.disconnection_count.lock().await += 1;
        *self.last_activity.lock().await = Instant::now();
    }

    pub async fn get_connection_count(&self) -> usize {
        *self.connection_count.lock().await
    }

    pub async fn get_disconnection_count(&self) -> usize {
        *self.disconnection_count.lock().await
    }

    pub async fn time_since_activity(&self) -> Duration {
        self.last_activity.lock().await.elapsed()
    }
}

/// Enhanced test fixture with connection lifecycle tracking
pub struct LifecycleTestFixture {
    pub server_addr: SocketAddr,
    pub client: IngestServiceClient<Channel>,
    pub tracker: Arc<ConnectionTracker>,
    pub _server_handle: tokio::task::JoinHandle<()>,
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl LifecycleTestFixture {
    /// Create new fixture with connection tracking
    pub async fn new() -> TestResult<Self> {
        Self::new_with_config(ServerConfig::new("127.0.0.1:0".parse()?)).await
    }

    /// Create fixture with custom configuration and tracking
    pub async fn new_with_config(mut config: ServerConfig) -> TestResult<Self> {
        let tracker = Arc::new(ConnectionTracker::new());

        // Find available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        config.bind_addr = addr;
        drop(listener);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        // Start server
        let server_config = config.clone();
        let tracker_clone = tracker.clone();
        let server_handle = tokio::spawn(async move {
            tracker_clone.set_state(ConnectionState::Connecting).await;

            let service = IngestionService::new_with_auth(server_config.auth_config.clone());
            let svc = IngestServiceServer::new(service);

            let server = Server::builder()
                .timeout(server_config.timeout_config.request_timeout)
                .add_service(svc);

            tracker_clone.set_state(ConnectionState::Connected).await;

            server
                .serve_with_shutdown(server_config.bind_addr, async {
                    shutdown_rx.await.ok();
                    tracker_clone.set_state(ConnectionState::ShuttingDown).await;
                })
                .await
                .expect("Server failed to start");

            tracker_clone.set_state(ConnectionState::Terminated).await;
        });

        // Wait for server to be ready
        tokio::time::sleep(Duration::from_millis(100)).await;

        tracker.set_state(ConnectionState::Connected).await;

        let channel = Channel::from_shared(format!("http://{}", addr))?
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .connect()
            .await?;

        tracker.increment_connections().await;

        let client = IngestServiceClient::new(channel);

        Ok(Self {
            server_addr: addr,
            client,
            tracker,
            _server_handle: server_handle,
            shutdown_tx,
        })
    }

    /// Gracefully shutdown with state tracking
    pub async fn shutdown(self) -> TestResult<()> {
        self.tracker.set_state(ConnectionState::Draining).await;
        let _ = self.shutdown_tx.send(());

        tokio::time::sleep(Duration::from_millis(100)).await;

        self.tracker.increment_disconnections().await;
        self.tracker.set_state(ConnectionState::Terminated).await;

        Ok(())
    }

    /// Verify connection is healthy
    pub async fn verify_healthy(&mut self) -> TestResult<()> {
        let response = self.client.health_check(Request::new(())).await?;
        let health = response.into_inner();

        if health.status != HealthStatus::Healthy as i32 {
            return Err(format!("Server not healthy: {:?}", health.message).into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod connection_establishment_tests {
    use super::*;

    async_test!(test_basic_connection_establishment, {
        let fixture = LifecycleTestFixture::new().await?;

        // Verify connection established
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Connected);
        assert_eq!(fixture.tracker.get_connection_count().await, 1);

        // Verify connection works
        let mut client = fixture.client.clone();
        let response = client.health_check(Request::new(())).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        fixture.shutdown().await?;

        // Verify shutdown
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Terminated);
        assert_eq!(fixture.tracker.get_disconnection_count().await, 1);

        Ok(())
    });

    async_test!(test_connection_with_custom_timeouts, {
        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(2),
            keepalive_interval: Duration::from_secs(10),
            keepalive_timeout: Duration::from_secs(3),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_timeouts(timeout_config);

        let fixture = LifecycleTestFixture::new_with_config(config).await?;

        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Connected);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_multiple_sequential_connections, {
        for i in 0..5 {
            let fixture = LifecycleTestFixture::new().await?;

            assert_eq!(fixture.tracker.get_connection_count().await, 1);

            let mut client = fixture.client.clone();
            let response = client.health_check(Request::new(())).await?;
            assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

            fixture.shutdown().await?;

            assert_eq!(fixture.tracker.get_disconnection_count().await, 1);

            if i < 4 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        Ok(())
    });

    async_test!(test_connection_pool_behavior, {
        let fixture = LifecycleTestFixture::new().await?;

        // Create multiple clients sharing the same channel
        let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
            .connect()
            .await?;

        let clients: Vec<IngestServiceClient<Channel>> = (0..10)
            .map(|_| IngestServiceClient::new(channel.clone()))
            .collect();

        // All clients should work using connection pooling
        for mut client in clients {
            let response = client.health_check(Request::new(())).await?;
            assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_connection_state_transitions, {
        let fixture = LifecycleTestFixture::new().await?;

        // Verify initial state
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Connected);

        // Make request to ensure connection is active
        let mut client = fixture.client.clone();
        client.health_check(Request::new(())).await?;

        // Still connected after request
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Connected);

        // Initiate shutdown
        fixture.tracker.set_state(ConnectionState::Draining).await;
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Draining);

        fixture.shutdown().await?;

        // Verify final state
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Terminated);

        Ok(())
    });
}

#[cfg(test)]
mod health_check_tests {
    use super::*;

    async_test!(test_health_check_during_startup, {
        let fixture = LifecycleTestFixture::new().await?;

        // Server should be healthy immediately after startup
        let mut client = fixture.client.clone();
        let response = client.health_check(Request::new(())).await?;
        let health = response.into_inner();

        assert!(
            health.status == HealthStatus::Healthy as i32 ||
            health.status == HealthStatus::Degraded as i32
        );
        assert!(!health.message.is_empty());
        assert!(!health.services.is_empty());

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_repeated_health_checks, {
        let fixture = LifecycleTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Perform multiple health checks
        for _ in 0..10 {
            let response = client.health_check(Request::new(())).await?;
            let health = response.into_inner();

            assert!(
                health.status == HealthStatus::Healthy as i32 ||
                health.status == HealthStatus::Degraded as i32
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_health_check_with_timeout, {
        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_millis(500),
            connection_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(5),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_timeouts(timeout_config);

        let fixture = LifecycleTestFixture::new_with_config(config).await?;
        let mut client = fixture.client.clone();

        // Health check should complete within timeout
        let start = Instant::now();
        let response = client.health_check(Request::new(())).await?;
        let duration = start.elapsed();

        assert!(duration < Duration::from_millis(500));
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_health_check_after_shutdown, {
        let fixture = LifecycleTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Initial health check succeeds
        let response = client.health_check(Request::new(())).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        // Shutdown server
        fixture.shutdown().await?;

        // Health check should fail
        let result = client.health_check(Request::new(())).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(
            error.code() == Code::Unavailable ||
            error.code() == Code::Internal ||
            error.message().contains("connection") ||
            error.message().contains("transport")
        );

        Ok(())
    });
}

#[cfg(test)]
mod graceful_shutdown_tests {
    use super::*;

    async_test!(test_graceful_shutdown_no_active_requests, {
        let fixture = LifecycleTestFixture::new().await?;

        // No active requests, should shutdown immediately
        let start = Instant::now();
        fixture.shutdown().await?;
        let duration = start.elapsed();

        assert!(duration < Duration::from_secs(1));
        assert_eq!(fixture.tracker.get_state().await, ConnectionState::Terminated);

        Ok(())
    });

    async_test!(test_graceful_shutdown_with_active_request, {
        let fixture = LifecycleTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Start a request
        let request_future = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            client.health_check(Request::new(())).await
        };

        // Start shutdown after brief delay
        let shutdown_future = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            fixture.shutdown().await
        };

        // Both should complete successfully
        let (request_result, shutdown_result) = tokio::join!(request_future, shutdown_future);

        // Request might succeed or fail depending on timing
        match request_result {
            Ok(_) => {}, // Request completed before shutdown
            Err(_) => {}, // Request failed due to shutdown - acceptable
        }

        assert!(shutdown_result.is_ok());

        Ok(())
    });

    async_test!(test_shutdown_timeout_behavior, {
        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(10),
            keepalive_timeout: Duration::from_secs(2),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_timeouts(timeout_config);

        let fixture = LifecycleTestFixture::new_with_config(config).await?;

        // Shutdown should respect timeout configuration
        let start = Instant::now();
        fixture.shutdown().await?;
        let duration = start.elapsed();

        // Should complete relatively quickly when no active requests
        assert!(duration < Duration::from_secs(5));

        Ok(())
    });

    async_test!(test_connection_cleanup_after_shutdown, {
        let fixture = LifecycleTestFixture::new().await?;
        let server_addr = fixture.server_addr;

        // Verify connection works
        let mut client = fixture.client.clone();
        client.health_check(Request::new(())).await?;

        fixture.shutdown().await?;

        // Try to create new connection to same address - should fail
        let connect_result = timeout(
            Duration::from_secs(2),
            Channel::from_shared(format!("http://{}", server_addr))
                .unwrap()
                .connect()
        ).await;

        assert!(connect_result.is_err() || connect_result.unwrap().is_err());

        Ok(())
    });
}

#[cfg(test)]
mod connection_resilience_tests {
    use super::*;

    async_test!(test_connection_recovery_after_brief_disconnect, {
        let fixture = LifecycleTestFixture::new().await?;
        let mut client = fixture.client.clone();

        // Initial connection works
        let response = client.health_check(Request::new(())).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        // Simulate brief network interruption by creating new client
        // (In real scenarios, the channel would handle reconnection)
        let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
            .connect()
            .await?;

        let mut new_client = IngestServiceClient::new(channel);

        // New connection should work
        let response = new_client.health_check(Request::new(())).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_multiple_connection_attempts, {
        let fixture = LifecycleTestFixture::new().await?;

        // Attempt multiple connections to same server
        for i in 0..5 {
            let channel = Channel::from_shared(format!("http://{}", fixture.server_addr))?
                .connect()
                .await?;

            let mut client = IngestServiceClient::new(channel);
            let response = client.health_check(Request::new(())).await?;

            assert_eq!(
                response.into_inner().status,
                HealthStatus::Healthy as i32,
                "Connection {} failed", i
            );
        }

        fixture.shutdown().await?;
        Ok(())
    });

    async_test!(test_connection_keepalive, {
        let timeout_config = TimeoutConfig {
            request_timeout: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(1), // Short keepalive for testing
            keepalive_timeout: Duration::from_secs(2),
        };

        let config = ServerConfig::new("127.0.0.1:0".parse()?)
            .with_timeouts(timeout_config);

        let fixture = LifecycleTestFixture::new_with_config(config).await?;
        let mut client = fixture.client.clone();

        // Make initial request
        client.health_check(Request::new(())).await?;

        // Wait for keepalive period
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Connection should still be alive
        let response = client.health_check(Request::new(())).await?;
        assert_eq!(response.into_inner().status, HealthStatus::Healthy as i32);

        fixture.shutdown().await?;
        Ok(())
    });
}
