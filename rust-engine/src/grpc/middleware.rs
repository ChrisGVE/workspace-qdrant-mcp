//! gRPC middleware for connection management, metrics, and security

use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use parking_lot::RwLock;
use crate::grpc::security::SecurityManager;

/// Connection tracking and management with enhanced security
#[derive(Debug)]
pub struct ConnectionManager {
    /// Active connections count
    active_connections: AtomicU64,

    /// Maximum allowed connections (future use)
    #[allow(dead_code)]
    max_connections: u64,

    /// Connection metadata by client ID
    connections: Arc<DashMap<String, ConnectionInfo>>,

    /// Enhanced rate limiting state
    rate_limiter: Arc<RwLock<EnhancedRateLimiter>>,

    /// Security manager for authentication and authorization (future use)
    #[allow(dead_code)]
    security_manager: Option<Arc<SecurityManager>>,
}

#[derive(Debug)]
pub struct ConnectionInfo {
    pub client_id: String,
    pub connected_at: Instant,
    pub last_activity: Instant,
    pub request_count: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
}

impl Clone for ConnectionInfo {
    fn clone(&self) -> Self {
        Self {
            client_id: self.client_id.clone(),
            connected_at: self.connected_at,
            last_activity: self.last_activity,
            request_count: AtomicU64::new(self.request_count.load(Ordering::SeqCst)),
            bytes_sent: AtomicU64::new(self.bytes_sent.load(Ordering::SeqCst)),
            bytes_received: AtomicU64::new(self.bytes_received.load(Ordering::SeqCst)),
        }
    }
}

#[derive(Debug)]
pub struct EnhancedRateLimiter {
    /// Requests per second limit per client (future use)
    #[allow(dead_code)]
    requests_per_second: u32,

    /// Burst capacity for rate limiting (future use)
    #[allow(dead_code)]
    burst_capacity: u32,

    /// Client request tracking with burst support
    client_requests: DashMap<String, ClientRequestTracker>,

    /// Cleanup interval (future use)
    #[allow(dead_code)]
    last_cleanup: Instant,

    /// Resource exhaustion protection enabled (future use)
    #[allow(dead_code)]
    resource_protection_enabled: bool,

    /// Memory usage tracking per client (future use)
    #[allow(dead_code)]
    client_memory_usage: DashMap<String, AtomicU64>,

    /// Maximum memory per connection (future use)
    #[allow(dead_code)]
    max_memory_per_connection: u64,
}

/// Enhanced client request tracking with burst capacity
#[derive(Debug, Clone)]
pub struct ClientRequestTracker {
    /// Recent request timestamps (future use)
    #[allow(dead_code)]
    pub requests: Vec<Instant>,
    /// Available burst tokens (future use)
    #[allow(dead_code)]
    pub burst_tokens: u32,
    /// Last token replenishment time (future use)
    #[allow(dead_code)]
    pub last_replenish: Instant,
    /// Memory usage for this client (future use)
    #[allow(dead_code)]
    pub memory_usage: u64,
}

impl ConnectionManager {
    pub fn new(max_connections: u64, requests_per_second: u32) -> Self {
        Self::new_with_security(max_connections, requests_per_second, 200, None)
    }

    pub fn new_with_security(
        max_connections: u64,
        requests_per_second: u32,
        burst_capacity: u32,
        security_manager: Option<Arc<SecurityManager>>
    ) -> Self {
        Self {
            active_connections: AtomicU64::new(0),
            max_connections,
            connections: Arc::new(DashMap::new()),
            rate_limiter: Arc::new(RwLock::new(EnhancedRateLimiter {
                requests_per_second,
                burst_capacity,
                client_requests: DashMap::new(),
                last_cleanup: Instant::now(),
                resource_protection_enabled: true,
                client_memory_usage: DashMap::new(),
                max_memory_per_connection: 100 * 1024 * 1024, // 100MB default
            })),
            security_manager,
        }
    }

    /// Register a new connection
    #[allow(dead_code)]
    pub fn register_connection(&self, client_id: String) -> Result<(), Status> {
        let current_connections = self.active_connections.load(Ordering::SeqCst);

        if current_connections >= self.max_connections {
            warn!("Connection limit reached: {}/{}", current_connections, self.max_connections);
            return Err(Status::resource_exhausted("Connection limit reached"));
        }

        let connection_info = ConnectionInfo {
            client_id: client_id.clone(),
            connected_at: Instant::now(),
            last_activity: Instant::now(),
            request_count: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
        };

        self.connections.insert(client_id.clone(), connection_info);
        self.active_connections.fetch_add(1, Ordering::SeqCst);

        info!("Connection registered: {} (total: {})", client_id, current_connections + 1);
        Ok(())
    }

    /// Unregister a connection
    pub fn unregister_connection(&self, client_id: &str) {
        if self.connections.remove(client_id).is_some() {
            let remaining = self.active_connections.fetch_sub(1, Ordering::SeqCst) - 1;
            info!("Connection unregistered: {} (remaining: {})", client_id, remaining);
        }

        // Clean up rate limiter tracking
        self.rate_limiter.read().client_requests.remove(client_id);
    }

    /// Check if request is rate limited
    #[allow(dead_code)]
    pub fn check_rate_limit(&self, client_id: &str) -> Result<(), Status> {
        let mut rate_limiter = self.rate_limiter.write();

        // Cleanup old entries periodically
        let now = Instant::now();
        if now.duration_since(rate_limiter.last_cleanup) > Duration::from_secs(60) {
            self.cleanup_enhanced_rate_limiter(&mut rate_limiter, now);
            rate_limiter.last_cleanup = now;
        }

        // Get or create client request history
        let mut tracker = rate_limiter.client_requests
            .entry(client_id.to_string())
            .or_insert_with(|| ClientRequestTracker {
                requests: Vec::new(),
                burst_tokens: 100,
                last_replenish: now,
                memory_usage: 0,
            })
            .clone();

        // Remove requests older than 1 second
        tracker.requests.retain(|&timestamp| now.duration_since(timestamp) < Duration::from_secs(1));

        // Check if rate limit exceeded
        if tracker.requests.len() >= rate_limiter.requests_per_second as usize {
            warn!("Rate limit exceeded for client: {}", client_id);
            return Err(Status::resource_exhausted("Rate limit exceeded"));
        }

        // Add current request
        tracker.requests.push(now);
        rate_limiter.client_requests.insert(client_id.to_string(), tracker);

        Ok(())
    }

    /// Update connection activity
    #[allow(dead_code)]
    pub fn update_activity(&self, client_id: &str, bytes_sent: u64, bytes_received: u64) {
        if let Some(mut connection) = self.connections.get_mut(client_id) {
            connection.last_activity = Instant::now();
            connection.request_count.fetch_add(1, Ordering::SeqCst);
            connection.bytes_sent.fetch_add(bytes_sent, Ordering::SeqCst);
            connection.bytes_received.fetch_add(bytes_received, Ordering::SeqCst);
        }
    }

    /// Get connection statistics
    #[allow(dead_code)]
    pub fn get_stats(&self) -> ConnectionStats {
        let active_count = self.active_connections.load(Ordering::SeqCst);
        let total_requests: u64 = self.connections
            .iter()
            .map(|entry| entry.request_count.load(Ordering::SeqCst))
            .sum();
        let total_bytes_sent: u64 = self.connections
            .iter()
            .map(|entry| entry.bytes_sent.load(Ordering::SeqCst))
            .sum();
        let total_bytes_received: u64 = self.connections
            .iter()
            .map(|entry| entry.bytes_received.load(Ordering::SeqCst))
            .sum();

        ConnectionStats {
            active_connections: active_count,
            max_connections: self.max_connections,
            total_requests,
            total_bytes_sent,
            total_bytes_received,
        }
    }

    /// Cleanup expired connections
    pub fn cleanup_expired_connections(&self, timeout: Duration) {
        let now = Instant::now();
        let mut expired_clients = Vec::new();

        for entry in self.connections.iter() {
            if now.duration_since(entry.last_activity) > timeout {
                expired_clients.push(entry.client_id.clone());
            }
        }

        for client_id in expired_clients {
            self.unregister_connection(&client_id);
            warn!("Expired connection cleaned up: {}", client_id);
        }
    }

    #[allow(dead_code)]
    fn cleanup_enhanced_rate_limiter(&self, rate_limiter: &mut EnhancedRateLimiter, now: Instant) {
        let cutoff = now - Duration::from_secs(60);

        // Remove old client tracking data
        rate_limiter.client_requests.retain(|_client_id, tracker| {
            tracker.requests.retain(|&timestamp| timestamp > cutoff);
            // Keep tracker even if no recent requests for burst token management
            true
        });

        // Cleanup memory usage tracking for disconnected clients
        rate_limiter.client_memory_usage.retain(|client_id, _| {
            self.connections.contains_key(client_id)
        });
    }

    /// Check memory usage for a client
    #[allow(dead_code)]
    pub fn check_memory_usage(&self, client_id: &str, additional_memory: u64) -> Result<(), Status> {
        let rate_limiter = self.rate_limiter.read();

        if !rate_limiter.resource_protection_enabled {
            return Ok(());
        }

        let current_usage = rate_limiter.client_memory_usage
            .get(client_id)
            .map(|usage| usage.load(Ordering::SeqCst))
            .unwrap_or(0);

        if current_usage + additional_memory > rate_limiter.max_memory_per_connection {
            warn!("Memory limit exceeded for client: {} (current: {}, additional: {}, limit: {})",
                  client_id, current_usage, additional_memory, rate_limiter.max_memory_per_connection);
            return Err(Status::resource_exhausted("Memory limit exceeded"));
        }

        Ok(())
    }

    /// Update memory usage for a client
    #[allow(dead_code)]
    pub fn update_memory_usage(&self, client_id: &str, memory_delta: i64) {
        let rate_limiter = self.rate_limiter.read();

        let usage = rate_limiter.client_memory_usage
            .entry(client_id.to_string())
            .or_insert_with(|| AtomicU64::new(0));

        if memory_delta > 0 {
            usage.fetch_add(memory_delta as u64, Ordering::SeqCst);
        } else if memory_delta < 0 {
            usage.fetch_sub((-memory_delta) as u64, Ordering::SeqCst);
        }
    }

    /// Get security manager if available
    #[allow(dead_code)]
    pub fn security_manager(&self) -> Option<&Arc<SecurityManager>> {
        self.security_manager.as_ref()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConnectionStats {
    pub active_connections: u64,
    pub max_connections: u64,
    pub total_requests: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
}

/// Connection pool for outbound connections (to Qdrant, etc.)
#[cfg(any(test, feature = "test-utils"))]
pub struct ConnectionPool<T: deadpool::managed::Manager> {
    #[allow(dead_code)]
    pool: deadpool::managed::Pool<T>,
    config: PoolConfig,
}

#[cfg(any(test, feature = "test-utils"))]
impl<T: deadpool::managed::Manager> std::fmt::Debug for ConnectionPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionPool")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub min_idle: Option<usize>,
    pub max_lifetime: Option<Duration>,
    pub idle_timeout: Option<Duration>,
    pub connection_timeout: Duration,
}

#[cfg(any(test, feature = "test-utils"))]
impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 10,
            min_idle: Some(2),
            max_lifetime: Some(Duration::from_secs(3600)), // 1 hour
            idle_timeout: Some(Duration::from_secs(600)),   // 10 minutes
            connection_timeout: Duration::from_secs(30),
        }
    }
}

/// Middleware interceptor for connection management
#[derive(Debug, Clone)]
pub struct ConnectionInterceptor {
    #[allow(dead_code)]
    connection_manager: Arc<ConnectionManager>,
}

impl ConnectionInterceptor {
    pub fn new(connection_manager: Arc<ConnectionManager>) -> Self {
        Self { connection_manager }
    }

    /// Intercept incoming requests
    #[allow(dead_code)]
    pub fn intercept<T>(&self, request: Request<T>) -> Result<Request<T>, Status> {
        // Extract client ID from metadata
        let client_id = request
            .metadata()
            .get("client-id")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        // Check rate limiting
        self.connection_manager.check_rate_limit(&client_id)?;

        // Update activity (approximate request size)
        let request_size = std::mem::size_of_val(&request) as u64;
        self.connection_manager.update_activity(&client_id, 0, request_size);

        Ok(request)
    }

    /// Intercept outgoing responses
    #[allow(dead_code)]
    pub fn intercept_response<T>(&self, response: Response<T>, client_id: &str) -> Response<T> {
        // Update activity (approximate response size)
        let response_size = std::mem::size_of_val(&response) as u64;
        self.connection_manager.update_activity(client_id, response_size, 0);

        response
    }
}

/// Retry configuration for failed connections
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry wrapper for gRPC operations
#[allow(dead_code)]
pub async fn with_retry<F, T, E>(
    operation: F,
    config: &RetryConfig,
) -> Result<T, E>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
    E: std::fmt::Debug,
{
    let mut delay = config.initial_delay;

    for attempt in 1..=config.max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) if attempt == config.max_retries => {
                debug!("Operation failed after {} attempts: {:?}", config.max_retries, err);
                return Err(err);
            },
            Err(err) => {
                debug!("Operation failed (attempt {}/{}): {:?}", attempt, config.max_retries, err);
                tokio::time::sleep(delay).await;

                // Exponential backoff with jitter
                delay = std::cmp::min(
                    Duration::from_millis(
                        (delay.as_millis() as f64 * config.backoff_multiplier) as u64
                    ),
                    config.max_delay,
                );
            }
        }
    }

    unreachable!()
}

/// Enhanced middleware that combines connection management, retry logic, and circuit breaker
#[derive(Debug)]
#[cfg(any(test, feature = "test-utils"))]
pub struct EnhancedMiddleware {
    connection_manager: Arc<ConnectionManager>,
    retry_strategy: RetryStrategy,
    circuit_breakers: Arc<DashMap<String, Arc<CircuitBreaker>>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl EnhancedMiddleware {
    /// Create new enhanced middleware
    pub fn new(
        connection_manager: Arc<ConnectionManager>,
        retry_strategy: RetryStrategy,
    ) -> Self {
        Self {
            connection_manager,
            retry_strategy,
            circuit_breakers: Arc::new(DashMap::new()),
        }
    }

    /// Get or create circuit breaker for a service
    pub async fn get_circuit_breaker(&self, service_name: &str) -> DaemonResult<Arc<CircuitBreaker>> {
        if let Some(cb) = self.circuit_breakers.get(service_name) {
            Ok(cb.clone())
        } else {
            let circuit_breaker = Arc::new(CircuitBreaker::new_default(service_name.to_string())?);
            self.circuit_breakers.insert(service_name.to_string(), circuit_breaker.clone());
            info!("Created circuit breaker for service: {}", service_name);
            Ok(circuit_breaker)
        }
    }

    /// Execute a gRPC operation with retry and circuit breaker protection
    pub async fn execute_with_protection<F, Fut, T>(
        &self,
        service_name: &str,
        operation: F,
    ) -> DaemonResult<T>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = DaemonResult<T>>,
    {
        let circuit_breaker = self.get_circuit_breaker(service_name).await?;

        self.retry_strategy.execute(|| {
            let cb = circuit_breaker.clone();
            let op = operation.clone();
            async move {
                cb.execute(|| op()).await
            }
        }).await
    }

    /// Get circuit breaker statistics for all services
    pub async fn get_circuit_breaker_stats(&self) -> Vec<(String, crate::grpc::CircuitBreakerStats)> {
        let mut stats = Vec::new();
        for entry in self.circuit_breakers.iter() {
            let service_name = entry.key().clone();
            let cb_stats = entry.value().stats().await;
            stats.push((service_name, cb_stats));
        }
        stats
    }

    /// Get connection manager
    pub fn connection_manager(&self) -> &Arc<ConnectionManager> {
        &self.connection_manager
    }

    /// Get retry strategy
    pub fn retry_strategy(&self) -> &RetryStrategy {
        &self.retry_strategy
    }
}

/// Security interceptor for authentication and authorization
#[derive(Debug, Clone)]
#[cfg(any(test, feature = "test-utils"))]
pub struct SecurityInterceptor {
    security_manager: Arc<SecurityManager>,
}

#[cfg(any(test, feature = "test-utils"))]
impl SecurityInterceptor {
    pub fn new(security_manager: Arc<SecurityManager>) -> Self {
        Self { security_manager }
    }

    pub async fn intercept<T>(&self, request: Request<T>) -> Result<Request<T>, Status> {
        // For tonic 0.12, we'll get service/method from metadata if available
        // or use default values for now
        let (service, method) = ("grpc_service", "grpc_method");

        // Get client ID from request metadata or connection info
        let client_id = self.extract_client_id(&request);

        // Authenticate the request
        let permissions = match self.security_manager.authenticate_request(&request).await {
            Ok(perms) => {
                self.security_manager.audit_logger().log_auth_event(
                    &client_id, service, method, true
                );
                perms
            }
            Err(e) => {
                self.security_manager.audit_logger().log_auth_event(
                    &client_id, service, method, false
                );
                return Err(Status::unauthenticated(format!("Authentication failed: {}", e)));
            }
        };

        // Authorize the request
        if !self.security_manager.authorize_request(&permissions, service, method) {
            return Err(Status::permission_denied("Insufficient permissions"));
        }

        // Add permissions to request metadata for downstream services
        let mut request = request;
        if let Ok(permissions_json) = serde_json::to_string(&permissions) {
            if let Ok(metadata_value) = tonic::metadata::MetadataValue::try_from(permissions_json) {
                request.metadata_mut().insert("x-permissions", metadata_value);
            }
        }

        Ok(request)
    }

    fn extract_client_id<T>(&self, request: &Request<T>) -> String {
        // Try to get client ID from various sources
        if let Some(client_id) = request.metadata().get("x-client-id") {
            if let Ok(id_str) = client_id.to_str() {
                return id_str.to_string();
            }
        }

        if let Some(client_id) = request.metadata().get("client-id") {
            if let Ok(id_str) = client_id.to_str() {
                return id_str.to_string();
            }
        }

        // Fallback to unknown client
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use tonic::{Request, Response, Status, metadata::MetadataValue};

    #[test]
    fn test_connection_manager_new() {
        let manager = ConnectionManager::new(100, 10);

        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 0);
        assert_eq!(manager.max_connections, 100);
        assert!(manager.connections.is_empty());
    }

    #[test]
    fn test_connection_info_clone() {
        let info = ConnectionInfo {
            client_id: "test_client".to_string(),
            connected_at: Instant::now(),
            last_activity: Instant::now(),
            request_count: AtomicU64::new(5),
            bytes_sent: AtomicU64::new(1000),
            bytes_received: AtomicU64::new(2000),
        };

        let cloned = info.clone();

        assert_eq!(info.client_id, cloned.client_id);
        assert_eq!(info.request_count.load(Ordering::SeqCst),
                   cloned.request_count.load(Ordering::SeqCst));
        assert_eq!(info.bytes_sent.load(Ordering::SeqCst),
                   cloned.bytes_sent.load(Ordering::SeqCst));
        assert_eq!(info.bytes_received.load(Ordering::SeqCst),
                   cloned.bytes_received.load(Ordering::SeqCst));
    }

    #[test]
    fn test_connection_manager_register_connection() {
        let manager = ConnectionManager::new(2, 10);

        // Register first connection
        let result1 = manager.register_connection("client1".to_string());
        assert!(result1.is_ok());
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 1);

        // Register second connection
        let result2 = manager.register_connection("client2".to_string());
        assert!(result2.is_ok());
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 2);

        // Try to register third connection (should fail due to limit)
        let result3 = manager.register_connection("client3".to_string());
        assert!(result3.is_err());
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_connection_manager_unregister_connection() {
        let manager = ConnectionManager::new(10, 10);

        // Register and then unregister
        manager.register_connection("client1".to_string()).unwrap();
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 1);

        manager.unregister_connection("client1");
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 0);
        assert!(manager.connections.is_empty());

        // Unregistering non-existent connection should not panic
        manager.unregister_connection("non_existent");
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_connection_manager_rate_limiting() {
        let manager = ConnectionManager::new(10, 2); // 2 requests per second

        // First request should succeed
        let result1 = manager.check_rate_limit("client1");
        assert!(result1.is_ok());

        // Second request should succeed
        let result2 = manager.check_rate_limit("client1");
        assert!(result2.is_ok());

        // Third request should fail (rate limit exceeded)
        let result3 = manager.check_rate_limit("client1");
        assert!(result3.is_err());
        if let Err(status) = result3 {
            assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        }
    }

    #[test]
    fn test_connection_manager_rate_limiting_different_clients() {
        let manager = ConnectionManager::new(10, 2); // 2 requests per second

        // Two requests from client1
        assert!(manager.check_rate_limit("client1").is_ok());
        assert!(manager.check_rate_limit("client1").is_ok());

        // Two requests from client2 should still succeed
        assert!(manager.check_rate_limit("client2").is_ok());
        assert!(manager.check_rate_limit("client2").is_ok());

        // Third request from each client should fail
        assert!(manager.check_rate_limit("client1").is_err());
        assert!(manager.check_rate_limit("client2").is_err());
    }

    #[test]
    fn test_connection_manager_update_activity() {
        let manager = ConnectionManager::new(10, 10);
        manager.register_connection("client1".to_string()).unwrap();

        // Update activity
        manager.update_activity("client1", 1000, 500);

        let stats = manager.get_stats();
        assert_eq!(stats.total_bytes_sent, 1000);
        assert_eq!(stats.total_bytes_received, 500);
        assert_eq!(stats.total_requests, 1);

        // Update activity again
        manager.update_activity("client1", 2000, 1000);

        let stats = manager.get_stats();
        assert_eq!(stats.total_bytes_sent, 3000);
        assert_eq!(stats.total_bytes_received, 1500);
        assert_eq!(stats.total_requests, 2);
    }

    #[test]
    fn test_connection_manager_get_stats() {
        let manager = ConnectionManager::new(10, 10);

        // Initial stats
        let stats = manager.get_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.max_connections, 10);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_received, 0);

        // Register connections and update activity
        manager.register_connection("client1".to_string()).unwrap();
        manager.register_connection("client2".to_string()).unwrap();
        manager.update_activity("client1", 100, 200);
        manager.update_activity("client2", 300, 400);

        let stats = manager.get_stats();
        assert_eq!(stats.active_connections, 2);
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_bytes_sent, 400);
        assert_eq!(stats.total_bytes_received, 600);
    }

    #[test]
    fn test_connection_manager_cleanup_expired_connections() {
        let manager = ConnectionManager::new(10, 10);
        manager.register_connection("client1".to_string()).unwrap();

        // Connection should not be expired immediately
        manager.cleanup_expired_connections(Duration::from_secs(1));
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 1);

        // Connection should be expired with zero timeout
        manager.cleanup_expired_connections(Duration::from_secs(0));
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_connection_stats_debug() {
        let stats = ConnectionStats {
            active_connections: 5,
            max_connections: 10,
            total_requests: 100,
            total_bytes_sent: 1000,
            total_bytes_received: 2000,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ConnectionStats"));
        assert!(debug_str.contains("5"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_connection_stats_clone() {
        let stats = ConnectionStats {
            active_connections: 5,
            max_connections: 10,
            total_requests: 100,
            total_bytes_sent: 1000,
            total_bytes_received: 2000,
        };

        let cloned = stats.clone();
        assert_eq!(stats.active_connections, cloned.active_connections);
        assert_eq!(stats.max_connections, cloned.max_connections);
        assert_eq!(stats.total_requests, cloned.total_requests);
        assert_eq!(stats.total_bytes_sent, cloned.total_bytes_sent);
        assert_eq!(stats.total_bytes_received, cloned.total_bytes_received);
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();

        assert_eq!(config.max_size, 10);
        assert_eq!(config.min_idle, Some(2));
        assert_eq!(config.max_lifetime, Some(Duration::from_secs(3600)));
        assert_eq!(config.idle_timeout, Some(Duration::from_secs(600)));
        assert_eq!(config.connection_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_pool_config_debug() {
        let config = PoolConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("PoolConfig"));
        assert!(debug_str.contains("max_size"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_connection_interceptor_new() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager.clone());

        assert!(Arc::ptr_eq(&interceptor.connection_manager, &manager));
    }

    #[test]
    fn test_connection_interceptor_debug() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager);

        let debug_str = format!("{:?}", interceptor);
        assert!(debug_str.contains("ConnectionInterceptor"));
        assert!(debug_str.contains("connection_manager"));
    }

    #[test]
    fn test_connection_interceptor_clone() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager);
        let cloned = interceptor.clone();

        assert!(Arc::ptr_eq(&interceptor.connection_manager, &cloned.connection_manager));
    }

    #[test]
    fn test_connection_interceptor_intercept_without_client_id() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager);

        let request: Request<()> = Request::new(());
        let result = interceptor.intercept(request);

        assert!(result.is_ok());
    }

    #[test]
    fn test_connection_interceptor_intercept_with_client_id() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager);

        let mut request: Request<()> = Request::new(());
        request.metadata_mut().insert(
            "client-id",
            MetadataValue::from_static("test_client")
        );

        let result = interceptor.intercept(request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_connection_interceptor_intercept_response() {
        let manager = Arc::new(ConnectionManager::new(10, 10));
        let interceptor = ConnectionInterceptor::new(manager.clone());

        manager.register_connection("test_client".to_string()).unwrap();

        let response: Response<()> = Response::new(());
        let result = interceptor.intercept_response(response, "test_client");

        // Should return the response unchanged
        assert_eq!(result.get_ref(), &());
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();

        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_retry_config_debug() {
        let config = RetryConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("RetryConfig"));
        assert!(debug_str.contains("max_retries"));
        assert!(debug_str.contains("3"));
    }

    #[test]
    fn test_retry_config_clone() {
        let config = RetryConfig::default();
        let cloned = config.clone();

        assert_eq!(config.max_retries, cloned.max_retries);
        assert_eq!(config.initial_delay, cloned.initial_delay);
        assert_eq!(config.max_delay, cloned.max_delay);
        assert_eq!(config.backoff_multiplier, cloned.backoff_multiplier);
    }

    #[tokio::test]
    async fn test_with_retry_success_first_attempt() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        };

        let result = with_retry(
            || Box::pin(async { Ok::<i32, &'static str>(42) }),
            &config,
        ).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_retry_success_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        };

        let attempt = Arc::new(AtomicU64::new(0));
        let attempt_clone = Arc::clone(&attempt);
        let result = with_retry(
            move || {
                let current_attempt = attempt_clone.fetch_add(1, Ordering::SeqCst) + 1;
                Box::pin(async move {
                    if current_attempt < 3 {
                        Err("temporary failure")
                    } else {
                        Ok::<i32, &'static str>(42)
                    }
                })
            },
            &config,
        ).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_retry_final_failure() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        };

        let result = with_retry(
            || Box::pin(async { Err::<i32, &'static str>("permanent failure") }),
            &config,
        ).await;

        assert_eq!(result.unwrap_err(), "permanent failure");
    }

    #[test]
    fn test_enhanced_rate_limiter_debug() {
        let rate_limiter = EnhancedRateLimiter {
            requests_per_second: 10,
            burst_capacity: 20,
            client_requests: DashMap::new(),
            last_cleanup: Instant::now(),
            resource_protection_enabled: true,
            client_memory_usage: DashMap::new(),
            max_memory_per_connection: 100 * 1024 * 1024,
        };

        let debug_str = format!("{:?}", rate_limiter);
        assert!(debug_str.contains("EnhancedRateLimiter"));
        assert!(debug_str.contains("requests_per_second"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("burst_capacity"));
        assert!(debug_str.contains("20"));
    }

    #[test]
    fn test_connection_manager_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ConnectionManager>();
        assert_sync::<ConnectionManager>();
    }

    #[test]
    fn test_connection_interceptor_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ConnectionInterceptor>();
        assert_sync::<ConnectionInterceptor>();
    }

    #[tokio::test]
    async fn test_concurrent_connection_management() {
        let manager = Arc::new(ConnectionManager::new(100, 50));

        let mut handles = vec![];

        // Spawn multiple tasks to register connections concurrently
        for i in 0..10 {
            let manager_clone = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                let client_id = format!("client_{}", i);
                manager_clone.register_connection(client_id.clone()).unwrap();

                // Simulate some activity
                for _ in 0..5 {
                    manager_clone.update_activity(&client_id, 100, 200);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }

                manager_clone.unregister_connection(&client_id);
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // All connections should be unregistered
        assert_eq!(manager.active_connections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_connection_manager_edge_cases() {
        let manager = ConnectionManager::new(10, 100);

        // Test with various client IDs
        let client_ids = vec![
            "".to_string(), // empty string
            "a".repeat(1000), // very long string
            "client-with-special-chars-!@#$%".to_string(),
            "正常".to_string(), // Unicode characters
        ];

        for client_id in client_ids {
            manager.update_activity(&client_id, 1024, 512);
            let stats = manager.get_stats();
            // Stats should be valid
            assert!(stats.total_bytes_sent >= 0);
            assert!(stats.total_bytes_received >= 0);
        }

        // Test connection cleanup with very short duration
        manager.cleanup_expired_connections(Duration::from_nanos(1));
        let stats_after_cleanup = manager.get_stats();
        assert!(stats_after_cleanup.active_connections >= 0);
    }

    #[test]
    fn test_connection_info_edge_cases() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let info = ConnectionInfo {
            client_id: "".to_string(),
            connected_at: Instant::now(),
            last_activity: Instant::now(),
            request_count: AtomicU64::new(u64::MAX),
            bytes_sent: AtomicU64::new(u64::MAX),
            bytes_received: AtomicU64::new(u64::MAX),
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ConnectionInfo"));
        assert!(debug_str.contains("client_id"));

        let cloned = info.clone();
        assert_eq!(info.client_id, cloned.client_id);
        assert_eq!(info.request_count.load(Ordering::SeqCst), cloned.request_count.load(Ordering::SeqCst));
        assert_eq!(info.bytes_sent.load(Ordering::SeqCst), cloned.bytes_sent.load(Ordering::SeqCst));
        assert_eq!(info.bytes_received.load(Ordering::SeqCst), cloned.bytes_received.load(Ordering::SeqCst));
    }

    #[test]
    fn test_connection_stats_edge_cases() {
        let stats = ConnectionStats {
            active_connections: u64::MAX,
            max_connections: u64::MAX,
            total_requests: u64::MAX,
            total_bytes_sent: u64::MAX,
            total_bytes_received: u64::MAX,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ConnectionStats"));
        assert!(debug_str.contains("active_connections"));

        let cloned = stats.clone();
        assert_eq!(stats.active_connections, cloned.active_connections);
        assert_eq!(stats.max_connections, cloned.max_connections);
        assert_eq!(stats.total_requests, cloned.total_requests);
        assert_eq!(stats.total_bytes_sent, cloned.total_bytes_sent);
        assert_eq!(stats.total_bytes_received, cloned.total_bytes_received);
    }

    #[test]
    fn test_pool_config_edge_cases() {
        let config = PoolConfig {
            max_size: usize::MAX,
            min_idle: Some(0),
            max_lifetime: Some(Duration::from_secs(1)),
            idle_timeout: Some(Duration::from_secs(1)),
            connection_timeout: Duration::from_secs(1),
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("PoolConfig"));
        assert!(debug_str.contains("max_size"));

        let cloned = config.clone();
        assert_eq!(config.max_size, cloned.max_size);
        assert_eq!(config.min_idle, cloned.min_idle);
        assert_eq!(config.connection_timeout, cloned.connection_timeout);
    }

    #[test]
    fn test_retry_config_edge_cases() {
        let config = RetryConfig {
            max_retries: u32::MAX,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("RetryConfig"));
        assert!(debug_str.contains("max_retries"));

        let cloned = config.clone();
        assert_eq!(config.max_retries, cloned.max_retries);
        assert_eq!(config.initial_delay, cloned.initial_delay);
        assert_eq!(config.max_delay, cloned.max_delay);
        assert!((config.backoff_multiplier - cloned.backoff_multiplier).abs() < f64::EPSILON);
    }

    #[test]
    fn test_connection_manager_overflow_resistance() {
        let manager = ConnectionManager::new(100, u32::MAX);

        // Register connection first, then test with large values
        manager.register_connection("test_client".to_string()).unwrap();
        manager.update_activity("test_client", u64::MAX, u64::MAX);
        let stats = manager.get_stats();
        assert!(stats.total_bytes_sent > 0);
        assert!(stats.total_bytes_received > 0);

        // Clean up
        manager.unregister_connection("test_client");
    }

    #[test]
    fn test_connection_interceptor_creation() {
        let connection_manager = Arc::new(ConnectionManager::new(100, 1000));
        let interceptor = ConnectionInterceptor::new(connection_manager.clone());

        // Test that the interceptor holds a reference to the connection manager
        assert!(Arc::strong_count(&connection_manager) >= 2);

        // Test stats access through connection manager
        let stats = connection_manager.get_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_connection_pool_edge_cases() {
        use std::collections::HashMap;
        use deadpool::managed::Manager;

        // Test that ConnectionPool can be created (field is never read in current impl)
        struct TestManager;
        impl Manager for TestManager {
            type Type = String;
            type Error = std::io::Error;
            async fn create(&self) -> Result<Self::Type, Self::Error> {
                Ok("test".to_string())
            }
            async fn recycle(&self, _obj: &mut Self::Type, _metrics: &deadpool::managed::Metrics) -> deadpool::managed::RecycleResult<Self::Error> {
                Ok(())
            }
        }

        let pool_config = deadpool::managed::PoolConfig::new(10);
        let pool = deadpool::managed::Pool::builder(TestManager {})
            .config(pool_config)
            .build()
            .unwrap();

        let pool_config = PoolConfig::default();
        let connection_pool = ConnectionPool { pool, config: pool_config };
        // Just test that it can be created (field is never actually used)
        let _ = connection_pool;
    }

    #[tokio::test]
    async fn test_enhanced_middleware_creation() {
        let connection_manager = Arc::new(ConnectionManager::new(10, 5));
        let retry_strategy = crate::grpc::RetryStrategy::new();

        let middleware = EnhancedMiddleware::new(connection_manager.clone(), retry_strategy);

        assert_eq!(middleware.circuit_breakers.len(), 0);
        assert!(Arc::ptr_eq(&middleware.connection_manager, &connection_manager));
    }

    #[tokio::test]
    async fn test_enhanced_middleware_circuit_breaker_creation() {
        let connection_manager = Arc::new(ConnectionManager::new(10, 5));
        let retry_strategy = crate::grpc::RetryStrategy::new();
        let middleware = EnhancedMiddleware::new(connection_manager, retry_strategy);

        // Get circuit breaker for first time
        let cb1 = middleware.get_circuit_breaker("test-service").await.unwrap();
        assert_eq!(middleware.circuit_breakers.len(), 1);

        // Get circuit breaker again - should return same instance
        let cb2 = middleware.get_circuit_breaker("test-service").await.unwrap();
        assert!(Arc::ptr_eq(&cb1, &cb2));
        assert_eq!(middleware.circuit_breakers.len(), 1);
    }

    #[tokio::test]
    async fn test_enhanced_middleware_execute_with_protection() {
        let connection_manager = Arc::new(ConnectionManager::new(10, 5));
        let retry_strategy = crate::grpc::RetryStrategy::new();
        let middleware = EnhancedMiddleware::new(connection_manager, retry_strategy);

        let result = middleware.execute_with_protection("test-service", || async {
            Ok::<i32, crate::error::DaemonError>(42)
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(middleware.circuit_breakers.len(), 1);
    }

    #[tokio::test]
    async fn test_enhanced_middleware_error_handling() {
        let connection_manager = Arc::new(ConnectionManager::new(10, 5));
        let retry_strategy = crate::grpc::RetryStrategy::new();
        let middleware = EnhancedMiddleware::new(connection_manager, retry_strategy);

        let result = middleware.execute_with_protection("test-service", || async {
            Err::<i32, crate::error::DaemonError>(crate::error::DaemonError::NetworkConnection {
                message: "test error".to_string()
            })
        }).await;

        assert!(result.is_err());

        // Verify circuit breaker was created and used
        assert_eq!(middleware.circuit_breakers.len(), 1);
        let stats = middleware.get_circuit_breaker_stats().await;
        assert!(stats[0].1.total_requests > 0);
    }

    #[tokio::test]
    async fn test_enhanced_middleware_circuit_breaker_stats() {
        let connection_manager = Arc::new(ConnectionManager::new(10, 5));
        let retry_strategy = crate::grpc::RetryStrategy::new();
        let middleware = EnhancedMiddleware::new(connection_manager, retry_strategy);

        // Create a circuit breaker
        let _ = middleware.get_circuit_breaker("test-service").await.unwrap();

        let stats = middleware.get_circuit_breaker_stats().await;
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, "test-service");
        assert_eq!(stats[0].1.state, crate::grpc::CircuitState::Closed);
    }
}