//! gRPC middleware for connection management, metrics, and security

use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use parking_lot::RwLock;

/// Connection tracking and management
#[derive(Debug)]
pub struct ConnectionManager {
    /// Active connections count
    active_connections: AtomicU64,

    /// Maximum allowed connections
    max_connections: u64,

    /// Connection metadata by client ID
    connections: Arc<DashMap<String, ConnectionInfo>>,

    /// Rate limiting state
    rate_limiter: Arc<RwLock<RateLimiter>>,
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
pub struct RateLimiter {
    /// Requests per second limit per client
    requests_per_second: u32,

    /// Client request tracking
    client_requests: DashMap<String, Vec<Instant>>,

    /// Cleanup interval
    last_cleanup: Instant,
}

impl ConnectionManager {
    pub fn new(max_connections: u64, requests_per_second: u32) -> Self {
        Self {
            active_connections: AtomicU64::new(0),
            max_connections,
            connections: Arc::new(DashMap::new()),
            rate_limiter: Arc::new(RwLock::new(RateLimiter {
                requests_per_second,
                client_requests: DashMap::new(),
                last_cleanup: Instant::now(),
            })),
        }
    }

    /// Register a new connection
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
    pub fn check_rate_limit(&self, client_id: &str) -> Result<(), Status> {
        let mut rate_limiter = self.rate_limiter.write();

        // Cleanup old entries periodically
        let now = Instant::now();
        if now.duration_since(rate_limiter.last_cleanup) > Duration::from_secs(60) {
            self.cleanup_rate_limiter(&mut rate_limiter, now);
            rate_limiter.last_cleanup = now;
        }

        // Get or create client request history
        let mut requests = rate_limiter.client_requests
            .entry(client_id.to_string())
            .or_insert_with(Vec::new)
            .clone();

        // Remove requests older than 1 second
        requests.retain(|&timestamp| now.duration_since(timestamp) < Duration::from_secs(1));

        // Check if rate limit exceeded
        if requests.len() >= rate_limiter.requests_per_second as usize {
            warn!("Rate limit exceeded for client: {}", client_id);
            return Err(Status::resource_exhausted("Rate limit exceeded"));
        }

        // Add current request
        requests.push(now);
        rate_limiter.client_requests.insert(client_id.to_string(), requests);

        Ok(())
    }

    /// Update connection activity
    pub fn update_activity(&self, client_id: &str, bytes_sent: u64, bytes_received: u64) {
        if let Some(mut connection) = self.connections.get_mut(client_id) {
            connection.last_activity = Instant::now();
            connection.request_count.fetch_add(1, Ordering::SeqCst);
            connection.bytes_sent.fetch_add(bytes_sent, Ordering::SeqCst);
            connection.bytes_received.fetch_add(bytes_received, Ordering::SeqCst);
        }
    }

    /// Get connection statistics
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

    fn cleanup_rate_limiter(&self, rate_limiter: &mut RateLimiter, now: Instant) {
        let cutoff = now - Duration::from_secs(60);

        // Remove old client tracking data
        rate_limiter.client_requests.retain(|_client_id, requests| {
            requests.retain(|&timestamp| timestamp > cutoff);
            !requests.is_empty()
        });
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub active_connections: u64,
    pub max_connections: u64,
    pub total_requests: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
}

/// Connection pool for outbound connections (to Qdrant, etc.)
pub struct ConnectionPool<T: deadpool::managed::Manager> {
    pool: deadpool::managed::Pool<T>,
    config: PoolConfig,
}

impl<T: deadpool::managed::Manager> std::fmt::Debug for ConnectionPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionPool")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub min_idle: Option<usize>,
    pub max_lifetime: Option<Duration>,
    pub idle_timeout: Option<Duration>,
    pub connection_timeout: Duration,
}

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
    connection_manager: Arc<ConnectionManager>,
}

impl ConnectionInterceptor {
    pub fn new(connection_manager: Arc<ConnectionManager>) -> Self {
        Self { connection_manager }
    }

    /// Intercept incoming requests
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