//! Health Checking and Monitoring
//!
//! This module provides health checking capabilities for discovered services,
//! including HTTP health checks, process validation, and service status monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

use super::registry::ServiceInfo;

/// Health check errors
#[derive(Error, Debug)]
pub enum HealthError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),
    
    #[error("Health check timeout")]
    Timeout,
    
    #[error("Invalid health response: {0}")]
    InvalidResponse(String),
    
    #[error("Service unreachable: {0}")]
    ServiceUnreachable(String),
    
    #[error("Process validation failed: PID {0} not running")]
    ProcessNotRunning(u32),
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Service is healthy and responding
    Healthy,
    /// Service is unhealthy but responsive
    Unhealthy,
    /// Service is not responding
    Unreachable,
    /// Service process is not running
    ProcessDead,
    /// Health check is in progress
    Checking,
    /// Health status is unknown
    Unknown,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Service name
    pub service_name: String,
    
    /// Health status
    pub status: HealthStatus,
    
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    
    /// Health check timestamp
    pub timestamp: SystemTime,
    
    /// Additional health metrics
    pub metrics: HashMap<String, String>,
    
    /// Error message if unhealthy
    pub error_message: Option<String>,
}

/// Health checker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// HTTP request timeout
    pub request_timeout: Duration,
    
    /// Health check interval
    pub check_interval: Duration,
    
    /// Maximum consecutive failures before marking unhealthy
    pub max_failures: u32,
    
    /// Enable process validation
    pub validate_process: bool,
    
    /// Custom HTTP headers for health checks
    pub custom_headers: HashMap<String, String>,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(10),
            check_interval: Duration::from_secs(30),
            max_failures: 3,
            validate_process: true,
            custom_headers: HashMap::new(),
        }
    }
}

/// Health checker for service monitoring
#[derive(Clone)]
pub struct HealthChecker {
    /// HTTP client for health checks
    client: reqwest::Client,
    
    /// Health check configuration
    config: HealthConfig,
    
    /// Service health status cache
    health_cache: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
    
    /// Failure counters for services
    failure_counters: Arc<RwLock<HashMap<String, u32>>>,
}

impl HealthChecker {
    /// Create a new health checker instance
    pub fn new(config: HealthConfig) -> Result<Self, HealthError> {
        let mut headers = reqwest::header::HeaderMap::new();
        
        // Add custom headers
        for (key, value) in &config.custom_headers {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| HealthError::InvalidResponse(format!("Invalid header name: {}", key)))?;
            let header_value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| HealthError::InvalidResponse(format!("Invalid header value: {}", value)))?;
            headers.insert(header_name, header_value);
        }

        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .default_headers(headers)
            .build()?;

        info!("Health checker initialized with timeout: {:?}", config.request_timeout);

        Ok(Self {
            client,
            config,
            health_cache: Arc::new(RwLock::new(HashMap::new())),
            failure_counters: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Perform health check on a service
    pub async fn check_service_health(
        &self,
        service_name: &str,
        service_info: &ServiceInfo,
    ) -> Result<HealthCheckResult, HealthError> {
        debug!("Checking health for service: {}", service_name);
        
        let start_time = Instant::now();
        let mut metrics = HashMap::new();
        let status;
        let mut error_message = None;

        // Step 1: Validate process if enabled
        if self.config.validate_process
            && !is_process_running(service_info.pid) {
                status = HealthStatus::ProcessDead;
                error_message = Some(format!("Process {} is not running", service_info.pid));
                
                return Ok(HealthCheckResult {
                    service_name: service_name.to_string(),
                    status,
                    response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                    timestamp: SystemTime::now(),
                    metrics,
                    error_message,
                });
            }

        // Step 2: Perform HTTP health check
        let health_url = format!("http://{}:{}{}", 
                               service_info.host, 
                               service_info.port, 
                               service_info.health_endpoint);

        match timeout(self.config.request_timeout, self.client.get(&health_url).send()).await {
            Ok(Ok(response)) => {
                let response_time = start_time.elapsed().as_millis() as u64;
                metrics.insert("response_time_ms".to_string(), response_time.to_string());
                metrics.insert("status_code".to_string(), response.status().as_u16().to_string());
                
                if response.status().is_success() {
                    // Try to parse response body for additional metrics
                    if let Ok(body) = response.text().await {
                        if let Ok(health_data) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&body) {
                            for (key, value) in health_data {
                                metrics.insert(key, value.to_string());
                            }
                        }
                    }
                    
                    status = HealthStatus::Healthy;
                    self.reset_failure_counter(service_name).await;
                } else {
                    status = HealthStatus::Unhealthy;
                    error_message = Some(format!("HTTP {} response", response.status()));
                    self.increment_failure_counter(service_name).await;
                }
            }
            
            Ok(Err(e)) => {
                status = HealthStatus::Unreachable;
                error_message = Some(format!("HTTP request failed: {}", e));
                self.increment_failure_counter(service_name).await;
            }
            
            Err(_) => {
                status = HealthStatus::Unreachable;
                error_message = Some("Health check timeout".to_string());
                self.increment_failure_counter(service_name).await;
            }
        }

        let result = HealthCheckResult {
            service_name: service_name.to_string(),
            status,
            response_time_ms: Some(start_time.elapsed().as_millis() as u64),
            timestamp: SystemTime::now(),
            metrics,
            error_message,
        };

        // Cache the result
        self.cache_health_result(&result).await;

        debug!("Health check completed for {}: {:?}", service_name, result.status);
        Ok(result)
    }

    /// Start continuous health monitoring for services
    pub async fn start_monitoring(
        &self,
        services: HashMap<String, ServiceInfo>,
    ) -> tokio::task::JoinHandle<()> {
        let health_checker = Self::new(self.config.clone()).unwrap();
        let check_interval = self.config.check_interval;
        
        tokio::spawn(async move {
            let mut interval = interval(check_interval);
            
            loop {
                interval.tick().await;
                
                for (service_name, service_info) in &services {
                    let checker = health_checker.clone();
                    let name = service_name.clone();
                    let info = service_info.clone();
                    
                    tokio::spawn(async move {
                        match checker.check_service_health(&name, &info).await {
                            Ok(result) => {
                                debug!("Health check result for {}: {:?}", name, result.status);
                            }
                            Err(e) => {
                                warn!("Health check failed for {}: {}", name, e);
                            }
                        }
                    });
                }
            }
        })
    }

    /// Get cached health status for a service
    pub async fn get_service_health(&self, service_name: &str) -> Option<HealthCheckResult> {
        let cache_guard = self.health_cache.read().await;
        cache_guard.get(service_name).cloned()
    }

    /// Get health status for all monitored services
    pub async fn get_all_health_status(&self) -> HashMap<String, HealthCheckResult> {
        let cache_guard = self.health_cache.read().await;
        cache_guard.clone()
    }

    /// Check if a service is currently healthy
    pub async fn is_service_healthy(&self, service_name: &str) -> bool {
        if let Some(result) = self.get_service_health(service_name).await {
            matches!(result.status, HealthStatus::Healthy)
        } else {
            false
        }
    }

    /// Get failure count for a service
    pub async fn get_failure_count(&self, service_name: &str) -> u32 {
        let counters_guard = self.failure_counters.read().await;
        counters_guard.get(service_name).copied().unwrap_or(0)
    }

    /// Check if service has exceeded maximum failures
    pub async fn has_exceeded_max_failures(&self, service_name: &str) -> bool {
        self.get_failure_count(service_name).await >= self.config.max_failures
    }

    /// Cache health check result
    async fn cache_health_result(&self, result: &HealthCheckResult) {
        let mut cache_guard = self.health_cache.write().await;
        cache_guard.insert(result.service_name.clone(), result.clone());
    }

    /// Increment failure counter for a service
    async fn increment_failure_counter(&self, service_name: &str) {
        let mut counters_guard = self.failure_counters.write().await;
        let count = counters_guard.entry(service_name.to_string()).or_insert(0);
        *count += 1;
        
        if *count >= self.config.max_failures {
            warn!("Service {} has exceeded maximum failures ({}/{})", 
                  service_name, count, self.config.max_failures);
        }
    }

    /// Reset failure counter for a service
    async fn reset_failure_counter(&self, service_name: &str) {
        let mut counters_guard = self.failure_counters.write().await;
        if let Some(count) = counters_guard.get_mut(service_name) {
            if *count > 0 {
                debug!("Resetting failure counter for service: {}", service_name);
                *count = 0;
            }
        }
    }
}

/// Check if a process is running by PID
fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let _pid_cstr = match CString::new(pid.to_string()) {
            Ok(cstr) => cstr,
            Err(_) => return false,
        };

        unsafe {
            libc::kill(pid as i32, 0) == 0
        }
    }

    #[cfg(windows)]
    {
        use std::ptr;
        use winapi::um::processthreadsapi::{GetExitCodeProcess, OpenProcess};
        use winapi::um::winnt::PROCESS_QUERY_INFORMATION;

        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_INFORMATION, 0, pid);
            if handle.is_null() {
                return false;
            }

            let mut exit_code: u32 = 0;
            let result = GetExitCodeProcess(handle, &mut exit_code);
            winapi::um::handleapi::CloseHandle(handle);

            result != 0 && exit_code == winapi::um::minwinbase::STILL_ACTIVE
        }
    }
}

impl HealthCheckResult {
    /// Check if this health result indicates the service is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy)
    }

    /// Check if this health result indicates the service is reachable
    pub fn is_reachable(&self) -> bool {
        !matches!(self.status, HealthStatus::Unreachable | HealthStatus::ProcessDead)
    }

    /// Get age of this health check result
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed().unwrap_or(Duration::ZERO)
    }
}

/// Helper function to create default health config with custom timeout
pub fn health_config_with_timeout(timeout_secs: u64) -> HealthConfig {
    HealthConfig {
        request_timeout: Duration::from_secs(timeout_secs),
        ..Default::default()
    }
}