//! File-based Service Registry
//!
//! This module implements a file-based service registry using JSON for
//! storing service information. It provides atomic operations for registration,
//! deregistration, and discovery with file locking for concurrent access.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Registry-specific errors
#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("Registry file not found: {0}")]
    FileNotFound(PathBuf),
    
    #[error("Invalid registry format: {0}")]
    InvalidFormat(String),
    
    #[error("File lock timeout")]
    LockTimeout,
    
    #[error("Permission denied: {0}")]
    PermissionDenied(PathBuf),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("Service already registered: {0}")]
    ServiceAlreadyExists(String),
    
    #[error("Process validation failed for PID {0}")]
    ProcessValidationFailed(u32),
}

/// Service status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Healthy,
    Unhealthy,
    Stopping,
}

/// Information about a registered service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    /// Service hostname or IP address
    pub host: String,
    
    /// Primary service port
    pub port: u16,
    
    /// Process ID of the service
    pub pid: u32,
    
    /// Service startup timestamp (ISO 8601)
    pub startup_time: String,
    
    /// Authentication token for secure communication
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
    
    /// Health check endpoint path
    pub health_endpoint: String,
    
    /// Additional service-specific ports
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub additional_ports: HashMap<String, u16>,
    
    /// Current service status
    pub status: ServiceStatus,
    
    /// Last health check timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_health_check: Option<String>,
    
    /// Service metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Service registry file format
#[derive(Debug, Serialize, Deserialize)]
struct RegistryFile {
    version: String,
    services: HashMap<String, ServiceInfo>,
    last_updated: String,
}

impl RegistryFile {
    fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            services: HashMap::new(),
            last_updated: current_iso_timestamp(),
        }
    }
}

/// File-based service registry implementation
pub struct ServiceRegistry {
    registry_path: PathBuf,
}

impl ServiceRegistry {
    /// Create a new service registry instance
    pub fn new<P: AsRef<Path>>(registry_path: Option<P>) -> Result<Self, RegistryError> {
        let registry_path = if let Some(path) = registry_path {
            path.as_ref().to_path_buf()
        } else {
            Self::default_registry_path()?
        };

        // Ensure the parent directory exists
        if let Some(parent) = registry_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!("Service registry initialized at: {}", registry_path.display());

        Ok(Self { registry_path })
    }

    /// Get default registry path (~/.workspace-qdrant/services.json)
    fn default_registry_path() -> Result<PathBuf, RegistryError> {
        let home_dir = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| RegistryError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Unable to determine home directory"
            )))?;

        Ok(PathBuf::from(home_dir)
            .join(".workspace-qdrant")
            .join("services.json"))
    }

    /// Register a service in the registry
    pub fn register_service(
        &self, 
        service_name: &str, 
        service_info: ServiceInfo
    ) -> Result<(), RegistryError> {
        debug!("Registering service: {} at {}:{}", service_name, service_info.host, service_info.port);

        // Validate process is running
        if !is_process_running(service_info.pid) {
            return Err(RegistryError::ProcessValidationFailed(service_info.pid));
        }

        let mut registry = self.load_or_create_registry()?;
        
        // Check if service already exists with different PID
        if let Some(existing) = registry.services.get(service_name) {
            if existing.pid != service_info.pid {
                warn!("Service {} already registered with different PID. Updating.", service_name);
            }
        }

        registry.services.insert(service_name.to_string(), service_info);
        registry.last_updated = current_iso_timestamp();

        self.save_registry(&registry)?;

        info!("Service {} registered successfully", service_name);
        Ok(())
    }

    /// Deregister a service from the registry
    pub fn deregister_service(&self, service_name: &str) -> Result<bool, RegistryError> {
        debug!("Deregistering service: {}", service_name);

        let mut registry = self.load_or_create_registry()?;
        let existed = registry.services.remove(service_name).is_some();
        
        if existed {
            registry.last_updated = current_iso_timestamp();
            self.save_registry(&registry)?;
            info!("Service {} deregistered successfully", service_name);
        } else {
            debug!("Service {} was not registered", service_name);
        }

        Ok(existed)
    }

    /// Discover a specific service by name
    pub fn discover_service(&self, service_name: &str) -> Result<Option<ServiceInfo>, RegistryError> {
        let registry = self.load_registry()?;
        
        if let Some(service_info) = registry.services.get(service_name) {
            // Validate the service process is still running
            if is_process_running(service_info.pid) {
                debug!("Service {} discovered at {}:{}", service_name, service_info.host, service_info.port);
                Ok(Some(service_info.clone()))
            } else {
                warn!("Service {} found but process {} is not running", service_name, service_info.pid);
                // Clean up stale entry
                let _ = self.deregister_service(service_name);
                Ok(None)
            }
        } else {
            debug!("Service {} not found in registry", service_name);
            Ok(None)
        }
    }

    /// List all registered services
    pub fn list_services(&self) -> Result<Vec<(String, ServiceInfo)>, RegistryError> {
        let registry = self.load_registry()?;
        let services: Vec<_> = registry.services.into_iter().collect();
        debug!("Listed {} registered services", services.len());
        Ok(services)
    }

    /// Update service status
    pub fn update_service_status(
        &self, 
        service_name: &str, 
        status: ServiceStatus
    ) -> Result<bool, RegistryError> {
        let mut registry = self.load_or_create_registry()?;
        
        if let Some(service_info) = registry.services.get_mut(service_name) {
            service_info.status = status;
            service_info.last_health_check = Some(current_iso_timestamp());
            registry.last_updated = current_iso_timestamp();
            
            let status_debug = service_info.status.clone();
            self.save_registry(&registry)?;
            debug!("Updated status for service {} to {:?}", service_name, status_debug);
            Ok(true)
        } else {
            debug!("Service {} not found for status update", service_name);
            Ok(false)
        }
    }

    /// Clean up stale registry entries
    pub fn cleanup_stale_entries(&self) -> Result<Vec<String>, RegistryError> {
        let mut registry = self.load_or_create_registry()?;
        let mut removed_services = Vec::new();
        
        registry.services.retain(|service_name, service_info| {
            if !is_process_running(service_info.pid) {
                warn!("Removing stale service {} (PID {} not running)", service_name, service_info.pid);
                removed_services.push(service_name.clone());
                false
            } else {
                true
            }
        });

        if !removed_services.is_empty() {
            registry.last_updated = current_iso_timestamp();
            self.save_registry(&registry)?;
            info!("Cleaned up {} stale service entries", removed_services.len());
        }

        Ok(removed_services)
    }

    /// Check if registry file exists
    pub fn exists(&self) -> bool {
        self.registry_path.exists()
    }

    /// Load the registry file or create a new one
    fn load_or_create_registry(&self) -> Result<RegistryFile, RegistryError> {
        if self.registry_path.exists() {
            self.load_registry()
        } else {
            Ok(RegistryFile::new())
        }
    }

    /// Load the registry file
    fn load_registry(&self) -> Result<RegistryFile, RegistryError> {
        if !self.registry_path.exists() {
            return Err(RegistryError::FileNotFound(self.registry_path.clone()));
        }

        let file = File::open(&self.registry_path)?;
        let reader = BufReader::new(file);
        
        let registry: RegistryFile = serde_json::from_reader(reader)
            .map_err(|e| RegistryError::InvalidFormat(e.to_string()))?;

        Ok(registry)
    }

    /// Save the registry file atomically
    fn save_registry(&self, registry: &RegistryFile) -> Result<(), RegistryError> {
        // Create temporary file for atomic write
        let temp_path = self.registry_path.with_extension("tmp");
        
        {
            let temp_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;
                
            let writer = BufWriter::new(temp_file);
            serde_json::to_writer_pretty(writer, registry)?;
        }

        // Atomically replace the original file
        std::fs::rename(&temp_path, &self.registry_path)?;

        debug!("Registry saved to {}", self.registry_path.display());
        Ok(())
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

/// Get current ISO 8601 timestamp
fn current_iso_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string()
}

impl ServiceInfo {
    /// Create a new ServiceInfo instance
    pub fn new(host: String, port: u16) -> Self {
        Self {
            host,
            port,
            pid: process::id(),
            startup_time: current_iso_timestamp(),
            auth_token: None,
            health_endpoint: "/health".to_string(),
            additional_ports: HashMap::new(),
            status: ServiceStatus::Starting,
            last_health_check: None,
            metadata: HashMap::new(),
        }
    }

    /// Set authentication token
    pub fn with_auth_token(mut self, token: String) -> Self {
        self.auth_token = Some(token);
        self
    }

    /// Set health endpoint
    pub fn with_health_endpoint(mut self, endpoint: String) -> Self {
        self.health_endpoint = endpoint;
        self
    }

    /// Add additional port
    pub fn with_additional_port(mut self, name: String, port: u16) -> Self {
        self.additional_ports.insert(name, port);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Generate a secure authentication token
    pub fn generate_auth_token() -> String {
        Uuid::new_v4().to_string()
    }
}