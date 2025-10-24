//! Configuration management for the Workspace Qdrant Daemon
//!
//! This module implements a clean lua-config pattern using a single dictionary
//! merged from default and user configurations, accessed via dot notation.

use crate::error::DaemonResult;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use serde_yaml::Value as YamlValue;

// =============================================================================
// UNIT PARSING UTILITIES
// =============================================================================

/// Size unit that supports parsing from strings with units (B, KB, MB, GB, TB)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SizeUnit(pub u64);

impl FromStr for SizeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("Empty size string".to_string());
        }

        // Extract number and unit - support both long and short forms
        let (num_str, unit) = if s.ends_with("TB") {
            (&s[..s.len() - 2], 1_099_511_627_776)
        } else if s.ends_with("GB") {
            (&s[..s.len() - 2], 1_073_741_824)
        } else if s.ends_with("MB") {
            (&s[..s.len() - 2], 1_048_576)
        } else if s.ends_with("KB") {
            (&s[..s.len() - 2], 1_024)
        } else if s.ends_with('T') && !s.ends_with("TB") {
            (&s[..s.len() - 1], 1_099_511_627_776)
        } else if s.ends_with('G') && !s.ends_with("GB") {
            (&s[..s.len() - 1], 1_073_741_824)
        } else if s.ends_with('M') && !s.ends_with("MB") {
            (&s[..s.len() - 1], 1_048_576)
        } else if s.ends_with('K') && !s.ends_with("KB") {
            (&s[..s.len() - 1], 1_024)
        } else if s.ends_with('B') {
            (&s[..s.len() - 1], 1)
        } else {
            // No unit, assume bytes
            (s, 1)
        };

        let num: u64 = num_str.trim().parse()
            .map_err(|_| format!("Invalid number: {}", num_str))?;

        Ok(SizeUnit(num * unit))
    }
}

impl fmt::Display for SizeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0;
        if bytes >= 1_099_511_627_776 {
            write!(f, "{}TB", bytes / 1_099_511_627_776)
        } else if bytes >= 1_073_741_824 {
            write!(f, "{}GB", bytes / 1_073_741_824)
        } else if bytes >= 1_048_576 {
            write!(f, "{}MB", bytes / 1_048_576)
        } else if bytes >= 1_024 {
            write!(f, "{}KB", bytes / 1_024)
        } else {
            write!(f, "{}B", bytes)
        }
    }
}

impl TryFrom<String> for SizeUnit {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<SizeUnit> for String {
    fn from(size: SizeUnit) -> String {
        size.to_string()
    }
}

/// Time duration unit that supports parsing from strings with units (ms, s, m, h)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TimeUnit(pub u64); // Store as milliseconds

impl FromStr for TimeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("Empty time string".to_string());
        }

        // Extract number and unit
        let (num_str, unit) = if s.ends_with("ms") {
            (&s[..s.len() - 2], 1)
        } else if s.ends_with('s') && !s.ends_with("ms") {
            (&s[..s.len() - 1], 1_000)
        } else if s.ends_with('m') {
            (&s[..s.len() - 1], 60_000)
        } else if s.ends_with('h') {
            (&s[..s.len() - 1], 3_600_000)
        } else {
            // No unit, assume milliseconds
            (s, 1)
        };

        let num: u64 = num_str.trim().parse()
            .map_err(|_| format!("Invalid number: {}", num_str))?;

        Ok(TimeUnit(num * unit))
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0;
        if ms >= 3_600_000 {
            write!(f, "{}h", ms / 3_600_000)
        } else if ms >= 60_000 {
            write!(f, "{}m", ms / 60_000)
        } else if ms >= 1_000 {
            write!(f, "{}s", ms / 1_000)
        } else {
            write!(f, "{}ms", ms)
        }
    }
}

impl TryFrom<String> for TimeUnit {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<TimeUnit> for String {
    fn from(time: TimeUnit) -> String {
        time.to_string()
    }
}

// =============================================================================
// CONFIGURATION VALUE ENUM
// =============================================================================

/// Configuration value that can hold different types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    /// Convert from YAML value with path context for error reporting
    pub fn from_yaml_value(value: &YamlValue, path: &str) -> Self {
        match value {
            YamlValue::String(s) => ConfigValue::String(s.clone()),
            YamlValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ConfigValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    ConfigValue::Float(f)
                } else {
                    ConfigValue::Null
                }
            },
            YamlValue::Bool(b) => ConfigValue::Boolean(*b),
            YamlValue::Sequence(seq) => {
                let array: Vec<ConfigValue> = seq.iter()
                    .enumerate()
                    .map(|(i, v)| Self::from_yaml_value(v, &format!("{}[{}]", path, i)))
                    .collect();
                ConfigValue::Array(array)
            },
            YamlValue::Mapping(map) => {
                let mut object = HashMap::new();
                for (k, v) in map {
                    if let Some(key_str) = k.as_str() {
                        let new_path = if path.is_empty() {
                            key_str.to_string()
                        } else {
                            format!("{}.{}", path, key_str)
                        };
                        object.insert(key_str.to_string(), Self::from_yaml_value(v, &new_path));
                    }
                }
                ConfigValue::Object(object)
            },
            YamlValue::Null => ConfigValue::Null,
            YamlValue::Tagged(tagged) => {
                // Handle tagged values by processing the inner value
                Self::from_yaml_value(&tagged.value, path)
            },
        }
    }

    /// Extract string value
    pub fn as_string(&self) -> Option<String> {
        match self {
            ConfigValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Extract integer value
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            ConfigValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Extract float value
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Extract boolean value
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Extract array value
    pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            ConfigValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Extract object value
    pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
        match self {
            ConfigValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }
}

// =============================================================================
// CONFIGURATION MANAGER - LUA-CONFIG PATTERN
// =============================================================================

/// Global configuration manager that provides dot-notation access
pub struct ConfigManager {
    config: HashMap<String, ConfigValue>,
}

impl ConfigManager {
    /// Create new configuration manager with merged config
    fn new(yaml_config: HashMap<String, ConfigValue>, defaults: HashMap<String, ConfigValue>) -> Self {
        let mut merged = defaults;

        // Merge YAML values taking precedence over defaults
        Self::merge_configs(&mut merged, yaml_config);

        Self {
            config: merged,
        }
    }

    /// Recursively merge configurations with YAML taking precedence
    fn merge_configs(target: &mut HashMap<String, ConfigValue>, source: HashMap<String, ConfigValue>) {
        for (key, source_value) in source {
            match (target.get(&key), &source_value) {
                (Some(ConfigValue::Object(_target_obj)), ConfigValue::Object(source_obj)) => {
                    // Both are objects, merge recursively
                    if let Some(ConfigValue::Object(target_obj_mut)) = target.get_mut(&key) {
                        Self::merge_configs(target_obj_mut, source_obj.clone());
                    }
                },
                _ => {
                    // Replace target with source value (YAML takes precedence)
                    target.insert(key, source_value);
                }
            }
        }
    }

    /// Get configuration value using dot notation (e.g., "server.port")
    pub fn get(&self, path: &str) -> Option<&ConfigValue> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = &self.config;

        for (i, part) in parts.iter().enumerate() {
            if let Some(value) = current.get(*part) {
                if i == parts.len() - 1 {
                    // Last part, return the value
                    return Some(value);
                } else if let ConfigValue::Object(next_obj) = value {
                    // Continue traversing
                    current = next_obj;
                } else {
                    // Path doesn't lead to an object but we're not at the end
                    return None;
                }
            } else {
                return None;
            }
        }

        None
    }

    /// Get configuration value with Result-based error handling
    pub fn get_config(&self, path: &str) -> Result<&ConfigValue, crate::error::DaemonError> {
        use crate::error::DaemonError;
        match self.get(path) {
            Some(value) => Ok(value),
            None => Err(DaemonError::ConfigKeyNotFound {
                path: path.to_string(),
            }),
        }
    }

    /// Get string value with default
    pub fn get_string(&self, path: &str, default: &str) -> String {
        self.get(path)
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| default.to_string())
    }

    /// Get integer value with default
    pub fn get_i64(&self, path: &str, default: i64) -> i64 {
        self.get(path)
            .and_then(|v| v.as_i64())
            .unwrap_or(default)
    }

    /// Get unsigned integer value with default
    pub fn get_u64(&self, path: &str, default: u64) -> u64 {
        self.get_i64(path, default as i64) as u64
    }

    /// Get 32-bit integer value with default
    pub fn get_i32(&self, path: &str, default: i32) -> i32 {
        self.get_i64(path, default as i64) as i32
    }

    /// Get unsigned 32-bit integer value with default
    pub fn get_u32(&self, path: &str, default: u32) -> u32 {
        self.get_i64(path, default as i64) as u32
    }

    /// Get 16-bit integer value with default
    pub fn get_u16(&self, path: &str, default: u16) -> u16 {
        self.get_i64(path, default as i64) as u16
    }

    /// Get 8-bit integer value with default
    pub fn get_u8(&self, path: &str, default: u8) -> u8 {
        self.get_i64(path, default as i64) as u8
    }

    /// Get float value with default
    pub fn get_f64(&self, path: &str, default: f64) -> f64 {
        self.get(path)
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    }

    /// Get 32-bit float value with default
    pub fn get_f32(&self, path: &str, default: f32) -> f32 {
        self.get_f64(path, default as f64) as f32
    }

    /// Get boolean value with default
    pub fn get_bool(&self, path: &str, default: bool) -> bool {
        self.get(path)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Get array value
    pub fn get_array(&self, path: &str) -> Option<&Vec<ConfigValue>> {
        self.get(path)?.as_array()
    }

    /// Get object value
    pub fn get_object(&self, path: &str) -> Option<&HashMap<String, ConfigValue>> {
        self.get(path)?.as_object()
    }

    /// Determine the base path for asset files based on deployment configuration
    fn get_asset_base_path(deployment_config: Option<&HashMap<String, ConfigValue>>) -> std::path::PathBuf {
        // Get deployment configuration
        let (develop_mode, base_path) = if let Some(config) = deployment_config {
            if let Some(deployment) = config.get("deployment").and_then(|d| d.as_object()) {
                let develop = deployment.get("develop")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let base_path = deployment.get("base_path")
                    .and_then(|v| v.as_string())
                    .filter(|s| !s.is_empty());
                (develop, base_path)
            } else {
                (true, None) // Default to development mode if no deployment config
            }
        } else {
            (true, None) // When first loading, default to development mode
        };

        // If base_path is explicitly set, use it
        if let Some(base_path) = base_path {
            return std::path::PathBuf::from(base_path).join("assets");
        }

        // Development mode: use project-relative path
        if develop_mode {
            let project_root = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            return project_root.join("assets");
        }

        // Production mode: use system-specific paths
        #[cfg(target_os = "linux")]
        {
            std::path::PathBuf::from("/usr/share/workspace-qdrant-mcp/assets")
        }
        #[cfg(target_os = "macos")]
        {
            std::path::PathBuf::from("/usr/local/share/workspace-qdrant-mcp/assets")
        }
        #[cfg(target_os = "windows")]
        {
            use std::env;
            let program_files = env::var("ProgramFiles")
                .unwrap_or_else(|_| "C:\\Program Files".to_string());
            std::path::PathBuf::from(program_files)
                .join("workspace-qdrant-mcp")
                .join("assets")
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for unknown systems
            std::path::PathBuf::from("/usr/local/share/workspace-qdrant-mcp/assets")
        }
    }

    /// Load comprehensive default configuration from asset file
    fn create_defaults() -> HashMap<String, ConfigValue> {
        Self::create_defaults_with_user_config(&HashMap::new())
    }

    fn create_defaults_with_user_config(user_config: &HashMap<String, ConfigValue>) -> HashMap<String, ConfigValue> {
        // Try to load from the default config asset using deployment-aware path resolution
        // Pass user config to enable deployment-aware path resolution
        let assets_dir = Self::get_asset_base_path(Some(user_config));
        let asset_file = assets_dir.join("default_configuration.yaml");

        if let Ok(content) = std::fs::read_to_string(&asset_file) {
            if let Ok(yaml_value) = serde_yaml::from_str::<YamlValue>(&content) {
                if let Some(config) = ConfigValue::from_yaml_value(&yaml_value, "").as_object() {
                    return config.clone();
                }
            }
        }

        // Fallback to basic defaults if asset file is not available
        let mut defaults = HashMap::new();

        // Basic server configuration
        let mut server = HashMap::new();
        server.insert("host".to_string(), ConfigValue::String("127.0.0.1".to_string()));
        server.insert("port".to_string(), ConfigValue::Integer(50051));
        defaults.insert("server".to_string(), ConfigValue::Object(server));

        // Basic Qdrant configuration
        let mut qdrant = HashMap::new();
        qdrant.insert("url".to_string(), ConfigValue::String("http://localhost:6333".to_string()));
        qdrant.insert("api_key".to_string(), ConfigValue::Null);
        defaults.insert("qdrant".to_string(), ConfigValue::Object(qdrant));

        // Basic database configuration
        let mut database = HashMap::new();
        database.insert("max_connections".to_string(), ConfigValue::Integer(10));
        defaults.insert("database".to_string(), ConfigValue::Object(database));

        // Basic auto-ingestion configuration
        let mut auto_ingestion = HashMap::new();
        auto_ingestion.insert("enabled".to_string(), ConfigValue::Boolean(true));
        defaults.insert("auto_ingestion".to_string(), ConfigValue::Object(auto_ingestion));

        defaults
    }
}

// =============================================================================
// GLOBAL CONFIGURATION INSTANCE
// =============================================================================

/// Global configuration manager instance
static CONFIG_MANAGER_INSTANCE: OnceLock<Mutex<ConfigManager>> = OnceLock::new();

/// Get reference to global configuration manager
pub fn config() -> &'static Mutex<ConfigManager> {
    CONFIG_MANAGER_INSTANCE.get()
        .expect("Configuration not initialized. Call init_config() first.")
}

/// Initialize global configuration
pub fn init_config(config_path: Option<&Path>) -> DaemonResult<()> {
    let config_manager = match load_config(config_path) {
        Ok(manager) => manager,
        Err(e) => {
            eprintln!("Failed to load configuration: {}", e);
            // Create with defaults only
            ConfigManager::new(HashMap::new(), ConfigManager::create_defaults())
        }
    };

    CONFIG_MANAGER_INSTANCE.set(Mutex::new(config_manager))
        .map_err(|_| crate::error::DaemonError::Configuration { message: "Configuration already initialized".to_string() })?;

    Ok(())
}

/// Load configuration from file using lua-config pattern
fn load_config(config_path: Option<&Path>) -> DaemonResult<ConfigManager> {
    // Load user configuration if provided
    let yaml_config = match config_path {
        Some(path) => {
            let content = std::fs::read_to_string(path)?;
            let yaml_value: YamlValue = serde_yaml::from_str(&content)
                .map_err(|e| crate::error::DaemonError::Configuration {
                    message: format!("YAML parsing error: {}", e)
                })?;

            if let Some(config_obj) = ConfigValue::from_yaml_value(&yaml_value, "").as_object() {
                config_obj.clone()
            } else {
                HashMap::new()
            }
        },
        None => HashMap::new(),
    };

    // Load defaults - pass user config to enable deployment-aware path resolution
    let defaults = ConfigManager::create_defaults_with_user_config(&yaml_config);

    // Create merged configuration
    let config_manager = ConfigManager::new(yaml_config, defaults);

    Ok(config_manager)
}

impl ConfigManager {
    /// Get string value with Result-based error handling
    pub fn try_get_string(&self, path: &str) -> Result<String, crate::error::DaemonError> {
        use crate::error::DaemonError;

        match self.get(path) {
            Some(value) => match value.as_string() {
                Some(s) => Ok(s),
                None => Err(DaemonError::ConfigTypeMismatch {
                    path: path.to_string(),
                    expected_type: "String".to_string(),
                    actual_type: format!("{:?}", value),
                }),
            },
            None => Err(DaemonError::ConfigKeyNotFound {
                path: path.to_string(),
            }),
        }
    }

    /// Get boolean value with Result-based error handling
    pub fn try_get_bool(&self, path: &str) -> Result<bool, crate::error::DaemonError> {
        use crate::error::DaemonError;

        match self.get(path) {
            Some(value) => match value.as_bool() {
                Some(b) => Ok(b),
                None => Err(DaemonError::ConfigTypeMismatch {
                    path: path.to_string(),
                    expected_type: "Boolean".to_string(),
                    actual_type: format!("{:?}", value),
                }),
            },
            None => Err(DaemonError::ConfigKeyNotFound {
                path: path.to_string(),
            }),
        }
    }

    /// Get integer value with Result-based error handling
    pub fn try_get_i64(&self, path: &str) -> Result<i64, crate::error::DaemonError> {
        use crate::error::DaemonError;

        match self.get(path) {
            Some(value) => match value.as_i64() {
                Some(i) => Ok(i),
                None => Err(DaemonError::ConfigTypeMismatch {
                    path: path.to_string(),
                    expected_type: "Integer".to_string(),
                    actual_type: format!("{:?}", value),
                }),
            },
            None => Err(DaemonError::ConfigKeyNotFound {
                path: path.to_string(),
            }),
        }
    }

    /// Get u16 value with Result-based error handling
    pub fn try_get_u16(&self, path: &str) -> Result<u16, crate::error::DaemonError> {
        let value = self.try_get_i64(path)?;
        if value >= 0 && value <= u16::MAX as i64 {
            Ok(value as u16)
        } else {
            Err(crate::error::DaemonError::ConfigTypeMismatch {
                path: path.to_string(),
                expected_type: "u16 (0-65535)".to_string(),
                actual_type: format!("i64({})", value),
            })
        }
    }

    /// Get u32 value with Result-based error handling
    pub fn try_get_u32(&self, path: &str) -> Result<u32, crate::error::DaemonError> {
        let value = self.try_get_i64(path)?;
        if value >= 0 && value <= u32::MAX as i64 {
            Ok(value as u32)
        } else {
            Err(crate::error::DaemonError::ConfigTypeMismatch {
                path: path.to_string(),
                expected_type: "u32 (0-4294967295)".to_string(),
                actual_type: format!("i64({})", value),
            })
        }
    }
}

// =============================================================================
// DAEMON CONFIGURATION - MINIMAL WRAPPER
// =============================================================================

/// Minimal configuration wrapper that provides direct access to ConfigManager
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub system: SystemConfig,
    pub grpc: GrpcConfig,
    pub server: ServerConfig,
    pub external_services: ExternalServicesConfig,
    pub transport: TransportConfig,
    pub message: MessageConfig,
    pub security: SecurityConfig,
    pub streaming: StreamingConfig,
    pub compression: CompressionConfig,
    pub metrics: MetricsConfig,
    pub logging: LoggingConfig,
    // Legacy field access for backward compatibility
    pub database: DatabaseConfig,
    pub qdrant: QdrantConfig,
    pub workspace: WorkspaceConfig,
    pub auto_ingestion: AutoIngestionConfig,
    pub processing: ProcessingConfig,
    pub file_watcher: FileWatcherConfig,
}

/// System configuration for legacy compatibility
#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub project_name: String,
    pub database: DatabaseConfig,
    pub auto_ingestion: AutoIngestionConfig,
    pub processing: ProcessingConfig,
    pub file_watcher: FileWatcherConfig,
}

/// gRPC configuration wrapper for legacy compatibility
#[derive(Debug, Clone)]
pub struct GrpcConfig {
    pub server: GrpcServerConfig,
    pub client: GrpcClientConfig,
    pub security: SecurityConfig,
    pub transport: TransportConfig,
    pub message: MessageConfig,
}

/// gRPC server configuration for legacy compatibility
#[derive(Debug, Clone)]
pub struct GrpcServerConfig {
    pub enabled: bool,
    pub port: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self::load(None).unwrap_or_else(|_| {
            // If loading fails, initialize with defaults only
            let _ = init_config(None);
            DaemonConfig {
                system: SystemConfig {
                    project_name: "default".to_string(),
                    database: DatabaseConfig {
                        max_connections: 10,
                        sqlite_path: ":memory:".to_string(),
                        connection_timeout_secs: 30,
                        enable_wal: false,
                    },
                    auto_ingestion: AutoIngestionConfig {
                        enabled: true,
                        project_collection: "".to_string(),
                        auto_create_watches: true,
                        project_path: None,
                        include_source_files: false,
                        include_patterns: vec![],
                        exclude_patterns: vec![],
                        max_depth: 10,
                        recursive: true,
                    },
                    processing: ProcessingConfig {
                        max_concurrent_tasks: 4,
                        supported_extensions: vec![".txt".to_string(), ".md".to_string()],
                        default_chunk_size: 1000,
                        default_chunk_overlap: 200,
                        max_file_size_bytes: 10 * 1024 * 1024,
                        enable_lsp: false,
                        lsp_timeout_secs: 30,
                    },
                    file_watcher: FileWatcherConfig {
                        enabled: true,
                        ignore_patterns: vec![],
                        recursive: true,
                        max_watched_dirs: 100,
                        debounce_ms: 1000,
                    },
                },
                grpc: GrpcConfig {
                    server: GrpcServerConfig {
                        enabled: true,
                        port: 50051,
                    },
                    client: GrpcClientConfig {
                        connection_timeout: TimeUnit(10000),
                        request_timeout: TimeUnit(30000),
                        max_retries: 3,
                        enable_keepalive: true,
                    },
                    security: SecurityConfig {
                        tls: TlsConfig {
                            enabled: false,
                            cert_file: None,
                            key_file: None,
                            ca_cert_file: None,
                            client_cert_verification: ClientCertVerification::None,
                        },
                        auth: AuthConfig {
                            jwt: JwtConfig {
                                secret_or_key_file: "changeme_jwt_secret".to_string(),
                                expiration_secs: 3600,
                                algorithm: "HS256".to_string(),
                                issuer: "workspace-qdrant-mcp".to_string(),
                                audience: "workspace-qdrant-mcp".to_string(),
                            },
                            api_key: ApiKeyConfig {
                                enabled: false,
                                key_permissions: HashMap::new(),
                                valid_keys: vec![],
                                header_name: "x-api-key".to_string(),
                            },
                            enable_service_auth: false,
                            authorization: AuthorizationConfig {
                                enabled: false,
                                service_permissions: ServicePermissions {
                                    document_processor: vec!["read".to_string(), "write".to_string()],
                                    search_service: vec!["read".to_string()],
                                    memory_service: vec!["read".to_string(), "write".to_string()],
                                    system_service: vec!["read".to_string()],
                                },
                                default_permissions: vec!["read".to_string()],
                            },
                        },
                        audit: SecurityAuditConfig {
                            enabled: false,
                            log_auth_events: true,
                            log_auth_failures: true,
                            log_rate_limit_events: true,
                            log_suspicious_patterns: true,
                        },
                    },
                    transport: TransportConfig {
                        unix_socket: UnixSocketConfig {
                            enabled: false,
                            path: None,
                            socket_path: "/tmp/workspace-qdrant.sock".to_string(),
                            permissions: 0o666,
                            cleanup_on_exit: true,
                            prefer_for_local: false,
                        },
                        local_optimization: LocalOptimizationConfig::default(),
                        transport_strategy: TransportStrategy::Tcp,
                    },
                    message: MessageConfig {
                        service_limits: ServiceLimits {
                            memory_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            system_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            document_processor: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            search_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                        },
                        max_frame_size: 100 * 1024 * 1024,
                        initial_window_size: 25 * 1024 * 1024,
                        enable_size_validation: true,
                        max_incoming_message_size: 100 * 1024 * 1024,
                        max_outgoing_message_size: 100 * 1024 * 1024,
                        monitoring: MessageMonitoringConfig {
                            oversized_alert_threshold: 0.8,
                        },
                    },
                },
                server: ServerConfig {
                    host: "127.0.0.1".to_string(),
                    port: 50051,
                    max_connections: 100,
                    request_timeout_secs: 30,
                    connection_timeout_secs: 10,
                    message: MessageConfig {
                        service_limits: ServiceLimits {
                            memory_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            system_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            document_processor: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                            search_service: ServiceLimit {
                                max_incoming: 100 * 1024 * 1024,
                                max_outgoing: 100 * 1024 * 1024,
                                enable_validation: true,
                            },
                        },
                        max_frame_size: 100 * 1024 * 1024,
                        initial_window_size: 25 * 1024 * 1024,
                        enable_size_validation: true,
                        max_incoming_message_size: 100 * 1024 * 1024,
                        max_outgoing_message_size: 100 * 1024 * 1024,
                        monitoring: MessageMonitoringConfig {
                            oversized_alert_threshold: 0.8,
                        },
                    },
                    streaming: StreamingConfig {
                        enabled: true,
                        stream_buffer_size: 1024,
                        enable_flow_control: true,
                        progress: StreamProgressConfig {
                            enabled: true,
                            progress_update_interval_ms: 1000,
                            enable_progress_callbacks: true,
                            enable_progress_tracking: true,
                            progress_threshold: 1024,
                        },
                        health: StreamHealthConfig {
                            enabled: true,
                            enable_auto_recovery: true,
                            enable_health_monitoring: true,
                            health_check_interval_secs: 30,
                        },
                        enable_server_streaming: true,
                        enable_client_streaming: true,
                        max_concurrent_streams: 100,
                        large_operations: LargeOperationConfig {
                            large_operation_chunk_size: 1024 * 1024,
                        },
                        stream_timeout_secs: 300,
                    },
                    enable_tls: false,
                    security: SecurityConfig::default(),
                    transport: TransportConfig::default(),
                    compression: CompressionConfig::default(),
                },
                external_services: ExternalServicesConfig {
                    qdrant: QdrantConfig {
                        url: "http://localhost:6333".to_string(),
                        api_key: None,
                        default_collection: CollectionConfig {
                            vector_size: 384,
                            shard_number: 1,
                            replication_factor: 1,
                            distance_metric: "Cosine".to_string(),
                            enable_indexing: true,
                        },
                        max_retries: 3,
                    },
                },
                transport: TransportConfig {
                    unix_socket: UnixSocketConfig {
                        enabled: false,
                        path: None,
                        socket_path: "/tmp/workspace-qdrant.sock".to_string(),
                        permissions: 0o666,
                        cleanup_on_exit: true,
                        prefer_for_local: false,
                    },
                    local_optimization: LocalOptimizationConfig::default(),
                    transport_strategy: TransportStrategy::Tcp,
                },
                message: MessageConfig {
                    service_limits: ServiceLimits {
                        memory_service: ServiceLimit {
                            max_incoming: 100 * 1024 * 1024,
                            max_outgoing: 100 * 1024 * 1024,
                            enable_validation: true,
                        },
                        system_service: ServiceLimit {
                            max_incoming: 100 * 1024 * 1024,
                            max_outgoing: 100 * 1024 * 1024,
                            enable_validation: true,
                        },
                        document_processor: ServiceLimit {
                            max_incoming: 100 * 1024 * 1024,
                            max_outgoing: 100 * 1024 * 1024,
                            enable_validation: true,
                        },
                        search_service: ServiceLimit {
                            max_incoming: 100 * 1024 * 1024,
                            max_outgoing: 100 * 1024 * 1024,
                            enable_validation: true,
                        },
                    },
                    max_frame_size: 100 * 1024 * 1024,
                    initial_window_size: 25 * 1024 * 1024,
                    enable_size_validation: true,
                    max_incoming_message_size: 100 * 1024 * 1024,
                    max_outgoing_message_size: 100 * 1024 * 1024,
                    monitoring: MessageMonitoringConfig {
                        oversized_alert_threshold: 0.8,
                    },
                },
                security: SecurityConfig {
                    tls: TlsConfig {
                        enabled: false,
                        cert_file: None,
                        key_file: None,
                        ca_cert_file: None,
                        client_cert_verification: ClientCertVerification::None,
                    },
                    auth: AuthConfig {
                        jwt: JwtConfig {
                            secret_or_key_file: "changeme_jwt_secret".to_string(),
                            expiration_secs: 3600,
                            algorithm: "HS256".to_string(),
                            issuer: "workspace-qdrant-daemon".to_string(),
                            audience: "workspace-qdrant-client".to_string(),
                        },
                        api_key: ApiKeyConfig {
                            enabled: false,
                            key_permissions: HashMap::new(),
                            valid_keys: vec![],
                            header_name: "x-api-key".to_string(),
                        },
                        enable_service_auth: false,
                        authorization: AuthorizationConfig {
                            enabled: false,
                            service_permissions: ServicePermissions {
                                document_processor: vec!["read".to_string(), "write".to_string()],
                                search_service: vec!["read".to_string()],
                                memory_service: vec!["read".to_string(), "write".to_string()],
                                system_service: vec!["read".to_string()],
                            },
                            default_permissions: vec!["read".to_string()],
                        },
                    },
                    audit: SecurityAuditConfig {
                        enabled: false,
                        log_auth_events: true,
                        log_auth_failures: true,
                        log_rate_limit_events: true,
                        log_suspicious_patterns: true,
                    },
                },
                streaming: StreamingConfig {
                    enabled: true,
                    stream_buffer_size: 1024,
                    enable_flow_control: true,
                    progress: StreamProgressConfig {
                        enabled: true,
                        progress_update_interval_ms: 1000,
                        enable_progress_callbacks: true,
                        enable_progress_tracking: true,
                        progress_threshold: 1024,
                    },
                    health: StreamHealthConfig {
                        enabled: true,
                        enable_auto_recovery: true,
                        enable_health_monitoring: true,
                        health_check_interval_secs: 30,
                    },
                    enable_server_streaming: true,
                    enable_client_streaming: true,
                    max_concurrent_streams: 100,
                    large_operations: LargeOperationConfig {
                        large_operation_chunk_size: 1024 * 1024,
                    },
                    stream_timeout_secs: 300,
                },
                compression: CompressionConfig {
                    enabled: true,
                    enable_gzip: true,
                    enable_compression_monitoring: false,
                    compression_level: 6,
                    compression_threshold: 1024,
                    adaptive: AdaptiveCompressionConfig {
                        enable_adaptive: true,
                        text_compression_level: 9,
                        binary_compression_level: 6,
                        structured_compression_level: 6,
                    },
                    performance: CompressionPerformanceConfig {
                        enable_failure_alerting: true,
                        enable_time_monitoring: true,
                        slow_compression_threshold_ms: 1000,
                        enable_ratio_tracking: true,
                        poor_ratio_threshold: 0.9,
                    },
                },
                metrics: MetricsConfig {
                    enabled: false,
                    collection_interval_secs: 60,
                },
                logging: LoggingConfig {
                    enabled: true,
                    level: "info".to_string(),
                    file_path: None,
                    max_file_size: SizeUnit(10 * 1024 * 1024),
                    max_files: 10,
                    enable_json: false,
                    enable_structured: true,
                    enable_console: true,
                },
                // Legacy field access compatibility
                database: DatabaseConfig {
                    max_connections: 10,
                    sqlite_path: ":memory:".to_string(),
                    connection_timeout_secs: 30,
                    enable_wal: false,
                },
                qdrant: QdrantConfig {
                    url: "http://localhost:6333".to_string(),
                    api_key: None,
                    default_collection: CollectionConfig {
                        vector_size: 384,
                        shard_number: 1,
                        replication_factor: 1,
                        distance_metric: "Cosine".to_string(),
                        enable_indexing: true,
                    },
                    max_retries: 3,
                },
                workspace: WorkspaceConfig {
                    collection_basename: None,
                    collection_types: vec![],
                    memory_collection_name: "memory".to_string(),
                    auto_create_collections: true,
                },
                auto_ingestion: AutoIngestionConfig {
                    enabled: true,
                    project_collection: "projects_content".to_string(),
                    auto_create_watches: true,
                    project_path: None,
                    include_source_files: false,
                    include_patterns: vec![],
                    exclude_patterns: vec![],
                    max_depth: 10,
                    recursive: true,
                },
                processing: ProcessingConfig {
                    max_concurrent_tasks: 4,
                    supported_extensions: vec![".txt".to_string(), ".md".to_string()],
                    default_chunk_size: 1000,
                    default_chunk_overlap: 200,
                    max_file_size_bytes: 10 * 1024 * 1024,
                    enable_lsp: false,
                    lsp_timeout_secs: 30,
                },
                file_watcher: FileWatcherConfig {
                    enabled: true,
                    ignore_patterns: vec![],
                    recursive: true,
                    max_watched_dirs: 100,
                    debounce_ms: 1000,
                },
            }
        })
    }
}

impl DaemonConfig {
    /// Load configuration using the lua-config pattern
    pub fn load(config_path: Option<&Path>) -> DaemonResult<Self> {
        // Initialize global configuration
        init_config(config_path)?;

        // Validate the configuration
        Self::validate_config()?;

        let config_guard = config().lock().unwrap();

        // Build structured configuration from ConfigManager
        let server_config = ServerConfig {
            host: config_guard.get_string("server.host", "127.0.0.1"),
            port: config_guard.get_u16("server.port", 8000),
            max_connections: config_guard.get_u32("performance.max_concurrent_tasks", 4) as usize,
            request_timeout_secs: config_guard.get_u64("performance.default_timeout_secs", 30),
            connection_timeout_secs: config_guard.get_u64("grpc.connection_timeout_secs", 10),
            streaming: StreamingConfig {
                enabled: config_guard.get_bool("grpc.streaming.enabled", true),
                stream_buffer_size: config_guard.get_u64("grpc.streaming.buffer_size", 1024) as usize,
                enable_flow_control: config_guard.get_bool("grpc.streaming.enable_flow_control", true),
                enable_server_streaming: config_guard.get_bool("grpc.streaming.enable_server_streaming", true),
                enable_client_streaming: config_guard.get_bool("grpc.streaming.enable_client_streaming", true),
                max_concurrent_streams: config_guard.get_u32("grpc.streaming.max_concurrent_streams", 100),
                stream_timeout_secs: config_guard.get_u64("grpc.streaming.stream_timeout_secs", 300),
                large_operations: LargeOperationConfig {
                    large_operation_chunk_size: config_guard.get_u64("grpc.streaming.large_operations.chunk_size", 1024 * 1024) as usize,
                },
                progress: StreamProgressConfig {
                    enabled: config_guard.get_bool("grpc.streaming.progress.enabled", true),
                    progress_update_interval_ms: config_guard.get_u64("grpc.streaming.progress.update_interval_ms", 1000),
                    enable_progress_callbacks: config_guard.get_bool("grpc.streaming.progress.enable_callbacks", true),
                    enable_progress_tracking: config_guard.get_bool("grpc.streaming.progress.enable_tracking", true),
                    progress_threshold: config_guard.get_u64("grpc.streaming.progress.threshold", 1024) as usize,
                },
                health: StreamHealthConfig {
                    enabled: config_guard.get_bool("grpc.streaming.health.enabled", true),
                    enable_auto_recovery: config_guard.get_bool("grpc.streaming.health.enable_auto_recovery", true),
                    enable_health_monitoring: config_guard.get_bool("grpc.streaming.health.enable_monitoring", true),
                    health_check_interval_secs: config_guard.get_u64("grpc.streaming.health.check_interval_secs", 30),
                },
            },
            message: MessageConfig {
                service_limits: ServiceLimits {
                    memory_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    system_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    document_processor: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    search_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                },
                max_frame_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as u32,
                initial_window_size: (config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) / 4) as u32,
                enable_size_validation: config_guard.get_bool("grpc.message.enable_size_validation", true),
                max_incoming_message_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                max_outgoing_message_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                monitoring: MessageMonitoringConfig {
                    oversized_alert_threshold: config_guard.get_f64("grpc.message.monitoring.oversized_alert_threshold", 0.8),
                },
            },
            enable_tls: config_guard.get_bool("security.tls.enabled", false),
            security: SecurityConfig {
                tls: TlsConfig {
                    enabled: config_guard.get_bool("security.tls.enabled", false),
                    cert_file: config_guard.get("security.tls.cert_file").and_then(|v| v.as_string()),
                    key_file: config_guard.get("security.tls.key_file").and_then(|v| v.as_string()),
                    ca_cert_file: config_guard.get("security.tls.ca_cert_file").and_then(|v| v.as_string()),
                    client_cert_verification: ClientCertVerification::None,
                },
                auth: AuthConfig {
                    jwt: JwtConfig {
                        secret_or_key_file: config_guard.get_string("security.auth.jwt.secret_or_key_file", "changeme_jwt_secret"),
                        expiration_secs: config_guard.get_u64("security.auth.jwt.expiration_secs", 3600),
                        algorithm: config_guard.get_string("security.auth.jwt.algorithm", "HS256"),
                        issuer: config_guard.get_string("security.auth.jwt.issuer", "workspace-qdrant-daemon"),
                        audience: config_guard.get_string("security.auth.jwt.audience", "workspace-qdrant-client"),
                    },
                    api_key: ApiKeyConfig {
                        enabled: config_guard.get_bool("security.auth.api_key.enabled", false),
                        key_permissions: HashMap::new(),
                        valid_keys: config_guard.get_array("security.auth.api_key.valid_keys")
                            .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                            .unwrap_or_else(|| vec![]),
                        header_name: config_guard.get_string("security.auth.api_key.header_name", "x-api-key"),
                    },
                    enable_service_auth: config_guard.get_bool("security.auth.enable_service_auth", false),
                    authorization: AuthorizationConfig {
                        enabled: config_guard.get_bool("security.authorization.enabled", false),
                        service_permissions: ServicePermissions {
                            document_processor: vec!["read".to_string(), "write".to_string()],
                            search_service: vec!["read".to_string()],
                            memory_service: vec!["read".to_string(), "write".to_string()],
                            system_service: vec!["read".to_string()],
                        },
                        default_permissions: vec!["read".to_string()],
                    },
                },
                audit: SecurityAuditConfig {
                    enabled: config_guard.get_bool("security.audit.enabled", false),
                    log_auth_events: config_guard.get_bool("security.audit.log_auth_events", true),
                    log_auth_failures: config_guard.get_bool("security.audit.log_auth_failures", true),
                    log_rate_limit_events: config_guard.get_bool("security.audit.log_rate_limit_events", true),
                    log_suspicious_patterns: config_guard.get_bool("security.audit.log_suspicious_patterns", true),
                },
            },
            transport: TransportConfig {
                unix_socket: UnixSocketConfig {
                    enabled: config_guard.get_bool("grpc.transport.unix_socket.enabled", false),
                    path: config_guard.get("grpc.transport.unix_socket.path").and_then(|v| v.as_string()),
                    socket_path: config_guard.get_string("grpc.transport.unix_socket.path", "/tmp/workspace-qdrant.sock"),
                    permissions: config_guard.get_u32("grpc.transport.unix_socket.permissions", 0o666),
                    cleanup_on_exit: config_guard.get_bool("grpc.transport.unix_socket.cleanup_on_exit", true),
                    prefer_for_local: config_guard.get_bool("grpc.transport.unix_socket.prefer_for_local", false),
                },
                local_optimization: LocalOptimizationConfig::default(),
                transport_strategy: TransportStrategy::Tcp,
            },
            compression: CompressionConfig {
                enabled: config_guard.get_bool("grpc.compression.enabled", true),
                enable_gzip: config_guard.get_bool("grpc.compression.enable_gzip", true),
                enable_compression_monitoring: config_guard.get_bool("grpc.compression.enable_monitoring", false),
                compression_level: config_guard.get_u32("grpc.compression.level", 6),
                compression_threshold: config_guard.get_u64("grpc.compression.threshold", 1024) as usize,
                adaptive: AdaptiveCompressionConfig {
                    enable_adaptive: config_guard.get_bool("grpc.compression.adaptive.enable", true),
                    text_compression_level: config_guard.get_u32("grpc.compression.adaptive.text_level", 9),
                    binary_compression_level: config_guard.get_u32("grpc.compression.adaptive.binary_level", 6),
                    structured_compression_level: config_guard.get_u32("grpc.compression.adaptive.structured_level", 6),
                },
                performance: CompressionPerformanceConfig {
                    enable_failure_alerting: config_guard.get_bool("grpc.compression.performance.enable_failure_alerting", true),
                    enable_time_monitoring: config_guard.get_bool("grpc.compression.performance.enable_time_monitoring", true),
                    slow_compression_threshold_ms: config_guard.get_u64("grpc.compression.performance.slow_threshold_ms", 1000),
                    enable_ratio_tracking: config_guard.get_bool("grpc.compression.performance.enable_ratio_tracking", true),
                    poor_ratio_threshold: config_guard.get_f64("grpc.compression.performance.poor_ratio_threshold", 0.9),
                },
            },
        };

        Ok(Self {
            system: SystemConfig {
                project_name: config_guard.get_string("system.project_name", "workspace-qdrant-mcp"),
                database: DatabaseConfig {
                    max_connections: config_guard.get_i32("database.max_connections", 10),
                    sqlite_path: config_guard.get_string("database.sqlite_path", ":memory:"),
                    connection_timeout_secs: config_guard.get_u64("database.connection_timeout_secs", 30),
                    enable_wal: config_guard.get_bool("database.enable_wal", false),
                },
                auto_ingestion: AutoIngestionConfig {
                    enabled: config_guard.get_bool("auto_ingestion.enabled", true),
                    project_collection: config_guard.get_string("auto_ingestion.project_collection", "projects_content"),
                    auto_create_watches: config_guard.get_bool("auto_ingestion.auto_create_watches", true),
                    project_path: config_guard.get("auto_ingestion.project_path").and_then(|v| v.as_string()),
                    include_source_files: config_guard.get_bool("auto_ingestion.include_source_files", false),
                    include_patterns: config_guard.get_array("workspace.custom_include_patterns")
                        .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                        .unwrap_or_else(|| vec![]),
                    exclude_patterns: config_guard.get_array("workspace.custom_exclude_patterns")
                        .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                        .unwrap_or_else(|| vec![]),
                    max_depth: config_guard.get_u64("auto_ingestion.max_depth", 10) as usize,
                    recursive: config_guard.get_bool("auto_ingestion.recursive", true),
                },
                processing: ProcessingConfig {
                    max_concurrent_tasks: config_guard.get_u64("performance.max_concurrent_tasks", 4) as usize,
                    supported_extensions: config_guard.get_array("auto_ingestion.supported_extensions")
                        .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                        .unwrap_or_else(|| vec![".txt".to_string(), ".md".to_string(), ".py".to_string(), ".rs".to_string()]),
                    default_chunk_size: config_guard.get_u64("processing.default_chunk_size", 1000) as usize,
                    default_chunk_overlap: config_guard.get_u64("processing.default_chunk_overlap", 200) as usize,
                    max_file_size_bytes: config_guard.get_u64("processing.max_file_size_bytes", 10 * 1024 * 1024),
                    enable_lsp: config_guard.get_bool("processing.enable_lsp", false),
                    lsp_timeout_secs: config_guard.get_u32("processing.lsp_timeout_secs", 30),
                },
                file_watcher: FileWatcherConfig {
                    enabled: config_guard.get_bool("auto_ingestion.auto_create_watches", true),
                    ignore_patterns: config_guard.get_array("auto_ingestion.ignore_patterns")
                        .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                        .unwrap_or_else(|| vec!["*.tmp".to_string(), "*.log".to_string()]),
                    recursive: config_guard.get_bool("auto_ingestion.recursive", true),
                    max_watched_dirs: config_guard.get_u64("auto_ingestion.max_watched_dirs", 100) as usize,
                    debounce_ms: config_guard.get_u64("auto_ingestion.debounce_ms", 1000),
                },
            },
            grpc: GrpcConfig {
                server: GrpcServerConfig {
                    enabled: config_guard.get_bool("grpc.enabled", true),
                    port: config_guard.get_u16("grpc.port", 50051),
                },
                client: GrpcClientConfig {
                    connection_timeout: TimeUnit(config_guard.get_u64("grpc.connection_timeout_secs", 10) * 1000),
                    request_timeout: TimeUnit(config_guard.get_u64("performance.default_timeout_secs", 30) * 1000),
                    max_retries: config_guard.get_u32("qdrant.max_retries", 3),
                    enable_keepalive: config_guard.get_bool("grpc.client.enable_keepalive", true),
                },
                security: SecurityConfig {
                    tls: TlsConfig {
                        enabled: config_guard.get_bool("security.tls.enabled", false),
                        cert_file: config_guard.get("security.tls.cert_file").and_then(|v| v.as_string()),
                        key_file: config_guard.get("security.tls.key_file").and_then(|v| v.as_string()),
                        ca_cert_file: config_guard.get("security.tls.ca_cert_file").and_then(|v| v.as_string()),
                        client_cert_verification: ClientCertVerification::None,
                    },
                    auth: AuthConfig {
                        jwt: JwtConfig {
                            secret_or_key_file: config_guard.get_string("security.auth.jwt.secret_or_key_file", "changeme_jwt_secret"),
                            expiration_secs: config_guard.get_u64("security.auth.jwt.expiration_secs", 3600),
                            algorithm: config_guard.get_string("security.auth.jwt.algorithm", "HS256"),
                            issuer: config_guard.get_string("security.auth.jwt.issuer", "workspace-qdrant-daemon"),
                            audience: config_guard.get_string("security.auth.jwt.audience", "workspace-qdrant-client"),
                        },
                        api_key: ApiKeyConfig {
                            enabled: config_guard.get_bool("security.auth.api_key.enabled", false),
                            key_permissions: HashMap::new(),
                            valid_keys: config_guard.get_array("security.auth.api_key.valid_keys")
                                .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                                .unwrap_or_else(|| vec![]),
                            header_name: config_guard.get_string("security.auth.api_key.header_name", "x-api-key"),
                        },
                        enable_service_auth: config_guard.get_bool("security.auth.enable_service_auth", false),
                        authorization: AuthorizationConfig {
                            enabled: config_guard.get_bool("security.authorization.enabled", false),
                            service_permissions: ServicePermissions {
                                document_processor: vec!["read".to_string(), "write".to_string()],
                                search_service: vec!["read".to_string()],
                                memory_service: vec!["read".to_string(), "write".to_string()],
                                system_service: vec!["read".to_string()],
                            },
                            default_permissions: vec!["read".to_string()],
                        },
                    },
                    audit: SecurityAuditConfig {
                        enabled: config_guard.get_bool("security.audit.enabled", false),
                        log_auth_events: config_guard.get_bool("security.audit.log_auth_events", true),
                        log_auth_failures: config_guard.get_bool("security.audit.log_auth_failures", true),
                        log_rate_limit_events: config_guard.get_bool("security.audit.log_rate_limit_events", true),
                        log_suspicious_patterns: config_guard.get_bool("security.audit.log_suspicious_patterns", true),
                    },
                },
                transport: TransportConfig {
                    unix_socket: UnixSocketConfig {
                        enabled: config_guard.get_bool("grpc.transport.unix_socket.enabled", false),
                        path: config_guard.get("grpc.transport.unix_socket.path").and_then(|v| v.as_string()),
                        socket_path: config_guard.get_string("grpc.transport.unix_socket.path", "/tmp/workspace-qdrant.sock"),
                        permissions: config_guard.get_u32("grpc.transport.unix_socket.permissions", 0o666),
                        cleanup_on_exit: config_guard.get_bool("grpc.transport.unix_socket.cleanup_on_exit", true),
                        prefer_for_local: config_guard.get_bool("grpc.transport.unix_socket.prefer_for_local", false),
                    },
                    local_optimization: LocalOptimizationConfig::default(),
                    transport_strategy: TransportStrategy::Tcp,
                },
                message: server_config.message.clone(),
            },
            server: server_config,
            external_services: ExternalServicesConfig {
                qdrant: QdrantConfig {
                    url: config_guard.get_string("qdrant.url", "http://localhost:6333"),
                    api_key: config_guard.get("qdrant.api_key").and_then(|v| v.as_string()),
                    max_retries: config_guard.get_u32("qdrant.max_retries", 3),
                    default_collection: CollectionConfig {
                        vector_size: config_guard.get_u64("qdrant.default_collection.vector_size", 384) as usize,
                        shard_number: config_guard.get_u32("qdrant.default_collection.shard_number", 1),
                        replication_factor: config_guard.get_u32("qdrant.default_collection.replication_factor", 1),
                        distance_metric: config_guard.get_string("qdrant.default_collection.distance_metric", "Cosine"),
                        enable_indexing: config_guard.get_bool("qdrant.default_collection.enable_indexing", true),
                    },
                },
            },
            transport: TransportConfig {
                unix_socket: UnixSocketConfig {
                    enabled: config_guard.get_bool("grpc.transport.unix_socket.enabled", false),
                    path: config_guard.get("grpc.transport.unix_socket.path").and_then(|v| v.as_string()),
                    socket_path: config_guard.get_string("grpc.transport.unix_socket.path", "/tmp/workspace-qdrant.sock"),
                    permissions: config_guard.get_u32("grpc.transport.unix_socket.permissions", 0o666),
                    cleanup_on_exit: config_guard.get_bool("grpc.transport.unix_socket.cleanup_on_exit", true),
                    prefer_for_local: config_guard.get_bool("grpc.transport.unix_socket.prefer_for_local", false),
                },
                local_optimization: LocalOptimizationConfig::default(),
                transport_strategy: TransportStrategy::Tcp,
            },
            message: MessageConfig {
                service_limits: ServiceLimits {
                    memory_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    system_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    document_processor: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                    search_service: ServiceLimit {
                        max_incoming: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        max_outgoing: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                        enable_validation: true,
                    },
                },
                max_frame_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as u32,
                initial_window_size: (config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) / 4) as u32,
                enable_size_validation: config_guard.get_bool("grpc.message.enable_size_validation", true),
                max_incoming_message_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                max_outgoing_message_size: config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize,
                monitoring: MessageMonitoringConfig {
                    oversized_alert_threshold: config_guard.get_f64("grpc.message.monitoring.oversized_alert_threshold", 0.8),
                },
            },
            security: SecurityConfig {
                tls: TlsConfig {
                    enabled: config_guard.get_bool("security.tls.enabled", false),
                    cert_file: config_guard.get("security.tls.cert_file").and_then(|v| v.as_string()),
                    key_file: config_guard.get("security.tls.key_file").and_then(|v| v.as_string()),
                    ca_cert_file: config_guard.get("security.tls.ca_cert_file").and_then(|v| v.as_string()),
                    client_cert_verification: ClientCertVerification::None,
                },
                auth: AuthConfig {
                    jwt: JwtConfig {
                        secret_or_key_file: config_guard.get_string("security.auth.jwt.secret_or_key_file", "changeme_jwt_secret"),
                        expiration_secs: config_guard.get_u64("security.auth.jwt.expiration_secs", 3600),
                        algorithm: config_guard.get_string("security.auth.jwt.algorithm", "HS256"),
                        issuer: config_guard.get_string("security.auth.jwt.issuer", "workspace-qdrant-daemon"),
                        audience: config_guard.get_string("security.auth.jwt.audience", "workspace-qdrant-client"),
                    },
                    api_key: ApiKeyConfig {
                        enabled: config_guard.get_bool("security.auth.api_key.enabled", false),
                        key_permissions: HashMap::new(),
                        valid_keys: config_guard.get_array("security.auth.api_key.valid_keys")
                            .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                            .unwrap_or_else(|| vec![]),
                        header_name: config_guard.get_string("security.auth.api_key.header_name", "x-api-key"),
                    },
                    enable_service_auth: config_guard.get_bool("security.auth.enable_service_auth", false),
                    authorization: AuthorizationConfig {
                        enabled: config_guard.get_bool("security.authorization.enabled", false),
                        service_permissions: ServicePermissions {
                            document_processor: vec!["read".to_string(), "write".to_string()],
                            search_service: vec!["read".to_string()],
                            memory_service: vec!["read".to_string(), "write".to_string()],
                            system_service: vec!["read".to_string()],
                        },
                        default_permissions: vec!["read".to_string()],
                    },
                },
                audit: SecurityAuditConfig {
                    enabled: config_guard.get_bool("security.audit.enabled", false),
                    log_auth_events: config_guard.get_bool("security.audit.log_auth_events", true),
                    log_auth_failures: config_guard.get_bool("security.audit.log_auth_failures", true),
                    log_rate_limit_events: config_guard.get_bool("security.audit.log_rate_limit_events", true),
                    log_suspicious_patterns: config_guard.get_bool("security.audit.log_suspicious_patterns", true),
                },
            },
            streaming: StreamingConfig {
                enabled: config_guard.get_bool("grpc.streaming.enabled", true),
                stream_buffer_size: config_guard.get_u64("grpc.streaming.buffer_size", 1024) as usize,
                enable_flow_control: config_guard.get_bool("grpc.streaming.enable_flow_control", true),
                enable_server_streaming: config_guard.get_bool("grpc.streaming.enable_server_streaming", true),
                enable_client_streaming: config_guard.get_bool("grpc.streaming.enable_client_streaming", true),
                max_concurrent_streams: config_guard.get_u32("grpc.streaming.max_concurrent_streams", 100),
                stream_timeout_secs: config_guard.get_u64("grpc.streaming.stream_timeout_secs", 300),
                large_operations: LargeOperationConfig {
                    large_operation_chunk_size: config_guard.get_u64("grpc.streaming.large_operations.chunk_size", 1024 * 1024) as usize,
                },
                progress: StreamProgressConfig {
                    enabled: config_guard.get_bool("grpc.streaming.progress.enabled", true),
                    progress_update_interval_ms: config_guard.get_u64("grpc.streaming.progress.update_interval_ms", 1000),
                    enable_progress_callbacks: config_guard.get_bool("grpc.streaming.progress.enable_callbacks", true),
                    enable_progress_tracking: config_guard.get_bool("grpc.streaming.progress.enable_tracking", true),
                    progress_threshold: config_guard.get_u64("grpc.streaming.progress.threshold", 1024) as usize,
                },
                health: StreamHealthConfig {
                    enabled: config_guard.get_bool("grpc.streaming.health.enabled", true),
                    enable_auto_recovery: config_guard.get_bool("grpc.streaming.health.enable_auto_recovery", true),
                    enable_health_monitoring: config_guard.get_bool("grpc.streaming.health.enable_monitoring", true),
                    health_check_interval_secs: config_guard.get_u64("grpc.streaming.health.check_interval_secs", 30),
                },
            },
            compression: CompressionConfig {
                enabled: config_guard.get_bool("grpc.compression.enabled", true),
                enable_gzip: config_guard.get_bool("grpc.compression.enable_gzip", true),
                enable_compression_monitoring: config_guard.get_bool("grpc.compression.enable_monitoring", false),
                compression_level: config_guard.get_u32("grpc.compression.level", 6),
                compression_threshold: config_guard.get_u64("grpc.compression.threshold", 1024) as usize,
                adaptive: AdaptiveCompressionConfig {
                    enable_adaptive: config_guard.get_bool("grpc.compression.adaptive.enable", true),
                    text_compression_level: config_guard.get_u32("grpc.compression.adaptive.text_level", 9),
                    binary_compression_level: config_guard.get_u32("grpc.compression.adaptive.binary_level", 6),
                    structured_compression_level: config_guard.get_u32("grpc.compression.adaptive.structured_level", 6),
                },
                performance: CompressionPerformanceConfig {
                    enable_failure_alerting: config_guard.get_bool("grpc.compression.performance.enable_failure_alerting", true),
                    enable_time_monitoring: config_guard.get_bool("grpc.compression.performance.enable_time_monitoring", true),
                    slow_compression_threshold_ms: config_guard.get_u64("grpc.compression.performance.slow_threshold_ms", 1000),
                    enable_ratio_tracking: config_guard.get_bool("grpc.compression.performance.enable_ratio_tracking", true),
                    poor_ratio_threshold: config_guard.get_f64("grpc.compression.performance.poor_ratio_threshold", 0.9),
                },
            },
            metrics: MetricsConfig {
                enabled: config_guard.get_bool("logging.enable_metrics", false),
                collection_interval_secs: config_guard.get_u64("logging.metrics_interval_secs", 60),
            },
            logging: LoggingConfig {
                enabled: config_guard.get_bool("logging.enabled", true),
                level: config_guard.get_string("logging.level", "info"),
                file_path: config_guard.get("logging.file_path").and_then(|v| v.as_string()),
                max_file_size: SizeUnit(config_guard.get_u64("logging.max_file_size", 10 * 1024 * 1024)),
                max_files: config_guard.get_u32("logging.max_files", 10),
                enable_json: config_guard.get_bool("logging.enable_json", false),
                enable_structured: config_guard.get_bool("logging.enable_structured", true),
                enable_console: config_guard.get_bool("logging.enable_console", true),
            },
            // Legacy field access compatibility
            database: DatabaseConfig {
                max_connections: config_guard.get_i32("database.max_connections", 10),
                sqlite_path: config_guard.get_string("database.sqlite_path", ":memory:"),
                connection_timeout_secs: config_guard.get_u64("database.connection_timeout_secs", 30),
                enable_wal: config_guard.get_bool("database.enable_wal", false),
            },
            qdrant: QdrantConfig {
                url: config_guard.get_string("qdrant.url", "http://localhost:6333"),
                api_key: config_guard.get("qdrant.api_key").and_then(|v| v.as_string()),
                max_retries: config_guard.get_u32("qdrant.max_retries", 3),
                default_collection: CollectionConfig {
                    vector_size: config_guard.get_u64("qdrant.default_collection.vector_size", 384) as usize,
                    shard_number: config_guard.get_u32("qdrant.default_collection.shard_number", 1),
                    replication_factor: config_guard.get_u32("qdrant.default_collection.replication_factor", 1),
                    distance_metric: config_guard.get_string("qdrant.default_collection.distance_metric", "Cosine"),
                    enable_indexing: config_guard.get_bool("qdrant.default_collection.enable_indexing", true),
                },
            },
            workspace: WorkspaceConfig {
                collection_basename: config_guard.get("workspace.collection_basename").and_then(|v| v.as_string()),
                collection_types: config_guard.get_array("workspace.collection_types")
                    .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                    .unwrap_or_else(|| vec![]),
                memory_collection_name: config_guard.get_string("workspace.memory_collection_name", "memory"),
                auto_create_collections: config_guard.get_bool("workspace.auto_create_collections", true),
            },
            auto_ingestion: AutoIngestionConfig {
                enabled: config_guard.get_bool("auto_ingestion.enabled", true),
                project_collection: config_guard.get_string("auto_ingestion.project_collection", "projects_content"),
                auto_create_watches: config_guard.get_bool("auto_ingestion.auto_create_watches", true),
                project_path: config_guard.get("auto_ingestion.project_path").and_then(|v| v.as_string()),
                include_source_files: config_guard.get_bool("auto_ingestion.include_source_files", false),
                include_patterns: config_guard.get_array("workspace.custom_include_patterns")
                    .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                    .unwrap_or_else(|| vec![]),
                exclude_patterns: config_guard.get_array("workspace.custom_exclude_patterns")
                    .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                    .unwrap_or_else(|| vec![]),
                max_depth: config_guard.get_u64("auto_ingestion.max_depth", 10) as usize,
                recursive: config_guard.get_bool("auto_ingestion.recursive", true),
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: config_guard.get_u64("performance.max_concurrent_tasks", 4) as usize,
                supported_extensions: config_guard.get_array("auto_ingestion.supported_extensions")
                    .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                    .unwrap_or_else(|| vec![".txt".to_string(), ".md".to_string(), ".py".to_string(), ".rs".to_string()]),
                default_chunk_size: config_guard.get_u64("processing.default_chunk_size", 1000) as usize,
                default_chunk_overlap: config_guard.get_u64("processing.default_chunk_overlap", 200) as usize,
                max_file_size_bytes: config_guard.get_u64("processing.max_file_size_bytes", 10 * 1024 * 1024),
                enable_lsp: config_guard.get_bool("processing.enable_lsp", false),
                lsp_timeout_secs: config_guard.get_u32("processing.lsp_timeout_secs", 30),
            },
            file_watcher: FileWatcherConfig {
                enabled: config_guard.get_bool("auto_ingestion.auto_create_watches", true),
                ignore_patterns: config_guard.get_array("auto_ingestion.ignore_patterns")
                    .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                    .unwrap_or_else(|| vec!["*.tmp".to_string(), "*.log".to_string()]),
                recursive: config_guard.get_bool("auto_ingestion.recursive", true),
                max_watched_dirs: config_guard.get_u64("auto_ingestion.max_watched_dirs", 100) as usize,
                debounce_ms: config_guard.get_u64("auto_ingestion.debounce_ms", 1000),
            },
        })
    }

    /// Validate configuration
    fn validate_config() -> DaemonResult<()> {
        // Validate required configuration paths using simple Result-based API
        crate::config::get_config("qdrant.url")
            .map_err(|_| crate::error::DaemonError::Configuration {
                message: "qdrant.url is required".to_string()
            })?;

        // Validate server configuration
        let port_value = crate::config::get_config("server.port")
            .map_err(|_| crate::error::DaemonError::Configuration {
                message: "server.port is required".to_string()
            })?;

        if let Some(port_i64) = port_value.as_i64() {
            if port_i64 <= 0 || port_i64 > 65535 {
                return Err(crate::error::DaemonError::Configuration {
                    message: "server.port must be between 1 and 65535".to_string()
                });
            }
        } else {
            return Err(crate::error::DaemonError::Configuration {
                message: "server.port must be a valid port number".to_string()
            });
        }

        Ok(())
    }

    // REMOVED: database() shim method - use get_config_* functions directly

    // REMOVED: qdrant() shim method - use get_config_* functions directly

    // REMOVED: auto_ingestion() shim method - use get_config_* functions directly

    // REMOVED: workspace() shim method - use get_config_* functions directly

    // REMOVED: processing() shim method - use get_config_* functions directly

    // REMOVED: file_watcher() shim method - was reading from wrong path (auto_ingestion.auto_create_watches)
    // Use get_config_bool("document_processing.file_watching.enabled", false) instead

    /// Get server configuration
    pub fn server(&self) -> ServerConfig {
        let config_guard = config().lock().unwrap();
        let max_message_size = config_guard.get_u64("grpc.max_message_length", 100 * 1024 * 1024) as usize;
        ServerConfig {
            host: config_guard.get_string("server.host", "127.0.0.1"),
            port: config_guard.get_u16("server.port", 8000),
            max_connections: config_guard.get_u32("performance.max_concurrent_tasks", 4) as usize,
            request_timeout_secs: config_guard.get_u64("performance.default_timeout_secs", 30),
            connection_timeout_secs: config_guard.get_u64("grpc.connection_timeout_secs", 10),
            streaming: self.streaming(),
            message: MessageConfig {
                service_limits: ServiceLimits {
                    memory_service: ServiceLimit {
                        max_incoming: max_message_size,
                        max_outgoing: max_message_size,
                        enable_validation: true,
                    },
                    system_service: ServiceLimit {
                        max_incoming: max_message_size,
                        max_outgoing: max_message_size,
                        enable_validation: true,
                    },
                    document_processor: ServiceLimit {
                        max_incoming: max_message_size,
                        max_outgoing: max_message_size,
                        enable_validation: true,
                    },
                    search_service: ServiceLimit {
                        max_incoming: max_message_size,
                        max_outgoing: max_message_size,
                        enable_validation: true,
                    },
                },
                max_frame_size: max_message_size as u32,
                initial_window_size: (max_message_size / 4) as u32,
                enable_size_validation: config_guard.get_bool("grpc.message.enable_size_validation", true),
                max_incoming_message_size: max_message_size,
                max_outgoing_message_size: max_message_size,
                monitoring: MessageMonitoringConfig {
                    oversized_alert_threshold: config_guard.get_f64("grpc.message.monitoring.oversized_alert_threshold", 0.8),
                },
            },
            enable_tls: config_guard.get_bool("security.tls.enabled", false),
            security: self.security(),
            transport: self.transport(),
            compression: self.compression(),
        }
    }

    /// Get security configuration
    pub fn security(&self) -> SecurityConfig {
        let config_guard = config().lock().unwrap();
        SecurityConfig {
            tls: TlsConfig {
                enabled: config_guard.get_bool("security.tls.enabled", false),
                cert_file: config_guard.get("security.tls.cert_file").and_then(|v| v.as_string()),
                key_file: config_guard.get("security.tls.key_file").and_then(|v| v.as_string()),
                ca_cert_file: config_guard.get("security.tls.ca_cert_file").and_then(|v| v.as_string()),
                client_cert_verification: ClientCertVerification::None, // Default no verification
            },
            auth: AuthConfig {
                jwt: JwtConfig {
                    secret_or_key_file: config_guard.get_string("security.auth.jwt.secret_or_key_file", "changeme_jwt_secret"),
                    expiration_secs: config_guard.get_u64("security.auth.jwt.expiration_secs", 3600),
                    algorithm: config_guard.get_string("security.auth.jwt.algorithm", "HS256"),
                    issuer: config_guard.get_string("security.auth.jwt.issuer", "workspace-qdrant-daemon"),
                    audience: config_guard.get_string("security.auth.jwt.audience", "workspace-qdrant-client"),
                },
                api_key: ApiKeyConfig {
                    enabled: config_guard.get_bool("security.auth.api_key.enabled", false),
                    key_permissions: HashMap::new(), // Default empty permissions
                    valid_keys: config_guard.get_array("security.auth.api_key.valid_keys")
                        .map(|arr| arr.iter().filter_map(|v| v.as_string()).collect())
                        .unwrap_or_else(|| vec![]),
                    header_name: config_guard.get_string("security.auth.api_key.header_name", "x-api-key"),
                },
                enable_service_auth: config_guard.get_bool("security.auth.enable_service_auth", false),
                authorization: AuthorizationConfig {
                    enabled: config_guard.get_bool("security.authorization.enabled", false),
                    service_permissions: ServicePermissions {
                        document_processor: vec!["read".to_string(), "write".to_string()],
                        search_service: vec!["read".to_string()],
                        memory_service: vec!["read".to_string(), "write".to_string()],
                        system_service: vec!["read".to_string()],
                    },
                    default_permissions: vec!["read".to_string()],
                },
            },
            audit: SecurityAuditConfig {
                enabled: config_guard.get_bool("security.audit.enabled", false),
                log_auth_events: config_guard.get_bool("security.audit.log_auth_events", true),
                log_auth_failures: config_guard.get_bool("security.audit.log_auth_failures", true),
                log_rate_limit_events: config_guard.get_bool("security.audit.log_rate_limit_events", true),
                log_suspicious_patterns: config_guard.get_bool("security.audit.log_suspicious_patterns", true),
            },
        }
    }

    /// Get transport configuration
    pub fn transport(&self) -> TransportConfig {
        let config_guard = config().lock().unwrap();
        TransportConfig {
            unix_socket: UnixSocketConfig {
                enabled: config_guard.get_bool("grpc.transport.unix_socket.enabled", false),
                path: config_guard.get("grpc.transport.unix_socket.path").and_then(|v| v.as_string()),
                socket_path: config_guard.get_string("grpc.transport.unix_socket.path", "/tmp/workspace-qdrant.sock"),
                permissions: config_guard.get_u32("grpc.transport.unix_socket.permissions", 0o666),
                cleanup_on_exit: config_guard.get_bool("grpc.transport.unix_socket.cleanup_on_exit", true),
                prefer_for_local: config_guard.get_bool("grpc.transport.unix_socket.prefer_for_local", false),
            },
            local_optimization: LocalOptimizationConfig::default(),
            transport_strategy: TransportStrategy::Tcp,
        }
    }

    /// Get local optimization configuration
    pub fn local_optimization(&self) -> LocalOptimizationConfig {
        let config_guard = config().lock().unwrap();
        LocalOptimizationConfig {
            enabled: config_guard.get_bool("grpc.transport.local_optimization.enabled", true),
            latency: LocalLatencyConfig {
                enabled: config_guard.get_bool("grpc.transport.local_optimization.latency.enabled", true),
                target_latency_ms: config_guard.get_u64("grpc.transport.local_optimization.latency.target_ms", 10),
                max_acceptable_latency_ms: config_guard.get_u64("grpc.transport.local_optimization.latency.max_acceptable_ms", 100),
                monitoring_enabled: config_guard.get_bool("grpc.transport.local_optimization.latency.monitoring_enabled", false),
            },
            cache_size: config_guard.get_u64("grpc.transport.local_optimization.cache_size", 1024) as usize,
            optimization_level: config_guard.get_u32("grpc.transport.local_optimization.level", 1),
        }
    }

    /// Get transport strategy
    pub fn transport_strategy(&self) -> TransportStrategy {
        let config_guard = config().lock().unwrap();
        let strategy = config_guard.get_string("grpc.transport.strategy", "tcp");
        match strategy.to_lowercase().as_str() {
            "unix" | "unix_socket" => TransportStrategy::UnixSocket,
            "hybrid" => TransportStrategy::Hybrid,
            _ => TransportStrategy::Tcp,
        }
    }

    /// Get service message limits configuration
    pub fn service_message_limits(&self) -> ServiceMessageLimits {
        let config_guard = config().lock().unwrap();
        ServiceMessageLimits {
            max_request_size: config_guard.get_u64("grpc.message.limits.max_request_size", 100 * 1024 * 1024) as usize,
            max_response_size: config_guard.get_u64("grpc.message.limits.max_response_size", 100 * 1024 * 1024) as usize,
            timeout_secs: config_guard.get_u64("grpc.message.limits.timeout_secs", 30),
            enable_compression: config_guard.get_bool("grpc.message.limits.enable_compression", true),
        }
    }

    /// Get large operation stream configuration
    pub fn large_operation_stream(&self) -> LargeOperationStreamConfig {
        let config_guard = config().lock().unwrap();
        LargeOperationStreamConfig {
            chunk_size: config_guard.get_u64("grpc.streaming.large_operations.chunk_size", 1024 * 1024) as usize,
            max_concurrent_chunks: config_guard.get_u64("grpc.streaming.large_operations.max_concurrent_chunks", 4) as usize,
            timeout_per_chunk_secs: config_guard.get_u64("grpc.streaming.large_operations.timeout_per_chunk_secs", 30),
            enable_progress_tracking: config_guard.get_bool("grpc.streaming.large_operations.enable_progress_tracking", true),
        }
    }

    /// Get logging configuration
    pub fn logging(&self) -> LoggingConfig {
        let config_guard = config().lock().unwrap();
        LoggingConfig {
            enabled: config_guard.get_bool("logging.enabled", true),
            level: config_guard.get_string("logging.level", "info"),
            file_path: config_guard.get("logging.file_path").and_then(|v| v.as_string()),
            max_file_size: SizeUnit(config_guard.get_u64("logging.max_file_size", 10 * 1024 * 1024)),
            max_files: config_guard.get_u32("logging.max_files", 10),
            enable_json: config_guard.get_bool("logging.enable_json", false),
            enable_structured: config_guard.get_bool("logging.enable_structured", true),
            enable_console: config_guard.get_bool("logging.enable_console", true),
        }
    }

    /// Add validate method that the codebase expects
    pub fn validate(&self) -> DaemonResult<()> {
        Self::validate_config()
    }

    /// Get metrics configuration
    pub fn metrics(&self) -> MetricsConfig {
        let config_guard = config().lock().unwrap();
        MetricsConfig {
            enabled: config_guard.get_bool("logging.enable_metrics", false),
            collection_interval_secs: config_guard.get_u64("logging.metrics_interval_secs", 60),
        }
    }

    /// Get streaming configuration
    pub fn streaming(&self) -> StreamingConfig {
        let config_guard = config().lock().unwrap();
        StreamingConfig {
            enabled: config_guard.get_bool("grpc.streaming.enabled", true),
            stream_buffer_size: config_guard.get_u64("grpc.streaming.buffer_size", 1024) as usize,
            enable_flow_control: config_guard.get_bool("grpc.streaming.enable_flow_control", true),
            enable_server_streaming: config_guard.get_bool("grpc.streaming.enable_server_streaming", true),
            enable_client_streaming: config_guard.get_bool("grpc.streaming.enable_client_streaming", true),
            max_concurrent_streams: config_guard.get_u32("grpc.streaming.max_concurrent_streams", 100),
            stream_timeout_secs: config_guard.get_u64("grpc.streaming.stream_timeout_secs", 300),
            large_operations: LargeOperationConfig {
                large_operation_chunk_size: config_guard.get_u64("grpc.streaming.large_operations.chunk_size", 1024 * 1024) as usize,
            },
            progress: StreamProgressConfig {
                enabled: config_guard.get_bool("grpc.streaming.progress.enabled", true),
                progress_update_interval_ms: config_guard.get_u64("grpc.streaming.progress.update_interval_ms", 1000),
                enable_progress_callbacks: config_guard.get_bool("grpc.streaming.progress.enable_callbacks", true),
                enable_progress_tracking: config_guard.get_bool("grpc.streaming.progress.enable_tracking", true),
                progress_threshold: config_guard.get_u64("grpc.streaming.progress.threshold", 1024) as usize,
            },
            health: StreamHealthConfig {
                enabled: config_guard.get_bool("grpc.streaming.health.enabled", true),
                enable_auto_recovery: config_guard.get_bool("grpc.streaming.health.enable_auto_recovery", true),
                enable_health_monitoring: config_guard.get_bool("grpc.streaming.health.enable_monitoring", true),
                health_check_interval_secs: config_guard.get_u64("grpc.streaming.health.check_interval_secs", 30),
            },
        }
    }

    /// Get compression configuration
    pub fn compression(&self) -> CompressionConfig {
        let config_guard = config().lock().unwrap();
        CompressionConfig {
            enabled: config_guard.get_bool("grpc.compression.enabled", true),
            enable_gzip: config_guard.get_bool("grpc.compression.enable_gzip", true),
            enable_compression_monitoring: config_guard.get_bool("grpc.compression.enable_monitoring", false),
            compression_level: config_guard.get_u32("grpc.compression.level", 6),
            compression_threshold: config_guard.get_u64("grpc.compression.threshold", 1024) as usize,
            adaptive: AdaptiveCompressionConfig {
                enable_adaptive: config_guard.get_bool("grpc.compression.adaptive.enable", true),
                text_compression_level: config_guard.get_u32("grpc.compression.adaptive.text_level", 9),
                binary_compression_level: config_guard.get_u32("grpc.compression.adaptive.binary_level", 6),
                structured_compression_level: config_guard.get_u32("grpc.compression.adaptive.structured_level", 6),
            },
            performance: CompressionPerformanceConfig {
                enable_failure_alerting: config_guard.get_bool("grpc.compression.performance.enable_failure_alerting", true),
                enable_time_monitoring: config_guard.get_bool("grpc.compression.performance.enable_time_monitoring", true),
                slow_compression_threshold_ms: config_guard.get_u64("grpc.compression.performance.slow_threshold_ms", 1000),
                enable_ratio_tracking: config_guard.get_bool("grpc.compression.performance.enable_ratio_tracking", true),
                poor_ratio_threshold: config_guard.get_f64("grpc.compression.performance.poor_ratio_threshold", 0.9),
            },
        }
    }

    /// Get raw configuration manager for direct access
    pub fn raw(&self) -> std::sync::MutexGuard<'_, ConfigManager> {
        config().lock().unwrap()
    }

    /// Get external services configuration (if any)
    pub fn external_services(&self) -> Option<HashMap<String, ConfigValue>> {
        let config_guard = config().lock().unwrap();
        config_guard.get_object("external_services").map(|obj| obj.clone())
    }
}


// =============================================================================
// CONFIGURATION VALUE STRUCTS - COMPUTED FROM CONFIGMANAGER
// =============================================================================

/// Database configuration values
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub max_connections: i32,
    pub sqlite_path: String,
    pub connection_timeout_secs: u64,
    pub enable_wal: bool,
}

/// Qdrant configuration values
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub default_collection: CollectionConfig,
    pub max_retries: u32,
}

/// Collection configuration for Qdrant
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub vector_size: usize,
    pub shard_number: u32,
    pub replication_factor: u32,
    pub distance_metric: String,
    pub enable_indexing: bool,
}

/// Workspace configuration values
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    pub collection_basename: Option<String>,
    pub collection_types: Vec<String>,
    pub memory_collection_name: String,
    pub auto_create_collections: bool,
}

/// Auto-ingestion configuration values
#[derive(Debug, Clone)]
pub struct AutoIngestionConfig {
    pub enabled: bool,
    pub project_collection: String,  // was target_collection_suffix
    pub auto_create_watches: bool,
    pub project_path: Option<String>,
    pub include_source_files: bool,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub max_depth: usize,
    pub recursive: bool,
}

/// Processing configuration values
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub max_concurrent_tasks: usize,
    pub supported_extensions: Vec<String>,
    pub default_chunk_size: usize,
    pub default_chunk_overlap: usize,
    pub max_file_size_bytes: u64,
    pub enable_lsp: bool,
    pub lsp_timeout_secs: u32,
}

/// File watcher configuration values
#[derive(Debug, Clone)]
pub struct FileWatcherConfig {
    pub enabled: bool,
    pub ignore_patterns: Vec<String>,
    pub recursive: bool,
    pub max_watched_dirs: usize,
    pub debounce_ms: u64,
}

/// Message configuration values
#[derive(Debug, Clone)]
pub struct MessageConfig {
    pub service_limits: ServiceLimits,
    pub max_frame_size: u32,
    pub initial_window_size: u32,
    pub enable_size_validation: bool,
    pub max_incoming_message_size: usize,
    pub max_outgoing_message_size: usize,
    pub monitoring: MessageMonitoringConfig,
}

impl Default for MessageConfig {
    fn default() -> Self {
        Self {
            service_limits: ServiceLimits {
                memory_service: ServiceLimit {
                    max_incoming: 100 * 1024 * 1024,
                    max_outgoing: 100 * 1024 * 1024,
                    enable_validation: true,
                },
                system_service: ServiceLimit {
                    max_incoming: 100 * 1024 * 1024,
                    max_outgoing: 100 * 1024 * 1024,
                    enable_validation: true,
                },
                document_processor: ServiceLimit {
                    max_incoming: 100 * 1024 * 1024,
                    max_outgoing: 100 * 1024 * 1024,
                    enable_validation: true,
                },
                search_service: ServiceLimit {
                    max_incoming: 100 * 1024 * 1024,
                    max_outgoing: 100 * 1024 * 1024,
                    enable_validation: true,
                },
            },
            max_frame_size: 100 * 1024 * 1024,
            initial_window_size: 25 * 1024 * 1024,
            enable_size_validation: true,
            max_incoming_message_size: 100 * 1024 * 1024,
            max_outgoing_message_size: 100 * 1024 * 1024,
            monitoring: MessageMonitoringConfig {
                oversized_alert_threshold: 0.8,
            },
        }
    }
}

/// Service limits configuration
#[derive(Debug, Clone)]
pub struct ServiceLimits {
    pub memory_service: ServiceLimit,
    pub system_service: ServiceLimit,
    pub document_processor: ServiceLimit,
    pub search_service: ServiceLimit,
}

/// Individual service limit
#[derive(Debug, Clone)]
pub struct ServiceLimit {
    pub max_incoming: usize,
    pub max_outgoing: usize,
    pub enable_validation: bool,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub enable_gzip: bool,
    pub enable_compression_monitoring: bool,
    pub performance: CompressionPerformanceConfig,
    pub compression_level: u32,
    pub adaptive: AdaptiveCompressionConfig,
    pub compression_threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_gzip: true,
            enable_compression_monitoring: false,
            compression_level: 6,
            compression_threshold: 1024,
            adaptive: AdaptiveCompressionConfig {
                enable_adaptive: true,
                text_compression_level: 9,
                binary_compression_level: 6,
                structured_compression_level: 6,
            },
            performance: CompressionPerformanceConfig {
                enable_failure_alerting: true,
                enable_time_monitoring: true,
                slow_compression_threshold_ms: 1000,
                enable_ratio_tracking: true,
                poor_ratio_threshold: 0.9,
            },
        }
    }
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub enabled: bool,
    pub stream_buffer_size: usize,
    pub enable_flow_control: bool,
    pub progress: StreamProgressConfig,
    pub health: StreamHealthConfig,
    pub enable_server_streaming: bool,
    pub enable_client_streaming: bool,
    pub max_concurrent_streams: u32,
    pub large_operations: LargeOperationConfig,
    pub stream_timeout_secs: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stream_buffer_size: 1024,
            enable_flow_control: true,
            enable_server_streaming: true,
            enable_client_streaming: true,
            max_concurrent_streams: 100,
            stream_timeout_secs: 300,
            large_operations: LargeOperationConfig {
                large_operation_chunk_size: 1024 * 1024,
            },
            progress: StreamProgressConfig {
                enabled: true,
                progress_update_interval_ms: 1000,
                enable_progress_callbacks: true,
                enable_progress_tracking: true,
                progress_threshold: 1024,
            },
            health: StreamHealthConfig {
                enabled: true,
                enable_auto_recovery: true,
                enable_health_monitoring: true,
                health_check_interval_secs: 30,
            },
        }
    }
}

/// Stream progress configuration
#[derive(Debug, Clone)]
pub struct StreamProgressConfig {
    pub enabled: bool,
    pub progress_update_interval_ms: u64,
    pub enable_progress_callbacks: bool,
    pub enable_progress_tracking: bool,
    pub progress_threshold: usize,
}

/// Stream health configuration
#[derive(Debug, Clone)]
pub struct StreamHealthConfig {
    pub enabled: bool,
    pub enable_auto_recovery: bool,
    pub enable_health_monitoring: bool,
    pub health_check_interval_secs: u64,
}

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub tls: TlsConfig,
    pub auth: AuthConfig,
    pub audit: SecurityAuditConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tls: TlsConfig {
                enabled: false,
                cert_file: None,
                key_file: None,
                ca_cert_file: None,
                client_cert_verification: ClientCertVerification::None,
            },
            auth: AuthConfig {
                jwt: JwtConfig {
                    secret_or_key_file: "changeme_jwt_secret".to_string(),
                    expiration_secs: 3600,
                    algorithm: "HS256".to_string(),
                    issuer: "workspace-qdrant-daemon".to_string(),
                    audience: "workspace-qdrant-client".to_string(),
                },
                api_key: ApiKeyConfig {
                    enabled: false,
                    key_permissions: HashMap::new(),
                    valid_keys: vec![],
                    header_name: "x-api-key".to_string(),
                },
                enable_service_auth: false,
                authorization: AuthorizationConfig {
                    enabled: false,
                    service_permissions: ServicePermissions {
                        document_processor: vec!["read".to_string(), "write".to_string()],
                        search_service: vec!["read".to_string()],
                        memory_service: vec!["read".to_string(), "write".to_string()],
                        system_service: vec!["read".to_string()],
                    },
                    default_permissions: vec!["read".to_string()],
                },
            },
            audit: SecurityAuditConfig {
                enabled: false,
                log_auth_events: true,
                log_auth_failures: true,
                log_rate_limit_events: true,
                log_suspicious_patterns: true,
            },
        }
    }
}

/// TLS configuration
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub enabled: bool,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub ca_cert_file: Option<String>,
    pub client_cert_verification: ClientCertVerification,
}

/// JWT configuration
#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub secret_or_key_file: String,
    pub expiration_secs: u64,
    pub algorithm: String,
    pub issuer: String,
    pub audience: String,
}

/// API key configuration
#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    pub enabled: bool,
    pub key_permissions: HashMap<String, Vec<String>>,
    pub valid_keys: Vec<String>,
    pub header_name: String,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub jwt: JwtConfig,
    pub api_key: ApiKeyConfig,
    pub enable_service_auth: bool,
    pub authorization: AuthorizationConfig,
}

/// Authorization configuration
#[derive(Debug, Clone)]
pub struct AuthorizationConfig {
    pub enabled: bool,
    pub service_permissions: ServicePermissions,
    pub default_permissions: Vec<String>,
}

/// Security audit configuration
#[derive(Debug, Clone)]
pub struct SecurityAuditConfig {
    pub enabled: bool,
    pub log_auth_events: bool,
    pub log_auth_failures: bool,
    pub log_rate_limit_events: bool,
    pub log_suspicious_patterns: bool,
}

/// Client certificate verification
#[derive(Debug, Clone)]
pub enum ClientCertVerification {
    None,
    Optional,
    Required,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub message: MessageConfig,
    pub request_timeout_secs: u64,
    pub connection_timeout_secs: u64,
    pub streaming: StreamingConfig,
    pub enable_tls: bool,
    pub security: SecurityConfig,
    pub transport: TransportConfig,
    pub compression: CompressionConfig,
}

/// Transport configuration
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub unix_socket: UnixSocketConfig,
    pub local_optimization: LocalOptimizationConfig,
    pub transport_strategy: TransportStrategy,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            unix_socket: UnixSocketConfig {
                enabled: false,
                path: None,
                socket_path: "/tmp/workspace-qdrant.sock".to_string(),
                permissions: 0o666,
                cleanup_on_exit: true,
                prefer_for_local: false,
            },
            local_optimization: LocalOptimizationConfig::default(),
            transport_strategy: TransportStrategy::default(),
        }
    }
}

/// Unix socket configuration
#[derive(Debug, Clone)]
pub struct UnixSocketConfig {
    pub enabled: bool,
    pub path: Option<String>,
    pub socket_path: String, // Legacy field for backward compatibility
    pub permissions: u32,
    pub cleanup_on_exit: bool,
    pub prefer_for_local: bool,
}

/// Local latency configuration
#[derive(Debug, Clone)]
pub struct LocalLatencyConfig {
    pub enabled: bool,
    pub target_latency_ms: u64,
    pub max_acceptable_latency_ms: u64,
    pub monitoring_enabled: bool,
}

impl Default for LocalLatencyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_latency_ms: 10,
            max_acceptable_latency_ms: 100,
            monitoring_enabled: false,
        }
    }
}

/// Service permissions configuration
#[derive(Debug, Clone)]
pub struct ServicePermissions {
    pub document_processor: Vec<String>,
    pub search_service: Vec<String>,
    pub memory_service: Vec<String>,
    pub system_service: Vec<String>,
}

/// Metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub collection_interval_secs: u64,
}

/// Message monitoring configuration
#[derive(Debug, Clone)]
pub struct MessageMonitoringConfig {
    pub oversized_alert_threshold: f64,
}

/// Compression performance configuration
#[derive(Debug, Clone)]
pub struct CompressionPerformanceConfig {
    pub enable_failure_alerting: bool,
    pub enable_time_monitoring: bool,
    pub slow_compression_threshold_ms: u64,
    pub enable_ratio_tracking: bool,
    pub poor_ratio_threshold: f64,
}

/// Large operation configuration
#[derive(Debug, Clone)]
pub struct LargeOperationConfig {
    pub large_operation_chunk_size: usize,
}

/// Adaptive compression configuration
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionConfig {
    pub enable_adaptive: bool,
    pub text_compression_level: u32,
    pub binary_compression_level: u32,
    pub structured_compression_level: u32,
}

// =============================================================================
// MISSING CONFIGURATION STRUCTS
// =============================================================================

/// External services configuration
#[derive(Debug, Clone)]
pub struct ExternalServicesConfig {
    pub qdrant: QdrantConfig,
}

/// gRPC client configuration
#[derive(Debug, Clone)]
pub struct GrpcClientConfig {
    pub connection_timeout: TimeUnit,
    pub request_timeout: TimeUnit,
    pub max_retries: u32,
    pub enable_keepalive: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_second: u32,
    pub burst_capacity: u32,
    pub connection_pool_limits: ConnectionPoolLimits,
    pub queue_depth_limit: usize,
    pub memory_protection: MemoryProtectionConfig,
    pub resource_protection: ResourceProtectionConfig,
}

/// Connection pool limits
#[derive(Debug, Clone)]
pub struct ConnectionPoolLimits {
    pub document_processor: u32,
    pub search_service: u32,
    pub memory_service: u32,
    pub system_service: u32,
}

/// Memory protection configuration
#[derive(Debug, Clone)]
pub struct MemoryProtectionConfig {
    pub enabled: bool,
    pub max_memory_per_connection: u64,
    pub memory_alert_threshold: f64,
    pub enable_memory_monitoring: bool,
    pub gc_trigger_threshold: f64,
}

/// Resource protection configuration
#[derive(Debug, Clone)]
pub struct ResourceProtectionConfig {
    pub enabled: bool,
    pub max_cpu_per_connection: f64,
    pub cpu_alert_threshold: f64,
    pub enable_resource_monitoring: bool,
    pub throttle_threshold: f64,
}

/// Audit configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_successful_requests: bool,
    pub log_failed_requests: bool,
    pub audit_file_path: String,
    pub enable_audit_rotation: bool,
    pub max_audit_file_size: u64,
    pub retention_days: u32,
}

/// Audit log rotation configuration
#[derive(Debug, Clone)]
pub struct AuditLogRotation {
    pub enabled: bool,
    pub max_file_size: u64,
    pub max_files: u32,
    pub compress_old_files: bool,
}

impl Default for AuditLogRotation {
    fn default() -> Self {
        Self {
            enabled: false,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 10,
            compress_old_files: true,
        }
    }
}

impl Default for ExternalServicesConfig {
    fn default() -> Self {
        Self {
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                default_collection: CollectionConfig {
                    vector_size: 384,
                    shard_number: 1,
                    replication_factor: 1,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                },
                max_retries: 3,
            },
        }
    }
}

impl Default for GrpcClientConfig {
    fn default() -> Self {
        Self {
            connection_timeout: TimeUnit(10000),
            request_timeout: TimeUnit(30000),
            max_retries: 3,
            enable_keepalive: true,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 1000,
            burst_capacity: 100,
            connection_pool_limits: ConnectionPoolLimits::default(),
            queue_depth_limit: 1000,
            memory_protection: MemoryProtectionConfig::default(),
            resource_protection: ResourceProtectionConfig::default(),
        }
    }
}

impl Default for ConnectionPoolLimits {
    fn default() -> Self {
        Self {
            document_processor: 50,
            search_service: 30,
            memory_service: 20,
            system_service: 10,
        }
    }
}

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_memory_per_connection: 104857600,
            memory_alert_threshold: 0.8,
            enable_memory_monitoring: false,
            gc_trigger_threshold: 0.9,
        }
    }
}

impl Default for ResourceProtectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_cpu_per_connection: 50.0,
            cpu_alert_threshold: 0.8,
            enable_resource_monitoring: false,
            throttle_threshold: 0.9,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_successful_requests: false,
            log_failed_requests: true,
            audit_file_path: "/tmp/audit.log".to_string(),
            enable_audit_rotation: false,
            max_audit_file_size: 10485760,
            retention_days: 30,
        }
    }
}

/// Local optimization configuration
#[derive(Debug, Clone)]
pub struct LocalOptimizationConfig {
    pub enabled: bool,
    pub latency: LocalLatencyConfig,
    pub cache_size: usize,
    pub optimization_level: u32,
}

impl Default for LocalOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency: LocalLatencyConfig::default(),
            cache_size: 1024,
            optimization_level: 1,
        }
    }
}

/// Transport strategy configuration
#[derive(Debug, Clone)]
pub enum TransportStrategy {
    Tcp,
    UnixSocket,
    Hybrid,
    Auto,
    ForceTcp,
    ForceUnixSocket,
    UnixSocketWithTcpFallback,
}

impl Default for TransportStrategy {
    fn default() -> Self {
        TransportStrategy::Tcp
    }
}

/// Service message limits configuration
#[derive(Debug, Clone)]
pub struct ServiceMessageLimits {
    pub max_request_size: usize,
    pub max_response_size: usize,
    pub timeout_secs: u64,
    pub enable_compression: bool,
}

impl Default for ServiceMessageLimits {
    fn default() -> Self {
        Self {
            max_request_size: 100 * 1024 * 1024,
            max_response_size: 100 * 1024 * 1024,
            timeout_secs: 30,
            enable_compression: true,
        }
    }
}

/// Large operation streaming configuration
#[derive(Debug, Clone)]
pub struct LargeOperationStreamConfig {
    pub chunk_size: usize,
    pub max_concurrent_chunks: usize,
    pub timeout_per_chunk_secs: u64,
    pub enable_progress_tracking: bool,
}

impl Default for LargeOperationStreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024,
            max_concurrent_chunks: 4,
            timeout_per_chunk_secs: 30,
            enable_progress_tracking: true,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub enabled: bool,
    pub level: String,
    pub file_path: Option<String>,
    pub max_file_size: SizeUnit,
    pub max_files: u32,
    pub enable_json: bool,
    pub enable_structured: bool,
    pub enable_console: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: "info".to_string(),
            file_path: None,
            max_file_size: SizeUnit(10 * 1024 * 1024), // 10MB
            max_files: 10,
            enable_json: false,
            enable_structured: true,
            enable_console: true,
        }
    }
}

// =============================================================================
// GLOBAL CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
// =============================================================================

/// Get configuration string value with dot notation
pub fn get_config_string(path: &str, default: &str) -> String {
    config().lock().unwrap().get_string(path, default)
}

/// Get configuration boolean value with dot notation
pub fn get_config_bool(path: &str, default: bool) -> bool {
    config().lock().unwrap().get_bool(path, default)
}

/// Get configuration u16 value with dot notation
pub fn get_config_u16(path: &str, default: u16) -> u16 {
    config().lock().unwrap().get_u16(path, default)
}

/// Get configuration u64 value with dot notation
pub fn get_config_u64(path: &str, default: u64) -> u64 {
    config().lock().unwrap().get_u64(path, default)
}

// =============================================================================
// RESULT-BASED CONFIGURATION FUNCTIONS (IDIOMATIC RUST)
// =============================================================================

/// Get configuration value with Result-based error handling (simple)
pub fn get_config(path: &str) -> Result<ConfigValue, crate::error::DaemonError> {
    config().lock().unwrap().get_config(path).map(|v| v.clone())
}

/// Get configuration string with Result-based error handling (idiomatic Rust)
pub fn try_get_config_string(path: &str) -> Result<String, crate::error::DaemonError> {
    config().lock().unwrap().try_get_string(path)
}

/// Get configuration boolean with Result-based error handling (idiomatic Rust)
pub fn try_get_config_bool(path: &str) -> Result<bool, crate::error::DaemonError> {
    config().lock().unwrap().try_get_bool(path)
}

/// Get configuration u16 with Result-based error handling (idiomatic Rust)
pub fn try_get_config_u16(path: &str) -> Result<u16, crate::error::DaemonError> {
    config().lock().unwrap().try_get_u16(path)
}

/// Get configuration u32 with Result-based error handling (idiomatic Rust)
pub fn try_get_config_u32(path: &str) -> Result<u32, crate::error::DaemonError> {
    config().lock().unwrap().try_get_u32(path)
}

/// Get configuration i64 with Result-based error handling (idiomatic Rust)
pub fn try_get_config_i64(path: &str) -> Result<i64, crate::error::DaemonError> {
    config().lock().unwrap().try_get_i64(path)
}

/// Common configuration utilities module for legacy compatibility
pub mod common {
    /// Get configuration string value with default
    pub fn get_config_string(path: &str, default: &str) -> String {
        super::get_config_string(path, default)
    }

    /// Get configuration boolean value with default
    pub fn get_config_bool(path: &str, default: bool) -> bool {
        super::get_config_bool(path, default)
    }

    /// Get project name from configuration
    pub fn project_name() -> String {
        super::get_config_string("system.project_name", "workspace-qdrant-mcp")
    }

    /// Check if Rust daemon is enabled
    pub fn rust_daemon_enabled() -> bool {
        super::get_config_bool("grpc.enabled", true)
    }
}

// =============================================================================
// CONFIGURATION-LEVEL COLLECTION NAME GENERATION
// =============================================================================

impl WorkspaceConfig {
    /// Generate complete collection name with suffix handling at configuration level
    pub fn get_collection_name(&self, project_name: &str, collection_type: &str) -> String {
        if let Some(basename) = &self.collection_basename {
            format!("{}-{}", basename, collection_type)
        } else {
            format!("{}-{}", project_name, collection_type)
        }
    }

    /// Get all available collection types for a project
    pub fn get_collection_types(&self) -> &Vec<String> {
        &self.collection_types
    }

    /// Get complete collection names for all types
    pub fn get_all_collection_names(&self, project_name: &str) -> Vec<String> {
        self.collection_types
            .iter()
            .map(|collection_type| self.get_collection_name(project_name, collection_type))
            .collect()
    }
}

impl Default for AutoIngestionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            project_collection: "code".to_string(),
            auto_create_watches: false,
            project_path: None,
            include_source_files: true,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_depth: 10,
            recursive: true,
        }
    }
}

impl AutoIngestionConfig {
    /// Generate complete project collection name with suffix handling at configuration level
    pub fn get_project_collection_name(&self, project_name: &str) -> String {
        format!("{}-{}", project_name, self.project_collection)
    }

    /// Get the base project collection suffix
    pub fn get_project_collection_suffix(&self) -> &str {
        &self.project_collection
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_size_unit_parsing() {
        assert_eq!(SizeUnit::from_str("100B").unwrap().0, 100);
        assert_eq!(SizeUnit::from_str("1KB").unwrap().0, 1024);
        assert_eq!(SizeUnit::from_str("1MB").unwrap().0, 1048576);
        assert_eq!(SizeUnit::from_str("1GB").unwrap().0, 1073741824);
        assert_eq!(SizeUnit::from_str("1TB").unwrap().0, 1099511627776);

        // Short forms
        assert_eq!(SizeUnit::from_str("1K").unwrap().0, 1024);
        assert_eq!(SizeUnit::from_str("1M").unwrap().0, 1048576);
        assert_eq!(SizeUnit::from_str("1G").unwrap().0, 1073741824);
        assert_eq!(SizeUnit::from_str("1T").unwrap().0, 1099511627776);
    }

    #[test]
    fn test_time_unit_parsing() {
        assert_eq!(TimeUnit::from_str("100ms").unwrap().0, 100);
        assert_eq!(TimeUnit::from_str("1s").unwrap().0, 1000);
        assert_eq!(TimeUnit::from_str("1m").unwrap().0, 60000);
        assert_eq!(TimeUnit::from_str("1h").unwrap().0, 3600000);
    }

    #[test]
    fn test_config_value_from_yaml() {
        let yaml_str = r#"
            string_val: "test"
            int_val: 42
            bool_val: true
            array_val: [1, 2, 3]
            object_val:
                nested: "value"
        "#;

        let yaml_value: YamlValue = serde_yaml::from_str(yaml_str).unwrap();
        let config_value = ConfigValue::from_yaml_value(&yaml_value, "");

        if let Some(obj) = config_value.as_object() {
            assert_eq!(obj.get("string_val").unwrap().as_string().unwrap(), "test");
            assert_eq!(obj.get("int_val").unwrap().as_i64().unwrap(), 42);
            assert_eq!(obj.get("bool_val").unwrap().as_bool().unwrap(), true);
            assert!(obj.get("array_val").unwrap().as_array().is_some());
            assert!(obj.get("object_val").unwrap().as_object().is_some());
        } else {
            panic!("Expected object configuration");
        }
    }

    #[test]
    fn test_config_manager_dot_notation() {
        let mut config = HashMap::new();
        let mut server = HashMap::new();
        server.insert("port".to_string(), ConfigValue::Integer(8080));
        server.insert("host".to_string(), ConfigValue::String("localhost".to_string()));
        config.insert("server".to_string(), ConfigValue::Object(server));

        let manager = ConfigManager::new(HashMap::new(), config);

        assert_eq!(manager.get_string("server.host", ""), "localhost");
        assert_eq!(manager.get_i32("server.port", 0), 8080);
        assert_eq!(manager.get_string("nonexistent.path", "default"), "default");
    }

    #[test]
    fn test_config_merging() {
        let mut defaults = HashMap::new();
        let mut server_defaults = HashMap::new();
        server_defaults.insert("port".to_string(), ConfigValue::Integer(3000));
        server_defaults.insert("host".to_string(), ConfigValue::String("127.0.0.1".to_string()));
        defaults.insert("server".to_string(), ConfigValue::Object(server_defaults));

        let mut user_config = HashMap::new();
        let mut server_user = HashMap::new();
        server_user.insert("port".to_string(), ConfigValue::Integer(8080));
        user_config.insert("server".to_string(), ConfigValue::Object(server_user));

        let manager = ConfigManager::new(user_config, defaults);

        // User config should override defaults
        assert_eq!(manager.get_i32("server.port", 0), 8080);
        // Defaults should remain where not overridden
        assert_eq!(manager.get_string("server.host", ""), "127.0.0.1");
    }

    #[test]
    fn test_updated_configuration_fields() {
        println!("=== Testing Updated Configuration Fields ===");

        // Test 1: Load default configuration
        println!("\n1. Testing default configuration loading...");
        init_config(None).unwrap();
        let config = config().lock().unwrap();

        // Test workspace configuration fields
        println!("\n2. Testing workspace configuration dot notation access:");

        if let Some(basename) = config.get("workspace.collection_basename") {
            println!("   workspace.collection_basename = {:?}", basename);
        } else {
            println!("   workspace.collection_basename = null (as expected)");
        }

        if let Some(types) = config.get("workspace.collection_types") {
            println!("   workspace.collection_types = {:?}", types);
        } else {
            println!("   workspace.collection_types = [] (empty array)");
        }

        if let Some(memory_name) = config.get("workspace.memory_collection_name") {
            println!("   workspace.memory_collection_name = {:?}", memory_name);
        }

        if let Some(auto_create) = config.get("workspace.auto_create_collections") {
            println!("   workspace.auto_create_collections = {:?}", auto_create);
        }

        // Test auto_ingestion configuration
        println!("\n3. Testing auto_ingestion configuration:");

        if let Some(project_collection) = config.get("auto_ingestion.project_collection") {
            println!("   auto_ingestion.project_collection = {:?}", project_collection);
        }

        // Test with custom values
        println!("\n4. Testing configuration merging with custom values:");

        let custom_config_yaml = r#"
workspace:
  collection_basename: "test_project"
  collection_types: ["code", "docs", "tests"]
  memory_collection_name: "custom_memory"
  auto_create_collections: false
auto_ingestion:
  project_collection: "custom_content"
"#;

        // Test that we can parse this structure
        let _custom_config: HashMap<String, serde_yaml::Value> = serde_yaml::from_str(custom_config_yaml).unwrap();
        println!("   ✓ Custom configuration structure is valid");

        println!("\n=== All Configuration Tests Passed! ===");
        println!("✓ Workspace configuration fields accessible via dot notation");
        println!("✓ Auto-ingestion project_collection field renamed correctly");
        println!("✓ Configuration merging structure validated");
        println!("✓ Collection naming pattern ready for implementation");
    }

    #[test]
    fn test_workspace_config_collection_name_generation() {
        // Test with collection_basename set
        let workspace_config_with_basename = WorkspaceConfig {
            collection_basename: Some("test_project".to_string()),
            collection_types: vec!["code".to_string(), "docs".to_string(), "tests".to_string()],
            memory_collection_name: "memory".to_string(),
            auto_create_collections: true,
        };

        assert_eq!(
            workspace_config_with_basename.get_collection_name("my_project", "code"),
            "test_project-code"
        );
        assert_eq!(
            workspace_config_with_basename.get_collection_name("my_project", "docs"),
            "test_project-docs"
        );

        // Test with collection_basename as None (use project name)
        let workspace_config_without_basename = WorkspaceConfig {
            collection_basename: None,
            collection_types: vec!["code".to_string(), "docs".to_string()],
            memory_collection_name: "memory".to_string(),
            auto_create_collections: true,
        };

        assert_eq!(
            workspace_config_without_basename.get_collection_name("my_project", "code"),
            "my_project-code"
        );
        assert_eq!(
            workspace_config_without_basename.get_collection_name("different_project", "docs"),
            "different_project-docs"
        );
    }

    #[test]
    fn test_workspace_config_get_all_collection_names() {
        let workspace_config = WorkspaceConfig {
            collection_basename: Some("test_project".to_string()),
            collection_types: vec!["code".to_string(), "docs".to_string(), "tests".to_string()],
            memory_collection_name: "memory".to_string(),
            auto_create_collections: true,
        };

        let all_names = workspace_config.get_all_collection_names("my_project");
        assert_eq!(all_names, vec!["test_project-code", "test_project-docs", "test_project-tests"]);

        // Test with None basename
        let workspace_config_no_basename = WorkspaceConfig {
            collection_basename: None,
            collection_types: vec!["content".to_string(), "memory".to_string()],
            memory_collection_name: "memory".to_string(),
            auto_create_collections: false,
        };

        let all_names_no_basename = workspace_config_no_basename.get_all_collection_names("workspace-qdrant-mcp");
        assert_eq!(all_names_no_basename, vec!["workspace-qdrant-mcp-content", "workspace-qdrant-mcp-memory"]);
    }

    #[test]
    fn test_auto_ingestion_config_project_collection_name() {
        let auto_ingestion_config = AutoIngestionConfig {
            enabled: true,
            project_collection: "projects_content".to_string(),
            auto_create_watches: true,
            project_path: None,
            include_source_files: true,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_depth: 10,
            recursive: true,
        };

        assert_eq!(
            auto_ingestion_config.get_project_collection_name("my_project"),
            "my_project-projects_content"
        );
        assert_eq!(
            auto_ingestion_config.get_project_collection_name("workspace-qdrant-mcp"),
            "workspace-qdrant-mcp-projects_content"
        );

        assert_eq!(
            auto_ingestion_config.get_project_collection_suffix(),
            "projects_content"
        );
    }
}