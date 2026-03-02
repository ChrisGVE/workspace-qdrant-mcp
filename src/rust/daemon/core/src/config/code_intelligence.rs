//! Code intelligence configuration (LSP and Tree-sitter grammars)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_true() -> bool { true }

// --- LSP Settings ---

fn default_max_servers_per_project() -> usize { 3 }
fn default_deactivation_delay() -> u64 { 60 }
fn default_cache_ttl() -> u64 { 300 }
fn default_startup_timeout() -> u64 { 30 }
fn default_request_timeout() -> u64 { 10 }
fn default_health_check_interval() -> u64 { 60 }
fn default_max_restart_attempts() -> u32 { 3 }
fn default_backoff_multiplier() -> f64 { 2.0 }
fn default_stability_reset() -> u64 { 3600 }

/// LSP (Language Server Protocol) configuration settings
///
/// These settings configure the daemon's LSP integration for code intelligence
/// features. LSP servers are started for active projects and provide enrichment
/// data like references, type information, and import resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspSettings {
    /// User PATH for finding language servers
    /// CLI stores user's PATH here so daemon can find servers
    #[serde(default)]
    pub user_path: Option<String>,

    /// Maximum number of LSP servers per active project
    #[serde(default = "default_max_servers_per_project")]
    pub max_servers_per_project: usize,

    /// Auto-start LSP servers when project becomes active
    #[serde(default = "default_true")]
    pub auto_start_on_activation: bool,

    /// Delay in seconds before stopping servers after project deactivation
    #[serde(default = "default_deactivation_delay")]
    pub deactivation_delay_secs: u64,

    /// Enable caching of LSP enrichment query results
    #[serde(default = "default_true")]
    pub enable_enrichment_cache: bool,

    /// TTL in seconds for cached enrichment data
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl_secs: u64,

    /// Timeout in seconds for LSP server startup
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Timeout in seconds for individual LSP requests
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Interval in seconds between health checks
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval_secs: u64,

    /// Maximum restart attempts before marking server unavailable
    #[serde(default = "default_max_restart_attempts")]
    pub max_restart_attempts: u32,

    /// Backoff multiplier for restart delays
    #[serde(default = "default_backoff_multiplier")]
    pub restart_backoff_multiplier: f64,

    /// Enable auto-restart of failed servers
    #[serde(default = "default_true")]
    pub enable_auto_restart: bool,

    /// Stability period in seconds before resetting restart count
    #[serde(default = "default_stability_reset")]
    pub stability_reset_secs: u64,
}

impl Default for LspSettings {
    fn default() -> Self {
        Self {
            user_path: None,
            max_servers_per_project: default_max_servers_per_project(),
            auto_start_on_activation: default_true(),
            deactivation_delay_secs: default_deactivation_delay(),
            enable_enrichment_cache: default_true(),
            cache_ttl_secs: default_cache_ttl(),
            startup_timeout_secs: default_startup_timeout(),
            request_timeout_secs: default_request_timeout(),
            health_check_interval_secs: default_health_check_interval(),
            max_restart_attempts: default_max_restart_attempts(),
            restart_backoff_multiplier: default_backoff_multiplier(),
            enable_auto_restart: default_true(),
            stability_reset_secs: default_stability_reset(),
        }
    }
}

impl LspSettings {
    /// Validate LSP configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.max_servers_per_project == 0 {
            return Err("max_servers_per_project must be greater than 0".to_string());
        }
        if self.cache_ttl_secs == 0 {
            return Err("cache_ttl_secs must be greater than 0".to_string());
        }
        if self.startup_timeout_secs == 0 {
            return Err("startup_timeout_secs must be greater than 0".to_string());
        }
        if self.request_timeout_secs == 0 {
            return Err("request_timeout_secs must be greater than 0".to_string());
        }
        if self.health_check_interval_secs == 0 {
            return Err("health_check_interval_secs must be greater than 0".to_string());
        }
        if self.restart_backoff_multiplier < 1.0 {
            return Err("restart_backoff_multiplier must be >= 1.0".to_string());
        }
        Ok(())
    }
}

// --- Grammar Config ---

fn default_grammar_check_interval() -> u32 { 168 } // Weekly
fn default_idle_update_check_delay() -> u64 { 300 } // 5 minutes
fn default_grammar_cache_dir() -> PathBuf {
    PathBuf::from("~/.workspace-qdrant/grammars")
}
fn default_required_grammars() -> Vec<String> {
    vec![]
}
fn default_tree_sitter_version() -> String { "0.24".to_string() }
fn default_download_base_url() -> String {
    "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}".to_string()
}

/// Tree-sitter grammar configuration for dynamic grammar loading
///
/// These settings configure how tree-sitter grammars are loaded at runtime.
/// Dynamic loading allows adding language support without recompiling the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarConfig {
    /// Directory for storing grammar shared libraries
    /// Default: ~/.workspace-qdrant/grammars
    #[serde(default = "default_grammar_cache_dir")]
    pub cache_dir: PathBuf,

    /// List of language grammars to pre-download on daemon startup
    /// Default: empty (grammars downloaded on first use via auto_download)
    #[serde(default = "default_required_grammars")]
    pub required: Vec<String>,

    /// Auto-download missing grammars from download sources
    #[serde(default = "default_true")]
    pub auto_download: bool,

    /// Expected tree-sitter ABI version for grammar compatibility
    #[serde(default = "default_tree_sitter_version")]
    pub tree_sitter_version: String,

    /// Base URL template for grammar downloads
    /// Supports {language}, {version}, {platform}, {ext} placeholders
    #[serde(default = "default_download_base_url")]
    pub download_base_url: String,

    /// Verify checksums of downloaded grammars
    #[serde(default = "default_true")]
    pub verify_checksums: bool,

    /// Load grammars on first use instead of at startup
    #[serde(default = "default_true")]
    pub lazy_loading: bool,

    /// Interval in hours to check for grammar updates
    /// Default: 168 (weekly)
    #[serde(default = "default_grammar_check_interval")]
    pub check_interval_hours: u32,

    /// Enable idle-time grammar and LSP update checks
    /// When true, daemon checks for updates when the queue is idle
    #[serde(default = "default_true")]
    pub idle_update_check_enabled: bool,

    /// Seconds the queue must be continuously empty before triggering update check
    /// Range: 1-3600
    #[serde(default = "default_idle_update_check_delay")]
    pub idle_update_check_delay_secs: u64,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_grammar_cache_dir(),
            required: default_required_grammars(),
            auto_download: default_true(),
            tree_sitter_version: default_tree_sitter_version(),
            download_base_url: default_download_base_url(),
            verify_checksums: default_true(),
            lazy_loading: default_true(),
            check_interval_hours: default_grammar_check_interval(),
            idle_update_check_enabled: default_true(),
            idle_update_check_delay_secs: default_idle_update_check_delay(),
        }
    }
}

impl GrammarConfig {
    /// Validate grammar configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.tree_sitter_version.is_empty() {
            return Err("tree_sitter_version cannot be empty".to_string());
        }
        if self.download_base_url.is_empty() && self.auto_download {
            return Err("download_base_url required when auto_download is enabled".to_string());
        }
        if self.idle_update_check_delay_secs == 0 {
            return Err("idle_update_check_delay_secs must be greater than 0".to_string());
        }
        if self.idle_update_check_delay_secs > 3600 {
            return Err("idle_update_check_delay_secs must not exceed 3600 (1 hour)".to_string());
        }
        Ok(())
    }

    /// Get the expanded cache directory (with ~ and env vars expanded)
    pub fn expanded_cache_dir(&self) -> PathBuf {
        let path_str = self.cache_dir.to_string_lossy().into_owned();
        let expanded = shellexpand::tilde(&path_str);
        PathBuf::from(expanded.into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_settings_defaults() {
        let settings = LspSettings::default();
        assert!(settings.user_path.is_none());
        assert_eq!(settings.max_servers_per_project, 3);
        assert!(settings.auto_start_on_activation);
        assert_eq!(settings.deactivation_delay_secs, 60);
        assert!(settings.enable_enrichment_cache);
        assert_eq!(settings.cache_ttl_secs, 300);
        assert_eq!(settings.startup_timeout_secs, 30);
        assert_eq!(settings.request_timeout_secs, 10);
        assert_eq!(settings.health_check_interval_secs, 60);
        assert_eq!(settings.max_restart_attempts, 3);
        assert_eq!(settings.restart_backoff_multiplier, 2.0);
        assert!(settings.enable_auto_restart);
        assert_eq!(settings.stability_reset_secs, 3600);
    }

    #[test]
    fn test_lsp_settings_validation() {
        let mut settings = LspSettings::default();

        // Valid settings
        assert!(settings.validate().is_ok());

        // Invalid max_servers_per_project
        settings.max_servers_per_project = 0;
        assert!(settings.validate().is_err());
        settings.max_servers_per_project = 3;

        // Invalid cache_ttl_secs
        settings.cache_ttl_secs = 0;
        assert!(settings.validate().is_err());
        settings.cache_ttl_secs = 300;

        // Invalid startup_timeout_secs
        settings.startup_timeout_secs = 0;
        assert!(settings.validate().is_err());
        settings.startup_timeout_secs = 30;

        // Invalid request_timeout_secs
        settings.request_timeout_secs = 0;
        assert!(settings.validate().is_err());
        settings.request_timeout_secs = 10;

        // Invalid health_check_interval_secs
        settings.health_check_interval_secs = 0;
        assert!(settings.validate().is_err());
        settings.health_check_interval_secs = 60;

        // Invalid restart_backoff_multiplier
        settings.restart_backoff_multiplier = 0.5;
        assert!(settings.validate().is_err());
        settings.restart_backoff_multiplier = 2.0;

        // All valid again
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_lsp_settings_serialization() {
        let settings = LspSettings {
            user_path: Some("/usr/local/bin".to_string()),
            max_servers_per_project: 5,
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("\"max_servers_per_project\":5"));
        assert!(json.contains("\"/usr/local/bin\""));

        // Deserialize back
        let deserialized: LspSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.user_path, Some("/usr/local/bin".to_string()));
        assert_eq!(deserialized.max_servers_per_project, 5);
    }

    #[test]
    fn test_grammar_config_defaults() {
        let config = GrammarConfig::default();
        assert_eq!(config.cache_dir, PathBuf::from("~/.workspace-qdrant/grammars"));
        assert!(config.required.is_empty(), "Default required should be empty");
        assert!(config.auto_download);
        assert_eq!(config.tree_sitter_version, "0.24");
        assert!(config.verify_checksums);
        assert!(config.lazy_loading);
        assert!(config.idle_update_check_enabled);
        assert_eq!(config.idle_update_check_delay_secs, 300);
    }

    #[test]
    fn test_grammar_config_validation() {
        let mut config = GrammarConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid tree_sitter_version
        config.tree_sitter_version = String::new();
        assert!(config.validate().is_err());
        config.tree_sitter_version = "0.24".to_string();

        // Invalid download_base_url when auto_download enabled
        config.download_base_url = String::new();
        assert!(config.validate().is_err());
        config.auto_download = false;
        assert!(config.validate().is_ok()); // Empty URL ok when auto_download disabled
        config = GrammarConfig::default(); // Reset

        // Invalid idle_update_check_delay_secs = 0
        config.idle_update_check_delay_secs = 0;
        assert!(config.validate().is_err());
        config.idle_update_check_delay_secs = 300;

        // Invalid idle_update_check_delay_secs > 3600
        config.idle_update_check_delay_secs = 3601;
        assert!(config.validate().is_err());
        config.idle_update_check_delay_secs = 300;

        // Valid edge: exactly 3600
        config.idle_update_check_delay_secs = 3600;
        assert!(config.validate().is_ok());

        // Valid edge: exactly 1
        config.idle_update_check_delay_secs = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_grammar_config_expanded_cache_dir() {
        let config = GrammarConfig::default();
        let expanded = config.expanded_cache_dir();
        // Should expand ~ to home directory
        assert!(!expanded.to_string_lossy().contains('~'));
        assert!(expanded.to_string_lossy().ends_with("grammars"));
    }

    #[test]
    fn test_grammar_config_serialization() {
        let config = GrammarConfig {
            cache_dir: PathBuf::from("/custom/grammars"),
            required: vec!["rust".to_string(), "python".to_string()],
            auto_download: false,
            tree_sitter_version: "0.24".to_string(),
            download_base_url: "https://example.com".to_string(),
            verify_checksums: true,
            lazy_loading: true,
            check_interval_hours: 168,
            idle_update_check_enabled: false,
            idle_update_check_delay_secs: 600,
        };

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"/custom/grammars\""));
        assert!(json.contains("\"auto_download\":false"));
        assert!(json.contains("\"idle_update_check_enabled\":false"));
        assert!(json.contains("\"idle_update_check_delay_secs\":600"));

        // Deserialize back
        let deserialized: GrammarConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cache_dir, PathBuf::from("/custom/grammars"));
        assert!(!deserialized.auto_download);
        assert!(!deserialized.idle_update_check_enabled);
        assert_eq!(deserialized.idle_update_check_delay_secs, 600);
    }

    #[test]
    fn test_grammar_config_missing_new_fields_uses_defaults() {
        // Simulate deserialization from config without new fields
        let json = r#"{
            "cache_dir": "/custom/grammars",
            "required": ["rust"],
            "auto_download": true,
            "tree_sitter_version": "0.24",
            "download_base_url": "https://example.com",
            "verify_checksums": true,
            "lazy_loading": true,
            "check_interval_hours": 168
        }"#;
        let config: GrammarConfig = serde_json::from_str(json).unwrap();
        // New fields should use defaults when missing
        assert!(config.idle_update_check_enabled);
        assert_eq!(config.idle_update_check_delay_secs, 300);
    }
}
