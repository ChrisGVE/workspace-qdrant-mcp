//! Language-specific LSP configuration
//!
//! Contains per-language configuration types (preferred servers, settings,
//! file patterns, enabled features, timeout overrides) and default language
//! configurations for supported programming languages.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::lsp::Language;

/// Language-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Preferred LSP servers in priority order
    pub preferred_servers: Vec<String>,

    /// Language-specific settings
    pub settings: HashMap<String, JsonValue>,

    /// File patterns for this language
    pub file_patterns: Vec<String>,

    /// Workspace configuration
    pub workspace_settings: HashMap<String, JsonValue>,

    /// Enable/disable features for this language
    pub enabled_features: Vec<String>,
    pub disabled_features: Vec<String>,

    /// Timeout overrides
    pub request_timeout_override: Option<Duration>,
    pub startup_timeout_override: Option<Duration>,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            preferred_servers: vec![],
            settings: HashMap::new(),
            file_patterns: vec![],
            workspace_settings: HashMap::new(),
            enabled_features: vec![],
            disabled_features: vec![],
            request_timeout_override: None,
            startup_timeout_override: None,
        }
    }
}

/// Create default language configurations for known programming languages.
pub fn default_language_configs() -> HashMap<Language, LanguageConfig> {
    let mut configs = HashMap::new();

    configs.insert(Language::Python, python_config());
    configs.insert(Language::Rust, rust_config());
    configs.insert(Language::TypeScript, typescript_config());
    configs.insert(Language::JavaScript, javascript_config());
    configs.insert(Language::Go, go_config());
    configs.insert(Language::C, c_config());
    configs.insert(Language::Cpp, cpp_config());

    configs
}

/// Standard LSP feature set shared by most languages.
fn standard_features() -> Vec<String> {
    vec![
        "completion".to_string(),
        "hover".to_string(),
        "definition".to_string(),
        "references".to_string(),
        "diagnostics".to_string(),
    ]
}

/// Extended feature set including code actions and formatting.
fn extended_features() -> Vec<String> {
    let mut features = standard_features();
    features.push("code_action".to_string());
    features.push("formatting".to_string());
    features
}

/// Standard feature set plus formatting (no code_action).
fn standard_with_formatting() -> Vec<String> {
    let mut features = standard_features();
    features.push("formatting".to_string());
    features
}

fn python_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec![
            "ruff-lsp".to_string(),
            "pylsp".to_string(),
            "pyright-langserver".to_string(),
        ],
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "python.analysis.autoSearchPaths".to_string(),
                JsonValue::Bool(true),
            );
            settings.insert(
                "python.analysis.useLibraryCodeForTypes".to_string(),
                JsonValue::Bool(true),
            );
            settings
        },
        file_patterns: vec!["*.py".to_string(), "*.pyw".to_string(), "*.pyi".to_string()],
        enabled_features: standard_features(),
        ..LanguageConfig::default()
    }
}

fn rust_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["rust-analyzer".to_string()],
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "rust-analyzer.checkOnSave.command".to_string(),
                JsonValue::String("clippy".to_string()),
            );
            settings.insert(
                "rust-analyzer.cargo.allFeatures".to_string(),
                JsonValue::Bool(true),
            );
            settings
        },
        file_patterns: vec!["*.rs".to_string()],
        enabled_features: extended_features(),
        // Rust-analyzer can be slow
        request_timeout_override: Some(Duration::from_secs(60)),
        startup_timeout_override: Some(Duration::from_secs(60)),
        ..LanguageConfig::default()
    }
}

fn typescript_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["typescript-language-server".to_string()],
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "typescript.preferences.includePackageJsonAutoImports".to_string(),
                JsonValue::String("on".to_string()),
            );
            settings
        },
        file_patterns: vec!["*.ts".to_string(), "*.tsx".to_string()],
        enabled_features: standard_with_formatting(),
        ..LanguageConfig::default()
    }
}

fn javascript_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["typescript-language-server".to_string()],
        file_patterns: vec!["*.js".to_string(), "*.jsx".to_string(), "*.mjs".to_string()],
        enabled_features: standard_features(),
        ..LanguageConfig::default()
    }
}

fn go_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["gopls".to_string()],
        settings: {
            let mut settings = HashMap::new();
            settings.insert(
                "gopls.analyses.unusedparams".to_string(),
                JsonValue::Bool(true),
            );
            settings.insert(
                "gopls.analyses.shadow".to_string(),
                JsonValue::Bool(true),
            );
            settings
        },
        file_patterns: vec!["*.go".to_string()],
        enabled_features: standard_with_formatting(),
        ..LanguageConfig::default()
    }
}

fn c_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["clangd".to_string(), "ccls".to_string()],
        file_patterns: vec!["*.c".to_string(), "*.h".to_string()],
        enabled_features: standard_features(),
        ..LanguageConfig::default()
    }
}

fn cpp_config() -> LanguageConfig {
    LanguageConfig {
        preferred_servers: vec!["clangd".to_string(), "ccls".to_string()],
        file_patterns: vec![
            "*.cpp".to_string(),
            "*.cc".to_string(),
            "*.cxx".to_string(),
            "*.hpp".to_string(),
        ],
        enabled_features: standard_features(),
        ..LanguageConfig::default()
    }
}
