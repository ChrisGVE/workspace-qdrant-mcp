//! Configuration types for Tier 3 LLM-assisted tagging.
//!
//! Defines provider enums, access modes, and the top-level `Tier3Config`.

use tracing::info;

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmProvider {
    Anthropic,
    OpenAi,
    Google,
    Ollama,
}

impl LlmProvider {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "openai" | "gpt" => Ok(Self::OpenAi),
            "google" | "gemini" => Ok(Self::Google),
            "ollama" => Ok(Self::Ollama),
            other => Err(format!("unknown LLM provider: '{}'", other)),
        }
    }

    /// Default base URL for API mode.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAi => "https://api.openai.com",
            Self::Google => "https://generativelanguage.googleapis.com",
            Self::Ollama => "http://localhost:11434",
        }
    }

    /// CLI binary name for CLI mode.
    pub(super) fn cli_binary(&self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("claude"),
            Self::OpenAi => Some("codex"),
            Self::Google => Some("gemini"),
            Self::Ollama => None, // Ollama always uses HTTP
        }
    }
}

/// How to access the provider.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessMode {
    /// Direct HTTP with API key (pay-per-token).
    Api,
    /// Shell out to provider CLI tool (uses existing subscription).
    Cli,
}

impl AccessMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "api" => Ok(Self::Api),
            "cli" => Ok(Self::Cli),
            other => Err(format!("unknown access mode: '{}' (expected 'api' or 'cli')", other)),
        }
    }
}

/// Configuration for a single provider slot (primary or fallback).
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: LlmProvider,
    pub access_mode: AccessMode,
    pub model: String,
    pub api_key_env: String,
    pub base_url: Option<String>,
}

impl ProviderConfig {
    /// Resolve the effective base URL (custom override or provider default).
    pub(super) fn effective_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.provider.default_base_url())
    }

    /// Resolve the effective access mode. Ollama always uses API (HTTP).
    pub(super) fn effective_access_mode(&self) -> &AccessMode {
        if self.provider == LlmProvider::Ollama {
            &AccessMode::Api
        } else {
            &self.access_mode
        }
    }
}

/// Top-level Tier 3 configuration.
#[derive(Debug, Clone)]
pub struct Tier3Config {
    pub enabled: bool,
    pub primary: ProviderConfig,
    pub fallback: Option<ProviderConfig>,
    pub max_chunks_per_doc: usize,
    pub max_tags_per_chunk: usize,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub rate_limit_rps: u32,
    pub temperature: f64,
    /// Hard ceiling for the entire `extract_tags()` call (seconds).
    /// Prevents a slow/down provider from blocking the ingestion pipeline.
    pub total_budget_secs: u64,
    /// Abort remaining chunks after this many consecutive per-chunk failures.
    /// Acts as a circuit breaker -- if the provider is consistently down, stop
    /// wasting time on subsequent chunks.
    pub max_consecutive_failures: u32,
}

impl Default for Tier3Config {
    fn default() -> Self {
        Self {
            enabled: false,
            primary: ProviderConfig {
                provider: LlmProvider::Anthropic,
                access_mode: AccessMode::Cli,
                model: "claude-haiku-4-5-20251001".to_string(),
                api_key_env: "ANTHROPIC_API_KEY".to_string(),
                base_url: None,
            },
            fallback: None,
            max_chunks_per_doc: 10,
            max_tags_per_chunk: 5,
            timeout_secs: 15,
            max_retries: 2,
            rate_limit_rps: 10,
            temperature: 0.3,
            total_budget_secs: 60,
            max_consecutive_failures: 2,
        }
    }
}

// ── Provider validation ──────────────────────────────────────────────────

/// Validate a provider configuration. Returns the resolved API key (if any).
pub(super) fn resolve_provider(provider: &ProviderConfig) -> Result<Option<String>, String> {
    let mode = provider.effective_access_mode();

    match mode {
        AccessMode::Api => {
            if provider.provider == LlmProvider::Ollama {
                return Ok(None);
            }
            let key = std::env::var(&provider.api_key_env).ok();
            if key.is_none() {
                info!(
                    "Tier3 {:?} API key not found in env var '{}'; \
                     API calls will fail until set",
                    provider.provider, provider.api_key_env
                );
            }
            Ok(key)
        }
        AccessMode::Cli => {
            if let Some(binary) = provider.provider.cli_binary() {
                if which::which(binary).is_err() {
                    return Err(format!(
                        "CLI binary '{}' not found for {:?}. \
                         Install it or switch to access_mode: api",
                        binary, provider.provider
                    ));
                }
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LlmProvider ──────────────────────────────────────────────────

    #[test]
    fn test_provider_from_str_valid() {
        assert_eq!(LlmProvider::from_str("anthropic").unwrap(), LlmProvider::Anthropic);
        assert_eq!(LlmProvider::from_str("claude").unwrap(), LlmProvider::Anthropic);
        assert_eq!(LlmProvider::from_str("openai").unwrap(), LlmProvider::OpenAi);
        assert_eq!(LlmProvider::from_str("gpt").unwrap(), LlmProvider::OpenAi);
        assert_eq!(LlmProvider::from_str("google").unwrap(), LlmProvider::Google);
        assert_eq!(LlmProvider::from_str("gemini").unwrap(), LlmProvider::Google);
        assert_eq!(LlmProvider::from_str("ollama").unwrap(), LlmProvider::Ollama);
    }

    #[test]
    fn test_provider_from_str_case_insensitive() {
        assert_eq!(LlmProvider::from_str("ANTHROPIC").unwrap(), LlmProvider::Anthropic);
        assert_eq!(LlmProvider::from_str("OpenAI").unwrap(), LlmProvider::OpenAi);
    }

    #[test]
    fn test_provider_from_str_unknown() {
        assert!(LlmProvider::from_str("unknown").is_err());
        assert!(LlmProvider::from_str("").is_err());
    }

    // ── AccessMode ───────────────────────────────────────────────────

    #[test]
    fn test_access_mode_from_str_valid() {
        assert_eq!(AccessMode::from_str("api").unwrap(), AccessMode::Api);
        assert_eq!(AccessMode::from_str("cli").unwrap(), AccessMode::Cli);
        assert_eq!(AccessMode::from_str("API").unwrap(), AccessMode::Api);
    }

    #[test]
    fn test_access_mode_from_str_unknown() {
        assert!(AccessMode::from_str("http").is_err());
        assert!(AccessMode::from_str("").is_err());
    }

    // ── Default base URLs ────────────────────────────────────────────

    #[test]
    fn test_default_base_urls() {
        assert_eq!(LlmProvider::Anthropic.default_base_url(), "https://api.anthropic.com");
        assert_eq!(LlmProvider::OpenAi.default_base_url(), "https://api.openai.com");
        assert_eq!(
            LlmProvider::Google.default_base_url(),
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(LlmProvider::Ollama.default_base_url(), "http://localhost:11434");
    }

    // ── Custom base URL override ─────────────────────────────────────

    #[test]
    fn test_custom_base_url_overrides_default() {
        let config = ProviderConfig {
            provider: LlmProvider::OpenAi,
            access_mode: AccessMode::Api,
            model: "gpt-4".to_string(),
            api_key_env: "OPENAI_API_KEY".to_string(),
            base_url: Some("https://my-proxy.example.com".to_string()),
        };
        assert_eq!(config.effective_base_url(), "https://my-proxy.example.com");
    }

    #[test]
    fn test_none_base_url_uses_default() {
        let config = ProviderConfig {
            provider: LlmProvider::Anthropic,
            access_mode: AccessMode::Api,
            model: "claude-haiku-4-5-20251001".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: None,
        };
        assert_eq!(config.effective_base_url(), "https://api.anthropic.com");
    }

    // ── Ollama always uses API mode ──────────────────────────────────

    #[test]
    fn test_ollama_forces_api_mode() {
        let config = ProviderConfig {
            provider: LlmProvider::Ollama,
            access_mode: AccessMode::Cli,
            model: "llama3.2".to_string(),
            api_key_env: String::new(),
            base_url: None,
        };
        assert_eq!(*config.effective_access_mode(), AccessMode::Api);
    }

    #[test]
    fn test_non_ollama_preserves_mode() {
        let config = ProviderConfig {
            provider: LlmProvider::Anthropic,
            access_mode: AccessMode::Cli,
            model: "claude-haiku-4-5-20251001".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: None,
        };
        assert_eq!(*config.effective_access_mode(), AccessMode::Cli);
    }

    // ── Config defaults ──────────────────────────────────────────────

    #[test]
    fn test_tier3_config_defaults() {
        let config = Tier3Config::default();
        assert!(!config.enabled);
        assert_eq!(config.primary.provider, LlmProvider::Anthropic);
        assert_eq!(config.primary.access_mode, AccessMode::Cli);
        assert_eq!(config.max_chunks_per_doc, 10);
        assert_eq!(config.max_tags_per_chunk, 5);
        assert_eq!(config.timeout_secs, 15);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.rate_limit_rps, 10);
        assert!((config.temperature - 0.3).abs() < 1e-6);
        assert_eq!(config.total_budget_secs, 60);
        assert_eq!(config.max_consecutive_failures, 2);
    }

    #[test]
    fn test_tier3_config_fallback_default_none() {
        let config = Tier3Config::default();
        assert!(config.fallback.is_none());
    }

    // ── CLI binary names ─────────────────────────────────────────────

    #[test]
    fn test_cli_binary_names() {
        assert_eq!(LlmProvider::Anthropic.cli_binary(), Some("claude"));
        assert_eq!(LlmProvider::OpenAi.cli_binary(), Some("codex"));
        assert_eq!(LlmProvider::Google.cli_binary(), Some("gemini"));
        assert_eq!(LlmProvider::Ollama.cli_binary(), None);
    }
}
