/// Tier 3 automated tagging — LLM-assisted classification.
///
/// Sends chunk text to an AI provider and receives structured concept tags.
/// Supports four providers (Anthropic, OpenAI, Google, Ollama) with two
/// access modes each:
/// - **API mode**: Direct HTTP with API key (pay-per-token)
/// - **CLI mode**: Shell out to provider CLI tool (uses existing subscription)
///
/// Disabled by default. Configured via `tagging.tier3` in YAML config.
/// All failures return empty tags — never blocks ingestion.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::time::Duration;

use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use tracing::{info, warn};

use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

// ─── Configuration ──────────────────────────────────────────────────────

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
    fn cli_binary(&self) -> Option<&'static str> {
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
    fn effective_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.provider.default_base_url())
    }

    /// Resolve the effective access mode. Ollama always uses API (HTTP).
    fn effective_access_mode(&self) -> &AccessMode {
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
    /// Acts as a circuit breaker — if the provider is consistently down, stop
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

// ─── Prompt ─────────────────────────────────────────────────────────────

const MAX_CHUNK_CHARS: usize = 2000;

fn build_prompt(chunk_text: &str) -> String {
    let truncated = if chunk_text.len() > MAX_CHUNK_CHARS {
        &chunk_text[..safe_char_boundary(chunk_text, MAX_CHUNK_CHARS)]
    } else {
        chunk_text
    };

    format!(
        "Extract 3-5 topic tags for the following code/text.\n\
         Tags should be general concepts, technologies, or domains.\n\
         Format: comma-separated lowercase kebab-case.\n\
         Only output the tags, nothing else.\n\n\
         Content:\n{}\n\nTags:",
        truncated
    )
}

/// Find the largest valid char boundary at or before `pos`.
fn safe_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut boundary = pos;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

// ─── Response parsing ───────────────────────────────────────────────────

/// Parse comma-separated kebab-case tags from LLM response text.
fn parse_tags_from_response(response: &str) -> Vec<String> {
    response
        .lines()
        .flat_map(|line| line.split(','))
        .map(|tag| {
            // Trim whitespace first, then strip bullet markers (-, *, .)
            let cleaned = tag
                .trim()
                .trim_start_matches(|c: char| c == '-' || c == '*' || c == '.')
                .trim_end_matches(|c: char| c == '.' || c == '*')
                .trim();
            cleaned.to_lowercase().replace(' ', "-")
        })
        .filter(|tag| {
            !tag.is_empty()
                && tag.len() >= 2
                && tag.len() <= 50
                && tag.chars().all(|c| c.is_alphanumeric() || c == '-')
        })
        .collect()
}

// ─── Tagger ─────────────────────────────────────────────────────────────

/// Tier 3 tagger: LLM-assisted tag extraction.
pub struct Tier3Tagger {
    client: reqwest::Client,
    config: Tier3Config,
    primary_key: Option<String>,
    fallback_key: Option<String>,
    rate_limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,
}

impl Tier3Tagger {
    /// Create a new tagger. Resolves API keys from environment and validates
    /// CLI binary availability for CLI mode providers.
    pub fn new(config: Tier3Config) -> Result<Self, String> {
        if !config.enabled {
            // Even when disabled, create a valid instance that returns empty results.
            let quota = Quota::per_second(NonZeroU32::new(1).unwrap());
            return Ok(Self {
                client: reqwest::Client::new(),
                config,
                primary_key: None,
                fallback_key: None,
                rate_limiter: RateLimiter::direct(quota),
            });
        }

        // Validate primary provider
        let primary_key = resolve_provider(&config.primary)?;

        // Validate fallback provider (if configured)
        let fallback_key = match &config.fallback {
            Some(fb) => Some(resolve_provider(fb)?).flatten(),
            None => None,
        };

        let rps = config.rate_limit_rps.max(1);
        let quota = Quota::per_second(
            NonZeroU32::new(rps).unwrap_or(NonZeroU32::new(1).unwrap()),
        );

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| format!("failed to create HTTP client: {}", e))?;

        Ok(Self {
            client,
            config,
            primary_key,
            fallback_key,
            rate_limiter: RateLimiter::direct(quota),
        })
    }

    /// Extract tags from document chunks using the configured LLM provider.
    ///
    /// Returns `Vec<SelectedTag>` with `TagType::Concept` and `"llm:"` prefix.
    /// Gracefully returns empty vec on all failure modes.
    ///
    /// Two safety mechanisms prevent blocking the ingestion pipeline:
    /// - **Total budget**: hard wall-clock ceiling (`total_budget_secs`).
    /// - **Circuit breaker**: aborts after `max_consecutive_failures` chunks
    ///   fail in a row (provider is likely down/unreachable).
    pub async fn extract_tags(&self, chunks: &[&str]) -> Vec<SelectedTag> {
        if !self.config.enabled || chunks.is_empty() {
            return Vec::new();
        }

        let limited_chunks = &chunks[..chunks.len().min(self.config.max_chunks_per_doc)];
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(self.config.total_budget_secs);
        let mut consecutive_failures: u32 = 0;
        let mut all_chunk_tags: Vec<Vec<String>> = Vec::new();

        for (i, chunk) in limited_chunks.iter().enumerate() {
            // Budget check: stop if we've exceeded the total time budget
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    "Tier3 total budget ({}s) exceeded after {}/{} chunks",
                    self.config.total_budget_secs, i, limited_chunks.len()
                );
                break;
            }

            // Circuit breaker: stop if too many consecutive chunks failed
            if consecutive_failures >= self.config.max_consecutive_failures {
                warn!(
                    "Tier3 circuit breaker: {} consecutive failures, \
                     aborting remaining {}/{} chunks",
                    consecutive_failures, i, limited_chunks.len()
                );
                break;
            }

            // Apply budget as a per-chunk timeout so we can't overshoot
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, self.extract_single_chunk(chunk)).await {
                Ok(tags) => {
                    if tags.is_empty() {
                        consecutive_failures += 1;
                    } else {
                        consecutive_failures = 0;
                    }
                    all_chunk_tags.push(tags);
                }
                Err(_) => {
                    warn!(
                        "Tier3 chunk {}/{} timed out (budget exhausted)",
                        i + 1,
                        limited_chunks.len()
                    );
                    break;
                }
            }
        }

        aggregate_tags(&all_chunk_tags, 10)
    }

    /// Try primary provider, fall back if configured.
    async fn extract_single_chunk(&self, chunk: &str) -> Vec<String> {
        // Rate limit
        self.rate_limiter.until_ready().await;

        let prompt = build_prompt(chunk);

        // Try primary
        match self
            .call_with_retries(
                &self.config.primary,
                self.primary_key.as_deref(),
                &prompt,
            )
            .await
        {
            Ok(tags) => return tags,
            Err(e) => {
                if self.config.fallback.is_some() {
                    warn!(
                        "Tier3 primary provider ({:?}) failed: {}, trying fallback",
                        self.config.primary.provider, e
                    );
                } else {
                    warn!(
                        "Tier3 primary provider ({:?}) failed: {}",
                        self.config.primary.provider, e
                    );
                    return Vec::new();
                }
            }
        }

        // Try fallback
        if let Some(ref fallback) = self.config.fallback {
            match self
                .call_with_retries(fallback, self.fallback_key.as_deref(), &prompt)
                .await
            {
                Ok(tags) => return tags,
                Err(e) => {
                    warn!(
                        "Tier3 fallback provider ({:?}) also failed: {}",
                        fallback.provider, e
                    );
                }
            }
        }

        Vec::new()
    }

    /// Retry a provider call up to `max_retries` times with exponential backoff.
    async fn call_with_retries(
        &self,
        provider: &ProviderConfig,
        api_key: Option<&str>,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let mut last_error = String::new();

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let backoff = Duration::from_secs(1 << (attempt - 1).min(3));
                tokio::time::sleep(backoff).await;
            }

            match self.call_provider(provider, api_key, prompt).await {
                Ok(tags) => return Ok(tags),
                Err(e) => {
                    last_error = e;
                    continue;
                }
            }
        }

        Err(last_error)
    }

    /// Dispatch to the appropriate provider and access mode.
    async fn call_provider(
        &self,
        provider: &ProviderConfig,
        api_key: Option<&str>,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let mode = provider.effective_access_mode();
        let base_url = provider.effective_base_url();

        match mode {
            AccessMode::Api => {
                let key = match (&provider.provider, api_key) {
                    (LlmProvider::Ollama, _) => "", // Ollama needs no key
                    (_, Some(k)) => k,
                    (_, None) => {
                        return Err(format!(
                            "API key not set for {:?} (env: {})",
                            provider.provider, provider.api_key_env
                        ));
                    }
                };
                self.call_api(&provider.provider, base_url, key, &provider.model, prompt)
                    .await
            }
            AccessMode::Cli => {
                self.call_cli(&provider.provider, &provider.model, prompt)
                    .await
            }
        }
    }

    // ─── API mode implementations ───────────────────────────────────

    async fn call_api(
        &self,
        provider: &LlmProvider,
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        match provider {
            LlmProvider::Anthropic => {
                self.call_anthropic_api(base_url, api_key, model, prompt).await
            }
            LlmProvider::OpenAi => {
                self.call_openai_api(base_url, api_key, model, prompt).await
            }
            LlmProvider::Google => {
                self.call_google_api(base_url, api_key, model, prompt).await
            }
            LlmProvider::Ollama => {
                self.call_ollama_api(base_url, model, prompt).await
            }
        }
    }

    async fn call_anthropic_api(
        &self,
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let url = format!("{}/v1/messages", base_url);
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 100,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Anthropic HTTP error: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Anthropic API {} : {}", status, text));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Anthropic parse error: {}", e))?;

        let text = json["content"][0]["text"]
            .as_str()
            .unwrap_or("");

        Ok(parse_tags_from_response(text))
    }

    async fn call_openai_api(
        &self,
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let url = format!("{}/v1/chat/completions", base_url);
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 100,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OpenAI HTTP error: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("OpenAI API {} : {}", status, text));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("OpenAI parse error: {}", e))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        Ok(parse_tags_from_response(text))
    }

    async fn call_google_api(
        &self,
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            base_url, model
        );
        let body = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": 100
            }
        });

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Google HTTP error: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Google API {} : {}", status, text));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Google parse error: {}", e))?;

        let text = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("");

        Ok(parse_tags_from_response(text))
    }

    async fn call_ollama_api(
        &self,
        base_url: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let url = format!("{}/api/generate", base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": {"temperature": self.config.temperature}
        });

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama HTTP error: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama API {} : {}", status, text));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Ollama parse error: {}", e))?;

        let text = json["response"].as_str().unwrap_or("");

        Ok(parse_tags_from_response(text))
    }

    // ─── CLI mode implementations ───────────────────────────────────

    async fn call_cli(
        &self,
        provider: &LlmProvider,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        match provider {
            LlmProvider::Anthropic => self.call_claude_cli(model, prompt).await,
            LlmProvider::OpenAi => self.call_codex_cli(model, prompt).await,
            LlmProvider::Google => self.call_gemini_cli(model, prompt).await,
            LlmProvider::Ollama => {
                // Ollama has no CLI mode; this path shouldn't be reached
                // because effective_access_mode() forces Api for Ollama.
                Err("Ollama does not support CLI mode".to_string())
            }
        }
    }

    async fn call_claude_cli(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let output = tokio::process::Command::new("claude")
            .args([
                "-p", prompt,
                "--output-format", "text",
                "--no-input",
                "--model", model,
            ])
            .output()
            .await
            .map_err(|e| format!("claude CLI exec error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("claude CLI exit {}: {}", output.status, stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(parse_tags_from_response(&stdout))
    }

    async fn call_codex_cli(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let output = tokio::process::Command::new("codex")
            .args(["--prompt", prompt, "--quiet", "--model", model])
            .output()
            .await
            .map_err(|e| format!("codex CLI exec error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("codex CLI exit {}: {}", output.status, stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(parse_tags_from_response(&stdout))
    }

    async fn call_gemini_cli(
        &self,
        _model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        let output = tokio::process::Command::new("gemini")
            .args(["-p", prompt])
            .output()
            .await
            .map_err(|e| format!("gemini CLI exec error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("gemini CLI exit {}: {}", output.status, stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(parse_tags_from_response(&stdout))
    }
}

// ─── Provider validation ────────────────────────────────────────────────

/// Validate a provider configuration. Returns the resolved API key (if any).
fn resolve_provider(provider: &ProviderConfig) -> Result<Option<String>, String> {
    let mode = provider.effective_access_mode();

    match mode {
        AccessMode::Api => {
            if provider.provider == LlmProvider::Ollama {
                // Ollama needs no API key
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

// ─── Tag aggregation ────────────────────────────────────────────────────

/// Aggregate tags from multiple chunks by frequency.
///
/// Tags are ranked by how many chunks mention them. The top `max_tags`
/// are returned as `SelectedTag` with `"llm:"` prefix.
fn aggregate_tags(chunk_tags: &[Vec<String>], max_tags: usize) -> Vec<SelectedTag> {
    if chunk_tags.is_empty() {
        return Vec::new();
    }

    let total_chunks = chunk_tags.len() as f64;
    let mut freq: HashMap<String, usize> = HashMap::new();

    for tags in chunk_tags {
        // Deduplicate within a single chunk before counting
        let mut seen = std::collections::HashSet::new();
        for tag in tags {
            if seen.insert(tag.clone()) {
                *freq.entry(tag.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(max_tags);

    ranked
        .into_iter()
        .map(|(tag, count)| SelectedTag {
            phrase: format!("llm:{}", tag),
            tag_type: TagType::Concept,
            score: count as f64 / total_chunks,
            diversity_score: 1.0,
            semantic_score: 0.0,
            ngram_size: 1,
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── LlmProvider ────────────────────────────────────────────────

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

    // ─── AccessMode ─────────────────────────────────────────────────

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

    // ─── Default base URLs ──────────────────────────────────────────

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

    // ─── Custom base URL override ───────────────────────────────────

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

    // ─── Ollama always uses API mode ────────────────────────────────

    #[test]
    fn test_ollama_forces_api_mode() {
        let config = ProviderConfig {
            provider: LlmProvider::Ollama,
            access_mode: AccessMode::Cli, // should be overridden
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

    // ─── Tag response parsing ───────────────────────────────────────

    #[test]
    fn test_parse_tags_happy_path() {
        let tags = parse_tags_from_response("web-server, database, authentication");
        assert_eq!(tags, vec!["web-server", "database", "authentication"]);
    }

    #[test]
    fn test_parse_tags_with_spaces() {
        let tags = parse_tags_from_response("machine learning, data pipeline, api design");
        assert_eq!(tags, vec!["machine-learning", "data-pipeline", "api-design"]);
    }

    #[test]
    fn test_parse_tags_with_bullets() {
        let tags = parse_tags_from_response("- web-server\n- database\n- caching");
        assert_eq!(tags, vec!["web-server", "database", "caching"]);
    }

    #[test]
    fn test_parse_tags_filters_empty_and_short() {
        let tags = parse_tags_from_response(",, a, ok-tag, ");
        assert_eq!(tags, vec!["ok-tag"]);
    }

    #[test]
    fn test_parse_tags_filters_invalid_chars() {
        let tags = parse_tags_from_response("good-tag, bad@tag, also_bad!, fine");
        assert_eq!(tags, vec!["good-tag", "fine"]);
    }

    #[test]
    fn test_parse_tags_empty_response() {
        let tags = parse_tags_from_response("");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_tags_multiline() {
        let tags = parse_tags_from_response("api-design, grpc\nstreaming, protobuf");
        assert_eq!(tags, vec!["api-design", "grpc", "streaming", "protobuf"]);
    }

    // ─── Tag aggregation ────────────────────────────────────────────

    #[test]
    fn test_aggregate_tags_frequency_ranking() {
        let chunk_tags = vec![
            vec!["web".to_string(), "api".to_string()],
            vec!["web".to_string(), "database".to_string()],
            vec!["web".to_string(), "api".to_string(), "caching".to_string()],
        ];
        let tags = aggregate_tags(&chunk_tags, 10);

        assert_eq!(tags[0].phrase, "llm:web");
        assert!((tags[0].score - 1.0).abs() < 1e-6); // 3/3
        assert_eq!(tags[1].phrase, "llm:api");
        assert!((tags[1].score - 2.0 / 3.0).abs() < 1e-6); // 2/3
    }

    #[test]
    fn test_aggregate_tags_deduplication_within_chunk() {
        let chunk_tags = vec![
            vec!["web".to_string(), "web".to_string(), "api".to_string()],
        ];
        let tags = aggregate_tags(&chunk_tags, 10);

        // "web" should count as 1 (not 2) from a single chunk
        let web_tag = tags.iter().find(|t| t.phrase == "llm:web").unwrap();
        assert!((web_tag.score - 1.0).abs() < 1e-6); // 1/1 chunk
    }

    #[test]
    fn test_aggregate_tags_respects_max() {
        let chunk_tags = vec![
            vec![
                "a".to_string(), "bb".to_string(), "cc".to_string(),
                "dd".to_string(), "ee".to_string(),
            ],
        ];
        let tags = aggregate_tags(&chunk_tags, 3);
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_aggregate_tags_empty() {
        let tags = aggregate_tags(&[], 10);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_aggregate_tags_prefix_and_type() {
        let chunk_tags = vec![vec!["rust".to_string()]];
        let tags = aggregate_tags(&chunk_tags, 10);

        assert_eq!(tags[0].phrase, "llm:rust");
        assert_eq!(tags[0].tag_type, TagType::Concept);
    }

    // ─── Prompt building ────────────────────────────────────────────

    #[test]
    fn test_prompt_contains_chunk_text() {
        let prompt = build_prompt("fn main() { println!(\"hello\"); }");
        assert!(prompt.contains("fn main()"));
        assert!(prompt.contains("Tags:"));
    }

    #[test]
    fn test_prompt_truncation() {
        let long_text = "x".repeat(3000);
        let prompt = build_prompt(&long_text);
        // The prompt should contain at most MAX_CHUNK_CHARS of the content
        assert!(prompt.len() < 3000 + 200); // prompt template overhead
    }

    #[test]
    fn test_prompt_truncation_multibyte() {
        // Ensure truncation handles multi-byte chars safely
        let text = "ä".repeat(1500); // each 'ä' is 2 bytes → 3000 bytes total
        let prompt = build_prompt(&text);
        // Should not panic and should produce valid UTF-8
        assert!(prompt.contains("Tags:"));
    }

    // ─── Config defaults ────────────────────────────────────────────

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

    // ─── Disabled tagger ────────────────────────────────────────────

    #[tokio::test]
    async fn test_disabled_tagger_returns_empty() {
        let config = Tier3Config {
            enabled: false,
            ..Tier3Config::default()
        };
        let tagger = Tier3Tagger::new(config).unwrap();
        let tags = tagger.extract_tags(&["some code here"]).await;
        assert!(tags.is_empty());
    }

    #[tokio::test]
    async fn test_empty_chunks_returns_empty() {
        let config = Tier3Config {
            enabled: true,
            ..Tier3Config::default()
        };
        // Construction may fail if CLI not found, so we use disabled
        let config_disabled = Tier3Config {
            enabled: false,
            ..config
        };
        let tagger = Tier3Tagger::new(config_disabled).unwrap();
        let tags = tagger.extract_tags(&[]).await;
        assert!(tags.is_empty());
    }

    // ─── safe_char_boundary ─────────────────────────────────────────

    #[test]
    fn test_safe_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(safe_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_safe_char_boundary_multibyte() {
        let s = "héllo"; // é is 2 bytes
        // byte index 2 is in the middle of é
        let boundary = safe_char_boundary(s, 2);
        assert!(s.is_char_boundary(boundary));
    }

    #[test]
    fn test_safe_char_boundary_beyond_len() {
        let s = "hi";
        assert_eq!(safe_char_boundary(s, 100), 2);
    }

    // ─── CLI binary names ───────────────────────────────────────────

    #[test]
    fn test_cli_binary_names() {
        assert_eq!(LlmProvider::Anthropic.cli_binary(), Some("claude"));
        assert_eq!(LlmProvider::OpenAi.cli_binary(), Some("codex"));
        assert_eq!(LlmProvider::Google.cli_binary(), Some("gemini"));
        assert_eq!(LlmProvider::Ollama.cli_binary(), None);
    }
}
