//! Tier 3 automated tagging -- LLM-assisted classification.
//!
//! Sends chunk text to an AI provider and receives structured concept tags.
//! Supports four providers (Anthropic, OpenAI, Google, Ollama) with two
//! access modes each:
//! - **API mode**: Direct HTTP with API key (pay-per-token)
//! - **CLI mode**: Shell out to provider CLI tool (uses existing subscription)
//!
//! Disabled by default. Configured via `tagging.tier3` in YAML config.
//! All failures return empty tags -- never blocks ingestion.

use std::num::NonZeroU32;
use std::time::Duration;

use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use tracing::warn;

use crate::keyword_extraction::tag_selector::SelectedTag;

use super::llm_aggregation::aggregate_llm_tags;
use super::tier3_config::{
    AccessMode, LlmProvider, ProviderConfig, Tier3Config,
    resolve_provider,
};

// ── Prompt ───────────────────────────────────────────────────────────────

const MAX_CHUNK_CHARS: usize = 2000;

pub(super) fn build_prompt(chunk_text: &str) -> String {
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

// ── Response parsing ─────────────────────────────────────────────────────

/// Parse comma-separated kebab-case tags from LLM response text.
pub(super) fn parse_tags_from_response(response: &str) -> Vec<String> {
    response
        .lines()
        .flat_map(|line| line.split(','))
        .map(|tag| {
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

// ── Tagger ───────────────────────────────────────────────────────────────

/// Tier 3 tagger: LLM-assisted tag extraction.
pub struct Tier3Tagger {
    pub(super) client: reqwest::Client,
    pub(super) config: Tier3Config,
    pub(super) primary_key: Option<String>,
    pub(super) fallback_key: Option<String>,
    pub(super) rate_limiter: RateLimiter<NotKeyed, InMemoryState, DefaultClock>,
}

impl Tier3Tagger {
    /// Create a new tagger. Resolves API keys from environment and validates
    /// CLI binary availability for CLI mode providers.
    pub fn new(config: Tier3Config) -> Result<Self, String> {
        if !config.enabled {
            let quota = Quota::per_second(NonZeroU32::new(1).unwrap());
            return Ok(Self {
                client: reqwest::Client::new(),
                config,
                primary_key: None,
                fallback_key: None,
                rate_limiter: RateLimiter::direct(quota),
            });
        }

        let primary_key = resolve_provider(&config.primary)?;

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
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    "Tier3 total budget ({}s) exceeded after {}/{} chunks",
                    self.config.total_budget_secs, i, limited_chunks.len()
                );
                break;
            }

            if consecutive_failures >= self.config.max_consecutive_failures {
                warn!(
                    "Tier3 circuit breaker: {} consecutive failures, \
                     aborting remaining {}/{} chunks",
                    consecutive_failures, i, limited_chunks.len()
                );
                break;
            }

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

        aggregate_llm_tags(&all_chunk_tags, 10)
    }

    /// Try primary provider, fall back if configured.
    async fn extract_single_chunk(&self, chunk: &str) -> Vec<String> {
        self.rate_limiter.until_ready().await;

        let prompt = build_prompt(chunk);

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
    pub(super) async fn call_with_retries(
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
    pub(super) async fn call_provider(
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
                    (LlmProvider::Ollama, _) => "",
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword_extraction::tag_selector::TagType;
    use super::super::llm_aggregation::aggregate_llm_tags;

    // ── Tag response parsing ─────────────────────────────────────────

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

    // ── Tag aggregation ──────────────────────────────────────────────

    #[test]
    fn test_aggregate_tags_frequency_ranking() {
        let chunk_tags = vec![
            vec!["web".to_string(), "api".to_string()],
            vec!["web".to_string(), "database".to_string()],
            vec!["web".to_string(), "api".to_string(), "caching".to_string()],
        ];
        let tags = aggregate_llm_tags(&chunk_tags, 10);

        assert_eq!(tags[0].phrase, "llm:web");
        assert!((tags[0].score - 1.0).abs() < 1e-6);
        assert_eq!(tags[1].phrase, "llm:api");
        assert!((tags[1].score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_tags_deduplication_within_chunk() {
        let chunk_tags = vec![
            vec!["web".to_string(), "web".to_string(), "api".to_string()],
        ];
        let tags = aggregate_llm_tags(&chunk_tags, 10);

        let web_tag = tags.iter().find(|t| t.phrase == "llm:web").unwrap();
        assert!((web_tag.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_tags_respects_max() {
        let chunk_tags = vec![
            vec![
                "a".to_string(), "bb".to_string(), "cc".to_string(),
                "dd".to_string(), "ee".to_string(),
            ],
        ];
        let tags = aggregate_llm_tags(&chunk_tags, 3);
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_aggregate_tags_empty() {
        let tags = aggregate_llm_tags(&[], 10);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_aggregate_tags_prefix_and_type() {
        let chunk_tags = vec![vec!["rust".to_string()]];
        let tags = aggregate_llm_tags(&chunk_tags, 10);

        assert_eq!(tags[0].phrase, "llm:rust");
        assert_eq!(tags[0].tag_type, TagType::Concept);
    }

    // ── Prompt building ──────────────────────────────────────────────

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
        assert!(prompt.len() < 3000 + 200);
    }

    #[test]
    fn test_prompt_truncation_multibyte() {
        let text = "\u{00e4}".repeat(1500);
        let prompt = build_prompt(&text);
        assert!(prompt.contains("Tags:"));
    }

    // ── safe_char_boundary ───────────────────────────────────────────

    #[test]
    fn test_safe_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(safe_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_safe_char_boundary_multibyte() {
        let s = "h\u{00e9}llo";
        let boundary = safe_char_boundary(s, 2);
        assert!(s.is_char_boundary(boundary));
    }

    #[test]
    fn test_safe_char_boundary_beyond_len() {
        let s = "hi";
        assert_eq!(safe_char_boundary(s, 100), 2);
    }

    // ── Disabled tagger ──────────────────────────────────────────────

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
        let config_disabled = Tier3Config {
            enabled: false,
            ..config
        };
        let tagger = Tier3Tagger::new(config_disabled).unwrap();
        let tags = tagger.extract_tags(&[]).await;
        assert!(tags.is_empty());
    }
}
