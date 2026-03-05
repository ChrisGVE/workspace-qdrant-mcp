//! LLM provider API and CLI implementations for Tier 3 tagging.
//!
//! Contains the HTTP API calls (Anthropic, OpenAI, Google, Ollama) and
//! CLI subprocess calls (Claude, Codex, Gemini) used by `Tier3Tagger`.

use super::tier3::{parse_tags_from_response, Tier3Tagger};
use super::tier3_config::LlmProvider;

// ── API mode implementations ─────────────────────────────────────────────

impl Tier3Tagger {
    pub(super) async fn call_api(
        &self,
        provider: &LlmProvider,
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
    ) -> Result<Vec<String>, String> {
        match provider {
            LlmProvider::Anthropic => {
                self.call_anthropic_api(base_url, api_key, model, prompt)
                    .await
            }
            LlmProvider::OpenAi => self.call_openai_api(base_url, api_key, model, prompt).await,
            LlmProvider::Google => self.call_google_api(base_url, api_key, model, prompt).await,
            LlmProvider::Ollama => self.call_ollama_api(base_url, model, prompt).await,
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

        let text = json["content"][0]["text"].as_str().unwrap_or("");

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
        let url = format!("{}/v1beta/models/{}:generateContent", base_url, model);
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

    // ── CLI mode implementations ─────────────────────────────────────

    pub(super) async fn call_cli(
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

    async fn call_claude_cli(&self, model: &str, prompt: &str) -> Result<Vec<String>, String> {
        let output = tokio::process::Command::new("claude")
            .args([
                "-p",
                prompt,
                "--output-format",
                "text",
                "--no-input",
                "--model",
                model,
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

    async fn call_codex_cli(&self, model: &str, prompt: &str) -> Result<Vec<String>, String> {
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

    async fn call_gemini_cli(&self, _model: &str, prompt: &str) -> Result<Vec<String>, String> {
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
