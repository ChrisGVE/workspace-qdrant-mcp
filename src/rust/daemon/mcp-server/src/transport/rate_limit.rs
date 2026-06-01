//! In-memory sliding-window rate limiter per client IP.
//!
//! Mirrors `SlidingWindowLimiter` in `src/typescript/mcp-server/src/auth-middleware.ts:252`.
//!
//! Default: 100 requests per 60-second window (env `MCP_HTTP_RATE_LIMIT`).
//! Exceeding the limit yields 429 + `Retry-After: 60`.
//!
//! This is a single-process tripwire, not a distributed DoS mitigation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Default rate limit (requests per minute per client IP).
///
/// Mirrors `DEFAULT_RATE_LIMIT_PER_MIN` in `auth-middleware.ts:33`.
pub const DEFAULT_RATE_LIMIT_PER_MIN: u32 = 100;

/// Rate-limit window.
///
/// Mirrors `RATE_LIMIT_WINDOW_MS = 60_000` in `auth-middleware.ts:36`.
pub const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Rate-limit configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per [`RATE_LIMIT_WINDOW`] per client IP.
    pub max_per_window: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_per_window: DEFAULT_RATE_LIMIT_PER_MIN,
        }
    }
}

impl RateLimitConfig {
    /// Parse from the `MCP_HTTP_RATE_LIMIT` environment variable.
    ///
    /// Returns the default when the variable is absent/empty.
    ///
    /// # Errors
    ///
    /// Returns an error string when the value is set but not a positive integer.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MCP_HTTP_RATE_LIMIT") {
            Err(_) => Ok(Self::default()),
            Ok(v) if v.trim().is_empty() => Ok(Self::default()),
            Ok(v) => {
                let n: u32 = v.trim().parse().map_err(|_| {
                    format!("MCP_HTTP_RATE_LIMIT must be a positive integer (got: {v})")
                })?;
                if n == 0 {
                    return Err(format!(
                        "MCP_HTTP_RATE_LIMIT must be a positive integer (got: {v})"
                    ));
                }
                Ok(Self { max_per_window: n })
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Limiter
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal in-memory sliding-window limiter.
///
/// One list of timestamps per IP; entries older than the window are dropped on
/// each touch. Thread-safe via an internal `Mutex`.
///
/// Mirrors `SlidingWindowLimiter` in `auth-middleware.ts:252`.
pub struct SlidingWindowLimiter {
    config: RateLimitConfig,
    window: Duration,
    buckets: Mutex<HashMap<String, Vec<Instant>>>,
}

impl SlidingWindowLimiter {
    /// Create a limiter with the supplied configuration and window.
    pub fn new(config: RateLimitConfig, window: Duration) -> Self {
        Self {
            config,
            window,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Create a limiter from the default configuration and the standard 60-second window.
    pub fn with_config(config: RateLimitConfig) -> Self {
        Self::new(config, RATE_LIMIT_WINDOW)
    }

    /// Returns `true` if the request is allowed; `false` if the limit is exceeded.
    ///
    /// Mirrors `SlidingWindowLimiter.allow()` in `auth-middleware.ts:260`.
    pub fn allow(&self, key: &str) -> bool {
        let now = Instant::now();
        let threshold = now - self.window;

        let mut map = self.buckets.lock().expect("rate-limit mutex poisoned");
        let bucket = map.entry(key.to_string()).or_default();

        // Drop stale entries.
        bucket.retain(|&ts| ts > threshold);

        if bucket.len() as u32 >= self.config.max_per_window {
            return false;
        }
        bucket.push(now);
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "rate_limit_tests.rs"]
mod tests;
