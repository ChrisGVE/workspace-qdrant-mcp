//! Adaptive HTTP rate-limit handling for remote embedding providers.
//!
//! `RateLimitAdapter` is shared between every in-flight request to one
//! provider instance. It learns from response headers (`Retry-After`,
//! `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens`) and replays
//! the implied waits before subsequent requests via `pre_request()`.
//!
//! Behaviour matches the specification in PRD §6.4:
//!
//! * `Retry-After` parsing accepts both the integer-seconds form
//!   (RFC 9110 §10.2.3) and the HTTP-date form via `httpdate`. Past
//!   HTTP-dates collapse to a zero wait and are logged at DEBUG.
//! * 503 responses are treated identically to 429 only when they carry a
//!   `Retry-After` header — bare 503s do not bump `consecutive_429s`.
//! * Endpoints that omit OpenAI-style headers fall through to a fixed
//!   exponential backoff schedule (`[1, 2, 4, 8, 16, 30]` seconds, capped).
//! * `consecutive_429s > 5` returns `EmbeddingError::RateLimitExhausted`
//!   from the caller; the adapter only tracks the streak.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use reqwest::header::HeaderMap;
use tokio::sync::Mutex;

const BACKOFF_SCHEDULE_SECS: &[u64] = &[1, 2, 4, 8, 16, 30];

/// Adaptive rate-limit / Retry-After tracker for a single provider client.
#[derive(Debug)]
pub struct RateLimitAdapter {
    /// Conservative pre-sleep heuristic: when the upstream reports fewer
    /// remaining tokens than `batch_size * PROACTIVE_BUDGET_MULTIPLIER`, the
    /// adapter sleeps until the reset window before the next request.
    batch_size: usize,
    /// Unbroken streak of 429/503-with-Retry-After responses.
    consecutive_429s: AtomicU32,
    /// Active wait window (seconds). 0 = no wait pending.
    retry_after_secs: AtomicU64,
    /// `Instant` at which the wait window began (set when `retry_after_secs`
    /// is updated). `None` once the window has elapsed.
    last_retry_after: Mutex<Option<Instant>>,
    /// Test-only counter incremented every time `pre_request` actually
    /// sleeps. The Prometheus counter (PRD task 12) reads this value.
    waits_observed: AtomicU64,
}

const PROACTIVE_BUDGET_MULTIPLIER: u64 = 4;

impl RateLimitAdapter {
    /// Construct a fresh adapter sized for the configured batch size.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            consecutive_429s: AtomicU32::new(0),
            retry_after_secs: AtomicU64::new(0),
            last_retry_after: Mutex::new(None),
            waits_observed: AtomicU64::new(0),
        }
    }

    /// Number of consecutive 429/503 responses observed since the last 2xx.
    pub fn consecutive_429s(&self) -> u32 {
        self.consecutive_429s.load(Ordering::Relaxed)
    }

    /// Total number of times `pre_request` slept on a wait window.
    /// Exposed for the Prometheus counter wiring in PRD task 12 and for unit
    /// tests; not part of the public protocol.
    pub fn waits_observed(&self) -> u64 {
        self.waits_observed.load(Ordering::Relaxed)
    }

    /// Enforce any active wait window before issuing the next request.
    /// Returns immediately when no wait is pending.
    pub async fn pre_request(&self) {
        let secs = self.retry_after_secs.load(Ordering::Relaxed);
        if secs == 0 {
            return;
        }

        let mut guard = self.last_retry_after.lock().await;
        let started = match *guard {
            Some(t) => t,
            None => return,
        };
        let elapsed = started.elapsed();
        let total = Duration::from_secs(secs);
        if elapsed < total {
            let remaining = total - elapsed;
            drop(guard);
            self.waits_observed.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(remaining).await;
        } else {
            // Window has elapsed — clear so we don't loop on stale state.
            *guard = None;
            self.retry_after_secs.store(0, Ordering::Relaxed);
        }
    }

    /// Update internal state from a fresh HTTP response.
    pub fn observe_response(&self, headers: &HeaderMap, status: u16) {
        let retry_after = parse_retry_after_header(headers);
        let is_throttle = match status {
            429 => true,
            503 => retry_after.is_some(),
            _ => false,
        };

        if is_throttle {
            let streak = self.consecutive_429s.fetch_add(1, Ordering::Relaxed) + 1;
            let wait_secs = retry_after.unwrap_or_else(|| backoff_secs_for_streak(streak));
            self.retry_after_secs.store(wait_secs, Ordering::Relaxed);
            self.set_window_now();
            return;
        }

        if (200..300).contains(&status) {
            self.consecutive_429s.store(0, Ordering::Relaxed);
            self.retry_after_secs.store(0, Ordering::Relaxed);
            // Drop the window if any stale value is still parked.
            if let Ok(mut guard) = self.last_retry_after.try_lock() {
                *guard = None;
            }

            if let Some(wait) = remaining_tokens_wait(headers, self.batch_size) {
                self.retry_after_secs.store(wait, Ordering::Relaxed);
                self.set_window_now();
            }
        }
    }

    fn set_window_now(&self) {
        if let Ok(mut guard) = self.last_retry_after.try_lock() {
            *guard = Some(Instant::now());
        }
    }
}

/// Map a streak count to a clamped backoff schedule.
fn backoff_secs_for_streak(streak: u32) -> u64 {
    if streak == 0 {
        return BACKOFF_SCHEDULE_SECS[0];
    }
    let idx = (streak as usize - 1).min(BACKOFF_SCHEDULE_SECS.len() - 1);
    BACKOFF_SCHEDULE_SECS[idx]
}

/// Parse the `Retry-After` header — integer seconds first, HTTP-date second.
/// Returns `None` when the header is absent or unparseable. Past HTTP-dates
/// collapse to `Some(0)` so the caller can clear pending state.
fn parse_retry_after_header(headers: &HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;

    if let Ok(secs) = raw.trim().parse::<u64>() {
        return Some(secs);
    }

    match httpdate::parse_http_date(raw.trim()) {
        Ok(when) => match when.duration_since(std::time::SystemTime::now()) {
            Ok(d) => Some(d.as_secs()),
            Err(_) => {
                tracing::debug!(
                    raw = %raw,
                    "Retry-After HTTP-date is in the past; treating as zero wait"
                );
                Some(0)
            }
        },
        Err(_) => None,
    }
}

/// Conservative pre-sleep heuristic on `x-ratelimit-remaining-tokens`.
/// Returns `Some(secs)` when remaining tokens dip below
/// `batch_size * PROACTIVE_BUDGET_MULTIPLIER` and a reset value is present.
fn remaining_tokens_wait(headers: &HeaderMap, batch_size: usize) -> Option<u64> {
    let remaining = header_u64(headers, "x-ratelimit-remaining-tokens")?;
    let threshold = (batch_size as u64).saturating_mul(PROACTIVE_BUDGET_MULTIPLIER);
    if remaining >= threshold {
        return None;
    }
    let reset = header_u64(headers, "x-ratelimit-reset-tokens")?;
    Some(reset)
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.trim().parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    fn empty_headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn headers_with_retry_after(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            reqwest::header::RETRY_AFTER,
            HeaderValue::from_str(value).unwrap(),
        );
        h
    }

    #[tokio::test]
    async fn no_wait_on_first() {
        let adapter = RateLimitAdapter::new(128);
        let start = Instant::now();
        adapter.pre_request().await;
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "fresh adapter must not sleep"
        );
        assert_eq!(adapter.consecutive_429s(), 0);
        assert_eq!(adapter.waits_observed(), 0);
    }

    #[tokio::test]
    async fn exponential_backoff() {
        let adapter = RateLimitAdapter::new(128);
        adapter.observe_response(&empty_headers(), 429);
        assert_eq!(adapter.consecutive_429s(), 1);
        assert_eq!(adapter.retry_after_secs.load(Ordering::Relaxed), 1);

        adapter.observe_response(&empty_headers(), 429);
        adapter.observe_response(&empty_headers(), 429);
        adapter.observe_response(&empty_headers(), 429);
        assert_eq!(adapter.consecutive_429s(), 4);
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            8,
            "schedule[3] = 8s for 4th 429"
        );

        for _ in 0..10 {
            adapter.observe_response(&empty_headers(), 429);
        }
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            30,
            "schedule clamps at the last bucket"
        );
    }

    #[tokio::test]
    async fn retry_after_overrides() {
        let adapter = RateLimitAdapter::new(128);
        adapter.observe_response(&headers_with_retry_after("17"), 429);
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            17,
            "explicit Retry-After must override exponential schedule"
        );
    }

    #[tokio::test]
    async fn http_date_parsed() {
        let adapter = RateLimitAdapter::new(128);
        let future =
            httpdate::fmt_http_date(std::time::SystemTime::now() + Duration::from_secs(45));
        adapter.observe_response(&headers_with_retry_after(&future), 429);
        let parsed = adapter.retry_after_secs.load(Ordering::Relaxed);
        assert!(
            (40..=46).contains(&parsed),
            "HTTP-date Retry-After should land near 45s, got {parsed}"
        );
    }

    #[tokio::test]
    async fn past_date_zero() {
        let adapter = RateLimitAdapter::new(128);
        let past = httpdate::fmt_http_date(std::time::SystemTime::now() - Duration::from_secs(120));
        adapter.observe_response(&headers_with_retry_after(&past), 429);
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            0,
            "past HTTP-date must collapse to zero wait"
        );

        let start = Instant::now();
        adapter.pre_request().await;
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn success_resets() {
        let adapter = RateLimitAdapter::new(128);
        adapter.observe_response(&empty_headers(), 429);
        adapter.observe_response(&empty_headers(), 429);
        assert_eq!(adapter.consecutive_429s(), 2);

        adapter.observe_response(&empty_headers(), 200);
        assert_eq!(adapter.consecutive_429s(), 0);
        assert_eq!(adapter.retry_after_secs.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn five_hundred_three_with_retry_after() {
        let adapter = RateLimitAdapter::new(128);
        adapter.observe_response(&empty_headers(), 503);
        assert_eq!(
            adapter.consecutive_429s(),
            0,
            "bare 503 (no Retry-After) is not throttling"
        );

        adapter.observe_response(&headers_with_retry_after("9"), 503);
        assert_eq!(adapter.consecutive_429s(), 1);
        assert_eq!(adapter.retry_after_secs.load(Ordering::Relaxed), 9);
    }

    #[tokio::test]
    async fn remaining_tokens_wait_triggers() {
        let adapter = RateLimitAdapter::new(128);
        let mut h = HeaderMap::new();
        // 100 < 128 * 4 = 512 → proactive sleep until reset window.
        h.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("100"),
        );
        h.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("3"));
        adapter.observe_response(&h, 200);
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            3,
            "low remaining-tokens budget must arm a proactive wait"
        );

        let mut plenty = HeaderMap::new();
        plenty.insert(
            "x-ratelimit-remaining-tokens",
            HeaderValue::from_static("100000"),
        );
        plenty.insert("x-ratelimit-reset-tokens", HeaderValue::from_static("1"));
        adapter.observe_response(&plenty, 200);
        assert_eq!(
            adapter.retry_after_secs.load(Ordering::Relaxed),
            0,
            "ample remaining-tokens budget must clear the wait"
        );
    }
}
