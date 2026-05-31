//! TTL-cached daemon availability check — mirrors `utils/daemon-check.ts`.
//!
//! [`ensure_daemon_available`] is a write-preflight guard that caches the last
//! health-check result to avoid hammering the daemon on every write operation.
//!
//! # Caching behaviour (daemon-check.ts:24-30)
//! - Positive result (daemon up): cached for [`POSITIVE_TTL_MS`] = 5 000 ms.
//! - Negative result (daemon down): cached for [`NEGATIVE_TTL_MS`] = 1 000 ms
//!   to allow fast recovery.
//!
//! # TS call-site analysis
//! Grepping the TypeScript source for `ensureDaemonAvailable` shows it is
//! **defined but never called** on any MCP tool path (store, rules, etc.).
//! The function exists as a utility but TS self-degrades gracefully without it.
//!
//! This Rust implementation follows the same pattern: the helper and its tests
//! are provided, but it is **not wired into any tool dispatch path** — to be
//! consistent with the TS behaviour.  A call site can be added later if needed.

use std::time::{Duration, Instant};

/// Positive-result TTL — mirrors `POSITIVE_TTL_MS = 5000` in daemon-check.ts.
pub const POSITIVE_TTL_MS: u64 = 5_000;

/// Negative-result TTL — mirrors `NEGATIVE_TTL_MS = 1000` in daemon-check.ts.
pub const NEGATIVE_TTL_MS: u64 = 1_000;

/// Error returned when the daemon is unavailable — exact TS message.
pub const DAEMON_UNAVAILABLE_MSG: &str = "Daemon unavailable. Cannot process write operation.";

/// Cached check result (opaque to callers — use via `Option<DaemonCheckCache>`).
#[derive(Debug, Clone)]
pub struct CachedCheck {
    available: bool,
    recorded_at: Instant,
}

impl CachedCheck {
    fn is_fresh(&self) -> bool {
        let ttl = if self.available {
            Duration::from_millis(POSITIVE_TTL_MS)
        } else {
            Duration::from_millis(NEGATIVE_TTL_MS)
        };
        self.recorded_at.elapsed() < ttl
    }
}

/// Result of a daemon availability probe (async callback).
pub type DaemonCheckResult = Result<(), String>;

/// Ensure the daemon is available for a write operation.
///
/// Mirrors `ensureDaemonAvailable(client)` in daemon-check.ts:20-39.
///
/// # Arguments
/// * `check_fn` — async callable that performs the actual health probe.
///   Returns `Ok(())` when available, `Err(msg)` when not.
/// * `cache` — mutable reference to the cached state (caller owns, enabling
///   injection and reset in tests).
///
/// # Errors
/// Returns `Err(DAEMON_UNAVAILABLE_MSG)` when the daemon is down.
pub async fn ensure_daemon_available<F, Fut>(
    check_fn: F,
    cache: &mut Option<CachedCheck>,
) -> Result<(), String>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = DaemonCheckResult>,
{
    // Check cache freshness.
    if let Some(ref cached) = *cache {
        if cached.is_fresh() {
            return if cached.available {
                Ok(())
            } else {
                Err(DAEMON_UNAVAILABLE_MSG.to_string())
            };
        }
    }

    // Cache miss or stale — run live probe.
    match check_fn().await {
        Ok(()) => {
            *cache = Some(CachedCheck {
                available: true,
                recorded_at: Instant::now(),
            });
            Ok(())
        }
        Err(_) => {
            *cache = Some(CachedCheck {
                available: false,
                recorded_at: Instant::now(),
            });
            Err(DAEMON_UNAVAILABLE_MSG.to_string())
        }
    }
}

/// Reset the cache (mirrors `resetDaemonCheck()` in daemon-check.ts:43-45).
pub fn reset_daemon_check(cache: &mut Option<CachedCheck>) {
    *cache = None;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── cache-miss path ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn returns_ok_when_probe_succeeds_no_cache() {
        let mut cache = None;
        let result = ensure_daemon_available(|| async { Ok(()) }, &mut cache).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn returns_err_when_probe_fails_no_cache() {
        let mut cache = None;
        let result =
            ensure_daemon_available(|| async { Err("probe failed".to_string()) }, &mut cache).await;
        assert_eq!(result.unwrap_err(), DAEMON_UNAVAILABLE_MSG);
    }

    // ── positive cache hit ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn positive_cache_hit_does_not_call_probe() {
        let mut cache = Some(CachedCheck {
            available: true,
            recorded_at: Instant::now(),
        });
        let mut probe_called = false;
        let result = ensure_daemon_available(
            || {
                probe_called = true;
                async { Ok(()) }
            },
            &mut cache,
        )
        .await;
        assert!(result.is_ok());
        assert!(!probe_called, "probe must not be called on cache hit");
    }

    // ── negative cache hit ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn negative_cache_hit_does_not_call_probe_and_returns_err() {
        let mut cache = Some(CachedCheck {
            available: false,
            recorded_at: Instant::now(),
        });
        let mut probe_called = false;
        let result = ensure_daemon_available(
            || {
                probe_called = true;
                async { Ok(()) }
            },
            &mut cache,
        )
        .await;
        assert_eq!(result.unwrap_err(), DAEMON_UNAVAILABLE_MSG);
        assert!(!probe_called, "probe must not be called on cache hit");
    }

    // ── stale positive cache re-probes ─────────────────────────────────────────

    #[tokio::test]
    async fn stale_positive_cache_re_probes() {
        // Stale = recorded more than POSITIVE_TTL_MS ago.
        let stale_at = Instant::now() - Duration::from_millis(POSITIVE_TTL_MS + 100);
        let mut cache = Some(CachedCheck {
            available: true,
            recorded_at: stale_at,
        });
        let mut probe_called = false;
        let _ = ensure_daemon_available(
            || {
                probe_called = true;
                async { Ok(()) }
            },
            &mut cache,
        )
        .await;
        assert!(probe_called, "stale cache must trigger re-probe");
    }

    // ── stale negative cache re-probes ────────────────────────────────────────

    #[tokio::test]
    async fn stale_negative_cache_re_probes() {
        let stale_at = Instant::now() - Duration::from_millis(NEGATIVE_TTL_MS + 100);
        let mut cache = Some(CachedCheck {
            available: false,
            recorded_at: stale_at,
        });
        let mut probe_called = false;
        let _ = ensure_daemon_available(
            || {
                probe_called = true;
                async { Ok(()) }
            },
            &mut cache,
        )
        .await;
        assert!(probe_called, "stale negative cache must trigger re-probe");
    }

    // ── cache is updated after probe ───────────────────────────────────────────

    #[tokio::test]
    async fn successful_probe_caches_positive() {
        let mut cache = None;
        ensure_daemon_available(|| async { Ok(()) }, &mut cache)
            .await
            .unwrap();
        let c = cache.as_ref().expect("cache must be set after probe");
        assert!(c.available);
        assert!(c.is_fresh());
    }

    #[tokio::test]
    async fn failed_probe_caches_negative() {
        let mut cache = None;
        let _ = ensure_daemon_available(|| async { Err("down".to_string()) }, &mut cache).await;
        let c = cache.as_ref().expect("cache must be set after probe");
        assert!(!c.available);
        assert!(c.is_fresh());
    }

    // ── reset ──────────────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_cache() {
        let mut cache = Some(CachedCheck {
            available: true,
            recorded_at: Instant::now(),
        });
        reset_daemon_check(&mut cache);
        assert!(cache.is_none());
    }

    // ── TTL constants ──────────────────────────────────────────────────────────

    #[test]
    fn positive_ttl_is_5000ms() {
        assert_eq!(POSITIVE_TTL_MS, 5_000);
    }

    #[test]
    fn negative_ttl_is_1000ms() {
        assert_eq!(NEGATIVE_TTL_MS, 1_000);
    }

    #[test]
    fn error_message_matches_ts() {
        assert_eq!(
            DAEMON_UNAVAILABLE_MSG,
            "Daemon unavailable. Cannot process write operation."
        );
    }
}
