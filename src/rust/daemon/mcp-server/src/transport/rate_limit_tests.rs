//! Tests for the sliding-window rate limiter (transport/rate_limit.rs).

use std::time::Duration;

use super::*;

fn limiter(limit: u32) -> SlidingWindowLimiter {
    SlidingWindowLimiter::new(
        RateLimitConfig {
            max_per_window: limit,
        },
        Duration::from_secs(60),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic allow / deny
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allows_up_to_limit() {
    let rl = limiter(3);
    assert!(rl.allow("1.2.3.4"));
    assert!(rl.allow("1.2.3.4"));
    assert!(rl.allow("1.2.3.4"));
    // 4th request exceeds limit of 3.
    assert!(!rl.allow("1.2.3.4"));
}

#[test]
fn different_ips_are_independent() {
    let rl = limiter(1);
    assert!(rl.allow("10.0.0.1"));
    assert!(!rl.allow("10.0.0.1")); // second from same IP → denied
    assert!(rl.allow("10.0.0.2")); // first from different IP → allowed
}

#[test]
fn first_request_always_allowed() {
    let rl = limiter(100);
    assert!(rl.allow("192.168.0.1"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Window rollover (short window)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn window_rollover_allows_after_expiry() {
    // Use a 1-millisecond window so we can expire entries without sleeping long.
    let rl = SlidingWindowLimiter::new(
        RateLimitConfig { max_per_window: 1 },
        Duration::from_millis(1),
    );
    let key = "172.16.0.1";
    assert!(rl.allow(key)); // OK — first request
    assert!(!rl.allow(key)); // Denied — limit 1 reached

    // Sleep slightly more than the window to ensure the first entry expires.
    std::thread::sleep(Duration::from_millis(5));

    // Now the old entry is stale; a fresh request must be allowed.
    assert!(rl.allow(key));
}

// ─────────────────────────────────────────────────────────────────────────────
// 429 response fields
// ─────────────────────────────────────────────────────────────────────────────

/// The limiter itself does not write HTTP responses, but the HTTP layer must
/// respond with 429 + Retry-After: 60. This test documents the contract.
#[test]
fn deny_signals_429_retry_after_60() {
    // Contract: when allow() returns false, the caller must respond:
    //   HTTP 429 Too Many Requests
    //   Retry-After: 60
    let rl = limiter(0); // limit 0 → always deny
                         // (limit=0 means max_per_window=0; allow() should always return false)
    assert!(!rl.allow("any-ip"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Config parsing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn config_default() {
    let cfg = RateLimitConfig::default();
    assert_eq!(cfg.max_per_window, DEFAULT_RATE_LIMIT_PER_MIN);
}
