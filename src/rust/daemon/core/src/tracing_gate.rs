//! Runtime trace cost-gate (PRD B1).
//!
//! A process-global `AtomicU8` "trace tier" consulted *before* constructing
//! expensive span attributes, so `#[instrument]`-heavy hot paths cost ~nothing
//! when tracing is off. This is **independent of the OTLP sampler**: the
//! sampler gates *export*, this gate guards *attribute construction* on the hot
//! path.
//!
//! Reading the tier is a single relaxed atomic load. The canonical hot-path
//! pattern is [`if_tier`], which evaluates its closure only when the current
//! tier is at least the requested one — so the attribute work is skipped
//! entirely (not merely discarded) when gated out.
//!
//! Tiers (ascending verbosity):
//! - [`TraceTier::Off`] — no spans/attributes on instrumented hot paths.
//! - [`TraceTier::Hot`] — only the cheap, high-value spans.
//! - [`TraceTier::Full`] — full span tree including per-chunk/per-point detail.
//!
//! The tier is configurable via the `WQM_TRACE_TIER` env override (parsed by
//! [`TraceTier::parse`]) and is set at startup via [`init_trace_tier_from_env`].
//! It can also be changed at runtime via [`set_trace_tier`] (a single atomic
//! store), so a future config-reload / signal handler flips span volume without
//! a restart. The config-file surface (`TelemetryConfig.tracing`) is wired by
//! B4; this module owns the gate itself and the env override.

use std::sync::atomic::{AtomicU8, Ordering};

/// Runtime trace verbosity tier. Encoded as a `u8` for the atomic gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TraceTier {
    /// Tracing off — instrumented hot paths do no attribute work.
    Off = 0,
    /// Only cheap, high-value spans.
    Hot = 1,
    /// Full span tree including per-chunk/per-point detail.
    Full = 2,
}

impl TraceTier {
    /// Parse a tier from a case-insensitive string (`off` | `hot` | `full`).
    /// Returns `None` for any other value.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "0" => Some(Self::Off),
            "hot" | "1" => Some(Self::Hot),
            "full" | "2" => Some(Self::Full),
            _ => None,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Off,
            1 => Self::Hot,
            _ => Self::Full,
        }
    }
}

/// Environment variable overriding the startup trace tier.
pub const TRACE_TIER_ENV: &str = "WQM_TRACE_TIER";

/// Process-global trace tier. Defaults to [`TraceTier::Off`] so tracing is
/// opt-in and costs nothing until explicitly enabled.
static TRACE_TIER: AtomicU8 = AtomicU8::new(TraceTier::Off as u8);

/// Current trace tier (single relaxed atomic load).
#[inline]
pub fn trace_tier() -> TraceTier {
    TraceTier::from_u8(TRACE_TIER.load(Ordering::Relaxed))
}

/// Set the trace tier at runtime (single relaxed atomic store). Takes effect
/// for subsequent [`trace_tier`] / [`if_tier`] reads without a restart.
pub fn set_trace_tier(tier: TraceTier) {
    TRACE_TIER.store(tier as u8, Ordering::Relaxed);
}

/// Whether the current tier is at least `min` (i.e. work guarded by `min`
/// should run). A single relaxed atomic load.
#[inline]
pub fn tier_enabled(min: TraceTier) -> bool {
    trace_tier() >= min
}

/// Evaluate `f` (typically expensive span-attribute construction) **only** when
/// the current tier is at least `min`; otherwise return `None` without calling
/// `f` at all. This is the hot-path gate: when tracing is off the only cost is
/// the atomic load — the closure body never runs.
#[inline]
pub fn if_tier<T>(min: TraceTier, f: impl FnOnce() -> T) -> Option<T> {
    if tier_enabled(min) {
        Some(f())
    } else {
        None
    }
}

/// Initialise the trace tier from the `WQM_TRACE_TIER` env override at startup.
/// An unset or unparseable value leaves the current tier unchanged (default
/// [`TraceTier::Off`]). Returns the tier in effect after applying the override.
pub fn init_trace_tier_from_env() -> TraceTier {
    if let Ok(raw) = std::env::var(TRACE_TIER_ENV) {
        if let Some(tier) = TraceTier::parse(&raw) {
            set_trace_tier(tier);
        }
    }
    trace_tier()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // These tests mutate the process-global tier; keep them in one #[test] so
    // they run serially and don't race other tests reading the gate.
    #[test]
    fn tier_gate_behaviour() {
        // Ordering: Off < Hot < Full.
        assert!(TraceTier::Off < TraceTier::Hot);
        assert!(TraceTier::Hot < TraceTier::Full);

        // Parsing (case-insensitive + numeric aliases); unknown → None.
        assert_eq!(TraceTier::parse("OFF"), Some(TraceTier::Off));
        assert_eq!(TraceTier::parse("Hot"), Some(TraceTier::Hot));
        assert_eq!(TraceTier::parse("2"), Some(TraceTier::Full));
        assert_eq!(TraceTier::parse("loud"), None);

        // Runtime switching (AC2): set then observe without restart.
        set_trace_tier(TraceTier::Off);
        assert_eq!(trace_tier(), TraceTier::Off);
        assert!(!tier_enabled(TraceTier::Hot));

        // AC3: closure NOT evaluated when gated out (side-effect counter stays 0).
        let calls = AtomicUsize::new(0);
        let gated = if_tier(TraceTier::Hot, || {
            calls.fetch_add(1, Ordering::Relaxed);
            "attrs"
        });
        assert_eq!(gated, None);
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        // Raise the tier: now the closure runs.
        set_trace_tier(TraceTier::Full);
        assert!(tier_enabled(TraceTier::Hot));
        let ran = if_tier(TraceTier::Hot, || {
            calls.fetch_add(1, Ordering::Relaxed);
            "attrs"
        });
        assert_eq!(ran, Some("attrs"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        // Restore default so other tests see a clean Off gate.
        set_trace_tier(TraceTier::Off);
    }
}
