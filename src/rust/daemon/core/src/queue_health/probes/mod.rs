//! Health probes (#133 F3/F4/F5).
//!
//! A probe is a pure function that reads one live signal — an EWMA lane snapshot,
//! a hard-state boolean, or the drain snapshot — and yields a [`ProbeResult`]: a
//! Red/Amber/Green status plus the culprit it blames and a one-line remediation.
//! The overall verdict ([`super::verdict`]) is the worst-of all probe results.
//!
//! Two probe families:
//! - **Family A — trend** ([`trend`]): three EWMA-ratio probes (ms/KB, embedder
//!   latency, DLQ delta-rate). 3-state RAG. Flap-prone, so their RAG is debounced
//!   (plurality, severity-biased) by the poll loop before the verdict reads it.
//! - **Family B — hard state** ([`hard_state`]): four binary predicates (Qdrant
//!   reachable, disk space, processing stall, all-items-failing). Green or Red —
//!   a hard failure has no "degraded" shade.
//!
//! plus the [`drain`] budget probe (Green/Amber).
//!
//! ## Remediation hygiene (S8 / SEC-03)
//!
//! Every remediation string is a fixed literal with at most a numeric ratio
//! interpolated — never an absolute path, a URL, an API key, or a `WQM_*`
//! config-variable name. A prefix-scan test enforces this over the whole canned
//! set (see `probes/tests.rs`).

pub mod drain;
pub mod hard_state;
pub mod trend;

use super::state::Rag;

/// The culprit/probe-name key each probe blames. Doubles as the debounce-ring key
/// (one ring per culprit), so the strings must be unique and stable.
pub const PROCESSING: &str = "processing";
/// Embedder-latency trend probe (A2).
pub const EMBEDDER: &str = "embedder";
/// Dead-letter-queue trend probe (A3).
pub const DLQ: &str = "dlq";
/// Qdrant reachability probe (B1).
pub const QDRANT: &str = "qdrant";
/// Free-disk-space probe (B2).
pub const DISK: &str = "disk";
/// Processing-stall probe (B3).
pub const STALL: &str = "stall";
/// All-items-failing-to-DLQ probe (B4).
pub const ALL_FAILING: &str = "all_failing";
/// Drain-budget probe (F5).
pub const DRAIN: &str = "drain";

/// The result of evaluating one probe.
///
/// `remediation` is `Some` only when `rag != Green` — a green probe needs no
/// operator action. `culprit` is a fixed `&'static str` (one of the consts above)
/// that attributes the verdict to a subsystem and keys the debounce ring.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeResult {
    /// This probe's Red/Amber/Green status.
    pub rag: Rag,
    /// The subsystem blamed (also the debounce-ring key).
    pub culprit: &'static str,
    /// One-line operator remediation; `None` when `rag == Green`.
    pub remediation: Option<String>,
}

impl ProbeResult {
    /// A healthy probe — no remediation.
    pub fn green(culprit: &'static str) -> Self {
        Self {
            rag: Rag::Green,
            culprit,
            remediation: None,
        }
    }

    /// A degraded (Amber) probe with its remediation.
    pub fn amber(culprit: &'static str, remediation: impl Into<String>) -> Self {
        Self {
            rag: Rag::Amber,
            culprit,
            remediation: Some(remediation.into()),
        }
    }

    /// A failing (Red) probe with its remediation.
    pub fn red(culprit: &'static str, remediation: impl Into<String>) -> Self {
        Self {
            rag: Rag::Red,
            culprit,
            remediation: Some(remediation.into()),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
