//! Granular performance telemetry with L0-L4 levels
//!
//! This module provides a compile-time-gated, per-module telemetry system
//! built on top of the `tracing` crate. Five granularity levels control the
//! overhead-vs-detail trade-off:
//!
//! | Level | Purpose                 | Overhead   | Default |
//! |-------|-------------------------|------------|---------|
//! | L0    | Module entry/exit       | Negligible | ON      |
//! | L1    | Function durations      | Low        | OFF     |
//! | L2    | Data flow metrics       | Moderate   | OFF     |
//! | L3    | Per-item tracing        | Higher     | OFF     |
//! | L4    | Debug diagnostics       | Full       | OFF     |
//!
//! Each level implies all lower levels (L2 enables L1 and L0).
//!
//! # Compile-time gating
//!
//! Feature flags `telemetry-l0` through `telemetry-l4` gate macro expansion.
//! When a level is disabled its macros compile to no-ops with zero cost.
//!
//! # Runtime filtering
//!
//! The [`TelemetryLayer`] filters spans/events by per-module overrides
//! stored in [`GranularTelemetryConfig`].

mod config;
mod levels;
#[macro_use]
mod macros;
mod subscriber;

pub use config::GranularTelemetryConfig;
pub use levels::TelemetryLevel;
pub use subscriber::TelemetryLayer;
