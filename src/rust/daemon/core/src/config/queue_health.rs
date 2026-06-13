//! Queue-health monitoring configuration (`[queue_health]`, #133 F6).
//!
//! All thresholds for the functional queue-health verdict are runtime-tunable,
//! mirroring the existing per-section config pattern (serde default fns +
//! per-field `#[serde(default = "…")]` + explicit `impl Default` + `validate()`).
//!
//! This module is the **definition authority** for [`QueueHealthConfig`]. The
//! subsystem module `crate::queue_health::config` is only a thin `pub use`
//! re-export of this type — it owns no fields, defaults, or validation. The two
//! full paths (`crate::config::queue_health::QueueHealthConfig` here vs.
//! `crate::queue_health::config::QueueHealthConfig` re-export) never collide.

use serde::{Deserialize, Serialize};

fn default_fast_alpha() -> f64 {
    0.3
}
fn default_slow_alpha() -> f64 {
    0.01
}
fn default_regression_ratio() -> f64 {
    2.0
}
fn default_embedder_ratio() -> f64 {
    2.0
}
fn default_dlq_flat_band() -> f64 {
    0.05
}
fn default_debounce_window() -> usize {
    5
}
fn default_drain_budget_secs() -> u64 {
    86_400
}
fn default_disk_low_bytes() -> u64 {
    1_073_741_824 // 1 GiB
}
fn default_disk_low_pct() -> f64 {
    0.05
}
fn default_min_item_bytes() -> u64 {
    256
}
fn default_item_bytes() -> u64 {
    65_536 // 64 KiB
}

/// Queue-health monitoring thresholds.
///
/// Defaults encode the design rationale (#133 §F6):
/// - `fast_alpha = 0.3` — fast EWMA lane weighs ≈ last ~20 items (the
///   perceptible-immediate window).
/// - `slow_alpha = 0.01` — slow lane (baseline) ≈ last ~200 items; stable but
///   not frozen.
/// - `regression_ratio` / `embedder_ratio = 2.0` — a 2× slowdown vs baseline is
///   the degrade trigger (clear regression, not noise).
/// - `dlq_flat_band = 0.05` — within 5% relative ⇒ DLQ slope is flat (stuck).
/// - `debounce_window = 5` (majority = 3) — five consecutive verdicts; three
///   agree to flip a per-metric state. Anti-flap without burying sustained
///   swings. MUST be odd so the majority vote is always well-defined.
/// - `drain_budget_secs = 86400` — one day of backlog is the "falling behind"
///   line.
/// - disk low ⇒ `free < 1 GiB OR < 5%`.
/// - `min_item_bytes = 256` — ms/KB size floor; known-size items below this are
///   clamped so a tiny file cannot produce an outlier ms/KB.
/// - `default_item_bytes = 65536` — imputed size when no pending row has a known
///   size, so the all-NULL drain fallback never divides by zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueHealthConfig {
    /// Fast-lane EWMA smoothing factor (responsive window).
    #[serde(default = "default_fast_alpha")]
    pub fast_alpha: f64,
    /// Slow-lane EWMA smoothing factor (baseline).
    #[serde(default = "default_slow_alpha")]
    pub slow_alpha: f64,
    /// fast/slow ratio above which ms/KB processing is "regressing".
    #[serde(default = "default_regression_ratio")]
    pub regression_ratio: f64,
    /// fast/slow ratio above which embedder latency is "regressing".
    #[serde(default = "default_embedder_ratio")]
    pub embedder_ratio: f64,
    /// Relative band within which the DLQ-depth slope counts as flat.
    #[serde(default = "default_dlq_flat_band")]
    pub dlq_flat_band: f64,
    /// Per-metric debounce window (consecutive verdicts; majority flips state).
    /// Must be odd.
    #[serde(default = "default_debounce_window")]
    pub debounce_window: usize,
    /// Drain budget in seconds; a backlog draining slower than this is "behind".
    #[serde(default = "default_drain_budget_secs")]
    pub drain_budget_secs: u64,
    /// Absolute free-disk low-water mark in bytes.
    #[serde(default = "default_disk_low_bytes")]
    pub disk_low_bytes: u64,
    /// Relative free-disk low-water mark as a fraction in (0,1).
    #[serde(default = "default_disk_low_pct")]
    pub disk_low_pct: f64,
    /// ms/KB size floor in bytes — known-size items below this are clamped.
    #[serde(default = "default_min_item_bytes")]
    pub min_item_bytes: u64,
    /// Imputed item size in bytes when no pending row has a known size.
    #[serde(default = "default_item_bytes")]
    pub default_item_bytes: u64,
}

impl Default for QueueHealthConfig {
    fn default() -> Self {
        Self {
            fast_alpha: default_fast_alpha(),
            slow_alpha: default_slow_alpha(),
            regression_ratio: default_regression_ratio(),
            embedder_ratio: default_embedder_ratio(),
            dlq_flat_band: default_dlq_flat_band(),
            debounce_window: default_debounce_window(),
            drain_budget_secs: default_drain_budget_secs(),
            disk_low_bytes: default_disk_low_bytes(),
            disk_low_pct: default_disk_low_pct(),
            min_item_bytes: default_min_item_bytes(),
            default_item_bytes: default_item_bytes(),
        }
    }
}

impl QueueHealthConfig {
    /// Validate the queue-health thresholds.
    ///
    /// A degenerate config is rejected at load (not silently accepted): EWMA
    /// alphas must be finite in (0,1]; the two regression ratios must exceed 1;
    /// `debounce_window` must be a non-zero **odd** number (an even window has no
    /// majority on a tie, leaving the vote undefined); `disk_low_pct` must be in
    /// (0,1); and the two byte floors must be non-zero.
    pub fn validate(&self) -> Result<(), String> {
        Self::validate_alpha("fast_alpha", self.fast_alpha)?;
        Self::validate_alpha("slow_alpha", self.slow_alpha)?;

        if !(self.regression_ratio > 1.0) {
            return Err(format!(
                "regression_ratio must be greater than 1.0 (got {})",
                self.regression_ratio
            ));
        }
        if !(self.embedder_ratio > 1.0) {
            return Err(format!(
                "embedder_ratio must be greater than 1.0 (got {})",
                self.embedder_ratio
            ));
        }
        if self.debounce_window == 0 {
            return Err("debounce_window must be at least 1".to_string());
        }
        if self.debounce_window % 2 == 0 {
            return Err(format!(
                "debounce_window must be odd so the majority vote is well-defined (got {})",
                self.debounce_window
            ));
        }
        if !(self.disk_low_pct > 0.0 && self.disk_low_pct < 1.0) {
            return Err(format!(
                "disk_low_pct must be in the open interval (0,1) (got {})",
                self.disk_low_pct
            ));
        }
        if self.min_item_bytes == 0 {
            return Err("min_item_bytes must be greater than 0".to_string());
        }
        if self.default_item_bytes == 0 {
            return Err("default_item_bytes must be greater than 0".to_string());
        }
        Ok(())
    }

    /// An EWMA alpha must be finite and in the half-open interval (0,1].
    fn validate_alpha(name: &str, alpha: f64) -> Result<(), String> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(format!(
                "{name} must be a finite value in (0,1] (got {alpha})"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let c = QueueHealthConfig::default();
        assert_eq!(c.fast_alpha, 0.3);
        assert_eq!(c.slow_alpha, 0.01);
        assert_eq!(c.regression_ratio, 2.0);
        assert_eq!(c.embedder_ratio, 2.0);
        assert_eq!(c.dlq_flat_band, 0.05);
        assert_eq!(c.debounce_window, 5);
        assert_eq!(c.drain_budget_secs, 86_400);
        assert_eq!(c.disk_low_bytes, 1_073_741_824);
        assert_eq!(c.disk_low_pct, 0.05);
        assert_eq!(c.min_item_bytes, 256);
        assert_eq!(c.default_item_bytes, 65_536);
    }

    #[test]
    fn default_config_validates() {
        assert!(QueueHealthConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_fast_alpha() {
        let c = QueueHealthConfig {
            fast_alpha: 0.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_alpha_above_one() {
        let c = QueueHealthConfig {
            slow_alpha: 1.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_non_finite_alpha() {
        let c = QueueHealthConfig {
            fast_alpha: f64::NAN,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = QueueHealthConfig {
            slow_alpha: f64::INFINITY,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_regression_ratio_at_or_below_one() {
        let c = QueueHealthConfig {
            regression_ratio: 1.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_embedder_ratio_at_or_below_one() {
        let c = QueueHealthConfig {
            embedder_ratio: 0.9,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_debounce_window() {
        let c = QueueHealthConfig {
            debounce_window: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_even_debounce_window() {
        // DOM-04: an even window has no majority on a tie.
        let c = QueueHealthConfig {
            debounce_window: 4,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_disk_low_pct_out_of_range() {
        for pct in [0.0, 1.0, 1.5, -0.1] {
            let c = QueueHealthConfig {
                disk_low_pct: pct,
                ..Default::default()
            };
            assert!(c.validate().is_err(), "disk_low_pct {pct} must be rejected");
        }
    }

    #[test]
    fn rejects_zero_byte_floors() {
        let c = QueueHealthConfig {
            min_item_bytes: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = QueueHealthConfig {
            default_item_bytes: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
}
