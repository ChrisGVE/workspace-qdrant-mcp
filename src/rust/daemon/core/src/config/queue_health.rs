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
fn default_dlq_rate_band() -> f64 {
    1.0
}
fn default_dlq_empty_eps() -> u64 {
    1
}
fn default_ms_per_kb_floor() -> f64 {
    0.1
}
fn default_embedder_latency_floor() -> f64 {
    1.0
}
fn default_stall_timeout_secs() -> u64 {
    60
}
fn default_all_failing_window() -> usize {
    3
}
fn default_qdrant_probe_timeout_secs() -> u64 {
    2
}
fn default_drain_snapshot_max_age_secs() -> u64 {
    15
}
fn default_baseline_ttl_secs() -> u64 {
    2_592_000 // 30 days
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
/// - `dlq_rate_band = 1.0` — DLQ delta-rate (counts/poll) at or below this is
///   "stuck"; above it the backlog is growing (Red). Absolute, not relative.
/// - probe floors (`ms_per_kb_floor = 0.1`, `embedder_latency_floor = 1.0`) — a
///   baseline below the floor is "too fast to matter" ⇒ Green (no near-zero
///   division). `stall_timeout_secs = 60`, `all_failing_window = 3` poll cycles,
///   `qdrant_probe_timeout_secs = 2`, `drain_snapshot_max_age_secs = 15`,
///   `baseline_ttl_secs = 2_592_000` (30d).
/// - `debounce_window = 5` — five consecutive verdicts feed a **plurality vote
///   with a severity-biased tie-break** (`state.rs` `DebounceRings`). Anti-flap
///   without burying sustained swings. Well-defined for any window size (even or
///   odd), so no odd-window constraint is imposed.
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
    /// DLQ delta-rate flat band (counts/poll): `|rate|` at or below this is
    /// "stuck" (A3). An absolute rate, not a relative band (DOM-03).
    #[serde(default = "default_dlq_rate_band")]
    pub dlq_rate_band: f64,
    /// DLQ "empty" threshold (count): `count < dlq_empty_eps` ⇒ Green regardless
    /// of rate (A3 emptiness test on the live sampled count).
    #[serde(default = "default_dlq_empty_eps")]
    pub dlq_empty_eps: u64,
    /// ms/KB absolute floor: a seeded baseline below this is "too fast to
    /// matter" ⇒ Green (A1, never divides by a near-zero baseline; DOM-05).
    #[serde(default = "default_ms_per_kb_floor")]
    pub ms_per_kb_floor: f64,
    /// Embedder-latency absolute floor (ms): analogous to `ms_per_kb_floor` for
    /// A2.
    #[serde(default = "default_embedder_latency_floor")]
    pub embedder_latency_floor: f64,
    /// Processing-stall timeout (secs): pending > 0 AND no poll/heartbeat within
    /// this window ⇒ Red (B3).
    #[serde(default = "default_stall_timeout_secs")]
    pub stall_timeout_secs: u64,
    /// All-items-failing detection window in **poll cycles** (B4).
    #[serde(default = "default_all_failing_window")]
    pub all_failing_window: usize,
    /// Qdrant reachability probe timeout (secs): the B1 health call must return
    /// within this or the probe is Red (B1).
    #[serde(default = "default_qdrant_probe_timeout_secs")]
    pub qdrant_probe_timeout_secs: u64,
    /// Drain snapshot staleness bound (secs): a pending-bytes snapshot older than
    /// this is insufficient data ⇒ Green (F5, SEC-05).
    #[serde(default = "default_drain_snapshot_max_age_secs")]
    pub drain_snapshot_max_age_secs: u64,
    /// Persisted-baseline TTL (secs): an orphaned `control_baseline` row older
    /// than this is pruned (F10, DATA-04). Default 30 days.
    #[serde(default = "default_baseline_ttl_secs")]
    pub baseline_ttl_secs: u64,
    /// Per-metric debounce window (consecutive verdicts; a plurality vote with a
    /// severity-biased tie-break sets the debounced state).
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
            dlq_rate_band: default_dlq_rate_band(),
            dlq_empty_eps: default_dlq_empty_eps(),
            ms_per_kb_floor: default_ms_per_kb_floor(),
            embedder_latency_floor: default_embedder_latency_floor(),
            stall_timeout_secs: default_stall_timeout_secs(),
            all_failing_window: default_all_failing_window(),
            qdrant_probe_timeout_secs: default_qdrant_probe_timeout_secs(),
            drain_snapshot_max_age_secs: default_drain_snapshot_max_age_secs(),
            baseline_ttl_secs: default_baseline_ttl_secs(),
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
    /// `debounce_window` must be non-zero (the plurality vote with a
    /// severity-biased tie-break is well-defined for any window size, so no
    /// odd-window constraint is imposed); `disk_low_pct` must be in (0,1); and
    /// the two byte floors must be non-zero.
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

        // Probe thresholds: every floor/band/timeout must be a positive value;
        // a zero would make its probe fire instantly or divide by zero.
        Self::validate_positive("dlq_rate_band", self.dlq_rate_band)?;
        Self::validate_positive("ms_per_kb_floor", self.ms_per_kb_floor)?;
        Self::validate_positive("embedder_latency_floor", self.embedder_latency_floor)?;
        if self.dlq_empty_eps == 0 {
            return Err("dlq_empty_eps must be at least 1".to_string());
        }
        if self.stall_timeout_secs == 0 {
            return Err(
                "stall_timeout_secs must be at least 1 (0 ⇒ instant false-Red)".to_string(),
            );
        }
        if self.all_failing_window == 0 {
            return Err("all_failing_window must be at least 1 poll cycle".to_string());
        }
        if self.qdrant_probe_timeout_secs == 0 {
            return Err("qdrant_probe_timeout_secs must be at least 1".to_string());
        }
        if self.drain_snapshot_max_age_secs == 0 {
            return Err("drain_snapshot_max_age_secs must be greater than 0".to_string());
        }
        if self.drain_budget_secs == 0 {
            return Err(
                "drain_budget_secs must be at least 1 (0 ⇒ permanent false Amber)".to_string(),
            );
        }
        // 30-day default; never accept a TTL shorter than a day (would risk
        // pruning an in-use baseline before the next idle flush).
        if self.baseline_ttl_secs < 86_400 {
            return Err(format!(
                "baseline_ttl_secs must be at least 86400 (1 day) (got {})",
                self.baseline_ttl_secs
            ));
        }
        Ok(())
    }

    /// A probe floor/band must be finite and strictly positive.
    fn validate_positive(name: &str, value: f64) -> Result<(), String> {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be a finite value > 0 (got {value})"));
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
        assert_eq!(c.dlq_rate_band, 1.0);
        assert_eq!(c.dlq_empty_eps, 1);
        assert_eq!(c.ms_per_kb_floor, 0.1);
        assert_eq!(c.embedder_latency_floor, 1.0);
        assert_eq!(c.stall_timeout_secs, 60);
        assert_eq!(c.all_failing_window, 3);
        assert_eq!(c.qdrant_probe_timeout_secs, 2);
        assert_eq!(c.drain_snapshot_max_age_secs, 15);
        assert_eq!(c.baseline_ttl_secs, 2_592_000);
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
    fn accepts_even_debounce_window() {
        // IMPL-12: the debounce is a plurality vote with a severity-biased
        // tie-break, well-defined for any window size — no odd constraint.
        let c = QueueHealthConfig {
            debounce_window: 4,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
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
    fn rejects_zero_probe_thresholds() {
        // Each new probe threshold must reject 0 / non-positive.
        let cases: Vec<QueueHealthConfig> = vec![
            QueueHealthConfig {
                dlq_rate_band: 0.0,
                ..Default::default()
            },
            QueueHealthConfig {
                ms_per_kb_floor: 0.0,
                ..Default::default()
            },
            QueueHealthConfig {
                embedder_latency_floor: -1.0,
                ..Default::default()
            },
            QueueHealthConfig {
                dlq_empty_eps: 0,
                ..Default::default()
            },
            QueueHealthConfig {
                stall_timeout_secs: 0,
                ..Default::default()
            },
            QueueHealthConfig {
                all_failing_window: 0,
                ..Default::default()
            },
            QueueHealthConfig {
                qdrant_probe_timeout_secs: 0,
                ..Default::default()
            },
            QueueHealthConfig {
                drain_snapshot_max_age_secs: 0,
                ..Default::default()
            },
        ];
        for c in cases {
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn rejects_zero_drain_budget_secs() {
        let c = QueueHealthConfig {
            drain_budget_secs: 0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_baseline_ttl_below_one_day() {
        let c = QueueHealthConfig {
            baseline_ttl_secs: 3_600,
            ..Default::default()
        };
        assert!(c.validate().is_err());
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
