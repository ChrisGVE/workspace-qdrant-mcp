//! Branch-lineage indexing configuration (`[branch_lineage]`).
//!
//! Runtime tolerances for the branch-lineage subsystem (resolver depth, the
//! re-key pass, the add/cleanup lock map, the read-path latency alert, and the
//! exclusion / over-threshold wire-shape switches). It mirrors the existing
//! per-section config pattern (serde `default_*` fns + per-field
//! `#[serde(default = "…")]` + an explicit `impl Default` + a `validate()`
//! returning `Result<(), String>`), exactly as `[queue_health]` does.
//!
//! This module is the **definition authority** for [`BranchLineageConfig`]: it
//! owns the field set, the defaults, and the validation. Later features read
//! these tolerances — the tagger (F6), the latency/telemetry alert path
//! (F15-telemetry), and the re-key pass (F16) — but none of them redefines a
//! default here.
//!
//! Field set + defaults are the authoritative PRD §15.1 / arch §9.6 table. Most
//! defaults carry a documented sizing/tuning basis; the capacities (4096 /
//! 1024) and the alert/latency tolerances are Chris's to tune (PERF-R2-03), and
//! `rekey_readgate_budget_s` has **no** compiled-in default at all — it stays
//! `None` until Chris sets a bound.

use serde::{Deserialize, Serialize};

fn default_lineage_depth_cap() -> u32 {
    64
}
fn default_max_excluded() -> u32 {
    1024
}
fn default_add_lock_map_capacity() -> u32 {
    4096
}
fn default_view_resolution_cache_capacity() -> u32 {
    1024
}
fn default_rekey_batch_size() -> u32 {
    100
}
fn default_rekey_stalled_alert_s() -> u32 {
    300
}
fn default_rekey_pause_ingest() -> bool {
    true
}
fn default_latency_regression_alert_ratio() -> f32 {
    1.5
}
fn default_grep_len1_latency_multiplier() -> f32 {
    2.0
}
fn default_branch_view_resolution_budget_ms() -> u32 {
    50
}
fn default_view_resolution_cache_ttl_s() -> u32 {
    5
}
fn default_latency_baseline_warmup_s() -> u32 {
    300
}
fn default_latency_alert_window_s() -> u32 {
    60
}
fn default_post_filter_page_budget() -> u32 {
    8
}
fn default_post_filter_timeout_ms() -> u32 {
    250
}

/// Strategy when a branch view's exclusion set exceeds `max_excluded`.
///
/// `Scroll` (default) keeps reads correct by post-filtering over a scroll loop;
/// `Reject` refuses the over-threshold query outright (ARCH-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverMaxExcludedStrategy {
    /// Post-filter over a bounded scroll loop (the correct-by-default path).
    Scroll,
    /// Refuse the query when the exclusion set is over threshold.
    Reject,
}

/// Policy for the broadest possible query: `branch="*"` combined with
/// `scope="all"` (the Layer-2 gate, UX-01).
///
/// `Confirm` (default) requires explicit confirmation before running it;
/// `Reject` refuses it outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WildcardAllPolicy {
    /// Require explicit confirmation before running the broadest query.
    Confirm,
    /// Refuse the `branch="*"` + `scope="all"` query outright.
    Reject,
}

/// Granularity of the add + cleanup-delete lock.
///
/// `ContentKey` (default) locks per content-key, allowing maximum concurrency;
/// `Tenant` locks the whole tenant, trading throughput for simplicity
/// (IMPL-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddLockGranularity {
    /// Lock per content-key (finest granularity, highest concurrency).
    ContentKey,
    /// Lock the entire tenant for the add/cleanup-delete critical section.
    Tenant,
}

/// Branch-lineage indexing tolerances (`[branch_lineage]`).
///
/// Defaults encode the design rationale (PRD §15.1, arch §9.6):
/// - `lineage_depth_cap = 64` — the resolver CTE depth bound (§5.2.1).
/// - `max_excluded = 1024` — the exclusion-MECHANISM and wire-shape switch; the
///   point at which exclusion flips from inline filter to the scroll path. It
///   is NOT a tombstone-set cap (ARCH-05).
/// - `over_max_excluded_strategy = scroll` — over-threshold behavior; scroll
///   keeps reads correct, reject refuses the query.
/// - `wildcard_all_policy = confirm` — the broadest query (`branch="*"` +
///   `scope="all"`) requires confirmation (UX-01).
/// - `rekey_pause_ingest = true` — a re-key pass pauses ingest for the tenant.
/// - `rekey_batch_size = 100` — re-key scroll/upsert/delete batch size, the
///   throughput-vs-latency trade (PERF-R2-02).
/// - `rekey_stalled_alert_s = 300` — alert if a tenant sits in
///   `status="in_progress"` past this (PERF-R2-02).
/// - `rekey_readgate_budget_s = None` — per-tenant read-unavailability budget,
///   no compiled-in default (Chris sets it). ALERT-ONLY (PERF-R3-02).
/// - `add_lock_granularity = content_key` and `add_lock_map_capacity = 4096` —
///   the add/cleanup-delete lock strictness and its bounded LRU capacity; the
///   capacity exceeds expected peak concurrent in-flight files (PERF-R2-03).
/// - `latency_regression_alert_ratio = 1.5` — read-path p95 alert multiple vs
///   baseline (PERF-01).
/// - `grep_len1_latency_multiplier = 2.0` — length-1 zero-tombstone grep budget
///   multiple (PERF-01).
/// - `branch_view_resolution_budget_ms = 50` — `branch_view` p95 budget
///   (PERF-04).
/// - `view_resolution_cache_ttl_s = 5` — `BranchViewCache` TTL = the
///   max-staleness bound.
/// - `view_resolution_cache_capacity = 1024` — bounded LRU `BranchViewCache`
///   capacity; exceeds the max distinct branch×scope triples per session
///   (PERF-02, PERF-R2-03).
/// - `latency_baseline_warmup_s = 300` — suppress latency alerts until the
///   baseline stabilizes (PERF-01).
/// - `latency_alert_window_s = 60` — trailing p95 window for the latency alert
///   (PERF-01).
/// - `post_filter_page_budget = 8` and `post_filter_timeout_ms = 250` — the
///   page count and wall-clock caps on the over-`max_excluded` scroll loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchLineageConfig {
    /// Resolver CTE depth bound (§5.2.1).
    #[serde(default = "default_lineage_depth_cap")]
    pub lineage_depth_cap: u32,
    /// Exclusion-MECHANISM + wire-shape switch (NOT a tombstone-set cap;
    /// ARCH-05). Above this the exclusion path flips to the scroll loop.
    #[serde(default = "default_max_excluded")]
    pub max_excluded: u32,
    /// Behavior when the exclusion set exceeds `max_excluded`.
    #[serde(default = "OverMaxExcludedStrategy::default")]
    pub over_max_excluded_strategy: OverMaxExcludedStrategy,
    /// Policy for `branch="*"` + `scope="all"` (the Layer-2 gate, UX-01).
    #[serde(default = "WildcardAllPolicy::default")]
    pub wildcard_all_policy: WildcardAllPolicy,
    /// Whether a re-key pass pauses ingest for the tenant.
    #[serde(default = "default_rekey_pause_ingest")]
    pub rekey_pause_ingest: bool,
    /// Re-key scroll/upsert/delete batch size (PERF-R2-02).
    #[serde(default = "default_rekey_batch_size")]
    pub rekey_batch_size: u32,
    /// Alert if a tenant sits in `status="in_progress"` past this many seconds
    /// (PERF-R2-02).
    #[serde(default = "default_rekey_stalled_alert_s")]
    pub rekey_stalled_alert_s: u32,
    /// Per-tenant read-unavailability budget (seconds). **No compiled-in
    /// default — Chris sets the bound, so it stays `None` until configured.**
    ///
    /// ALERT-ONLY (PERF-R3-02): the read-gate holds until the re-key completes
    /// for correctness, so breaching this budget only emits a warning/alert —
    /// it NEVER aborts the re-key nor denies reads. The window is bounded,
    /// observable, and alert-wired regardless of the configured value.
    #[serde(default)]
    pub rekey_readgate_budget_s: Option<u32>,
    /// Add + cleanup-delete lock granularity (IMPL-05).
    #[serde(default = "AddLockGranularity::default")]
    pub add_lock_granularity: AddLockGranularity,
    /// Bounded LRU lock-map capacity (IMPL-05; sizing basis: exceeds expected
    /// peak concurrent in-flight files — PERF-R2-03).
    #[serde(default = "default_add_lock_map_capacity")]
    pub add_lock_map_capacity: u32,
    /// Read-path p95 alert multiple vs baseline (PERF-01).
    #[serde(default = "default_latency_regression_alert_ratio")]
    pub latency_regression_alert_ratio: f32,
    /// Length-1 zero-tombstone grep budget multiple (PERF-01).
    #[serde(default = "default_grep_len1_latency_multiplier")]
    pub grep_len1_latency_multiplier: f32,
    /// `branch_view` p95 budget in milliseconds (PERF-04).
    #[serde(default = "default_branch_view_resolution_budget_ms")]
    pub branch_view_resolution_budget_ms: u32,
    /// `BranchViewCache` TTL in seconds — the max-staleness bound.
    #[serde(default = "default_view_resolution_cache_ttl_s")]
    pub view_resolution_cache_ttl_s: u32,
    /// Bounded LRU `BranchViewCache` capacity (PERF-02; sizing basis: exceeds
    /// the max distinct branch×scope triples per session — PERF-R2-03).
    #[serde(default = "default_view_resolution_cache_capacity")]
    pub view_resolution_cache_capacity: u32,
    /// Suppress latency alerts until the baseline stabilizes (seconds; PERF-01).
    #[serde(default = "default_latency_baseline_warmup_s")]
    pub latency_baseline_warmup_s: u32,
    /// Trailing p95 window for the latency alert (seconds; PERF-01).
    #[serde(default = "default_latency_alert_window_s")]
    pub latency_alert_window_s: u32,
    /// Max pages of the over-`max_excluded` scroll loop.
    #[serde(default = "default_post_filter_page_budget")]
    pub post_filter_page_budget: u32,
    /// Wall-clock cap (milliseconds) on the over-`max_excluded` scroll loop.
    #[serde(default = "default_post_filter_timeout_ms")]
    pub post_filter_timeout_ms: u32,
}

impl Default for OverMaxExcludedStrategy {
    fn default() -> Self {
        Self::Scroll
    }
}

impl Default for WildcardAllPolicy {
    fn default() -> Self {
        Self::Confirm
    }
}

impl Default for AddLockGranularity {
    fn default() -> Self {
        Self::ContentKey
    }
}

impl Default for BranchLineageConfig {
    fn default() -> Self {
        Self {
            lineage_depth_cap: default_lineage_depth_cap(),
            max_excluded: default_max_excluded(),
            over_max_excluded_strategy: OverMaxExcludedStrategy::default(),
            wildcard_all_policy: WildcardAllPolicy::default(),
            rekey_pause_ingest: default_rekey_pause_ingest(),
            rekey_batch_size: default_rekey_batch_size(),
            rekey_stalled_alert_s: default_rekey_stalled_alert_s(),
            rekey_readgate_budget_s: None,
            add_lock_granularity: AddLockGranularity::default(),
            add_lock_map_capacity: default_add_lock_map_capacity(),
            latency_regression_alert_ratio: default_latency_regression_alert_ratio(),
            grep_len1_latency_multiplier: default_grep_len1_latency_multiplier(),
            branch_view_resolution_budget_ms: default_branch_view_resolution_budget_ms(),
            view_resolution_cache_ttl_s: default_view_resolution_cache_ttl_s(),
            view_resolution_cache_capacity: default_view_resolution_cache_capacity(),
            latency_baseline_warmup_s: default_latency_baseline_warmup_s(),
            latency_alert_window_s: default_latency_alert_window_s(),
            post_filter_page_budget: default_post_filter_page_budget(),
            post_filter_timeout_ms: default_post_filter_timeout_ms(),
        }
    }
}

impl BranchLineageConfig {
    /// Validate the branch-lineage tolerances.
    ///
    /// A degenerate config is rejected at load (not silently accepted): every
    /// bound that a value of zero would break — the resolver depth, the re-key
    /// batch size and stalled-alert horizon, the two LRU capacities, the cache
    /// TTL, the view-resolution budget, the latency window, and the scroll-loop
    /// page/timeout caps — must be non-zero. The two ratios are multiples vs a
    /// baseline, so each must be finite and strictly greater than 1.0 (a
    /// multiple at or below 1.0 would fire on no regression at all). When set,
    /// `rekey_readgate_budget_s` must be non-zero (a zero budget would alert
    /// instantly on every re-key).
    pub fn validate(&self) -> Result<(), String> {
        Self::validate_nonzero("lineage_depth_cap", self.lineage_depth_cap)?;
        Self::validate_nonzero("max_excluded", self.max_excluded)?;
        Self::validate_nonzero("rekey_batch_size", self.rekey_batch_size)?;
        Self::validate_nonzero("rekey_stalled_alert_s", self.rekey_stalled_alert_s)?;
        Self::validate_nonzero("add_lock_map_capacity", self.add_lock_map_capacity)?;
        Self::validate_nonzero(
            "view_resolution_cache_capacity",
            self.view_resolution_cache_capacity,
        )?;
        Self::validate_nonzero(
            "view_resolution_cache_ttl_s",
            self.view_resolution_cache_ttl_s,
        )?;
        Self::validate_nonzero(
            "branch_view_resolution_budget_ms",
            self.branch_view_resolution_budget_ms,
        )?;
        Self::validate_nonzero("latency_baseline_warmup_s", self.latency_baseline_warmup_s)?;
        Self::validate_nonzero("latency_alert_window_s", self.latency_alert_window_s)?;
        Self::validate_nonzero("post_filter_page_budget", self.post_filter_page_budget)?;
        Self::validate_nonzero("post_filter_timeout_ms", self.post_filter_timeout_ms)?;

        Self::validate_ratio(
            "latency_regression_alert_ratio",
            self.latency_regression_alert_ratio,
        )?;
        Self::validate_ratio(
            "grep_len1_latency_multiplier",
            self.grep_len1_latency_multiplier,
        )?;

        // The read-gate budget has no default; only validate a value once set.
        if let Some(budget) = self.rekey_readgate_budget_s {
            Self::validate_nonzero("rekey_readgate_budget_s", budget)?;
        }
        Ok(())
    }

    /// A capacity/horizon/bound must be at least 1 (a zero would either fire its
    /// alert instantly or bound a loop/cache to nothing).
    fn validate_nonzero(name: &str, value: u32) -> Result<(), String> {
        if value == 0 {
            return Err(format!("{name} must be at least 1 (got 0)"));
        }
        Ok(())
    }

    /// An alert multiple-vs-baseline must be finite and strictly above 1.0.
    fn validate_ratio(name: &str, value: f32) -> Result<(), String> {
        if !value.is_finite() || value <= 1.0 {
            return Err(format!(
                "{name} must be a finite value greater than 1.0 (got {value})"
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
        let c = BranchLineageConfig::default();
        assert_eq!(c.lineage_depth_cap, 64);
        assert_eq!(c.max_excluded, 1024);
        assert_eq!(
            c.over_max_excluded_strategy,
            OverMaxExcludedStrategy::Scroll
        );
        assert_eq!(c.wildcard_all_policy, WildcardAllPolicy::Confirm);
        assert!(c.rekey_pause_ingest);
        assert_eq!(c.rekey_batch_size, 100);
        assert_eq!(c.rekey_stalled_alert_s, 300);
        assert_eq!(c.rekey_readgate_budget_s, None);
        assert_eq!(c.add_lock_granularity, AddLockGranularity::ContentKey);
        assert_eq!(c.add_lock_map_capacity, 4096);
        assert_eq!(c.latency_regression_alert_ratio, 1.5);
        assert_eq!(c.grep_len1_latency_multiplier, 2.0);
        assert_eq!(c.branch_view_resolution_budget_ms, 50);
        assert_eq!(c.view_resolution_cache_ttl_s, 5);
        assert_eq!(c.view_resolution_cache_capacity, 1024);
        assert_eq!(c.latency_baseline_warmup_s, 300);
        assert_eq!(c.latency_alert_window_s, 60);
        assert_eq!(c.post_filter_page_budget, 8);
        assert_eq!(c.post_filter_timeout_ms, 250);
    }

    #[test]
    fn default_config_validates() {
        assert!(BranchLineageConfig::default().validate().is_ok());
    }

    #[test]
    fn readgate_budget_defaults_to_none_and_accepts_explicit_value() {
        assert_eq!(BranchLineageConfig::default().rekey_readgate_budget_s, None);
        let c = BranchLineageConfig {
            rekey_readgate_budget_s: Some(120),
            ..Default::default()
        };
        assert_eq!(c.rekey_readgate_budget_s, Some(120));
        assert!(c.validate().is_ok());
    }

    #[test]
    fn partial_toml_defaults_the_rest() {
        // A user config that sets only a subset of fields deserialises with the
        // remaining fields taking their compiled-in defaults.
        let toml = r#"
            rekey_batch_size = 250
            wildcard_all_policy = "reject"
            rekey_readgate_budget_s = 90
        "#;
        let c: BranchLineageConfig = toml::from_str(toml).expect("partial config parses");
        assert_eq!(c.rekey_batch_size, 250);
        assert_eq!(c.wildcard_all_policy, WildcardAllPolicy::Reject);
        assert_eq!(c.rekey_readgate_budget_s, Some(90));
        // Untouched fields keep their defaults.
        assert_eq!(c.lineage_depth_cap, 64);
        assert_eq!(
            c.over_max_excluded_strategy,
            OverMaxExcludedStrategy::Scroll
        );
        assert_eq!(c.add_lock_granularity, AddLockGranularity::ContentKey);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn enums_parse_from_snake_case() {
        #[derive(Deserialize)]
        struct Wrap {
            strategy: OverMaxExcludedStrategy,
            policy: WildcardAllPolicy,
            granularity: AddLockGranularity,
        }
        let w: Wrap = toml::from_str(
            r#"
            strategy = "reject"
            policy = "confirm"
            granularity = "tenant"
        "#,
        )
        .expect("enum strings parse");
        assert_eq!(w.strategy, OverMaxExcludedStrategy::Reject);
        assert_eq!(w.policy, WildcardAllPolicy::Confirm);
        assert_eq!(w.granularity, AddLockGranularity::Tenant);
    }

    #[test]
    fn round_trips_through_toml() {
        let original = BranchLineageConfig::default();
        let serialised = toml::to_string(&original).expect("serialise");
        let restored: BranchLineageConfig = toml::from_str(&serialised).expect("deserialise");
        assert_eq!(restored.lineage_depth_cap, original.lineage_depth_cap);
        assert_eq!(
            restored.rekey_readgate_budget_s,
            original.rekey_readgate_budget_s
        );
        assert_eq!(
            restored.over_max_excluded_strategy,
            original.over_max_excluded_strategy
        );
        assert!(restored.validate().is_ok());
    }

    #[test]
    fn rejects_zero_bounds() {
        let zero_cases: Vec<BranchLineageConfig> = vec![
            BranchLineageConfig {
                lineage_depth_cap: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                max_excluded: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                rekey_batch_size: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                rekey_stalled_alert_s: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                add_lock_map_capacity: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                view_resolution_cache_capacity: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                view_resolution_cache_ttl_s: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                branch_view_resolution_budget_ms: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                latency_baseline_warmup_s: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                latency_alert_window_s: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                post_filter_page_budget: 0,
                ..Default::default()
            },
            BranchLineageConfig {
                post_filter_timeout_ms: 0,
                ..Default::default()
            },
        ];
        for c in zero_cases {
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn rejects_zero_readgate_budget_when_set() {
        let c = BranchLineageConfig {
            rekey_readgate_budget_s: Some(0),
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_ratios_at_or_below_one() {
        for ratio in [1.0_f32, 0.9, 0.0] {
            let c = BranchLineageConfig {
                latency_regression_alert_ratio: ratio,
                ..Default::default()
            };
            assert!(
                c.validate().is_err(),
                "latency_regression_alert_ratio {ratio} must be rejected"
            );
        }
        let c = BranchLineageConfig {
            grep_len1_latency_multiplier: 1.0,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_non_finite_ratio() {
        let c = BranchLineageConfig {
            latency_regression_alert_ratio: f32::NAN,
            ..Default::default()
        };
        assert!(c.validate().is_err());
        let c = BranchLineageConfig {
            grep_len1_latency_multiplier: f32::INFINITY,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }
}
