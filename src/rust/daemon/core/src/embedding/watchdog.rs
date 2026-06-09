//! Embedding subsystem watchdog (spec 18 §3.3).
//!
//! Complements the periodic [`ProviderHealthMonitor`](super::provider::ProviderHealthMonitor)
//! liveness probe with autonomous recovery. While the dense provider is
//! healthy the watchdog idles at a long backstop interval. Once a probe fails
//! it switches to escalating re-init attempts and flips the shared
//! [`EmbeddingHealth`] flag to unavailable; if the provider stays down for
//! `max_attempts` consecutive probes it writes a one-shot diagnostic file and
//! keeps probing at the backstop interval, recovering automatically when the
//! provider returns.
//!
//! It never shuts the daemon down: a provider outage *degrades* the system
//! rather than killing it. The shared [`EmbeddingHealth`] flag is the canonical
//! availability signal the queue processor reads to decide whether to dispatch
//! embedding work or re-lease (park) it until the provider recovers — so
//! search, graph, and delete operations keep serving throughout.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::provider::DenseProvider;

/// Escalating re-init intervals after a failure, capped at 10 minutes
/// (spec 18 §3.3). Index 0 is used after the first failure; the loop holds at
/// the last value for all further attempts and uses it as the healthy-state
/// backstop interval.
pub const DEFAULT_RETRY_INTERVALS_SECS: [u64; 5] = [30, 60, 120, 300, 600];

/// Consecutive failed re-init attempts tolerated before the watchdog writes a
/// one-shot diagnostic. Embedding work stays parked (via [`EmbeddingHealth`])
/// and the watchdog keeps probing; the daemon is never shut down.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 10;

/// Cheaply-cloneable availability flag for the dense embedding subsystem.
///
/// The watchdog owns the write side; future degraded-mode consumers (e.g. the
/// queue processor) read [`EmbeddingHealth::is_available`] to decide whether to
/// dispatch embedding work or re-lease the item.
#[derive(Clone, Debug)]
pub struct EmbeddingHealth {
    available: Arc<AtomicBool>,
}

impl EmbeddingHealth {
    /// Create a handle with the given initial availability.
    pub fn new(available: bool) -> Self {
        Self {
            available: Arc::new(AtomicBool::new(available)),
        }
    }

    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    pub fn set_available(&self) {
        self.available.store(true, Ordering::Release);
    }

    pub fn set_unavailable(&self) {
        self.available.store(false, Ordering::Release);
    }
}

/// Tunable parameters for [`EmbeddingWatchdog`].
#[derive(Clone, Debug)]
pub struct WatchdogConfig {
    /// Re-init intervals; the watchdog steps through these on consecutive
    /// failures and holds at the last value. An empty list falls back to a 600s
    /// backstop rather than panicking.
    pub retry_intervals: Vec<Duration>,
    /// Consecutive failures tolerated before the watchdog writes a one-shot
    /// diagnostic. It keeps probing afterward; the daemon is never shut down.
    pub max_attempts: u32,
    /// Where the failure diagnostic JSON is written once the provider has been
    /// unrecoverable for `max_attempts` probes.
    pub diagnostic_path: PathBuf,
}

impl WatchdogConfig {
    /// Build a config with the spec defaults, writing diagnostics to
    /// `diagnostic_path`.
    pub fn new(diagnostic_path: PathBuf) -> Self {
        Self {
            retry_intervals: DEFAULT_RETRY_INTERVALS_SECS
                .iter()
                .map(|s| Duration::from_secs(*s))
                .collect(),
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            diagnostic_path,
        }
    }
}

/// Diagnostic record written when the watchdog gives up (spec 18 §3.3).
#[derive(Serialize)]
struct FailureDiagnostic {
    timestamp: String,
    daemon_start: String,
    total_attempts: u32,
    last_error: String,
    provider_label: String,
    output_dim: usize,
    action: &'static str,
}

/// Autonomous recovery supervisor for the dense embedding provider.
pub struct EmbeddingWatchdog {
    provider: Arc<dyn DenseProvider>,
    health: EmbeddingHealth,
    config: WatchdogConfig,
    daemon_start: DateTime<Utc>,
}

impl EmbeddingWatchdog {
    /// Create a watchdog. It starts in the `Available` state; the first failed
    /// probe flips it to unavailable.
    pub fn new(provider: Arc<dyn DenseProvider>, config: WatchdogConfig) -> Self {
        Self {
            provider,
            health: EmbeddingHealth::new(true),
            config,
            daemon_start: Utc::now(),
        }
    }

    /// Shared availability handle other subsystems observe to decide whether to
    /// dispatch embedding work or park it. The queue processor reads this to
    /// re-lease (rather than fail) embedding items while the provider is down.
    pub fn health(&self) -> EmbeddingHealth {
        self.health.clone()
    }

    /// Run the recovery loop until `cancel` fires (daemon shutdown).
    ///
    /// The watchdog never shuts the daemon down: a provider outage degrades the
    /// system (embedding work is parked via [`EmbeddingHealth`]) but search,
    /// graph, and delete operations keep serving. On a failed probe it escalates
    /// re-probe intervals and flips health to unavailable; once it has been
    /// unrecoverable for `max_attempts` consecutive probes it writes a one-shot
    /// diagnostic, then keeps probing at the backstop interval so it recovers
    /// automatically when the provider returns.
    pub async fn run(self, cancel: CancellationToken) {
        let mut consecutive_failures: u32 = 0;
        let mut diagnostic_written = false;

        loop {
            let wait = self.interval_for(consecutive_failures);
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("EmbeddingWatchdog: shutdown signal received");
                    return;
                }
                _ = tokio::time::sleep(wait) => {}
            }

            match self.provider.probe().await {
                Ok(()) => {
                    if !self.health.is_available() {
                        info!(
                            recovered_after = consecutive_failures,
                            "embedding provider recovered; resuming embedding work"
                        );
                    }
                    self.health.set_available();
                    consecutive_failures = 0;
                    diagnostic_written = false;
                }
                Err(e) => {
                    consecutive_failures += 1;
                    self.health.set_unavailable();
                    warn!(
                        attempt = consecutive_failures,
                        max = self.config.max_attempts,
                        error = %e,
                        "embedding provider probe failed; embedding work is parked"
                    );
                    if consecutive_failures >= self.config.max_attempts && !diagnostic_written {
                        error!(
                            attempts = consecutive_failures,
                            "embedding provider unrecoverable; writing diagnostic. \
                             Embedding work stays parked and the watchdog keeps \
                             probing — it resumes automatically on recovery."
                        );
                        self.write_diagnostic(consecutive_failures, &e.to_string())
                            .await;
                        diagnostic_written = true;
                    }
                }
            }
        }
    }

    /// Pick the wait before the next probe. Index 0 (healthy) uses the longest
    /// interval as a light backstop; failures step through the escalation. An
    /// empty interval list (only possible via a hand-built `WatchdogConfig`)
    /// falls back to a 600s backstop rather than panicking.
    fn interval_for(&self, consecutive_failures: u32) -> Duration {
        let len = self.config.retry_intervals.len();
        if len == 0 {
            return Duration::from_secs(600);
        }
        let idx = if consecutive_failures == 0 {
            len - 1
        } else {
            ((consecutive_failures - 1) as usize).min(len - 1)
        };
        self.config.retry_intervals[idx]
    }

    async fn write_diagnostic(&self, attempts: u32, last_error: &str) {
        let diag = FailureDiagnostic {
            timestamp: wqm_common::timestamps::now_utc(),
            daemon_start: wqm_common::timestamps::format_utc(&self.daemon_start),
            total_attempts: attempts,
            last_error: last_error.to_string(),
            provider_label: self.provider.provider_label().to_string(),
            output_dim: self.provider.output_dim(),
            action: "degraded_embedding_parked",
        };
        let json = match serde_json::to_string_pretty(&diag) {
            Ok(j) => j,
            Err(e) => {
                error!(error = %e, "failed to serialize embedding failure diagnostic");
                return;
            }
        };
        if let Some(parent) = self.config.diagnostic_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!(path = %parent.display(), error = %e, "could not create diagnostic dir");
            }
        }
        match tokio::fs::write(&self.config.diagnostic_path, json).await {
            Ok(()) => info!(
                path = %self.config.diagnostic_path.display(),
                "wrote embedding failure diagnostic"
            ),
            Err(e) => error!(
                path = %self.config.diagnostic_path.display(),
                error = %e,
                "failed to write embedding failure diagnostic"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use crate::embedding::types::{DenseEmbedding, EmbeddingError};

    /// Provider whose `probe` replays a scripted queue of outcomes, then falls
    /// back to `default_ok` once the queue is drained.
    #[derive(Debug)]
    struct ScriptedProvider {
        results: Mutex<VecDeque<bool>>,
        default_ok: bool,
    }

    impl ScriptedProvider {
        fn new(results: impl IntoIterator<Item = bool>, default_ok: bool) -> Arc<Self> {
            Arc::new(Self {
                results: Mutex::new(results.into_iter().collect()),
                default_ok,
            })
        }
    }

    #[async_trait]
    impl DenseProvider for ScriptedProvider {
        async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(Vec::new())
        }
        fn output_dim(&self) -> usize {
            384
        }
        fn provider_label(&self) -> &str {
            "scripted"
        }
        fn metrics_label(&self) -> &'static str {
            "fastembed"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            let ok = self
                .results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(self.default_ok);
            if ok {
                Ok(())
            } else {
                Err(EmbeddingError::GenerationError {
                    message: "scripted failure".to_string(),
                })
            }
        }
    }

    fn fast_config(max_attempts: u32, tmp: PathBuf) -> WatchdogConfig {
        WatchdogConfig {
            retry_intervals: vec![Duration::from_millis(5), Duration::from_millis(5)],
            max_attempts,
            diagnostic_path: tmp,
        }
    }

    #[test]
    fn interval_for_escalates_then_holds() {
        let cfg = WatchdogConfig::new(PathBuf::from("x"));
        let wd = EmbeddingWatchdog::new(ScriptedProvider::new([], true), cfg);
        // Healthy state uses the longest (backstop) interval.
        assert_eq!(wd.interval_for(0), Duration::from_secs(600));
        // Failures step through the escalation and hold at the last value.
        assert_eq!(wd.interval_for(1), Duration::from_secs(30));
        assert_eq!(wd.interval_for(2), Duration::from_secs(60));
        assert_eq!(wd.interval_for(5), Duration::from_secs(600));
        assert_eq!(wd.interval_for(99), Duration::from_secs(600));
    }

    #[test]
    fn interval_for_empty_intervals_does_not_panic() {
        let cfg = WatchdogConfig {
            retry_intervals: Vec::new(),
            max_attempts: 3,
            diagnostic_path: PathBuf::from("x"),
        };
        let wd = EmbeddingWatchdog::new(ScriptedProvider::new([], true), cfg);
        // Falls back to the 600s backstop rather than panicking on len-1.
        assert_eq!(wd.interval_for(0), Duration::from_secs(600));
        assert_eq!(wd.interval_for(5), Duration::from_secs(600));
    }

    #[tokio::test(start_paused = true)]
    async fn parks_and_keeps_probing_without_shutdown() {
        let dir = std::env::temp_dir().join(format!("wqm-wd-{}", std::process::id()));
        let diag = dir.join("embedding-failure.json");
        let _ = std::fs::remove_file(&diag);

        let provider = ScriptedProvider::new([], false); // every probe fails
        let wd = EmbeddingWatchdog::new(provider, fast_config(3, diag.clone()));
        let health = wd.health();

        // Loop cancel = the only way the watchdog ever exits. A provider failure
        // must NOT terminate it (degrade, not shut down).
        let loop_cancel = CancellationToken::new();
        let handle = tokio::spawn(wd.run(loop_cancel.clone()));

        // Drive the paused clock past several (tiny) intervals so it crosses
        // max_attempts and writes its one-shot diagnostic.
        for _ in 0..8 {
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }

        assert!(
            !handle.is_finished(),
            "watchdog must keep running (parked), not shut down, on provider failure"
        );
        assert!(
            !health.is_available(),
            "health must be unavailable while the provider is down so work is parked"
        );

        // The one-shot diagnostic is written via tokio::fs (blocking pool). The
        // paused test clock freezes timers but not those OS threads, so we must
        // yield BOTH the task (to re-poll the watchdog across its create_dir_all
        // -> write awaits) and a little real wall-time (to let the blocking
        // threads actually run) before asserting — a bare yield spin races on a
        // loaded runner.
        for _ in 0..200 {
            if diag.exists() {
                break;
            }
            tokio::task::yield_now().await;
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(diag.exists(), "diagnostic file must be written");
        let body = std::fs::read_to_string(&diag).unwrap();
        assert!(body.contains("degraded_embedding_parked"));
        assert!(!body.contains("controlled_shutdown"));
        let _ = std::fs::remove_file(&diag);

        loop_cancel.cancel();
        handle.await.expect("watchdog must exit on loop cancel");
    }

    #[tokio::test(start_paused = true)]
    async fn recovers_and_resumes() {
        let provider = ScriptedProvider::new([false, false], true); // fail twice, then heal
        let loop_cancel = CancellationToken::new();
        let wd = EmbeddingWatchdog::new(
            provider,
            fast_config(10, std::env::temp_dir().join("wqm-wd-recover.json")),
        );
        let health = wd.health();
        let handle = tokio::spawn(wd.run(loop_cancel.clone()));

        // Let several intervals elapse so the two failures and a success run.
        for _ in 0..6 {
            tokio::time::advance(Duration::from_secs(700)).await;
            tokio::task::yield_now().await;
        }

        assert!(
            health.is_available(),
            "provider should have recovered and flipped health back to available"
        );

        loop_cancel.cancel();
        handle.await.expect("watchdog must exit on cancel");
    }

    #[tokio::test]
    async fn exits_promptly_on_cancel() {
        let provider = ScriptedProvider::new([], true);
        let wd = EmbeddingWatchdog::new(
            provider,
            WatchdogConfig::new(std::env::temp_dir().join("wqm-wd-cancel.json")),
        );
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(wd.run(cancel.clone()));
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("watchdog must exit promptly after cancel")
            .expect("watchdog task must not panic");
    }
}
