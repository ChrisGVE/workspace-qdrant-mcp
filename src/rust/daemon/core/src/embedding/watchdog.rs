//! Embedding subsystem watchdog (spec 18 §3.3).
//!
//! Complements the periodic [`ProviderHealthMonitor`](super::provider::ProviderHealthMonitor)
//! liveness probe with autonomous recovery. While the dense provider is
//! healthy the watchdog idles at a long backstop interval. Once a probe fails
//! it switches to escalating re-init attempts; if the provider does not recover
//! within `max_attempts` consecutive failures it writes a diagnostic file and
//! cancels its `shutdown_request` token so the daemon can perform a controlled
//! shutdown (a supervising service manager then restarts the process).
//!
//! The shared [`EmbeddingHealth`] flag it maintains is the canonical
//! availability signal other subsystems read to decide whether embedding work
//! can proceed.

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
/// diagnostic and requests a controlled shutdown.
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
    /// failures and holds at the last value. Must be non-empty.
    pub retry_intervals: Vec<Duration>,
    /// Consecutive failures tolerated before requesting controlled shutdown.
    pub max_attempts: u32,
    /// Where the failure diagnostic JSON is written on give-up.
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
    shutdown_request: CancellationToken,
}

impl EmbeddingWatchdog {
    /// Create a watchdog. It starts in the `Available` state; the first failed
    /// probe flips it to unavailable. `shutdown_request` is cancelled if the
    /// provider proves unrecoverable.
    pub fn new(
        provider: Arc<dyn DenseProvider>,
        config: WatchdogConfig,
        shutdown_request: CancellationToken,
    ) -> Self {
        Self {
            provider,
            health: EmbeddingHealth::new(true),
            config,
            daemon_start: Utc::now(),
            shutdown_request,
        }
    }

    /// Shared availability handle for other subsystems to observe.
    pub fn health(&self) -> EmbeddingHealth {
        self.health.clone()
    }

    /// Run the recovery loop until `cancel` fires (daemon shutdown) or the
    /// provider is declared unrecoverable.
    pub async fn run(self, cancel: CancellationToken) {
        let mut consecutive_failures: u32 = 0;

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
                            "embedding provider recovered"
                        );
                    }
                    self.health.set_available();
                    consecutive_failures = 0;
                }
                Err(e) => {
                    consecutive_failures += 1;
                    self.health.set_unavailable();
                    warn!(
                        attempt = consecutive_failures,
                        max = self.config.max_attempts,
                        error = %e,
                        "embedding provider probe failed"
                    );
                    if consecutive_failures >= self.config.max_attempts {
                        error!(
                            attempts = consecutive_failures,
                            "embedding provider unrecoverable; writing diagnostic and \
                             requesting controlled shutdown"
                        );
                        self.write_diagnostic(consecutive_failures, &e.to_string());
                        self.shutdown_request.cancel();
                        return;
                    }
                }
            }
        }
    }

    /// Pick the wait before the next probe. Index 0 (healthy) uses the longest
    /// interval as a light backstop; failures step through the escalation.
    fn interval_for(&self, consecutive_failures: u32) -> Duration {
        let len = self.config.retry_intervals.len();
        let idx = if consecutive_failures == 0 {
            len - 1
        } else {
            ((consecutive_failures - 1) as usize).min(len - 1)
        };
        self.config.retry_intervals[idx]
    }

    fn write_diagnostic(&self, attempts: u32, last_error: &str) {
        let diag = FailureDiagnostic {
            timestamp: Utc::now().to_rfc3339(),
            daemon_start: self.daemon_start.to_rfc3339(),
            total_attempts: attempts,
            last_error: last_error.to_string(),
            provider_label: self.provider.provider_label().to_string(),
            output_dim: self.provider.output_dim(),
            action: "controlled_shutdown",
        };
        let json = match serde_json::to_string_pretty(&diag) {
            Ok(j) => j,
            Err(e) => {
                error!(error = %e, "failed to serialize embedding failure diagnostic");
                return;
            }
        };
        if let Some(parent) = self.config.diagnostic_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&self.config.diagnostic_path, json) {
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
        let wd = EmbeddingWatchdog::new(
            ScriptedProvider::new([], true),
            cfg,
            CancellationToken::new(),
        );
        // Healthy state uses the longest (backstop) interval.
        assert_eq!(wd.interval_for(0), Duration::from_secs(600));
        // Failures step through the escalation and hold at the last value.
        assert_eq!(wd.interval_for(1), Duration::from_secs(30));
        assert_eq!(wd.interval_for(2), Duration::from_secs(60));
        assert_eq!(wd.interval_for(5), Duration::from_secs(600));
        assert_eq!(wd.interval_for(99), Duration::from_secs(600));
    }

    #[tokio::test(start_paused = true)]
    async fn gives_up_after_max_attempts() {
        let dir = std::env::temp_dir().join(format!("wqm-wd-{}", std::process::id()));
        let diag = dir.join("embedding-failure.json");
        let _ = std::fs::remove_file(&diag);

        let provider = ScriptedProvider::new([], false); // every probe fails
        let shutdown = CancellationToken::new();
        let wd = EmbeddingWatchdog::new(provider, fast_config(3, diag.clone()), shutdown.clone());
        let health = wd.health();

        let handle = tokio::spawn(wd.run(CancellationToken::new()));
        // Drive the paused clock far past the (tiny) intervals.
        tokio::time::advance(Duration::from_secs(1)).await;
        handle.await.expect("watchdog task must finish on give-up");

        assert!(shutdown.is_cancelled(), "give-up must request shutdown");
        assert!(
            !health.is_available(),
            "state must be unavailable on give-up"
        );
        assert!(diag.exists(), "diagnostic file must be written");
        let body = std::fs::read_to_string(&diag).unwrap();
        assert!(body.contains("controlled_shutdown"));
        assert!(body.contains("\"total_attempts\": 3"));
        let _ = std::fs::remove_file(&diag);
    }

    #[tokio::test(start_paused = true)]
    async fn recovers_without_shutdown() {
        let provider = ScriptedProvider::new([false, false], true); // fail twice, then heal
        let shutdown = CancellationToken::new();
        let loop_cancel = CancellationToken::new();
        let wd = EmbeddingWatchdog::new(
            provider,
            fast_config(10, std::env::temp_dir().join("wqm-wd-recover.json")),
            shutdown.clone(),
        );
        let health = wd.health();
        let handle = tokio::spawn(wd.run(loop_cancel.clone()));

        // Let several intervals elapse so the two failures and a success run.
        for _ in 0..6 {
            tokio::time::advance(Duration::from_secs(700)).await;
            tokio::task::yield_now().await;
        }

        assert!(health.is_available(), "provider should have recovered");
        assert!(
            !shutdown.is_cancelled(),
            "recovery must not request shutdown"
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
            CancellationToken::new(),
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
