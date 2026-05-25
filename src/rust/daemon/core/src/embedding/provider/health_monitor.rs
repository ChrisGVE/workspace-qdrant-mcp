//! Background health probe loop for the active dense embedding provider.
//!
//! Spawned once at daemon startup (`memexd/src/main.rs`, Phase 5) and runs
//! until the daemon's shutdown `CancellationToken` fires. Each tick calls
//! `provider.probe()`; the probe is responsible for both reachability checks
//! and any output-dim drift updates (the monitor never writes to the
//! provider's atomic itself).
//!
//! Shutdown is cooperative: the loop awaits `cancel.cancelled()` alongside
//! the next interval tick using `tokio::select!`, so the task exits within
//! the duration of one probe call after `cancel.cancel()` is invoked.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::DenseProvider;
use crate::embedding::types::EmbeddingError;

/// Default monitor tick interval (seconds). Override via `new()` argument.
pub const DEFAULT_PROBE_INTERVAL_SECS: u64 = 30;

/// Shared probe result cache written by `ProviderHealthMonitor` and read by
/// gRPC health endpoints. Lives in `core` so both crates can reference it.
#[derive(Debug, Default)]
pub struct SharedProbeCache {
    pub last_probe_at: Option<Instant>,
    pub last_result: Option<Result<(), EmbeddingError>>,
}

impl SharedProbeCache {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }
}

/// Owns the probe schedule for a single active provider.
#[derive(Debug)]
pub struct ProviderHealthMonitor {
    provider: Arc<dyn DenseProvider>,
    interval: Duration,
    cache: Arc<Mutex<SharedProbeCache>>,
}

impl ProviderHealthMonitor {
    pub fn new(
        provider: Arc<dyn DenseProvider>,
        interval: Duration,
        cache: Arc<Mutex<SharedProbeCache>>,
    ) -> Self {
        Self {
            provider,
            interval,
            cache,
        }
    }

    /// Run the probe loop. Terminates cleanly when `cancel` is cancelled.
    pub async fn run(self, cancel: CancellationToken) {
        let mut ticker = interval(self.interval);
        // First tick fires immediately; skip it so the daemon comes up
        // before issuing any network probe.
        ticker.tick().await;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("ProviderHealthMonitor: shutdown signal received");
                    return;
                }
                _ = ticker.tick() => {
                    let result = self.provider.probe().await;
                    match &result {
                        Ok(()) => {
                            debug!(
                                provider = self.provider.provider_label(),
                                output_dim = self.provider.output_dim(),
                                "embedding provider probe succeeded"
                            );
                        }
                        Err(e) => {
                            warn!(
                                provider = self.provider.provider_label(),
                                error = %e,
                                "embedding provider probe failed"
                            );
                        }
                    }
                    let mut cache = self.cache.lock().await;
                    cache.last_probe_at = Some(Instant::now());
                    cache.last_result = Some(result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::embedding::types::{DenseEmbedding, EmbeddingError};

    /// Trivial provider whose `probe` returns immediately; used to verify
    /// the monitor loop terminates on cancellation without performing any
    /// actual network I/O.
    #[derive(Debug)]
    struct StubProvider {
        probe_calls: AtomicUsize,
    }

    impl StubProvider {
        fn new() -> Self {
            Self {
                probe_calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl DenseProvider for StubProvider {
        async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(Vec::new())
        }
        fn output_dim(&self) -> usize {
            1
        }
        fn provider_label(&self) -> &str {
            "stub"
        }
        fn metrics_label(&self) -> &'static str {
            "openai_compatible_other"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            self.probe_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_health_monitor_exits_on_cancel() {
        let provider = Arc::new(StubProvider::new());
        let cache = SharedProbeCache::new();
        let monitor = ProviderHealthMonitor::new(
            provider.clone() as Arc<dyn DenseProvider>,
            Duration::from_millis(50),
            cache,
        );
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(monitor.run(cancel_clone));

        // Cancel almost immediately and verify the spawned task winds up
        // within a generous bound. Use timeout to keep the test deterministic
        // even if the cooperative cancel is racing the interval tick.
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("monitor must exit promptly after cancel")
            .expect("monitor task must not panic");
    }

    #[tokio::test]
    async fn test_health_monitor_writes_cache() {
        let provider = Arc::new(StubProvider::new());
        let cache = SharedProbeCache::new();
        let cache_clone = Arc::clone(&cache);
        let monitor = ProviderHealthMonitor::new(
            provider.clone() as Arc<dyn DenseProvider>,
            Duration::from_millis(20),
            cache,
        );
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(monitor.run(cancel_clone));

        // Wait enough for at least one probe tick
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel.cancel();
        let _ = handle.await;

        let lock = cache_clone.lock().await;
        assert!(lock.last_probe_at.is_some());
        assert!(lock.last_result.is_some());
        assert!(lock.last_result.as_ref().unwrap().is_ok());
        assert!(provider.probe_calls.load(Ordering::Relaxed) >= 1);
    }
}
