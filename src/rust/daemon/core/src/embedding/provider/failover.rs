//! `FailoverDenseProvider` — preferred endpoint with explicit, configured
//! fallback.
//!
//! Wraps two providers serving the SAME model (e.g. a GPU embedding server as
//! the preferred endpoint and a CPU server as standby). Every call tries the
//! primary first; on any error it logs a WARN, memoizes the failure for
//! `retry_after` (so a dead primary is not re-dialed on every request), and
//! serves from the fallback. After the memo expires the primary is retried —
//! recovery is automatic, no restart needed.
//!
//! This is NOT the "silent fallback to a different provider" the
//! `DenseProvider` contract forbids: both endpoints MUST serve the same
//! model/dimensionality (vectors are model-bound, not server-bound), the
//! fallback is explicitly configured (`embedding.fallback_base_url`), and
//! every switchover is logged.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{info, warn};

use super::DenseProvider;
use crate::embedding::types::{DenseEmbedding, EmbeddingError};

/// How long a primary failure is memoized before the primary is retried.
const PRIMARY_RETRY_SECS: u64 = 60;

/// Preferred + fallback dense provider pair (same model on both endpoints).
pub struct FailoverDenseProvider {
    primary: Arc<dyn DenseProvider>,
    fallback: Arc<dyn DenseProvider>,
    /// While `Some(t)` and `now < t`, skip the primary without dialing it.
    primary_down_until: Mutex<Option<Instant>>,
    retry_after: Duration,
    label: String,
}

impl std::fmt::Debug for FailoverDenseProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailoverDenseProvider")
            .field("primary", &self.primary.provider_label())
            .field("fallback", &self.fallback.provider_label())
            .field("retry_after", &self.retry_after)
            .finish_non_exhaustive()
    }
}

impl FailoverDenseProvider {
    pub fn new(primary: Arc<dyn DenseProvider>, fallback: Arc<dyn DenseProvider>) -> Self {
        Self::with_retry_after(primary, fallback, Duration::from_secs(PRIMARY_RETRY_SECS))
    }

    /// Constructor with an explicit memo duration (used by tests).
    pub fn with_retry_after(
        primary: Arc<dyn DenseProvider>,
        fallback: Arc<dyn DenseProvider>,
        retry_after: Duration,
    ) -> Self {
        let label = format!(
            "{} (fallback: {})",
            primary.provider_label(),
            fallback.provider_label()
        );
        Self {
            primary,
            fallback,
            primary_down_until: Mutex::new(None),
            retry_after,
            label,
        }
    }

    /// True when the primary should be attempted (no active down-memo).
    fn primary_eligible(&self) -> bool {
        let guard = self.primary_down_until.lock().expect("memo lock poisoned");
        match *guard {
            Some(until) => Instant::now() >= until,
            None => true,
        }
    }

    fn memoize_primary_down(&self) {
        let mut guard = self.primary_down_until.lock().expect("memo lock poisoned");
        *guard = Some(Instant::now() + self.retry_after);
    }

    fn clear_primary_down(&self) {
        let mut guard = self.primary_down_until.lock().expect("memo lock poisoned");
        if guard.take().is_some() {
            info!(
                provider = %self.primary.provider_label(),
                "Primary embedding endpoint recovered — leaving fallback"
            );
        }
    }
}

#[async_trait]
impl DenseProvider for FailoverDenseProvider {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
        if self.primary_eligible() {
            match self.primary.embed(texts).await {
                Ok(out) => {
                    self.clear_primary_down();
                    return Ok(out);
                }
                Err(e) => {
                    warn!(
                        primary = %self.primary.provider_label(),
                        fallback = %self.fallback.provider_label(),
                        retry_after_secs = self.retry_after.as_secs(),
                        error = %e,
                        "Primary embedding endpoint failed — switching to fallback"
                    );
                    self.memoize_primary_down();
                }
            }
        }
        self.fallback.embed(texts).await
    }

    fn output_dim(&self) -> usize {
        // Both endpoints serve the same model, so the dims agree; report the
        // one we are currently routing to so probe-driven drift updates win.
        if self.primary_eligible() {
            self.primary.output_dim()
        } else {
            self.fallback.output_dim()
        }
    }

    fn provider_label(&self) -> &str {
        &self.label
    }

    fn metrics_label(&self) -> &'static str {
        self.primary.metrics_label()
    }

    async fn probe(&self) -> Result<(), EmbeddingError> {
        match self.primary.probe().await {
            Ok(()) => {
                self.clear_primary_down();
                Ok(())
            }
            Err(primary_err) => {
                warn!(
                    primary = %self.primary.provider_label(),
                    error = %primary_err,
                    "Primary embedding endpoint probe failed — probing fallback"
                );
                self.memoize_primary_down();
                self.fallback.probe().await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct MockProvider {
        name: &'static str,
        dim: usize,
        fail: AtomicBool,
        embed_calls: AtomicUsize,
    }

    impl MockProvider {
        fn new(name: &'static str, fail: bool) -> Arc<Self> {
            Arc::new(Self {
                name,
                dim: 4,
                fail: AtomicBool::new(fail),
                embed_calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl DenseProvider for MockProvider {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(EmbeddingError::GenerationError {
                    message: format!("{} down", self.name),
                });
            }
            Ok(texts
                .iter()
                .map(|t| {
                    let mut vector = vec![0.0_f32; self.dim];
                    vector[0] = 1.0;
                    DenseEmbedding {
                        vector,
                        model_name: self.name.to_string(),
                        sequence_length: t.len(),
                    }
                })
                .collect())
        }

        fn output_dim(&self) -> usize {
            self.dim
        }

        fn provider_label(&self) -> &str {
            self.name
        }

        fn metrics_label(&self) -> &'static str {
            "openai_compatible_other"
        }

        async fn probe(&self) -> Result<(), EmbeddingError> {
            if self.fail.load(Ordering::SeqCst) {
                Err(EmbeddingError::GenerationError {
                    message: format!("{} down", self.name),
                })
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn healthy_primary_serves_and_fallback_is_untouched() {
        let primary = MockProvider::new("gpu", false);
        let fallback = MockProvider::new("cpu", false);
        let failover = FailoverDenseProvider::new(primary.clone(), fallback.clone());

        let out = failover.embed(&["a", "b"]).await.expect("embed ok");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].model_name, "gpu");
        assert_eq!(primary.embed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback.embed_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn failed_primary_switches_to_fallback_and_is_memoized() {
        let primary = MockProvider::new("gpu", true);
        let fallback = MockProvider::new("cpu", false);
        let failover = FailoverDenseProvider::with_retry_after(
            primary.clone(),
            fallback.clone(),
            Duration::from_secs(3600),
        );

        let out = failover.embed(&["a"]).await.expect("fallback serves");
        assert_eq!(out[0].model_name, "cpu");
        assert_eq!(primary.embed_calls.load(Ordering::SeqCst), 1);

        // Second call within the memo window must NOT re-dial the primary.
        let out = failover.embed(&["b"]).await.expect("fallback serves");
        assert_eq!(out[0].model_name, "cpu");
        assert_eq!(primary.embed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback.embed_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn primary_is_retried_after_memo_expires_and_recovers() {
        let primary = MockProvider::new("gpu", true);
        let fallback = MockProvider::new("cpu", false);
        let failover = FailoverDenseProvider::with_retry_after(
            primary.clone(),
            fallback.clone(),
            Duration::ZERO,
        );

        let out = failover.embed(&["a"]).await.expect("fallback serves");
        assert_eq!(out[0].model_name, "cpu");

        // Primary comes back; with an expired memo it is retried and wins.
        primary.fail.store(false, Ordering::SeqCst);
        let out = failover.embed(&["b"]).await.expect("primary serves");
        assert_eq!(out[0].model_name, "gpu");
        assert_eq!(primary.embed_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn both_down_propagates_fallback_error() {
        let primary = MockProvider::new("gpu", true);
        let fallback = MockProvider::new("cpu", true);
        let failover = FailoverDenseProvider::new(primary, fallback);

        let err = failover.embed(&["a"]).await.expect_err("both down");
        assert!(err.to_string().contains("cpu down"));
    }

    #[tokio::test]
    async fn probe_is_healthy_when_only_fallback_responds() {
        let primary = MockProvider::new("gpu", true);
        let fallback = MockProvider::new("cpu", false);
        let failover = FailoverDenseProvider::new(primary, fallback.clone());

        failover.probe().await.expect("fallback probe ok");
        // And the memo now routes embeds straight to the fallback.
        let out = failover.embed(&["a"]).await.expect("fallback serves");
        assert_eq!(out[0].model_name, "cpu");
    }
}
