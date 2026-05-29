//! Tests for the embedding_provider health component and
//! GetEmbeddingProviderStatus RPC (spec §7.9).

use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use tonic::Request;
use workspace_qdrant_core::config::EmbeddingSettings;
use workspace_qdrant_core::embedding::provider::DenseProvider;
use workspace_qdrant_core::embedding::{DenseEmbedding, EmbeddingError};

use crate::proto::{system_service_server::SystemService, ServiceStatus};

use super::super::service_impl::SystemServiceImpl;

// ─── Stub provider ───────────────────────────────────────────────────────────

/// Stub provider whose `probe()` returns whatever the supplied closure
/// produces. Counts probe invocations so tests can assert on caching.
struct StubProvider {
    result_factory: StdMutex<Box<dyn FnMut() -> Result<(), EmbeddingError> + Send>>,
    probe_calls: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for StubProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StubProvider")
            .field("probe_calls", &self.probe_calls)
            .finish()
    }
}

impl StubProvider {
    fn new<F>(f: F) -> Arc<Self>
    where
        F: FnMut() -> Result<(), EmbeddingError> + Send + 'static,
    {
        Arc::new(Self {
            result_factory: StdMutex::new(Box::new(f)),
            probe_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.probe_calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl DenseProvider for StubProvider {
    async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
        Ok(Vec::new())
    }
    fn output_dim(&self) -> usize {
        1536
    }
    fn provider_label(&self) -> &str {
        "stub-openai"
    }
    fn metrics_label(&self) -> &'static str {
        "openai"
    }
    async fn probe(&self) -> Result<(), EmbeddingError> {
        self.probe_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (self.result_factory.lock().unwrap())()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn settings_with(ttl_secs: u64) -> Arc<EmbeddingSettings> {
    let mut s = EmbeddingSettings::default();
    s.health_probe_cache_secs = ttl_secs;
    s.output_dim = 1536;
    Arc::new(s)
}

fn find_component<'a>(
    components: &'a [crate::proto::ComponentHealth],
    name: &str,
) -> Option<&'a crate::proto::ComponentHealth> {
    components.iter().find(|c| c.component_name == name)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_health_includes_embedding_provider_component() {
    let provider = StubProvider::new(|| Ok(()));
    let svc = SystemServiceImpl::new()
        .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(60));
    // Seed the cache so Health does not report `probe pending`.
    svc.seed_embedding_probe_cache(Ok(())).await;

    let resp = svc.health(Request::new(())).await.unwrap().into_inner();
    let comp = find_component(&resp.components, "embedding_provider")
        .expect("Health response must include embedding_provider component");
    assert_eq!(comp.status, ServiceStatus::Healthy as i32);
    assert!(comp.message.contains("Running"));
}

#[tokio::test]
async fn test_health_embedding_provider_unhealthy_propagates() {
    let provider = StubProvider::new(|| {
        Err(EmbeddingError::RemoteError {
            status_code: 401,
            message: "Unauthorized".to_string(),
        })
    });
    let svc = SystemServiceImpl::new()
        .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(60));
    svc.seed_embedding_probe_cache(Err(EmbeddingError::RemoteError {
        status_code: 401,
        message: "Unauthorized".to_string(),
    }))
    .await;

    let resp = svc.health(Request::new(())).await.unwrap().into_inner();
    let comp = find_component(&resp.components, "embedding_provider").unwrap();
    assert_eq!(comp.status, ServiceStatus::Unhealthy as i32);
    assert!(
        comp.message.contains("auth failure"),
        "expected auth-failure message, got: {}",
        comp.message
    );

    // Overall health must reflect the unhealthy component.
    assert_eq!(resp.status, ServiceStatus::Unhealthy as i32);
}

#[tokio::test]
async fn test_health_probe_cached_within_ttl() {
    let provider = StubProvider::new(|| Ok(()));
    let svc = SystemServiceImpl::new()
        .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(60));

    // Two on-demand status RPCs in quick succession should produce only
    // one underlying probe call thanks to the TTL cache.
    let _ = svc
        .get_embedding_provider_status(Request::new(()))
        .await
        .unwrap();
    let _ = svc
        .get_embedding_provider_status(Request::new(()))
        .await
        .unwrap();
    assert_eq!(
        provider.calls(),
        1,
        "second status call within TTL must reuse cached probe; got {} probes",
        provider.calls()
    );
}

#[tokio::test]
async fn test_health_embedding_provider_probe_pending() {
    // Provider wired but no probe has run yet — Health must report
    // Degraded with a `probe pending` message, not block on a probe.
    let provider = StubProvider::new(|| Ok(()));
    let svc = SystemServiceImpl::new()
        .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(60));

    let resp = svc.health(Request::new(())).await.unwrap().into_inner();
    let comp = find_component(&resp.components, "embedding_provider").unwrap();
    assert_eq!(comp.status, ServiceStatus::Degraded as i32);
    assert!(
        comp.message.contains("probe pending"),
        "got: {}",
        comp.message
    );
    assert_eq!(
        provider.calls(),
        0,
        "Health must not issue a probe when the cache is empty"
    );
}

#[tokio::test]
async fn test_background_probe_loop_seeds_health_cache() {
    // The background loop is what turns "probe pending" into the provider's
    // real state — without anyone calling GetEmbeddingProviderStatus.
    let provider = StubProvider::new(|| Ok(()));
    let svc = SystemServiceImpl::new()
        .with_dense_provider(provider.clone() as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(0));

    // Baseline: empty cache reports `probe pending`.
    let resp = svc.health(Request::new(())).await.unwrap().into_inner();
    let comp = find_component(&resp.components, "embedding_provider").unwrap();
    assert_eq!(comp.status, ServiceStatus::Degraded as i32);
    assert!(comp.message.contains("probe pending"));

    // Start the loop (first tick fires immediately) and wait for it to seed.
    let _handle = svc
        .spawn_embedding_health_probe_loop(std::time::Duration::from_secs(30))
        .expect("loop must spawn when a provider is wired");

    let mut became_healthy = false;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let resp = svc.health(Request::new(())).await.unwrap().into_inner();
        let comp = find_component(&resp.components, "embedding_provider").unwrap();
        if comp.status == ServiceStatus::Healthy as i32 {
            assert!(comp.message.contains("Running"));
            became_healthy = true;
            break;
        }
    }
    assert!(
        became_healthy,
        "background probe loop should seed the cache to Healthy"
    );
    assert!(
        provider.calls() >= 1,
        "the loop must have probed the provider"
    );
}

#[tokio::test]
async fn test_probe_timeout_records_degraded_not_pending() {
    // A probe that never returns within the 3s timeout must leave a real
    // (degraded) result in the cache, NOT None — otherwise Health would read
    // back "probe pending" forever for a persistently-timing-out provider.
    #[derive(Debug)]
    struct SlowProvider;
    #[async_trait]
    impl DenseProvider for SlowProvider {
        async fn embed(&self, _t: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(Vec::new())
        }
        fn output_dim(&self) -> usize {
            1536
        }
        fn provider_label(&self) -> &str {
            "slow-stub"
        }
        fn metrics_label(&self) -> &'static str {
            "openai"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            Ok(())
        }
    }

    let svc = SystemServiceImpl::new()
        .with_dense_provider(Arc::new(SlowProvider) as Arc<dyn DenseProvider>)
        .with_embedding_settings(settings_with(0));

    // Inline probe hits the 3s timeout and returns a degraded status.
    let (status, _msg) = svc.probe_embedding_provider().await;
    assert_eq!(status, "degraded");

    // Crucially, the cache now holds a real result — Health no longer pends.
    let resp = svc.health(Request::new(())).await.unwrap().into_inner();
    let comp = find_component(&resp.components, "embedding_provider").unwrap();
    assert_eq!(comp.status, ServiceStatus::Degraded as i32);
    assert!(
        !comp.message.contains("probe pending"),
        "timeout must surface as degraded, not probe-pending; got: {}",
        comp.message
    );
}
