//! Service instantiation and server startup logic.
//!
//! Extracts the `start()` method from `GrpcServer`, splitting it into
//! focused helpers: service wiring, TLS configuration, and server launch.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic::{Request, Status};
use workspace_qdrant_core::storage::{StorageClient, StorageConfig};
use workspace_qdrant_core::{LanguageServerManager, ProjectLspConfig};

use crate::auth::AuthInterceptor;
use crate::services::{
    AdminWriteServiceImpl, CollectionServiceImpl, DocumentServiceImpl, EmbeddingServiceImpl,
    LibraryWriteServiceImpl, ProjectServiceImpl, QueueWriteServiceImpl, SystemServiceImpl,
    TextSearchServiceImpl, TrackingWriteServiceImpl, WatchWriteServiceImpl,
};
use crate::{GrpcError, GrpcServer, ServerConfig};

/// Build a cloneable auth-checking closure from the server configuration.
///
/// The returned closure is `Clone` (the inner `AuthInterceptor` is `Clone`) and
/// applies the API-key / origin checks defined in `auth.rs`. Always wrapping
/// every service with this closure — even when auth is disabled — keeps the
/// `tonic::Router` type identical in both modes; the closure short-circuits
/// to `Ok` when no auth is configured.
pub fn make_auth_fn(
    config: &ServerConfig,
) -> impl FnMut(Request<()>) -> Result<Request<()>, Status> + Clone {
    let interceptor = AuthInterceptor::new(config.auth_config.clone());
    move |req: Request<()>| -> Result<Request<()>, Status> {
        interceptor.check(&req)?;
        Ok(req)
    }
}

type LayerStack =
    tower::layer::util::Stack<crate::metrics_layer::MetricsLayer, tower::layer::util::Identity>;

impl GrpcServer {
    pub async fn start(&mut self) -> Result<(), GrpcError> {
        log_debug("GrpcServer::start() called");

        let storage_client = self.create_storage_client();
        let system_service = self.build_system_service(&storage_client);
        let collection_service = CollectionServiceImpl::new(Arc::clone(&storage_client));
        let dense_provider = self.dense_provider.clone().ok_or_else(|| {
            GrpcError::Configuration(
                "GrpcServer requires a DenseProvider — call .with_dense_provider(...) before start()".to_string(),
            )
        })?;
        let document_service =
            DocumentServiceImpl::new(Arc::clone(&storage_client), Arc::clone(&dense_provider));
        let embedding_service = EmbeddingServiceImpl::new(dense_provider);
        let project_service = self.build_project_service(&storage_client).await;

        self.log_startup_info();

        let server_builder = self.configure_transport().await?;
        self.serve(
            server_builder,
            system_service,
            collection_service,
            document_service,
            embedding_service,
            project_service,
            storage_client,
        )
        .await
    }

    pub fn get_metrics(&self) -> Arc<crate::ServerMetrics> {
        Arc::clone(&self.metrics)
    }

    fn create_storage_client(&self) -> Arc<StorageClient> {
        log_debug("About to create StorageClient with daemon_mode()");
        Arc::new(StorageClient::with_config(StorageConfig::daemon_mode()))
    }

    fn build_system_service(&self, storage_client: &Arc<StorageClient>) -> SystemServiceImpl {
        let mut svc = if let Some(pool) = self.db_pool.as_ref() {
            SystemServiceImpl::new().with_database_pool(pool.clone())
        } else {
            SystemServiceImpl::new()
        };

        if let Some(flag) = &self.pause_flag {
            svc = svc.with_pause_flag(Arc::clone(flag));
        }
        if let Some(ref signal) = self.watch_refresh_signal {
            svc = svc.with_watch_refresh_signal(Arc::clone(signal));
        }
        if let Some(ref health) = self.queue_health {
            svc = svc.with_queue_health(Arc::clone(health));
        }
        if let Some(ref ewma_state) = self.ewma_state {
            svc = svc.with_ewma_state(Arc::clone(ewma_state));
        }
        if let Some(ref adaptive_state) = self.adaptive_state {
            svc = svc.with_adaptive_state(Arc::clone(adaptive_state));
        }
        if let Some(ref hierarchy_builder) = self.hierarchy_builder {
            svc = svc.with_hierarchy_builder(Arc::clone(hierarchy_builder));
        }
        if let Some(ref search_db) = self.search_db {
            svc = svc.with_search_db(Arc::clone(search_db));
        }
        if let Some(ref lexicon_manager) = self.lexicon_manager {
            svc = svc.with_lexicon_manager(Arc::clone(lexicon_manager));
        }
        // Use externally provided storage client or fall back to the local one
        if let Some(ref sc) = self.storage_client {
            svc = svc.with_storage_client(Arc::clone(sc));
        } else {
            svc = svc.with_storage_client(Arc::clone(storage_client));
        }

        if let Some(ref provider) = self.dense_provider {
            svc = svc.with_dense_provider(Arc::clone(provider));
        }
        if let Some(ref settings) = self.embedding_settings {
            svc = svc.with_embedding_settings(Arc::clone(settings));
        }
        if let Some(ref cache) = self.probe_cache {
            svc = svc.with_probe_cache(Arc::clone(cache));
        }

        svc
    }

    async fn build_project_service(
        &mut self,
        storage_client: &Arc<StorageClient>,
    ) -> Option<ProjectServiceImpl> {
        let pool = self.db_pool.as_ref()?;

        tracing::info!("Creating ProjectService with database pool");

        let svc = self.create_project_service_impl(pool.clone()).await;

        // Wire watch refresh signal
        let svc = if let Some(ref signal) = self.watch_refresh_signal {
            svc.map(|s| s.with_watch_refresh_signal(Arc::clone(signal)))
        } else {
            svc
        };

        // Wire storage client for DeleteProject
        svc.map(|s| s.with_storage(Arc::clone(storage_client)))
    }

    async fn create_project_service_impl(
        &mut self,
        pool: sqlx::SqlitePool,
    ) -> Option<ProjectServiceImpl> {
        // Use external LSP manager if provided
        if let Some(lsp_manager) = self.lsp_manager.take() {
            tracing::info!("Using external LSP manager for ProjectService");
            return Some(ProjectServiceImpl::with_lsp_manager(pool, lsp_manager));
        }

        // Create internal LSP manager if enabled
        if self.enable_lsp {
            return self.create_with_internal_lsp(pool).await;
        }

        Some(ProjectServiceImpl::new(pool))
    }

    async fn create_with_internal_lsp(&self, pool: sqlx::SqlitePool) -> Option<ProjectServiceImpl> {
        match LanguageServerManager::new(ProjectLspConfig::default()).await {
            Ok(mut lsp_manager) => {
                if let Err(e) = lsp_manager.initialize().await {
                    tracing::warn!("Failed to initialize LSP manager: {}", e);
                }
                let lsp_manager = Arc::new(RwLock::new(lsp_manager));
                tracing::info!("LSP lifecycle management enabled for ProjectService (internal)");
                Some(ProjectServiceImpl::with_lsp_manager(pool, lsp_manager))
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to create LSP manager, continuing without LSP: {}",
                    e
                );
                Some(ProjectServiceImpl::new(pool))
            }
        }
    }

    /// Get or spawn a WriteActorHandle.
    ///
    /// If an external handle was injected via `with_write_actor`, use it.
    /// Otherwise, spawn a new WriteActor from the db_pool.
    ///
    /// **Note**: The fallback path spawns a new, untracked WriteActor each call.
    /// This method must only be called once during `serve()`. The injected handle
    /// path (`with_write_actor`) is preferred for production use.
    fn resolve_write_actor(&self) -> Option<workspace_qdrant_core::write_actor::WriteActorHandle> {
        if let Some(ref handle) = self.write_actor {
            return Some(handle.clone());
        }
        // Fallback: spawn a WriteActor from the pool (single-call only)
        let pool = self.db_pool.as_ref()?;
        let handle = workspace_qdrant_core::write_actor::WriteActor::spawn(pool.clone());
        Some(handle)
    }

    fn log_startup_info(&self) {
        tracing::info!("Starting gRPC server on {}", self.config.bind_addr);
        let auth_enabled = self
            .config
            .auth_config
            .as_ref()
            .map(|a| a.enabled)
            .unwrap_or(false);
        tracing::info!(
            "gRPC server configuration: TLS={}, Auth={}, Timeouts={:?}",
            self.config.tls_config.is_some(),
            auth_enabled,
            self.config.timeout_config
        );

        // Surface the chosen auth mode unambiguously so operators can tell at
        // a glance whether the auth interceptor is enforcing API keys.
        if auth_enabled {
            tracing::info!(
                "gRPC auth mode: SECURE — auth interceptor enforcing API key on all services"
            );
        } else {
            tracing::warn!(
                "gRPC auth mode: INSECURE — auth interceptor wired but disabled; all requests pass"
            );
        }

        log_security_warnings(&self.config);
    }

    async fn configure_transport(
        &self,
    ) -> Result<tonic::transport::server::Server<LayerStack>, GrpcError> {
        let mut server_builder = Server::builder()
            .timeout(self.config.timeout_config.request_timeout)
            .concurrency_limit_per_connection(
                self.config.performance_config.max_concurrent_streams as usize,
            )
            .tcp_nodelay(self.config.performance_config.tcp_nodelay);

        if let Some(tls_config) = &self.config.tls_config {
            server_builder = apply_tls(server_builder, tls_config).await?;
        }

        // Install the Prometheus metrics tower layer so every RPC is counted
        // and timed via the shared DaemonMetrics registry.
        Ok(server_builder.layer(crate::metrics_layer::MetricsLayer))
    }

    #[allow(clippy::too_many_arguments)]
    async fn serve(
        &mut self,
        mut server_builder: tonic::transport::server::Server<LayerStack>,
        system_service: SystemServiceImpl,
        collection_service: CollectionServiceImpl,
        document_service: DocumentServiceImpl,
        embedding_service: EmbeddingServiceImpl,
        project_service: Option<ProjectServiceImpl>,
        local_storage_client: Arc<StorageClient>,
    ) -> Result<(), GrpcError> {
        use crate::proto;

        // Build the auth-checking closure once and clone it into each service
        // wrap. When auth is disabled or unconfigured, the closure is a pass-
        // through, so wrapping is type-uniform across modes.
        let auth_fn = make_auth_fn(&self.config);

        let mut router = server_builder
            .add_service(InterceptedService::new(
                proto::system_service_server::SystemServiceServer::new(system_service),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::collection_service_server::CollectionServiceServer::new(collection_service),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::document_service_server::DocumentServiceServer::new(document_service),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::embedding_service_server::EmbeddingServiceServer::new(embedding_service),
                auth_fn.clone(),
            ));

        if let Some(project_svc_impl) = project_service {
            project_svc_impl.start_deferred_shutdown_monitor();
            tracing::info!(
                "Registering ProjectService gRPC endpoint with deferred shutdown monitor"
            );
            router = router.add_service(InterceptedService::new(
                proto::project_service_server::ProjectServiceServer::new(project_svc_impl),
                auth_fn.clone(),
            ));
        }
        if let Some(search_db) = self.search_db.take() {
            tracing::info!("Registering TextSearchService gRPC endpoint");
            let mut text_search_svc = TextSearchServiceImpl::new(search_db);
            // State pool enables tenant index_status in responses (#97).
            if let Some(pool) = self.db_pool.clone() {
                text_search_svc = text_search_svc.with_state_pool(pool);
            }
            router = router.add_service(InterceptedService::new(
                proto::text_search_service_server::TextSearchServiceServer::new(text_search_svc),
                auth_fn.clone(),
            ));
        }
        if let Some(graph_store) = self.graph_store.take() {
            tracing::info!("Registering GraphService gRPC endpoint");
            router = router.add_service(InterceptedService::new(
                proto::graph_service_server::GraphServiceServer::new(
                    crate::services::GraphServiceImpl::new(graph_store, self.graph_sqlite.take()),
                ),
                auth_fn.clone(),
            ));
        }
        if let Some(language_manager) = self.language_manager.take() {
            tracing::info!("Registering LanguageService gRPC endpoint");
            router = router.add_service(InterceptedService::new(
                proto::language_service_server::LanguageServiceServer::new(
                    crate::services::LanguageServiceImpl::new(language_manager),
                ),
                auth_fn.clone(),
            ));
        }

        router = self.register_write_services(router, &local_storage_client, auth_fn.clone());

        let addr = self.config.bind_addr;
        match self.shutdown_signal.take() {
            Some(shutdown) => {
                tracing::info!("gRPC server started with graceful shutdown support");
                router
                    .serve_with_shutdown(addr, async {
                        shutdown.await.ok();
                        tracing::info!("gRPC server received shutdown signal");
                    })
                    .await?;
            }
            None => {
                tracing::info!("gRPC server started without shutdown signal");
                router.serve(addr).await?;
            }
        }

        tracing::info!("gRPC server stopped");
        Ok(())
    }

    fn register_write_services<F>(
        &mut self,
        router: tonic::transport::server::Router<LayerStack>,
        local_storage_client: &Arc<StorageClient>,
        auth_fn: F,
    ) -> tonic::transport::server::Router<LayerStack>
    where
        F: FnMut(Request<()>) -> Result<Request<()>, Status> + Clone + Send + Sync + 'static,
    {
        use crate::proto;

        let Some(write_handle) = self.resolve_write_actor() else {
            tracing::warn!(
                "Write services NOT registered: no db_pool or write_actor available. \
                 Queue, watch, library, tracking, and admin mutations will be unavailable."
            );
            return router;
        };

        let mut admin_impl = AdminWriteServiceImpl::new(write_handle.clone());
        if let (Some(settings), Some(provider), Some(pool), Some(pause_flag)) = (
            self.embedding_settings.clone(),
            self.dense_provider.clone(),
            self.db_pool.clone(),
            self.pause_flag.clone(),
        ) {
            let storage_for_reembed = self
                .storage_client
                .clone()
                .unwrap_or_else(|| Arc::clone(local_storage_client));
            let ctx = Arc::new(crate::services::reembed::ReembedContext {
                settings,
                provider,
                storage_client: storage_for_reembed,
                pool,
                pause_flag,
            });
            admin_impl = admin_impl.with_reembed_context(ctx);
            tracing::info!("AdminWriteService TriggerReembed wiring complete");
        } else {
            tracing::warn!(
                "AdminWriteService TriggerReembed NOT wired: missing one of \
                 (embedding_settings, dense_provider, db_pool, pause_flag)"
            );
        }

        // RebalanceIdf engine wiring: Qdrant client + SQLite pool (WI-f1).
        if let Some(pool) = self.db_pool.clone() {
            let storage_for_rebalance = self
                .storage_client
                .clone()
                .unwrap_or_else(|| Arc::clone(local_storage_client));
            admin_impl = admin_impl.with_rebalance_context(storage_for_rebalance, pool);
            tracing::info!("AdminWriteService RebalanceIdf engine wiring complete");
        } else {
            tracing::warn!("AdminWriteService RebalanceIdf NOT wired: missing db_pool");
        }

        tracing::info!(
            "Registered 5 write services via WriteActor for serialized SQLite mutations"
        );
        router
            .add_service(InterceptedService::new(
                proto::queue_write_service_server::QueueWriteServiceServer::new(
                    QueueWriteServiceImpl::new(write_handle.clone()),
                ),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::watch_write_service_server::WatchWriteServiceServer::new(
                    WatchWriteServiceImpl::new(write_handle.clone()),
                ),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::library_write_service_server::LibraryWriteServiceServer::new({
                    let mut svc = LibraryWriteServiceImpl::new(write_handle.clone());
                    // Wire the recover cascade deps (#140): state.db pool +
                    // Qdrant client. Other library RPCs ignore them.
                    if let Some(pool) = self.db_pool.clone() {
                        let storage = self
                            .storage_client
                            .clone()
                            .unwrap_or_else(|| Arc::clone(local_storage_client));
                        svc = svc.with_recover_deps(pool, storage);
                    } else {
                        tracing::warn!(
                            "LibraryWriteService RecoverLibrary NOT wired: missing db_pool"
                        );
                    }
                    svc
                }),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::tracking_write_service_server::TrackingWriteServiceServer::new(
                    TrackingWriteServiceImpl::new(write_handle),
                ),
                auth_fn.clone(),
            ))
            .add_service(InterceptedService::new(
                proto::admin_write_service_server::AdminWriteServiceServer::new(admin_impl),
                auth_fn,
            ))
    }
}

/// Apply TLS configuration to a tonic server builder.
async fn apply_tls(
    server_builder: tonic::transport::server::Server,
    tls_config: &crate::TlsConfig,
) -> Result<tonic::transport::server::Server, GrpcError> {
    let cert = tokio::fs::read(&tls_config.cert_path)
        .await
        .map_err(|e| GrpcError::Configuration(format!("Failed to read TLS certificate: {e}")))?;

    let key = tokio::fs::read(&tls_config.key_path)
        .await
        .map_err(|e| GrpcError::Configuration(format!("Failed to read TLS key: {e}")))?;

    let identity = Identity::from_pem(cert, key);
    let mut tls = ServerTlsConfig::new().identity(identity);

    if let Some(ca_cert_path) = &tls_config.ca_cert_path {
        let ca_cert = tokio::fs::read(ca_cert_path)
            .await
            .map_err(|e| GrpcError::Configuration(format!("Failed to read CA certificate: {e}")))?;
        tls = tls.client_ca_root(tonic::transport::Certificate::from_pem(ca_cert));
    }

    if tls_config.require_client_cert {
        tls = tls.client_auth_optional(false);
    }

    server_builder
        .tls_config(tls)
        .map_err(|e| GrpcError::Configuration(format!("Failed to configure TLS: {e}")))
}

/// Log security warnings for the current server configuration.
fn log_security_warnings(config: &ServerConfig) {
    let warnings = config.get_security_warnings();
    if warnings.is_empty() {
        tracing::info!("gRPC server security configuration validated");
        return;
    }

    tracing::warn!("===== SECURITY WARNINGS =====");
    for warning in &warnings {
        tracing::warn!("  - {}", warning);
    }
    tracing::warn!("============================");
    if !config.is_secure() {
        tracing::error!("gRPC server is running in INSECURE mode - not suitable for production");
    }
}

/// Write a debug message to `/tmp/grpc_debug.log` (best-effort).
fn log_debug(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/grpc_debug.log")
    {
        use std::io::Write;
        let _ = writeln!(f, "{msg}");
    }
}
