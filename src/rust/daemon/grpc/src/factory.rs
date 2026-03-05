//! Service instantiation and server startup logic.
//!
//! Extracts the `start()` method from `GrpcServer`, splitting it into
//! focused helpers: service wiring, TLS configuration, and server launch.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use workspace_qdrant_core::storage::{StorageClient, StorageConfig};
use workspace_qdrant_core::{LanguageServerManager, ProjectLspConfig};

use crate::services::{
    CollectionServiceImpl, DocumentServiceImpl, EmbeddingServiceImpl, ProjectServiceImpl,
    SystemServiceImpl, TextSearchServiceImpl,
};
use crate::{GrpcError, GrpcServer, ServerConfig};

impl GrpcServer {
    pub async fn start(&mut self) -> Result<(), GrpcError> {
        log_debug("GrpcServer::start() called");

        let storage_client = self.create_storage_client();
        let system_service = self.build_system_service(&storage_client);
        let collection_service = CollectionServiceImpl::new(Arc::clone(&storage_client));
        let document_service = DocumentServiceImpl::new(Arc::clone(&storage_client));
        let embedding_service = EmbeddingServiceImpl::new();
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

    fn log_startup_info(&self) {
        tracing::info!("Starting gRPC server on {}", self.config.bind_addr);
        tracing::info!(
            "gRPC server configuration: TLS={}, Auth={}, Timeouts={:?}",
            self.config.tls_config.is_some(),
            self.config
                .auth_config
                .as_ref()
                .map(|a| a.enabled)
                .unwrap_or(false),
            self.config.timeout_config
        );

        log_security_warnings(&self.config);
    }

    async fn configure_transport(&self) -> Result<tonic::transport::server::Server, GrpcError> {
        let mut server_builder = Server::builder()
            .timeout(self.config.timeout_config.request_timeout)
            .concurrency_limit_per_connection(
                self.config.performance_config.max_concurrent_streams as usize,
            )
            .tcp_nodelay(self.config.performance_config.tcp_nodelay);

        if let Some(tls_config) = &self.config.tls_config {
            server_builder = apply_tls(server_builder, tls_config).await?;
        }

        Ok(server_builder)
    }

    #[allow(clippy::too_many_arguments)]
    async fn serve(
        &mut self,
        mut server_builder: tonic::transport::server::Server,
        system_service: SystemServiceImpl,
        collection_service: CollectionServiceImpl,
        document_service: DocumentServiceImpl,
        embedding_service: EmbeddingServiceImpl,
        project_service: Option<ProjectServiceImpl>,
    ) -> Result<(), GrpcError> {
        use crate::proto;

        // Register core services
        let system_svc = proto::system_service_server::SystemServiceServer::new(system_service);
        let collection_svc =
            proto::collection_service_server::CollectionServiceServer::new(collection_service);
        let document_svc =
            proto::document_service_server::DocumentServiceServer::new(document_service);
        let embedding_svc =
            proto::embedding_service_server::EmbeddingServiceServer::new(embedding_service);

        let mut router = server_builder
            .add_service(system_svc)
            .add_service(collection_svc)
            .add_service(document_svc)
            .add_service(embedding_svc);

        // Conditionally add ProjectService
        if let Some(project_svc_impl) = project_service {
            project_svc_impl.start_deferred_shutdown_monitor();
            let project_svc =
                proto::project_service_server::ProjectServiceServer::new(project_svc_impl);
            tracing::info!(
                "Registering ProjectService gRPC endpoint with deferred shutdown monitor"
            );
            router = router.add_service(project_svc);
        }

        // Conditionally add TextSearchService
        if let Some(search_db) = self.search_db.take() {
            let text_search_svc_impl = TextSearchServiceImpl::new(search_db);
            let text_search_svc = proto::text_search_service_server::TextSearchServiceServer::new(
                text_search_svc_impl,
            );
            tracing::info!("Registering TextSearchService gRPC endpoint");
            router = router.add_service(text_search_svc);
        }

        // Conditionally add GraphService
        if let Some(graph_store) = self.graph_store.take() {
            let graph_svc_impl = crate::services::GraphServiceImpl::new(graph_store);
            let graph_svc = proto::graph_service_server::GraphServiceServer::new(graph_svc_impl);
            tracing::info!("Registering GraphService gRPC endpoint");
            router = router.add_service(graph_svc);
        }

        // Start server with graceful shutdown
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
