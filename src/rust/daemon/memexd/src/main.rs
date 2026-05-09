//! memexd - Memory eXchange Daemon
//!
//! A daemon process that manages document processing, file watching, and embedding
//! generation for the workspace-qdrant-mcp system.
//!
//! The daemon lifecycle is split into modules by phase:
//! - [`startup`]: CLI parsing, instance check, PID file, config loading
//! - [`database`]: Pool creation, migrations, reconciliation
//! - [`background`]: Periodic maintenance task spawns
//! - [`queue_init`]: Embedding, storage, queue processor setup
//! - [`grpc_setup`]: gRPC server creation and dependency wiring
//! - [`shutdown`]: Graceful shutdown with cleanup

mod background;
mod database;
mod grpc_setup;
mod queue_init;
mod shutdown;
mod startup;
#[cfg(windows)]
mod windows_service;

use std::process;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{debug, error, info};

use workspace_qdrant_core::{config::Config, config::DaemonConfig, HierarchyBuilder, WatchManager};

pub use startup::DaemonArgs;

/// Execute Phase 1: instance check, PID, config conversion, nice level.
/// Returns the resolved `Config` and a scopeguard that removes the PID file on drop.
fn run_phase1(
    args: &DaemonArgs,
    daemon_config: &DaemonConfig,
) -> Result<(Config, impl Drop), Box<dyn std::error::Error>> {
    startup::check_existing_instance(&args.pid_file, args.project_id.as_ref())?;
    startup::create_pid_file(&args.pid_file, args.project_id.as_ref())?;
    startup::check_stale_legacy_directory();

    let mut config = Config::from(daemon_config.clone());
    config.resource_limits.resolve_auto_values();
    startup::set_process_nice_level(config.resource_limits.nice_level);

    let pid_file_cleanup = args.pid_file.clone();
    let cleanup_guard = scopeguard::guard((), move |_| {
        startup::remove_pid_file(&pid_file_cleanup);
    });

    Ok((config, cleanup_guard))
}

/// Wire the gRPC server with all phase-5 dependencies and return its handle.
fn wire_grpc(
    args: &DaemonArgs,
    db_handles: &database::DatabaseHandles,
    qc: &queue_init::QueueComponents,
    lsp_manager: &Option<Arc<tokio::sync::RwLock<workspace_qdrant_core::LanguageServerManager>>>,
    watch_refresh_signal: &Arc<Notify>,
    embedding_settings: Arc<workspace_qdrant_core::config::EmbeddingSettings>,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    grpc_setup::spawn_grpc_server(
        args,
        db_handles.queue_pool.clone(),
        Arc::clone(&db_handles.pause_flag),
        Arc::clone(watch_refresh_signal),
        Arc::clone(&qc.queue_health),
        Arc::clone(&qc.adaptive_state),
        Arc::clone(&db_handles.search_db),
        db_handles.graph_store.clone(),
        lsp_manager.clone(),
        Arc::clone(&qc.hierarchy_builder),
        qc.watch_pool.clone(),
        Arc::clone(&qc.dense_provider),
        embedding_settings,
    )
}

/// Main daemon orchestrator: runs phases 1-7 in sequence.
async fn run_daemon(
    daemon_config: DaemonConfig,
    args: DaemonArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let suffix = args
        .project_id
        .as_deref()
        .map_or(String::new(), |id| format!(" for project {id}"));
    info!(
        "Starting memexd daemon (version {} ({})){suffix}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_NUMBER")
    );

    // Phase 1: Startup
    let (config, _cleanup_guard) = run_phase1(&args, &daemon_config)?;
    // Phase 2: Database
    let db_handles = database::initialize_all(&config).await?;
    // Phase 3: Background tasks
    let prometheus_config = background::resolve_prometheus_config(
        daemon_config.observability.telemetry.prometheus.clone(),
        args.metrics_port,
    );
    let mut bg_handles = background::spawn_all(
        &db_handles.queue_pool,
        &db_handles.pause_flag,
        &prometheus_config,
    );
    // Phase 4: LSP manager
    let lsp_manager = grpc_setup::init_lsp_manager(&daemon_config).await;
    let watch_refresh_signal = Arc::new(Notify::new());

    // Phase 5: IPC + Queue processor
    let _ipc_server = queue_init::setup_ipc_server(&config, &db_handles.queue_pool).await?;
    let mut qc = queue_init::initialize(
        &config,
        &daemon_config,
        db_handles.queue_pool.clone(),
        &db_handles.search_db,
        &db_handles.graph_store,
        &lsp_manager,
        &watch_refresh_signal,
    )
    .await?;

    // Phase 5a: Embedding-dim consistency guard.
    //
    // Refuse to start when the active provider's `output_dim` disagrees with
    // the dim of the existing `projects` collection. `--bootstrap-reembed`
    // suppresses the guard for provider migration. See PRD §6.5.
    let active_dim = daemon_config.embedding.output_dim;
    if let Err(e) = workspace_qdrant_core::specs::check_dim_consistency(
        &qc.mirror_storage,
        active_dim,
        args.bootstrap_reembed,
    )
    .await
    {
        if let workspace_qdrant_core::embedding::EmbeddingError::DimensionMismatch {
            actual_dim,
            stored_dim,
        } = &e
        {
            eprintln!(
                "FATAL: active embedding provider outputs {}-dim vectors but the \
                 'projects' Qdrant collection was created with {}-dim vectors.\n\
                 Run `wqm admin reembed --confirm` to rebuild all collections at \
                 the new dimension.\n\n\
                 If you are in the middle of a provider migration, start the \
                 daemon with:\n  memexd --bootstrap-reembed\n\
                 to suppress this check, run `wqm admin reembed --confirm`, \
                 then restart normally.",
                actual_dim, stored_dim
            );
        }
        return Err(Box::new(e));
    }

    // Phase 5b: Background dense-provider health probe loop.
    let health_monitor = workspace_qdrant_core::embedding::provider::ProviderHealthMonitor::new(
        Arc::clone(&qc.dense_provider),
        std::time::Duration::from_secs(
            workspace_qdrant_core::embedding::provider::health_monitor::DEFAULT_PROBE_INTERVAL_SECS,
        ),
    );
    let health_monitor_token = qc.adaptive_shutdown_token.child_token();
    tokio::spawn(async move { health_monitor.run(health_monitor_token).await });

    // Phase 6: gRPC server + queue start + recovery
    let embedding_settings = Arc::new(daemon_config.embedding.clone());
    bg_handles.grpc_handle = Some(wire_grpc(
        &args,
        &db_handles,
        &qc,
        &lsp_manager,
        &watch_refresh_signal,
        embedding_settings,
    )?);
    queue_init::start_processor(&mut qc.unified_queue_processor, &daemon_config).await?;
    queue_init::spawn_recovery_tasks(
        &qc.unified_queue_processor,
        &qc.allowed_extensions,
        &qc.mirror_storage,
        &daemon_config,
    );

    // gRPC is serving — run the slow ignore-rule reconciliation in the
    // background so projects with many files no longer delay readiness
    // (issue #59).
    let _ignore_reconcile_handle =
        database::spawn_background_reconciliation(db_handles.queue_pool.clone());

    // Phase 6b: File watching + hierarchy
    let watch_manager = start_file_watchers(&qc, &watch_refresh_signal).await;
    spawn_git_event_consumer(&qc.unified_queue_processor, &watch_manager);
    let hierarchy_cancel = tokio_util::sync::CancellationToken::new();
    let _hierarchy_handle =
        Arc::clone(&qc.hierarchy_builder).start_scheduled(hierarchy_cancel.clone());
    info!("Canonical tag hierarchy rebuild scheduled (nightly at 2 AM)");
    spawn_hierarchy_bootstrap(&qc.hierarchy_builder);
    info!(
        "memexd daemon is running. gRPC on port {}, send SIGTERM or SIGINT to stop.",
        args.grpc_port
    );

    // Phase 7: Wait for shutdown signal, then clean up
    if let Err(e) = shutdown::wait_for_signal().await {
        error!("Error in signal handling: {}", e);
    }
    shutdown::stop_watchers(&watch_manager).await;
    shutdown::stop_queue_processor(&mut qc.unified_queue_processor).await;
    shutdown::stop_lsp(lsp_manager).await;
    shutdown::abort_background_tasks(bg_handles, qc.adaptive_shutdown_token, hierarchy_cancel);
    Ok(())
}

/// Start file watchers for all enabled watch folders.
async fn start_file_watchers(
    qc: &queue_init::QueueComponents,
    watch_refresh_signal: &Arc<Notify>,
) -> Arc<WatchManager> {
    let watch_manager = Arc::new(
        WatchManager::new(qc.watch_pool.clone(), Arc::clone(&qc.allowed_extensions))
            .with_refresh_signal(Arc::clone(watch_refresh_signal)),
    );
    if let Err(e) = watch_manager.start_all_watches().await {
        tracing::warn!("Failed to start file watchers (non-fatal): {}", e);
    } else {
        let count = watch_manager.active_watcher_count().await;
        info!("File watchers started: {} active watches", count);
    }
    let _watch_poll_handle = Arc::clone(&watch_manager).start_polling(300);
    watch_manager
}

/// Spawn the git event consumer that processes branch switches and commits.
fn spawn_git_event_consumer(
    uqp: &workspace_qdrant_core::UnifiedQueueProcessor,
    watch_manager: &Arc<WatchManager>,
) {
    let git_pool = uqp.pool().clone();
    let git_qm = uqp.queue_manager().clone();
    let git_watch_manager = Arc::clone(watch_manager);
    tokio::spawn(async move {
        if let Some(mut rx) = git_watch_manager.take_git_event_rx().await {
            info!("Git event consumer started");
            while let Some(event) = rx.recv().await {
                debug!(
                    "Processing git event: {:?} for {}",
                    event.event_type, event.watch_folder_id
                );
                handle_single_git_event(&event, &git_pool, &git_qm).await;
            }
            info!("Git event consumer stopped (channel closed)");
        } else {
            debug!("No git event receiver available");
        }
    });
}

/// Process a single git event and log the outcome.
async fn handle_single_git_event(
    event: &workspace_qdrant_core::git::GitEvent,
    pool: &sqlx::SqlitePool,
    qm: &workspace_qdrant_core::queue_operations::QueueManager,
) {
    match workspace_qdrant_core::branch_switch::handle_git_event(event, pool, qm).await {
        Ok(stats) => {
            let total = stats.batch_updated
                + stats.enqueued_changed
                + stats.enqueued_added
                + stats.enqueued_deleted;
            if total > 0 {
                info!(
                    "Git event {:?} processed: {} batch-updated, \
                     {} changed, {} added, {} deleted",
                    event.event_type,
                    stats.batch_updated,
                    stats.enqueued_changed,
                    stats.enqueued_added,
                    stats.enqueued_deleted
                );
            }
        }
        Err(e) => {
            tracing::warn!("Git event processing failed (non-fatal): {}", e);
        }
    }
}

/// Spawn bootstrap hierarchy rebuild if tags exist but canonical_tags is empty.
fn spawn_hierarchy_bootstrap(hierarchy_builder: &Arc<HierarchyBuilder>) {
    let builder = Arc::clone(hierarchy_builder);
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
        if builder.needs_rebuild().await {
            info!("Bootstrap: tags exist but no canonical hierarchy -- running initial rebuild");
            match builder.rebuild_all().await {
                Ok(result) => {
                    info!(
                        "Bootstrap hierarchy rebuild complete: {} tenants, \
                         {} canonical tags, {} edges",
                        result.tenants_processed, result.total_canonical_tags, result.total_edges
                    );
                }
                Err(e) => {
                    error!("Bootstrap hierarchy rebuild failed: {}", e);
                }
            }
        }
    });
}

/// Redirect stderr to /dev/null (or NUL on Windows) in daemon mode.
fn redirect_stderr_for_daemon() {
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        if let Ok(null_file) = OpenOptions::new().write(true).open("/dev/null") {
            unsafe {
                libc::dup2(null_file.as_raw_fd(), libc::STDERR_FILENO);
            }
        }
    }

    #[cfg(windows)]
    {
        use std::fs::OpenOptions;
        use std::os::windows::io::AsRawHandle;

        if let Ok(null_file) = OpenOptions::new().write(true).open("NUL") {
            unsafe {
                winapi::um::processenv::SetStdHandle(
                    winapi::um::winbase::STD_ERROR_HANDLE,
                    null_file.as_raw_handle() as *mut winapi::ctypes::c_void,
                );
            }
        }
    }
}

/// Cross-platform entry point.
///
/// On Windows, first attempt to run as an SCM-managed service. If
/// `service_dispatcher::start` reports that the binary was not started
/// by the SCM (`ERROR_FAILED_SERVICE_CONTROLLER_CONNECT`, raw
/// errno 1063), fall through to the normal `tokio::main` interactive
/// path. On non-Windows the SCM step is a no-op.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(windows)]
    {
        if windows_service::try_start_dispatcher() {
            // Service ran under the SCM and has now stopped.
            return Ok(());
        }
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("TTY_DETECTION_SILENT", "1");
    std::env::set_var("ATTY_FORCE_DISABLE_DEBUG", "1");
    std::env::set_var("NO_COLOR", "1");

    let is_daemon_mode = startup::detect_daemon_mode();

    if is_daemon_mode {
        workspace_qdrant_core::logging::suppress_tty_debug_output();
        startup::suppress_third_party_output();
        std::env::set_var("WQM_SERVICE_MODE", "true");
        redirect_stderr_for_daemon();
    }

    let args = startup::parse_args()?;
    // Load config first so we can honor the OTLP settings during
    // subscriber initialization. Config loading uses tracing calls that are
    // dropped until the subscriber is installed — acceptable tradeoff for
    // getting OTLP export on `#[instrument]` spans right from startup.
    let config = startup::load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;
    startup::init_logging_with_telemetry(
        &args.log_level,
        args.foreground,
        Some(&config.observability.telemetry),
    )?;

    info!("memexd daemon starting up");
    info!("Command-line arguments: {:?}", args);

    if let Err(e) = run_daemon(config, args).await {
        error!("Daemon failed: {}", e);
        process::exit(1);
    }

    Ok(())
}
