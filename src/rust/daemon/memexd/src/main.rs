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
mod relative_path_hook;
mod shutdown;
mod startup;

use memexd::control_port;
#[cfg(windows)]
mod windows_service;

use std::process;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{debug, error, info, warn};

use workspace_qdrant_core::{config::Config, config::DaemonConfig, HierarchyBuilder, WatchManager};

pub use startup::DaemonArgs;

/// Execute Phase 1: instance check, control-port lock, PID, config conversion, nice level.
///
/// Returns the resolved `Config` plus a tuple of drop guards: the
/// PID-file scopeguard and the control-port lock holder. Both are
/// released when the returned tuple is dropped at process exit.
///
/// The control-port bind (spec 16 §10.1) happens *before* the PID file
/// is created so that a second memexd attempting to start cannot leave
/// a stale PID file behind on rejection. SQLite is opened in phase 2,
/// strictly after the lock is acquired.
fn run_phase1(
    args: &DaemonArgs,
    daemon_config: &DaemonConfig,
) -> Result<(Config, (impl Drop, control_port::ControlPortGuard)), Box<dyn std::error::Error>> {
    startup::check_existing_instance(&args.pid_file, args.project_id.as_ref())?;

    // Acquire the cross-process single-instance lock BEFORE SQLite is opened.
    let mode = control_port::DeploymentMode::detect();
    let port = control_port::resolve_port(args.control_port, daemon_config.control_port)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let cp_guard = control_port::acquire(port, mode).map_err(|e| {
        error!(
            "Failed to acquire memexd control port (single-instance lock): {}",
            e
        );
        Box::<dyn std::error::Error>::from(format!("{e}"))
    })?;

    startup::create_pid_file(&args.pid_file, args.project_id.as_ref())?;
    startup::check_stale_legacy_directory();

    let mut config = Config::from(daemon_config.clone());
    config.resource_limits.resolve_auto_values();
    startup::set_process_nice_level(config.resource_limits.nice_level);

    let pid_file_cleanup = args.pid_file.clone();
    let cleanup_guard = scopeguard::guard((), move |_| {
        startup::remove_pid_file(&pid_file_cleanup);
    });

    Ok((config, (cleanup_guard, cp_guard)))
}

/// Wire the gRPC server with all phase-5 dependencies and return its handle.
fn wire_grpc(
    args: &DaemonArgs,
    db_handles: &database::DatabaseHandles,
    qc: &queue_init::QueueComponents,
    lsp_manager: &Option<Arc<tokio::sync::RwLock<workspace_qdrant_core::LanguageServerManager>>>,
    watch_refresh_signal: &Arc<Notify>,
    embedding_settings: Arc<workspace_qdrant_core::config::EmbeddingSettings>,
    probe_cache: Arc<
        tokio::sync::Mutex<workspace_qdrant_core::embedding::provider::SharedProbeCache>,
    >,
    grammar_config: workspace_qdrant_core::config::GrammarConfig,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    grpc_setup::spawn_grpc_server(
        args,
        db_handles.queue_pool.clone(),
        Arc::clone(&db_handles.pause_flag),
        Arc::clone(watch_refresh_signal),
        Arc::clone(&qc.queue_health),
        Arc::clone(&qc.adaptive_state),
        Arc::clone(&db_handles.search_db),
        db_handles.graph_sqlite.clone(),
        lsp_manager.clone(),
        Arc::clone(&qc.hierarchy_builder),
        qc.watch_pool.clone(),
        Arc::clone(&qc.dense_provider),
        embedding_settings,
        probe_cache,
        grammar_config,
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
    // Telemetry kill switch: gate the dimensional-processing emission path (A2)
    // so it creates no series when metrics collection is disabled.
    workspace_qdrant_core::monitoring::metrics_core::METRICS
        .set_enabled(daemon_config.observability.metrics.enabled);
    // Trace cost-gate (B1/B4): set the tier from the telemetry config. The
    // tier already reflects env > YAML > default precedence — the WQM_TRACE_TIER
    // override is folded into the config during env-override application at load.
    let trace_tier = daemon_config
        .observability
        .telemetry
        .tracing
        .effective_tier();
    workspace_qdrant_core::tracing_gate::set_trace_tier(trace_tier);
    info!("Trace tier: {:?}", trace_tier);
    // Phase 2: Database
    let db_handles = database::initialize_all(&config).await?;
    // Graph metrics snapshotter (D5): snapshot the code-graph gauges on the
    // collection_interval. Uses the SQLite graph pool directly for a single
    // read transaction per tick; no-ops when telemetry is disabled.
    if let Some(graph_store) = db_handles.graph_sqlite.as_ref() {
        match graph_store.read().await {
            Ok(guard) => {
                let graph_pool = guard.pool().clone();
                drop(guard);
                background::start_graph_metrics_collection(
                    graph_pool,
                    daemon_config.observability.collection_interval,
                );
            }
            Err(e) => warn!("Graph metrics: unable to acquire graph pool: {}", e),
        }
    }
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

    // Phase 5a-5b: Dimension consistency + health monitor
    let probe_cache = check_dim_and_start_health_monitor(&daemon_config, &args, &mut qc).await?;

    // Phase 6: gRPC server + queue start + recovery
    let embedding_settings = Arc::new(daemon_config.embedding.clone());
    bg_handles.grpc_handle = Some(wire_grpc(
        &args,
        &db_handles,
        &qc,
        &lsp_manager,
        &watch_refresh_signal,
        embedding_settings,
        probe_cache,
        daemon_config.grammars.clone(),
    )?);
    queue_init::start_processor(&mut qc.unified_queue_processor, &daemon_config).await?;
    queue_init::spawn_recovery_tasks(
        &qc.unified_queue_processor,
        &qc.allowed_extensions,
        &qc.mirror_storage,
        &daemon_config,
    );

    let _ignore_reconcile_handle =
        database::spawn_background_reconciliation(db_handles.queue_pool.clone());

    // Phase 6b: File watching + hierarchy
    let (watch_manager, hierarchy_cancel) =
        start_watchers_and_hierarchy(&qc, &watch_refresh_signal).await;

    // Phase 6c: Post-startup hook for relative-path migration (phases 2b–4).
    let _migration_hook = relative_path_hook::spawn_if_needed(
        db_handles.queue_pool.clone(),
        Arc::clone(&qc.mirror_storage),
    );

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

/// Phase 5a-5b: Verify embedding dimension consistency and start health monitor.
/// Returns the shared probe cache so gRPC can read background probe results.
async fn check_dim_and_start_health_monitor(
    daemon_config: &DaemonConfig,
    args: &DaemonArgs,
    qc: &mut queue_init::QueueComponents,
) -> Result<
    Arc<tokio::sync::Mutex<workspace_qdrant_core::embedding::provider::SharedProbeCache>>,
    Box<dyn std::error::Error>,
> {
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

    let probe_cache = workspace_qdrant_core::embedding::provider::SharedProbeCache::new();
    let health_monitor = workspace_qdrant_core::embedding::provider::ProviderHealthMonitor::new(
        Arc::clone(&qc.dense_provider),
        std::time::Duration::from_secs(
            workspace_qdrant_core::embedding::provider::health_monitor::DEFAULT_PROBE_INTERVAL_SECS,
        ),
        Arc::clone(&probe_cache),
    );
    let health_monitor_token = qc.adaptive_shutdown_token.child_token();
    tokio::spawn(async move { health_monitor.run(health_monitor_token).await });

    Ok(probe_cache)
}

/// Phase 6b: Start file watchers, git event consumer, and hierarchy builder.
async fn start_watchers_and_hierarchy(
    qc: &queue_init::QueueComponents,
    watch_refresh_signal: &Arc<Notify>,
) -> (Arc<WatchManager>, tokio_util::sync::CancellationToken) {
    let watch_manager = start_file_watchers(qc, watch_refresh_signal).await;
    spawn_git_event_consumer(&qc.unified_queue_processor, &watch_manager);
    spawn_branch_event_consumer(&qc.unified_queue_processor, &watch_manager);
    spawn_branch_reconciliation(&qc.unified_queue_processor);
    let hierarchy_cancel = tokio_util::sync::CancellationToken::new();
    let _hierarchy_handle =
        Arc::clone(&qc.hierarchy_builder).start_scheduled(hierarchy_cancel.clone());
    info!("Canonical tag hierarchy rebuild scheduled (nightly at 2 AM)");
    spawn_hierarchy_bootstrap(&qc.hierarchy_builder);
    (watch_manager, hierarchy_cancel)
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
    let branch_ctx = Arc::new(workspace_qdrant_core::branch_switch::BranchUpdateContext {
        storage_client: Arc::clone(uqp.storage_client()),
        search_db: uqp.search_db().cloned(),
        branch_locks: Arc::clone(uqp.branch_locks()),
    });
    let git_watch_manager = Arc::clone(watch_manager);
    tokio::spawn(async move {
        if let Some(mut rx) = git_watch_manager.take_git_event_rx().await {
            info!("Git event consumer started");
            while let Some(event) = rx.recv().await {
                debug!(
                    "Processing git event: {:?} for {}",
                    event.event_type, event.watch_folder_id
                );
                handle_single_git_event(&event, &git_pool, &git_qm, &branch_ctx).await;
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
    branch_ctx: &workspace_qdrant_core::branch_switch::BranchUpdateContext,
) {
    match workspace_qdrant_core::branch_switch::handle_git_event(event, pool, qm, branch_ctx).await
    {
        Ok(stats) => {
            let total = stats.branch_added
                + stats.enqueued_changed
                + stats.enqueued_added
                + stats.enqueued_deleted;
            if total > 0 {
                info!(
                    "Git event {:?} processed: {} branch-added, \
                     {} changed, {} added, {} deleted",
                    event.event_type,
                    stats.branch_added,
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

/// Spawn the branch lifecycle event consumer that handles branch creation/deletion.
fn spawn_branch_event_consumer(
    uqp: &workspace_qdrant_core::UnifiedQueueProcessor,
    watch_manager: &Arc<WatchManager>,
) {
    let pool = uqp.pool().clone();
    let branch_ctx = Arc::new(workspace_qdrant_core::branch_switch::BranchUpdateContext {
        storage_client: Arc::clone(uqp.storage_client()),
        search_db: uqp.search_db().cloned(),
        branch_locks: Arc::clone(uqp.branch_locks()),
    });
    let wm = Arc::clone(watch_manager);
    tokio::spawn(async move {
        if let Some(mut rx) = wm.take_branch_event_rx().await {
            info!("Branch lifecycle event consumer started");
            while let Some((watch_folder_id, tenant_id, event)) = rx.recv().await {
                handle_branch_lifecycle_event(
                    &pool,
                    &branch_ctx,
                    &watch_folder_id,
                    &tenant_id,
                    &event,
                )
                .await;
            }
            info!("Branch lifecycle event consumer stopped");
        }
    });
}

/// Process a single branch lifecycle event.
async fn handle_branch_lifecycle_event(
    pool: &sqlx::SqlitePool,
    branch_ctx: &workspace_qdrant_core::branch_switch::BranchUpdateContext,
    watch_folder_id: &str,
    tenant_id: &str,
    event: &workspace_qdrant_core::git::BranchEvent,
) {
    use workspace_qdrant_core::git::BranchEvent;

    match event {
        BranchEvent::Deleted { branch } => {
            info!(
                "Branch lifecycle: '{}' deleted for {}",
                branch, watch_folder_id
            );
            let project_root: Option<String> =
                sqlx::query_scalar("SELECT path FROM watch_folders WHERE watch_id = ?1")
                    .bind(watch_folder_id)
                    .fetch_optional(pool)
                    .await
                    .unwrap_or(None);

            if let Some(root) = project_root {
                let result = workspace_qdrant_core::branch_cleanup::cleanup_deleted_branch(
                    pool,
                    branch_ctx,
                    watch_folder_id,
                    tenant_id,
                    std::path::Path::new(&root),
                    branch,
                )
                .await;
                info!(
                    "Branch cleanup for '{}': {} updated, {} deleted, skipped={}",
                    branch, result.updated, result.deleted, result.skipped
                );
            }
        }
        BranchEvent::Created { branch, .. } => {
            debug!(
                "Branch lifecycle: '{}' created for {} (discovery on first file event)",
                branch, watch_folder_id
            );
        }
        BranchEvent::Renamed {
            old_name, new_name, ..
        } => {
            info!(
                "Branch lifecycle: '{}' renamed to '{}' for {}",
                old_name, new_name, watch_folder_id
            );
            let result = workspace_qdrant_core::branch_cleanup::handle_branch_rename(
                pool,
                branch_ctx,
                watch_folder_id,
                tenant_id,
                old_name,
                new_name,
            )
            .await;
            info!(
                "Branch rename for '{}'->'{}': {} updated, {} errors",
                old_name, new_name, result.updated, result.errors
            );
        }
        _ => {
            debug!(
                "Branch lifecycle event: {:?} for {}",
                event, watch_folder_id
            );
        }
    }
}

/// Delay before the first stale-branch reconciliation sweep, leaving startup
/// recovery and initial queue bursts room to settle.
const BRANCH_RECONCILE_STARTUP_DELAY_SECS: u64 = 300;

/// Interval between periodic stale-branch reconciliation sweeps.
const BRANCH_RECONCILE_INTERVAL_SECS: u64 = 24 * 3600;

/// Spawn the stale-branch reconciliation sweep (#102).
///
/// Branch deletions that happen while the daemon is down never reach the
/// lifecycle event consumer, leaving stale rows in `tracked_files.branches[]`
/// and search.db `file_metadata` (each of which duplicates grep matches).
/// This task sweeps once shortly after startup and then daily, routing every
/// stale branch through the same `cleanup_deleted_branch` path the live
/// consumer uses (local + remote existence re-checked before any deletion).
fn spawn_branch_reconciliation(uqp: &workspace_qdrant_core::UnifiedQueueProcessor) {
    let pool = uqp.pool().clone();
    let branch_ctx = workspace_qdrant_core::branch_switch::BranchUpdateContext {
        storage_client: Arc::clone(uqp.storage_client()),
        search_db: uqp.search_db().cloned(),
        branch_locks: Arc::clone(uqp.branch_locks()),
    };
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(
            BRANCH_RECONCILE_STARTUP_DELAY_SECS,
        ))
        .await;
        loop {
            let stats =
                workspace_qdrant_core::branch_cleanup::reconcile_stale_branches(&pool, &branch_ctx)
                    .await;
            if stats.branches_pruned > 0 || stats.orphaned_lines_pruned > 0 || stats.errors > 0 {
                info!(
                    "Branch reconcile sweep: {} folders checked, {} branches pruned, \
                     {} skipped, {} orphaned lines pruned, {} errors",
                    stats.folders_checked,
                    stats.branches_pruned,
                    stats.branches_skipped,
                    stats.orphaned_lines_pruned,
                    stats.errors
                );
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(
                BRANCH_RECONCILE_INTERVAL_SECS,
            ))
            .await;
        }
    });
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
    // dropped until the subscriber is installed -- acceptable tradeoff for
    // getting OTLP export on `#[instrument]` spans right from startup.
    let config = startup::load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;
    if let Err(e) = config.validate() {
        error!("Invalid daemon configuration: {}", e);
        return Err(format!("Invalid configuration: {}", e).into());
    }
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
