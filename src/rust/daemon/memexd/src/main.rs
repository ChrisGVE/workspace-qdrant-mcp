//! memexd - Memory eXchange Daemon
//!
//! A daemon process that manages document processing, file watching, and embedding
//! generation for the workspace-qdrant-mcp system.

use clap::{Arg, Command};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use workspace_qdrant_core::{
    config::{Config, DaemonConfig},
    LoggingConfig, initialize_logging,
    unified_config::{UnifiedConfigManager, UnifiedConfigError},
    ipc::IpcServer,
    MetricsServer, METRICS,
    // Queue processor imports (Task 21)
    // Note: Legacy QueueProcessor removed - all processing via unified_queue
    // ProcessorConfig still used for configuration extraction
    ProcessorConfig,
    UnifiedQueueProcessor, UnifiedProcessorConfig,
    DocumentProcessor, EmbeddingGenerator, EmbeddingConfig,
    StorageClient, StorageConfig,
    queue_config::QueueConnectionConfig,
    // LSP lifecycle management (Task 1.1)
    LanguageServerManager, ProjectLspConfig,
    // Schema management (ADR-003: daemon owns database)
    SchemaManager,
};

// gRPC server for Python MCP server and CLI communication (Task 421)
use workspace_qdrant_grpc::{GrpcServer, ServerConfig as GrpcServerConfig};

/// Command-line arguments for memexd daemon
#[derive(Debug, Clone)]
struct DaemonArgs {
    /// Path to configuration file
    config_file: Option<PathBuf>,
    /// Port for IPC communication
    port: Option<u16>,
    /// Port for gRPC server (default: 50051)
    grpc_port: u16,
    /// Logging level
    log_level: String,
    /// PID file path
    pid_file: PathBuf,
    /// Run in foreground (don't daemonize)
    foreground: bool,
    /// Project identifier for multi-instance support
    project_id: Option<String>,
    /// Port for Prometheus metrics endpoint (disabled if not specified)
    metrics_port: Option<u16>,
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            config_file: None,
            port: None,
            grpc_port: 50051,
            log_level: "info".to_string(),
            pid_file: PathBuf::from("/tmp/memexd.pid"),
            foreground: false,
            project_id: None,
            metrics_port: None,
        }
    }
}

/// Parse command-line arguments with graceful error handling
fn parse_args() -> Result<DaemonArgs, Box<dyn std::error::Error>> {
    let is_daemon = detect_daemon_mode();

    let matches = Command::new("memexd")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Christian C. Berclaz <christian.berclaz@mac.com>")
        .about("Memory eXchange Daemon - Document processing and embedding generation service")
        .disable_help_flag(is_daemon)
        .disable_version_flag(is_daemon)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("IPC communication port")
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("log-level")
                .short('l')
                .long("log-level")
                .value_name("LEVEL")
                .help("Logging level (error, warn, info, debug, trace)")
                .default_value("info")
                .value_parser(["error", "warn", "info", "debug", "trace"]),
        )
        .arg(
            Arg::new("pid-file")
                .long("pid-file")
                .value_name("FILE")
                .help("PID file path")
                .default_value("/tmp/memexd.pid")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("foreground")
                .short('f')
                .long("foreground")
                .help("Run in foreground (don't daemonize)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("project-id")
                .long("project-id")
                .value_name("ID")
                .help("Project identifier for multi-instance support")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            Arg::new("metrics-port")
                .long("metrics-port")
                .value_name("PORT")
                .help("Enable Prometheus metrics endpoint on this port (e.g., 9090)")
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("grpc-port")
                .long("grpc-port")
                .value_name("PORT")
                .help("Port for gRPC server (for MCP server and CLI communication)")
                .default_value("50051")
                .value_parser(clap::value_parser!(u16)),
        )
        .try_get_matches();

    let matches = match matches {
        Ok(m) => m,
        Err(e) => {
            // Handle version/help specially - they should exit with code 0
            if e.kind() == clap::error::ErrorKind::DisplayHelp
                || e.kind() == clap::error::ErrorKind::DisplayVersion
            {
                print!("{}", e);
                process::exit(0);
            }
            if is_daemon {
                process::exit(1);
            } else {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    };

    let log_level = matches.get_one::<String>("log-level")
        .ok_or("Missing log-level parameter")?;

    let pid_file = matches.get_one::<PathBuf>("pid-file")
        .ok_or("Missing pid-file parameter")?;

    let grpc_port = matches.get_one::<u16>("grpc-port")
        .copied()
        .unwrap_or(50051);

    Ok(DaemonArgs {
        config_file: matches.get_one::<PathBuf>("config").cloned(),
        port: matches.get_one::<u16>("port").copied(),
        grpc_port,
        log_level: log_level.clone(),
        pid_file: pid_file.clone(),
        foreground: matches.get_flag("foreground"),
        project_id: matches.get_one::<String>("project-id").cloned(),
        metrics_port: matches.get_one::<u16>("metrics-port").copied(),
    })
}

/// Suppress third-party library output in daemon mode
fn suppress_third_party_output() {
    let suppression_vars = [
        ("ORT_LOGGING_LEVEL", "4"),
        ("OMP_NUM_THREADS", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
        ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("NO_COLOR", "1"),
        ("TERM", "dumb"),
        ("RUST_BACKTRACE", "0"),
        ("TTY_DETECTION_SILENT", "1"),
        ("ATTY_FORCE_DISABLE_DEBUG", "1"),
        ("WQM_TTY_DEBUG", "0"),
    ];

    for (key, value) in &suppression_vars {
        std::env::set_var(key, value);
    }
}

/// Initialize comprehensive logging with daemon mode support
fn init_logging(log_level: &str, foreground: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !foreground {
        std::env::set_var("WQM_SERVICE_MODE", "true");
        workspace_qdrant_core::logging::suppress_tty_debug_output();
        suppress_third_party_output();
    }

    let mut config = if foreground {
        LoggingConfig::development()
    } else {
        let mut prod_config = LoggingConfig::production();
        prod_config.console_output = false;
        prod_config.file_logging = false;
        prod_config.force_disable_ansi = Some(true);
        prod_config
    };

    use tracing::Level;
    config.level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    if !foreground {
        config.level = Level::ERROR;
    }

    initialize_logging(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Create PID file with current process ID
fn create_pid_file(pid_file: &Path, project_id: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    let pid = process::id();

    if let Some(parent) = pid_file.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let temp_file = pid_file.with_extension("tmp");
    fs::write(&temp_file, format!("{}\n", pid))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&temp_file)?.permissions();
        perms.set_mode(0o644);
        fs::set_permissions(&temp_file, perms)?;
    }

    fs::rename(&temp_file, pid_file)?;

    let project_info = project_id.map(|id| format!(" for project {}", id)).unwrap_or_default();
    info!("Created PID file at {} with PID {}{}", pid_file.display(), pid, project_info);
    Ok(())
}

/// Remove PID file and any temporary files
fn remove_pid_file(pid_file: &Path) {
    if let Err(e) = fs::remove_file(pid_file) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!("Failed to remove PID file {}: {}", pid_file.display(), e);
        }
    } else {
        info!("Removed PID file {}", pid_file.display());
    }

    let temp_file = pid_file.with_extension("tmp");
    if temp_file.exists() {
        let _ = fs::remove_file(&temp_file);
    }
}

/// Check if another instance is already running
fn check_existing_instance(pid_file: &Path, project_id: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        let pid_content = fs::read_to_string(pid_file)?;
        let pid: u32 = pid_content.trim().parse()?;

        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "comm="])
                .output()?;

            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let process_name = stdout.trim();

                if process_name.contains("memexd") {
                    let project_info = project_id.map(|id| format!(" for project {}", id)).unwrap_or_default();
                    return Err(format!(
                        "Another memexd instance is already running{} with PID {}",
                        project_info, pid
                    ).into());
                } else {
                    warn!("PID file contains non-memexd process, removing stale file");
                }
            }
        }

        #[cfg(windows)]
        {
            use std::process::Command;
            let output = Command::new("tasklist")
                .args(["/FI", &format!("PID eq {}", pid), "/FO", "CSV", "/NH"])
                .output()?;

            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !stdout.trim().is_empty() && stdout.contains("memexd") {
                    return Err(format!("Another memexd instance is already running with PID {}", pid).into());
                }
            }
        }

        warn!("Found stale PID file {}, removing it", pid_file.display());
        fs::remove_file(pid_file)?;
    }
    Ok(())
}

/// Load configuration from file or use defaults
fn load_config(args: &DaemonArgs) -> Result<DaemonConfig, Box<dyn std::error::Error>> {
    let is_daemon_mode = detect_daemon_mode();
    let config_manager = UnifiedConfigManager::new(None::<PathBuf>);

    let daemon_config = match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            match config_manager.load_config(Some(config_path)) {
                Ok(daemon_config) => {
                    info!("Configuration loaded successfully");
                    daemon_config
                },
                Err(UnifiedConfigError::FileNotFound(path)) => {
                    error!("Configuration file not found: {}", path.display());
                    return Err(format!("Configuration file not found: {}", path.display()).into());
                },
                Err(e) => {
                    error!("Configuration loading error: {}", e);
                    return Err(format!("Configuration loading error: {}", e).into());
                }
            }
        }
        None => {
            info!("Auto-discovering configuration files");
            match config_manager.load_config(None) {
                Ok(daemon_config) => {
                    info!("Configuration auto-discovered");
                    daemon_config
                },
                Err(_) => {
                    info!("Using default configuration");
                    if is_daemon_mode {
                        DaemonConfig::daemon_mode()
                    } else {
                        DaemonConfig::default()
                    }
                }
            }
        }
    };

    if args.port.is_some() {
        info!("Port override specified: {}", args.port.unwrap());
    }

    Ok(daemon_config)
}

/// Set up signal handlers for graceful shutdown
async fn setup_signal_handlers() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;
        let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;

        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM, initiating graceful shutdown");
            }
            _ = sigint.recv() => {
                info!("Received SIGINT, initiating graceful shutdown");
            }
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C, initiating graceful shutdown");
            }
        }
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c().await?;
        info!("Received Ctrl+C, initiating graceful shutdown");
    }

    Ok(())
}

/// Main daemon loop
async fn run_daemon(daemon_config: DaemonConfig, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    let project_info = args.project_id.as_ref().map(|id| format!(" for project {}", id)).unwrap_or_default();
    info!("Starting memexd daemon (version {}){}", env!("CARGO_PKG_VERSION"), project_info);

    check_existing_instance(&args.pid_file, args.project_id.as_ref())?;
    create_pid_file(&args.pid_file, args.project_id.as_ref())?;

    // Convert DaemonConfig to Config for queue processor settings
    let config = Config::from(daemon_config.clone());

    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });

    // Start metrics server if port is specified
    let metrics_handle = if let Some(port) = args.metrics_port {
        info!("Starting Prometheus metrics endpoint on port {}", port);
        let mut metrics_server = MetricsServer::new(port);
        let handle = tokio::spawn(async move {
            if let Err(e) = metrics_server.start().await {
                error!("Metrics server error: {}", e);
            }
        });
        Some(handle)
    } else {
        info!("Metrics endpoint disabled (use --metrics-port to enable)");
        None
    };

    // Start uptime tracking
    let start_time = std::time::Instant::now();
    let uptime_handle = tokio::spawn(async move {
        loop {
            METRICS.set_uptime(start_time.elapsed().as_secs_f64());
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });

    // Initialize SQLite database pool early so it can be shared with gRPC server
    // Get SQLite database path from config or use default
    let db_path = config.database_path.clone().unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(format!("{}/.workspace-qdrant/state.db", home))
    });

    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    // Create SQLite connection pool for queue operations and ProjectService
    let queue_config = QueueConnectionConfig::with_database_path(&db_path);
    let queue_pool = queue_config.create_pool().await
        .map_err(|e| format!("Failed to create queue database pool: {}", e))?;

    info!("Queue database pool created at: {}", db_path.display());

    // Run schema migrations (ADR-003: daemon owns database and schema)
    // This creates watch_folders, unified_queue tables with correct schema
    info!("Running database schema migrations...");
    let schema_manager = SchemaManager::new(queue_pool.clone());
    schema_manager.run_migrations().await
        .map_err(|e| format!("Failed to run schema migrations: {}", e))?;
    info!("Schema migrations complete");

    // Clone pool for gRPC server (ProjectService needs it)
    let grpc_db_pool = queue_pool.clone();

    // Initialize LSP lifecycle manager (Task 1.1)
    // Created centrally to share between gRPC server and UnifiedQueueProcessor
    // Configuration loaded from daemon config.yaml (Task 1.15)
    let lsp_config = ProjectLspConfig::from(daemon_config.lsp.clone());
    info!(
        "Initializing LSP lifecycle manager (max_servers={}, health_interval={}s)...",
        lsp_config.max_servers_per_project,
        lsp_config.health_check_interval_secs
    );
    let lsp_manager = match LanguageServerManager::new(lsp_config).await {
        Ok(mut manager) => {
            // Initialize the LSP manager
            if let Err(e) = manager.initialize().await {
                warn!("Failed to initialize LSP manager: {}", e);
            }
            let manager = Arc::new(RwLock::new(manager));
            info!("LSP lifecycle manager initialized");
            Some(manager)
        }
        Err(e) => {
            warn!("Failed to create LSP manager, continuing without LSP: {}", e);
            None
        }
    };

    // Clone LSP manager for gRPC server
    let grpc_lsp_manager = lsp_manager.clone();

    // Start gRPC server for MCP server and CLI communication (Task 421)
    let grpc_port = args.grpc_port;
    let grpc_addr = format!("127.0.0.1:{}", grpc_port).parse()
        .map_err(|e| format!("Invalid gRPC address: {}", e))?;
    let grpc_config = GrpcServerConfig::new(grpc_addr);

    info!("Starting gRPC server on port {}", grpc_port);
    let grpc_handle = tokio::spawn(async move {
        let mut grpc_server = GrpcServer::new(grpc_config)
            .with_database_pool(grpc_db_pool);

        // Enable LSP if manager was created successfully
        if let Some(lsp_manager) = grpc_lsp_manager {
            grpc_server = grpc_server.with_lsp_manager(lsp_manager);
        }

        if let Err(e) = grpc_server.start().await {
            error!("gRPC server error: {}", e);
        }
    });
    info!("gRPC server started on 127.0.0.1:{} with ProjectService enabled", grpc_port);

    info!("Initializing IPC server");
    let max_concurrent = config.max_concurrent_tasks.unwrap_or(8);
    let (ipc_server, _ipc_client) = IpcServer::new(max_concurrent);

    info!("Starting IPC server");
    ipc_server.start().await.map_err(|e| {
        error!("Failed to start IPC server: {}", e);
        e
    })?;

    info!("IPC server started successfully");

    // Initialize queue processor (Task 21)
    // Note: queue_pool was created earlier for sharing with gRPC server
    info!("Initializing queue processor...");

    // Initialize queue processor components
    let processor_config = ProcessorConfig {
        batch_size: config.queue_batch_size.unwrap_or(10) as i32,
        poll_interval_ms: config.queue_poll_interval_ms.unwrap_or(500),
        worker_count: config.queue_worker_count.unwrap_or(4),
        parallel_processing: config.queue_parallel_processing.unwrap_or(true),
        backpressure_threshold: config.queue_backpressure_threshold.unwrap_or(1000),
        ..ProcessorConfig::default()
    };

    // Create processing components
    let document_processor = Arc::new(DocumentProcessor::new());

    // Build EmbeddingConfig from daemon configuration
    let embedding_config = EmbeddingConfig {
        max_cache_size: daemon_config.embedding.cache_max_entries,
        model_cache_dir: daemon_config.embedding.model_cache_dir.clone(),
        ..EmbeddingConfig::default()
    };

    if let Some(ref cache_dir) = embedding_config.model_cache_dir {
        info!("Using model cache directory: {}", cache_dir.display());
    }

    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .map_err(|e| format!("Failed to create embedding generator: {}", e))?
    );
    // Use daemon_mode() for gRPC port 6334 and skip compatibility check
    let storage_config = StorageConfig::daemon_mode();
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    // Initialize multi-tenant Qdrant collections (projects, libraries, memory)
    // This is idempotent - existing collections are skipped
    info!("Initializing Qdrant collections...");
    match storage_client.initialize_multi_tenant_collections(None).await {
        Ok(result) => {
            info!(
                "Qdrant collections initialized: projects={}, libraries={}, memory={}",
                result.projects_created, result.libraries_created, result.memory_created
            );
            if result.is_complete() {
                info!("All multi-tenant collections ready with dense+sparse vector support");
            }
        }
        Err(e) => {
            // Log warning but continue - Qdrant might not be available yet
            // Collections will be created on first use
            warn!("Failed to initialize Qdrant collections (will retry on use): {}", e);
        }
    }

    // Initialize unified queue processor (Task 37.26)
    // Note: Legacy QueueProcessor removed per Task 21 - all processing via unified_queue
    info!("Initializing unified queue processor...");
    let unified_config = UnifiedProcessorConfig {
        batch_size: processor_config.batch_size,
        poll_interval_ms: processor_config.poll_interval_ms,
        worker_id: format!("memexd-{}", std::process::id()),
        lease_duration_secs: 300, // 5 minutes
        max_retries: 3,
        ..UnifiedProcessorConfig::default()
    };

    let mut unified_queue_processor = UnifiedQueueProcessor::with_components(
        queue_pool,
        unified_config.clone(),
        document_processor,
        embedding_generator,
        storage_client,
    );

    // Add LSP manager for code enrichment during file processing (Task 1.1)
    if let Some(ref lsp) = lsp_manager {
        unified_queue_processor = unified_queue_processor.with_lsp_manager(Arc::clone(lsp));
        info!("LSP manager attached to unified queue processor for code enrichment");
    }

    // Recover stale leases from previous daemon crashes (Task 37.19)
    if let Err(e) = unified_queue_processor.recover_stale_leases().await {
        warn!("Failed to recover stale unified queue leases: {}", e);
    }

    unified_queue_processor.start()
        .map_err(|e| format!("Failed to start unified queue processor: {}", e))?;

    info!(
        "Unified queue processor started (batch_size={}, poll_interval={}ms, worker_id={})",
        unified_config.batch_size,
        unified_config.poll_interval_ms,
        unified_config.worker_id
    );

    info!("memexd daemon is running. gRPC on port {}, send SIGTERM or SIGINT to stop.", grpc_port);

    let shutdown_future = setup_signal_handlers();
    if let Err(e) = shutdown_future.await {
        error!("Error in signal handling: {}", e);
    }

    // Cleanup - stop unified queue processor first for graceful shutdown
    // Note: Legacy QueueProcessor removed per Task 21 - all processing via unified_queue
    info!("Stopping unified queue processor...");
    if let Err(e) = unified_queue_processor.stop().await {
        error!("Error stopping unified queue processor: {}", e);
    } else {
        info!("Unified queue processor stopped");
    }

    // Shutdown LSP manager - stop all language servers (Task 1.1)
    if let Some(lsp) = lsp_manager {
        info!("Shutting down LSP manager...");
        let manager = lsp.write().await;
        if let Err(e) = manager.shutdown().await {
            error!("Error shutting down LSP manager: {}", e);
        } else {
            info!("LSP manager shutdown complete");
        }
    }

    uptime_handle.abort();
    grpc_handle.abort();
    info!("gRPC server stopped");
    if let Some(handle) = metrics_handle {
        handle.abort();
        info!("Metrics server stopped");
    }

    info!("memexd daemon shutdown complete");
    Ok(())
}

/// Detect if we're running in daemon/service mode
fn detect_daemon_mode() -> bool {
    let args: Vec<String> = std::env::args().collect();

    // If asking for help or version, don't treat as daemon mode
    if args.iter().any(|arg| arg == "--help" || arg == "-h" || arg == "--version" || arg == "-V") {
        return false;
    }

    let is_daemon = !args.iter().any(|arg| arg == "--foreground" || arg == "-f");

    // macOS: XPC_SERVICE_NAME is set to "0" in regular terminal sessions,
    // so we need to check it's not empty and not "0" to detect actual service context
    let xpc_is_service = std::env::var("XPC_SERVICE_NAME")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let service_context = std::env::var("WQM_SERVICE_MODE").unwrap_or_default() == "true" ||
        xpc_is_service ||
        std::env::var("LAUNCHD_SOCKET_PATH").is_ok() ||
        std::env::var("SYSTEMD_EXEC_PID").is_ok() ||
        std::env::var("SERVICE_NAME").is_ok();

    is_daemon || service_context
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("TTY_DETECTION_SILENT", "1");
    std::env::set_var("ATTY_FORCE_DISABLE_DEBUG", "1");
    std::env::set_var("NO_COLOR", "1");

    let is_daemon_mode = detect_daemon_mode();

    if is_daemon_mode {
        workspace_qdrant_core::logging::suppress_tty_debug_output();
        suppress_third_party_output();
        std::env::set_var("WQM_SERVICE_MODE", "true");

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            use std::fs::OpenOptions;

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
                        null_file.as_raw_handle() as *mut std::ffi::c_void
                    );
                }
            }
        }
    }

    let args = parse_args()?;
    init_logging(&args.log_level, args.foreground)?;

    info!("memexd daemon starting up");
    info!("Command-line arguments: {:?}", args);

    let config = load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;

    if let Err(e) = run_daemon(config, args).await {
        error!("Daemon failed: {}", e);
        process::exit(1);
    }

    Ok(())
}
