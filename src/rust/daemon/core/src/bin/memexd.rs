//! memexd - Memory eXchange Daemon
//!
//! A daemon process that manages document processing, file watching, and embedding
//! generation for the workspace-qdrant-mcp system.

use clap::{Arg, Command};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use tokio::signal;
use tracing::{error, info, warn};

use workspace_qdrant_core::{
    config::Config,
    LoggingConfig, initialize_logging,
    unified_config::{UnifiedConfigManager, UnifiedConfigError},
    ipc::IpcServer,
};

/// Command-line arguments for memexd daemon
#[derive(Debug, Clone)]
struct DaemonArgs {
    /// Path to configuration file
    config_file: Option<PathBuf>,
    /// Port for IPC communication
    port: Option<u16>,
    /// Logging level
    log_level: String,
    /// PID file path
    pid_file: PathBuf,
    /// Run in foreground (don't daemonize)
    foreground: bool,
    /// Project identifier for multi-instance support
    project_id: Option<String>,
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            config_file: None,
            port: None,
            log_level: "info".to_string(),
            pid_file: PathBuf::from("/tmp/memexd.pid"),
            foreground: false,
            project_id: None,
        }
    }
}

/// Parse command-line arguments with graceful error handling
fn parse_args() -> Result<DaemonArgs, Box<dyn std::error::Error>> {
    let is_daemon = detect_daemon_mode();

    let matches = Command::new("memexd")
        .version("0.2.0")
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
        .try_get_matches();

    let matches = match matches {
        Ok(m) => m,
        Err(e) => {
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

    Ok(DaemonArgs {
        config_file: matches.get_one::<PathBuf>("config").cloned(),
        port: matches.get_one::<u16>("port").copied(),
        log_level: log_level.clone(),
        pid_file: pid_file.clone(),
        foreground: matches.get_flag("foreground"),
        project_id: matches.get_one::<String>("project-id").cloned(),
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
fn load_config(args: &DaemonArgs) -> Result<Config, Box<dyn std::error::Error>> {
    let is_daemon_mode = detect_daemon_mode();
    let config_manager = UnifiedConfigManager::new(None::<PathBuf>);

    let config = match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            match config_manager.load_config(Some(config_path)) {
                Ok(daemon_config) => {
                    info!("Configuration loaded successfully");
                    Config::from(daemon_config)
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
                    Config::from(daemon_config)
                },
                Err(_) => {
                    info!("Using default configuration");
                    if is_daemon_mode {
                        let mut cfg = Config::default();
                        cfg.enable_metrics = false;
                        cfg
                    } else {
                        Config::default()
                    }
                }
            }
        }
    };

    if args.port.is_some() {
        info!("Port override specified: {}", args.port.unwrap());
    }

    Ok(config)
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
async fn run_daemon(config: Config, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    let project_info = args.project_id.as_ref().map(|id| format!(" for project {}", id)).unwrap_or_default();
    info!("Starting memexd daemon (version 0.2.0){}", project_info);

    check_existing_instance(&args.pid_file, args.project_id.as_ref())?;
    create_pid_file(&args.pid_file, args.project_id.as_ref())?;

    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });

    info!("Initializing IPC server");
    let max_concurrent = config.max_concurrent_tasks.unwrap_or(8);
    let (ipc_server, _ipc_client) = IpcServer::new(max_concurrent);

    info!("Starting IPC server");
    ipc_server.start().await.map_err(|e| {
        error!("Failed to start IPC server: {}", e);
        e
    })?;

    info!("IPC server started successfully");
    info!("memexd daemon is running. Send SIGTERM or SIGINT to stop.");

    let shutdown_future = setup_signal_handlers();
    if let Err(e) = shutdown_future.await {
        error!("Error in signal handling: {}", e);
    }

    info!("memexd daemon shutdown complete");
    Ok(())
}

/// Detect if we're running in daemon/service mode
fn detect_daemon_mode() -> bool {
    let args: Vec<String> = std::env::args().collect();
    let is_daemon = !args.iter().any(|arg| arg == "--foreground" || arg == "-f");

    let service_context = std::env::var("WQM_SERVICE_MODE").unwrap_or_default() == "true" ||
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
