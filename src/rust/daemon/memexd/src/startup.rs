//! Phase 1: CLI parsing, instance check, PID file management, configuration loading.
//!
//! Handles early daemon bootstrap before any async resources are created:
//! argument parsing, daemon mode detection, logging initialization,
//! existing-instance detection, PID file creation/removal, legacy directory
//! warnings, and configuration loading from file or defaults.

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use clap::{Arg, Command};
use tracing::{error, info, warn};

use workspace_qdrant_core::{
    config::DaemonConfig,
    initialize_logging,
    unified_config::{UnifiedConfigError, UnifiedConfigManager},
    LoggingConfig,
};

/// Command-line arguments for memexd daemon.
#[derive(Debug, Clone)]
pub struct DaemonArgs {
    /// Path to configuration file.
    pub config_file: Option<PathBuf>,
    /// Port for IPC communication.
    pub port: Option<u16>,
    /// Port for gRPC server (default: 50051).
    pub grpc_port: u16,
    /// Logging level.
    pub log_level: String,
    /// PID file path.
    pub pid_file: PathBuf,
    /// Run in foreground (don't daemonize).
    pub foreground: bool,
    /// Project identifier for multi-instance support.
    pub project_id: Option<String>,
    /// Port for Prometheus metrics endpoint (disabled if not specified).
    pub metrics_port: Option<u16>,
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

/// Define all CLI arguments for the memexd command.
fn build_cli(is_daemon: bool) -> Command {
    Command::new("memexd")
        .version(concat!(
            env!("CARGO_PKG_VERSION"),
            " (",
            env!("BUILD_NUMBER"),
            ")"
        ))
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
}

/// Parse command-line arguments with graceful error handling.
pub fn parse_args() -> Result<DaemonArgs, Box<dyn std::error::Error>> {
    let is_daemon = detect_daemon_mode();
    let matches = match build_cli(is_daemon).try_get_matches() {
        Ok(m) => m,
        Err(e) => {
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

    let log_level = matches
        .get_one::<String>("log-level")
        .ok_or("Missing log-level parameter")?;

    let pid_file = matches
        .get_one::<PathBuf>("pid-file")
        .ok_or("Missing pid-file parameter")?;

    let grpc_port = matches
        .get_one::<u16>("grpc-port")
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

/// Detect if we're running in daemon/service mode.
pub fn detect_daemon_mode() -> bool {
    let args: Vec<String> = std::env::args().collect();

    if args
        .iter()
        .any(|arg| arg == "--help" || arg == "-h" || arg == "--version" || arg == "-V")
    {
        return false;
    }

    let is_daemon = !args.iter().any(|arg| arg == "--foreground" || arg == "-f");

    let xpc_is_service = std::env::var("XPC_SERVICE_NAME")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let service_context = std::env::var("WQM_SERVICE_MODE").unwrap_or_default() == "true"
        || xpc_is_service
        || std::env::var("LAUNCHD_SOCKET_PATH").is_ok()
        || std::env::var("SYSTEMD_EXEC_PID").is_ok()
        || std::env::var("SERVICE_NAME").is_ok();

    is_daemon || service_context
}

/// Suppress third-party library output in daemon mode.
pub fn suppress_third_party_output() {
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

/// Initialize comprehensive logging with daemon mode support.
pub fn init_logging(log_level: &str, foreground: bool) -> Result<(), Box<dyn std::error::Error>> {
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

    if !foreground && config.level > Level::INFO {
        config.level = Level::INFO;
    }

    initialize_logging(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Check if another memexd instance is already running by inspecting the PID file.
pub fn check_existing_instance(
    pid_file: &Path,
    project_id: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        let pid_content = fs::read_to_string(pid_file)?;
        let pid: u32 = pid_content.trim().parse()?;

        #[cfg(unix)]
        {
            let output = process::Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "comm="])
                .output()?;

            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let process_name = stdout.trim();

                if process_name.contains("memexd") {
                    let project_info = project_id
                        .map(|id| format!(" for project {}", id))
                        .unwrap_or_default();
                    return Err(format!(
                        "Another memexd instance is already running{} with PID {}",
                        project_info, pid
                    )
                    .into());
                } else {
                    warn!("PID file contains non-memexd process, removing stale file");
                }
            }
        }

        #[cfg(windows)]
        {
            let output = process::Command::new("tasklist")
                .args(["/FI", &format!("PID eq {}", pid), "/FO", "CSV", "/NH"])
                .output()?;

            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !stdout.trim().is_empty() && stdout.contains("memexd") {
                    return Err(format!(
                        "Another memexd instance is already running with PID {}",
                        pid
                    )
                    .into());
                }
            }
        }

        warn!("Found stale PID file {}, removing it", pid_file.display());
        fs::remove_file(pid_file)?;
    }
    Ok(())
}

/// Create PID file with the current process ID.
pub fn create_pid_file(
    pid_file: &Path,
    project_id: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
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

    let project_info = project_id
        .map(|id| format!(" for project {}", id))
        .unwrap_or_default();
    info!(
        "Created PID file at {} with PID {}{}",
        pid_file.display(),
        pid,
        project_info
    );
    Ok(())
}

/// Remove PID file and any temporary files.
pub fn remove_pid_file(pid_file: &Path) {
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

/// Warn if the stale legacy directory from the old Python MCP server exists.
pub fn check_stale_legacy_directory() {
    if let Ok(home) = std::env::var("HOME") {
        let legacy_dir = PathBuf::from(&home).join(".workspace-qdrant-mcp");
        if legacy_dir.exists() {
            warn!(
                path = %legacy_dir.display(),
                "Stale legacy directory found from old Python MCP server. \
                 This directory is no longer used and can be safely removed: \
                 rm -rf ~/.workspace-qdrant-mcp"
            );
        }
    }
}

/// Load configuration from file or auto-discover defaults.
pub fn load_config(args: &DaemonArgs) -> Result<DaemonConfig, Box<dyn std::error::Error>> {
    let is_daemon_mode = detect_daemon_mode();
    let config_manager = UnifiedConfigManager::new(None::<PathBuf>);

    let daemon_config = match &args.config_file {
        Some(config_path) => load_from_file(&config_manager, config_path)?,
        None => load_auto_discover(&config_manager, is_daemon_mode),
    };

    if let Some(port) = args.port {
        info!("Port override specified: {}", port);
    }

    Ok(daemon_config)
}

/// Load config from an explicit file path.
fn load_from_file(
    config_manager: &UnifiedConfigManager,
    config_path: &Path,
) -> Result<DaemonConfig, Box<dyn std::error::Error>> {
    info!("Loading configuration from {}", config_path.display());
    match config_manager.load_config(Some(config_path)) {
        Ok(daemon_config) => {
            info!("Configuration loaded successfully");
            Ok(daemon_config)
        }
        Err(UnifiedConfigError::FileNotFound(path)) => {
            error!("Configuration file not found: {}", path.display());
            Err(format!("Configuration file not found: {}", path.display()).into())
        }
        Err(e) => {
            error!("Configuration loading error: {}", e);
            Err(format!("Configuration loading error: {}", e).into())
        }
    }
}

/// Auto-discover config or fall back to defaults.
fn load_auto_discover(config_manager: &UnifiedConfigManager, is_daemon_mode: bool) -> DaemonConfig {
    info!("Auto-discovering configuration files");
    match config_manager.load_config(None) {
        Ok(daemon_config) => {
            info!("Configuration auto-discovered");
            daemon_config
        }
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

/// Set process nice level for resource management (Task 504).
pub fn set_process_nice_level(nice_level: i32) {
    #[cfg(unix)]
    {
        let result = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice_level) };
        if result == 0 {
            info!("Set process nice level to {}", nice_level);
        } else {
            let err = std::io::Error::last_os_error();
            warn!(
                "Failed to set nice level to {}: {} (continuing with default)",
                nice_level, err
            );
        }
    }

    #[cfg(not(unix))]
    {
        let _ = nice_level;
        info!("Nice level not supported on this platform, skipping");
    }
}
