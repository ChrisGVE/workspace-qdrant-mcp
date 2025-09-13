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

// Platform-specific imports for stderr suppression
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
// Removed unused imports: EnvFilter, FmtSubscriber
use workspace_qdrant_core::{
    ProcessingEngine, config::{Config, DaemonConfig}, 
    LoggingConfig, initialize_logging,
    // Removed unused imports: ErrorRecovery, ErrorRecoveryStrategy, track_async_operation, LoggingErrorMonitor
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
        .disable_help_flag(is_daemon) // Disable help in daemon mode to prevent output
        .disable_version_flag(is_daemon) // Disable version in daemon mode
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
            // Handle argument parsing errors gracefully
            if is_daemon {
                // In daemon mode, exit silently without stderr output
                process::exit(1);
            } else {
                // In interactive mode, show helpful error message
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    };

    // Extract arguments with proper error handling instead of unwrap()
    let log_level = matches.get_one::<String>("log-level")
        .ok_or("Missing log-level parameter (this should not happen with default value)")?;

    let pid_file = matches.get_one::<PathBuf>("pid-file")
        .ok_or("Missing pid-file parameter (this should not happen with default value)")?;

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
    // Set environment variables to suppress third-party library output
    let suppression_vars = [
        ("ORT_LOGGING_LEVEL", "4"),     // ONNX Runtime - only fatal errors
        ("OMP_NUM_THREADS", "1"),       // Disable OpenMP threading messages
        ("TOKENIZERS_PARALLELISM", "false"), // Disable tokenizers parallel output
        ("HF_HUB_DISABLE_PROGRESS_BARS", "1"), // Disable HuggingFace progress bars
        ("HF_HUB_DISABLE_TELEMETRY", "1"), // Disable HuggingFace telemetry
        ("NO_COLOR", "1"),              // Disable all ANSI color output
        ("TERM", "dumb"),               // Force dumb terminal mode
        ("RUST_BACKTRACE", "0"),        // Disable Rust panic backtraces
        ("TTY_DETECTION_SILENT", "1"),  // Suppress TTY detection debug messages
        ("ATTY_FORCE_DISABLE_DEBUG", "1"), // Suppress atty crate debug output
        ("WQM_TTY_DEBUG", "0"),         // Disable internal TTY debug output
        ("TTY_DEBUG", "0"),             // CRITICAL: Disable Claude CLI TTY debug messages
    ];

    for (key, value) in &suppression_vars {
        std::env::set_var(key, value);
    }
}

/// Initialize comprehensive logging with daemon mode support
fn init_logging(log_level: &str, foreground: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Set service mode environment variable if not running in foreground
    if !foreground {
        std::env::set_var("WQM_SERVICE_MODE", "true");
        // Suppress TTY detection debug output first, before any TTY checks
        workspace_qdrant_core::logging::suppress_tty_debug_output();
        // Suppress third-party output early in daemon mode
        suppress_third_party_output();
    }

    let mut config = if foreground {
        LoggingConfig::development()
    } else {
        // In daemon mode, configure for complete silence
        let mut prod_config = LoggingConfig::production();
        prod_config.console_output = false; // Disable console output completely
        prod_config.file_logging = false;   // Let launchd handle file logging
        prod_config.force_disable_ansi = Some(true);
        prod_config
    };

    // Parse log level
    use tracing::Level;
    config.level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    // In daemon mode, force level to OFF for complete silence
    if !foreground {
        config.level = Level::ERROR; // Minimal logging level
    }

    initialize_logging(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Create PID file with current process ID, with project-specific naming support
fn create_pid_file(pid_file: &Path, project_id: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    let pid = process::id();
    
    // Create parent directory if it doesn't exist
    if let Some(parent) = pid_file.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    
    // Write PID file atomically
    let temp_file = pid_file.with_extension("tmp");
    fs::write(&temp_file, format!("{}\n", pid))?;
    
    // Set appropriate permissions (readable by owner and group)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&temp_file)?.permissions();
        perms.set_mode(0o644);
        fs::set_permissions(&temp_file, perms)?;
    }
    
    // Atomically move temp file to final location
    fs::rename(&temp_file, pid_file)?;
    
    let project_info = project_id.map(|id| format!(" for project {}", id)).unwrap_or_default();
    info!("Created PID file at {} with PID {}{}", pid_file.display(), pid, project_info);
    Ok(())
}

/// Remove PID file and any temporary files
fn remove_pid_file(pid_file: &Path) {
    // Remove main PID file
    if let Err(e) = fs::remove_file(pid_file) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!("Failed to remove PID file {}: {}", pid_file.display(), e);
        }
    } else {
        info!("Removed PID file {}", pid_file.display());
    }
    
    // Also clean up any temporary files
    let temp_file = pid_file.with_extension("tmp");
    if temp_file.exists() {
        if let Err(e) = fs::remove_file(&temp_file) {
            warn!("Failed to remove temporary PID file {}: {}", temp_file.display(), e);
        }
    }
}

/// Check if another instance is already running for this project
fn check_existing_instance(pid_file: &Path, project_id: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        let pid_content = fs::read_to_string(pid_file)?;
        let pid: u32 = pid_content.trim().parse()?;
        
        // Check if process with this PID is still running and is memexd
        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "comm="])
                .output()?;
            
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let process_name = stdout.trim();
                
                // Check if it's actually a memexd process
                if process_name.contains("memexd") {
                    let project_info = project_id.map(|id| format!(" for project {}", id)).unwrap_or_default();
                    return Err(format!(
                        "Another memexd instance is already running{} with PID {} (process: {}). \
                         Use 'kill {}' to stop it or remove stale PID file at {}",
                        project_info, pid, process_name, pid, pid_file.display()
                    ).into());
                } else {
                    // PID exists but it's not memexd - remove stale file
                    warn!(
                        "PID file {} contains PID {} but process '{}' is not memexd, removing stale file",
                        pid_file.display(), pid, process_name
                    );
                }
            }
        }
        
        // Windows process detection
        #[cfg(windows)]
        {
            use std::process::Command;
            let output = Command::new("tasklist")
                .args(["/FI", &format!("PID eq {}", pid), "/FO", "CSV", "/NH"])
                .output()?;
            
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !stdout.trim().is_empty() && stdout.contains("memexd") {
                    return Err(format!(
                        "Another memexd instance is already running with PID {}. \
                         Use 'taskkill /PID {}' to stop it or remove stale PID file at {}",
                        pid, pid, pid_file.display()
                    ).into());
                } else if !stdout.trim().is_empty() {
                    // PID exists but it's not memexd - remove stale file
                    warn!(
                        "PID file {} contains PID {} but process is not memexd, removing stale file",
                        pid_file.display(), pid
                    );
                }
            }
        }
        
        // PID file exists but process is not running - remove stale file
        warn!("Found stale PID file {}, removing it", pid_file.display());
        fs::remove_file(pid_file)?;
    }
    Ok(())
}

/// Load configuration from file or use defaults
fn load_config(args: &DaemonArgs) -> Result<(Config, DaemonConfig), Box<dyn std::error::Error>> {
    let is_daemon_mode = detect_daemon_mode();

    match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            let config_content = fs::read_to_string(config_path)?;
            let mut daemon_config: DaemonConfig = toml::from_str(&config_content)?;

            // Override with daemon-mode settings if in daemon mode for MCP compliance
            if is_daemon_mode {
                daemon_config.qdrant.check_compatibility = false;
            }

            // Convert to engine config for backward compatibility
            let engine_config = Config::from(daemon_config.clone());

            // Note: Port configuration would be handled by IPC layer if needed
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }

            info!("Configuration loaded successfully - Qdrant transport: {:?}", daemon_config.qdrant.transport);
            Ok((engine_config, daemon_config))
        }
        None => {
            info!("Using default configuration");

            // Use daemon-mode configuration if running as daemon for MCP compliance
            let daemon_config = if is_daemon_mode {
                DaemonConfig::daemon_mode()
            } else {
                DaemonConfig::default()
            };

            let engine_config = Config::from(daemon_config.clone());

            // Note: Port configuration would be handled by IPC layer if needed
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }

            Ok((engine_config, daemon_config))
        }
    }
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
        // On non-Unix systems (Windows), only handle Ctrl+C
        signal::ctrl_c().await?;
        info!("Received Ctrl+C, initiating graceful shutdown");
    }
    
    Ok(())
}

/// Main daemon loop
async fn run_daemon(_config: Config, daemon_config: DaemonConfig, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    let project_info = args.project_id.as_ref().map(|id| format!(" for project {}", id)).unwrap_or_default();
    info!("Starting memexd daemon (version 0.2.0){}", project_info);
    
    // Check for existing instances
    check_existing_instance(&args.pid_file, args.project_id.as_ref())?;
    
    // Create PID file
    create_pid_file(&args.pid_file, args.project_id.as_ref())?;
    
    // Ensure PID file is cleaned up on exit
    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });
    
    // Initialize the processing engine with daemon configuration
    info!("Initializing ProcessingEngine with daemon configuration");
    let mut engine = ProcessingEngine::with_daemon_config(daemon_config);
    
    // Start the engine with IPC support
    info!("Starting ProcessingEngine with IPC support");
    let _ipc_client = engine.start_with_ipc().await.map_err(|e| {
        error!("Failed to start processing engine: {}", e);
        e
    })?;
    
    info!("ProcessingEngine started successfully");
    info!("IPC client available for Python integration");
    
    // Set up graceful shutdown handling
    let shutdown_future = setup_signal_handlers();
    
    // Main daemon loop
    info!("memexd daemon is running. Send SIGTERM or SIGINT to stop.");
    
    // Wait for shutdown signal
    if let Err(e) = shutdown_future.await {
        error!("Error in signal handling: {}", e);
    }
    
    // Graceful shutdown
    info!("Shutting down ProcessingEngine...");
    if let Err(e) = engine.shutdown().await {
        error!("Error during engine shutdown: {}", e);
    }
    
    info!("memexd daemon shutdown complete");
    Ok(())
}

/// Detect if we're running in daemon/service mode
fn detect_daemon_mode() -> bool {
    // Check command-line arguments for daemon indicators
    let args: Vec<String> = std::env::args().collect();
    let is_daemon = !args.iter().any(|arg| arg == "--foreground" || arg == "-f");

    // Also check environment variables that indicate service mode
    let service_context = std::env::var("WQM_SERVICE_MODE").unwrap_or_default() == "true" ||
        std::env::var("LAUNCHD_SOCKET_PATH").is_ok() || // macOS launchd
        std::env::var("SYSTEMD_EXEC_PID").is_ok() ||    // systemd
        std::env::var("SERVICE_NAME").is_ok();          // Windows service

    is_daemon || service_context
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CRITICAL: Suppress third-party library output before ANY initialization
    // This must be the very first operation to prevent early output
    let is_daemon_mode = detect_daemon_mode();

    if is_daemon_mode {
        // Suppress TTY detection debug output first, before any TTY checks
        workspace_qdrant_core::logging::suppress_tty_debug_output();
        suppress_third_party_output();
        // Set environment variable for daemon mode
        std::env::set_var("WQM_SERVICE_MODE", "true");

        // CRITICAL: Redirect stderr to suppress Qdrant client compatibility warnings
        // This must be done before any Qdrant client initialization
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            use std::fs::OpenOptions;

            // Redirect stderr to /dev/null to suppress third-party library output
            if let Ok(null_file) = OpenOptions::new().write(true).open("/dev/null") {
                unsafe {
                    libc::dup2(null_file.as_raw_fd(), libc::STDERR_FILENO);
                }
            }
        }

        #[cfg(windows)]
        {
            // Windows stderr suppression - redirect to NUL
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
    
    // Initialize comprehensive logging early
    init_logging(&args.log_level, args.foreground)?;
    
    info!("memexd daemon starting up");
    info!("Command-line arguments: {:?}", args);
    
    // Load configuration
    let (config, daemon_config) = load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;
    
    // Run the daemon
    if let Err(e) = run_daemon(config, daemon_config, args).await {
        error!("Daemon failed: {}", e);
        process::exit(1);
    }
    
    Ok(())
}