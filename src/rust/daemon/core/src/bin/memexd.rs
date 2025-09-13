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
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use workspace_qdrant_core::{
    ProcessingEngine, config::{Config, DaemonConfig}, 
    LoggingConfig, initialize_logging,
    ErrorRecovery, ErrorRecoveryStrategy,
    track_async_operation, LoggingErrorMonitor,
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

/// Parse command-line arguments
fn parse_args() -> DaemonArgs {
    let matches = Command::new("memexd")
        .version("0.2.0")
        .author("Christian C. Berclaz <christian.berclaz@mac.com>")
        .about("Memory eXchange Daemon - Document processing and embedding generation service")
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
        .get_matches();

    DaemonArgs {
        config_file: matches.get_one::<PathBuf>("config").cloned(),
        port: matches.get_one::<u16>("port").copied(),
        log_level: matches.get_one::<String>("log-level").unwrap().clone(),
        pid_file: matches.get_one::<PathBuf>("pid-file").unwrap().clone(),
        foreground: matches.get_flag("foreground"),
        project_id: matches.get_one::<String>("project-id").cloned(),
    }
}

/// Initialize comprehensive logging based on the specified level
fn init_logging(log_level: &str, foreground: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Set service mode environment variable if not running in foreground
    if !foreground {
        std::env::set_var("WQM_SERVICE_MODE", "true");
    }
    
    let mut config = if foreground {
        LoggingConfig::development()
    } else {
        LoggingConfig::production()
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
    
    // Configure based on daemon mode
    if !foreground {
        // For daemon mode, disable file logging since launchd handles redirection
        // The plist already redirects stdout/stderr to user-writable log files
        config.json_format = false; // Keep readable format for launchd logs
        config.file_logging = false; // Let launchd handle file logging
        config.force_disable_ansi = Some(true); // Force disable ANSI colors in service mode
        
        // Also set environment variable for consistency
        std::env::set_var("NO_COLOR", "1");
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
    match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            let config_content = fs::read_to_string(config_path)?;
            let daemon_config: DaemonConfig = toml::from_str(&config_content)?;
            
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
            let daemon_config = DaemonConfig::default();
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
async fn run_daemon(config: Config, daemon_config: DaemonConfig, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
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

/// Suppress third-party library output in daemon mode
fn suppress_third_party_output() {
    // ONNX Runtime suppression - prevent initialization messages
    std::env::set_var("ORT_LOGGING_LEVEL", "4"); // Silent mode (0=verbose, 4=fatal only)
    std::env::set_var("OMP_NUM_THREADS", "1"); // Reduce OpenMP threading messages
    std::env::set_var("OPENBLAS_NUM_THREADS", "1"); // OpenBLAS threading control

    // Tokenizers suppression - prevent HuggingFace tokenizer messages
    std::env::set_var("TOKENIZERS_PARALLELISM", "false");
    std::env::set_var("HF_HUB_DISABLE_PROGRESS_BARS", "1");
    std::env::set_var("HF_HUB_DISABLE_TELEMETRY", "1");
    std::env::set_var("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1");

    // Tracing/logging suppression for third-party crates
    // Note: RUST_LOG is set by our logging system, but ensure no conflicts
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "off");
    }

    // Python integration suppression (if FastEmbed uses Python bindings)
    std::env::set_var("PYTHONPATH", ""); // Clear Python integration
    std::env::set_var("PYTHONIOENCODING", "utf-8"); // Prevent encoding warnings
    std::env::set_var("PYTHONUNBUFFERED", "1"); // Immediate output (easier to suppress)

    // CUDA/GPU library suppression
    std::env::set_var("CUDA_VISIBLE_DEVICES", ""); // Disable CUDA if not needed
    std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "3"); // TensorFlow fatal only

    // General ML library suppression
    std::env::set_var("MPLCONFIGDIR", "/tmp"); // matplotlib config to temp
    std::env::set_var("TRANSFORMERS_OFFLINE", "1"); // Prevent network calls
    std::env::set_var("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1"); // Disable warnings

    // Disable terminal detection and TTY-related output
    std::env::set_var("TERM", "dumb"); // Disable terminal features
    std::env::set_var("NO_COLOR", "1"); // Force disable colors globally
    std::env::set_var("FORCE_COLOR", "0"); // Explicitly disable color
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
        suppress_third_party_output();
    }

    let args = parse_args();
    
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