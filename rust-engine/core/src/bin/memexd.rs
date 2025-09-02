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
use workspace_qdrant_core::{ProcessingEngine, config::Config};

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
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            config_file: None,
            port: None,
            log_level: "info".to_string(),
            pid_file: PathBuf::from("/tmp/memexd.pid"),
            foreground: false,
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
        .get_matches();

    DaemonArgs {
        config_file: matches.get_one::<PathBuf>("config").cloned(),
        port: matches.get_one::<u16>("port").copied(),
        log_level: matches.get_one::<String>("log-level").unwrap().clone(),
        pid_file: matches.get_one::<PathBuf>("pid-file").unwrap().clone(),
        foreground: matches.get_flag("foreground"),
    }
}

/// Initialize logging based on the specified level
fn init_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))?;

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

/// Create PID file with current process ID
fn create_pid_file(pid_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let pid = process::id();
    fs::write(pid_file, pid.to_string())?;
    info!("Created PID file at {} with PID {}", pid_file.display(), pid);
    Ok(())
}

/// Remove PID file
fn remove_pid_file(pid_file: &Path) {
    if let Err(e) = fs::remove_file(pid_file) {
        warn!("Failed to remove PID file {}: {}", pid_file.display(), e);
    } else {
        info!("Removed PID file {}", pid_file.display());
    }
}

/// Check if another instance is already running
fn check_existing_instance(pid_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        let pid_content = fs::read_to_string(pid_file)?;
        let pid: u32 = pid_content.trim().parse()?;
        
        // Check if process with this PID is still running
        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new("ps")
                .args(["-p", &pid.to_string()])
                .output()?;
            
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.lines().count() > 1 { // Header + process line
                    return Err(format!(
                        "Another memexd instance is already running with PID {}. \
                         Use 'kill {}' to stop it or remove stale PID file at {}",
                        pid, pid, pid_file.display()
                    ).into());
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
fn load_config(args: &DaemonArgs) -> Result<Config, Box<dyn std::error::Error>> {
    match &args.config_file {
        Some(config_path) => {
            info!("Loading configuration from {}", config_path.display());
            let config_content = fs::read_to_string(config_path)?;
            let config: Config = toml::from_str(&config_content)?;
            
            // Note: Port configuration would be handled by IPC layer if needed
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }
            
            Ok(config)
        }
        None => {
            info!("Using default configuration");
            let config = Config::default();
            
            // Note: Port configuration would be handled by IPC layer if needed
            if args.port.is_some() {
                info!("Port override specified: {}, but will be handled by IPC layer", args.port.unwrap());
            }
            
            Ok(config)
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
async fn run_daemon(config: Config, args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting memexd daemon (version 0.2.0)");
    
    // Check for existing instances
    check_existing_instance(&args.pid_file)?;
    
    // Create PID file
    create_pid_file(&args.pid_file)?;
    
    // Ensure PID file is cleaned up on exit
    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });
    
    // Initialize the processing engine
    info!("Initializing ProcessingEngine with configuration");
    let mut engine = ProcessingEngine::with_config(config);
    
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    
    // Initialize logging early
    init_logging(&args.log_level)?;
    
    info!("memexd daemon starting up");
    info!("Command-line arguments: {:?}", args);
    
    // Load configuration
    let config = load_config(&args).map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;
    
    // Run the daemon
    if let Err(e) = run_daemon(config, args).await {
        error!("Daemon failed: {}", e);
        process::exit(1);
    }
    
    Ok(())
}