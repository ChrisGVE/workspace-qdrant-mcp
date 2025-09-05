//! memexd - Service Installation Demo
//!
//! A simplified daemon to demonstrate pure daemon architecture
//! and service installation capabilities.

use clap::{Arg, Command};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Duration;
use tokio::signal;
use tokio::time::interval;

/// Command-line arguments for demo daemon
#[derive(Debug, Clone)]
struct DaemonArgs {
    pid_file: PathBuf,
}

impl Default for DaemonArgs {
    fn default() -> Self {
        Self {
            pid_file: PathBuf::from("/tmp/memexd-demo.pid"),
        }
    }
}

/// Parse command-line arguments
fn parse_args() -> DaemonArgs {
    let matches = Command::new("memexd-service-demo")
        .version("0.2.0")
        .author("Christian C. Berclaz <christian.berclaz@mac.com>")
        .about("Memory eXchange Daemon - Service Installation Demo")
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
                .default_value("/tmp/memexd-demo.pid")
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
        pid_file: matches.get_one::<PathBuf>("pid-file").unwrap().clone(),
    }
}

/// Create PID file
fn create_pid_file(pid_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let pid = process::id();
    fs::write(pid_file, pid.to_string())?;
    println!("Created PID file at {} with PID {}", pid_file.display(), pid);
    Ok(())
}

/// Remove PID file
fn remove_pid_file(pid_file: &PathBuf) {
    if let Err(e) = fs::remove_file(pid_file) {
        eprintln!("Failed to remove PID file {}: {}", pid_file.display(), e);
    } else {
        println!("Removed PID file {}", pid_file.display());
    }
}

/// Set up signal handlers for graceful shutdown
async fn setup_signal_handlers() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        tokio::select! {
            _ = signal::ctrl_c() => {
                println!("Received Ctrl+C, initiating graceful shutdown");
            }
        }
    }
    
    #[cfg(not(unix))]
    {
        signal::ctrl_c().await?;
        println!("Received Ctrl+C, initiating graceful shutdown");
    }
    
    Ok(())
}

/// Simulate priority-based task processing
async fn run_priority_demo() {
    let mut interval = interval(Duration::from_secs(5));
    let mut task_count = 0;
    
    loop {
        tokio::select! {
            _ = interval.tick() => {
                task_count += 1;
                
                // Simulate priority-based processing
                let priority_mode = env::var("MEMEXD_PRIORITY_MODE").unwrap_or_else(|_| "enabled".to_string());
                let high_queue_size = env::var("MEMEXD_HIGH_PRIORITY_QUEUE_SIZE")
                    .unwrap_or_else(|_| "100".to_string());
                let low_queue_size = env::var("MEMEXD_LOW_PRIORITY_QUEUE_SIZE")
                    .unwrap_or_else(|_| "1000".to_string());
                
                println!(
                    "Task #{}: Priority mode={}, High queue limit={}, Low queue limit={}",
                    task_count, priority_mode, high_queue_size, low_queue_size
                );
                
                // Simulate different types of tasks
                match task_count % 3 {
                    0 => println!("  Processing high-priority MCP request"),
                    1 => println!("  Processing current project task"),
                    2 => println!("  Processing background ingestion (throttled if MCP active)"),
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                println!("Service health check: OK");
            }
        }
    }
}

/// Main daemon loop
async fn run_daemon(args: DaemonArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting memexd service demo (version 0.2.0)");
    println!("Args: {:?}", args);
    
    // Create PID file
    create_pid_file(&args.pid_file)?;
    
    // Ensure PID file is cleaned up on exit
    let pid_file_cleanup = args.pid_file.clone();
    let _cleanup_guard = scopeguard::guard((), move |_| {
        remove_pid_file(&pid_file_cleanup);
    });
    
    println!("memexd service demo is running with priority-based resource management");
    println!("  Environment variables:");
    println!("    MEMEXD_PRIORITY_MODE={}", env::var("MEMEXD_PRIORITY_MODE").unwrap_or_else(|_| "enabled".to_string()));
    println!("    MEMEXD_HIGH_PRIORITY_QUEUE_SIZE={}", env::var("MEMEXD_HIGH_PRIORITY_QUEUE_SIZE").unwrap_or_else(|_| "100".to_string()));
    println!("    MEMEXD_LOW_PRIORITY_QUEUE_SIZE={}", env::var("MEMEXD_LOW_PRIORITY_QUEUE_SIZE").unwrap_or_else(|_| "1000".to_string()));
    println!("    MEMEXD_RESOURCE_THROTTLE={}", env::var("MEMEXD_RESOURCE_THROTTLE").unwrap_or_else(|_| "enabled".to_string()));
    println!("Send SIGTERM or SIGINT to stop.");
    
    // Start priority demo
    let priority_demo = tokio::spawn(run_priority_demo());
    
    // Set up graceful shutdown handling
    let shutdown_future = setup_signal_handlers();
    
    // Wait for shutdown signal
    if let Err(e) = shutdown_future.await {
        eprintln!("Error in signal handling: {}", e);
    }
    
    // Graceful shutdown
    println!("Shutting down priority demo...");
    priority_demo.abort();
    
    println!("memexd service demo shutdown complete");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    
    println!("memexd service demo starting with pure daemon architecture");
    
    // Run the daemon
    if let Err(e) = run_daemon(args).await {
        eprintln!("Service demo failed: {}", e);
        process::exit(1);
    }
    
    Ok(())
}