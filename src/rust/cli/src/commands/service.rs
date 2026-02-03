//! Service command - daemon lifecycle management
//!
//! Manages the memexd daemon runtime.
//! Subcommands: start, stop, restart, status
//!
//! Note: Installation/uninstallation is handled by platform-specific installers,
//! not by the CLI. Use `wqm debug logs` to view daemon logs.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::process::Command;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Service command arguments
#[derive(Args)]
pub struct ServiceArgs {
    #[command(subcommand)]
    command: ServiceCommand,
}

/// Service subcommands
#[derive(Subcommand)]
enum ServiceCommand {
    /// Start the daemon
    Start,

    /// Stop the daemon
    Stop,

    /// Restart the daemon
    Restart,

    /// Show daemon status
    Status,
}

/// Execute service command
pub async fn execute(args: ServiceArgs) -> Result<()> {
    match args.command {
        ServiceCommand::Start => start().await,
        ServiceCommand::Stop => stop().await,
        ServiceCommand::Restart => restart().await,
        ServiceCommand::Status => status().await,
    }
}

/// Check if daemon is running by attempting gRPC connection
async fn is_daemon_running() -> bool {
    DaemonClient::connect_default().await.is_ok()
}

/// Get platform-specific service manager
fn get_service_manager() -> ServiceManager {
    #[cfg(target_os = "macos")]
    {
        ServiceManager::Launchctl
    }
    #[cfg(target_os = "linux")]
    {
        ServiceManager::Systemd
    }
    #[cfg(target_os = "windows")]
    {
        ServiceManager::WindowsService
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        ServiceManager::Unknown
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Variants used based on target platform
enum ServiceManager {
    Launchctl,
    Systemd,
    WindowsService,
    Unknown,
}

const SERVICE_NAME: &str = "com.workspace-qdrant.memexd";
const DAEMON_BINARY: &str = "memexd";

async fn start() -> Result<()> {
    output::info("Starting daemon...");

    if is_daemon_running().await {
        output::info("Daemon is already running");
        return Ok(());
    }

    match get_service_manager() {
        ServiceManager::Launchctl => {
            let plist_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", SERVICE_NAME));

            if plist_path.exists() {
                let status = Command::new("launchctl")
                    .args(["load", "-w"])
                    .arg(&plist_path)
                    .status()?;

                if status.success() {
                    // Wait a moment for daemon to start
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    if is_daemon_running().await {
                        output::success("Daemon started");
                    } else {
                        output::warning("Service loaded but daemon not responding");
                    }
                } else {
                    output::error("Failed to start daemon");
                }
            } else {
                output::error("Service not installed. Run: wqm service install");
            }
        }
        ServiceManager::Systemd => {
            let status = Command::new("systemctl")
                .args(["--user", "start", "memexd"])
                .status()?;

            if status.success() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if is_daemon_running().await {
                    output::success("Daemon started");
                } else {
                    output::warning("Service started but daemon not responding");
                }
            } else {
                output::error("Failed to start daemon");
            }
        }
        ServiceManager::WindowsService => {
            #[cfg(windows)]
            {
                let status = Command::new("sc.exe")
                    .args(["start", "memexd"])
                    .status()?;

                if status.success() {
                    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                    if is_daemon_running().await {
                        output::success("Daemon started");
                    } else {
                        output::warning("Service started but daemon not responding yet");
                        output::info("Try: wqm service status");
                    }
                } else {
                    output::error("Failed to start daemon");
                    output::info("Check if service is installed: wqm service status");
                }
            }
            #[cfg(not(windows))]
            {
                output::error("Windows service start only available on Windows");
            }
        }
        _ => {
            output::error("Service management not supported on this platform");
        }
    }

    Ok(())
}

async fn stop() -> Result<()> {
    output::info("Stopping daemon...");

    match get_service_manager() {
        ServiceManager::Launchctl => {
            let plist_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", SERVICE_NAME));

            if plist_path.exists() {
                let _ = Command::new("launchctl")
                    .args(["unload"])
                    .arg(&plist_path)
                    .status();
            }

            // Also try killing by name
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();

            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

            if !is_daemon_running().await {
                output::success("Daemon stopped");
            } else {
                output::warning("Daemon may still be running");
            }
        }
        ServiceManager::Systemd => {
            let status = Command::new("systemctl")
                .args(["--user", "stop", "memexd"])
                .status()?;

            if status.success() {
                output::success("Daemon stopped");
            } else {
                output::warning("Failed to stop daemon via systemd");
            }
        }
        ServiceManager::WindowsService => {
            #[cfg(windows)]
            {
                let status = Command::new("sc.exe")
                    .args(["stop", "memexd"])
                    .status()?;

                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                if !is_daemon_running().await {
                    output::success("Daemon stopped");
                } else {
                    // Try force kill as fallback
                    let _ = Command::new("taskkill")
                        .args(["/F", "/IM", "memexd.exe"])
                        .status();

                    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

                    if !is_daemon_running().await {
                        output::success("Daemon stopped (force kill)");
                    } else {
                        output::warning("Daemon may still be running");
                    }
                }
            }
            #[cfg(not(windows))]
            {
                output::error("Windows service stop only available on Windows");
            }
        }
        _ => {
            // Fallback: try to kill by process name
            #[cfg(unix)]
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();
            #[cfg(windows)]
            let _ = Command::new("taskkill").args(["/F", "/IM", "memexd.exe"]).status();
            output::info("Attempted to stop daemon");
        }
    }

    Ok(())
}

async fn restart() -> Result<()> {
    output::info("Restarting daemon...");
    stop().await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    start().await?;
    Ok(())
}

async fn status() -> Result<()> {
    output::section("Daemon Status");

    // Try to connect and get health
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Connection", ServiceStatus::Healthy);

            // Get health check
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let status = ServiceStatus::from_proto(health.status);
                    output::status_line("Health", status);

                    // Show component health
                    if !health.components.is_empty() {
                        output::separator();
                        for comp in health.components {
                            let comp_status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&comp.component_name, comp_status);
                            if !comp.message.is_empty() {
                                output::kv("  Message", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => {
                    output::status_line("Health", ServiceStatus::Unknown);
                    output::warning(format!("Could not get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Connection", ServiceStatus::Unhealthy);
            output::error("Daemon not running or not reachable");
            output::info("Start with: wqm service start");
        }
    }

    Ok(())
}
