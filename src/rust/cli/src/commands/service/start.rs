//! Service start subcommand

use anyhow::{Context, Result};
use std::process::Command;

use crate::output;

use super::platform::{get_service_manager, is_daemon_running, ServiceManager, SERVICE_NAME};

/// Maximum time to wait for daemon to become responsive after loading the service.
const STARTUP_TIMEOUT_SECS: u64 = 30;

/// Interval between gRPC connectivity checks during startup.
const POLL_INTERVAL_MS: u64 = 500;

/// Start the daemon service
pub async fn execute() -> Result<()> {
    output::info("Starting daemon...");

    if is_daemon_running().await {
        output::info("Daemon is already running");
        return Ok(());
    }

    match get_service_manager() {
        ServiceManager::Launchctl => start_launchctl().await,
        ServiceManager::Systemd => start_systemd().await,
        ServiceManager::WindowsService => start_windows().await,
        _ => {
            output::error("Service management not supported on this platform");
            Ok(())
        }
    }
}

/// Poll gRPC until the daemon responds or timeout is reached.
///
/// Returns `true` if the daemon became responsive.
async fn wait_for_daemon(timeout_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);

    while tokio::time::Instant::now() < deadline {
        if is_daemon_running().await {
            return true;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
    }
    false
}

async fn start_launchctl() -> Result<()> {
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
            if wait_for_daemon(STARTUP_TIMEOUT_SECS).await {
                output::success("Daemon started");
            } else {
                output::error("Daemon failed to start within timeout");
                output::info("Check logs: wqm service logs");
            }
        } else {
            output::error("Failed to load service plist");
        }
    } else {
        output::error("Service not installed. Run: wqm service install");
    }

    Ok(())
}

async fn start_systemd() -> Result<()> {
    let status = Command::new("systemctl")
        .args(["--user", "start", "memexd"])
        .status()?;

    if status.success() {
        if wait_for_daemon(STARTUP_TIMEOUT_SECS).await {
            output::success("Daemon started");
        } else {
            output::error("Daemon failed to start within timeout");
            output::info("Check logs: journalctl --user -u memexd");
        }
    } else {
        output::error("Failed to start daemon");
    }

    Ok(())
}

async fn start_windows() -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("sc.exe").args(["start", "memexd"]).status()?;

        if status.success() {
            if wait_for_daemon(STARTUP_TIMEOUT_SECS).await {
                output::success("Daemon started");
            } else {
                output::error("Daemon failed to start within timeout");
                output::info("Check Windows Event Viewer for details");
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

    Ok(())
}
