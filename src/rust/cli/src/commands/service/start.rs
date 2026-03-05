//! Service start subcommand

use anyhow::{Context, Result};
use std::process::Command;

use crate::output;

use super::platform::{get_service_manager, is_daemon_running, ServiceManager, SERVICE_NAME};

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

    Ok(())
}

async fn start_systemd() -> Result<()> {
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

    Ok(())
}

async fn start_windows() -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("sc.exe").args(["start", "memexd"]).status()?;

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

    Ok(())
}
