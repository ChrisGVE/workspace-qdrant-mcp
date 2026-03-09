//! Service stop subcommand

use anyhow::{Context, Result};
use std::process::Command;

use crate::output;

use super::platform::{
    get_service_manager, is_daemon_running, ServiceManager, DAEMON_BINARY, SERVICE_NAME,
};

/// Maximum time to wait for daemon to fully shut down.
const SHUTDOWN_TIMEOUT_SECS: u64 = 10;

/// Interval between checks during shutdown.
const POLL_INTERVAL_MS: u64 = 300;

/// Stop the daemon service
pub async fn execute() -> Result<()> {
    output::info("Stopping daemon...");

    match get_service_manager() {
        ServiceManager::Launchctl => stop_launchctl().await,
        ServiceManager::Systemd => stop_systemd().await,
        ServiceManager::WindowsService => stop_windows().await,
        _ => {
            // Fallback: try to kill by process name
            #[cfg(unix)]
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();
            #[cfg(windows)]
            let _ = Command::new("taskkill")
                .args(["/F", "/IM", "memexd.exe"])
                .status();
            output::info("Attempted to stop daemon");
            Ok(())
        }
    }
}

/// Wait until the daemon is no longer responding, or timeout.
///
/// Returns `true` if the daemon stopped successfully.
async fn wait_for_shutdown(timeout_secs: u64) -> bool {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);

    while tokio::time::Instant::now() < deadline {
        if !is_daemon_running().await {
            return true;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
    }
    false
}

async fn stop_launchctl() -> Result<()> {
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

    // Also try killing by name (catches orphaned processes)
    let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();

    if wait_for_shutdown(SHUTDOWN_TIMEOUT_SECS).await {
        output::success("Daemon stopped");
    } else {
        // Force kill as last resort
        let _ = Command::new("pkill")
            .args(["-9", "-f", DAEMON_BINARY])
            .status();
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        if !is_daemon_running().await {
            output::success("Daemon stopped (force kill)");
        } else {
            output::warning("Daemon may still be running");
        }
    }

    Ok(())
}

async fn stop_systemd() -> Result<()> {
    let status = Command::new("systemctl")
        .args(["--user", "stop", "memexd"])
        .status()?;

    if status.success() {
        if wait_for_shutdown(SHUTDOWN_TIMEOUT_SECS).await {
            output::success("Daemon stopped");
        } else {
            output::warning("Daemon may still be running");
        }
    } else {
        output::warning("Failed to stop daemon via systemd");
    }

    Ok(())
}

async fn stop_windows() -> Result<()> {
    #[cfg(windows)]
    {
        let status = Command::new("sc.exe").args(["stop", "memexd"]).status()?;

        if wait_for_shutdown(SHUTDOWN_TIMEOUT_SECS).await {
            output::success("Daemon stopped");
        } else {
            // Try force kill as fallback
            let _ = Command::new("taskkill")
                .args(["/F", "/IM", "memexd.exe"])
                .status();
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
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

    Ok(())
}
