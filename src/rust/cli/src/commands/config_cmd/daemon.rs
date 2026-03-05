//! Daemon lifecycle helpers used during config migration.
//!
//! Provides functions to check, stop, and start the daemon process
//! so that configuration files can be safely moved while the daemon
//! is not running.

use anyhow::{bail, Context, Result};

use crate::grpc::client::DaemonClient;
use crate::output;

/// Check if daemon is reachable via gRPC.
pub(super) async fn is_daemon_running() -> bool {
    DaemonClient::connect_default().await.is_ok()
}

/// Stop the daemon and wait for it to shut down.
/// Returns true if the daemon was stopped or already not running.
pub(super) async fn stop_daemon() -> Result<bool> {
    if !is_daemon_running().await {
        return Ok(true);
    }

    output::info("Stopping daemon...");

    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .context("Could not find home directory")?
            .join("Library/LaunchAgents/com.workspace-qdrant.memexd.plist");

        if plist_path.exists() {
            let _ = std::process::Command::new("launchctl")
                .args(["unload"])
                .arg(&plist_path)
                .status();
        }
        let _ = std::process::Command::new("pkill")
            .args(["-f", "memexd"])
            .status();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "stop", "memexd"])
            .status();
    }

    // Wait for daemon to stop
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    if is_daemon_running().await {
        bail!("Could not stop daemon cleanly. Please stop it manually and retry.");
    }

    output::success("Daemon stopped");
    Ok(true)
}

/// Start the daemon after config migration.
pub(super) async fn start_daemon() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .context("Could not find home directory")?
            .join("Library/LaunchAgents/com.workspace-qdrant.memexd.plist");

        if plist_path.exists() {
            let status = std::process::Command::new("launchctl")
                .args(["load", "-w"])
                .arg(&plist_path)
                .status()?;
            if status.success() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if is_daemon_running().await {
                    output::success("Daemon restarted");
                } else {
                    output::warning("Service loaded but daemon not responding yet");
                }
                return Ok(());
            }
        }
        output::warning("Could not restart daemon. Start manually: wqm service start");
    }

    #[cfg(target_os = "linux")]
    {
        let status = std::process::Command::new("systemctl")
            .args(["--user", "start", "memexd"])
            .status()?;
        if status.success() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            if is_daemon_running().await {
                output::success("Daemon restarted");
            } else {
                output::warning("Service started but daemon not responding yet");
            }
            return Ok(());
        }
        output::warning("Could not restart daemon. Start manually: wqm service start");
    }

    Ok(())
}
