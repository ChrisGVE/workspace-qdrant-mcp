//! Service stop subcommand
//!
//! Behavior matrix driven by [`DaemonSource`](super::detect::DaemonSource):
//!
//! | Source                  | Behavior                                         |
//! |-------------------------|--------------------------------------------------|
//! | `LocalOnly` / `Both`    | platform-local stop (warning first for `Both`)   |
//! | `DockerOnly`            | stdout hint pointing at `docker compose down`    |
//! | `RemoteOnly { addr }`   | stderr error, non-zero exit                      |
//! | `None`                  | stderr "nothing running", non-zero exit          |
//!
//! `plan_stop()` is a pure mapping from `DaemonSource` to [`StopAction`],
//! keeping all branch selection unit-testable.

use anyhow::{anyhow, Context, Result};
use std::process::Command;

use crate::output;

use super::detect::{detect_daemon_source, DaemonSource};
use super::messages;
use super::platform::{
    get_service_manager, is_daemon_running, ServiceManager, DAEMON_BINARY, SERVICE_NAME,
};

/// Maximum time to wait for daemon to fully shut down.
const SHUTDOWN_TIMEOUT_SECS: u64 = 10;

/// Interval between checks during shutdown.
const POLL_INTERVAL_MS: u64 = 300;

/// Decision computed from a `DaemonSource` for the `stop` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopAction {
    /// Run the platform-local stop flow. `warn_both` triggers the
    /// "local+docker" stderr warning before stopping.
    Local { warn_both: bool },
    /// Print the `docker compose down` hint and exit success.
    InfoDocker,
    /// Print "cannot stop remote memexd at {addr}" and exit non-zero.
    ErrorRemote { addr: String },
    /// Print "nothing running" and exit non-zero.
    ErrorNothing,
}

/// Pure decision: map a `DaemonSource` to the action `stop` should take.
pub fn plan_stop(source: &DaemonSource) -> StopAction {
    match source {
        DaemonSource::LocalOnly { .. } => StopAction::Local { warn_both: false },
        DaemonSource::Both { .. } => StopAction::Local { warn_both: true },
        DaemonSource::DockerOnly => StopAction::InfoDocker,
        DaemonSource::RemoteOnly { addr } => StopAction::ErrorRemote { addr: addr.clone() },
        DaemonSource::None => StopAction::ErrorNothing,
    }
}

/// Stop the daemon service
pub async fn execute() -> Result<()> {
    let source = detect_daemon_source().await;
    run(plan_stop(&source)).await
}

/// Dispatch a precomputed `StopAction`.
pub(super) async fn run(action: StopAction) -> Result<()> {
    match action {
        StopAction::InfoDocker => {
            messages::info_docker("down");
            Ok(())
        }
        StopAction::ErrorRemote { addr } => {
            messages::err_remote("stop", &addr);
            Err(anyhow!("cannot stop remote memexd at {}", addr))
        }
        StopAction::ErrorNothing => {
            messages::err_nothing_running();
            Err(anyhow!("nothing running"))
        }
        StopAction::Local { warn_both } => {
            if warn_both {
                messages::warn_both();
            }
            local_stop().await
        }
    }
}

/// Platform-local stop flow (previously inline in `execute()`).
async fn local_stop() -> Result<()> {
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

        if !status.success() {
            output::warning(
                "sc.exe stop returned non-zero — service may not be installed or already stopped; falling back to direct process check",
            );
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_local_only_runs_local() {
        assert_eq!(
            plan_stop(&DaemonSource::LocalOnly { pid: 1 }),
            StopAction::Local { warn_both: false }
        );
    }

    #[test]
    fn plan_both_warns_then_local() {
        assert_eq!(
            plan_stop(&DaemonSource::Both { pid: 42 }),
            StopAction::Local { warn_both: true }
        );
    }

    #[test]
    fn plan_docker_only_prints_info() {
        assert_eq!(plan_stop(&DaemonSource::DockerOnly), StopAction::InfoDocker);
    }

    #[test]
    fn plan_remote_only_errors_with_addr() {
        assert_eq!(
            plan_stop(&DaemonSource::RemoteOnly {
                addr: "172.16.0.5:50051".to_string()
            }),
            StopAction::ErrorRemote {
                addr: "172.16.0.5:50051".to_string(),
            }
        );
    }

    #[test]
    fn plan_none_errors_nothing_running() {
        assert_eq!(plan_stop(&DaemonSource::None), StopAction::ErrorNothing);
    }

    #[tokio::test]
    async fn run_docker_only_is_ok() {
        assert!(run(StopAction::InfoDocker).await.is_ok());
    }

    #[tokio::test]
    async fn run_remote_returns_error_with_addr() {
        let err = run(StopAction::ErrorRemote {
            addr: "5.5.5.5:50051".to_string(),
        })
        .await
        .expect_err("remote stop must error");
        assert!(err.to_string().contains("5.5.5.5:50051"));
        assert!(err.to_string().contains("cannot stop"));
    }

    #[tokio::test]
    async fn run_nothing_returns_error() {
        let err = run(StopAction::ErrorNothing)
            .await
            .expect_err("nothing-running stop must error");
        assert!(err.to_string().contains("nothing running"));
    }
}
