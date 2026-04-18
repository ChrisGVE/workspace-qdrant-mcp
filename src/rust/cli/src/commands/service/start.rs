//! Service start subcommand
//!
//! Behavior is driven by [`DaemonSource`](super::detect::DaemonSource):
//!
//! | Source                  | Behavior                                         |
//! |-------------------------|--------------------------------------------------|
//! | `None` / `LocalOnly`    | platform-local start (launchctl/systemd/sc.exe)  |
//! | `DockerOnly`            | stdout hint pointing at `docker compose up`      |
//! | `Both`                  | stderr warning, then platform-local start        |
//! | `RemoteOnly { addr }`   | stderr error, non-zero exit                      |
//!
//! The `plan_start()` function is a pure mapping from `DaemonSource` to
//! [`StartAction`] so every branch is unit-testable without spawning a
//! daemon.

use anyhow::{anyhow, Context, Result};
use std::process::Command;

use crate::output;

use super::detect::{detect_daemon_source, DaemonSource};
use super::messages;
use super::platform::{get_service_manager, is_daemon_running, ServiceManager, SERVICE_NAME};

/// Maximum time to wait for daemon to become responsive after loading the service.
const STARTUP_TIMEOUT_SECS: u64 = 30;

/// Interval between gRPC connectivity checks during startup.
const POLL_INTERVAL_MS: u64 = 500;

/// Decision computed from a `DaemonSource` for the `start` command.
///
/// Kept separate from the side-effecting dispatcher so every variant of the
/// behavior matrix can be asserted in unit tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartAction {
    /// Run the platform-local start flow. `warn_both` is true when the
    /// caller should first emit the `Both` warning to stderr.
    Local { warn_both: bool },
    /// Print the docker-compose up hint and exit success.
    InfoDocker,
    /// Print "cannot start remote memexd at {addr}" and exit non-zero.
    ErrorRemote { addr: String },
}

/// Pure decision: map a `DaemonSource` to the action `start` should take.
pub fn plan_start(source: &DaemonSource) -> StartAction {
    match source {
        DaemonSource::LocalOnly { .. } | DaemonSource::None => {
            StartAction::Local { warn_both: false }
        }
        DaemonSource::Both { .. } => StartAction::Local { warn_both: true },
        DaemonSource::DockerOnly => StartAction::InfoDocker,
        DaemonSource::RemoteOnly { addr } => StartAction::ErrorRemote { addr: addr.clone() },
    }
}

/// Start the daemon service
pub async fn execute() -> Result<()> {
    let source = detect_daemon_source().await;
    run(plan_start(&source)).await
}

/// Dispatch a precomputed `StartAction`. Public so tests and restart can
/// reuse the same branch handling.
pub(super) async fn run(action: StartAction) -> Result<()> {
    match action {
        StartAction::InfoDocker => {
            messages::info_docker("up");
            Ok(())
        }
        StartAction::ErrorRemote { addr } => {
            messages::err_remote("start", &addr);
            Err(anyhow!("cannot start remote memexd at {}", addr))
        }
        StartAction::Local { warn_both } => {
            if warn_both {
                messages::warn_both();
            }
            local_start().await
        }
    }
}

/// Platform-local start flow (previously inline in `execute()`).
async fn local_start() -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_local_only_runs_local() {
        assert_eq!(
            plan_start(&DaemonSource::LocalOnly { pid: 1 }),
            StartAction::Local { warn_both: false }
        );
    }

    #[test]
    fn plan_none_runs_local() {
        assert_eq!(
            plan_start(&DaemonSource::None),
            StartAction::Local { warn_both: false }
        );
    }

    #[test]
    fn plan_both_warns_then_local() {
        assert_eq!(
            plan_start(&DaemonSource::Both { pid: 42 }),
            StartAction::Local { warn_both: true }
        );
    }

    #[test]
    fn plan_docker_only_prints_info() {
        assert_eq!(
            plan_start(&DaemonSource::DockerOnly),
            StartAction::InfoDocker
        );
    }

    #[test]
    fn plan_remote_only_errors_with_addr() {
        assert_eq!(
            plan_start(&DaemonSource::RemoteOnly {
                addr: "10.0.0.1:50051".to_string()
            }),
            StartAction::ErrorRemote {
                addr: "10.0.0.1:50051".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn run_docker_only_is_ok() {
        // InfoDocker path must not error.
        let result = run(StartAction::InfoDocker).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn run_remote_only_returns_error_with_addr() {
        let result = run(StartAction::ErrorRemote {
            addr: "1.2.3.4:50051".to_string(),
        })
        .await;
        let err = result.expect_err("remote start must error");
        assert!(err.to_string().contains("1.2.3.4:50051"));
        assert!(err.to_string().contains("cannot start"));
    }
}
