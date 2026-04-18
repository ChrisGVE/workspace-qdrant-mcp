//! Service restart subcommand
//!
//! Behavior matrix driven by [`DaemonSource`](super::detect::DaemonSource):
//!
//! | Source                  | Behavior                                         |
//! |-------------------------|--------------------------------------------------|
//! | `LocalOnly` / `Both`    | local stop, grace period, local start            |
//! | `DockerOnly`            | single hint: `docker compose restart`            |
//! | `RemoteOnly { addr }`   | error, skip                                      |
//! | `None`                  | skip stop; run start (which starts locally)      |
//!
//! `plan_restart()` is a pure mapping so the decision is unit-testable
//! without real processes.

use anyhow::{anyhow, Result};

use super::detect::{detect_daemon_source, DaemonSource};
use super::messages;
use super::start::{self, StartAction};
use super::stop::{self, StopAction};

/// Decision computed from a `DaemonSource` for the `restart` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAction {
    /// Stop locally, wait, then start locally. `warn_both` triggers the
    /// local+docker warning exactly once, before the stop phase.
    Local { warn_both: bool },
    /// Print a single `docker compose restart` hint and exit success.
    InfoDocker,
    /// Print "cannot restart remote memexd at {addr}" and exit non-zero.
    ErrorRemote { addr: String },
    /// No daemon was detected at all — just run the local start flow.
    StartOnly,
}

/// Pure decision: map a `DaemonSource` to the action `restart` should take.
pub fn plan_restart(source: &DaemonSource) -> RestartAction {
    match source {
        DaemonSource::LocalOnly { .. } => RestartAction::Local { warn_both: false },
        DaemonSource::Both { .. } => RestartAction::Local { warn_both: true },
        DaemonSource::DockerOnly => RestartAction::InfoDocker,
        DaemonSource::RemoteOnly { addr } => RestartAction::ErrorRemote { addr: addr.clone() },
        DaemonSource::None => RestartAction::StartOnly,
    }
}

/// Restart the daemon service (stop then start)
///
/// For local sources, waits for confirmed shutdown before starting the new
/// instance. This prevents port conflicts and ensures clean state
/// transitions.
pub async fn execute() -> Result<()> {
    let source = detect_daemon_source().await;
    run(plan_restart(&source)).await
}

/// Dispatch a precomputed `RestartAction`.
pub(super) async fn run(action: RestartAction) -> Result<()> {
    match action {
        RestartAction::InfoDocker => {
            messages::info_docker("restart");
            Ok(())
        }
        RestartAction::ErrorRemote { addr } => {
            messages::err_remote("restart", &addr);
            Err(anyhow!("cannot restart remote memexd at {}", addr))
        }
        RestartAction::StartOnly => {
            // No daemon detected: nothing to stop; just start locally.
            start::run(StartAction::Local { warn_both: false }).await
        }
        RestartAction::Local { warn_both } => {
            // The warning belongs to the stop phase; do not re-emit during start.
            stop::run(StopAction::Local { warn_both }).await?;
            // Small grace period after confirmed shutdown to release OS
            // resources (file locks, TCP TIME_WAIT, etc.)
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            start::run(StartAction::Local { warn_both: false }).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_local_only_is_local_no_warn() {
        assert_eq!(
            plan_restart(&DaemonSource::LocalOnly { pid: 1 }),
            RestartAction::Local { warn_both: false }
        );
    }

    #[test]
    fn plan_both_is_local_with_warn() {
        assert_eq!(
            plan_restart(&DaemonSource::Both { pid: 42 }),
            RestartAction::Local { warn_both: true }
        );
    }

    #[test]
    fn plan_docker_only_is_info() {
        assert_eq!(
            plan_restart(&DaemonSource::DockerOnly),
            RestartAction::InfoDocker
        );
    }

    #[test]
    fn plan_remote_only_errors_with_addr() {
        assert_eq!(
            plan_restart(&DaemonSource::RemoteOnly {
                addr: "10.1.1.1:50051".to_string()
            }),
            RestartAction::ErrorRemote {
                addr: "10.1.1.1:50051".to_string(),
            }
        );
    }

    #[test]
    fn plan_none_is_start_only() {
        assert_eq!(plan_restart(&DaemonSource::None), RestartAction::StartOnly);
    }

    #[tokio::test]
    async fn run_docker_only_is_ok() {
        assert!(run(RestartAction::InfoDocker).await.is_ok());
    }

    #[tokio::test]
    async fn run_remote_returns_error_with_addr() {
        let err = run(RestartAction::ErrorRemote {
            addr: "9.9.9.9:50051".to_string(),
        })
        .await
        .expect_err("remote restart must error");
        assert!(err.to_string().contains("9.9.9.9:50051"));
        assert!(err.to_string().contains("cannot restart"));
    }
}
