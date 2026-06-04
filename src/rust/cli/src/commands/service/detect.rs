//! Daemon runtime detection.
//!
//! Inspects the host environment to classify where (if anywhere) a `memexd`
//! daemon is currently running: a locally-launched process, a Docker
//! container, a remote gRPC endpoint, or none of the above.
//!
//! This module is intentionally isolated: it only exposes pure detection
//! primitives and an orchestrator. Wiring into `wqm service start/stop/
//! restart/status` happens in a separate task.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Default PID file location written by `memexd` at startup.
const DEFAULT_PID_FILE: &str = "/tmp/memexd.pid";

/// Default local gRPC address probed for a running daemon.
const DEFAULT_GRPC_ADDR: &str = "127.0.0.1:50051";

/// gRPC health probe timeout. Detection must be snappy so CLI commands are
/// responsive even when no daemon is listening.
const GRPC_PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Where a `memexd` daemon is running, from the CLI's point of view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonSource {
    /// A locally-spawned `memexd` process is alive (no container, no remote).
    LocalOnly { pid: u32 },
    /// A Docker container named `memexd` is running (no local PID, no remote).
    DockerOnly,
    /// Both a local process and a Docker container are running. Ambiguous —
    /// callers should surface a warning to the user.
    Both { pid: u32 },
    /// Only a remote/unclassified gRPC endpoint responds on the default port.
    RemoteOnly { addr: String },
    /// No daemon detected anywhere.
    None,
}

impl fmt::Display for DaemonSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DaemonSource::LocalOnly { pid } => write!(f, "local (pid {pid})"),
            DaemonSource::DockerOnly => write!(f, "docker container"),
            DaemonSource::Both { pid } => {
                write!(f, "local (pid {pid}) and docker container")
            }
            DaemonSource::RemoteOnly { addr } => write!(f, "remote gRPC @ {addr}"),
            DaemonSource::None => write!(f, "none"),
        }
    }
}

/// Default PID file path. Wrapped so tests can use the `_at` variants.
fn pid_file_path() -> PathBuf {
    PathBuf::from(DEFAULT_PID_FILE)
}

/// Read and validate the daemon PID file at the default location.
///
/// Returns `Some(pid)` iff the file exists, parses as a `u32`, and the
/// corresponding process is alive. Any other condition (missing file,
/// corrupted contents, dead pid) yields `None`.
#[allow(dead_code)] // wiring into service commands lands in task 11
fn check_local_pid() -> Option<u32> {
    check_local_pid_at(&pid_file_path())
}

/// Same as [`check_local_pid`] but takes an explicit path, to make the
/// liveness check unit-testable without touching `/tmp/memexd.pid`.
fn check_local_pid_at(path: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pid: u32 = raw.trim().parse().ok()?;
    if is_pid_alive(pid) {
        Some(pid)
    } else {
        None
    }
}

/// Verify `pid` corresponds to a live process.
///
/// Uses `kill(pid, 0)` (signal 0) on Unix, which performs the permission /
/// existence check without delivering a signal. On non-Unix targets we
/// conservatively return `false` so detection degrades to the gRPC / Docker
/// checks rather than reporting a false `LocalOnly`.
fn is_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        use nix::sys::signal::kill;
        use nix::unistd::Pid;
        let Ok(pid_i32) = i32::try_from(pid) else {
            return false;
        };
        // `None` = signal 0; succeeds iff the process exists and we have
        // permission to signal it. `ESRCH` means no such process.
        kill(Pid::from_raw(pid_i32), None).is_ok()
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// Check whether a Docker container named `memexd` is currently running.
///
/// Runs `docker ps --filter name=^memexd$ --quiet`. If the `docker` CLI is
/// missing or otherwise unreachable, returns `false` rather than propagating
/// the error so detection stays best-effort.
#[allow(dead_code)] // wiring into service commands lands in task 11
fn check_docker_container() -> bool {
    let result = Command::new("docker")
        .args(["ps", "--filter", "name=^memexd$", "--quiet"])
        .stderr(Stdio::null())
        .output();

    match result {
        Ok(output) if output.status.success() => {
            // Non-empty stdout = at least one matching container id.
            !output.stdout.iter().all(u8::is_ascii_whitespace)
        }
        // ENOENT (docker not installed) or non-zero exit: treat as absent.
        _ => false,
    }
}

/// Probe the default local gRPC endpoint for a responsive daemon.
///
/// NOTE: Rather than pulling in `tonic-health` and the `grpc.health.v1`
/// service (which the daemon does not currently implement), this performs a
/// best-effort `crate::grpc::connect_default()` under a 1s deadline. A
/// successful connect means the TCP port is accepting gRPC traffic, which is
/// sufficient signal for "something is listening on 50051".
#[allow(dead_code)] // wiring into service commands lands in task 11
async fn check_grpc_health() -> bool {
    match tokio::time::timeout(GRPC_PROBE_TIMEOUT, crate::grpc::connect_default()).await {
        Ok(Ok(_client)) => true,
        // Connect failed OR outer timeout elapsed.
        _ => false,
    }
}

/// Orchestrate all three probes and classify the result.
///
/// Preserves the layering required for unit testing: the classification
/// logic is a separate pure function ([`classify`]), so the public entry
/// point is effectively just wiring.
#[allow(dead_code)] // wiring into service commands lands in task 11
pub async fn detect_daemon_source() -> DaemonSource {
    let local_pid = check_local_pid();
    let docker = check_docker_container();
    let grpc_ok = check_grpc_health().await;
    classify(local_pid, docker, grpc_ok, DEFAULT_GRPC_ADDR)
}

/// Pure classification of the three probe outcomes.
///
/// Kept separate from [`detect_daemon_source`] so every branch of the
/// decision matrix is exercised without needing real processes, containers,
/// or sockets.
fn classify(local_pid: Option<u32>, docker: bool, grpc_ok: bool, grpc_addr: &str) -> DaemonSource {
    match (local_pid, docker, grpc_ok) {
        (Some(pid), false, _) => DaemonSource::LocalOnly { pid },
        (None, true, _) => DaemonSource::DockerOnly,
        (Some(pid), true, _) => DaemonSource::Both { pid },
        (None, false, true) => DaemonSource::RemoteOnly {
            addr: grpc_addr.to_string(),
        },
        (None, false, false) => DaemonSource::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ---- check_local_pid_at ------------------------------------------------

    #[test]
    fn local_pid_live_process_returns_pid() {
        // The current process is guaranteed to be alive.
        let pid = std::process::id();
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "{pid}").unwrap();
        assert_eq!(check_local_pid_at(f.path()), Some(pid));
    }

    #[test]
    fn local_pid_dead_process_returns_none() {
        // u32::MAX is extremely unlikely to map to a live process.
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "{}", u32::MAX).unwrap();
        assert_eq!(check_local_pid_at(f.path()), None);
    }

    #[test]
    fn local_pid_missing_file_returns_none() {
        let f = NamedTempFile::new().unwrap();
        let path = f.path().to_path_buf();
        drop(f); // delete the file
        assert_eq!(check_local_pid_at(&path), None);
    }

    #[test]
    fn local_pid_corrupted_contents_returns_none() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "not-a-number").unwrap();
        assert_eq!(check_local_pid_at(f.path()), None);
    }

    #[test]
    fn local_pid_empty_file_returns_none() {
        let f = NamedTempFile::new().unwrap();
        assert_eq!(check_local_pid_at(f.path()), None);
    }

    // ---- is_pid_alive ------------------------------------------------------

    #[test]
    #[cfg(unix)]
    fn pid_alive_for_self() {
        assert!(is_pid_alive(std::process::id()));
    }

    #[test]
    #[cfg(unix)]
    fn pid_alive_false_for_absurd_pid() {
        assert!(!is_pid_alive(u32::MAX));
    }

    // ---- classify matrix ---------------------------------------------------

    const ADDR: &str = "127.0.0.1:50051";

    #[test]
    fn classify_local_only() {
        assert_eq!(
            classify(Some(123), false, false, ADDR),
            DaemonSource::LocalOnly { pid: 123 }
        );
        // gRPC responding is fine — a local daemon is expected to listen.
        assert_eq!(
            classify(Some(123), false, true, ADDR),
            DaemonSource::LocalOnly { pid: 123 }
        );
    }

    #[test]
    fn classify_docker_only() {
        assert_eq!(classify(None, true, false, ADDR), DaemonSource::DockerOnly);
        assert_eq!(classify(None, true, true, ADDR), DaemonSource::DockerOnly);
    }

    #[test]
    fn classify_both() {
        assert_eq!(
            classify(Some(7), true, false, ADDR),
            DaemonSource::Both { pid: 7 }
        );
        assert_eq!(
            classify(Some(7), true, true, ADDR),
            DaemonSource::Both { pid: 7 }
        );
    }

    #[test]
    fn classify_remote_only() {
        assert_eq!(
            classify(None, false, true, ADDR),
            DaemonSource::RemoteOnly {
                addr: ADDR.to_string(),
            }
        );
    }

    #[test]
    fn classify_none() {
        assert_eq!(classify(None, false, false, ADDR), DaemonSource::None);
    }

    // ---- Display -----------------------------------------------------------

    #[test]
    fn display_variants() {
        assert_eq!(
            DaemonSource::LocalOnly { pid: 42 }.to_string(),
            "local (pid 42)"
        );
        assert_eq!(DaemonSource::DockerOnly.to_string(), "docker container");
        assert_eq!(
            DaemonSource::Both { pid: 99 }.to_string(),
            "local (pid 99) and docker container"
        );
        assert_eq!(
            DaemonSource::RemoteOnly {
                addr: "1.2.3.4:50051".to_string()
            }
            .to_string(),
            "remote gRPC @ 1.2.3.4:50051"
        );
        assert_eq!(DaemonSource::None.to_string(), "none");
    }
}
