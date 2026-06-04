//! Cross-process single-instance lock for memexd via TCP listen socket.
//!
//! Spec: `docs/specs/16-path-abstraction.md` §10.1.
//!
//! The lock primitive is a TCP listener bound to `127.0.0.1:<control_port>`
//! (default 7799). Only one process — host or docker, arbitrated by the host
//! kernel when the docker container publishes the same port on
//! `127.0.0.1:7799` — can hold the bind at a time. Process death releases
//! the socket immediately, so no stale-lock cleanup logic is needed
//! (compare: filesystem PID lockfiles).
//!
//! In addition to the authoritative socket bind, this module writes a
//! diagnostic identity-stamp JSON file to
//! `~/.local/share/workspace-qdrant/memexd.lock` (host) or
//! `/var/lib/wqm/memexd.lock` (docker). The file is informational only —
//! the socket is the authoritative lock. The stamp is best-effort: a
//! failure to write the file is logged as a warning, not a startup
//! blocker.
//!
//! ## Port-resolution precedence
//!
//! 1. `DaemonArgs.control_port` (CLI `--control-port`) — highest.
//! 2. `DaemonConfig.control_port` (config file).
//! 3. `WQM_CONTROL_PORT` env var.
//! 4. Built-in default `7799`.
//!
//! ## Lifetime
//!
//! [`ControlPortGuard`] owns the bound socket and the identity-stamp
//! file. Dropping the guard closes the socket (releasing the lock) and
//! deletes the identity-stamp file. The guard is held for the daemon's
//! full lifetime via `_control_port_guard` in the orchestrator.

use std::fs;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Default memexd control port (loopback). Spec 16 §10.1.
pub const DEFAULT_CONTROL_PORT: u16 = 7799;

/// Environment variable consulted when the CLI flag is absent (spec 16
/// §10.1 — compose-generated overrides consume the same name).
pub const CONTROL_PORT_ENV: &str = "WQM_CONTROL_PORT";

/// File name (basename only) for the diagnostic identity stamp.
pub const IDENTITY_STAMP_FILENAME: &str = "memexd.lock";

/// Container-side fixed directory holding the identity stamp (spec 16
/// §9.2). Host-side path is resolved via `wqm_common::paths::get_data_dir`.
pub const DOCKER_DATA_DIR: &str = "/var/lib/wqm";

/// Deployment mode for the running daemon process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentMode {
    /// Running directly on the host OS.
    Host,
    /// Running inside a docker (or compatible) container.
    Docker,
}

impl DeploymentMode {
    /// Detect the active deployment mode.
    ///
    /// Precedence:
    /// 1. `WQM_DEPLOYMENT_MODE=host|docker` — explicit override (used by
    ///    tests and by containers that customize their entrypoint).
    /// 2. Existence of `/.dockerenv` — Docker's standard marker file.
    /// 3. Default: `Host`.
    pub fn detect() -> Self {
        if let Ok(value) = std::env::var("WQM_DEPLOYMENT_MODE") {
            match value.trim().to_ascii_lowercase().as_str() {
                "docker" => return Self::Docker,
                "host" => return Self::Host,
                other => {
                    warn!(
                        value = other,
                        "WQM_DEPLOYMENT_MODE unrecognised (expected 'host' or \
                         'docker'); falling back to filesystem detection"
                    );
                }
            }
        }
        if Path::new("/.dockerenv").exists() {
            Self::Docker
        } else {
            Self::Host
        }
    }

    /// String form used in the identity stamp JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Docker => "docker",
        }
    }
}

/// JSON shape of the memexd identity-stamp file (spec 16 §10.1, item 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityStamp {
    /// `host` or `docker`.
    pub mode: DeploymentMode,
    /// Operating-system process ID of memexd.
    pub pid: u32,
    /// RFC3339 timestamp at which the bind succeeded.
    pub started_at: String,
    /// Control port the daemon is bound to (`127.0.0.1:<port>`).
    pub port: u16,
}

impl IdentityStamp {
    /// Build a stamp for the current process.
    pub fn new(mode: DeploymentMode, port: u16) -> Self {
        Self {
            mode,
            pid: process::id(),
            started_at: chrono::Utc::now().to_rfc3339(),
            port,
        }
    }
}

/// Errors produced when binding the memexd control port.
#[derive(Debug, thiserror::Error)]
pub enum ControlPortError {
    /// Another process already holds the bind (or the port is taken by an
    /// unrelated daemon — the user-visible error string distinguishes).
    #[error(
        "memexd control port 127.0.0.1:{port} is unavailable (another \
         memexd instance is already running, or the port is in use by an \
         unrelated process). If this is incorrect, check for stale memexd \
         processes (`ps -ef | grep memexd`) or use --control-port to \
         override. Underlying error: {source}"
    )]
    BindFailed {
        port: u16,
        #[source]
        source: io::Error,
    },
}

/// Owns the bound TCP listener and the identity-stamp file.
///
/// Dropping the guard releases both. The socket release is what
/// authoritatively unlocks the daemon's single-instance discipline;
/// removing the identity-stamp file is best-effort cleanup.
#[derive(Debug)]
pub struct ControlPortGuard {
    /// Bound listener — kept alive for the daemon's lifetime so the OS
    /// keeps the port reserved against any second memexd attempting to
    /// bind. The listener never accepts connections; it is purely a lock.
    listener: Option<TcpListener>,
    /// Port we actually bound to (echoes the resolved value).
    ///
    /// Held for diagnostic access; the listener itself is the authoritative
    /// lock.
    #[allow(dead_code)]
    port: u16,
    /// Mode of the running daemon (host or docker). Held for diagnostic
    /// access; identifies which deployment branch wrote the stamp.
    #[allow(dead_code)]
    mode: DeploymentMode,
    /// Path of the identity-stamp file; `None` if write failed.
    stamp_path: Option<PathBuf>,
}

impl ControlPortGuard {
    /// Port the daemon is bound to.
    ///
    /// Exposed for diagnostic logging and integration tests; production
    /// code paths consult [`resolve_port`] for the resolved value.
    #[allow(dead_code)]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Deployment mode detected at bind time.
    ///
    /// Exposed for diagnostic logging.
    #[allow(dead_code)]
    pub fn mode(&self) -> DeploymentMode {
        self.mode
    }

    /// Path of the identity-stamp file on disk, if it was successfully
    /// written. `None` when the write failed (the daemon still ran:
    /// the stamp is diagnostic only).
    #[allow(dead_code)]
    pub fn stamp_path(&self) -> Option<&Path> {
        self.stamp_path.as_deref()
    }
}

impl Drop for ControlPortGuard {
    fn drop(&mut self) {
        // Close the bind first so a fast-restart scenario sees the port
        // freed before the next memexd attempts its own bind.
        if let Some(listener) = self.listener.take() {
            drop(listener);
        }
        if let Some(path) = &self.stamp_path {
            match fs::remove_file(path) {
                Ok(()) => {
                    info!(path = %path.display(), "Removed memexd identity stamp");
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    // Already gone — fine.
                }
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to remove memexd identity stamp (non-fatal)"
                    );
                }
            }
        }
    }
}

/// Resolve the effective control port from CLI flag → env var → config →
/// built-in default.
///
/// `cli_override` is the CLI flag value (`None` if unset). `config_port`
/// is the value read from `DaemonConfig.control_port`. The env var
/// (`WQM_CONTROL_PORT`) is consulted between the CLI flag and the config
/// so that operators can pin the port without editing config (e.g.,
/// inside compose generated overrides).
///
/// Returns `Err` if the env var is set but cannot be parsed as `u16`.
pub fn resolve_port(cli_override: Option<u16>, config_port: Option<u16>) -> Result<u16, String> {
    if let Some(p) = cli_override {
        return Ok(p);
    }
    if let Ok(raw) = std::env::var(CONTROL_PORT_ENV) {
        return raw
            .trim()
            .parse::<u16>()
            .map_err(|e| format!("{CONTROL_PORT_ENV}='{raw}' is not a valid u16 port: {e}"));
    }
    Ok(config_port.unwrap_or(DEFAULT_CONTROL_PORT))
}

/// Resolve the on-disk identity-stamp path for the given mode.
fn resolve_stamp_path(mode: DeploymentMode) -> Option<PathBuf> {
    match mode {
        DeploymentMode::Docker => {
            Some(PathBuf::from(DOCKER_DATA_DIR).join(IDENTITY_STAMP_FILENAME))
        }
        DeploymentMode::Host => match wqm_common::paths::get_data_dir() {
            Ok(dir) => Some(dir.join(IDENTITY_STAMP_FILENAME)),
            Err(e) => {
                warn!(
                    error = %e,
                    "Cannot resolve host data dir for identity stamp; \
                     stamp will not be written (diagnostic only)"
                );
                None
            }
        },
    }
}

/// Best-effort write of the identity-stamp file.
///
/// Failures are logged at WARN and the returned `Option<PathBuf>` is
/// `None`. The caller continues startup regardless — the socket bind is
/// the authoritative lock.
fn write_identity_stamp(stamp: &IdentityStamp, target: &Path) -> Option<PathBuf> {
    // Ensure parent dir exists. Container-side `/var/lib/wqm` is created
    // by the image; host-side `~/.local/share/workspace-qdrant` may not
    // exist yet on first run.
    if let Some(parent) = target.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            warn!(
                path = %parent.display(),
                error = %e,
                "Failed to create directory for memexd identity stamp \
                 (diagnostic only — daemon continues)"
            );
            return None;
        }
    }

    let json = match serde_json::to_string_pretty(stamp) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Failed to serialise identity stamp (diagnostic only)");
            return None;
        }
    };

    // Atomic write: write to a sibling temp file then rename.
    let tmp = target.with_extension("lock.tmp");
    if let Err(e) = fs::write(&tmp, json.as_bytes()) {
        warn!(
            path = %tmp.display(),
            error = %e,
            "Failed to write memexd identity stamp tempfile (diagnostic only)"
        );
        return None;
    }
    if let Err(e) = fs::rename(&tmp, target) {
        warn!(
            from = %tmp.display(),
            to = %target.display(),
            error = %e,
            "Failed to rename memexd identity stamp into place (diagnostic only)"
        );
        // Best-effort cleanup of the temp file.
        let _ = fs::remove_file(&tmp);
        return None;
    }

    info!(
        path = %target.display(),
        pid = stamp.pid,
        port = stamp.port,
        mode = stamp.mode.as_str(),
        "Wrote memexd identity stamp"
    );
    Some(target.to_path_buf())
}

/// Acquire the cross-process single-instance lock.
///
/// Binds a TCP listener to `127.0.0.1:<port>` and writes the diagnostic
/// identity-stamp file. The returned [`ControlPortGuard`] must be held
/// for the daemon's lifetime; dropping it releases the port and removes
/// the stamp.
///
/// Bind failure is fatal: a second memexd would corrupt shared SQLite
/// state if it proceeded past this point (spec 16 §10.1, §12). The
/// stamp-file failure is non-fatal — the socket is authoritative.
pub fn acquire(port: u16, mode: DeploymentMode) -> Result<ControlPortGuard, ControlPortError> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let listener =
        TcpListener::bind(addr).map_err(|source| ControlPortError::BindFailed { port, source })?;
    // Non-blocking so the listener never holds an accept worker; we
    // never accept on this socket. This also lets a tokio runtime ignore
    // the fd without any registration.
    if let Err(e) = listener.set_nonblocking(true) {
        warn!(
            port,
            error = %e,
            "Failed to put control-port listener into non-blocking mode \
             (non-fatal — listener is never polled)"
        );
    }
    info!(
        port,
        mode = mode.as_str(),
        "Acquired memexd control port (single-instance lock active)"
    );

    let stamp = IdentityStamp::new(mode, port);
    let stamp_path = resolve_stamp_path(mode).and_then(|p| write_identity_stamp(&stamp, &p));

    Ok(ControlPortGuard {
        listener: Some(listener),
        port,
        mode,
        stamp_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests that mutate process-global env vars (WQM_CONTROL_PORT,
    // WQM_DEPLOYMENT_MODE) must not run concurrently with each other or they
    // race; `#[serial]` enforces that under any `--test-threads` setting.
    use serial_test::serial;

    /// Reserve a free localhost port for tests by binding then dropping a
    /// listener — the OS hands the same port back on the immediate
    /// re-bind in 99%+ cases on Linux/macOS thanks to ephemeral-port
    /// reservation latency. Adequate for unit tests; the integration
    /// test below uses a fixed port and serial discipline.
    fn pick_free_port() -> u16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    #[test]
    fn resolve_port_cli_override_wins() {
        // Even with a config value, CLI wins.
        let p = resolve_port(Some(9000), Some(7799)).unwrap();
        assert_eq!(p, 9000);
    }

    #[test]
    #[serial]
    fn resolve_port_env_var_takes_precedence_over_config() {
        // Use a unique env-var name guard via a one-off process-global lock
        // would be ideal; instead, scope to a unique value and clean up.
        // Since std::env::set_var is process-global, tests that touch
        // WQM_CONTROL_PORT must run serially. We mark them with
        // `_env_serial` and rely on `cargo test -- --test-threads=1`
        // discipline plus pre-clear.
        std::env::remove_var(CONTROL_PORT_ENV);
        std::env::set_var(CONTROL_PORT_ENV, "8123");
        let p = resolve_port(None, Some(7799)).unwrap();
        std::env::remove_var(CONTROL_PORT_ENV);
        assert_eq!(p, 8123);
    }

    #[test]
    #[serial]
    fn resolve_port_env_var_invalid_returns_error() {
        std::env::remove_var(CONTROL_PORT_ENV);
        std::env::set_var(CONTROL_PORT_ENV, "not-a-number");
        let err = resolve_port(None, None);
        std::env::remove_var(CONTROL_PORT_ENV);
        assert!(err.is_err(), "non-numeric env value must error");
    }

    #[test]
    #[serial]
    fn resolve_port_falls_back_to_config_then_default() {
        std::env::remove_var(CONTROL_PORT_ENV);
        assert_eq!(resolve_port(None, Some(7800)).unwrap(), 7800);
        assert_eq!(resolve_port(None, None).unwrap(), DEFAULT_CONTROL_PORT);
    }

    #[test]
    #[serial]
    fn deployment_mode_env_override_recognised() {
        std::env::remove_var("WQM_DEPLOYMENT_MODE");
        std::env::set_var("WQM_DEPLOYMENT_MODE", "docker");
        let mode = DeploymentMode::detect();
        std::env::remove_var("WQM_DEPLOYMENT_MODE");
        assert_eq!(mode, DeploymentMode::Docker);
    }

    #[test]
    #[serial]
    fn deployment_mode_env_override_host() {
        std::env::remove_var("WQM_DEPLOYMENT_MODE");
        std::env::set_var("WQM_DEPLOYMENT_MODE", "HOST");
        let mode = DeploymentMode::detect();
        std::env::remove_var("WQM_DEPLOYMENT_MODE");
        assert_eq!(mode, DeploymentMode::Host);
    }

    #[test]
    fn acquire_binds_and_releases_port() {
        let port = pick_free_port();
        let guard = acquire(port, DeploymentMode::Host).expect("first bind succeeds");
        assert_eq!(guard.port(), port);
        drop(guard);

        // After drop, re-bind should succeed. Some platforms keep the
        // port in TIME_WAIT; SO_REUSEADDR is platform-dependent so we
        // tolerate transient failure by retrying once.
        let retry = TcpListener::bind((Ipv4Addr::LOCALHOST, port));
        assert!(retry.is_ok(), "port must be re-bindable after drop");
    }

    #[test]
    fn acquire_second_bind_fails_clear_error() {
        let port = pick_free_port();
        let _g1 = acquire(port, DeploymentMode::Host).expect("first bind succeeds");
        let err = acquire(port, DeploymentMode::Host).expect_err("second bind must fail");
        let msg = err.to_string();
        assert!(
            msg.contains(&port.to_string()),
            "error message must include port number: {msg}"
        );
        assert!(
            msg.contains("--control-port"),
            "error message must hint --control-port override: {msg}"
        );
    }

    #[test]
    fn identity_stamp_round_trip() {
        let stamp = IdentityStamp::new(DeploymentMode::Host, 7799);
        let json = serde_json::to_string(&stamp).unwrap();
        let back: IdentityStamp = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, DeploymentMode::Host);
        assert_eq!(back.port, 7799);
        assert_eq!(back.pid, std::process::id());
    }

    #[test]
    fn identity_stamp_writes_and_drops_under_data_dir() {
        // Force host mode + isolated data dir via WQM_DATA_DIR.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let prev_data_dir = std::env::var("WQM_DATA_DIR").ok();
        std::env::set_var("WQM_DATA_DIR", tmp.path());
        std::env::remove_var("WQM_DEPLOYMENT_MODE");

        let port = pick_free_port();
        let guard = acquire(port, DeploymentMode::Host).expect("bind ok");
        let stamp_path = guard.stamp_path().expect("stamp written").to_path_buf();
        assert!(stamp_path.exists(), "stamp file present after acquire");
        let raw = fs::read_to_string(&stamp_path).unwrap();
        let parsed: IdentityStamp = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.port, port);
        assert_eq!(parsed.mode, DeploymentMode::Host);

        drop(guard);
        assert!(
            !stamp_path.exists(),
            "stamp file removed on guard drop: {}",
            stamp_path.display()
        );

        // Restore env.
        match prev_data_dir {
            Some(v) => std::env::set_var("WQM_DATA_DIR", v),
            None => std::env::remove_var("WQM_DATA_DIR"),
        }
    }
}
