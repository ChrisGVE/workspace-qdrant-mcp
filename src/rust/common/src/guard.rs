//! Daemon-running guard shared across CLI commands that require the daemon to be
//! stopped (F20 / AC-F20.4).
//!
//! `assert_daemon_stopped` is the single authoritative daemon-running check in
//! the workspace (FP-2 / DR GP-9). It probes `<data_dir>/daemon.lock` with a
//! non-blocking `flock(LOCK_EX|LOCK_NB)` -- the same mechanism `DaemonLock`
//! (wqm-storage-write) uses to hold the lock while the daemon is live.
//!
//! ## Effectiveness caveat (#175 transition path)
//!
//! The guard becomes FULLY effective only once `memexd` acquires `DaemonLock`
//! at startup (which rides the #175 daemon/write-crate cutover -- the daemon
//! does not yet depend on `wqm-storage-write`). Until #175 lands, a running
//! daemon will not hold `daemon.lock`, so the guard will return `Ok(())` even
//! when the daemon is live. This matches the branch's posture: build the
//! correct structure now; daemon wiring rides #175. Documented in
//! ARCHITECTURE.md section 7.7 and `docs/cli/backup-restore.md`.
//!
//! ## Call sites
//!
//! - `wqm restore --full` (AC-F20.2 / AC-F20.4) -- required, refuses on live daemon.
//! - `wqm recover-state` (AC-F20.4 repoint) -- replaces the local gRPC probe.
//!
//! `wqm backup --full` does NOT call this guard; it is read-only (AC-F20.4).

use std::path::Path;

use crate::error::StorageError;

/// Probe `<data_dir>/daemon.lock` with a non-blocking exclusive flock and
/// return immediately.
///
/// - Lock file **absent**: `Ok(())` (daemon definitely not running).
/// - Lock file **free** (flock succeeds): release immediately and return `Ok(())`.
/// - Lock file **held by another process** (EWOULDBLOCK/EAGAIN): return
///   `Err(StorageError::LockConflict(...))` with a human-readable message.
///
/// # Platform
///
/// Unix-only. On non-Unix targets this function always returns `Ok(())`.
pub fn assert_daemon_stopped(data_dir: impl AsRef<Path>) -> Result<(), StorageError> {
    assert_daemon_stopped_inner(data_dir.as_ref())
}

// ---- Unix implementation ---------------------------------------------------

#[cfg(unix)]
fn assert_daemon_stopped_inner(data_dir: &Path) -> Result<(), StorageError> {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    let lock_path = data_dir.join("daemon.lock");

    // If the lock file does not exist the daemon is definitely not running
    // (DaemonLock::acquire creates it at daemon startup).
    if !lock_path.exists() {
        return Ok(());
    }

    let file = OpenOptions::new()
        .read(true)
        .open(&lock_path)
        .map_err(|e| {
            StorageError::Sqlite(format!(
                "guard: could not open daemon.lock at {}: {}",
                lock_path.display(),
                e
            ))
        })?;

    let fd = file.as_raw_fd();

    // LOCK_EX | LOCK_NB: exclusive, non-blocking.
    //   ret == 0   => acquired (lock was free, daemon not running); release immediately.
    //   ret == -1  => check errno.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

    if ret == 0 {
        // Lock was free -- release it immediately and report daemon stopped.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        return Ok(());
    }

    // Capture errno BEFORE any other syscall could overwrite it.
    let io_err = std::io::Error::last_os_error();
    let raw = io_err.raw_os_error().unwrap_or(0);

    if raw == libc::EWOULDBLOCK || raw == libc::EAGAIN {
        return Err(StorageError::LockConflict(format!(
            "daemon.lock at {} is held by another process (the daemon is running). \
             Stop the daemon first: wqm service stop \
             (or: launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist)",
            lock_path.display()
        )));
    }

    Err(StorageError::Sqlite(format!(
        "guard: flock on daemon.lock at {} failed: {}",
        lock_path.display(),
        io_err
    )))
}

// ---- Non-Unix stub ---------------------------------------------------------

#[cfg(not(unix))]
fn assert_daemon_stopped_inner(_data_dir: &Path) -> Result<(), StorageError> {
    // The daemon does not run on non-Unix targets; the guard is always a no-op.
    Ok(())
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "guard_tests.rs"]
mod tests;
