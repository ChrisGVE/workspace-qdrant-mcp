//! Cross-process singleton advisory lock for single-writer enforcement (F14).
//!
//! File: `wqm-storage-write/src/single_writer.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: GP-9 (arch §2 B1, §6.3) requires that `memexd` is the SOLE process
//!   allowed to write to the per-project store.db files and the Qdrant `projects`
//!   collection. This module implements that invariant at the OS process boundary
//!   via an exclusive advisory file lock (`flock(2)`) on `<data_dir>/daemon.lock`.
//!
//!   Key design choices (locked by AC-F14.1):
//!   * POSIX `flock(2)` with `LOCK_EX | LOCK_NB` — exclusive, non-blocking. The OS
//!     releases the lock automatically when the file descriptor is closed, so a
//!     crashed/killed daemon's lock is freed without manual stale detection.
//!   * Fail-closed: a second writer that finds the OS lock held is REFUSED with a
//!     clear `StorageError::LockConflict` — never force-reclaimed.
//!   * Heartbeat: on top of the OS lock the daemon writes its PID + a monotonically-
//!     updated Unix timestamp into the lock file. The heartbeat is DIAGNOSTIC ONLY —
//!     the staleness check returns a human-readable string and NEVER auto-steals the
//!     lock. Manual operator intervention is required to remove a genuinely stale lock.
//!   * Staleness window default: 30 seconds (configurable via `DaemonLockConfig`).
//!
//!   DISTINCTION from `blob::lock` (`ContentKeyLockManager`):
//!   `ContentKeyLockManager` is an IN-PROCESS async `Mutex` per `content_key` that
//!   serializes concurrent Tokio tasks within a single daemon instance. `DaemonLock`
//!   is a CROSS-PROCESS OS file lock that prevents two separate daemon processes from
//!   sharing the same data directory. They are complementary, not interchangeable.
//!
//! Neighbors: `crate::connection` (write connection opener), `wqm_common::StorageError`
//!   (canonical error type, DR GP-9).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use wqm_common::error::StorageError;

/// Configuration for `DaemonLock`.
///
/// The staleness window is purely diagnostic — it controls when the heartbeat
/// is considered old enough to display a warning. It NEVER triggers auto-reclaim.
#[derive(Debug, Clone, Copy)]
pub struct DaemonLockConfig {
    /// Duration after which a heartbeat timestamp is considered stale for the
    /// purpose of the diagnostic message (AC-F14.1 advisory). Default: 30 s.
    pub heartbeat_staleness_window: Duration,
}

impl Default for DaemonLockConfig {
    fn default() -> Self {
        Self {
            heartbeat_staleness_window: Duration::from_secs(30),
        }
    }
}

/// Cross-process singleton OS advisory lock for the single-writer invariant (GP-9).
///
/// Acquires an exclusive `flock(2)` on `<data_dir>/daemon.lock` at construction.
/// Holds the open `File` (which owns the fd) for its lifetime; `Drop` closes the
/// fd and the OS releases the lock automatically — no manual cleanup required.
///
/// Use [`DaemonLock::acquire`] to obtain the lock. If the lock is already held by
/// another process, it returns `StorageError::LockConflict` immediately (non-blocking).
pub struct DaemonLock {
    /// The open file whose fd holds the OS advisory lock. Kept alive for the
    /// duration of the lock.
    _file: File,
    /// Path to the lock file (for diagnostics).
    lock_path: PathBuf,
    /// Configuration (staleness window for heartbeat assessment).
    config: DaemonLockConfig,
}

impl std::fmt::Debug for DaemonLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonLock")
            .field("lock_path", &self.lock_path)
            .field("config", &self.config)
            .finish()
    }
}

impl DaemonLock {
    /// Acquire the exclusive singleton advisory lock for `data_dir`.
    ///
    /// Opens (or creates) `<data_dir>/daemon.lock`, calls `flock(LOCK_EX|LOCK_NB)`,
    /// and writes an initial heartbeat (pid + current Unix timestamp) into the file.
    ///
    /// # Errors
    ///
    /// * `StorageError::LockConflict` — another process already holds the lock.
    /// * `StorageError::Sqlite` — the lock file could not be opened or the
    ///   `flock` syscall returned an unexpected error.
    pub fn acquire(data_dir: impl AsRef<Path>) -> Result<Self, StorageError> {
        Self::acquire_with_config(data_dir, DaemonLockConfig::default())
    }

    /// Acquire with an explicit `DaemonLockConfig`.
    pub fn acquire_with_config(
        data_dir: impl AsRef<Path>,
        config: DaemonLockConfig,
    ) -> Result<Self, StorageError> {
        let lock_path = data_dir.as_ref().join("daemon.lock");

        let mut file = OpenOptions::new()
            .create(true)
            // Do not truncate on open: we write the heartbeat explicitly after
            // acquiring the flock, using set_len(0) + write_all.
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|e| {
                StorageError::Sqlite(format!(
                    "daemon lock: cannot open {}: {e}",
                    lock_path.display()
                ))
            })?;

        // LOCK_EX | LOCK_NB: exclusive, non-blocking.
        // EWOULDBLOCK / EAGAIN -> another process holds the lock -> fail-closed.
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            let raw = err.raw_os_error().unwrap_or(0);
            if raw == libc::EWOULDBLOCK || raw == libc::EAGAIN {
                return Err(StorageError::LockConflict(format!(
                    "daemon.lock at {} is held by another process; \
                     start only one memexd per data directory",
                    lock_path.display()
                )));
            }
            return Err(StorageError::Sqlite(format!(
                "daemon lock: flock failed on {}: {err}",
                lock_path.display()
            )));
        }

        // Write initial heartbeat.
        let beat = Self::heartbeat_bytes();
        file.set_len(0).map_err(|e| {
            StorageError::Sqlite(format!(
                "daemon lock: truncate failed on {}: {e}",
                lock_path.display()
            ))
        })?;
        file.write_all(&beat).map_err(|e| {
            StorageError::Sqlite(format!(
                "daemon lock: write heartbeat failed on {}: {e}",
                lock_path.display()
            ))
        })?;

        Ok(Self {
            _file: file,
            lock_path,
            config,
        })
    }

    /// Refresh the heartbeat (pid + current timestamp) in the lock file.
    ///
    /// Should be called periodically by the daemon (e.g. every 30 s) so that an
    /// operator can detect a genuinely stale lock after a crash. The OS lock
    /// itself persists regardless of heartbeat updates.
    pub fn refresh_heartbeat(&mut self) -> Result<(), StorageError> {
        let beat = Self::heartbeat_bytes();
        // Seek to beginning and overwrite.
        use std::io::Seek;
        self._file.seek(std::io::SeekFrom::Start(0)).map_err(|e| {
            StorageError::Sqlite(format!(
                "daemon lock: seek failed on {}: {e}",
                self.lock_path.display()
            ))
        })?;
        self._file.set_len(0).map_err(|e| {
            StorageError::Sqlite(format!(
                "daemon lock: truncate failed on {}: {e}",
                self.lock_path.display()
            ))
        })?;
        self._file.write_all(&beat).map_err(|e| {
            StorageError::Sqlite(format!(
                "daemon lock: write heartbeat failed on {}: {e}",
                self.lock_path.display()
            ))
        })
    }

    /// Assess the heartbeat in the lock file at `lock_path` and return a diagnostic
    /// string if the timestamp is older than the staleness window.
    ///
    /// Returns `None` if the heartbeat is current (within the staleness window) or
    /// if the file cannot be read. NEVER force-reclaims the lock — diagnostic only.
    ///
    /// This is a static method so it can be called by a prospective second writer
    /// BEFORE attempting to acquire (to surface a better error message), or by an
    /// operator tool.
    pub fn assess_staleness(
        data_dir: impl AsRef<Path>,
        config: DaemonLockConfig,
    ) -> Option<String> {
        let lock_path = data_dir.as_ref().join("daemon.lock");
        let mut file = File::open(&lock_path).ok()?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).ok()?;
        let (pid, ts_secs) = parse_heartbeat(&contents)?;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let age = now_secs.saturating_sub(ts_secs);
        if age >= config.heartbeat_staleness_window.as_secs() {
            Some(format!(
                "daemon.lock at {} looks stale: pid {pid} last heartbeat {age} s ago \
                 (window {} s). If process {pid} is dead, remove the lock file manually \
                 and restart memexd.",
                lock_path.display(),
                config.heartbeat_staleness_window.as_secs()
            ))
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Serialize the current PID + Unix timestamp into ASCII bytes.
    /// Format: `pid=<N> ts=<N>\n`
    fn heartbeat_bytes() -> Vec<u8> {
        let pid = std::process::id();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("pid={pid} ts={ts}\n").into_bytes()
    }
}

/// Parse `pid=<N> ts=<N>` from a heartbeat string. Returns `(pid, ts_secs)`.
fn parse_heartbeat(s: &str) -> Option<(u32, u64)> {
    // Tolerant line-by-line scan — the file may have multiple heartbeat lines
    // if a previous write was interrupted; we take the last complete one.
    let mut result = None;
    for line in s.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let pid = parts
            .iter()
            .find(|p| p.starts_with("pid="))?
            .trim_start_matches("pid=")
            .parse::<u32>()
            .ok()?;
        let ts = parts
            .iter()
            .find(|p| p.starts_with("ts="))?
            .trim_start_matches("ts=")
            .parse::<u64>()
            .ok()?;
        result = Some((pid, ts));
    }
    result
}

// ---------------------------------------------------------------------------
// Tests (AC-F14.1, AC-F14.3)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // AC-F14.1 — second writer is refused (mutual-exclusion proof)
    // -----------------------------------------------------------------------

    /// Acquiring the lock on a temp dir succeeds; a second acquire on the same
    /// dir fails with LockConflict. Both lock-holders are in the same process,
    /// but because `flock` is per-fd (not per-PID) two separate File descriptors
    /// on the same path from the same process produce two separate lock requests,
    /// and the second one is correctly refused when the first is still held.
    /// This proves the mechanism the rebuild/migrate verbs use (AC-F14.3).
    #[test]
    fn t_f14_second_acquire_refused() {
        let dir = TempDir::new().expect("tempdir");
        let _lock1 = DaemonLock::acquire(dir.path()).expect("first acquire");

        let result = DaemonLock::acquire(dir.path());
        assert!(
            matches!(result, Err(StorageError::LockConflict(_))),
            "second acquire must fail with LockConflict, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // AC-F14.1 — OS releases lock on fd close (crash-release proof)
    //
    // The strongest feasible proof in a unit-test context: acquire, drop (closes
    // the fd), then re-acquire in the same process. The OS releases the lock on
    // fd close, so the re-acquire succeeds. This proves the "crashed/killed daemon"
    // recovery path: the OS releases the lock when the process exits, allowing a
    // fresh daemon restart to acquire it.
    //
    // Note: a cross-process proof (spawn child, child acquires, child exits, parent
    // acquires) is even stronger but requires spawning `std::process::Command`,
    // which is heavy for a unit test and would depend on the binary being built.
    // The fd-close proof is the canonical mechanism; the OS close-on-exit path is
    // identical (the kernel closes all fds on process exit).
    // -----------------------------------------------------------------------

    /// Drop closes the fd; re-acquire succeeds.
    #[test]
    fn t_f14_lock_released_on_drop() {
        let dir = TempDir::new().expect("tempdir");

        {
            let _lock = DaemonLock::acquire(dir.path()).expect("first acquire");
            // `_lock` dropped here -> fd closed -> OS releases flock.
        }

        // Re-acquire must succeed now.
        let result = DaemonLock::acquire(dir.path());
        assert!(
            result.is_ok(),
            "re-acquire after drop must succeed, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // AC-F14.1 — heartbeat staleness diagnostic (never force-reclaims)
    // -----------------------------------------------------------------------

    /// Write a heartbeat with an old timestamp; staleness assessment returns
    /// the diagnostic string and does NOT modify or remove the lock file.
    #[test]
    fn t_f14_heartbeat_staleness_diagnostic() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("daemon.lock");

        // Write a heartbeat with a timestamp 120 s in the past.
        let old_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(120);
        std::fs::write(&lock_path, format!("pid=99999 ts={old_ts}\n"))
            .expect("write stale heartbeat");

        let config = DaemonLockConfig {
            heartbeat_staleness_window: Duration::from_secs(30),
        };

        let diag = DaemonLock::assess_staleness(dir.path(), config);
        assert!(
            diag.is_some(),
            "120s-old heartbeat must trigger staleness diagnostic"
        );
        let msg = diag.unwrap();
        assert!(
            msg.contains("stale"),
            "diagnostic must mention 'stale': {msg}"
        );
        assert!(
            msg.contains("manually"),
            "diagnostic must say 'manually': {msg}"
        );
        // The lock file must still exist — no auto-removal.
        assert!(
            lock_path.exists(),
            "staleness check must not remove lock file"
        );
    }

    /// A recent heartbeat does NOT trigger the diagnostic.
    #[test]
    fn t_f14_heartbeat_current_no_diagnostic() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("daemon.lock");

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        std::fs::write(&lock_path, format!("pid=12345 ts={now_ts}\n"))
            .expect("write fresh heartbeat");

        let config = DaemonLockConfig {
            heartbeat_staleness_window: Duration::from_secs(30),
        };

        let diag = DaemonLock::assess_staleness(dir.path(), config);
        assert!(
            diag.is_none(),
            "fresh heartbeat must not trigger staleness diagnostic"
        );
    }

    // -----------------------------------------------------------------------
    // Heartbeat round-trip: refresh updates the timestamp.
    // -----------------------------------------------------------------------

    #[test]
    fn t_f14_refresh_heartbeat_updates_content() {
        let dir = TempDir::new().expect("tempdir");
        let mut lock = DaemonLock::acquire(dir.path()).expect("acquire");
        lock.refresh_heartbeat().expect("refresh");

        let lock_path = dir.path().join("daemon.lock");
        let content = std::fs::read_to_string(&lock_path).expect("read lock file");
        assert!(
            content.starts_with("pid="),
            "lock file must contain pid= after refresh"
        );
        assert!(
            content.contains("ts="),
            "lock file must contain ts= after refresh"
        );
    }

    // -----------------------------------------------------------------------
    // AC-F14.3 — two DaemonLock acquisitions on the same data-dir are mutually
    // exclusive: the second is refused. This is the mechanism that the
    // rebuild/migrate verbs use (they acquire DaemonLock or run in the daemon).
    // -----------------------------------------------------------------------

    /// Same-dir mutual exclusion (proof of AC-F14.3 gate mechanism).
    #[test]
    fn t_f14_same_dir_mutually_exclusive() {
        let dir = TempDir::new().expect("tempdir");

        let _guard = DaemonLock::acquire(dir.path()).expect("first acquire");

        let second = DaemonLock::acquire(dir.path());
        assert!(
            matches!(second, Err(StorageError::LockConflict(_))),
            "two DaemonLock acquires on same data-dir must be mutually exclusive"
        );
    }

    /// Different dirs get independent locks — each succeeds.
    #[test]
    fn t_f14_different_dirs_independent() {
        let dir1 = TempDir::new().expect("tempdir1");
        let dir2 = TempDir::new().expect("tempdir2");

        let _lock1 = DaemonLock::acquire(dir1.path()).expect("dir1 acquire");
        let _lock2 = DaemonLock::acquire(dir2.path()).expect("dir2 acquire");
        // Both held simultaneously — no conflict.
    }
}
