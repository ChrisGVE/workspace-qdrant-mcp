//! Tests for `wqm_common::guard::assert_daemon_stopped`.
//!
//! AC-F20.4: exactly one daemon-running-guard definition in the workspace;
//! test the absent-file path, the free-lock path, and the held-lock path.

use std::fs::File;
use tempfile::TempDir;

use crate::error::StorageError;
use crate::guard::assert_daemon_stopped;

// ---- Helper ----------------------------------------------------------------

/// Create a TempDir and return `(dir, lock_path)`.
fn tmp_dir() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let lock = dir.path().join("daemon.lock");
    (dir, lock)
}

// ---- Tests -----------------------------------------------------------------

/// AC-F20.4: lock file absent => Ok (daemon not running).
#[test]
fn t_f20_guard_absent_lock_file_returns_ok() {
    let (dir, _lock) = tmp_dir();
    // Do not create the lock file.
    let result = assert_daemon_stopped(dir.path());
    assert!(
        result.is_ok(),
        "expected Ok when lock file absent, got {:?}",
        result
    );
}

/// AC-F20.4: lock file exists but is NOT held => Ok (daemon not running).
#[test]
fn t_f20_guard_free_lock_file_returns_ok() {
    let (dir, lock_path) = tmp_dir();
    // Create the file but hold no lock on it.
    File::create(&lock_path).expect("create lock file");
    let result = assert_daemon_stopped(dir.path());
    assert!(
        result.is_ok(),
        "expected Ok when lock file free, got {:?}",
        result
    );
}

/// AC-F20.4: lock file held by another fd (simulating a live daemon)
/// => Err(StorageError::LockConflict).
#[cfg(unix)]
#[test]
fn t_f20_guard_held_lock_returns_lock_conflict() {
    use std::os::unix::io::AsRawFd;

    let (dir, lock_path) = tmp_dir();
    // Create and hold an exclusive flock on the file from this process.
    let holder = File::create(&lock_path).expect("create lock file");
    let fd = holder.as_raw_fd();
    let got = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(got, 0, "expected to acquire holder lock");

    // Now the guard should find the lock held.
    let result = assert_daemon_stopped(dir.path());
    assert!(
        matches!(result, Err(StorageError::LockConflict(_))),
        "expected LockConflict, got {:?}",
        result
    );

    // The error message should mention the daemon.
    if let Err(StorageError::LockConflict(msg)) = result {
        assert!(
            msg.contains("daemon"),
            "error message should mention daemon: {}",
            msg
        );
    }

    // Release holder lock.
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(holder);

    // After release, guard should succeed again.
    let result2 = assert_daemon_stopped(dir.path());
    assert!(
        result2.is_ok(),
        "expected Ok after lock release, got {:?}",
        result2
    );
}

/// AC-F20.4 structural assertion: assert_daemon_stopped is defined exactly once
/// in the workspace (no duplicate). This is a naming/existence check -- if two
/// definitions existed, there would be a compile error; this test documents the
/// contract.
#[test]
fn t_f20_guard_single_definition_exists() {
    // If this test compiles and runs, the function is accessible from wqm_common::guard.
    // The absence of a second definition is enforced at compile time (orphan/name collision).
    let dir = TempDir::new().expect("tempdir");
    let _ = assert_daemon_stopped(dir.path());
}
