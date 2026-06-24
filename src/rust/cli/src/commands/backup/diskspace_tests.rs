//! Tests for diskspace pre-flight check (AC-F20.1b).

use std::path::Path;
use tempfile::TempDir;

use super::{check_free_space, free_bytes};

/// AC-F20.1b: free_bytes returns a nonzero positive value for a real temp dir.
#[test]
fn t_f20_disk_free_bytes_nonzero_on_real_dir() {
    let dir = TempDir::new().expect("tempdir");
    let avail = free_bytes(dir.path()).expect("free_bytes");
    assert!(avail > 0, "expected positive free bytes, got 0");
}

/// AC-F20.1b: check_free_space succeeds when required is 0.
#[test]
fn t_f20_disk_check_succeeds_for_zero_required() {
    let dir = TempDir::new().expect("tempdir");
    let result = check_free_space(dir.path(), 0);
    assert!(
        result.is_ok(),
        "check_free_space(0) should always succeed: {:?}",
        result
    );
}

/// AC-F20.1b: check_free_space refuses when required exceeds available space.
///
/// We request u64::MAX bytes which no filesystem will have free.
#[test]
fn t_f20_disk_check_refuses_when_below_required() {
    let dir = TempDir::new().expect("tempdir");
    let result = check_free_space(dir.path(), u64::MAX);
    assert!(
        result.is_err(),
        "check_free_space(u64::MAX) should fail but returned Ok"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("insufficient disk space"),
        "error message should mention insufficient disk space: {}",
        msg
    );
}

/// AC-F20.1b: check_free_space proceeds when required is well below available.
#[test]
fn t_f20_disk_check_proceeds_when_above_required() {
    let dir = TempDir::new().expect("tempdir");
    // Require 1 byte -- always satisfiable on a real filesystem.
    let result = check_free_space(dir.path(), 1);
    assert!(
        result.is_ok(),
        "check_free_space(1) on a real dir should succeed: {:?}",
        result
    );
}

/// AC-F20.1b: error message includes required and available byte counts.
#[test]
fn t_f20_disk_check_error_message_includes_counts() {
    let dir = TempDir::new().expect("tempdir");
    let result = check_free_space(dir.path(), u64::MAX);
    let msg = result.unwrap_err().to_string();
    // Message must name both the required and available quantities.
    assert!(
        msg.contains("need") || msg.contains("have"),
        "error must include required/available counts: {}",
        msg
    );
}

/// AC-F20.1b: check_free_space fails with a clear message on a non-existent path.
#[cfg(unix)]
#[test]
fn t_f20_disk_check_nonexistent_path_returns_err() {
    let result = free_bytes(Path::new("/nonexistent/path/that/does/not/exist"));
    assert!(result.is_err(), "expected error for nonexistent path");
}
