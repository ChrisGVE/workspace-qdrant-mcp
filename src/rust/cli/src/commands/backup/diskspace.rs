//! Pre-flight free-space check for `wqm backup --full` (AC-F20.1b).
//!
//! Peak transient disk footprint formula:
//!   `sum(SQLite store sizes) + Qdrant snapshot size`
//!
//! Both the destination directory and any temp staging directory must have at
//! least this many free bytes BEFORE the backup starts.  The check refuses
//! with a clear required-vs-available error rather than failing partway and
//! leaving a torn/partial archive.
//!
//! Free-space detection uses `nix::sys::statvfs::statvfs` (already available
//! in wqm-cli via the `nix` crate with `features = ["fs"]`).

use std::path::Path;

use anyhow::{Context, Result};

/// Check that `dir` has at least `required_bytes` of free space.
///
/// Returns `Ok(available_bytes)` on success, or `Err` with a clear
/// required-vs-available message when space is insufficient.
pub(crate) fn check_free_space(dir: &Path, required_bytes: u64) -> Result<u64> {
    let available = free_bytes(dir)
        .with_context(|| format!("could not query free space on {}", dir.display()))?;

    if available < required_bytes {
        anyhow::bail!(
            "insufficient disk space on {}: need {} ({} bytes), have {} ({} bytes). \
             Free at least {} more bytes before running the full backup.",
            dir.display(),
            crate::output::format_bytes(required_bytes as i64),
            required_bytes,
            crate::output::format_bytes(available as i64),
            available,
            crate::output::format_bytes((required_bytes - available) as i64),
        );
    }

    Ok(available)
}

/// Return the number of bytes available (not just free) to the current user on
/// the filesystem that contains `path`.
///
/// Uses `nix::sys::statvfs::statvfs` (`bavail * frsize`) which accounts for
/// the reserved-blocks allowance and reflects what an unprivileged process can
/// actually write.
#[cfg(unix)]
pub(crate) fn free_bytes(path: &Path) -> Result<u64> {
    let stat = nix::sys::statvfs::statvfs(path)
        .with_context(|| format!("statvfs failed for {}", path.display()))?;
    // Cast both fields to u64 -- `blocks_available` and `fragment_size` may be
    // u32 or u64 depending on the platform/nix version.
    Ok(stat.blocks_available() as u64 * stat.fragment_size() as u64)
}

#[cfg(not(unix))]
pub(crate) fn free_bytes(_path: &Path) -> Result<u64> {
    // Non-Unix: return a large value so the check never fails.
    Ok(u64::MAX)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "diskspace_tests.rs"]
mod tests;
