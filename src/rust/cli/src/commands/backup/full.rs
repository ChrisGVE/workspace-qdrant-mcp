//! `wqm backup --full <destination>` implementation (F20 / AC-F20.1).
//!
//! Produces a single compressed archive bundling:
//!   (a) every SQLite truth store (`state.db` + per-project/global/libraries
//!       `store.db`) copied read-consistently via `VACUUM INTO`;
//!   (b) a Qdrant snapshot (reusing `backup::create::trigger_snapshot` and
//!       `download_snapshot` -- no second snapshot path, FP-2 / DR GP-9);
//!   (c) `manifest.json` recording bundle contents and metadata (AC-F20.1).
//!
//! The archive is produced by streaming raw tar bytes through the external
//! compressor's stdin to the destination file.  Peak transient disk is
//! bounded by the pre-flight formula: sum(store sizes) + Qdrant snapshot size
//! (AC-F20.1b).
//!
//! ## Daemon guard
//!
//! `backup --full` is read-only and may run with the daemon up (AC-F20.4).
//! It records `daemon_running` in the manifest (DATA-N01) so a restore knows
//! the bundle may have a mild SQLite<->Qdrant temporal skew.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tempfile::TempDir;

use super::compressor::detect as detect_compressor;
use super::diskspace::check_free_space;
use super::manifest::{BackupManifest, StoreEntry};
use super::stores::{discover_stores, to_store_entry, total_store_bytes, vacuum_into};
use super::types::{build_client, qdrant_url};
use crate::output;

/// Run `wqm backup --full <destination>`.
///
/// `destination` is the path (including filename) for the output archive.
/// If Qdrant is unreachable the Qdrant snapshot leg is skipped with a warning
/// and the archive contains the SQLite stores only.
pub async fn backup_full(destination: &Path) -> Result<()> {
    output::section("Full Backup");
    output::kv("Destination", destination.display().to_string());

    // 1. Detect compressor.
    let compressor = detect_compressor()
        .context("no compressor found; install zstd, xz, or gzip before running backup --full")?;
    output::kv("Compressor", compressor.name);

    // 2-6. Stage SQLite copies + Qdrant snapshot + pre-flight free-space check.
    let prepared = prepare_backup(destination).await?;

    // 7. Build manifest.
    let manifest_bytes = build_manifest_bytes(&prepared, compressor.name)?;

    // 8. Build tar archive, stream through compressor to destination file.
    output::info(format!(
        "Writing archive ({} compression)...",
        compressor.name
    ));
    write_archive(destination, &compressor, &prepared, &manifest_bytes)?;

    // 9. Summary.
    print_backup_summary(destination, &prepared, compressor.name);

    Ok(())
}

// ---- Steps -----------------------------------------------------------------

/// Result of the staging phase: SQLite copies, the optional Qdrant snapshot,
/// and the daemon-running flag.  Holds the `TempDir` alive until the archive
/// is written.
struct PreparedBackup {
    _stage_dir: TempDir,
    staged_stores: Vec<(String, PathBuf)>,
    store_entries: Vec<StoreEntry>,
    qdrant_snapshot_path: Option<PathBuf>,
    qdrant_snapshot_name: Option<String>,
    daemon_running: bool,
}

/// Discover stores, stage read-consistent SQLite copies, attempt a Qdrant
/// snapshot, and run the AC-F20.1b pre-flight free-space check.
async fn prepare_backup(destination: &Path) -> Result<PreparedBackup> {
    // Discover stores.
    let data_dir = wqm_common::paths::get_data_dir()
        .map_err(|e| anyhow::anyhow!("could not determine data directory: {}", e))?;
    let stores = discover_stores(&data_dir);
    if stores.is_empty() {
        output::warning("No SQLite stores found under data directory; archive will be minimal.");
    }
    output::kv("Stores found", stores.len().to_string());
    for s in &stores {
        output::info(format!("  {}", s.rel_path));
    }

    // Daemon-running probe (DATA-N01: for manifest only -- backup is read-only,
    // it does NOT refuse).
    let daemon_running = is_daemon_running_probe(&data_dir);
    if daemon_running {
        output::warning(
            "Daemon appears to be running. The archive may have a mild SQLite<->Qdrant \
             temporal skew. After restoring, run `wqm admin rebuild` to re-derive a \
             consistent Qdrant index (AC-F20.1 DATA-N01).",
        );
    }

    // Stage SQLite copies + attempt Qdrant snapshot.
    let stage_dir = TempDir::new().context("could not create temp staging directory")?;
    let staged_stores = stage_sqlite_copies(&stores, stage_dir.path())?;
    let (qdrant_snapshot_path, qdrant_snapshot_name) =
        attempt_qdrant_snapshot(stage_dir.path()).await;

    // Pre-flight free-space check (AC-F20.1b).
    let qdrant_bytes: u64 = qdrant_snapshot_path
        .as_deref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let required = total_store_bytes(&stores) + qdrant_bytes;
    let dest_parent = ensure_dest_parent(destination)?;
    output::kv("Required space", output::format_bytes(required as i64));
    check_free_space(&dest_parent, required).context("pre-flight free-space check failed")?;
    check_free_space(stage_dir.path(), 0).ok(); // staging dir check (informational)

    output::kv("Qdrant URL", qdrant_url());
    output::separator();

    let store_entries: Vec<StoreEntry> = stores.iter().map(to_store_entry).collect();
    Ok(PreparedBackup {
        _stage_dir: stage_dir,
        staged_stores,
        store_entries,
        qdrant_snapshot_path,
        qdrant_snapshot_name,
        daemon_running,
    })
}

/// Resolve and create the destination's parent directory.
fn ensure_dest_parent(destination: &Path) -> Result<PathBuf> {
    let dest_parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    if !dest_parent.exists() {
        std::fs::create_dir_all(&dest_parent).with_context(|| {
            format!(
                "could not create destination directory: {}",
                dest_parent.display()
            )
        })?;
    }
    Ok(dest_parent)
}

/// Build the `manifest.json` bytes from the prepared bundle.
fn build_manifest_bytes(
    prepared: &PreparedBackup,
    compressor_name: &'static str,
) -> Result<Vec<u8>> {
    let manifest = BackupManifest::new(
        prepared.store_entries.clone(),
        prepared.qdrant_snapshot_name.clone(),
        compressor_name,
        prepared.daemon_running,
    );
    manifest.to_json_bytes()
}

/// Stream a tar archive of the prepared bundle through the compressor into the
/// destination file.
fn write_archive(
    destination: &Path,
    compressor: &super::compressor::Compressor,
    prepared: &PreparedBackup,
    manifest_bytes: &[u8],
) -> Result<()> {
    let dest_file = File::create(destination)
        .with_context(|| format!("could not create archive: {}", destination.display()))?;

    let mut child = compressor
        .spawn_compress(std::process::Stdio::from(dest_file))
        .context("could not spawn compressor")?;

    {
        let stdin = child
            .stdin
            .take()
            .context("compressor stdin not captured")?;
        write_tar_to_writer(
            stdin,
            &prepared.staged_stores,
            prepared.qdrant_snapshot_path.as_deref(),
            &prepared.qdrant_snapshot_name,
            manifest_bytes,
        )?;
    }

    let status = child.wait().context("compressor did not exit cleanly")?;
    if !status.success() {
        anyhow::bail!("compressor exited with status: {}", status);
    }
    Ok(())
}

/// Print the post-backup summary.
fn print_backup_summary(
    destination: &Path,
    prepared: &PreparedBackup,
    compressor_name: &'static str,
) {
    let archive_size = std::fs::metadata(destination).map(|m| m.len()).unwrap_or(0);
    output::separator();
    output::success(format!("Archive written: {}", destination.display()));
    output::kv("Archive size", output::format_bytes(archive_size as i64));
    output::kv("SQLite stores", prepared.staged_stores.len().to_string());
    output::kv(
        "Qdrant snapshot",
        prepared
            .qdrant_snapshot_name
            .as_deref()
            .unwrap_or("skipped (Qdrant unreachable)"),
    );
    output::kv("Compressor", compressor_name);
    if prepared.daemon_running {
        output::info(
            "Reminder: daemon was running during backup. \
             Run `wqm admin rebuild` after restore to ensure index consistency.",
        );
    }
}

// ---- Helpers ---------------------------------------------------------------

/// Stage SQLite copies via VACUUM INTO; returns list of (rel_path, staged_abs_path).
fn stage_sqlite_copies(
    stores: &[super::stores::DiscoveredStore],
    stage_dir: &Path,
) -> Result<Vec<(String, PathBuf)>> {
    let mut staged: Vec<(String, PathBuf)> = Vec::with_capacity(stores.len());
    for store in stores {
        let safe_name = store.rel_path.replace('/', "_");
        let dest = stage_dir.join(&safe_name);
        vacuum_into(&store.abs_path, &dest)
            .with_context(|| format!("VACUUM INTO failed for {}", store.rel_path))?;
        staged.push((store.rel_path.clone(), dest));
    }
    Ok(staged)
}

/// Attempt to trigger and download a Qdrant snapshot to `stage_dir`.
///
/// Returns `(Some(path), Some(name))` on success, `(None, None)` on any error
/// (Qdrant unreachable, etc.).  Errors are logged as warnings; they do not
/// abort the backup -- the SQLite truth is always the primary goal.
async fn attempt_qdrant_snapshot(stage_dir: &Path) -> (Option<PathBuf>, Option<String>) {
    use super::create::{download_snapshot, trigger_snapshot};

    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            output::warning(format!(
                "Skipping Qdrant snapshot: could not build HTTP client: {}",
                e
            ));
            return (None, None);
        }
    };

    let snapshot = match trigger_snapshot(&client, "all", true).await {
        Ok(s) => s,
        Err(e) => {
            output::warning(format!(
                "Skipping Qdrant snapshot: Qdrant unreachable or snapshot failed: {}",
                e
            ));
            return (None, None);
        }
    };

    let stage_path = stage_dir.to_path_buf();
    if let Err(e) = download_snapshot(&client, "all", &snapshot, &stage_path, true).await {
        output::warning(format!("Skipping Qdrant snapshot: download failed: {}", e));
        return (None, None);
    }

    let name = snapshot.name.clone();
    let path = stage_dir.join(&name);
    if path.exists() {
        (Some(path), Some(name))
    } else {
        output::warning("Qdrant snapshot downloaded but file not found at expected path");
        (None, None)
    }
}

/// Write all staged files into a tar archive on `writer`.
///
/// Archive members:
///   - `stores/<rel_path>` for each SQLite copy
///   - `qdrant/<snapshot_name>` for the Qdrant snapshot (if present)
///   - `manifest.json` at the root
fn write_tar_to_writer<W: std::io::Write>(
    writer: W,
    staged_stores: &[(String, PathBuf)],
    qdrant_path: Option<&Path>,
    qdrant_name: &Option<String>,
    manifest_bytes: &[u8],
) -> Result<()> {
    let mut builder = tar::Builder::new(writer);

    // SQLite stores.
    for (rel_path, staged_path) in staged_stores {
        let archive_name = format!("stores/{}", rel_path);
        let mut file = File::open(staged_path)
            .with_context(|| format!("open staged store: {}", staged_path.display()))?;
        builder
            .append_file(&archive_name, &mut file)
            .with_context(|| format!("tar: append {}", archive_name))?;
    }

    // Qdrant snapshot.
    if let (Some(path), Some(name)) = (qdrant_path, qdrant_name) {
        let archive_name = format!("qdrant/{}", name);
        let mut file = File::open(path)
            .with_context(|| format!("open qdrant snapshot: {}", path.display()))?;
        builder
            .append_file(&archive_name, &mut file)
            .with_context(|| format!("tar: append {}", archive_name))?;
    }

    // manifest.json.
    let mut header = tar::Header::new_gnu();
    header.set_size(manifest_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, "manifest.json", manifest_bytes)
        .context("tar: append manifest.json")?;

    builder.finish().context("tar: finalize archive")?;
    Ok(())
}

/// Probe whether the daemon lock file is held (best-effort; see #175 caveat).
fn is_daemon_running_probe(data_dir: &Path) -> bool {
    wqm_common::guard::assert_daemon_stopped(data_dir).is_err()
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "full_tests.rs"]
mod tests;
