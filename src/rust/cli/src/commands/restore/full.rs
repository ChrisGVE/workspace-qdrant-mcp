//! `wqm restore --full <archive>` implementation (F20 / AC-F20.2).
//!
//! Restores the truth-inclusive bundle produced by `wqm backup --full`:
//!   1. Calls `wqm_common::guard::assert_daemon_stopped` -- refuses if daemon
//!      is live (AC-F20.4 / DR GP-4).
//!   2. Decompresses the archive by piping the file through the external
//!      compressor's `-d -c` mode into a streaming `tar::Archive` (PERF-NN-02:
//!      never buffers the whole archive in memory).
//!   3. Extracts SQLite stores from `stores/*` members to the data directory,
//!      reproducing the original directory layout.
//!   4. Writes the Qdrant snapshot member (`qdrant/*`) to a temp file and
//!      uploads it via the existing REST upload path (reusing
//!      `restore/from_backup.rs` Qdrant upload logic -- no second upload path,
//!      FP-2 / DR GP-9).  The Qdrant leg is skipped with a warning when
//!      Qdrant is unreachable or the snapshot is absent from the bundle.
//!   5. Reads and displays the `manifest.json` member.
//!
//! ## Guard effectiveness caveat (#175)
//!
//! The flock guard becomes fully effective only once `memexd` acquires
//! `DaemonLock` at startup (rides #175 daemon/write-crate cutover).  Until
//! #175 lands the guard is a best-effort probe.  See `wqm_common::guard` docs.

use std::fs;
use std::io::{BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tempfile::TempDir;

use crate::commands::backup::compressor::{detect as detect_compressor, Compressor};
use crate::commands::backup::manifest::BackupManifest;
use crate::output;

/// Run `wqm restore --full <archive>`.
///
/// The archive must have been produced by `wqm backup --full`.
pub async fn restore_full(archive: &Path, force: bool) -> Result<()> {
    output::section("Full Restore");
    output::kv("Archive", archive.display().to_string());

    if !archive.exists() {
        anyhow::bail!("archive not found: {}", archive.display());
    }

    // 1. Guard: refuse if daemon is running (AC-F20.4).
    let data_dir = guard_daemon_stopped()?;

    // 2. Detect compressor.
    let compressor = detect_compressor()
        .context("no compressor found; install zstd, xz, or gzip (same tool used during backup)")?;
    output::kv("Decompressor", compressor.name);

    // 3. Confirmation prompt unless --force.
    if !confirm_restore(force)? {
        output::info("Restore cancelled.");
        return Ok(());
    }

    output::separator();
    output::info("Decompressing and extracting archive (streaming)...");

    // 4-5. Decompress (streaming) + extract members.
    let tmp_qdrant = TempDir::new().context("could not create temp dir for Qdrant snapshot")?;
    let (manifest, qdrant_staging, store_count) =
        extract_archive(archive, &compressor, &data_dir, tmp_qdrant.path())?;

    // 6. Display manifest.
    display_manifest(&manifest, store_count);

    // 7. Upload Qdrant snapshot (optional leg).
    if let Some((snap_name, snap_path)) = qdrant_staging {
        upload_qdrant_snapshot(&snap_path, &snap_name).await;
    } else {
        output::info("No Qdrant snapshot in archive (or Qdrant leg was skipped during backup).");
        output::info("Run `wqm admin rebuild` to re-derive the Qdrant index from SQLite truth.");
    }

    output::separator();
    output::success("Full restore complete.");
    output::info(
        "Restart the daemon to resume normal operation: wqm service start \
         (or: launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist)",
    );

    Ok(())
}

// ---- Steps -----------------------------------------------------------------

/// Resolve the data dir and refuse if the daemon is running (AC-F20.4).
fn guard_daemon_stopped() -> Result<PathBuf> {
    let data_dir = wqm_common::paths::get_data_dir()
        .map_err(|e| anyhow::anyhow!("could not determine data directory: {}", e))?;

    if let Err(e) = wqm_common::guard::assert_daemon_stopped(&data_dir) {
        output::error(format!("{}", e));
        output::info(
            "Note: the daemon-running guard is best-effort until #175 (daemon/write-crate \
             cutover) lands.",
        );
        anyhow::bail!("Cannot restore while daemon is running");
    }
    Ok(data_dir)
}

/// Prompt for confirmation unless `force`.  Returns `Ok(true)` to proceed.
fn confirm_restore(force: bool) -> Result<bool> {
    if force {
        return Ok(true);
    }
    output::warning(
        "This will overwrite SQLite stores in the data directory and upload a Qdrant snapshot!",
    );
    output::separator();
    eprint!("Type 'yes' to confirm restore: ");
    std::io::stderr().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("could not read confirmation")?;
    Ok(input.trim() == "yes")
}

/// Classification of one extracted tar entry.
enum EntryOutcome {
    Manifest(BackupManifest),
    Store,
    Qdrant(String, PathBuf),
    Skip,
}

/// Decompress the archive (streaming) and extract its members.
///
/// Returns `(manifest, qdrant_staging, store_count)`.  Decompression pipes the
/// archive file through the compressor's `-d -c` mode into a streaming
/// `tar::Archive`; the whole archive is never buffered in memory (PERF-NN-02).
fn extract_archive(
    archive: &Path,
    compressor: &Compressor,
    data_dir: &Path,
    tmp_qdrant: &Path,
) -> Result<(Option<BackupManifest>, Option<(String, PathBuf)>, usize)> {
    let archive_file = fs::File::open(archive)
        .with_context(|| format!("could not open archive: {}", archive.display()))?;

    let mut decomp_child = compressor
        .spawn_decompress(std::process::Stdio::from(archive_file))
        .context("could not spawn decompressor")?;

    let decomp_stdout = decomp_child
        .stdout
        .take()
        .context("decompressor stdout not captured")?;

    // BufReader smooths small reads; the source is a process pipe so reads are
    // already bounded -- this is purely for ergonomics.
    let mut tar_archive = tar::Archive::new(BufReader::new(decomp_stdout));

    let mut manifest: Option<BackupManifest> = None;
    let mut qdrant_staging: Option<(String, PathBuf)> = None;
    let mut store_count = 0usize;

    for entry in tar_archive
        .entries()
        .context("could not read tar entries")?
    {
        let mut entry = entry.context("corrupt tar entry")?;
        let path = entry
            .path()
            .context("tar entry has no path")?
            .to_string_lossy()
            .to_string();

        match restore_one_entry(&mut entry, &path, data_dir, tmp_qdrant)? {
            EntryOutcome::Manifest(m) => manifest = Some(m),
            EntryOutcome::Store => store_count += 1,
            EntryOutcome::Qdrant(name, p) => qdrant_staging = Some((name, p)),
            EntryOutcome::Skip => {}
        }
    }

    let status = decomp_child
        .wait()
        .context("decompressor process did not exit cleanly")?;
    if !status.success() {
        anyhow::bail!("decompressor exited with status: {}", status);
    }

    Ok((manifest, qdrant_staging, store_count))
}

/// Classify and write one tar entry.  Unknown members are skipped for forward
/// compatibility.
fn restore_one_entry<R: std::io::Read>(
    entry: &mut tar::Entry<'_, R>,
    path: &str,
    data_dir: &Path,
    tmp_qdrant: &Path,
) -> Result<EntryOutcome> {
    if path == "manifest.json" {
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).context("read manifest.json")?;
        let m = BackupManifest::from_json_bytes(&buf).context("manifest.json is not valid JSON")?;
        return Ok(EntryOutcome::Manifest(m));
    }

    if let Some(rel) = path.strip_prefix("stores/") {
        // Restore SQLite store to data_dir/<rel>, rename-safe (write .tmp then rename).
        let dest = data_dir.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("could not create parent dir: {}", parent.display()))?;
        }
        let tmp = dest.with_extension("db.restoring");
        {
            let mut out = fs::File::create(&tmp)
                .with_context(|| format!("create tmp restore file: {}", tmp.display()))?;
            std::io::copy(entry, &mut out).with_context(|| format!("write {}", tmp.display()))?;
        }
        fs::rename(&tmp, &dest)
            .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
        output::info(format!("  restored stores/{}", rel));
        return Ok(EntryOutcome::Store);
    }

    if let Some(snap_name) = path.strip_prefix("qdrant/") {
        // Write Qdrant snapshot to a temp file for upload.
        let staging_path = tmp_qdrant.join(snap_name);
        let mut out = fs::File::create(&staging_path)
            .with_context(|| format!("create staging file: {}", staging_path.display()))?;
        std::io::copy(entry, &mut out).context("write qdrant snapshot to staging")?;
        return Ok(EntryOutcome::Qdrant(snap_name.to_string(), staging_path));
    }

    Ok(EntryOutcome::Skip)
}

/// Display the manifest summary (or a warning when absent).
fn display_manifest(manifest: &Option<BackupManifest>, store_count: usize) {
    if let Some(m) = manifest {
        output::separator();
        output::kv("Backup version", &m.wqm_version);
        output::kv("Timestamp", &m.archive_timestamp);
        output::kv("Compressor", &m.compressor);
        output::kv(
            "Daemon was running at backup",
            if m.daemon_running { "yes" } else { "no" },
        );
        if m.daemon_running {
            output::warning(
                "This backup was taken with the daemon running. \
                 Run `wqm admin rebuild` after restore to ensure Qdrant index consistency.",
            );
        }
    } else {
        output::warning("No manifest.json found in archive -- proceeding without manifest.");
    }

    output::kv("SQLite stores restored", store_count.to_string());
}

// ---- Helpers ---------------------------------------------------------------

/// Upload a staged Qdrant snapshot via REST multipart upload.
///
/// Errors are logged as warnings rather than aborting the restore -- the SQLite
/// truth is already restored and is the durable recovery anchor.  The Qdrant
/// index can always be rebuilt with `wqm admin rebuild`.
async fn upload_qdrant_snapshot(staging_path: &Path, _snap_name: &str) {
    use crate::commands::restore::client::{build_client, qdrant_url};

    output::info("Uploading Qdrant snapshot...");

    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            output::warning(format!(
                "Skipping Qdrant upload: could not build HTTP client: {}",
                e
            ));
            output::info("Run `wqm admin rebuild` to re-derive the Qdrant index.");
            return;
        }
    };

    // Read the snapshot bytes.
    let bytes = match tokio::fs::read(staging_path).await {
        Ok(b) => b,
        Err(e) => {
            output::warning(format!("Skipping Qdrant upload: read staging file: {}", e));
            return;
        }
    };

    let file_name = staging_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("snapshot.snapshot")
        .to_string();

    let base = qdrant_url();
    let upload_url = format!("{}/snapshots/upload", base);

    let part = match reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str("application/octet-stream")
    {
        Ok(p) => p,
        Err(e) => {
            output::warning(format!("Skipping Qdrant upload: multipart error: {}", e));
            return;
        }
    };

    let form = reqwest::multipart::Form::new().part("snapshot", part);

    let resp = match client
        .post(&upload_url)
        .query(&[("wait", "true"), ("priority", "snapshot")])
        .multipart(form)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            output::warning(format!("Skipping Qdrant upload: request failed: {}", e));
            output::info("Run `wqm admin rebuild` to re-derive the Qdrant index.");
            return;
        }
    };

    if resp.status().is_success() {
        output::success("Qdrant snapshot uploaded successfully.");
    } else {
        let code = resp.status();
        let body = resp.text().await.unwrap_or_default();
        output::warning(format!(
            "Qdrant upload returned {}: {} -- run `wqm admin rebuild` to recover the index.",
            code, body
        ));
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[path = "full_restore_tests.rs"]
mod tests;
