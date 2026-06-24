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

use crate::commands::backup::compressor::detect as detect_compressor;
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

    // 2. Detect compressor.
    let compressor = detect_compressor()
        .context("no compressor found; install zstd, xz, or gzip (same tool used during backup)")?;
    output::kv("Decompressor", compressor.name);

    // 3. Confirmation prompt unless --force.
    if !force {
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
        if input.trim() != "yes" {
            output::info("Restore cancelled.");
            return Ok(());
        }
    }

    output::separator();
    output::info("Decompressing and extracting archive (streaming)...");

    // 4. Decompress archive as a streaming reader (PERF-NN-02).
    let archive_file = fs::File::open(archive)
        .with_context(|| format!("could not open archive: {}", archive.display()))?;

    let mut decomp_child = compressor
        .spawn_decompress(std::process::Stdio::from(archive_file))
        .context("could not spawn decompressor")?;

    let decomp_stdout = decomp_child
        .stdout
        .take()
        .context("decompressor stdout not captured")?;

    // Use a BufReader to smooth out small reads -- the underlying source is a
    // process pipe so reads are already bounded; this is purely for ergonomics.
    let buffered = BufReader::new(decomp_stdout);
    let mut tar_archive = tar::Archive::new(buffered);

    let tmp_qdrant = TempDir::new().context("could not create temp dir for Qdrant snapshot")?;
    let mut manifest: Option<BackupManifest> = None;
    let mut qdrant_snapshot_staging: Option<(String, PathBuf)> = None;
    let mut store_count = 0usize;

    // 5. Stream through tar entries.
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

        if path == "manifest.json" {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).context("read manifest.json")?;
            manifest = Some(
                BackupManifest::from_json_bytes(&buf).context("manifest.json is not valid JSON")?,
            );
            continue;
        }

        if let Some(rel) = path.strip_prefix("stores/") {
            // Restore SQLite store to data_dir/<rel>.
            let dest = data_dir.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("could not create parent dir: {}", parent.display())
                })?;
            }
            // Overwrite in-place (rename-safe: write to .tmp then rename).
            let tmp = dest.with_extension("db.restoring");
            {
                let mut out = fs::File::create(&tmp)
                    .with_context(|| format!("create tmp restore file: {}", tmp.display()))?;
                std::io::copy(&mut entry, &mut out)
                    .with_context(|| format!("write {}", tmp.display()))?;
            }
            fs::rename(&tmp, &dest)
                .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
            store_count += 1;
            output::info(format!("  restored stores/{}", rel));
            continue;
        }

        if let Some(snap_name) = path.strip_prefix("qdrant/") {
            // Write Qdrant snapshot to a temp file for upload.
            let staging_path = tmp_qdrant.path().join(snap_name);
            let mut out = fs::File::create(&staging_path)
                .with_context(|| format!("create staging file: {}", staging_path.display()))?;
            std::io::copy(&mut entry, &mut out)
                .with_context(|| format!("write qdrant snapshot to staging"))?;
            qdrant_snapshot_staging = Some((snap_name.to_string(), staging_path));
            continue;
        }
        // Unknown members are silently skipped for forward compatibility.
    }

    let status = decomp_child
        .wait()
        .context("decompressor process did not exit cleanly")?;
    if !status.success() {
        anyhow::bail!("decompressor exited with status: {}", status);
    }

    // 6. Display manifest.
    if let Some(ref m) = manifest {
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

    // 7. Upload Qdrant snapshot (optional leg).
    if let Some((snap_name, snap_path)) = qdrant_snapshot_staging {
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
