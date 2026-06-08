//! Zero-byte file handling for the ingestion pipeline.
//!
//! Empty files cannot be embedded or chunked, but should still be recorded in
//! `tracked_files` for inventory purposes and marked as done in the queue so
//! they do not remain in an errored state indefinitely.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::debug;

use crate::context::ProcessingContext;
use crate::file_classification::{get_extension_for_storage, is_test_file};
use crate::tracked_files_schema::{self, ProcessingStatus};
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{DestinationStatus, FilePayload, UnifiedQueueItem};

/// SHA-256 hash of the empty byte string.
#[cfg(test)]
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Returns `true` when the file at `path` exists, is a regular file, and has
/// zero bytes.
///
/// Rejects non-regular file types (symlinks, directories, FIFOs, device nodes)
/// to prevent DoS from sources like `/dev/zero` where `metadata().len()` is 0
/// but reading would never terminate.
pub(super) fn is_zero_byte(path: &Path) -> bool {
    path.metadata()
        .map_or(false, |m| m.file_type().is_file() && m.len() == 0)
}

/// Record a zero-byte file in `tracked_files` (chunk_count = 0, both statuses
/// set to `Skipped`) and mark both queue destinations as `Done`.
///
/// No embedding or Qdrant upsert is performed.
pub(super) async fn handle_zero_byte_file(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    watch_folder_id: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    debug!("Skipping 0-byte file: {}", payload.file_path);

    // Compute the file hash — for an empty file this is the SHA-256 of "".
    let file_hash = tracked_files_schema::compute_file_hash(file_path).unwrap_or_else(|_| {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string()
    });

    let file_mtime = tracked_files_schema::get_file_mtime(file_path)
        .unwrap_or_else(|_| wqm_common::timestamps::now_utc());

    let extension = get_extension_for_storage(file_path);
    let is_test = is_test_file(file_path);
    let base_point =
        wqm_common::hashing::compute_base_point(&item.tenant_id, relative_path, &file_hash);

    record_tracked_file(
        pool,
        item,
        watch_folder_id,
        relative_path,
        &file_mtime,
        &file_hash,
        extension.as_deref(),
        is_test,
        &base_point,
        payload.file_type.as_deref(),
    )
    .await?;

    mark_destinations_done(ctx, item).await;

    Ok(())
}

/// Insert or update the `tracked_files` row for a zero-byte file.
#[allow(clippy::too_many_arguments)]
pub(super) async fn record_tracked_file(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    watch_folder_id: &str,
    relative_path: &str,
    file_mtime: &str,
    file_hash: &str,
    extension: Option<&str>,
    is_test: bool,
    base_point: &str,
    payload_file_type: Option<&str>,
) -> UnifiedProcessorResult<()> {
    let existing = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("tracked_files lookup failed: {}", e))
    })?;

    let mut tx = pool.begin().await.map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e))
    })?;

    if let Some(existing_file) = existing {
        tracked_files_schema::update_tracked_file_tx(
            &mut tx,
            existing_file.file_id,
            file_mtime,
            file_hash,
            0,    // chunk_count
            None, // chunking_method
            ProcessingStatus::Skipped,
            ProcessingStatus::Skipped,
            Some(base_point),
            None, // component
        )
        .await
        .map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!("tracked_files update failed: {}", e))
        })?;
    } else {
        tracked_files_schema::insert_tracked_file_tx(
            &mut tx,
            watch_folder_id,
            relative_path,
            Some(item.branch.as_str()),
            payload_file_type,
            None, // language
            file_mtime,
            file_hash,
            0,    // chunk_count
            None, // chunking_method
            ProcessingStatus::Skipped,
            ProcessingStatus::Skipped,
            Some(&item.collection),
            extension,
            is_test,
            Some(base_point),
            None, // component
        )
        .await
        .map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!("tracked_files insert failed: {}", e))
        })?;
    }

    tx.commit().await.map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e))
    })?;

    Ok(())
}

/// Mark both qdrant and search destinations as done so the queue item is
/// dequeued cleanly.
pub(super) async fn mark_destinations_done(ctx: &ProcessingContext, item: &UnifiedQueueItem) {
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn zero_byte_file_detected() {
        let tmp = NamedTempFile::new().expect("create temp file");
        // NamedTempFile creates a 0-byte file by default
        assert!(is_zero_byte(tmp.path()), "fresh temp file should be 0-byte");
    }

    #[test]
    fn non_empty_file_not_detected() {
        let mut tmp = NamedTempFile::new().expect("create temp file");
        writeln!(tmp, "hello").expect("write content");
        assert!(
            !is_zero_byte(tmp.path()),
            "file with content should not be 0-byte"
        );
    }

    #[test]
    fn nonexistent_path_returns_false() {
        let path = Path::new("/nonexistent/path/that/does/not/exist.txt");
        assert!(
            !is_zero_byte(path),
            "nonexistent path should return false, not panic"
        );
    }

    #[test]
    fn empty_file_hash_is_sha256_of_empty() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let hash = tracked_files_schema::compute_file_hash(tmp.path())
            .expect("should hash a 0-byte file successfully");
        assert_eq!(hash, EMPTY_SHA256);
    }

    #[test]
    fn directory_not_detected_as_zero_byte() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        assert!(
            !is_zero_byte(dir.path()),
            "directory must not be treated as zero-byte file"
        );
    }

    #[test]
    fn dev_zero_not_detected_as_zero_byte() {
        let path = Path::new("/dev/zero");
        if !path.exists() {
            return;
        }
        assert!(
            !is_zero_byte(path),
            "/dev/zero must not be classified as zero-byte"
        );
    }

    #[test]
    fn dev_null_not_detected_as_zero_byte() {
        let path = Path::new("/dev/null");
        if !path.exists() {
            return;
        }
        assert!(
            !is_zero_byte(path),
            "/dev/null must not be classified as zero-byte"
        );
    }

    #[test]
    fn fifo_not_detected_as_zero_byte() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let fifo_path = dir.path().join("test_fifo");
        let fifo_cstr =
            std::ffi::CString::new(fifo_path.to_str().unwrap()).expect("cstring from path");
        let ret = unsafe { libc::mkfifo(fifo_cstr.as_ptr(), 0o644) };
        assert_eq!(ret, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        assert!(
            !is_zero_byte(&fifo_path),
            "FIFO must not be classified as zero-byte"
        );
    }

    #[test]
    fn broken_symlink_returns_false() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let link_path = dir.path().join("broken_link");
        std::os::unix::fs::symlink("/nonexistent/target", &link_path).expect("create symlink");
        assert!(
            !is_zero_byte(&link_path),
            "broken symlink should return false"
        );
    }
}
