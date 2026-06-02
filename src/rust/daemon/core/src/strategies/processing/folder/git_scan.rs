//! Git-aware fast-path discovery.
//!
//! The default project discovery is a progressive, single-level filesystem
//! walk (`scan.rs`): each `Folder/Scan` queue item reads one directory level
//! and enqueues child files + child `Folder/Scan` items. The queue `total`
//! therefore grows *organically* as the breadth-first walk drains, so the
//! reported percentage is computed against a denominator that is still
//! expanding — misleading for large trees.
//!
//! For git repositories we can do better: the git **index** already enumerates
//! every tracked file (exactly what `git ls-files` returns). Reading it once,
//! up front, lets us enqueue all `File/Add` items in a single pass so `total`
//! reflects the real file count from the very first scan. Non-git directories
//! and any git error fall back to the progressive FS walk transparently.
//!
//! Gates applied per file mirror the FS scan exactly:
//! `.wqmignore`/`.gitignore` (parent-aware), [`should_exclude_file`], the
//! extension allowlist, mtime pruning, and the max-size cap — all reused from
//! [`process_file_entry`]. Submodule gitlinks are enqueued as their own
//! `Tenant/Add`, same as the FS scan.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use git2::Repository;
use tracing::{debug, warn};
use wqm_common::paths::CanonicalPath;

use crate::allowed_extensions::AllowedExtensions;
use crate::patterns::gitignore::ProjectIgnoreMatcher;
use crate::queue_operations::QueueManager;
use crate::unified_queue_processor::UnifiedProcessorResult;
use crate::unified_queue_schema::UnifiedQueueItem;

use super::scan::{enqueue_submodule, parse_iso8601_to_system_time, process_file_entry};

/// git filemode for a gitlink (submodule) index entry.
const GIT_FILEMODE_COMMIT: u32 = 0o160000;

/// Whether the git fast-path discovery is enabled (default: yes).
///
/// Set `WQM_GIT_FASTPATH_DISCOVERY=0` (or `false`/`no`) to force the legacy
/// progressive FS scan for git repositories too.
pub(crate) fn git_fastpath_enabled() -> bool {
    !matches!(
        std::env::var("WQM_GIT_FASTPATH_DISCOVERY")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

/// One enumerated git index entry: path relative to the repo workdir, plus the
/// filemode (used to detect submodule gitlinks). Collected synchronously so no
/// `!Send` git2 handle is held across an `.await`.
struct IndexEntry {
    rel_path: String,
    mode: u32,
}

/// Enumerate `root_dir`'s git index and enqueue every tracked file (and any
/// submodule gitlinks), applying the same exclusion gates as the FS scan.
///
/// Returns `Some(Ok((files, submodules, excluded, errors)))` on success,
/// `Some(Err(..))` if enqueueing failed mid-way, or `None` when `root_dir` is
/// not a git repository / the index can't be read — in which case the caller
/// falls back to the progressive FS scan.
pub(crate) async fn enumerate_git_index(
    root_dir: &Path,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
    last_scan: Option<&str>,
) -> Option<UnifiedProcessorResult<(u64, u64, u64, u64)>> {
    // ── Phase 1 (sync): read the index, drop all git2 handles before await. ──
    // git2 types are `!Send`; holding them across an `.await` would make this
    // future non-Send and fail to compile under the multi-threaded runtime.
    let (entries, decode_errors) = match collect_index_entries(root_dir) {
        Some(v) => v,
        None => return None, // not a git repo / unreadable index -> FS fallback
    };

    // ── Phase 2 (async): enqueue from the owned, Send-safe Vec. ──
    Some(
        enqueue_entries(
            entries,
            decode_errors,
            watch_folder_root,
            item,
            queue_manager,
            allowed_extensions,
            last_scan,
        )
        .await,
    )
}

/// Synchronous git index read. Returns the tracked entries plus a count of
/// non-UTF-8 paths that had to be skipped, or `None` if `root_dir` is not a
/// git repo / the index cannot be opened.
fn collect_index_entries(root_dir: &Path) -> Option<(Vec<IndexEntry>, u64)> {
    let repo = match Repository::open(root_dir) {
        Ok(r) => r,
        Err(e) => {
            debug!(
                "git fast-path: {} is not a git repo ({}); using FS scan",
                root_dir.display(),
                e
            );
            return None;
        }
    };
    let index = match repo.index() {
        Ok(i) => i,
        Err(e) => {
            warn!(
                "git fast-path: cannot read index for {} ({}); using FS scan",
                root_dir.display(),
                e
            );
            return None;
        }
    };

    let mut entries = Vec::with_capacity(index.len());
    let mut decode_errors = 0u64;
    for entry in index.iter() {
        match std::str::from_utf8(&entry.path) {
            Ok(s) => entries.push(IndexEntry {
                rel_path: s.to_string(),
                mode: entry.mode,
            }),
            Err(_) => decode_errors += 1,
        }
    }
    Some((entries, decode_errors))
}

#[allow(clippy::too_many_arguments)]
async fn enqueue_entries(
    entries: Vec<IndexEntry>,
    decode_errors: u64,
    watch_folder_root: &CanonicalPath,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
    last_scan: Option<&str>,
) -> UnifiedProcessorResult<(u64, u64, u64, u64)> {
    let baseline = last_scan.and_then(parse_iso8601_to_system_time);
    let root = Path::new(watch_folder_root.as_str());

    let mut files_queued = 0u64;
    let mut submodules_queued = 0u64;
    let mut files_excluded = 0u64;
    let mut errors = decode_errors;

    // One ignore matcher per parent directory (cascade root -> dir), mirroring
    // the per-directory matcher the FS scan builds. `None` means "no ignore
    // files apply to this directory".
    let mut matcher_cache: HashMap<PathBuf, Option<ProjectIgnoreMatcher>> = HashMap::new();

    for entry in entries {
        // Index paths are relative to the repo workdir, which equals the
        // watch_folder root for a top-level project.
        let abs = root.join(&entry.rel_path);

        // Submodule gitlink -> enqueue as its own tenant (same as the FS scan).
        if entry.mode == GIT_FILEMODE_COMMIT {
            submodules_queued += enqueue_submodule(&abs, item, queue_manager, &mut errors).await;
            continue;
        }

        // Gate 0: `.gitignore` is already enforced by git (the index only holds
        // tracked files), but re-apply the matcher so `.wqmignore` exclusions
        // (and any force-added paths) are honored — including directory rules
        // inherited from ancestor directories.
        let parent = abs.parent().unwrap_or(root).to_path_buf();
        let matcher = matcher_cache
            .entry(parent.clone())
            .or_insert_with(|| ProjectIgnoreMatcher::for_dir(&parent, Some(root)));
        if let Some(m) = matcher {
            if is_ignored_with_ancestors(m, root, &abs) {
                files_excluded += 1;
                continue;
            }
        }

        files_queued += process_file_entry(
            &abs,
            watch_folder_root,
            item,
            queue_manager,
            allowed_extensions,
            baseline.as_ref(),
            &mut files_excluded,
            &mut errors,
        )
        .await;
    }

    Ok((files_queued, submodules_queued, files_excluded, errors))
}

/// Parent-aware ignore check for a file enumerated directly from the index.
///
/// The progressive FS scan excludes an ignored directory at the directory
/// level and never descends into it, so a rule like `reports/` is only ever
/// tested against the directory. Here we reach the nested file directly, so we
/// must replay that logic: test the file itself, then each ancestor directory
/// up to (but not including) the watch-folder root.
fn is_ignored_with_ancestors(
    matcher: &ProjectIgnoreMatcher,
    root: &Path,
    abs_file: &Path,
) -> bool {
    if matcher.is_ignored(abs_file, false) {
        return true;
    }
    let mut current = abs_file.parent();
    while let Some(dir) = current {
        if dir == root || !dir.starts_with(root) {
            break;
        }
        if matcher.is_ignored(dir, true) {
            return true;
        }
        current = dir.parent();
    }
    false
}
