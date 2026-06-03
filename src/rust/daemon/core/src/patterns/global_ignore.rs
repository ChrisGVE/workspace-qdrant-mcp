//! Daemon-wide global ignore matcher (`global.wqmignore`).
//!
//! The directory-walk path (folder-scan strategy, see
//! `strategies/processing/folder/scan.rs`) and the ignore-sync reconciler (see
//! `startup/reconciliation/ignore_sync.rs`) both apply `global.wqmignore`, but
//! the file-watcher → queue enqueue path historically did NOT. That let the
//! watcher enqueue `Add`/`Update`/`Delete` operations for paths the user
//! globally excluded — most damagingly the daemon's own Qdrant storage under
//! `<repo>/state/qdrant/`: an embedding write rotates a Qdrant segment file →
//! the watcher fires an `Update` event → the daemon re-indexes the segment →
//! more Qdrant writes → more rotation → a self-sustaining feedback loop
//! (observed `reindex_count` up to 46 on a single segment).
//!
//! This module gives the watcher the SAME global exclusion check, built from
//! the SAME file the reconciler uses. The matcher is cached process-wide and
//! transparently rebuilt when the file's mtime changes, so edits via the admin
//! UI take effect on the next event without a daemon restart.

use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use once_cell::sync::Lazy;
use tracing::{debug, warn};

/// Cached compiled matcher plus the inputs used to detect staleness.
struct CachedMatcher {
    matcher: Gitignore,
    /// `global.wqmignore` mtime when this matcher was built (None if absent).
    loaded_mtime: Option<SystemTime>,
    /// Resolved path the matcher was built from.
    path: PathBuf,
}

static GLOBAL_IGNORE: Lazy<RwLock<Option<CachedMatcher>>> = Lazy::new(|| RwLock::new(None));

/// Resolve `dirname(WQM_DATABASE_PATH)/global.wqmignore`.
///
/// Mirrors `watching_queue::ignore_watch::run_reconciliation` and
/// `startup::reconciliation::ignore_sync`, so the watcher and the reconciler
/// always read the exact same file.
fn resolve_global_ignore_path() -> Option<PathBuf> {
    wqm_common::paths::get_database_path()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("global.wqmignore")))
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Build a `Gitignore` matcher from `global.wqmignore`.
///
/// Anchored at the filesystem root (`/`) rather than the file's own parent: the
/// watcher feeds ABSOLUTE paths from arbitrary project trees (e.g.
/// `/home/u/repos/<proj>/state/qdrant/...`) that do not live under the ignore
/// file's directory (`/var/lib/memexd/`). With root `/`, every absolute path
/// strips cleanly to a root-relative path so the `**/`-prefixed and
/// extension-glob patterns in `global.wqmignore` (see its header) match at any
/// depth — matching the reconciler's effective behaviour for those patterns
/// while staying robust for cross-tree absolute inputs.
fn build_matcher(path: &Path) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(Path::new("/"));
    if let Some(err) = builder.add(path) {
        warn!("[global_ignore] error reading {}: {}", path.display(), err);
    }
    match builder.build() {
        Ok(m) => Some(m),
        Err(e) => {
            warn!(
                "[global_ignore] failed to build matcher from {}: {}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Match `path` against the cached matcher, checking the path and every parent
/// directory so a directory pattern (e.g. `**/state/qdrant/`) excludes the
/// files nested beneath it.
fn matches(cached: &CachedMatcher, path: &Path, is_dir: bool) -> bool {
    cached
        .matcher
        .matched_path_or_any_parents(path, is_dir)
        .is_ignore()
}

/// Returns `true` when `path` is excluded by `global.wqmignore`.
///
/// Cheap on the hot path: a single `stat` of `global.wqmignore` detects edits;
/// the compiled matcher is rebuilt only when the mtime changes. Returns `false`
/// (do not exclude) when the file is absent or unreadable — callers keep their
/// other filters, so a missing global ignore never blocks indexing.
///
/// Pass `is_dir = true` only when `path` is known to be a directory; for a
/// regular file `false` is correct (parent directories are still consulted).
pub fn is_globally_ignored(path: &Path, is_dir: bool) -> bool {
    let Some(ignore_path) = resolve_global_ignore_path() else {
        return false;
    };
    let current_mtime = file_mtime(&ignore_path);

    // Fast path: a valid cached matcher for the same file + mtime.
    {
        let guard = GLOBAL_IGNORE.read().unwrap();
        if let Some(cached) = guard.as_ref() {
            if cached.path == ignore_path && cached.loaded_mtime == current_mtime {
                return matches(cached, path, is_dir);
            }
        }
    }

    // Slow path: (re)build under the write lock.
    let mut guard = GLOBAL_IGNORE.write().unwrap();
    // Re-check: another thread may have rebuilt between the locks.
    if let Some(cached) = guard.as_ref() {
        if cached.path == ignore_path && cached.loaded_mtime == current_mtime {
            return matches(cached, path, is_dir);
        }
    }

    let Some(matcher) = build_matcher(&ignore_path) else {
        // File missing/unreadable: drop any stale cache and do not exclude.
        *guard = None;
        return false;
    };

    let cached = CachedMatcher {
        matcher,
        loaded_mtime: current_mtime,
        path: ignore_path.clone(),
    };
    let result = matches(&cached, path, is_dir);
    debug!(
        "[global_ignore] (re)loaded matcher from {}",
        ignore_path.display()
    );
    *guard = Some(cached);
    result
}

/// Test-only hook: build a matcher from an explicit file and query a path,
/// bypassing the process-wide cache and the `WQM_DATABASE_PATH` resolution.
/// Lets tests validate matching semantics deterministically.
#[cfg(test)]
pub(crate) fn is_ignored_by_file(ignore_file: &Path, query: &Path, is_dir: bool) -> bool {
    match build_matcher(ignore_file) {
        Some(m) => m.matched_path_or_any_parents(query, is_dir).is_ignore(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    /// Write a global.wqmignore with the given body inside a temp "memexd" dir.
    fn write_global(body: &str) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("global.wqmignore");
        fs::write(&path, body).unwrap();
        (dir, path)
    }

    #[test]
    fn state_qdrant_segment_file_is_ignored() {
        // The real regression: a deep, ABSOLUTE path outside the ignore file's
        // own directory tree must still be excluded via the `**/` pattern.
        let (_d, ig) = write_global("**/state/qdrant/\nstate/\n");
        let p = Path::new(
            "/home/u/repos/workspace-qdrant-mcp/state/qdrant/storage/collections/projects/0/segments/abc/segment.json",
        );
        assert!(
            is_ignored_by_file(&ig, p, false),
            "state/qdrant segment file must be globally ignored"
        );
    }

    #[test]
    fn generated_protobuf_dart_is_ignored() {
        let (_d, ig) = write_global("**/*.pb.dart\n**/generated/\n");
        let p = Path::new("/home/u/repos/app/doc-frontend/packages/generated/lib/protos/shifts.pb.dart");
        assert!(is_ignored_by_file(&ig, p, false));
    }

    #[test]
    fn generated_dir_java_is_ignored() {
        let (_d, ig) = write_global("**/generated/\n**/proto/**/*.java\n");
        let p = Path::new("/home/u/repos/app/doc-backend/proto/src/generated/doc/ScheduleOuterClass.java");
        assert!(is_ignored_by_file(&ig, p, false));
    }

    #[test]
    fn ordinary_source_file_is_not_ignored() {
        let (_d, ig) = write_global("**/state/qdrant/\n**/*.pb.dart\n**/generated/\n");
        let p = Path::new("/home/u/repos/workspace-qdrant-mcp/src/rust/daemon/core/src/lib.rs");
        assert!(
            !is_ignored_by_file(&ig, p, false),
            "hand-authored source must NOT be globally ignored"
        );
    }

    #[test]
    fn missing_global_file_does_not_exclude() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("global.wqmignore");
        let p = Path::new("/home/u/repos/app/state/qdrant/segment.json");
        assert!(!is_ignored_by_file(&absent, p, false));
    }

    // ── exclusion-with-re-inclusion patterns (testlink / zabbix) ──────────────

    #[test]
    fn zabbix_vendored_tree_is_ignored() {
        let (_d, ig) = write_global("**/zabbix/zabbix/**\n");
        let p = Path::new("/home/u/repos/bws-engineer/zabbix/zabbix/Dockerfiles/web/conf.d/x.conf");
        assert!(is_ignored_by_file(&ig, p, false));
    }

    #[test]
    fn testlink_non_reincluded_subtree_is_ignored() {
        let (_d, ig) = write_global(
            "**/bws-dev-plataform/testlink/**\n!**/bws-dev-plataform/testlink/cfg/**\n",
        );
        let p = Path::new("/home/u/repos/x/bws-dev-plataform/testlink/lib/api/foo.php");
        assert!(
            is_ignored_by_file(&ig, p, false),
            "testlink files outside cfg/custom must stay excluded"
        );
    }

    // Characterises the GOTCHA precisely. Contents-only `!.../cfg/**`
    // re-includes the cfg FILE (matched_path_or_any_parents honours the `/**`
    // negation), but does NOT re-include the bare cfg DIRECTORY. A
    // directory-pruning walk (folder-scan / reconciler) tests the dir with
    // is_dir=true, finds it still ignored, and never descends — so the cfg
    // files are never reached during a scan. This is why cfg/custom showed 0
    // indexed files despite the `!.../cfg/**` re-include.
    #[test]
    fn testlink_cfg_contents_only_reincludes_file_but_not_dir() {
        let (_d, ig) = write_global(
            "**/bws-dev-plataform/testlink/**\n!**/bws-dev-plataform/testlink/cfg/**\n",
        );
        let cfg_file = Path::new("/home/u/x/bws-dev-plataform/testlink/cfg/const.inc.php");
        let cfg_dir = Path::new("/home/u/x/bws-dev-plataform/testlink/cfg");
        assert!(
            !is_ignored_by_file(&ig, cfg_file, false),
            "the cfg FILE is re-included by !.../cfg/**"
        );
        assert!(
            is_ignored_by_file(&ig, cfg_dir, true),
            "but the bare cfg DIR is NOT re-included — a walk prunes it before descending"
        );
    }

    // The fix: re-include the cfg DIRECTORY as well as its contents, so the walk
    // descends into cfg and indexes the re-included files. Both the dir
    // (is_dir=true) and the file (is_dir=false) must report not-ignored.
    #[test]
    fn testlink_cfg_reincluded_when_dir_and_contents_both_negated() {
        let (_d, ig) = write_global(
            "**/bws-dev-plataform/testlink/**\n\
             !**/bws-dev-plataform/testlink/cfg/\n\
             !**/bws-dev-plataform/testlink/cfg/**\n",
        );
        let cfg_file = Path::new("/home/u/x/bws-dev-plataform/testlink/cfg/const.inc.php");
        let cfg_dir = Path::new("/home/u/x/bws-dev-plataform/testlink/cfg");
        assert!(
            !is_ignored_by_file(&ig, cfg_dir, true),
            "dir-form !.../cfg/ re-includes the dir so the walk descends"
        );
        assert!(
            !is_ignored_by_file(&ig, cfg_file, false),
            "and the file stays re-included"
        );
    }

    // End-to-end through the PUBLIC `is_globally_ignored` — the exact function
    // the file-watcher enqueue path calls. Exercises WQM_DATABASE_PATH
    // resolution, the process-wide cache, and the matcher together, so it
    // regression-guards the watcher feedback-loop fix. `#[serial]` because it
    // mutates the WQM_DATABASE_PATH process env.
    #[serial_test::serial]
    #[test]
    fn public_api_resolves_global_ignore_via_env_and_excludes_state_qdrant() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("global.wqmignore"),
            "**/state/qdrant/\n**/*.pb.dart\n**/generated/\n",
        )
        .unwrap();
        // get_database_path() honours WQM_DATABASE_PATH; global.wqmignore is its sibling.
        let db_path = dir.path().join("memexd.db");
        std::env::set_var("WQM_DATABASE_PATH", &db_path);

        let ignored = Path::new(
            "/home/u/repos/workspace-qdrant-mcp/state/qdrant/storage/collections/projects/0/segments/abc/segment.json",
        );
        let source = Path::new("/home/u/repos/workspace-qdrant-mcp/src/rust/daemon/core/src/lib.rs");

        let ignored_hit = is_globally_ignored(ignored, false);
        let source_hit = is_globally_ignored(source, false);

        std::env::remove_var("WQM_DATABASE_PATH");

        assert!(
            ignored_hit,
            "watcher path must treat state/qdrant segment files as globally ignored"
        );
        assert!(!source_hit, "hand-authored source must not be filtered");
    }
}
