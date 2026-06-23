//! F0 structural guard (AC-F0.1): the four relocated shared nexuses are defined
//! exactly once, in `wqm-common`, and the old daemon-core homes no longer define
//! them (they re-export). This is the FP-2 / DR GP-9 invariant — one canonical
//! home, no duplicate that could silently drift.
//!
//! Location: `wqm-common/tests/f0_nexus_single_home.rs`. The test walks the
//! workspace Rust source and counts `pub`-item definition sites by exact prefix.
//!
//! Note on same-named, UNRELATED types (intentionally NOT nexuses): `SearchResult`
//! also names a distinct client model (`client/src/models.rs`), and `FileChange`
//! also names a distinct FTS batch-apply payload (`fts_batch_processor`). Those
//! are separate types under separate module paths, so the guard asserts "exactly
//! one definition UNDER common/src + zero at the old daemon home" rather than a
//! naive tree-wide count of one.

use std::path::{Path, PathBuf};

/// Walk up from this crate's manifest dir to the workspace source root (the dir
/// containing both `common` and `daemon` members). Returns None outside the repo.
fn workspace_root() -> Option<PathBuf> {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if root.join("common").is_dir() && root.join("daemon").is_dir() {
            return Some(root);
        }
        if !root.pop() {
            return None;
        }
    }
}

/// Count lines whose trimmed start equals `pat` across every `.rs` file under
/// `dir` (skipping `target/` and dotted dirs).
fn count_defs(dir: &Path, pat: &str) -> usize {
    fn walk(dir: &Path, pat: &str, acc: &mut usize) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                walk(&p, pat, acc);
            } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(src) = std::fs::read_to_string(&p) {
                    *acc += src
                        .lines()
                        .filter(|l| l.trim_start().starts_with(pat))
                        .count();
                }
            }
        }
    }
    let mut acc = 0;
    walk(dir, pat, &mut acc);
    acc
}

#[test]
fn f0_nexuses_have_single_canonical_home_in_wqm_common() {
    let Some(root) = workspace_root() else {
        return; // not running in-repo (e.g. packaged crate) — guard no-ops.
    };
    let common = root.join("common").join("src");
    let daemon_storage = root.join("daemon").join("core").join("src").join("storage");
    let daemon_git = root.join("daemon").join("core").join("src").join("git");

    // (pattern, old daemon home that must NOT redefine it)
    let cases: &[(&str, &Path)] = &[
        ("pub enum StorageError", &daemon_storage),
        ("pub struct SearchResult", &daemon_storage),
        ("pub fn rrf_merge", &daemon_storage),
        ("pub struct FileChange", &daemon_git),
        ("pub enum FileChangeStatus", &daemon_git),
    ];

    for (pat, old_home) in cases {
        let in_common = count_defs(&common, pat);
        assert_eq!(
            in_common, 1,
            "nexus `{pat}` must be defined exactly once under wqm-common/src (found {in_common})"
        );
        let in_old = count_defs(old_home, pat);
        assert_eq!(
            in_old,
            0,
            "nexus `{pat}` must NOT be redefined at its old daemon home {} (found {in_old}); \
             daemon-core must re-export from wqm-common (FP-2)",
            old_home.display()
        );
    }
}
