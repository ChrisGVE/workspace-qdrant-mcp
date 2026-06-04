//! Tarball extraction and `src/` directory discovery for downloaded grammars.

use std::path::{Component, Path, PathBuf};

use super::{DownloadError, DownloadResult};

/// Join an untrusted tar-header `entry_path` under `dest`, stripping any
/// component that could escape `dest` (`..`, absolute roots, drive prefixes,
/// `.`). Returns `None` when nothing safe remains so the caller skips the entry.
///
/// Defends against tar path traversal / zip-slip (CWE-22): a malicious or
/// MITM'd release tarball can carry entries like `../../../../etc/cron.d/evil`,
/// and `dest.join(path)` would resolve above `dest` without error. Filtering to
/// `Normal` components neutralizes both `..` sequences and absolute paths; the
/// `starts_with(dest)` check is a final guard. (Symlink entries are never
/// unpacked — `extract_tarball` only handles dir/file types — so symlink-based
/// traversal is not reachable here.)
fn sanitized_dest(dest: &Path, entry_path: &Path) -> Option<PathBuf> {
    let mut rel = PathBuf::new();
    for comp in entry_path.components() {
        match comp {
            Component::Normal(c) => rel.push(c),
            Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_)
            | Component::CurDir => {}
        }
    }
    if rel.as_os_str().is_empty() {
        return None;
    }
    let full = dest.join(&rel);
    full.starts_with(dest).then_some(full)
}

/// Extract a gzipped tarball into a directory.
///
/// Uses entry-by-entry extraction to handle tarballs that lack directory
/// entries (common in GitHub release tarballs).
pub(super) fn extract_tarball(bytes: &[u8], dest: &Path) -> DownloadResult<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let decoder = GzDecoder::new(std::io::Cursor::new(bytes));
    let mut archive = Archive::new(decoder);
    // Some release tarballs omit directory entries, so we extract
    // entry-by-entry and create parent directories as needed.
    for entry in archive
        .entries()
        .map_err(|e| DownloadError::ExtractionFailed(format!("Failed to read tarball: {}", e)))?
    {
        let mut entry = entry
            .map_err(|e| DownloadError::ExtractionFailed(format!("Failed to read entry: {}", e)))?;
        let path = entry
            .path()
            .map_err(|e| DownloadError::ExtractionFailed(format!("Invalid entry path: {}", e)))?;
        // Sanitize the untrusted header path; skip entries that would escape dest.
        let full_path = match sanitized_dest(dest, &path) {
            Some(p) => p,
            None => continue,
        };

        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&full_path).ok();
        } else if entry.header().entry_type().is_file() {
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    DownloadError::ExtractionFailed(format!(
                        "Failed to create directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
            entry.unpack(&full_path).map_err(|e| {
                DownloadError::ExtractionFailed(format!(
                    "Failed to unpack {}: {}",
                    full_path.display(),
                    e
                ))
            })?;
        }
    }
    Ok(())
}

/// Find the `src/` directory containing `parser.c` within an extracted tarball.
///
/// Handles multiple tarball formats:
/// - Files at root with `./src/parser.c` (release tarballs, e.g. tree-sitter-rust)
/// - Single top-level directory (archive tarballs, e.g. `tree-sitter-rust-main/src/parser.c`)
/// - Grammar in a subdirectory (monorepos, e.g. `typescript/src/parser.c`)
pub(super) fn find_src_dir(extract_dir: &Path, subdir: Option<&str>) -> DownloadResult<PathBuf> {
    // Strategy 1: Check if src/parser.c exists directly in extract_dir
    // (release tarballs often extract files directly)
    let direct_src = if let Some(sub) = subdir {
        extract_dir.join(sub).join("src")
    } else {
        extract_dir.join("src")
    };
    if direct_src.join("parser.c").exists() {
        return Ok(direct_src);
    }

    // Strategy 2: Look for a single top-level directory (archive tarballs)
    let entries: Vec<_> = std::fs::read_dir(extract_dir)
        .map_err(|e| DownloadError::ExtractionFailed(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    if entries.len() == 1 {
        let top_dir = entries[0].path();
        let grammar_root = if let Some(sub) = subdir {
            top_dir.join(sub)
        } else {
            top_dir
        };

        let src_dir = grammar_root.join("src");
        if src_dir.join("parser.c").exists() {
            return Ok(src_dir);
        }
        if grammar_root.join("parser.c").exists() {
            return Ok(grammar_root);
        }
    }

    // Strategy 3: Search recursively for parser.c (last resort)
    for e in walkdir::WalkDir::new(extract_dir)
        .max_depth(4)
        .into_iter()
        .flatten()
    {
        if e.file_name() == "parser.c" {
            if let Some(parent) = e.path().parent() {
                return Ok(parent.to_path_buf());
            }
        }
    }

    Err(DownloadError::ExtractionFailed(format!(
        "Could not find parser.c in extracted archive at {}",
        extract_dir.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_dest_keeps_normal_paths_inside_dest() {
        let dest = Path::new("/tmp/extract");
        let got = sanitized_dest(dest, Path::new("pkg/src/parser.c")).unwrap();
        assert_eq!(got, Path::new("/tmp/extract/pkg/src/parser.c"));
        assert!(got.starts_with(dest));
    }

    #[test]
    fn sanitized_dest_strips_parent_dir_escapes() {
        let dest = Path::new("/tmp/extract");
        // Classic zip-slip payload must not escape dest.
        let got = sanitized_dest(dest, Path::new("../../../../etc/cron.d/evil")).unwrap();
        assert_eq!(got, Path::new("/tmp/extract/etc/cron.d/evil"));
        assert!(got.starts_with(dest));
    }

    #[test]
    fn sanitized_dest_strips_absolute_root() {
        let dest = Path::new("/tmp/extract");
        let got = sanitized_dest(dest, Path::new("/etc/passwd")).unwrap();
        assert_eq!(got, Path::new("/tmp/extract/etc/passwd"));
        assert!(got.starts_with(dest));
    }

    #[test]
    fn sanitized_dest_strips_interior_parent_refs() {
        let dest = Path::new("/tmp/extract");
        // Interleaved `..` must not climb above dest.
        let got = sanitized_dest(dest, Path::new("a/../../../b")).unwrap();
        assert_eq!(got, Path::new("/tmp/extract/a/b"));
        assert!(got.starts_with(dest));
    }

    #[test]
    fn sanitized_dest_skips_empty_after_strip() {
        let dest = Path::new("/tmp/extract");
        assert!(sanitized_dest(dest, Path::new("../..")).is_none());
        assert!(sanitized_dest(dest, Path::new("/")).is_none());
        assert!(sanitized_dest(dest, Path::new(".")).is_none());
    }
}
