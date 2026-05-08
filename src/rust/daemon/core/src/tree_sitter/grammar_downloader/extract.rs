//! Tarball extraction and `src/` directory discovery for downloaded grammars.

use std::path::{Path, PathBuf};

use super::{DownloadError, DownloadResult};

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
        let full_path = dest.join(&*path);

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
