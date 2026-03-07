//! Editor-managed LSP server discovery.
//!
//! Scans well-known editor installation directories (mason.nvim, VS Code,
//! Zed, Emacs lsp-mode, etc.) to find LSP servers not on `$PATH`.

use std::path::{Path, PathBuf};

/// Source of an LSP server detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionSource {
    /// Found via system PATH
    Path,
    /// Found in an editor-managed directory
    Editor(String),
}

impl std::fmt::Display for DetectionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionSource::Path => write!(f, "PATH"),
            DetectionSource::Editor(name) => write!(f, "{name}"),
        }
    }
}

/// Result of searching for an LSP binary across PATH and editor locations.
#[derive(Debug, Clone)]
pub struct LspDetectionResult {
    /// Full path to the found binary.
    pub path: PathBuf,
    /// Where the binary was found.
    pub source: DetectionSource,
}

/// An editor-managed search location.
#[derive(Debug, Clone)]
struct EditorSearchPath {
    /// Human-readable editor name (e.g., "mason.nvim").
    name: &'static str,
    /// Base directory relative to home.
    base_dir: &'static str,
    /// Whether to search subdirectories recursively.
    recursive: bool,
    /// Maximum recursion depth (only if recursive).
    max_depth: usize,
}

/// Get the list of editor-managed search paths.
///
/// All paths are relative to the user's home directory.
fn editor_search_paths() -> Vec<EditorSearchPath> {
    vec![
        EditorSearchPath {
            name: "mason.nvim",
            base_dir: ".local/share/nvim/mason/bin",
            recursive: false,
            max_depth: 0,
        },
        EditorSearchPath {
            name: "VS Code",
            base_dir: ".vscode/extensions",
            recursive: true,
            max_depth: 3,
        },
        EditorSearchPath {
            name: "VS Code Insiders",
            base_dir: ".vscode-insiders/extensions",
            recursive: true,
            max_depth: 3,
        },
        EditorSearchPath {
            name: "Zed",
            base_dir: ".local/share/zed/languages",
            recursive: true,
            max_depth: 3,
        },
        EditorSearchPath {
            name: "Emacs lsp-mode",
            base_dir: ".emacs.d/.cache/lsp",
            recursive: true,
            max_depth: 3,
        },
        EditorSearchPath {
            name: "Emacs eglot",
            base_dir: ".emacs.d/eglot",
            recursive: true,
            max_depth: 2,
        },
        EditorSearchPath {
            name: "Helix",
            base_dir: ".config/helix/runtime",
            recursive: true,
            max_depth: 2,
        },
        #[cfg(target_os = "macos")]
        EditorSearchPath {
            name: "VS Code (macOS)",
            base_dir: "Library/Application Support/Code/User/globalStorage",
            recursive: true,
            max_depth: 3,
        },
    ]
}

/// Find an LSP binary by name, searching PATH first then editor locations.
///
/// Returns the first match with its detection source, or `None`.
pub fn find_lsp_binary(binary: &str) -> Option<LspDetectionResult> {
    // 1. Check PATH first
    if let Ok(path) = which::which(binary) {
        return Some(LspDetectionResult {
            path,
            source: DetectionSource::Path,
        });
    }

    // 2. Search editor-managed locations
    let home = dirs::home_dir()?;
    find_in_editor_paths(&home, binary)
}

/// Search all editor-managed locations for a binary.
fn find_in_editor_paths(home: &Path, binary: &str) -> Option<LspDetectionResult> {
    for search_path in editor_search_paths() {
        let base = home.join(search_path.base_dir);
        if !base.exists() {
            continue;
        }

        if let Some(path) = find_binary_in_dir(&base, binary, search_path.recursive, search_path.max_depth) {
            return Some(LspDetectionResult {
                path,
                source: DetectionSource::Editor(search_path.name.to_string()),
            });
        }
    }
    None
}

/// Search a directory (optionally recursively) for an executable binary.
fn find_binary_in_dir(
    dir: &Path,
    binary: &str,
    recursive: bool,
    max_depth: usize,
) -> Option<PathBuf> {
    if !recursive {
        // Direct lookup only
        let candidate = dir.join(binary);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
        return None;
    }

    // Recursive search with depth limit
    for entry in walkdir::WalkDir::new(dir)
        .max_depth(max_depth)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name().to_string_lossy() == binary && is_executable_file(entry.path()) {
            return Some(entry.path().to_path_buf());
        }
    }
    None
}

/// Check if a path is an executable file.
fn is_executable_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = path.metadata() {
            return (meta.permissions().mode() & 0o111) != 0;
        }
        false
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Find all LSP binaries from a list of binary names.
///
/// Returns results for each found binary with its detection source.
pub fn find_all_lsp_binaries(binaries: &[&str]) -> Vec<(String, LspDetectionResult)> {
    let mut results = Vec::new();
    for binary in binaries {
        if let Some(result) = find_lsp_binary(binary) {
            results.push((binary.to_string(), result));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_search_paths_non_empty() {
        let paths = editor_search_paths();
        assert!(paths.len() >= 5);
    }

    #[test]
    fn test_find_lsp_binary_on_path() {
        // `ls` should always be on PATH
        let result = find_lsp_binary("ls");
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.source, DetectionSource::Path);
    }

    #[test]
    fn test_find_lsp_binary_not_found() {
        let result = find_lsp_binary("definitely-not-an-lsp-server-xyz");
        assert!(result.is_none());
    }

    #[test]
    fn test_detection_source_display() {
        assert_eq!(DetectionSource::Path.to_string(), "PATH");
        assert_eq!(
            DetectionSource::Editor("mason.nvim".to_string()).to_string(),
            "mason.nvim"
        );
    }

    #[test]
    fn test_is_executable_file() {
        // /bin/ls should be executable
        let ls = PathBuf::from("/bin/ls");
        if ls.exists() {
            assert!(is_executable_file(&ls));
        }
        // A non-existent path should not
        assert!(!is_executable_file(Path::new("/nonexistent/binary")));
    }

    #[test]
    fn test_find_binary_in_dir_non_recursive() {
        // Find a common binary in a known location
        let (dir, name) = if Path::new("/bin/ls").exists() {
            ("/bin", "ls")
        } else if Path::new("/usr/bin/ls").exists() {
            ("/usr/bin", "ls")
        } else {
            // Skip on unusual systems
            return;
        };
        let result = find_binary_in_dir(Path::new(dir), name, false, 0);
        assert!(result.is_some());
    }

    #[test]
    fn test_find_binary_in_nonexistent_dir() {
        let result = find_binary_in_dir(Path::new("/nonexistent"), "ls", false, 0);
        assert!(result.is_none());
    }
}
