//! File Type Allowlist (Task 511)
//!
//! Provides an allowlist of file extensions that the daemon will process.
//! Files whose extensions are not in the allowlist are silently skipped,
//! preventing the ingestion of binary files, media, caches, and other
//! non-textual content that caused the home directory contamination incident.

use std::collections::HashSet;
use std::path::Path;

/// Allowlist of file extensions for project and library ingestion.
///
/// The daemon will only process files whose extension (case-insensitive)
/// matches an entry in the appropriate set. Extension-less files are
/// rejected by default.
#[derive(Debug, Clone)]
pub struct AllowedExtensions {
    /// Extensions allowed for project collections (source code, config, docs).
    project_extensions: HashSet<String>,
    /// Extensions allowed for library collections (documents, reference material).
    library_extensions: HashSet<String>,
}

impl Default for AllowedExtensions {
    fn default() -> Self {
        let project_extensions: HashSet<String> = [
            // Rust
            ".rs",
            // Python
            ".py",
            // JavaScript / TypeScript
            ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
            // Go
            ".go",
            // Java / JVM
            ".java", ".kt", ".scala", ".groovy", ".clj", ".cljs",
            // C / C++
            ".c", ".cpp", ".h", ".hpp",
            // Swift
            ".swift",
            // Ruby
            ".rb",
            // Lua
            ".lua",
            // Shell
            ".sh", ".bash", ".zsh", ".fish",
            // Config / Data
            ".toml", ".yaml", ".yml", ".json", ".xml",
            // Web
            ".html", ".css", ".scss", ".less", ".vue", ".svelte", ".astro",
            // SQL / GraphQL / Proto
            ".sql", ".graphql", ".proto",
            // Documentation
            ".md", ".txt", ".rst", ".tex",
            // Elixir / Erlang
            ".ex", ".exs", ".erl", ".hrl",
            // Haskell / ML / Elm
            ".hs", ".ml", ".mli", ".elm",
            // R
            ".r", ".R",  // note: kept separate for case-insensitive matching
            // Dart
            ".dart",
            // .NET
            ".cs", ".fs", ".vb",
            // Perl / PHP
            ".pl", ".pm", ".php",
            // Nix
            ".nix",
            // Lean
            ".lean",
            // Zig
            ".zig",
            // Nim
            ".nim",
            // V / Odin / D
            ".v", ".odin", ".d",
            // Fortran
            ".f90", ".f95",
            // Pascal
            ".pas",
            // COBOL
            ".cob", ".cbl",
            // Build / CI files (by extension)
            ".dockerfile", ".makefile", ".cmake", ".mk",
            // PowerShell / Batch
            ".ps1", ".bat", ".cmd",
            // Text processing
            ".awk", ".sed",
            // Build tool configs
            ".sbt", ".gradle", ".pom",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let library_extensions: HashSet<String> = [
            // Documents
            ".pdf", ".epub", ".docx", ".pptx", ".ppt", ".pages", ".key",
            ".odt", ".odp", ".ods", ".rtf", ".doc",
            // Text / Markup
            ".md", ".txt", ".html", ".htm",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            project_extensions,
            library_extensions,
        }
    }
}

impl AllowedExtensions {
    /// Check whether a file is allowed for ingestion into the given collection.
    ///
    /// Returns `true` when the file's extension (case-insensitive) is present
    /// in the allowlist for the target collection. Extension-less files are
    /// always rejected.
    ///
    /// # Arguments
    /// * `file_path` - Absolute or relative path to the file.
    /// * `collection` - Target collection name (`"libraries"` or anything else
    ///   which falls back to the project allowlist).
    pub fn is_allowed(&self, file_path: &str, collection: &str) -> bool {
        let path = Path::new(file_path);

        // Extract extension; reject extension-less files
        let ext = match path.extension() {
            Some(ext) => ext.to_string_lossy().to_lowercase(),
            None => return false,
        };

        // Prepend dot for lookup
        let dotted = format!(".{}", ext);

        if collection == "libraries" {
            self.library_extensions.contains(&dotted)
        } else {
            self.project_extensions.contains(&dotted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_has_common_project_extensions() {
        let ae = AllowedExtensions::default();
        for ext in &[".rs", ".py", ".js", ".ts", ".go", ".java", ".c", ".cpp", ".md", ".toml", ".yaml"] {
            assert!(
                ae.project_extensions.contains(*ext),
                "Expected project extension {} to be present",
                ext
            );
        }
    }

    #[test]
    fn test_default_has_common_library_extensions() {
        let ae = AllowedExtensions::default();
        for ext in &[".pdf", ".epub", ".docx", ".md", ".txt", ".html"] {
            assert!(
                ae.library_extensions.contains(*ext),
                "Expected library extension {} to be present",
                ext
            );
        }
    }

    #[test]
    fn test_is_allowed_project_source_files() {
        let ae = AllowedExtensions::default();
        assert!(ae.is_allowed("/home/user/project/src/main.rs", "projects"));
        assert!(ae.is_allowed("/home/user/project/lib.py", "projects"));
        assert!(ae.is_allowed("/home/user/project/index.ts", "projects"));
        assert!(ae.is_allowed("README.md", "projects"));
    }

    #[test]
    fn test_is_allowed_library_documents() {
        let ae = AllowedExtensions::default();
        assert!(ae.is_allowed("/docs/manual.pdf", "libraries"));
        assert!(ae.is_allowed("/docs/book.epub", "libraries"));
        assert!(ae.is_allowed("/docs/notes.md", "libraries"));
        assert!(ae.is_allowed("/docs/report.docx", "libraries"));
    }

    #[test]
    fn test_rejects_binary_and_media_files() {
        let ae = AllowedExtensions::default();
        // These should not be in either allowlist
        assert!(!ae.is_allowed("image.png", "projects"));
        assert!(!ae.is_allowed("photo.jpg", "projects"));
        assert!(!ae.is_allowed("video.mp4", "projects"));
        assert!(!ae.is_allowed("archive.zip", "projects"));
        assert!(!ae.is_allowed("binary.exe", "projects"));
        assert!(!ae.is_allowed("data.sqlite", "projects"));
        assert!(!ae.is_allowed("model.onnx", "projects"));
    }

    #[test]
    fn test_rejects_extension_less_files() {
        let ae = AllowedExtensions::default();
        assert!(!ae.is_allowed("Makefile", "projects"));
        assert!(!ae.is_allowed("Dockerfile", "projects"));
        assert!(!ae.is_allowed("LICENSE", "projects"));
        assert!(!ae.is_allowed("/home/user/.bashrc", "projects"));
    }

    #[test]
    fn test_case_insensitive_matching() {
        let ae = AllowedExtensions::default();
        assert!(ae.is_allowed("file.RS", "projects"));
        assert!(ae.is_allowed("file.Py", "projects"));
        assert!(ae.is_allowed("file.PDF", "libraries"));
        assert!(ae.is_allowed("FILE.Html", "libraries"));
    }

    #[test]
    fn test_library_collection_uses_library_set() {
        let ae = AllowedExtensions::default();
        // .pdf is in library but not project
        assert!(ae.is_allowed("doc.pdf", "libraries"));
        assert!(!ae.is_allowed("doc.pdf", "projects"));
    }

    #[test]
    fn test_project_collection_uses_project_set() {
        let ae = AllowedExtensions::default();
        // .rs is in project but not library
        assert!(ae.is_allowed("main.rs", "projects"));
        assert!(!ae.is_allowed("main.rs", "libraries"));
    }

    #[test]
    fn test_unknown_collection_falls_back_to_project() {
        let ae = AllowedExtensions::default();
        // Any collection name other than "libraries" uses project set
        assert!(ae.is_allowed("main.rs", "some_custom_collection"));
        assert!(!ae.is_allowed("doc.pdf", "some_custom_collection"));
    }

    #[test]
    fn test_empty_path() {
        let ae = AllowedExtensions::default();
        assert!(!ae.is_allowed("", "projects"));
    }

    #[test]
    fn test_dot_only_extension() {
        let ae = AllowedExtensions::default();
        // A file like "file." has an empty extension
        assert!(!ae.is_allowed("file.", "projects"));
    }

    #[test]
    fn test_r_case_sensitivity() {
        let ae = AllowedExtensions::default();
        // Both .r and .R should work via case-insensitive matching
        assert!(ae.is_allowed("analysis.r", "projects"));
        assert!(ae.is_allowed("analysis.R", "projects"));
    }

    #[test]
    fn test_shared_extensions_between_project_and_library() {
        let ae = AllowedExtensions::default();
        // .md and .txt are in both sets
        assert!(ae.is_allowed("notes.md", "projects"));
        assert!(ae.is_allowed("notes.md", "libraries"));
        assert!(ae.is_allowed("readme.txt", "projects"));
        assert!(ae.is_allowed("readme.txt", "libraries"));
    }

    #[test]
    fn test_path_with_dots_in_directory() {
        let ae = AllowedExtensions::default();
        // Directories with dots should not confuse extension extraction
        assert!(ae.is_allowed("/home/user/my.project/src/main.rs", "projects"));
        assert!(!ae.is_allowed("/home/user/my.project/src/data.bin", "projects"));
    }
}
