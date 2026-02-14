//! File Type Allowlist
//!
//! Provides a two-tier allowlist of file extensions that the daemon will process.
//! The library allowlist is a superset of the project allowlist, so reference
//! material containing code examples can be fully processed. Files whose
//! extensions are not in the appropriate allowlist are silently skipped.

use std::collections::HashSet;
use std::path::Path;

/// Routing decision for a file based on its extension and source context.
///
/// Determines which Qdrant collection a file should be ingested into:
/// - `ProjectCollection` for source code and project config files
/// - `LibraryCollection` for reference/document formats (even when found in project folders)
/// - `Excluded` for file types not in any allowlist
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileRoute {
    /// File routes to the `projects` collection.
    ProjectCollection,
    /// File routes to the `libraries` collection.
    /// `source_project_id` is set when a library-format file is found inside a project folder,
    /// allowing the library entry to be associated back to its originating project.
    LibraryCollection { source_project_id: Option<String> },
    /// File is not in any allowlist and should be skipped.
    Excluded,
}

/// Extensions for binary/reference formats that route to the `libraries` collection
/// even when discovered inside a project folder.
///
/// These are document formats (PDF, EPUB, etc.) that are unlikely to be "source code"
/// and are better served by the library ingestion pipeline. Source-like formats
/// (e.g., `.md`, `.txt`, `.html`) stay in `projects` because they are typically
/// project documentation meant to be searched alongside code.
const LIBRARY_ROUTED_EXTENSIONS: &[&str] = &[
    ".pdf", ".epub", ".docx", ".doc", ".rtf", ".odt",
    ".mobi",
    ".pptx", ".ppt", ".pages", ".key", ".odp",
    ".xlsx", ".xls", ".ods", ".parquet",
];

/// Two-tier allowlist of file extensions for project and library ingestion.
///
/// The library set is a superset of the project set: `library_extensions ⊇ project_extensions`.
/// This allows reference material (books, papers, documentation) containing code examples
/// to be fully processed when ingested into the libraries collection.
///
/// Extension-less files are always rejected.
#[derive(Debug, Clone)]
pub struct AllowedExtensions {
    /// Extensions allowed for project collections (source code, config, docs).
    project_extensions: HashSet<String>,
    /// Extensions allowed for library collections (superset of project_extensions
    /// plus document/reference formats like .pdf, .epub, .docx, etc.).
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
            // Spreadsheets and data
            ".csv", ".tsv",
            // Notebooks
            ".ipynb",
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

        // Library-only extensions: document/reference formats not in project set.
        // library_extensions = project_extensions ∪ library_only_extensions
        let library_only: HashSet<String> = [
            // Documents
            ".pdf", ".epub", ".docx", ".doc", ".rtf", ".odt",
            // Ebooks
            ".mobi",
            // Presentations
            ".pptx", ".ppt", ".pages", ".key", ".odp",
            // Spreadsheets (formats not already in project_extensions)
            ".xlsx", ".xls", ".ods", ".parquet",
            // Web (variant not in project set)
            ".htm",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let mut library_extensions = project_extensions.clone();
        library_extensions.extend(library_only);

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

    /// Route a file to the appropriate Qdrant collection based on its extension
    /// and the watch folder's configured collection.
    ///
    /// # Routing logic
    ///
    /// 1. **Library watch folders** (`watch_collection == "libraries"`):
    ///    Files with extensions in the library allowlist route to `LibraryCollection`.
    ///    All others are `Excluded`.
    ///
    /// 2. **Project watch folders** (`watch_collection == "projects"`):
    ///    - If the extension is in `LIBRARY_ROUTED_EXTENSIONS` (binary document formats
    ///      like `.pdf`, `.docx`, `.epub`), the file routes to `LibraryCollection` with
    ///      `source_project_id` set to the project's tenant_id, so the library entry
    ///      can be traced back to its origin project.
    ///    - If the extension is in the project allowlist, it routes to `ProjectCollection`.
    ///    - Otherwise, the file is `Excluded`.
    ///
    /// # Arguments
    /// * `file_path` - Path to the file being routed.
    /// * `watch_collection` - The collection configured on the watch folder (`"projects"` or `"libraries"`).
    /// * `tenant_id` - The tenant identifier (project ID or library name) for the watch folder.
    pub fn route_file(&self, file_path: &str, watch_collection: &str, tenant_id: &str) -> FileRoute {
        let path = Path::new(file_path);

        // Extract extension; reject extension-less files
        let ext = match path.extension() {
            Some(ext) => ext.to_string_lossy().to_lowercase(),
            None => return FileRoute::Excluded,
        };

        let dotted = format!(".{}", ext);

        if watch_collection == "libraries" {
            // Library watch folder: accept any library-allowed extension
            if self.library_extensions.contains(&dotted) {
                FileRoute::LibraryCollection { source_project_id: None }
            } else {
                FileRoute::Excluded
            }
        } else {
            // Project watch folder: check for library-routed override first
            if LIBRARY_ROUTED_EXTENSIONS.contains(&dotted.as_str()) {
                FileRoute::LibraryCollection {
                    source_project_id: Some(tenant_id.to_string()),
                }
            } else if self.project_extensions.contains(&dotted) {
                FileRoute::ProjectCollection
            } else {
                FileRoute::Excluded
            }
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
        // Library-only document formats
        for ext in &[".pdf", ".epub", ".docx", ".mobi", ".parquet"] {
            assert!(
                ae.library_extensions.contains(*ext),
                "Expected library-only extension {} to be present",
                ext
            );
        }
        // Project extensions should also be present (superset)
        for ext in &[".rs", ".py", ".js", ".md", ".txt", ".html"] {
            assert!(
                ae.library_extensions.contains(*ext),
                "Expected project extension {} to also be in library set",
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
    fn test_project_extensions_allowed_in_libraries() {
        let ae = AllowedExtensions::default();
        // Project extensions like .rs are now also in library set (superset)
        assert!(ae.is_allowed("main.rs", "projects"));
        assert!(ae.is_allowed("main.rs", "libraries"));
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

    #[test]
    fn test_library_extensions_is_superset_of_project() {
        let ae = AllowedExtensions::default();
        // Every project extension must also be in library extensions
        for ext in &ae.project_extensions {
            assert!(
                ae.library_extensions.contains(ext),
                "Project extension {} missing from library set (superset violation)",
                ext
            );
        }
    }

    #[test]
    fn test_library_only_extensions_rejected_for_projects() {
        let ae = AllowedExtensions::default();
        // Document/reference formats should not be allowed in projects
        for path in &["doc.pdf", "book.epub", "report.docx",
                       "novel.mobi", "slides.pptx", "data.parquet", "budget.xlsx"] {
            assert!(
                !ae.is_allowed(path, "projects"),
                "Library-only file {} should be rejected for projects",
                path
            );
            assert!(
                ae.is_allowed(path, "libraries"),
                "Library-only file {} should be accepted for libraries",
                path
            );
        }
    }

    #[test]
    fn test_case_insensitive_library_only_extensions() {
        let ae = AllowedExtensions::default();
        assert!(ae.is_allowed("doc.PDF", "libraries"));
        assert!(ae.is_allowed("book.EPUB", "libraries"));
        assert!(ae.is_allowed("report.DOCX", "libraries"));
    }

    // --- FileRoute / route_file() tests ---

    #[test]
    fn test_route_source_file_in_project() {
        let ae = AllowedExtensions::default();
        assert_eq!(
            ae.route_file("/project/src/main.rs", "projects", "my-project"),
            FileRoute::ProjectCollection
        );
        assert_eq!(
            ae.route_file("lib.py", "projects", "my-project"),
            FileRoute::ProjectCollection
        );
        assert_eq!(
            ae.route_file("index.ts", "projects", "my-project"),
            FileRoute::ProjectCollection
        );
    }

    #[test]
    fn test_route_pdf_in_project_goes_to_library() {
        let ae = AllowedExtensions::default();
        let route = ae.route_file("/project/docs/manual.pdf", "projects", "my-project");
        assert_eq!(
            route,
            FileRoute::LibraryCollection {
                source_project_id: Some("my-project".to_string())
            }
        );
    }

    #[test]
    fn test_route_docx_in_project_goes_to_library() {
        let ae = AllowedExtensions::default();
        let route = ae.route_file("report.docx", "projects", "my-project");
        assert_eq!(
            route,
            FileRoute::LibraryCollection {
                source_project_id: Some("my-project".to_string())
            }
        );
    }

    #[test]
    fn test_route_all_library_routed_extensions_in_project() {
        let ae = AllowedExtensions::default();
        for ext in LIBRARY_ROUTED_EXTENSIONS {
            let filename = format!("file{}", ext);
            let route = ae.route_file(&filename, "projects", "proj-1");
            assert_eq!(
                route,
                FileRoute::LibraryCollection {
                    source_project_id: Some("proj-1".to_string())
                },
                "Extension {} in project should route to LibraryCollection",
                ext
            );
        }
    }

    #[test]
    fn test_route_source_file_in_library() {
        let ae = AllowedExtensions::default();
        // .rs is in the library set (superset), so it's allowed in library folders
        assert_eq!(
            ae.route_file("main.rs", "libraries", "my-lib"),
            FileRoute::LibraryCollection { source_project_id: None }
        );
        assert_eq!(
            ae.route_file("example.py", "libraries", "my-lib"),
            FileRoute::LibraryCollection { source_project_id: None }
        );
    }

    #[test]
    fn test_route_pdf_in_library() {
        let ae = AllowedExtensions::default();
        assert_eq!(
            ae.route_file("book.pdf", "libraries", "my-lib"),
            FileRoute::LibraryCollection { source_project_id: None }
        );
    }

    #[test]
    fn test_route_binary_file_excluded() {
        let ae = AllowedExtensions::default();
        assert_eq!(ae.route_file("image.png", "projects", "proj"), FileRoute::Excluded);
        assert_eq!(ae.route_file("photo.jpg", "libraries", "lib"), FileRoute::Excluded);
        assert_eq!(ae.route_file("archive.zip", "projects", "proj"), FileRoute::Excluded);
    }

    #[test]
    fn test_route_extensionless_excluded() {
        let ae = AllowedExtensions::default();
        assert_eq!(ae.route_file("Makefile", "projects", "proj"), FileRoute::Excluded);
        assert_eq!(ae.route_file("Dockerfile", "libraries", "lib"), FileRoute::Excluded);
    }

    #[test]
    fn test_route_case_insensitive() {
        let ae = AllowedExtensions::default();
        assert_eq!(
            ae.route_file("file.RS", "projects", "proj"),
            FileRoute::ProjectCollection
        );
        assert_eq!(
            ae.route_file("doc.PDF", "projects", "proj"),
            FileRoute::LibraryCollection {
                source_project_id: Some("proj".to_string())
            }
        );
        assert_eq!(
            ae.route_file("doc.PDF", "libraries", "lib"),
            FileRoute::LibraryCollection { source_project_id: None }
        );
    }

    #[test]
    fn test_route_md_stays_in_project() {
        // .md is a project extension, not in LIBRARY_ROUTED_EXTENSIONS,
        // so it stays in the project collection when found in project folders
        let ae = AllowedExtensions::default();
        assert_eq!(
            ae.route_file("README.md", "projects", "proj"),
            FileRoute::ProjectCollection
        );
    }

    #[test]
    fn test_route_empty_path() {
        let ae = AllowedExtensions::default();
        assert_eq!(ae.route_file("", "projects", "proj"), FileRoute::Excluded);
    }

    #[test]
    fn test_route_unknown_collection_uses_project_logic() {
        let ae = AllowedExtensions::default();
        // Non-"libraries" collections fall through to project logic
        assert_eq!(
            ae.route_file("main.rs", "custom", "proj"),
            FileRoute::ProjectCollection
        );
        assert_eq!(
            ae.route_file("doc.pdf", "custom", "proj"),
            FileRoute::LibraryCollection {
                source_project_id: Some("proj".to_string())
            }
        );
    }
}
