use std::collections::HashSet;
use std::path::Path;

use super::types::FileRoute;

/// Extensions for binary/reference formats that route to the `libraries` collection
/// even when discovered inside a project folder.
///
/// These are document formats (PDF, EPUB, etc.) that are unlikely to be "source code"
/// and are better served by the library ingestion pipeline. Source-like formats
/// (e.g., `.md`, `.txt`, `.html`) stay in `projects` because they are typically
/// project documentation meant to be searched alongside code.
pub(super) const LIBRARY_ROUTED_EXTENSIONS: &[&str] = &[
    ".pdf", ".epub", ".docx", ".doc", ".rtf", ".odt",
    ".mobi",
    ".pptx", ".ppt", ".pages", ".key", ".odp",
    ".xlsx", ".xls", ".ods", ".numbers", ".parquet",
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
    pub(super) project_extensions: HashSet<String>,
    /// Extensions allowed for library collections (superset of project_extensions
    /// plus document/reference formats like .pdf, .epub, .docx, etc.).
    pub(super) library_extensions: HashSet<String>,
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
            ".xlsx", ".xls", ".ods", ".numbers", ".parquet",
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
