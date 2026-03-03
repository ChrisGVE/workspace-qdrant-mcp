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
