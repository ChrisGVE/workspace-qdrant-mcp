//! Component auto-detection from workspace definition files.
//!
//! Parses Cargo.toml `[workspace]` members and package.json `workspaces`
//! to derive dot-separated hierarchical component names.
//!
//! Examples:
//!   Cargo.toml member `"daemon/core"`  → component `"daemon.core"`
//!   package.json workspace `"packages/ui"` → component `"packages.ui"`

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use tracing::debug;

/// Detected workspace component.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Dot-separated component ID, e.g. "daemon.core"
    pub id: String,
    /// Base directory relative to project root, e.g. "daemon/core"
    pub base_path: String,
    /// Glob patterns matching files in this component
    pub patterns: Vec<String>,
    /// Detection source
    pub source: ComponentSource,
}

/// How the component was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentSource {
    Cargo,
    Npm,
    Directory,
}

impl ComponentSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cargo => "cargo",
            Self::Npm => "npm",
            Self::Directory => "directory",
        }
    }
}

/// Map from component ID to its info.
pub type ComponentMap = HashMap<String, ComponentInfo>;

// ── Detection ────────────────────────────────────────────────────────────

/// Detect project components from workspace definition files.
///
/// Tries Cargo.toml first, then package.json, then falls back to
/// top-level directory heuristic.
pub fn detect_components(project_path: &Path) -> ComponentMap {
    let mut components = ComponentMap::new();

    // Try Cargo workspace first
    let cargo = detect_cargo_workspace(project_path);
    if !cargo.is_empty() {
        components.extend(cargo);
    }

    // Try npm/yarn workspaces
    let npm = detect_npm_workspace(project_path);
    for (id, info) in npm {
        components.entry(id).or_insert(info);
    }

    // Fallback: top-level directories if nothing detected
    if components.is_empty() {
        components.extend(detect_from_directories(project_path));
    }

    components
}

// ── Cargo workspace ──────────────────────────────────────────────────────

/// Search for Cargo.toml workspace definitions and parse members.
fn detect_cargo_workspace(project_path: &Path) -> ComponentMap {
    let mut components = ComponentMap::new();

    let candidates = [
        project_path.join("Cargo.toml"),
        project_path.join("src/rust/Cargo.toml"),
    ];

    for cargo_path in &candidates {
        let content = match fs::read_to_string(cargo_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let members = parse_cargo_members(&content);
        if members.is_empty() {
            continue;
        }

        // Compute base directory relative to project root
        let cargo_dir = cargo_path.parent().unwrap_or(project_path);
        let relative_base = if cargo_dir == project_path {
            String::new()
        } else {
            cargo_dir
                .strip_prefix(project_path)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
        };

        for member in &members {
            let full_path = if relative_base.is_empty() {
                member.clone()
            } else {
                format!("{}/{}", relative_base, member)
            };
            let id = path_to_component_id(member);

            components.insert(
                id.clone(),
                ComponentInfo {
                    id,
                    base_path: full_path.clone(),
                    patterns: vec![format!("{}/**", full_path)],
                    source: ComponentSource::Cargo,
                },
            );
        }

        // Found a workspace, stop searching
        break;
    }

    components
}

/// Extract workspace members from Cargo.toml content.
///
/// Parses the `members = [...]` array from a `[workspace]` section.
/// Handles multi-line arrays and inline comments.
pub fn parse_cargo_members(content: &str) -> Vec<String> {
    let workspace_idx = match content.find("[workspace]") {
        Some(idx) => idx,
        None => return Vec::new(),
    };

    let after_workspace = &content[workspace_idx..];

    // Find members = [...] — using a simple search since we need the content inside brackets
    let members_start = match after_workspace.find("members") {
        Some(idx) => idx,
        None => return Vec::new(),
    };

    let after_members = &after_workspace[members_start..];
    let bracket_start = match after_members.find('[') {
        Some(idx) => idx,
        None => return Vec::new(),
    };
    let bracket_end = match after_members[bracket_start..].find(']') {
        Some(idx) => bracket_start + idx,
        None => return Vec::new(),
    };

    let array_content = &after_members[bracket_start + 1..bracket_end];

    // Strip line comments before extracting strings
    let cleaned: String = array_content
        .lines()
        .map(|line| {
            if let Some(hash_pos) = line.find('#') {
                &line[..hash_pos]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Extract quoted strings (both " and ')
    let mut members = Vec::new();
    let mut chars = cleaned.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch == '"' || ch == '\'' {
            let quote = ch;
            chars.next(); // consume opening quote
            let value: String = chars.by_ref().take_while(|&c| c != quote).collect();
            if !value.is_empty() {
                members.push(value);
            }
        } else {
            chars.next();
        }
    }

    members
}

// ── npm workspace ────────────────────────────────────────────────────────

/// Parse package.json workspaces into components.
fn detect_npm_workspace(project_path: &Path) -> ComponentMap {
    let mut components = ComponentMap::new();

    let pkg_path = project_path.join("package.json");
    let content = match fs::read_to_string(&pkg_path) {
        Ok(c) => c,
        Err(_) => return components,
    };

    let pkg: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return components,
    };

    let workspace_paths = extract_npm_workspace_paths(&pkg);

    for ws_path in &workspace_paths {
        if ws_path.contains('*') {
            // Resolve glob: "packages/*" → list actual subdirectories
            let base_dir = ws_path.split('*').next().unwrap_or("").trim_end_matches('/');
            let full_base = project_path.join(base_dir);

            if let Ok(entries) = fs::read_dir(&full_base) {
                for entry in entries.flatten() {
                    if let Ok(ft) = entry.file_type() {
                        if ft.is_dir() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            let rel_path = if base_dir.is_empty() {
                                name.clone()
                            } else {
                                format!("{}/{}", base_dir, name)
                            };
                            let id = path_to_component_id(&rel_path);
                            components.insert(
                                id.clone(),
                                ComponentInfo {
                                    id,
                                    base_path: rel_path.clone(),
                                    patterns: vec![format!("{}/**", rel_path)],
                                    source: ComponentSource::Npm,
                                },
                            );
                        }
                    }
                }
            }
        } else {
            let id = path_to_component_id(ws_path);
            components.insert(
                id.clone(),
                ComponentInfo {
                    id,
                    base_path: ws_path.clone(),
                    patterns: vec![format!("{}/**", ws_path)],
                    source: ComponentSource::Npm,
                },
            );
        }
    }

    components
}

/// Extract workspace paths from package.json value.
fn extract_npm_workspace_paths(pkg: &serde_json::Value) -> Vec<String> {
    match &pkg["workspaces"] {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        serde_json::Value::Object(obj) => {
            if let Some(serde_json::Value::Array(arr)) = obj.get("packages") {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

// ── Directory fallback ───────────────────────────────────────────────────

const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "node_modules",
    "target",
    "dist",
    "build",
    ".taskmaster",
    ".claude",
    ".serena",
    "tmp",
];

/// Fallback: use top-level directories as components.
fn detect_from_directories(project_path: &Path) -> ComponentMap {
    let mut components = ComponentMap::new();

    let entries = match fs::read_dir(project_path) {
        Ok(e) => e,
        Err(_) => return components,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || IGNORED_DIRS.contains(&name.as_str()) {
            continue;
        }

        if let Ok(ft) = entry.file_type() {
            if ft.is_dir() {
                components.insert(
                    name.clone(),
                    ComponentInfo {
                        id: name.clone(),
                        base_path: name.clone(),
                        patterns: vec![format!("{}/**", name)],
                        source: ComponentSource::Directory,
                    },
                );
            }
        }
    }

    components
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Convert a path to a dot-separated component ID.
///
/// `"daemon/core"` → `"daemon.core"`
/// `"cli"` → `"cli"`
pub fn path_to_component_id(path: &str) -> String {
    path.trim_matches('/').replace('/', ".")
}

/// Check if a relative file path belongs to a component.
pub fn file_matches_component(relative_path: &str, component: &ComponentInfo) -> bool {
    relative_path == component.base_path
        || relative_path.starts_with(&format!("{}/", component.base_path))
}

/// Check if a component ID matches a filter (exact or prefix).
///
/// `"daemon.core"` matches filter `"daemon.core"` (exact)
/// `"daemon.core"` matches filter `"daemon"` (prefix)
pub fn component_matches_filter(component_id: &str, filter: &str) -> bool {
    component_id == filter || component_id.starts_with(&format!("{}.", filter))
}

/// Assign a component to a file based on its relative path.
///
/// Returns the most specific (longest base_path) matching component.
pub fn assign_component<'a>(
    relative_path: &str,
    components: &'a ComponentMap,
) -> Option<&'a ComponentInfo> {
    let mut best: Option<&ComponentInfo> = None;
    let mut best_len: usize = 0;

    for component in components.values() {
        if file_matches_component(relative_path, component) && component.base_path.len() > best_len
        {
            best = Some(component);
            best_len = component.base_path.len();
        }
    }

    best
}

/// Persist detected components to the project_components table.
pub async fn persist_components(
    pool: &sqlx::SqlitePool,
    watch_folder_id: &str,
    components: &ComponentMap,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();

    for component in components.values() {
        let component_id = format!("{}:{}", watch_folder_id, component.id);
        let patterns_json = serde_json::to_string(&component.patterns).unwrap_or_default();

        sqlx::query(
            "INSERT OR REPLACE INTO project_components
             (component_id, watch_folder_id, component_name, base_path, source, patterns, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        )
        .bind(&component_id)
        .bind(watch_folder_id)
        .bind(&component.id)
        .bind(&component.base_path)
        .bind(component.source.as_str())
        .bind(&patterns_json)
        .bind(&now)
        .execute(pool)
        .await?;
    }

    debug!(
        "Persisted {} components for watch folder {}",
        components.len(),
        watch_folder_id
    );
    Ok(())
}

/// Load components from the project_components table.
pub async fn load_components(
    pool: &sqlx::SqlitePool,
    watch_folder_id: &str,
) -> Result<ComponentMap, sqlx::Error> {
    use sqlx::Row;

    let rows = sqlx::query(
        "SELECT component_name, base_path, source, patterns
         FROM project_components WHERE watch_folder_id = ?1",
    )
    .bind(watch_folder_id)
    .fetch_all(pool)
    .await?;

    let mut components = ComponentMap::new();
    for row in &rows {
        let name: String = row.get("component_name");
        let base_path: String = row.get("base_path");
        let source_str: String = row.get("source");
        let patterns_json: Option<String> = row.get("patterns");

        let source = match source_str.as_str() {
            "cargo" => ComponentSource::Cargo,
            "npm" => ComponentSource::Npm,
            _ => ComponentSource::Directory,
        };

        let patterns: Vec<String> = patterns_json
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or_else(|| vec![format!("{}/**", base_path)]);

        components.insert(
            name.clone(),
            ComponentInfo {
                id: name,
                base_path,
                patterns,
                source,
            },
        );
    }

    Ok(components)
}

// ── Backfill ─────────────────────────────────────────────────────────────

/// Stats from a component backfill run.
#[derive(Debug, Clone, Default)]
pub struct BackfillStats {
    /// Watch folders processed
    pub folders_processed: u64,
    /// Files updated with component assignment
    pub files_updated: u64,
    /// Files that couldn't be assigned (no matching component)
    pub files_unmatched: u64,
    /// Errors during processing
    pub errors: u64,
}

/// Backfill NULL component assignments for all active watch folders.
///
/// For each enabled watch folder, detects components from the workspace
/// definition files, then assigns components to tracked_files that have
/// `component IS NULL`. Batches updates in transactions of `batch_size`.
pub async fn backfill_components(
    pool: &sqlx::SqlitePool,
    batch_size: usize,
) -> Result<BackfillStats, String> {
    use sqlx::Row;

    let mut stats = BackfillStats::default();

    // Get all enabled, non-archived watch folders with their paths
    let folders: Vec<(String, String)> = sqlx::query_as(
        "SELECT watch_id, path FROM watch_folders WHERE enabled = 1 AND is_archived = 0",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query watch_folders: {e}"))?;

    for (watch_id, path) in &folders {
        let project_path = Path::new(path.as_str());
        if !project_path.is_dir() {
            debug!("Skipping backfill for {}: path does not exist", watch_id);
            continue;
        }

        // Detect components for this project
        let components = detect_components(project_path);
        if components.is_empty() {
            debug!("No components detected for {}, skipping backfill", watch_id);
            stats.folders_processed += 1;
            continue;
        }

        // Persist the detected components (idempotent)
        if let Err(e) = persist_components(pool, watch_id, &components).await {
            debug!("Failed to persist components for {}: {}", watch_id, e);
            stats.errors += 1;
        }

        // Find all tracked files with NULL component in this watch folder
        let null_files: Vec<(i64, String)> = sqlx::query(
            "SELECT file_id, relative_path FROM tracked_files
             WHERE watch_folder_id = ?1 AND component IS NULL AND relative_path IS NOT NULL",
        )
        .bind(watch_id)
        .map(|row: sqlx::sqlite::SqliteRow| {
            let file_id: i64 = row.get("file_id");
            let rel_path: String = row.get("relative_path");
            (file_id, rel_path)
        })
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to query tracked_files for {}: {e}", watch_id))?;

        if null_files.is_empty() {
            stats.folders_processed += 1;
            continue;
        }

        debug!(
            "Backfilling components for {}: {} files with NULL component",
            watch_id,
            null_files.len()
        );

        // Batch update in transactions
        for chunk in null_files.chunks(batch_size) {
            let mut tx = pool
                .begin()
                .await
                .map_err(|e| format!("Failed to begin transaction: {e}"))?;

            for (file_id, rel_path) in chunk {
                match assign_component(rel_path, &components) {
                    Some(comp) => {
                        if let Err(e) = sqlx::query(
                            "UPDATE tracked_files SET component = ?1 WHERE file_id = ?2",
                        )
                        .bind(&comp.id)
                        .bind(file_id)
                        .execute(&mut *tx)
                        .await
                        {
                            debug!("Failed to update file_id {}: {}", file_id, e);
                            stats.errors += 1;
                        } else {
                            stats.files_updated += 1;
                        }
                    }
                    None => {
                        stats.files_unmatched += 1;
                    }
                }
            }

            tx.commit()
                .await
                .map_err(|e| format!("Failed to commit batch: {e}"))?;
        }

        stats.folders_processed += 1;
    }

    Ok(stats)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_path_to_component_id() {
        assert_eq!(path_to_component_id("daemon/core"), "daemon.core");
        assert_eq!(path_to_component_id("cli"), "cli");
        assert_eq!(path_to_component_id("src/typescript/mcp"), "src.typescript.mcp");
        assert_eq!(path_to_component_id("trailing/"), "trailing");
        assert_eq!(path_to_component_id("/leading"), "leading");
    }

    #[test]
    fn test_parse_cargo_members_basic() {
        let content = r#"
[workspace]
resolver = "2"
members = [
    "daemon/core",
    "daemon/grpc",
    "cli",
]
"#;
        let members = parse_cargo_members(content);
        assert_eq!(members, vec!["daemon/core", "daemon/grpc", "cli"]);
    }

    #[test]
    fn test_parse_cargo_members_inline() {
        let content = r#"
[workspace]
members = ["a", "b"]
"#;
        let members = parse_cargo_members(content);
        assert_eq!(members, vec!["a", "b"]);
    }

    #[test]
    fn test_parse_cargo_members_with_comments() {
        let content = r#"
[workspace]
members = [
    "a",
    # "commented-out",
    "b",
]
"#;
        let members = parse_cargo_members(content);
        assert_eq!(members, vec!["a", "b"]);
    }

    #[test]
    fn test_parse_cargo_members_no_workspace() {
        let content = r#"
[package]
name = "my-crate"
"#;
        assert!(parse_cargo_members(content).is_empty());
    }

    #[test]
    fn test_file_matches_component() {
        let comp = ComponentInfo {
            id: "daemon.core".into(),
            base_path: "daemon/core".into(),
            patterns: vec!["daemon/core/**".into()],
            source: ComponentSource::Cargo,
        };

        assert!(file_matches_component("daemon/core/src/lib.rs", &comp));
        assert!(file_matches_component("daemon/core", &comp));
        assert!(!file_matches_component("daemon/grpc/src/lib.rs", &comp));
        assert!(!file_matches_component("daemon/core_extra/foo.rs", &comp));
    }

    #[test]
    fn test_component_matches_filter() {
        assert!(component_matches_filter("daemon.core", "daemon.core"));
        assert!(component_matches_filter("daemon.core", "daemon"));
        assert!(!component_matches_filter("daemon.core", "cli"));
        assert!(!component_matches_filter("daemon", "daemon.core"));
    }

    #[test]
    fn test_assign_component_most_specific() {
        let mut components = ComponentMap::new();
        components.insert(
            "daemon".into(),
            ComponentInfo {
                id: "daemon".into(),
                base_path: "daemon".into(),
                patterns: vec!["daemon/**".into()],
                source: ComponentSource::Cargo,
            },
        );
        components.insert(
            "daemon.core".into(),
            ComponentInfo {
                id: "daemon.core".into(),
                base_path: "daemon/core".into(),
                patterns: vec!["daemon/core/**".into()],
                source: ComponentSource::Cargo,
            },
        );

        let result = assign_component("daemon/core/src/lib.rs", &components);
        assert_eq!(result.unwrap().id, "daemon.core");

        let result = assign_component("daemon/grpc/src/lib.rs", &components);
        assert_eq!(result.unwrap().id, "daemon");

        let result = assign_component("cli/src/main.rs", &components);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_cargo_workspace() {
        let dir = TempDir::new().unwrap();
        let cargo_toml = dir.path().join("Cargo.toml");
        fs::write(
            &cargo_toml,
            r#"
[workspace]
resolver = "2"
members = ["crate-a", "crate-b"]
"#,
        )
        .unwrap();

        // Create the member directories (not strictly needed for detection,
        // but validates the path computation)
        fs::create_dir_all(dir.path().join("crate-a")).unwrap();
        fs::create_dir_all(dir.path().join("crate-b")).unwrap();

        let components = detect_components(dir.path());
        assert_eq!(components.len(), 2);
        assert!(components.contains_key("crate-a"));
        assert!(components.contains_key("crate-b"));
        assert_eq!(components["crate-a"].source, ComponentSource::Cargo);
    }

    #[test]
    fn test_detect_cargo_workspace_nested() {
        let dir = TempDir::new().unwrap();
        // No root Cargo.toml, but one in src/rust/
        let nested = dir.path().join("src/rust");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            nested.join("Cargo.toml"),
            r#"
[workspace]
members = ["daemon/core", "cli"]
"#,
        )
        .unwrap();

        let components = detect_components(dir.path());
        assert_eq!(components.len(), 2);

        // Component IDs are based on member path, not full path
        assert!(components.contains_key("daemon.core"));
        assert!(components.contains_key("cli"));

        // base_path should include the relative prefix
        assert_eq!(components["daemon.core"].base_path, "src/rust/daemon/core");
        assert_eq!(components["cli"].base_path, "src/rust/cli");
    }

    #[test]
    fn test_detect_npm_workspace() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"workspaces": ["packages/ui", "packages/api"]}"#,
        )
        .unwrap();

        let components = detect_components(dir.path());
        assert_eq!(components.len(), 2);
        assert!(components.contains_key("packages.ui"));
        assert!(components.contains_key("packages.api"));
        assert_eq!(components["packages.ui"].source, ComponentSource::Npm);
    }

    #[test]
    fn test_detect_npm_workspace_glob() {
        let dir = TempDir::new().unwrap();
        let pkgs = dir.path().join("packages");
        fs::create_dir_all(pkgs.join("alpha")).unwrap();
        fs::create_dir_all(pkgs.join("beta")).unwrap();
        // Create a file that should be ignored (not a dir)
        fs::write(pkgs.join("README.md"), "").unwrap();

        fs::write(
            dir.path().join("package.json"),
            r#"{"workspaces": ["packages/*"]}"#,
        )
        .unwrap();

        let components = detect_components(dir.path());
        assert_eq!(components.len(), 2);
        assert!(components.contains_key("packages.alpha"));
        assert!(components.contains_key("packages.beta"));
    }

    #[test]
    fn test_detect_directory_fallback() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::create_dir_all(dir.path().join("tests")).unwrap();
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        fs::create_dir_all(dir.path().join("node_modules")).unwrap();
        fs::write(dir.path().join("README.md"), "").unwrap();

        let components = detect_components(dir.path());
        assert!(components.contains_key("src"));
        assert!(components.contains_key("tests"));
        assert!(!components.contains_key(".git"));
        assert!(!components.contains_key("node_modules"));
        assert_eq!(components["src"].source, ComponentSource::Directory);
    }

    #[test]
    fn test_cargo_takes_priority_over_npm() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"
[workspace]
members = ["shared"]
"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"workspaces": ["shared", "web"]}"#,
        )
        .unwrap();

        let components = detect_components(dir.path());
        // "shared" should be Cargo (takes priority)
        assert_eq!(components["shared"].source, ComponentSource::Cargo);
        // "web" should be npm (no conflict)
        assert!(components.contains_key("web"));
        assert_eq!(components["web"].source, ComponentSource::Npm);
    }

    #[tokio::test]
    async fn test_backfill_components() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"
[workspace]
members = ["alpha", "beta"]
"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("alpha")).unwrap();
        fs::create_dir_all(dir.path().join("beta")).unwrap();

        // Create in-memory SQLite with required schema
        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'projects',
                tenant_id TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                is_archived INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_folder_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                relative_path TEXT,
                component TEXT,
                UNIQUE(watch_folder_id, file_path)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE project_components (
                component_id TEXT PRIMARY KEY,
                watch_folder_id TEXT NOT NULL,
                component_name TEXT NOT NULL,
                base_path TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'auto',
                patterns TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(watch_folder_id, component_name)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a watch folder pointing to our temp dir
        let path_str = dir.path().to_string_lossy().to_string();
        sqlx::query("INSERT INTO watch_folders (watch_id, path) VALUES ('wf1', ?1)")
            .bind(&path_str)
            .execute(&pool)
            .await
            .unwrap();

        // Insert tracked files with NULL component
        for rel in &["alpha/src/lib.rs", "beta/src/main.rs", "README.md"] {
            let abs = format!("{}/{}", path_str, rel);
            sqlx::query(
                "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
                 VALUES ('wf1', ?1, ?2)",
            )
            .bind(&abs)
            .bind(rel)
            .execute(&pool)
            .await
            .unwrap();
        }

        let stats = backfill_components(&pool, 100).await.unwrap();
        assert_eq!(stats.folders_processed, 1);
        assert_eq!(stats.files_updated, 2); // alpha/src/lib.rs + beta/src/main.rs
        assert_eq!(stats.files_unmatched, 1); // README.md at root

        // Verify the component column was set
        let row: (Option<String>,) = sqlx::query_as(
            "SELECT component FROM tracked_files WHERE relative_path = 'alpha/src/lib.rs'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.0.as_deref(), Some("alpha"));

        let row: (Option<String>,) = sqlx::query_as(
            "SELECT component FROM tracked_files WHERE relative_path = 'beta/src/main.rs'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.0.as_deref(), Some("beta"));

        // README.md should still be NULL
        let row: (Option<String>,) = sqlx::query_as(
            "SELECT component FROM tracked_files WHERE relative_path = 'README.md'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.0.is_none());

        // Components should be persisted
        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM project_components WHERE watch_folder_id = 'wf1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 2); // alpha + beta
    }
}
