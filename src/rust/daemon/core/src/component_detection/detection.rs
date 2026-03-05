//! Component detection logic: workspace parsing, directory fallback, and matching.

use std::fs;
use std::path::Path;

use super::{ComponentInfo, ComponentMap, ComponentSource};

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
            let base_dir = ws_path
                .split('*')
                .next()
                .unwrap_or("")
                .trim_end_matches('/');
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
