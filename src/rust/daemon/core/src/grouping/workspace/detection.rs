/// Workspace detection for Cargo, npm, and Go workspaces.
///
/// Walks up from a project directory to find workspace root files,
/// parses member lists, and resolves paths.
use std::path::{Path, PathBuf};

/// Information about a detected workspace.
#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    /// Unique identifier (hash of workspace root path).
    pub workspace_id: String,
    /// Absolute path to the workspace root.
    pub workspace_root: PathBuf,
    /// Absolute paths to workspace member directories.
    pub members: Vec<PathBuf>,
    /// Workspace type for labeling.
    pub workspace_type: &'static str,
}

/// Detect Cargo workspace from a project directory.
///
/// Walks up from `project_path` looking for a `Cargo.toml` with
/// a `[workspace]` section. Returns member paths resolved from
/// the workspace root.
pub fn detect_cargo_workspace(project_path: &Path) -> Option<WorkspaceInfo> {
    walk_up_detect(project_path, "Cargo.toml", parse_cargo_workspace)
}

/// Detect npm workspace from a project directory.
///
/// Walks up from `project_path` looking for a `package.json` with
/// a `workspaces` field.
pub fn detect_npm_workspace(project_path: &Path) -> Option<WorkspaceInfo> {
    walk_up_detect(project_path, "package.json", parse_npm_workspace)
}

/// Detect Go multi-module workspace.
///
/// Looks for `go.work` file in parent directories (Go 1.18+ workspaces).
pub fn detect_go_workspace(project_path: &Path) -> Option<WorkspaceInfo> {
    walk_up_detect(project_path, "go.work", parse_go_workspace)
}

// ---- Walk-up helper ---------------------------------------------------------

/// Walk up from `project_path` looking for `filename`, then parse with `parser`.
fn walk_up_detect(
    project_path: &Path,
    filename: &str,
    parser: fn(&Path, &str) -> Option<WorkspaceInfo>,
) -> Option<WorkspaceInfo> {
    let mut current = if project_path.is_file() {
        project_path.parent()?.to_path_buf()
    } else {
        project_path.to_path_buf()
    };

    for _ in 0..10 {
        let candidate = current.join(filename);
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if let Some(info) = parser(&current, &content) {
                    return Some(info);
                }
            }
        }
        if !current.pop() {
            break;
        }
    }

    None
}

// ---- Parsers ----------------------------------------------------------------

/// Parse a Cargo.toml for workspace members.
fn parse_cargo_workspace(workspace_root: &Path, content: &str) -> Option<WorkspaceInfo> {
    if !content.contains("[workspace]") {
        return None;
    }

    let mut members = Vec::new();
    let mut in_members = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("members") && trimmed.contains('=') {
            in_members = true;
            if let Some(bracket_start) = trimmed.find('[') {
                let rest = &trimmed[bracket_start..];
                members.extend(extract_toml_array_strings(rest));
                if rest.contains(']') {
                    in_members = false;
                }
            }
            continue;
        }

        if in_members {
            if trimmed == "]" || trimmed.starts_with(']') {
                in_members = false;
                continue;
            }
            let cleaned = trimmed.trim_matches(',').trim_matches('"').trim();
            if !cleaned.is_empty() && !cleaned.starts_with('#') {
                members.push(cleaned.to_string());
            }
        }
    }

    build_workspace_info(workspace_root, &members, "cargo")
}

/// Parse package.json for workspace members.
fn parse_npm_workspace(workspace_root: &Path, content: &str) -> Option<WorkspaceInfo> {
    let parsed: serde_json::Value = serde_json::from_str(content).ok()?;
    let workspaces = parsed.get("workspaces")?;

    let patterns: Vec<String> = match workspaces {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        serde_json::Value::Object(obj) => obj
            .get("packages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        _ => return None,
    };

    build_workspace_info(workspace_root, &patterns, "npm")
}

/// Parse go.work for workspace members.
fn parse_go_workspace(workspace_root: &Path, content: &str) -> Option<WorkspaceInfo> {
    let mut members = Vec::new();
    let mut in_use = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed == "use (" {
            in_use = true;
            continue;
        }
        if trimmed == ")" {
            in_use = false;
            continue;
        }

        if in_use {
            let path = trimmed.trim();
            if !path.is_empty() && !path.starts_with("//") {
                members.push(path.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("use ") {
            let path = rest.trim();
            if !path.is_empty() {
                members.push(path.to_string());
            }
        }
    }

    build_workspace_info(workspace_root, &members, "go")
}

// ---- Shared helpers ---------------------------------------------------------

/// Build a WorkspaceInfo from resolved member patterns, returning None if empty.
fn build_workspace_info(
    workspace_root: &Path,
    patterns: &[String],
    workspace_type: &'static str,
) -> Option<WorkspaceInfo> {
    if patterns.is_empty() {
        return None;
    }

    let resolved = resolve_workspace_members(workspace_root, patterns);
    if resolved.is_empty() {
        return None;
    }

    Some(WorkspaceInfo {
        workspace_id: generate_workspace_id(workspace_root),
        workspace_root: workspace_root.to_path_buf(),
        members: resolved,
        workspace_type,
    })
}

/// Extract strings from a TOML inline array fragment like `["a", "b"]`.
pub(crate) fn extract_toml_array_strings(s: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut in_quote = false;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '"' => {
                if in_quote {
                    results.push(current.clone());
                    current.clear();
                }
                in_quote = !in_quote;
            }
            _ if in_quote => current.push(ch),
            _ => {}
        }
    }

    results
}

/// Resolve workspace member patterns to absolute directories.
///
/// Handles glob patterns like `packages/*` or `daemon/*`.
/// Only returns directories that actually exist.
pub(crate) fn resolve_workspace_members(root: &Path, patterns: &[String]) -> Vec<PathBuf> {
    let mut resolved = Vec::new();

    for pattern in patterns {
        if pattern.contains('*') {
            let full_pattern = root.join(pattern);
            if let Ok(entries) = glob::glob(&full_pattern.to_string_lossy()) {
                for entry in entries.flatten() {
                    if entry.is_dir() {
                        resolved.push(entry);
                    }
                }
            }
        } else {
            let member_path = root.join(pattern);
            if member_path.is_dir() {
                resolved.push(member_path);
            }
        }
    }

    resolved
}

/// Generate a deterministic workspace_id from the workspace root path.
pub(crate) fn generate_workspace_id(workspace_root: &Path) -> String {
    use sha2::{Digest, Sha256};
    let input = workspace_root.to_string_lossy();
    let hash = Sha256::digest(input.as_bytes());
    format!("ws:{:x}", hash)[..15].to_string()
}
