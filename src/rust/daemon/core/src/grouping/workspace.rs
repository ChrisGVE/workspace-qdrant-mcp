/// Automated project grouping by workspace membership.
///
/// Detects Cargo workspaces, npm workspaces, and Go multi-module repos,
/// then groups member projects in `project_groups` with `group_type = "workspace"`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use super::schema;

// ---- Workspace detection ---------------------------------------------------

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
    let mut current = if project_path.is_file() {
        project_path.parent()?.to_path_buf()
    } else {
        project_path.to_path_buf()
    };

    // Walk up to find workspace root (max 10 levels)
    for _ in 0..10 {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if let Some(info) = parse_cargo_workspace(&current, &content) {
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

/// Parse a Cargo.toml for workspace members.
fn parse_cargo_workspace(workspace_root: &Path, content: &str) -> Option<WorkspaceInfo> {
    // Check for [workspace] section
    if !content.contains("[workspace]") {
        return None;
    }

    let mut members = Vec::new();
    let mut in_members = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("members") && trimmed.contains('=') {
            in_members = true;
            // Handle inline: members = ["a", "b"]
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
            // Lines like: "daemon/*",
            let cleaned = trimmed.trim_matches(',').trim_matches('"').trim();
            if !cleaned.is_empty() && !cleaned.starts_with('#') {
                members.push(cleaned.to_string());
            }
        }
    }

    if members.is_empty() {
        return None;
    }

    // Resolve glob patterns and member paths
    let resolved = resolve_workspace_members(workspace_root, &members);
    if resolved.is_empty() {
        return None;
    }

    let workspace_id = generate_workspace_id(workspace_root);

    Some(WorkspaceInfo {
        workspace_id,
        workspace_root: workspace_root.to_path_buf(),
        members: resolved,
        workspace_type: "cargo",
    })
}

/// Detect npm workspace from a project directory.
///
/// Walks up from `project_path` looking for a `package.json` with
/// a `workspaces` field.
pub fn detect_npm_workspace(project_path: &Path) -> Option<WorkspaceInfo> {
    let mut current = if project_path.is_file() {
        project_path.parent()?.to_path_buf()
    } else {
        project_path.to_path_buf()
    };

    for _ in 0..10 {
        let pkg_json = current.join("package.json");
        if pkg_json.exists() {
            if let Ok(content) = std::fs::read_to_string(&pkg_json) {
                if let Some(info) = parse_npm_workspace(&current, &content) {
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

/// Parse package.json for workspace members.
fn parse_npm_workspace(workspace_root: &Path, content: &str) -> Option<WorkspaceInfo> {
    let parsed: serde_json::Value = serde_json::from_str(content).ok()?;

    let workspaces = parsed.get("workspaces")?;

    // workspaces can be an array or an object with "packages" key
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

    if patterns.is_empty() {
        return None;
    }

    let resolved = resolve_workspace_members(workspace_root, &patterns);
    if resolved.is_empty() {
        return None;
    }

    let workspace_id = generate_workspace_id(workspace_root);

    Some(WorkspaceInfo {
        workspace_id,
        workspace_root: workspace_root.to_path_buf(),
        members: resolved,
        workspace_type: "npm",
    })
}

/// Detect Go multi-module workspace.
///
/// Looks for `go.work` file in parent directories (Go 1.18+ workspaces),
/// or multiple `go.mod` files in the same repository.
pub fn detect_go_workspace(project_path: &Path) -> Option<WorkspaceInfo> {
    let mut current = if project_path.is_file() {
        project_path.parent()?.to_path_buf()
    } else {
        project_path.to_path_buf()
    };

    for _ in 0..10 {
        let go_work = current.join("go.work");
        if go_work.exists() {
            if let Ok(content) = std::fs::read_to_string(&go_work) {
                if let Some(info) = parse_go_workspace(&current, &content) {
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

    if members.is_empty() {
        return None;
    }

    let resolved = resolve_workspace_members(workspace_root, &members);
    if resolved.is_empty() {
        return None;
    }

    let workspace_id = generate_workspace_id(workspace_root);

    Some(WorkspaceInfo {
        workspace_id,
        workspace_root: workspace_root.to_path_buf(),
        members: resolved,
        workspace_type: "go",
    })
}

// ---- Helpers ---------------------------------------------------------------

/// Extract strings from a TOML inline array fragment like `["a", "b"]`.
fn extract_toml_array_strings(s: &str) -> Vec<String> {
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
fn resolve_workspace_members(root: &Path, patterns: &[String]) -> Vec<PathBuf> {
    let mut resolved = Vec::new();

    for pattern in patterns {
        if pattern.contains('*') {
            // Glob expand
            let full_pattern = root.join(pattern);
            if let Ok(entries) = glob::glob(&full_pattern.to_string_lossy()) {
                for entry in entries.flatten() {
                    if entry.is_dir() {
                        resolved.push(entry);
                    }
                }
            }
        } else {
            // Direct path
            let member_path = root.join(pattern);
            if member_path.is_dir() {
                resolved.push(member_path);
            }
        }
    }

    resolved
}

/// Generate a deterministic workspace_id from the workspace root path.
fn generate_workspace_id(workspace_root: &Path) -> String {
    use sha2::{Digest, Sha256};
    let input = workspace_root.to_string_lossy();
    let hash = Sha256::digest(input.as_bytes());
    format!("ws:{:x}", hash)[..15].to_string()
}

// ---- Grouping logic --------------------------------------------------------

/// Recompute workspace-based groups for all registered projects.
///
/// For each project in `watch_folders`, detects workspace membership
/// and creates groups linking projects that share a workspace root.
///
/// Returns the number of workspace groups created.
pub async fn compute_workspace_groups(pool: &SqlitePool) -> Result<usize, sqlx::Error> {
    // Clear existing workspace groups
    sqlx::query("DELETE FROM project_groups WHERE group_type = 'workspace'")
        .execute(pool)
        .await?;

    // Load all project watch folders
    let rows = sqlx::query(
        r#"
        SELECT tenant_id, folder_path
        FROM watch_folders
        WHERE watch_type = 'project'
        "#,
    )
    .fetch_all(pool)
    .await?;

    // Collect workspace memberships: workspace_id -> vec of tenant_ids
    let mut workspace_tenants: HashMap<String, Vec<String>> = HashMap::new();

    for row in &rows {
        let tenant_id: String = row.get("tenant_id");
        let folder_path: String = row.get("folder_path");
        let project_path = Path::new(&folder_path);

        // Try each workspace detector
        let workspace = detect_cargo_workspace(project_path)
            .or_else(|| detect_npm_workspace(project_path))
            .or_else(|| detect_go_workspace(project_path));

        if let Some(ws) = workspace {
            // Check if this project's path is actually a member
            let is_member = ws.members.iter().any(|m| {
                // Member path matches or is a parent of the project path
                project_path.starts_with(m) || m.starts_with(project_path)
            }) || project_path.starts_with(&ws.workspace_root);

            if is_member {
                workspace_tenants
                    .entry(ws.workspace_id.clone())
                    .or_default()
                    .push(tenant_id.clone());

                debug!(
                    tenant = tenant_id.as_str(),
                    workspace_id = ws.workspace_id.as_str(),
                    workspace_type = ws.workspace_type,
                    "Detected workspace membership"
                );
            }
        }
    }

    let groups_created =
        create_workspace_groups(pool, &workspace_tenants).await?;

    info!(
        projects = rows.len(),
        workspaces = workspace_tenants.len(),
        groups = groups_created,
        "Workspace group computation complete"
    );

    Ok(groups_created)
}

/// Write workspace groups to the database for all workspaces with 2+ members.
///
/// Returns the number of groups created.
async fn create_workspace_groups(
    pool: &SqlitePool,
    workspace_tenants: &HashMap<String, Vec<String>>,
) -> Result<usize, sqlx::Error> {
    let mut groups_created = 0;

    for (workspace_id, tenants) in workspace_tenants {
        if tenants.len() < 2 {
            debug!(workspace_id = workspace_id.as_str(), "Skipping single-member workspace");
            continue;
        }

        let group_id = format!("workspace:{}", workspace_id);
        for tenant_id in tenants {
            schema::add_to_group(pool, &group_id, tenant_id, "workspace", 1.0).await?;
        }
        debug!(
            workspace_id = workspace_id.as_str(),
            members = tenants.len(),
            "Created workspace group"
        );
        groups_created += 1;
    }

    Ok(groups_created)
}

/// Detect workspace for a single project and update its group membership.
///
/// Call this when a new project is registered. More efficient than
/// recomputing all groups.
pub async fn update_project_workspace_group(
    pool: &SqlitePool,
    tenant_id: &str,
    project_path: &Path,
) -> Result<bool, sqlx::Error> {
    // Remove any existing workspace membership for this tenant
    sqlx::query("DELETE FROM project_groups WHERE tenant_id = ? AND group_type = 'workspace'")
        .bind(tenant_id)
        .execute(pool)
        .await?;

    let workspace = detect_cargo_workspace(project_path)
        .or_else(|| detect_npm_workspace(project_path))
        .or_else(|| detect_go_workspace(project_path));

    let ws = match workspace {
        Some(ws) => ws,
        None => {
            debug!(tenant_id, "No workspace detected for project");
            return Ok(false);
        }
    };

    let group_id = format!("workspace:{}", ws.workspace_id);

    // Add this project
    schema::add_to_group(pool, &group_id, tenant_id, "workspace", 1.0).await?;

    // Find other registered projects that are also members of this workspace
    let peer_rows = sqlx::query(
        r#"
        SELECT tenant_id, folder_path
        FROM watch_folders
        WHERE watch_type = 'project' AND tenant_id != ?
        "#,
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await?;

    for row in &peer_rows {
        let peer_tenant: String = row.get("tenant_id");
        let peer_path_str: String = row.get("folder_path");
        let peer_path = Path::new(&peer_path_str);

        // Check if peer is a member of this workspace
        let is_member = ws.members.iter().any(|m| {
            peer_path.starts_with(m) || m.starts_with(peer_path)
        }) || peer_path.starts_with(&ws.workspace_root);

        if is_member {
            schema::add_to_group(pool, &group_id, &peer_tenant, "workspace", 1.0)
                .await?;

            debug!(
                peer = peer_tenant.as_str(),
                workspace_id = ws.workspace_id.as_str(),
                "Added peer to workspace group"
            );
        }
    }

    debug!(
        tenant_id,
        workspace_id = ws.workspace_id.as_str(),
        workspace_type = ws.workspace_type,
        "Updated project workspace group"
    );

    Ok(true)
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;
