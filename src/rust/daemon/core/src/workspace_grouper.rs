/// Automated project grouping by workspace membership.
///
/// Detects Cargo workspaces, npm workspaces, and Go multi-module repos,
/// then groups member projects in `project_groups` with `group_type = "workspace"`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use crate::project_groups_schema;

// ─── Workspace detection ───────────────────────────────────────────────

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

// ─── Helpers ───────────────────────────────────────────────────────────

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

// ─── Grouping logic ────────────────────────────────────────────────────

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

    // Collect workspace memberships: workspace_id → vec of tenant_ids
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

    let mut groups_created = 0;

    // Create groups for workspaces with 2+ members
    for (workspace_id, tenants) in &workspace_tenants {
        if tenants.len() < 2 {
            debug!(
                workspace_id = workspace_id.as_str(),
                "Skipping single-member workspace"
            );
            continue;
        }

        let group_id = format!("workspace:{}", workspace_id);

        for tenant_id in tenants {
            project_groups_schema::add_to_group(pool, &group_id, tenant_id, "workspace", 1.0)
                .await?;
        }

        debug!(
            workspace_id = workspace_id.as_str(),
            members = tenants.len(),
            "Created workspace group"
        );
        groups_created += 1;
    }

    info!(
        projects = rows.len(),
        workspaces = workspace_tenants.len(),
        groups = groups_created,
        "Workspace group computation complete"
    );

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
    project_groups_schema::add_to_group(pool, &group_id, tenant_id, "workspace", 1.0).await?;

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
            project_groups_schema::add_to_group(pool, &group_id, &peer_tenant, "workspace", 1.0)
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
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use tempfile::TempDir;

    // ─── Parsing tests ──────────────────────────────────────────────────

    #[test]
    fn test_parse_cargo_workspace_basic() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create workspace Cargo.toml
        fs::write(
            root.join("Cargo.toml"),
            r#"
[workspace]
members = [
    "crate-a",
    "crate-b",
]
"#,
        )
        .unwrap();

        // Create member directories
        fs::create_dir_all(root.join("crate-a")).unwrap();
        fs::create_dir_all(root.join("crate-b")).unwrap();

        let info = detect_cargo_workspace(&root.join("crate-a")).unwrap();
        assert_eq!(info.workspace_type, "cargo");
        assert_eq!(info.workspace_root, root);
        assert_eq!(info.members.len(), 2);
    }

    #[test]
    fn test_parse_cargo_workspace_glob() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("Cargo.toml"),
            r#"
[workspace]
members = ["packages/*"]
"#,
        )
        .unwrap();

        fs::create_dir_all(root.join("packages/lib-a")).unwrap();
        fs::create_dir_all(root.join("packages/lib-b")).unwrap();

        let info = detect_cargo_workspace(&root.join("packages/lib-a")).unwrap();
        assert_eq!(info.members.len(), 2);
    }

    #[test]
    fn test_parse_cargo_workspace_inline() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"a\", \"b\"]\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("a")).unwrap();
        fs::create_dir_all(root.join("b")).unwrap();

        let info = detect_cargo_workspace(&root.join("a")).unwrap();
        assert_eq!(info.members.len(), 2);
    }

    #[test]
    fn test_no_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Regular Cargo.toml (no workspace section)
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        assert!(detect_cargo_workspace(root).is_none());
    }

    #[test]
    fn test_parse_npm_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("package.json"),
            r#"{
  "name": "monorepo",
  "workspaces": ["packages/*"]
}"#,
        )
        .unwrap();

        fs::create_dir_all(root.join("packages/app")).unwrap();
        fs::create_dir_all(root.join("packages/lib")).unwrap();

        let info = detect_npm_workspace(&root.join("packages/app")).unwrap();
        assert_eq!(info.workspace_type, "npm");
        assert_eq!(info.members.len(), 2);
    }

    #[test]
    fn test_parse_npm_workspace_object() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("package.json"),
            r#"{
  "name": "monorepo",
  "workspaces": { "packages": ["services/*"] }
}"#,
        )
        .unwrap();

        fs::create_dir_all(root.join("services/api")).unwrap();

        let info = detect_npm_workspace(&root.join("services/api")).unwrap();
        assert_eq!(info.workspace_type, "npm");
        assert_eq!(info.members.len(), 1);
    }

    #[test]
    fn test_no_npm_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("package.json"),
            r#"{"name": "standalone", "version": "1.0.0"}"#,
        )
        .unwrap();

        assert!(detect_npm_workspace(root).is_none());
    }

    #[test]
    fn test_parse_go_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("go.work"),
            "go 1.21\n\nuse (\n\t./cmd\n\t./pkg\n)\n",
        )
        .unwrap();

        fs::create_dir_all(root.join("cmd")).unwrap();
        fs::create_dir_all(root.join("pkg")).unwrap();

        let info = detect_go_workspace(&root.join("cmd")).unwrap();
        assert_eq!(info.workspace_type, "go");
        assert_eq!(info.members.len(), 2);
    }

    #[test]
    fn test_parse_go_workspace_single_use() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(root.join("go.work"), "go 1.21\n\nuse ./app\n").unwrap();
        fs::create_dir_all(root.join("app")).unwrap();

        let info = detect_go_workspace(&root.join("app")).unwrap();
        assert_eq!(info.members.len(), 1);
    }

    #[test]
    fn test_no_go_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // go.mod only (no go.work) → not a workspace
        fs::write(
            root.join("go.mod"),
            "module example.com/myapp\n\ngo 1.21\n",
        )
        .unwrap();

        assert!(detect_go_workspace(root).is_none());
    }

    #[test]
    fn test_workspace_id_deterministic() {
        let path = Path::new("/home/user/projects/my-workspace");
        let id1 = generate_workspace_id(path);
        let id2 = generate_workspace_id(path);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_workspace_id_unique() {
        let id1 = generate_workspace_id(Path::new("/a"));
        let id2 = generate_workspace_id(Path::new("/b"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_extract_toml_array_strings() {
        let result = extract_toml_array_strings(r#"["foo", "bar/baz"]"#);
        assert_eq!(result, vec!["foo", "bar/baz"]);
    }

    // ─── SQLite integration tests ───────────────────────────────────────

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Minimal watch_folders table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS watch_folders (
                watch_id TEXT PRIMARY KEY,
                folder_path TEXT NOT NULL,
                watch_type TEXT NOT NULL DEFAULT 'project',
                tenant_id TEXT NOT NULL,
                git_remote_url TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(crate::project_groups_schema::CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in crate::project_groups_schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }

        pool
    }

    #[tokio::test]
    async fn test_compute_workspace_groups_cargo() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create a Cargo workspace
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"app\", \"lib\"]\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("app")).unwrap();
        fs::create_dir_all(root.join("lib")).unwrap();

        let pool = setup_pool().await;

        // Register two projects in the workspace
        let app_path = root.join("app").to_string_lossy().to_string();
        let lib_path = root.join("lib").to_string_lossy().to_string();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)",
        )
        .bind("w1")
        .bind(&app_path)
        .bind("tenant-app")
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)",
        )
        .bind("w2")
        .bind(&lib_path)
        .bind("tenant-lib")
        .execute(&pool)
        .await
        .unwrap();

        let groups = compute_workspace_groups(&pool).await.unwrap();
        assert_eq!(groups, 1);

        let members = project_groups_schema::get_group_members(&pool, "tenant-app")
            .await
            .unwrap();
        assert_eq!(members.len(), 2);
        assert!(members.contains(&"tenant-app".to_string()));
        assert!(members.contains(&"tenant-lib".to_string()));
    }

    #[tokio::test]
    async fn test_update_single_project_workspace() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"svc-a\", \"svc-b\"]\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("svc-a")).unwrap();
        fs::create_dir_all(root.join("svc-b")).unwrap();

        let pool = setup_pool().await;

        // Register svc-a first
        let svc_a_path = root.join("svc-a").to_string_lossy().to_string();
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)",
        )
        .bind("w1")
        .bind(&svc_a_path)
        .bind("tenant-a")
        .execute(&pool)
        .await
        .unwrap();

        // Then register svc-b and update groups
        let svc_b_path = root.join("svc-b").to_string_lossy().to_string();
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)",
        )
        .bind("w2")
        .bind(&svc_b_path)
        .bind("tenant-b")
        .execute(&pool)
        .await
        .unwrap();

        let added = update_project_workspace_group(&pool, "tenant-b", &root.join("svc-b"))
            .await
            .unwrap();
        assert!(added);

        // Both should be in the same group
        let members = project_groups_schema::get_group_members(&pool, "tenant-a")
            .await
            .unwrap();
        assert_eq!(members.len(), 2);
    }

    #[tokio::test]
    async fn test_no_workspace_no_group() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // No workspace manifests
        let pool = setup_pool().await;

        let path = root.to_string_lossy().to_string();
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)",
        )
        .bind("w1")
        .bind(&path)
        .bind("tenant-solo")
        .execute(&pool)
        .await
        .unwrap();

        let groups = compute_workspace_groups(&pool).await.unwrap();
        assert_eq!(groups, 0);
    }
}
