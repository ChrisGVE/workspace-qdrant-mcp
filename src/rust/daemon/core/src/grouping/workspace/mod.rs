/// Automated project grouping by workspace membership.
///
/// Detects Cargo workspaces, npm workspaces, and Go multi-module repos,
/// then groups member projects in `project_groups` with `group_type = "workspace"`.

mod detection;

use std::collections::HashMap;
use std::path::Path;

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use super::schema;

pub use detection::{
    WorkspaceInfo, detect_cargo_workspace, detect_npm_workspace, detect_go_workspace,
};
#[cfg(test)]
pub(crate) use detection::{extract_toml_array_strings, generate_workspace_id};

// ---- Grouping logic --------------------------------------------------------

/// Recompute workspace-based groups for all registered projects.
///
/// For each project in `watch_folders`, detects workspace membership
/// and creates groups linking projects that share a workspace root.
///
/// Returns the number of workspace groups created.
pub async fn compute_workspace_groups(pool: &SqlitePool) -> Result<usize, sqlx::Error> {
    sqlx::query("DELETE FROM project_groups WHERE group_type = 'workspace'")
        .execute(pool)
        .await?;

    let rows = sqlx::query(
        r#"
        SELECT tenant_id, folder_path
        FROM watch_folders
        WHERE watch_type = 'project'
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut workspace_tenants: HashMap<String, Vec<String>> = HashMap::new();

    for row in &rows {
        let tenant_id: String = row.get("tenant_id");
        let folder_path: String = row.get("folder_path");
        let project_path = Path::new(&folder_path);

        let workspace = detect_cargo_workspace(project_path)
            .or_else(|| detect_npm_workspace(project_path))
            .or_else(|| detect_go_workspace(project_path));

        if let Some(ws) = workspace {
            let is_member = ws.members.iter().any(|m| {
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

    let groups_created = create_workspace_groups(pool, &workspace_tenants).await?;

    info!(
        projects = rows.len(),
        workspaces = workspace_tenants.len(),
        groups = groups_created,
        "Workspace group computation complete"
    );

    Ok(groups_created)
}

/// Write workspace groups to the database for all workspaces with 2+ members.
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
    schema::add_to_group(pool, &group_id, tenant_id, "workspace", 1.0).await?;

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

        let is_member = ws.members.iter().any(|m| {
            peer_path.starts_with(m) || m.starts_with(peer_path)
        }) || peer_path.starts_with(&ws.workspace_root);

        if is_member {
            schema::add_to_group(pool, &group_id, &peer_tenant, "workspace", 1.0).await?;
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
