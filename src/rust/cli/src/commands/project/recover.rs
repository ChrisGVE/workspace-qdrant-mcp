//! Reconcile (recover) a drifted project registration (#140).
//!
//! `wqm project recover [<id|path>] [--new-path <dir>] [--rescan-remote]
//! [--dry-run]`. Re-points a moved project and/or flips its tenancy, rewriting
//! stored file paths and migrating tenant-keyed data across SQLite and Qdrant.
//! The daemon does all the work in the `RecoverProject` RPC; this command only
//! resolves the target project, ships the request, and renders the result.

use std::path::PathBuf;

use anyhow::Result;

use crate::grpc::proto::RecoverProjectRequest;
use crate::output;
use crate::output::style::home_to_tilde;

use super::resolver::resolve_project_id_or_cwd;

pub(super) async fn recover_project(
    project: Option<&str>,
    new_path: Option<PathBuf>,
    rescan_remote: bool,
    dry_run: bool,
) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    output::section(if dry_run {
        "Recover Project (dry run)"
    } else {
        "Recover Project"
    });
    output::kv("Project ID", &project_id);
    if let Some(p) = new_path.as_ref() {
        output::kv("New path", &home_to_tilde(&p.display().to_string()));
    }
    if rescan_remote {
        output::kv("Rescan remote", "Yes (recompute tenancy)");
    }
    output::separator();

    let new_path_str = new_path
        .as_ref()
        .map(|p| p.display().to_string())
        .filter(|p| !p.is_empty());

    match crate::grpc::connect_default().await {
        Ok(mut client) => {
            let request = RecoverProjectRequest {
                project_id: project_id.clone(),
                new_path: new_path_str,
                rescan_remote,
                dry_run,
            };

            match client.project().recover_project(request).await {
                Ok(response) => render(response.into_inner()),
                Err(e) => {
                    let msg = e.message();
                    if msg.contains("not found") {
                        output::error(format!("Project not found: {}", project_id));
                    } else {
                        output::error(format!("Failed to recover project: {}", msg));
                    }
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

/// Render a `RecoverProjectResponse` as human-readable output.
fn render(r: crate::grpc::proto::RecoverProjectResponse) {
    if !r.changed {
        output::info(r.message);
        return;
    }

    if r.old_tenant_id != r.new_tenant_id {
        output::kv(
            "Tenancy",
            &format!("{} -> {}", r.old_tenant_id, r.new_tenant_id),
        );
    }
    if r.old_path != r.new_path {
        output::kv(
            "Path",
            &format!(
                "{} -> {}",
                home_to_tilde(&r.old_path),
                home_to_tilde(&r.new_path)
            ),
        );
    }
    output::kv("SQLite rows", &r.sqlite_rows_updated.to_string());
    output::kv("Qdrant points", &r.qdrant_points_updated.to_string());
    output::separator();

    if r.dry_run {
        output::info(r.message);
    } else {
        output::success(r.message);
    }
}
