//! Show project status
//!
//! Comprehensive project status using the columnar display template.

use anyhow::Result;

use crate::data::db::connect_readonly;
use crate::data::queries;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::GetProjectStatusRequest;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::path::format_path;
use crate::output::{self, canvas, ServiceStatus};

use super::resolver::resolve_project_id_or_cwd_quiet;

/// Get the current git branch for a path, if available.
fn current_git_branch(path: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", path, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch.is_empty() || branch == "HEAD" {
            None
        } else {
            Some(branch)
        }
    } else {
        None
    }
}

pub(super) async fn project_status(project: Option<&str>) -> Result<()> {
    let (project_id, auto_detected) = resolve_project_id_or_cwd_quiet(project)?;
    let locale = NumberLocale::default();

    canvas::print_title("Project Status");
    canvas::print_blank();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = GetProjectStatusRequest {
                project_id: project_id.clone(),
            };

            match client.project().get_project_status(request).await {
                Ok(response) => {
                    let status = response.into_inner();

                    if status.found {
                        // Section 1: Project Identity
                        let mut builder = ColumnarBuilder::new()
                            .kv("Project Name", &status.project_name)
                            .kv(
                                "Detection Method",
                                if auto_detected {
                                    "current working directory"
                                } else {
                                    "user argument"
                                },
                            );

                        // If CWD differs from project path, show both
                        if let Ok(cwd) = std::env::current_dir() {
                            let cwd_str = cwd.to_string_lossy().to_string();
                            if cwd_str != status.project_root {
                                builder = builder.kv("Current Directory", format_path(&cwd_str));
                                if status.is_worktree {
                                    builder = builder.kv("Worktree", "yes");
                                }
                            }
                        }

                        builder = builder
                            .kv("Project Path", format_path(&status.project_root))
                            .kv("Project Id", &status.project_id);

                        if let Some(remote) = &status.git_remote {
                            builder = builder.kv("Git Remote", remote);
                        }

                        if let Some(branch) = current_git_branch(&status.project_root) {
                            builder = builder.kv("Active Branch", branch);
                        }

                        if status.is_worktree {
                            if let Some(main_path) = &status.main_worktree_path {
                                builder = builder.kv("Main Working Tree", format_path(main_path));
                            }
                        }

                        // Section 2: Content and Database Status
                        builder = builder.section(Some("Project Content and Database Status"));

                        // Get file stats from canonical queries
                        let (stats, languages) = match connect_readonly() {
                            Ok(conn) => {
                                let s = queries::get_project_file_stats(&conn, &project_id)
                                    .unwrap_or_default();
                                let l = queries::get_languages(&conn, &project_id, "projects")
                                    .unwrap_or_default();
                                (s, l)
                            }
                            Err(_) => (queries::ReconcileStats::default(), Vec::new()),
                        };

                        if !languages.is_empty() {
                            builder = builder.kv("Languages", languages.join(", "));
                        }

                        builder = builder
                            .kv_gutter(
                                "Chunks in Database",
                                format_usize(stats.chunk_count, &locale),
                                Gutter::None,
                            )
                            .kv_gutter(
                                "Tracked Files",
                                format_usize(stats.tracked_files, &locale),
                                Gutter::None,
                            )
                            .kv_gutter(
                                "Files in Sync",
                                format_usize(stats.in_sync, &locale),
                                Gutter::Sync,
                            )
                            .kv_gutter(
                                "Files to Add",
                                format_usize(stats.to_add, &locale),
                                Gutter::Add,
                            )
                            .kv_gutter(
                                "Files to Update",
                                format_usize(stats.to_update, &locale),
                                Gutter::Update,
                            )
                            .kv_gutter(
                                "Files to Remove",
                                format_usize(stats.to_remove, &locale),
                                Gutter::Remove,
                            );

                        builder.render();

                        output::status_line(
                            "Status",
                            if status.is_active {
                                ServiceStatus::Active
                            } else {
                                ServiceStatus::Inactive
                            },
                        );
                    } else {
                        ColumnarBuilder::new()
                            .kv("Project Id", &project_id)
                            .render();
                        output::status_line("Registered", ServiceStatus::Unknown);
                        output::info("Project not registered with daemon");
                        output::info("Register with: wqm project register");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not get status: {}", e));
                }
            }
        }
        Err(_) => {
            ColumnarBuilder::new()
                .kv("Project Id", &project_id)
                .render();
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}
