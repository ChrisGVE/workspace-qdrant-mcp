//! Show project status
//!
//! Comprehensive project status using the columnar display template.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::GetProjectStatusRequest;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::path::format_path;
use crate::output::{self, canvas, ServiceStatus};

use super::resolver::resolve_project_id_or_cwd_quiet;

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

                        if status.is_worktree {
                            if let Some(main_path) = &status.main_worktree_path {
                                builder = builder.kv("Main Working Tree", format_path(main_path));
                            }
                        }

                        // Section 2: Content and Database Status
                        builder = builder.section(Some("Project Content and Database Status"));

                        // Get file stats from SQLite
                        let stats = get_file_stats(&project_id);

                        if !stats.languages.is_empty() {
                            builder = builder.kv("Languages", stats.languages.join(", "));
                        }

                        builder = builder
                            .kv_gutter(
                                "Chunks in Database",
                                format_usize(stats.chunk_count, &locale),
                                Gutter::None,
                            )
                            .kv_gutter(
                                "Files on Disk",
                                format_usize(stats.files_on_disk, &locale),
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

/// File statistics for a project.
struct FileStats {
    languages: Vec<String>,
    chunk_count: usize,
    files_on_disk: usize,
    tracked_files: usize,
    in_sync: usize,
    to_add: usize,
    to_update: usize,
    to_remove: usize,
}

/// Get file statistics from SQLite for a project.
fn get_file_stats(project_id: &str) -> FileStats {
    let mut stats = FileStats {
        languages: Vec::new(),
        chunk_count: 0,
        files_on_disk: 0,
        tracked_files: 0,
        in_sync: 0,
        to_add: 0,
        to_update: 0,
        to_remove: 0,
    };

    let db_path = match crate::config::get_database_path_checked() {
        Ok(p) => p,
        Err(_) => return stats,
    };

    let conn = match rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(_) => return stats,
    };
    let _ = conn.execute_batch("PRAGMA busy_timeout=5000;");

    // Get tracked file count and chunk count
    if let Ok(mut stmt) = conn.prepare(
        "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = ?1 AND wf.collection = 'projects'",
    ) {
        if let Ok(row) = stmt.query_row(rusqlite::params![project_id], |row| {
            Ok((row.get::<_, usize>(0)?, row.get::<_, usize>(1)?))
        }) {
            stats.tracked_files = row.0;
            stats.chunk_count = row.1;
        }
    }

    // Get languages
    if let Ok(mut stmt) = conn.prepare(
        "SELECT DISTINCT tf.language \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = ?1 AND wf.collection = 'projects' \
         AND tf.language IS NOT NULL AND tf.language != '' \
         ORDER BY tf.language",
    ) {
        if let Ok(rows) =
            stmt.query_map(rusqlite::params![project_id], |row| row.get::<_, String>(0))
        {
            stats.languages = rows.flatten().collect();
        }
    }

    // Get reconciliation stats (to_add, to_update, to_remove)
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tf.reconcile_reason, COUNT(*) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = ?1 AND wf.collection = 'projects' \
         AND tf.needs_reconcile = 1 \
         GROUP BY tf.reconcile_reason",
    ) {
        if let Ok(rows) = stmt.query_map(rusqlite::params![project_id], |row| {
            Ok((row.get::<_, Option<String>>(0)?, row.get::<_, usize>(1)?))
        }) {
            for row in rows.flatten() {
                match row.0.as_deref() {
                    Some("new") | Some("added") => stats.to_add += row.1,
                    Some("modified") | Some("updated") | Some("content_changed") => {
                        stats.to_update += row.1
                    }
                    Some("deleted") | Some("removed") => stats.to_remove += row.1,
                    _ => stats.to_update += row.1,
                }
            }
        }
    }

    // Files in sync = tracked - needs_reconcile
    let needs_reconcile = stats.to_add + stats.to_update + stats.to_remove;
    stats.in_sync = stats.tracked_files.saturating_sub(needs_reconcile);

    // Files on disk: use tracked count as approximation (actual disk walk would be slow)
    stats.files_on_disk = stats.tracked_files;

    stats
}
