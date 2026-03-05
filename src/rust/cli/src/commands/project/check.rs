//! Check ingestion status: compare tracked files against filesystem

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use colored::Colorize;
use serde::Serialize;
use walkdir::WalkDir;

use crate::output;

use super::resolver::resolve_project_id_or_cwd_quiet;

/// File check status
#[derive(Debug, Clone, Serialize)]
pub struct FileCheckEntry {
    pub path: String,
    pub status: &'static str,
}

/// Summary of check results
#[derive(Debug, Serialize)]
pub struct CheckSummary {
    pub project_id: String,
    pub project_root: String,
    pub up_to_date: u64,
    pub to_add: u64,
    pub to_update: u64,
    pub to_delete: u64,
    pub total_tracked: u64,
    pub total_on_disk: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileCheckEntry>,
}

pub(super) async fn check_project(project: Option<&str>, verbose: bool, json: bool) -> Result<()> {
    let (project_id, auto_detected) = resolve_project_id_or_cwd_quiet(project)?;
    if auto_detected && !json {
        output::info(format!("Auto-detected project: {}", project_id));
    }

    let db_path = crate::config::get_database_path_checked()
        .map_err(|e| anyhow::anyhow!("Database not found: {}", e))?;

    let conn = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context("Failed to open state database")?;

    let (watch_id, project_root) = find_watch_folder(&conn, &project_id, json)?;
    let (watch_id, project_root) = match (watch_id, project_root) {
        (Some(w), Some(p)) => (w, p),
        _ => return Ok(()),
    };

    let tracked_files = load_tracked_files(&conn, &watch_id)?;
    let disk_files = scan_disk_files(&project_root, json)?;
    let disk_files = match disk_files {
        Some(f) => f,
        None => return Ok(()),
    };

    let (up_to_date, to_add, to_update, to_delete) =
        compare_files(&tracked_files, &disk_files, &project_root);

    let mut files = Vec::new();
    if verbose || json {
        for p in &to_add {
            files.push(FileCheckEntry {
                path: p.clone(),
                status: "add",
            });
        }
        for p in &to_update {
            files.push(FileCheckEntry {
                path: p.clone(),
                status: "update",
            });
        }
        for p in &to_delete {
            files.push(FileCheckEntry {
                path: p.clone(),
                status: "delete",
            });
        }
    }

    let summary = CheckSummary {
        project_id: project_id.clone(),
        project_root: project_root.clone(),
        up_to_date,
        to_add: to_add.len() as u64,
        to_update: to_update.len() as u64,
        to_delete: to_delete.len() as u64,
        total_tracked: tracked_files.len() as u64,
        total_on_disk: disk_files.len() as u64,
        files,
    };

    if json {
        output::print_json(&summary);
    } else {
        print_check_summary(&summary, verbose, &to_add, &to_update, &to_delete);
    }

    Ok(())
}

/// Find the watch_folder row for a project. Returns (None, None) if not found
/// (after printing an error).
fn find_watch_folder(
    conn: &rusqlite::Connection,
    project_id: &str,
    json: bool,
) -> Result<(Option<String>, Option<String>)> {
    let result: Result<(String, String), _> = conn.query_row(
        "SELECT watch_id, path FROM watch_folders \
         WHERE tenant_id = ?1 AND collection = 'projects' LIMIT 1",
        rusqlite::params![project_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    match result {
        Ok((w, p)) => Ok((Some(w), Some(p))),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            if json {
                output::print_json(&serde_json::json!({
                    "error": "Project not found",
                    "project_id": project_id
                }));
            } else {
                output::error(format!("Project not found: {}", project_id));
            }
            Ok((None, None))
        }
        Err(e) => Err(e.into()),
    }
}

/// Load all tracked files from SQLite for a watch folder.
fn load_tracked_files(
    conn: &rusqlite::Connection,
    watch_id: &str,
) -> Result<HashMap<String, String>> {
    let mut stmt =
        conn.prepare("SELECT file_path, file_hash FROM tracked_files WHERE watch_folder_id = ?1")?;
    let rows: Vec<(String, String)> = stmt
        .query_map(rusqlite::params![watch_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows.into_iter().collect())
}

/// Walk the filesystem and return the set of eligible relative paths.
/// Returns None if the project root does not exist (after printing error).
fn scan_disk_files(project_root: &str, json: bool) -> Result<Option<HashSet<String>>> {
    let project_root_path = std::path::Path::new(project_root);
    if !project_root_path.exists() {
        if json {
            output::print_json(&serde_json::json!({
                "error": "Project root does not exist",
                "project_root": project_root,
            }));
        } else {
            output::error(format!("Project root does not exist: {}", project_root));
        }
        return Ok(None);
    }

    let allowed = workspace_qdrant_core::allowed_extensions::AllowedExtensions::default();
    let mut disk_files: HashSet<String> = HashSet::new();

    for entry in WalkDir::new(project_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !workspace_qdrant_core::patterns::exclusion::should_exclude_directory(&name)
            } else {
                true
            }
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let abs_path = entry.path();
        let rel_path = match abs_path.strip_prefix(project_root_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        if workspace_qdrant_core::patterns::exclusion::should_exclude_file(&rel_path) {
            continue;
        }
        if !allowed.is_allowed(&rel_path, "projects") {
            continue;
        }

        disk_files.insert(rel_path);
    }

    Ok(Some(disk_files))
}

/// Compare tracked files against disk files. Returns (up_to_date, to_add,
/// to_update, to_delete) with the add/update/delete vectors sorted.
fn compare_files(
    tracked_files: &HashMap<String, String>,
    disk_files: &HashSet<String>,
    project_root: &str,
) -> (u64, Vec<String>, Vec<String>, Vec<String>) {
    let project_root_path = std::path::Path::new(project_root);
    let mut up_to_date: u64 = 0;
    let mut to_update: Vec<String> = Vec::new();
    let mut to_delete: Vec<String> = Vec::new();
    let mut to_add: Vec<String> = Vec::new();

    for (rel_path, tracked_hash) in tracked_files {
        if disk_files.contains(rel_path) {
            let abs = project_root_path.join(rel_path);
            match wqm_common::hashing::compute_file_hash(&abs) {
                Ok(disk_hash) => {
                    if disk_hash == *tracked_hash {
                        up_to_date += 1;
                    } else {
                        to_update.push(rel_path.clone());
                    }
                }
                Err(_) => {
                    to_update.push(rel_path.clone());
                }
            }
        } else {
            to_delete.push(rel_path.clone());
        }
    }

    for rel_path in disk_files {
        if !tracked_files.contains_key(rel_path) {
            to_add.push(rel_path.clone());
        }
    }

    to_add.sort();
    to_update.sort();
    to_delete.sort();

    (up_to_date, to_add, to_update, to_delete)
}

fn print_check_summary(
    summary: &CheckSummary,
    verbose: bool,
    to_add: &[String],
    to_update: &[String],
    to_delete: &[String],
) {
    output::section(format!("Project Check: {}", summary.project_id));
    output::kv("Path", &summary.project_root);
    output::separator();

    output::kv("Tracked files", &summary.total_tracked.to_string());
    output::kv("Files on disk", &summary.total_on_disk.to_string());
    output::separator();

    if summary.up_to_date > 0 {
        println!(
            "  {} {}",
            "=".green(),
            format!("{} up to date", summary.up_to_date)
        );
    }
    if summary.to_add > 0 {
        println!(
            "  {} {}",
            "+".yellow(),
            format!("{} to add", summary.to_add)
        );
    }
    if summary.to_update > 0 {
        println!(
            "  {} {}",
            "~".blue(),
            format!("{} to update", summary.to_update)
        );
    }
    if summary.to_delete > 0 {
        println!(
            "  {} {}",
            "-".red(),
            format!("{} to delete", summary.to_delete)
        );
    }

    if summary.to_add == 0 && summary.to_update == 0 && summary.to_delete == 0 {
        output::success("All tracked files are up to date");
    }

    if verbose {
        print_verbose_files(to_add, to_update, to_delete);
    }
}

fn print_verbose_files(to_add: &[String], to_update: &[String], to_delete: &[String]) {
    if !to_add.is_empty() {
        output::separator();
        println!("{}", "Files to add:".bold());
        for p in to_add {
            println!("  {} {}", "+".yellow(), p);
        }
    }
    if !to_update.is_empty() {
        output::separator();
        println!("{}", "Files to update:".bold());
        for p in to_update {
            println!("  {} {}", "~".blue(), p);
        }
    }
    if !to_delete.is_empty() {
        output::separator();
        println!("{}", "Files to delete:".bold());
        for p in to_delete {
            println!("  {} {}", "-".red(), p);
        }
    }
}
