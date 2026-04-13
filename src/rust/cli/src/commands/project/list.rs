//! List all registered projects
//!
//! Uses SQLite canonical queries for project data and shared orphan
//! detection. No gRPC dependency — works even when daemon is down.

use anyhow::Result;
use tabled::Tabled;
use wqm_common::constants::COLLECTION_PROJECTS;

use crate::data::db::connect_readonly;
use crate::data::{orphans, queries};
use crate::output::style::home_to_tilde;
use crate::output::{self, ColumnHints};

/// Project table row
#[derive(Tabled)]
struct ProjectRow {
    #[tabled(rename = " ")]
    gutter: String,
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Path")]
    path: String,
    #[tabled(rename = "Status")]
    status: String,
    #[tabled(rename = "Documents")]
    documents: String,
}

impl ColumnHints for ProjectRow {
    fn content_columns() -> &'static [usize] {
        &[2] // Path is the content column (index shifted by gutter)
    }
}

pub(super) async fn list_projects(active_only: bool, _priority: Option<String>) -> Result<()> {
    output::section("Registered Projects");

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(e) => {
            output::error(format!("Database not available: {}", e));
            return Ok(());
        }
    };

    // Get projects from SQLite (canonical source)
    let projects = queries::get_projects(&conn).unwrap_or_default();

    // Detect orphans via shared module
    let orphan_list = orphans::detect_orphans(&conn, COLLECTION_PROJECTS)
        .await
        .unwrap_or_default();

    if projects.is_empty() && orphan_list.is_empty() {
        output::info("No projects registered. Use `wqm project register` to add one.");
        return Ok(());
    }

    let mut rows: Vec<ProjectRow> = Vec::new();

    for proj in &projects {
        if active_only && !proj.is_active {
            continue;
        }

        let doc_counts = queries::get_document_counts(&conn, &proj.tenant_id, COLLECTION_PROJECTS)
            .unwrap_or_default();

        let name = proj
            .path
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or(&proj.tenant_id)
            .to_string();

        let status = if proj.is_active {
            "Active".to_string()
        } else {
            "Inactive".to_string()
        };

        rows.push(ProjectRow {
            gutter: " ".to_string(),
            name,
            path: home_to_tilde(&proj.path),
            status,
            documents: doc_counts.tracked_files.to_string(),
        });
    }

    // Always include orphans (with warning gutter) unless active-only filter
    if !active_only {
        for orphan in &orphan_list {
            let short_id = if orphan.tenant_id.len() > 12 {
                &orphan.tenant_id[..12]
            } else {
                &orphan.tenant_id
            };
            rows.push(ProjectRow {
                gutter: "⚠".to_string(),
                name: short_id.to_string(),
                path: "-".to_string(),
                status: "Orphan".to_string(),
                documents: orphan.document_count.to_string(),
            });
        }
    }

    if !rows.is_empty() {
        output::print_table_auto(&rows);
    }

    output::separator();
    let total = rows.len();
    output::summary(output::summary_line(total, total, "projects"));

    Ok(())
}
