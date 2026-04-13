//! List all registered projects
//!
//! Uses SQLite canonical queries only. No Qdrant scroll, no gRPC.
//! Instant response regardless of dataset size.

use anyhow::Result;
use tabled::Tabled;
use wqm_common::constants::COLLECTION_PROJECTS;

use crate::data::db::connect_readonly;
use crate::data::queries;
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
        &[2] // Path is the content column
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

    let projects = queries::get_projects(&conn).unwrap_or_default();

    if projects.is_empty() {
        output::info("No projects registered. Use `wqm project register` to add one.");
        return Ok(());
    }

    // Batch query: all doc counts in one GROUP BY (not N queries)
    let doc_counts =
        queries::get_all_document_counts(&conn, COLLECTION_PROJECTS).unwrap_or_default();

    let mut rows: Vec<ProjectRow> = Vec::new();

    for proj in &projects {
        if active_only && !proj.is_active {
            continue;
        }

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

        let docs = doc_counts
            .get(&proj.tenant_id)
            .map(|c| c.tracked_files.to_string())
            .unwrap_or_else(|| "0".to_string());

        rows.push(ProjectRow {
            gutter: " ".to_string(),
            name,
            path: home_to_tilde(&proj.path),
            status,
            documents: docs,
        });
    }

    if !rows.is_empty() {
        output::print_table_auto(&rows);
    }

    output::separator();
    let total = rows.len();
    output::summary(output::summary_line(total, total, "projects"));

    Ok(())
}
