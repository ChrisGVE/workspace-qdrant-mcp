//! List all registered projects
//!
//! Uses SQLite canonical queries only. No Qdrant scroll, no gRPC.
//! Instant response regardless of dataset size.
//! Table follows cli-feedback.md template: gutter, full-width,
//! closing separator, summary line.

use anyhow::Result;
use tabled::Tabled;
use wqm_common::constants::COLLECTION_PROJECTS;

use crate::data::db::connect_readonly;
use crate::data::queries;
use crate::output::canvas;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::style::home_to_tilde;
use crate::output::{render_table, ColumnHints, GutterRow};

/// Project table row (without gutter — render_table handles gutter separately)
#[derive(Tabled)]
struct ProjectRow {
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
        &[1] // Path is the content column
    }

    fn numeric_columns() -> &'static [usize] {
        &[3] // Documents is numeric (right-aligned)
    }
}

pub(super) async fn list_projects(active_only: bool, _priority: Option<String>) -> Result<()> {
    canvas::print_title("Registered Projects");
    canvas::print_blank();

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(e) => {
            crate::output::error(format!("Database not available: {}", e));
            return Ok(());
        }
    };

    let projects = queries::get_projects(&conn).unwrap_or_default();

    if projects.is_empty() {
        crate::output::info("No projects registered. Use `wqm project register` to add one.");
        return Ok(());
    }

    let doc_counts =
        queries::get_all_document_counts(&conn, COLLECTION_PROJECTS).unwrap_or_default();

    let locale = NumberLocale::default();
    let mut rows: Vec<GutterRow<ProjectRow>> = Vec::new();

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
            .map(|c| format_usize(c.tracked_files, &locale))
            .unwrap_or_else(|| "0".to_string());

        rows.push(GutterRow {
            gutter: Gutter::None,
            data: ProjectRow {
                name,
                path: home_to_tilde(&proj.path),
                status,
                documents: docs,
            },
        });
    }

    // Sort: Active first, then by name (case-insensitive)
    rows.sort_by(|a, b| {
        a.data
            .status
            .cmp(&b.data.status)
            .then_with(|| a.data.name.to_lowercase().cmp(&b.data.name.to_lowercase()))
    });

    let total = rows.len();
    let summary = format!("{} projects", total);
    render_table(&rows, Some(&summary));

    Ok(())
}
