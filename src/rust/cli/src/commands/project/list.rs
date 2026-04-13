//! List all registered projects
//!
//! Uses SQLite canonical queries only. No Qdrant scroll, no gRPC.
//! Instant response regardless of dataset size.
//! Table follows cli-feedback.md template: gutter, full-width,
//! closing separator, summary line.
//!
//! Optional date columns (Created, Last Scan, Last Activity) are included
//! when terminal width allows, per rule 14 — fill large gaps with useful
//! information rather than empty space.

use anyhow::Result;
use tabled::builder::Builder;
use tabled::settings::object::{Columns, Rows};
use tabled::settings::peaker::PriorityMax;
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Alignment, Color, Modify, Width};
use wqm_common::constants::COLLECTION_PROJECTS;

use crate::data::db::connect_readonly;
use crate::data::queries;
use crate::output::canvas;
use crate::output::gutter::Gutter;
use crate::output::number::{format_date_short, format_usize, NumberLocale};
use crate::output::peakers::ExpandEven;
use crate::output::style::home_to_tilde;
use crate::output::table::{print_table_summary, terminal_width};

/// Minimum terminal width to show each optional date column.
const WIDTH_CREATED: usize = 130;
const WIDTH_LAST_SCAN: usize = 150;
const WIDTH_LAST_ACTIVITY: usize = 170;

/// Internal row data (not Tabled — we use Builder for dynamic columns).
struct ProjectRowData {
    gutter: Gutter,
    name: String,
    tenant_id: String,
    path: String,
    status: String,
    documents: String,
    languages: String,
    chunks: String,
    created: String,
    last_scan: String,
    last_activity: String,
}

pub(super) async fn list_projects(
    active_only: bool,
    verbose: bool,
    _priority: Option<String>,
) -> Result<()> {
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
    let mut rows: Vec<ProjectRowData> = Vec::new();

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

        let counts = doc_counts.get(&proj.tenant_id);
        let docs = counts
            .map(|c| format_usize(c.tracked_files, &locale))
            .unwrap_or_else(|| "0".to_string());
        let chunks = counts
            .map(|c| format_usize(c.chunk_count, &locale))
            .unwrap_or_else(|| "0".to_string());

        let languages = if verbose {
            queries::get_languages(&conn, &proj.tenant_id, COLLECTION_PROJECTS)
                .unwrap_or_default()
                .join(", ")
        } else {
            String::new()
        };

        let created = proj
            .created_at
            .as_deref()
            .map(format_date_short)
            .unwrap_or_else(|| "—".to_string());

        let last_scan = proj
            .last_scan
            .as_deref()
            .map(format_date_short)
            .unwrap_or_else(|| "—".to_string());

        let last_activity = proj
            .last_activity_at
            .as_deref()
            .map(format_date_short)
            .unwrap_or_else(|| "—".to_string());

        rows.push(ProjectRowData {
            gutter: Gutter::None,
            name,
            tenant_id: proj.tenant_id.clone(),
            path: home_to_tilde(&proj.path),
            status,
            documents: docs,
            languages,
            chunks,
            created,
            last_scan,
            last_activity,
        });
    }

    // Sort: Active first, then by name (case-insensitive)
    rows.sort_by(|a, b| {
        a.status
            .cmp(&b.status)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    let total = rows.len();
    let width = terminal_width();

    // Determine which optional columns to show (rule 14: fill large gaps)
    let show_created = width >= WIDTH_CREATED;
    let show_last_scan = width >= WIDTH_LAST_SCAN;
    let show_last_activity = width >= WIDTH_LAST_ACTIVITY;

    render_project_table(
        &rows,
        width,
        verbose,
        show_created,
        show_last_scan,
        show_last_activity,
    );

    let summary = format!("{} projects", total);
    print_table_summary(&summary);

    Ok(())
}

/// Render the project table with dynamic columns and gutter prefix.
fn render_project_table(
    rows: &[ProjectRowData],
    width: usize,
    verbose: bool,
    show_created: bool,
    show_last_scan: bool,
    show_last_activity: bool,
) {
    if rows.is_empty() {
        return;
    }

    let table_width = width.saturating_sub(Gutter::SYMBOL_WIDTH);

    // Build table with dynamic columns
    let mut builder = Builder::default();

    // Track which column indices are numeric for right-alignment
    let mut numeric_cols: Vec<usize> = Vec::new();
    let mut col_idx = 0usize;

    // Header row
    let mut headers: Vec<String> = Vec::new();
    headers.push("Name".into());
    col_idx += 1; // 0
    if verbose {
        headers.push("Id".into());
        col_idx += 1; // 1
    }
    headers.push("Path".into());
    col_idx += 1;
    headers.push("Status".into());
    col_idx += 1;
    if verbose {
        headers.push("Languages".into());
        col_idx += 1;
    }
    headers.push("Documents".into());
    numeric_cols.push(col_idx);
    col_idx += 1;
    if verbose {
        headers.push("Chunks".into());
        numeric_cols.push(col_idx);
        col_idx += 1;
    }
    if show_created {
        headers.push("Created".into());
        col_idx += 1;
    }
    if show_last_scan {
        headers.push("Last Scan".into());
        col_idx += 1;
    }
    if show_last_activity {
        headers.push("Last Activity".into());
        col_idx += 1;
    }
    let _ = col_idx;
    builder.push_record(headers);

    // Data rows
    for row in rows {
        let mut record: Vec<String> = Vec::new();
        record.push(row.name.clone());
        if verbose {
            record.push(row.tenant_id.clone());
        }
        record.push(row.path.clone());
        record.push(row.status.clone());
        if verbose {
            record.push(row.languages.clone());
        }
        record.push(row.documents.clone());
        if verbose {
            record.push(row.chunks.clone());
        }
        if show_created {
            record.push(row.created.clone());
        }
        if show_last_scan {
            record.push(row.last_scan.clone());
        }
        if show_last_activity {
            record.push(row.last_activity.clone());
        }
        builder.push_record(record);
    }

    let mut table = builder.build();

    // Style: borderless with header separator
    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));

    // Right-align numeric columns
    for &col in &numeric_cols {
        table.with(Modify::new(Columns::single(col)).with(Alignment::right()));
    }

    // Width management: wrap then even spread
    table.with(
        Width::wrap(table_width)
            .priority::<PriorityMax>()
            .keep_words(),
    );
    table.with(Width::increase(table_width).priority::<ExpandEven>());

    let output = table.to_string();
    let lines: Vec<&str> = output.lines().collect();

    // Render with gutter prefix
    let mut data_row_idx = 0;
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            // Header row
            println!("{}{line}", Gutter::None.colored());
        } else if line.chars().all(|c| c == '─' || c == ' ') {
            // Separator line — extend to full width
            println!("{}", "─".repeat(width));
        } else {
            // Data row
            let g = rows
                .get(data_row_idx)
                .map(|r| r.gutter)
                .unwrap_or(Gutter::None);
            println!("{}{line}", g.colored());
            data_row_idx += 1;
        }
    }

    // Closing separator
    println!("{}", "─".repeat(width));
}
