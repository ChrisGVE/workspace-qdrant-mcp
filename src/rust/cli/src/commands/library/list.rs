//! Library list subcommand
//!
//! Uses SQLite canonical queries. Table template per cli-feedback.md.

use std::collections::HashSet;

use anyhow::{Context, Result};
use tabled::builder::Builder;
use tabled::settings::object::{Columns, Rows};
use tabled::settings::peaker::PriorityMax;
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Alignment, Color, Modify, Width};

use crate::data::db::connect_readonly;
use crate::output::canvas;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::peakers::ExpandEven;
use crate::output::render::print_table_summary;
use crate::output::style::home_to_tilde;
use crate::output::table::terminal_width;
use wqm_common::constants::COLLECTION_LIBRARIES;

/// Internal row data for the library table.
struct LibraryRowData {
    gutter: Gutter,
    name: String,
    path: String,
    mode: String,
    status: String,
    documents: String,
}

/// List all libraries using table template.
pub async fn execute(verbose: bool) -> Result<()> {
    canvas::print_title("Registered Libraries");
    canvas::print_blank();

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => {
            crate::output::info("No libraries configured. Use `wqm library add` to add one.");
            return Ok(());
        }
    };

    let locale = NumberLocale::default();
    let mut rows: Vec<LibraryRowData> = Vec::new();
    let mut known_tags = HashSet::new();

    // Get document counts from SQLite (instant)
    let doc_counts = get_library_doc_counts(&conn);

    // Watch folders
    collect_watch_folders(&conn, &doc_counts, &mut known_tags, &mut rows, &locale)?;

    // Format-routed libraries from projects
    collect_format_routed(&conn, &doc_counts, &mut known_tags, &mut rows, &locale)?;

    // Orphans (in doc_counts but not in known_tags)
    collect_orphans(&doc_counts, &known_tags, &mut rows, &locale);

    if rows.is_empty() {
        crate::output::info(
            "No libraries found. Use `wqm library add` to add one, \
             or add documents to a watched project folder.",
        );
        return Ok(());
    }

    // Sort: watching first, then by name (case-insensitive)
    rows.sort_by(|a, b| {
        let a_active = a.status == "Watching";
        let b_active = b.status == "Watching";
        b_active
            .cmp(&a_active)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    let total = rows.len();
    let width = terminal_width();

    render_library_table(&rows, width, verbose);

    print_table_summary(&format!("{} libraries", total));

    Ok(())
}

/// Get document counts per library from SQLite tracked_files table.
fn get_library_doc_counts(conn: &rusqlite::Connection) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT wf.tenant_id, COUNT(tf.file_id) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE tf.collection = 'libraries' \
         GROUP BY wf.tenant_id",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        }) {
            for row in rows.flatten() {
                counts.insert(row.0, row.1);
            }
        }
    }
    counts
}

fn collect_watch_folders(
    conn: &rusqlite::Connection,
    doc_counts: &std::collections::HashMap<String, usize>,
    known_tags: &mut HashSet<String>,
    rows: &mut Vec<LibraryRowData>,
    locale: &NumberLocale,
) -> Result<usize> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT tenant_id, path, library_mode, enabled \
             FROM watch_folders WHERE collection = '{}' ORDER BY tenant_id",
            COLLECTION_LIBRARIES
        ))
        .context("Failed to query watch_folders")?;

    let libraries: Vec<(String, String, Option<String>, bool)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i32>(3)? != 0,
            ))
        })
        .context("Failed to read library rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")?;

    let count = libraries.len();
    for (tenant_id, path, mode, enabled) in &libraries {
        known_tags.insert(tenant_id.clone());
        let docs = doc_counts
            .get(tenant_id)
            .map(|c| format_usize(*c, locale))
            .unwrap_or_else(|| "0".to_string());

        let gutter = if *enabled { Gutter::Sync } else { Gutter::Add };

        rows.push(LibraryRowData {
            gutter,
            name: tenant_id.clone(),
            path: home_to_tilde(path),
            mode: mode.as_deref().unwrap_or("incremental").to_string(),
            status: if *enabled {
                "Watching".to_string()
            } else {
                "Paused".to_string()
            },
            documents: docs,
        });
    }

    Ok(count)
}

fn collect_format_routed(
    conn: &rusqlite::Connection,
    doc_counts: &std::collections::HashMap<String, usize>,
    known_tags: &mut HashSet<String>,
    rows: &mut Vec<LibraryRowData>,
    locale: &NumberLocale,
) -> Result<usize> {
    let mut stmt = conn
        .prepare(
            "SELECT wf.tenant_id, wf.path, COUNT(tf.file_id) as file_count
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
             WHERE tf.collection = 'libraries' AND wf.collection = 'projects'
             GROUP BY wf.tenant_id
             ORDER BY wf.tenant_id",
        )
        .context("Failed to query format-routed library files")?;

    let routed: Vec<(String, String, usize)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2)?,
            ))
        })
        .context("Failed to read format-routed rows")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse format-routed rows")?;

    let count = routed.len();
    for (tenant_id, path, _file_count) in &routed {
        known_tags.insert(tenant_id.clone());
        let docs = doc_counts
            .get(tenant_id)
            .map(|c| format_usize(*c, locale))
            .unwrap_or_else(|| "0".to_string());

        rows.push(LibraryRowData {
            gutter: Gutter::Info,
            name: tenant_id.clone(),
            path: home_to_tilde(path),
            mode: "auto-routed".to_string(),
            status: "Active".to_string(),
            documents: docs,
        });
    }

    Ok(count)
}

fn collect_orphans(
    doc_counts: &std::collections::HashMap<String, usize>,
    known_tags: &HashSet<String>,
    rows: &mut Vec<LibraryRowData>,
    locale: &NumberLocale,
) {
    let mut orphans: Vec<(&String, &usize)> = doc_counts
        .iter()
        .filter(|(tag, _)| !known_tags.contains(*tag))
        .collect();
    orphans.sort_by_key(|(tag, _)| (*tag).clone());

    for (tag, count) in orphans {
        rows.push(LibraryRowData {
            gutter: Gutter::Warning,
            name: tag.clone(),
            path: "—".to_string(),
            mode: "—".to_string(),
            status: "Orphan".to_string(),
            documents: format_usize(*count, locale),
        });
    }
}

fn render_library_table(rows: &[LibraryRowData], width: usize, verbose: bool) {
    if rows.is_empty() {
        return;
    }

    let table_width = width.saturating_sub(Gutter::SYMBOL_WIDTH);

    let mut builder = Builder::default();

    // Headers
    let mut headers: Vec<String> = vec![
        "Name".into(),
        "Path".into(),
        "Mode".into(),
        "Status".into(),
        "Documents".into(),
    ];
    if verbose {
        // Verbose could add more columns in the future
    }
    let _ = verbose; // suppress unused warning
    builder.push_record(headers);

    // Data rows
    for row in rows {
        let mut record: Vec<String> = vec![
            row.name.clone(),
            row.path.clone(),
            row.mode.clone(),
            row.status.clone(),
            row.documents.clone(),
        ];
        builder.push_record(record);
    }

    let mut table = builder.build();

    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));

    // Right-align Documents column (index 4)
    table.with(Modify::new(Columns::single(4)).with(Alignment::right()));

    table.with(
        Width::wrap(table_width)
            .priority::<PriorityMax>()
            .keep_words(),
    );
    table.with(Width::increase(table_width).priority::<ExpandEven>());

    let output = table.to_string();
    let lines: Vec<&str> = output.lines().collect();

    let mut data_row_idx = 0;
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            println!("{}{line}", Gutter::None.colored());
        } else if line.chars().all(|c| c == '─' || c == ' ') {
            println!("{}", "─".repeat(width));
        } else {
            let g = rows
                .get(data_row_idx)
                .map(|r| r.gutter)
                .unwrap_or(Gutter::None);
            println!("{}{line}", g.colored());
            data_row_idx += 1;
        }
    }

    println!("{}", "─".repeat(width));
}
