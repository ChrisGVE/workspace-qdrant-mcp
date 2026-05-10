//! Gutter-aware table rendering.
//!
//! Provides [`render_table`] — the primary table rendering function for
//! PRD-compliant output with gutter indicators, bold title-cased headers,
//! header/closing separators, and optional summary lines.

use colored::Colorize;
use tabled::settings::object::Rows;
use tabled::settings::peaker::PriorityMax;
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Color, Modify, Width};
use tabled::{Table, Tabled};

use super::gutter::Gutter;
use super::messages::info;
use super::peakers::{ExpandEven, ShrinkCategoricalFirst, COLUMN_MIN_WIDTHS, CONTENT_COLUMNS};
use super::table::{terminal_width, ColumnHints, CELL_PADDING};

/// A row with an optional gutter indicator, for use with [`render_table`].
pub struct GutterRow<T> {
    pub gutter: Gutter,
    pub data: T,
}

/// Print a full-width separator line to stdout.
pub fn print_table_separator() {
    let width = terminal_width();
    println!("{}", "─".repeat(width));
}

/// Print a full-width closing separator (same style as header separator).
pub fn print_table_closing_separator() {
    let width = terminal_width();
    println!("{}", "─".repeat(width));
}

/// Print a summary line below a table (left-aligned, gutter-aware).
///
/// Simple summaries (e.g., "22 projects") are left-aligned under the first
/// data column (after the gutter space).
pub fn print_table_summary(text: &str) {
    // Gutter width (2) for alignment with table content
    println!("  {}", text.dimmed());
}

fn apply_width_management(
    table: &mut tabled::Table,
    table_width: usize,
    hints: &[usize],
    col_mins: Option<Vec<usize>>,
) {
    if let Some(col_mins) = col_mins {
        CONTENT_COLUMNS.with(|cc| {
            *cc.borrow_mut() = hints.to_vec();
        });
        COLUMN_MIN_WIDTHS.with(|cmw| {
            *cmw.borrow_mut() = col_mins;
        });

        table.with(
            Width::wrap(table_width)
                .priority::<ShrinkCategoricalFirst>()
                .keep_words(),
        );

        CONTENT_COLUMNS.with(|cc| cc.borrow_mut().clear());
        COLUMN_MIN_WIDTHS.with(|cmw| cmw.borrow_mut().clear());
    } else {
        table.with(
            Width::wrap(table_width)
                .priority::<PriorityMax>()
                .keep_words(),
        );
    }
    table.with(Width::increase(table_width).priority::<ExpandEven>());
}

/// Compute column min widths from a slice of references.
fn compute_column_min_widths_from_refs<T: Tabled>(
    data: &[&T],
    content_columns: &[usize],
) -> Vec<usize> {
    let headers = T::headers();
    let num_cols = headers.len();
    let mut min_widths = vec![0usize; num_cols];

    for (i, header) in headers.iter().enumerate() {
        min_widths[i] = header.len();
    }

    for row in data {
        let fields = row.fields();
        for (i, field) in fields.iter().enumerate() {
            if i >= num_cols || content_columns.contains(&i) {
                continue;
            }
            let widest = field.split_whitespace().map(|s| s.len()).max().unwrap_or(0);
            min_widths[i] = min_widths[i].max(widest);
        }
    }

    for w in &mut min_widths {
        *w += CELL_PADDING;
    }

    min_widths
}

/// Render a table with PRD-compliant formatting:
/// - Gutter column (first position)
/// - Bold, title-cased headers
/// - Header separator line (full terminal width)
/// - Closing separator line (full terminal width)
/// - Optional summary below closing line
///
/// This is the primary table rendering function for new code.
pub fn render_table<T: Tabled + ColumnHints>(rows: &[GutterRow<T>], summary: Option<&str>) {
    if rows.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();

    // Print each row with gutter prefix
    // We build the table from just the data, then prepend gutter to each output line
    let data: Vec<&T> = rows.iter().map(|r| &r.data).collect();
    let gutters: Vec<Gutter> = rows.iter().map(|r| r.gutter).collect();

    let hints = T::content_columns();
    let table_width = width.saturating_sub(Gutter::SYMBOL_WIDTH);

    // Compute min widths before Table::new consumes the data vec
    let col_mins = if !hints.is_empty() {
        Some(compute_column_min_widths_from_refs(&data, hints))
    } else {
        None
    };

    let mut table = Table::new(data);
    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));

    // Right-align numeric columns
    let numeric_cols = T::numeric_columns();
    for &col_idx in numeric_cols {
        use tabled::settings::object::Columns;
        use tabled::settings::Alignment;
        table.with(Modify::new(Columns::single(col_idx)).with(Alignment::right()));
    }

    apply_width_management(&mut table, table_width, hints, col_mins);

    let output = table.to_string();
    let lines: Vec<&str> = output.lines().collect();

    // Render with gutter: symbol (1 char) immediately before tabled's
    // left padding. Total = 1 gutter + 1 tabled padding = 2 chars before content.
    let mut data_row_idx = 0;
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            // Header row — empty gutter
            println!("{}{line}", Gutter::None.colored());
        } else if line.chars().all(|c| c == '─' || c == ' ') {
            // Separator line — extend to full width
            println!("{}", "─".repeat(width));
        } else {
            // Data row — use corresponding gutter
            let g = gutters.get(data_row_idx).copied().unwrap_or(Gutter::None);
            println!("{}{line}", g.colored());
            data_row_idx += 1;
        }
    }

    // Closing separator
    println!("{}", "─".repeat(width));

    // Summary
    if let Some(s) = summary {
        print_table_summary(s);
    }
}
