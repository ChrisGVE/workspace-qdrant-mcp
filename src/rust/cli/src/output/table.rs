//! Table printing and layout helpers.
//!
//! Provides JSON, plain, script, and tabled output functions together with
//! the `ColumnHints` trait and content-aware width computation used by the
//! custom peakers in `peakers.rs`.

use colored::Colorize;
use serde::Serialize;
use tabled::settings::object::Rows;
use tabled::settings::peaker::PriorityMax;
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Color, Modify, Width};
use tabled::{Table, Tabled};

use super::canvas::title_case;
use super::formatters::strip_ansi;
use super::gutter::Gutter;
use super::messages::info;
use super::peakers::{
    ExpandContentOnly, ExpandEven, ShrinkCategoricalFirst, COLUMN_MIN_WIDTHS, CONTENT_COLUMNS,
};
use crate::config::OutputFormat;

/// Cell padding used by tabled's default layout (1 space each side).
const CELL_PADDING: usize = 2;

/// Compute content-aware minimum widths for each column.
///
/// For categorical columns: minimum = max(header width, widest word/segment)
/// where words are split at commas and whitespace boundaries across all cells.
/// This ensures categorical columns never shrink below their widest atomic value.
///
/// For content columns: minimum = header width (flexible, gets extra space).
///
/// The returned widths include `CELL_PADDING` to match tabled's internal
/// column width representation (content + padding).
fn compute_column_min_widths<T: Tabled>(data: &[T], content_columns: &[usize]) -> Vec<usize> {
    let headers = T::headers();
    let num_cols = headers.len();
    let mut min_widths = vec![0usize; num_cols];

    // Start with header widths for all columns
    for (i, header) in headers.iter().enumerate() {
        min_widths[i] = header.len();
    }

    // For categorical columns, find widest word/segment across all cells.
    // Split on whitespace only — this matches how tabled's `split_keeping_words`
    // processes text, so trailing punctuation (commas, semicolons) is included
    // in the word width measurement.
    for row in data {
        let fields = row.fields();
        for (i, field) in fields.iter().enumerate() {
            if i >= num_cols || content_columns.contains(&i) {
                continue; // Content columns keep header width as min
            }
            let widest = field.split_whitespace().map(|s| s.len()).max().unwrap_or(0);
            min_widths[i] = min_widths[i].max(widest);
        }
    }

    // Add cell padding — tabled's peaker widths include padding
    for w in &mut min_widths {
        *w += CELL_PADDING;
    }

    min_widths
}

/// Apply borderless formatting to a table: no outer borders, bold headers,
/// a dim `─` separator under the header, and space-separated columns.
fn apply_borderless(table: &mut Table) {
    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));
}

/// Minimum terminal width to produce readable output.
const MIN_TERMINAL_WIDTH: usize = 40;

/// Default width when stdout is not a terminal (piped output).
const DEFAULT_PIPE_WIDTH: usize = 120;

/// Get the current terminal width.
///
/// Returns the actual terminal width when stdout is a TTY, clamped to
/// at least `MIN_TERMINAL_WIDTH`. Falls back to `DEFAULT_PIPE_WIDTH`
/// when stdout is piped or not a terminal.
pub fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| (w.0 as usize).max(MIN_TERMINAL_WIDTH))
        .unwrap_or(DEFAULT_PIPE_WIDTH)
}

/// Format and print data based on output format
pub fn print_data<T: Serialize + Tabled>(data: &[T], format: OutputFormat) {
    match format {
        OutputFormat::Json => print_json(data),
        OutputFormat::Table => print_table(data),
        OutputFormat::Plain => print_plain(data),
        OutputFormat::Script => print_script(data, true),
    }
}

/// Print data as JSON
///
/// ANSI escape codes are stripped from the output to prevent colored fields
/// from leaking terminal sequences into the JSON.
pub fn print_json<T: Serialize + ?Sized>(data: &T) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => println!("{}", strip_ansi(&json)),
        Err(e) => super::messages::error(format!("Failed to serialize to JSON: {}", e)),
    }
}

/// Print data in script-friendly format: space-separated columns, one row per line.
///
/// Uses `Tabled::headers()` for the header row and `Tabled::fields()` for data rows.
/// All ANSI escape codes are stripped. Suitable for piping through `awk`, `cut`, `grep`.
pub fn print_script<T: Tabled>(data: &[T], include_headers: bool) {
    if include_headers {
        let headers: Vec<String> = T::headers()
            .into_iter()
            .map(|h| strip_ansi(&h).replace(' ', "_"))
            .collect();
        println!("{}", headers.join(" "));
    }

    for row in data {
        let fields: Vec<String> = row
            .fields()
            .into_iter()
            .map(|f| {
                let clean = strip_ansi(&f);
                // Replace any whitespace in field values with underscores
                // so each field is a single "word" for awk/cut
                if clean.contains(' ') {
                    clean.replace(' ', "_")
                } else if clean.is_empty() {
                    "-".to_string()
                } else {
                    clean
                }
            })
            .collect();
        println!("{}", fields.join(" "));
    }
}

/// Print data as a formatted table that fills the full terminal width.
///
/// Uses a borderless layout with bold headers and a dim separator line.
/// Wraps content to fit within terminal width (`PriorityMax` shrinks the
/// widest column first, `keep_words` breaks at word/comma boundaries),
/// then stretches to fill the full terminal width.
pub fn print_table<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();
    let mut table = Table::new(data);
    apply_borderless(&mut table);
    let output = table
        .with(Width::wrap(width).priority::<PriorityMax>().keep_words())
        .with(Width::increase(width).priority::<PriorityMax>())
        .to_string();
    println!("{}", output);
}

/// Print data as plain text (one item per line)
pub fn print_plain<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        return;
    }

    // Use table but with minimal style
    let table = Table::new(data).with(Style::blank()).to_string();
    println!("{}", table);
}

/// Print a table with column layout hints for optimal width distribution.
///
/// `content_columns` lists 0-based indices of columns that contain
/// variable-length text (titles, descriptions, paths). These columns
/// receive the most available width. All other columns are treated as
/// categorical (IDs, dates, enums, comma-delimited tags) and are
/// constrained to the width of their widest atomic value (word/segment).
///
/// Extra space is distributed exclusively to content columns.
pub fn print_table_with_hints<T: Tabled>(data: &[T], content_columns: &[usize]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();

    // Compute content-aware minimums and set thread-locals for peakers
    let col_mins = compute_column_min_widths(data, content_columns);
    CONTENT_COLUMNS.with(|cc| {
        *cc.borrow_mut() = content_columns.to_vec();
    });
    COLUMN_MIN_WIDTHS.with(|cmw| {
        *cmw.borrow_mut() = col_mins;
    });

    let mut table = Table::new(data);
    apply_borderless(&mut table);
    let output = table
        .with(
            Width::wrap(width)
                .priority::<ShrinkCategoricalFirst>()
                .keep_words(),
        )
        .with(Width::increase(width).priority::<ExpandContentOnly>())
        .to_string();

    // Clean up thread-locals
    CONTENT_COLUMNS.with(|cc| cc.borrow_mut().clear());
    COLUMN_MIN_WIDTHS.with(|cmw| cmw.borrow_mut().clear());

    println!("{}", output);
}

/// Trait for structs that know which of their columns are content columns.
///
/// Implement this on `Tabled` structs to enable automatic layout hints
/// via [`print_table_auto`].
pub trait ColumnHints {
    /// Returns 0-based indices of content columns (variable-length text).
    /// Empty slice means all columns are categorical (no special treatment).
    fn content_columns() -> &'static [usize];

    /// Returns 0-based indices of numeric columns (right-aligned).
    /// Default: empty (all left-aligned).
    fn numeric_columns() -> &'static [usize] {
        &[]
    }
}

/// Print a table with layout hints derived from the `ColumnHints` trait.
///
/// Delegates to [`print_table_with_hints`] when content columns exist,
/// or [`print_table`] when all columns are categorical.
pub fn print_table_auto<T: Tabled + ColumnHints>(data: &[T]) {
    let hints = T::content_columns();
    if hints.is_empty() {
        print_table(data);
    } else {
        print_table_with_hints(data, hints);
    }
}

/// Build a [`Table`] with borderless styling applied but no width adjustments.
///
/// Callers can apply additional modifications (e.g. `Disable::column`) before
/// calling [`finish_table`] for final width layout and printing.
pub fn build_table<T: Tabled>(data: &[T]) -> Table {
    let mut table = Table::new(data);
    apply_borderless(&mut table);
    table
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

/// A row with an optional gutter indicator, for use with the enhanced table.
pub struct GutterRow<T> {
    pub gutter: Gutter,
    pub data: T,
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

    // Apply width management: shrink content columns first when too wide,
    // but distribute extra width evenly across ALL columns (rule 14: even spread).
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
    // Even spread: distribute extra width across all columns (rule 14)
    table.with(Width::increase(table_width).priority::<ExpandEven>());

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

/// Apply terminal-width layout to a table and print it.
///
/// When `content_columns` is non-empty, uses content-aware peakers that
/// give priority to those columns. Otherwise falls back to `PriorityMax`.
pub fn finish_table<T: Tabled>(table: &mut Table, data: &[T], content_columns: &[usize]) {
    let width = terminal_width();

    if content_columns.is_empty() {
        let output = table
            .with(Width::wrap(width).priority::<PriorityMax>().keep_words())
            .with(Width::increase(width).priority::<PriorityMax>())
            .to_string();
        println!("{}", output);
    } else {
        let col_mins = compute_column_min_widths(data, content_columns);
        CONTENT_COLUMNS.with(|cc| {
            *cc.borrow_mut() = content_columns.to_vec();
        });
        COLUMN_MIN_WIDTHS.with(|cmw| {
            *cmw.borrow_mut() = col_mins;
        });

        let output = table
            .with(
                Width::wrap(width)
                    .priority::<ShrinkCategoricalFirst>()
                    .keep_words(),
            )
            .with(Width::increase(width).priority::<ExpandContentOnly>())
            .to_string();

        CONTENT_COLUMNS.with(|cc| cc.borrow_mut().clear());
        COLUMN_MIN_WIDTHS.with(|cmw| cmw.borrow_mut().clear());

        println!("{}", output);
    }
}

#[cfg(test)]
mod tests {
    use tabled::Tabled;

    use super::*;
    use crate::output::formatters::strip_ansi;

    /// Test struct for print_script
    #[derive(Tabled)]
    struct ScriptTestRow {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Name")]
        name: String,
    }

    #[test]
    fn test_print_script_with_headers() {
        // Capture output by testing the logic directly
        let data = [ScriptTestRow {
            id: "1".into(),
            status: "done".into(),
            name: "test".into(),
        }];
        let headers: Vec<String> = <ScriptTestRow as Tabled>::headers()
            .into_iter()
            .map(|h| strip_ansi(&h).replace(' ', "_"))
            .collect();
        assert_eq!(headers.join(" "), "ID Status Name");

        let fields: Vec<String> = data[0]
            .fields()
            .into_iter()
            .map(|f| strip_ansi(&f))
            .collect();
        assert_eq!(fields.join(" "), "1 done test");
    }

    #[test]
    fn test_print_script_strips_ansi_from_fields() {
        let data = [ScriptTestRow {
            id: "abc".into(),
            status: "\x1b[31mfailed\x1b[0m".into(),
            name: "my item".into(),
        }];
        let fields: Vec<String> = data[0]
            .fields()
            .into_iter()
            .map(|f| {
                let clean = strip_ansi(&f);
                if clean.contains(' ') {
                    clean.replace(' ', "_")
                } else if clean.is_empty() {
                    "-".to_string()
                } else {
                    clean
                }
            })
            .collect();
        assert_eq!(fields[0], "abc");
        assert_eq!(fields[1], "failed"); // ANSI stripped
        assert_eq!(fields[2], "my_item"); // space replaced with underscore
    }

    #[test]
    fn test_print_script_empty_field_becomes_dash() {
        let data = [ScriptTestRow {
            id: "1".into(),
            status: "".into(),
            name: "ok".into(),
        }];
        let fields: Vec<String> = data[0]
            .fields()
            .into_iter()
            .map(|f| {
                let clean = strip_ansi(&f);
                if clean.is_empty() {
                    "-".to_string()
                } else {
                    clean
                }
            })
            .collect();
        assert_eq!(fields[1], "-");
    }

    /// Test struct for column min width computation
    #[derive(Tabled)]
    struct TestRow {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Title")]
        title: String,
        #[tabled(rename = "Tags")]
        tags: String,
    }

    #[test]
    fn test_compute_column_min_widths_header_floor() {
        // When cells are shorter than headers, header width + padding is the minimum
        let data = vec![TestRow {
            id: "1".into(),
            title: "Hi".into(),
            tags: "a".into(),
        }];
        // Title(1) is content column
        let mins = compute_column_min_widths(&data, &[1]);
        // All values include CELL_PADDING (2)
        assert_eq!(mins[0], 2 + CELL_PADDING); // "ID" header = 2
        assert_eq!(mins[1], 5 + CELL_PADDING); // "Title" header = 5 (content col)
        assert_eq!(mins[2], 4 + CELL_PADDING); // "Tags" header = 4 > "a" = 1
    }

    #[test]
    fn test_compute_column_min_widths_whitespace_split() {
        // Splits on whitespace only — trailing punctuation stays with the word
        let data = vec![TestRow {
            id: "abc".into(),
            title: "A long title".into(),
            tags: "short, verylongtag, mid".into(),
        }];
        let mins = compute_column_min_widths(&data, &[1]);
        assert_eq!(mins[0], 3 + CELL_PADDING); // "abc" > "ID"(2)
        assert_eq!(mins[1], 5 + CELL_PADDING); // content column, stays at header width
                                               // "verylongtag," (with trailing comma) = 12 > "Tags"(4)
        assert_eq!(mins[2], 12 + CELL_PADDING);
    }

    #[test]
    fn test_compute_column_min_widths_across_rows() {
        // Min width should be the max across all rows
        let data = vec![
            TestRow {
                id: "1".into(),
                title: "x".into(),
                tags: "a, b".into(),
            },
            TestRow {
                id: "42".into(),
                title: "y".into(),
                tags: "longvalue,".into(),
            },
            TestRow {
                id: "7".into(),
                title: "z".into(),
                tags: "c".into(),
            },
        ];
        let mins = compute_column_min_widths(&data, &[1]);
        assert_eq!(mins[0], 2 + CELL_PADDING); // "42" = 2, "ID" = 2
                                               // "longvalue," = 10 (no whitespace, so entire string is one word)
        assert_eq!(mins[2], 10 + CELL_PADDING);
    }

    #[test]
    fn test_borderless_style_has_no_outer_borders() {
        let data = vec![TestRow {
            id: "1".into(),
            title: "Hello".into(),
            tags: "a".into(),
        }];
        let mut table = Table::new(&data);
        apply_borderless(&mut table);
        let output = table.to_string();
        let clean = strip_ansi(&output);

        // No box-drawing border characters
        assert!(!clean.contains('╭'), "should not contain top-left corner");
        assert!(!clean.contains('╮'), "should not contain top-right corner");
        assert!(
            !clean.contains('╰'),
            "should not contain bottom-left corner"
        );
        assert!(
            !clean.contains('╯'),
            "should not contain bottom-right corner"
        );
        assert!(!clean.contains('│'), "should not contain vertical border");
        assert!(!clean.contains('├'), "should not contain left tee");
        assert!(!clean.contains('┤'), "should not contain right tee");
        assert!(!clean.contains('┼'), "should not contain cross");
    }

    #[test]
    fn test_borderless_style_has_header_separator() {
        let data = vec![TestRow {
            id: "1".into(),
            title: "Hello".into(),
            tags: "a".into(),
        }];
        let mut table = Table::new(&data);
        apply_borderless(&mut table);
        let output = table.to_string();
        let clean = strip_ansi(&output);

        // Should have a separator line made of '─' characters
        let lines: Vec<&str> = clean.lines().collect();
        assert!(
            lines.len() >= 3,
            "expected at least 3 lines (header, sep, data)"
        );
        // Second line should be the separator
        assert!(
            lines[1].contains('─'),
            "separator line should contain ─, got: {:?}",
            lines[1]
        );
    }

    #[test]
    fn test_borderless_style_bold_header() {
        let data = vec![TestRow {
            id: "1".into(),
            title: "Hello".into(),
            tags: "a".into(),
        }];
        let mut table = Table::new(&data);
        apply_borderless(&mut table);
        let output = table.to_string();

        // Header row should contain bold ANSI escape codes
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("\x1b[1m"),
            "header should contain bold ANSI code"
        );
    }
}
