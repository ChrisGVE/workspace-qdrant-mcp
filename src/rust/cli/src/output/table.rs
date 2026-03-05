//! Table printing and layout helpers.
//!
//! Provides JSON, plain, script, and tabled output functions together with
//! the `ColumnHints` trait and content-aware width computation used by the
//! custom peakers in `peakers.rs`.

use serde::Serialize;
use tabled::settings::peaker::PriorityMax;
use tabled::settings::{Style, Width};
use tabled::{Table, Tabled};

use super::formatters::strip_ansi;
use super::messages::info;
use super::peakers::{
    ExpandContentOnly, ShrinkCategoricalFirst, COLUMN_MIN_WIDTHS, CONTENT_COLUMNS,
};
use crate::config::OutputFormat;

/// Cell padding used by `Style::rounded()` (1 space each side).
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

/// Get the current terminal width, falling back to 120 columns
pub fn terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(120)
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
/// 1. Wraps content to fit within terminal width (`PriorityMax` shrinks the
///    widest column first, `keep_words` breaks at word/comma boundaries).
/// 2. Stretches the table to fill the full terminal width, giving extra
///    space to the widest columns (text/content) via `PriorityMax`.
pub fn print_table<T: Tabled>(data: &[T]) {
    if data.is_empty() {
        info("No data to display");
        return;
    }

    let width = terminal_width();
    let table = Table::new(data)
        .with(Style::rounded())
        .with(Width::wrap(width).priority::<PriorityMax>().keep_words())
        .with(Width::increase(width).priority::<PriorityMax>())
        .to_string();
    println!("{}", table);
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

    let table = Table::new(data)
        .with(Style::rounded())
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

    println!("{}", table);
}

/// Trait for structs that know which of their columns are content columns.
///
/// Implement this on `Tabled` structs to enable automatic layout hints
/// via [`print_table_auto`].
pub trait ColumnHints {
    /// Returns 0-based indices of content columns (variable-length text).
    /// Empty slice means all columns are categorical (no special treatment).
    fn content_columns() -> &'static [usize];
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
        let data = vec![ScriptTestRow {
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
        let data = vec![ScriptTestRow {
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
        let data = vec![ScriptTestRow {
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
}
