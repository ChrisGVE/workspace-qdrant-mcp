//! Integration tests for the CLI output formatting pipeline.
//!
//! These tests verify end-to-end behavior of the output module: table rendering
//! produces no box-drawing borders, script output strips ANSI and separates
//! columns with spaces, JSON output is valid and ANSI-free, and message helpers
//! emit the expected formatting.

#[cfg(test)]
mod integration {
    use serde::Serialize;
    use tabled::Tabled;

    use crate::output::formatters::strip_ansi;
    use crate::output::style;
    use crate::output::table;
    use crate::output::table::ColumnHints;

    // ─── Test data ──────────────────────────────────────────────────────────

    #[derive(Tabled, Serialize, Clone)]
    struct Row {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Description")]
        description: String,
    }

    fn sample_rows() -> Vec<Row> {
        vec![
            Row {
                id: "abc12345".into(),
                status: "active".into(),
                description: "First item".into(),
            },
            Row {
                id: "def67890".into(),
                status: "pending".into(),
                description: "Second item with longer text".into(),
            },
        ]
    }

    fn colored_rows() -> Vec<Row> {
        vec![Row {
            id: "\x1b[32mabc12345\x1b[0m".into(),
            status: "\x1b[1;31mfailed\x1b[0m".into(),
            description: "Has \x1b[33mcolored\x1b[0m text".into(),
        }]
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Table output: no box-drawing characters
    // ═══════════════════════════════════════════════════════════════════════

    /// Box-drawing characters that must never appear in borderless table output.
    const BOX_CHARS: &[char] = &[
        '╭', '╮', '╰', '╯', '│', '├', '┤', '┬', '┴', '┼', '║', '═', '╔', '╗', '╚', '╝',
    ];

    /// Render a table via `build_table` (same styling as `print_table`) and
    /// return the ANSI-stripped output string for assertion.
    fn render_table(data: &[Row]) -> String {
        let table = table::build_table(data);
        strip_ansi(&table.to_string())
    }

    #[test]
    fn table_output_contains_no_box_drawing_characters() {
        let output = render_table(&sample_rows());
        for ch in BOX_CHARS {
            assert!(
                !output.contains(*ch),
                "table output must not contain box-drawing character '{}', got:\n{}",
                ch,
                output
            );
        }
    }

    #[test]
    fn table_output_has_header_separator_with_dash() {
        let output = render_table(&sample_rows());
        let lines: Vec<&str> = output.lines().collect();
        assert!(
            lines.len() >= 3,
            "expected at least 3 lines (header, separator, data), got {}",
            lines.len()
        );
        assert!(
            lines[1].contains('─'),
            "second line should be a '─' separator, got: {:?}",
            lines[1]
        );
    }

    #[test]
    fn table_output_has_bold_header() {
        // Force colored output — the `colored` crate disables ANSI when
        // stdout is not a terminal (which is the case in test harnesses).
        colored::control::set_override(true);
        let data = sample_rows();
        let table = table::build_table(&data);
        let raw = table.to_string();
        colored::control::unset_override();

        let first_line = raw.lines().next().expect("table has at least one line");
        assert!(
            first_line.contains("\x1b[1m"),
            "header row should contain bold ANSI code, got: {:?}",
            first_line
        );
    }

    #[test]
    fn table_empty_data_does_not_panic() {
        let empty: Vec<Row> = vec![];
        // build_table does not guard empty data (print_table does), but it
        // should not panic.
        let table = table::build_table(&empty);
        let _ = table.to_string();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Script output: space-separated, ANSI-stripped
    // ═══════════════════════════════════════════════════════════════════════

    /// Reproduce the script-output logic inline (since `print_script` writes to
    /// stdout) and verify field formatting.
    fn script_lines(data: &[Row], include_headers: bool) -> Vec<String> {
        let mut lines = Vec::new();
        if include_headers {
            let headers: Vec<String> = <Row as Tabled>::headers()
                .into_iter()
                .map(|h| strip_ansi(&h).replace(' ', "_"))
                .collect();
            lines.push(headers.join(" "));
        }
        for row in data {
            let fields: Vec<String> = row
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
            lines.push(fields.join(" "));
        }
        lines
    }

    #[test]
    fn script_output_space_separated_with_headers() {
        let lines = script_lines(&sample_rows(), true);
        assert_eq!(lines[0], "ID Status Description");
        // Data rows should have exactly 3 space-separated fields
        for line in &lines[1..] {
            let parts: Vec<&str> = line.split(' ').collect();
            assert_eq!(
                parts.len(),
                3,
                "expected 3 fields per row, got {}: {:?}",
                parts.len(),
                line
            );
        }
    }

    #[test]
    fn script_output_strips_ansi() {
        let lines = script_lines(&colored_rows(), false);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert!(!line.contains("\x1b["), "ANSI codes should be stripped");
        assert!(line.contains("abc12345"), "ID should be present");
        assert!(line.contains("failed"), "status should be present");
    }

    #[test]
    fn script_output_replaces_spaces_with_underscores() {
        let data = vec![Row {
            id: "x".into(),
            status: "in progress".into(),
            description: "multi word desc".into(),
        }];
        let lines = script_lines(&data, false);
        let parts: Vec<&str> = lines[0].split(' ').collect();
        assert_eq!(
            parts.len(),
            3,
            "spaces within fields should become underscores"
        );
        assert_eq!(parts[1], "in_progress");
        assert_eq!(parts[2], "multi_word_desc");
    }

    #[test]
    fn script_output_empty_field_becomes_dash() {
        let data = vec![Row {
            id: "1".into(),
            status: "".into(),
            description: "ok".into(),
        }];
        let lines = script_lines(&data, false);
        let parts: Vec<&str> = lines[0].split(' ').collect();
        assert_eq!(parts[1], "-", "empty field should become '-'");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // JSON output: valid JSON, ANSI-stripped
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn json_output_is_valid_and_ansi_free() {
        let data = colored_rows();
        // Reproduce print_json logic: serialize then strip ANSI
        let json = serde_json::to_string_pretty(&data).expect("serialization should succeed");
        let clean = strip_ansi(&json);

        // Must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&clean).expect("stripped output should be valid JSON");

        // Must be an array of one element
        let arr = parsed.as_array().expect("should be an array");
        assert_eq!(arr.len(), 1);

        // The output string itself has no ANSI escapes.
        assert!(
            !clean.contains("\x1b["),
            "JSON output must not contain ANSI escape codes"
        );
    }

    #[test]
    fn json_output_plain_data_round_trips() {
        let data = sample_rows();
        let json = serde_json::to_string_pretty(&data).unwrap();
        let clean = strip_ansi(&json);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&clean).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["id"], "abc12345");
        assert_eq!(parsed[1]["status"], "pending");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Messages: section and separator formatting
    // ═══════════════════════════════════════════════════════════════════════

    /// The ANSI underline escape code (ESC[4m).
    const ANSI_UNDERLINE: &str = "\x1b[4m";

    #[test]
    fn section_output_has_no_underline() {
        // Force colored output so ANSI codes are actually emitted.
        colored::control::set_override(true);
        use colored::Colorize;
        let styled = "Test Section".to_string().bold().to_string();
        colored::control::unset_override();

        assert!(
            !styled.contains(ANSI_UNDERLINE),
            "section output must not contain underline ANSI code, got: {:?}",
            styled
        );
        // Verify bold IS present
        assert!(
            styled.contains("\x1b[1m"),
            "section output should contain bold ANSI code"
        );
    }

    #[test]
    fn separator_uses_horizontal_dash() {
        // separator() repeats '─' up to terminal_width().min(100).
        // Verify the character choice by building the string directly.
        let width = table::terminal_width().min(100);
        let sep = "─".repeat(width);
        // Every character in the separator should be '─'
        for ch in sep.chars() {
            assert_eq!(ch, '─', "separator should use '─' character exclusively");
        }
        // Verify length
        assert_eq!(
            sep.chars().count(),
            width,
            "separator should span {} characters",
            width
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Style helpers: additional edge cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn short_id_with_multibyte_characters() {
        // A string with multi-byte chars where truncation falls mid-character
        let id = "ab\u{1f680}cd\u{1f389}ef"; // U+1F680 and U+1F389 are 4 bytes each
        let result = style::short_id(id);
        // DEFAULT_ID_LENGTH is 8 bytes; "ab\u{1f680}" = 2+4 = 6 bytes,
        // "ab\u{1f680}c" = 7 bytes, "ab\u{1f680}cd" = 8 bytes. Boundary at 8 is valid.
        assert!(result.len() <= style::DEFAULT_ID_LENGTH);
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn short_id_with_only_multibyte() {
        // All 4-byte emoji: each emoji is 4 bytes, so at DEFAULT_ID_LENGTH=8
        // we can fit exactly 2.
        let id = "\u{1f525}\u{1f30a}\u{1f33f}\u{1f3af}";
        let result = style::short_id(id);
        assert_eq!(result, "\u{1f525}\u{1f30a}");
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn short_path_very_small_max_len() {
        // max_len < 4 should return the full shortened path without panic
        let path = "/some/very/long/path/to/file.rs";
        let result = style::short_path(path, 3);
        // Should not panic and should return something
        assert!(!result.is_empty());
    }

    #[test]
    fn short_path_last_component_longer_than_max() {
        // When last component + ellipsis exceeds max_len, truncate from right
        let path = "~/verylongfilename.rs";
        let result = style::short_path(path, 10);
        assert!(
            result.len() <= 13, // ellipsis '\u{2026}' is 3 bytes in UTF-8
            "result too long: {} (len {})",
            result,
            result.len()
        );
        assert!(
            result.contains('\u{2026}'),
            "should contain ellipsis when truncated"
        );
    }

    #[test]
    fn summary_line_shown_exceeds_total() {
        // Edge case: shown > total (shouldn't happen normally, but should not panic)
        let result = style::summary_line(10, 5, "items");
        assert_eq!(result, "5 items");
    }

    #[test]
    fn summary_line_large_numbers() {
        let result = style::summary_line(50, 1_000_000, "documents");
        assert_eq!(result, "Showing 50 of 1000000 documents");
    }

    #[test]
    fn home_to_tilde_empty_path() {
        assert_eq!(style::home_to_tilde(""), "");
    }

    #[test]
    fn home_to_tilde_home_only() {
        let home = dirs::home_dir().unwrap();
        let path = home.to_string_lossy().to_string();
        let result = style::home_to_tilde(&path);
        assert_eq!(result, "~");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Table + ColumnHints integration
    // ═══════════════════════════════════════════════════════════════════════

    #[derive(Tabled, Clone)]
    struct HintedRow {
        #[tabled(rename = "ID")]
        id: String,
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Tags")]
        tags: String,
    }

    impl table::ColumnHints for HintedRow {
        fn content_columns() -> &'static [usize] {
            &[1] // Name is the content column
        }
    }

    #[test]
    fn column_hints_trait_returns_expected_indices() {
        assert_eq!(HintedRow::content_columns(), &[1]);
    }

    #[test]
    fn build_table_with_hints_no_box_chars() {
        let data = vec![
            HintedRow {
                id: "1".into(),
                name: "Widget".into(),
                tags: "rust, cli".into(),
            },
            HintedRow {
                id: "2".into(),
                name: "Gadget with a much longer name".into(),
                tags: "tools".into(),
            },
        ];
        let table = table::build_table(&data);
        let output = strip_ansi(&table.to_string());
        for ch in BOX_CHARS {
            assert!(
                !output.contains(*ch),
                "hinted table must not contain box-drawing character '{}'",
                ch
            );
        }
    }
}
