//! Output formatting module
//!
//! Provides JSON and table output formatting with colored terminal support.
//! Used across all CLI commands for consistent output.
//!
//! Note: Some formatting functions are infrastructure for future CLI commands.

#![allow(dead_code, unused_imports)]

mod formatters;
mod messages;
mod peakers;
pub mod style;
mod table;

// ─── Re-exports ───────────────────────────────────────────────────────────

// Messages
pub use messages::{
    confirm, error, info, kv, section, separator, status_line, success, summary, warning,
    ServiceStatus,
};

// Formatters
pub use formatters::{format_bytes, format_date, format_duration, strip_ansi, truncate};

// Style (design system)
pub use style::{
    bold_style, dim_style, error_style, info_style, short_id, short_path, success_style,
    summary_line, warning_style, COLUMN_SPACING, DEFAULT_ID_LENGTH, DEFAULT_PAGE_SIZE,
    DEFAULT_PATH_MAX, INDENT_WIDTH,
};

// Table
pub use table::{
    print_data, print_json, print_plain, print_script, print_table, print_table_auto,
    print_table_with_hints, terminal_width, ColumnHints,
};
