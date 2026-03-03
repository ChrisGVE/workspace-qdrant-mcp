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
mod table;

// ─── Re-exports ───────────────────────────────────────────────────────────

// Messages
pub use messages::{
    confirm, error, info, kv, section, separator, status_line, success, warning, ServiceStatus,
};

// Formatters
pub use formatters::{format_bytes, format_date, format_duration, strip_ansi, truncate};

// Table
pub use table::{
    print_data, print_json, print_plain, print_script, print_table, print_table_auto,
    print_table_with_hints, terminal_width, ColumnHints,
};
