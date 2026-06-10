//! Output formatting module
//!
//! Provides a comprehensive formatting library for consistent CLI output:
//! - **Canvas**: Title/content/footnotes wrapper for all command output
//! - **Table**: Full-width tables with gutter, headers, separators, summaries
//! - **Columnar**: Key-value displays with sections and nesting
//! - **Gutter**: Status indicator symbols (sync, add, update, remove, etc.)
//! - **Number**: Locale-aware integer/float formatting with thousands separators
//! - **Path**: Smart path display with ~ and XDG variable substitution
//! - **Style**: Semantic color styles and layout constants
//! - **Formatters**: Pure string utilities (ANSI strip, truncate, dates, sizes)
//! - **Messages**: Simple colored terminal messages (success, error, warning, info)

#![allow(dead_code, unused_imports)]

pub mod canvas;
pub mod columnar;
mod formatters;
pub mod gutter;
mod messages;
pub mod number;
pub mod path;
pub(crate) mod peakers;
pub(crate) mod render;
pub mod style;
pub(crate) mod table;
#[cfg(test)]
mod tests;

// ─── Re-exports ───────────────────────────────────────────────────────────

// Canvas
pub use canvas::{
    print_blank, print_dim_separator, print_double_separator, print_footnote, print_separator,
    print_sized_dim_separator, print_sized_separator, print_title, title_case, Canvas,
};

// Columnar
pub use columnar::ColumnarBuilder;

// Gutter
pub use gutter::Gutter;

// Messages
pub use messages::{
    confirm, error, info, kv, section, separator, status_line, success, summary, typed_confirm,
    typed_confirm_matches, warning, ServiceStatus,
};

// Formatters
pub use formatters::{format_bytes, format_date, format_duration, strip_ansi, truncate};

// Number
pub use number::{
    format_date_short, format_float, format_integer, format_percentage, format_usize, NumberLocale,
};

// Path
pub use path::format_path;

// Style (design system)
pub use style::{
    bold_style, dim_style, error_style, info_style, short_id, short_path, success_style,
    summary_line, warning_style, COLUMN_SPACING, DEFAULT_ID_LENGTH, DEFAULT_PAGE_SIZE,
    DEFAULT_PATH_MAX, INDENT_WIDTH,
};

// Table
pub use table::{
    build_table, finish_table, print_data, print_json, print_plain, print_script, print_table,
    print_table_auto, print_table_with_hints, terminal_width, ColumnHints,
};

// Render (gutter-aware table rendering)
pub use render::{
    print_table_closing_separator, print_table_separator, print_table_summary, render_table,
    GutterRow,
};
