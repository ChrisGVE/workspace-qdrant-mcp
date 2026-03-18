//! CLI output design system constants and helpers.
//!
//! Provides semantic color styles, layout constants, and formatting utilities
//! for consistent terminal output across all CLI commands.

use colored::{ColoredString, Colorize};

// ─── Layout constants ────────────────────────────────────────────────────────

/// Number of characters to show for truncated hash IDs.
pub const DEFAULT_ID_LENGTH: usize = 8;

/// Maximum display width for truncated file paths.
pub const DEFAULT_PATH_MAX: usize = 40;

/// Column gap for borderless table layouts.
pub const COLUMN_SPACING: usize = 2;

/// Indentation width in spaces.
pub const INDENT_WIDTH: usize = 2;

/// Default number of items per page in list output.
pub const DEFAULT_PAGE_SIZE: usize = 50;

// ─── Semantic color styles ───────────────────────────────────────────────────

/// Green style for success, healthy, active, done.
pub fn success_style(text: &str) -> ColoredString {
    text.green()
}

/// Yellow style for warning, degraded, pending.
pub fn warning_style(text: &str) -> ColoredString {
    text.yellow()
}

/// Red style for error, unhealthy, failed.
pub fn error_style(text: &str) -> ColoredString {
    text.red()
}

/// Blue style for informational indicators.
pub fn info_style(text: &str) -> ColoredString {
    text.blue()
}

/// Dimmed style for secondary info, inactive, metadata.
pub fn dim_style(text: &str) -> ColoredString {
    text.dimmed()
}

/// Bold style for headers, labels, emphasis.
pub fn bold_style(text: &str) -> ColoredString {
    text.bold()
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Return the first [`DEFAULT_ID_LENGTH`] characters of `id`, or the full
/// string if it is shorter.
pub fn short_id(id: &str) -> String {
    if id.len() <= DEFAULT_ID_LENGTH {
        id.to_string()
    } else {
        // Find a valid char boundary at or before DEFAULT_ID_LENGTH
        let mut boundary = DEFAULT_ID_LENGTH;
        while boundary > 0 && !id.is_char_boundary(boundary) {
            boundary -= 1;
        }
        id[..boundary].to_string()
    }
}

/// Shorten a filesystem path for display.
///
/// 1. Replaces the user's home directory prefix with `~`.
/// 2. If the result exceeds `max_len`, truncates the middle with `…`,
///    keeping the last path component visible.
pub fn short_path(path: &str, max_len: usize) -> String {
    let home = dirs::home_dir()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_default();

    let shortened = if !home.is_empty() && path.starts_with(&home) {
        format!("~{}", &path[home.len()..])
    } else {
        path.to_string()
    };

    if shortened.len() <= max_len || max_len < 4 {
        return shortened;
    }

    // Keep the last path component visible.
    let last_component = shortened
        .rfind('/')
        .map(|pos| &shortened[pos..])
        .unwrap_or(&shortened);

    // If the last component alone (plus ellipsis) already exceeds max_len,
    // just truncate from the start.
    let ellipsis = "\u{2026}"; // …
    let ellipsis_len = ellipsis.len();

    if last_component.len() + ellipsis_len >= max_len {
        // Truncate from the right
        let target = max_len.saturating_sub(ellipsis_len);
        let mut boundary = target;
        while boundary > 0 && !shortened.is_char_boundary(boundary) {
            boundary -= 1;
        }
        return format!("{}{ellipsis}", &shortened[..boundary]);
    }

    // prefix_len = what we can keep from the front
    let prefix_len = max_len - ellipsis_len - last_component.len();
    let mut boundary = prefix_len;
    while boundary > 0 && !shortened.is_char_boundary(boundary) {
        boundary -= 1;
    }

    format!("{}{ellipsis}{}", &shortened[..boundary], last_component)
}

/// Produce a human-readable summary line for paginated lists.
///
/// Returns `"Showing {shown} of {total} {item_name}"` when `shown < total`,
/// or `"{total} {item_name}"` when all items are displayed.
pub fn summary_line(shown: usize, total: usize, item_name: &str) -> String {
    if shown >= total {
        format!("{total} {item_name}")
    } else {
        format!("Showing {shown} of {total} {item_name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── short_id ─────────────────────────────────────────────────────────

    #[test]
    fn short_id_long_hash() {
        let id = "a1b2c3d4e5f6a7b8";
        assert_eq!(short_id(id), "a1b2c3d4");
    }

    #[test]
    fn short_id_exact_length() {
        let id = "abcdefgh";
        assert_eq!(short_id(id), "abcdefgh");
    }

    #[test]
    fn short_id_shorter_than_limit() {
        assert_eq!(short_id("abc"), "abc");
        assert_eq!(short_id(""), "");
    }

    // ── short_path ───────────────────────────────────────────────────────

    #[test]
    fn short_path_replaces_home() {
        let home = dirs::home_dir().unwrap();
        let path = format!("{}/projects/foo/bar.rs", home.display());
        let result = short_path(&path, 80);
        assert!(result.starts_with("~/"), "expected ~ prefix, got: {result}");
        assert!(!result.contains(&home.to_string_lossy().to_string()));
    }

    #[test]
    fn short_path_truncates_middle() {
        let path = "~/projects/workspace-qdrant-mcp/src/rust/cli/src/output/style.rs";
        let result = short_path(path, 30);
        assert!(result.len() <= 34, "got len {}: {result}", result.len());
        // last component should be preserved
        assert!(result.ends_with("/style.rs"), "got: {result}");
        assert!(
            result.contains('\u{2026}'),
            "expected ellipsis, got: {result}"
        );
    }

    #[test]
    fn short_path_no_truncation_when_fits() {
        let path = "~/short/path.rs";
        assert_eq!(short_path(path, 40), path);
    }

    // ── summary_line ─────────────────────────────────────────────────────

    #[test]
    fn summary_line_all_shown() {
        assert_eq!(summary_line(3, 3, "projects"), "3 projects");
    }

    #[test]
    fn summary_line_partial() {
        assert_eq!(
            summary_line(50, 632, "queue items"),
            "Showing 50 of 632 queue items"
        );
    }

    #[test]
    fn summary_line_zero() {
        assert_eq!(summary_line(0, 0, "items"), "0 items");
    }

    // ── style functions return expected types ────────────────────────────

    #[test]
    fn style_functions_produce_output() {
        // Verify they compile and don't panic; content is colored terminal output.
        let _ = success_style("ok");
        let _ = warning_style("warn");
        let _ = error_style("err");
        let _ = info_style("info");
        let _ = dim_style("dim");
        let _ = bold_style("bold");
    }
}
