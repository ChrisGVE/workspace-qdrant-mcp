//! Public entry points for content-type-aware rendering.
//!
//! `src/rust/cli/src/tui/render/content.rs`
//!
//! This module is the single hook the rest of the TUI calls for rendering text.
//! It dispatches to the right sub-renderer based on file extension or caller
//! context, and handles the binary-detection gate so callers never need to
//! inspect raw bytes themselves.
//!
//! ## Public API
//! - [`render_for_path`] — primary entry point for the file viewer: detects
//!   binary, picks markdown / code / plain based on extension.
//! - [`render_markdown`] — render a Markdown string (rules popup, scratchpad).
//! - [`render_code`] — render source text with optional language hint.
//! - [`render_plain`] — render plain text with word-wrap.

use ratatui::text::Line;

use super::code::{highlight_line, language_for_extension};
use super::markdown;

// ─── Public entry points ──────────────────────────────────────────────────────

/// Render Markdown text to styled lines, word-wrapped to `width` columns.
///
/// Used by the rules detail popup and the scratchpad detail popup to display
/// stored text with basic structure (headings, lists, inline code, etc.).
pub fn render_markdown(text: &str, width: usize) -> Vec<Line<'static>> {
    markdown::render_markdown(text, width)
}

/// Render source code with optional syntax highlighting, word-wrapped to `width`.
///
/// `language` is a canonical name as returned by
/// [`language_for_extension`] (e.g. `"rust"`, `"python"`).  Pass `None` for
/// plain rendering with no highlighting.
pub fn render_code(text: &str, language: Option<&str>, width: usize) -> Vec<Line<'static>> {
    let safe_w = width.max(10);
    let mut out = Vec::new();
    for raw in text.lines() {
        match language {
            Some(lang) => {
                // Each source line is highlighted as-is (no word-wrap: code
                // lines that exceed the viewport simply overflow — consistent
                // with every real terminal editor's behaviour).
                let _ = safe_w;
                out.push(highlight_line(raw, lang));
            }
            None => {
                // Plain fallback: emit one raw span per line.
                out.push(Line::from(raw.to_owned()));
            }
        }
    }
    out
}

/// Render plain text with word-wrap to `width` columns.
///
/// Each paragraph line is wrapped independently; blank lines are preserved as
/// paragraph breaks.
pub fn render_plain(text: &str, width: usize) -> Vec<Line<'static>> {
    use super::markdown::word_wrap;
    let safe_w = width.max(10);
    let mut out = Vec::new();
    for raw in text.lines() {
        if raw.trim().is_empty() {
            out.push(Line::from(""));
        } else {
            for wrapped in word_wrap(raw, safe_w) {
                out.push(Line::from(wrapped));
            }
        }
    }
    out
}

/// Render raw file bytes to styled lines, dispatching on binary detection and
/// file extension.
///
/// This is the single entry point the file viewer calls. Decision tree:
/// 1. Binary (NUL byte or high UTF-8 replacement-char ratio) → one-line notice.
/// 2. Extension `.md` / `.markdown` → [`render_markdown`].
/// 3. Known source extension → [`render_code`] with the detected language.
/// 4. Unknown extension → [`render_plain`].
pub fn render_for_path(path: &str, bytes: &[u8], width: usize) -> Vec<Line<'static>> {
    // ── Binary detection ────────────────────────────────────────────────────
    if bytes.contains(&0u8) {
        return binary_notice(bytes.len());
    }
    let text = String::from_utf8_lossy(bytes);
    let replacement_count = text.chars().filter(|&c| c == '\u{FFFD}').count();
    let char_count = text.chars().count();
    if char_count > 0 && replacement_count * 10 > char_count {
        return binary_notice(bytes.len());
    }

    // ── Extension dispatch ──────────────────────────────────────────────────
    let ext = file_extension(path).to_ascii_lowercase();
    match ext.as_str() {
        "md" | "markdown" => render_markdown(&text, width),
        other => match language_for_extension(other) {
            Some(lang) => render_code(&text, Some(lang), width),
            None => render_plain(&text, width),
        },
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// The text that leads a binary-file notice line. Also used in tests.
pub const BINARY_NOTICE_PREFIX: &str = "(binary file,";

/// Build the single-line binary-file notice.
fn binary_notice(byte_count: usize) -> Vec<Line<'static>> {
    use ratatui::style::Style;
    use ratatui::text::Span;
    vec![Line::from(Span::styled(
        format!("{BINARY_NOTICE_PREFIX} {byte_count} bytes)"),
        Style::default().fg(crate::tui::theme::COLOR_DIM),
    ))]
}

/// Extract the file extension from a path string, returning an empty string if
/// there is no extension.
fn file_extension(path: &str) -> &str {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── render_plain ──────────────────────────────────────────────────────────

    #[test]
    fn plain_wraps_long_line() {
        let text = "word ".repeat(30);
        let lines = render_plain(text.trim(), 20);
        for line in &lines {
            let len: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            assert!(len <= 20, "line too wide: {len}");
        }
    }

    #[test]
    fn plain_blank_line_preserved() {
        let lines = render_plain("para1\n\npara2", 80);
        let empty = lines
            .iter()
            .filter(|l| l.spans.iter().all(|s| s.content.trim().is_empty()))
            .count();
        assert!(empty >= 1);
    }

    // ── render_code ──────────────────────────────────────────────────────────

    #[test]
    fn code_rust_produces_lines_equal_to_input() {
        let src = "fn main() {\n    println!(\"hi\");\n}\n";
        let lines = render_code(src, Some("rust"), 80);
        // One ratatui Line per source line (trailing newline = empty last line).
        assert_eq!(lines.len(), src.lines().count());
    }

    #[test]
    fn code_none_language_plain_lines() {
        let src = "hello\nworld\n";
        let lines = render_code(src, None, 80);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].spans[0].content, "hello");
        assert_eq!(lines[1].spans[0].content, "world");
    }

    // ── render_markdown ───────────────────────────────────────────────────────

    #[test]
    fn markdown_heading_via_entry_point() {
        use ratatui::style::Modifier;
        let lines = render_markdown("# Title", 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
    }

    // ── render_for_path ───────────────────────────────────────────────────────

    #[test]
    fn binary_nul_gives_notice() {
        let bytes = vec![0u8; 64];
        let lines = render_for_path("file.bin", &bytes, 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].spans[0].content.starts_with(BINARY_NOTICE_PREFIX));
        assert!(lines[0].spans[0].content.contains("64"));
    }

    #[test]
    fn md_extension_uses_markdown_renderer() {
        use ratatui::style::Modifier;
        let bytes = b"# Heading\n\nSome text.";
        let lines = render_for_path("README.md", bytes, 80);
        // First line should be a bold heading span.
        assert!(lines[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
    }

    #[test]
    fn rs_extension_uses_code_renderer() {
        use crate::tui::theme;
        // A Rust source line with a comment should produce a dim span.
        let bytes = b"// this is a comment\nfn main() {}";
        let lines = render_for_path("main.rs", bytes, 80);
        assert!(!lines.is_empty());
        // First line is the comment — its first span must be dim.
        assert_eq!(lines[0].spans[0].style.fg, Some(theme::COLOR_DIM));
    }

    #[test]
    fn unknown_extension_uses_plain_renderer() {
        let bytes = b"hello world";
        let lines = render_for_path("file.xyz", bytes, 80);
        assert_eq!(lines.len(), 1);
        // Plain renderer: single raw span with no special fg.
        assert_eq!(lines[0].spans[0].content, "hello world");
    }

    #[test]
    fn empty_file_produces_empty_vec() {
        let lines = render_for_path("empty.txt", b"", 80);
        // render_plain on empty text: 0 lines (no content).
        assert!(lines.is_empty());
    }

    #[test]
    fn binary_high_replacement_ratio_gives_notice() {
        // Craft bytes that decode as many replacement chars: invalid UTF-8 sequences.
        let bytes: Vec<u8> = (0..50).map(|_| 0xFF_u8).collect();
        let lines = render_for_path("weird.dat", &bytes, 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].spans[0].content.starts_with(BINARY_NOTICE_PREFIX));
    }
}
