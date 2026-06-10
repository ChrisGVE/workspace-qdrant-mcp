//! Tests for the hand-rolled Markdown renderer.
//!
//! `src/rust/cli/src/tui/render/markdown_tests.rs`
//!
//! Included by `markdown.rs` via `#[cfg(test)] #[path = "markdown_tests.rs"] mod tests;`.

use ratatui::style::Modifier;

use super::{parse_inline, render_markdown, word_wrap};
use crate::tui::theme;

// ── Headings ──────────────────────────────────────────────────────────────────

#[test]
fn h1_heading_is_bold() {
    let lines = render_markdown("# Hello World", 80);
    assert_eq!(lines.len(), 1);
    let span = &lines[0].spans[0];
    assert!(span.style.add_modifier.contains(Modifier::BOLD));
    assert_eq!(span.content, "Hello World");
}

#[test]
fn h3_heading_renders() {
    let lines = render_markdown("### Section", 80);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].spans[0]
        .style
        .add_modifier
        .contains(Modifier::BOLD));
}

#[test]
fn non_heading_hash_renders_plain() {
    // "##no-space" is not a valid ATX heading
    let lines = render_markdown("##nospace", 80);
    // Should render as ordinary text, not a heading
    let combined: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
        .collect::<Vec<_>>()
        .join("");
    assert!(combined.contains("##nospace"));
}

// ── Fenced code blocks ────────────────────────────────────────────────────────

#[test]
fn fenced_code_block_is_dim() {
    let md = "```\nlet x = 1;\n```";
    let lines = render_markdown(md, 80);
    // The fence delimiters are stripped; only the content line remains.
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].spans[0].style.fg, Some(theme::COLOR_DIM));
    assert_eq!(lines[0].spans[0].content, "let x = 1;");
}

#[test]
fn fenced_block_multiple_lines() {
    let md = "```\nline1\nline2\n```";
    let lines = render_markdown(md, 80);
    assert_eq!(lines.len(), 2);
    for line in &lines {
        assert_eq!(line.spans[0].style.fg, Some(theme::COLOR_DIM));
    }
}

// ── Bullet lists ──────────────────────────────────────────────────────────────

#[test]
fn bullet_list_item_has_glyph() {
    let lines = render_markdown("- item one", 80);
    assert_eq!(lines.len(), 1);
    // First span should be the bullet glyph.
    assert_eq!(lines[0].spans[0].content, "• ");
}

#[test]
fn bullet_star_and_plus_work() {
    let lines_star = render_markdown("* item", 80);
    let lines_plus = render_markdown("+ item", 80);
    assert_eq!(lines_star[0].spans[0].content, "• ");
    assert_eq!(lines_plus[0].spans[0].content, "• ");
}

// ── Ordered lists ─────────────────────────────────────────────────────────────

#[test]
fn ordered_list_item_renders() {
    let lines = render_markdown("1. First item", 80);
    assert_eq!(lines.len(), 1);
    // First span is the number prefix.
    assert_eq!(lines[0].spans[0].content, "1. ");
}

// ── Blockquotes ───────────────────────────────────────────────────────────────

#[test]
fn blockquote_has_bar_prefix() {
    let lines = render_markdown("> quoted text", 80);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].spans[0].content, "▏ ");
    assert_eq!(lines[0].spans[0].style.fg, Some(theme::COLOR_DIM));
}

// ── Inline markup ─────────────────────────────────────────────────────────────

#[test]
fn inline_code_is_accent() {
    let spans = parse_inline("Use `cargo build` to compile");
    let code_span = spans.iter().find(|s| s.content == "cargo build");
    assert!(code_span.is_some(), "expected inline code span");
    assert_eq!(code_span.unwrap().style.fg, Some(theme::COLOR_ACCENT));
}

#[test]
fn bold_double_star_is_bold() {
    let spans = parse_inline("**important** text");
    let bold_span = spans.iter().find(|s| s.content == "important");
    assert!(bold_span.is_some(), "expected bold span");
    assert!(bold_span
        .unwrap()
        .style
        .add_modifier
        .contains(Modifier::BOLD));
}

#[test]
fn bold_double_underscore_is_bold() {
    let spans = parse_inline("__bold__ text");
    let bold_span = spans.iter().find(|s| s.content == "bold");
    assert!(bold_span.is_some());
    assert!(bold_span
        .unwrap()
        .style
        .add_modifier
        .contains(Modifier::BOLD));
}

#[test]
fn italic_single_star_is_italic() {
    let spans = parse_inline("*italic* text");
    let span = spans.iter().find(|s| s.content == "italic");
    assert!(span.is_some());
    assert!(span.unwrap().style.add_modifier.contains(Modifier::ITALIC));
}

#[test]
fn italic_single_underscore_is_italic() {
    let spans = parse_inline("_italic_ text");
    let span = spans.iter().find(|s| s.content == "italic");
    assert!(span.is_some());
    assert!(span.unwrap().style.add_modifier.contains(Modifier::ITALIC));
}

#[test]
fn plain_text_unchanged() {
    let spans = parse_inline("hello world");
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].content, "hello world");
}

// ── word_wrap ─────────────────────────────────────────────────────────────────

#[test]
fn word_wrap_short_line_unchanged() {
    let out = word_wrap("hello", 80);
    assert_eq!(out, vec!["hello".to_string()]);
}

#[test]
fn word_wrap_splits_at_width() {
    let out = word_wrap("one two three four", 10);
    for line in &out {
        assert!(line.chars().count() <= 10, "line too wide: {line}");
    }
    assert!(out.len() > 1);
}

#[test]
fn word_wrap_empty_gives_one_empty_line() {
    let out = word_wrap("", 80);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], "");
}

#[test]
fn word_wrap_hard_breaks_long_word() {
    let long_word = "a".repeat(20);
    let out = word_wrap(&long_word, 8);
    for line in &out {
        assert!(line.chars().count() <= 8, "line too wide: {line}");
    }
}

// ── blank line preservation ───────────────────────────────────────────────────

#[test]
fn blank_lines_preserved() {
    let md = "para one\n\npara two";
    let lines = render_markdown(md, 80);
    // The blank line should appear as an empty Line.
    let empty_count = lines
        .iter()
        .filter(|l| l.spans.iter().all(|s| s.content.trim().is_empty()))
        .count();
    assert!(empty_count >= 1, "expected at least one blank line");
}
