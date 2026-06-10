//! Tests for the code syntax highlighter.
//!
//! `src/rust/cli/src/tui/render/code_tests.rs`
//!
//! Included by `code.rs` via `#[cfg(test)] #[path = "code_tests.rs"] mod tests;`.

use super::{highlight_line, language_for_extension};
use crate::tui::theme;

// ── extension → language ──────────────────────────────────────────────────

#[test]
fn known_extensions_resolve() {
    assert_eq!(language_for_extension("rs"), Some("rust"));
    assert_eq!(language_for_extension("py"), Some("python"));
    assert_eq!(language_for_extension("go"), Some("go"));
    assert_eq!(language_for_extension("ts"), Some("typescript"));
    assert_eq!(language_for_extension("js"), Some("javascript"));
    assert_eq!(language_for_extension("lua"), Some("lua"));
    assert_eq!(language_for_extension("sql"), Some("sql"));
    assert_eq!(language_for_extension("toml"), Some("toml"));
    assert_eq!(language_for_extension("yaml"), Some("yaml"));
    assert_eq!(language_for_extension("yml"), Some("yaml"));
    assert_eq!(language_for_extension("json"), Some("json"));
    assert_eq!(language_for_extension("sh"), Some("shell"));
}

#[test]
fn unknown_extension_returns_none() {
    assert_eq!(language_for_extension("xyz"), None);
    assert_eq!(language_for_extension(""), None);
}

// ── comment detection ─────────────────────────────────────────────────────

#[test]
fn rust_line_comment_yields_dim_span() {
    let line = highlight_line("// this is a comment", "rust");
    assert_eq!(line.spans.len(), 1);
    assert_eq!(line.spans[0].style.fg, Some(theme::COLOR_DIM));
}

#[test]
fn python_hash_comment_is_dim() {
    let line = highlight_line("# python comment", "python");
    assert_eq!(line.spans.len(), 1);
    assert_eq!(line.spans[0].style.fg, Some(theme::COLOR_DIM));
}

#[test]
fn lua_double_dash_comment_is_dim() {
    let line = highlight_line("-- lua comment", "lua");
    assert_eq!(line.spans[0].style.fg, Some(theme::COLOR_DIM));
}

// ── keyword detection ─────────────────────────────────────────────────────

#[test]
fn rust_fn_keyword_is_accented() {
    let line = highlight_line("fn main() {", "rust");
    let kw_span = line.spans.iter().find(|s| s.content == "fn");
    assert!(kw_span.is_some(), "expected 'fn' keyword span");
    let kw = kw_span.unwrap();
    assert_eq!(kw.style.fg, Some(theme::COLOR_ACCENT));
}

#[test]
fn python_def_keyword_is_accented() {
    let line = highlight_line("def hello():", "python");
    let kw_span = line.spans.iter().find(|s| s.content == "def");
    assert!(kw_span.is_some());
    assert_eq!(kw_span.unwrap().style.fg, Some(theme::COLOR_ACCENT));
}

// ── string literals ───────────────────────────────────────────────────────

#[test]
fn double_quoted_string_is_success_color() {
    let line = highlight_line(r#"let x = "hello world";"#, "rust");
    let str_span = line
        .spans
        .iter()
        .find(|s| s.content.contains("hello world"));
    assert!(str_span.is_some(), "expected string literal span");
    assert_eq!(str_span.unwrap().style.fg, Some(theme::COLOR_SUCCESS));
}

#[test]
fn single_quoted_string_is_success_color() {
    let line = highlight_line("x = 'hello'", "python");
    let str_span = line.spans.iter().find(|s| s.content.contains("hello"));
    assert!(str_span.is_some());
    assert_eq!(str_span.unwrap().style.fg, Some(theme::COLOR_SUCCESS));
}

// ── plain text ────────────────────────────────────────────────────────────

#[test]
fn plain_unknown_language_has_raw_span() {
    let line = highlight_line("hello world", "unknown_lang");
    // Should produce a single raw span with no special styling.
    assert_eq!(line.spans.len(), 1);
    assert_eq!(line.spans[0].content, "hello world");
}

#[test]
fn empty_line_produces_one_span() {
    let line = highlight_line("", "rust");
    assert_eq!(line.spans.len(), 1);
    assert_eq!(line.spans[0].content, "");
}
