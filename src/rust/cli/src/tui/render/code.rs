//! Simple per-line code syntax highlighter.
//!
//! `src/rust/cli/src/tui/render/code.rs`
//!
//! Uses the `regex` crate (already a `wqm-cli` dependency under the `tui`
//! feature) to identify three token classes on each source line:
//!
//! - **Line comments** → `COLOR_DIM` (gray)
//! - **String literals** (double- or single-quoted) → `COLOR_SUCCESS` (green)
//! - **Keywords** (small per-language set) → `COLOR_ACCENT` (cyan)
//!
//! Multi-line strings and block comments are intentionally out of scope; line-
//! level best-effort is acceptable for a TUI file viewer.
//!
//! Patterns are compiled once per language via `std::sync::OnceLock` and reused
//! across calls.
//!
//! Language metadata (extension map, keyword tables, comment prefixes) lives in
//! the sibling module [`code_lang`].

use std::sync::OnceLock;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use regex::Regex;

use crate::tui::theme;

pub use super::code_lang::{comment_prefix, keywords_for, language_for_extension};

// ─── Compiled patterns ────────────────────────────────────────────────────────

/// Patterns compiled once per language via `OnceLock`.
struct LangPatterns {
    /// Matches a double-quoted string: `"..."` (non-greedy, no newline).
    double_quote: Regex,
    /// Matches a single-quoted string: `'...'` (non-greedy, no newline).
    single_quote: Regex,
    /// Matches any keyword in the language's set as a whole word.
    keywords: Option<Regex>,
}

/// Return cached patterns for `language`, building them on first call.
fn patterns_for(language: &str) -> &'static LangPatterns {
    use std::collections::HashMap;
    use std::sync::Mutex;

    // One global map: language name → compiled patterns.
    static CACHE: OnceLock<Mutex<HashMap<String, &'static LangPatterns>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let mut guard = cache.lock().unwrap();
    if let Some(p) = guard.get(language) {
        return p;
    }

    let kws = keywords_for(language);
    let kw_re = if kws.is_empty() {
        None
    } else {
        // Escape each keyword and join with `|` inside a word-boundary group.
        let alts: Vec<String> = kws.iter().map(|k| regex::escape(k)).collect();
        let pat = format!(r"\b(?:{})\b", alts.join("|"));
        Regex::new(&pat).ok()
    };

    let patterns: &'static LangPatterns = Box::leak(Box::new(LangPatterns {
        double_quote: Regex::new(r#""(?:[^"\\]|\\.)*""#).unwrap(),
        single_quote: Regex::new(r"'(?:[^'\\]|\\.)*'").unwrap(),
        keywords: kw_re,
    }));

    guard.insert(language.to_string(), patterns);
    patterns
}

// ─── Per-line highlighting ────────────────────────────────────────────────────

/// Highlight a single source line, returning a styled `Line<'static>`.
///
/// Tokens are identified left-to-right in priority order:
/// 1. Line comment (rest of line) → `COLOR_DIM`
/// 2. String literals → `COLOR_SUCCESS`
/// 3. Keywords → `COLOR_ACCENT` with BOLD
/// 4. Everything else → default foreground
///
/// The function operates on owned `String` values so the returned `Line` has
/// a `'static` lifetime (all spans own their content).
pub fn highlight_line(raw: &str, language: &str) -> Line<'static> {
    let cp = comment_prefix(language);
    let p = patterns_for(language);
    highlight_with(raw, cp, p)
}

/// Core highlighting logic, split out for testability.
fn highlight_with(raw: &str, comment_pfx: Option<&str>, p: &LangPatterns) -> Line<'static> {
    // If the (trimmed) line starts with a comment prefix, color the whole line.
    if let Some(pfx) = comment_pfx {
        let trimmed = raw.trim_start();
        if trimmed.starts_with(pfx) {
            return Line::from(Span::styled(
                raw.to_owned(),
                Style::default().fg(theme::COLOR_DIM),
            ));
        }
        // Inline comment: find first occurrence not inside a string.
        if let Some(comment_start) = find_comment_start(raw, pfx, p) {
            let before = &raw[..comment_start];
            let comment_part = &raw[comment_start..];
            let mut spans = spans_for_code(before, p);
            spans.push(Span::styled(
                comment_part.to_owned(),
                Style::default().fg(theme::COLOR_DIM),
            ));
            return Line::from(spans);
        }
    }
    Line::from(spans_for_code(raw, p))
}

/// Build styled spans for a code segment (no comment handling).
fn spans_for_code(text: &str, p: &LangPatterns) -> Vec<Span<'static>> {
    // Collect all token ranges (string literals + keywords) sorted by start.
    // Overlapping tokens: earlier wins (first match takes precedence).
    let mut tokens: Vec<(usize, usize, TokenKind)> = Vec::new();

    collect_string_tokens(&mut tokens, text, &p.double_quote);
    collect_string_tokens(&mut tokens, text, &p.single_quote);
    if let Some(kw_re) = &p.keywords {
        collect_keyword_tokens(&mut tokens, text, kw_re);
    }

    // Sort by start position; remove overlaps (keep first).
    tokens.sort_by_key(|(start, _, _)| *start);
    let tokens = remove_overlaps(tokens);

    build_spans(text, &tokens)
}

/// Collect string-literal token ranges from a regex match.
fn collect_string_tokens(out: &mut Vec<(usize, usize, TokenKind)>, text: &str, re: &Regex) {
    for m in re.find_iter(text) {
        out.push((m.start(), m.end(), TokenKind::StringLiteral));
    }
}

/// Collect keyword token ranges from a regex match.
fn collect_keyword_tokens(out: &mut Vec<(usize, usize, TokenKind)>, text: &str, re: &Regex) {
    for m in re.find_iter(text) {
        out.push((m.start(), m.end(), TokenKind::Keyword));
    }
}

/// Remove overlapping token ranges, keeping earlier starts.
fn remove_overlaps(sorted: Vec<(usize, usize, TokenKind)>) -> Vec<(usize, usize, TokenKind)> {
    let mut result: Vec<(usize, usize, TokenKind)> = Vec::new();
    let mut end_watermark = 0usize;
    for tok in sorted {
        if tok.0 >= end_watermark {
            end_watermark = tok.1;
            result.push(tok);
        }
    }
    result
}

/// Convert token ranges into a `Vec<Span<'static>>` filling all gaps with plain text.
fn build_spans(text: &str, tokens: &[(usize, usize, TokenKind)]) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut cursor = 0usize;
    for &(start, end, kind) in tokens {
        if cursor < start {
            spans.push(Span::raw(text[cursor..start].to_owned()));
        }
        let slice = text[start..end].to_owned();
        let style = match kind {
            TokenKind::StringLiteral => Style::default().fg(theme::COLOR_SUCCESS),
            TokenKind::Keyword => Style::default()
                .fg(theme::COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        };
        spans.push(Span::styled(slice, style));
        cursor = end;
    }
    if cursor < text.len() {
        spans.push(Span::raw(text[cursor..].to_owned()));
    }
    if spans.is_empty() {
        spans.push(Span::raw(text.to_owned()));
    }
    spans
}

/// What kind of syntax token a range represents.
#[derive(Clone, Copy)]
enum TokenKind {
    StringLiteral,
    Keyword,
}

/// Find the byte offset of the first comment prefix that is NOT inside a string literal.
fn find_comment_start(text: &str, prefix: &str, p: &LangPatterns) -> Option<usize> {
    // Collect string literal ranges to exclude.
    let mut string_ranges: Vec<(usize, usize)> = Vec::new();
    for re in [&p.double_quote, &p.single_quote] {
        for m in re.find_iter(text) {
            string_ranges.push((m.start(), m.end()));
        }
    }
    string_ranges.sort_by_key(|(s, _)| *s);

    // Walk through the text looking for the prefix.
    let mut i = 0usize;
    while i + prefix.len() <= text.len() {
        if text[i..].starts_with(prefix) {
            // Check if position i is inside any string range.
            let inside_string = string_ranges.iter().any(|&(s, e)| i >= s && i < e);
            if !inside_string {
                return Some(i);
            }
        }
        // Advance by one char boundary.
        i += text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
