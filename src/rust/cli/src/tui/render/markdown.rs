//! Hand-rolled line-based Markdown renderer for TUI overlays.
//!
//! `src/rust/cli/src/tui/render/markdown.rs`
//!
//! Parses Markdown line-by-line and converts it to styled ratatui `Line`s.
//! This is deliberately pragmatic rather than CommonMark-complete: the goal is
//! readable rendering of rule text and scratchpad notes in a terminal overlay,
//! not full spec compliance. Unknown or edge-case syntax renders literally —
//! that is always acceptable.
//!
//! ## Supported constructs
//! - ATX headings `#`..`######` → bold white, level indicated by weight
//! - Fenced code blocks (` ``` `) → dim verbatim content
//! - Bullet lists (`-`, `*`, `+`) and ordered (`1.`) → glyph + indent
//! - Blockquotes `> ` → indented dim text
//! - Inline: `` `code` `` → accent, `**bold**`/`__bold__` → bold,
//!   `*italic*`/`_italic_` → italic
//! - Blank lines → preserved as paragraph breaks
//! - All other lines → word-wrapped plain text with inline markup

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::theme;

// ─── Block-level parser ───────────────────────────────────────────────────────

/// Render Markdown text to styled `Line`s, word-wrapped to `width` columns.
///
/// Returns an owned `Vec<Line<'static>>` so the result can be cached in view
/// state and passed directly to `Paragraph::new`.
pub fn render_markdown(text: &str, width: usize) -> Vec<Line<'static>> {
    let safe_width = width.max(10);
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_fence = false;

    for raw_line in text.lines() {
        // ── Fenced code block boundary ─────────────────────────────────────
        if raw_line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            // Don't emit the fence line itself — it's structural punctuation.
            continue;
        }

        if in_fence {
            // Verbatim: emit with dim color, no inline parsing, no wrap.
            out.push(Line::from(Span::styled(
                raw_line.to_owned(),
                Style::default().fg(theme::COLOR_DIM),
            )));
            continue;
        }

        // ── ATX heading ────────────────────────────────────────────────────
        if let Some(heading) = parse_heading(raw_line) {
            out.extend(render_heading(heading, safe_width));
            continue;
        }

        // ── Blank line ─────────────────────────────────────────────────────
        if raw_line.trim().is_empty() {
            out.push(Line::from(""));
            continue;
        }

        // ── Blockquote ─────────────────────────────────────────────────────
        if let Some(content) = raw_line.strip_prefix("> ").or(raw_line.strip_prefix(">")) {
            let indent_w = safe_width.saturating_sub(2);
            for wrapped in word_wrap(content, indent_w) {
                out.push(Line::from(vec![
                    Span::styled("▏ ", Style::default().fg(theme::COLOR_DIM)),
                    Span::styled(wrapped, Style::default().fg(theme::COLOR_DIM)),
                ]));
            }
            continue;
        }

        // ── Bullet list ────────────────────────────────────────────────────
        if let Some(item_text) = parse_bullet(raw_line) {
            let item_w = safe_width.saturating_sub(2);
            let mut wrapped = word_wrap(item_text, item_w);
            // First wrapped line gets the bullet glyph; continuation lines are indented.
            for (i, segment) in wrapped.drain(..).enumerate() {
                if i == 0 {
                    let mut spans =
                        vec![Span::styled("• ", Style::default().fg(theme::COLOR_ACCENT))];
                    spans.extend(parse_inline(&segment));
                    out.push(Line::from(spans));
                } else {
                    let mut spans = vec![Span::raw("  ")];
                    spans.extend(parse_inline(&segment));
                    out.push(Line::from(spans));
                }
            }
            continue;
        }

        // ── Ordered list ───────────────────────────────────────────────────
        if let Some((number, item_text)) = parse_ordered(raw_line) {
            let prefix = format!("{number}. ");
            let prefix_w = prefix.chars().count();
            let item_w = safe_width.saturating_sub(prefix_w);
            let indent = " ".repeat(prefix_w);
            let mut wrapped = word_wrap(item_text, item_w);
            for (i, segment) in wrapped.drain(..).enumerate() {
                if i == 0 {
                    let mut spans = vec![Span::styled(
                        prefix.clone(),
                        Style::default().fg(theme::COLOR_ACCENT),
                    )];
                    spans.extend(parse_inline(&segment));
                    out.push(Line::from(spans));
                } else {
                    let mut spans = vec![Span::raw(indent.clone())];
                    spans.extend(parse_inline(&segment));
                    out.push(Line::from(spans));
                }
            }
            continue;
        }

        // ── Normal paragraph text ───────────────────────────────────────────
        for segment in word_wrap(raw_line.trim(), safe_width) {
            out.push(Line::from(parse_inline(&segment)));
        }
    }

    // If a fenced block was never closed, the parser just consumed all remaining
    // lines verbatim — that is the correct fallback.
    out
}

// ─── Heading ──────────────────────────────────────────────────────────────────

/// Data for a parsed ATX heading.
struct Heading<'a> {
    level: usize,
    text: &'a str,
}

/// Parse an ATX heading line. Returns `None` if the line is not a heading.
fn parse_heading(line: &str) -> Option<Heading<'_>> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level > 6 {
        return None;
    }
    let rest = trimmed[level..].trim_start();
    // ATX heading spec: must be separated from text by a space (or be empty).
    if !rest.is_empty() && !trimmed[level..].starts_with(' ') {
        return None;
    }
    Some(Heading { level, text: rest })
}

/// Render a heading as styled lines. Higher levels get `COLOR_FG`; lower levels
/// progressively dim toward `COLOR_MUTED`.
fn render_heading(h: Heading, width: usize) -> Vec<Line<'static>> {
    let fg = if h.level <= 2 {
        theme::COLOR_FG
    } else if h.level <= 4 {
        theme::COLOR_MUTED
    } else {
        theme::COLOR_DIM
    };

    let mut lines = Vec::new();
    for segment in word_wrap(h.text, width) {
        lines.push(Line::from(Span::styled(
            segment,
            Style::default().fg(fg).add_modifier(Modifier::BOLD),
        )));
    }
    lines
}

// ─── List parsers ─────────────────────────────────────────────────────────────

/// Parse a bullet list item. Returns the item text if the line starts with
/// `- `, `* `, or `+ ` (with optional leading whitespace).
fn parse_bullet(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

/// Parse an ordered list item. Returns `(number_str, item_text)` for lines
/// matching `N. text` where N is one or more digits.
fn parse_ordered(line: &str) -> Option<(String, &str)> {
    let trimmed = line.trim_start();
    // Collect leading digits.
    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let rest = &trimmed[digits.len()..];
    if let Some(text) = rest.strip_prefix(". ") {
        Some((digits, text))
    } else {
        None
    }
}

// ─── Inline markup parser ─────────────────────────────────────────────────────

/// Parse a single line of text for inline markup, returning `Vec<Span<'static>>`.
///
/// Recognised patterns (single-pass, left-to-right, first match wins):
/// - `` `code` `` → accent color
/// - `**text**` or `__text__` → bold
/// - `*text*` or `_text_` → italic
///
/// Anything else is emitted as a plain span. Nested or malformed markers render
/// literally — no backtracking.
pub fn parse_inline(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut plain_start = 0;

    while i < chars.len() {
        // ── Backtick inline code ──────────────────────────────────────────
        if chars[i] == '`' {
            if let Some(end) = find_closing(&chars, i + 1, '`') {
                flush_plain(text, plain_start, i, &chars, &mut spans);
                let code: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(code, Style::default().fg(theme::COLOR_ACCENT)));
                i = end + 1;
                plain_start = i;
                continue;
            }
        }

        // ── Double-star / double-underscore bold ──────────────────────────
        if i + 1 < chars.len() {
            let two = (chars[i], chars[i + 1]);
            if two == ('*', '*') || two == ('_', '_') {
                let marker = chars[i];
                if let Some(end) = find_double_closing(&chars, i + 2, marker) {
                    flush_plain(text, plain_start, i, &chars, &mut spans);
                    let content: String = chars[i + 2..end].iter().collect();
                    spans.push(Span::styled(
                        content,
                        Style::default().add_modifier(Modifier::BOLD),
                    ));
                    i = end + 2;
                    plain_start = i;
                    continue;
                }
            }
        }

        // ── Single-star / single-underscore italic ────────────────────────
        if chars[i] == '*' || chars[i] == '_' {
            let marker = chars[i];
            // Only treat as italic if it's NOT a double marker.
            let is_double = i + 1 < chars.len() && chars[i + 1] == marker;
            if !is_double {
                if let Some(end) = find_closing(&chars, i + 1, marker) {
                    flush_plain(text, plain_start, i, &chars, &mut spans);
                    let content: String = chars[i + 1..end].iter().collect();
                    spans.push(Span::styled(
                        content,
                        Style::default().add_modifier(Modifier::ITALIC),
                    ));
                    i = end + 1;
                    plain_start = i;
                    continue;
                }
            }
        }

        i += 1;
    }

    // Flush any remaining plain text.
    if plain_start < chars.len() {
        let tail: String = chars[plain_start..].iter().collect();
        spans.push(Span::raw(tail));
    }
    if spans.is_empty() {
        spans.push(Span::raw(text.to_owned()));
    }
    spans
}

/// Emit accumulated plain text from `chars[plain_start..end]` as a raw span.
fn flush_plain(
    _text: &str,
    plain_start: usize,
    end: usize,
    chars: &[char],
    spans: &mut Vec<Span<'static>>,
) {
    if plain_start < end {
        let s: String = chars[plain_start..end].iter().collect();
        spans.push(Span::raw(s));
    }
}

/// Find the next occurrence of `marker` in `chars` starting at `from`.
/// Returns the index (exclusive) of the closing marker, or `None`.
fn find_closing(chars: &[char], from: usize, marker: char) -> Option<usize> {
    chars[from..]
        .iter()
        .position(|&c| c == marker)
        .map(|p| p + from)
}

/// Find the next occurrence of two consecutive `marker` chars starting at `from`.
fn find_double_closing(chars: &[char], from: usize, marker: char) -> Option<usize> {
    let mut i = from;
    while i + 1 < chars.len() {
        if chars[i] == marker && chars[i + 1] == marker {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ─── Word-wrap ────────────────────────────────────────────────────────────────

/// Word-wrap `text` to at most `max_width` columns, splitting at whitespace.
///
/// Returns at least one element even for empty input. Lines that contain no
/// whitespace and exceed `max_width` are hard-wrapped at character boundaries.
pub fn word_wrap(text: &str, max_width: usize) -> Vec<String> {
    let safe_w = max_width.max(1);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        // Hard-wrap any word that is by itself longer than the column limit.
        let word_chars: Vec<char> = word.chars().collect();
        let mut word_start = 0;
        while word_start < word_chars.len() {
            let word_slice: String = word_chars[word_start..].iter().collect();
            let word_len = word_chars.len() - word_start;
            let cur_len = current.chars().count();

            if current.is_empty() {
                if word_len <= safe_w {
                    current = word_slice;
                    word_start = word_chars.len();
                } else {
                    // Hard-wrap: take as many chars as fit.
                    let chunk: String =
                        word_chars[word_start..word_start + safe_w].iter().collect();
                    lines.push(chunk);
                    word_start += safe_w;
                }
            } else if cur_len + 1 + word_len <= safe_w {
                current.push(' ');
                current.push_str(&word_slice);
                word_start = word_chars.len();
            } else {
                lines.push(current.clone());
                current.clear();
                // Don't advance word_start; reprocess the full remaining slice.
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "markdown_tests.rs"]
mod tests;
