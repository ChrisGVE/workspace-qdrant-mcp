//! Content-type-aware rendering for TUI text overlays.
//!
//! `src/rust/cli/src/tui/render/mod.rs`
//!
//! Sits between raw file bytes / stored text and ratatui's `Paragraph` widget.
//! Four sub-modules each own one rendering concern:
//!
//! - [`content`] — public entry points; dispatches to the right renderer.
//! - [`markdown`] — hand-rolled line-based Markdown renderer (no dependencies).
//! - [`code`] — simple per-line code highlighter with an extension→language map.
//! - [`code_lang`] — language metadata: extension map, keyword tables, comment prefixes.
//!
//! All public functions return `Vec<Line<'static>>` (owned spans) so callers
//! can cache the result in view state and hand it directly to `Paragraph::new`.

#[cfg(feature = "tui")]
pub mod code;
#[cfg(feature = "tui")]
pub mod code_lang;
#[cfg(feature = "tui")]
pub mod content;
#[cfg(feature = "tui")]
pub mod markdown;
