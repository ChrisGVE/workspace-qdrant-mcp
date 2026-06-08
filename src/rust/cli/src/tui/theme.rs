//! Shared TUI theme — color palette, styles, and visual constants.
//!
//! Centralizes all color and style definitions used across TUI views
//! for consistency. Maps to the same semantic meanings as the CLI
//! output module's `style.rs` and `gutter.rs`.

use ratatui::style::{Color, Modifier, Style};

// ─── Semantic colors ────────────────────────────────────────────────────────

/// Green — success, healthy, active, done, in-sync.
pub const COLOR_SUCCESS: Color = Color::Green;

/// Yellow — warning, degraded, pending, paused.
pub const COLOR_WARNING: Color = Color::Yellow;

/// Red — error, unhealthy, failed, to-remove.
pub const COLOR_ERROR: Color = Color::Red;

/// Blue — informational, in-progress, updating.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub const COLOR_INFO: Color = Color::Blue;

/// Cyan — accent for counts, highlights.
pub const COLOR_ACCENT: Color = Color::Cyan;

/// Gray — secondary info, inactive, metadata.
pub const COLOR_DIM: Color = Color::DarkGray;

/// Light gray — borders, separators, labels.
pub const COLOR_MUTED: Color = Color::Gray;

/// White — primary text, headers.
pub const COLOR_FG: Color = Color::White;

/// Black — backgrounds for popups.
pub const COLOR_BG: Color = Color::Black;

// ─── Alarm state ────────────────────────────────────────────────────────────

/// Dark red background tint for service/Qdrant down alarm.
pub const COLOR_ALARM_BG: Color = Color::Rgb(60, 0, 0);

/// Muted/dimmed background for paused items.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub const COLOR_PAUSED_BG: Color = Color::Rgb(30, 30, 30);

// ─── Composite styles ───────────────────────────────────────────────────────

/// Style for the tab bar title ("Workspace Qdrant MCP").
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn title_style() -> Style {
    Style::default().fg(COLOR_FG).add_modifier(Modifier::BOLD)
}

/// Style for the active tab label.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn tab_active_style() -> Style {
    Style::default()
        .fg(COLOR_BG)
        .bg(COLOR_FG)
        .add_modifier(Modifier::BOLD)
}

/// Style for inactive tab labels.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn tab_inactive_style() -> Style {
    Style::default().fg(COLOR_MUTED)
}

/// Style for tab number indicators.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn tab_number_style() -> Style {
    Style::default().fg(COLOR_WARNING)
}

/// Style for tab separators.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn tab_separator_style() -> Style {
    Style::default().fg(COLOR_MUTED)
}

/// Style for help key labels.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn help_key_style() -> Style {
    Style::default().fg(COLOR_WARNING)
}

/// Style for table headers in list views.
pub fn table_header_style() -> Style {
    Style::default().fg(COLOR_FG).add_modifier(Modifier::BOLD)
}

/// Style for the selected row in a list — the TUI-wide cursor.
///
/// This is the single source of truth for the selection highlight. Apply it as
/// the `Row`'s base style (not per-cell) so the highlight spans the entire line
/// including inter-column gaps. Cell spans must set only `fg` (never `bg`) so
/// this background shows through; the background is a light slate that contrasts
/// with every column foreground (including muted metadata) to keep values
/// readable under the cursor. The cursor deliberately does NOT use bold —
/// bold is reserved to mark active projects/libraries, so the two signals stay
/// independent.
pub fn selected_row_style() -> Style {
    Style::default().bg(Color::Rgb(70, 70, 100))
}

/// Background tint marking a row that matches the active `/` search. Distinct
/// from the cursor's slate so matches stay visible even when the cursor is on a
/// different row. Applied as the row's base style; cell spans keep their own fg.
pub fn search_match_style() -> Style {
    Style::default().bg(Color::Rgb(0, 55, 55))
}

/// Style for popup/overlay backgrounds.
pub fn popup_style() -> Style {
    Style::default().bg(COLOR_BG)
}

/// Style for loading/placeholder text.
pub fn loading_style() -> Style {
    Style::default().fg(COLOR_DIM)
}

/// Style for the bottom status bar.
pub fn status_bar_style() -> Style {
    Style::default().fg(COLOR_DIM)
}

/// Style for alarm state — dark reddish background across the interface.
pub fn alarm_style() -> Style {
    Style::default().bg(COLOR_ALARM_BG)
}

/// Style for paused items — dimmed background.
#[allow(dead_code)] // reserved: TUI design-language parity with output/style.rs
pub fn paused_style() -> Style {
    Style::default().fg(COLOR_DIM).bg(COLOR_PAUSED_BG)
}

/// Style for the search input prompt.
pub fn search_style() -> Style {
    Style::default().fg(COLOR_ACCENT)
}

// ─── Gutter symbols (matching CLI gutter.rs) ────────────────────────────────

/// Gutter symbol for in-sync/healthy items.
pub const GUTTER_SYNC: &str = "●";
/// Gutter symbol for pending/to-add items.
#[allow(dead_code)] // reserved: TUI design-language parity with output/gutter.rs
pub const GUTTER_ADD: &str = "○";
/// Gutter symbol for in-progress/updating items.
#[allow(dead_code)] // reserved: TUI design-language parity with output/gutter.rs
pub const GUTTER_UPDATE: &str = "◆";
/// Gutter symbol for failed/to-remove items.
pub const GUTTER_REMOVE: &str = "✗";
/// Gutter symbol for warning/orphan.
#[allow(dead_code)] // reserved: TUI design-language parity with output/gutter.rs
pub const GUTTER_WARNING: &str = "▲";
/// Gutter symbol for informational.
#[allow(dead_code)] // reserved: TUI design-language parity with output/gutter.rs
pub const GUTTER_INFO: &str = "·";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_colors_are_distinct() {
        // Ensure key colors don't accidentally alias
        assert_ne!(COLOR_SUCCESS, COLOR_ERROR);
        assert_ne!(COLOR_WARNING, COLOR_INFO);
        assert_ne!(COLOR_SUCCESS, COLOR_WARNING);
    }

    #[test]
    fn styles_do_not_panic() {
        let _ = title_style();
        let _ = tab_active_style();
        let _ = tab_inactive_style();
        let _ = tab_number_style();
        let _ = tab_separator_style();
        let _ = help_key_style();
        let _ = table_header_style();
        let _ = selected_row_style();
        let _ = popup_style();
        let _ = loading_style();
        let _ = status_bar_style();
        let _ = alarm_style();
        let _ = paused_style();
        let _ = search_style();
    }

    #[test]
    fn alarm_bg_is_dark_red() {
        assert_eq!(COLOR_ALARM_BG, Color::Rgb(60, 0, 0));
    }

    #[test]
    fn cursor_and_match_backgrounds_are_distinct() {
        // The search-match tint must not collide with the cursor highlight,
        // and the cursor must not rely on bold (reserved for active rows).
        assert_ne!(selected_row_style().bg, search_match_style().bg);
        assert!(!selected_row_style().add_modifier.contains(Modifier::BOLD));
    }
}
