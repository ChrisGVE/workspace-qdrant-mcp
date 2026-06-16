//! Gutter symbols for table and columnar displays.
//!
//! The gutter is the first terminal column, used to display status indicators
//! (errors, warnings, info, sync state) alongside content rows.

use colored::{ColoredString, Colorize};

/// A gutter symbol with its colored representation and a plain-text fallback.
///
/// Symbols use standard UTF-8 characters (no Nerd Font dependency):
/// - `●` filled circle — good / in sync (green)
/// - `○` empty circle — pending / to be added (yellow)
/// - `◆` diamond — in progress / updating (blue)
/// - `✗` ballot x — failed / to remove (red)
/// - `▲` triangle — warning / attention (yellow)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gutter {
    /// No gutter indicator (empty space).
    None,
    /// Green `●` — file/item in sync / healthy / good.
    Sync,
    /// Yellow `○` — file/item to be added / pending.
    Add,
    /// Blue `◆` — file/item to be updated / in progress.
    Update,
    /// Red `✗` — file/item to be removed / failed.
    Remove,
    /// Red `✗` — error condition.
    Error,
    /// Yellow `▲` — warning condition.
    Warning,
    /// Blue `·` — informational.
    Info,
    /// Yellow `▲` — orphan / unknown state.
    Orphan,
    /// Cyan `…` — probing / learning baseline (cold-start health verdict, #133 F8).
    Probing,
}

impl Gutter {
    /// Render the gutter symbol with color for terminal display.
    pub fn colored(self) -> ColoredString {
        match self {
            Gutter::None => " ".normal(),
            Gutter::Sync => "●".green(),
            Gutter::Add => "○".yellow(),
            Gutter::Update => "◆".blue(),
            Gutter::Remove => "✗".red(),
            Gutter::Error => "✗".red(),
            Gutter::Warning => "▲".yellow(),
            Gutter::Info => "·".blue(),
            Gutter::Orphan => "▲".yellow(),
            Gutter::Probing => "…".cyan(),
        }
    }

    /// Plain-text representation (no color), for script/pipe output.
    pub fn plain(self) -> &'static str {
        match self {
            Gutter::None => " ",
            Gutter::Sync => "*",
            Gutter::Add => "o",
            Gutter::Update => ">",
            Gutter::Remove => "x",
            Gutter::Error => "x",
            Gutter::Warning => "!",
            Gutter::Info => ".",
            Gutter::Orphan => "!",
            Gutter::Probing => "~",
        }
    }

    /// Width consumed by the gutter column in columnar displays (symbol + separator space).
    pub const WIDTH: usize = 2;

    /// Width of just the gutter symbol (for table displays where tabled provides its own padding).
    pub const SYMBOL_WIDTH: usize = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_plain_symbols_are_single_char() {
        let variants = [
            Gutter::None,
            Gutter::Sync,
            Gutter::Add,
            Gutter::Update,
            Gutter::Remove,
            Gutter::Error,
            Gutter::Warning,
            Gutter::Info,
            Gutter::Orphan,
            Gutter::Probing,
        ];
        for g in variants {
            assert_eq!(
                g.plain().chars().count(),
                1,
                "{:?} plain should be 1 char",
                g
            );
        }
    }

    #[test]
    fn colored_does_not_panic() {
        let variants = [
            Gutter::None,
            Gutter::Sync,
            Gutter::Add,
            Gutter::Update,
            Gutter::Remove,
            Gutter::Error,
            Gutter::Warning,
            Gutter::Info,
            Gutter::Orphan,
            Gutter::Probing,
        ];
        for g in variants {
            let _ = g.colored();
        }
    }

    #[test]
    fn gutter_width_is_two() {
        assert_eq!(Gutter::WIDTH, 2);
    }

    #[test]
    fn probing_renders_non_color_channel() {
        // #133 F8: the cold-start gutter has a real non-color channel.
        assert_eq!(Gutter::Probing.plain(), "~");
        assert!(Gutter::Probing.colored().to_string().contains('…'));
    }
}
