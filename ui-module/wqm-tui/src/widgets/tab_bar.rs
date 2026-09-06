//! The tab bar — VISUAL-LANGUAGE.md §3, plus the must-see rule from §4.
//!
//! Inversion is *the* selector mechanism, and cyan is reserved to it: if it is a cyan
//! block, it is what you have selected. The leading number stays outside the inverted
//! block because it is a jump hint, not part of the selection.
//!
//! §4's must-see rule overrides the hue but not the mechanism: a tab owning a degraded or
//! offline store is recoloured, and it keeps that colour *under* inversion — so a selected
//! error tab is a coloured inverse block, never a cyan one.
//!
//! # The number was a hint that never looked like one
//!
//! §3 keeps the leading number outside the inverted block because it is a **jump hint**, and
//! then drew it at the same muted rung as the label it precedes — which says "metadata", not
//! "press this". It now carries [`crate::tokens::accent`], the hue §10 measured as unclaimed
//! headroom (Chris, 20260906). Only the *key* does: tab ten is reached with `0`, so its `1`
//! stays muted and its `0` is the accent. That split lives on [`Tab`] ([`Tab::prefix`] and
//! [`Tab::hotkey_char`]) rather than in the render loop, because a loop that special-cased the
//! number ten would be right about one tab and wrong about every future two-digit one.
//!
//! # Under a modal the bar carries no highlight at all
//!
//! Chris, 20260906: *"remove all highlighting to the underlying page when in modal mode"*. So
//! [`TabBar::under_modal`] drops the accent, the inversion and the §4 alarm hues together, and
//! leaves the selected tab distinguished by **weight alone**. That is §1's two axes doing what
//! they are for: emphasis says *which one you are on*, highlight says *this is live*, and while
//! a modal owns the input nothing behind it is live.
//!
//! # A tab that names a collection derives its label
//!
//! Tabs that stand for an N8 collection are built with [`Tab::for_collection`], which reads
//! the label out of [`crate::names::display`] rather than spelling one beside the registry.
//! That is `CR-038`'s invariant, and it is why this tab now says *Libraries* where it used
//! to say *Library*. `Dashboard`, `Service` and `Config` name no collection, so they carry
//! UI-only labels and always will.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use wqm_common::names::Collection;

use crate::{
    names,
    tokens::{self, Health},
};

pub struct Tab {
    pub number: u8,
    /// Owned, because a derived label is computed rather than written — see the module
    /// docs. A UI-only tab still passes a literal, and `impl Into<String>` takes both.
    pub label: String,
    /// Set when this tab owns a store in trouble; drives the §4 recolour.
    pub alarm: Option<Health>,
}

impl Tab {
    pub fn new(number: u8, label: impl Into<String>) -> Self {
        Self {
            number,
            label: label.into(),
            alarm: None,
        }
    }

    /// A tab standing for an N8 collection, labelled from the registry.
    pub fn for_collection(number: u8, collection: Collection) -> Self {
        Self::new(number, names::display(collection))
    }

    pub fn alarming(number: u8, label: impl Into<String>, health: Health) -> Self {
        Self {
            number,
            label: label.into(),
            alarm: Some(health),
        }
    }

    /// The number as printed, minus the one character that jumps here — `"1"` for tab ten,
    /// empty for tabs one through nine.
    ///
    /// Derived from the number rather than listed, so a hypothetical eleventh tab splits the
    /// same way without anyone remembering to add it.
    pub fn prefix(&self) -> String {
        let printed = self.number.to_string();
        let mut chars = printed.chars();
        chars.next_back();
        chars.as_str().to_string()
    }

    /// The single key that jumps to this tab: `1`–`9`, and `0` for tab ten.
    pub fn hotkey_char(&self) -> char {
        self.number
            .to_string()
            .chars()
            .next_back()
            .expect("a number always prints at least one digit")
    }

    /// The hue this tab carries when selected — its alarm colour if it has one, else the
    /// reserved selector cyan.
    fn selected_bg(&self) -> Color {
        match self.alarm {
            Some(health) => health.color(),
            None => tokens::selector(),
        }
    }

    /// The hue this tab carries when unselected. An alarm still shows; otherwise the tab
    /// recedes to muted.
    ///
    /// # An alarm here is carried by hue alone, and that is now visibly a gap
    ///
    /// r02 §3 asks for a structural signature first with colour reserved, and
    /// [`Health::glyph`] is that signature everywhere else. This surface has no glyph, so
    /// under an encoding that emits no colour ([`crate::encoding`]) an alarming tab is
    /// indistinguishable from a calm one — measured, not reasoned: `CLICOLOR_FORCE=no_color
    /// cargo pantry dump "Tab Bar"` renders *Alarm, Unselected* identically to *Default*.
    ///
    /// The gap is not new; the encoding axis only made it observable. Adding a glyph to a tab
    /// label changes the visual language, so it is recorded as an open decision in
    /// `handover.md` §7 rather than fixed here.
    ///
    /// # An alarm is a highlight, so a modal takes it too
    ///
    /// Under a modal `modal` is true and this is muted whatever the alarm says. That is not a
    /// loss of information: the modal is the thing being answered, and §4's must-see rule is
    /// about the tab *you could jump to*, which is exactly what a modal has suspended.
    fn unselected_fg(&self, modal: bool) -> Color {
        match self.alarm {
            Some(health) if !modal => health.color(),
            _ => tokens::muted(),
        }
    }
}

pub struct TabBar {
    tabs: Vec<Tab>,
    active: usize,
    modal: bool,
}

/// Columns between one tab and the next.
const TAB_GAP: u16 = 2;

impl TabBar {
    pub fn new(tabs: Vec<Tab>, active: usize) -> Self {
        Self {
            tabs,
            active,
            modal: false,
        }
    }

    /// Whether a modal owns the input. See the module docs: under one, every highlight drops.
    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
        self
    }

    /// Columns this row would occupy if nothing truncated it.
    ///
    /// Exists because [`crate::widgets::chrome::app_bar`] has to decide whether the title and
    /// the ten tabs both fit **before** it draws either, and a caller that measured the row by
    /// rendering it into a scratch buffer would be measuring a second implementation.
    pub fn width(&self) -> u16 {
        Self::width_of(&self.tabs, self.active)
    }

    /// [`TabBar::width`] for a row that has not been built yet — what
    /// [`crate::widgets::chrome::app_bar`] asks, since it must decide the layout before it
    /// hands the tabs over.
    pub fn width_of(tabs: &[Tab], active: usize) -> u16 {
        let drawn: u16 = tabs
            .iter()
            .enumerate()
            .map(|(i, tab)| {
                let number = tab.number.to_string().chars().count() as u16;
                let label = tab.label.chars().count() as u16;
                // Selected: `N` + space + the block's own two padding spaces around the label.
                // Unselected: `N` + space + label.
                number + label + if i == active { 3 } else { 1 }
            })
            .sum();
        drawn + TAB_GAP * (tabs.len().saturating_sub(1) as u16)
    }

    /// STORYBOARD §4.1's actual tab row, in its order.
    ///
    /// `Dashboard · Queue · Projects · Libraries · Rules · Scratchpad · Search · Graph ·
    /// Tags · Service` — ten, of which four name an N8 collection and therefore derive their
    /// label rather than spelling one (`CR-038`). This is what a **screen** draws;
    /// [`TabBar::standard`] is the abbreviated set the widget previews use, which exists so a
    /// tab-bar variant fits in a preview cell.
    pub fn storyboard_tabs() -> Vec<Tab> {
        vec![
            Tab::new(1, "Dashboard"),
            Tab::new(2, "Queue"),
            Tab::for_collection(3, Collection::Projects),
            Tab::for_collection(4, Collection::Libraries),
            Tab::for_collection(5, Collection::Rules),
            Tab::for_collection(6, Collection::Scratchpad),
            Tab::new(7, "Search"),
            Tab::new(8, "Graph"),
            Tab::new(9, "Tags"),
            Tab::new(10, "Service"),
        ]
    }

    /// The abbreviated tab row the widget previews use — see [`TabBar::storyboard_tabs`] for
    /// the row a screen draws.
    pub fn standard(active: usize) -> Self {
        Self::new(
            vec![
                Tab::new(1, "Dashboard"),
                Tab::for_collection(2, Collection::Libraries),
                Tab::for_collection(3, Collection::Rules),
                Tab::new(4, "Service"),
                Tab::new(5, "Config"),
            ],
            active,
        )
    }
}

impl Widget for TabBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut spans: Vec<Span> = Vec::new();

        for (i, tab) in self.tabs.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw("  "));
            }

            // The number stays out of the block so the jump hint is never inverted, and it is
            // spelled in two spans so only the key that actually jumps carries the accent.
            let prefix = tab.prefix();
            if !prefix.is_empty() {
                spans.push(Span::styled(prefix, tokens::muted_style()));
            }
            spans.push(Span::styled(
                tab.hotkey_char().to_string(),
                Style::default().fg(if self.modal {
                    tokens::muted()
                } else {
                    tokens::accent()
                }),
            ));

            if i == self.active {
                spans.push(Span::styled(" ", tokens::muted_style()));
                spans.push(if self.modal {
                    // Weight alone: `inverted` would put a fill behind a page that a modal has
                    // already taken the input from.
                    Span::styled(
                        format!(" {} ", tab.label),
                        tokens::muted_style().add_modifier(Modifier::BOLD),
                    )
                } else {
                    Span::styled(
                        format!(" {} ", tab.label),
                        tokens::inverted(tab.selected_bg()),
                    )
                });
            } else {
                spans.push(Span::styled(
                    format!(" {}", tab.label),
                    Style::default().fg(tab.unselected_fg(self.modal)),
                ));
            }
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::chrome::test_support::Restore;

    fn render(bar: TabBar, width: u16) -> Buffer {
        let area = Rect::new(0, 0, width, 1);
        let mut buf = Buffer::empty(area);
        bar.render(area, &mut buf);
        buf
    }

    fn row(buf: &Buffer) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).expect("cell in area").symbol())
            .collect()
    }

    /// The style of the first cell of a needle, read out of the buffer rather than recomputed.
    fn style_of(buf: &Buffer, needle: &str) -> Style {
        let line = row(buf);
        let byte = line.find(needle).unwrap_or_else(|| panic!("{needle:?} not drawn in {line:?}"));
        let x = line[..byte].chars().count() as u16;
        buf.cell((x, 0)).expect("cell in area").style()
    }

    /// Tab 10's jump KEY is `0`, so only the `0` may carry the hue that says "press this".
    ///
    /// The split is a property of [`Tab`], not of the render loop: a bar that special-cased
    /// the number ten would pass a pixel test and still leave every other two-digit tab
    /// painting its whole number as a hotkey.
    #[test]
    fn a_two_digit_tab_offers_only_its_last_digit_as_the_jump_key() {
        assert_eq!(Tab::new(1, "Dashboard").prefix(), "");
        assert_eq!(Tab::new(1, "Dashboard").hotkey_char(), '1');
        assert_eq!(Tab::new(9, "Tags").prefix(), "");
        assert_eq!(Tab::new(9, "Tags").hotkey_char(), '9');
        assert_eq!(Tab::new(10, "Service").prefix(), "1");
        assert_eq!(Tab::new(10, "Service").hotkey_char(), '0');
    }

    /// The digit is the accent hue and the label is not — the whole point of the split.
    ///
    /// Compared against the STATED tokens rather than against each other: "the digit differs
    /// from the label" would also pass if both had drifted onto some third colour.
    #[test]
    fn the_jump_digit_carries_the_accent_and_the_label_stays_muted() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(TabBar::new(vec![Tab::new(1, "Dashboard"), Tab::new(2, "Queue")], 0), 40);
        assert_eq!(style_of(&buf, "2").fg, Some(tokens::accent()));
        assert_eq!(style_of(&buf, "Queue").fg, Some(tokens::muted()));
    }

    /// Tab 10 paints ONLY the `0`; the leading `1` recedes with the label.
    #[test]
    fn tab_ten_accents_the_zero_and_mutes_the_one() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(TabBar::new(vec![Tab::new(10, "Service")], 0), 40);
        assert_eq!(row(&buf).trim_end(), "10  Service");
        assert_eq!(style_of(&buf, "1").fg, Some(tokens::muted()), "the 1 is not a key");
        assert_eq!(style_of(&buf, "0").fg, Some(tokens::accent()), "the 0 is the key");
    }

    /// Under a modal the page beneath it carries no highlight at all (Chris, 20260906).
    ///
    /// Three things drop together, and each is checked against its own stated token: the
    /// digit's accent, the selected tab's inverted block, and an alarming tab's hue.
    #[test]
    fn a_modal_removes_every_highlight_from_the_bar_beneath_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let build = || {
            TabBar::new(
                vec![
                    Tab::new(1, "Dashboard"),
                    Tab::alarming(2, "Queue", Health::Offline),
                ],
                0,
            )
        };

        let live = render(build(), 40);
        let under = render(build().under_modal(true), 40);

        assert_eq!(style_of(&live, "1").fg, Some(tokens::accent()));
        assert_eq!(style_of(&under, "1").fg, Some(tokens::muted()), "no accent under a modal");

        assert_eq!(
            style_of(&live, "Dashboard").bg,
            Some(tokens::selector()),
            "the selected tab is an inverted block while nothing is over it"
        );
        // `Color::Reset` is a buffer cell that was never filled — the terminal's own
        // background, which is what "no highlight" is in a buffer rather than `None`.
        assert_eq!(
            style_of(&under, "Dashboard").bg,
            Some(Color::Reset),
            "no fill under a modal — emphasis is not highlight"
        );
        assert!(
            style_of(&under, "Dashboard")
                .add_modifier
                .contains(Modifier::BOLD),
            "bold is what is left to distinguish the selected tab"
        );

        assert_eq!(style_of(&live, "Queue").fg, Some(Health::Offline.color()));
        assert_eq!(
            style_of(&under, "Queue").fg,
            Some(tokens::muted()),
            "an alarm hue is a highlight too, and drops with the rest"
        );
    }

    /// A bar knows how wide it wants to be, because the app bar has to decide whether the
    /// title and the ten tabs fit before it draws either.
    #[test]
    fn the_measured_width_is_the_width_actually_drawn() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let bar = TabBar::new(TabBar::storyboard_tabs(), 0);
        let want = bar.width();
        let drawn = row(&render(TabBar::new(TabBar::storyboard_tabs(), 0), 200));
        assert_eq!(
            drawn.trim_end().chars().count() as u16,
            want,
            "the measurement and the drawing must be the same arithmetic"
        );
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient;
