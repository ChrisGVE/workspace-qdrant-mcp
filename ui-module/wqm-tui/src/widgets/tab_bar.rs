//! The tab bar — VISUAL-LANGUAGE.md §3, plus the must-see rule from §4.
//!
//! Inversion is *the* selector mechanism, and cyan is reserved to it: if it is a cyan
//! block, it is what you have selected. The leading number stays outside the inverted
//! block because it is a jump hint, not part of the selection.
//!
//! §4's must-see rule overrides the hue but not the mechanism: a tab owning a degraded or
//! offline store is recoloured, and it keeps that colour *under* inversion — so a selected
//! error tab is a coloured inverse block, never a cyan one.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, Health};

pub struct Tab {
    pub number: u8,
    pub label: &'static str,
    /// Set when this tab owns a store in trouble; drives the §4 recolour.
    pub alarm: Option<Health>,
}

impl Tab {
    pub fn new(number: u8, label: &'static str) -> Self {
        Self {
            number,
            label,
            alarm: None,
        }
    }

    pub fn alarming(number: u8, label: &'static str, health: Health) -> Self {
        Self {
            number,
            label,
            alarm: Some(health),
        }
    }

    /// The hue this tab carries when selected — its alarm colour if it has one, else the
    /// reserved selector cyan.
    fn selected_bg(&self) -> Color {
        match self.alarm {
            Some(health) => health.color(),
            None => tokens::SELECTOR,
        }
    }

    /// The hue this tab carries when unselected. An alarm still shows; otherwise the tab
    /// recedes to muted.
    fn unselected_fg(&self) -> Color {
        match self.alarm {
            Some(health) => health.color(),
            None => tokens::muted(),
        }
    }
}

pub struct TabBar {
    tabs: Vec<Tab>,
    active: usize,
}

impl TabBar {
    pub fn new(tabs: Vec<Tab>, active: usize) -> Self {
        Self { tabs, active }
    }

    /// The five top-level tabs the storyboard works with.
    pub fn standard(active: usize) -> Self {
        Self::new(
            vec![
                Tab::new(1, "Dashboard"),
                Tab::new(2, "Library"),
                Tab::new(3, "Rules"),
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

            if i == self.active {
                // Number stays out of the block so the jump hint is never inverted.
                spans.push(Span::styled(
                    format!("{} ", tab.number),
                    tokens::muted_style(),
                ));
                spans.push(Span::styled(
                    format!(" {} ", tab.label),
                    Style::default()
                        .fg(tokens::SELECTOR_FG)
                        .bg(tab.selected_bg()),
                ));
            } else {
                spans.push(Span::styled(
                    format!("{} {}", tab.number, tab.label),
                    Style::default().fg(tab.unselected_fg()),
                ));
            }
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "tabs",
            ty: "Vec<Tab>",
            description: "Ordered tabs; each carries its jump number and optional alarm",
        },
        PropInfo {
            name: "active",
            ty: "usize",
            description: "Index of the selected tab — the one inverted block on screen",
        },
    ];

    macro_rules! variant {
        ($ty:ident, $name:literal, $desc:literal, $build:expr) => {
            struct $ty;
            impl Ingredient for $ty {
                fn group(&self) -> &str {
                    "Tab Bar"
                }
                fn name(&self) -> &str {
                    $name
                }
                fn source(&self) -> &str {
                    "wqm_tui::widgets::tab_bar"
                }
                fn description(&self) -> &str {
                    $desc
                }
                fn props(&self) -> &[PropInfo] {
                    PROPS
                }
                fn render(&self, area: Rect, buf: &mut Buffer) {
                    let w: TabBar = $build;
                    w.render(area, buf);
                }
            }
        };
    }

    variant!(
        Default_,
        "Default",
        "Dashboard selected. One cyan block; everything else recedes to muted",
        TabBar::standard(0)
    );

    variant!(
        MidSelection,
        "Mid Selection",
        "Selection moved to Rules — checks the block reads the same away from the left edge",
        TabBar::standard(2)
    );

    variant!(
        AlarmUnselected,
        "Alarm, Unselected",
        "Service owns an offline store while Dashboard is selected: does the alarm compete with the selector?",
        TabBar::new(
            vec![
                Tab::new(1, "Dashboard"),
                Tab::new(2, "Library"),
                Tab::new(3, "Rules"),
                Tab::alarming(4, "Service", Health::Offline),
                Tab::new(5, "Config"),
            ],
            0,
        )
    );

    variant!(
        AlarmSelected,
        "Alarm, Selected",
        "The §4 override: a selected error tab is a RED inverse block, never a cyan one",
        TabBar::new(
            vec![
                Tab::new(1, "Dashboard"),
                Tab::new(2, "Library"),
                Tab::new(3, "Rules"),
                Tab::alarming(4, "Service", Health::Offline),
                Tab::new(5, "Config"),
            ],
            3,
        )
    );

    variant!(
        AlarmDegraded,
        "Alarm, Degraded",
        "Yellow rather than red — the softer alarm still has to beat muted without beating the selector",
        TabBar::new(
            vec![
                Tab::new(1, "Dashboard"),
                Tab::new(2, "Library"),
                Tab::new(3, "Rules"),
                Tab::alarming(4, "Service", Health::Degraded),
                Tab::new(5, "Config"),
            ],
            0,
        )
    );

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Default_),
            Box::new(MidSelection),
            Box::new(AlarmUnselected),
            Box::new(AlarmSelected),
            Box::new(AlarmDegraded),
        ]
    }
}
