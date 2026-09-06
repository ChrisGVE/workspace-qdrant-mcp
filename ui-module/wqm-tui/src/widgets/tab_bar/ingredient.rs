//! The pantry variants for [`crate::widgets::tab_bar`].
//!
//! Split out of the widget rather than sitting under it because the two grew past this
//! crate's 500-line file limit together: five frames of a five-tab row is as much text as the
//! row itself. The module path is unchanged — `tab_bar.rs` beside `tab_bar/` is how a file
//! module keeps its own children — so `pantry.toml` and `crate::tier` see exactly what they
//! saw before.

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
            Tab::for_collection(2, Collection::Libraries),
            Tab::for_collection(3, Collection::Rules),
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
            Tab::for_collection(2, Collection::Libraries),
            Tab::for_collection(3, Collection::Rules),
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
            Tab::for_collection(2, Collection::Libraries),
            Tab::for_collection(3, Collection::Rules),
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
