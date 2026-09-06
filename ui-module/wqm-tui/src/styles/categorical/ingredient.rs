//! The pantry variants for [`super`].
//!
//! Six frames: the four Catppuccin flavours, one non-Catppuccin theme to show what the
//! fallback honestly amounts to, and the bare strip. The fallback frame is not a curiosity —
//! it is what eleven of the fifteen bundled themes give, so a consumer that assumed a
//! categorical tier exists would be wrong most of the time, and the frame says so out loud.

use super::*;
use ratatui_themes::ThemeName;
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "theme",
        ty: "ThemePalette",
        description: "The palette the tier is read from — its bg is what identifies the flavour",
    },
    PropInfo {
        name: "mirrored",
        ty: "bool",
        description: "The flavour is not bundled; its role mapping is reproduced, not selected",
    },
    PropInfo {
        name: "frame",
        ty: "Frame",
        description: "Table (both ΔE columns) or Strip (swatches only, judged by eye)",
    },
];

struct Variant {
    name: &'static str,
    description: &'static str,
    build: fn() -> CategoricalFrame,
}

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Styles"
    }
    fn group(&self) -> &str {
        "Categorical"
    }
    fn name(&self) -> &str {
        self.name
    }
    fn source(&self) -> &str {
        "wqm_tui::styles::categorical"
    }
    fn description(&self) -> &str {
        self.description
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        (self.build)().render(area, buf);
    }
}

/// A flavour by its Catppuccin identifier, as a palette. Panics on an unknown name, which is a
/// typo in this file and nothing a user can reach.
fn flavour(identifier: &str) -> ThemePalette {
    let flavour = catppuccin::PALETTE
        .all_flavors()
        .into_iter()
        .find(|f| f.identifier() == identifier)
        .expect("a Catppuccin flavour identifier");
    crate::categorical::mirrored_palette(flavour)
}

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    vec![
        Box::new(Variant {
            name: "Mocha",
            description: "The harness's own theme: ten hues, every one clear of the four reserved roles",
            build: || {
                CategoricalFrame::new(ThemeName::CatppuccinMocha.palette(), false, Frame::Table)
            },
        }),
        Box::new(Variant {
            name: "Macchiato",
            description: "One step lighter. Nine hues — the exclusion is measured per flavour, so the count moves",
            build: || CategoricalFrame::new(flavour("macchiato"), true, Frame::Table),
        }),
        Box::new(Variant {
            name: "Frappé",
            description: "Lighter again, and the order changes hands: mauve leads here, not blue",
            build: || CategoricalFrame::new(flavour("frappe"), true, Frame::Table),
        }),
        Box::new(Variant {
            name: "Latte",
            description: "The light flavour: eight hues, and the ΔE columns are much larger — dark accents on a light base",
            build: || {
                CategoricalFrame::new(ThemeName::CatppuccinLatte.palette(), false, Frame::Table)
            },
        }),
        Box::new(Variant {
            name: "Everforest (fallback)",
            description: "What eleven of the fifteen themes actually offer. Read `len()` before relying on distinct hues",
            build: || CategoricalFrame::new(ThemeName::Everforest.palette(), false, Frame::Table),
        }),
        Box::new(Variant {
            name: "Strip",
            description: "Mocha's ten side by side, no numbers — the only frame that answers 'are these telling apart at a glance?'",
            build: || {
                CategoricalFrame::new(ThemeName::CatppuccinMocha.palette(), false, Frame::Strip)
            },
        }),
    ]
}
