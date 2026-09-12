//! What the modal switch is pinned to.
//!
//! These guards are about the **switch** — every hue answers muted, the block loses its fill,
//! the bright text rungs collapse to muted while the quiet neutrals survive, the scope restores
//! and nests. The two guards that sweep a whole 125 × 34 frame live beside the screens they
//! sweep (`views::dashboard::tests::modal` and `views::shell`), because they are readings of
//! those screens rather than of this module.
//!
//! # Every one of these reads its tokens OUTSIDE the scope first
//!
//! Inside a scope `tokens::accent()` *is* the muted rung, so an assertion taken there compares
//! muted with muted and passes however wrong the switch is. So each guard reads the live value
//! first and asserts it is not already muted — otherwise the guard is checking nothing, which
//! is the failure mode a colour test is most prone to.

use ratatui::style::{Color, Modifier};

use crate::categorical::Categorical;
use crate::encoding::Encoding;
use crate::tokens::{self, Health, ModalScope, ModalTint, Palette, DISC};
use crate::widgets::chrome::test_support::Restore;

/// Every hue the vocabulary can emit, by the accessor a widget would reach for.
///
/// Named pairs rather than a bare list, so a failure says *which* colour survived.
/// `selector_fg` is here with the rest: it is a hue slot (`Color::Black`), not a rung, and
/// although [`tokens::inverted`] no longer reaches it under a modal, a rule with one silent
/// exception is a rule the next caller gets wrong.
fn hue_accessors() -> Vec<(&'static str, Color)> {
    vec![
        ("accent", tokens::accent()),
        ("selector", tokens::selector()),
        ("selector_fg", tokens::selector_fg()),
        ("in_flight", tokens::in_flight()),
        ("healthy", tokens::healthy()),
        ("degraded", tokens::degraded()),
        ("offline", tokens::offline()),
    ]
}

/// Every hue in the vocabulary is the muted rung under a modal — checked against the rung
/// itself rather than against "not what it was", which would pass on any drift.
#[test]
fn every_hue_the_vocabulary_can_emit_goes_muted_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = hue_accessors();
    let muted = tokens::muted();
    assert!(
        live.iter().all(|(_, colour)| *colour != muted),
        "every hue is already muted with no modal open — this guard would check nothing: {live:?}"
    );

    for tint in ModalTint::ALL {
        ModalTint::set(tint);
        let _modal = ModalScope::enter();
        for (name, _) in &live {
            let under = hue_accessors()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("the same accessors, re-read inside the scope")
                .1;
            assert_eq!(under, muted, "`{name}` survived a {tint:?} modal");
        }
    }
}

/// The bright text rungs are NOT exempt — the other half of the rule, and the half a
/// heavy-handed implementation of the *first* half would miss. A page under a modal is grey
/// **and quiet**: emphasis is a highlight too (VL §6, Chris 2026-09-07: *"we still have colors
/// on the screen while all should be muted"*), so every text rung brighter than `muted` — the
/// baseline, the strong rung, the header and the cursor mark — collapses onto `muted`.
#[test]
fn the_bright_text_rungs_collapse_onto_muted_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let muted = tokens::muted();
    // Read live first: if a rung were already muted with no modal open, the guard below would
    // pass on nothing.
    let live = [
        ("normal", tokens::normal()),
        ("strong", tokens::strong()),
        ("header", tokens::header()),
        ("cursor_mark", tokens::cursor_mark()),
    ];
    assert!(
        live.iter().all(|(_, colour)| *colour != muted),
        "a bright rung is already muted with no modal open — this guard checks nothing"
    );

    let _modal = ModalScope::enter();
    for (name, colour) in [
        ("normal", tokens::normal()),
        ("strong", tokens::strong()),
        ("header", tokens::header()),
        ("cursor_mark", tokens::cursor_mark()),
    ] {
        assert_eq!(colour, muted, "`{name}` stayed bright under a modal");
    }
}

/// The quiet neutral rungs are NOT touched — structure survives. The emphasis ladder's quiet
/// half, the two rule weights, the SELECTION's tint and the layer fills all sit at or below
/// `muted` already, so a modal has nothing to take from them.
///
/// The data cursor's fill left this list on 20260912 (ruling 7): it is a hue now, not a rung,
/// and a modal takes it like every other hue. `tokens`'s own guards hold that end.
#[test]
fn the_quiet_neutral_rungs_do_not_move_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = [
        tokens::faint(),
        tokens::muted(),
        tokens::rule_frame(),
        tokens::rule_internal(),
        tokens::selection_bg(),
        tokens::edit_bg(),
        tokens::layer1_bg(),
        tokens::layer2_bg(),
    ];

    for tint in ModalTint::ALL {
        ModalTint::set(tint);
        let _modal = ModalScope::enter();
        let under = [
            tokens::faint(),
            tokens::muted(),
            tokens::rule_frame(),
            tokens::rule_internal(),
            tokens::selection_bg(),
            tokens::edit_bg(),
            tokens::layer1_bg(),
            tokens::layer2_bg(),
        ];

        assert_eq!(under, live, "a quiet neutral rung moved under a {tint:?} modal");
    }
}

#[test]
fn modal_border_uses_each_selected_hue_even_inside_the_modal_scope() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();
    Palette::set(Palette::Bundled);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    for tint in ModalTint::ALL {
        ModalTint::set(tint);
        let border = tokens::modal_border();
        let role = match tint {
            ModalTint::Neutral => tokens::muted(),
            ModalTint::Accent => tokens::accent(),
            ModalTint::Selected => tokens::selected(),
            ModalTint::InFlight => tokens::in_flight(),
        };
        assert_eq!(border, role, "{tint:?} uses the wrong role");
        if tint == ModalTint::Neutral {
            assert_eq!(border, tokens::muted());
        } else {
            assert_ne!(border, tokens::muted(), "{tint:?} has no border hue");
        }
        let _modal = ModalScope::enter();
        assert_eq!(tokens::modal_border(), border, "{tint:?} chrome was muted");
    }
}

#[test]
fn modal_fill_is_a_subtle_wash_of_each_layer() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();
    Palette::set(Palette::Bundled);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    for tint in ModalTint::ALL {
        ModalTint::set(tint);
        for layer in [tokens::layer1_bg(), tokens::layer2_bg()] {
            let fill = tokens::modal_fill(layer);
            if tint == ModalTint::Neutral {
                assert_eq!(fill, layer);
            } else {
                assert_ne!(fill, layer, "{tint:?} did not tint the layer");
                assert!(tokens::delta_e(fill, layer) < 30.0, "{tint:?} wash is too strong");
                let _modal = ModalScope::enter();
                assert_eq!(tokens::modal_fill(layer), fill, "{tint:?} fill was muted");
            }
        }
    }
}

/// Every health state's disc keeps its SHAPE and loses its hue (VL §4).
#[test]
fn every_health_state_paints_muted_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let states = [Health::Healthy, Health::Degraded, Health::Offline];
    let live: Vec<Color> = states.iter().map(|state| state.color()).collect();
    let muted = tokens::muted();
    assert!(
        live.iter().all(|colour| *colour != muted),
        "the three states are already muted — this guard would check nothing"
    );

    let _modal = ModalScope::enter();
    for state in states {
        assert_eq!(state.color(), muted, "{state:?} kept its hue under a modal");
        assert_eq!(
            state.glyph(),
            DISC,
            "{state:?} changed shape under a modal — only the hue goes"
        );
    }
}

/// A data hue is a hue: the categorical tier mutes with the roles.
///
/// Measured on Mocha, which is the one bundled flavour that carries a tier worth the name —
/// on the eleven non-Catppuccin themes the tier is a single entry and "they all went muted"
/// would be one comparison dressed as eight.
#[test]
fn the_categorical_tier_mutes_with_the_roles() {
    let _serial = crate::global_state_lock();
    let previous = (Palette::current(), Encoding::current(), tokens::theme());
    Palette::set(Palette::Bundled);
    Encoding::set(Encoding::TrueColor);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    let tier = Categorical::current();
    assert!(
        tier.len() > 1,
        "Mocha carries a real categorical tier, or this guard is checking nothing"
    );
    let live: Vec<Color> = tier.iter().map(|(_, colour)| colour).collect();

    let under: Vec<Color> = {
        let _modal = ModalScope::enter();
        let read: Vec<Color> = tier.iter().map(|(_, colour)| colour).collect();
        for (i, colour) in read.iter().enumerate() {
            assert_eq!(
                tier.color(i),
                Some(*colour),
                "`color` and `iter` disagree about entry {i} under a modal"
            );
        }
        read
    };

    assert_ne!(live, under, "the tier painted straight through a modal");
    let muted = tokens::muted();
    assert!(
        under.iter().all(|colour| *colour == muted),
        "a data hue survived a modal: {under:?}"
    );

    Palette::set(previous.0);
    Encoding::set(previous.1);
    if let Some(theme) = previous.2 {
        tokens::set_theme(theme);
    }
}

/// The inverted block loses its fill and keeps its weight.
///
/// `bg: None` rather than a darker fill: a block behind a page a modal has taken the input from
/// is precisely the highlight §6 removes, and the padding stays so nothing on the row moves.
#[test]
fn an_inverted_block_under_a_modal_has_no_fill_at_all() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let selector = tokens::selector();
    assert_eq!(
        tokens::inverted(selector).bg,
        Some(selector),
        "with nothing over it the block IS the fill"
    );

    let _modal = ModalScope::enter();
    let under = tokens::inverted(selector);
    assert_eq!(under.bg, None, "a fill survived a modal");
    assert_eq!(under.fg, Some(tokens::muted()));
    assert!(
        under.add_modifier.contains(Modifier::BOLD),
        "weight is what is left to say which one is selected"
    );
    assert!(
        !under.add_modifier.contains(Modifier::REVERSED),
        "reverse video is a fill by another name"
    );
}

/// The scope restores on drop, and nests — an inner scope's end must not switch the page back
/// on underneath an outer one.
#[test]
fn the_scope_restores_on_drop_and_nests() {
    let _serial = crate::global_state_lock();

    assert!(!tokens::under_modal(), "a test left a scope open");
    {
        let outer = ModalScope::enter();
        assert!(tokens::under_modal());
        {
            let _inner = ModalScope::enter();
            assert!(tokens::under_modal());
        }
        assert!(
            tokens::under_modal(),
            "the inner scope's drop switched the page back on under the outer one"
        );
        drop(outer);
        assert!(!tokens::under_modal(), "the outer scope outlived its value");
    }
    assert!(!tokens::under_modal());
}
